"""Validated NetCDF loader, train-only Normalizer, and leakage-free sliding window dataset."""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

from monsoon_r1.contract import CHANNELS, LAT, LON, NC, NLAT, NLON, T_IN, doy_sincos as contract_doy_sincos

DEFAULT_NC_PATH = Path(
    os.environ.get(
        "MONSOONVERSE_DATASET_PATH",
        str(Path(__file__).resolve().parents[3] / "official-kit" / "release" / "india_climate_train.nc"),
    )
)
REF_TRAIN_END_DATE = dt.date(2019, 12, 31)
REF_VAL_START_DATE = dt.date(2020, 1, 1)


@dataclass
class ClimateArchive:
    data: np.ndarray       # [T, 6, 32, 36] float32 physical units
    static: np.ndarray     # [2, 32, 36] float32 (land_sea_mask, orography_m)
    dates: np.ndarray      # [T] datetime.date objects
    dates_int: np.ndarray  # [T] YYYYMMDD int64
    lat: np.ndarray        # [32]
    lon: np.ndarray        # [36]


def load_climate_archive(nc_path: str | Path = DEFAULT_NC_PATH) -> ClimateArchive:
    """Load india_climate_train.nc, verifying variables, grid, units, continuity, and finiteness."""
    p = Path(nc_path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset not found: {p}")

    try:
        ctx = xr.open_dataset(p, engine="h5netcdf")
    except Exception:
        ctx = xr.open_dataset(p)

    with ctx as ds:
        if "data" not in ds or "static" not in ds:
            raise ValueError(f"Expected 'data' and 'static' variables, got {list(ds.data_vars)}")

        channels = tuple(str(c) for c in ds.coords["channel"].values)
        if channels != CHANNELS:
            raise ValueError(f"Channel order mismatch: {channels} != {CHANNELS}")

        lat = np.asarray(ds.coords["lat"].values, dtype=np.float32)
        lon = np.asarray(ds.coords["lon"].values, dtype=np.float32)
        if lat.shape != (NLAT,) or not np.allclose(lat, LAT):
            raise ValueError(f"Unexpected lat coords: {lat}")
        if lon.shape != (NLON,) or not np.allclose(lon, LON):
            raise ValueError(f"Unexpected lon coords: {lon}")

        data = np.asarray(ds["data"].values, dtype=np.float32)
        static = np.asarray(ds["static"].values, dtype=np.float32)
        if data.ndim != 4 or data.shape[1:] != (NC, NLAT, NLON):
            raise ValueError(f"Unexpected data shape: {data.shape}")
        if static.shape != (2, NLAT, NLON):
            raise ValueError(f"Unexpected static shape: {static.shape}")

        if not np.isfinite(data).all():
            raise ValueError("Non-finite (NaN/Inf) values found in dynamic data")
        if not np.isfinite(static).all():
            raise ValueError("Non-finite (NaN/Inf) values found in static data")
        if float(data[:, 1].min()) < -1e-6:
            raise ValueError(f"Invalid negative precipitation found: min={data[:, 1].min()}")
        data[:, 1] = np.maximum(data[:, 1], 0.0)

        raw_times = ds.coords["time"].values
        dates = np.array([dt.date.fromisoformat(str(t)[:10]) for t in raw_times], dtype=object)
        dates_int = np.array([int(f"{d:%Y%m%d}") for d in dates], dtype=np.int64)

        for i in range(len(dates) - 1):
            if (dates[i + 1] - dates[i]).days != 1:
                raise ValueError(f"Daily continuity violation at idx {i}: {dates[i]} -> {dates[i+1]}")

    return ClimateArchive(
        data=data,
        static=static,
        dates=dates,
        dates_int=dates_int,
        lat=lat,
        lon=lon,
    )


def chronological_split_indices(
    dates: np.ndarray,
    mode: str = "reference",
    train_frac: float = 0.8,
    gap_days: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return strictly chronological, non-overlapping train/val index arrays."""
    if mode == "reference":
        train_idx = np.where(dates <= REF_TRAIN_END_DATE)[0]
        val_idx = np.where(dates >= REF_VAL_START_DATE)[0]
    elif mode == "incumbent":
        n_total = len(dates)
        train_end = int(n_total * train_frac)
        val_start = train_end + int(gap_days)
        train_idx = np.arange(0, train_end, dtype=np.int64)
        val_idx = np.arange(val_start, n_total, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported split mode: {mode!r}")

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Empty train or validation split")
    if int(train_idx[-1]) >= int(val_idx[0]):
        raise ValueError("Train/val split overlap detected")
    return train_idx, val_idx


@dataclass
class Normalizer:
    """Train-only channel normalization with zero-div protection and physical tp>=0."""
    mean: np.ndarray
    std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    log1p_tp: bool = False

    @classmethod
    def fit(
        cls,
        train_data: np.ndarray,
        static_fields: np.ndarray,
        log1p_tp: bool = False,
    ) -> "Normalizer":
        x = np.asarray(train_data, dtype=np.float64).copy()
        if log1p_tp:
            x[:, 1] = np.log1p(np.maximum(x[:, 1], 0.0))
        mean = x.mean(axis=(0, 2, 3)).astype(np.float32)
        std = np.maximum(x.std(axis=(0, 2, 3)), 1e-6).astype(np.float32)
        s_mean = np.array([0.0, float(static_fields[1].mean())], dtype=np.float32)
        s_std = np.array([1.0, max(float(static_fields[1].std()), 1e-6)], dtype=np.float32)
        return cls(mean=mean, std=std, static_mean=s_mean, static_std=s_std, log1p_tp=bool(log1p_tp))

    def normalize(self, phys: np.ndarray) -> np.ndarray:
        arr = np.asarray(phys, dtype=np.float32).copy()
        if self.log1p_tp:
            arr[..., 1, :, :] = np.log1p(np.maximum(arr[..., 1, :, :], 0.0))
        view = (1,) * (arr.ndim - 3) + (NC, 1, 1)
        return ((arr - self.mean.reshape(view)) / np.maximum(self.std.reshape(view), 1e-6)).astype(np.float32)

    def denormalize(self, norm: np.ndarray) -> np.ndarray:
        arr = np.asarray(norm, dtype=np.float32)
        view = (1,) * (arr.ndim - 3) + (NC, 1, 1)
        out = arr * np.maximum(self.std.reshape(view), 1e-6) + self.mean.reshape(view)
        if self.log1p_tp:
            out[..., 1, :, :] = np.expm1(np.clip(out[..., 1, :, :], 0.0, 6.5))
        else:
            out[..., 1, :, :] = np.maximum(out[..., 1, :, :], 0.0)
        return out.astype(np.float32)

    def normalize_static(self, static: np.ndarray) -> np.ndarray:
        return (
            (static - self.static_mean[:, None, None])
            / np.maximum(self.static_std[:, None, None], 1e-6)
        ).astype(np.float32)

    def validate_round_trip(self, phys: np.ndarray, atol: float = 1e-3) -> float:
        norm = self.normalize(phys)
        rec = self.denormalize(norm)
        diff = float(np.max(np.abs(rec - phys)))
        if diff > atol:
            raise ValueError(f"Normalization round-trip error {diff:.6g} > {atol}")
        return diff

    def to_dict(self) -> dict:
        return {
            "channels": list(CHANNELS),
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "static_channels": ["land_sea_mask", "orography_m"],
            "static_mean": [float(v) for v in self.static_mean],
            "static_std": [float(v) for v in self.static_std],
            "log1p_tp": bool(self.log1p_tp),
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            mean=np.asarray(d["mean"], dtype=np.float32),
            std=np.asarray(d["std"], dtype=np.float32),
            static_mean=np.asarray(d["static_mean"], dtype=np.float32),
            static_std=np.asarray(d["static_std"], dtype=np.float32),
            log1p_tp=bool(d.get("log1p_tp", False)),
        )


def date_to_doy_sincos(d: dt.date) -> np.ndarray:
    return np.asarray(contract_doy_sincos(d.timetuple().tm_yday), dtype=np.float32)


class MonsoonWindowDataset(Dataset):
    """Sliding windows strictly bounded within one split: (init_norm, doy_sc, target_norm, target_phys)."""

    def __init__(
        self,
        data_phys: np.ndarray,
        data_norm: np.ndarray,
        dates: np.ndarray,
        t_in: int = T_IN,
        horizon: int = 1,
        stride: int = 1,
    ) -> None:
        if not (len(data_phys) == len(data_norm) == len(dates)):
            raise ValueError("Mismatched split lengths")
        self.data_phys = torch.from_numpy(np.ascontiguousarray(data_phys, dtype=np.float32))
        self.data_norm = torch.from_numpy(np.ascontiguousarray(data_norm, dtype=np.float32))
        self.doy_sc = torch.from_numpy(
            np.stack([date_to_doy_sincos(d) for d in dates]).astype(np.float32)
        )
        self.t_in = int(t_in)
        self.horizon = int(horizon)

        max_start = len(dates) - self.t_in - self.horizon
        if max_start < 0:
            raise ValueError(f"Split too short ({len(dates)} days) for t_in={self.t_in}, horizon={self.horizon}")
        self.starts = np.arange(0, max_start + 1, int(stride), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = int(self.starts[idx])
        last_in = s + self.t_in - 1
        init_norm = self.data_norm[s : s + self.t_in]
        doy_sc = self.doy_sc[last_in]
        target_norm = self.data_norm[last_in + 1 : last_in + 1 + self.horizon]
        target_phys = self.data_phys[last_in + 1 : last_in + 1 + self.horizon]
        return init_norm, doy_sc, target_norm, target_phys
