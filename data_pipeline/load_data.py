"""Load india_climate_train.nc, stats.json, and static geographic fields."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
STATIC_CHANNELS = ("land_sea_mask", "orography_m")
DEFAULT_NC_PATH = Path(__file__).resolve().parents[2] / "official-kit" / "release" / "india_climate_train.nc"
DEFAULT_STATS_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "stats.json"


def load_stats(stats_path: Path | str = DEFAULT_STATS_PATH) -> dict[str, Any]:
    """Load normalisation statistics JSON and validate channel order."""
    p = Path(stats_path)
    with p.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    if tuple(stats.get("channels", [])) != CHANNELS:
        raise ValueError(f"Channel order mismatch in {p}: {stats.get('channels')}")
    return stats


def load_climate_dataset(
    nc_path: Path | str = DEFAULT_NC_PATH,
) -> dict[str, Any]:
    """Load the NetCDF dataset into numpy arrays with verified orientation and shapes."""
    p = Path(nc_path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset not found: {p}")
    try:
        ctx = xr.open_dataset(p, engine="h5netcdf")
    except Exception:
        ctx = xr.open_dataset(p)
    with ctx as ds:
        data = np.asarray(ds["data"].values, dtype=np.float32)  # [T, 6, 32, 36]
        static = np.asarray(ds["static"].values, dtype=np.float32)  # [2, 32, 36]
        time_vals = ds["time"].values
        lat = np.asarray(ds["lat"].values, dtype=np.float32)
        lon = np.asarray(ds["lon"].values, dtype=np.float32)
        channels = tuple(str(c) for c in ds["channel"].values)
        static_channels = tuple(str(c) for c in ds["static_channel"].values)

    if channels != CHANNELS:
        raise ValueError(f"Unexpected channels: {channels} != {CHANNELS}")
    if static_channels != STATIC_CHANNELS:
        raise ValueError(f"Unexpected static channels: {static_channels} != {STATIC_CHANNELS}")
    if data.ndim != 4 or data.shape[1:] != (6, 32, 36):
        raise ValueError(f"Unexpected data shape: {data.shape}")
    if static.shape != (2, 32, 36):
        raise ValueError(f"Unexpected static shape: {static.shape}")

    return {
        "data": data,
        "static": static,
        "time": time_vals,
        "lat": lat,
        "lon": lon,
        "channels": channels,
        "static_channels": static_channels,
    }


def save_static_fields(static: np.ndarray, out_path: Path | str) -> None:
    """Save [2, 32, 36] static fields (and normalised versions) to artifacts/static_fields.npz."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lsm = static[0].astype(np.float32)
    oro = static[1].astype(np.float32)
    oro_mean = float(oro.mean())
    oro_std = float(oro.std()) if float(oro.std()) > 0 else 1.0
    oro_norm = ((oro - oro_mean) / oro_std).astype(np.float32)
    static_norm = np.stack([lsm, oro_norm], axis=0).astype(np.float32)
    np.savez(
        out,
        static=static.astype(np.float32),
        static_norm=static_norm,
        oro_mean=np.float32(oro_mean),
        oro_std=np.float32(oro_std),
    )
