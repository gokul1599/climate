"""PyTorch Dataset for MonsoonVerse spatiotemporal training and validation."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from data_pipeline.create_windows import build_window_indices
from data_pipeline.load_data import load_climate_dataset, load_stats
from data_pipeline.preprocess import chronological_split, compute_doy_sincos, normalise_dynamic


class ClimateWindowDataset(Dataset):
    """Yields normalised 3-day dynamic history, static fields, DOY sin/cos, and H-step future targets."""

    def __init__(
        self,
        data_norm: np.ndarray,
        static_norm: np.ndarray,
        doy_sincos: np.ndarray,
        t_in: int = 3,
        horizon: int = 1,
    ) -> None:
        self.data_norm = data_norm.astype(np.float32)
        self.static_norm = static_norm.astype(np.float32)  # [2, 32, 36]
        self.doy_sincos = doy_sincos.astype(np.float32)    # [N, 2]
        self.t_in = t_in
        self.horizon = horizon
        self.starts = build_window_indices(len(data_norm), t_in=t_in, horizon=horizon)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = int(self.starts[idx])
        hist = self.data_norm[s : s + self.t_in]                  # [3, 6, 32, 36]
        target = self.data_norm[s + self.t_in : s + self.t_in + self.horizon]  # [H, 6, 32, 36]
        doy_last = self.doy_sincos[s + self.t_in - 1]             # [2] (last input day)
        doy_future = self.doy_sincos[s + self.t_in - 1 : s + self.t_in - 1 + self.horizon]  # [H, 2]
        return {
            "hist": torch.from_numpy(hist),
            "static": torch.from_numpy(self.static_norm),
            "doy_last": torch.from_numpy(doy_last),
            "doy_seq": torch.from_numpy(doy_future),
            "target": torch.from_numpy(target),
        }


def load_train_val_datasets(
    t_in: int = 3,
    horizon: int = 1,
    train_frac: float = 0.8,
    gap_days: int = 10,
) -> tuple[ClimateWindowDataset, ClimateWindowDataset, dict]:
    """Load dataset, normalise with stats.json, and return chronological train/val Datasets + metadata."""
    ds = load_climate_dataset()
    stats = load_stats()
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)

    data_phys = ds["data"]
    data_norm = normalise_dynamic(data_phys, mean, std)
    static = ds["static"]
    oro_mean = float(static[1].mean())
    oro_std = float(static[1].std()) if float(static[1].std()) > 0 else 1.0
    static_norm = np.stack([static[0], (static[1] - oro_mean) / oro_std], axis=0).astype(np.float32)
    doy_sc = compute_doy_sincos(ds["time"])

    tr_sl, va_sl = chronological_split(len(data_phys), train_frac=train_frac, gap_days=gap_days)

    train_ds = ClimateWindowDataset(data_norm[tr_sl], static_norm, doy_sc[tr_sl], t_in=t_in, horizon=horizon)
    val_ds = ClimateWindowDataset(data_norm[va_sl], static_norm, doy_sc[va_sl], t_in=t_in, horizon=horizon)

    info = {
        "mean": mean,
        "std": std,
        "static_norm": static_norm,
        "static_phys": static,
        "data_phys": data_phys,
        "time": ds["time"],
        "train_slice": tr_sl,
        "val_slice": va_sl,
    }
    return train_ds, val_ds, info
