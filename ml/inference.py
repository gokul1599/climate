"""Inference interface for MonsoonVerse."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ml.model import MonsoonResCNN
from ml.normalisation import load_norm_stats
from ml.rollout import autoregressive_rollout


class MonsoonPredictor:
    """Holds loaded model, normalisation stats, static fields, and training climatology for CPU inference."""

    def __init__(self, artifacts_dir: Path | str | None = None) -> None:
        root = Path(__file__).resolve().parents[1]
        art = Path(artifacts_dir) if artifacts_dir is not None else root / "artifacts"

        ckpt = torch.load(art / "model.pt", map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        use_clim = bool(cfg.get("use_clim_cond", False))
        self.model_name = "MonsoonVerse"
        self.model = MonsoonResCNN(
            hidden_channels=int(cfg["hidden_channels"]),
            num_blocks=int(cfg["num_blocks"]),
            use_clim_cond=use_clim,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.mean, self.std = load_norm_stats(art / "stats.json")
        st = np.load(art / "static_fields.npz")
        self.static_norm = np.asarray(st["static_norm"], dtype=np.float32)

        self.clim_norm_366: np.ndarray | None = None
        clim_path = art / "climatology.npz"
        if use_clim and clim_path.is_file():
            self.clim_norm_366 = np.asarray(np.load(clim_path)["clim_norm"], dtype=np.float32)

    def predict_rollout(
        self,
        init_states: np.ndarray,
        doy_sincos: list[float] | np.ndarray,
        steps: int = 10,
        perturbation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict `steps` days in physical units given [3,6,32,36] init_states."""
        return autoregressive_rollout(
            model=self.model,
            init_states_phys=init_states,
            doy_sincos=doy_sincos,
            steps=steps,
            mean=self.mean,
            std=self.std,
            static_norm=self.static_norm,
            perturbation_phys=perturbation,
            clim_norm_366=self.clim_norm_366,
        )


_DEFAULT_PREDICTOR: MonsoonPredictor | None = None


def get_predictor() -> MonsoonPredictor:
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = MonsoonPredictor()
    return _DEFAULT_PREDICTOR


def predict_rollout(
    init_states: np.ndarray,
    doy_sincos: list[float] | np.ndarray,
    steps: int = 10,
    perturbation: np.ndarray | None = None,
) -> np.ndarray:
    """Contract-compatible functional inference entrypoint."""
    return get_predictor().predict_rollout(init_states, doy_sincos, steps=steps, perturbation=perturbation)
