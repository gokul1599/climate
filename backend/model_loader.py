"""Singleton model loader supporting incumbent, LocalResNet-v3, and promoted LocalResNet-v4."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ml.inference import MonsoonPredictor
from ml.local_resnet_v3.inference import LocalResNetV3Predictor

ROOT_DIR = Path(__file__).resolve().parents[1]
_PREDICTOR: MonsoonPredictor | LocalResNetV3Predictor | None = None
_ACTIVE_VARIANT: str = "incumbent"


def resolve_variant(variant: str | None = None) -> str:
    env_v = os.environ.get("MONSOONVERSE_MODEL_VARIANT")
    if variant is not None:
        raw = variant.strip().lower()
    elif env_v is not None and env_v.strip():
        raw = env_v.strip().lower()
    else:
        # Promoted default when verified LocalResNet-v4 artifacts are present (Protocol A: 65.56/70, 100% warm_ocean, 2/13 stall)
        v4_dir = ROOT_DIR / "artifacts" / "local_resnet_v4"
        if (v4_dir / "model.pt").is_file() and (v4_dir / "norm_stats.json").is_file():
            return "local_resnet_v4"
        v3_dir = ROOT_DIR / "artifacts" / "local_resnet_v3"
        if (v3_dir / "model.pt").is_file() and (v3_dir / "norm_stats.json").is_file():
            return "local_resnet_v3"
        return "incumbent"
    if raw in ("local_resnet_v4", "v4"):
        return "local_resnet_v4"
    if raw in ("local_resnet_v3", "challenger", "v3"):
        return "local_resnet_v3"
    return "incumbent"


def create_predictor(variant: str | None = None) -> MonsoonPredictor | LocalResNetV3Predictor:
    v = resolve_variant(variant)
    if v in ("local_resnet_v4", "local_resnet_v3"):
        cand_dir = ROOT_DIR / "artifacts" / v
        if (cand_dir / "model.pt").is_file() and (cand_dir / "norm_stats.json").is_file():
            return LocalResNetV3Predictor(artifacts_dir=cand_dir)
        raise FileNotFoundError(f"Requested {v} artifacts not found under {cand_dir}")
    return MonsoonPredictor(artifacts_dir=ROOT_DIR / "artifacts")


def load_predictor(variant: str | None = None) -> MonsoonPredictor | LocalResNetV3Predictor:
    """Load configured model weights once at startup."""
    global _PREDICTOR, _ACTIVE_VARIANT
    requested = resolve_variant(variant)
    if _PREDICTOR is None or requested != _ACTIVE_VARIANT:
        _PREDICTOR = create_predictor(requested)
        _ACTIVE_VARIANT = requested
    return _PREDICTOR


def get_model_info() -> dict[str, Any]:
    """Return active model identity, architecture summary, and SHA-256 provenance."""
    pred = load_predictor()
    art = ROOT_DIR / "artifacts"
    challenger_ready = (art / "local_resnet_v4" / "model.pt").is_file() or (art / "local_resnet_v3" / "model.pt").is_file()

    if _ACTIVE_VARIANT == "local_resnet_v4":
        mdir = art / "local_resnet_v4"
        ckpt_p = mdir / "model.pt"
        man_p = mdir / "manifest.json"
        display_name = getattr(pred, "model_name", "MonsoonVerse-LocalResNet-v4")
        arch_summary = "24ch -> stem(64) -> 4x LocalResBlock(dil=[1,2,4,1], init_scale=0.25, no GroupNorm) -> 313,030 params"
    elif _ACTIVE_VARIANT == "local_resnet_v3":
        mdir = art / "local_resnet_v3"
        ckpt_p = mdir / "model.pt"
        man_p = mdir / "manifest.json"
        display_name = getattr(pred, "model_name", "MonsoonVerse-LocalResNet-v3")
        arch_summary = "24ch -> stem(64) -> 4x LocalResBlock(dil=[1,2,2,1], init_scale=0.25, no GroupNorm) -> 313,030 params"
    else:
        ckpt_p = art / "model.pt"
        man_p = art / "incumbent_manifest.json"
        display_name = "MonsoonVerse-ClimResCNN"
        arch_summary = "28ch (+6 clim) -> stem(32) -> 5x ResBlock(GroupNorm, SiLU) -> 103,014 params"

    ckpt_sha = hashlib.sha256(ckpt_p.read_bytes()).hexdigest() if ckpt_p.is_file() else None
    manifest = json.loads(man_p.read_text(encoding="utf-8")) if man_p.is_file() else None
    return {
        "active_variant": _ACTIVE_VARIANT,
        "model_name": display_name,
        "architecture_summary": arch_summary,
        "checkpoint_sha256": ckpt_sha,
        "challenger_available": challenger_ready,
        "manifest": manifest,
    }
