"""Tests for upgraded API provenance/uncertainty endpoints, model selector, and import isolation."""
from __future__ import annotations

import subprocess
import sys

from backend.api import evaluation_data, system_status
from backend.model_loader import get_model_info, resolve_variant


def test_variant_selector_supports_promoted_default_and_incumbent_fallback(monkeypatch):
    monkeypatch.delenv("MONSOONVERSE_MODEL_VARIANT", raising=False)
    # Default promotes local_resnet_v3 when trained artifacts exist, else incumbent
    default_v = resolve_variant()
    assert default_v in ("local_resnet_v3", "incumbent")
    assert resolve_variant("incumbent") == "incumbent"
    assert resolve_variant("local_resnet_v3") == "local_resnet_v3"
    info = get_model_info()
    assert info["active_variant"] == default_v
    assert info["checkpoint_sha256"] is not None and len(info["checkpoint_sha256"]) == 64


def test_system_status_and_evaluation_provenance_endpoints():
    st = system_status()
    assert st["active_variant"] in ("local_resnet_v3", "incumbent")
    assert st["architecture_summary"]
    assert len(st["checkpoint_sha256"]) == 64

    ev = evaluation_data()
    assert ev["incumbent_manifest"]["model_name"] == "MonsoonVerse-ClimResCNN"
    assert ev["incumbent_uncertainty_bands"]["is_probabilistic_ensemble"] is False
    assert len(ev["incumbent_uncertainty_bands"]["leads"]) == 10
    assert ev["active_model"]["active_variant"] in ("local_resnet_v3", "incumbent")
    if ev["active_model"]["active_variant"] == "local_resnet_v3":
        assert ev["challenger_manifest"]["model_name"] == "MonsoonVerse-LocalResNet-v3"
        assert ev["model_comparison"]["promotion_decision"]["decision"] == "PROMOTE_CHALLENGER"


def test_serving_import_graph_has_no_xarray_or_train():
    code = (
        "import sys, backend.api; "
        "assert 'xarray' not in sys.modules, 'xarray leaked into serving import graph'; "
        "assert 'ml.local_resnet_v3.train' not in sys.modules, 'train leaked'; "
        "assert 'ml.train' not in sys.modules, 'ml.train leaked'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
