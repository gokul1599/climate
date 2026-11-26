"""Comprehensive unit & integration tests for LocalResNet-v3 architecture, losses, dataset, inference & stress."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
import torch

from impact.anomaly_analysis import compute_scenario_anomalies
from impact.regional_analysis import compute_regional_summary
from ml.local_resnet_v3.dataset import (
    MonsoonWindowDataset,
    Normalizer,
    chronological_split_indices,
)
from ml.local_resnet_v3.inference import LocalResNetV3Predictor, advance_doy_sincos, make_coord_grid
from ml.local_resnet_v3.loss import (
    MonsoonRolloutLoss,
    lat_weighted_huber_loss,
    make_lat_weights_tensor,
    spatial_gradient_loss,
    spatial_variance_floor_loss,
)
from ml.local_resnet_v3.model import LocalResBlock, MonsoonResNet
from ml.local_resnet_v3.train import rollout_forward


def test_local_resnet_v3_architecture_and_params():
    model = MonsoonResNet(hidden_dim=64, num_blocks=4, dilations=(1, 2, 2, 1), init_scale=0.25)
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 313030
    assert model.dilations == [1, 2, 2, 1]
    assert isinstance(model.blocks[0], LocalResBlock)
    assert float(model.blocks[0].scale.mean().item()) == pytest.approx(0.25)

    model.eval()
    hist = torch.randn(2, 3, 6, 32, 36)
    static = torch.randn(2, 2, 32, 36)
    doy = torch.tensor([[0.0, 1.0], [0.5, 0.866]], dtype=torch.float32)
    coord = torch.from_numpy(make_coord_grid()[None]).expand(2, -1, -1, -1)
    out1 = model(hist, static, doy, coord)
    out2 = model(hist, static, doy, coord)
    assert out1.shape == (2, 6, 32, 36)
    assert torch.allclose(out1, out2)

    # Zero weights head -> residual equals last input frame
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    out_zero = model(hist, static, doy, coord)
    assert torch.allclose(out_zero, hist[:, -1], atol=1e-6)


def test_loss_components_behavior_and_gradients():
    lat_w = make_lat_weights_tensor()
    ch_w = torch.tensor([1.0, 1.4, 1.0, 1.0, 1.0, 1.0]).view(1, 6, 1, 1)

    target = torch.randn(2, 4, 6, 32, 36)
    pred = (target * 0.2 + 0.1).clone().requires_grad_(True)

    # 1. Constant fields have zero spatial gradient & variance floor loss
    const_f = torch.ones(2, 2, 6, 32, 36)
    assert float(spatial_gradient_loss(const_f.reshape(4, 6, 32, 36), const_f.reshape(4, 6, 32, 36), lat_w)) == 0.0
    assert float(spatial_variance_floor_loss(const_f, const_f, min_ratio=0.65)) == 0.0

    # 2. Variance collapse penalty activates when pred var < min_ratio * target var
    vf = spatial_variance_floor_loss(pred, target, min_ratio=0.75)
    assert float(vf.detach()) > 0.0
    vf_same = spatial_variance_floor_loss(target, target, min_ratio=0.75)
    assert float(vf_same.detach()) == pytest.approx(0.0)

    # 3. Heavy rain weighting boosts tp errors only when weight > 1
    l_base, extra0 = lat_weighted_huber_loss(pred, target, lat_w, ch_w, rain_thresh_norm=0.5, heavy_rain_boost=1.0)
    l_boost, extra2 = lat_weighted_huber_loss(pred, target, lat_w, ch_w, rain_thresh_norm=0.5, heavy_rain_boost=2.2)
    assert float(extra0.detach()) == 0.0
    assert float(extra2.detach()) >= 0.0
    assert float(l_boost.detach()) >= float(l_base.detach())

    # 4. Combined loss finite forward + backward grads + zero-weight disable
    crit_zero = MonsoonRolloutLoss(grad_weight=0.0, var_weight=0.0, heavy_rain_boost=1.0)
    t_z, comps_z = crit_zero.forward_with_components(pred, target)
    assert comps_z["grad_weighted"] == 0.0 and comps_z["var_floor_weighted"] == 0.0

    crit = MonsoonRolloutLoss(grad_weight=0.14, var_weight=0.08, var_min_ratio=0.65, rain_thresh_norm=0.5, heavy_rain_boost=2.2)
    total, comps = crit.forward_with_components(pred, target)
    assert torch.isfinite(total)
    total.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_dataset_split_normalization_and_rollout_alignment():
    dates = np.array([dt.date(2019, 12, 20) + dt.timedelta(days=i) for i in range(30)], dtype=object)
    tr_idx, va_idx = chronological_split_indices(dates, mode="reference")
    assert dates[tr_idx[-1]] <= dt.date(2019, 12, 31)
    assert dates[va_idx[0]] >= dt.date(2020, 1, 1)
    assert tr_idx[-1] < va_idx[0]

    rng = np.random.default_rng(7)
    data = rng.normal(0.0, 2.0, size=(30, 6, 32, 36)).astype(np.float32)
    data[:, 0] += 298.0
    data[:, 1] = np.abs(data[:, 1]) * 3.0
    static = rng.uniform(0.0, 1.0, size=(2, 32, 36)).astype(np.float32)

    norm = Normalizer.fit(data[tr_idx], static)
    err = norm.validate_round_trip(data[tr_idx], atol=1e-4)
    assert err < 1e-4

    # Check exact dataset horizon alignment inside train split (never crosses to val)
    ds = MonsoonWindowDataset(data[tr_idx], norm.normalize(data[tr_idx]), dates[tr_idx], horizon=3, stride=1)
    assert len(ds) == len(tr_idx) - 3 - 3 + 1
    init_n, doy_sc, tgt_n, tgt_p = ds[0]
    assert init_n.shape == (3, 6, 32, 36)
    assert doy_sc.shape == (2,)
    assert tgt_n.shape == (3, 6, 32, 36)
    assert np.allclose(tgt_p.numpy(), data[tr_idx][3:6])

    # Check DOY advances smoothly around circle
    s0, c0 = 0.0, 1.0
    s1, c1 = advance_doy_sincos(s0, c0)
    assert abs(s1 * s1 + c1 * c1 - 1.0) < 1e-6


def test_local_resnet_predictor_contract_and_prefix_consistency(tmp_path: Path):
    # Create temporary lightweight LocalResNet-v3 checkpoint to test predictor contract offline
    adir = tmp_path / "cand"
    adir.mkdir()
    rng = np.random.default_rng(11)
    init_s = rng.normal(0.0, 1.0, size=(3, 6, 32, 36)).astype(np.float32)
    init_s[:, 0] = 300.0
    init_s[:, 1] = np.abs(init_s[:, 1]) + 1.0
    init_s[:, 4] = 101000.0
    init_s[:, 5] = 40.0
    static = np.ones((2, 32, 36), dtype=np.float32)

    norm = Normalizer.fit(init_s, static)
    norm.save(adir / "norm_stats.json")
    np.savez_compressed(adir / "static_fields.npz", static=static)
    model = MonsoonResNet(hidden_dim=64, num_blocks=4, dilations=[1, 2, 2, 1])
    torch.save(
        {
            "model_name": "MonsoonVerse-LocalResNet-v3",
            "model_config": {"t_in": 3, "n_dyn": 6, "n_static": 2, "n_time": 2, "n_coord": 2, "hidden_dim": 64, "num_blocks": 4, "dilations": [1, 2, 2, 1], "init_scale": 0.25},
            "state_dict": model.state_dict(),
        },
        adir / "model.pt",
    )

    pred = LocalResNetV3Predictor(artifacts_dir=adir)
    out4 = pred.predict_rollout(init_s, [0.0, 1.0], steps=4, perturbation=None)
    out10 = pred.predict_rollout(init_s, [0.0, 1.0], steps=10, perturbation=None)
    out_zeros = pred.predict_rollout(init_s, [0.0, 1.0], steps=10, perturbation=np.zeros((6, 32, 36), dtype=np.float32))

    assert out4.shape == (4, 6, 32, 36)
    assert out10.shape == (10, 6, 32, 36)
    assert np.allclose(out4, out10[:4], atol=1e-6)
    assert np.allclose(out10, out_zeros, atol=1e-6)
    assert np.isfinite(out10).all()
    assert float(out10[:, 1].min()) >= 0.0
    assert float(out10[:, 0].min()) >= 233.15 and float(out10[:, 0].max()) <= 333.15

    with pytest.raises(ValueError):
        pred.predict_rollout(init_s, [0.0, 1.0], steps=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        pred.predict_rollout(init_s, [0.0, 1.0], steps=15)


def test_regional_stress_indicators_and_anomalies():
    roll_norm = np.zeros((5, 6, 32, 36), dtype=np.float32)
    roll_norm[:, 0] = 305.15  # 32 C
    roll_norm[:, 1] = 12.0    # >= 10 mm/day on all 5 days -> persistent streak = 5
    roll_norm[:, 2] = 12.0    # wind >= 10 m/s
    roll_norm[:, 5] = 45.0

    roll_pert = roll_norm.copy()
    roll_pert[:, 1] = 20.0
    roll_pert[:, 0] = 310.15  # 37 C -> triggers >= 35 C hot day count

    rep = compute_regional_summary(roll_pert)
    assert len(rep) == 6
    for row in rep:
        assert row["persistent_heavy_streak_days"] == 5
        assert row["heavy_rain_exposure_pct"] == 100.0
        assert row["hot_day_cell_days"] == 5 * row["grid_cells"]
        assert row["strong_wind_cell_days"] == 5 * row["grid_cells"]
        assert "heat_stress_proxy_c" in row

    anom = compute_scenario_anomalies(roll_norm, roll_pert)
    assert "formulas" in anom
    assert anom["global_mean_delta_tp"] == pytest.approx(8.0)
