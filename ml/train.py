"""Training script for MonsoonVerse Residual CNN with multi-step autoregressive rollout loss."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline.dataset import load_train_val_datasets
from ml.loss import ClimateRolloutLoss
from ml.model import MonsoonResCNN


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def rollout_train_step(
    model: MonsoonResCNN,
    hist: torch.Tensor,
    static: torch.Tensor,
    doy_seq: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Differentiable H-step autoregressive rollout in normalised space."""
    cur_hist = hist
    preds = []
    for k in range(horizon):
        pred_k = model(cur_hist, static, doy_seq[:, k])  # [B, 6, 32, 36]
        preds.append(pred_k)
        cur_hist = torch.cat([cur_hist[:, 1:], pred_k.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)  # [B, H, 6, 32, 36]


@torch.no_grad()
def evaluate_val_loss(
    model: MonsoonResCNN,
    loader: DataLoader,
    criterion: ClimateRolloutLoss,
    horizon: int,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        pred = rollout_train_step(
            model,
            batch["hist"],
            batch["static"],
            batch["doy_seq"],
            horizon=horizon,
        )
        loss = criterion(pred, batch["target"])
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def train() -> dict:
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "ml" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))

    horizon = int(cfg["rollout_horizon"])
    train_ds, val_ds, info = load_train_val_datasets(
        t_in=int(cfg["t_in"]),
        horizon=horizon,
        train_frac=float(cfg["train_frac"]),
        gap_days=int(cfg["gap_days"]),
    )

    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, drop_last=False)

    model = MonsoonResCNN(
        hidden_channels=int(cfg["hidden_channels"]),
        num_blocks=int(cfg["num_blocks"]),
    )
    criterion = ClimateRolloutLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]), eta_min=1e-5)

    best_val = float("inf")
    best_epoch = -1
    history = []
    t_start = time.perf_counter()

    ckpt_path = root / "artifacts" / "model.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        tr_loss = 0.0
        nb = 0
        # Curriculum: 1-step for epoch 1, 2-step for epoch 2, full horizon afterwards
        cur_h = 1 if epoch == 1 else (2 if epoch == 2 else horizon)

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = rollout_train_step(
                model,
                batch["hist"],
                batch["static"],
                batch["doy_seq"][:, :cur_h],
                horizon=cur_h,
            )
            loss = criterion(pred, batch["target"][:, :cur_h])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += float(loss.item())
            nb += 1

        scheduler.step()
        tr_loss /= max(nb, 1)
        val_loss = evaluate_val_loss(model, val_loader, criterion, horizon=horizon)

        history.append({"epoch": epoch, "horizon": cur_h, "train_loss": tr_loss, "val_loss": val_loss})
        print(
            f"Epoch {epoch:02d}/{cfg['epochs']} (h={cur_h}) | "
            f"train_loss={tr_loss:.5f} | val_loss={val_loss:.5f} | "
            f"elapsed={time.perf_counter() - t_start:.1f}s",
            flush=True,
        )

        # Only select best checkpoint once full rollout horizon is active
        if cur_h == horizon and val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )

    total_time = time.perf_counter() - t_start
    n_params = sum(p.numel() for p in model.parameters())
    meta = {
        "model_name": cfg["model_name"],
        "num_parameters": int(n_params),
        "checkpoint_size_bytes": int(ckpt_path.stat().st_size),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "training_time_s": round(total_time, 2),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "config": cfg,
        "history": history,
    }
    (root / "artifacts" / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Training complete in {total_time:.1f}s. Best epoch={best_epoch}, val_loss={best_val:.5f}")
    return meta


if __name__ == "__main__":
    train()
