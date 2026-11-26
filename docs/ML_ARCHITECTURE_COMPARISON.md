# MONSOONVERSE — ML ARCHITECTURE COMPARISON

This document records the architectural design choices between the production incumbent (**MonsoonVerse-ClimResCNN**) and the reference-adapted challenger (**MonsoonVerse-LocalResNet-v3**).

| Dimension | Incumbent: `MonsoonVerse-ClimResCNN` | Challenger: `MonsoonVerse-LocalResNet-v3` |
|---|---|---|
| **Module Path** | [ml/model.py](file:///C:/Users/gokul/OneDrive/Documents/projects/Yugaantar-Track2/MonsoonVerse/ml/model.py) | [ml/local_resnet_v3/model.py](file:///C:/Users/gokul/OneDrive/Documents/projects/Yugaantar-Track2/MonsoonVerse/ml/local_resnet_v3/model.py) |
| **Input Channels (`32×36`)** | **28:** 3×6 dyn (18) + static (2) + DOY sin/cos (2) + future daily climatology (6) | **24:** 3×6 dyn (18) + static (2) + DOY sin/cos (2) + normalized lat/lon coord grid (2) |
| **Climatology Conditioning** | Explicit future `C(doy + k)` at every rollout step | None (learns seasonal advection purely from DOY sin/cos + spatial coordinates) |
| **Hidden Width / Depth** | `32` hidden channels · `5` `ResBlock`s | `64` hidden channels · `4` `LocalResBlock`s |
| **Dilations** | Uniform `[1, 1, 1, 1, 1]` | Multi-scale local `[1, 2, 2, 1]` |
| **Normalization** | `GroupNorm(8)` in every block | None (unnormalized spatial residual preservation) |
| **Residual Gate** | Fixed identity `+1.0` | Learnable per-channel scale (`init_scale = 0.25`) |
| **Parameter Count** | `103,014` (`424,753` B checkpoint) | **`313,030`** (matching reference spec exactly) |
| **Output Contract** | `hist_norm[:, -1] + delta` in physical-unit adapter | `hist_norm[:, -1] + delta` in physical-unit adapter (`tp >= 0`, `t2m in [233.15, 333.15]`) |
| **Objective Function** | Lead-weighted `0.6*SmoothL1 + 0.4*MSE` | Lat-weighted Huber + Heavy-Rain Boost (`>=10 mm/d`, `2.2x`) + Spatial Gradient (`0.14`) + Variance Floor (`0.08`, `min_ratio=0.65`) |
| **Training Schedule** | 7-epoch curriculum `h=1->4->8` (`80/20` gap `10d`) | Two-stage: Stage 1 (14 epochs, `h=1`, `lr=1.5e-3`) + Stage 2 (8 epochs, `h=8`, `lr=4.5e-4`, `2005–2019` / `2020–2021`) |
