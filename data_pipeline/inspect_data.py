"""Inspect india_climate_train.nc and stats.json, and write analysis/data_summary.md and data_pipeline/metadata.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.load_data import CHANNELS, load_climate_dataset, load_stats, save_static_fields


def run_inspection(
    nc_path: Path | str | None = None,
    stats_path: Path | str | None = None,
) -> dict:
    ds = load_climate_dataset() if nc_path is None else load_climate_dataset(nc_path)
    stats = load_stats() if stats_path is None else load_stats(stats_path)

    data = ds["data"]
    static = ds["static"]
    times = pd.to_datetime(ds["time"])
    lat = ds["lat"]
    lon = ds["lon"]

    diffs = np.diff(times.values).astype("timedelta64[D]").astype(int)
    gaps = int((diffs != 1).sum())

    nans = int(np.isnan(data).sum())
    infs = int(np.isinf(data).sum())
    static_nans = int(np.isnan(static).sum())

    channel_summary = []
    for i, name in enumerate(CHANNELS):
        ch = data[:, i]
        channel_summary.append(
            {
                "index": i,
                "name": name,
                "min": float(ch.min()),
                "max": float(ch.max()),
                "mean": float(ch.mean()),
                "std": float(ch.std()),
                "stats_json_mean": float(stats["mean"][i]),
                "stats_json_std": float(stats["std"][i]),
            }
        )

    static_summary = [
        {
            "name": "land_sea_mask",
            "min": float(static[0].min()),
            "max": float(static[0].max()),
            "mean": float(static[0].mean()),
        },
        {
            "name": "orography_m",
            "min": float(static[1].min()),
            "max": float(static[1].max()),
            "mean": float(static[1].mean()),
            "std": float(static[1].std()),
        },
    ]

    meta = {
        "num_days": int(data.shape[0]),
        "date_start": str(times[0].date()),
        "date_end": str(times[-1].date()),
        "daily_gaps": gaps,
        "grid_shape": [int(data.shape[2]), int(data.shape[3])],
        "lat_range": [float(lat[0]), float(lat[-1])],
        "lon_range": [float(lon[0]), float(lon[-1])],
        "lat_ascending": bool(np.all(np.diff(lat) > 0)),
        "lon_ascending": bool(np.all(np.diff(lon) > 0)),
        "nan_count": nans,
        "inf_count": infs,
        "static_nan_count": static_nans,
        "channels": channel_summary,
        "static_channels": static_summary,
    }

    root = Path(__file__).resolve().parents[1]
    save_static_fields(static, root / "artifacts" / "static_fields.npz")

    meta_path = root / "data_pipeline" / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "# MonsoonVerse Dataset Inspection Summary",
        "",
        f"- **Days**: {meta['num_days']} ({meta['date_start']} to {meta['date_end']})",
        f"- **Daily gaps**: {meta['daily_gaps']}",
        f"- **Grid shape**: {meta['grid_shape'][0]} x {meta['grid_shape'][1]}",
        f"- **Latitude**: {meta['lat_range'][0]} to {meta['lat_range'][1]} (ascending={meta['lat_ascending']})",
        f"- **Longitude**: {meta['lon_range'][0]} to {meta['lon_range'][1]} (ascending={meta['lon_ascending']})",
        f"- **NaNs / Infs**: {nans} / {infs} (static NaNs: {static_nans})",
        "",
        "## Dynamic Channels",
        "",
        "| Index | Name | Min | Max | Mean | Std | stats.json Mean | stats.json Std |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in channel_summary:
        lines.append(
            f"| {c['index']} | {c['name']} | {c['min']:.4f} | {c['max']:.4f} | "
            f"{c['mean']:.4f} | {c['std']:.4f} | {c['stats_json_mean']:.4f} | {c['stats_json_std']:.4f} |"
        )
    lines += [
        "",
        "## Static Channels",
        "",
        "| Name | Min | Max | Mean |",
        "|---|---:|---:|---:|",
    ]
    for s in static_summary:
        lines.append(f"| {s['name']} | {s['min']:.4f} | {s['max']:.4f} | {s['mean']:.4f} |")

    (root / "analysis" / "data_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return meta


if __name__ == "__main__":
    summary = run_inspection()
    print(json.dumps(summary, indent=2))
