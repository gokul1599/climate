"""Generate the exact two-page technical brief PDF (brief.pdf) using ReportLab."""
from __future__ import annotations

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_brief(out_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle("T", parent=styles["Title"], fontSize=15, leading=18, textColor=colors.HexColor("#0f172a"), spaceAfter=2)
    sub_s = ParagraphStyle("S", parent=styles["Normal"], fontSize=9, leading=11, textColor=colors.HexColor("#475569"), spaceAfter=6)
    h2_s = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=10.5, leading=13, textColor=colors.HexColor("#0369a1"), spaceBefore=5, spaceAfter=2)
    body_s = ParagraphStyle("B", parent=styles["Normal"], fontSize=8.5, leading=11.3, textColor=colors.HexColor("#1e293b"), spaceAfter=3)

    # ---------------- PAGE 1: MODEL DESIGN & SCIENTIFIC VALIDATION ----------------
    p1 = [
        Paragraph("MONSOONVERSE — Multiverse of Monsoon", title_s),
        Paragraph("Page 1: Model Design, Training & Scientific Validation · Yugaantar DevForge Track 2", sub_s),
        Paragraph("1. Problem Statement, Dataset & Preprocessing", h2_s),
        Paragraph(
            "India's monsoon exhibits tightly coupled temperature, wind, pressure, moisture, and precipitation dynamics. "
            "MonsoonVerse is an offline, CPU-efficient climate digital twin trained on <i>india_climate_train.nc</i> "
            "(2005–2021, 6,209 daily steps, 0 missing days/NaNs) on a 32×36 1°×1° ERA5-derived grid (6.5°–37.5°N, 65.5°–100.5°E). "
            "Dynamic channels in fixed order are <i>t2m</i> (K), <i>tp</i> (mm/day), <i>u10</i> (m/s), <i>v10</i> (m/s), "
            "<i>msl</i> (Pa), and <i>tcwv</i> (kg/m²), paired with static <i>land_sea_mask</i> and <i>orography_m</i>. "
            "We enforce a strict chronological split (2005-01-01 to 2018-08-07 train; 10-day gap; 2018-08-18 to 2021-12-31 val) "
            "and z-score normalise internally via <i>stats.json</i> while keeping all API boundaries in physical units.",
            body_s,
        ),
        Paragraph("2. Model Architecture, Conditioning & Training Strategy", h2_s),
        Paragraph(
            "<i>MonsoonResCNN</i> is a compact 2D Residual CNN (32 hidden channels, 5 residual blocks, 3×3 replicate-padded "
            "convolutions, GroupNorm, SiLU) ingesting 28 channels [B, 28, 32, 36]: 3 historical days × 6 dynamic fields (18), "
            "2 static fields (2), broadcast day-of-year sin/cos (2), and 6 training-only daily climatology channels (<i>clim_norm</i>) "
            "for the target day. It predicts the normalised residual <i>next_state = latest_state + delta</i>. "
            "Training uses a 7-epoch curriculum multi-step rollout (h=1→4→8) with latitude-, channel- (1.35× <i>tp</i>), and "
            "lead-weighted SmoothL1+MSE loss. During 10-day autoregressive rollout, <i>doy_sincos</i> rotates +1 day/step, "
            "inference is deterministic (eval mode), and <i>tp ≥ 0</i> is enforced in physical space.",
            body_s,
        ),
        Paragraph("3. Quantitative Validation, Baselines, Drift & Blurring", h2_s),
    ]

    t1 = Table(
        [
            ["Metric (82 Val Windows, 2018–2021)", "Baseline (Ep 4)", "Accepted (ClimResCNN)", "Persistence / Clim", "Status"],
            ["Day-1 Mean SS vs Best / CSI", "+0.1583 / 0.4339", "+0.1564 / 0.4374", "0.3841 / 0.2576", "Beats both"],
            ["Day-4 Mean SS vs Best / CSI", "-0.0184 / 0.3048", "+0.0441 / 0.3023", "0.1897 / 0.2702", "Positive (+4.41%)"],
            ["Day-8 Mean SS vs Best / CSI", "-0.1461 / 0.3207", "-0.0292 / 0.2835", "0.1896 / 0.2861", "+11.7% gain vs Ep4"],
            ["Day-4 Mean SS vs Persistence", "+0.1856", "+0.2363", "0.0000", "Passes hard gate"],
            ["Day-8 Spatial Variance (Anti-Blur)", "10.0 / 10.0 pts", "10.0 / 10.0 pts", "—", "Within [0.40, 2.50]"],
            ["CPU 10-Day Latency / Validity", "69.7 ms / 82/82", "73.5 ms / 82/82", "—", "0 NaNs, < 5,000 ms"],
        ],
        colWidths=[54 * mm, 30 * mm, 35 * mm, 32 * mm, 29 * mm],
    )
    t1.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7.6),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("PADDING", (0, 0), (-1, -1), 3.2),
            ]
        )
    )
    p1.append(t1)
    p1.append(Spacer(1, 3 * mm))
    p1.extend(
        [
            Paragraph("4. Drift, Failure Modes & Scientific Limitations", h2_s),
            Paragraph(
                "Conditioning on training climatology prevents multi-day drift (0/82 validity failures) and preserves Day-8 spatial "
                "variance (10/10 anti-blur). However, (1) 1°×1° (~111 km) resolution smooths mesoscale convective extremes and steep "
                "orographic rainfall; (2) <i>t2m</i> (-19.1%) and <i>tcwv</i> (-2.8%) still trail climatology at Day 8; "
                "(3) hidden post-2023-01-10 judge windows cannot be tested locally.",
                body_s,
            ),
            PageBreak(),
        ]
    )

    # ---------------- PAGE 2: VISUALIZATION, PERTURBATIONS & IMPACT ----------------
    p2 = [
        Paragraph("MONSOONVERSE — Visualization, Perturbations & Impact", title_s),
        Paragraph("Page 2: Scenario Simulation, Impact Explorer & Offline Reproducibility", sub_s),
        Paragraph("5. Perturbation Simulation & Spatial Propagation", h2_s),
        Paragraph(
            "Physical-unit perturbations are added exclusively to <i>init_states[-1]</i> prior to normalisation. On the official "
            "monsoon validation test: (a) <i>warm_ocean</i> (+3.0 K <i>t2m</i>, +15% <i>tcwv</i> over Bay of Bengal 8°–20°N, 82°–94°E) "
            "achieves 99.85% Day-1 localisation, propagates outward over Days 4–8 without teleportation (far-field 0.00054), "
            "and remains stable; (b) <i>monsoon_stall</i> (-40% <i>u10/v10</i> over Arabian Sea 8°–20°N, 65°–75°E) achieves 99.93% "
            "Day-1 localisation and stability, though its outward Days 4–8 growth stayed below the official propagation threshold "
            "(13.13/15 causality pts); (c) <i>null/zeros</i> yields 0.00 max difference (exact determinism).",
            body_s,
        ),
        Paragraph("6. Interactive Dashboard & Normal-vs-Shock Comparison", h2_s),
        Paragraph(
            "The React + TypeScript + Tailwind + Recharts dashboard connects live to <i>POST /health</i>, the simulation API, "
            "and <i>/api/sample-windows</i>. Users select a validation window, climate variable (all 6 channels), forecast day (D1–D10), "
            "and scenario (Normal, Warm Bay of Bengal, Arabian Sea Stall, Zero Check). The UI renders side-by-side Normal, Perturbed, "
            "and diverging Difference (<i>Perturbed − Normal</i>) 32×36 India grid maps (south-to-north) alongside 10-day domain-mean "
            "and peak-rainfall evolution charts.",
            body_s,
        ),
        Paragraph("7. Regional Impact Explorer & Defensible Use Case", h2_s),
        Paragraph(
            "The Impact Explorer (<i>impact/regional_analysis.py</i>) aggregates 10-day rollouts across six Indian regions "
            "(Western Ghats &amp; Konkan, Central India Monsoon Core, Indo-Gangetic Plain, Northeast India, Bay of Bengal, Arabian Sea) "
            "to compute regional mean rainfall, rainfall anomalies, heavy-rain cell-days (≥10 mm/day), ΔT, and ΔTCWV. "
            "<b>Use Case — Agricultural &amp; Water-Resource Screening:</b> Comparing Normal vs. Arabian Sea Stall scenarios highlights "
            "simulated moisture-transport reductions and rainfall deficits over the Western Ghats and Central India core. "
            "<b>Caveat:</b> All outputs are exploratory model-derived indicators, not operational flood, crop-loss, or disaster warnings.",
            body_s,
        ),
        Paragraph("8. Offline Reproducibility & Repository Integrity", h2_s),
        Paragraph(
            "Root <i>serve.py</i> listens on <i>0.0.0.0:8000</i> with zero external network calls or data downloads at inference. "
            "All inference assets (<i>model.pt</i>, <i>stats.json</i>, <i>static_fields.npz</i>, <i>climatology.npz</i>) are "
            "bundled in-repo. Official <i>monsoon_r1.repo_scan</i> scores <b>15.0/15.0</b> (0 flags); "
            "<i>monsoon_r1.schema_validator</i> passes all 6 checks (2.1s–2.3s wall time, 1.17 MB JSON); "
            "<i>pytest</i> passes 8/8 tests. Repository: <i>https://github.com/gokul1599/climate</i> (MIT / CC BY 4.0 ERA5 attribution).",
            body_s,
        ),
    ]

    doc.build(p1 + p2)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = root / "brief.pdf"
    build_brief(out)
    print(f"Generated {out} ({out.stat().st_size} bytes)")
