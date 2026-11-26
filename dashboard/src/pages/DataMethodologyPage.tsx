import React from 'react';
import { BookOpen, Cpu, Database, Layers, ShieldAlert } from 'lucide-react';
import { EvaluationBundle } from '../types/climate';
import { VARIABLES } from '../services/api';

interface DataMethodologyPageProps {
  evaluation: EvaluationBundle | null;
}

export const DataMethodologyPage: React.FC<DataMethodologyPageProps> = ({ evaluation }) => {
  const ds = evaluation?.dataset_metadata;
  const mm = evaluation?.model_metadata;

  return (
    <div className="space-y-6 text-xs text-slate-300">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <section className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-5 space-y-2.5">
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <Database className="h-4 w-4 text-cyan-400" />
            Dataset Provenance &amp; Coverage
          </div>
          <p>
            Trained on <code className="text-cyan-300">india_climate_train.nc</code>, derived from
            Copernicus ERA5 reanalysis (Hersbach et al. 2020) via Google WeatherBench 2 (Rasp et al.
            2024).
          </p>
          <ul className="list-disc space-y-1 pl-5 text-slate-400">
            <li>
              <strong>Temporal coverage:</strong> {ds ? `${ds.date_start} to ${ds.date_end} (${ds.num_days} days, ${ds.daily_gaps} gaps)` : '2005-01-01 to 2021-12-31 (6,209 days)'}
            </li>
            <li>
              <strong>Spatial grid:</strong> 32 × 36 cells of 1°×1°, centres 6.5°–37.5°N (S→N) and
              65.5°–100.5°E (W→E).
            </li>
            <li>
              <strong>Daily aggregation:</strong> <code>t2m, u10, v10, msl, tcwv</code> are daily
              means of 00/06/12/18 UTC; <code>tp</code> is 24-hour accumulated precipitation.
            </li>
            <li>
              <strong>Static fields:</strong> <code>land_sea_mask</code> and <code>orography_m</code>{' '}
              bundled in <code>artifacts/static_fields.npz</code>.
            </li>
          </ul>
        </section>

        <section className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-5 space-y-2.5">
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <Cpu className="h-4 w-4 text-emerald-400" />
            Model Architecture &amp; Training
          </div>
          <p>
            <strong className="text-white">{mm?.model_name ?? 'MonsoonVerse-ClimResCNN'}</strong> is
            a compact CPU-efficient 2D Residual CNN (32 hidden channels, 5 replicate-padded residual
            blocks, GroupNorm, SiLU).
          </p>
          <ul className="list-disc space-y-1 pl-5 text-slate-400">
            <li>
              <strong>28 Input Channels:</strong> 3 historical days × 6 dynamic fields (18) + 2
              static fields (2) + broadcast day-of-year sin/cos (2) + 6 training-only daily
              climatology channels (6).
            </li>
            <li>
              <strong>Residual Output:</strong> Predicts normalised delta added to latest state;
              enforces physical <code>tp ≥ 0</code> between autoregressive steps.
            </li>
            <li>
              <strong>Chronological Split:</strong> 80% training (2005–2018-08-07), 10-day
              anti-leakage gap, 20% validation (2018-08-18–2021-12-31).
            </li>
            <li>
              <strong>Curriculum Rollout:</strong> 7 epochs (h=1→4→8) with latitude-, lead-, and
              precipitation-weighted (1.35×) SmoothL1 + MSE loss.
            </li>
          </ul>
        </section>
      </div>

      {/* Six Channels Table */}
      <section className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-5">
        <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-white">
          <Layers className="h-4 w-4 text-cyan-400" />
          Six Dynamic Atmospheric Channels (Exact Contract Order)
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2 pr-3">Index</th>
                <th className="py-2 px-3">Variable</th>
                <th className="py-2 px-3">Physical Name</th>
                <th className="py-2 px-3">Unit</th>
                <th className="py-2 pl-3">Physical Meaning</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {VARIABLES.map((v) => (
                <tr key={v.key}>
                  <td className="py-2 pr-3 font-mono text-slate-400">{v.index}</td>
                  <td className="py-2 px-3 font-mono font-semibold text-cyan-300">{v.key}</td>
                  <td className="py-2 px-3 text-slate-100">{v.label}</td>
                  <td className="py-2 px-3 font-mono text-amber-300">{v.unit}</td>
                  <td className="py-2 pl-3 text-slate-400">{v.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Perturbation Math & Limitations */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <section className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-5 space-y-2">
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <BookOpen className="h-4 w-4 text-amber-400" />
            Perturbation Mathematics &amp; Baselines
          </div>
          <p className="text-slate-400">
            Perturbations Δx in physical units are added exclusively to the most recent input state{' '}
            <code>init_states[-1]</code> before z-score normalisation. Official scenarios:
          </p>
          <ul className="list-disc space-y-1 pl-5 text-slate-400">
            <li>
              <strong>Warm Bay of Bengal:</strong> +3.0 K <code>t2m</code>, +15% current{' '}
              <code>tcwv</code> over 8°–20°N, 82°–94°E.
            </li>
            <li>
              <strong>Arabian Sea Monsoon Stall:</strong> −40% <code>u10</code> and{' '}
              <code>v10</code> over 8°–20°N, 65°–75°E.
            </li>
            <li>
              <strong>Baselines:</strong> Persistence (repeat <code>init_states[-1]</code>) and
              15-day smoothed 366-day training climatology.
            </li>
          </ul>
        </section>

        <section className="rounded-xl border border-amber-500/30 bg-amber-950/15 p-5 space-y-2">
          <div className="flex items-center gap-2 text-sm font-semibold text-amber-200">
            <ShieldAlert className="h-4 w-4 text-amber-400" />
            Scientific Limitations &amp; Attribution
          </div>
          <ul className="list-disc space-y-1 pl-5 text-amber-100/80">
            <li>1°×1° (~111 km) spatial resolution smooths mesoscale convective extremes.</li>
            <li>
              Day-8 <code>t2m</code> (-19.1%) and <code>tcwv</code> (-2.8%) trail climatology due
              to synoptic phase-error double penalty.
            </li>
            <li>
              <code>monsoon_stall</code> exhibits dissipative relaxation rather than sustained
              outward energy growth above the 2% threshold.
            </li>
            <li>
              Code licensed under MIT; ERA5 data licensed under CC BY 4.0 (Copernicus / WeatherBench
              2 attribution in <code>NOTICE</code>).
            </li>
          </ul>
        </section>
      </div>
    </div>
  );
};
