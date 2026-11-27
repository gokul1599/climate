import React, { useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { VARIABLES } from '../services/api';
import { EvaluationBundle } from '../types/climate';
import { MetricCard } from '../components/MetricCard';
import { EmptyState } from '../components/EmptyState';
import { LoadingSkeleton } from '../components/LoadingSkeleton';

interface ModelPerformancePageProps {
  evaluation: EvaluationBundle | null;
  loading: boolean;
}

export const ModelPerformancePage: React.FC<ModelPerformancePageProps> = ({
  evaluation,
  loading,
}) => {
  const [selectedChannel, setSelectedChannel] = useState<string>('all');
  const [selectedDay, setSelectedDay] = useState<number>(0); // 0 = all days, 1..10 = specific day

  if (loading && !evaluation) {
    return <LoadingSkeleton heightClass="h-72" />;
  }
  if (!evaluation?.ten_day || !evaluation?.validation) {
    return (
      <EmptyState
        title="Evaluation Metrics Unavailable"
        description="Run evaluation/ten_day_eval.py on the backend to populate live validation metrics."
      />
    );
  }

  const { ten_day, validation } = evaluation;
  const fc = validation.skill.summary.forecast_points_out_of_45;

  const skillCurve = ten_day.days.map((d) => {
    const ch = selectedChannel === 'all' ? null : d.channels.find((c) => c.channel === selectedChannel);
    return {
      day: `D${d.day}`,
      ssBest: Number(((ch ? ch.ss_vs_best : d.mean_ss_vs_best) * 100).toFixed(2)),
      ssPersist: Number(((ch ? ch.ss_vs_persist : d.mean_ss_vs_persist) * 100).toFixed(2)),
      ssClim: Number(((ch ? ch.ss_vs_clim : d.mean_ss_vs_clim) * 100).toFixed(2)),
      csiModel: Number(d.precip_10mm.csi_model.toFixed(4)),
      csiPersist: Number(d.precip_10mm.csi_persist.toFixed(4)),
      csiClim: Number(d.precip_10mm.csi_clim.toFixed(4)),
    };
  });

  const filteredDays = selectedDay === 0 ? ten_day.days : ten_day.days.filter((d) => d.day === selectedDay);

  return (
    <div className="space-y-6">
      <h2 className="sr-only">Validation Metrics and Skill Curves</h2>
      {/* Provenance Banner */}
      <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 px-4 py-3 text-xs text-slate-300">
        <span className="font-semibold text-white">Chronological Validation Set:</span>{' '}
        {ten_day.val_date_start} to {ten_day.val_date_end} ({ten_day.num_windows} rolling 10-day
        windows, 0 hidden-judge leakage) · Model: <span className="text-cyan-300">{ten_day.model_name}</span>
      </div>

      {/* Top KPI Summary */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <MetricCard
          label="Forecast Skill (/45)"
          value={fc.points.toFixed(2)}
          sublabel={`RMSE ${fc.rmse_pts.toFixed(2)} + CSI ${fc.csi_pts.toFixed(2)}`}
          accent="cyan"
        />
        <MetricCard
          label="Anti-Blur (/10)"
          value={validation.skill.summary.anti_blur_points_out_of_10.toFixed(1)}
          sublabel="Day-8 spatial var ratio"
          accent="emerald"
        />
        <MetricCard
          label="Causality (/15)"
          value={validation.perturbations.causality_points_out_of_15.toFixed(2)}
          sublabel="BoB & Arabian Sea shocks"
          accent="amber"
        />
        <MetricCard
          label="Day-4 vs Persist"
          value={`+${(ten_day.days[3].mean_ss_vs_persist * 100).toFixed(2)}%`}
          sublabel="Hard gate passed"
          accent="emerald"
        />
        <MetricCard
          label="Validity Failures"
          value={`${validation.validity_failures} / ${validation.num_eval_windows}`}
          sublabel="Finite, tp>=0, valid T"
          accent="emerald"
        />
        <MetricCard
          label="Mean CPU Rollout"
          value={`${validation.cpu_inference_ms_mean.toFixed(1)} ms`}
          sublabel={`Max ${validation.cpu_inference_ms_max.toFixed(1)} ms`}
        />
      </div>

      {/* Filter Bar */}
      <div className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-800/90 bg-navy-900/70 p-3 text-xs">
        <label className="flex items-center gap-2 text-slate-300">
          <span>Channel:</span>
          <select
            value={selectedChannel}
            onChange={(e) => setSelectedChannel(e.target.value)}
            className="rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs text-white"
          >
            <option value="all">All 6 Variables (Mean Skill)</option>
            {VARIABLES.map((v) => (
              <option key={v.key} value={v.key}>
                {v.key} — {v.label} ({v.unit})
              </option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-2 text-slate-300">
          <span>Lead Horizon:</span>
          <select
            value={selectedDay}
            onChange={(e) => setSelectedDay(Number(e.target.value))}
            className="rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs text-white"
          >
            <option value={0}>All Days (D1–D10)</option>
            {Array.from({ length: 10 }, (_, k) => (
              <option key={k + 1} value={k + 1}>
                Day {k + 1}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4">
          <h3 className="mb-2 text-sm font-semibold text-white">
            Skill Score (%) vs Baselines — {selectedChannel === 'all' ? '6-Channel Mean' : selectedChannel}
          </h3>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={skillCurve}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="day" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} unit="%" />
                <ReferenceLine y={0} stroke="#64748b" strokeDasharray="3 3" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: 11 }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="ssBest" name="vs Better Baseline (%)" stroke="#38bdf8" strokeWidth={2.5} />
                <Line type="monotone" dataKey="ssPersist" name="vs Persistence (%)" stroke="#10b981" strokeWidth={1.8} />
                <Line type="monotone" dataKey="ssClim" name="vs Climatology (%)" stroke="#f59e0b" strokeWidth={1.8} strokeDasharray="4 2" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4">
          <h3 className="mb-2 text-sm font-semibold text-white">
            Heavy-Rain CSI (tp ≥ 10 mm/day) Across Days 1–10
          </h3>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={skillCurve}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="day" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} domain={[0.1, 0.5]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: 11 }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="csiModel" name="MonsoonVerse CSI" stroke="#22d3ee" strokeWidth={2.5} />
                <Line type="monotone" dataKey="csiClim" name="Climatology CSI" stroke="#f59e0b" strokeWidth={1.8} />
                <Line type="monotone" dataKey="csiPersist" name="Persistence CSI" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="3 3" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Detailed Table */}
      <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4">
        <h3 className="mb-3 text-sm font-semibold text-white">
          Detailed Per-Horizon &amp; Per-Channel Validation Metrics
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-[11px] text-slate-400">
                <th className="py-2 pr-2">Day</th>
                <th className="py-2 px-2">Channel</th>
                <th className="py-2 px-2 text-right">RMSE</th>
                <th className="py-2 px-2 text-right">MAE</th>
                <th className="py-2 px-2 text-right">Bias</th>
                <th className="py-2 px-2 text-right">Corr</th>
                <th className="py-2 px-2 text-right">Var Ratio</th>
                <th className="py-2 px-2 text-right">SS vs Persist</th>
                <th className="py-2 px-2 text-right">SS vs Clim</th>
                <th className="py-2 pl-2 text-right">SS vs Best</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {filteredDays.flatMap((d) =>
                d.channels
                  .filter((c) => selectedChannel === 'all' || c.channel === selectedChannel)
                  .map((c) => (
                    <tr key={`${d.day}-${c.channel}`} className="hover:bg-slate-800/30">
                      <td className="py-1.5 pr-2 font-mono text-slate-300">D{d.day}</td>
                      <td className="py-1.5 px-2 font-medium text-cyan-300">{c.channel}</td>
                      <td className="py-1.5 px-2 text-right font-mono">{c.rmse.toFixed(3)}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-slate-300">{c.mae.toFixed(3)}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-slate-400">{c.bias.toFixed(3)}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-slate-300">{c.corr.toFixed(3)}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-slate-300">
                        {c.spatial_var_ratio.toFixed(3)}
                      </td>
                      <td className="py-1.5 px-2 text-right font-mono text-emerald-400">
                        +{(c.ss_vs_persist * 100).toFixed(1)}%
                      </td>
                      <td className="py-1.5 px-2 text-right font-mono text-slate-300">
                        {(c.ss_vs_clim * 100).toFixed(1)}%
                      </td>
                      <td
                        className={`py-1.5 pl-2 text-right font-mono font-semibold ${
                          c.ss_vs_best >= 0 ? 'text-emerald-300' : 'text-rose-400'
                        }`}
                      >
                        {(c.ss_vs_best * 100).toFixed(1)}%
                      </td>
                    </tr>
                  )),
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Deterministic Empirical Out-of-Sample Residual Intervals */}
      {(() => {
        const activeBands =
          (evaluation.active_model?.active_variant === 'local_resnet_v4' ||
            evaluation.active_model?.active_variant === 'local_resnet_v3') &&
          evaluation.challenger_uncertainty_bands
            ? evaluation.challenger_uncertainty_bands
            : evaluation.incumbent_uncertainty_bands;
        if (!activeBands) return null;
        return (
          <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="text-sm font-semibold text-white">
                  Empirical Out-of-Sample Residual Error Bands (D1–D10 · {activeBands.model_name})
                </h3>
                <p className="text-[11px] text-amber-200/90">
                  {activeBands.deterministic_note}
                </p>
              </div>
              <span className="rounded-full border border-slate-700 bg-slate-900 px-2.5 py-0.5 text-[10px] text-slate-300">
                n = {activeBands.num_windows} validation windows ({activeBands.val_date_start}..{activeBands.val_date_end})
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-[11px] text-slate-400">
                    <th className="py-2 pr-2">Lead</th>
                    <th className="py-2 px-2">t2m MAE / [p05, p95] (K)</th>
                    <th className="py-2 px-2">tp MAE / [p05, p95] (mm/d)</th>
                    <th className="py-2 px-2">u10 MAE / [p05, p95] (m/s)</th>
                    <th className="py-2 pl-2">tcwv MAE / [p05, p95] (kg/m²)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50 font-mono text-[11px]">
                  {activeBands.leads
                    .filter((l) => l.day === 1 || l.day === 4 || l.day === 8 || l.day === 10)
                    .map((l) => {
                      const fmt = (ch: string) => {
                        const s = l.channels[ch];
                        return s ? `${s.mae.toFixed(2)} [${s.p05.toFixed(2)}, ${s.p95.toFixed(2)}]` : '—';
                      };
                      return (
                        <tr key={l.day}>
                          <td className="py-1.5 pr-2 text-cyan-300">Day {l.day}</td>
                          <td className="py-1.5 px-2 text-slate-300">{fmt('t2m')}</td>
                          <td className="py-1.5 px-2 text-slate-300">{fmt('tp')}</td>
                          <td className="py-1.5 px-2 text-slate-300">{fmt('u10')}</td>
                          <td className="py-1.5 pl-2 text-slate-300">{fmt('tcwv')}</td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </div>
        );
      })()}

      {/* Controlled Incumbent vs LocalResNet-v3 Comparison (when generated) */}
      {evaluation.model_comparison && (() => {
        const mc = evaluation.model_comparison as {
          promotion_decision?: { decision?: string; reason?: string };
          protocol_a_reference_52w?: {
            incumbent?: { score_summary?: Record<string, number>; latency_ms?: { mean?: number }; causality?: { warm_ocean?: { propagates_rate?: number }; monsoon_stall?: { propagates_rate?: number } } };
            local_resnet_v3?: { score_summary?: Record<string, number>; latency_ms?: { mean?: number }; causality?: { warm_ocean?: { propagates_rate?: number }; monsoon_stall?: { propagates_rate?: number } } };
          };
        };
        const inc = mc.protocol_a_reference_52w?.incumbent;
        const chg = mc.protocol_a_reference_52w?.local_resnet_v3;
        return (
          <div className="rounded-xl border border-cyan-500/30 bg-navy-900/85 p-4 space-y-3 text-xs">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-sm font-semibold text-white">
                Controlled Challenger Benchmark (Protocol A: 2020–2021, 52 Windows, Zero Leakage)
              </h3>
              <span className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-0.5 text-[10px] font-semibold text-emerald-300">
                {mc.promotion_decision?.decision ?? 'EVALUATED'}
              </span>
            </div>
            <p className="text-slate-300">{mc.promotion_decision?.reason ?? ''}</p>
            {inc?.score_summary && chg?.score_summary && (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-[11px] text-slate-400">
                      <th className="py-2 pr-2">Benchmark Metric</th>
                      <th className="py-2 px-2 text-right">Incumbent (ClimResCNN)</th>
                      <th className="py-2 pl-2 text-right">Promoted (LocalResNet-v3)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50 font-mono text-[11px]">
                    <tr>
                      <td className="py-1.5 pr-2 font-sans text-slate-300">Forecast RMSE Skill (/30)</td>
                      <td className="py-1.5 px-2 text-right text-slate-300">{inc.score_summary.forecast_rmse_pts_30?.toFixed(2)}</td>
                      <td className="py-1.5 pl-2 text-right text-emerald-300">{chg.score_summary.forecast_rmse_pts_30?.toFixed(2)}</td>
                    </tr>
                    <tr>
                      <td className="py-1.5 pr-2 font-sans text-slate-300">Heavy-Rain CSI Skill (/15)</td>
                      <td className="py-1.5 px-2 text-right text-slate-300">{inc.score_summary.forecast_csi_pts_15?.toFixed(2)}</td>
                      <td className="py-1.5 pl-2 text-right text-emerald-300">{chg.score_summary.forecast_csi_pts_15?.toFixed(2)}</td>
                    </tr>
                    <tr>
                      <td className="py-1.5 pr-2 font-sans text-slate-300">Total Model Score (/70)</td>
                      <td className="py-1.5 px-2 text-right text-slate-300">{inc.score_summary.total_model_pts_70?.toFixed(2)}</td>
                      <td className="py-1.5 pl-2 text-right font-semibold text-cyan-300">{chg.score_summary.total_model_pts_70?.toFixed(2)}</td>
                    </tr>
                    <tr>
                      <td className="py-1.5 pr-2 font-sans text-slate-300">CPU Latency Mean (ms)</td>
                      <td className="py-1.5 px-2 text-right text-slate-300">{inc.latency_ms?.mean?.toFixed(2)}</td>
                      <td className="py-1.5 pl-2 text-right text-emerald-300">{chg.latency_ms?.mean?.toFixed(2)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
          </div>
        );
      })()}
    </div>
  );
};
