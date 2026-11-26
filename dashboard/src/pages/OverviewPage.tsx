import React, { useState } from 'react';
import {
  Activity,
  Clock,
  Database,
  FlaskConical,
  Grid,
  Layers,
  Play,
} from 'lucide-react';
import {
  Line,
  LineChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  computeDomainDaySeries,
  SCENARIOS,
  VARIABLES,
} from '../services/api';
import {
  NavSection,
  RegionalIndicator,
  SavedSimulationRun,
  ScenarioKey,
  VariableIndex,
} from '../types/climate';
import { ForecastMap } from '../components/ForecastMap';
import { MetricCard } from '../components/MetricCard';
import { EmptyState } from '../components/EmptyState';
import { LoadingSkeleton } from '../components/LoadingSkeleton';

interface OverviewPageProps {
  normalRollout: number[][][][] | null;
  perturbedRollout: number[][][][] | null;
  dayIdx: number;
  onChangeDay: (d: number) => void;
  varIdx: VariableIndex;
  onChangeVar: (v: VariableIndex) => void;
  scenario: ScenarioKey;
  onChangeScenario: (s: ScenarioKey) => void;
  inferenceMs: number | null;
  backendHealthy: boolean | null;
  savedTotal: number;
  recentRuns: SavedSimulationRun[];
  indicators: RegionalIndicator[];
  loading: boolean;
  showGeoOverlay: boolean;
  showGridLines: boolean;
  onRunNew: () => void;
  onNavigate: (sec: NavSection) => void;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({
  normalRollout,
  perturbedRollout,
  dayIdx,
  onChangeDay,
  varIdx,
  onChangeVar,
  scenario,
  onChangeScenario,
  inferenceMs,
  backendHealthy,
  savedTotal,
  recentRuns,
  indicators,
  loading,
  showGeoOverlay,
  showGridLines,
  onRunNew,
  onNavigate,
}) => {
  const [mapMode, setMapMode] = useState<'normal' | 'perturbed' | 'difference'>('normal');
  const variable = VARIABLES[varIdx];
  const scenarioMeta = SCENARIOS.find((s) => s.key === scenario) ?? SCENARIOS[0];

  const activeMode =
    mapMode !== 'normal' && !perturbedRollout ? 'normal' : mapMode;

  const activeGrid = normalRollout
    ? activeMode === 'normal'
      ? normalRollout[dayIdx][varIdx]
      : activeMode === 'perturbed' && perturbedRollout
      ? perturbedRollout[dayIdx][varIdx]
      : normalRollout[dayIdx][varIdx].map((row, i) =>
          row.map((v, j) => (perturbedRollout ? perturbedRollout[dayIdx][varIdx][i][j] - v : 0)),
        )
    : null;

  const trendSeries = normalRollout
    ? computeDomainDaySeries(normalRollout, perturbedRollout, varIdx)
    : [];

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <section className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-800/90 bg-gradient-to-r from-[#0b1426] to-[#0f1d36] p-5">
        <div>
          <h2 className="text-lg font-bold tracking-tight text-white sm:text-xl">
            Climate Intelligence Overview
          </h2>
          <p className="mt-1 max-w-2xl text-xs text-slate-300 sm:text-sm">
            Explore 10-day Indian regional weather forecasts and controlled atmospheric perturbation
            experiments powered by <span className="text-cyan-300 font-medium">MonsoonVerse-ClimResCNN</span>.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={onRunNew}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400 disabled:opacity-50"
          >
            <Play className="h-3.5 w-3.5" />
            Run New Simulation
          </button>
          <button
            type="button"
            onClick={() => onNavigate('scenarios')}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900/80 px-4 py-2 text-xs font-medium text-slate-200 hover:bg-slate-800"
          >
            <FlaskConical className="h-3.5 w-3.5 text-amber-400" />
            Explore Scenarios
          </button>
        </div>
      </section>

      {/* 6 KPI Cards */}
      <section className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <MetricCard
          label="Forecast Horizon"
          value="10 Days"
          sublabel="Autoregressive rollout"
          icon={<Clock className="h-4 w-4" />}
        />
        <MetricCard
          label="Model Variables"
          value="6 Channels"
          sublabel="t2m, tp, u10, v10, msl, tcwv"
          icon={<Layers className="h-4 w-4" />}
        />
        <MetricCard
          label="Spatial Grid"
          value="32 × 36"
          sublabel="1°×1° (6.5°–37.5°N)"
          icon={<Grid className="h-4 w-4" />}
        />
        <MetricCard
          label="CPU Inference"
          value={inferenceMs !== null ? `${inferenceMs.toFixed(1)} ms` : '—'}
          sublabel="Last live /simulate call"
          accent="emerald"
          icon={<Activity className="h-4 w-4" />}
        />
        <MetricCard
          label="Saved Runs"
          value={String(savedTotal)}
          sublabel="Persisted in Neon DB"
          accent="amber"
          icon={<Database className="h-4 w-4" />}
        />
        <MetricCard
          label="Model Readiness"
          value={backendHealthy ? 'Ready' : backendHealthy === false ? 'Offline' : 'Checking'}
          sublabel="FastAPI CPU runtime"
          accent={backendHealthy ? 'emerald' : 'rose'}
          icon={<Activity className="h-4 w-4" />}
        />
      </section>

      {/* Main Map + Quick Controls */}
      {loading && !normalRollout ? (
        <LoadingSkeleton heightClass="h-80" />
      ) : activeGrid ? (
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
          <div className="space-y-3 xl:col-span-2">
            <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-800 bg-navy-900/70 p-3 text-xs">
              <div className="flex flex-wrap items-center gap-2">
                <select
                  value={varIdx}
                  onChange={(e) => onChangeVar(Number(e.target.value) as VariableIndex)}
                  aria-label="Overview climate variable"
                  className="rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-100"
                >
                  {VARIABLES.map((v) => (
                    <option key={v.key} value={v.index}>
                      {v.label} ({v.unit})
                    </option>
                  ))}
                </select>

                <select
                  value={dayIdx}
                  onChange={(e) => onChangeDay(Number(e.target.value))}
                  aria-label="Overview forecast day"
                  className="rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-100"
                >
                  {Array.from({ length: 10 }, (_, d) => (
                    <option key={d} value={d}>
                      Forecast Day {d + 1}
                    </option>
                  ))}
                </select>

                <select
                  value={scenario}
                  onChange={(e) => onChangeScenario(e.target.value as ScenarioKey)}
                  aria-label="Overview scenario"
                  className="rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-100"
                >
                  {SCENARIOS.map((s) => (
                    <option key={s.key} value={s.key}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="inline-flex rounded-lg border border-slate-800 bg-slate-950 p-0.5">
                {(['normal', 'perturbed', 'difference'] as const).map((m) => (
                  <button
                    key={m}
                    type="button"
                    disabled={m !== 'normal' && !perturbedRollout}
                    onClick={() => setMapMode(m)}
                    className={`rounded-md px-2.5 py-1 text-[11px] font-medium capitalize transition ${
                      activeMode === m
                        ? 'bg-cyan-500/20 text-cyan-300'
                        : 'text-slate-400 hover:text-slate-200 disabled:opacity-35'
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>

            <ForecastMap
              title={`India Regional Grid — Day ${dayIdx + 1} (${variable.label})`}
              subtitle={`Mode: ${activeMode} · Scenario: ${scenarioMeta.label}`}
              grid={activeGrid}
              unit={variable.unit}
              digits={activeMode === 'difference' ? 3 : variable.digits}
              palette={activeMode === 'difference' ? 'diverging' : variable.palette}
              diverging={activeMode === 'difference'}
              scenarioBox={activeMode !== 'normal' ? scenarioMeta.box : undefined}
              showGeoOverlay={showGeoOverlay}
              showGridLines={showGridLines}
            />
          </div>

          {/* Right Column Previews */}
          <div className="space-y-4">
            <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4">
              <div className="mb-2 flex items-center justify-between">
                <h3 className="text-xs font-semibold text-slate-100">
                  10-Day Domain Trend ({variable.shortLabel})
                </h3>
                <button
                  type="button"
                  onClick={() => onNavigate('forecast')}
                  className="text-[11px] text-cyan-400 hover:underline"
                >
                  Open Studio →
                </button>
              </div>
              <div className="h-40 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trendSeries}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="day" stroke="#94a3b8" fontSize={10} />
                    <YAxis stroke="#94a3b8" fontSize={10} domain={['auto', 'auto']} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: 11 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="normalMean"
                      name="Normal"
                      stroke="#38bdf8"
                      strokeWidth={2}
                      dot={false}
                    />
                    {perturbedRollout && (
                      <Line
                        type="monotone"
                        dataKey="perturbedMean"
                        name="Perturbed"
                        stroke="#f59e0b"
                        strokeWidth={2}
                        dot={false}
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Regional Impact Preview */}
            <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4">
              <div className="mb-2 flex items-center justify-between">
                <h3 className="text-xs font-semibold text-slate-100">
                  Regional Rainfall Preview (10-Day)
                </h3>
                <button
                  type="button"
                  onClick={() => onNavigate('impact')}
                  className="text-[11px] text-cyan-400 hover:underline"
                >
                  Impact Explorer →
                </button>
              </div>
              <div className="space-y-1.5 text-xs">
                {indicators.slice(0, 4).map((r) => (
                  <div
                    key={r.region}
                    className="flex items-center justify-between rounded-lg bg-slate-950/70 px-2.5 py-1.5"
                  >
                    <span className="truncate text-slate-300">{r.region}</span>
                    <span className="font-mono text-[11px] text-cyan-300">
                      {r.normalMeanTp.toFixed(2)} mm/d
                      {perturbedRollout && (
                        <span
                          className={`ml-1.5 ${
                            r.deltaTp >= 0 ? 'text-emerald-400' : 'text-rose-400'
                          }`}
                        >
                          ({r.deltaTp >= 0 ? '+' : ''}
                          {r.deltaTp.toFixed(2)})
                        </span>
                      )}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Recent Saved Runs */}
            <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4">
              <div className="mb-2 flex items-center justify-between">
                <h3 className="text-xs font-semibold text-slate-100">
                  Recently Saved in Neon
                </h3>
                <button
                  type="button"
                  onClick={() => onNavigate('saved')}
                  className="text-[11px] text-cyan-400 hover:underline"
                >
                  View All →
                </button>
              </div>
              {recentRuns.length === 0 ? (
                <p className="text-[11px] text-slate-500">No saved runs yet.</p>
              ) : (
                <div className="space-y-1.5 text-[11px]">
                  {recentRuns.slice(0, 3).map((r) => (
                    <div
                      key={r.id}
                      className="flex items-center justify-between rounded-lg bg-slate-950/70 px-2.5 py-1.5"
                    >
                      <span className="truncate text-slate-200">
                        #{r.id} {r.name}
                      </span>
                      <span className="shrink-0 font-mono text-cyan-300">
                        {r.inference_ms.toFixed(0)} ms
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      ) : (
        <EmptyState
          title="No Forecast Loaded"
          description="Click Run New Simulation above to execute a live 10-day forecast on the FastAPI backend."
        />
      )}
    </div>
  );
};
