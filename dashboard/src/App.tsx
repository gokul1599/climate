import React, { useEffect, useMemo, useState } from 'react';
import { Activity, CloudRain, Play, RefreshCw } from 'lucide-react';
import {
  buildPerturbation,
  checkHealth,
  computeRegionalIndicators,
  DbStatus,
  deleteSimulationRun,
  fetchDbStatus,
  fetchSampleWindows,
  fetchSavedSimulations,
  runSimulation,
  SampleWindow,
  SavedSimulationRun,
  saveSimulationRun,
  SCENARIOS,
  ScenarioKey,
  VariableIndex,
  VARIABLES,
} from './services/api';
import { IndiaGridMap } from './components/IndiaGridMap';
import { ForecastCharts } from './components/ForecastCharts';
import { ImpactExplorer } from './components/ImpactExplorer';
import { SavedSimulations } from './components/SavedSimulations';

export function App() {
  const [health, setHealth] = useState<{ status: string; model_name: string } | null>(null);
  const [windows, setWindows] = useState<SampleWindow[]>([]);
  const [windowIdx, setWindowIdx] = useState<number>(0);
  const [scenario, setScenario] = useState<ScenarioKey>('warm_ocean');
  const [varIdx, setVarIdx] = useState<VariableIndex>(1);
  const [dayIdx, setDayIdx] = useState<number>(0);

  const [normalRollout, setNormalRollout] = useState<number[][][][] | null>(null);
  const [perturbedRollout, setPerturbedRollout] = useState<number[][][][] | null>(null);
  const [inferenceMs, setInferenceMs] = useState<number | null>(null);
  const [dbStatus, setDbStatus] = useState<DbStatus | null>(null);
  const [savedRuns, setSavedRuns] = useState<SavedSimulationRun[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [h, w, db] = await Promise.all([
          checkHealth(),
          fetchSampleWindows(),
          fetchDbStatus().catch(() => ({ connected: false, provider: 'neon', mode: 'offline' })),
        ]);
        setHealth(h);
        setWindows(w);
        setDbStatus(db);
        if (db.connected) {
          const hist = await fetchSavedSimulations();
          setSavedRuns(hist.runs);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  const executeSimulation = async (wIdx = windowIdx, sc = scenario) => {
    if (!windows[wIdx]) return;
    setLoading(true);
    setError(null);
    try {
      const win = windows[wIdx];
      const normRes = await runSimulation(win.init_states, win.doy_sincos, 10, null);
      setNormalRollout(normRes.rollout);

      let activePertRollout: number[][][][] | null = null;
      let activeInfMs = normRes.inference_ms;
      if (sc === 'normal') {
        setPerturbedRollout(null);
        setInferenceMs(normRes.inference_ms);
      } else {
        const pert = buildPerturbation(sc, win.init_states[2]);
        const pertRes = await runSimulation(win.init_states, win.doy_sincos, 10, pert);
        activePertRollout = pertRes.rollout;
        activeInfMs = pertRes.inference_ms;
        setPerturbedRollout(pertRes.rollout);
        setInferenceMs(pertRes.inference_ms);
      }

      if (dbStatus?.connected) {
        const summary = computeRegionalIndicators(normRes.rollout, activePertRollout);
        const saved = await saveSimulationRun({
          scenario: sc,
          window_id: win.id,
          window_label: win.label,
          doy_sincos: win.doy_sincos,
          steps: 10,
          inference_ms: activeInfMs,
          regional_summary: summary,
          anomalies: activePertRollout ? summary : null,
        }).catch(() => null);
        if (saved) {
          setSavedRuns((prev) => [saved, ...prev].slice(0, 20));
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteRun = async (id: number) => {
    try {
      await deleteSimulationRun(id);
      setSavedRuns((prev) => prev.filter((r) => r.id !== id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  useEffect(() => {
    if (windows.length > 0 && normalRollout === null && !loading) {
      executeSimulation(0, scenario);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [windows]);

  const activeVar = VARIABLES[varIdx];
  const activeScenario = SCENARIOS.find((s) => s.key === scenario)!;

  const diffGrid = useMemo(() => {
    if (!normalRollout || !perturbedRollout) return null;
    const n = normalRollout[dayIdx][varIdx];
    const p = perturbedRollout[dayIdx][varIdx];
    return n.map((row, i) => row.map((v, j) => p[i][j] - v));
  }, [normalRollout, perturbedRollout, dayIdx, varIdx]);

  const regionalIndicators = useMemo(() => {
    if (!normalRollout) return [];
    return computeRegionalIndicators(normalRollout, perturbedRollout);
  }, [normalRollout, perturbedRollout]);

  return (
    <div className="min-h-screen bg-navy-950 text-slate-100">
      <header className="border-b border-slate-800 bg-navy-900/90 px-6 py-4">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-cyan-500/15 text-cyan-400">
              <CloudRain className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white">
                MONSOONVERSE — Multiverse of Monsoon
              </h1>
              <p className="text-xs text-slate-400">
                India Interactive Climate Digital Twin · 32×36 ERA5 Grid (6.5°–37.5°N, 65.5°–100.5°E)
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 text-xs">
            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 font-medium ${
                health?.status === 'ready'
                  ? 'bg-emerald-500/15 text-emerald-300'
                  : 'bg-rose-500/15 text-rose-300'
              }`}
            >
              <Activity className="h-3.5 w-3.5" />
              {health ? `${health.model_name}: ${health.status}` : 'Backend Offline'}
            </span>
            {inferenceMs !== null && (
              <span className="rounded-full bg-slate-800 px-3 py-1 text-cyan-300">
                Live CPU Inference: {inferenceMs.toFixed(1)} ms
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-6 px-6 py-6">
        {error && (
          <div className="rounded-lg border border-rose-700 bg-rose-950/60 px-4 py-3 text-sm text-rose-200">
            {error}
          </div>
        )}

        {/* Controls */}
        <section className="grid grid-cols-1 gap-4 rounded-xl border border-slate-800 bg-navy-900/70 p-4 md:grid-cols-4">
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-400">
              Initial Validation Window
            </label>
            <select
              value={windowIdx}
              onChange={(e) => setWindowIdx(Number(e.target.value))}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
            >
              {windows.map((w, idx) => (
                <option key={w.id} value={idx}>
                  {w.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium text-slate-400">
              Climate Variable
            </label>
            <select
              value={varIdx}
              onChange={(e) => setVarIdx(Number(e.target.value) as VariableIndex)}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
            >
              {VARIABLES.map((v) => (
                <option key={v.key} value={v.index}>
                  {v.label} ({v.unit})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium text-slate-400">
              Perturbation Scenario
            </label>
            <select
              value={scenario}
              onChange={(e) => setScenario(e.target.value as ScenarioKey)}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
            >
              {SCENARIOS.map((s) => (
                <option key={s.key} value={s.key}>
                  {s.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-end">
            <button
              onClick={() => executeSimulation(windowIdx, scenario)}
              disabled={loading || windows.length === 0}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-cyan-500 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:opacity-50"
            >
              {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {loading ? 'Simulating...' : 'Run 10-Day Simulation'}
            </button>
          </div>
        </section>

        {/* Forecast Day Slider */}
        <section className="rounded-xl border border-slate-800 bg-navy-900/70 p-4">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="font-semibold text-slate-200">
              Forecast Horizon: Day {dayIdx + 1} of 10
            </span>
            <span className="text-slate-400">{activeScenario.description}</span>
          </div>
          <input
            type="range"
            min={0}
            max={9}
            value={dayIdx}
            onChange={(e) => setDayIdx(Number(e.target.value))}
            className="w-full accent-cyan-400"
          />
          <div className="mt-1 flex justify-between text-[11px] text-slate-500">
            {Array.from({ length: 10 }, (_, d) => (
              <button
                key={d}
                onClick={() => setDayIdx(d)}
                className={`rounded px-1.5 py-0.5 ${
                  dayIdx === d ? 'bg-cyan-500/20 text-cyan-300 font-semibold' : ''
                }`}
              >
                D{d + 1}
              </button>
            ))}
          </div>
        </section>

        {/* Spatial Grid Maps */}
        {normalRollout && (
          <section
            className={`grid grid-cols-1 gap-4 ${
              perturbedRollout ? 'lg:grid-cols-3' : 'lg:grid-cols-1'
            }`}
          >
            <IndiaGridMap
              title={`Normal Forecast — Day ${dayIdx + 1} (${activeVar.key})`}
              grid={normalRollout[dayIdx][varIdx]}
              unit={activeVar.unit}
              digits={activeVar.digits}
            />
            {perturbedRollout && (
              <IndiaGridMap
                title={`${activeScenario.label} — Day ${dayIdx + 1}`}
                grid={perturbedRollout[dayIdx][varIdx]}
                unit={activeVar.unit}
                digits={activeVar.digits}
                box={activeScenario.box}
              />
            )}
            {diffGrid && (
              <IndiaGridMap
                title={`Difference (Perturbed − Normal) — Day ${dayIdx + 1}`}
                grid={diffGrid}
                unit={activeVar.unit}
                digits={3}
                diverging
                box={activeScenario.box}
              />
            )}
          </section>
        )}

        {/* Evolution Charts */}
        {normalRollout && (
          <ForecastCharts
            normalRollout={normalRollout}
            perturbedRollout={perturbedRollout}
            variable={activeVar}
          />
        )}

        {/* Impact Explorer */}
        {normalRollout && (
          <ImpactExplorer
            indicators={regionalIndicators}
            hasPerturbation={perturbedRollout !== null}
          />
        )}

        {/* Saved Simulations (Neon PostgreSQL) */}
        <SavedSimulations
          dbStatus={dbStatus}
          runs={savedRuns}
          onDelete={handleDeleteRun}
        />
      </main>
    </div>
  );
}

export default App;
