import React from 'react';
import { Download, Play, RefreshCw, RotateCcw, Save } from 'lucide-react';
import { SCENARIOS, VARIABLES } from '../services/api';
import { SampleWindow, ScenarioKey, VariableIndex } from '../types/climate';

interface ForecastControlsProps {
  windows: SampleWindow[];
  windowIdx: number;
  onChangeWindow: (idx: number) => void;
  varIdx: VariableIndex;
  onChangeVar: (idx: VariableIndex) => void;
  scenario: ScenarioKey;
  onChangeScenario: (sc: ScenarioKey) => void;
  dayIdx: number;
  onChangeDay: (d: number) => void;
  loading: boolean;
  canSave: boolean;
  dbConnected: boolean;
  onRun: () => void;
  onReset: () => void;
  onOpenSave: () => void;
  onOpenExport: () => void;
}

export const ForecastControls: React.FC<ForecastControlsProps> = ({
  windows,
  windowIdx,
  onChangeWindow,
  varIdx,
  onChangeVar,
  scenario,
  onChangeScenario,
  dayIdx,
  onChangeDay,
  loading,
  canSave,
  dbConnected,
  onRun,
  onReset,
  onOpenSave,
  onOpenExport,
}) => {
  const activeScenario = SCENARIOS.find((s) => s.key === scenario) ?? SCENARIOS[0];

  return (
    <div className="space-y-4 rounded-xl border border-slate-800/90 bg-navy-900/80 p-4">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
        <div>
          <label
            htmlFor="ctrl-window"
            className="mb-1 block text-[11px] font-medium text-slate-400"
          >
            Initial Validation Window
          </label>
          <select
            id="ctrl-window"
            value={windowIdx}
            onChange={(e) => onChangeWindow(Number(e.target.value))}
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-100"
          >
            {windows.map((w, idx) => (
              <option key={w.id} value={idx}>
                {w.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor="ctrl-var" className="mb-1 block text-[11px] font-medium text-slate-400">
            Climate Variable
          </label>
          <select
            id="ctrl-var"
            value={varIdx}
            onChange={(e) => onChangeVar(Number(e.target.value) as VariableIndex)}
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-100"
          >
            {VARIABLES.map((v) => (
              <option key={v.key} value={v.index}>
                {v.label} ({v.unit})
              </option>
            ))}
          </select>
        </div>

        <div>
          <label
            htmlFor="ctrl-scenario"
            className="mb-1 block text-[11px] font-medium text-slate-400"
          >
            Perturbation Scenario
          </label>
          <select
            id="ctrl-scenario"
            value={scenario}
            onChange={(e) => onChangeScenario(e.target.value as ScenarioKey)}
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-100"
          >
            {SCENARIOS.map((s) => (
              <option key={s.key} value={s.key}>
                {s.label} ({s.badge})
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-wrap items-end gap-2">
          <button
            type="button"
            onClick={onRun}
            disabled={loading || windows.length === 0}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-cyan-500 px-3 py-2 text-xs font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:opacity-50"
          >
            {loading ? (
              <RefreshCw className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
            ) : (
              <Play className="h-3.5 w-3.5" aria-hidden="true" />
            )}
            {loading ? 'Running...' : 'Run Simulation'}
          </button>

          <button
            type="button"
            onClick={onReset}
            title="Reset controls to Normal Day 1"
            aria-label="Reset forecast controls"
            className="rounded-lg border border-slate-700 bg-slate-950 p-2 text-slate-300 hover:bg-slate-800 hover:text-white"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>

          <button
            type="button"
            onClick={onOpenSave}
            disabled={!canSave || !dbConnected}
            title={
              dbConnected
                ? 'Save simulation summary to Neon PostgreSQL'
                : 'Neon DB offline'
            }
            className="inline-flex items-center gap-1 rounded-lg border border-emerald-500/40 bg-emerald-500/15 px-2.5 py-2 text-xs font-medium text-emerald-300 hover:bg-emerald-500/25 disabled:opacity-40"
          >
            <Save className="h-3.5 w-3.5" aria-hidden="true" />
            Save
          </button>

          <button
            type="button"
            onClick={onOpenExport}
            disabled={!canSave}
            className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-2 text-xs font-medium text-slate-200 hover:bg-slate-800 disabled:opacity-40"
          >
            <Download className="h-3.5 w-3.5" aria-hidden="true" />
            Export
          </button>
        </div>
      </div>

      <div className="border-t border-slate-800/80 pt-3">
        <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2 text-xs">
          <label htmlFor="ctrl-day-slider" className="font-semibold text-slate-200">
            Forecast Horizon: Day {dayIdx + 1} of 10
          </label>
          <span className="text-[11px] text-slate-400">{activeScenario.description}</span>
        </div>
        <input
          id="ctrl-day-slider"
          type="range"
          min={0}
          max={9}
          value={dayIdx}
          onChange={(e) => onChangeDay(Number(e.target.value))}
          className="w-full accent-cyan-400"
        />
        <div className="mt-1.5 flex justify-between text-[11px] text-slate-500">
          {Array.from({ length: 10 }, (_, d) => (
            <button
              key={d}
              type="button"
              onClick={() => onChangeDay(d)}
              className={`rounded px-1.5 py-0.5 transition ${
                dayIdx === d
                  ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/30'
                  : 'hover:text-slate-200'
              }`}
            >
              D{d + 1}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
