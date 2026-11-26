import React from 'react';
import { Database, Trash2 } from 'lucide-react';
import { DbStatus, SavedSimulationRun } from '../services/api';

interface SavedSimulationsProps {
  dbStatus: DbStatus | null;
  runs: SavedSimulationRun[];
  onDelete: (id: number) => void;
}

export const SavedSimulations: React.FC<SavedSimulationsProps> = ({
  dbStatus,
  runs,
  onDelete,
}) => {
  return (
    <div className="rounded-xl border border-slate-800 bg-navy-900/80 p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-slate-100">
            Saved Simulation Runs (Neon PostgreSQL)
          </h3>
        </div>
        <span
          className={`rounded-full px-2.5 py-0.5 text-[11px] font-medium ${
            dbStatus?.connected
              ? 'bg-emerald-500/15 text-emerald-300'
              : 'bg-slate-800 text-slate-400'
          }`}
        >
          {dbStatus?.connected
            ? `Neon Connected (${dbStatus.database || 'neondb'})`
            : `Neon: ${dbStatus?.mode || 'offline'}`}
        </span>
      </div>

      {!dbStatus?.connected ? (
        <p className="text-xs text-slate-400">
          Database persistence is inactive (offline mode). Configure DATABASE_URL to persist simulation summaries.
        </p>
      ) : runs.length === 0 ? (
        <p className="text-xs text-slate-400">
          No simulation runs persisted yet. Run a simulation above to record its summary to Neon.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2 pr-3">ID</th>
                <th className="py-2 px-3">Timestamp</th>
                <th className="py-2 px-3">Initial Window</th>
                <th className="py-2 px-3">Scenario</th>
                <th className="py-2 px-3 text-right">Inference (ms)</th>
                <th className="py-2 px-3 text-right">Peak |Δ Rain| (mm/d)</th>
                <th className="py-2 pl-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {runs.map((run) => {
                const maxDeltaTp = run.regional_summary.reduce(
                  (acc, r) => Math.max(acc, Math.abs(r.deltaTp || 0)),
                  0,
                );
                return (
                  <tr key={run.id} className="hover:bg-slate-800/30">
                    <td className="py-2 pr-3 font-mono text-slate-300">#{run.id}</td>
                    <td className="py-2 px-3 text-slate-400">
                      {new Date(run.created_at).toLocaleString()}
                    </td>
                    <td className="py-2 px-3 text-slate-200">{run.window_label}</td>
                    <td className="py-2 px-3 text-cyan-300">{run.scenario}</td>
                    <td className="py-2 px-3 text-right text-slate-300">
                      {run.inference_ms.toFixed(1)}
                    </td>
                    <td className="py-2 px-3 text-right text-amber-300">
                      {maxDeltaTp.toFixed(3)}
                    </td>
                    <td className="py-2 pl-3 text-right">
                      <button
                        onClick={() => onDelete(run.id)}
                        title="Delete run"
                        className="rounded p-1 text-slate-400 hover:bg-rose-500/15 hover:text-rose-300"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
