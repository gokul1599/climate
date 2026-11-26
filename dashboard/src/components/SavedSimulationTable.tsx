import React, { useState } from 'react';
import {
  Check,
  Database,
  Download,
  Edit3,
  Eye,
  GitCompare,
  Search,
  Trash2,
  X,
} from 'lucide-react';
import { DbStatus, SavedSimulationRun } from '../types/climate';
import { SCENARIOS } from '../services/api';
import { exportSavedRunJson } from '../services/exporters';
import { EmptyState } from './EmptyState';

interface SavedSimulationTableProps {
  dbStatus: DbStatus | null;
  runs: SavedSimulationRun[];
  total: number;
  page: number;
  pageSize: number;
  search: string;
  scenarioFilter: string;
  onSearchChange: (v: string) => void;
  onScenarioFilterChange: (v: string) => void;
  onPageChange: (nextPage: number) => void;
  onRename: (id: number, newName: string) => Promise<void>;
  onDelete: (id: number) => Promise<void>;
  onOpenRun: (run: SavedSimulationRun) => void;
  compareIds: number[];
  onToggleCompare: (id: number) => void;
}

export const SavedSimulationTable: React.FC<SavedSimulationTableProps> = ({
  dbStatus,
  runs,
  total,
  page,
  pageSize,
  search,
  scenarioFilter,
  onSearchChange,
  onScenarioFilterChange,
  onPageChange,
  onRename,
  onDelete,
  onOpenRun,
  compareIds,
  onToggleCompare,
}) => {
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editName, setEditName] = useState<string>('');
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  return (
    <div className="space-y-4 rounded-xl border border-slate-800/90 bg-navy-900/80 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-white">
            Saved Simulation Runs ({total})
          </h3>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="relative">
            <Search className="pointer-events-none absolute left-2.5 top-2 h-3.5 w-3.5 text-slate-500" />
            <input
              type="search"
              value={search}
              onChange={(e) => onSearchChange(e.target.value)}
              placeholder="Search name or window..."
              aria-label="Search saved simulations"
              className="rounded-lg border border-slate-700 bg-slate-950 pl-8 pr-3 py-1.5 text-xs text-slate-100 placeholder:text-slate-500"
            />
          </div>

          <select
            value={scenarioFilter}
            onChange={(e) => onScenarioFilterChange(e.target.value)}
            aria-label="Filter saved simulations by scenario"
            className="rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-100"
          >
            <option value="all">All Scenarios</option>
            {SCENARIOS.map((s) => (
              <option key={s.key} value={s.key}>
                {s.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {!dbStatus?.connected ? (
        <EmptyState
          title="Neon PostgreSQL Offline"
          description="Inference continues offline without database access. Connect DATABASE_URL on the backend to persist and manage simulation history."
        />
      ) : runs.length === 0 ? (
        <EmptyState
          title="No Saved Simulation Runs Found"
          description="Adjust your search filter or save a simulation from Forecast Studio or Scenario Lab."
        />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-[11px] text-slate-400">
                <th className="py-2 pr-2">Cmp</th>
                <th className="py-2 px-2">ID</th>
                <th className="py-2 px-2">Name</th>
                <th className="py-2 px-2">Scenario</th>
                <th className="py-2 px-2">Window</th>
                <th className="py-2 px-2 text-right">Inf (ms)</th>
                <th className="py-2 px-2 text-right">Peak |Δ Rain|</th>
                <th className="py-2 px-2">Created</th>
                <th className="py-2 pl-2 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {runs.map((run) => {
                const maxDeltaTp = run.regional_summary.reduce(
                  (acc, r) => Math.max(acc, Math.abs(r.deltaTp || 0)),
                  0,
                );
                const isEditing = editingId === run.id;
                const isConfirming = confirmDeleteId === run.id;
                const isCompared = compareIds.includes(run.id);

                return (
                  <tr key={run.id} className="hover:bg-slate-800/30">
                    <td className="py-2.5 pr-2">
                      <button
                        type="button"
                        onClick={() => onToggleCompare(run.id)}
                        aria-label={`Select run #${run.id} for comparison`}
                        className={`rounded p-1 border ${
                          isCompared
                            ? 'border-cyan-400 bg-cyan-500/20 text-cyan-300'
                            : 'border-slate-700 text-slate-500 hover:text-slate-200'
                        }`}
                      >
                        <GitCompare className="h-3 w-3" />
                      </button>
                    </td>
                    <td className="py-2.5 px-2 font-mono text-slate-400">#{run.id}</td>
                    <td className="py-2.5 px-2 font-medium text-slate-100">
                      {isEditing ? (
                        <div className="flex items-center gap-1">
                          <input
                            type="text"
                            value={editName}
                            onChange={(e) => setEditName(e.target.value)}
                            className="w-36 rounded border border-cyan-500/50 bg-slate-950 px-2 py-0.5 text-xs text-white"
                          />
                          <button
                            type="button"
                            onClick={async () => {
                              if (editName.trim()) {
                                await onRename(run.id, editName.trim());
                                setEditingId(null);
                              }
                            }}
                            aria-label="Save name"
                            className="rounded p-1 text-emerald-400 hover:bg-emerald-500/15"
                          >
                            <Check className="h-3.5 w-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => setEditingId(null)}
                            aria-label="Cancel rename"
                            className="rounded p-1 text-slate-400 hover:bg-slate-800"
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      ) : (
                        <span>{run.name}</span>
                      )}
                    </td>
                    <td className="py-2.5 px-2 text-cyan-300">{run.scenario}</td>
                    <td className="py-2.5 px-2 text-slate-300">{run.window_label}</td>
                    <td className="py-2.5 px-2 text-right font-mono text-slate-300">
                      {run.inference_ms.toFixed(1)}
                    </td>
                    <td className="py-2.5 px-2 text-right font-mono text-amber-300">
                      {maxDeltaTp.toFixed(3)} mm/d
                    </td>
                    <td className="py-2.5 px-2 text-[11px] text-slate-400">
                      {new Date(run.created_at).toLocaleString()}
                    </td>
                    <td className="py-2.5 pl-2 text-right">
                      {isConfirming ? (
                        <span className="inline-flex items-center gap-1">
                          <button
                            type="button"
                            onClick={async () => {
                              await onDelete(run.id);
                              setConfirmDeleteId(null);
                            }}
                            className="rounded bg-rose-600 px-2 py-0.5 text-[10px] font-semibold text-white hover:bg-rose-500"
                          >
                            Confirm
                          </button>
                          <button
                            type="button"
                            onClick={() => setConfirmDeleteId(null)}
                            className="rounded border border-slate-700 px-1.5 py-0.5 text-[10px] text-slate-300"
                          >
                            Cancel
                          </button>
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1">
                          <button
                            type="button"
                            onClick={() => onOpenRun(run)}
                            title="Inspect saved run"
                            aria-label={`Inspect run #${run.id}`}
                            className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-cyan-300"
                          >
                            <Eye className="h-3.5 w-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              setEditingId(run.id);
                              setEditName(run.name);
                            }}
                            title="Rename run"
                            aria-label={`Rename run #${run.id}`}
                            className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-amber-300"
                          >
                            <Edit3 className="h-3.5 w-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => exportSavedRunJson(run)}
                            title="Export JSON"
                            aria-label={`Export run #${run.id} JSON`}
                            className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-emerald-300"
                          >
                            <Download className="h-3.5 w-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => setConfirmDeleteId(run.id)}
                            title="Delete run"
                            aria-label={`Delete run #${run.id}`}
                            className="rounded p-1 text-slate-400 hover:bg-rose-500/15 hover:text-rose-300"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {dbStatus?.connected && totalPages > 1 && (
        <div className="flex items-center justify-between border-t border-slate-800 pt-3 text-xs text-slate-400">
          <span>
            Page {page + 1} of {totalPages}
          </span>
          <div className="flex gap-2">
            <button
              type="button"
              disabled={page <= 0}
              onClick={() => onPageChange(page - 1)}
              className="rounded border border-slate-700 px-2.5 py-1 hover:bg-slate-800 disabled:opacity-40"
            >
              Previous
            </button>
            <button
              type="button"
              disabled={page + 1 >= totalPages}
              onClick={() => onPageChange(page + 1)}
              className="rounded border border-slate-700 px-2.5 py-1 hover:bg-slate-800 disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
