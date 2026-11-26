import React, { useState } from 'react';
import { GitCompare, Save, X } from 'lucide-react';
import { DbStatus, SavedSimulationRun } from '../types/climate';
import { SavedSimulationTable } from '../components/SavedSimulationTable';

interface SavedSimulationsPageProps {
  dbStatus: DbStatus | null;
  runs: SavedSimulationRun[];
  total: number;
  page: number;
  pageSize: number;
  search: string;
  scenarioFilter: string;
  canSaveCurrent: boolean;
  onSearchChange: (v: string) => void;
  onScenarioFilterChange: (v: string) => void;
  onPageChange: (next: number) => void;
  onSaveCurrent: (customName: string, notes: string) => Promise<void>;
  onRename: (id: number, name: string) => Promise<void>;
  onDelete: (id: number) => Promise<void>;
}

export const SavedSimulationsPage: React.FC<SavedSimulationsPageProps> = ({
  dbStatus,
  runs,
  total,
  page,
  pageSize,
  search,
  scenarioFilter,
  canSaveCurrent,
  onSearchChange,
  onScenarioFilterChange,
  onPageChange,
  onSaveCurrent,
  onRename,
  onDelete,
}) => {
  const [saveName, setSaveName] = useState('');
  const [saveNotes, setSaveNotes] = useState('');
  const [saving, setSaving] = useState(false);
  const [inspectedRun, setInspectedRun] = useState<SavedSimulationRun | null>(null);
  const [compareIds, setCompareIds] = useState<number[]>([]);

  const toggleCompare = (id: number) => {
    setCompareIds((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 2) return [prev[1], id];
      return [...prev, id];
    });
  };

  const comparedRuns = compareIds
    .map((id) => runs.find((r) => r.id === id))
    .filter((r): r is SavedSimulationRun => Boolean(r));

  return (
    <div className="space-y-6">
      {/* Save Current Simulation Bar */}
      <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-white">
            Save Current Simulation Summary to Neon PostgreSQL
          </h3>
          <span className="text-[11px] text-slate-400">
            Stores lightweight metadata &amp; regional indicators (~3 KB/row)
          </span>
        </div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <input
            type="text"
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            placeholder="Custom simulation name (e.g., Jul BoB +3K test)"
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-100 placeholder:text-slate-500"
          />
          <input
            type="text"
            value={saveNotes}
            onChange={(e) => setSaveNotes(e.target.value)}
            placeholder="Optional analyst notes..."
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-100 placeholder:text-slate-500"
          />
          <button
            type="button"
            disabled={!canSaveCurrent || !dbStatus?.connected || saving}
            onClick={async () => {
              setSaving(true);
              try {
                await onSaveCurrent(saveName, saveNotes);
                setSaveName('');
                setSaveNotes('');
              } finally {
                setSaving(false);
              }
            }}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-500 px-4 py-2 text-xs font-semibold text-slate-950 hover:bg-emerald-400 disabled:opacity-40"
          >
            <Save className="h-3.5 w-3.5" />
            {saving ? 'Saving...' : 'Save Current Run'}
          </button>
        </div>
      </div>

      {/* Side-by-side comparison if 2 runs selected */}
      {comparedRuns.length === 2 && (
        <div className="rounded-xl border border-cyan-500/40 bg-navy-900/90 p-4">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <GitCompare className="h-4 w-4 text-cyan-400" />
              <h3 className="text-sm font-semibold text-white">
                Comparing #{comparedRuns[0].id} ({comparedRuns[0].name}) vs #{comparedRuns[1].id} (
                {comparedRuns[1].name})
              </h3>
            </div>
            <button
              type="button"
              onClick={() => setCompareIds([])}
              className="rounded border border-slate-700 px-2 py-0.5 text-[11px] text-slate-300 hover:bg-slate-800"
            >
              Clear Compare
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2 pr-3">Region</th>
                  <th className="py-2 px-3 text-right">#{comparedRuns[0].id} Δ Rain</th>
                  <th className="py-2 px-3 text-right">#{comparedRuns[1].id} Δ Rain</th>
                  <th className="py-2 px-3 text-right">#{comparedRuns[0].id} Δ T2m (K)</th>
                  <th className="py-2 pl-3 text-right">#{comparedRuns[1].id} Δ T2m (K)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {comparedRuns[0].regional_summary.map((r0, idx) => {
                  const r1 = comparedRuns[1].regional_summary[idx];
                  return (
                    <tr key={r0.region}>
                      <td className="py-2 pr-3 font-medium text-slate-200">{r0.region}</td>
                      <td className="py-2 px-3 text-right font-mono text-cyan-300">
                        {(r0.deltaTp ?? 0).toFixed(3)}
                      </td>
                      <td className="py-2 px-3 text-right font-mono text-amber-300">
                        {(r1?.deltaTp ?? 0).toFixed(3)}
                      </td>
                      <td className="py-2 px-3 text-right font-mono text-slate-300">
                        {(r0.deltaT2m ?? 0).toFixed(3)}
                      </td>
                      <td className="py-2 pl-3 text-right font-mono text-slate-300">
                        {(r1?.deltaT2m ?? 0).toFixed(3)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <SavedSimulationTable
        dbStatus={dbStatus}
        runs={runs}
        total={total}
        page={page}
        pageSize={pageSize}
        search={search}
        scenarioFilter={scenarioFilter}
        onSearchChange={onSearchChange}
        onScenarioFilterChange={onScenarioFilterChange}
        onPageChange={onPageChange}
        onRename={onRename}
        onDelete={onDelete}
        onOpenRun={(run) => setInspectedRun(run)}
        compareIds={compareIds}
        onToggleCompare={toggleCompare}
      />

      {/* Run Detail Modal */}
      {inspectedRun && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4"
          role="dialog"
          aria-modal="true"
        >
          <div className="max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded-xl border border-slate-800 bg-[#0b1426] p-5 shadow-2xl">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-white">
                  #{inspectedRun.id} — {inspectedRun.name}
                </h3>
                <p className="text-[11px] text-slate-400">
                  {inspectedRun.scenario} · {inspectedRun.window_label} · {inspectedRun.model_version} ·{' '}
                  {inspectedRun.inference_ms.toFixed(1)} ms
                </p>
              </div>
              <button
                type="button"
                onClick={() => setInspectedRun(null)}
                aria-label="Close run details"
                className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-white"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            {inspectedRun.notes && (
              <p className="mb-3 rounded-lg bg-slate-900/80 px-3 py-2 text-xs text-slate-300">
                Notes: {inspectedRun.notes}
              </p>
            )}
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2 pr-2">Region</th>
                  <th className="py-2 px-2 text-right">Norm Rain</th>
                  <th className="py-2 px-2 text-right">Scen Rain</th>
                  <th className="py-2 px-2 text-right">Δ Rain</th>
                  <th className="py-2 pl-2 text-right">Δ T2m (K)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {inspectedRun.regional_summary.map((r) => (
                  <tr key={r.region}>
                    <td className="py-2 pr-2 text-slate-200">{r.region}</td>
                    <td className="py-2 px-2 text-right font-mono text-cyan-300">
                      {(r.normalMeanTp ?? 0).toFixed(2)}
                    </td>
                    <td className="py-2 px-2 text-right font-mono text-amber-300">
                      {(r.perturbedMeanTp ?? 0).toFixed(2)}
                    </td>
                    <td className="py-2 px-2 text-right font-mono text-slate-200">
                      {(r.deltaTp ?? 0).toFixed(3)}
                    </td>
                    <td className="py-2 pl-2 text-right font-mono text-slate-300">
                      {(r.deltaT2m ?? 0).toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
