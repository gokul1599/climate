import React, { useState } from 'react';
import { Download, Info } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { computeRegionalIndicators, computeRegionalTimeSeries, REGIONS } from '../services/api';
import { exportRegionalIndicatorsCsv } from '../services/exporters';
import { ScenarioKey, VariableMeta } from '../types/climate';
import { ForecastMap } from '../components/ForecastMap';
import { RegionalImpactTable } from '../components/RegionalImpactTable';
import { EmptyState } from '../components/EmptyState';

interface ImpactExplorerPageProps {
  normalRollout: number[][][][] | null;
  perturbedRollout: number[][][][] | null;
  dayIdx: number;
  scenario: ScenarioKey;
  variable: VariableMeta;
  selectedRegion: string;
  onSelectRegion: (r: string) => void;
  showGeoOverlay: boolean;
  showGridLines: boolean;
}

export const ImpactExplorerPage: React.FC<ImpactExplorerPageProps> = ({
  normalRollout,
  perturbedRollout,
  dayIdx,
  scenario,
  variable,
  selectedRegion,
  onSelectRegion,
  showGeoOverlay,
  showGridLines,
}) => {
  const [horizonScope, setHorizonScope] = useState<'all' | 'day'>('all');

  if (!normalRollout) {
    return (
      <EmptyState
        title="No Active Simulation"
        description="Run a simulation in Overview or Forecast Studio to compute regional model-derived impact indicators."
      />
    );
  }

  const indicators = computeRegionalIndicators(
    normalRollout,
    perturbedRollout,
    horizonScope === 'day' ? dayIdx : null,
  );
  const regionMeta = REGIONS.find((r) => r.name === selectedRegion) ?? REGIONS[0];
  const regionalRainSeries = computeRegionalTimeSeries(
    normalRollout,
    perturbedRollout,
    1,
    regionMeta.box,
    null,
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-amber-500/30 bg-amber-950/20 px-4 py-3 text-xs text-amber-200">
        <div className="flex items-start gap-2.5">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
          <div>
            <span className="font-semibold">Scientific Caveat:</span> All regional statistics below
            are exploratory, 1°×1° model-derived indicators for comparative scenario screening. They
            are <strong>not</strong> validated flood warnings, crop-loss diagnoses, or operational
            weather alerts.
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="inline-flex rounded-lg border border-slate-700 bg-slate-950 p-0.5">
            <button
              type="button"
              onClick={() => setHorizonScope('all')}
              className={`rounded-md px-2.5 py-1 text-[11px] ${
                horizonScope === 'all' ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400'
              }`}
            >
              10-Day Aggregate
            </button>
            <button
              type="button"
              onClick={() => setHorizonScope('day')}
              className={`rounded-md px-2.5 py-1 text-[11px] ${
                horizonScope === 'day' ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400'
              }`}
            >
              Day {dayIdx + 1} Only
            </button>
          </div>
          <button
            type="button"
            onClick={() => exportRegionalIndicatorsCsv(indicators, scenario)}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-800"
          >
            <Download className="h-3.5 w-3.5" />
            Export CSV
          </button>
        </div>
      </div>

      {/* Region Pills */}
      <div className="flex flex-wrap gap-2">
        {REGIONS.map((r) => (
          <button
            key={r.name}
            type="button"
            onClick={() => onSelectRegion(r.name)}
            className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition ${
              selectedRegion === r.name
                ? 'border-emerald-400/60 bg-emerald-500/15 text-emerald-200'
                : 'border-slate-800 bg-navy-900/70 text-slate-400 hover:text-slate-200'
            }`}
          >
            {r.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <ForecastMap
          title={`Selected Region Highlight: ${regionMeta.name}`}
          subtitle={regionMeta.focus}
          grid={
            perturbedRollout
              ? normalRollout[dayIdx][1].map((row, i) =>
                  row.map((v, j) => perturbedRollout[dayIdx][1][i][j] - v),
                )
              : normalRollout[dayIdx][variable.index]
          }
          unit={perturbedRollout ? 'mm/day Δ' : variable.unit}
          digits={perturbedRollout ? 3 : variable.digits}
          palette={perturbedRollout ? 'diverging' : variable.palette}
          diverging={Boolean(perturbedRollout)}
          regionBox={regionMeta.box}
          showGeoOverlay={showGeoOverlay}
          showGridLines={showGridLines}
        />

        <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4">
          <h3 className="mb-1 text-sm font-semibold text-white">
            {regionMeta.name} — Daily Mean Rainfall Trajectory (mm/day)
          </h3>
          <p className="mb-3 text-[11px] text-slate-400">{regionMeta.focus}</p>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={regionalRainSeries}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="label" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: 11 }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="normalValue" name="Normal Rain (mm/d)" fill="#38bdf8" radius={[3, 3, 0, 0]} />
                {perturbedRollout && (
                  <Bar
                    dataKey="perturbedValue"
                    name="Scenario Rain (mm/d)"
                    fill="#f59e0b"
                    radius={[3, 3, 0, 0]}
                  />
                )}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <RegionalImpactTable
        indicators={indicators}
        hasPerturbation={perturbedRollout !== null}
        selectedRegion={selectedRegion}
        onSelectRegion={onSelectRegion}
      />
    </div>
  );
};
