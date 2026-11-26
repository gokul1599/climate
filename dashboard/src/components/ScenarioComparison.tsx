import React from 'react';
import { FlaskConical, Sliders } from 'lucide-react';
import { CustomPerturbationConfig, GridCellSelection, ScenarioMeta, VariableMeta } from '../types/climate';
import { ForecastMap } from './ForecastMap';

interface ScenarioComparisonProps {
  normalRollout: number[][][][];
  perturbedRollout: number[][][][] | null;
  dayIdx: number;
  onChangeDay: (d: number) => void;
  variable: VariableMeta;
  scenario: ScenarioMeta;
  customPert: CustomPerturbationConfig;
  onChangeCustomPert: (cfg: CustomPerturbationConfig) => void;
  onRunCustom: () => void;
  selectedCell: GridCellSelection | null;
  onSelectCell: (c: GridCellSelection) => void;
  showGeoOverlay: boolean;
  showGridLines: boolean;
}

export const ScenarioComparison: React.FC<ScenarioComparisonProps> = ({
  normalRollout,
  perturbedRollout,
  dayIdx,
  onChangeDay,
  variable,
  scenario,
  customPert,
  onChangeCustomPert,
  onRunCustom,
  selectedCell,
  onSelectCell,
  showGeoOverlay,
  showGridLines,
}) => {
  const activeBox: [number, number, number, number] | undefined =
    scenario.key === 'custom'
      ? [customPert.latMin, customPert.latMax, customPert.lonMin, customPert.lonMax]
      : scenario.box;

  const diffGrid = perturbedRollout
    ? normalRollout[dayIdx][variable.index].map((row, i) =>
        row.map((val, j) => perturbedRollout[dayIdx][variable.index][i][j] - val),
      )
    : null;

  // Compute spatial response energy across key lead days (1, 4, 8, 10)
  const leadSnapshots = [0, 3, 7, 9].map((d) => {
    if (!perturbedRollout) {
      return { day: d + 1, meanAbsDiff: 0, maxAbsDiff: 0, activeCells: 0 };
    }
    let sumAbs = 0;
    let maxAbs = 0;
    let active = 0;
    for (let i = 0; i < 32; i++) {
      for (let j = 0; j < 36; j++) {
        const diff = Math.abs(
          perturbedRollout[d][variable.index][i][j] - normalRollout[d][variable.index][i][j],
        );
        sumAbs += diff;
        if (diff > maxAbs) maxAbs = diff;
        if (diff > 1e-3) active++;
      }
    }
    return {
      day: d + 1,
      meanAbsDiff: sumAbs / (32 * 36),
      maxAbsDiff: maxAbs,
      activeCells: active,
    };
  });

  return (
    <div className="space-y-4">
      {scenario.key === 'custom' && (
        <div className="rounded-xl border border-amber-500/30 bg-amber-950/20 p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Sliders className="h-4 w-4 text-amber-400" />
              <h3 className="text-xs font-semibold text-amber-200 sm:text-sm">
                Experimental Custom Perturbation Builder (Physical Units on init_states[-1])
              </h3>
            </div>
            <button
              type="button"
              onClick={onRunCustom}
              className="rounded-lg bg-amber-500 px-3 py-1.5 text-xs font-semibold text-slate-950 hover:bg-amber-400"
            >
              Simulate Custom Shock
            </button>
          </div>
          <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-4 lg:grid-cols-7">
            {(
              [
                ['Lat Min (°N)', 'latMin', 6.5, 35],
                ['Lat Max (°N)', 'latMax', 8, 37.5],
                ['Lon Min (°E)', 'lonMin', 65.5, 98],
                ['Lon Max (°E)', 'lonMax', 68, 100.5],
                ['Δ t2m (K)', 'deltaT2mK', -5, 5],
                ['Δ tcwv (%)', 'deltaTcwvPct', -30, 30],
                ['Δ Wind (%)', 'deltaWindPct', -60, 40],
              ] as const
            ).map(([lbl, key, minV, maxV]) => (
              <label key={key} className="block">
                <span className="mb-1 block text-[10px] text-slate-400">{lbl}</span>
                <input
                  type="number"
                  step="0.5"
                  min={minV}
                  max={maxV}
                  value={customPert[key]}
                  onChange={(e) =>
                    onChangeCustomPert({ ...customPert, [key]: Number(e.target.value) })
                  }
                  className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-100"
                />
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Three Coordinated Maps */}
      <div
        className={`grid grid-cols-1 gap-4 ${
          perturbedRollout ? 'xl:grid-cols-3' : 'xl:grid-cols-1'
        }`}
      >
        <ForecastMap
          title={`1. Normal Forecast — Day ${dayIdx + 1} (${variable.shortLabel})`}
          subtitle="Unperturbed autoregressive baseline"
          grid={normalRollout[dayIdx][variable.index]}
          unit={variable.unit}
          digits={variable.digits}
          palette={variable.palette}
          selectedCell={selectedCell}
          onSelectCell={onSelectCell}
          showGeoOverlay={showGeoOverlay}
          showGridLines={showGridLines}
          compact={Boolean(perturbedRollout)}
        />
        {perturbedRollout && (
          <ForecastMap
            title={`2. ${scenario.label} — Day ${dayIdx + 1}`}
            subtitle={scenario.isOfficial ? 'Official perturbation preset' : 'Experimental shock'}
            grid={perturbedRollout[dayIdx][variable.index]}
            unit={variable.unit}
            digits={variable.digits}
            palette={variable.palette}
            scenarioBox={activeBox}
            selectedCell={selectedCell}
            onSelectCell={onSelectCell}
            showGeoOverlay={showGeoOverlay}
            showGridLines={showGridLines}
            compact
          />
        )}
        {diffGrid && (
          <ForecastMap
            title={`3. Spatial Difference (Perturbed − Normal) — Day ${dayIdx + 1}`}
            subtitle="Amber outline marks initial perturbation box"
            grid={diffGrid}
            unit={variable.unit}
            digits={3}
            palette="diverging"
            diverging
            scenarioBox={activeBox}
            selectedCell={selectedCell}
            onSelectCell={onSelectCell}
            showGeoOverlay={showGeoOverlay}
            showGridLines={showGridLines}
            compact
          />
        )}
      </div>

      {/* Day 1 / 4 / 8 / 10 Propagation Diagnostics */}
      <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <FlaskConical className="h-4 w-4 text-amber-400" />
            <h3 className="text-xs font-semibold text-slate-100 sm:text-sm">
              Spatial Ripple &amp; Lead-Day Diagnostics ({variable.shortLabel})
            </h3>
          </div>
          <span className="text-[11px] text-slate-400">
            Click a lead card to jump directly to that forecast day
          </span>
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {leadSnapshots.map((s) => (
            <button
              key={s.day}
              type="button"
              onClick={() => onChangeDay(s.day - 1)}
              className={`rounded-lg border p-3 text-left transition ${
                dayIdx === s.day - 1
                  ? 'border-cyan-500/60 bg-cyan-500/10'
                  : 'border-slate-800 bg-slate-950/70 hover:border-slate-700'
              }`}
            >
              <div className="text-xs font-semibold text-slate-200">Day {s.day}</div>
              <div className="mt-1 text-[11px] text-slate-400">
                Max |Δ|: <span className="font-mono text-amber-300">{s.maxAbsDiff.toFixed(3)}</span>{' '}
                {variable.unit}
              </div>
              <div className="text-[11px] text-slate-400">
                Mean |Δ|: <span className="font-mono text-cyan-300">{s.meanAbsDiff.toFixed(4)}</span>
              </div>
              <div className="text-[11px] text-slate-500">Active cells (&gt;1e-3): {s.activeCells}/1152</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
