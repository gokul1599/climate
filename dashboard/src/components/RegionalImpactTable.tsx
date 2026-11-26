import React from 'react';
import { RegionalIndicator } from '../types/climate';

interface RegionalImpactTableProps {
  indicators: RegionalIndicator[];
  hasPerturbation: boolean;
  selectedRegion: string;
  onSelectRegion: (regionName: string) => void;
}

export const RegionalImpactTable: React.FC<RegionalImpactTableProps> = ({
  indicators,
  hasPerturbation,
  selectedRegion,
  onSelectRegion,
}) => {
  return (
    <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-xs font-semibold text-slate-100 sm:text-sm">
            Regional Monsoon Stress &amp; Impact Indicators (Model-Derived)
          </h3>
          <p className="text-[11px] text-slate-400">
            Derived strictly from live model rollouts across 6 Indian sub-regions — includes exploratory
            apparent-temperature proxy; not operational disaster warnings.
          </p>
        </div>
        <span className="rounded-full border border-slate-700 bg-slate-900 px-2.5 py-0.5 text-[10px] text-slate-300">
          Click a row to highlight region on map
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-slate-800 text-[11px] text-slate-400">
              <th className="py-2 pr-3">Region</th>
              <th className="py-2 px-2 text-right">Cells</th>
              <th className="py-2 px-3 text-right">Norm Rain (mm/d)</th>
              {hasPerturbation && (
                <>
                  <th className="py-2 px-3 text-right">Scen Rain</th>
                  <th className="py-2 px-3 text-right">Δ Rain</th>
                  <th className="py-2 px-3 text-right">Δ T2m (K)</th>
                  <th className="py-2 px-3 text-right">Δ Wind (m/s)</th>
                </>
              )}
              <th className="py-2 px-3 text-right">Heavy Exposure (&gt;=10mm)</th>
              <th className="py-2 px-3 text-right">Persistent Rain Streak (d)</th>
              <th className="py-2 px-3 text-right">Heat-Stress Proxy (°C)*</th>
              <th className="py-2 pl-3 text-right">Strong Wind (&gt;=10 m/s)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {indicators.map((r) => {
              const isSelected = r.region === selectedRegion;
              const streakNorm = r.normalPersistentHeavyStreak ?? 0;
              const streakPert = r.perturbedPersistentHeavyStreak ?? streakNorm;
              const heatNorm = r.normalHeatStressProxyC ?? r.normalMeanT2mC;
              const heatPert = r.perturbedHeatStressProxyC ?? heatNorm;
              const windNorm = r.normalStrongWindCells ?? 0;
              const windPert = r.perturbedStrongWindCells ?? windNorm;
              return (
                <tr
                  key={r.region}
                  onClick={() => onSelectRegion(r.region)}
                  className={`cursor-pointer transition ${
                    isSelected ? 'bg-emerald-500/10' : 'hover:bg-slate-800/40'
                  }`}
                >
                  <td className="py-2.5 pr-3 font-medium text-slate-100">
                    <span className="inline-flex items-center gap-1.5">
                      {isSelected && (
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" aria-hidden="true" />
                      )}
                      {r.region}
                    </span>
                  </td>
                  <td className="py-2.5 px-2 text-right font-mono text-slate-400">{r.gridCells}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-cyan-300">
                    {r.normalMeanTp.toFixed(2)}
                  </td>
                  {hasPerturbation && (
                    <>
                      <td className="py-2.5 px-3 text-right font-mono text-amber-300">
                        {r.perturbedMeanTp.toFixed(2)}
                      </td>
                      <td
                        className={`py-2.5 px-3 text-right font-mono font-medium ${
                          r.deltaTp >= 0 ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {r.deltaTp >= 0 ? '+' : ''}
                        {r.deltaTp.toFixed(3)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-slate-300">
                        {r.deltaT2m >= 0 ? '+' : ''}
                        {r.deltaT2m.toFixed(3)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-slate-300">
                        {r.deltaWind >= 0 ? '+' : ''}
                        {r.deltaWind.toFixed(3)}
                      </td>
                    </>
                  )}
                  <td className="py-2.5 px-3 text-right font-mono text-slate-200">
                    {hasPerturbation
                      ? `${r.normalHeavyCells} (${(r.normalHeavyExposurePct ?? 0).toFixed(1)}%) → ${r.perturbedHeavyCells} (${(r.perturbedHeavyExposurePct ?? 0).toFixed(1)}%)`
                      : `${r.normalHeavyCells} (${(r.normalHeavyExposurePct ?? 0).toFixed(1)}%)`}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-300">
                    {hasPerturbation ? `${streakNorm}d → ${streakPert}d` : `${streakNorm}d`}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-amber-200">
                    {hasPerturbation ? `${heatNorm.toFixed(1)} → ${heatPert.toFixed(1)}` : heatNorm.toFixed(1)}
                  </td>
                  <td className="py-2.5 pl-3 text-right font-mono text-slate-300">
                    {hasPerturbation ? `${windNorm} → ${windPert}` : windNorm}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2.5 text-[10px] text-slate-400">
        *Heat-Stress Proxy is an unvalidated apparent-temperature indicator (T_c + 0.33·0.65·tcwv − 0.70·|V| − 4.0) for exploratory scenario delta screening only.
      </p>
    </div>
  );
};
