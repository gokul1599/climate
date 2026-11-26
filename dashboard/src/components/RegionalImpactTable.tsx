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
            Regional Climate Impact Indicators (Model-Derived)
          </h3>
          <p className="text-[11px] text-slate-400">
            Exploratory model-derived indicators across 6 Indian sub-regions — not validated flood,
            crop-loss, or operational disaster forecasts.
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
                  <th className="py-2 px-3 text-right">Δ TCWV (kg/m²)</th>
                  <th className="py-2 px-3 text-right">Δ Wind (m/s)</th>
                </>
              )}
              <th className="py-2 pl-3 text-right">Heavy-Rain Cell-Days (≥10mm)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {indicators.map((r) => {
              const isSelected = r.region === selectedRegion;
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
                        {r.deltaTcwv >= 0 ? '+' : ''}
                        {r.deltaTcwv.toFixed(3)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-slate-300">
                        {r.deltaWind >= 0 ? '+' : ''}
                        {r.deltaWind.toFixed(3)}
                      </td>
                    </>
                  )}
                  <td className="py-2.5 pl-3 text-right font-mono text-slate-200">
                    {hasPerturbation
                      ? `${r.normalHeavyCells} → ${r.perturbedHeavyCells}`
                      : r.normalHeavyCells}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
