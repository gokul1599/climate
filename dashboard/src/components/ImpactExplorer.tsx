import React from 'react';
import { RegionalIndicator } from '../services/api';

interface ImpactExplorerProps {
  indicators: RegionalIndicator[];
  hasPerturbation: boolean;
}

export const ImpactExplorer: React.FC<ImpactExplorerProps> = ({
  indicators,
  hasPerturbation,
}) => {
  return (
    <div className="rounded-xl border border-slate-800 bg-navy-900/80 p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">
            Regional Climate Impact Explorer (10-Day Model Indicators)
          </h3>
          <p className="text-xs text-slate-400">
            Model-derived indicative summaries only — not operational flood, disaster, or crop forecasts.
          </p>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400">
              <th className="py-2 pr-3">Region</th>
              <th className="py-2 px-3 text-right">Normal Mean Rain (mm/d)</th>
              {hasPerturbation && (
                <>
                  <th className="py-2 px-3 text-right">Scenario Mean Rain</th>
                  <th className="py-2 px-3 text-right">Rain Anomaly</th>
                  <th className="py-2 px-3 text-right">Δ Temp (K)</th>
                  <th className="py-2 px-3 text-right">Δ TCWV (kg/m²)</th>
                </>
              )}
              <th className="py-2 pl-3 text-right">Heavy-Rain Cell-Days (≥10mm)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {indicators.map((r) => (
              <tr key={r.region} className="hover:bg-slate-800/30">
                <td className="py-2 pr-3 font-medium text-slate-200">{r.region}</td>
                <td className="py-2 px-3 text-right text-cyan-300">{r.normalMeanTp.toFixed(2)}</td>
                {hasPerturbation && (
                  <>
                    <td className="py-2 px-3 text-right text-amber-300">
                      {r.perturbedMeanTp.toFixed(2)}
                    </td>
                    <td
                      className={`py-2 px-3 text-right font-medium ${
                        r.deltaTp >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {r.deltaTp >= 0 ? '+' : ''}
                      {r.deltaTp.toFixed(3)}
                    </td>
                    <td className="py-2 px-3 text-right text-slate-300">
                      {r.deltaT2m >= 0 ? '+' : ''}
                      {r.deltaT2m.toFixed(3)}
                    </td>
                    <td className="py-2 px-3 text-right text-slate-300">
                      {r.deltaTcwv >= 0 ? '+' : ''}
                      {r.deltaTcwv.toFixed(3)}
                    </td>
                  </>
                )}
                <td className="py-2 pl-3 text-right text-slate-200">
                  {hasPerturbation
                    ? `${r.normalHeavyCells} → ${r.perturbedHeavyCells}`
                    : r.normalHeavyCells}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
