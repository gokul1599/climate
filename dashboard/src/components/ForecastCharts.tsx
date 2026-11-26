import React from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { computeDomainDaySeries, computeRegionalTimeSeries, REGIONS } from '../services/api';
import { GridCellSelection, VariableMeta } from '../types/climate';

interface ForecastChartsProps {
  normalRollout: number[][][][];
  perturbedRollout: number[][][][] | null;
  variable: VariableMeta;
  selectedRegionName: string;
  selectedCell: GridCellSelection | null;
  onClearCell?: () => void;
  chartRef?: React.RefObject<HTMLDivElement>;
}

export const ForecastCharts: React.FC<ForecastChartsProps> = ({
  normalRollout,
  perturbedRollout,
  variable,
  selectedRegionName,
  selectedCell,
  onClearCell,
  chartRef,
}) => {
  const domainSeries = computeDomainDaySeries(normalRollout, perturbedRollout, variable.index);
  const regionObj = REGIONS.find((r) => r.name === selectedRegionName) ?? REGIONS[0];
  const localSeries = computeRegionalTimeSeries(
    normalRollout,
    perturbedRollout,
    variable.index,
    regionObj.box,
    selectedCell,
  );

  const envelopeData = domainSeries.map((d) => ({
    ...d,
    range: [d.normalMin, d.normalMax] as [number, number],
  }));

  return (
    <div ref={chartRef} className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      {/* Chart 1: Domain Mean + Min/Max Envelope */}
      <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4">
        <h3 className="mb-1 text-xs font-semibold text-slate-100 sm:text-sm">
          Domain Mean &amp; Spread — {variable.label} ({variable.unit})
        </h3>
        <p className="mb-3 text-[11px] text-slate-400">
          32×36 domain mean alongside spatial min/max envelope across Days 1–10.
        </p>
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={envelopeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="label" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} domain={['auto', 'auto']} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: 11 }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Area
                type="monotone"
                dataKey="range"
                name="Normal Min–Max"
                fill="#0ea5e9"
                fillOpacity={0.14}
                stroke="none"
              />
              <Line
                type="monotone"
                dataKey="normalMean"
                name="Normal Mean"
                stroke="#38bdf8"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
              {perturbedRollout && (
                <Line
                  type="monotone"
                  dataKey="perturbedMean"
                  name="Perturbed Mean"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 2: Precipitation Mean & Peak */}
      <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4">
        <h3 className="mb-1 text-xs font-semibold text-slate-100 sm:text-sm">
          Precipitation Intensity — Peak &amp; Mean (mm/day)
        </h3>
        <p className="mb-3 text-[11px] text-slate-400">
          Daily grid maximum and domain average precipitation evolution.
        </p>
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={domainSeries}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="label" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} domain={['auto', 'auto']} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: 11 }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line
                type="monotone"
                dataKey="normalPeakRain"
                name="Normal Peak Rain"
                stroke="#22d3ee"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
              {perturbedRollout && (
                <Line
                  type="monotone"
                  dataKey="perturbedPeakRain"
                  name="Perturbed Peak Rain"
                  stroke="#fb7185"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              )}
              <Line
                type="monotone"
                dataKey="normalMeanRain"
                name="Normal Mean Rain"
                stroke="#10b981"
                strokeWidth={1.5}
                strokeDasharray="4 2"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 3: Selected Region or Grid Cell Time Series */}
      <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4">
        <div className="mb-1 flex items-center justify-between gap-2">
          <h3 className="text-xs font-semibold text-slate-100 sm:text-sm">
            {selectedCell
              ? `Grid Cell (${selectedCell.lat.toFixed(1)}°N, ${selectedCell.lon.toFixed(1)}°E) — ${variable.shortLabel}`
              : `Regional Mean (${regionObj.name}) — ${variable.shortLabel}`}
          </h3>
          {selectedCell && onClearCell && (
            <button
              type="button"
              onClick={onClearCell}
              className="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-300 hover:bg-slate-800"
            >
              Use Region Mean
            </button>
          )}
        </div>
        <p className="mb-3 text-[11px] text-slate-400">
          {selectedCell
            ? 'Time series at the clicked 1°×1° grid cell.'
            : 'Click any cell on the Forecast Map to inspect its individual time series.'}
        </p>
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={localSeries}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="label" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} domain={['auto', 'auto']} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: 11 }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line
                type="monotone"
                dataKey="normalValue"
                name={`Normal (${variable.unit})`}
                stroke="#38bdf8"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
              {perturbedRollout && (
                <Line
                  type="monotone"
                  dataKey="perturbedValue"
                  name={`Perturbed (${variable.unit})`}
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 4: Anomaly Evolution (Perturbed - Normal) */}
      <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4">
        <h3 className="mb-1 text-xs font-semibold text-slate-100 sm:text-sm">
          Scenario Anomaly Evolution (Perturbed − Normal)
        </h3>
        <p className="mb-3 text-[11px] text-slate-400">
          Domain-wide &amp; local {variable.shortLabel} response trajectory ({variable.unit}).
        </p>
        <div className="h-56 w-full">
          {perturbedRollout ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={domainSeries.map((d, i) => ({
                  label: d.label,
                  domainAnomaly: d.anomalyMean ?? 0,
                  localAnomaly: localSeries[i]?.anomaly ?? 0,
                }))}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="label" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} domain={['auto', 'auto']} />
                <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 3" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: 11 }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line
                  type="monotone"
                  dataKey="domainAnomaly"
                  name="Domain Anomaly"
                  stroke="#a78bfa"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="localAnomaly"
                  name={selectedCell ? 'Cell Anomaly' : `${regionObj.name} Anomaly`}
                  stroke="#fb7185"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-slate-800 text-xs text-slate-500">
              Select a perturbation scenario (e.g., Warm Bay of Bengal) to view anomaly evolution.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
