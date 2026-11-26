import React from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { VariableMeta } from '../services/api';

interface ForecastChartsProps {
  normalRollout: number[][][][];
  perturbedRollout: number[][][][] | null;
  variable: VariableMeta;
}

function gridMean(grid: number[][]): number {
  let sum = 0;
  let cnt = 0;
  for (let i = 0; i < grid.length; i++) {
    for (let j = 0; j < grid[i].length; j++) {
      sum += grid[i][j];
      cnt++;
    }
  }
  return cnt ? sum / cnt : 0;
}

function gridMax(grid: number[][]): number {
  let m = -Infinity;
  for (let i = 0; i < grid.length; i++) {
    for (let j = 0; j < grid[i].length; j++) {
      if (grid[i][j] > m) m = grid[i][j];
    }
  }
  return m;
}

export const ForecastCharts: React.FC<ForecastChartsProps> = ({
  normalRollout,
  perturbedRollout,
  variable,
}) => {
  const data = normalRollout.map((stepGrid, idx) => {
    const nMean = gridMean(stepGrid[variable.index]);
    const pMean = perturbedRollout ? gridMean(perturbedRollout[idx][variable.index]) : null;
    const nRainMax = gridMax(stepGrid[1]);
    const pRainMax = perturbedRollout ? gridMax(perturbedRollout[idx][1]) : null;
    return {
      day: `Day ${idx + 1}`,
      normalMean: Number(nMean.toFixed(3)),
      perturbedMean: pMean !== null ? Number(pMean.toFixed(3)) : undefined,
      normalMaxRain: Number(nRainMax.toFixed(2)),
      perturbedMaxRain: pRainMax !== null ? Number(pRainMax.toFixed(2)) : undefined,
    };
  });

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <div className="rounded-xl border border-slate-800 bg-navy-900/80 p-4">
        <h3 className="mb-3 text-sm font-semibold text-slate-100">
          Domain Mean Evolution — {variable.label} ({variable.unit})
        </h3>
        <div className="h-60 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="day" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} domain={['auto', 'auto']} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: 12 }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="normalMean"
                name="Normal"
                stroke="#38bdf8"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
              {perturbedRollout && (
                <Line
                  type="monotone"
                  dataKey="perturbedMean"
                  name="Perturbed"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="rounded-xl border border-slate-800 bg-navy-900/80 p-4">
        <h3 className="mb-3 text-sm font-semibold text-slate-100">
          Peak Grid Rainfall Evolution (mm/day)
        </h3>
        <div className="h-60 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="day" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} domain={['auto', 'auto']} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: 12 }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="normalMaxRain"
                name="Normal Peak Rain"
                stroke="#22d3ee"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
              {perturbedRollout && (
                <Line
                  type="monotone"
                  dataKey="perturbedMaxRain"
                  name="Perturbed Peak Rain"
                  stroke="#fb7185"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
