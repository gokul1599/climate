import React, { useState } from 'react';
import { LAT, LON } from '../services/api';

interface IndiaGridMapProps {
  title: string;
  grid: number[][]; // [32, 36], index 0 = 6.5N (south) .. 31 = 37.5N (north)
  unit: string;
  digits?: number;
  diverging?: boolean;
  box?: [number, number, number, number];
}

function interpolateColor(t: number, diverging: boolean): string {
  const clamped = Math.max(0, Math.min(1, t));
  if (diverging) {
    // Blue (#2563eb) -> Slate (#1e293b) -> Red/Amber (#ef4444)
    if (clamped < 0.5) {
      const u = clamped * 2;
      const r = Math.round(37 + u * (30 - 37));
      const g = Math.round(99 + u * (41 - 99));
      const b = Math.round(235 + u * (59 - 235));
      return `rgb(${r},${g},${b})`;
    } else {
      const u = (clamped - 0.5) * 2;
      const r = Math.round(30 + u * (239 - 30));
      const g = Math.round(41 + u * (68 - 41));
      const b = Math.round(59 + u * (68 - 59));
      return `rgb(${r},${g},${b})`;
    }
  }
  // Sequential dark navy -> cyan -> emerald/yellow
  if (clamped < 0.5) {
    const u = clamped * 2;
    const r = Math.round(12 + u * (6 - 12));
    const g = Math.round(25 + u * (182 - 25));
    const b = Math.round(58 + u * (212 - 58));
    return `rgb(${r},${g},${b})`;
  }
  const u = (clamped - 0.5) * 2;
  const r = Math.round(6 + u * (250 - 6));
  const g = Math.round(182 + u * (204 - 182));
  const b = Math.round(212 + u * (21 - 212));
  return `rgb(${r},${g},${b})`;
}

export const IndiaGridMap: React.FC<IndiaGridMapProps> = ({
  title,
  grid,
  unit,
  digits = 2,
  diverging = false,
  box,
}) => {
  const [hover, setHover] = useState<{ lat: number; lon: number; val: number } | null>(null);

  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let i = 0; i < 32; i++) {
    for (let j = 0; j < 36; j++) {
      const v = grid[i][j];
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
  }

  let scaleMin = minVal;
  let scaleMax = maxVal;
  if (diverging) {
    const absMax = Math.max(Math.abs(minVal), Math.abs(maxVal), 1e-6);
    scaleMin = -absMax;
    scaleMax = absMax;
  } else if (Math.abs(scaleMax - scaleMin) < 1e-6) {
    scaleMax = scaleMin + 1;
  }

  // Render rows from North (i=31, 37.5N) at top to South (i=0, 6.5N) at bottom
  const rowsNorthToSouth = Array.from({ length: 32 }, (_, idx) => 31 - idx);

  return (
    <div className="rounded-xl border border-slate-800 bg-navy-900/80 p-4">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-100">{title}</h3>
        <span className="text-xs text-cyan-400">
          {hover
            ? `${hover.lat.toFixed(1)}°N, ${hover.lon.toFixed(1)}°E: ${hover.val.toFixed(digits)} ${unit}`
            : `Range: ${minVal.toFixed(digits)} to ${maxVal.toFixed(digits)} ${unit}`}
        </span>
      </div>

      <div className="relative overflow-hidden rounded-lg border border-slate-800 bg-slate-950 p-1">
        <div
          className="grid gap-0"
          style={{ gridTemplateColumns: 'repeat(36, minmax(0, 1fr))' }}
          onMouseLeave={() => setHover(null)}
        >
          {rowsNorthToSouth.map((i) =>
            LON.map((lon, j) => {
              const val = grid[i][j];
              const norm = (val - scaleMin) / (scaleMax - scaleMin);
              const inBox =
                box &&
                LAT[i] >= box[0] &&
                LAT[i] <= box[1] &&
                lon >= box[2] &&
                lon <= box[3];
              return (
                <div
                  key={`${i}-${j}`}
                  onMouseEnter={() => setHover({ lat: LAT[i], lon, val })}
                  style={{ backgroundColor: interpolateColor(norm, diverging) }}
                  className={`aspect-square w-full ${
                    inBox ? 'ring-1 ring-inset ring-amber-300/70' : ''
                  }`}
                />
              );
            }),
          )}
        </div>
      </div>

      <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400">
        <span>65.5°E (West)</span>
        <span>6.5°N – 37.5°N</span>
        <span>100.5°E (East)</span>
      </div>
    </div>
  );
};
