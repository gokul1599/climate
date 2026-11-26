import React, { useMemo, useState } from 'react';
import { RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';
import { LAT, LON } from '../services/api';
import { GridCellSelection, PaletteKind } from '../types/climate';
import { MapLegend, samplePaletteColor } from './MapLegend';

// Simplified India + Sri Lanka reference coastline polylines in lon/lat degrees for 1x1 deg orientation context
const INDIA_COAST_REF: [number, number][] = [
  [68.2, 23.8], [69.0, 22.8], [70.2, 22.3], [69.1, 22.2], [70.5, 20.9], [72.2, 21.6],
  [72.8, 21.1], [72.8, 19.0], [73.4, 17.0], [74.1, 14.8], [74.9, 12.9], [76.2, 10.0],
  [77.5, 8.1], [78.1, 8.8], [79.8, 10.3], [80.3, 13.1], [80.1, 15.3], [82.3, 16.9],
  [83.3, 17.7], [85.2, 19.8], [86.9, 20.7], [88.1, 21.7], [89.0, 21.8],
];

const INDIA_NORTH_REF: [number, number][] = [
  [68.2, 23.8], [70.8, 27.8], [74.5, 31.0], [74.0, 34.4], [76.5, 35.8], [78.5, 34.6],
  [79.2, 31.2], [81.0, 30.1], [84.5, 27.5], [88.2, 27.9], [89.5, 26.8], [92.0, 27.6],
  [96.8, 28.4], [95.0, 26.2], [94.0, 23.8], [92.8, 22.0], [89.0, 21.8],
];

const SRI_LANKA_REF: [number, number][] = [
  [79.9, 9.7], [81.2, 8.6], [81.8, 7.2], [80.6, 6.0], [79.8, 7.0], [79.9, 9.7],
];

function toSvgPath(pts: [number, number][]): string {
  return pts
    .map(([lon, lat], idx) => {
      const x = ((lon - 65.0) / 36.0) * 360;
      const y = ((38.0 - lat) / 32.0) * 320;
      return `${idx === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

interface ForecastMapProps {
  title: string;
  subtitle?: string;
  grid: number[][]; // [32, 36]
  unit: string;
  digits?: number;
  palette: PaletteKind;
  diverging?: boolean;
  scenarioBox?: [number, number, number, number];
  regionBox?: [number, number, number, number];
  selectedCell?: GridCellSelection | null;
  onSelectCell?: (cell: GridCellSelection) => void;
  showGeoOverlay?: boolean;
  showGridLines?: boolean;
  compact?: boolean;
}

export const ForecastMap: React.FC<ForecastMapProps> = ({
  title,
  subtitle,
  grid,
  unit,
  digits = 2,
  palette,
  diverging = false,
  scenarioBox,
  regionBox,
  selectedCell,
  onSelectCell,
  showGeoOverlay = true,
  showGridLines = false,
  compact = false,
}) => {
  const [hover, setHover] = useState<{ lat: number; lon: number; val: number } | null>(null);
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const { scaleMin, scaleMax, minVal, maxVal } = useMemo(() => {
    let mn = Infinity;
    let mx = -Infinity;
    for (let i = 0; i < 32; i++) {
      for (let j = 0; j < 36; j++) {
        const v = grid[i][j];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    let sMin = mn;
    let sMax = mx;
    if (diverging) {
      const absMax = Math.max(Math.abs(mn), Math.abs(mx), 1e-6);
      sMin = -absMax;
      sMax = absMax;
    } else if (Math.abs(sMax - sMin) < 1e-6) {
      sMax = sMin + 1;
    }
    return { scaleMin: sMin, scaleMax: sMax, minVal: mn, maxVal: mx };
  }, [grid, diverging]);

  const rowsNorthToSouth = useMemo(() => Array.from({ length: 32 }, (_, idx) => 31 - idx), []);
  const activePalette: PaletteKind = diverging ? 'diverging' : palette;

  const coastPath = useMemo(() => toSvgPath(INDIA_COAST_REF), []);
  const northPath = useMemo(() => toSvgPath(INDIA_NORTH_REF), []);
  const lankaPath = useMemo(() => toSvgPath(SRI_LANKA_REF), []);

  return (
    <div
      className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-3.5"
      role="region"
      aria-label={`${title} spatial climate grid map`}
    >
      <div className="mb-2 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="text-xs font-semibold text-slate-100 sm:text-sm">{title}</h3>
          {subtitle && <p className="text-[11px] text-slate-400">{subtitle}</p>}
        </div>
        <div className="flex items-center gap-1.5">
          <span className="rounded bg-slate-900/90 px-2 py-0.5 font-mono text-[11px] text-cyan-300 border border-slate-800">
            {hover
              ? `${hover.lat.toFixed(1)}°N, ${hover.lon.toFixed(1)}°E: ${hover.val.toFixed(digits)} ${unit}`
              : `${minVal.toFixed(digits)} .. ${maxVal.toFixed(digits)} ${unit}`}
          </span>
          {!compact && (
            <div className="inline-flex items-center rounded-md border border-slate-800 bg-slate-950/90 p-0.5">
              <button
                type="button"
                onClick={() => setZoom((z) => Math.min(2.4, Number((z + 0.4).toFixed(1))))}
                aria-label="Zoom in map"
                className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-white"
              >
                <ZoomIn className="h-3 w-3" />
              </button>
              <button
                type="button"
                onClick={() => {
                  setZoom((z) => {
                    const next = Math.max(1, Number((z - 0.4).toFixed(1)));
                    if (next === 1) setPan({ x: 0, y: 0 });
                    return next;
                  });
                }}
                aria-label="Zoom out map"
                className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-white"
              >
                <ZoomOut className="h-3 w-3" />
              </button>
              {zoom > 1 && (
                <button
                  type="button"
                  onClick={() => {
                    setZoom(1);
                    setPan({ x: 0, y: 0 });
                  }}
                  aria-label="Reset map zoom"
                  className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-white"
                >
                  <RotateCcw className="h-3 w-3" />
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {zoom > 1 && !compact && (
        <div className="mb-1.5 flex items-center justify-end gap-2 text-[10px] text-slate-400">
          <span>Zoom {zoom.toFixed(1)}×</span>
          <button
            type="button"
            onClick={() => setPan((p) => ({ ...p, x: Math.min(80, p.x + 25) }))}
            className="rounded border border-slate-800 px-1.5 py-0.5 hover:bg-slate-800"
          >
            ←
          </button>
          <button
            type="button"
            onClick={() => setPan((p) => ({ ...p, x: Math.max(-80, p.x - 25) }))}
            className="rounded border border-slate-800 px-1.5 py-0.5 hover:bg-slate-800"
          >
            →
          </button>
          <button
            type="button"
            onClick={() => setPan((p) => ({ ...p, y: Math.min(80, p.y + 25) }))}
            className="rounded border border-slate-800 px-1.5 py-0.5 hover:bg-slate-800"
          >
            ↑
          </button>
          <button
            type="button"
            onClick={() => setPan((p) => ({ ...p, y: Math.max(-80, p.y - 25) }))}
            className="rounded border border-slate-800 px-1.5 py-0.5 hover:bg-slate-800"
          >
            ↓
          </button>
        </div>
      )}

      <div className="relative overflow-hidden rounded-lg border border-slate-800 bg-slate-950 p-1">
        <div
          className="relative transition-transform duration-150"
          style={{ transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)` }}
        >
          <div
            className="grid gap-0"
            style={{ gridTemplateColumns: 'repeat(36, minmax(0, 1fr))' }}
            onMouseLeave={() => setHover(null)}
          >
            {rowsNorthToSouth.map((i) =>
              LON.map((lon, j) => {
                const val = grid[i][j];
                const norm = (val - scaleMin) / (scaleMax - scaleMin);
                const inScenario =
                  scenarioBox &&
                  LAT[i] >= scenarioBox[0] &&
                  LAT[i] <= scenarioBox[1] &&
                  lon >= scenarioBox[2] &&
                  lon <= scenarioBox[3];
                const inRegion =
                  regionBox &&
                  LAT[i] >= regionBox[0] &&
                  LAT[i] <= regionBox[1] &&
                  lon >= regionBox[2] &&
                  lon <= regionBox[3];
                const isSelected =
                  selectedCell && selectedCell.latIdx === i && selectedCell.lonIdx === j;

                return (
                  <button
                    key={`${i}-${j}`}
                    type="button"
                    aria-label={`${LAT[i].toFixed(1)}N ${lon.toFixed(1)}E: ${val.toFixed(digits)} ${unit}`}
                    onMouseEnter={() => setHover({ lat: LAT[i], lon, val })}
                    onClick={() =>
                      onSelectCell?.({ latIdx: i, lonIdx: j, lat: LAT[i], lon })
                    }
                    style={{ backgroundColor: samplePaletteColor(norm, activePalette) }}
                    className={`aspect-square w-full cursor-crosshair p-0 border-0 ${
                      showGridLines ? 'ring-[0.5px] ring-inset ring-slate-950/30' : ''
                    } ${inScenario ? 'ring-1 ring-inset ring-amber-300/80' : ''} ${
                      inRegion ? 'ring-1 ring-inset ring-emerald-300/85' : ''
                    } ${isSelected ? 'ring-2 ring-inset ring-white' : ''}`}
                  />
                );
              }),
            )}
          </div>

          {showGeoOverlay && (
            <svg
              viewBox="0 0 360 320"
              className="pointer-events-none absolute inset-0 h-full w-full"
              aria-hidden="true"
            >
              <path
                d={coastPath}
                fill="none"
                stroke="rgba(248, 250, 252, 0.55)"
                strokeWidth="1.2"
              />
              <path
                d={northPath}
                fill="none"
                stroke="rgba(248, 250, 252, 0.35)"
                strokeDasharray="3 2"
                strokeWidth="1.0"
              />
              <path
                d={lankaPath}
                fill="none"
                stroke="rgba(248, 250, 252, 0.5)"
                strokeWidth="1.0"
              />
            </svg>
          )}
        </div>
      </div>

      <MapLegend
        minVal={scaleMin}
        maxVal={scaleMax}
        unit={unit}
        digits={digits}
        palette={activePalette}
      />

      <div className="mt-1.5 flex items-center justify-between text-[10px] text-slate-400">
        <span>65.5°E (West)</span>
        <span>6.5°N – 37.5°N · 1°×1° Grid</span>
        <span>100.5°E (East)</span>
      </div>
    </div>
  );
};
