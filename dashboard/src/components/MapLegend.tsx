import React from 'react';
import { PaletteKind } from '../types/climate';

export function samplePaletteColor(t: number, palette: PaletteKind): string {
  const c = Math.max(0, Math.min(1, t));
  if (palette === 'diverging') {
    // Blue (#2563eb) -> Slate (#1e293b) -> Red/Amber (#ef4444)
    if (c < 0.5) {
      const u = c * 2;
      return `rgb(${Math.round(37 + u * (30 - 37))},${Math.round(99 + u * (41 - 99))},${Math.round(
        235 + u * (59 - 235),
      )})`;
    }
    const u = (c - 0.5) * 2;
    return `rgb(${Math.round(30 + u * (239 - 30))},${Math.round(41 + u * (68 - 41))},${Math.round(
      59 + u * (68 - 59),
    )})`;
  }
  if (palette === 'thermal') {
    // Deep indigo -> teal -> amber -> crimson
    if (c < 0.35) {
      const u = c / 0.35;
      return `rgb(${Math.round(15 + u * 16)},${Math.round(23 + u * 95)},${Math.round(66 + u * 84)})`;
    }
    if (c < 0.7) {
      const u = (c - 0.35) / 0.35;
      return `rgb(${Math.round(31 + u * 214)},${Math.round(118 + u * 40)},${Math.round(150 - u * 139)})`;
    }
    const u = (c - 0.7) / 0.3;
    return `rgb(${Math.round(245 - u * 25)},${Math.round(158 - u * 120)},${Math.round(11 + u * 27)})`;
  }
  if (palette === 'precip') {
    // Dry slate -> cyan -> deep blue -> bright violet
    if (c < 0.35) {
      const u = c / 0.35;
      return `rgb(${Math.round(12 + u * 2)},${Math.round(22 + u * 94)},${Math.round(44 + u * 124)})`;
    }
    if (c < 0.7) {
      const u = (c - 0.35) / 0.35;
      return `rgb(${Math.round(14 + u * 45)},${Math.round(116 + u * 54)},${Math.round(168 + u * 78)})`;
    }
    const u = (c - 0.7) / 0.3;
    return `rgb(${Math.round(59 + u * 133)},${Math.round(170 - u * 60)},${Math.round(246 - u * 10)})`;
  }
  if (palette === 'wind') {
    // Teal-blue negative -> neutral slate -> amber positive
    if (c < 0.5) {
      const u = c * 2;
      return `rgb(${Math.round(14 + u * 16)},${Math.round(116 - u * 75)},${Math.round(144 - u * 85)})`;
    }
    const u = (c - 0.5) * 2;
    return `rgb(${Math.round(30 + u * 215)},${Math.round(41 + u * 117)},${Math.round(59 - u * 48)})`;
  }
  if (palette === 'pressure') {
    // Cool navy -> sky -> warm sand
    if (c < 0.5) {
      const u = c * 2;
      return `rgb(${Math.round(15 + u * 41)},${Math.round(23 + u * 166)},${Math.round(42 + u * 206)})`;
    }
    const u = (c - 0.5) * 2;
    return `rgb(${Math.round(56 + u * 194)},${Math.round(189 + u * 15)},${Math.round(248 - u * 170)})`;
  }
  // moisture: slate -> emerald -> cyan
  if (c < 0.5) {
    const u = c * 2;
    return `rgb(${Math.round(12 + u * 4)},${Math.round(25 + u * 140)},${Math.round(50 + u * 80)})`;
  }
  const u = (c - 0.5) * 2;
  return `rgb(${Math.round(16 + u * 18)},${Math.round(165 + u * 46)},${Math.round(130 + u * 108)})`;
}

interface MapLegendProps {
  minVal: number;
  maxVal: number;
  unit: string;
  digits?: number;
  palette: PaletteKind;
}

export const MapLegend: React.FC<MapLegendProps> = ({
  minVal,
  maxVal,
  unit,
  digits = 2,
  palette,
}) => {
  const stops = [0, 0.25, 0.5, 0.75, 1].map((t) => samplePaletteColor(t, palette));
  const midVal = (minVal + maxVal) / 2;

  return (
    <div className="mt-2.5 space-y-1">
      <div
        className="h-2 w-full rounded-full border border-slate-800"
        style={{ background: `linear-gradient(to right, ${stops.join(', ')})` }}
        aria-hidden="true"
      />
      <div className="flex items-center justify-between text-[10px] font-mono text-slate-400">
        <span>
          {minVal.toFixed(digits)} {unit}
        </span>
        <span>{midVal.toFixed(digits)}</span>
        <span>
          {maxVal.toFixed(digits)} {unit}
        </span>
      </div>
    </div>
  );
};
