import { jsPDF } from 'jspdf';
import { LAT, LON, VARIABLES } from './api';
import { RegionalIndicator, SavedSimulationRun, ScenarioMeta, VariableMeta } from '../types/climate';

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function exportForecastCsv(
  rollout: number[][][][],
  dayIdx: number,
  scenarioKey: string,
  allDays = false,
): void {
  const header = 'day,lat,lon,t2m_K,tp_mm_day,u10_ms,v10_ms,msl_Pa,tcwv_kg_m2';
  const lines = [header];
  const dayList = allDays
    ? Array.from({ length: rollout.length }, (_, k) => k)
    : [dayIdx];

  for (const d of dayList) {
    for (let i = 0; i < 32; i++) {
      for (let j = 0; j < 36; j++) {
        const vals = VARIABLES.map((v) => rollout[d][v.index][i][j].toFixed(4));
        lines.push(`${d + 1},${LAT[i].toFixed(1)},${LON[j].toFixed(1)},${vals.join(',')}`);
      }
    }
  }
  const suffix = allDays ? '10day' : `day${dayIdx + 1}`;
  triggerDownload(
    new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' }),
    `monsoonverse_${scenarioKey}_${suffix}.csv`,
  );
}

export function exportMetadataJson(payload: Record<string, unknown>, filename: string): void {
  triggerDownload(
    new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }),
    filename,
  );
}

export function exportRegionalIndicatorsCsv(
  indicators: RegionalIndicator[],
  scenarioKey: string,
): void {
  const header =
    'region,grid_cells,normal_tp_mm_d,scenario_tp_mm_d,delta_tp_mm_d,normal_heavy_cells,scenario_heavy_cells,delta_t2m_K,delta_tcwv_kg_m2,delta_wind_ms';
  const rows = indicators.map((r) =>
    [
      `"${r.region}"`,
      r.gridCells,
      r.normalMeanTp.toFixed(4),
      r.perturbedMeanTp.toFixed(4),
      r.deltaTp.toFixed(4),
      r.normalHeavyCells,
      r.perturbedHeavyCells,
      r.deltaT2m.toFixed(4),
      r.deltaTcwv.toFixed(4),
      r.deltaWind.toFixed(4),
    ].join(','),
  );
  triggerDownload(
    new Blob([[header, ...rows].join('\n')], { type: 'text/csv;charset=utf-8' }),
    `monsoonverse_regional_impact_${scenarioKey}.csv`,
  );
}

export function exportSavedRunJson(run: SavedSimulationRun): void {
  exportMetadataJson(
    run as unknown as Record<string, unknown>,
    `monsoonverse_saved_run_${run.id}.json`,
  );
}

export async function exportChartPng(container: HTMLElement | null, filename: string): Promise<boolean> {
  if (!container) return false;
  const svg = container.querySelector('svg');
  if (!svg) return false;

  const serializer = new XMLSerializer();
  const svgStr = serializer.serializeToString(svg);
  const svgBlob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(svgBlob);

  return new Promise<boolean>((resolve) => {
    const img = new Image();
    img.onload = () => {
      const rect = svg.getBoundingClientRect();
      const canvas = document.createElement('canvas');
      canvas.width = Math.max(rect.width * 2, 800);
      canvas.height = Math.max(rect.height * 2, 420);
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        URL.revokeObjectURL(url);
        resolve(false);
        return;
      }
      ctx.fillStyle = '#0b1426';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((pngBlob) => {
        URL.revokeObjectURL(url);
        if (!pngBlob) {
          resolve(false);
          return;
        }
        triggerDownload(pngBlob, filename);
        resolve(true);
      }, 'image/png');
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(false);
    };
    img.src = url;
  });
}

export function exportSimulationPdfReport(params: {
  windowLabel: string;
  scenario: ScenarioMeta;
  variable: VariableMeta;
  dayIdx: number;
  inferenceMs: number | null;
  indicators: RegionalIndicator[];
  hasPerturbation: boolean;
}): void {
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  doc.setFontSize(16);
  doc.setTextColor(14, 165, 233);
  doc.text('MONSOONVERSE — Simulation Report', 15, 18);

  doc.setFontSize(9.5);
  doc.setTextColor(71, 85, 105);
  doc.text(`Generated: ${new Date().toISOString()} · Model: MonsoonVerse-ClimResCNN (ERA5 2005–2021, 32×36)`, 15, 24);

  doc.setTextColor(15, 23, 42);
  doc.setFontSize(11);
  doc.text('1. Simulation Configuration & Provenance', 15, 34);
  doc.setFontSize(9.5);
  doc.text(`Initial Window: ${params.windowLabel}`, 15, 41);
  doc.text(`Scenario: ${params.scenario.label} (${params.scenario.badge})`, 15, 46.5);
  doc.text(`Perturbation: ${params.scenario.description}`, 15, 52, { maxWidth: 180 });
  doc.text(
    `Selected Focus: Day ${params.dayIdx + 1}/10 · ${params.variable.label} (${params.variable.unit}) · CPU Inference: ${
      params.inferenceMs !== null ? `${params.inferenceMs.toFixed(1)} ms` : 'N/A'
    }`,
    15,
    62,
  );

  doc.setFontSize(11);
  doc.text('2. Regional Model-Derived Indicators (10-Day Aggregate)', 15, 74);
  let y = 82;
  doc.setFontSize(8.5);
  doc.text('Region', 15, y);
  doc.text('Norm Rain', 75, y);
  doc.text('Scen Rain', 98, y);
  doc.text('Delta Rain', 120, y);
  doc.text('Delta T2m(K)', 144, y);
  doc.text('Heavy Cells', 172, y);
  y += 5;

  for (const r of params.indicators) {
    doc.text(r.region, 15, y);
    doc.text(r.normalMeanTp.toFixed(2), 75, y);
    doc.text(r.perturbedMeanTp.toFixed(2), 98, y);
    doc.text(`${r.deltaTp >= 0 ? '+' : ''}${r.deltaTp.toFixed(3)}`, 120, y);
    doc.text(`${r.deltaT2m >= 0 ? '+' : ''}${r.deltaT2m.toFixed(3)}`, 144, y);
    doc.text(
      params.hasPerturbation ? `${r.normalHeavyCells} -> ${r.perturbedHeavyCells}` : String(r.normalHeavyCells),
      172,
      y,
    );
    y += 5.5;
  }

  y += 6;
  doc.setFontSize(10.5);
  doc.text('3. Scientific Limitations & Disclaimer', 15, y);
  y += 6;
  doc.setFontSize(8.5);
  doc.setTextColor(71, 85, 105);
  doc.text(
    'MonsoonVerse is a compact 1°×1° regional research prototype trained on 2005–2021 ERA5 data. Outputs are deterministic model-derived scenario comparisons, NOT validated flood alerts, crop-loss diagnoses, or operational meteorological warnings.',
    15,
    y,
    { maxWidth: 180 },
  );

  doc.save(`monsoonverse_report_${params.scenario.key}_day${params.dayIdx + 1}.pdf`);
}
