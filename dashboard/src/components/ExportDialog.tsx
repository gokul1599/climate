import React from 'react';
import { Download, FileJson, FileSpreadsheet, FileText, Image as ImageIcon, X } from 'lucide-react';
import { RegionalIndicator, ScenarioMeta, VariableMeta } from '../types/climate';
import {
  exportChartPng,
  exportForecastCsv,
  exportMetadataJson,
  exportRegionalIndicatorsCsv,
  exportSimulationPdfReport,
} from '../services/exporters';

interface ExportDialogProps {
  open: boolean;
  onClose: () => void;
  normalRollout: number[][][][] | null;
  perturbedRollout: number[][][][] | null;
  dayIdx: number;
  windowLabel: string;
  scenario: ScenarioMeta;
  variable: VariableMeta;
  inferenceMs: number | null;
  indicators: RegionalIndicator[];
  chartContainerRef?: React.RefObject<HTMLDivElement>;
}

export const ExportDialog: React.FC<ExportDialogProps> = ({
  open,
  onClose,
  normalRollout,
  perturbedRollout,
  dayIdx,
  windowLabel,
  scenario,
  variable,
  inferenceMs,
  indicators,
  chartContainerRef,
}) => {
  if (!open) return null;

  const activeRollout = perturbedRollout ?? normalRollout;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="export-dialog-title"
    >
      <div className="w-full max-w-md rounded-xl border border-slate-800 bg-[#0b1426] p-5 shadow-2xl">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Download className="h-4 w-4 text-cyan-400" aria-hidden="true" />
            <h2 id="export-dialog-title" className="text-sm font-semibold text-white">
              Export Forecast &amp; Scientific Report
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close export dialog"
            className="rounded-md p-1 text-slate-400 hover:bg-slate-800 hover:text-white"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-2.5 text-xs">
          <button
            type="button"
            disabled={!activeRollout}
            onClick={() => activeRollout && exportForecastCsv(activeRollout, dayIdx, scenario.key, false)}
            className="flex w-full items-center justify-between rounded-lg border border-slate-800 bg-slate-900/70 px-3.5 py-2.5 text-left text-slate-200 hover:border-cyan-500/40 hover:bg-slate-800/80 disabled:opacity-40"
          >
            <span className="flex items-center gap-2.5">
              <FileSpreadsheet className="h-4 w-4 text-cyan-400" />
              Export Selected Day CSV (Day {dayIdx + 1}, 6 channels)
            </span>
            <span className="text-[11px] text-slate-400">.csv</span>
          </button>

          <button
            type="button"
            disabled={!activeRollout}
            onClick={() => activeRollout && exportForecastCsv(activeRollout, dayIdx, scenario.key, true)}
            className="flex w-full items-center justify-between rounded-lg border border-slate-800 bg-slate-900/70 px-3.5 py-2.5 text-left text-slate-200 hover:border-cyan-500/40 hover:bg-slate-800/80 disabled:opacity-40"
          >
            <span className="flex items-center gap-2.5">
              <FileSpreadsheet className="h-4 w-4 text-emerald-400" />
              Export Full 10-Day Rollout CSV (11,520 rows)
            </span>
            <span className="text-[11px] text-slate-400">.csv</span>
          </button>

          <button
            type="button"
            disabled={!indicators.length}
            onClick={() => exportRegionalIndicatorsCsv(indicators, scenario.key)}
            className="flex w-full items-center justify-between rounded-lg border border-slate-800 bg-slate-900/70 px-3.5 py-2.5 text-left text-slate-200 hover:border-cyan-500/40 hover:bg-slate-800/80 disabled:opacity-40"
          >
            <span className="flex items-center gap-2.5">
              <FileSpreadsheet className="h-4 w-4 text-amber-400" />
              Export Regional Indicators CSV (6 regions)
            </span>
            <span className="text-[11px] text-slate-400">.csv</span>
          </button>

          <button
            type="button"
            onClick={() =>
              exportMetadataJson(
                {
                  exported_at: new Date().toISOString(),
                  model: 'MonsoonVerse-ClimResCNN',
                  window: windowLabel,
                  scenario,
                  focus_day: dayIdx + 1,
                  focus_variable: variable.key,
                  inference_ms: inferenceMs,
                  regional_indicators: indicators,
                },
                `monsoonverse_metadata_${scenario.key}.json`,
              )
            }
            className="flex w-full items-center justify-between rounded-lg border border-slate-800 bg-slate-900/70 px-3.5 py-2.5 text-left text-slate-200 hover:border-cyan-500/40 hover:bg-slate-800/80"
          >
            <span className="flex items-center gap-2.5">
              <FileJson className="h-4 w-4 text-violet-400" />
              Export Simulation Metadata JSON
            </span>
            <span className="text-[11px] text-slate-400">.json</span>
          </button>

          <button
            type="button"
            disabled={!chartContainerRef?.current}
            onClick={async () => {
              if (chartContainerRef?.current) {
                await exportChartPng(
                  chartContainerRef.current,
                  `monsoonverse_chart_${scenario.key}_${variable.key}.png`,
                );
              }
            }}
            className="flex w-full items-center justify-between rounded-lg border border-slate-800 bg-slate-900/70 px-3.5 py-2.5 text-left text-slate-200 hover:border-cyan-500/40 hover:bg-slate-800/80 disabled:opacity-40"
          >
            <span className="flex items-center gap-2.5">
              <ImageIcon className="h-4 w-4 text-cyan-300" />
              Export Forecast Chart Snapshot
            </span>
            <span className="text-[11px] text-slate-400">.png</span>
          </button>

          <button
            type="button"
            disabled={!indicators.length}
            onClick={() =>
              exportSimulationPdfReport({
                windowLabel,
                scenario,
                variable,
                dayIdx,
                inferenceMs,
                indicators,
                hasPerturbation: perturbedRollout !== null,
              })
            }
            className="flex w-full items-center justify-between rounded-lg border border-cyan-500/40 bg-cyan-500/15 px-3.5 py-2.5 text-left font-medium text-cyan-200 hover:bg-cyan-500/25 disabled:opacity-40"
          >
            <span className="flex items-center gap-2.5">
              <FileText className="h-4 w-4 text-cyan-300" />
              Generate Simulation PDF Report
            </span>
            <span className="text-[11px] text-cyan-300">.pdf</span>
          </button>
        </div>
      </div>
    </div>
  );
};
