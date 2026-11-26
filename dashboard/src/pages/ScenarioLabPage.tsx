import React, { useState } from 'react';
import { AlertCircle, CheckCircle2, FileJson, FlaskConical, Play, Upload } from 'lucide-react';
import {
  CustomExperimentPayload,
  CustomPerturbationConfig,
  GridCellSelection,
  RegionalIndicator,
  SampleWindow,
  ScenarioKey,
  ScenarioMeta,
  VariableIndex,
  VariableMeta,
} from '../types/climate';
import { buildPerturbation, validateExperimentJsonPayload } from '../services/api';
import { ForecastControls } from '../components/ForecastControls';
import { ScenarioComparison } from '../components/ScenarioComparison';
import { RegionalImpactTable } from '../components/RegionalImpactTable';
import { EmptyState } from '../components/EmptyState';
import { LoadingSkeleton } from '../components/LoadingSkeleton';

interface ScenarioLabPageProps {
  windows: SampleWindow[];
  windowIdx: number;
  onChangeWindow: (idx: number) => void;
  varIdx: VariableIndex;
  onChangeVar: (v: VariableIndex) => void;
  variable: VariableMeta;
  scenario: ScenarioKey;
  scenarioMeta: ScenarioMeta;
  onChangeScenario: (s: ScenarioKey) => void;
  dayIdx: number;
  onChangeDay: (d: number) => void;
  normalRollout: number[][][][] | null;
  perturbedRollout: number[][][][] | null;
  customPert: CustomPerturbationConfig;
  onChangeCustomPert: (cfg: CustomPerturbationConfig) => void;
  indicators: RegionalIndicator[];
  selectedRegion: string;
  onSelectRegion: (r: string) => void;
  selectedCell: GridCellSelection | null;
  onSelectCell: (c: GridCellSelection | null) => void;
  loading: boolean;
  dbConnected: boolean;
  showGeoOverlay: boolean;
  showGridLines: boolean;
  onRun: () => void;
  onRunCustomJson?: (payload: CustomExperimentPayload, isSyntheticTemplate: boolean) => Promise<void>;
  modelName?: string | null;
  simSourceNote?: string;
  onReset: () => void;
  onOpenSave: () => void;
  onOpenExport: () => void;
  chartRef?: React.RefObject<HTMLDivElement>;
}

export const ScenarioLabPage: React.FC<ScenarioLabPageProps> = ({
  windows,
  windowIdx,
  onChangeWindow,
  varIdx,
  onChangeVar,
  variable,
  scenario,
  scenarioMeta,
  onChangeScenario,
  dayIdx,
  onChangeDay,
  normalRollout,
  perturbedRollout,
  customPert,
  onChangeCustomPert,
  indicators,
  selectedRegion,
  onSelectRegion,
  selectedCell,
  onSelectCell,
  loading,
  dbConnected,
  showGeoOverlay,
  showGridLines,
  onRun,
  onRunCustomJson,
  modelName,
  simSourceNote,
  onReset,
  onOpenSave,
  onOpenExport,
  chartRef,
}) => {
  const [jsonOpen, setJsonOpen] = useState(false);
  const [jsonText, setJsonText] = useState('');
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [isSyntheticTemplate, setIsSyntheticTemplate] = useState(false);
  const [validatedSummary, setValidatedSummary] = useState<string | null>(null);

  const loadTemplateFromActiveWindow = () => {
    const win = windows[windowIdx];
    if (!win) return;
    const pert = buildPerturbation(
      scenario === 'normal' ? 'warm_ocean' : scenario,
      win.init_states[2],
      customPert,
    );
    const tpl: CustomExperimentPayload = {
      init_states: win.init_states,
      doy_sincos: win.doy_sincos,
      steps: 10,
      perturbation: pert,
    };
    setJsonText(JSON.stringify(tpl));
    setIsSyntheticTemplate(true);
    setJsonError(null);
    setValidatedSummary('Loaded Synthetic Demo Template [init_states (3x6x32x36), doy_sincos, steps=10, perturbation]');
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const txt = await file.text();
    setJsonText(txt);
    setIsSyntheticTemplate(false);
    setJsonError(null);
    setValidatedSummary(`Loaded file "${file.name}" (${(file.size / 1024).toFixed(1)} KB). Click Validate & Run.`);
  };

  const handleValidateAndRunJson = async () => {
    const res = validateExperimentJsonPayload(jsonText);
    if (!res.ok) {
      setJsonError(res.error);
      setValidatedSummary(null);
      return;
    }
    setJsonError(null);
    const p = res.payload;
    setValidatedSummary(
      `Validated R1 JSON: init=[3,6,32,36], doy=[${p.doy_sincos[0].toFixed(3)}, ${p.doy_sincos[1].toFixed(3)}], steps=${p.steps}, pert=${p.perturbation ? '[6,32,36]' : 'null'}`,
    );
    if (onRunCustomJson) {
      await onRunCustomJson(p, isSyntheticTemplate);
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="sr-only">Scenario Simulation Lab Controls and Maps</h2>

      {/* Active Provenance & Run Mode Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-800/90 bg-navy-900/75 px-4 py-2.5 text-xs">
        <div className="flex flex-wrap items-center gap-2 text-slate-300">
          <FlaskConical className="h-4 w-4 text-cyan-400" />
          <span className="font-semibold text-white">Active Model:</span>
          <span className="rounded bg-cyan-500/15 px-2 py-0.5 font-mono text-[11px] text-cyan-300">
            {modelName ?? 'MonsoonVerse-ClimResCNN'}
          </span>
          {simSourceNote && <span className="text-slate-400">· {simSourceNote}</span>}
        </div>
        <button
          type="button"
          onClick={() => setJsonOpen((o) => !o)}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-cyan-300 hover:border-cyan-500/50 hover:bg-slate-800"
        >
          <FileJson className="h-3.5 w-3.5" />
          {jsonOpen ? 'Hide Custom JSON Experiment' : 'Custom JSON Experiment Input'}
        </button>
      </div>

      <ForecastControls
        windows={windows}
        windowIdx={windowIdx}
        onChangeWindow={onChangeWindow}
        varIdx={varIdx}
        onChangeVar={onChangeVar}
        scenario={scenario}
        onChangeScenario={onChangeScenario}
        dayIdx={dayIdx}
        onChangeDay={onChangeDay}
        loading={loading}
        canSave={Boolean(normalRollout)}
        dbConnected={dbConnected}
        onRun={onRun}
        onReset={onReset}
        onOpenSave={onOpenSave}
        onOpenExport={onOpenExport}
      />

      {jsonOpen && (
        <div className="rounded-xl border border-cyan-500/35 bg-navy-900/90 p-4 space-y-3 text-xs">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                Validated Custom R1 Experiment JSON
                {isSyntheticTemplate && (
                  <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium text-amber-300">
                    Synthetic Demo Template
                  </span>
                )}
              </h3>
              <p className="text-[11px] text-slate-400">
                Strict R1 contract: <code>init_states [3,6,32,36]</code>, <code>doy_sincos [2]</code>, integer{' '}
                <code>steps (1..14)</code>, and <code>perturbation (null | [6,32,36])</code>. Executes live via{' '}
                <code>POST /simulate</code>.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={loadTemplateFromActiveWindow}
                className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-1.5 text-xs font-medium text-amber-200 hover:bg-amber-500/20"
              >
                Load Synthetic Demo Template
              </button>
              <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-800">
                <Upload className="h-3.5 w-3.5" />
                Upload .json
                <input type="file" accept=".json,application/json" onChange={handleFileUpload} className="hidden" />
              </label>
              <button
                type="button"
                onClick={handleValidateAndRunJson}
                disabled={loading || !jsonText.trim()}
                className="inline-flex items-center gap-1.5 rounded-lg bg-cyan-500 px-3.5 py-1.5 text-xs font-semibold text-slate-950 hover:bg-cyan-400 disabled:opacity-50"
              >
                <Play className="h-3.5 w-3.5" />
                Validate &amp; Run JSON
              </button>
            </div>
          </div>

          <textarea
            aria-label="Custom Experiment JSON Payload"
            rows={5}
            value={jsonText}
            onChange={(e) => {
              setJsonText(e.target.value);
              setIsSyntheticTemplate(false);
            }}
            placeholder='{"init_states": [[[...]]], "doy_sincos": [-0.3, -0.95], "steps": 10, "perturbation": null}'
            className="w-full rounded-lg border border-slate-700 bg-slate-950 p-3 font-mono text-[11px] text-slate-100 placeholder:text-slate-600"
          />

          {jsonError && (
            <div role="alert" className="flex items-start gap-2 rounded-lg border border-rose-500/40 bg-rose-950/40 px-3 py-2 text-rose-200">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
              <span>{jsonError}</span>
            </div>
          )}

          {validatedSummary && !jsonError && (
            <div className="flex items-start gap-2 rounded-lg border border-emerald-500/35 bg-emerald-950/30 px-3 py-2 text-emerald-200">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
              <span>{validatedSummary}</span>
            </div>
          )}
        </div>
      )}

      {loading && !normalRollout ? (
        <LoadingSkeleton heightClass="h-80" />
      ) : normalRollout ? (
        <>
          <ScenarioComparison
            normalRollout={normalRollout}
            perturbedRollout={perturbedRollout}
            dayIdx={dayIdx}
            onChangeDay={onChangeDay}
            variable={variable}
            scenario={scenarioMeta}
            customPert={customPert}
            onChangeCustomPert={onChangeCustomPert}
            onRunCustom={onRun}
            selectedCell={selectedCell}
            onSelectCell={(c) => onSelectCell(c)}
            showGeoOverlay={showGeoOverlay}
            showGridLines={showGridLines}
            chartRef={chartRef}
          />
          <RegionalImpactTable
            indicators={indicators}
            hasPerturbation={perturbedRollout !== null}
            selectedRegion={selectedRegion}
            onSelectRegion={onSelectRegion}
          />
        </>
      ) : (
        <EmptyState
          title="Scenario Lab Ready"
          description="Choose Warm Bay of Bengal, Arabian Sea Monsoon Stall, Zero Perturbation Check, Custom Shock, or Custom JSON Experiment Input."
        />
      )}
    </div>
  );
};
