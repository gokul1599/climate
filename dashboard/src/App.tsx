import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  buildPerturbation,
  checkHealth,
  computeRegionalIndicators,
  DEFAULT_CUSTOM_PERT,
  deleteSimulationRun,
  fetchDbStatus,
  fetchEvaluationData,
  fetchSampleWindows,
  fetchSavedSimulations,
  fetchSystemStatus,
  REGIONS,
  renameSimulationRun,
  runSimulation,
  saveSimulationRun,
  SCENARIOS,
  VARIABLES,
} from './services/api';
import {
  AppSettings,
  CustomExperimentPayload,
  CustomPerturbationConfig,
  DbStatus,
  EvaluationBundle,
  GridCellSelection,
  NavSection,
  SampleWindow,
  SavedSimulationRun,
  ScenarioKey,
  SystemStatus,
  VariableIndex,
} from './types/climate';
import { AppShell } from './components/AppShell';
import { ErrorState } from './components/ErrorState';
import { ExportDialog } from './components/ExportDialog';
import { OverviewPage } from './pages/OverviewPage';
import { ForecastStudioPage } from './pages/ForecastStudioPage';
import { ScenarioLabPage } from './pages/ScenarioLabPage';
import { ImpactExplorerPage } from './pages/ImpactExplorerPage';
import { SavedSimulationsPage } from './pages/SavedSimulationsPage';
import { ModelPerformancePage } from './pages/ModelPerformancePage';
import { DataMethodologyPage } from './pages/DataMethodologyPage';
import { SystemStatusPage } from './pages/SystemStatusPage';
import { SettingsPage } from './pages/SettingsPage';

const PAGE_SIZE = 10;
const SETTINGS_STORAGE_KEY = 'monsoonverse_settings_v1';

const DEFAULT_SETTINGS: AppSettings = {
  reducedMotion: false,
  showGeoOverlay: true,
  showGridLines: false,
  autoSaveToNeon: false,
};

function loadInitialSettings(): AppSettings {
  try {
    const raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    return { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<AppSettings>) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function App() {
  const [activeSection, setActiveSection] = useState<NavSection>('overview');
  const [backendHealthy, setBackendHealthy] = useState<boolean | null>(null);
  const [modelName, setModelName] = useState<string | null>(null);
  const [dbStatus, setDbStatus] = useState<DbStatus | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);

  const [windows, setWindows] = useState<SampleWindow[]>([]);
  const [windowIdx, setWindowIdx] = useState<number>(0);
  const [scenario, setScenario] = useState<ScenarioKey>('warm_ocean');
  const [customPert, setCustomPert] = useState<CustomPerturbationConfig>(DEFAULT_CUSTOM_PERT);
  const [varIdx, setVarIdx] = useState<VariableIndex>(1);
  const [dayIdx, setDayIdx] = useState<number>(0);

  const [normalRollout, setNormalRollout] = useState<number[][][][] | null>(null);
  const [perturbedRollout, setPerturbedRollout] = useState<number[][][][] | null>(null);
  const [inferenceMs, setInferenceMs] = useState<number | null>(null);
  const [lastSimAt, setLastSimAt] = useState<string | null>(null);
  const [simSourceNote, setSimSourceNote] = useState<string>('Sample ERA5 validation window (unmodified)');

  const executeCustomJsonExperiment = async (payload: CustomExperimentPayload, isSyntheticTemplate: boolean) => {
    setLoading(true);
    setError(null);
    setDbWarning(null);
    try {
      const baseRes = await runSimulation(payload.init_states, payload.doy_sincos, payload.steps, null);
      setNormalRollout(baseRes.rollout);
      let activeMs = baseRes.inference_ms;
      if (payload.perturbation) {
        const pertRes = await runSimulation(
          payload.init_states,
          payload.doy_sincos,
          payload.steps,
          payload.perturbation,
        );
        setPerturbedRollout(pertRes.rollout);
        activeMs = pertRes.inference_ms;
      } else {
        setPerturbedRollout(null);
      }
      setInferenceMs(activeMs);
      setLastSimAt(new Date().toISOString());
      setDayIdx((prev) => Math.min(prev, Math.max(0, payload.steps - 1)));
      setBackendHealthy(true);
      setSimSourceNote(
        isSyntheticTemplate
          ? `Custom JSON run (${payload.steps}d) from Synthetic Demo Template · ${modelName ?? 'Active Model'}`
          : `Custom JSON run (${payload.steps}d) from User JSON Payload · ${modelName ?? 'Active Model'}`,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const [selectedRegion, setSelectedRegion] = useState<string>(REGIONS[0].name);
  const [selectedCell, setSelectedCell] = useState<GridCellSelection | null>(null);

  const [savedRuns, setSavedRuns] = useState<SavedSimulationRun[]>([]);
  const [savedTotal, setSavedTotal] = useState<number>(0);
  const [savedPage, setSavedPage] = useState<number>(0);
  const [savedSearch, setSavedSearch] = useState<string>('');
  const [savedScenarioFilter, setSavedScenarioFilter] = useState<string>('all');

  const [evaluation, setEvaluation] = useState<EvaluationBundle | null>(null);
  const [evalLoading, setEvalLoading] = useState<boolean>(false);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [dbWarning, setDbWarning] = useState<string | null>(null);

  const [exportOpen, setExportOpen] = useState<boolean>(false);
  const [saveModalOpen, setSaveModalOpen] = useState<boolean>(false);
  const [saveModalName, setSaveModalName] = useState<string>('');
  const [settings, setSettings] = useState<AppSettings>(loadInitialSettings);

  const handleUpdateSettings = (next: AppSettings) => {
    setSettings(next);
    try {
      window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // ignore storage quota / private mode errors
    }
  };

  const studioChartRef = useRef<HTMLDivElement>(null);

  const loadSavedRuns = useCallback(
    async (pg = savedPage, q = savedSearch, sc = savedScenarioFilter) => {
      try {
        const res = await fetchSavedSimulations({
          limit: PAGE_SIZE,
          offset: pg * PAGE_SIZE,
          search: q,
          scenario: sc,
        });
        setSavedRuns(res.runs);
        setSavedTotal(res.total);
      } catch (e) {
        setDbWarning(e instanceof Error ? e.message : 'Unable to fetch saved simulations');
      }
    },
    [savedPage, savedSearch, savedScenarioFilter],
  );

  // Initial load: health, sample windows, DB status, system status, evaluation bundle
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const [h, w, db, sys] = await Promise.all([
          checkHealth(),
          fetchSampleWindows(),
          fetchDbStatus().catch(() => ({ connected: false, provider: 'neon', mode: 'offline' })),
          fetchSystemStatus().catch(() => null),
        ]);
        if (!active) return;
        setBackendHealthy(h.status === 'ready');
        setModelName(h.model_name);
        setWindows(w);
        setDbStatus(db);
        if (sys) setSystemStatus(sys);
        if (db.connected) {
          const res = await fetchSavedSimulations({ limit: PAGE_SIZE, offset: 0 });
          if (active) {
            setSavedRuns(res.runs);
            setSavedTotal(res.total);
          }
        }
      } catch (e) {
        if (!active) return;
        setBackendHealthy(false);
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  // Load evaluation metrics lazily when entering Performance or Methodology pages
  useEffect(() => {
    if ((activeSection === 'performance' || activeSection === 'methodology') && !evaluation && !evalLoading) {
      setEvalLoading(true);
      fetchEvaluationData()
        .then((ev) => setEvaluation(ev))
        .catch((e) => setError(e instanceof Error ? e.message : String(e)))
        .finally(() => setEvalLoading(false));
    }
  }, [activeSection, evaluation, evalLoading]);

  // Reload saved runs when filters/pagination change
  useEffect(() => {
    if (dbStatus?.connected) {
      loadSavedRuns(savedPage, savedSearch, savedScenarioFilter);
    }
  }, [dbStatus?.connected, savedPage, savedSearch, savedScenarioFilter, loadSavedRuns]);

  const executeSimulation = async (
    wIdx = windowIdx,
    sc = scenario,
    persistName?: string,
    persistNotes?: string,
  ) => {
    const win = windows[wIdx];
    if (!win) return;
    setLoading(true);
    setError(null);
    setDbWarning(null);
    try {
      const normRes = await runSimulation(win.init_states, win.doy_sincos, 10, null);
      setNormalRollout(normRes.rollout);

      let activePert: number[][][][] | null = null;
      let activeMs = normRes.inference_ms;
      if (sc === 'normal') {
        setPerturbedRollout(null);
      } else {
        const pert = buildPerturbation(sc, win.init_states[2], customPert);
        const pertRes = await runSimulation(win.init_states, win.doy_sincos, 10, pert);
        activePert = pertRes.rollout;
        activeMs = pertRes.inference_ms;
        setPerturbedRollout(pertRes.rollout);
      }

      setInferenceMs(activeMs);
      setLastSimAt(new Date().toISOString());
      setBackendHealthy(true);
      setSimSourceNote(`Preset "${sc}" on "${win.label}" (10d) · ${modelName ?? 'MonsoonVerse'}`);

      if (dbStatus?.connected && (settings.autoSaveToNeon || persistName !== undefined)) {
        const summary = computeRegionalIndicators(normRes.rollout, activePert);
        try {
          await saveSimulationRun({
            name: persistName,
            scenario: sc,
            window_id: win.id,
            window_label: win.label,
            doy_sincos: win.doy_sincos,
            steps: 10,
            inference_ms: activeMs,
            configuration: sc === 'custom' ? { customPert } : { officialPreset: sc },
            regional_summary: summary,
            anomalies: activePert ? summary : null,
            notes: persistNotes,
          });
          await loadSavedRuns(0, savedSearch, savedScenarioFilter);
          setSavedPage(0);
        } catch (dbErr) {
          setDbWarning(
            `Simulation succeeded, but Neon save failed: ${
              dbErr instanceof Error ? dbErr.message : String(dbErr)
            }`,
          );
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  // Initial simulation once sample windows arrive
  useEffect(() => {
    if (windows.length > 0 && normalRollout === null && !loading) {
      executeSimulation(0, scenario);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [windows]);

  const handleSaveCurrentExplicit = async (customName: string, notes = '') => {
    if (!normalRollout || !windows[windowIdx]) return;
    const win = windows[windowIdx];
    const summary = computeRegionalIndicators(normalRollout, perturbedRollout);
    try {
      await saveSimulationRun({
        name: customName || undefined,
        scenario,
        window_id: win.id,
        window_label: win.label,
        doy_sincos: win.doy_sincos,
        steps: 10,
        inference_ms: inferenceMs ?? 0,
        configuration: scenario === 'custom' ? { customPert } : { officialPreset: scenario },
        regional_summary: summary,
        anomalies: perturbedRollout ? summary : null,
        notes,
      });
      await loadSavedRuns(0, savedSearch, savedScenarioFilter);
      setSavedPage(0);
    } catch (e) {
      setDbWarning(e instanceof Error ? e.message : 'Failed to save simulation to Neon');
    }
  };

  const handleRenameRun = async (id: number, name: string) => {
    await renameSimulationRun(id, name);
    await loadSavedRuns(savedPage, savedSearch, savedScenarioFilter);
  };

  const handleDeleteRun = async (id: number) => {
    await deleteSimulationRun(id);
    await loadSavedRuns(savedPage, savedSearch, savedScenarioFilter);
  };

  const handleReset = () => {
    setWindowIdx(0);
    setScenario('normal');
    setVarIdx(1);
    setDayIdx(0);
    setSelectedCell(null);
  };

  const activeVar = VARIABLES[varIdx];
  const activeScenario = SCENARIOS.find((s) => s.key === scenario) ?? SCENARIOS[0];
  const currentIndicators = useMemo(
    () => (normalRollout ? computeRegionalIndicators(normalRollout, perturbedRollout) : []),
    [normalRollout, perturbedRollout],
  );

  return (
    <AppShell
      activeSection={activeSection}
      onSelectSection={setActiveSection}
      backendHealthy={backendHealthy}
      modelName={modelName}
      dbStatus={dbStatus}
      inferenceMs={inferenceMs}
      settings={settings}
    >
      {error && (
        <ErrorState
          title="Backend / Simulation Error"
          message={error}
          onRetry={() => executeSimulation(windowIdx, scenario)}
        />
      )}
      {dbWarning && (
        <ErrorState title="Neon Persistence Warning (Inference Unaffected)" message={dbWarning} />
      )}

      {activeSection === 'overview' && (
        <OverviewPage
          normalRollout={normalRollout}
          perturbedRollout={perturbedRollout}
          dayIdx={dayIdx}
          onChangeDay={setDayIdx}
          varIdx={varIdx}
          onChangeVar={setVarIdx}
          scenario={scenario}
          onChangeScenario={setScenario}
          inferenceMs={inferenceMs}
          backendHealthy={backendHealthy}
          savedTotal={savedTotal}
          recentRuns={savedRuns}
          indicators={currentIndicators}
          loading={loading}
          showGeoOverlay={settings.showGeoOverlay}
          showGridLines={settings.showGridLines}
          onRunNew={() => executeSimulation(windowIdx, scenario)}
          onNavigate={setActiveSection}
        />
      )}

      {activeSection === 'forecast' && (
        <ForecastStudioPage
          windows={windows}
          windowIdx={windowIdx}
          onChangeWindow={setWindowIdx}
          varIdx={varIdx}
          onChangeVar={setVarIdx}
          variable={activeVar}
          scenario={scenario}
          scenarioMeta={activeScenario}
          onChangeScenario={setScenario}
          dayIdx={dayIdx}
          onChangeDay={setDayIdx}
          normalRollout={normalRollout}
          perturbedRollout={perturbedRollout}
          loading={loading}
          dbConnected={Boolean(dbStatus?.connected)}
          selectedRegion={selectedRegion}
          selectedCell={selectedCell}
          onSelectCell={setSelectedCell}
          showGeoOverlay={settings.showGeoOverlay}
          showGridLines={settings.showGridLines}
          onRun={() => executeSimulation(windowIdx, scenario)}
          onReset={handleReset}
          onOpenSave={() => setSaveModalOpen(true)}
          onOpenExport={() => setExportOpen(true)}
          chartRef={studioChartRef}
        />
      )}

      {activeSection === 'scenarios' && (
        <ScenarioLabPage
          windows={windows}
          windowIdx={windowIdx}
          onChangeWindow={setWindowIdx}
          varIdx={varIdx}
          onChangeVar={setVarIdx}
          variable={activeVar}
          scenario={scenario}
          scenarioMeta={activeScenario}
          onChangeScenario={setScenario}
          dayIdx={dayIdx}
          onChangeDay={setDayIdx}
          normalRollout={normalRollout}
          perturbedRollout={perturbedRollout}
          customPert={customPert}
          onChangeCustomPert={setCustomPert}
          indicators={currentIndicators}
          selectedRegion={selectedRegion}
          onSelectRegion={setSelectedRegion}
          selectedCell={selectedCell}
          onSelectCell={setSelectedCell}
          loading={loading}
          dbConnected={Boolean(dbStatus?.connected)}
          showGeoOverlay={settings.showGeoOverlay}
          showGridLines={settings.showGridLines}
          onRun={() => executeSimulation(windowIdx, scenario)}
          onRunCustomJson={executeCustomJsonExperiment}
          modelName={modelName}
          simSourceNote={simSourceNote}
          onReset={handleReset}
          onOpenSave={() => setSaveModalOpen(true)}
          onOpenExport={() => setExportOpen(true)}
          chartRef={studioChartRef}
        />
      )}

      {activeSection === 'impact' && (
        <ImpactExplorerPage
          normalRollout={normalRollout}
          perturbedRollout={perturbedRollout}
          dayIdx={dayIdx}
          scenario={scenario}
          variable={activeVar}
          selectedRegion={selectedRegion}
          onSelectRegion={setSelectedRegion}
          showGeoOverlay={settings.showGeoOverlay}
          showGridLines={settings.showGridLines}
        />
      )}

      {activeSection === 'saved' && (
        <SavedSimulationsPage
          dbStatus={dbStatus}
          runs={savedRuns}
          total={savedTotal}
          page={savedPage}
          pageSize={PAGE_SIZE}
          search={savedSearch}
          scenarioFilter={savedScenarioFilter}
          canSaveCurrent={Boolean(normalRollout)}
          onSearchChange={(q) => {
            setSavedSearch(q);
            setSavedPage(0);
          }}
          onScenarioFilterChange={(sc) => {
            setSavedScenarioFilter(sc);
            setSavedPage(0);
          }}
          onPageChange={setSavedPage}
          onSaveCurrent={handleSaveCurrentExplicit}
          onRename={handleRenameRun}
          onDelete={handleDeleteRun}
        />
      )}

      {activeSection === 'performance' && (
        <ModelPerformancePage evaluation={evaluation} loading={evalLoading} />
      )}

      {activeSection === 'methodology' && <DataMethodologyPage evaluation={evaluation} />}

      {activeSection === 'status' && (
        <SystemStatusPage
          systemStatus={systemStatus}
          lastInferenceMs={inferenceMs}
          lastSimAt={lastSimAt}
          onRefreshStatus={(st) => {
            setSystemStatus(st);
            setDbStatus(st.neon);
            setBackendHealthy(st.fastapi === 'operational' && st.model_loaded);
          }}
        />
      )}

      {activeSection === 'settings' && (
        <SettingsPage settings={settings} onChangeSettings={handleUpdateSettings} />
      )}

      {/* Save Modal */}
      {saveModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="save-dialog-title"
          onKeyDown={(e) => {
            if (e.key === 'Escape') setSaveModalOpen(false);
          }}
        >
          <div className="w-full max-w-sm rounded-xl border border-slate-800 bg-[#0b1426] p-5 shadow-2xl">
            <h3 id="save-dialog-title" className="text-sm font-semibold text-white">
              Save Simulation to Neon
            </h3>
            <p className="mt-1 text-xs text-slate-400">
              Give this run a name so you can inspect or compare it later.
            </p>
            <label htmlFor="save-run-name-input" className="sr-only">
              Simulation name
            </label>
            <input
              id="save-run-name-input"
              type="text"
              value={saveModalName}
              onChange={(e) => setSaveModalName(e.target.value)}
              placeholder={`${windows[windowIdx]?.label ?? 'Window'} (${scenario})`}
              className="mt-3 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-white"
            />
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setSaveModalOpen(false)}
                className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-800"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={async () => {
                  await handleSaveCurrentExplicit(saveModalName);
                  setSaveModalName('');
                  setSaveModalOpen(false);
                }}
                className="rounded-lg bg-emerald-500 px-3 py-1.5 text-xs font-semibold text-slate-950 hover:bg-emerald-400"
              >
                Save Run
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Export Dialog */}
      <ExportDialog
        open={exportOpen}
        onClose={() => setExportOpen(false)}
        normalRollout={normalRollout}
        perturbedRollout={perturbedRollout}
        dayIdx={dayIdx}
        windowLabel={windows[windowIdx]?.label ?? 'Sample Window'}
        scenario={activeScenario}
        variable={activeVar}
        inferenceMs={inferenceMs}
        indicators={currentIndicators}
        chartContainerRef={studioChartRef}
      />
    </AppShell>
  );
}

export default App;
