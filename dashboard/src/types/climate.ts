export type NavSection =
  | 'overview'
  | 'forecast'
  | 'scenarios'
  | 'impact'
  | 'saved'
  | 'performance'
  | 'methodology'
  | 'status'
  | 'settings';

export type VariableIndex = 0 | 1 | 2 | 3 | 4 | 5;

export type PaletteKind = 'thermal' | 'precip' | 'wind' | 'pressure' | 'moisture' | 'diverging';

export interface VariableMeta {
  index: VariableIndex;
  key: 't2m' | 'tp' | 'u10' | 'v10' | 'msl' | 'tcwv';
  label: string;
  shortLabel: string;
  unit: string;
  digits: number;
  palette: PaletteKind;
  description: string;
}

export type ScenarioKey = 'normal' | 'warm_ocean' | 'monsoon_stall' | 'zeros' | 'custom';

export interface ScenarioMeta {
  key: ScenarioKey;
  label: string;
  badge: string;
  isOfficial: boolean;
  description: string;
  box?: [number, number, number, number];
}

export interface CustomPerturbationConfig {
  latMin: number;
  latMax: number;
  lonMin: number;
  lonMax: number;
  deltaT2mK: number;
  deltaTcwvPct: number;
  deltaWindPct: number;
}

export interface SampleWindow {
  id: number;
  label: string;
  init_states: number[][][][]; // [3, 6, 32, 36]
  doy_sincos: [number, number];
}

export interface SimulateResponse {
  rollout: number[][][][]; // [steps, 6, 32, 36]
  inference_ms: number;
}

export interface RegionalIndicator {
  region: string;
  box: [number, number, number, number];
  gridCells: number;
  normalMeanTp: number;
  perturbedMeanTp: number;
  deltaTp: number;
  normalHeavyCells: number;
  perturbedHeavyCells: number;
  normalHeavyExposurePct?: number;
  perturbedHeavyExposurePct?: number;
  normalPersistentHeavyStreak?: number;
  perturbedPersistentHeavyStreak?: number;
  normalPersistentWetStreak?: number;
  perturbedPersistentWetStreak?: number;
  normalMeanT2mC: number;
  perturbedMeanT2mC: number;
  deltaT2m: number;
  normalHeatStressProxyC?: number;
  perturbedHeatStressProxyC?: number;
  deltaHeatStressProxyC?: number;
  normalHotDays?: number;
  perturbedHotDays?: number;
  normalMeanTcwv: number;
  perturbedMeanTcwv: number;
  deltaTcwv: number;
  normalMeanWind: number;
  perturbedMeanWind: number;
  deltaWind: number;
  normalStrongWindCells?: number;
  perturbedStrongWindCells?: number;
  normalMaxWindMs?: number;
  perturbedMaxWindMs?: number;
}

export interface DayDomainMetric {
  day: number;
  label: string;
  normalMean: number;
  normalMin: number;
  normalMax: number;
  perturbedMean?: number;
  perturbedMin?: number;
  perturbedMax?: number;
  anomalyMean?: number;
  normalMeanRain: number;
  normalPeakRain: number;
  perturbedMeanRain?: number;
  perturbedPeakRain?: number;
}

export interface DayRegionMetric {
  day: number;
  label: string;
  normalValue: number;
  perturbedValue?: number;
  anomaly?: number;
}

export interface GridCellSelection {
  latIdx: number;
  lonIdx: number;
  lat: number;
  lon: number;
}

export interface DbStatus {
  connected: boolean;
  provider: string;
  mode: string;
  database?: string;
  run_count?: number;
  last_run_at?: string | null;
  detail?: string;
}

export interface SavedSimulationRun {
  id: number;
  created_at: string;
  name: string;
  scenario: string;
  window_id: number;
  window_label: string;
  doy_sincos: number[];
  steps: number;
  inference_ms: number;
  model_version: string;
  configuration: Record<string, unknown>;
  regional_summary: RegionalIndicator[];
  anomalies?: RegionalIndicator[] | null;
  notes: string;
}

export interface TenDayChannelMetric {
  channel: string;
  rmse: number;
  mae: number;
  bias: number;
  corr: number;
  spatial_corr_mean: number;
  spatial_var_ratio: number;
  rmse_persist: number;
  rmse_clim: number;
  ss_vs_persist: number;
  ss_vs_clim: number;
  ss_vs_best: number;
  valid_samples: number;
  invalid_count: number;
}

export interface TenDayLeadEntry {
  day: number;
  mean_ss_vs_best: number;
  mean_ss_vs_persist: number;
  mean_ss_vs_clim: number;
  precip_10mm: {
    csi_model: number;
    csi_persist: number;
    csi_clim: number;
    csi_skill: number;
    precision: number;
    recall: number;
    pred_event_freq: number;
    true_event_freq: number;
  };
  channels: TenDayChannelMetric[];
}

export interface TenDayMetrics {
  model_name: string;
  val_date_start: string;
  val_date_end: string;
  num_windows: number;
  days: TenDayLeadEntry[];
  regional_rmse: Record<string, Record<string, Record<string, number>>>;
}

export interface ValidationResults {
  model_name: string;
  val_date_start: string;
  val_date_end: string;
  num_eval_windows: number;
  validity_failures: number;
  cpu_inference_ms_mean: number;
  cpu_inference_ms_max: number;
  skill: {
    summary: {
      forecast_points_out_of_45: {
        mean_ss: number;
        mean_csi_skill: number;
        rmse_pts: number;
        csi_pts: number;
        points: number;
      };
      anti_blur_points_out_of_10: number;
      beats_persistence_day4: boolean;
    };
  };
  perturbations: {
    null_consistent: boolean;
    null_max_std_diff: number;
    warm_ocean: Record<string, unknown>;
    monsoon_stall: Record<string, unknown>;
    causality_points_out_of_15: number;
  };
}

export interface UncertaintyChannelStats {
  bias: number;
  mae: number;
  rmse: number;
  p05: number;
  p95: number;
}

export interface UncertaintyBands {
  model_name: string;
  method?: string;
  is_probabilistic_ensemble: boolean;
  deterministic_note: string;
  num_windows: number;
  val_date_start: string;
  val_date_end: string;
  leads: {
    day: number;
    channels: Record<string, UncertaintyChannelStats>;
  }[];
}

export interface EvaluationBundle {
  ten_day: TenDayMetrics | null;
  validation: ValidationResults | null;
  model_metadata: {
    model_name: string;
    epoch: number;
    val_loss: number;
    checkpoint_size_bytes: number;
    train_time_s: number;
    train_samples: number;
    val_samples: number;
    config: Record<string, unknown>;
  } | null;
  dataset_metadata: {
    num_days: number;
    date_start: string;
    date_end: string;
    daily_gaps: number;
    grid_shape: [number, number];
    lat_range: [number, number];
    lon_range: [number, number];
    nan_count: number;
    inf_count: number;
  } | null;
  active_model?: {
    active_variant: string;
    model_name: string;
    architecture_summary: string;
    checkpoint_sha256: string | null;
    challenger_available: boolean;
    manifest?: Record<string, unknown> | null;
  } | null;
  incumbent_manifest?: Record<string, unknown> | null;
  incumbent_uncertainty_bands?: UncertaintyBands | null;
  challenger_manifest?: Record<string, unknown> | null;
  challenger_uncertainty_bands?: UncertaintyBands | null;
  model_comparison?: Record<string, unknown> | null;
}

export interface CustomExperimentPayload {
  init_states: number[][][][];
  doy_sincos: [number, number];
  steps: number;
  perturbation: number[][][] | null;
}

export interface SystemStatus {
  fastapi: string;
  model_loaded: boolean;
  model_name: string;
  active_variant?: string;
  architecture_summary?: string;
  checkpoint_sha256?: string | null;
  challenger_available?: boolean;
  inference_mode: string;
  assets_present: boolean;
  climatology_loaded: boolean;
  neon: DbStatus;
}

export interface AppSettings {
  reducedMotion: boolean;
  showGeoOverlay: boolean;
  showGridLines: boolean;
  autoSaveToNeon: boolean;
}
