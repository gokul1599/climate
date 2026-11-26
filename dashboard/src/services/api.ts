import {
  CustomPerturbationConfig,
  DayDomainMetric,
  DayRegionMetric,
  DbStatus,
  EvaluationBundle,
  GridCellSelection,
  RegionalIndicator,
  SampleWindow,
  SavedSimulationRun,
  ScenarioKey,
  ScenarioMeta,
  SimulateResponse,
  SystemStatus,
  VariableIndex,
  VariableMeta,
} from '../types/climate';

export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');

export const VARIABLES: VariableMeta[] = [
  {
    index: 0,
    key: 't2m',
    label: '2m Temperature',
    shortLabel: 't2m',
    unit: 'K',
    digits: 2,
    palette: 'thermal',
    description: 'Daily mean surface air temperature at 2 metres (00/06/12/18 UTC mean).',
  },
  {
    index: 1,
    key: 'tp',
    label: 'Total Precipitation',
    shortLabel: 'tp',
    unit: 'mm/day',
    digits: 2,
    palette: 'precip',
    description: '24-hour accumulated daily precipitation (non-negative physical constraint).',
  },
  {
    index: 2,
    key: 'u10',
    label: 'Eastward Wind (u10)',
    shortLabel: 'u10',
    unit: 'm/s',
    digits: 2,
    palette: 'wind',
    description: 'Zonal wind component at 10 metres (+ eastward, − westward).',
  },
  {
    index: 3,
    key: 'v10',
    label: 'Northward Wind (v10)',
    shortLabel: 'v10',
    unit: 'm/s',
    digits: 2,
    palette: 'wind',
    description: 'Meridional wind component at 10 metres (+ northward, − southward).',
  },
  {
    index: 4,
    key: 'msl',
    label: 'Mean Sea-Level Pressure',
    shortLabel: 'msl',
    unit: 'Pa',
    digits: 0,
    palette: 'pressure',
    description: 'Atmospheric pressure reduced to mean sea level.',
  },
  {
    index: 5,
    key: 'tcwv',
    label: 'Total Column Water Vapour',
    shortLabel: 'tcwv',
    unit: 'kg/m²',
    digits: 2,
    palette: 'moisture',
    description: 'Vertically integrated atmospheric water vapour content.',
  },
];

export const SCENARIOS: ScenarioMeta[] = [
  {
    key: 'normal',
    label: 'Normal Forecast',
    badge: 'Baseline',
    isOfficial: true,
    description: 'Unperturbed 10-day deterministic autoregressive rollout from the 3-day initial state.',
  },
  {
    key: 'warm_ocean',
    label: 'Warm Bay of Bengal',
    badge: 'Official Shock',
    isOfficial: true,
    description: '+3.0 K t2m and +15% current TCWV over Bay of Bengal (8°–20°N, 82°–94°E) on init_states[-1].',
    box: [8, 20, 82, 94],
  },
  {
    key: 'monsoon_stall',
    label: 'Arabian Sea Monsoon Stall',
    badge: 'Official Shock',
    isOfficial: true,
    description: '−40% u10 and −40% v10 wind components over Arabian Sea (8°–20°N, 65°–75°E) on init_states[-1].',
    box: [8, 20, 65, 75],
  },
  {
    key: 'zeros',
    label: 'Zero Perturbation Check',
    badge: 'Official Gate',
    isOfficial: true,
    description: 'All-zero [6, 32, 36] delta array; verifies deterministic equivalence (diff = 0.0).',
  },
  {
    key: 'custom',
    label: 'Custom Experimental Shock',
    badge: 'Experimental',
    isOfficial: false,
    description: 'Exploratory regional perturbation applied strictly in physical units to init_states[-1].',
  },
];

export const REGIONS: { name: string; box: [number, number, number, number]; focus: string }[] = [
  {
    name: 'Western Ghats & Konkan',
    box: [10, 20, 72, 77],
    focus: 'Orographic monsoon inflow & coastal heavy rainfall belt',
  },
  {
    name: 'Central India Monsoon Core',
    box: [18, 26, 75, 85],
    focus: 'Core rainfed agricultural zone sensitive to monsoon breaks',
  },
  {
    name: 'Indo-Gangetic Plain',
    box: [24, 31, 75, 88],
    focus: 'Alluvial basin heat exposure & monsoon trough rainfall',
  },
  {
    name: 'Northeast India',
    box: [22, 29, 88, 97],
    focus: 'High-precipitation moisture convergence foothills',
  },
  {
    name: 'Bay of Bengal',
    box: [8, 20, 82, 94],
    focus: 'Primary cyclogenesis & warm-ocean moisture source box',
  },
  {
    name: 'Arabian Sea',
    box: [8, 20, 65, 75],
    focus: 'Cross-equatorial Somali jet & monsoon stall box',
  },
];

export const LAT = Array.from({ length: 32 }, (_, i) => 6.5 + i);
export const LON = Array.from({ length: 36 }, (_, j) => 65.5 + j);

export const DEFAULT_CUSTOM_PERT: CustomPerturbationConfig = {
  latMin: 12,
  latMax: 22,
  lonMin: 74,
  lonMax: 84,
  deltaT2mK: 2.0,
  deltaTcwvPct: 10,
  deltaWindPct: 0,
};

function makeEmptyPert(): number[][][] {
  return Array.from({ length: 6 }, () =>
    Array.from({ length: 32 }, () => Array<number>(36).fill(0)),
  );
}

export function buildPerturbation(
  scenario: ScenarioKey,
  lastState: number[][][],
  custom?: CustomPerturbationConfig,
): number[][][] | null {
  if (scenario === 'normal') return null;
  const pert = makeEmptyPert();
  if (scenario === 'zeros') return pert;

  if (scenario === 'warm_ocean') {
    for (let i = 0; i < 32; i++) {
      if (LAT[i] >= 8 && LAT[i] <= 20) {
        for (let j = 0; j < 36; j++) {
          if (LON[j] >= 82 && LON[j] <= 94) {
            pert[0][i][j] = 3.0;
            pert[5][i][j] = 0.15 * lastState[5][i][j];
          }
        }
      }
    }
  } else if (scenario === 'monsoon_stall') {
    for (let i = 0; i < 32; i++) {
      if (LAT[i] >= 8 && LAT[i] <= 20) {
        for (let j = 0; j < 36; j++) {
          if (LON[j] >= 65 && LON[j] <= 75) {
            pert[2][i][j] = -0.4 * lastState[2][i][j];
            pert[3][i][j] = -0.4 * lastState[3][i][j];
          }
        }
      }
    }
  } else if (scenario === 'custom' && custom) {
    for (let i = 0; i < 32; i++) {
      if (LAT[i] >= custom.latMin && LAT[i] <= custom.latMax) {
        for (let j = 0; j < 36; j++) {
          if (LON[j] >= custom.lonMin && LON[j] <= custom.lonMax) {
            pert[0][i][j] = custom.deltaT2mK;
            pert[5][i][j] = (custom.deltaTcwvPct / 100) * lastState[5][i][j];
            pert[2][i][j] = (custom.deltaWindPct / 100) * lastState[2][i][j];
            pert[3][i][j] = (custom.deltaWindPct / 100) * lastState[3][i][j];
          }
        }
      }
    }
  }
  return pert;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}${txt ? `: ${txt.slice(0, 160)}` : ''}`);
  }
  return (await res.json()) as T;
}

export function checkHealth(): Promise<{ status: string; model_name: string }> {
  return fetchJson('/health', { method: 'POST' });
}

export async function fetchSampleWindows(): Promise<SampleWindow[]> {
  const data = await fetchJson<{ windows: SampleWindow[] }>('/api/sample-windows');
  return data.windows;
}

export function runSimulation(
  initStates: number[][][][],
  doySincos: [number, number],
  steps: number,
  perturbation: number[][][] | null,
): Promise<SimulateResponse> {
  return fetchJson('/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      init_states: initStates,
      doy_sincos: doySincos,
      steps,
      perturbation,
    }),
  });
}

export function fetchDbStatus(): Promise<DbStatus> {
  return fetchJson('/api/db-status');
}

export function fetchSystemStatus(): Promise<SystemStatus> {
  return fetchJson('/api/system-status');
}

export function fetchEvaluationData(): Promise<EvaluationBundle> {
  return fetchJson('/api/evaluation');
}

export function fetchSavedSimulations(params?: {
  limit?: number;
  offset?: number;
  search?: string;
  scenario?: string;
}): Promise<{ enabled: boolean; runs: SavedSimulationRun[]; total: number }> {
  const qs = new URLSearchParams();
  if (params?.limit !== undefined) qs.set('limit', String(params.limit));
  if (params?.offset !== undefined) qs.set('offset', String(params.offset));
  if (params?.search) qs.set('search', params.search);
  if (params?.scenario && params.scenario !== 'all') qs.set('scenario', params.scenario);
  const suffix = qs.toString() ? `?${qs.toString()}` : '';
  return fetchJson(`/api/simulations${suffix}`);
}

export async function saveSimulationRun(payload: {
  name?: string;
  scenario: string;
  window_id: number;
  window_label: string;
  doy_sincos: [number, number];
  steps: number;
  inference_ms: number;
  configuration: Record<string, unknown>;
  regional_summary: RegionalIndicator[];
  anomalies?: RegionalIndicator[] | null;
  notes?: string;
}): Promise<SavedSimulationRun> {
  const data = await fetchJson<{ saved: boolean; run: SavedSimulationRun }>('/api/simulations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return data.run;
}

export async function renameSimulationRun(
  runId: number,
  name: string,
  notes?: string,
): Promise<SavedSimulationRun> {
  const data = await fetchJson<{ updated: boolean; run: SavedSimulationRun }>(
    `/api/simulations/${runId}`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, notes }),
    },
  );
  return data.run;
}

export async function deleteSimulationRun(runId: number): Promise<void> {
  await fetchJson(`/api/simulations/${runId}`, { method: 'DELETE' });
}

export function computeRegionalIndicators(
  normalRollout: number[][][][],
  perturbedRollout: number[][][][] | null,
  dayFilter?: number | null,
): RegionalIndicator[] {
  const dayIndices =
    dayFilter !== undefined && dayFilter !== null
      ? [dayFilter]
      : Array.from({ length: normalRollout.length }, (_, k) => k);

  return REGIONS.map(({ name, box }) => {
    const [la0, la1, lo0, lo1] = box;
    let nTp = 0;
    let pTp = 0;
    let nT2m = 0;
    let pT2m = 0;
    let nTcwv = 0;
    let pTcwv = 0;
    let nWind = 0;
    let pWind = 0;
    let nHeavy = 0;
    let pHeavy = 0;
    let count = 0;
    let cellCount = 0;

    for (let i = 0; i < 32; i++) {
      if (LAT[i] < la0 || LAT[i] > la1) continue;
      for (let j = 0; j < 36; j++) {
        if (LON[j] < lo0 || LON[j] > lo1) continue;
        cellCount++;
      }
    }

    for (const d of dayIndices) {
      for (let i = 0; i < 32; i++) {
        if (LAT[i] < la0 || LAT[i] > la1) continue;
        for (let j = 0; j < 36; j++) {
          if (LON[j] < lo0 || LON[j] > lo1) continue;
          const nvTp = normalRollout[d][1][i][j];
          const nvT2m = normalRollout[d][0][i][j];
          const nvTcwv = normalRollout[d][5][i][j];
          const nvW = Math.hypot(normalRollout[d][2][i][j], normalRollout[d][3][i][j]);

          nTp += nvTp;
          nT2m += nvT2m;
          nTcwv += nvTcwv;
          nWind += nvW;
          if (nvTp >= 10) nHeavy++;

          const src = perturbedRollout ?? normalRollout;
          const pvTp = src[d][1][i][j];
          const pvT2m = src[d][0][i][j];
          const pvTcwv = src[d][5][i][j];
          const pvW = Math.hypot(src[d][2][i][j], src[d][3][i][j]);

          pTp += pvTp;
          pT2m += pvT2m;
          pTcwv += pvTcwv;
          pWind += pvW;
          if (pvTp >= 10) pHeavy++;
          count++;
        }
      }
    }

    const denom = Math.max(count, 1);
    return {
      region: name,
      box,
      gridCells: cellCount,
      normalMeanTp: nTp / denom,
      perturbedMeanTp: pTp / denom,
      deltaTp: (pTp - nTp) / denom,
      normalHeavyCells: nHeavy,
      perturbedHeavyCells: pHeavy,
      normalMeanT2mC: nT2m / denom - 273.15,
      perturbedMeanT2mC: pT2m / denom - 273.15,
      deltaT2m: (pT2m - nT2m) / denom,
      normalMeanTcwv: nTcwv / denom,
      perturbedMeanTcwv: pTcwv / denom,
      deltaTcwv: (pTcwv - nTcwv) / denom,
      normalMeanWind: nWind / denom,
      perturbedMeanWind: pWind / denom,
      deltaWind: (pWind - nWind) / denom,
    };
  });
}

function gridStats(grid: number[][]): { mean: number; min: number; max: number } {
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  let cnt = 0;
  for (let i = 0; i < 32; i++) {
    for (let j = 0; j < 36; j++) {
      const v = grid[i][j];
      sum += v;
      if (v < min) min = v;
      if (v > max) max = v;
      cnt++;
    }
  }
  return { mean: sum / Math.max(cnt, 1), min, max };
}

export function computeDomainDaySeries(
  normalRollout: number[][][][],
  perturbedRollout: number[][][][] | null,
  varIdx: VariableIndex,
): DayDomainMetric[] {
  return normalRollout.map((step, idx) => {
    const nSt = gridStats(step[varIdx]);
    const nRain = gridStats(step[1]);
    const pSt = perturbedRollout ? gridStats(perturbedRollout[idx][varIdx]) : undefined;
    const pRain = perturbedRollout ? gridStats(perturbedRollout[idx][1]) : undefined;
    return {
      day: idx + 1,
      label: `Day ${idx + 1}`,
      normalMean: Number(nSt.mean.toFixed(3)),
      normalMin: Number(nSt.min.toFixed(3)),
      normalMax: Number(nSt.max.toFixed(3)),
      perturbedMean: pSt ? Number(pSt.mean.toFixed(3)) : undefined,
      perturbedMin: pSt ? Number(pSt.min.toFixed(3)) : undefined,
      perturbedMax: pSt ? Number(pSt.max.toFixed(3)) : undefined,
      anomalyMean: pSt ? Number((pSt.mean - nSt.mean).toFixed(4)) : undefined,
      normalMeanRain: Number(nRain.mean.toFixed(3)),
      normalPeakRain: Number(nRain.max.toFixed(2)),
      perturbedMeanRain: pRain ? Number(pRain.mean.toFixed(3)) : undefined,
      perturbedPeakRain: pRain ? Number(pRain.max.toFixed(2)) : undefined,
    };
  });
}

export function computeRegionalTimeSeries(
  normalRollout: number[][][][],
  perturbedRollout: number[][][][] | null,
  varIdx: VariableIndex,
  box: [number, number, number, number],
  cell?: GridCellSelection | null,
): DayRegionMetric[] {
  return normalRollout.map((step, idx) => {
    if (cell) {
      const nv = step[varIdx][cell.latIdx][cell.lonIdx];
      const pv = perturbedRollout ? perturbedRollout[idx][varIdx][cell.latIdx][cell.lonIdx] : undefined;
      return {
        day: idx + 1,
        label: `Day ${idx + 1}`,
        normalValue: Number(nv.toFixed(3)),
        perturbedValue: pv !== undefined ? Number(pv.toFixed(3)) : undefined,
        anomaly: pv !== undefined ? Number((pv - nv).toFixed(4)) : undefined,
      };
    }

    const [la0, la1, lo0, lo1] = box;
    let nSum = 0;
    let pSum = 0;
    let cnt = 0;
    for (let i = 0; i < 32; i++) {
      if (LAT[i] < la0 || LAT[i] > la1) continue;
      for (let j = 0; j < 36; j++) {
        if (LON[j] < lo0 || LON[j] > lo1) continue;
        nSum += step[varIdx][i][j];
        if (perturbedRollout) pSum += perturbedRollout[idx][varIdx][i][j];
        cnt++;
      }
    }
    const nv = nSum / Math.max(cnt, 1);
    const pv = perturbedRollout ? pSum / Math.max(cnt, 1) : undefined;
    return {
      day: idx + 1,
      label: `Day ${idx + 1}`,
      normalValue: Number(nv.toFixed(3)),
      perturbedValue: pv !== undefined ? Number(pv.toFixed(3)) : undefined,
      anomaly: pv !== undefined ? Number((pv - nv).toFixed(4)) : undefined,
    };
  });
}
