import {
  CustomExperimentPayload,
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

export const STRESS_FORMULAS: { id: string; title: string; formula: string; badge: string }[] = [
  {
    id: 'heavy_rain',
    title: 'Heavy-Rain Exposure (>=10 mm/day)',
    formula: 'Cell-days with tp >= 10.0 mm/day; Exposure % = 100 * cell_days / (days * grid_cells)',
    badge: 'Official Rain Metric',
  },
  {
    id: 'persistent_streak',
    title: 'Persistent Rainfall Duration',
    formula: 'Max consecutive days tp >= 10.0 mm/day (or wet >= 2.5 mm/day) at any cell in the region',
    badge: 'Derived Model Indicator',
  },
  {
    id: 'heat_stress_proxy',
    title: 'Heat-Stress Proxy (Unvalidated Indicator)',
    formula: 'Apparent Temp Proxy = T_c + 0.33 * (0.65 * tcwv) - 0.70 * wind - 4.0 (deg C); plus T >= 35 C cell-days',
    badge: 'Unvalidated Proxy',
  },
  {
    id: 'strong_wind',
    title: 'Strong-Wind Exposure (>=10 m/s)',
    formula: 'Cell-days with |V| = sqrt(u10^2 + v10^2) >= 10.0 m/s and regional peak wind speed',
    badge: 'Derived Model Indicator',
  },
];

function checkGrid3D(arr: unknown, expectedD0: number, label: string): string | null {
  if (!Array.isArray(arr) || arr.length !== expectedD0) {
    return `${label} must be an array of length ${expectedD0}.`;
  }
  for (let i = 0; i < expectedD0; i++) {
    const plane = arr[i];
    if (!Array.isArray(plane) || plane.length !== 32) {
      return `${label}[${i}] must have 32 latitude rows.`;
    }
    for (let r = 0; r < 32; r++) {
      const row = plane[r];
      if (!Array.isArray(row) || row.length !== 36) {
        return `${label}[${i}][${r}] must have 36 longitude columns.`;
      }
      for (let c = 0; c < 36; c++) {
        const v = row[c];
        if (typeof v !== 'number' || !Number.isFinite(v)) {
          return `${label}[${i}][${r}][${c}] must be a finite number.`;
        }
      }
    }
  }
  return null;
}

export function validateExperimentJsonPayload(
  rawText: string,
): { ok: true; payload: CustomExperimentPayload } | { ok: false; error: string } {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawText);
  } catch (err) {
    return {
      ok: false,
      error: `Invalid JSON syntax: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { ok: false, error: 'Root JSON value must be an object { init_states, doy_sincos, steps, perturbation }.' };
  }
  const obj = parsed as Record<string, unknown>;

  if (!Array.isArray(obj.init_states) || obj.init_states.length !== 3) {
    return { ok: false, error: 'init_states must be shape [3, 6, 32, 36] (3 consecutive daily states).' };
  }
  for (let t = 0; t < 3; t++) {
    const err = checkGrid3D(obj.init_states[t], 6, `init_states[${t}]`);
    if (err) return { ok: false, error: err };
    const tpGrid = (obj.init_states[t] as number[][][])[1];
    for (let r = 0; r < 32; r++) {
      for (let c = 0; c < 36; c++) {
        if (tpGrid[r][c] < 0) {
          return { ok: false, error: `init_states[${t}][1][${r}][${c}] precipitation (tp) cannot be negative.` };
        }
      }
    }
  }

  if (
    !Array.isArray(obj.doy_sincos) ||
    obj.doy_sincos.length !== 2 ||
    typeof obj.doy_sincos[0] !== 'number' ||
    typeof obj.doy_sincos[1] !== 'number' ||
    !Number.isFinite(obj.doy_sincos[0]) ||
    !Number.isFinite(obj.doy_sincos[1])
  ) {
    return { ok: false, error: 'doy_sincos must be a finite 2-element number array [sin, cos].' };
  }
  const doyNormSq = obj.doy_sincos[0] ** 2 + obj.doy_sincos[1] ** 2;
  if (doyNormSq < 0.8 || doyNormSq > 1.2) {
    return { ok: false, error: 'doy_sincos must lie near the unit circle (sin^2 + cos^2 ~= 1.0).' };
  }

  if (
    typeof obj.steps !== 'number' ||
    !Number.isInteger(obj.steps) ||
    obj.steps < 1 ||
    obj.steps > 14
  ) {
    return { ok: false, error: 'steps must be an integer between 1 and 14.' };
  }

  let pert: number[][][] | null = null;
  if (obj.perturbation !== undefined && obj.perturbation !== null) {
    const pErr = checkGrid3D(obj.perturbation, 6, 'perturbation');
    if (pErr) return { ok: false, error: pErr };
    pert = obj.perturbation as number[][][];
    const lastTp = (obj.init_states[2] as number[][][])[1];
    const pertTp = pert[1];
    for (let r = 0; r < 32; r++) {
      for (let c = 0; c < 36; c++) {
        if (lastTp[r][c] + pertTp[r][c] < -1e-6) {
          return {
            ok: false,
            error: `Post-perturbation precipitation at cell (${r},${c}) becomes negative (${(lastTp[r][c] + pertTp[r][c]).toFixed(3)} mm/day).`,
          };
        }
      }
    }
  }

  return {
    ok: true,
    payload: {
      init_states: obj.init_states as number[][][][],
      doy_sincos: [obj.doy_sincos[0], obj.doy_sincos[1]],
      steps: obj.steps,
      perturbation: pert,
    },
  };
}

function longestTrueStreak(boolMatrix: boolean[][]): number {
  let maxStreak = 0;
  for (let c = 0; c < boolMatrix.length; c++) {
    let cur = 0;
    const series = boolMatrix[c];
    for (let d = 0; d < series.length; d++) {
      if (series[d]) {
        cur++;
        if (cur > maxStreak) maxStreak = cur;
      } else {
        cur = 0;
      }
    }
  }
  return maxStreak;
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
    let nHeatProxy = 0;
    let pHeatProxy = 0;
    let nMaxWind = 0;
    let pMaxWind = 0;
    let nHeavy = 0;
    let pHeavy = 0;
    let nStrongWind = 0;
    let pStrongWind = 0;
    let nHotDays = 0;
    let pHotDays = 0;
    let count = 0;

    const cells: [number, number][] = [];
    for (let i = 0; i < 32; i++) {
      if (LAT[i] < la0 || LAT[i] > la1) continue;
      for (let j = 0; j < 36; j++) {
        if (LON[j] < lo0 || LON[j] > lo1) continue;
        cells.push([i, j]);
      }
    }

    const nHeavyMat: boolean[][] = cells.map(() => []);
    const pHeavyMat: boolean[][] = cells.map(() => []);
    const nWetMat: boolean[][] = cells.map(() => []);
    const pWetMat: boolean[][] = cells.map(() => []);

    for (const d of dayIndices) {
      for (let cIdx = 0; cIdx < cells.length; cIdx++) {
        const [i, j] = cells[cIdx];
        const nvTp = normalRollout[d][1][i][j];
        const nvT2m = normalRollout[d][0][i][j];
        const nvT2mC = nvT2m - 273.15;
        const nvTcwv = normalRollout[d][5][i][j];
        const nvW = Math.hypot(normalRollout[d][2][i][j], normalRollout[d][3][i][j]);
        const nvHeat = nvT2mC + 0.33 * (0.65 * Math.max(nvTcwv, 0)) - 0.7 * nvW - 4.0;

        nTp += nvTp;
        nT2m += nvT2m;
        nTcwv += nvTcwv;
        nWind += nvW;
        nHeatProxy += nvHeat;
        if (nvW > nMaxWind) nMaxWind = nvW;
        if (nvTp >= 10) nHeavy++;
        if (nvW >= 10) nStrongWind++;
        if (nvT2mC >= 35) nHotDays++;
        nHeavyMat[cIdx].push(nvTp >= 10);
        nWetMat[cIdx].push(nvTp >= 2.5);

        const src = perturbedRollout ?? normalRollout;
        const pvTp = src[d][1][i][j];
        const pvT2m = src[d][0][i][j];
        const pvT2mC = pvT2m - 273.15;
        const pvTcwv = src[d][5][i][j];
        const pvW = Math.hypot(src[d][2][i][j], src[d][3][i][j]);
        const pvHeat = pvT2mC + 0.33 * (0.65 * Math.max(pvTcwv, 0)) - 0.7 * pvW - 4.0;

        pTp += pvTp;
        pT2m += pvT2m;
        pTcwv += pvTcwv;
        pWind += pvW;
        pHeatProxy += pvHeat;
        if (pvW > pMaxWind) pMaxWind = pvW;
        if (pvTp >= 10) pHeavy++;
        if (pvW >= 10) pStrongWind++;
        if (pvT2mC >= 35) pHotDays++;
        pHeavyMat[cIdx].push(pvTp >= 10);
        pWetMat[cIdx].push(pvTp >= 2.5);
        count++;
      }
    }

    const denom = Math.max(count, 1);
    const nHeatMean = nHeatProxy / denom;
    const pHeatMean = pHeatProxy / denom;
    return {
      region: name,
      box,
      gridCells: cells.length,
      normalMeanTp: nTp / denom,
      perturbedMeanTp: pTp / denom,
      deltaTp: (pTp - nTp) / denom,
      normalHeavyCells: nHeavy,
      perturbedHeavyCells: pHeavy,
      normalHeavyExposurePct: Number(((100 * nHeavy) / denom).toFixed(2)),
      perturbedHeavyExposurePct: Number(((100 * pHeavy) / denom).toFixed(2)),
      normalPersistentHeavyStreak: longestTrueStreak(nHeavyMat),
      perturbedPersistentHeavyStreak: longestTrueStreak(pHeavyMat),
      normalPersistentWetStreak: longestTrueStreak(nWetMat),
      perturbedPersistentWetStreak: longestTrueStreak(pWetMat),
      normalMeanT2mC: nT2m / denom - 273.15,
      perturbedMeanT2mC: pT2m / denom - 273.15,
      deltaT2m: (pT2m - nT2m) / denom,
      normalHeatStressProxyC: Number(nHeatMean.toFixed(2)),
      perturbedHeatStressProxyC: Number(pHeatMean.toFixed(2)),
      deltaHeatStressProxyC: Number((pHeatMean - nHeatMean).toFixed(3)),
      normalHotDays: nHotDays,
      perturbedHotDays: pHotDays,
      normalMeanTcwv: nTcwv / denom,
      perturbedMeanTcwv: pTcwv / denom,
      deltaTcwv: (pTcwv - nTcwv) / denom,
      normalMeanWind: nWind / denom,
      perturbedMeanWind: pWind / denom,
      deltaWind: (pWind - nWind) / denom,
      normalStrongWindCells: nStrongWind,
      perturbedStrongWindCells: pStrongWind,
      normalMaxWindMs: Number(nMaxWind.toFixed(2)),
      perturbedMaxWindMs: Number(pMaxWind.toFixed(2)),
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
