export type VariableIndex = 0 | 1 | 2 | 3 | 4 | 5;

export interface VariableMeta {
  index: VariableIndex;
  key: string;
  label: string;
  unit: string;
  digits: number;
}

export const VARIABLES: VariableMeta[] = [
  { index: 0, key: 't2m', label: '2m Temperature', unit: 'K', digits: 2 },
  { index: 1, key: 'tp', label: 'Total Precipitation', unit: 'mm/day', digits: 2 },
  { index: 2, key: 'u10', label: 'Eastward Wind (u10)', unit: 'm/s', digits: 2 },
  { index: 3, key: 'v10', label: 'Northward Wind (v10)', unit: 'm/s', digits: 2 },
  { index: 4, key: 'msl', label: 'Mean Sea-Level Pressure', unit: 'Pa', digits: 0 },
  { index: 5, key: 'tcwv', label: 'Total Column Water Vapour', unit: 'kg/m²', digits: 2 },
];

export type ScenarioKey = 'normal' | 'warm_ocean' | 'monsoon_stall' | 'zeros';

export interface ScenarioMeta {
  key: ScenarioKey;
  label: string;
  description: string;
  box?: [number, number, number, number]; // latMin, latMax, lonMin, lonMax
}

export const SCENARIOS: ScenarioMeta[] = [
  {
    key: 'normal',
    label: 'Normal Forecast',
    description: 'Unperturbed 10-day autoregressive forecast from the 3-day initial state.',
  },
  {
    key: 'warm_ocean',
    label: 'Warm Bay of Bengal',
    description: '+3.0 K temperature and +15% TCWV over Bay of Bengal (8°–20°N, 82°–94°E).',
    box: [8, 20, 82, 94],
  },
  {
    key: 'monsoon_stall',
    label: 'Arabian Sea Monsoon Stall',
    description: '-40% u10 and -40% v10 wind over Arabian Sea (8°–20°N, 65°–75°E).',
    box: [8, 20, 65, 75],
  },
  {
    key: 'zeros',
    label: 'Zero Perturbation Check',
    description: 'All-zero [6, 32, 36] perturbation array; verifies deterministic consistency.',
  },
];

export const LAT = Array.from({ length: 32 }, (_, i) => 6.5 + i);
export const LON = Array.from({ length: 36 }, (_, j) => 65.5 + j);

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

export function buildPerturbation(
  scenario: ScenarioKey,
  lastState: number[][][],
): number[][][] | null {
  if (scenario === 'normal') return null;
  const pert: number[][][] = Array.from({ length: 6 }, () =>
    Array.from({ length: 32 }, () => Array(36).fill(0)),
  );
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
  }
  return pert;
}

export async function checkHealth(): Promise<{ status: string; model_name: string }> {
  const res = await fetch('/health', { method: 'POST' });
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function fetchSampleWindows(): Promise<SampleWindow[]> {
  const res = await fetch('/api/sample-windows');
  if (!res.ok) throw new Error(`Failed to fetch sample windows: ${res.status}`);
  const data = await res.json();
  return data.windows;
}

export async function runSimulation(
  initStates: number[][][][],
  doySincos: [number, number],
  steps: number,
  perturbation: number[][][] | null,
): Promise<SimulateResponse> {
  const res = await fetch('/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      init_states: initStates,
      doy_sincos: doySincos,
      steps,
      perturbation,
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Simulation failed (${res.status}): ${text}`);
  }
  return res.json();
}

export interface RegionalIndicator {
  region: string;
  normalMeanTp: number;
  perturbedMeanTp: number;
  deltaTp: number;
  normalHeavyCells: number;
  perturbedHeavyCells: number;
  deltaT2m: number;
  deltaTcwv: number;
}

const REGIONS: { name: string; box: [number, number, number, number] }[] = [
  { name: 'Western Ghats & Konkan', box: [10, 20, 72, 77] },
  { name: 'Central India Monsoon Core', box: [18, 26, 75, 85] },
  { name: 'Indo-Gangetic Plain', box: [24, 31, 75, 88] },
  { name: 'Northeast India', box: [22, 29, 88, 97] },
  { name: 'Bay of Bengal', box: [8, 20, 82, 94] },
  { name: 'Arabian Sea', box: [8, 20, 65, 75] },
];

export function computeRegionalIndicators(
  normalRollout: number[][][][],
  perturbedRollout: number[][][][] | null,
): RegionalIndicator[] {
  const steps = normalRollout.length;
  return REGIONS.map(({ name, box: [la0, la1, lo0, lo1] }) => {
    let nTp = 0;
    let pTp = 0;
    let nT2m = 0;
    let pT2m = 0;
    let nTcwv = 0;
    let pTcwv = 0;
    let nHeavy = 0;
    let pHeavy = 0;
    let count = 0;

    for (let d = 0; d < steps; d++) {
      for (let i = 0; i < 32; i++) {
        if (LAT[i] < la0 || LAT[i] > la1) continue;
        for (let j = 0; j < 36; j++) {
          if (LON[j] < lo0 || LON[j] > lo1) continue;
          const nvTp = normalRollout[d][1][i][j];
          nTp += nvTp;
          nT2m += normalRollout[d][0][i][j];
          nTcwv += normalRollout[d][5][i][j];
          if (nvTp >= 10) nHeavy++;

          const pvTp = perturbedRollout ? perturbedRollout[d][1][i][j] : nvTp;
          pTp += pvTp;
          pT2m += perturbedRollout ? perturbedRollout[d][0][i][j] : normalRollout[d][0][i][j];
          pTcwv += perturbedRollout ? perturbedRollout[d][5][i][j] : normalRollout[d][5][i][j];
          if (pvTp >= 10) pHeavy++;
          count++;
        }
      }
    }

    const nMeanTp = count ? nTp / count : 0;
    const pMeanTp = count ? pTp / count : 0;
    return {
      region: name,
      normalMeanTp: nMeanTp,
      perturbedMeanTp: pMeanTp,
      deltaTp: pMeanTp - nMeanTp,
      normalHeavyCells: nHeavy,
      perturbedHeavyCells: pHeavy,
      deltaT2m: count ? (pT2m - nT2m) / count : 0,
      deltaTcwv: count ? (pTcwv - nTcwv) / count : 0,
    };
  });
}
