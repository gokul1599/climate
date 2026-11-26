import React, { useRef } from 'react';
import {
  GridCellSelection,
  SampleWindow,
  ScenarioKey,
  ScenarioMeta,
  VariableIndex,
  VariableMeta,
} from '../types/climate';
import { REGIONS } from '../services/api';
import { ForecastControls } from '../components/ForecastControls';
import { ForecastMap } from '../components/ForecastMap';
import { ForecastCharts } from '../components/ForecastCharts';
import { EmptyState } from '../components/EmptyState';
import { LoadingSkeleton } from '../components/LoadingSkeleton';

interface ForecastStudioPageProps {
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
  loading: boolean;
  dbConnected: boolean;
  selectedRegion: string;
  selectedCell: GridCellSelection | null;
  onSelectCell: (c: GridCellSelection | null) => void;
  showGeoOverlay: boolean;
  showGridLines: boolean;
  onRun: () => void;
  onReset: () => void;
  onOpenSave: () => void;
  onOpenExport: () => void;
  chartRef: React.RefObject<HTMLDivElement>;
}

export const ForecastStudioPage: React.FC<ForecastStudioPageProps> = ({
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
  loading,
  dbConnected,
  selectedRegion,
  selectedCell,
  onSelectCell,
  showGeoOverlay,
  showGridLines,
  onRun,
  onReset,
  onOpenSave,
  onOpenExport,
  chartRef,
}) => {
  const localChartRef = useRef<HTMLDivElement>(null);
  const activeChartRef = chartRef ?? localChartRef;
  const activeRegionBox = REGIONS.find((r) => r.name === selectedRegion)?.box;

  return (
    <div className="space-y-6">
      <h2 className="sr-only">Forecast Controls and Spatial Maps</h2>
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

      {loading && !normalRollout ? (
        <LoadingSkeleton heightClass="h-96" />
      ) : normalRollout ? (
        <>
          <div className={`grid grid-cols-1 gap-4 ${perturbedRollout ? 'lg:grid-cols-2' : ''}`}>
            <ForecastMap
              title={`Normal Forecast — Day ${dayIdx + 1} (${variable.label})`}
              subtitle={variable.description}
              grid={normalRollout[dayIdx][varIdx]}
              unit={variable.unit}
              digits={variable.digits}
              palette={variable.palette}
              regionBox={activeRegionBox}
              selectedCell={selectedCell}
              onSelectCell={(c) => onSelectCell(c)}
              showGeoOverlay={showGeoOverlay}
              showGridLines={showGridLines}
            />
            {perturbedRollout && (
              <ForecastMap
                title={`${scenarioMeta.label} — Day ${dayIdx + 1}`}
                subtitle="Perturbed physical-unit rollout"
                grid={perturbedRollout[dayIdx][varIdx]}
                unit={variable.unit}
                digits={variable.digits}
                palette={variable.palette}
                scenarioBox={scenarioMeta.box}
                regionBox={activeRegionBox}
                selectedCell={selectedCell}
                onSelectCell={(c) => onSelectCell(c)}
                showGeoOverlay={showGeoOverlay}
                showGridLines={showGridLines}
              />
            )}
          </div>

          <ForecastCharts
            normalRollout={normalRollout}
            perturbedRollout={perturbedRollout}
            variable={variable}
            selectedRegionName={selectedRegion}
            selectedCell={selectedCell}
            onClearCell={() => onSelectCell(null)}
            chartRef={activeChartRef}
          />
        </>
      ) : (
        <EmptyState
          title="Ready to Simulate"
          description="Select an initial validation window, variable, and scenario, then click Run Simulation."
        />
      )}
    </div>
  );
};
