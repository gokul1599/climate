import React from 'react';
import {
  CustomPerturbationConfig,
  GridCellSelection,
  SampleWindow,
  ScenarioKey,
  ScenarioMeta,
  VariableIndex,
  VariableMeta,
} from '../types/climate';
import { ForecastControls } from '../components/ForecastControls';
import { ScenarioComparison } from '../components/ScenarioComparison';
import { RegionalImpactTable } from '../components/RegionalImpactTable';
import { EmptyState } from '../components/EmptyState';
import { LoadingSkeleton } from '../components/LoadingSkeleton';
import { RegionalIndicator } from '../types/climate';

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
  onReset,
  onOpenSave,
  onOpenExport,
  chartRef,
}) => {
  return (
    <div className="space-y-6">
      <h2 className="sr-only">Scenario Simulation Lab Controls and Maps</h2>
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
          description="Choose Warm Bay of Bengal, Arabian Sea Monsoon Stall, Zero Perturbation Check, or Custom Shock and click Run Simulation."
        />
      )}
    </div>
  );
};
