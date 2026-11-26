import React, { useState } from 'react';
import { Activity, Cpu, Database, RefreshCw, Server } from 'lucide-react';
import { checkHealth, fetchDbStatus, fetchSystemStatus } from '../services/api';
import { SystemStatus } from '../types/climate';
import { StatusIndicator, StatusKind } from '../components/StatusIndicator';

interface SystemStatusPageProps {
  systemStatus: SystemStatus | null;
  lastInferenceMs: number | null;
  lastSimAt: string | null;
  onRefreshStatus: (st: SystemStatus) => void;
}

export const SystemStatusPage: React.FC<SystemStatusPageProps> = ({
  systemStatus,
  lastInferenceMs,
  lastSimAt,
  onRefreshStatus,
}) => {
  const [checking, setChecking] = useState(false);
  const [probeLatencyMs, setProbeLatencyMs] = useState<number | null>(null);
  const [probeError, setProbeError] = useState<string | null>(null);

  const runLiveChecks = async () => {
    setChecking(true);
    setProbeError(null);
    const t0 = performance.now();
    try {
      await checkHealth();
      const [sys] = await Promise.all([fetchSystemStatus(), fetchDbStatus()]);
      setProbeLatencyMs(performance.now() - t0);
      onRefreshStatus(sys);
    } catch (e) {
      setProbeError(e instanceof Error ? e.message : String(e));
    } finally {
      setChecking(false);
    }
  };

  const apiStatus: StatusKind = systemStatus
    ? systemStatus.fastapi === 'operational'
      ? 'operational'
      : 'degraded'
    : 'untested';
  const modelStatus: StatusKind = systemStatus
    ? systemStatus.model_loaded && systemStatus.assets_present
      ? 'operational'
      : 'degraded'
    : 'untested';
  const dbStatus: StatusKind = !systemStatus
    ? 'untested'
    : systemStatus.neon.connected
    ? 'operational'
    : systemStatus.neon.mode === 'error'
    ? 'degraded'
    : 'offline';

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-800/90 bg-navy-900/80 p-4">
        <div>
          <h2 className="text-sm font-semibold text-white">
            Independent Runtime &amp; Infrastructure Health
          </h2>
          <p className="text-xs text-slate-400">
            Inference (/simulate) and Neon persistence are decoupled; database unavailability never
            blocks model forecasts.
          </p>
        </div>
        <button
          type="button"
          onClick={runLiveChecks}
          disabled={checking}
          className="inline-flex items-center gap-2 rounded-lg bg-cyan-500 px-3.5 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400 disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${checking ? 'animate-spin' : ''}`} />
          {checking ? 'Probing...' : 'Run Live Checks'}
        </button>
      </div>

      {probeError && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-950/40 px-4 py-3 text-xs text-rose-200">
          Probe failed: {probeError}
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-2 text-xs font-medium text-slate-300">
              <Server className="h-4 w-4 text-cyan-400" />
              FastAPI Backend
            </span>
            <StatusIndicator
              label={apiStatus === 'operational' ? 'Operational' : apiStatus === 'untested' ? 'Not tested' : 'Offline'}
              status={apiStatus}
            />
          </div>
          <p className="text-[11px] text-slate-400">
            Probe RTT: {probeLatencyMs !== null ? `${probeLatencyMs.toFixed(1)} ms` : 'Click Run Live Checks'}
          </p>
        </div>

        <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-2 text-xs font-medium text-slate-300">
              <Cpu className="h-4 w-4 text-emerald-400" />
              Model &amp; Assets
            </span>
            <StatusIndicator
              label={modelStatus === 'operational' ? 'Operational' : modelStatus === 'untested' ? 'Not tested' : 'Degraded'}
              status={modelStatus}
            />
          </div>
          <p className="text-[11px] text-slate-400">
            {systemStatus?.model_name ?? 'MonsoonVerse-ClimResCNN'} (
            {systemStatus?.active_variant ?? 'incumbent'}) ·{' '}
            {systemStatus?.climatology_loaded ? 'Climatology loaded' : 'No clim conditioning'}
          </p>
          {systemStatus?.architecture_summary && (
            <p className="font-mono text-[10px] text-slate-400">{systemStatus.architecture_summary}</p>
          )}
          {systemStatus?.checkpoint_sha256 && (
            <p className="font-mono text-[10px] text-cyan-300/80">
              SHA-256: {systemStatus.checkpoint_sha256.slice(0, 16)}... · Challenger{' '}
              {systemStatus.challenger_available ? 'ready' : 'not built'}
            </p>
          )}
        </div>

        <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-2 text-xs font-medium text-slate-300">
              <Database className="h-4 w-4 text-amber-400" />
              Neon PostgreSQL
            </span>
            <StatusIndicator
              label={
                dbStatus === 'operational'
                  ? 'Operational'
                  : dbStatus === 'degraded'
                  ? 'Degraded'
                  : dbStatus === 'offline'
                  ? 'Offline'
                  : 'Not tested'
              }
              status={dbStatus}
            />
          </div>
          <p className="text-[11px] text-slate-400">
            {systemStatus?.neon.connected
              ? `Live (${systemStatus.neon.database}) · ${systemStatus.neon.run_count ?? 0} saved runs`
              : systemStatus?.neon.detail || systemStatus?.neon.mode || 'Unknown'}
          </p>
        </div>

        <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-2 text-xs font-medium text-slate-300">
              <Activity className="h-4 w-4 text-violet-400" />
              CPU Simulation Engine
            </span>
            <StatusIndicator
              label={lastInferenceMs !== null ? 'Operational' : 'Not tested'}
              status={lastInferenceMs !== null ? 'operational' : 'untested'}
            />
          </div>
          <p className="text-[11px] text-slate-400">
            Last run: {lastInferenceMs !== null ? `${lastInferenceMs.toFixed(1)} ms` : 'None yet'}
            {lastSimAt ? ` · ${new Date(lastSimAt).toLocaleTimeString()}` : ''}
          </p>
        </div>
      </div>
    </div>
  );
};
