import React from 'react';
import { Menu } from 'lucide-react';
import { DbStatus, NavSection } from '../types/climate';
import { NAV_ITEMS } from './Sidebar';
import { StatusIndicator, StatusKind } from './StatusIndicator';

interface TopbarProps {
  activeSection: NavSection;
  backendHealthy: boolean | null;
  modelName: string | null;
  dbStatus: DbStatus | null;
  inferenceMs: number | null;
  onOpenMobileMenu: () => void;
}

export const Topbar: React.FC<TopbarProps> = ({
  activeSection,
  backendHealthy,
  modelName,
  dbStatus,
  inferenceMs,
  onOpenMobileMenu,
}) => {
  const title = NAV_ITEMS.find((i) => i.id === activeSection)?.label ?? 'Overview';

  const backendStatus: StatusKind =
    backendHealthy === null ? 'untested' : backendHealthy ? 'operational' : 'offline';
  const modelStatus: StatusKind =
    backendHealthy === null ? 'untested' : backendHealthy ? 'operational' : 'offline';
  const neonStatus: StatusKind = !dbStatus
    ? 'untested'
    : dbStatus.connected
    ? 'operational'
    : dbStatus.mode === 'error'
    ? 'degraded'
    : 'offline';

  return (
    <header className="sticky top-0 z-30 border-b border-slate-800/90 bg-[#070d1b]/95 px-4 py-3 backdrop-blur-sm sm:px-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onOpenMobileMenu}
            aria-label="Open navigation menu"
            className="rounded-lg border border-slate-800 p-1.5 text-slate-300 hover:bg-slate-800 lg:hidden"
          >
            <Menu className="h-4 w-4" />
          </button>
          <div>
            <h1 className="text-base font-semibold tracking-tight text-white">{title}</h1>
            <p className="text-[11px] text-slate-400">
              India Regional Grid (6.5°–37.5°N, 65.5°–100.5°E) · ERA5 Daily 2005–2021
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <StatusIndicator
            label="Backend"
            status={backendStatus}
            detail={backendHealthy ? 'FastAPI Online' : backendHealthy === false ? 'Unreachable' : 'Checking'}
          />
          <StatusIndicator
            label="Model"
            status={modelStatus}
            detail={
              backendHealthy
                ? `${modelName || 'ClimResCNN'}${inferenceMs !== null ? ` · ${inferenceMs.toFixed(0)}ms` : ''}`
                : 'Unavailable'
            }
          />
          <StatusIndicator
            label="Neon DB"
            status={neonStatus}
            detail={
              dbStatus?.connected
                ? dbStatus.database || 'neondb'
                : dbStatus?.mode === 'offline-disabled'
                ? 'Offline Mode'
                : dbStatus?.mode || 'Checking'
            }
          />
        </div>
      </div>
    </header>
  );
};
