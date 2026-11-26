import React from 'react';
import {
  Activity,
  BarChart3,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  CloudRain,
  Database,
  FlaskConical,
  Globe2,
  LayoutDashboard,
  Map as MapIcon,
  Settings,
  X,
} from 'lucide-react';
import { NavSection } from '../types/climate';

export interface NavItem {
  id: NavSection;
  label: string;
  icon: React.ReactNode;
}

export const NAV_ITEMS: NavItem[] = [
  { id: 'overview', label: 'Overview', icon: <LayoutDashboard className="h-4 w-4" /> },
  { id: 'forecast', label: 'Forecast Studio', icon: <MapIcon className="h-4 w-4" /> },
  { id: 'scenarios', label: 'Scenario Lab', icon: <FlaskConical className="h-4 w-4" /> },
  { id: 'impact', label: 'Impact Explorer', icon: <Globe2 className="h-4 w-4" /> },
  { id: 'saved', label: 'Saved Simulations', icon: <Database className="h-4 w-4" /> },
  { id: 'performance', label: 'Model Performance', icon: <BarChart3 className="h-4 w-4" /> },
  { id: 'methodology', label: 'Data & Methodology', icon: <BookOpen className="h-4 w-4" /> },
  { id: 'status', label: 'System Status', icon: <Activity className="h-4 w-4" /> },
  { id: 'settings', label: 'Settings', icon: <Settings className="h-4 w-4" /> },
];

interface SidebarProps {
  active: NavSection;
  onSelect: (section: NavSection) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  active,
  onSelect,
  collapsed,
  onToggleCollapse,
  mobileOpen,
  onCloseMobile,
}) => {
  const navContent = (
    <div className="flex h-full flex-col justify-between bg-[#070d1b] border-r border-slate-800/90">
      <div>
        <div className="flex items-center justify-between border-b border-slate-800/90 px-3 py-3.5">
          <div className="flex items-center gap-2.5 overflow-hidden">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-cyan-500/15 text-cyan-400 border border-cyan-500/30">
              <CloudRain className="h-4 w-4" aria-hidden="true" />
            </div>
            {!collapsed && (
              <div className="truncate">
                <div className="text-sm font-bold tracking-tight text-white">MONSOONVERSE</div>
                <div className="text-[10px] text-slate-400">Climate Digital Twin</div>
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={onToggleCollapse}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className="hidden rounded-md p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-200 lg:inline-flex"
          >
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </button>
          <button
            type="button"
            onClick={onCloseMobile}
            aria-label="Close navigation menu"
            className="rounded-md p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-200 lg:hidden"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <nav aria-label="Primary navigation" className="space-y-1 p-2">
          {NAV_ITEMS.map((item) => {
            const isActive = active === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  onSelect(item.id);
                  onCloseMobile();
                }}
                aria-current={isActive ? 'page' : undefined}
                title={collapsed ? item.label : undefined}
                className={`flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-xs font-medium transition-colors ${
                  isActive
                    ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
                    : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200 border border-transparent'
                }`}
              >
                <span className="shrink-0">{item.icon}</span>
                {!collapsed && <span className="truncate">{item.label}</span>}
              </button>
            );
          })}
        </nav>
      </div>

      {!collapsed && (
        <div className="border-t border-slate-800/80 p-3 text-[11px] text-slate-400">
          <div className="font-medium text-slate-300">MonsoonVerse-ClimResCNN</div>
          <div>ERA5 2005–2021 · 32×36 (1°×1°)</div>
        </div>
      )}
    </div>
  );

  return (
    <>
      <aside
        className={`hidden shrink-0 lg:block transition-all ${
          collapsed ? 'w-14' : 'w-56'
        }`}
      >
        <div className="sticky top-0 h-screen">{navContent}</div>
      </aside>

      {mobileOpen && (
        <div
          className="fixed inset-0 z-50 flex lg:hidden"
          role="dialog"
          aria-modal="true"
          aria-label="Mobile navigation drawer"
        >
          <div
            className="fixed inset-0 bg-black/70"
            onClick={onCloseMobile}
            aria-hidden="true"
          />
          <div className="relative z-10 h-full w-64">{navContent}</div>
        </div>
      )}
    </>
  );
};
