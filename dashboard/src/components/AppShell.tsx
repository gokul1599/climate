import React, { useState } from 'react';
import { AppSettings, DbStatus, NavSection } from '../types/climate';
import { Sidebar } from './Sidebar';
import { Topbar } from './Topbar';

interface AppShellProps {
  activeSection: NavSection;
  onSelectSection: (section: NavSection) => void;
  backendHealthy: boolean | null;
  modelName: string | null;
  dbStatus: DbStatus | null;
  inferenceMs: number | null;
  settings: AppSettings;
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({
  activeSection,
  onSelectSection,
  backendHealthy,
  modelName,
  dbStatus,
  inferenceMs,
  settings,
  children,
}) => {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div
      className={`min-h-screen bg-[#050a14] text-slate-100 flex ${
        settings.reducedMotion ? 'reduced-motion' : ''
      }`}
    >
      <Sidebar
        active={activeSection}
        onSelect={onSelectSection}
        collapsed={collapsed}
        onToggleCollapse={() => setCollapsed((c) => !c)}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar
          activeSection={activeSection}
          backendHealthy={backendHealthy}
          modelName={modelName}
          dbStatus={dbStatus}
          inferenceMs={inferenceMs}
          onOpenMobileMenu={() => setMobileOpen(true)}
        />
        <main className="mx-auto w-full max-w-7xl flex-1 space-y-6 px-4 py-6 sm:px-6">
          {children}
        </main>
      </div>
    </div>
  );
};
