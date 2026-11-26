import React from 'react';
import { API_BASE } from '../services/api';
import { AppSettings } from '../types/climate';

interface SettingsPageProps {
  settings: AppSettings;
  onChangeSettings: (next: AppSettings) => void;
}

export const SettingsPage: React.FC<SettingsPageProps> = ({ settings, onChangeSettings }) => {
  return (
    <div className="max-w-2xl space-y-6">
      <section className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-5 space-y-4">
        <h2 className="text-sm font-semibold text-white">Visualization &amp; Accessibility Preferences</h2>

        {(
          [
            [
              'showGeoOverlay',
              'Show Reference Coastline Overlay',
              'Display simplified India & Sri Lanka coastal reference vectors over the 1°×1° grid.',
            ],
            [
              'showGridLines',
              'Show 1°×1° Cell Borders',
              'Draw subtle cell borders to highlight individual grid cells on maps.',
            ],
            [
              'reducedMotion',
              'Reduced Motion Mode',
              'Disable transitions and spinner animations across the interface.',
            ],
            [
              'autoSaveToNeon',
              'Auto-Save Completed Runs to Neon',
              'Automatically persist lightweight regional summaries after each live simulation when Neon is online.',
            ],
          ] as const
        ).map(([key, title, desc]) => (
          <label
            key={key}
            className="flex items-start justify-between gap-4 rounded-lg border border-slate-800 bg-slate-950/60 p-3.5 cursor-pointer"
          >
            <div>
              <div className="text-xs font-medium text-slate-100">{title}</div>
              <div className="mt-0.5 text-[11px] text-slate-400">{desc}</div>
            </div>
            <input
              type="checkbox"
              checked={settings[key]}
              onChange={(e) => onChangeSettings({ ...settings, [key]: e.target.checked })}
              className="mt-1 h-4 w-4 accent-cyan-400"
            />
          </label>
        ))}
      </section>

      <section className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-5 space-y-2 text-xs">
        <h2 className="text-sm font-semibold text-white">Backend Connection Context</h2>
        <div className="rounded-lg bg-slate-950 p-3 font-mono text-[11px] text-cyan-300">
          API_BASE = {API_BASE || '(relative Vite proxy → http://127.0.0.1:8000)'}
        </div>
        <p className="text-[11px] text-slate-400">
          Physical units (K, mm/day, m/s, Pa, kg/m²) are locked to the R1 API contract and never
          altered in frontend bundles. Database credentials remain strictly backend-only.
        </p>
      </section>
    </div>
  );
};
