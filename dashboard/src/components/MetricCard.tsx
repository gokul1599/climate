import React from 'react';

interface MetricCardProps {
  label: string;
  value: string;
  sublabel?: string;
  accent?: 'cyan' | 'emerald' | 'amber' | 'rose' | 'slate';
  icon?: React.ReactNode;
}

const ACCENT_MAP: Record<NonNullable<MetricCardProps['accent']>, string> = {
  cyan: 'text-cyan-300',
  emerald: 'text-emerald-300',
  amber: 'text-amber-300',
  rose: 'text-rose-300',
  slate: 'text-slate-100',
};

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  sublabel,
  accent = 'cyan',
  icon,
}) => {
  return (
    <div className="rounded-xl border border-slate-800/90 bg-navy-900/80 p-4 shadow-sm">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-slate-400">{label}</span>
        {icon && <span className="text-slate-500">{icon}</span>}
      </div>
      <div className={`mt-2 text-xl font-semibold tracking-tight ${ACCENT_MAP[accent]}`}>
        {value}
      </div>
      {sublabel && <p className="mt-1 text-[11px] text-slate-400">{sublabel}</p>}
    </div>
  );
};
