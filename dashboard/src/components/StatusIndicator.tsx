import React from 'react';

export type StatusKind = 'operational' | 'degraded' | 'offline' | 'untested';

interface StatusIndicatorProps {
  label: string;
  status: StatusKind;
  detail?: string;
}

const STYLES: Record<StatusKind, { bg: string; dot: string; text: string }> = {
  operational: {
    bg: 'border-emerald-500/30 bg-emerald-500/10',
    dot: 'bg-emerald-400',
    text: 'text-emerald-300',
  },
  degraded: {
    bg: 'border-amber-500/30 bg-amber-500/10',
    dot: 'bg-amber-400',
    text: 'text-amber-300',
  },
  offline: {
    bg: 'border-rose-500/30 bg-rose-500/10',
    dot: 'bg-rose-400',
    text: 'text-rose-300',
  },
  untested: {
    bg: 'border-slate-700 bg-slate-800/60',
    dot: 'bg-slate-400',
    text: 'text-slate-300',
  },
};

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({ label, status, detail }) => {
  const s = STYLES[status];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium ${s.bg} ${s.text}`}
      title={detail}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${s.dot}`} aria-hidden="true" />
      <span>{label}</span>
      {detail && <span className="hidden opacity-80 sm:inline">· {detail}</span>}
    </span>
  );
};
