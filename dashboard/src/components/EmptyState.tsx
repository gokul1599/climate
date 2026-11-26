import React from 'react';
import { Inbox } from 'lucide-react';

interface EmptyStateProps {
  title: string;
  description: string;
  action?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ title, description, action }) => (
  <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-800 bg-navy-900/40 px-6 py-10 text-center">
    <div className="mb-3 rounded-full bg-slate-800/70 p-3 text-slate-400">
      <Inbox className="h-5 w-5" aria-hidden="true" />
    </div>
    <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
    <p className="mt-1 max-w-md text-xs text-slate-400">{description}</p>
    {action && <div className="mt-4">{action}</div>}
  </div>
);
