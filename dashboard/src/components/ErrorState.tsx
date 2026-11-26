import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'Operation Error',
  message,
  onRetry,
}) => (
  <div
    role="alert"
    className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-rose-500/40 bg-rose-950/40 px-4 py-3 text-xs text-rose-200"
  >
    <div className="flex items-start gap-2.5">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" aria-hidden="true" />
      <div>
        <div className="font-semibold text-rose-100">{title}</div>
        <div className="mt-0.5 text-rose-200/90">{message}</div>
      </div>
    </div>
    {onRetry && (
      <button
        type="button"
        onClick={onRetry}
        className="inline-flex items-center gap-1.5 rounded-lg border border-rose-400/40 bg-rose-900/50 px-3 py-1.5 font-medium text-rose-100 hover:bg-rose-800/60"
      >
        <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
        Retry
      </button>
    )}
  </div>
);
