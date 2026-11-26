import React from 'react';

interface LoadingSkeletonProps {
  lines?: number;
  heightClass?: string;
}

export const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({
  lines = 3,
  heightClass = 'h-24',
}) => (
  <div className="space-y-3 rounded-xl border border-slate-800 bg-navy-900/60 p-4" aria-busy="true">
    <div className={`w-full animate-pulse rounded-lg bg-slate-800/70 ${heightClass}`} />
    {Array.from({ length: lines }, (_, idx) => (
      <div
        key={idx}
        className="h-3 animate-pulse rounded bg-slate-800/60"
        style={{ width: `${90 - idx * 12}%` }}
      />
    ))}
  </div>
);
