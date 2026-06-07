const TREND_CONFIG = {
  improving:         { icon: '↑', color: '#4ade80' },
  declining:         { icon: '↓', color: '#ef4444' },
  plateau:           { icon: '→', color: '#f59e0b' },
  insufficient_data: { icon: '–', color: 'rgba(148,163,184,0.35)' },
};

export default function SkillProgressCard({
  label,
  currentScore,
  trend,
  totalImprovement,
  focusSessions,
  isSelected,
  onClick,
  color,
}) {
  const trendInfo = TREND_CONFIG[trend] ?? TREND_CONFIG.insufficient_data;
  const score = currentScore != null ? Math.round(currentScore) : null;

  const deltaText = (() => {
    if (totalImprovement == null) return null;
    const n = Math.round(totalImprovement);
    return n > 0 ? `+${n}` : String(n);
  })();

  const deltaColor =
    totalImprovement > 0 ? '#4ade80' :
    totalImprovement < 0 ? '#ef4444' :
    '#f59e0b';

  return (
    <button
      onClick={onClick}
      className="p-4 rounded-xl text-left transition-all duration-200 w-full"
      style={{
        background: isSelected ? `${color}12` : 'rgba(255,255,255,0.02)',
        border: `1px solid ${isSelected ? color + '55' : 'rgba(255,255,255,0.06)'}`,
      }}
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-2.5">
        <span
          className="text-xs font-semibold truncate mr-2"
          style={{ color: isSelected ? color : 'rgba(148,163,184,0.65)' }}
        >
          {label}
        </span>
        <span className="text-sm font-bold flex-shrink-0" style={{ color: trendInfo.color }}>
          {trendInfo.icon}
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-1 rounded-full mb-3" style={{ background: 'rgba(255,255,255,0.06)' }}>
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{
            width: `${Math.min(score ?? 0, 100)}%`,
            background: `linear-gradient(90deg, ${color}70, ${color})`,
          }}
        />
      </div>

      {/* Score + delta + focus sessions */}
      <div className="flex items-end justify-between">
        <span className="text-2xl font-bold text-white leading-none">
          {score ?? '—'}
        </span>
        <div className="text-right leading-tight">
          {deltaText && (
            <div className="text-xs font-semibold" style={{ color: deltaColor }}>
              {deltaText} pts
            </div>
          )}
          {focusSessions > 0 && (
            <div className="text-xs" style={{ color: 'rgba(148,163,184,0.35)' }}>
              {focusSessions} focus
            </div>
          )}
        </div>
      </div>
    </button>
  );
}
