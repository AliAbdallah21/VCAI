const TREND_CONFIG = {
  improving:         { icon: '↑', color: '#a5d6a7' },
  declining:         { icon: '↓', color: '#ffb4ab' },
  plateau:           { icon: '→', color: '#e9c46a' },
  insufficient_data: { icon: '–', color: 'var(--text-subtle)' },
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
    totalImprovement > 0 ? '#a5d6a7' :
    totalImprovement < 0 ? '#ffb4ab' :
    '#e9c46a';

  return (
    <button
      onClick={onClick}
      className="p-4 rounded-xl text-left transition-all duration-200 w-full"
      style={{
        background: isSelected ? `${color}12` : 'var(--bg-card-alt)',
        border: `1px solid ${isSelected ? color + '55' : 'var(--border)'}`,
      }}
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-2.5">
        <span
          className="text-xs font-semibold truncate mr-2"
          style={{ color: isSelected ? color : 'var(--text-secondary)' }}
        >
          {label}
        </span>
        <span className="text-sm font-bold flex-shrink-0" style={{ color: trendInfo.color }}>
          {trendInfo.icon}
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-1 rounded-full mb-3" style={{ background: 'var(--border)' }}>
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
            <div className="text-xs" style={{ color: 'var(--text-subtle)' }}>
              {focusSessions} focus
            </div>
          )}
        </div>
      </div>
    </button>
  );
}
