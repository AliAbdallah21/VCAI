const INSIGHT_STYLES = {
  improvement: {
    bg: 'rgba(74,222,128,0.07)',
    border: 'rgba(74,222,128,0.18)',
    iconBg: 'rgba(74,222,128,0.15)',
    iconColor: '#4ade80',
    Icon: () => (
      <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <path d="M12 19V5M5 12l7-7 7 7"/>
      </svg>
    ),
  },
  strength: {
    bg: 'rgba(245,158,11,0.07)',
    border: 'rgba(245,158,11,0.18)',
    iconBg: 'rgba(245,158,11,0.15)',
    iconColor: '#f59e0b',
    Icon: () => (
      <svg width="13" height="13" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
      </svg>
    ),
  },
  plateau: {
    bg: 'rgba(245,158,11,0.07)',
    border: 'rgba(245,158,11,0.18)',
    iconBg: 'rgba(245,158,11,0.15)',
    iconColor: '#f59e0b',
    Icon: () => (
      <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <path d="M5 12h14"/>
      </svg>
    ),
  },
  milestone: {
    bg: 'rgba(139,92,246,0.07)',
    border: 'rgba(139,92,246,0.18)',
    iconBg: 'rgba(139,92,246,0.15)',
    iconColor: '#a78bfa',
    Icon: () => (
      <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <path d="M8.21 13.89L7 23l5-3 5 3-1.21-9.11M12 2a5 5 0 100 10A5 5 0 0012 2z"/>
      </svg>
    ),
  },
  best_score: {
    bg: 'rgba(59,130,246,0.07)',
    border: 'rgba(59,130,246,0.18)',
    iconBg: 'rgba(59,130,246,0.15)',
    iconColor: '#60a5fa',
    Icon: () => (
      <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <path d="M12 15l-3.09 1.64.59-3.45L7 11.14l3.46-.5L12 7.5l1.54 3.14 3.46.5-2.5 2.05.59 3.45L12 15z"/>
      </svg>
    ),
  },
};

const DEFAULT_STYLE = INSIGHT_STYLES.improvement;

export default function InsightsPanel({ insights }) {
  if (!insights.length) return null;

  return (
    <div className="mb-6">
      <h3 className="text-xs font-semibold mb-3" style={{ color: 'rgba(148,163,184,0.5)' }}>
        Coaching Insights
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
        {insights.map((insight, i) => {
          const style = INSIGHT_STYLES[insight.type] ?? DEFAULT_STYLE;
          const { Icon } = style;
          return (
            <div
              key={i}
              className="flex items-start gap-3 p-3.5 rounded-xl"
              style={{ background: style.bg, border: `1px solid ${style.border}` }}
            >
              <div
                className="w-6 h-6 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5"
                style={{ background: style.iconBg, color: style.iconColor }}
              >
                <Icon />
              </div>
              <p
                className="text-xs leading-relaxed"
                style={{ color: 'rgba(226,232,240,0.8)', direction: 'rtl', textAlign: 'right' }}
              >
                {insight.text}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
