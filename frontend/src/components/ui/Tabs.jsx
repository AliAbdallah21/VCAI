/* Tabs — segmented pill-style tab control (Amethyst Precision) */

export default function Tabs({ tabs, active, onChange, badge }) {
  return (
    <div
      style={{
        display: 'flex',
        gap: 2,
        padding: 3,
        borderRadius: 11,
        background: 'rgba(222,183,255,0.04)',
        border: '1px solid var(--border)',
        width: 'fit-content',
        flexWrap: 'wrap',
      }}
    >
      {tabs.map(t => {
        const isActive = t.key === active;
        const count = badge?.[t.key];
        return (
          <button
            key={t.key}
            onClick={() => onChange(t.key)}
            style={{
              padding: '6px 14px',
              borderRadius: 8,
              fontSize: 12.5,
              fontWeight: isActive ? 600 : 500,
              color: isActive ? 'var(--text-primary)' : 'var(--text-muted)',
              background: isActive ? 'rgba(222,183,255,0.12)' : 'transparent',
              border: `1px solid ${isActive ? 'rgba(222,183,255,0.3)' : 'transparent'}`,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              transition: 'all 0.13s ease',
              whiteSpace: 'nowrap',
            }}
            onMouseEnter={e => {
              if (!isActive) {
                e.currentTarget.style.color = 'var(--text-secondary)';
                e.currentTarget.style.background = 'rgba(222,183,255,0.06)';
              }
            }}
            onMouseLeave={e => {
              if (!isActive) {
                e.currentTarget.style.color = 'var(--text-muted)';
                e.currentTarget.style.background = 'transparent';
              }
            }}
          >
            {t.label}
            {count > 0 && (
              <span
                style={{
                  minWidth: 18,
                  height: 18,
                  borderRadius: 9,
                  background: isActive ? '#ffb4ab' : 'rgba(255,180,171,0.7)',
                  color: isActive ? '#93000a' : '#fff',
                  fontSize: 10,
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '0 4px',
                }}
              >
                {count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
