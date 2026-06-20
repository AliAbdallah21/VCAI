/* StatCard — KPI metric card used across all dashboards */

export default function StatCard({ label, value, sub, icon: Icon, accent = '#deb7ff', trend }) {
  return (
    <div
      style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-card)',
        padding: '20px 22px',
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        position: 'relative',
        overflow: 'hidden',
        transition: 'border-color 0.15s',
      }}
    >
      {/* Top row: label + icon */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8 }}>
        <span style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)' }}>
          {label}
        </span>
        {Icon && (
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 9,
              background: `${accent}18`,
              border: `1px solid ${accent}28`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
            }}
          >
            <Icon size={15} color={accent} />
          </div>
        )}
      </div>

      {/* Value + trend */}
      <div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span style={{ fontSize: 28, fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.04em', lineHeight: 1 }}>
            {value}
          </span>
          {trend !== undefined && (
            <span style={{
              fontSize: 11.5,
              fontWeight: 600,
              color: trend >= 0 ? '#a5d6a7' : '#ffb4ab',
              display: 'flex',
              alignItems: 'center',
              gap: 2,
            }}>
              {trend >= 0 ? '↑' : '↓'} {Math.abs(trend)}%
            </span>
          )}
        </div>
        {sub && (
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>{sub}</p>
        )}
      </div>

      {/* Bottom accent gradient line */}
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: 2,
          background: `linear-gradient(90deg, transparent, ${accent}40, transparent)`,
        }}
      />
    </div>
  );
}
