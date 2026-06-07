/* ChartCard — container for recharts charts with title, subtitle, optional right slot */

export default function ChartCard({ title, subtitle, right, children, minHeight = 220 }) {
  return (
    <div
      style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-card)',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 12,
          padding: '16px 20px',
          borderBottom: '1px solid var(--border)',
        }}
      >
        <div>
          <p style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text-primary)' }}>{title}</p>
          {subtitle && (
            <p style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 2 }}>{subtitle}</p>
          )}
        </div>
        {right && <div style={{ flexShrink: 0 }}>{right}</div>}
      </div>

      {/* Body */}
      <div style={{ padding: '16px 20px', minHeight }}>
        {children}
      </div>
    </div>
  );
}
