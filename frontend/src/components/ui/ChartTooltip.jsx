/* ChartTooltip — shared recharts tooltip (Amethyst Precision) */

export default function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div
      style={{
        background: 'var(--glass-bg)',
        backdropFilter: 'blur(20px)',
        border: '1px solid var(--border-strong)',
        borderRadius: 10,
        padding: '10px 14px',
        fontSize: 12,
        boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
      }}
    >
      <p style={{ fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 6 }}>{label}</p>
      {payload.map(e => (
        <div key={e.dataKey} style={{ display: 'flex', justifyContent: 'space-between', gap: 16, alignItems: 'center' }}>
          <span style={{ color: e.color, fontWeight: 500 }}>{e.name}</span>
          <span style={{ fontWeight: 700, color: '#e5e1e4' }}>{e.value}</span>
        </div>
      ))}
    </div>
  );
}
