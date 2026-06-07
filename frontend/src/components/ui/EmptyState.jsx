/* EmptyState — clean empty/no-data states with icon, title, description, optional CTA */

export default function EmptyState({ icon: Icon, title, description, action }) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '48px 24px',
        textAlign: 'center',
        gap: 12,
      }}
    >
      {Icon && (
        <div
          style={{
            width: 48,
            height: 48,
            borderRadius: 14,
            background: 'var(--primary-soft)',
            border: '1px solid rgba(222,183,255,0.15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: 4,
          }}
        >
          <Icon size={22} color="var(--text-muted)" />
        </div>
      )}
      <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-secondary)' }}>{title}</p>
      {description && (
        <p style={{ fontSize: 12.5, color: 'var(--text-muted)', maxWidth: 280 }}>{description}</p>
      )}
      {action && <div style={{ marginTop: 4 }}>{action}</div>}
    </div>
  );
}
