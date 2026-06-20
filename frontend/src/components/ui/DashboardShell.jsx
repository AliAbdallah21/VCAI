/* DashboardShell — top-bar shell for manager and admin dashboards (no sidebar) */
import { useNavigate } from 'react-router-dom';

function VcaiLogo({ accent = '#b472f1', accent2 = '#deb7ff' }) {
  return (
    <div
      style={{
        width: 34,
        height: 34,
        borderRadius: 10,
        background: `linear-gradient(135deg, ${accent} 0%, ${accent2} 100%)`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        boxShadow: `0 2px 12px ${accent}50`,
      }}
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#4a007f" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
      </svg>
    </div>
  );
}

export default function DashboardShell({
  children,
  user,
  logout,
  title = 'VCAI',
  subtitle,
  accent = '#b472f1',
  accent2 = '#deb7ff',
  right,
}) {
  const navigate  = useNavigate();
  const initials  = user?.full_name?.split(' ').map(n => n[0]).join('').slice(0, 2).toUpperCase() || 'U';

  const handleLogout = () => { logout(); navigate('/login'); };

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-app)' }}>
      {/* Header */}
      <header
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 30,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 32px',
          height: 64,
          background: 'var(--bg-card)',
          borderBottom: '1px solid var(--border)',
          backdropFilter: 'blur(20px)',
          gap: 16,
        }}
      >
        {/* Left: logo + title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <VcaiLogo accent={accent} accent2={accent2} />
          <div>
            <p style={{ fontWeight: 700, fontSize: 14, color: 'var(--text-primary)', letterSpacing: '0.03em', lineHeight: 1.2 }}>
              {title}
            </p>
            {subtitle && (
              <p style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.2 }}>{subtitle}</p>
            )}
          </div>
        </div>

        {/* Right slot + avatar + sign out */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {right}

          {/* Avatar */}
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: '50%',
              background: `linear-gradient(135deg, ${accent}, ${accent2})`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 11,
              fontWeight: 700,
              color: '#4a007f',
              flexShrink: 0,
              cursor: 'default',
            }}
            title={user?.email}
          >
            {initials}
          </div>

          {/* Sign out */}
          <button
            onClick={handleLogout}
            className="btn-secondary"
            style={{ padding: '6px 14px', fontSize: 12 }}
          >
            Sign out
          </button>
        </div>
      </header>

      {/* Page content */}
      <main style={{ padding: '32px', maxWidth: 1200, margin: '0 auto' }}>
        {children}
      </main>
    </div>
  );
}
