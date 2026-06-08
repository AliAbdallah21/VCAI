import { useState, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

/* ── Breakpoint hook ── */
function useIsMobile(breakpoint = 768) {
  const [isMobile, setIsMobile] = useState(() => window.innerWidth < breakpoint);
  useEffect(() => {
    const handler = () => setIsMobile(window.innerWidth < breakpoint);
    window.addEventListener('resize', handler);
    return () => window.removeEventListener('resize', handler);
  }, [breakpoint]);
  return isMobile;
}

/* ── Icons ─────────────────────────────────────────── */

const IconDashboard = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <rect x="3" y="3" width="7" height="7" rx="1.5" />
    <rect x="14" y="3" width="7" height="7" rx="1.5" />
    <rect x="3" y="14" width="7" height="7" rx="1.5" />
    <rect x="14" y="14" width="7" height="7" rx="1.5" />
  </svg>
);

const IconMic = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
  </svg>
);

const IconPhone = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M2.25 6.75c0 8.284 6.716 15 15 15h2.25a2.25 2.25 0 002.25-2.25v-1.372c0-.516-.351-.966-.852-1.091l-4.423-1.106c-.44-.11-.902.055-1.173.417l-.97 1.293c-.282.376-.769.542-1.21.38a12.035 12.035 0 01-7.143-7.143c-.162-.441.004-.928.38-1.21l1.293-.97c.363-.271.527-.734.417-1.173L6.963 3.102a1.125 1.125 0 00-1.091-.852H4.5A2.25 2.25 0 002.25 4.5v2.25z" />
  </svg>
);

const IconChart = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
  </svg>
);

const IconCompare = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M3 6h6m0 0v12m0-12L6 9m12 9h-6m0 0V6m0 12l3-3" />
  </svg>
);

const IconProgress = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" />
    <polyline points="16 7 22 7 22 13" />
  </svg>
);

const IconSettings = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);

const IconLogout = () => (
  <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M15.75 9V5.25A2.25 2.25 0 0013.5 3h-6a2.25 2.25 0 00-2.25 2.25v13.5A2.25 2.25 0 007.5 21h6a2.25 2.25 0 002.25-2.25V15m3 0l3-3m0 0l-3-3m3 3H9" />
  </svg>
);

/* ── Nav config ─────────────────────────────────────── */

const NAV = [
  { path: '/dashboard', label: 'Dashboard',       shortLabel: 'Home',    Icon: IconDashboard },
  { path: '/setup',     label: 'New Session',     shortLabel: 'New',     Icon: IconMic       },
  { path: '/sessions',  label: 'Resume a Call',   shortLabel: 'Resume',  Icon: IconPhone     },
  { path: '/evaluate',  label: 'Evaluate a Call', shortLabel: 'Eval',    Icon: IconChart     },
  { path: '/compare',   label: 'Compare',         shortLabel: 'Compare', Icon: IconCompare   },
  { path: '/progress',  label: 'Progress',        shortLabel: 'Track',   Icon: IconProgress  },
  { path: '/settings',  label: 'Settings',        shortLabel: 'Settings',Icon: IconSettings  },
];

/* ── Logo mark (Amethyst) ──────────────────────────── */

function VcaiLogo() {
  return (
    <div
      style={{
        width: 34,
        height: 34,
        borderRadius: 10,
        background: 'linear-gradient(135deg, #b472f1 0%, #deb7ff 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        boxShadow: '0 2px 12px rgba(180,114,241,0.4)',
      }}
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#4a007f" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
      </svg>
    </div>
  );
}

/* ── NavItem (Amethyst active state) ────────────────── */

function NavItem({ path, label, Icon, active }) {
  return (
    <Link
      to={path}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '8px 10px',
        borderRadius: 9,
        fontSize: 13,
        fontWeight: active ? 600 : 500,
        textDecoration: 'none',
        color: active ? '#e5e1e4' : 'rgba(207,194,212,0.55)',
        background: active ? 'rgba(222,183,255,0.1)' : 'transparent',
        borderLeft: active ? '2px solid #deb7ff' : '2px solid transparent',
        transition: 'all 0.13s ease',
      }}
      onMouseEnter={e => {
        if (!active) {
          e.currentTarget.style.background = 'rgba(222,183,255,0.06)';
          e.currentTarget.style.color = 'rgba(207,194,212,0.85)';
        }
      }}
      onMouseLeave={e => {
        if (!active) {
          e.currentTarget.style.background = 'transparent';
          e.currentTarget.style.color = 'rgba(207,194,212,0.55)';
        }
      }}
    >
      <span style={{ flexShrink: 0, color: active ? '#deb7ff' : 'inherit' }}>
        <Icon />
      </span>
      <span>{label}</span>
    </Link>
  );
}

/* ── Main Layout ────────────────────────────────────── */

export default function Layout({ children }) {
  const { user, logout } = useAuth();
  const location = useLocation();
  const navigate  = useNavigate();
  const isMobile  = useIsMobile();

  const handleLogout = () => { logout(); navigate('/login'); };
  const initials = user?.full_name?.split(' ').map(n => n[0]).join('').slice(0, 2).toUpperCase() || 'U';

  return (
    <div style={{ minHeight: '100vh', display: 'flex', background: 'var(--bg-app)' }}>

      {/* ── Mobile top bar ─────────────────────────── */}
      {isMobile && (
        <header
          style={{
            position: 'fixed',
            top: 0, left: 0, right: 0,
            zIndex: 30,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 16px',
            height: 56,
            background: 'rgba(19,19,21,0.97)',
            borderBottom: '1px solid var(--border)',
            backdropFilter: 'blur(20px)',
          }}
        >
          <Link to="/dashboard" style={{ display: 'flex', alignItems: 'center', gap: 10, textDecoration: 'none' }}>
            <VcaiLogo />
            <span style={{ fontWeight: 700, fontSize: 14, color: '#e5e1e4', letterSpacing: '0.04em' }}>VCAI</span>
          </Link>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div
              style={{
                width: 30, height: 30, borderRadius: '50%',
                background: 'linear-gradient(135deg, #b472f1, #deb7ff)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 11, fontWeight: 700, color: '#4a007f',
              }}
            >
              {initials}
            </div>
            <button
              onClick={handleLogout}
              style={{
                padding: '6px 10px',
                borderRadius: 8,
                fontSize: 12,
                fontWeight: 500,
                color: 'var(--text-muted)',
                background: 'var(--bg-card)',
                border: '1px solid var(--border)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                transition: 'color 0.15s',
              }}
              onMouseEnter={e => e.currentTarget.style.color = '#ffb4ab'}
              onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
            >
              <IconLogout />
            </button>
          </div>
        </header>
      )}

      {/* ── Desktop Sidebar ────────────────────────── */}
      {!isMobile && (
        <aside
          style={{
            width: 248,
            flexShrink: 0,
            display: 'flex',
            flexDirection: 'column',
            position: 'sticky',
            top: 0,
            height: '100vh',
            overflowY: 'auto',
            background: 'var(--bg-sidebar)',
            borderRight: '1px solid var(--border)',
          }}
        >
          {/* Logo area */}
          <div style={{ padding: '24px 20px 20px', borderBottom: '1px solid var(--border)' }}>
            <Link to="/dashboard" style={{ display: 'flex', alignItems: 'center', gap: 12, textDecoration: 'none' }}>
              <VcaiLogo />
              <div>
                <p style={{ fontWeight: 700, fontSize: 14, color: '#e5e1e4', letterSpacing: '0.04em', lineHeight: 1.2 }}>VCAI</p>
                <p style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.2 }}>Sales Training AI</p>
              </div>
            </Link>
          </div>

          {/* Navigation */}
          <nav style={{ flex: 1, padding: '16px 12px 8px' }}>
            <p style={{ fontSize: 10.5, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--text-subtle)', padding: '0 10px', marginBottom: 8 }}>
              Platform
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {NAV.map(({ path, label, Icon }) => (
                <NavItem
                  key={path}
                  path={path}
                  label={label}
                  Icon={Icon}
                  active={location.pathname === path}
                />
              ))}
            </div>
          </nav>

          {/* User / bottom area */}
          <div style={{ padding: '12px', borderTop: '1px solid var(--border)' }}>
            <div
              style={{
                borderRadius: 12,
                padding: '12px',
                background: 'rgba(222,183,255,0.04)',
                border: '1px solid var(--border)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                <div
                  style={{
                    width: 32, height: 32,
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, #b472f1, #deb7ff)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 11, fontWeight: 700, color: '#4a007f',
                    flexShrink: 0,
                  }}
                >
                  {initials}
                </div>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <p style={{ fontSize: 13, fontWeight: 600, color: '#e5e1e4', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {user?.full_name}
                  </p>
                  <p style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {user?.email}
                  </p>
                </div>
              </div>
              <button
                onClick={handleLogout}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 6,
                  padding: '7px 0',
                  borderRadius: 8,
                  fontSize: 12,
                  fontWeight: 500,
                  color: 'var(--text-muted)',
                  background: 'transparent',
                  border: '1px solid var(--border)',
                  cursor: 'pointer',
                  transition: 'color 0.15s, border-color 0.15s, background 0.15s',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.color = '#ffb4ab';
                  e.currentTarget.style.borderColor = 'rgba(255,180,171,0.3)';
                  e.currentTarget.style.background = 'rgba(255,180,171,0.06)';
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.color = 'var(--text-muted)';
                  e.currentTarget.style.borderColor = 'var(--border)';
                  e.currentTarget.style.background = 'transparent';
                }}
              >
                <IconLogout />
                Sign out
              </button>
            </div>
          </div>
        </aside>
      )}

      {/* ── Main content ───────────────────────────── */}
      <main
        style={{
          flex: 1,
          overflow: 'auto',
          paddingTop: isMobile ? 56 : 0,
          paddingBottom: isMobile ? 64 : 0,
        }}
      >
        {children}
      </main>

      {/* ── Mobile bottom tab bar ───────────────────── */}
      {isMobile && (
        <nav
          style={{
            position: 'fixed',
            bottom: 0, left: 0, right: 0,
            zIndex: 30,
            display: 'flex',
            alignItems: 'stretch',
            justifyContent: 'space-around',
            padding: '8px 4px',
            background: 'rgba(19,19,21,0.97)',
            borderTop: '1px solid var(--border)',
            backdropFilter: 'blur(20px)',
          }}
        >
          {NAV.map(({ path, shortLabel, Icon }) => {
            const active = location.pathname === path;
            return (
              <Link
                key={path}
                to={path}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 3,
                  flex: 1,
                  padding: '6px 2px',
                  borderRadius: 10,
                  color: active ? '#deb7ff' : 'rgba(207,194,212,0.45)',
                  textDecoration: 'none',
                  transition: 'color 0.13s',
                }}
              >
                <Icon />
                <span style={{ fontSize: 9.5, fontWeight: active ? 600 : 500, letterSpacing: '0.02em' }}>
                  {shortLabel}
                </span>
              </Link>
            );
          })}
        </nav>
      )}
    </div>
  );
}
