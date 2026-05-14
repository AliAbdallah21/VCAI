import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const IconDashboard = () => (
  <svg width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <rect x="3" y="3" width="7" height="7" rx="1.5" />
    <rect x="14" y="3" width="7" height="7" rx="1.5" />
    <rect x="3" y="14" width="7" height="7" rx="1.5" />
    <rect x="14" y="14" width="7" height="7" rx="1.5" />
  </svg>
);

const IconMic = () => (
  <svg width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
  </svg>
);

const IconPhone = () => (
  <svg width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M2.25 6.75c0 8.284 6.716 15 15 15h2.25a2.25 2.25 0 002.25-2.25v-1.372c0-.516-.351-.966-.852-1.091l-4.423-1.106c-.44-.11-.902.055-1.173.417l-.97 1.293c-.282.376-.769.542-1.21.38a12.035 12.035 0 01-7.143-7.143c-.162-.441.004-.928.38-1.21l1.293-.97c.363-.271.527-.734.417-1.173L6.963 3.102a1.125 1.125 0 00-1.091-.852H4.5A2.25 2.25 0 002.25 4.5v2.25z" />
  </svg>
);

const IconChart = () => (
  <svg width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
  </svg>
);

const IconLogout = () => (
  <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M15.75 9V5.25A2.25 2.25 0 0013.5 3h-6a2.25 2.25 0 00-2.25 2.25v13.5A2.25 2.25 0 007.5 21h6a2.25 2.25 0 002.25-2.25V15m3 0l3-3m0 0l-3-3m3 3H9" />
  </svg>
);

const NAV = [
  { path: '/dashboard', label: 'Dashboard',      Icon: IconDashboard, accent: '#60a5fa', accentBg: 'rgba(37,99,235,0.1)',   accentBorder: 'rgba(37,99,235,0.18)'  },
  { path: '/setup',     label: 'New Session',    Icon: IconMic,       accent: '#a78bfa', accentBg: 'rgba(124,58,237,0.1)', accentBorder: 'rgba(124,58,237,0.18)' },
  { path: '/sessions',  label: 'Resume a Call',  Icon: IconPhone,     accent: '#34d399', accentBg: 'rgba(16,185,129,0.1)', accentBorder: 'rgba(16,185,129,0.18)' },
  { path: '/evaluate',  label: 'Evaluate a Call',Icon: IconChart,     accent: '#fbbf24', accentBg: 'rgba(245,158,11,0.1)', accentBorder: 'rgba(245,158,11,0.18)' },
];

export default function Layout({ children }) {
  const { user, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();

  const handleLogout = () => { logout(); navigate('/login'); };
  const initials = user?.full_name?.split(' ').map(n => n[0]).join('').slice(0, 2).toUpperCase() || 'U';

  return (
    <div className="min-h-screen flex" style={{ background: '#030712' }}>
      {/* ── Sidebar ───────────────────────────────────────── */}
      <aside
        className="w-56 flex-shrink-0 flex flex-col sticky top-0 h-screen overflow-y-auto"
        style={{ background: 'rgba(8, 14, 28, 0.98)', borderRight: '1px solid rgba(255,255,255,0.05)' }}
      >
        {/* Logo */}
        <div className="px-5 pt-6 pb-5" style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
          <Link to="/dashboard" className="flex items-center gap-3 group">
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
              style={{ background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)', boxShadow: '0 0 18px rgba(37,99,235,0.35)' }}
            >
              <span className="heading text-white font-bold text-sm">V</span>
            </div>
            <div>
              <p className="heading font-bold text-white text-sm tracking-wider">VCAI</p>
              <p className="text-xs" style={{ color: 'rgba(148,163,184,0.4)' }}>Sales Training AI</p>
            </div>
          </Link>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-5 space-y-0.5">
          <p className="text-xs px-3 mb-3 font-semibold tracking-widest uppercase" style={{ color: 'rgba(148,163,184,0.25)' }}>
            Menu
          </p>
          {NAV.map(({ path, label, Icon, accent, accentBg, accentBorder }) => {
            const active = location.pathname === path;
            return (
              <Link
                key={path}
                to={path}
                className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-150"
                style={active
                  ? { color: accent, background: accentBg, border: `1px solid ${accentBorder}` }
                  : { color: 'rgba(148,163,184,0.55)', border: '1px solid transparent' }
                }
                onMouseEnter={e => { if (!active) e.currentTarget.style.color = '#e2e8f0'; }}
                onMouseLeave={e => { if (!active) e.currentTarget.style.color = 'rgba(148,163,184,0.55)'; }}
              >
                <span className="flex-shrink-0" style={active ? { color: accent } : {}}>
                  <Icon />
                </span>
                <span>{label}</span>
                {active && (
                  <span
                    className="ml-auto w-1 h-4 rounded-full flex-shrink-0"
                    style={{ background: `linear-gradient(to bottom, ${accent}, ${accent}88)` }}
                  />
                )}
              </Link>
            );
          })}
        </nav>

        {/* User */}
        <div className="px-3 pb-5" style={{ borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '12px' }}>
          <div className="rounded-xl p-3" style={{ background: 'rgba(255,255,255,0.02)' }}>
            <div className="flex items-center gap-3 mb-3">
              <div
                className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white flex-shrink-0"
                style={{ background: 'linear-gradient(135deg, #3b82f6, #7c3aed)' }}
              >
                {initials}
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-slate-200 truncate">{user?.full_name}</p>
                <p className="text-xs truncate" style={{ color: 'rgba(148,163,184,0.4)' }}>{user?.email}</p>
              </div>
            </div>
            <button
              onClick={handleLogout}
              className="w-full flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-medium transition-all duration-200 text-slate-500 hover:text-red-400"
              style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}
            >
              <IconLogout />
              Sign out
            </button>
          </div>
        </div>
      </aside>

      {/* ── Main Content ──────────────────────────────────── */}
      <main className="flex-1 overflow-auto">{children}</main>
    </div>
  );
}
