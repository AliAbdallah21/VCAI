import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { sessionsAPI, evaluationAPI } from '../services/api';
import Layout from '../components/Layout';

const StatCard = ({ label, value, sub, color, Icon }) => (
  <div
    className="rounded-2xl p-5 relative overflow-hidden"
    style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}
  >
    <div className="flex items-start justify-between">
      <div>
        <p className="text-xs font-medium tracking-wide uppercase mb-3" style={{ color: 'rgba(148,163,184,0.5)' }}>
          {label}
        </p>
        <p className="heading text-3xl font-bold" style={{ color }}>{value}</p>
        {sub && <p className="text-xs mt-1" style={{ color: 'rgba(148,163,184,0.4)' }}>{sub}</p>}
      </div>
      <div
        className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
        style={{ background: `${color}18` }}
      >
        <Icon color={color} />
      </div>
    </div>
    <div
      className="absolute bottom-0 left-0 right-0 h-px"
      style={{ background: `linear-gradient(90deg, transparent, ${color}40, transparent)` }}
    />
  </div>
);

const IconChart = ({ color }) => (
  <svg width="20" height="20" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
  </svg>
);
const IconTarget = ({ color }) => (
  <svg width="20" height="20" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>
  </svg>
);
const IconClock = ({ color }) => (
  <svg width="20" height="20" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
  </svg>
);

const difficultyStyle = {
  easy:   { bg: 'rgba(16,185,129,0.1)', text: '#34d399', border: 'rgba(16,185,129,0.2)' },
  medium: { bg: 'rgba(245,158,11,0.1)', text: '#fbbf24', border: 'rgba(245,158,11,0.2)' },
  hard:   { bg: 'rgba(239,68,68,0.1)',  text: '#f87171', border: 'rgba(239,68,68,0.2)'  },
};

const SESSION_TABS = [
  { key: 'all',       label: 'All' },
  { key: 'active',    label: 'Active' },
  { key: 'evaluated', label: 'Evaluated' },
];

export default function Dashboard() {
  const { user }  = useAuth();
  const navigate  = useNavigate();
  const [sessions, setSessions]   = useState([]);
  const [loading, setLoading]     = useState(true);
  const [sessionTab, setSessionTab] = useState('all');
  const [triggering, setTriggering] = useState({});
  const [stats, setStats]         = useState({ total: 0, avgScore: 0, totalMinutes: 0 });

  useEffect(() => {
    sessionsAPI.getAll(5)
      .then(data => {
        setSessions(data.sessions);
        const completed = data.sessions.filter(s => s.overall_score);
        setStats({
          total: data.total,
          avgScore: completed.length
            ? Math.round(completed.reduce((a, b) => a + b.overall_score, 0) / completed.length)
            : 0,
          totalMinutes: Math.round(data.sessions.reduce((a, b) => a + (b.duration_seconds || 0), 0) / 60),
        });
      })
      .finally(() => setLoading(false));
  }, []);

  const formatDate = dateStr =>
    new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });

  const scoreColor = s => s >= 80 ? '#34d399' : s >= 60 ? '#fbbf24' : '#f87171';

  return (
    <Layout>
      <div className="p-4 md:p-8 max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8 slide-up">
          <h1 className="heading text-2xl font-bold text-white mb-1">
            Welcome back, {user?.full_name?.split(' ')[0]}
          </h1>
          <p className="text-sm" style={{ color: 'rgba(148,163,184,0.55)' }}>
            Here is your training overview
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <StatCard label="Total Sessions"  value={stats.total}               color="#60a5fa" Icon={IconChart}  />
          <StatCard label="Average Score"   value={stats.avgScore || '—'}     color="#34d399" Icon={IconTarget} sub={stats.avgScore ? 'out of 100' : null} />
          <StatCard label="Training Time"   value={`${stats.totalMinutes}m`}  color="#a78bfa" Icon={IconClock}  />
        </div>

        {/* CTA */}
        <Link
          to="/setup"
          className="flex items-center justify-between gap-4 rounded-2xl p-5 md:p-7 mb-8 group transition-all duration-300"
          style={{
            background: 'linear-gradient(135deg, rgba(37,99,235,0.2) 0%, rgba(124,58,237,0.2) 100%)',
            border: '1px solid rgba(37,99,235,0.25)',
          }}
        >
          <div className="min-w-0">
            <h3 className="heading text-base md:text-lg font-bold text-white mb-1 group-hover:text-blue-200 transition-colors">
              Start New Training Session
            </h3>
            <p className="text-xs md:text-sm" style={{ color: 'rgba(148,163,184,0.6)' }}>
              Practice with AI-powered virtual customers
            </p>
          </div>
          <div
            className="w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0 transition-transform duration-200 group-hover:translate-x-1"
            style={{ background: 'rgba(37,99,235,0.3)', border: '1px solid rgba(37,99,235,0.3)' }}
          >
            <svg width="18" height="18" fill="none" stroke="#93c5fd" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
              <path d="M5 12h14M12 5l7 7-7 7"/>
            </svg>
          </div>
        </Link>

        {/* Recent Sessions */}
        <div className="rounded-2xl overflow-hidden" style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}>
          <div className="px-6 py-4 flex items-center justify-between gap-4 flex-wrap" style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            <h2 className="heading text-sm font-bold text-white tracking-wide">Recent Sessions</h2>
            <div className="flex items-center gap-3">
              {/* Tabs */}
              <div className="flex gap-0.5 p-0.5 rounded-lg" style={{ background: 'rgba(255,255,255,0.04)' }}>
                {SESSION_TABS.map(t => (
                  <button
                    key={t.key}
                    onClick={() => setSessionTab(t.key)}
                    className="px-3 py-1 rounded-md text-xs font-medium transition-all duration-150"
                    style={sessionTab === t.key
                      ? { background: 'rgba(37,99,235,0.2)', color: '#93c5fd' }
                      : { color: 'rgba(148,163,184,0.5)' }
                    }
                  >
                    {t.label}
                  </button>
                ))}
              </div>
              <Link to="/sessions" className="text-xs font-medium transition-all" style={{ color: 'rgba(96,165,250,0.7)' }}
                onMouseEnter={e => e.currentTarget.style.color = '#93c5fd'}
                onMouseLeave={e => e.currentTarget.style.color = 'rgba(96,165,250,0.7)'}
              >
                View all →
              </Link>
            </div>
          </div>

          {loading ? (
            <div className="p-10 text-center">
              <div className="w-6 h-6 spin-ring mx-auto mb-3" style={{ border: '2px solid rgba(255,255,255,0.08)', borderTopColor: '#3b82f6', borderRadius: '50%' }} />
              <p className="text-sm" style={{ color: 'rgba(148,163,184,0.4)' }}>Loading sessions…</p>
            </div>
          ) : sessions.length === 0 ? (
            <div className="p-14 text-center">
              <div
                className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-4"
                style={{ background: 'rgba(37,99,235,0.1)', border: '1px solid rgba(37,99,235,0.15)' }}
              >
                <svg width="24" height="24" fill="none" stroke="#60a5fa" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                  <path d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
                </svg>
              </div>
              <h3 className="font-semibold text-white mb-1">No sessions yet</h3>
              <p className="text-sm" style={{ color: 'rgba(148,163,184,0.45)' }}>Start your first training session above</p>
            </div>
          ) : (() => {
            const filtered = sessions.filter(s => {
              if (sessionTab === 'active')    return s.status === 'active';
              if (sessionTab === 'evaluated') return !!s.overall_score;
              return true;
            });
            if (filtered.length === 0) return (
              <div className="p-10 text-center">
                <p className="text-sm" style={{ color: 'rgba(148,163,184,0.4)' }}>
                  {sessionTab === 'active' ? 'No active calls' : sessionTab === 'evaluated' ? 'No evaluated sessions yet' : 'No sessions yet'}
                </p>
              </div>
            );
            return (
              <div>
                {filtered.map((s, i) => {
                  const diff      = difficultyStyle[s.difficulty] || difficultyStyle.medium;
                  const isActive  = s.status === 'active';
                  const evaluated = !!s.overall_score;
                  const isBusy    = triggering[s.id];
                  return (
                    <div
                      key={s.id}
                      className="flex items-center justify-between px-6 py-4 transition-colors duration-150"
                      style={{ borderBottom: i < filtered.length - 1 ? '1px solid rgba(255,255,255,0.04)' : 'none' }}
                      onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.025)'}
                      onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                    >
                      <div className="flex items-center gap-4">
                        <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
                          style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.07)' }}>
                          <svg width="15" height="15" fill="none" stroke="rgba(148,163,184,0.6)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                            <path d="M17.982 18.725A7.488 7.488 0 0012 15.75a7.488 7.488 0 00-5.982 2.975m11.963 0a9 9 0 10-11.963 0m11.963 0A8.966 8.966 0 0112 21a8.966 8.966 0 01-5.982-2.275M15 9.75a3 3 0 11-6 0 3 3 0 016 0z" />
                          </svg>
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <p className="text-sm font-medium text-slate-200">{s.persona_name || s.persona_id}</p>
                            {isActive && (
                              <span className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold"
                                style={{ background: 'rgba(16,185,129,0.12)', color: '#34d399', border: '1px solid rgba(16,185,129,0.2)' }}>
                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse inline-block" />
                                Live
                              </span>
                            )}
                          </div>
                          <p className="text-xs mt-0.5" style={{ color: 'rgba(148,163,184,0.45)' }}>{formatDate(s.started_at)}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {!isActive && (
                          <span className="px-2.5 py-1 rounded-lg text-xs font-semibold capitalize hidden sm:block"
                            style={{ background: diff.bg, color: diff.text, border: `1px solid ${diff.border}` }}>
                            {s.difficulty}
                          </span>
                        )}
                        {isActive ? (
                          <button onClick={() => navigate(`/session/${s.id}`)}
                            className="px-3.5 py-1.5 rounded-xl text-xs font-semibold"
                            style={{ background: 'rgba(16,185,129,0.12)', color: '#34d399', border: '1px solid rgba(16,185,129,0.2)' }}>
                            Resume →
                          </button>
                        ) : evaluated ? (
                          <>
                            <span className="heading text-base font-bold" style={{ color: scoreColor(s.overall_score) }}>{s.overall_score}</span>
                            <button onClick={() => navigate(`/evaluation/${s.id}`)}
                              className="px-3 py-1.5 rounded-xl text-xs font-semibold"
                              style={{ background: 'rgba(37,99,235,0.12)', color: '#93c5fd', border: '1px solid rgba(37,99,235,0.2)' }}>
                              Report
                            </button>
                            <button onClick={async () => {
                              setTriggering(p => ({...p, [s.id]: true}));
                              try { await evaluationAPI.triggerEvaluation(s.id,'training',true); navigate(`/evaluation/${s.id}`); }
                              catch { setTriggering(p => ({...p, [s.id]: false})); }
                            }} disabled={isBusy}
                              className="px-3 py-1.5 rounded-xl text-xs font-semibold disabled:opacity-40"
                              style={{ background: 'rgba(245,158,11,0.08)', color: '#fbbf24', border: '1px solid rgba(245,158,11,0.15)' }}>
                              {isBusy ? '…' : '↻'}
                            </button>
                          </>
                        ) : (
                          <button onClick={async () => {
                            setTriggering(p => ({...p, [s.id]: true}));
                            try { await evaluationAPI.triggerEvaluation(s.id,'training',false); navigate(`/evaluation/${s.id}`); }
                            catch { setTriggering(p => ({...p, [s.id]: false})); }
                          }} disabled={isBusy}
                            className="px-3.5 py-1.5 rounded-xl text-xs font-semibold disabled:opacity-40"
                            style={{ background: 'rgba(245,158,11,0.12)', color: '#fbbf24', border: '1px solid rgba(245,158,11,0.2)' }}>
                            {isBusy ? 'Starting…' : 'Evaluate →'}
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            );
          })()}
        </div>
      </div>
    </Layout>
  );
}
