import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { sessionsAPI, evaluationAPI, learningAPI } from '../services/api';
import Layout from '../components/Layout';
import Badge from '../components/ui/Badge';
import EmptyState from '../components/ui/EmptyState';

/* ── Inline icons ── */
const IcoActivity = ({ size = 15, color }) => (
  <svg width={size} height={size} fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
  </svg>
);
const IcoTarget = ({ size = 15, color }) => (
  <svg width={size} height={size} fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="6" /><circle cx="12" cy="12" r="2" />
  </svg>
);
const IcoClock = ({ size = 15, color }) => (
  <svg width={size} height={size} fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" />
  </svg>
);
const IcoMic = ({ size = 22, color = 'rgba(222,183,255,0.3)' }) => (
  <svg width={size} height={size} fill="none" stroke={color} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
  </svg>
);
const IcoUser = ({ size = 16, color = 'var(--text-muted)' }) => (
  <svg width={size} height={size} fill="none" stroke={color} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M17.982 18.725A7.488 7.488 0 0012 15.75a7.488 7.488 0 00-5.982 2.975m11.963 0a9 9 0 10-11.963 0m11.963 0A8.966 8.966 0 0112 21a8.966 8.966 0 01-5.982-2.275M15 9.75a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);
const IcoArrow = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M5 12h14M12 5l7 7-7 7" />
  </svg>
);

/* ── KPI stat card ── */
function StatCard({ label, value, sub, Icon, accent }) {
  return (
    <div
      style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-card)',
        padding: '20px 22px',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 12 }}>
        <span style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)' }}>
          {label}
        </span>
        <div style={{ width: 30, height: 30, borderRadius: 8, background: `${accent}18`, border: `1px solid ${accent}28`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Icon size={14} color={accent} />
        </div>
      </div>
      <span style={{ fontSize: 30, fontWeight: 800, color: '#e5e1e4', letterSpacing: '-0.04em', lineHeight: 1 }}>
        {value}
      </span>
      {sub && <p style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 5 }}>{sub}</p>}
      <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, transparent, ${accent}40, transparent)` }} />
    </div>
  );
}

/* ── Skill color/label/trend maps ── */
const SKILL_COLORS = {
  communication:     '#deb7ff',
  product_knowledge: '#e9c46a',
  objection_handling:'#ffb4ab',
  rapport:           '#a5d6a7',
  closing:           '#b472f1',
};
const SKILL_LABELS = {
  communication:     'Communication',
  product_knowledge: 'Product Knowledge',
  objection_handling:'Objection Handling',
  rapport:           'Rapport',
  closing:           'Closing',
};
const TREND_META = {
  improving:         { symbol: '↑', color: '#a5d6a7' },
  declining:         { symbol: '↓', color: '#ffb4ab' },
  plateau:           { symbol: '→', color: '#e9c46a' },
  insufficient_data: { symbol: '–', color: 'var(--text-muted)' },
};

const scoreColor = s => s >= 80 ? '#a5d6a7' : s >= 60 ? '#e9c46a' : '#ffb4ab';
const formatDate = d => new Date(d).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });

const SESSION_TABS = [
  { key: 'all',        label: 'All' },
  { key: 'active',     label: 'Active' },
  { key: 'incomplete', label: 'Needs Evaluation' },
  { key: 'evaluated',  label: 'Evaluated' },
];

export default function Dashboard() {
  const { user }   = useAuth();
  const navigate   = useNavigate();
  const [sessions, setSessions]     = useState([]);
  const [loading, setLoading]       = useState(true);
  const [sessionTab, setSessionTab] = useState('all');
  const [triggering, setTriggering] = useState({});
  const [resuming, setResuming]     = useState(null);

  // Re-open an ended session: reactivate it server-side, then enter the call.
  const handleResume = async (s) => {
    setResuming(s.id);
    try {
      if (s.status !== 'active') await sessionsAPI.reactivate(s.id);
      navigate(`/session/${s.id}`);
    } catch {
      setResuming(null);
    }
  };
  const [stats, setStats]           = useState({ total: 0, avgScore: 0, totalMinutes: 0 });
  const [profile, setProfile]       = useState(null);

  useEffect(() => {
    sessionsAPI.getAll(5)
      .then(data => {
        setSessions(data.sessions);
        const done = data.sessions.filter(s => s.overall_score);
        setStats({
          total: data.total,
          avgScore: done.length ? Math.round(done.reduce((a, b) => a + b.overall_score, 0) / done.length) : 0,
          totalMinutes: Math.round(data.sessions.reduce((a, b) => a + (b.duration_seconds || 0), 0) / 60),
        });
      })
      .finally(() => setLoading(false));

    learningAPI.getProfile().then(setProfile).catch(() => {});
  }, []);

  const filtered = sessions.filter(s => {
    if (sessionTab === 'active')     return s.status === 'active';
    if (sessionTab === 'incomplete') return s.status !== 'active' && !s.overall_score;
    if (sessionTab === 'evaluated')  return !!s.overall_score;
    return true;
  });

  const firstName = user?.full_name?.split(' ')[0] ?? 'there';

  return (
    <Layout>
      <div style={{ padding: '32px 36px', maxWidth: 900, margin: '0 auto' }}>

        {/* ── Page header ── */}
        <div className="slide-up" style={{ marginBottom: 28 }}>
          <h1 style={{ fontSize: 26, fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.02em', margin: 0 }}>
            Welcome back, {firstName}
          </h1>
          <p style={{ fontSize: 13.5, color: 'var(--text-muted)', marginTop: 6 }}>
            Here is your training overview
          </p>
        </div>

        {/* ── KPI Stats ── */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 24 }}>
          <StatCard label="Total Sessions"  value={stats.total}              Icon={IcoActivity} accent="#deb7ff" />
          <StatCard label="Average Score"   value={stats.avgScore || '—'}    Icon={IcoTarget}   accent="#a5d6a7" sub={stats.avgScore ? 'out of 100' : 'No sessions yet'} />
          <StatCard label="Training Time"   value={`${stats.totalMinutes}m`} Icon={IcoClock}    accent="#b472f1" />
        </div>

        {/* ── Start New Session CTA ── */}
        <Link
          to="/setup"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 16,
            padding: '20px 24px',
            marginBottom: 24,
            borderRadius: 'var(--radius-card)',
            background: 'rgba(222,183,255,0.06)',
            border: '1px solid rgba(222,183,255,0.18)',
            textDecoration: 'none',
            transition: 'border-color 0.15s, background 0.15s',
          }}
          onMouseEnter={e => {
            e.currentTarget.style.borderColor = 'rgba(222,183,255,0.35)';
            e.currentTarget.style.background  = 'rgba(222,183,255,0.1)';
          }}
          onMouseLeave={e => {
            e.currentTarget.style.borderColor = 'rgba(222,183,255,0.18)';
            e.currentTarget.style.background  = 'rgba(222,183,255,0.06)';
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <div
              style={{
                width: 44,
                height: 44,
                borderRadius: 12,
                background: 'rgba(222,183,255,0.12)',
                border: '1px solid rgba(222,183,255,0.22)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
              }}
            >
              <svg width="20" height="20" fill="none" stroke="#deb7ff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                <path d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
              </svg>
            </div>
            <div>
              <p style={{ fontSize: 15, fontWeight: 600, color: '#e5e1e4', margin: 0 }}>
                Start New Training Session
              </p>
              <p style={{ fontSize: 12.5, color: 'var(--text-muted)', marginTop: 3 }}>
                Practice with AI-powered virtual customers
              </p>
            </div>
          </div>
          <div
            style={{
              width: 34,
              height: 34,
              borderRadius: 9,
              background: 'rgba(222,183,255,0.12)',
              border: '1px solid rgba(222,183,255,0.22)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              color: '#deb7ff',
            }}
          >
            <IcoArrow />
          </div>
        </Link>

        {/* ── Skill Progress widget ── */}
        {profile?.has_enough_data && profile.skill_averages?.length > 0 && (() => {
          const weakest = [...profile.skill_averages].sort((a, b) => a.avg_score - b.avg_score).slice(0, 3);
          return (
            <div
              style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-card)',
                padding: '20px 22px',
                marginBottom: 24,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
                <p style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text-primary)' }}>Skill Progress</p>
                <Link
                  to="/progress"
                  style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-muted)', textDecoration: 'none', transition: 'color 0.13s' }}
                  onMouseEnter={e => e.currentTarget.style.color = '#deb7ff'}
                  onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
                >
                  View all →
                </Link>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {weakest.map(skill => {
                  const color = SKILL_COLORS[skill.skill_key] ?? '#deb7ff';
                  const label = SKILL_LABELS[skill.skill_key] ?? skill.skill_key;
                  const trend = TREND_META[skill.trend] ?? TREND_META.insufficient_data;
                  const score = Math.round(skill.avg_score);
                  return (
                    <div key={skill.skill_key} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      <span style={{ fontSize: 12.5, fontWeight: 500, width: 148, flexShrink: 0, color: 'var(--text-secondary)' }}>
                        {label}
                      </span>
                      <div style={{ flex: 1, height: 5, borderRadius: 99, background: 'rgba(222,183,255,0.08)', overflow: 'hidden' }}>
                        <div style={{ height: '100%', borderRadius: 99, width: `${score}%`, background: color, transition: 'width 0.5s ease' }} />
                      </div>
                      <span style={{ fontSize: 12.5, fontWeight: 700, width: 28, textAlign: 'right', flexShrink: 0, color }}>
                        {score}
                      </span>
                      <span style={{ fontSize: 12, fontWeight: 700, width: 14, textAlign: 'right', flexShrink: 0, color: trend.color }}>
                        {trend.symbol}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })()}

        {/* ── Recent Sessions ── */}
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
              padding: '16px 20px',
              borderBottom: '1px solid var(--border)',
              gap: 12,
              flexWrap: 'wrap',
            }}
          >
            <p style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text-primary)' }}>Recent Sessions</p>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              {/* Pill tabs */}
              <div style={{ display: 'flex', gap: 2, padding: 3, borderRadius: 9, background: 'rgba(222,183,255,0.04)', border: '1px solid var(--border)' }}>
                {SESSION_TABS.map(t => {
                  const isActive = t.key === sessionTab;
                  return (
                    <button
                      key={t.key}
                      onClick={() => setSessionTab(t.key)}
                      style={{
                        padding: '5px 12px',
                        borderRadius: 7,
                        fontSize: 12,
                        fontWeight: isActive ? 600 : 500,
                        color: isActive ? '#e5e1e4' : 'var(--text-muted)',
                        background: isActive ? 'rgba(222,183,255,0.12)' : 'transparent',
                        border: `1px solid ${isActive ? 'rgba(222,183,255,0.3)' : 'transparent'}`,
                        cursor: 'pointer',
                        transition: 'all 0.13s',
                      }}
                    >
                      {t.label}
                    </button>
                  );
                })}
              </div>
              <Link
                to="/sessions"
                style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-muted)', textDecoration: 'none', transition: 'color 0.13s' }}
                onMouseEnter={e => e.currentTarget.style.color = '#deb7ff'}
                onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
              >
                View all →
              </Link>
            </div>
          </div>

          {/* Body */}
          {loading ? (
            <div style={{ padding: '48px 24px', textAlign: 'center' }}>
              <div className="spin-ring" style={{ width: 22, height: 22, border: '2px solid rgba(222,183,255,0.08)', borderTopColor: '#deb7ff', borderRadius: '50%', margin: '0 auto 10px' }} />
              <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>Loading sessions…</p>
            </div>
          ) : sessions.length === 0 ? (
            <EmptyState
              icon={IcoMic}
              title="No sessions yet"
              description="Start your first training session to see your progress here"
              action={
                <Link to="/setup" className="btn-primary" style={{ fontSize: 12 }}>
                  Start Session
                </Link>
              }
            />
          ) : filtered.length === 0 ? (
            <div style={{ padding: '36px 24px', textAlign: 'center' }}>
              <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                {sessionTab === 'active' ? 'No active calls' : sessionTab === 'evaluated' ? 'No evaluated sessions yet' : 'No sessions'}
              </p>
            </div>
          ) : (
            <div>
              {filtered.map((s, i) => {
                const isActive  = s.status === 'active';
                const evaluated = !!s.overall_score;
                const isBusy    = triggering[s.id];

                return (
                  <div
                    key={s.id}
                    onClick={() => handleResume(s)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={e => { if (e.key === 'Enter') handleResume(s); }}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '13px 20px',
                      borderBottom: i < filtered.length - 1 ? '1px solid var(--border)' : 'none',
                      gap: 12,
                      cursor: 'pointer',
                      transition: 'background 0.12s',
                    }}
                    onMouseEnter={e => e.currentTarget.style.background = 'rgba(222,183,255,0.03)'}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                  >
                    {/* Left: icon + name + date */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, minWidth: 0 }}>
                      <div style={{ width: 36, height: 36, borderRadius: 10, background: 'rgba(222,183,255,0.06)', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                        <IcoUser size={16} color="var(--text-muted)" />
                      </div>
                      <div style={{ minWidth: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 7, flexWrap: 'wrap' }}>
                          <span style={{ fontSize: 13.5, fontWeight: 500, color: 'var(--text-secondary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {s.persona_name || s.persona_id}
                          </span>
                          {isActive && <Badge variant="live" label="Live" dot />}
                        </div>
                        <p style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 2 }}>
                          {formatDate(s.started_at)}
                        </p>
                      </div>
                    </div>

                    {/* Right: difficulty · score · actions — aligned into fixed slots */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
                      {/* Difficulty — fixed slot so badges line up across rows */}
                      <div style={{ width: 64, display: 'flex', justifyContent: 'flex-end' }}>
                        {!isActive && s.difficulty && (
                          <Badge variant={s.difficulty} label={s.difficulty.charAt(0).toUpperCase() + s.difficulty.slice(1)} />
                        )}
                      </div>

                      {/* Score — fixed slot, right-aligned (N/A when unscored) */}
                      <div style={{ width: 36, textAlign: 'right' }}>
                        {evaluated ? (
                          <span style={{ fontSize: 16, fontWeight: 700, color: scoreColor(s.overall_score) }}>
                            {s.overall_score}
                          </span>
                        ) : (
                          <span style={{ fontSize: 12, fontWeight: 600, letterSpacing: '0.02em', color: 'var(--text-subtle)' }}>
                            N/A
                          </span>
                        )}
                      </div>

                      {/* Action buttons — uniform padding, primary always first */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <button
                          onClick={e => { e.stopPropagation(); handleResume(s); }}
                          disabled={resuming === s.id}
                          className="btn-primary"
                          style={{ fontSize: 12, padding: '6px 14px', minWidth: 96, opacity: resuming === s.id ? 0.5 : 1 }}
                        >
                          {resuming === s.id ? '…' : isActive ? 'Resume →' : 'Continue →'}
                        </button>

                        {!isActive && (evaluated ? (
                          <button
                            onClick={e => { e.stopPropagation(); navigate(`/evaluation/${s.id}`); }}
                            className="btn-secondary"
                            style={{ fontSize: 12, padding: '6px 14px', minWidth: 96 }}
                          >
                            Report
                          </button>
                        ) : (
                          <button
                            onClick={async e => {
                              e.stopPropagation();
                              setTriggering(p => ({ ...p, [s.id]: true }));
                              try { await evaluationAPI.triggerEvaluation(s.id, 'training', false); navigate(`/evaluation/${s.id}`); }
                              catch { setTriggering(p => ({ ...p, [s.id]: false })); }
                            }}
                            disabled={isBusy}
                            className="btn-secondary"
                            style={{ fontSize: 12, padding: '6px 14px', minWidth: 96, color: '#e9c46a', borderColor: 'rgba(233,196,106,0.3)' }}
                          >
                            {isBusy ? 'Starting…' : 'Evaluate →'}
                          </button>
                        ))}

                        {/* Re-evaluate — fixed slot so rows end at the same edge */}
                        <div style={{ width: 32 }}>
                          {evaluated && (
                            <button
                              onClick={async e => {
                                e.stopPropagation();
                                setTriggering(p => ({ ...p, [s.id]: true }));
                                try { await evaluationAPI.triggerEvaluation(s.id, 'training', true); navigate(`/evaluation/${s.id}`); }
                                catch { setTriggering(p => ({ ...p, [s.id]: false })); }
                              }}
                              disabled={isBusy}
                              className="btn-secondary"
                              style={{ fontSize: 12, padding: '6px 0', width: 32 }}
                              title="Re-evaluate"
                            >
                              {isBusy ? '…' : '↻'}
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
