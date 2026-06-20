import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { managerAPI } from '../services/api';
import AbuseQueue from '../components/AbuseQueue';
import DashboardShell from '../components/ui/DashboardShell';
import Tabs from '../components/ui/Tabs';
import Badge from '../components/ui/Badge';
import EmptyState from '../components/ui/EmptyState';
import ChartTooltip from '../components/ui/ChartTooltip';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

/* ── Inline icons ── */
const IcoUsers = ({ size = 15, color }) => (
  <svg width={size} height={size} fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M16 11c1.5 0 2.5-1 2.5-2.5S17.5 6 16 6M20 21c0-2.5-2-4.5-4-4.5H8C6 16.5 4 18.5 4 21m6-8.5c2 0 3.5-1.5 3.5-3.5S12 5.5 10 5.5 6.5 7 6.5 9s1.5 3.5 3.5 3.5z" />
  </svg>
);
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
const IcoFlag = ({ size = 15, color }) => (
  <svg width={size} height={size} fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" /><line x1="4" y1="22" x2="4" y2="15" />
  </svg>
);
const IcoInbox = ({ size = 22, color = 'rgba(222,183,255,0.25)' }) => (
  <svg width={size} height={size} fill="none" stroke={color} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <polyline points="22 12 16 12 14 15 10 15 8 12 2 12" />
    <path d="M5.45 5.11L2 12v6a2 2 0 002 2h16a2 2 0 002-2v-6l-3.45-6.89A2 2 0 0016.76 4H7.24a2 2 0 00-1.79 1.11z" />
  </svg>
);

const SKILLS = [
  { key: 'communication',     label: 'Communication' },
  { key: 'product_knowledge', label: 'Product Knowledge' },
  { key: 'objection_handling',label: 'Objection Handling' },
  { key: 'rapport',           label: 'Rapport' },
  { key: 'closing',           label: 'Closing' },
];

/* ── KPI stat card (Amethyst) ── */
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
      <span style={{ fontSize: 30, fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.04em', lineHeight: 1 }}>
        {value}
      </span>
      {sub && <p style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 5 }}>{sub}</p>}
      <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, transparent, ${accent}40, transparent)` }} />
    </div>
  );
}

/* ── Section card ── */
function Card({ title, children, right }) {
  return (
    <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-card)', overflow: 'hidden' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px 20px', borderBottom: '1px solid var(--border)', gap: 12 }}>
        <p style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text-primary)' }}>{title}</p>
        {right}
      </div>
      <div style={{ padding: '20px' }}>
        {children}
      </div>
    </div>
  );
}

/* ── Shared chart axis props ── */
const axisTick = { fill: '#988d9d', fontSize: 11 };
const axisLine = { stroke: 'rgba(76,68,82,0.6)' };
const gridProps = { strokeDasharray: '3 3', stroke: 'rgba(76,68,82,0.3)', vertical: false };

const TABS = [
  { key: 'Overview',       label: 'Overview' },
  { key: 'Agents',         label: 'Agents' },
  { key: 'Emotion trends', label: 'Emotion Trends' },
  { key: 'Abuse',          label: 'Abuse' },
];

const toArray = v => (Array.isArray(v) ? v : []);

const scoreColor = s => s >= 80 ? '#a5d6a7' : s >= 60 ? '#e9c46a' : 'var(--text-secondary)';

export default function ManagerDashboard() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [tab, setTab]           = useState('Overview');
  const [analytics, setAnalytics] = useState(null);
  const [agentsData, setAgentsData] = useState(null);
  const [emotion, setEmotion]   = useState(null);
  const [flags, setFlags]       = useState([]);
  const [loading, setLoading]   = useState(true);

  const loadFlags = () => managerAPI.getAbuse().then(d => setFlags(toArray(d))).catch(() => setFlags([]));

  useEffect(() => {
    Promise.all([
      managerAPI.getAnalytics().then(setAnalytics).catch(() => setAnalytics(null)),
      managerAPI.getAgents().then(setAgentsData).catch(() => setAgentsData(null)),
      managerAPI.getEmotionTrends().then(setEmotion).catch(() => setEmotion(null)),
      loadFlags(),
    ]).finally(() => setLoading(false));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const openFlags = toArray(flags).filter(f => f.status === 'open').length;
  const tabBadge = { Abuse: openFlags };

  return (
    <DashboardShell
      user={user}
      logout={logout}
      title="VCAI Manager"
      subtitle="Team dashboard"
      right={
        <Link
          to="/seats"
          className="btn-secondary"
          style={{ fontSize: 12, padding: '6px 14px' }}
        >
          Manage Seats
        </Link>
      }
    >
      {/* Page header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.02em', margin: 0 }}>
          Team Overview
        </h1>
        <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 5 }}>
          Monitor your team's training activity and performance
        </p>
      </div>

      {/* Tabs */}
      <div style={{ marginBottom: 24 }}>
        <Tabs tabs={TABS} active={tab} onChange={setTab} badge={tabBadge} />
      </div>

      {loading ? (
        <div style={{ padding: '64px 0', textAlign: 'center' }}>
          <div className="spin-ring" style={{ width: 24, height: 24, border: '2px solid rgba(222,183,255,0.08)', borderTopColor: '#deb7ff', borderRadius: '50%', margin: '0 auto 12px' }} />
          <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>Loading dashboard…</p>
        </div>
      ) : (
        <>
          {/* ── Overview ── */}
          {tab === 'Overview' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
              {/* KPI grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
                <StatCard label="Active Agents"        value={analytics?.active_agents ?? 0}                              Icon={IcoUsers}    accent="#deb7ff" />
                <StatCard label="Sessions This Period" value={analytics?.sessions_this_period ?? 0} sub={`${analytics?.total_sessions ?? 0} all-time`} Icon={IcoActivity} accent="#a5d6a7" />
                <StatCard label="Team Avg Score"       value={analytics?.avg_score ?? '—'}                                Icon={IcoTarget}   accent="#b472f1" />
                <StatCard label="Open Abuse Flags"     value={openFlags}                                                  Icon={IcoFlag}     accent={openFlags > 0 ? '#ffb4ab' : '#a5d6a7'} />
              </div>

              {/* Weakest skill alert */}
              {analytics?.weakest_skill && (
                <div
                  style={{
                    background: 'rgba(255,180,171,0.05)',
                    border: '1px solid rgba(255,180,171,0.18)',
                    borderRadius: 'var(--radius-card)',
                    padding: '14px 18px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                  }}
                >
                  <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                    Weakest team skill:{' '}
                    <span style={{ fontWeight: 700, color: '#ffb4ab' }}>
                      {SKILLS.find(s => s.key === analytics.weakest_skill.skill)?.label || analytics.weakest_skill.skill}
                    </span>
                    <span style={{ color: 'var(--text-muted)', marginLeft: 6 }}>
                      (avg {analytics.weakest_skill.average})
                    </span>
                  </span>
                </div>
              )}

              {/* Sessions chart */}
              <Card title="Sessions per Day" subtitle="Last 30 days">
                {(analytics?.sessions_per_day?.length ?? 0) === 0 ? (
                  <EmptyState icon={IcoInbox} title="No sessions yet" description="Sessions will appear here once your team starts training" />
                ) : (
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={analytics.sessions_per_day} margin={{ top: 8, right: 8, left: -24, bottom: 4 }}>
                      <CartesianGrid {...gridProps} />
                      <XAxis dataKey="date" tick={axisTick} tickLine={false} axisLine={axisLine} />
                      <YAxis allowDecimals={false} tick={axisTick} tickLine={false} axisLine={false} />
                      <Tooltip content={<ChartTooltip />} />
                      <Line type="monotone" dataKey="count" name="Sessions" stroke="#deb7ff" strokeWidth={2} dot={{ r: 2.5, fill: '#deb7ff' }} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </Card>

              {/* Score distribution */}
              <Card title="Score Distribution" subtitle="Sessions by score range">
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={analytics?.score_distribution ?? []} margin={{ top: 8, right: 8, left: -24, bottom: 4 }}>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="bucket" tick={axisTick} tickLine={false} axisLine={axisLine} />
                    <YAxis allowDecimals={false} tick={axisTick} tickLine={false} axisLine={false} />
                    <Tooltip content={<ChartTooltip />} />
                    <Bar dataKey="count" name="Sessions" fill="#b472f1" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </div>
          )}

          {/* ── Agents ── */}
          {tab === 'Agents' && (
            <Card
              title="Team Members"
              right={
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  {agentsData?.pending_invites ?? 0} pending invite{agentsData?.pending_invites !== 1 ? 's' : ''}
                </span>
              }
            >
              {(agentsData?.agents?.length ?? 0) === 0 ? (
                <EmptyState icon={IcoUsers} title="No agents yet" description="Invite agents from the Seats page to get started" />
              ) : (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border)' }}>
                        {['Agent', 'Sessions', 'Avg Score', 'Streak', 'Last Session', 'Status'].map(h => (
                          <th key={h} style={{ textAlign: 'left', padding: '0 12px 12px 0', fontSize: 11.5, fontWeight: 600, color: 'var(--text-muted)', letterSpacing: '0.04em', whiteSpace: 'nowrap' }}>
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {agentsData.agents.map(a => (
                        <tr
                          key={a.user_id}
                          onClick={() => navigate(`/manager/agents/${a.user_id}`)}
                          style={{ cursor: 'pointer', borderBottom: '1px solid var(--border)', transition: 'background 0.12s' }}
                          onMouseEnter={e => e.currentTarget.style.background = 'rgba(222,183,255,0.03)'}
                          onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                        >
                          <td style={{ padding: '12px 12px 12px 0' }}>
                            <p style={{ fontSize: 13.5, fontWeight: 500, color: 'var(--text-secondary)' }}>{a.full_name}</p>
                            <p style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 2 }}>{a.email}</p>
                          </td>
                          <td style={{ padding: '12px 12px 12px 0', fontSize: 13, color: 'var(--text-secondary)' }}>
                            {a.completed_sessions}/{a.total_sessions}
                          </td>
                          <td style={{ padding: '12px 12px 12px 0', fontSize: 13, fontWeight: 600, color: scoreColor(a.avg_overall_score) }}>
                            {a.avg_overall_score ?? '—'}
                          </td>
                          <td style={{ padding: '12px 12px 12px 0', fontSize: 13, color: 'var(--text-secondary)' }}>
                            {a.current_streak > 0 ? `${a.current_streak} 🔥` : a.current_streak}
                          </td>
                          <td style={{ padding: '12px 12px 12px 0', fontSize: 12.5, color: 'var(--text-muted)' }}>
                            {a.last_session_date ? new Date(a.last_session_date).toLocaleDateString() : '—'}
                          </td>
                          <td style={{ padding: '12px 0' }}>
                            <Badge variant={a.is_active ? 'active' : 'inactive'} label={a.is_active ? 'Active' : 'Inactive'} dot={a.is_active} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          )}

          {/* ── Emotion Trends ── */}
          {tab === 'Emotion trends' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
                <StatCard label="Avg Customer Mood"   value={emotion?.avg_mood_score ?? '—'}                                                                          Icon={IcoActivity} accent="#deb7ff" />
                <StatCard label="High-Risk Endings"   value={`${Math.round((emotion?.high_risk_session_share ?? 0) * 100)}%`} sub={`${emotion?.total_sessions_with_emotion ?? 0} sessions`} Icon={IcoFlag}     accent="#ffb4ab" />
                <StatCard label="Emotions Logged"     value={(emotion?.emotion_distribution ?? []).reduce((s, e) => s + e.count, 0)}                                  Icon={IcoTarget}   accent="#b472f1" />
              </div>

              <Card title="Customer Emotion Distribution" subtitle="All sessions">
                {(emotion?.emotion_distribution?.length ?? 0) === 0 ? (
                  <EmptyState icon={IcoInbox} title="No emotion data yet" description="Emotion data will appear after training sessions complete" />
                ) : (
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={emotion.emotion_distribution} margin={{ top: 8, right: 8, left: -24, bottom: 4 }}>
                      <CartesianGrid {...gridProps} />
                      <XAxis dataKey="emotion" tick={axisTick} tickLine={false} axisLine={axisLine} />
                      <YAxis allowDecimals={false} tick={axisTick} tickLine={false} axisLine={false} />
                      <Tooltip content={<ChartTooltip />} />
                      <Bar dataKey="count" name="Logs" fill="#a5d6a7" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </Card>
            </div>
          )}

          {/* ── Abuse ── */}
          {tab === 'Abuse' && <AbuseQueue flags={toArray(flags)} onResolved={loadFlags} />}
        </>
      )}
    </DashboardShell>
  );
}
