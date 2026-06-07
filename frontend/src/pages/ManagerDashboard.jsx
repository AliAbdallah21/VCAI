import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { managerAPI } from '../services/api';
import AbuseQueue from '../components/AbuseQueue';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

const SKILLS = [
  { key: 'communication', label: 'Communication' },
  { key: 'product_knowledge', label: 'Product Knowledge' },
  { key: 'objection_handling', label: 'Objection Handling' },
  { key: 'rapport', label: 'Rapport' },
  { key: 'closing', label: 'Closing' },
];

function Shell({ children, user, logout }) {
  const navigate = useNavigate();
  const initials = user?.full_name?.split(' ').map((n) => n[0]).join('').slice(0, 2).toUpperCase() || 'M';
  return (
    <div className="min-h-screen" style={{ background: '#030712' }}>
      <header
        className="sticky top-0 z-30 flex items-center justify-between px-5 py-3"
        style={{ background: 'rgba(8,14,28,0.96)', borderBottom: '1px solid rgba(255,255,255,0.05)', backdropFilter: 'blur(20px)' }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)', boxShadow: '0 0 18px rgba(37,99,235,0.35)' }}
          >
            <span className="heading text-white font-bold text-sm">V</span>
          </div>
          <div>
            <p className="heading font-bold text-white text-sm tracking-wider">VCAI Manager</p>
            <p className="text-xs" style={{ color: 'rgba(148,163,184,0.4)' }}>Team dashboard</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Link to="/seats" className="text-xs font-semibold px-3 py-1.5 rounded-lg" style={{ color: '#60a5fa', background: 'rgba(37,99,235,0.1)', border: '1px solid rgba(37,99,235,0.18)' }}>
            Seats
          </Link>
          <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white" style={{ background: 'linear-gradient(135deg, #3b82f6, #7c3aed)' }} title={user?.email}>
            {initials}
          </div>
          <button
            onClick={() => { logout(); navigate('/login'); }}
            className="px-3 py-1.5 rounded-lg text-xs font-medium text-slate-500 hover:text-red-400"
            style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}
          >
            Sign out
          </button>
        </div>
      </header>
      <main className="p-4 md:p-8 max-w-6xl mx-auto">{children}</main>
    </div>
  );
}

function MetricCard({ label, value, sub, color }) {
  return (
    <div className="rounded-2xl p-5 relative overflow-hidden" style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}>
      <p className="text-xs font-medium tracking-wide uppercase mb-3" style={{ color: 'rgba(148,163,184,0.5)' }}>{label}</p>
      <p className="heading text-3xl font-bold" style={{ color }}>{value}</p>
      {sub && <p className="text-xs mt-1" style={{ color: 'rgba(148,163,184,0.4)' }}>{sub}</p>}
      <div className="absolute bottom-0 left-0 right-0 h-px" style={{ background: `linear-gradient(90deg, transparent, ${color}40, transparent)` }} />
    </div>
  );
}

function Card({ title, children, right }) {
  return (
    <div className="rounded-2xl p-5" style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-white">{title}</h3>
        {right}
      </div>
      {children}
    </div>
  );
}

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl p-3 text-xs" style={{ background: 'rgba(8,14,28,0.97)', border: '1px solid rgba(255,255,255,0.1)' }}>
      <p className="font-semibold text-white mb-1">{label}</p>
      {payload.map((e) => (
        <div key={e.dataKey} className="flex justify-between gap-4">
          <span style={{ color: e.color }}>{e.name}</span>
          <span className="font-bold text-white">{e.value}</span>
        </div>
      ))}
    </div>
  );
};

const TABS = ['Overview', 'Agents', 'Emotion trends', 'Abuse'];

const toArray = (value) => (Array.isArray(value) ? value : []);

export default function ManagerDashboard() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [tab, setTab] = useState('Overview');
  const [analytics, setAnalytics] = useState(null);
  const [agentsData, setAgentsData] = useState(null);
  const [emotion, setEmotion] = useState(null);
  const [flags, setFlags] = useState([]);
  const [loading, setLoading] = useState(true);

  const loadFlags = () => managerAPI.getAbuse().then((data) => setFlags(toArray(data))).catch(() => setFlags([]));

  useEffect(() => {
    Promise.all([
      managerAPI.getAnalytics().then(setAnalytics).catch(() => setAnalytics(null)),
      managerAPI.getAgents().then(setAgentsData).catch(() => setAgentsData(null)),
      managerAPI.getEmotionTrends().then(setEmotion).catch(() => setEmotion(null)),
      loadFlags(),
    ]).finally(() => setLoading(false));
  }, []);

  const abuseFlags = toArray(flags);
  const openFlags = abuseFlags.filter((f) => f.status === 'open').length;

  return (
    <Shell user={user} logout={logout}>
      {/* Tabs */}
      <div className="flex gap-2 mb-6 flex-wrap">
        {TABS.map((t) => {
          const active = tab === t;
          return (
            <button
              key={t}
              onClick={() => setTab(t)}
              className="px-3.5 py-1.5 rounded-lg text-xs font-semibold"
              style={{
                background: active ? 'rgba(59,130,246,0.15)' : 'rgba(255,255,255,0.03)',
                border: `1px solid ${active ? 'rgba(59,130,246,0.45)' : 'rgba(255,255,255,0.07)'}`,
                color: active ? '#60a5fa' : 'rgba(148,163,184,0.55)',
              }}
            >
              {t}{t === 'Abuse' && openFlags > 0 ? ` (${openFlags})` : ''}
            </button>
          );
        })}
      </div>

      {loading ? (
        <p className="text-sm py-16 text-center" style={{ color: 'rgba(148,163,184,0.5)' }}>Loading...</p>
      ) : (
        <>
          {/* ── Overview ── */}
          {tab === 'Overview' && (
            <div className="space-y-5">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard label="Active agents" value={analytics?.active_agents ?? 0} color="#60a5fa" />
                <MetricCard label="Sessions this period" value={analytics?.sessions_this_period ?? 0} sub={`${analytics?.total_sessions ?? 0} total`} color="#34d399" />
                <MetricCard label="Team avg score" value={analytics?.avg_score ?? '—'} color="#a78bfa" />
                <MetricCard label="Open abuse flags" value={openFlags} color={openFlags > 0 ? '#ef4444' : '#34d399'} />
              </div>

              {analytics?.weakest_skill && (
                <div className="rounded-2xl p-4" style={{ background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.18)' }}>
                  <p className="text-sm text-white">
                    Weakest team skill:{' '}
                    <span className="font-bold" style={{ color: '#f87171' }}>
                      {SKILLS.find((s) => s.key === analytics.weakest_skill.skill)?.label || analytics.weakest_skill.skill}
                    </span>{' '}
                    <span style={{ color: 'rgba(148,163,184,0.6)' }}>(avg {analytics.weakest_skill.average})</span>
                  </p>
                </div>
              )}

              <Card title="Sessions per day (last 30 days)">
                {(analytics?.sessions_per_day?.length ?? 0) === 0 ? (
                  <p className="text-xs py-8 text-center" style={{ color: 'rgba(148,163,184,0.4)' }}>No sessions yet.</p>
                ) : (
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={analytics.sessions_per_day} margin={{ top: 8, right: 16, left: -24, bottom: 4 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                      <XAxis dataKey="date" tick={{ fill: 'rgba(148,163,184,0.45)', fontSize: 11 }} tickLine={false} axisLine={{ stroke: 'rgba(255,255,255,0.06)' }} />
                      <YAxis allowDecimals={false} tick={{ fill: 'rgba(148,163,184,0.45)', fontSize: 11 }} tickLine={false} axisLine={false} />
                      <Tooltip content={<ChartTooltip />} />
                      <Line type="monotone" dataKey="count" name="Sessions" stroke="#60a5fa" strokeWidth={2} dot={{ r: 2.5, fill: '#60a5fa' }} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </Card>

              <Card title="Score distribution">
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={analytics?.score_distribution ?? []} margin={{ top: 8, right: 16, left: -24, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                    <XAxis dataKey="bucket" tick={{ fill: 'rgba(148,163,184,0.45)', fontSize: 11 }} tickLine={false} axisLine={{ stroke: 'rgba(255,255,255,0.06)' }} />
                    <YAxis allowDecimals={false} tick={{ fill: 'rgba(148,163,184,0.45)', fontSize: 11 }} tickLine={false} axisLine={false} />
                    <Tooltip content={<ChartTooltip />} />
                    <Bar dataKey="count" name="Sessions" fill="#7c3aed" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </div>
          )}

          {/* ── Agents ── */}
          {tab === 'Agents' && (
            <Card
              title="Agents"
              right={<span className="text-xs" style={{ color: 'rgba(148,163,184,0.45)' }}>{agentsData?.pending_invites ?? 0} pending invite(s)</span>}
            >
              {(agentsData?.agents?.length ?? 0) === 0 ? (
                <p className="text-xs py-8 text-center" style={{ color: 'rgba(148,163,184,0.4)' }}>No agents yet. Invite agents from Seats.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left" style={{ color: 'rgba(148,163,184,0.5)' }}>
                        <th className="font-medium pb-2 pr-4">Agent</th>
                        <th className="font-medium pb-2 pr-4">Sessions</th>
                        <th className="font-medium pb-2 pr-4">Avg score</th>
                        <th className="font-medium pb-2 pr-4">Streak</th>
                        <th className="font-medium pb-2 pr-4">Last session</th>
                        <th className="font-medium pb-2">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {agentsData.agents.map((a) => (
                        <tr
                          key={a.user_id}
                          onClick={() => navigate(`/manager/agents/${a.user_id}`)}
                          className="cursor-pointer hover:bg-white/[0.02]"
                          style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
                        >
                          <td className="py-2.5 pr-4">
                            <p className="text-slate-200 font-medium">{a.full_name}</p>
                            <p className="text-xs" style={{ color: 'rgba(148,163,184,0.45)' }}>{a.email}</p>
                          </td>
                          <td className="py-2.5 pr-4 text-slate-300">{a.completed_sessions}/{a.total_sessions}</td>
                          <td className="py-2.5 pr-4 text-slate-300">{a.avg_overall_score ?? '—'}</td>
                          <td className="py-2.5 pr-4 text-slate-300">{a.current_streak}</td>
                          <td className="py-2.5 pr-4" style={{ color: 'rgba(148,163,184,0.55)' }}>
                            {a.last_session_date ? new Date(a.last_session_date).toLocaleDateString() : '—'}
                          </td>
                          <td className="py-2.5">
                            <span className="text-xs font-semibold" style={{ color: a.is_active ? '#34d399' : 'rgba(148,163,184,0.5)' }}>
                              {a.is_active ? 'Active' : 'Inactive'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          )}

          {/* ── Emotion trends ── */}
          {tab === 'Emotion trends' && (
            <div className="space-y-5">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <MetricCard label="Avg customer mood" value={emotion?.avg_mood_score ?? '—'} color="#60a5fa" />
                <MetricCard label="High-risk endings" value={`${Math.round((emotion?.high_risk_session_share ?? 0) * 100)}%`} sub={`${emotion?.total_sessions_with_emotion ?? 0} sessions`} color="#ef4444" />
                <MetricCard label="Emotions logged" value={(emotion?.emotion_distribution ?? []).reduce((s, e) => s + e.count, 0)} color="#a78bfa" />
              </div>
              <Card title="Customer emotion distribution">
                {(emotion?.emotion_distribution?.length ?? 0) === 0 ? (
                  <p className="text-xs py-8 text-center" style={{ color: 'rgba(148,163,184,0.4)' }}>No emotion data yet.</p>
                ) : (
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={emotion.emotion_distribution} margin={{ top: 8, right: 16, left: -24, bottom: 4 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                      <XAxis dataKey="emotion" tick={{ fill: 'rgba(148,163,184,0.45)', fontSize: 11 }} tickLine={false} axisLine={{ stroke: 'rgba(255,255,255,0.06)' }} />
                      <YAxis allowDecimals={false} tick={{ fill: 'rgba(148,163,184,0.45)', fontSize: 11 }} tickLine={false} axisLine={false} />
                      <Tooltip content={<ChartTooltip />} />
                      <Bar dataKey="count" name="Logs" fill="#34d399" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </Card>
            </div>
          )}

          {/* ── Abuse ── */}
          {tab === 'Abuse' && <AbuseQueue flags={abuseFlags} onResolved={loadFlags} />}
        </>
      )}
    </Shell>
  );
}
