import { useState, useEffect } from 'react';
import { Link, useParams, useNavigate } from 'react-router-dom';
import { managerAPI } from '../services/api';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

const SKILLS = [
  { key: 'communication', label: 'Communication', color: '#b472f1' },
  { key: 'product_knowledge', label: 'Product Knowledge', color: '#f4a261' },
  { key: 'objection_handling', label: 'Objection Handling', color: '#ffb4ab' },
  { key: 'rapport', label: 'Rapport', color: '#a5d6a7' },
  { key: 'closing', label: 'Closing', color: '#deb7ff' },
];

function SkillBar({ label, color, value }) {
  const pct = value == null ? 0 : Math.max(0, Math.min(100, value));
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>{label}</span>
        <span className="text-xs font-bold" style={{ color: 'var(--text-primary)' }}>{value == null ? '—' : value}</span>
      </div>
      <div className="h-2 rounded-full" style={{ background: 'var(--border)' }}>
        <div className="h-2 rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl p-3 text-xs" style={{ background: 'var(--glass-bg)', border: '1px solid var(--border-strong)' }}>
      <p className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>{label}</p>
      <p className="font-bold" style={{ color: 'var(--primary)' }}>{payload[0].value}</p>
    </div>
  );
};

export default function AgentProgress() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    setLoading(true);
    managerAPI.getAgentProgress(id)
      .then(setData)
      .catch((e) => setError(e?.response?.data?.detail || 'Could not load agent'))
      .finally(() => setLoading(false));
  }, [id]);

  const trendData = (data?.score_trend ?? []).map((p, i) => ({
    label: p.date ? new Date(p.date).toLocaleDateString() : `#${i + 1}`,
    overall_score: p.overall_score,
  }));

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg-app)' }}>
      <header
        className="sticky top-0 z-30 flex items-center gap-3 px-5 py-3"
        style={{ background: 'var(--glass-bg)', borderBottom: '1px solid var(--border)', backdropFilter: 'blur(20px)' }}
      >
        <button
          onClick={() => navigate('/dashboard')}
          className="text-xs font-semibold px-3 py-1.5 rounded-lg"
          style={{ color: 'var(--primary)', background: 'var(--primary-soft)', border: '1px solid rgba(222,183,255,0.18)' }}
        >
          ← Back
        </button>
        <p className="heading font-bold text-sm tracking-wider" style={{ color: 'var(--text-primary)' }}>Agent detail</p>
      </header>

      <main className="p-4 md:p-8 max-w-5xl mx-auto">
        {loading ? (
          <p className="text-sm py-16 text-center" style={{ color: 'var(--text-muted)' }}>Loading...</p>
        ) : error ? (
          <p className="text-sm py-16 text-center" style={{ color: 'var(--error)' }}>{error}</p>
        ) : (
          <div className="space-y-5">
            <div>
              <h1 className="heading text-2xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>{data.full_name}</h1>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                {data.email} · {data.is_active ? 'Active' : 'Inactive'}
              </p>
            </div>

            <div className="ds-card p-5">
              <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Per-skill averages</h3>
              <div className="space-y-3">
                {SKILLS.map((s) => (
                  <SkillBar key={s.key} label={s.label} color={s.color} value={data.skill_averages?.[s.key] ?? null} />
                ))}
              </div>
            </div>

            <div className="ds-card p-5">
              <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Overall score trend</h3>
              {trendData.length < 2 ? (
                <p className="text-xs py-8 text-center" style={{ color: 'var(--text-muted)' }}>
                  Needs at least 2 scored sessions to show a trend.
                </p>
              ) : (
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={trendData} margin={{ top: 8, right: 16, left: -24, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="label" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickLine={false} axisLine={{ stroke: 'var(--border)' }} interval="preserveStartEnd" />
                    <YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickLine={false} axisLine={false} />
                    <Tooltip content={<ChartTooltip />} />
                    <Line type="monotone" dataKey="overall_score" name="Overall" stroke="#b472f1" strokeWidth={2} dot={{ r: 3, fill: '#b472f1' }} connectNulls={false} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>

            <div className="ds-card p-5">
              <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Session history</h3>
              {(data.sessions?.length ?? 0) === 0 ? (
                <p className="text-xs py-8 text-center" style={{ color: 'var(--text-muted)' }}>No sessions yet.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left" style={{ color: 'var(--text-muted)' }}>
                        <th className="font-medium pb-2 pr-4">Persona</th>
                        <th className="font-medium pb-2 pr-4">Difficulty</th>
                        <th className="font-medium pb-2 pr-4">Score</th>
                        <th className="font-medium pb-2 pr-4">Turns</th>
                        <th className="font-medium pb-2 pr-4">Date</th>
                        <th className="font-medium pb-2">Report</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.sessions.map((s) => (
                        <tr key={s.id} style={{ borderTop: '1px solid var(--border)' }}>
                          <td className="py-2.5 pr-4" style={{ color: 'var(--text-primary)' }}>{s.persona_name || s.persona_id}</td>
                          <td className="py-2.5 pr-4" style={{ color: 'var(--text-secondary)' }}>{s.difficulty || '—'}</td>
                          <td className="py-2.5 pr-4" style={{ color: 'var(--text-secondary)' }}>{s.overall_score ?? '—'}</td>
                          <td className="py-2.5 pr-4" style={{ color: 'var(--text-secondary)' }}>{s.turn_count}</td>
                          <td className="py-2.5 pr-4" style={{ color: 'var(--text-muted)' }}>
                            {s.started_at ? new Date(s.started_at).toLocaleDateString() : '—'}
                          </td>
                          <td className="py-2.5">
                            <Link to={`/evaluation/${s.id}`} className="text-xs font-semibold" style={{ color: 'var(--primary)' }}>
                              View
                            </Link>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
