import { useState, useEffect } from 'react';
import { Link, useParams, useNavigate } from 'react-router-dom';
import { managerAPI } from '../services/api';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

const SKILLS = [
  { key: 'communication', label: 'Communication', color: '#3b82f6' },
  { key: 'product_knowledge', label: 'Product Knowledge', color: '#f59e0b' },
  { key: 'objection_handling', label: 'Objection Handling', color: '#ef4444' },
  { key: 'rapport', label: 'Rapport', color: '#10b981' },
  { key: 'closing', label: 'Closing', color: '#8b5cf6' },
];

function SkillBar({ label, color, value }) {
  const pct = value == null ? 0 : Math.max(0, Math.min(100, value));
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-medium" style={{ color: 'rgba(148,163,184,0.7)' }}>{label}</span>
        <span className="text-xs font-bold text-white">{value == null ? '—' : value}</span>
      </div>
      <div className="h-2 rounded-full" style={{ background: 'rgba(255,255,255,0.05)' }}>
        <div className="h-2 rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl p-3 text-xs" style={{ background: 'rgba(8,14,28,0.97)', border: '1px solid rgba(255,255,255,0.1)' }}>
      <p className="font-semibold text-white mb-1">{label}</p>
      <p className="font-bold" style={{ color: '#60a5fa' }}>{payload[0].value}</p>
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
    <div className="min-h-screen" style={{ background: '#030712' }}>
      <header
        className="sticky top-0 z-30 flex items-center gap-3 px-5 py-3"
        style={{ background: 'rgba(8,14,28,0.96)', borderBottom: '1px solid rgba(255,255,255,0.05)', backdropFilter: 'blur(20px)' }}
      >
        <button
          onClick={() => navigate('/dashboard')}
          className="text-xs font-semibold px-3 py-1.5 rounded-lg"
          style={{ color: '#60a5fa', background: 'rgba(37,99,235,0.1)', border: '1px solid rgba(37,99,235,0.18)' }}
        >
          ← Back
        </button>
        <p className="heading font-bold text-white text-sm tracking-wider">Agent detail</p>
      </header>

      <main className="p-4 md:p-8 max-w-5xl mx-auto">
        {loading ? (
          <p className="text-sm py-16 text-center" style={{ color: 'rgba(148,163,184,0.5)' }}>Loading...</p>
        ) : error ? (
          <p className="text-sm py-16 text-center" style={{ color: '#f87171' }}>{error}</p>
        ) : (
          <div className="space-y-5">
            <div>
              <h1 className="heading text-2xl font-bold text-white mb-1">{data.full_name}</h1>
              <p className="text-sm" style={{ color: 'rgba(148,163,184,0.55)' }}>
                {data.email} · {data.is_active ? 'Active' : 'Inactive'}
              </p>
            </div>

            <div className="rounded-2xl p-5" style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}>
              <h3 className="text-sm font-semibold text-white mb-4">Per-skill averages</h3>
              <div className="space-y-3">
                {SKILLS.map((s) => (
                  <SkillBar key={s.key} label={s.label} color={s.color} value={data.skill_averages?.[s.key] ?? null} />
                ))}
              </div>
            </div>

            <div className="rounded-2xl p-5" style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}>
              <h3 className="text-sm font-semibold text-white mb-4">Overall score trend</h3>
              {trendData.length < 2 ? (
                <p className="text-xs py-8 text-center" style={{ color: 'rgba(148,163,184,0.4)' }}>
                  Needs at least 2 scored sessions to show a trend.
                </p>
              ) : (
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={trendData} margin={{ top: 8, right: 16, left: -24, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                    <XAxis dataKey="label" tick={{ fill: 'rgba(148,163,184,0.45)', fontSize: 11 }} tickLine={false} axisLine={{ stroke: 'rgba(255,255,255,0.06)' }} interval="preserveStartEnd" />
                    <YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tick={{ fill: 'rgba(148,163,184,0.45)', fontSize: 11 }} tickLine={false} axisLine={false} />
                    <Tooltip content={<ChartTooltip />} />
                    <Line type="monotone" dataKey="overall_score" name="Overall" stroke="#60a5fa" strokeWidth={2} dot={{ r: 3, fill: '#60a5fa' }} connectNulls={false} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>

            <div className="rounded-2xl p-5" style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}>
              <h3 className="text-sm font-semibold text-white mb-4">Session history</h3>
              {(data.sessions?.length ?? 0) === 0 ? (
                <p className="text-xs py-8 text-center" style={{ color: 'rgba(148,163,184,0.4)' }}>No sessions yet.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left" style={{ color: 'rgba(148,163,184,0.5)' }}>
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
                        <tr key={s.id} style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                          <td className="py-2.5 pr-4 text-slate-200">{s.persona_name || s.persona_id}</td>
                          <td className="py-2.5 pr-4" style={{ color: 'rgba(148,163,184,0.6)' }}>{s.difficulty || '—'}</td>
                          <td className="py-2.5 pr-4 text-slate-300">{s.overall_score ?? '—'}</td>
                          <td className="py-2.5 pr-4 text-slate-300">{s.turn_count}</td>
                          <td className="py-2.5 pr-4" style={{ color: 'rgba(148,163,184,0.55)' }}>
                            {s.started_at ? new Date(s.started_at).toLocaleDateString() : '—'}
                          </td>
                          <td className="py-2.5">
                            <Link to={`/evaluation/${s.id}`} className="text-xs font-semibold" style={{ color: '#60a5fa' }}>
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
