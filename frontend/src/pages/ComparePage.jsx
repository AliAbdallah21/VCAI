import { useState, useEffect, useMemo } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { sessionsAPI, evaluationAPI } from '../services/api';
import Layout from '../components/Layout';

/* ──────────────────────────────────────────────────────────────────────── */
/* Helpers                                                                  */
/* ──────────────────────────────────────────────────────────────────────── */

const fmtDate = iso => {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
};

const fmtDuration = s => {
  if (!s || s <= 0) return '—';
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${sec.toString().padStart(2, '0')}`;
};

const scoreTone = (score) => {
  if (score == null) return { color: '#94a3b8', bg: 'rgba(148,163,184,0.08)', border: 'rgba(148,163,184,0.15)' };
  if (score >= 75) return { color: '#34d399', bg: 'rgba(16,185,129,0.08)', border: 'rgba(16,185,129,0.2)' };
  if (score >= 50) return { color: '#fbbf24', bg: 'rgba(245,158,11,0.08)', border: 'rgba(245,158,11,0.2)' };
  return { color: '#f87171', bg: 'rgba(239,68,68,0.08)', border: 'rgba(239,68,68,0.2)' };
};

const diffStyle = {
  easy:   { color: '#34d399', bg: 'rgba(16,185,129,0.1)' },
  medium: { color: '#fbbf24', bg: 'rgba(245,158,11,0.1)' },
  hard:   { color: '#f87171', bg: 'rgba(239,68,68,0.1)' },
};

/* ──────────────────────────────────────────────────────────────────────── */
/* Sub-components                                                           */
/* ──────────────────────────────────────────────────────────────────────── */

const SessionSelector = ({ sessions, value, onChange, otherValue, label }) => (
  <div className="mb-4">
    <label className="text-xs font-semibold uppercase tracking-wider mb-2 block" style={{ color: 'rgba(148,163,184,0.5)' }}>
      {label}
    </label>
    <select
      value={value || ''}
      onChange={e => onChange(e.target.value || null)}
      className="w-full px-3 py-2.5 rounded-xl text-sm"
      style={{
        background: 'rgba(13,21,38,0.9)',
        border: '1px solid rgba(255,255,255,0.08)',
        color: '#e2e8f0',
      }}
    >
      <option value="">Select a session…</option>
      {sessions.map(s => (
        <option key={s.id} value={s.id} disabled={s.id === otherValue}>
          {s.persona_name || 'Unknown'} · {s.difficulty} · {fmtDate(s.started_at)}
          {s.overall_score != null ? ` · ${s.overall_score}%` : ''}
        </option>
      ))}
    </select>
  </div>
);

const ScoreRing = ({ score }) => {
  const tone = scoreTone(score);
  const display = score == null ? '—' : `${score}`;
  return (
    <div
      className="w-28 h-28 rounded-full flex items-center justify-center flex-shrink-0"
      style={{
        background: `radial-gradient(circle, ${tone.bg} 0%, transparent 70%)`,
        border: `2px solid ${tone.border}`,
      }}
    >
      <div className="text-center">
        <p className="text-3xl font-bold" style={{ color: tone.color }}>{display}</p>
        <p className="text-xs" style={{ color: 'rgba(148,163,184,0.5)' }}>out of 100</p>
      </div>
    </div>
  );
};

const StatLine = ({ label, value, tone }) => (
  <div className="flex items-center justify-between py-1.5 text-xs">
    <span style={{ color: 'rgba(148,163,184,0.55)' }}>{label}</span>
    <span className="font-semibold" style={{ color: tone || '#e2e8f0' }}>{value}</span>
  </div>
);

const SkillBar = ({ name, score, otherScore }) => {
  const tone = scoreTone(score);
  const diff = otherScore != null && score != null ? score - otherScore : null;
  return (
    <div className="mb-2.5">
      <div className="flex items-center justify-between mb-1 text-xs">
        <span className="font-medium" style={{ color: '#cbd5e1' }}>{name}</span>
        <span className="flex items-center gap-1.5">
          {diff !== null && (
            <span className="text-xs font-medium" style={{
              color: diff > 0 ? '#34d399' : diff < 0 ? '#f87171' : 'rgba(148,163,184,0.4)'
            }}>
              {diff > 0 ? `+${diff}` : diff < 0 ? `${diff}` : '='}
            </span>
          )}
          <span className="font-semibold" style={{ color: tone.color }}>{score}%</span>
        </span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.04)' }}>
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${score}%`, background: tone.color, boxShadow: `0 0 6px ${tone.color}60` }}
        />
      </div>
    </div>
  );
};

/* ──────────────────────────────────────────────────────────────────────── */
/* Side panel — one session                                                 */
/* ──────────────────────────────────────────────────────────────────────── */

const Side = ({ data, error, loading, otherData }) => {
  if (loading) {
    return (
      <div className="py-16 text-center">
        <div
          className="w-6 h-6 spin-ring mx-auto"
          style={{ border: '2px solid rgba(255,255,255,0.08)', borderTopColor: '#3b82f6', borderRadius: '50%' }}
        />
        <p className="text-xs mt-3" style={{ color: 'rgba(148,163,184,0.4)' }}>Loading report…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="py-12 text-center px-4">
        <p className="text-sm" style={{ color: '#f87171' }}>{error}</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="py-16 text-center">
        <svg className="mx-auto mb-3" width="32" height="32" fill="none" stroke="rgba(148,163,184,0.3)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
          <path d="M9 17.25v1.007a3 3 0 01-.879 2.122L7.5 21h9l-.621-.621A3 3 0 0115 18.257V17.25m6-12V15a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 15V5.25m18 0A2.25 2.25 0 0018.75 3H5.25A2.25 2.25 0 003 5.25m18 0V12a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 12V5.25" />
        </svg>
        <p className="text-sm" style={{ color: 'rgba(148,163,184,0.45)' }}>Pick a session to compare</p>
      </div>
    );
  }

  // data is the FinalReport JSON
  const overall = data.scores?.overall_score ?? data.overall_score ?? null;
  const stats = data.quick_stats || {};
  const skills = data.scores?.skills || [];
  const fc = data.fact_check;
  const tone = scoreTone(overall);

  // Build a {skill_key: score} map for the "other" side so we can compute deltas
  const otherSkills = otherData?.scores?.skills || [];
  const otherMap = Object.fromEntries(otherSkills.map(s => [s.skill_key, s.score]));

  return (
    <div className="space-y-4">
      {/* Header — persona + overall score */}
      <div className="flex items-center gap-4">
        <ScoreRing score={overall} />
        <div className="min-w-0 flex-1">
          <p className="font-bold text-white text-base truncate">{data.persona_name || 'Session'}</p>
          <p className="text-xs mt-1 mb-2" style={{ color: 'rgba(148,163,184,0.5)' }}>
            {fmtDate(data.created_at)}
          </p>
          <span className="text-xs px-2 py-0.5 rounded-lg font-medium"
            style={{ color: tone.color, background: tone.bg, border: `1px solid ${tone.border}` }}>
            {overall >= 75 ? 'Passed' : overall >= 50 ? 'Needs improvement' : 'Failed'}
          </span>
        </div>
      </div>

      {/* Quick stats */}
      <div className="rounded-xl p-4"
        style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.05)' }}>
        <StatLine label="Duration" value={fmtDuration(stats.duration_seconds)} />
        <StatLine label="Total turns" value={stats.total_turns ?? '—'} />
        <StatLine label="Salesperson turns" value={stats.salesperson_turns ?? '—'} />
        <StatLine label="Final customer emotion" value={stats.final_customer_emotion ?? '—'} />
        <StatLine label="Checkpoints achieved" value={`${stats.checkpoints_achieved ?? 0} / ${stats.checkpoints_total ?? 6}`} />
      </div>

      {/* Skill breakdown with deltas */}
      <div className="rounded-xl p-4"
        style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.05)' }}>
        <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'rgba(148,163,184,0.5)' }}>
          Skills
        </p>
        {skills.length > 0 ? (
          skills.map((s, i) => (
            <SkillBar
              key={s.skill_key || i}
              name={s.skill_name}
              score={s.score}
              otherScore={otherMap[s.skill_key] ?? null}
            />
          ))
        ) : (
          <p className="text-xs" style={{ color: 'rgba(148,163,184,0.4)' }}>No skill data</p>
        )}
      </div>

      {/* Fact-check */}
      <div className="rounded-xl p-4"
        style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.05)' }}>
        <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'rgba(148,163,184,0.5)' }}>
          Factual Accuracy
        </p>
        {fc && fc.claims_checked > 0 ? (
          <>
            <div className="flex items-center justify-between mb-2">
              <p className="text-2xl font-bold" style={{ color: scoreTone(Math.round((fc.accuracy_rate || 0) * 100)).color }}>
                {Math.round((fc.accuracy_rate || 0) * 100)}%
              </p>
              <div className="text-right text-xs leading-tight" style={{ color: 'rgba(148,163,184,0.6)' }}>
                <p>{fc.accurate_count} accurate · {fc.inaccurate_count} wrong</p>
                {fc.properties_discussed?.length > 0 && (
                  <p className="truncate max-w-[150px]">{fc.properties_discussed.join(', ')}</p>
                )}
              </div>
            </div>
            {fc.errors?.length > 0 && (
              <div className="space-y-1 mt-2">
                {fc.errors.slice(0, 3).map((err, i) => (
                  <p key={i} className="text-xs" style={{ color: 'rgba(248,113,113,0.85)' }}>
                    • <span className="capitalize">{err.claim_type}</span>: {err.claimed_value}
                    <span style={{ color: 'rgba(148,163,184,0.5)' }}> → {err.correct_value}</span>
                  </p>
                ))}
                {fc.errors.length > 3 && (
                  <p className="text-xs italic" style={{ color: 'rgba(148,163,184,0.4)' }}>
                    + {fc.errors.length - 3} more
                  </p>
                )}
              </div>
            )}
          </>
        ) : (
          <p className="text-xs" style={{ color: 'rgba(148,163,184,0.4)' }}>No factual claims checked</p>
        )}
      </div>
    </div>
  );
};

/* ──────────────────────────────────────────────────────────────────────── */
/* Main page                                                                */
/* ──────────────────────────────────────────────────────────────────────── */

export default function ComparePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const [sessions, setSessions]   = useState([]);
  const [loadingList, setLoadingList] = useState(true);

  const sessionA = searchParams.get('a') || null;
  const sessionB = searchParams.get('b') || null;

  const [reportA, setReportA] = useState(null);
  const [reportB, setReportB] = useState(null);
  const [loadA, setLoadA] = useState(false);
  const [loadB, setLoadB] = useState(false);
  const [errA, setErrA] = useState(null);
  const [errB, setErrB] = useState(null);

  // Load session list once
  useEffect(() => {
    setLoadingList(true);
    sessionsAPI.getAll(50, 0)
      .then(data => setSessions(data.sessions || []))
      .catch(() => setSessions([]))
      .finally(() => setLoadingList(false));
  }, []);

  // Load report A
  useEffect(() => {
    if (!sessionA) { setReportA(null); setErrA(null); return; }
    setLoadA(true); setErrA(null);
    evaluationAPI.getReport(sessionA)
      .then(r => {
        if (r.status === 'completed') setReportA(r.report);
        else if (r.status === 'failed') { setReportA(null); setErrA(r.error || 'Evaluation failed for this session.'); }
        else if (r.status === 'not_started') { setReportA(null); setErrA('This session has no evaluation yet. Run it from the session report page first.'); }
        else { setReportA(null); setErrA(`Evaluation is ${r.status} — wait for it to complete, then try again.`); }
      })
      .catch(() => setErrA('Failed to load this session\'s report.'))
      .finally(() => setLoadA(false));
  }, [sessionA]);

  // Load report B
  useEffect(() => {
    if (!sessionB) { setReportB(null); setErrB(null); return; }
    setLoadB(true); setErrB(null);
    evaluationAPI.getReport(sessionB)
      .then(r => {
        if (r.status === 'completed') setReportB(r.report);
        else if (r.status === 'failed') { setReportB(null); setErrB(r.error || 'Evaluation failed for this session.'); }
        else if (r.status === 'not_started') { setReportB(null); setErrB('This session has no evaluation yet. Run it from the session report page first.'); }
        else { setReportB(null); setErrB(`Evaluation is ${r.status} — wait for it to complete, then try again.`); }
      })
      .catch(() => setErrB('Failed to load this session\'s report.'))
      .finally(() => setLoadB(false));
  }, [sessionB]);

  const setSide = (side, id) => {
    const next = new URLSearchParams(searchParams);
    if (id) next.set(side, id); else next.delete(side);
    setSearchParams(next);
  };

  // Pre-compute the overall delta (for the verdict banner)
  const verdict = useMemo(() => {
    if (!reportA || !reportB) return null;
    const a = reportA.scores?.overall_score ?? null;
    const b = reportB.scores?.overall_score ?? null;
    if (a == null || b == null) return null;
    return { a, b, diff: a - b };
  }, [reportA, reportB]);

  return (
    <Layout>
      <div className="p-4 md:p-8 max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-6 flex items-center justify-between flex-wrap gap-3">
          <div>
            <h1 className="heading text-2xl font-bold text-white mb-1">Compare Sessions</h1>
            <p className="text-sm" style={{ color: 'rgba(148,163,184,0.55)' }}>
              Pick any two completed sessions to see scores, skills, and factual accuracy side by side.
            </p>
          </div>
          <button
            onClick={() => navigate('/sessions')}
            className="text-xs px-3 py-1.5 rounded-lg font-medium"
            style={{ background: 'rgba(255,255,255,0.04)', color: 'rgba(148,163,184,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}
          >
            View all sessions
          </button>
        </div>

        {/* Verdict banner — only when both loaded */}
        {verdict && (
          <div
            className="rounded-2xl px-5 py-4 mb-5 flex items-center gap-4"
            style={{
              background: verdict.diff > 0 ? 'rgba(16,185,129,0.07)' : verdict.diff < 0 ? 'rgba(239,68,68,0.07)' : 'rgba(148,163,184,0.05)',
              border: `1px solid ${verdict.diff > 0 ? 'rgba(16,185,129,0.18)' : verdict.diff < 0 ? 'rgba(239,68,68,0.18)' : 'rgba(148,163,184,0.12)'}`,
            }}
          >
            <div className="text-2xl">
              {verdict.diff > 0 ? '📈' : verdict.diff < 0 ? '📉' : '⚖️'}
            </div>
            <div className="text-sm" style={{ color: '#cbd5e1' }}>
              {verdict.diff === 0
                ? <>Both sessions scored the same — <span className="font-semibold">{verdict.a}%</span></>
                : verdict.diff > 0
                  ? <>Left session scored <span className="font-semibold text-emerald-400">+{verdict.diff} points higher</span> ({verdict.a}% vs {verdict.b}%)</>
                  : <>Right session scored <span className="font-semibold text-emerald-400">+{Math.abs(verdict.diff)} points higher</span> ({verdict.b}% vs {verdict.a}%)</>
              }
            </div>
          </div>
        )}

        {/* Two-column compare */}
        <div className="grid md:grid-cols-2 gap-5">
          {/* ── Left ── */}
          <div className="rounded-2xl p-5" style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}>
            <SessionSelector
              sessions={sessions}
              value={sessionA}
              onChange={id => setSide('a', id)}
              otherValue={sessionB}
              label="Session A"
            />
            <Side data={reportA} loading={loadA || (loadingList && sessionA)} error={errA} otherData={reportB} />
          </div>

          {/* ── Right ── */}
          <div className="rounded-2xl p-5" style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}>
            <SessionSelector
              sessions={sessions}
              value={sessionB}
              onChange={id => setSide('b', id)}
              otherValue={sessionA}
              label="Session B"
            />
            <Side data={reportB} loading={loadB || (loadingList && sessionB)} error={errB} otherData={reportA} />
          </div>
        </div>

        {/* Help footer */}
        {!sessionA && !sessionB && !loadingList && sessions.length === 0 && (
          <div className="mt-6 rounded-2xl p-6 text-center"
            style={{ background: 'rgba(13,21,38,0.5)', border: '1px solid rgba(255,255,255,0.05)' }}>
            <p className="text-sm" style={{ color: 'rgba(148,163,184,0.55)' }}>
              You don't have any sessions yet. Start one from{' '}
              <a href="/setup" className="font-semibold" style={{ color: '#60a5fa' }}>New Session</a>.
            </p>
          </div>
        )}
      </div>
    </Layout>
  );
}
