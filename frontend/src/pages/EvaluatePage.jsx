import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import Layout from '../components/Layout';
import { sessionsAPI, evaluationAPI } from '../services/api';

const TABS = [
  { key: 'needs',     label: 'Needs Evaluation' },
  { key: 'evaluated', label: 'Evaluated' },
  { key: 'all',       label: 'All' },
];

const scoreColor = s => s >= 80 ? '#a5d6a7' : s >= 60 ? '#e9c46a' : '#ffb4ab';
const fmtDate = d => new Date(d).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });

const diffStyle = {
  easy:   { bg: 'rgba(165,214,167,0.1)',  text: '#a5d6a7', border: 'rgba(165,214,167,0.2)'  },
  medium: { bg: 'rgba(233,196,106,0.1)',  text: '#e9c46a', border: 'rgba(233,196,106,0.2)'  },
  hard:   { bg: 'rgba(255,180,171,0.1)',  text: '#ffb4ab', border: 'rgba(255,180,171,0.2)'  },
};

export default function EvaluatePage() {
  const navigate = useNavigate();
  const [tab, setTab]             = useState('needs');
  const [sessions, setSessions]   = useState([]);
  const [total, setTotal]         = useState(0);
  const [loading, setLoading]     = useState(true);
  const [triggering, setTriggering] = useState({});
  const [offset, setOffset]       = useState(0);
  const LIMIT = 20;

  const load = useCallback((off = 0) => {
    setLoading(true);
    sessionsAPI.getAll(LIMIT, off)
      .then(data => {
        const ended = (data.sessions || []).filter(s => s.status !== 'active');
        if (off === 0) setSessions(ended);
        else setSessions(prev => [...prev, ...ended]);
        setTotal(data.total || 0);
        setOffset(off);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(0); }, [load]);

  const filtered = sessions.filter(s => {
    if (tab === 'needs')     return !s.overall_score;
    if (tab === 'evaluated') return !!s.overall_score;
    return true;
  });

  const handleEvaluate = async (s, force = false) => {
    setTriggering(prev => ({ ...prev, [s.id]: true }));
    try {
      await evaluationAPI.triggerEvaluation(s.id, 'training', force);
      navigate(`/evaluation/${s.id}`);
    } catch {
      setTriggering(prev => ({ ...prev, [s.id]: false }));
    }
  };

  return (
    <Layout>
      <div className="p-4 md:p-8 max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-8 slide-up">
          <h1 className="heading text-2xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>Evaluate a Call</h1>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Generate or re-run AI evaluations for any completed session
          </p>
        </div>

        {/* Tabs */}
        <div
          className="flex gap-1 p-1 rounded-xl mb-6 w-fit"
          style={{ background: 'var(--bg-card-alt)', border: '1px solid var(--border)' }}
        >
          {TABS.map(t => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className="px-4 py-2 rounded-lg text-sm font-medium transition-all duration-150"
              style={tab === t.key
                ? { background: 'var(--primary-soft)', color: 'var(--primary)', border: '1px solid rgba(222,183,255,0.25)' }
                : { color: 'var(--text-muted)', border: '1px solid transparent' }
              }
            >
              {t.label}
              {t.key !== 'all' && (
                <span className="ml-2 text-xs opacity-60">
                  {t.key === 'needs'     ? sessions.filter(s => !s.overall_score).length    : ''}
                  {t.key === 'evaluated' ? sessions.filter(s => !!s.overall_score).length   : ''}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Sessions list */}
        <div
          className="rounded-2xl overflow-hidden"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
        >
          {loading && offset === 0 ? (
            <div className="p-14 text-center">
              <div className="w-6 h-6 spin-ring mx-auto mb-3" style={{ border: '2px solid var(--border)', borderTopColor: 'var(--primary-container)', borderRadius: '50%' }} />
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading sessions…</p>
            </div>
          ) : filtered.length === 0 ? (
            <div className="p-14 text-center">
              <p className="font-medium mb-1" style={{ color: 'var(--text-muted)' }}>No sessions here</p>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                {tab === 'needs' ? 'All sessions have been evaluated' : 'No evaluated sessions yet'}
              </p>
            </div>
          ) : (
            <>
              {filtered.map((s, i) => {
                const evaluated = !!s.overall_score;
                const df = diffStyle[s.difficulty] || diffStyle.medium;
                const isBusy = triggering[s.id];

                return (
                  <div
                    key={s.id}
                    className="flex items-center px-6 py-4 transition-colors duration-150"
                    style={{ borderBottom: i < filtered.length - 1 ? '1px solid var(--border)' : 'none' }}
                    onMouseEnter={e => e.currentTarget.style.background = 'var(--primary-soft)'}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                  >
                    {/* Left */}
                    <div className="flex items-center gap-4 flex-1 min-w-0">
                      <div
                        className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
                        style={{ background: 'var(--bg-card-alt)', border: '1px solid var(--border)' }}
                      >
                        <svg width="15" height="15" fill="none" stroke="var(--text-muted)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                          <path d="M17.982 18.725A7.488 7.488 0 0012 15.75a7.488 7.488 0 00-5.982 2.975m11.963 0a9 9 0 10-11.963 0m11.963 0A8.966 8.966 0 0112 21a8.966 8.966 0 01-5.982-2.275M15 9.75a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                      </div>
                      <div className="min-w-0">
                        <p className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>{s.persona_name || 'Unknown Persona'}</p>
                        <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{fmtDate(s.started_at)}</p>
                      </div>
                    </div>

                    {/* Mid: difficulty + score */}
                    <div className="hidden sm:flex items-center gap-3 mx-4">
                      <span className="px-2.5 py-1 rounded-lg text-xs font-semibold capitalize"
                        style={{ background: df.bg, color: df.text, border: `1px solid ${df.border}` }}>
                        {s.difficulty}
                      </span>
                      {s.turn_count > 0 && (
                        <span className="px-2.5 py-1 rounded-lg text-xs font-medium"
                          style={{ background: 'var(--bg-card-alt)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}>
                          {s.turn_count} turns
                        </span>
                      )}
                      {evaluated ? (
                        <div className="flex items-center gap-1.5">
                          <span className="heading text-lg font-bold" style={{ color: scoreColor(s.overall_score) }}>
                            {s.overall_score}
                          </span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>/100</span>
                        </div>
                      ) : (
                        <span className="text-xs px-2.5 py-1 rounded-lg"
                          style={{ background: 'rgba(233,196,106,0.08)', color: '#e9c46a', border: '1px solid rgba(233,196,106,0.15)' }}>
                          Not evaluated
                        </span>
                      )}
                    </div>

                    {/* Right: actions */}
                    <div className="flex items-center gap-2 flex-shrink-0">
                      {evaluated ? (
                        <>
                          {/* View report */}
                          <button
                            onClick={() => navigate(`/evaluation/${s.id}`)}
                            className="px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all duration-150"
                            style={{ background: 'var(--primary-soft)', color: 'var(--primary)', border: '1px solid rgba(222,183,255,0.2)' }}
                            onMouseEnter={e => { e.currentTarget.style.background = 'var(--primary-soft-hover)'; }}
                            onMouseLeave={e => { e.currentTarget.style.background = 'var(--primary-soft)'; }}
                          >
                            View Report
                          </button>
                          {/* Re-evaluate */}
                          <button
                            onClick={() => handleEvaluate(s, true)}
                            disabled={isBusy}
                            className="px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all duration-150 disabled:opacity-50"
                            style={{ background: 'rgba(233,196,106,0.08)', color: '#e9c46a', border: '1px solid rgba(233,196,106,0.15)' }}
                            onMouseEnter={e => { e.currentTarget.style.background = 'rgba(233,196,106,0.18)'; }}
                            onMouseLeave={e => { e.currentTarget.style.background = 'rgba(233,196,106,0.08)'; }}
                          >
                            {isBusy ? '…' : '↻ Re-evaluate'}
                          </button>
                        </>
                      ) : (
                        <button
                          onClick={() => handleEvaluate(s, false)}
                          disabled={isBusy}
                          className="px-4 py-1.5 rounded-xl text-xs font-semibold transition-all duration-150 disabled:opacity-50"
                          style={{ background: 'var(--primary)', color: 'var(--primary-on)', border: '1px solid var(--primary)' }}
                          onMouseEnter={e => { e.currentTarget.style.background = 'var(--primary-hover)'; }}
                          onMouseLeave={e => { e.currentTarget.style.background = 'var(--primary)'; }}
                        >
                          {isBusy ? 'Starting…' : 'Evaluate Now →'}
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}

              {sessions.length < total && (
                <div className="p-4 text-center" style={{ borderTop: '1px solid var(--border)' }}>
                  <button
                    onClick={() => load(sessions.length)}
                    disabled={loading}
                    className="px-6 py-2 rounded-xl text-sm font-medium disabled:opacity-40"
                    style={{ background: 'var(--bg-card-alt)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
                  >
                    {loading ? 'Loading…' : `Load more (${total - sessions.length} remaining)`}
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </Layout>
  );
}
