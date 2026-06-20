import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import Layout from '../components/Layout';
import { sessionsAPI } from '../services/api';

const TABS = [
  { key: 'all',        label: 'All' },
  { key: 'active',     label: 'Active' },
  { key: 'incomplete', label: 'Needs Evaluation' },
  { key: 'evaluated',  label: 'Evaluated' },
];

const statusStyle = {
  active:    { bg: 'rgba(165,214,167,0.1)',  text: '#a5d6a7', border: 'rgba(165,214,167,0.2)',  label: 'Live'      },
  ended:     { bg: 'rgba(152,141,157,0.08)', text: 'var(--text-muted)', border: 'rgba(152,141,157,0.15)', label: 'Ended'    },
  completed: { bg: 'rgba(152,141,157,0.08)', text: 'var(--text-muted)', border: 'rgba(152,141,157,0.15)', label: 'Ended'   },
};

const diffStyle = {
  easy:   { bg: 'rgba(165,214,167,0.1)',  text: '#a5d6a7', border: 'rgba(165,214,167,0.2)'  },
  medium: { bg: 'rgba(233,196,106,0.1)',  text: '#e9c46a', border: 'rgba(233,196,106,0.2)'  },
  hard:   { bg: 'rgba(255,180,171,0.1)',  text: '#ffb4ab', border: 'rgba(255,180,171,0.2)'  },
};

const scoreColor = s => s >= 80 ? '#a5d6a7' : s >= 60 ? '#e9c46a' : '#ffb4ab';

const fmtDate = d => new Date(d).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
const fmtDuration = s => { if (!s) return '—'; const m = Math.floor(s / 60); const sec = s % 60; return `${m}:${String(sec).padStart(2,'0')}`; };

export default function SessionsPage() {
  const navigate = useNavigate();
  const [tab, setTab]           = useState('all');
  const [sessions, setSessions] = useState([]);
  const [total, setTotal]       = useState(0);
  const [loading, setLoading]   = useState(true);
  const [resuming, setResuming] = useState(null);
  const [offset, setOffset]     = useState(0);
  const LIMIT = 20;

  const load = useCallback((off = 0) => {
    setLoading(true);
    sessionsAPI.getAll(LIMIT, off)
      .then(data => {
        if (off === 0) setSessions(data.sessions || []);
        else setSessions(prev => [...prev, ...(data.sessions || [])]);
        setTotal(data.total || 0);
        setOffset(off);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(0); }, [load]);

  const filtered = sessions.filter(s => {
    if (tab === 'active')     return s.status === 'active';
    if (tab === 'incomplete') return s.status !== 'active' && !s.overall_score;
    if (tab === 'evaluated')  return !!s.overall_score;
    return true;
  });

  const handleResume = async (s) => {
    setResuming(s.id);
    try {
      if (s.status !== 'active') await sessionsAPI.reactivate(s.id);
      navigate(`/session/${s.id}`);
    } catch {
      setResuming(null);
    }
  };

  return (
    <Layout>
      <div className="p-4 md:p-8 max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-8 slide-up">
          <h1 className="heading text-2xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>Resume a Call</h1>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Continue any previous session or start where you left off
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
                  {t.key === 'active'     ? sessions.filter(s => s.status === 'active').length             : ''}
                  {t.key === 'incomplete' ? sessions.filter(s => s.status !== 'active' && !s.overall_score).length : ''}
                  {t.key === 'evaluated'  ? sessions.filter(s => !!s.overall_score).length                 : ''}
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
              <p className="font-medium mb-1" style={{ color: 'var(--text-muted)' }}>No sessions found</p>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                {tab === 'active' ? 'No active calls right now' : 'Try a different filter'}
              </p>
            </div>
          ) : (
            <>
              {filtered.map((s, i) => {
                const isActive   = s.status === 'active';
                const evaluated  = !!s.overall_score;
                const st         = statusStyle[s.status] || statusStyle.ended;
                const df         = diffStyle[s.difficulty] || diffStyle.medium;
                const isResuming = resuming === s.id;

                return (
                  <div
                    key={s.id}
                    className="flex items-center px-4 md:px-6 py-3 md:py-4 gap-2 transition-colors duration-150"
                    style={{
                      borderBottom: i < filtered.length - 1 ? '1px solid var(--border)' : 'none',
                    }}
                    onMouseEnter={e => e.currentTarget.style.background = 'var(--primary-soft)'}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                  >
                    {/* Left: info */}
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
                        <div className="flex items-center gap-2 flex-wrap">
                          <p className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>{s.persona_name || 'Unknown Persona'}</p>
                          {isActive && (
                            <span className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold"
                              style={{ background: 'rgba(165,214,167,0.12)', color: '#a5d6a7', border: '1px solid rgba(165,214,167,0.2)' }}>
                              <span className="w-1.5 h-1.5 rounded-full animate-pulse inline-block" style={{ background: '#a5d6a7' }} />
                              Live
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-3 mt-0.5">
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{fmtDate(s.started_at)}</p>
                          {s.duration_seconds > 0 && (
                            <p className="text-xs" style={{ color: 'var(--text-subtle)' }}>
                              {fmtDuration(s.duration_seconds)}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Mid: badges */}
                    <div className="hidden sm:flex items-center gap-2 mx-4">
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
                      {evaluated && (
                        <span className="heading text-base font-bold" style={{ color: scoreColor(s.overall_score) }}>
                          {s.overall_score}
                        </span>
                      )}
                    </div>

                    {/* Right: actions */}
                    <div className="flex items-center gap-2 flex-shrink-0">
                      {/* Resume — always available */}
                      <button
                        onClick={() => handleResume(s)}
                        disabled={isResuming}
                        className="px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all duration-150 disabled:opacity-50"
                        style={{ background: 'var(--primary-soft)', color: 'var(--primary)', border: '1px solid rgba(222,183,255,0.2)' }}
                        onMouseEnter={e => { e.currentTarget.style.background = 'var(--primary-soft-hover)'; }}
                        onMouseLeave={e => { e.currentTarget.style.background = 'var(--primary-soft)'; }}
                      >
                        {isResuming ? '…' : isActive ? 'Resume →' : 'Continue →'}
                      </button>

                      {/* Evaluate / View Report — for ended sessions */}
                      {!isActive && (
                        <button
                          onClick={() => navigate(`/evaluation/${s.id}`)}
                          className="px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all duration-150"
                          style={evaluated
                            ? { background: 'var(--primary-soft)', color: 'var(--primary)', border: '1px solid rgba(222,183,255,0.2)' }
                            : { background: 'rgba(233,196,106,0.1)', color: '#e9c46a', border: '1px solid rgba(233,196,106,0.2)' }
                          }
                          onMouseEnter={e => { e.currentTarget.style.opacity = '0.8'; }}
                          onMouseLeave={e => { e.currentTarget.style.opacity = '1'; }}
                        >
                          {evaluated ? 'View Report' : 'Evaluate'}
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}

              {/* Load more */}
              {sessions.length < total && (
                <div className="p-4 text-center" style={{ borderTop: '1px solid var(--border)' }}>
                  <button
                    onClick={() => load(sessions.length)}
                    disabled={loading}
                    className="px-6 py-2 rounded-xl text-sm font-medium transition-all duration-150 disabled:opacity-40"
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
