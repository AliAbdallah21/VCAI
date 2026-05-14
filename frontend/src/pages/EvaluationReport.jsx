import { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { evaluationAPI, sessionsAPI } from '../services/api';
import Layout from '../components/Layout';

/* ── Analysis steps — ordered to match the ACTUAL backend pipeline ── */
// progressStart/End define when each step is "active" based on real backend progress.
const ANALYSIS_STEPS = [
  {
    id: 'db_fetch',
    label: 'Loading conversation from database',
    sublabel: 'Fetching turns, emotion logs & session metadata',
    progressStart: 10,
    progressEnd: 22,
    Icon: () => (
      <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4.03 3-9 3S3 13.66 3 12"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/>
      </svg>
    ),
  },
  {
    id: 'factcheck',
    label: 'Fact-checking property claims',
    sublabel: 'Cross-referencing prices, payment plans & project details',
    progressStart: 22,
    progressEnd: 40,
    Icon: () => (
      <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <path d="M2.25 12l8.954-8.955c.44-.439 1.152-.439 1.591 0L21.75 12M4.5 9.75v10.125c0 .621.504 1.125 1.125 1.125H9.75v-4.875c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21h4.125c.621 0 1.125-.504 1.125-1.125V9.75M8.25 21h8.25"/>
      </svg>
    ),
  },
  {
    id: 'analyzer',
    label: 'Analyzing sales techniques with Claude',
    sublabel: 'Evaluating rapport, negotiation & objection handling',
    progressStart: 40,
    progressEnd: 75,
    Icon: () => (
      <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>
      </svg>
    ),
  },
  {
    id: 'synthesis',
    label: 'Generating performance report',
    sublabel: 'Synthesising insights & personalised coaching recommendations',
    progressStart: 75,
    progressEnd: 100,
    Icon: () => (
      <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <path d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
      </svg>
    ),
  },
];

/** Map backend progress (0-100) → which step index is currently active. */
function progressToStep(progress) {
  for (let i = ANALYSIS_STEPS.length - 1; i >= 0; i--) {
    if (progress >= ANALYSIS_STEPS[i].progressStart) return i;
  }
  return 0;
}

/* ── Circular score ring ─────────────────────────────── */
function ScoreRing({ score }) {
  const [displayed, setDisplayed] = useState(0);
  const [ready, setReady]         = useState(false);
  const raf = useRef(null);

  useEffect(() => {
    const t = setTimeout(() => setReady(true), 200);
    return () => clearTimeout(t);
  }, []);

  useEffect(() => {
    if (!ready) return;
    let start = null;
    const duration = 1400;
    const animate = (ts) => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayed(Math.round(eased * score));
      if (progress < 1) raf.current = requestAnimationFrame(animate);
    };
    raf.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(raf.current);
  }, [ready, score]);

  const radius = 52;
  const circ   = 2 * Math.PI * radius;
  const offset = circ - (displayed / 100) * circ;
  const color  = score >= 80 ? '#10b981' : score >= 60 ? '#f59e0b' : '#ef4444';

  return (
    <div className="score-pop" style={{ display: 'inline-block' }}>
      <svg width="140" height="140" viewBox="0 0 140 140">
        <circle cx="70" cy="70" r={radius} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="9"/>
        <circle
          cx="70" cy="70" r={radius}
          fill="none"
          stroke={color}
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={ready ? offset : circ}
          transform="rotate(-90 70 70)"
          style={{ transition: 'stroke-dashoffset 1.4s cubic-bezier(0.4, 0, 0.2, 1)', filter: `drop-shadow(0 0 8px ${color}60)` }}
        />
        <text x="70" y="66" textAnchor="middle" fill="white" fontSize="30" fontWeight="700" fontFamily="Syne, sans-serif">
          {displayed}
        </text>
        <text x="70" y="86" textAnchor="middle" fill="rgba(148,163,184,0.5)" fontSize="11" fontFamily="DM Sans, sans-serif">
          out of 100
        </text>
      </svg>
    </div>
  );
}

/* ── Animated score bar ──────────────────────────────── */
function ScoreBar({ label, score, delay = 0 }) {
  const [width, setWidth] = useState(0);
  const color = score >= 80 ? '#10b981' : score >= 60 ? '#f59e0b' : '#ef4444';
  const textColor = score >= 80 ? '#34d399' : score >= 60 ? '#fbbf24' : '#f87171';

  useEffect(() => {
    const t = setTimeout(() => setWidth(score), 100 + delay);
    return () => clearTimeout(t);
  }, [score, delay]);

  return (
    <div>
      <div className="flex justify-between text-xs mb-2">
        <span style={{ color: 'rgba(148,163,184,0.7)' }} className="capitalize">
          {label.replace(/_/g, ' ')}
        </span>
        <span className="font-semibold" style={{ color: textColor }}>{score}%</span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: `${width}%`,
            background: color,
            boxShadow: `0 0 8px ${color}50`,
            transitionTimingFunction: 'cubic-bezier(0.4, 0, 0.2, 1)',
          }}
        />
      </div>
    </div>
  );
}

/* ── Emotion journey ──────────────────────────────────── */
const EMOTION_EMOJI = {
  curious: '🤔', interested: '😊', happy: '😄', satisfied: '😌',
  neutral: '😐', confused: '😕', frustrated: '😤', angry: '😠',
  skeptical: '🤨', excited: '🤩',
};

/* ── Analysis progress panel ─────────────────────────── */
function AnalysisPanel({ progress, stage }) {
  const activeStep = progressToStep(progress);

  return (
    <div
      className="rounded-2xl p-8 mb-6"
      style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}
    >
      {/* Header */}
      <div className="text-center mb-8">
        <div
          className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-4"
          style={{ background: 'rgba(37,99,235,0.12)', border: '1px solid rgba(37,99,235,0.2)' }}
        >
          <svg className="spin-ring" width="22" height="22" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="rgba(59,130,246,0.2)" strokeWidth="3"/>
            <path d="M12 2a10 10 0 0110 10" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round"/>
          </svg>
        </div>
        <h2 className="heading text-xl font-bold text-white mb-1">Analyzing Your Session</h2>
        <p className="text-sm" style={{ color: 'rgba(148,163,184,0.5)' }}>
          {stage || 'Initializing…'}
        </p>
      </div>

      {/* Steps */}
      <div className="space-y-2 mb-8">
        {ANALYSIS_STEPS.map((step, i) => {
          const isDone   = i < activeStep;
          const isActive = i === activeStep;

          return (
            <div
              key={step.id}
              className="flex items-center gap-4 px-4 py-3.5 rounded-xl transition-all duration-500"
              style={{
                background: isActive ? 'rgba(37,99,235,0.08)' : isDone ? 'rgba(16,185,129,0.04)' : 'transparent',
                border: isActive
                  ? '1px solid rgba(37,99,235,0.2)'
                  : isDone
                  ? '1px solid rgba(16,185,129,0.1)'
                  : '1px solid transparent',
                opacity: (!isDone && !isActive) ? 0.4 : 1,
              }}
            >
              {/* Icon bubble */}
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                style={{
                  background: isDone
                    ? 'rgba(16,185,129,0.15)'
                    : isActive
                    ? 'rgba(37,99,235,0.15)'
                    : 'rgba(255,255,255,0.04)',
                  color: isDone ? '#34d399' : isActive ? '#60a5fa' : 'rgba(148,163,184,0.3)',
                }}
              >
                {isDone ? (
                  <svg width="14" height="14" fill="none" stroke="#34d399" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                    <path d="M20 6L9 17l-5-5"/>
                  </svg>
                ) : isActive ? (
                  <svg className="spin-ring" width="14" height="14" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="rgba(59,130,246,0.2)" strokeWidth="3.5"/>
                    <path d="M12 2a10 10 0 0110 10" stroke="#60a5fa" strokeWidth="3.5" strokeLinecap="round"/>
                  </svg>
                ) : (
                  <step.Icon />
                )}
              </div>

              {/* Labels */}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium" style={{ color: isDone ? '#34d399' : isActive ? '#e2e8f0' : 'rgba(148,163,184,0.6)' }}>
                  {step.label}
                </p>
                <p className="text-xs mt-0.5" style={{ color: 'rgba(148,163,184,0.35)' }}>
                  {step.sublabel}
                </p>
              </div>

              {/* Status badge */}
              <div className="flex-shrink-0">
                {isDone && (
                  <span className="text-xs font-semibold px-2.5 py-1 rounded-lg"
                    style={{ background: 'rgba(16,185,129,0.1)', color: '#34d399', border: '1px solid rgba(16,185,129,0.2)' }}>
                    Done
                  </span>
                )}
                {isActive && (
                  <span className="text-xs font-semibold px-2.5 py-1 rounded-lg"
                    style={{ background: 'rgba(37,99,235,0.1)', color: '#60a5fa', border: '1px solid rgba(37,99,235,0.2)' }}>
                    Running
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Progress bar */}
      <div>
        <div className="flex justify-between text-xs mb-2" style={{ color: 'rgba(148,163,184,0.45)' }}>
          <span>Overall progress</span>
          <span className="font-medium" style={{ color: '#60a5fa' }}>{Math.round(progress)}%</span>
        </div>
        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{
              width: `${progress}%`,
              background: 'linear-gradient(90deg, #2563eb, #7c3aed)',
              boxShadow: '0 0 12px rgba(37,99,235,0.4)',
            }}
          />
        </div>
      </div>
    </div>
  );
}

/* ── Main Component ──────────────────────────────────── */
export default function EvaluationReport() {
  const { sessionId } = useParams();
  const navigate      = useNavigate();

  const [loading, setLoading]         = useState(true);
  const [session, setSession]         = useState(null);
  const [evaluation, setEvaluation]   = useState(null);
  const [quickStats, setQuickStats]   = useState(null);
  const [error, setError]             = useState(null);

  const [reportVisible, setReportVisible] = useState(false);
  const [reEvaluating, setReEvaluating]   = useState(false);

  const handleReEvaluate = async () => {
    setReEvaluating(true);
    try {
      await evaluationAPI.triggerEvaluation(sessionId, 'training', true);
      setReportVisible(false);
      setEvaluation({ status: 'pending', progress: 0, stage: null });
    } catch {
      // leave report visible on error
    } finally {
      setReEvaluating(false);
    }
  };

  const fetchEvaluation = useCallback(async () => {
    try {
      const sessionData = await sessionsAPI.getById(sessionId);
      setSession(sessionData);

      const report = await evaluationAPI.getReport(sessionId);
      // Trigger evaluation for not_started OR stale pending (task died before progressing)
      if (report.status === 'not_started' || report.status === 'pending') {
        await evaluationAPI.triggerEvaluation(sessionId);
        setEvaluation({ status: 'pending', progress: report.progress || 0, stage: report.stage || null });
      } else {
        setEvaluation(report);
        if (report.quick_stats) setQuickStats(report.quick_stats);
      }

      try {
        const stats = await evaluationAPI.getQuickStats(sessionId);
        setQuickStats(stats.stats);
      } catch {}
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load evaluation');
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => { fetchEvaluation(); }, [fetchEvaluation]);

  // Poll for completion
  useEffect(() => {
    if (!evaluation || evaluation.status === 'completed' || evaluation.status === 'failed') return;
    const interval = setInterval(async () => {
      try {
        const report = await evaluationAPI.getReport(sessionId);
        setEvaluation(report);
        if (report.quick_stats) setQuickStats(report.quick_stats);
        if (report.status === 'completed' || report.status === 'failed') clearInterval(interval);
      } catch {}
    }, 3000);
    return () => clearInterval(interval);
  }, [evaluation, sessionId]);

  // Reveal report once evaluation completes
  useEffect(() => {
    if (evaluation?.status !== 'completed') return;
    const t = setTimeout(() => setReportVisible(true), 600);
    return () => clearTimeout(t);
  }, [evaluation?.status]);

  const formatDuration = (s) => {
    if (!s) return '0:00';
    return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`;
  };

  const isProcessing  = evaluation?.status === 'pending' || evaluation?.status === 'processing';
  const isCompleted   = evaluation?.status === 'completed';
  const isFailed      = evaluation?.status === 'failed';
  const stepsProgress = evaluation?.progress || 0;

  /* ── Loading skeleton ─── */
  if (loading) {
    return (
      <Layout>
        <div className="min-h-screen flex items-center justify-center" style={{ background: '#030712' }}>
          <div className="text-center">
            <svg className="spin-ring mx-auto mb-4" width="32" height="32" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="rgba(59,130,246,0.15)" strokeWidth="3"/>
              <path d="M12 2a10 10 0 0110 10" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round"/>
            </svg>
            <p className="text-sm" style={{ color: 'rgba(148,163,184,0.45)' }}>Loading evaluation…</p>
          </div>
        </div>
      </Layout>
    );
  }

  /* ── Error ─── */
  if (error) {
    return (
      <Layout>
        <div className="min-h-screen flex items-center justify-center p-8" style={{ background: '#030712' }}>
          <div className="text-center max-w-sm">
            <div
              className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-5"
              style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)' }}
            >
              <svg width="24" height="24" fill="none" stroke="#f87171" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                <path d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"/>
              </svg>
            </div>
            <h2 className="heading font-bold text-white mb-2">Error Loading Report</h2>
            <p className="text-sm mb-6" style={{ color: 'rgba(148,163,184,0.5)' }}>{error}</p>
            <Link to="/dashboard" className="btn-primary inline-block px-6 py-2.5 rounded-xl text-sm font-semibold text-white">
              Back to Dashboard
            </Link>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div style={{ background: '#030712', minHeight: '100vh' }}>
        <div className="max-w-3xl mx-auto px-6 py-10">

          {/* Page Header */}
          <div className="flex items-start justify-between mb-8">
            <div>
              <h1 className="heading text-2xl font-bold text-white mb-1">Evaluation Report</h1>
              <p className="text-sm" style={{ color: 'rgba(148,163,184,0.5)' }}>
                {session?.persona_name || 'Training Session'} · {session?.difficulty || 'medium'} difficulty
              </p>
            </div>
            <div className="flex items-center gap-2">
              {isCompleted && (
                <button
                  onClick={handleReEvaluate}
                  disabled={reEvaluating}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 disabled:opacity-50"
                  style={{ background: 'rgba(245,158,11,0.08)', color: '#fbbf24', border: '1px solid rgba(245,158,11,0.18)' }}
                  onMouseEnter={e => { e.currentTarget.style.background = 'rgba(245,158,11,0.16)'; }}
                  onMouseLeave={e => { e.currentTarget.style.background = 'rgba(245,158,11,0.08)'; }}
                >
                  {reEvaluating ? (
                    <svg className="spin-ring" width="13" height="13" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="10" stroke="rgba(245,158,11,0.25)" strokeWidth="3"/>
                      <path d="M12 2a10 10 0 0110 10" stroke="#fbbf24" strokeWidth="3" strokeLinecap="round"/>
                    </svg>
                  ) : (
                    <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                      <path d="M1 4v6h6M23 20v-6h-6"/><path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15"/>
                    </svg>
                  )}
                  {reEvaluating ? 'Starting…' : 'Re-evaluate'}
                </button>
              )}
              <Link
                to="/dashboard"
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200"
                style={{ color: 'rgba(148,163,184,0.55)', border: '1px solid rgba(255,255,255,0.06)' }}
              >
                <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                  <path d="M19 12H5M12 19l-7-7 7-7"/>
                </svg>
                Dashboard
              </Link>
            </div>
          </div>

          {/* ── PROCESSING STATE ── */}
          {isProcessing && <AnalysisPanel progress={stepsProgress} stage={evaluation?.stage} />}

          {/* ── FAILED STATE ── */}
          {isFailed && (
            <div
              className="rounded-2xl p-8 mb-6 text-center"
              style={{ background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.15)' }}
            >
              <div
                className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-4"
                style={{ background: 'rgba(239,68,68,0.12)' }}
              >
                <svg width="22" height="22" fill="none" stroke="#f87171" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                  <path d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"/>
                </svg>
              </div>
              <h2 className="heading font-bold text-white mb-2">Evaluation Failed</h2>
              <p className="text-sm mb-6" style={{ color: 'rgba(239,68,68,0.7)' }}>
                {evaluation?.error || 'Something went wrong during evaluation.'}
              </p>
              <button
                onClick={() => { setEvaluation({ status: 'pending', progress: 0 }); evaluationAPI.triggerEvaluation(sessionId); }}
                className="btn-primary px-6 py-2.5 rounded-xl text-sm font-semibold text-white"
              >
                Retry Evaluation
              </button>
            </div>
          )}

          {/* ── QUICK STATS (always shown when available) ── */}
          {quickStats && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
              {[
                { label: 'Duration',        val: formatDuration(quickStats.duration_seconds) },
                { label: 'Total Turns',     val: quickStats.total_turns },
                { label: 'Your Responses',  val: quickStats.salesperson_turns },
                { label: 'Customer Turns',  val: quickStats.customer_turns },
              ].map(({ label, val }) => (
                <div
                  key={label}
                  className="rounded-xl p-4"
                  style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}
                >
                  <p className="text-xs mb-2" style={{ color: 'rgba(148,163,184,0.45)' }}>{label}</p>
                  <p className="heading text-xl font-bold text-white">{val}</p>
                </div>
              ))}
            </div>
          )}

          {/* ── EMOTION JOURNEY ── */}
          {quickStats?.emotion_journey?.length > 0 && (
            <div
              className="rounded-2xl p-5 mb-6"
              style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}
            >
              <h3 className="heading text-sm font-bold text-white mb-4">Customer Emotion Journey</h3>
              <div className="flex items-center gap-2 flex-wrap">
                {quickStats.emotion_journey.map((em, i) => (
                  <div key={i} className="flex items-center">
                    <div
                      className="px-3 py-2 rounded-xl text-center"
                      style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)' }}
                    >
                      <span className="text-xl block">{EMOTION_EMOJI[em] || '😐'}</span>
                      <span className="text-xs mt-1 block capitalize" style={{ color: 'rgba(148,163,184,0.5)' }}>{em}</span>
                    </div>
                    {i < quickStats.emotion_journey.length - 1 && (
                      <svg width="16" height="16" fill="none" stroke="rgba(148,163,184,0.2)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24" className="mx-1">
                        <path d="M5 12h14M12 5l7 7-7 7"/>
                      </svg>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── COMPLETED REPORT ── */}
          {isCompleted && evaluation?.report && reportVisible && (() => {
            const rep    = evaluation.report;
            const scores = rep.scores || {};
            const skills = scores.skills || [];
            const status = scores.status || (evaluation.overall_score >= (scores.pass_threshold || 75) ? 'passed' : 'failed');
            const isPassed = status === 'passed';

            return (
              <div className="slide-up">

                {/* ── Overall score card ── */}
                <div
                  className="rounded-2xl p-8 mb-5 flex flex-col md:flex-row items-center gap-8"
                  style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}
                >
                  <div className="flex-shrink-0">
                    <ScoreRing score={evaluation.overall_score} />
                  </div>
                  <div className="flex-1 text-center md:text-left">
                    <h2 className="heading text-xl font-bold text-white mb-3">Overall Performance</h2>
                    <div className="flex items-center gap-3 justify-center md:justify-start flex-wrap mb-4">
                      {isPassed ? (
                        <span className="px-3 py-1.5 rounded-xl text-sm font-semibold flex items-center gap-2"
                          style={{ background: 'rgba(16,185,129,0.1)', color: '#34d399', border: '1px solid rgba(16,185,129,0.2)' }}>
                          <svg width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24"><path d="M20 6L9 17l-5-5"/></svg>
                          Passed
                        </span>
                      ) : (
                        <span className="px-3 py-1.5 rounded-xl text-sm font-semibold"
                          style={{ background: 'rgba(239,68,68,0.1)', color: '#f87171', border: '1px solid rgba(239,68,68,0.2)' }}>
                          Needs Improvement
                        </span>
                      )}
                      <span className="text-sm" style={{ color: 'rgba(148,163,184,0.4)' }}>
                        Pass threshold: {scores.pass_threshold || 75}%
                      </span>
                    </div>
                    {rep.executive_summary && (
                      <p className="text-sm leading-relaxed" style={{ color: 'rgba(148,163,184,0.65)' }}>
                        {rep.executive_summary}
                      </p>
                    )}
                  </div>
                </div>

                {/* ── Skill score bars ── */}
                {skills.length > 0 && (
                  <div className="rounded-2xl p-6 mb-5"
                    style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}>
                    <h3 className="heading text-sm font-bold text-white mb-5">Skill Scores</h3>
                    <div className="space-y-4">
                      {skills.map((sk, i) => (
                        <ScoreBar key={sk.skill_key || i} label={sk.skill_name} score={sk.score} delay={i * 80} />
                      ))}
                    </div>
                  </div>
                )}

                {/* ── Per-skill detail ── */}
                {skills.filter(sk => (sk.strengths?.length || sk.areas_to_improve?.length) && sk.was_tested).length > 0 && (
                  <div className="rounded-2xl p-6 mb-5"
                    style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}>
                    <h3 className="heading text-sm font-bold text-white mb-5">Skill Breakdown</h3>
                    <div className="space-y-5">
                      {skills.filter(sk => sk.was_tested).map((sk, i) => {
                        const color = sk.score >= 80 ? '#34d399' : sk.score >= 60 ? '#fbbf24' : '#f87171';
                        const bg    = sk.score >= 80 ? 'rgba(16,185,129,0.06)' : sk.score >= 60 ? 'rgba(245,158,11,0.06)' : 'rgba(239,68,68,0.06)';
                        const border = sk.score >= 80 ? 'rgba(16,185,129,0.12)' : sk.score >= 60 ? 'rgba(245,158,11,0.12)' : 'rgba(239,68,68,0.12)';
                        return (
                          <div key={sk.skill_key || i} className="rounded-xl p-4" style={{ background: bg, border: `1px solid ${border}` }}>
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-sm font-semibold text-white">{sk.skill_name}</span>
                              <span className="text-sm font-bold" style={{ color }}>{sk.score}/100</span>
                            </div>
                            {sk.summary && (
                              <p className="text-xs mb-3" style={{ color: 'rgba(148,163,184,0.6)' }}>{sk.summary}</p>
                            )}
                            <div className="grid md:grid-cols-2 gap-3">
                              {sk.strengths?.length > 0 && (
                                <div>
                                  <p className="text-xs font-semibold mb-1.5" style={{ color: '#34d399' }}>What worked</p>
                                  <ul className="space-y-1">
                                    {sk.strengths.map((s, j) => (
                                      <li key={j} className="text-xs flex items-start gap-1.5" style={{ color: 'rgba(148,163,184,0.7)' }}>
                                        <svg className="flex-shrink-0 mt-0.5" width="10" height="10" fill="none" stroke="#34d399" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24"><path d="M20 6L9 17l-5-5"/></svg>
                                        {s}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                              {sk.areas_to_improve?.length > 0 && (
                                <div>
                                  <p className="text-xs font-semibold mb-1.5" style={{ color: '#fbbf24' }}>To improve</p>
                                  <ul className="space-y-1">
                                    {sk.areas_to_improve.map((s, j) => (
                                      <li key={j} className="text-xs flex items-start gap-1.5" style={{ color: 'rgba(148,163,184,0.7)' }}>
                                        <svg className="flex-shrink-0 mt-0.5" width="10" height="10" fill="none" stroke="#fbbf24" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                                        {s}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* ── Fact-check accuracy (against KB) ── */}
                {rep.fact_check && rep.fact_check.claims_checked > 0 && (() => {
                  const fc = rep.fact_check;
                  const pct = Math.round((fc.accuracy_rate || 0) * 100);
                  const tone = pct >= 80
                    ? { ring: '#34d399', bg: 'rgba(16,185,129,0.05)', border: 'rgba(16,185,129,0.18)', label: 'Accurate' }
                    : pct >= 50
                      ? { ring: '#fbbf24', bg: 'rgba(245,158,11,0.05)', border: 'rgba(245,158,11,0.18)', label: 'Mixed' }
                      : { ring: '#f87171', bg: 'rgba(239,68,68,0.05)', border: 'rgba(239,68,68,0.18)', label: 'Inaccurate' };
                  return (
                    <div className="rounded-2xl p-6 mb-5"
                      style={{ background: tone.bg, border: `1px solid ${tone.border}` }}>
                      <div className="flex items-start justify-between mb-4 gap-4 flex-wrap">
                        <div>
                          <h3 className="heading text-sm font-bold text-white mb-1">Factual Accuracy (vs. Knowledge Base)</h3>
                          <p className="text-xs" style={{ color: 'rgba(148,163,184,0.55)' }}>
                            Salesperson claims about prices, sizes, delivery, and features, verified against the property database.
                          </p>
                        </div>
                        <div className="flex items-center gap-3 flex-shrink-0">
                          <div className="text-right">
                            <p className="text-2xl font-bold" style={{ color: tone.ring }}>{pct}%</p>
                            <p className="text-xs" style={{ color: 'rgba(148,163,184,0.5)' }}>{tone.label}</p>
                          </div>
                          <div className="text-right text-xs leading-tight" style={{ color: 'rgba(148,163,184,0.6)' }}>
                            <p>{fc.accurate_count} accurate</p>
                            <p>{fc.inaccurate_count} wrong</p>
                            {fc.unverifiable_count > 0 && <p>{fc.unverifiable_count} unverifiable</p>}
                          </div>
                        </div>
                      </div>

                      {fc.properties_discussed?.length > 0 && (
                        <div className="mb-3 flex flex-wrap gap-1.5">
                          {fc.properties_discussed.map((name, i) => (
                            <span key={i} className="text-xs px-2 py-0.5 rounded-lg"
                              style={{ background: 'rgba(255,255,255,0.05)', color: 'rgba(148,163,184,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}>
                              {name}
                            </span>
                          ))}
                        </div>
                      )}

                      {fc.errors?.length > 0 ? (
                        <div className="space-y-2.5">
                          {fc.errors.map((err, i) => (
                            <div key={i} className="rounded-xl p-3.5"
                              style={{
                                background: 'rgba(0,0,0,0.25)',
                                border: `1px solid ${err.severity === 'critical' ? 'rgba(239,68,68,0.25)' : 'rgba(245,158,11,0.2)'}`,
                              }}>
                              <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                                <span className="text-xs px-2 py-0.5 rounded font-medium capitalize"
                                  style={{
                                    background: err.severity === 'critical' ? 'rgba(239,68,68,0.15)' : 'rgba(245,158,11,0.15)',
                                    color: err.severity === 'critical' ? '#f87171' : '#fbbf24',
                                  }}>
                                  {err.severity}
                                </span>
                                <span className="text-xs px-2 py-0.5 rounded font-medium"
                                  style={{ background: 'rgba(255,255,255,0.06)', color: 'rgba(148,163,184,0.7)' }}>
                                  {err.claim_type}
                                </span>
                                {err.turn_number != null && (
                                  <span className="text-xs" style={{ color: 'rgba(148,163,184,0.45)' }}>Turn {err.turn_number}</span>
                                )}
                                {err.property_name && (
                                  <span className="text-xs" style={{ color: 'rgba(148,163,184,0.5)' }}>· {err.property_name}</span>
                                )}
                              </div>
                              <p className="text-sm mb-1" style={{ color: '#f87171' }}>
                                <span className="opacity-60">Said:</span> "{err.claimed_value}"
                              </p>
                              <p className="text-sm mb-1" style={{ color: '#34d399' }}>
                                <span className="opacity-60">KB says:</span> {err.correct_value}
                              </p>
                              {err.explanation_ar && (
                                <p className="text-xs mt-2" style={{ color: 'rgba(148,163,184,0.65)' }} dir="rtl">
                                  {err.explanation_ar}
                                </p>
                              )}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-sm" style={{ color: '#34d399' }}>
                          ✓ All checked claims match the knowledge base.
                        </p>
                      )}
                    </div>
                  );
                })()}

                {/* ── Top strengths + improvements ── */}
                <div className="grid md:grid-cols-2 gap-4 mb-5">
                  {rep.top_strengths?.length > 0 && (
                    <div className="rounded-2xl p-6"
                      style={{ background: 'rgba(16,185,129,0.05)', border: '1px solid rgba(16,185,129,0.12)' }}>
                      <h3 className="heading text-sm font-bold mb-4" style={{ color: '#34d399' }}>Key Strengths</h3>
                      <ul className="space-y-2.5">
                        {rep.top_strengths.map((item, i) => (
                          <li key={i} className="flex items-start gap-2.5 text-sm" style={{ color: 'rgba(148,163,184,0.7)' }}>
                            <svg className="flex-shrink-0 mt-0.5" width="13" height="13" fill="none" stroke="#34d399" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24"><path d="M20 6L9 17l-5-5"/></svg>
                            {item}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {rep.top_improvements?.length > 0 && (
                    <div className="rounded-2xl p-6"
                      style={{ background: 'rgba(245,158,11,0.05)', border: '1px solid rgba(245,158,11,0.12)' }}>
                      <h3 className="heading text-sm font-bold mb-4" style={{ color: '#fbbf24' }}>Areas to Improve</h3>
                      <ul className="space-y-2.5">
                        {rep.top_improvements.map((item, i) => (
                          <li key={i} className="flex items-start gap-2.5 text-sm" style={{ color: 'rgba(148,163,184,0.7)' }}>
                            <svg className="flex-shrink-0 mt-0.5" width="13" height="13" fill="none" stroke="#fbbf24" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                            {item}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                {/* ── Turn-by-turn feedback ── */}
                {rep.turn_feedback?.length > 0 && (
                  <div className="rounded-2xl p-6 mb-5"
                    style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}>
                    <h3 className="heading text-sm font-bold text-white mb-4">Turn-by-Turn Feedback</h3>
                    <div className="space-y-3">
                      {rep.turn_feedback.filter(t => t.speaker === 'salesperson').map((t, i) => {
                        const aColor = { excellent: '#34d399', good: '#60a5fa', adequate: '#94a3b8', needs_improvement: '#fbbf24', poor: '#f87171' }[t.assessment] || '#94a3b8';
                        return (
                          <div key={i} className="rounded-xl p-4"
                            style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                            <div className="flex items-center gap-2 mb-2">
                              <span className="text-xs px-2 py-0.5 rounded-lg font-medium capitalize"
                                style={{ background: `${aColor}15`, color: aColor, border: `1px solid ${aColor}30` }}>
                                {t.assessment?.replace(/_/g, ' ') || 'neutral'}
                              </span>
                              <span className="text-xs" style={{ color: 'rgba(148,163,184,0.35)' }}>Turn {t.turn_number}</span>
                            </div>
                            <p className="text-xs italic mb-2" style={{ color: 'rgba(148,163,184,0.5)' }}>"{t.text}"</p>
                            {t.what_was_good && (
                              <p className="text-xs mb-1" style={{ color: '#34d399' }}>✓ {t.what_was_good}</p>
                            )}
                            {t.what_to_improve && (
                              <p className="text-xs mb-1" style={{ color: '#fbbf24' }}>→ {t.what_to_improve}</p>
                            )}
                            {t.suggested_alternative && (
                              <p className="text-xs mt-1 pl-3" style={{ color: 'rgba(148,163,184,0.55)', borderLeft: '2px solid rgba(37,99,235,0.4)' }}>
                                Try: "{t.suggested_alternative}"
                              </p>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* ── Recommended practice ── */}
                {rep.recommended_practice?.length > 0 && (
                  <div className="rounded-2xl p-6 mb-5"
                    style={{ background: 'rgba(37,99,235,0.05)', border: '1px solid rgba(37,99,235,0.12)' }}>
                    <h3 className="heading text-sm font-bold mb-4" style={{ color: '#60a5fa' }}>Recommended Practice</h3>
                    <ul className="space-y-2.5">
                      {rep.recommended_practice.map((item, i) => (
                        <li key={i} className="flex items-start gap-2.5 text-sm" style={{ color: 'rgba(148,163,184,0.7)' }}>
                          <span className="flex-shrink-0 w-5 h-5 rounded-lg flex items-center justify-center text-xs font-bold mt-0.5"
                            style={{ background: 'rgba(37,99,235,0.15)', color: '#60a5fa' }}>{i + 1}</span>
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

              </div>
            );
          })()}

          {/* Action buttons */}
          <div className="flex flex-col sm:flex-row gap-3 justify-center pt-2">
            <Link
              to="/dashboard"
              className="px-6 py-3 rounded-xl text-sm font-semibold text-center transition-all duration-200"
              style={{
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.08)',
                color: 'rgba(148,163,184,0.8)',
              }}
            >
              Back to Dashboard
            </Link>
            {isCompleted && (
              <button
                onClick={handleReEvaluate}
                disabled={reEvaluating}
                className="px-6 py-3 rounded-xl text-sm font-semibold text-center transition-all duration-200 disabled:opacity-50 flex items-center justify-center gap-2"
                style={{ background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.2)', color: '#fbbf24' }}
              >
                {reEvaluating ? (
                  <svg className="spin-ring" width="13" height="13" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="rgba(245,158,11,0.25)" strokeWidth="3"/>
                    <path d="M12 2a10 10 0 0110 10" stroke="#fbbf24" strokeWidth="3" strokeLinecap="round"/>
                  </svg>
                ) : (
                  <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                    <path d="M1 4v6h6M23 20v-6h-6"/><path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15"/>
                  </svg>
                )}
                {reEvaluating ? 'Starting…' : 'Re-evaluate'}
              </button>
            )}
            <Link
              to="/setup"
              className="btn-primary px-6 py-3 rounded-xl text-sm font-semibold text-white text-center flex items-center justify-center gap-2"
            >
              Start New Session
              <svg width="14" height="14" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                <path d="M5 12h14M12 5l7 7-7 7"/>
              </svg>
            </Link>
          </div>
        </div>
      </div>
    </Layout>
  );
}
