import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Layout from '../components/Layout';
import SkillProgressCard from '../components/progress/SkillProgressCard';
import SkillTrendChart from '../components/progress/SkillTrendChart';
import FocusTimeline from '../components/progress/FocusTimeline';
import InsightsPanel from '../components/progress/InsightsPanel';
import { learningAPI } from '../services/api';

const SKILL_CONFIG = [
  { key: 'communication',      label: 'Communication',      color: '#3b82f6' },
  { key: 'product_knowledge',  label: 'Product Knowledge',  color: '#f59e0b' },
  { key: 'objection_handling', label: 'Objection Handling', color: '#ef4444' },
  { key: 'rapport',            label: 'Rapport',            color: '#10b981' },
  { key: 'closing',            label: 'Closing',            color: '#8b5cf6' },
];

const PERIOD_OPTIONS = [
  { id: 'session', label: 'Per Session' },
  { id: 'week',    label: 'Weekly' },
  { id: 'month',   label: 'Monthly' },
  { id: 'year',    label: 'Yearly' },
];

function Spinner() {
  return (
    <div className="flex items-center justify-center py-24">
      <div
        className="w-6 h-6"
        style={{
          border: '2px solid rgba(255,255,255,0.08)',
          borderTopColor: '#3b82f6',
          borderRadius: '50%',
          animation: 'spin 0.8s linear infinite',
        }}
      />
    </div>
  );
}

function EmptyState({ navigate }) {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <div
        className="w-14 h-14 rounded-2xl flex items-center justify-center mb-5"
        style={{ background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.2)' }}
      >
        <svg width="24" height="24" fill="none" stroke="#60a5fa" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
          <path d="M3 3v18h18M7 16l4-4 4 4 4-5"/>
        </svg>
      </div>
      <h3 className="font-bold text-white mb-2">No data yet</h3>
      <p className="text-sm mb-6" style={{ color: 'rgba(148,163,184,0.5)' }}>
        Complete your first training session to start tracking progress.
      </p>
      <button
        onClick={() => navigate('/setup')}
        className="btn-primary px-5 py-2.5 rounded-xl text-sm font-semibold text-white"
      >
        Start a Session
      </button>
    </div>
  );
}

export default function ProgressPage() {
  const navigate = useNavigate();
  const [groupBy, setGroupBy] = useState('month');
  const [progressData, setProgressData] = useState(null);
  const [insights, setInsights] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedSkill, setSelectedSkill] = useState(null);

  // Re-fetch when grouping changes
  useEffect(() => {
    setLoading(true);
    learningAPI.getProgress(groupBy)
      .then(setProgressData)
      .catch(() => setProgressData(null))
      .finally(() => setLoading(false));
  }, [groupBy]);

  // Insights fetched once
  useEffect(() => {
    learningAPI.getInsights()
      .then(data => setInsights(data.insights ?? []))
      .catch(() => {});
  }, []);

  const summaryBySkill = Object.fromEntries(
    (progressData?.skill_summaries ?? []).map(s => [s.skill_key, s])
  );

  const toggleSkill = (key) =>
    setSelectedSkill(prev => (prev === key ? null : key));

  return (
    <Layout>
      <div className="p-4 md:p-8 max-w-5xl mx-auto">

        {/* Header */}
        <div className="mb-8 slide-up">
          <h1 className="heading text-2xl font-bold text-white mb-1">Progress</h1>
          <p className="text-sm" style={{ color: 'rgba(148,163,184,0.55)' }}>
            Track how each skill has evolved across your training sessions
          </p>
        </div>

        {loading ? <Spinner /> : !progressData || progressData.total_sessions === 0 ? (
          <EmptyState navigate={navigate} />
        ) : (
          <>
            {/* Coaching insights */}
            <InsightsPanel insights={insights} />

            {/* Period selector */}
            <div className="flex gap-2 mb-5">
              {PERIOD_OPTIONS.map(opt => {
                const active = groupBy === opt.id;
                return (
                  <button
                    key={opt.id}
                    onClick={() => setGroupBy(opt.id)}
                    className="px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-150"
                    style={{
                      background: active ? 'rgba(59,130,246,0.15)' : 'rgba(255,255,255,0.03)',
                      border: `1px solid ${active ? 'rgba(59,130,246,0.45)' : 'rgba(255,255,255,0.07)'}`,
                      color: active ? '#60a5fa' : 'rgba(148,163,184,0.5)',
                    }}
                  >
                    {opt.label}
                  </button>
                );
              })}
              <span className="ml-auto text-xs self-center" style={{ color: 'rgba(148,163,184,0.35)' }}>
                {progressData.total_sessions} session{progressData.total_sessions !== 1 ? 's' : ''}
              </span>
            </div>

            {/* Skill cards grid — click to highlight in chart */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3 mb-5">
              {SKILL_CONFIG.map(s => {
                const summary = summaryBySkill[s.key];
                return (
                  <SkillProgressCard
                    key={s.key}
                    label={s.label}
                    color={s.color}
                    currentScore={summary?.current_score ?? null}
                    firstScore={summary?.first_score ?? null}
                    trend={summary?.trend ?? 'insufficient_data'}
                    totalImprovement={summary?.total_improvement ?? null}
                    focusSessions={summary?.focus_sessions ?? 0}
                    isSelected={selectedSkill === s.key}
                    onClick={() => toggleSkill(s.key)}
                  />
                );
              })}
            </div>

            {/* Trend chart */}
            <div
              className="rounded-2xl p-5 mb-5"
              style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-white">
                  {selectedSkill
                    ? `${SKILL_CONFIG.find(s => s.key === selectedSkill)?.label} over time`
                    : 'All skills over time'}
                </h3>
                {selectedSkill && (
                  <button
                    onClick={() => setSelectedSkill(null)}
                    className="text-xs px-2 py-1 rounded-lg"
                    style={{ color: 'rgba(148,163,184,0.45)', background: 'rgba(255,255,255,0.03)' }}
                  >
                    Show all
                  </button>
                )}
              </div>

              {progressData.periods.length < 2 ? (
                <p className="text-xs py-8 text-center" style={{ color: 'rgba(148,163,184,0.4)' }}>
                  Complete at least 2 sessions to see a trend line.
                </p>
              ) : (
                <SkillTrendChart
                  periods={progressData.periods}
                  selectedSkill={selectedSkill}
                  skillConfig={SKILL_CONFIG}
                />
              )}
            </div>

            {/* Focus timeline — only meaningful per-session */}
            {groupBy === 'session' && progressData.periods.length > 0 && (
              <div
                className="rounded-2xl p-5"
                style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}
              >
                <FocusTimeline
                  periods={progressData.periods}
                  skillConfig={SKILL_CONFIG}
                />
              </div>
            )}
          </>
        )}
      </div>
    </Layout>
  );
}
