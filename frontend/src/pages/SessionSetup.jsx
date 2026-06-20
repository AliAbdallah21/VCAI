import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { personasAPI, sessionsAPI, learningAPI } from '../services/api';
import Layout from '../components/Layout';
import Avatar from '../components/Avatar';

const DIFFICULTIES = [
  {
    id: 'easy',
    label: 'Easy',
    desc: 'Receptive customer, straightforward objections',
    color: '#a5d6a7',
    bg: 'rgba(165,214,167,0.08)',
    border: 'rgba(165,214,167,0.2)',
    borderActive: 'rgba(165,214,167,0.5)',
    Icon: () => (
      <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"/>
        <path d="M8 13s1.5 2 4 2 4-2 4-2M9 9h.01M15 9h.01"/>
      </svg>
    ),
  },
  {
    id: 'medium',
    label: 'Medium',
    desc: 'Skeptical customer, price-conscious',
    color: '#e9c46a',
    bg: 'rgba(233,196,106,0.08)',
    border: 'rgba(233,196,106,0.2)',
    borderActive: 'rgba(233,196,106,0.5)',
    Icon: () => (
      <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"/>
        <path d="M8 15h8M9 9h.01M15 9h.01"/>
      </svg>
    ),
  },
  {
    id: 'hard',
    label: 'Hard',
    desc: 'Aggressive, demanding, difficult to close',
    color: '#ffb4ab',
    bg: 'rgba(255,180,171,0.08)',
    border: 'rgba(255,180,171,0.2)',
    borderActive: 'rgba(255,180,171,0.5)',
    Icon: () => (
      <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"/>
        <path d="M8 15s1.5-2 4-2 4 2 4 2M9 9h.01M15 9h.01"/>
      </svg>
    ),
  },
];

const StepBadge = ({ n, active }) => (
  <div
    className="w-7 h-7 rounded-lg flex items-center justify-center text-xs font-bold flex-shrink-0"
    style={active
      ? { background: 'linear-gradient(135deg, #b472f1, #deb7ff)', color: '#4a007f' }
      : { background: 'var(--bg-card-alt)', color: 'var(--text-muted)' }
    }
  >
    {n}
  </div>
);

// Buyer contexts for the "custom" scenario mode (4 fixed values, mirrors
// shared/scenarios.py BUYER_CONTEXTS).
const BUYER_CONTEXTS = [
  { id: 'first_time',     label: 'First-time buyer' },
  { id: 'investor',       label: 'Property investor' },
  { id: 'family_upgrade', label: 'Growing family' },
  { id: 'downsizer',      label: 'Downsizer' },
];
const SCENARIO_TIMELINES = [
  { id: 'urgent',    label: 'Urgent (~1 month)' },
  { id: 'flexible',  label: 'Flexible (3–6 months)' },
  { id: 'exploring', label: 'Just exploring' },
];

export default function SessionSetup() {
  const navigate  = useNavigate();
  const [personas, setPersonas] = useState([]);
  const [selected, setSelected] = useState(null);
  const [difficulty, setDifficulty] = useState('medium');
  const [genderFilter, setGenderFilter] = useState('all'); // all | male | female
  const [loading, setLoading]   = useState(true);
  const [creating, setCreating] = useState(false);

  // Scenario picker — 3 modes: random | preset | custom
  const [scenarioMode, setScenarioMode]   = useState('random');
  const [presets, setPresets]             = useState([]);
  const [selectedPreset, setSelectedPreset] = useState('');
  const [customContext, setCustomContext]   = useState('');
  const [customTimeline, setCustomTimeline] = useState('');

  // Adaptive learning recommendation
  const [learningProfile, setLearningProfile] = useState(null);
  const [recommendationDismissed, setRecommendationDismissed] = useState(false);
  const [trainingFocus, setTrainingFocus] = useState(null);

  useEffect(() => {
    personasAPI.getAll()
      .then(data => setPersonas(data.personas))
      .finally(() => setLoading(false));
    sessionsAPI.getScenarioPresets()
      .then(data => setPresets(data.presets || []))
      .catch(() => setPresets([]));
    learningAPI.getProfile()
      .then(profile => setLearningProfile(profile))
      .catch(() => {}); // non-critical — silently ignore if no history yet
  }, []);

  // Build the scenario spec the backend expects from the current picker state.
  const buildScenarioSpec = () => {
    if (scenarioMode === 'preset' && selectedPreset) {
      return { mode: 'preset', preset_id: selectedPreset };
    }
    if (scenarioMode === 'custom') {
      const pins = {};
      if (customContext)  pins.buyer_context = customContext;
      if (customTimeline) pins.timeline      = customTimeline;
      return { mode: 'custom', pins };
    }
    return { mode: 'random' };
  };

  const applyRecommendation = (allPersonas) => {
    const rec = learningProfile?.recommendation;
    if (!rec) return;
    setDifficulty(rec.recommended_difficulty);
    setTrainingFocus(rec.focus_skill);
    // Pre-select the recommended persona if it exists in the list
    const match = allPersonas.find(p => p.id === rec.recommended_persona_id);
    if (match) setSelected(match);
    setRecommendationDismissed(false);
  };

  const handleStart = async () => {
    if (!selected) return;
    setCreating(true);
    try {
      const session = await sessionsAPI.create(
        selected.id,
        difficulty,
        buildScenarioSpec(),
        trainingFocus,
      );
      navigate(`/session/${session.id}`);
    } catch {
      alert('Failed to create session');
      setCreating(false);
    }
  };

  // Difficulty is decoupled from persona — any persona is playable at any
  // difficulty. Difficulty controls how much friction the customer puts up.
  // Gender filter is purely cosmetic — filter the list client-side.
  const filtered = genderFilter === 'all'
    ? personas
    : personas.filter(p => p.gender === genderFilter);

  // Scenario presets filtered to match the selected persona's gender (or all
  // if no persona is selected yet). Presets tagged "any" always appear.
  const filteredPresets = selected
    ? presets.filter(p => !p.gender_fit || p.gender_fit === 'any' || p.gender_fit === selected.gender)
    : presets;

  const activeDiff = DIFFICULTIES.find(d => d.id === difficulty);

  return (
    <Layout>
      <div className="p-4 md:p-8 max-w-3xl mx-auto">
        <div className="mb-8 slide-up">
          <h1 className="heading text-2xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>New Training Session</h1>
          <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
            Choose your difficulty level and customer persona
          </p>
        </div>

        {/* Adaptive Recommendation Card */}
        {learningProfile?.has_enough_data && learningProfile?.recommendation && !recommendationDismissed && (
          <div
            className="rounded-2xl p-5 mb-5 slide-up"
            style={{
              background: 'linear-gradient(135deg, rgba(180,114,241,0.12), rgba(222,183,255,0.08))',
              border: '1px solid rgba(180,114,241,0.3)',
            }}
          >
            <div className="flex items-start justify-between gap-3 mb-3">
              <div className="flex items-center gap-2">
                <div
                  className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
                  style={{ background: 'rgba(180,114,241,0.2)' }}
                >
                  <svg width="14" height="14" fill="none" stroke="var(--primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                    <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
                  </svg>
                </div>
                <span className="text-xs font-bold tracking-wide" style={{ color: 'var(--text-primary)' }}>Recommended for You</span>
              </div>
              <button
                onClick={() => setRecommendationDismissed(true)}
                className="text-xs px-2 py-1 rounded-lg transition-all"
                style={{ color: 'var(--text-muted)', background: 'var(--bg-card-alt)' }}
              >
                Dismiss
              </button>
            </div>

            <p className="text-xs mb-3 leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
              {learningProfile.recommendation.reason}
            </p>

            <div className="flex flex-wrap gap-2 mb-4">
              <span
                className="text-xs px-2.5 py-1 rounded-lg font-medium"
                style={{ background: 'rgba(180,114,241,0.15)', color: 'var(--primary)', border: '1px solid rgba(180,114,241,0.25)' }}
              >
                Focus: {learningProfile.recommendation.focus_skill_name_ar}
              </span>
              <span
                className="text-xs px-2.5 py-1 rounded-lg font-medium capitalize"
                style={{ background: 'rgba(180,114,241,0.12)', color: 'var(--primary)', border: '1px solid rgba(180,114,241,0.2)' }}
              >
                {learningProfile.recommendation.recommended_difficulty} difficulty
              </span>
              {learningProfile.recommendation.scenario_hint && (
                <span
                  className="text-xs px-2.5 py-1 rounded-lg"
                  style={{ background: 'var(--bg-card-alt)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
                >
                  {learningProfile.recommendation.scenario_hint}
                </span>
              )}
            </div>

            <button
              onClick={() => applyRecommendation(personas)}
              className="w-full py-2.5 rounded-xl text-sm font-semibold transition-all duration-200"
              style={{
                background: 'linear-gradient(135deg, rgba(180,114,241,0.3), rgba(222,183,255,0.2))',
                border: '1px solid rgba(180,114,241,0.4)',
                color: 'var(--text-primary)',
              }}
            >
              Apply Recommendation
            </button>

            {trainingFocus && (
              <p className="text-xs mt-2 text-center" style={{ color: 'rgba(165,214,167,0.8)' }}>
                ✓ Recommendation applied — persona and difficulty pre-selected
              </p>
            )}
          </div>
        )}

        {/* Step 1 — Difficulty */}
        <div
          className="rounded-2xl p-6 mb-5"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
        >
          <div className="flex items-center gap-3 mb-5">
            <StepBadge n="1" active />
            <h2 className="heading font-bold text-sm tracking-wide" style={{ color: 'var(--text-primary)' }}>Select Difficulty</h2>
          </div>
          <div className="grid grid-cols-3 gap-3">
            {DIFFICULTIES.map(d => {
              const active = difficulty === d.id;
              return (
                <button
                  key={d.id}
                  onClick={() => setDifficulty(d.id)}
                  className="p-4 rounded-xl text-left transition-all duration-200"
                  style={{
                    background: active ? d.bg : 'var(--bg-card-alt)',
                    border: `1px solid ${active ? d.borderActive : 'var(--border)'}`,
                  }}
                >
                  <div style={{ color: active ? d.color : 'var(--text-muted)' }} className="mb-3">
                    <d.Icon />
                  </div>
                  <p className="font-bold text-sm mb-1" style={{ color: 'var(--text-primary)' }}>{d.label}</p>
                  <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>{d.desc}</p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Step 2 — Persona */}
        <div
          className="rounded-2xl p-6 mb-5"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
        >
          <div className="flex items-center gap-3 mb-4">
            <StepBadge n="2" active={!!selected === false} />
            <h2 className="heading font-bold text-sm tracking-wide" style={{ color: 'var(--text-primary)' }}>Choose Customer</h2>
            {activeDiff && (
              <span
                className="ml-auto text-xs font-semibold px-2.5 py-1 rounded-lg"
                style={{ background: activeDiff.bg, color: activeDiff.color, border: `1px solid ${activeDiff.border}` }}
              >
                {activeDiff.label}
              </span>
            )}
          </div>

          {/* Gender filter tabs */}
          <div className="flex gap-2 mb-4">
            {[
              { id: 'all',    label: 'All' },
              { id: 'male',   label: '♂ Male' },
              { id: 'female', label: '♀ Female' },
            ].map(g => {
              const active = genderFilter === g.id;
              return (
                <button
                  key={g.id}
                  onClick={() => { setGenderFilter(g.id); setSelected(null); }}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-150"
                  style={{
                    background: active ? 'var(--primary-soft-hover)' : 'var(--bg-card-alt)',
                    border: `1px solid ${active ? 'var(--primary)' : 'var(--border)'}`,
                    color: active ? 'var(--primary)' : 'var(--text-muted)',
                  }}
                >
                  {g.label}
                </button>
              );
            })}
          </div>

          {loading ? (
            <div className="py-10 text-center">
              <div
                className="w-5 h-5 spin-ring mx-auto"
                style={{ border: '2px solid var(--border)', borderTopColor: 'var(--primary)', borderRadius: '50%' }}
              />
            </div>
          ) : filtered.length === 0 ? (
            <div className="py-10 text-center text-sm" style={{ color: 'var(--text-muted)' }}>
              No personas for this difficulty level
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {filtered.map(p => {
                const active = selected?.id === p.id;
                return (
                  <button
                    key={p.id}
                    onClick={() => setSelected(p)}
                    className="p-4 rounded-xl text-left transition-all duration-200"
                    style={{
                      background: active ? 'var(--primary-soft)' : 'var(--bg-card-alt)',
                      border: `1px solid ${active ? 'var(--primary)' : 'var(--border)'}`,
                    }}
                  >
                    <div className="flex items-start gap-3">
                      <Avatar src={p.avatar_url} name={p.name_en} size={40} active={active} />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-1.5 mb-1">
                          <p className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>{p.name_en}</p>
                          <span
                            className="text-xs px-1.5 py-0.5 rounded"
                            style={{
                              background: p.gender === 'female' ? 'rgba(236,72,153,0.12)' : 'rgba(180,114,241,0.12)',
                              color: p.gender === 'female' ? '#f472b6' : 'var(--primary)',
                              border: `1px solid ${p.gender === 'female' ? 'rgba(236,72,153,0.2)' : 'rgba(180,114,241,0.2)'}`,
                            }}
                          >
                            {p.gender === 'female' ? '♀' : '♂'}
                          </span>
                        </div>
                        <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                          {p.description_en}
                        </p>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Step 3 — Scenario */}
        <div
          className="rounded-2xl p-6 mb-5"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
        >
          <div className="flex items-center gap-3 mb-2">
            <StepBadge n="3" active />
            <h2 className="heading font-bold text-sm tracking-wide" style={{ color: 'var(--text-primary)' }}>Buyer Scenario</h2>
          </div>
          <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
            The customer's situation — budget, timeline, and what they're looking for.
          </p>

          {/* Mode toggle */}
          <div className="grid grid-cols-3 gap-2 mb-4">
            {[
              { id: 'random', label: 'Surprise me', desc: 'Fully random' },
              { id: 'preset', label: 'Pick a preset', desc: 'Curated' },
              { id: 'custom', label: 'Customize', desc: 'Pin some, rest random' },
            ].map(m => {
              const active = scenarioMode === m.id;
              return (
                <button
                  key={m.id}
                  onClick={() => setScenarioMode(m.id)}
                  className="p-3 rounded-xl text-left transition-all duration-200"
                  style={{
                    background: active ? 'var(--primary-soft)' : 'var(--bg-card-alt)',
                    border: `1px solid ${active ? 'var(--primary)' : 'var(--border)'}`,
                  }}
                >
                  <p className="font-semibold text-xs mb-0.5" style={{ color: 'var(--text-primary)' }}>{m.label}</p>
                  <p className="text-xs leading-tight" style={{ color: 'var(--text-muted)' }}>{m.desc}</p>
                </button>
              );
            })}
          </div>

          {/* Mode-specific content */}
          {scenarioMode === 'random' && (
            <p className="text-xs px-3 py-2.5 rounded-xl"
              style={{ background: 'var(--bg-card-alt)', color: 'var(--text-secondary)' }}>
              A coherent buyer scenario will be generated for this session.
            </p>
          )}

          {scenarioMode === 'preset' && (
            <>
              <select
                value={selectedPreset}
                onChange={e => setSelectedPreset(e.target.value)}
                className="w-full px-3 py-2.5 rounded-xl text-sm"
                style={{ background: 'var(--bg-card-alt)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              >
                <option value="">Select a scenario…</option>
                {filteredPresets.map(p => (
                  <option key={p.preset_id} value={p.preset_id}>{p.label}</option>
                ))}
              </select>
              {selected && filteredPresets.length < presets.length && (
                <p className="text-xs mt-1.5" style={{ color: 'var(--text-muted)' }}>
                  Showing {filteredPresets.length} of {presets.length} scenarios — filtered for {selected.gender} personas.
                </p>
              )}
            </>
          )}

          {scenarioMode === 'custom' && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium mb-1.5 block" style={{ color: 'var(--text-muted)' }}>
                  Buyer type
                </label>
                <select
                  value={customContext}
                  onChange={e => setCustomContext(e.target.value)}
                  className="w-full px-3 py-2.5 rounded-xl text-sm"
                  style={{ background: 'var(--bg-card-alt)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                >
                  <option value="">Any (random)</option>
                  {BUYER_CONTEXTS.map(c => <option key={c.id} value={c.id}>{c.label}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs font-medium mb-1.5 block" style={{ color: 'var(--text-muted)' }}>
                  Timeline
                </label>
                <select
                  value={customTimeline}
                  onChange={e => setCustomTimeline(e.target.value)}
                  className="w-full px-3 py-2.5 rounded-xl text-sm"
                  style={{ background: 'var(--bg-card-alt)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                >
                  <option value="">Any (random)</option>
                  {SCENARIO_TIMELINES.map(t => <option key={t.id} value={t.id}>{t.label}</option>)}
                </select>
              </div>
              <p className="text-xs sm:col-span-2" style={{ color: 'var(--text-muted)' }}>
                Budget and must-haves are drawn to match the buyer type.
              </p>
            </div>
          )}
        </div>

        {/* Selected preview */}
        {selected && (
          <div
            className="rounded-2xl p-4 mb-5 flex items-center gap-4"
            style={{ background: 'rgba(180,114,241,0.07)', border: '1px solid rgba(180,114,241,0.2)' }}
          >
            <Avatar src={selected.avatar_url} name={selected.name_en} size={40} active />
            <div>
              <p className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>{selected.name_en}</p>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {activeDiff?.label} difficulty · Ready to start
              </p>
            </div>
            <div className="ml-auto">
              <svg width="16" height="16" fill="none" stroke="#a5d6a7" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                <path d="M20 6L9 17l-5-5"/>
              </svg>
            </div>
          </div>
        )}

        {/* Start Button */}
        <button
          onClick={handleStart}
          disabled={!selected || creating}
          className="btn-primary w-full py-4 rounded-xl font-semibold flex items-center justify-center gap-3"
        >
          {creating ? (
            <>
              <svg className="w-5 h-5 spin-ring" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="rgba(74,0,127,0.2)" strokeWidth="3"/>
                <path d="M12 2a10 10 0 0110 10" stroke="#4a007f" strokeWidth="3" strokeLinecap="round"/>
              </svg>
              Creating Session…
            </>
          ) : (
            <>
              <svg width="18" height="18" fill="none" stroke="#4a007f" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                <path d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
              </svg>
              Start Training Session
            </>
          )}
        </button>
      </div>
    </Layout>
  );
}
