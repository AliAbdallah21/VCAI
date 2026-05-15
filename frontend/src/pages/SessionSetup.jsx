import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { personasAPI, sessionsAPI } from '../services/api';
import Layout from '../components/Layout';
import Avatar from '../components/Avatar';

const DIFFICULTIES = [
  {
    id: 'easy',
    label: 'Easy',
    desc: 'Receptive customer, straightforward objections',
    color: '#10b981',
    bg: 'rgba(16,185,129,0.08)',
    border: 'rgba(16,185,129,0.2)',
    borderActive: 'rgba(16,185,129,0.5)',
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
    color: '#f59e0b',
    bg: 'rgba(245,158,11,0.08)',
    border: 'rgba(245,158,11,0.2)',
    borderActive: 'rgba(245,158,11,0.5)',
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
    color: '#ef4444',
    bg: 'rgba(239,68,68,0.08)',
    border: 'rgba(239,68,68,0.2)',
    borderActive: 'rgba(239,68,68,0.5)',
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
      ? { background: 'linear-gradient(135deg, #2563eb, #7c3aed)', color: '#fff' }
      : { background: 'rgba(255,255,255,0.06)', color: 'rgba(148,163,184,0.5)' }
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
  const [loading, setLoading]   = useState(true);
  const [creating, setCreating] = useState(false);

  // Scenario picker — 3 modes: random | preset | custom
  const [scenarioMode, setScenarioMode]   = useState('random');
  const [presets, setPresets]             = useState([]);
  const [selectedPreset, setSelectedPreset] = useState('');
  const [customContext, setCustomContext]   = useState('');
  const [customTimeline, setCustomTimeline] = useState('');

  useEffect(() => {
    personasAPI.getAll()
      .then(data => setPersonas(data.personas))
      .finally(() => setLoading(false));
    sessionsAPI.getScenarioPresets()
      .then(data => setPresets(data.presets || []))
      .catch(() => setPresets([]));
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

  const handleStart = async () => {
    if (!selected) return;
    setCreating(true);
    try {
      const session = await sessionsAPI.create(selected.id, difficulty, buildScenarioSpec());
      navigate(`/session/${session.id}`);
    } catch {
      alert('Failed to create session');
      setCreating(false);
    }
  };

  // Difficulty is decoupled from persona — any persona is playable at any
  // difficulty, so we show the full persona list regardless of the selected
  // difficulty. Difficulty controls how much friction the customer puts up.
  const filtered = personas;
  const activeDiff = DIFFICULTIES.find(d => d.id === difficulty);

  return (
    <Layout>
      <div className="p-4 md:p-8 max-w-3xl mx-auto">
        <div className="mb-8 slide-up">
          <h1 className="heading text-2xl font-bold text-white mb-1">New Training Session</h1>
          <p className="text-sm" style={{ color: 'rgba(148,163,184,0.55)' }}>
            Choose your difficulty level and customer persona
          </p>
        </div>

        {/* Step 1 — Difficulty */}
        <div
          className="rounded-2xl p-6 mb-5"
          style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}
        >
          <div className="flex items-center gap-3 mb-5">
            <StepBadge n="1" active />
            <h2 className="heading font-bold text-white text-sm tracking-wide">Select Difficulty</h2>
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
                    background: active ? d.bg : 'rgba(255,255,255,0.02)',
                    border: `1px solid ${active ? d.borderActive : 'rgba(255,255,255,0.06)'}`,
                  }}
                >
                  <div style={{ color: active ? d.color : 'rgba(148,163,184,0.4)' }} className="mb-3">
                    <d.Icon />
                  </div>
                  <p className="font-bold text-sm text-white mb-1">{d.label}</p>
                  <p className="text-xs leading-relaxed" style={{ color: 'rgba(148,163,184,0.5)' }}>{d.desc}</p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Step 2 — Persona */}
        <div
          className="rounded-2xl p-6 mb-5"
          style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}
        >
          <div className="flex items-center gap-3 mb-5">
            <StepBadge n="2" active={!!selected === false} />
            <h2 className="heading font-bold text-white text-sm tracking-wide">Choose Customer</h2>
            {activeDiff && (
              <span
                className="ml-auto text-xs font-semibold px-2.5 py-1 rounded-lg"
                style={{ background: activeDiff.bg, color: activeDiff.color, border: `1px solid ${activeDiff.border}` }}
              >
                {activeDiff.label}
              </span>
            )}
          </div>

          {loading ? (
            <div className="py-10 text-center">
              <div
                className="w-5 h-5 spin-ring mx-auto"
                style={{ border: '2px solid rgba(255,255,255,0.08)', borderTopColor: '#3b82f6', borderRadius: '50%' }}
              />
            </div>
          ) : filtered.length === 0 ? (
            <div className="py-10 text-center text-sm" style={{ color: 'rgba(148,163,184,0.4)' }}>
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
                      background: active ? 'rgba(37,99,235,0.1)' : 'rgba(255,255,255,0.02)',
                      border: `1px solid ${active ? 'rgba(37,99,235,0.35)' : 'rgba(255,255,255,0.06)'}`,
                    }}
                  >
                    <div className="flex items-start gap-3">
                      <Avatar src={p.avatar_url} name={p.name_en} size={40} active={active} />
                      <div className="min-w-0">
                        <p className="font-semibold text-sm text-white mb-1">{p.name_en}</p>
                        <p className="text-xs leading-relaxed" style={{ color: 'rgba(148,163,184,0.5)' }}>
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
          style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}
        >
          <div className="flex items-center gap-3 mb-2">
            <StepBadge n="3" active />
            <h2 className="heading font-bold text-white text-sm tracking-wide">Buyer Scenario</h2>
          </div>
          <p className="text-xs mb-4" style={{ color: 'rgba(148,163,184,0.5)' }}>
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
                    background: active ? 'rgba(236,72,153,0.1)' : 'rgba(255,255,255,0.02)',
                    border: `1px solid ${active ? 'rgba(236,72,153,0.4)' : 'rgba(255,255,255,0.06)'}`,
                  }}
                >
                  <p className="font-semibold text-xs text-white mb-0.5">{m.label}</p>
                  <p className="text-xs leading-tight" style={{ color: 'rgba(148,163,184,0.5)' }}>{m.desc}</p>
                </button>
              );
            })}
          </div>

          {/* Mode-specific content */}
          {scenarioMode === 'random' && (
            <p className="text-xs px-3 py-2.5 rounded-xl"
              style={{ background: 'rgba(255,255,255,0.02)', color: 'rgba(148,163,184,0.6)' }}>
              A coherent buyer scenario will be generated for this session.
            </p>
          )}

          {scenarioMode === 'preset' && (
            <select
              value={selectedPreset}
              onChange={e => setSelectedPreset(e.target.value)}
              className="w-full px-3 py-2.5 rounded-xl text-sm"
              style={{ background: 'rgba(8,14,28,0.9)', border: '1px solid rgba(255,255,255,0.08)', color: '#e2e8f0' }}
            >
              <option value="">Select a scenario…</option>
              {presets.map(p => (
                <option key={p.preset_id} value={p.preset_id}>{p.label}</option>
              ))}
            </select>
          )}

          {scenarioMode === 'custom' && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium mb-1.5 block" style={{ color: 'rgba(148,163,184,0.5)' }}>
                  Buyer type
                </label>
                <select
                  value={customContext}
                  onChange={e => setCustomContext(e.target.value)}
                  className="w-full px-3 py-2.5 rounded-xl text-sm"
                  style={{ background: 'rgba(8,14,28,0.9)', border: '1px solid rgba(255,255,255,0.08)', color: '#e2e8f0' }}
                >
                  <option value="">Any (random)</option>
                  {BUYER_CONTEXTS.map(c => <option key={c.id} value={c.id}>{c.label}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs font-medium mb-1.5 block" style={{ color: 'rgba(148,163,184,0.5)' }}>
                  Timeline
                </label>
                <select
                  value={customTimeline}
                  onChange={e => setCustomTimeline(e.target.value)}
                  className="w-full px-3 py-2.5 rounded-xl text-sm"
                  style={{ background: 'rgba(8,14,28,0.9)', border: '1px solid rgba(255,255,255,0.08)', color: '#e2e8f0' }}
                >
                  <option value="">Any (random)</option>
                  {SCENARIO_TIMELINES.map(t => <option key={t.id} value={t.id}>{t.label}</option>)}
                </select>
              </div>
              <p className="text-xs sm:col-span-2" style={{ color: 'rgba(148,163,184,0.4)' }}>
                Budget and must-haves are drawn to match the buyer type.
              </p>
            </div>
          )}
        </div>

        {/* Selected preview */}
        {selected && (
          <div
            className="rounded-2xl p-4 mb-5 flex items-center gap-4"
            style={{ background: 'rgba(37,99,235,0.07)', border: '1px solid rgba(37,99,235,0.2)' }}
          >
            <Avatar src={selected.avatar_url} name={selected.name_en} size={40} active />
            <div>
              <p className="font-semibold text-sm text-white">{selected.name_en}</p>
              <p className="text-xs" style={{ color: 'rgba(148,163,184,0.5)' }}>
                {activeDiff?.label} difficulty · Ready to start
              </p>
            </div>
            <div className="ml-auto">
              <svg width="16" height="16" fill="none" stroke="#4ade80" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                <path d="M20 6L9 17l-5-5"/>
              </svg>
            </div>
          </div>
        )}

        {/* Start Button */}
        <button
          onClick={handleStart}
          disabled={!selected || creating}
          className="btn-primary w-full py-4 rounded-xl font-semibold text-white flex items-center justify-center gap-3"
        >
          {creating ? (
            <>
              <svg className="w-5 h-5 spin-ring" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.2)" strokeWidth="3"/>
                <path d="M12 2a10 10 0 0110 10" stroke="white" strokeWidth="3" strokeLinecap="round"/>
              </svg>
              Creating Session…
            </>
          ) : (
            <>
              <svg width="18" height="18" fill="none" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
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
