import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { personasAPI, sessionsAPI } from '../services/api';
import Layout from '../components/Layout';

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

export default function SessionSetup() {
  const navigate  = useNavigate();
  const [personas, setPersonas] = useState([]);
  const [selected, setSelected] = useState(null);
  const [difficulty, setDifficulty] = useState('medium');
  const [loading, setLoading]   = useState(true);
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    personasAPI.getAll()
      .then(data => setPersonas(data.personas))
      .finally(() => setLoading(false));
  }, []);

  const handleStart = async () => {
    if (!selected) return;
    setCreating(true);
    try {
      const session = await sessionsAPI.create(selected.id, difficulty);
      navigate(`/session/${session.id}`);
    } catch {
      alert('Failed to create session');
      setCreating(false);
    }
  };

  const filtered = personas.filter(p => p.difficulty === difficulty);
  const activeDiff = DIFFICULTIES.find(d => d.id === difficulty);

  return (
    <Layout>
      <div className="p-8 max-w-3xl mx-auto">
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
                  onClick={() => { setDifficulty(d.id); setSelected(null); }}
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
                      <div
                        className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 text-sm font-bold"
                        style={{
                          background: active ? 'rgba(37,99,235,0.25)' : 'rgba(255,255,255,0.06)',
                          color: active ? '#93c5fd' : 'rgba(148,163,184,0.5)',
                        }}
                      >
                        {p.name_en?.charAt(0) || '?'}
                      </div>
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

        {/* Selected preview */}
        {selected && (
          <div
            className="rounded-2xl p-4 mb-5 flex items-center gap-4"
            style={{ background: 'rgba(37,99,235,0.07)', border: '1px solid rgba(37,99,235,0.2)' }}
          >
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center font-bold text-sm flex-shrink-0"
              style={{ background: 'rgba(37,99,235,0.2)', color: '#93c5fd' }}
            >
              {selected.name_en?.charAt(0)}
            </div>
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
