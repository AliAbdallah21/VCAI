import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { getPlans } from '../services/api';
import PricingCards from '../components/PricingCards';
import { useScrollReveal } from '../hooks/useScrollReveal';
import { useTheme } from '../context/ThemeContext';

function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const isDark = theme === 'dark';
  return (
    <button onClick={toggle} title={isDark ? 'Light mode' : 'Dark mode'}
      style={{
        width: 34, height: 34, borderRadius: 8, border: '1px solid var(--border)',
        background: 'transparent', cursor: 'pointer', display: 'flex',
        alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)',
        transition: 'all 0.2s ease',
      }}
      onMouseEnter={e => { e.currentTarget.style.background = 'var(--primary-soft)'; e.currentTarget.style.color = 'var(--primary)'; }}
      onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-muted)'; }}
    >
      {isDark ? (
        <svg width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
          <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
        </svg>
      ) : (
        <svg width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
          <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/>
        </svg>
      )}
    </button>
  );
}

/* ── Data ──────────────────────────────────────────────── */
const STATS = [
  { value: '8', label: 'Sales skills scored' },
  { value: '152', label: 'Knowledge documents' },
  { value: '<3s', label: 'First audio latency' },
  { value: '100%', label: 'Egyptian Arabic' },
];

const STEPS = [
  {
    icon: (
      <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <path d="M18 21a8 8 0 00-16 0"/><circle cx="10" cy="8" r="4"/><path d="M22 21a8 8 0 00-6-7.7"/>
      </svg>
    ),
    title: 'Sign up your firm',
    body: 'Create a workspace for your real-estate sales team in minutes — no credit card required.',
  },
  {
    icon: (
      <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/>
      </svg>
    ),
    title: 'Invite your agents',
    body: 'Add salespeople to your workspace. Each gets their own training history and skill profile.',
  },
  {
    icon: (
      <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
    ),
    title: 'Train and track',
    body: 'Agents practice against AI customers while you monitor progress with automated skill reports.',
  },
];

const FEATURES = [
  {
    icon: (
      <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <path d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z"/>
      </svg>
    ),
    title: 'Egyptian-dialect voice',
    body: 'A Chatterbox TTS fine-tuned on Egyptian Arabic delivers natural, realistic speech — not robotic text-to-speech.',
    accent: '#deb7ff',
  },
  {
    icon: (
      <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>
      </svg>
    ),
    title: 'Dual-modal emotion',
    body: 'Voice tone and text sentiment are fused in real time so the AI customer reacts like a real person would.',
    accent: '#a5d6a7',
  },
  {
    icon: (
      <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
      </svg>
    ),
    title: 'Cross-session memory',
    body: 'The virtual customer remembers every prior conversation — building a continuous, realistic relationship.',
    accent: '#e9c46a',
  },
  {
    icon: (
      <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <path d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
      </svg>
    ),
    title: 'Automated scoring',
    body: 'Every session is scored across 8 sales skills — rapport, objection handling, negotiation, and more.',
    accent: '#f4a261',
  },
  {
    icon: (
      <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>
      </svg>
    ),
    title: 'Personalised coaching',
    body: 'AI-generated coaching feedback pinpoints weak spots and recommends targeted practice scenarios.',
    accent: '#deb7ff',
  },
  {
    icon: (
      <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
      </svg>
    ),
    title: 'Manager dashboard',
    body: 'Team leads see every agent\'s skill radar, session history, and trend lines in one place.',
    accent: '#a5d6a7',
  },
];

const TEAM = [
  { name: 'Ali Abdallah', role: 'AI Pipeline & Backend', initials: 'AA' },
  { name: 'Bakr', role: 'TTS & Voice Fine-tuning', initials: 'B' },
  { name: 'Ismail', role: 'LLM & Evaluation', initials: 'I' },
  { name: 'Menna', role: 'Emotion Detection', initials: 'M' },
];

/* ── Inline SVG logo ── */
function VCAILogo({ size = 36 }) {
  return (
    <div
      style={{
        width: size,
        height: size,
        borderRadius: Math.round(size * 0.28),
        background: 'linear-gradient(135deg, #b472f1, #deb7ff)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
      }}
    >
      <span style={{ fontWeight: 800, fontSize: size * 0.45, color: '#4a007f', lineHeight: 1 }}>V</span>
    </div>
  );
}

/* ── Header ── */
function Header({ isAuthenticated }) {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <header
      className="sticky top-0 z-50"
      style={{
        background: scrolled ? 'var(--glass-bg)' : 'transparent',
        backdropFilter: scrolled ? 'blur(20px)' : 'none',
        borderBottom: scrolled ? '1px solid var(--border)' : '1px solid transparent',
        transition: 'all 0.3s ease',
      }}
    >
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <VCAILogo size={34} />
          <span className="font-bold text-xl" style={{ color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>VCAI</span>
        </div>
        <nav className="flex items-center gap-2">
          <a href="#features" className="hidden md:block text-sm font-medium px-3 py-1.5 rounded-lg transition-colors" style={{ color: 'var(--text-secondary)' }}
            onMouseEnter={e => e.target.style.color='var(--primary)'}
            onMouseLeave={e => e.target.style.color='var(--text-secondary)'}
          >Features</a>
          <a href="#about" className="hidden md:block text-sm font-medium px-3 py-1.5 rounded-lg transition-colors" style={{ color: 'var(--text-secondary)' }}
            onMouseEnter={e => e.target.style.color='var(--primary)'}
            onMouseLeave={e => e.target.style.color='var(--text-secondary)'}
          >About</a>
          <a href="#pricing" className="hidden md:block text-sm font-medium px-3 py-1.5 rounded-lg transition-colors" style={{ color: 'var(--text-secondary)' }}
            onMouseEnter={e => e.target.style.color='var(--primary)'}
            onMouseLeave={e => e.target.style.color='var(--text-secondary)'}
          >Pricing</a>
          <Link to="/contact" className="hidden md:block text-sm font-medium px-3 py-1.5 rounded-lg transition-colors" style={{ color: 'var(--text-secondary)' }}
            onMouseEnter={e => e.target.style.color='var(--primary)'}
            onMouseLeave={e => e.target.style.color='var(--text-secondary)'}
          >Contact</Link>
          <ThemeToggle />
          {isAuthenticated ? (
            <Link to="/dashboard" className="btn-primary" style={{ padding: '8px 18px' }}>Go to dashboard</Link>
          ) : (
            <>
              <Link to="/login" className="text-sm font-medium px-3 py-1.5" style={{ color: 'var(--text-secondary)' }}>Sign in</Link>
              <Link to="/onboarding?plan=free" className="btn-primary" style={{ padding: '8px 18px' }}>Start free</Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}

/* ── Main ── */
export default function Landing() {
  const { isAuthenticated } = useAuth();
  const [plans, setPlans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useScrollReveal();

  useEffect(() => {
    getPlans()
      .then(setPlans)
      .catch(() => setError('Could not load pricing. Please try again later.'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ background: 'var(--bg-app)', color: 'var(--text-primary)', overflowX: 'hidden' }}>
      <Header isAuthenticated={isAuthenticated} />

      {/* ── Hero ── */}
      <section style={{ position: 'relative', padding: '100px 24px 120px', textAlign: 'center', overflow: 'hidden' }}>
        {/* Background orbs */}
        <div className="orb-float" style={{
          position: 'absolute', top: '10%', left: '15%',
          width: 400, height: 400, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(180,114,241,0.12) 0%, transparent 70%)',
          pointerEvents: 'none',
        }} />
        <div className="orb-float-slow" style={{
          position: 'absolute', top: '20%', right: '10%',
          width: 300, height: 300, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(124,58,237,0.1) 0%, transparent 70%)',
          pointerEvents: 'none',
        }} />
        <div style={{
          position: 'absolute', bottom: '5%', left: '50%', transform: 'translateX(-50%)',
          width: 600, height: 200, borderRadius: '50%',
          background: 'radial-gradient(ellipse, rgba(180,114,241,0.06) 0%, transparent 70%)',
          pointerEvents: 'none',
        }} />

        <div className="max-w-4xl mx-auto" style={{ position: 'relative', zIndex: 1 }}>
          {/* Eyebrow */}
          <div className="slide-up" style={{ display: 'inline-flex', alignItems: 'center', gap: 8, marginBottom: 24 }}>
            <span style={{
              background: 'rgba(222,183,255,0.1)',
              border: '1px solid rgba(222,183,255,0.2)',
              borderRadius: 999,
              padding: '5px 14px',
              fontSize: 12,
              fontWeight: 600,
              color: 'var(--primary)',
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
            }}>
              Graduation Project · MIU 2026
            </span>
          </div>

          <h1 className="display-special hero-gradient-text slide-up" style={{ animationDelay: '0.1s', marginBottom: 24 }}>
            Train Smarter.<br />Sell Better.
          </h1>

          <p className="body-lg slide-up" style={{
            animationDelay: '0.2s',
            color: 'var(--text-muted)',
            maxWidth: 600,
            margin: '0 auto 16px',
          }}>
            The first Egyptian-Arabic AI sales-training platform. Practice real conversations,
            get emotion-aware feedback, and track improvement — automatically.
          </p>

          <div className="slide-up" style={{ animationDelay: '0.35s', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12, marginTop: 36 }}>
            <Link to="/onboarding?plan=free" className="btn-primary" style={{ padding: '12px 28px', fontSize: 14 }}>
              Start for free
            </Link>
            <a href="#features" className="btn-secondary" style={{ padding: '12px 28px', fontSize: 14 }}>
              See how it works
            </a>
          </div>
        </div>
      </section>

      {/* ── Stats strip ── */}
      <section style={{ borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)', padding: '32px 24px' }}>
        <div className="max-w-4xl mx-auto" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 16 }}>
          {STATS.map((s, i) => (
            <div key={s.label} className="reveal stat-card-landing" data-delay={i * 80} style={{ textAlign: 'center' }}>
              <div className="stats-high-contrast" style={{ fontSize: 32, color: 'var(--primary)', lineHeight: 1 }}>{s.value}</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 6, fontWeight: 500 }}>{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── How it works ── */}
      <section id="features" className="max-w-6xl mx-auto px-6 py-20">
        <div className="reveal" style={{ textAlign: 'center', marginBottom: 48 }}>
          <h2 className="headline-md" style={{ color: 'var(--text-primary)' }}>How it works</h2>
          <p style={{ color: 'var(--text-muted)', marginTop: 12, fontSize: 16 }}>Three steps from sign-up to measurable progress.</p>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 20 }}>
          {STEPS.map((step, i) => (
            <div key={step.title} className="ds-card reveal" data-delay={i * 100} style={{ padding: '28px 24px' }}>
              <div style={{
                width: 44, height: 44, borderRadius: 12,
                background: 'var(--primary-soft)',
                border: '1px solid rgba(222,183,255,0.2)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: 'var(--primary)', marginBottom: 16,
              }}>
                {step.icon}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                <span style={{
                  width: 22, height: 22, borderRadius: '50%',
                  background: 'rgba(222,183,255,0.15)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 11, fontWeight: 700, color: 'var(--primary)',
                }}>{i + 1}</span>
                <h3 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>{step.title}</h3>
              </div>
              <p style={{ color: 'var(--text-muted)', fontSize: 14, lineHeight: 1.6 }}>{step.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features grid ── */}
      <section style={{ background: 'var(--surface-container-lowest)', padding: '80px 24px' }}>
        <div className="max-w-6xl mx-auto">
          <div className="reveal" style={{ textAlign: 'center', marginBottom: 48 }}>
            <h2 className="headline-md" style={{ color: 'var(--text-primary)' }}>Built for realistic practice</h2>
            <p style={{ color: 'var(--text-muted)', marginTop: 12, fontSize: 16 }}>
              Every component is engineered for authentic Egyptian-Arabic sales conversations.
            </p>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 20 }}>
            {FEATURES.map((f, i) => (
              <div key={f.title} className="feature-card reveal" data-delay={i * 60}>
                <div style={{
                  width: 44, height: 44, borderRadius: 12,
                  background: `${f.accent}15`,
                  border: `1px solid ${f.accent}25`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: f.accent, marginBottom: 16,
                }}>
                  {f.icon}
                </div>
                <h3 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 8 }}>{f.title}</h3>
                <p style={{ color: 'var(--text-muted)', fontSize: 14, lineHeight: 1.6 }}>{f.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── About / Team ── */}
      <section id="about" className="max-w-6xl mx-auto px-6 py-20">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 64, alignItems: 'center' }} className="about-grid">
          <div className="reveal-left">
            <span style={{
              fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.1em',
              color: 'var(--primary)', marginBottom: 12, display: 'block',
            }}>About the project</span>
            <h2 className="headline-md" style={{ color: 'var(--text-primary)', marginBottom: 20 }}>
              A graduation project from MIU
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: 15, lineHeight: 1.8, marginBottom: 16 }}>
              VCAI was built as a Computer Science graduation project at Misr International University, 2026.
              Our goal: give real-estate sales teams in Egypt an AI practice partner that actually speaks their language —
              Egyptian Arabic — and reacts like a real human customer.
            </p>
            <p style={{ color: 'var(--text-muted)', fontSize: 15, lineHeight: 1.8, marginBottom: 28 }}>
              The platform combines a fine-tuned Egyptian TTS model, dual-modal emotion detection,
              a LangGraph orchestration pipeline, and an automated multi-skill evaluation engine.
            </p>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              <Link to="/about" className="btn-secondary" style={{ padding: '10px 20px' }}>Learn more</Link>
              <Link to="/contact" className="btn-secondary" style={{ padding: '10px 20px' }}>Contact us</Link>
            </div>
          </div>

          <div className="reveal" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {TEAM.map((member, i) => (
              <div key={member.name} className="team-card" data-delay={i * 80}>
                <div style={{
                  width: 52, height: 52, borderRadius: '50%',
                  background: 'linear-gradient(135deg, #b472f1, #7c3aed)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  margin: '0 auto 12px',
                  fontSize: 16, fontWeight: 700, color: 'white',
                }}>{member.initials}</div>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: 14 }}>{member.name}</div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>{member.role}</div>
              </div>
            ))}
            <div className="team-card" style={{ gridColumn: '1 / -1', background: 'var(--bg-card-alt)' }} data-delay={400}>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>Supervised by</div>
              <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: 15 }}>Dr. Ahmed Mansour</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>T.A. Karim Mohamed · Misr International University</div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Pricing ── */}
      <section id="pricing" style={{ background: 'var(--surface-container-lowest)', padding: '80px 24px' }}>
        <div className="max-w-6xl mx-auto">
          <div className="reveal" style={{ textAlign: 'center', marginBottom: 48 }}>
            <h2 className="headline-md" style={{ color: 'var(--text-primary)' }}>Simple, transparent pricing</h2>
            <p style={{ color: 'var(--text-muted)', marginTop: 12, fontSize: 16 }}>Pick a plan that fits your team.</p>
          </div>
          {loading && <p style={{ textAlign: 'center', color: 'var(--text-muted)' }}>Loading plans…</p>}
          {error && !loading && (
            <p style={{ textAlign: 'center', color: 'var(--error)', background: 'rgba(255,180,171,0.08)', borderRadius: 12, padding: '12px 24px', maxWidth: 400, margin: '0 auto' }}>
              {error}
            </p>
          )}
          {!loading && !error && plans.length > 0 && <PricingCards plans={plans} />}
        </div>
      </section>

      {/* ── CTA banner ── */}
      <section style={{ padding: '80px 24px', textAlign: 'center' }}>
        <div className="max-w-2xl mx-auto reveal">
          <h2 className="headline-md" style={{ color: 'var(--text-primary)', marginBottom: 16 }}>
            Ready to train smarter?
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: 16, marginBottom: 32 }}>
            Join your team on VCAI and start measuring real sales improvement from day one.
          </p>
          <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
            <Link to="/onboarding?plan=free" className="btn-primary" style={{ padding: '12px 32px', fontSize: 14 }}>
              Get started free
            </Link>
            <Link to="/contact" className="btn-secondary" style={{ padding: '12px 32px', fontSize: 14 }}>
              Talk to us
            </Link>
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer style={{ borderTop: '1px solid var(--border)', padding: '48px 24px 32px' }}>
        <div className="max-w-6xl mx-auto">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 32, marginBottom: 40 }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                <VCAILogo size={30} />
                <span style={{ fontWeight: 700, fontSize: 16, color: 'var(--text-primary)' }}>VCAI</span>
              </div>
              <p style={{ fontSize: 13, color: 'var(--text-subtle)', lineHeight: 1.7 }}>
                Egyptian-Arabic AI sales training for real-estate teams.
              </p>
            </div>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: 12 }}>Platform</div>
              {[{ label: 'Features', href: '#features' }, { label: 'Pricing', href: '#pricing' }, { label: 'Sign in', href: '/login' }, { label: 'Start free', href: '/onboarding?plan=free' }].map(l => (
                <div key={l.label} style={{ marginBottom: 8 }}>
                  {l.href.startsWith('/') ? (
                    <Link to={l.href} style={{ fontSize: 13, color: 'var(--text-muted)', textDecoration: 'none' }}
                      onMouseEnter={e => e.target.style.color='var(--primary)'}
                      onMouseLeave={e => e.target.style.color='var(--text-muted)'}
                    >{l.label}</Link>
                  ) : (
                    <a href={l.href} style={{ fontSize: 13, color: 'var(--text-muted)', textDecoration: 'none' }}
                      onMouseEnter={e => e.target.style.color='var(--primary)'}
                      onMouseLeave={e => e.target.style.color='var(--text-muted)'}
                    >{l.label}</a>
                  )}
                </div>
              ))}
            </div>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: 12 }}>Company</div>
              {[{ label: 'About', href: '/about' }, { label: 'Contact', href: '/contact' }, { label: 'Privacy Policy', href: '/privacy' }].map(l => (
                <div key={l.label} style={{ marginBottom: 8 }}>
                  <Link to={l.href} style={{ fontSize: 13, color: 'var(--text-muted)', textDecoration: 'none' }}
                    onMouseEnter={e => e.target.style.color='var(--primary)'}
                    onMouseLeave={e => e.target.style.color='var(--text-muted)'}
                  >{l.label}</Link>
                </div>
              ))}
            </div>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: 12 }}>Contact</div>
              <a href="mailto:gradproject11234@gmail.com" style={{ fontSize: 13, color: 'var(--text-muted)', textDecoration: 'none', display: 'block', marginBottom: 8 }}
                onMouseEnter={e => e.target.style.color='var(--primary)'}
                onMouseLeave={e => e.target.style.color='var(--text-muted)'}
              >gradproject11234@gmail.com</a>
              <p style={{ fontSize: 13, color: 'var(--text-subtle)', lineHeight: 1.6 }}>
                Misr International University<br />Cairo, Egypt
              </p>
            </div>
          </div>
          <div style={{ borderTop: '1px solid var(--border)', paddingTop: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
            <span style={{ fontSize: 12, color: 'var(--text-subtle)' }}>© 2026 VCAI — Graduation Project, Misr International University</span>
            <div style={{ display: 'flex', gap: 16 }}>
              <Link to="/privacy" style={{ fontSize: 12, color: 'var(--text-subtle)', textDecoration: 'none' }}>Privacy Policy</Link>
              <Link to="/contact" style={{ fontSize: 12, color: 'var(--text-subtle)', textDecoration: 'none' }}>Contact</Link>
            </div>
          </div>
        </div>
      </footer>

      <style>{`
        @media (max-width: 768px) {
          .about-grid { grid-template-columns: 1fr !important; gap: 40px !important; }
        }
      `}</style>
    </div>
  );
}
