import { Link } from 'react-router-dom';
import { useScrollReveal } from '../hooks/useScrollReveal';

const TEAM = [
  { name: 'Ali Abdallah', role: 'AI Pipeline & Backend', desc: 'LangGraph orchestration, FastAPI backend, WebSocket pipeline, TTS integration.', initials: 'AA' },
  { name: 'Bakr', role: 'TTS & Voice Fine-tuning', desc: 'Egyptian Arabic Chatterbox fine-tune, multi-speaker training, audio preprocessing.', initials: 'B' },
  { name: 'Ismail', role: 'LLM & Evaluation', desc: 'OpenRouter integration, Qwen local model, post-session evaluation pipeline.', initials: 'I' },
  { name: 'Menna', role: 'Emotion Detection', desc: 'Dual-modal emotion fusion, emotion2vec + AraBERT, emotional context engine.', initials: 'M' },
];

const TECH = [
  { name: 'Faster-Whisper', desc: 'Real-time Arabic speech recognition on GPU' },
  { name: 'Chatterbox TTS', desc: 'Egyptian Arabic fine-tuned text-to-speech' },
  { name: 'emotion2vec + AraBERT', desc: 'Dual-modal emotion detection' },
  { name: 'LangGraph', desc: 'Stateful AI pipeline orchestration' },
  { name: 'OpenRouter / Qwen', desc: 'LLM backbone for virtual customer' },
  { name: 'FAISS + RAG', desc: 'Knowledge-base fact-checking in evaluation' },
  { name: 'FastAPI + PostgreSQL', desc: 'Backend API and session storage' },
  { name: 'React + Vite', desc: 'Frontend application' },
];

function VCAILogo({ size = 36 }) {
  return (
    <div style={{
      width: size, height: size, borderRadius: Math.round(size * 0.28),
      background: 'linear-gradient(135deg, #b472f1, #deb7ff)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
    }}>
      <span style={{ fontWeight: 800, fontSize: size * 0.45, color: '#4a007f', lineHeight: 1 }}>V</span>
    </div>
  );
}

export default function About() {
  useScrollReveal();

  return (
    <div style={{ background: 'var(--bg-app)', color: 'var(--text-primary)', minHeight: '100vh' }}>

      {/* Header */}
      <header style={{ borderBottom: '1px solid var(--border)', padding: '0 24px' }}>
        <div className="max-w-6xl mx-auto h-16 flex items-center justify-between">
          <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: 10, textDecoration: 'none' }}>
            <VCAILogo size={32} />
            <span style={{ fontWeight: 700, fontSize: 17, color: 'var(--text-primary)' }}>VCAI</span>
          </Link>
          <div style={{ display: 'flex', gap: 12 }}>
            <Link to="/contact" className="btn-secondary" style={{ padding: '7px 16px' }}>Contact</Link>
            <Link to="/login" className="btn-primary" style={{ padding: '7px 16px' }}>Sign in</Link>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section style={{ padding: '80px 24px 60px', textAlign: 'center', position: 'relative', overflow: 'hidden' }}>
        <div style={{
          position: 'absolute', top: 0, left: '50%', transform: 'translateX(-50%)',
          width: 600, height: 300, borderRadius: '50%',
          background: 'radial-gradient(ellipse, rgba(180,114,241,0.1) 0%, transparent 70%)',
          pointerEvents: 'none',
        }} />
        <div className="max-w-3xl mx-auto" style={{ position: 'relative' }}>
          <span style={{
            fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.1em',
            color: 'var(--primary)', marginBottom: 16, display: 'block',
          }}>About VCAI</span>
          <h1 className="headline-lg slide-up" style={{ color: 'var(--text-primary)', marginBottom: 20 }}>
            AI Sales Training,<br />Built for Egypt
          </h1>
          <p className="body-lg slide-up" style={{ color: 'var(--text-muted)', animationDelay: '0.1s' }}>
            VCAI is a Computer Science graduation project from Misr International University, 2026.
            We built the first Egyptian-Arabic AI platform for real-estate sales training.
          </p>
        </div>
      </section>

      {/* Mission */}
      <section style={{ padding: '40px 24px 80px' }}>
        <div className="max-w-4xl mx-auto" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 40 }}>
          <div className="ds-card reveal" style={{ padding: '32px 28px' }}>
            <div style={{ fontSize: 28, marginBottom: 16 }}>🎯</div>
            <h3 style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 12 }}>The Problem</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: 14, lineHeight: 1.8 }}>
              Real-estate sales agents in Egypt have no scalable way to practice conversations before
              meeting real clients. Role-plays with managers are rare, inconsistent, and don't provide
              objective feedback. Poor practice = poor performance.
            </p>
          </div>
          <div className="ds-card reveal" data-delay="100" style={{ padding: '32px 28px' }}>
            <div style={{ fontSize: 28, marginBottom: 16 }}>💡</div>
            <h3 style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 12 }}>Our Solution</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: 14, lineHeight: 1.8 }}>
              A tireless AI customer that speaks Egyptian Arabic, reacts emotionally to the salesperson's
              tone and words, remembers prior sessions, and automatically scores performance across
              8 professional sales skills after every practice session.
            </p>
          </div>
        </div>
      </section>

      {/* Tech stack */}
      <section style={{ background: 'var(--surface-container-lowest)', padding: '80px 24px' }}>
        <div className="max-w-6xl mx-auto">
          <div className="reveal" style={{ textAlign: 'center', marginBottom: 48 }}>
            <h2 className="headline-md" style={{ color: 'var(--text-primary)' }}>Technology Stack</h2>
            <p style={{ color: 'var(--text-muted)', marginTop: 12 }}>Built end-to-end with state-of-the-art open-source models.</p>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 16 }}>
            {TECH.map((t, i) => (
              <div key={t.name} className="ds-card reveal" data-delay={i * 50} style={{ padding: '20px 20px' }}>
                <div style={{ fontWeight: 700, color: 'var(--primary)', fontSize: 14, marginBottom: 6 }}>{t.name}</div>
                <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>{t.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Team */}
      <section style={{ padding: '80px 24px' }}>
        <div className="max-w-6xl mx-auto">
          <div className="reveal" style={{ textAlign: 'center', marginBottom: 48 }}>
            <h2 className="headline-md" style={{ color: 'var(--text-primary)' }}>The Team</h2>
            <p style={{ color: 'var(--text-muted)', marginTop: 12 }}>Computer Science students at Misr International University.</p>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 20, marginBottom: 24 }}>
            {TEAM.map((m, i) => (
              <div key={m.name} className="team-card reveal" data-delay={i * 80}>
                <div style={{
                  width: 64, height: 64, borderRadius: '50%',
                  background: 'linear-gradient(135deg, #b472f1, #7c3aed)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  margin: '0 auto 16px',
                  fontSize: 20, fontWeight: 700, color: 'white',
                }}>{m.initials}</div>
                <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: 15, marginBottom: 4 }}>{m.name}</div>
                <div style={{ fontSize: 12, color: 'var(--primary)', fontWeight: 600, marginBottom: 10 }}>{m.role}</div>
                <div style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.6 }}>{m.desc}</div>
              </div>
            ))}
          </div>

          {/* Supervisors */}
          <div className="ds-card reveal" style={{ padding: '28px 32px', display: 'flex', gap: 40, flexWrap: 'wrap', alignItems: 'center', justifyContent: 'center', textAlign: 'center', background: 'var(--bg-card-alt)' }}>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: 8 }}>Supervisor</div>
              <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: 16 }}>Dr. Ahmed Mansour</div>
              <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>Misr International University</div>
            </div>
            <div style={{ width: 1, height: 48, background: 'var(--border)' }} />
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: 8 }}>Teaching Assistant</div>
              <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: 16 }}>T.A. Karim Mohamed</div>
              <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>Misr International University</div>
            </div>
            <div style={{ width: 1, height: 48, background: 'var(--border)' }} />
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: 8 }}>Institution</div>
              <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: 16 }}>Misr International University</div>
              <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>Faculty of Computer Science · Class of 2026</div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section style={{ background: 'var(--surface-container-lowest)', padding: '64px 24px', textAlign: 'center' }}>
        <div className="max-w-xl mx-auto reveal">
          <h2 style={{ fontSize: 28, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }}>Try VCAI today</h2>
          <p style={{ color: 'var(--text-muted)', marginBottom: 28 }}>Free to start — no credit card required.</p>
          <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
            <Link to="/onboarding?plan=free" className="btn-primary" style={{ padding: '11px 28px' }}>Get started</Link>
            <Link to="/contact" className="btn-secondary" style={{ padding: '11px 28px' }}>Contact us</Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer style={{ borderTop: '1px solid var(--border)', padding: '24px', textAlign: 'center' }}>
        <div style={{ display: 'flex', gap: 24, justifyContent: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
          <Link to="/" style={{ fontSize: 13, color: 'var(--text-muted)', textDecoration: 'none' }}>Home</Link>
          <Link to="/contact" style={{ fontSize: 13, color: 'var(--text-muted)', textDecoration: 'none' }}>Contact</Link>
          <Link to="/privacy" style={{ fontSize: 13, color: 'var(--text-muted)', textDecoration: 'none' }}>Privacy Policy</Link>
        </div>
        <p style={{ fontSize: 12, color: 'var(--text-subtle)' }}>© 2026 VCAI — Graduation Project, Misr International University</p>
      </footer>
    </div>
  );
}
