import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useScrollReveal } from '../hooks/useScrollReveal';

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

const CONTACTS = [
  {
    icon: (
      <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/>
      </svg>
    ),
    label: 'Email',
    value: 'gradproject11234@gmail.com',
    href: 'mailto:gradproject11234@gmail.com',
  },
  {
    icon: (
      <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/><circle cx="12" cy="10" r="3"/>
      </svg>
    ),
    label: 'Location',
    value: 'Misr International University, Cairo, Egypt',
    href: null,
  },
  {
    icon: (
      <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
        <path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07A19.5 19.5 0 013.07 9.81a19.79 19.79 0 01-3.07-8.68A2 2 0 012 .18h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L6.91 7.91a16 16 0 006.18 6.18l1.79-1.79a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 16.92z"/>
      </svg>
    ),
    label: 'Response time',
    value: 'We reply within 1–2 business days',
    href: null,
  },
];

export default function Contact() {
  const [form, setForm] = useState({ name: '', email: '', subject: '', message: '' });
  const [sent, setSent] = useState(false);

  useScrollReveal();

  const handleSubmit = (e) => {
    e.preventDefault();
    const body = `Name: ${form.name}%0D%0AEmail: ${form.email}%0D%0A%0D%0A${form.message}`;
    window.location.href = `mailto:gradproject11234@gmail.com?subject=${encodeURIComponent(form.subject || 'VCAI Enquiry')}&body=${body}`;
    setSent(true);
  };

  return (
    <div style={{ background: 'var(--bg-app)', color: 'var(--text-primary)', minHeight: '100vh' }}>

      {/* Header */}
      <header style={{ borderBottom: '1px solid var(--border)', padding: '0 24px' }}>
        <div className="max-w-6xl mx-auto h-16 flex items-center justify-between">
          <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: 10, textDecoration: 'none' }}>
            <VCAILogo size={30} />
            <span style={{ fontWeight: 700, fontSize: 16, color: 'var(--text-primary)' }}>VCAI</span>
          </Link>
          <div style={{ display: 'flex', gap: 12 }}>
            <Link to="/about" className="btn-secondary" style={{ padding: '7px 16px' }}>About</Link>
            <Link to="/login" className="btn-primary" style={{ padding: '7px 16px' }}>Sign in</Link>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section style={{ padding: '64px 24px 48px', textAlign: 'center', position: 'relative', overflow: 'hidden' }}>
        <div style={{
          position: 'absolute', top: 0, left: '50%', transform: 'translateX(-50%)',
          width: 500, height: 250, borderRadius: '50%',
          background: 'radial-gradient(ellipse, rgba(180,114,241,0.09) 0%, transparent 70%)',
          pointerEvents: 'none',
        }} />
        <div className="max-w-2xl mx-auto" style={{ position: 'relative' }}>
          <span style={{
            fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.1em',
            color: 'var(--primary)', marginBottom: 12, display: 'block',
          }}>Get in touch</span>
          <h1 className="headline-lg slide-up" style={{ color: 'var(--text-primary)', marginBottom: 16 }}>Contact Us</h1>
          <p className="body-md slide-up" style={{ color: 'var(--text-muted)', animationDelay: '0.1s' }}>
            Questions about VCAI? Want to discuss enterprise plans or research collaboration? We'd love to hear from you.
          </p>
        </div>
      </section>

      {/* Body */}
      <section style={{ padding: '0 24px 80px' }}>
        <div className="max-w-5xl mx-auto" style={{ display: 'grid', gridTemplateColumns: '1fr 1.6fr', gap: 40, alignItems: 'start' }}>

          {/* Left — contact info */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {CONTACTS.map((c, i) => (
              <div key={c.label} className="ds-card reveal" data-delay={i * 80} style={{ padding: '20px 22px', display: 'flex', gap: 16, alignItems: 'flex-start' }}>
                <div style={{
                  width: 40, height: 40, borderRadius: 10, flexShrink: 0,
                  background: 'var(--primary-soft)', border: '1px solid rgba(222,183,255,0.2)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: 'var(--primary)',
                }}>{c.icon}</div>
                <div>
                  <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-muted)', marginBottom: 4 }}>{c.label}</div>
                  {c.href ? (
                    <a href={c.href} style={{ fontSize: 14, color: 'var(--primary)', textDecoration: 'none', fontWeight: 500 }}>{c.value}</a>
                  ) : (
                    <span style={{ fontSize: 14, color: 'var(--text-secondary)' }}>{c.value}</span>
                  )}
                </div>
              </div>
            ))}

            <div className="ds-card reveal" data-delay="280" style={{ padding: '20px 22px', background: 'var(--bg-card-alt)' }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>Project Context</div>
              <p style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.7 }}>
                VCAI is a Computer Science graduation project at Misr International University, Class of 2026.
                Supervised by Dr. Ahmed Mansour.
              </p>
            </div>
          </div>

          {/* Right — form */}
          <div className="ds-card reveal" data-delay="100" style={{ padding: '32px 28px' }}>
            {sent ? (
              <div style={{ textAlign: 'center', padding: '40px 0' }}>
                <div style={{ fontSize: 48, marginBottom: 16 }}>✅</div>
                <h3 style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 8 }}>Email client opened</h3>
                <p style={{ color: 'var(--text-muted)', fontSize: 14 }}>Your default mail app should have opened with the message pre-filled. Send it and we'll get back to you shortly.</p>
                <button className="btn-secondary" style={{ marginTop: 24 }} onClick={() => setSent(false)}>Send another</button>
              </div>
            ) : (
              <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
                <h2 style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>Send us a message</h2>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                  <div>
                    <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>Your name</label>
                    <input className="input-dark" style={{ width: '100%' }} placeholder="Ahmed" value={form.name}
                      onChange={e => setForm(f => ({ ...f, name: e.target.value }))} required />
                  </div>
                  <div>
                    <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>Email address</label>
                    <input className="input-dark" style={{ width: '100%' }} type="email" placeholder="ahmed@company.com" value={form.email}
                      onChange={e => setForm(f => ({ ...f, email: e.target.value }))} required />
                  </div>
                </div>

                <div>
                  <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>Subject</label>
                  <input className="input-dark" style={{ width: '100%' }} placeholder="Enterprise plan / Research collaboration / General question" value={form.subject}
                    onChange={e => setForm(f => ({ ...f, subject: e.target.value }))} />
                </div>

                <div>
                  <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>Message</label>
                  <textarea className="input-dark" style={{ width: '100%', minHeight: 140, resize: 'vertical' }}
                    placeholder="Tell us about your team, your use case, or any questions you have…"
                    value={form.message} onChange={e => setForm(f => ({ ...f, message: e.target.value }))} required />
                </div>

                <button type="submit" className="btn-primary" style={{ padding: '12px', width: '100%', fontSize: 14 }}>
                  Send message
                </button>
                <p style={{ fontSize: 12, color: 'var(--text-subtle)', textAlign: 'center' }}>
                  This will open your email client with the message pre-filled.
                </p>
              </form>
            )}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer style={{ borderTop: '1px solid var(--border)', padding: '24px', textAlign: 'center' }}>
        <div style={{ display: 'flex', gap: 24, justifyContent: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
          <Link to="/" style={{ fontSize: 13, color: 'var(--text-muted)', textDecoration: 'none' }}>Home</Link>
          <Link to="/about" style={{ fontSize: 13, color: 'var(--text-muted)', textDecoration: 'none' }}>About</Link>
          <Link to="/privacy" style={{ fontSize: 13, color: 'var(--text-muted)', textDecoration: 'none' }}>Privacy Policy</Link>
        </div>
        <p style={{ fontSize: 12, color: 'var(--text-subtle)' }}>© 2026 VCAI — Graduation Project, Misr International University</p>
      </footer>

      <style>{`
        @media (max-width: 768px) {
          section > div[style*="grid-template-columns: 1fr 1.6fr"] {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </div>
  );
}
