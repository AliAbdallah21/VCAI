import { Link } from 'react-router-dom';

const SECTIONS = [
  {
    title: 'Information We Collect',
    body: `When you register, we collect your name, email address, and company name. During training sessions, we process audio recordings for real-time speech-to-text transcription. We also store session transcripts, emotion analysis results, and evaluation scores to power your progress tracking. Audio data is processed in memory and is not permanently stored on our servers.`,
  },
  {
    title: 'How We Use Your Information',
    body: `We use your information to: provide and improve the VCAI training platform; generate personalised evaluation reports and coaching feedback; display your skill progress to you and (if applicable) your team manager; communicate important service updates. We do not sell your personal data to third parties.`,
  },
  {
    title: 'Audio & Voice Data',
    body: `Audio recordings are used solely to transcribe your speech during training sessions using our on-device Faster-Whisper model. Transcriptions are stored to enable evaluation and memory features. Raw audio is not permanently stored after transcription is complete. You may request deletion of your session data at any time by contacting us.`,
  },
  {
    title: 'Data Sharing',
    body: `We share data with: your workspace manager (session scores and skill summaries only, not raw transcripts); cloud AI providers for generating evaluation feedback (data is sent over encrypted connections and not retained by providers beyond the request). We do not share your data with advertisers or unrelated third parties.`,
  },
  {
    title: 'Data Security',
    body: `All data is transmitted over HTTPS/TLS. Passwords are hashed using industry-standard bcrypt. Database access is restricted and audited. We apply the principle of least privilege across all services. While we take security seriously, no system is perfectly secure — please use a strong, unique password.`,
  },
  {
    title: 'Data Retention',
    body: `Account data is retained for as long as your account is active. You may request account deletion at any time; upon deletion, your personal data and session history will be removed within 30 days. Anonymised, aggregated statistics may be retained for research purposes.`,
  },
  {
    title: 'Your Rights',
    body: `You have the right to access, correct, or delete your personal data. You may export your session history and evaluation reports at any time from within the platform. To exercise any right or make a data request, contact us at gradproject11234@gmail.com.`,
  },
  {
    title: 'Cookies',
    body: `We use only essential session cookies required for authentication. We do not use advertising, tracking, or analytics cookies. You can disable cookies in your browser settings, but this will prevent you from logging in.`,
  },
  {
    title: 'Changes to This Policy',
    body: `We may update this Privacy Policy from time to time. We will notify registered users of significant changes via email. Continued use of the platform after changes constitutes acceptance of the updated policy. The effective date at the top of this page reflects the latest revision.`,
  },
  {
    title: 'Contact',
    body: `For any privacy-related questions or requests, email us at gradproject11234@gmail.com or use the Contact page. We aim to respond within 5 business days.`,
  },
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

export default function Privacy() {
  return (
    <div style={{ background: 'var(--bg-app)', color: 'var(--text-primary)', minHeight: '100vh' }}>

      <header style={{ borderBottom: '1px solid var(--border)', padding: '0 24px' }}>
        <div className="max-w-4xl mx-auto h-16 flex items-center justify-between">
          <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: 10, textDecoration: 'none' }}>
            <VCAILogo size={30} />
            <span style={{ fontWeight: 700, fontSize: 16, color: 'var(--text-primary)' }}>VCAI</span>
          </Link>
          <Link to="/" style={{ fontSize: 13, color: 'var(--text-muted)', textDecoration: 'none' }}>← Back to home</Link>
        </div>
      </header>

      <main style={{ maxWidth: 720, margin: '0 auto', padding: '64px 24px' }}>
        <div style={{ marginBottom: 48 }}>
          <h1 style={{ fontSize: 36, fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.02em', marginBottom: 12 }}>
            Privacy Policy
          </h1>
          <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>Effective date: June 2026 · Misr International University — VCAI Graduation Project</p>
          <p style={{ color: 'var(--text-muted)', marginTop: 16, lineHeight: 1.8 }}>
            VCAI ("we", "us", "our") is committed to protecting your privacy. This policy explains what data
            we collect, how we use it, and your rights regarding your information.
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 32 }}>
          {SECTIONS.map((s, i) => (
            <div key={s.title} style={{ paddingBottom: 32, borderBottom: i < SECTIONS.length - 1 ? '1px solid var(--border)' : 'none' }}>
              <h2 style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 12 }}>
                {i + 1}. {s.title}
              </h2>
              <p style={{ color: 'var(--text-muted)', lineHeight: 1.8, fontSize: 14 }}>{s.body}</p>
            </div>
          ))}
        </div>
      </main>

      <footer style={{ borderTop: '1px solid var(--border)', padding: '24px', textAlign: 'center' }}>
        <div style={{ display: 'flex', gap: 24, justifyContent: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
          <Link to="/" style={{ fontSize: 13, color: 'var(--text-muted)', textDecoration: 'none' }}>Home</Link>
          <Link to="/about" style={{ fontSize: 13, color: 'var(--text-muted)', textDecoration: 'none' }}>About</Link>
          <Link to="/contact" style={{ fontSize: 13, color: 'var(--text-muted)', textDecoration: 'none' }}>Contact</Link>
        </div>
        <p style={{ fontSize: 12, color: 'var(--text-subtle)' }}>© 2026 VCAI — Graduation Project, Misr International University</p>
      </footer>
    </div>
  );
}
