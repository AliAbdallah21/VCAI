import { Link } from 'react-router-dom';

export default function Contact() {
  return (
    <div className="min-h-screen flex items-center justify-center p-8" style={{ background: 'var(--bg-app)' }}>
      <div className="ds-card p-10 max-w-md text-center">
        <div className="inline-flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #b472f1, #deb7ff)' }}>
            <span className="font-bold text-lg" style={{ color: '#4a007f' }}>V</span>
          </div>
          <span className="font-bold text-xl" style={{ color: 'var(--text-primary)' }}>VCAI</span>
        </div>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>Talk to us about Enterprise</h1>
        <p className="mt-3" style={{ color: 'var(--text-muted)' }}>
          For enterprise plans and custom support, reach out and our team will get back to you.
        </p>
        <a
          href="mailto:fitai.sub@gmail.com"
          className="btn-primary inline-block mt-6"
          style={{ padding: '12px 24px' }}
        >
          Email us
        </a>
        <div className="mt-6">
          <Link to="/" className="text-sm font-medium hover:underline" style={{ color: 'var(--primary)' }}>
            Back to home
          </Link>
        </div>
      </div>
    </div>
  );
}
