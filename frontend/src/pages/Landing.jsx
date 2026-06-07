import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { getPlans } from '../services/api';
import PricingCards from '../components/PricingCards';

const STEPS = [
  {
    title: 'Sign up your firm',
    body: 'Create a workspace for your real-estate sales team in minutes.',
  },
  {
    title: 'Invite your agents',
    body: 'Add salespeople to your workspace within your plan seat limit.',
  },
  {
    title: 'Train and track',
    body: 'Agents practice against AI customers while you track their progress.',
  },
];

const FEATURES = [
  {
    title: 'Egyptian-dialect voice',
    body: 'A Chatterbox fine-tune speaks natural Egyptian Arabic for realistic role-play.',
  },
  {
    title: 'Emotional reaction',
    body: 'Dual-modal emotion detection reads tone and text so customers react like real people.',
  },
  {
    title: 'Cross-session memory',
    body: 'The virtual customer remembers prior conversations across training sessions.',
  },
  {
    title: 'Automated scoring',
    body: 'Every session is scored across eight sales skills with actionable feedback.',
  },
];

function Header({ isAuthenticated }) {
  return (
    <header className="glass sticky top-0 z-10" style={{ borderBottom: '1px solid var(--border)' }}>
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #b472f1, #deb7ff)' }}>
            <span className="font-bold text-lg" style={{ color: '#4a007f' }}>V</span>
          </div>
          <span className="font-bold text-xl" style={{ color: 'var(--text-primary)' }}>VCAI</span>
        </div>
        <nav className="flex items-center gap-3">
          <a href="#pricing" className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Pricing
          </a>
          {isAuthenticated ? (
            <Link
              to="/dashboard"
              className="btn-primary"
            >
              Go to dashboard
            </Link>
          ) : (
            <>
              <Link to="/login" className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
                Sign in
              </Link>
              <Link
                to="/onboarding?plan=free"
                className="btn-primary"
              >
                Start free
              </Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}

export default function Landing() {
  const { isAuthenticated } = useAuth();
  const [plans, setPlans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    getPlans()
      .then((data) => setPlans(data))
      .catch(() => setError('Could not load pricing. Please try again later.'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg-app)', color: 'var(--text-primary)' }}>
      <Header isAuthenticated={isAuthenticated} />

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-6 pt-20 pb-24 text-center">
        <h1 className="display-special tracking-tight">
          <span className="text-gradient">
            VCAI
          </span>
        </h1>
        <p className="mt-6 text-2xl font-semibold" style={{ color: 'var(--text-primary)' }}>
          The first Egyptian-Arabic AI sales-training customer
        </p>
        <p className="mt-4 max-w-2xl mx-auto body-lg" style={{ color: 'var(--text-muted)' }}>
          Give your real-estate agents a tireless practice partner that talks, reacts, and
          remembers, then measure their progress automatically.
        </p>
        <div className="mt-10 flex items-center justify-center gap-4">
          <Link
            to="/onboarding?plan=free"
            className="btn-primary"
            style={{ padding: '12px 24px' }}
          >
            Start free
          </Link>
          <a
            href="#pricing"
            className="btn-secondary"
            style={{ padding: '12px 24px' }}
          >
            See pricing
          </a>
        </div>
      </section>

      {/* How it works */}
      <section className="max-w-6xl mx-auto px-6 py-16">
        <h2 className="headline-md text-center mb-12" style={{ color: 'var(--text-primary)' }}>How it works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {STEPS.map((step, i) => (
            <div key={step.title} className="ds-card p-6">
              <div className="w-10 h-10 rounded-xl font-bold flex items-center justify-center mb-4" style={{ background: 'var(--primary-soft)', color: 'var(--primary)' }}>
                {i + 1}
              </div>
              <h3 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>{step.title}</h3>
              <p className="mt-2" style={{ color: 'var(--text-muted)' }}>{step.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="max-w-6xl mx-auto px-6 py-16">
        <h2 className="headline-md text-center mb-12" style={{ color: 'var(--text-primary)' }}>Built for realistic practice</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {FEATURES.map((feature) => (
            <div key={feature.title} className="ds-card p-6">
              <h3 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>{feature.title}</h3>
              <p className="mt-2 text-sm" style={{ color: 'var(--text-muted)' }}>{feature.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Pricing */}
      <section id="pricing" className="max-w-6xl mx-auto px-6 py-16 scroll-mt-20">
        <h2 className="headline-md text-center mb-3" style={{ color: 'var(--text-primary)' }}>Simple, transparent pricing</h2>
        <p className="text-center mb-12" style={{ color: 'var(--text-muted)' }}>Pick a plan that fits your team.</p>

        {loading && <p className="text-center" style={{ color: 'var(--text-muted)' }}>Loading plans...</p>}
        {error && !loading && (
          <p className="text-center rounded-xl py-4 max-w-md mx-auto" style={{ color: 'var(--error)', background: 'rgba(255,180,171,0.08)' }}>
            {error}
          </p>
        )}
        {!loading && !error && plans.length > 0 && <PricingCards plans={plans} />}
      </section>

      {/* Footer */}
      <footer className="mt-16" style={{ borderTop: '1px solid var(--border)' }}>
        <div className="max-w-6xl mx-auto px-6 py-8 text-center text-sm" style={{ color: 'var(--text-subtle)' }}>
          VCAI — Graduation Project, Misr International University, 2026
        </div>
      </footer>
    </div>
  );
}
