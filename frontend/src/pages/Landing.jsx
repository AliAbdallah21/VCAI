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
    <header className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b border-slate-100">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-lg">V</span>
          </div>
          <span className="font-bold text-xl text-slate-800">VCAI</span>
        </div>
        <nav className="flex items-center gap-3">
          <a href="#pricing" className="text-sm font-medium text-slate-600 hover:text-slate-900">
            Pricing
          </a>
          {isAuthenticated ? (
            <Link
              to="/dashboard"
              className="px-4 py-2 rounded-xl text-sm font-medium bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:opacity-90"
            >
              Go to dashboard
            </Link>
          ) : (
            <>
              <Link to="/login" className="text-sm font-medium text-slate-600 hover:text-slate-900">
                Sign in
              </Link>
              <Link
                to="/onboarding?plan=free"
                className="px-4 py-2 rounded-xl text-sm font-medium bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:opacity-90"
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
    <div className="min-h-screen bg-slate-50 text-slate-800">
      <Header isAuthenticated={isAuthenticated} />

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-6 pt-20 pb-24 text-center">
        <h1 className="text-5xl sm:text-6xl font-bold tracking-tight">
          <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            VCAI
          </span>
        </h1>
        <p className="mt-6 text-2xl font-semibold text-slate-800">
          The first Egyptian-Arabic AI sales-training customer
        </p>
        <p className="mt-4 max-w-2xl mx-auto text-lg text-slate-500">
          Give your real-estate agents a tireless practice partner that talks, reacts, and
          remembers, then measure their progress automatically.
        </p>
        <div className="mt-10 flex items-center justify-center gap-4">
          <Link
            to="/onboarding?plan=free"
            className="px-6 py-3 rounded-xl font-medium bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:opacity-90"
          >
            Start free
          </Link>
          <a
            href="#pricing"
            className="px-6 py-3 rounded-xl font-medium bg-white border border-slate-200 text-slate-800 hover:bg-slate-100"
          >
            See pricing
          </a>
        </div>
      </section>

      {/* How it works */}
      <section className="max-w-6xl mx-auto px-6 py-16">
        <h2 className="text-3xl font-bold text-center mb-12">How it works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {STEPS.map((step, i) => (
            <div key={step.title} className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100">
              <div className="w-10 h-10 rounded-xl bg-blue-100 text-blue-600 font-bold flex items-center justify-center mb-4">
                {i + 1}
              </div>
              <h3 className="text-lg font-bold text-slate-800">{step.title}</h3>
              <p className="mt-2 text-slate-500">{step.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="max-w-6xl mx-auto px-6 py-16">
        <h2 className="text-3xl font-bold text-center mb-12">Built for realistic practice</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {FEATURES.map((feature) => (
            <div key={feature.title} className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100">
              <h3 className="text-lg font-bold text-slate-800">{feature.title}</h3>
              <p className="mt-2 text-slate-500 text-sm">{feature.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Pricing */}
      <section id="pricing" className="max-w-6xl mx-auto px-6 py-16 scroll-mt-20">
        <h2 className="text-3xl font-bold text-center mb-3">Simple, transparent pricing</h2>
        <p className="text-center text-slate-500 mb-12">Pick a plan that fits your team.</p>

        {loading && <p className="text-center text-slate-500">Loading plans...</p>}
        {error && !loading && (
          <p className="text-center text-red-600 bg-red-50 rounded-xl py-4 max-w-md mx-auto">
            {error}
          </p>
        )}
        {!loading && !error && plans.length > 0 && <PricingCards plans={plans} />}
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-100 mt-16">
        <div className="max-w-6xl mx-auto px-6 py-8 text-center text-sm text-slate-400">
          VCAI — Graduation Project, Misr International University, 2026
        </div>
      </footer>
    </div>
  );
}
