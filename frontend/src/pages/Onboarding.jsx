import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { getPlans, onboardingAPI } from '../services/api';

const UNLIMITED = 1_000_000;
const fmtLimit = (n) => (n >= UNLIMITED ? 'Unlimited' : String(n));

const fmtPrice = (plan, annual) => {
  const value = annual ? plan.price_annual_usd : plan.price_monthly_usd;
  if (value === null || value === undefined) return 'Contact us';
  if (value === 0) return '$0';
  return `$${value}`;
};

export default function Onboarding() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { setAuth } = useAuth();

  const [plans, setPlans] = useState([]);
  const [planName, setPlanName] = useState(searchParams.get('plan') || 'free');
  const [annual, setAnnual] = useState(false);
  const [step, setStep] = useState(1);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const [companyName, setCompanyName] = useState('');
  const [managerName, setManagerName] = useState('');
  const [managerEmail, setManagerEmail] = useState('');
  const [password, setPassword] = useState('');

  useEffect(() => {
    getPlans()
      .then((data) => {
        setPlans(data);
        // Enterprise is contact-sales only; fall back to growth.
        if (planName === 'enterprise') setPlanName('growth');
      })
      .catch(() => setError('Could not load plans'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedPlan = useMemo(
    () => plans.find((p) => p.name === planName),
    [plans, planName]
  );
  const isFree = planName === 'free';

  const submit = async () => {
    setError('');
    setLoading(true);
    try {
      const result = await onboardingAPI.signup({
        company_name: companyName,
        plan_name: planName,
        billing_cycle: annual ? 'annual' : 'monthly',
        manager_name: managerName,
        manager_email: managerEmail,
        password,
      });
      setAuth(result);
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not create your workspace');
    } finally {
      setLoading(false);
    }
  };

  const goToDetails = (e) => {
    e.preventDefault();
    setStep(2);
  };

  const afterDetails = (e) => {
    e.preventDefault();
    setError('');
    // Free plans skip the mocked-payment step.
    if (isFree) {
      submit();
    } else {
      setStep(3);
    }
  };

  const stepLabels = isFree ? ['Plan', 'Your details'] : ['Plan', 'Your details', 'Checkout'];

  return (
    <div className="min-h-screen py-10 px-4" style={{ background: 'var(--bg-app)' }}>
      <div className="max-w-2xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #b472f1, #deb7ff)' }}>
              <span className="font-bold" style={{ color: '#4a007f' }}>V</span>
            </div>
            <span className="font-bold text-xl" style={{ color: 'var(--text-primary)' }}>VCAI</span>
          </Link>
          <span className="text-xs px-3 py-1.5 rounded-full" style={{ color: 'var(--warning)', background: 'rgba(233,196,106,0.1)', border: '1px solid rgba(233,196,106,0.25)' }}>
            Demo billing - Stripe integration coming later
          </span>
        </div>

        <div className="flex items-center gap-2 mb-6">
          {stepLabels.map((label, i) => (
            <div key={label} className="flex items-center gap-2">
              <span
                className="w-7 h-7 rounded-full flex items-center justify-center text-sm font-semibold"
                style={step >= i + 1 ? { background: '#b472f1', color: '#4a007f' } : { background: 'var(--surface-container-highest)', color: 'var(--text-muted)' }}
              >
                {i + 1}
              </span>
              <span className="text-sm" style={{ color: step >= i + 1 ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                {label}
              </span>
              {i < stepLabels.length - 1 && <span className="w-6 h-px" style={{ background: 'var(--border-strong)' }} />}
            </div>
          ))}
        </div>

        {error && (
          <div className="px-4 py-3 rounded-xl text-sm mb-4" style={{ color: 'var(--error)', background: 'rgba(255,180,171,0.08)', border: '1px solid rgba(255,180,171,0.25)' }}>{error}</div>
        )}

        {/* Step 1 - Plan confirm */}
        {step === 1 && (
          <div className="ds-card p-6">
            <h2 className="text-xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>Confirm your plan</h2>
            <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>You can switch plans before continuing.</p>

            <div className="flex items-center justify-center gap-4 mb-6">
              <span className="text-sm font-medium" style={{ color: annual ? 'var(--text-muted)' : 'var(--text-primary)' }}>Monthly</span>
              <button
                type="button"
                role="switch"
                aria-checked={annual}
                onClick={() => setAnnual((v) => !v)}
                className="relative w-14 h-7 rounded-full transition"
                style={{ background: annual ? '#b472f1' : 'var(--surface-container-highest)' }}
              >
                <span
                  className={`absolute top-0.5 left-0.5 w-6 h-6 bg-white rounded-full shadow transition-transform ${annual ? 'translate-x-7' : ''}`}
                />
              </button>
              <span className="text-sm font-medium" style={{ color: annual ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                Annual <span className="ml-1 text-xs font-semibold" style={{ color: 'var(--success-green)' }}>save ~2 months</span>
              </span>
            </div>

            <div className="grid sm:grid-cols-2 gap-3 mb-6">
              {plans
                .filter((p) => p.name !== 'enterprise')
                .map((p) => (
                  <button
                    key={p.name}
                    type="button"
                    onClick={() => setPlanName(p.name)}
                    className="text-left rounded-xl p-4 border transition"
                    style={planName === p.name
                      ? { borderColor: '#b472f1', background: 'var(--primary-soft)', boxShadow: '0 0 0 2px rgba(222,183,255,0.15)' }
                      : { borderColor: 'var(--border)' }}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>{p.display_name}</span>
                      <span className="font-bold" style={{ color: 'var(--text-primary)' }}>{fmtPrice(p, annual)}</span>
                    </div>
                    <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                      {fmtLimit(p.seat_limit)} seats · {fmtLimit(p.session_limit_monthly)} sessions/mo ·{' '}
                      {p.name === 'free' ? '2 personas' : 'All 12 personas'}
                    </p>
                  </button>
                ))}
            </div>

            <button
              type="button"
              onClick={goToDetails}
              disabled={!selectedPlan}
              className="btn-primary w-full"
              style={{ padding: '12px 0' }}
            >
              Continue
            </button>
          </div>
        )}

        {/* Step 2 - Company + manager details */}
        {step === 2 && (
          <form onSubmit={afterDetails} className="ds-card p-6 space-y-5">
            <div>
              <h2 className="text-xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>Your details</h2>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                You will be the manager of this workspace ({selectedPlan?.display_name} plan).
              </p>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Company name</label>
              <input
                value={companyName}
                onChange={(e) => setCompanyName(e.target.value)}
                className="input-dark w-full px-4 py-3"
                placeholder="Acme Real Estate"
                required
                minLength={2}
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Your name</label>
              <input
                value={managerName}
                onChange={(e) => setManagerName(e.target.value)}
                className="input-dark w-full px-4 py-3"
                placeholder="Jane Manager"
                required
                minLength={2}
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Work email</label>
              <input
                type="email"
                value={managerEmail}
                onChange={(e) => setManagerEmail(e.target.value)}
                className="input-dark w-full px-4 py-3"
                placeholder="you@company.com"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="input-dark w-full px-4 py-3"
                placeholder="At least 6 characters"
                required
                minLength={6}
              />
            </div>
            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => setStep(1)}
                className="btn-secondary"
                style={{ padding: '12px 20px' }}
              >
                Back
              </button>
              <button
                type="submit"
                disabled={loading}
                className="btn-primary flex-1"
                style={{ padding: '12px 0' }}
              >
                {isFree ? (loading ? 'Creating workspace...' : 'Create workspace') : 'Continue to checkout'}
              </button>
            </div>
          </form>
        )}

        {/* Step 3 - Mocked payment (paid plans only) */}
        {step === 3 && !isFree && (
          <div className="ds-card p-6 space-y-5">
            <div>
              <h2 className="text-xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>Checkout</h2>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                {selectedPlan?.display_name} · {annual ? 'Annual' : 'Monthly'} ·{' '}
                {fmtPrice(selectedPlan, annual)} {annual ? '/ year' : '/ month'}
              </p>
            </div>

            <div className="rounded-xl border-2 border-dashed p-5" style={{ borderColor: 'rgba(233,196,106,0.4)', background: 'rgba(233,196,106,0.08)' }}>
              <p className="text-sm font-semibold mb-3" style={{ color: 'var(--warning)' }}>Demo checkout - no card charged</p>
              <div className="space-y-3 opacity-60 pointer-events-none select-none">
                <div>
                  <label className="block text-xs font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>Card number</label>
                  <input
                    disabled
                    value="4242 4242 4242 4242"
                    className="input-dark w-full px-4 py-2.5"
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>Expiry</label>
                    <input disabled value="12 / 30" className="input-dark w-full px-4 py-2.5" />
                  </div>
                  <div>
                    <label className="block text-xs font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>CVC</label>
                    <input disabled value="123" className="input-dark w-full px-4 py-2.5" />
                  </div>
                </div>
              </div>
            </div>

            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Demo billing - Stripe integration coming later. Your 14-day trial starts now; no payment is captured.
            </p>

            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => setStep(2)}
                className="btn-secondary"
                style={{ padding: '12px 20px' }}
              >
                Back
              </button>
              <button
                type="button"
                onClick={submit}
                disabled={loading}
                className="btn-primary flex-1"
                style={{ padding: '12px 0' }}
              >
                {loading ? 'Starting trial...' : 'Start 14-day trial'}
              </button>
            </div>
          </div>
        )}

        <p className="text-center mt-6 text-sm" style={{ color: 'var(--text-muted)' }}>
          Already have an account?{' '}
          <Link to="/login" className="font-medium hover:underline" style={{ color: 'var(--primary)' }}>Sign in</Link>
        </p>
      </div>
    </div>
  );
}
