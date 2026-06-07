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
    <div className="min-h-screen bg-slate-50 py-10 px-4">
      <div className="max-w-2xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-9 h-9 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold">V</span>
            </div>
            <span className="font-bold text-xl text-slate-800">VCAI</span>
          </Link>
          <span className="text-xs text-amber-700 bg-amber-50 border border-amber-200 px-3 py-1.5 rounded-full">
            Demo billing - Stripe integration coming later
          </span>
        </div>

        <div className="flex items-center gap-2 mb-6">
          {stepLabels.map((label, i) => (
            <div key={label} className="flex items-center gap-2">
              <span
                className={`w-7 h-7 rounded-full flex items-center justify-center text-sm font-semibold ${
                  step >= i + 1 ? 'bg-blue-600 text-white' : 'bg-slate-200 text-slate-500'
                }`}
              >
                {i + 1}
              </span>
              <span className={`text-sm ${step >= i + 1 ? 'text-slate-800' : 'text-slate-400'}`}>
                {label}
              </span>
              {i < stepLabels.length - 1 && <span className="w-6 h-px bg-slate-300" />}
            </div>
          ))}
        </div>

        {error && (
          <div className="bg-red-50 text-red-600 px-4 py-3 rounded-xl text-sm mb-4">{error}</div>
        )}

        {/* Step 1 - Plan confirm */}
        {step === 1 && (
          <div className="bg-white rounded-2xl shadow-sm p-6">
            <h2 className="text-xl font-bold text-slate-800 mb-1">Confirm your plan</h2>
            <p className="text-slate-500 text-sm mb-6">You can switch plans before continuing.</p>

            <div className="flex items-center justify-center gap-4 mb-6">
              <span className={`text-sm font-medium ${annual ? 'text-slate-400' : 'text-slate-800'}`}>Monthly</span>
              <button
                type="button"
                role="switch"
                aria-checked={annual}
                onClick={() => setAnnual((v) => !v)}
                className={`relative w-14 h-7 rounded-full transition ${annual ? 'bg-blue-600' : 'bg-slate-300'}`}
              >
                <span
                  className={`absolute top-0.5 left-0.5 w-6 h-6 bg-white rounded-full shadow transition-transform ${annual ? 'translate-x-7' : ''}`}
                />
              </button>
              <span className={`text-sm font-medium ${annual ? 'text-slate-800' : 'text-slate-400'}`}>
                Annual <span className="ml-1 text-xs text-emerald-600 font-semibold">save ~2 months</span>
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
                    className={`text-left rounded-xl p-4 border transition ${
                      planName === p.name
                        ? 'border-blue-600 ring-2 ring-blue-100 bg-blue-50'
                        : 'border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-slate-800">{p.display_name}</span>
                      <span className="font-bold text-slate-800">{fmtPrice(p, annual)}</span>
                    </div>
                    <p className="text-xs text-slate-500 mt-2">
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
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-xl font-medium hover:opacity-90 disabled:opacity-50"
            >
              Continue
            </button>
          </div>
        )}

        {/* Step 2 - Company + manager details */}
        {step === 2 && (
          <form onSubmit={afterDetails} className="bg-white rounded-2xl shadow-sm p-6 space-y-5">
            <div>
              <h2 className="text-xl font-bold text-slate-800 mb-1">Your details</h2>
              <p className="text-slate-500 text-sm">
                You will be the manager of this workspace ({selectedPlan?.display_name} plan).
              </p>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Company name</label>
              <input
                value={companyName}
                onChange={(e) => setCompanyName(e.target.value)}
                className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl outline-none focus:ring-2 focus:ring-blue-500 text-slate-900 placeholder:text-slate-400"
                placeholder="Acme Real Estate"
                required
                minLength={2}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Your name</label>
              <input
                value={managerName}
                onChange={(e) => setManagerName(e.target.value)}
                className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl outline-none focus:ring-2 focus:ring-blue-500 text-slate-900 placeholder:text-slate-400"
                placeholder="Jane Manager"
                required
                minLength={2}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Work email</label>
              <input
                type="email"
                value={managerEmail}
                onChange={(e) => setManagerEmail(e.target.value)}
                className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl outline-none focus:ring-2 focus:ring-blue-500 text-slate-900 placeholder:text-slate-400"
                placeholder="you@company.com"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl outline-none focus:ring-2 focus:ring-blue-500 text-slate-900 placeholder:text-slate-400"
                placeholder="At least 6 characters"
                required
                minLength={6}
              />
            </div>
            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => setStep(1)}
                className="px-5 py-3 rounded-xl font-medium bg-slate-100 text-slate-700 hover:bg-slate-200"
              >
                Back
              </button>
              <button
                type="submit"
                disabled={loading}
                className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-xl font-medium hover:opacity-90 disabled:opacity-50"
              >
                {isFree ? (loading ? 'Creating workspace...' : 'Create workspace') : 'Continue to checkout'}
              </button>
            </div>
          </form>
        )}

        {/* Step 3 - Mocked payment (paid plans only) */}
        {step === 3 && !isFree && (
          <div className="bg-white rounded-2xl shadow-sm p-6 space-y-5">
            <div>
              <h2 className="text-xl font-bold text-slate-800 mb-1">Checkout</h2>
              <p className="text-slate-500 text-sm">
                {selectedPlan?.display_name} · {annual ? 'Annual' : 'Monthly'} ·{' '}
                {fmtPrice(selectedPlan, annual)} {annual ? '/ year' : '/ month'}
              </p>
            </div>

            <div className="rounded-xl border-2 border-dashed border-amber-300 bg-amber-50 p-5">
              <p className="text-sm font-semibold text-amber-800 mb-3">Demo checkout - no card charged</p>
              <div className="space-y-3 opacity-60 pointer-events-none select-none">
                <div>
                  <label className="block text-xs font-medium text-slate-600 mb-1">Card number</label>
                  <input
                    disabled
                    value="4242 4242 4242 4242"
                    className="w-full px-4 py-2.5 bg-white border border-slate-200 rounded-lg"
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-medium text-slate-600 mb-1">Expiry</label>
                    <input disabled value="12 / 30" className="w-full px-4 py-2.5 bg-white border border-slate-200 rounded-lg" />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-slate-600 mb-1">CVC</label>
                    <input disabled value="123" className="w-full px-4 py-2.5 bg-white border border-slate-200 rounded-lg" />
                  </div>
                </div>
              </div>
            </div>

            <p className="text-xs text-slate-500">
              Demo billing - Stripe integration coming later. Your 14-day trial starts now; no payment is captured.
            </p>

            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => setStep(2)}
                className="px-5 py-3 rounded-xl font-medium bg-slate-100 text-slate-700 hover:bg-slate-200"
              >
                Back
              </button>
              <button
                type="button"
                onClick={submit}
                disabled={loading}
                className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-xl font-medium hover:opacity-90 disabled:opacity-50"
              >
                {loading ? 'Starting trial...' : 'Start 14-day trial'}
              </button>
            </div>
          </div>
        )}

        <p className="text-center mt-6 text-sm text-slate-500">
          Already have an account?{' '}
          <Link to="/login" className="text-blue-600 font-medium hover:underline">Sign in</Link>
        </p>
      </div>
    </div>
  );
}
