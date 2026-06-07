import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const UNLIMITED = 1_000_000;

const fmtLimit = (n) => (n >= UNLIMITED ? 'Unlimited' : String(n));

const fmtPrice = (plan, annual) => {
  const value = annual ? plan.price_annual_usd : plan.price_monthly_usd;
  if (value === null || value === undefined) return 'Contact us';
  if (value === 0) return '$0';
  return `$${value}`;
};

const planFeatures = (plan) => [
  `${fmtLimit(plan.seat_limit)} ${plan.seat_limit === 1 ? 'seat' : 'seats'}`,
  `${fmtLimit(plan.session_limit_monthly)} sessions / month`,
  plan.name === 'free' ? '2 personas (Easy)' : 'All 12 personas',
  plan.gaas_enabled ? 'Agentic auto-training' : 'Manual training',
];

function PlanCard({ plan, annual, recommended, onSelect }) {
  const isEnterprise = plan.name === 'enterprise';
  const isContact = plan.price_monthly_usd === null || plan.price_monthly_usd === undefined;
  const price = fmtPrice(plan, annual);
  const showPeriod = !isContact && plan.price_monthly_usd !== 0;

  return (
    <div
      className={`relative flex flex-col bg-white rounded-2xl p-6 shadow-sm ${
        recommended ? 'border-2 border-blue-600' : 'border border-slate-100'
      }`}
    >
      {recommended && (
        <span className="absolute -top-3 left-1/2 -translate-x-1/2 bg-gradient-to-r from-blue-600 to-purple-600 text-white text-xs font-semibold px-3 py-1 rounded-full">
          Most popular
        </span>
      )}

      <h3 className="text-lg font-bold text-slate-800">{plan.display_name}</h3>

      <div className="mt-4 mb-6">
        <span className="text-4xl font-bold text-slate-800">{price}</span>
        {showPeriod && (
          <span className="text-slate-500 text-sm ml-1">/ {annual ? 'year' : 'month'}</span>
        )}
      </div>

      <ul className="space-y-3 mb-6 flex-1">
        {planFeatures(plan).map((feature) => (
          <li key={feature} className="flex items-start gap-2 text-sm text-slate-600">
            <span className="text-blue-600 mt-0.5">✓</span>
            <span>{feature}</span>
          </li>
        ))}
      </ul>

      <button
        type="button"
        onClick={() => onSelect(plan.name)}
        className={`w-full py-3 rounded-xl font-medium transition ${
          recommended
            ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:opacity-90'
            : 'bg-slate-100 text-slate-800 hover:bg-slate-200'
        }`}
      >
        {isEnterprise ? 'Contact us' : 'Get started'}
      </button>
    </div>
  );
}

export default function PricingCards({ plans, onSelect }) {
  const navigate = useNavigate();
  const [annual, setAnnual] = useState(false);

  const handleSelect = (name) => {
    if (onSelect) {
      onSelect(name);
      return;
    }
    if (name === 'enterprise') {
      navigate('/contact');
      return;
    }
    navigate(`/onboarding?plan=${name}`);
  };

  const freePlan = plans.find((p) => p.name === 'free');
  const paidPlans = plans.filter((p) => p.name !== 'free');

  return (
    <div>
      <div className="flex items-center justify-center gap-4 mb-10">
        <span className={`text-sm font-medium ${annual ? 'text-slate-400' : 'text-slate-800'}`}>
          Monthly
        </span>
        <button
          type="button"
          role="switch"
          aria-checked={annual}
          onClick={() => setAnnual((v) => !v)}
          className={`relative w-14 h-7 rounded-full transition ${
            annual ? 'bg-blue-600' : 'bg-slate-300'
          }`}
        >
          <span
            className={`absolute top-0.5 left-0.5 w-6 h-6 bg-white rounded-full shadow transition-transform ${
              annual ? 'translate-x-7' : ''
            }`}
          />
        </button>
        <span className={`text-sm font-medium ${annual ? 'text-slate-800' : 'text-slate-400'}`}>
          Annual
          <span className="ml-2 text-xs text-emerald-600 font-semibold">save ~2 months</span>
        </span>
      </div>

      {freePlan && (
        <div className="mb-8 flex flex-col sm:flex-row items-center justify-between gap-4 bg-white border border-slate-100 rounded-2xl px-6 py-5 shadow-sm">
          <div>
            <h3 className="text-lg font-bold text-slate-800">{freePlan.display_name}</h3>
            <p className="text-sm text-slate-500 mt-1">
              {planFeatures(freePlan).join(' · ')}
            </p>
          </div>
          <div className="flex items-center gap-4 shrink-0">
            <span className="text-2xl font-bold text-slate-800">{fmtPrice(freePlan, annual)}</span>
            <button
              type="button"
              onClick={() => handleSelect(freePlan.name)}
              className="px-5 py-2.5 rounded-xl font-medium bg-slate-100 text-slate-800 hover:bg-slate-200 transition"
            >
              Get started
            </button>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {paidPlans.map((plan) => (
          <PlanCard
            key={plan.name}
            plan={plan}
            annual={annual}
            recommended={plan.name === 'growth'}
            onSelect={handleSelect}
          />
        ))}
      </div>
    </div>
  );
}
