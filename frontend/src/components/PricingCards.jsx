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
      className="relative flex flex-col ds-card p-6"
      style={recommended ? { border: '2px solid #b472f1', background: 'var(--primary-soft)' } : undefined}
    >
      {recommended && (
        <span className="absolute -top-3 left-1/2 -translate-x-1/2 btn-primary text-xs font-semibold px-3 py-1 rounded-full">
          Most popular
        </span>
      )}

      <h3 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>{plan.display_name}</h3>

      <div className="mt-4 mb-6">
        <span className="text-4xl font-bold" style={{ color: 'var(--text-primary)' }}>{price}</span>
        {showPeriod && (
          <span className="text-sm ml-1" style={{ color: 'var(--text-muted)' }}>/ {annual ? 'year' : 'month'}</span>
        )}
      </div>

      <ul className="space-y-3 mb-6 flex-1">
        {planFeatures(plan).map((feature) => (
          <li key={feature} className="flex items-start gap-2 text-sm" style={{ color: 'var(--text-secondary)' }}>
            <span className="mt-0.5" style={{ color: 'var(--primary)' }}>✓</span>
            <span>{feature}</span>
          </li>
        ))}
      </ul>

      <button
        type="button"
        onClick={() => onSelect(plan.name)}
        className={`w-full ${recommended ? 'btn-primary' : 'btn-secondary'}`}
        style={{ padding: '12px 0' }}
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
        <span className="text-sm font-medium" style={{ color: annual ? 'var(--text-muted)' : 'var(--text-primary)' }}>
          Monthly
        </span>
        <button
          type="button"
          role="switch"
          aria-checked={annual}
          onClick={() => setAnnual((v) => !v)}
          className="relative w-14 h-7 rounded-full transition"
          style={{ background: annual ? '#b472f1' : 'var(--surface-container-highest)' }}
        >
          <span
            className={`absolute top-0.5 left-0.5 w-6 h-6 bg-white rounded-full shadow transition-transform ${
              annual ? 'translate-x-7' : ''
            }`}
          />
        </button>
        <span className="text-sm font-medium" style={{ color: annual ? 'var(--text-primary)' : 'var(--text-muted)' }}>
          Annual
          <span className="ml-2 text-xs font-semibold" style={{ color: 'var(--success-green)' }}>save ~2 months</span>
        </span>
      </div>

      {freePlan && (
        <div className="mb-8 flex flex-col sm:flex-row items-center justify-between gap-4 ds-card px-6 py-5">
          <div>
            <h3 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>{freePlan.display_name}</h3>
            <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
              {planFeatures(freePlan).join(' · ')}
            </p>
          </div>
          <div className="flex items-center gap-4 shrink-0">
            <span className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>{fmtPrice(freePlan, annual)}</span>
            <button
              type="button"
              onClick={() => handleSelect(freePlan.name)}
              className="btn-secondary"
              style={{ padding: '10px 20px' }}
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
