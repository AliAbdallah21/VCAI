import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import { adminAPI } from '../services/api';
import DashboardShell from '../components/ui/DashboardShell';
import Tabs from '../components/ui/Tabs';
import Badge from '../components/ui/Badge';
import EmptyState from '../components/ui/EmptyState';
import ChartTooltip from '../components/ui/ChartTooltip';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

/* ── Inline icons ── */
const IcoBuilding = ({ size = 15, color }) => (
  <svg width={size} height={size} fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M3 21h18M3 7l9-4 9 4M4 11v10M20 11v10M8 11v4M12 11v4M16 11v4" />
  </svg>
);
const IcoLayers = ({ size = 15, color }) => (
  <svg width={size} height={size} fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <polygon points="12 2 2 7 12 12 22 7 12 2" /><polyline points="2 17 12 22 22 17" /><polyline points="2 12 12 17 22 12" />
  </svg>
);
const IcoActivity = ({ size = 15, color }) => (
  <svg width={size} height={size} fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
  </svg>
);
const IcoFlag = ({ size = 15, color }) => (
  <svg width={size} height={size} fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" /><line x1="4" y1="22" x2="4" y2="15" />
  </svg>
);
const IcoInbox = ({ size = 22, color = 'rgba(222,183,255,0.25)' }) => (
  <svg width={size} height={size} fill="none" stroke={color} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <polyline points="22 12 16 12 14 15 10 15 8 12 2 12" />
    <path d="M5.45 5.11L2 12v6a2 2 0 002 2h16a2 2 0 002-2v-6l-3.45-6.89A2 2 0 0016.76 4H7.24a2 2 0 00-1.79 1.11z" />
  </svg>
);
const IcoServer = ({ size = 22, color = 'rgba(222,183,255,0.25)' }) => (
  <svg width={size} height={size} fill="none" stroke={color} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <rect x="2" y="2" width="20" height="8" rx="2" ry="2" /><rect x="2" y="14" width="20" height="8" rx="2" ry="2" />
    <line x1="6" y1="6" x2="6.01" y2="6" /><line x1="6" y1="18" x2="6.01" y2="18" />
  </svg>
);

const HEALTH_COLOR = { ok: '#a5d6a7', warn: '#e9c46a', error: '#ffb4ab' };

/* ── Shared KPI card (Amethyst) ── */
function StatCard({ label, value, sub, Icon, accent }) {
  return (
    <div
      style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-card)',
        padding: '20px 22px',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 12 }}>
        <span style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)' }}>
          {label}
        </span>
        <div style={{ width: 30, height: 30, borderRadius: 8, background: `${accent}18`, border: `1px solid ${accent}28`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Icon size={14} color={accent} />
        </div>
      </div>
      <span style={{ fontSize: 30, fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.04em', lineHeight: 1 }}>
        {value}
      </span>
      {sub && <p style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 5 }}>{sub}</p>}
      <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, transparent, ${accent}40, transparent)` }} />
    </div>
  );
}

/* ── Section card ── */
function Card({ title, subtitle, children, right }) {
  return (
    <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-card)', overflow: 'hidden' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px 20px', borderBottom: '1px solid var(--border)', gap: 12 }}>
        <div>
          <p style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text-primary)' }}>{title}</p>
          {subtitle && <p style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 2 }}>{subtitle}</p>}
        </div>
        {right}
      </div>
      <div style={{ padding: '20px' }}>{children}</div>
    </div>
  );
}

/* ── Shared axis styles (Amethyst) ── */
const axisTick = { fill: '#988d9d', fontSize: 11 };
const axisLine = { stroke: 'rgba(76,68,82,0.6)' };
const gridProps = { strokeDasharray: '3 3', stroke: 'rgba(76,68,82,0.3)', vertical: false };

/* ── Tenant detail slide-over ── */
function TenantDetail({ companyId, onClose, onChanged }) {
  const [detail, setDetail] = useState(null);
  const [busy, setBusy]     = useState(false);

  const load = useCallback(() => {
    adminAPI.getTenant(companyId).then(setDetail).catch(() => setDetail(null));
  }, [companyId]);
  useEffect(() => { load(); }, [load]);

  const toggle = async () => {
    if (!detail) return;
    setBusy(true);
    try {
      if (detail.company.is_active) await adminAPI.suspendTenant(companyId);
      else await adminAPI.reactivateTenant(companyId);
      load();
      onChanged?.();
    } finally { setBusy(false); }
  };

  if (!detail) return null;
  const active = detail.company.is_active;

  return (
    <div
      style={{ position: 'fixed', inset: 0, zIndex: 40, display: 'flex', justifyContent: 'flex-end', background: 'rgba(0,0,0,0.65)' }}
      onClick={onClose}
    >
      <div
        style={{ width: '100%', maxWidth: 480, height: '100%', overflowY: 'auto', background: '#1c1b1d', borderLeft: '1px solid var(--border)', padding: '28px 24px' }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 20 }}>
          <div>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>{detail.company.name}</h2>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>{detail.company.slug}</p>
          </div>
          <button
            onClick={onClose}
            style={{ fontSize: 20, color: 'var(--text-muted)', background: 'none', border: 'none', cursor: 'pointer', lineHeight: 1, padding: '0 4px' }}
          >
            ×
          </button>
        </div>

        {/* Status + toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
          <Badge variant={active ? 'active' : 'suspended'} label={active ? 'Active' : 'Suspended'} dot />
          <button
            onClick={toggle}
            disabled={busy}
            className={active ? 'btn-danger' : 'btn-secondary'}
            style={{ marginLeft: 'auto', fontSize: 12, padding: '6px 14px', ...(active ? {} : { color: '#a5d6a7', borderColor: 'rgba(165,214,167,0.3)' }) }}
          >
            {busy ? '…' : active ? 'Suspend Tenant' : 'Reactivate Tenant'}
          </button>
        </div>

        {/* Subscription */}
        {detail.subscription && (
          <Section label="Subscription">
            <p style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              {detail.subscription.display_name || detail.subscription.plan_name} · {detail.subscription.billing_cycle} · {detail.subscription.billing_status}
            </p>
          </Section>
        )}

        {/* Agents */}
        <Section label={`Agents (${detail.agents.length})`}>
          {detail.agents.length === 0 ? (
            <p style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>No agents.</p>
          ) : detail.agents.map(a => (
            <div key={a.user_id} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--border)' }}>
              <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{a.full_name} <span style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>({a.role})</span></span>
              <Badge variant={a.is_active ? 'active' : 'inactive'} label={a.is_active ? 'Active' : 'Inactive'} />
            </div>
          ))}
        </Section>

        {/* Usage */}
        <Section label="Usage (last 6 periods)">
          {detail.usage_history.length === 0 ? (
            <p style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>No usage yet.</p>
          ) : detail.usage_history.map((u, i) => (
            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderBottom: '1px solid var(--border)' }}>
              <span style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>{u.period_start}</span>
              <span style={{ fontSize: 12.5, color: 'var(--text-secondary)' }}>{u.sessions_used} sessions</span>
            </div>
          ))}
        </Section>

        {/* Abuse flags */}
        <Section label={`Open Abuse Flags (${detail.open_abuse_flags.length})`}>
          {detail.open_abuse_flags.length === 0 ? (
            <p style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>None.</p>
          ) : detail.open_abuse_flags.map(f => (
            <div key={f.id} style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderBottom: '1px solid var(--border)' }}>
              <span style={{ fontSize: 12.5, color: 'var(--text-secondary)' }}>{f.reason}</span>
              <Badge variant="open" label={f.severity} />
            </div>
          ))}
        </Section>

        {/* Audit */}
        <Section label="Recent Audit">
          {detail.recent_audit.map(a => (
            <div key={a.id} style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderBottom: '1px solid var(--border)' }}>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{a.action}</span>
              <span style={{ fontSize: 11.5, color: 'var(--text-subtle)' }}>{a.created_at ? new Date(a.created_at).toLocaleDateString() : ''}</span>
            </div>
          ))}
        </Section>
      </div>
    </div>
  );
}

function Section({ label, children }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <p style={{ fontSize: 10.5, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-muted)', marginBottom: 8 }}>{label}</p>
      {children}
    </div>
  );
}

/* ── Main component ── */

const TABS = [
  { key: 'Overview', label: 'Overview' },
  { key: 'Tenants',  label: 'Tenants'  },
  { key: 'Abuse',    label: 'Abuse'    },
  { key: 'Audit',    label: 'Audit'    },
  { key: 'Health',   label: 'Health'   },
];

export default function AdminDashboard() {
  const { user, logout } = useAuth();
  const [tab, setTab]     = useState('Overview');
  const [usage, setUsage] = useState(null);
  const [tenants, setTenants] = useState(null);
  const [search, setSearch]   = useState('');
  const [abuse, setAbuse]     = useState([]);
  const [audit, setAudit]     = useState([]);
  const [health, setHealth]   = useState(null);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading]   = useState(true);

  const loadTenants = useCallback(() => {
    adminAPI.getTenants(search ? { search } : {}).then(setTenants).catch(() => setTenants(null));
  }, [search]);

  useEffect(() => {
    Promise.all([
      adminAPI.getUsage().then(setUsage).catch(() => setUsage(null)),
      loadTenants(),
      adminAPI.getAbuse().then(setAbuse).catch(() => setAbuse([])),
      adminAPI.getAudit().then(setAudit).catch(() => setAudit([])),
      adminAPI.getHealth().then(setHealth).catch(() => setHealth(null)),
    ]).finally(() => setLoading(false));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { loadTenants(); }, [loadTenants]);

  const subsByPlan = (usage?.active_subs_by_plan ?? []).map(p => ({ label: p.plan, count: p.count }));

  return (
    <DashboardShell
      user={user}
      logout={logout}
      title="VCAI Admin"
      subtitle="Platform console"
      accent="#b472f1"
      accent2="#deb7ff"
    >
      {/* Page header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.02em', margin: 0 }}>
          Platform Console
        </h1>
        <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 5 }}>
          Manage tenants, monitor platform health, and review activity
        </p>
      </div>

      {/* Tabs */}
      <div style={{ marginBottom: 24 }}>
        <Tabs tabs={TABS} active={tab} onChange={setTab} />
      </div>

      {loading ? (
        <div style={{ padding: '64px 0', textAlign: 'center' }}>
          <div className="spin-ring" style={{ width: 24, height: 24, border: '2px solid rgba(222,183,255,0.08)', borderTopColor: '#deb7ff', borderRadius: '50%', margin: '0 auto 12px' }} />
          <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>Loading console…</p>
        </div>
      ) : (
        <>
          {/* ── Overview ── */}
          {tab === 'Overview' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
                <StatCard label="Total Tenants"         value={usage?.total_companies ?? 0}       sub={`${usage?.active_companies ?? 0} active`}                       Icon={IcoBuilding} accent="#ffb4ab" />
                <StatCard label="Active Subscriptions"  value={usage?.active_subscriptions ?? 0}                                                                        Icon={IcoLayers}   accent="#deb7ff" />
                <StatCard label="Sessions This Period"  value={usage?.sessions_this_period ?? 0}  sub={`${usage?.total_sessions ?? 0} all-time`}                        Icon={IcoActivity} accent="#a5d6a7" />
                <StatCard label="Open Abuse Flags"      value={usage?.open_abuse_flags ?? 0}                                                                             Icon={IcoFlag}     accent={(usage?.open_abuse_flags ?? 0) > 0 ? '#ffb4ab' : '#a5d6a7'} />
              </div>

              {/* Platform sessions chart */}
              <Card title="Platform Sessions" subtitle="Last 30 days">
                {(usage?.sessions_per_day?.length ?? 0) === 0 ? (
                  <EmptyState icon={IcoInbox} title="No sessions yet" description="Session data will appear here once tenants start using the platform" />
                ) : (
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={usage.sessions_per_day} margin={{ top: 8, right: 8, left: -24, bottom: 4 }}>
                      <CartesianGrid {...gridProps} />
                      <XAxis dataKey="date" tick={axisTick} tickLine={false} axisLine={axisLine} />
                      <YAxis allowDecimals={false} tick={axisTick} tickLine={false} axisLine={false} />
                      <Tooltip content={<ChartTooltip />} />
                      <Line type="monotone" dataKey="count" name="Sessions" stroke="#deb7ff" strokeWidth={2} dot={{ r: 2.5, fill: '#deb7ff' }} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </Card>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
                {/* Subs by plan */}
                <Card title="Subscriptions by Plan">
                  {subsByPlan.length === 0 ? (
                    <EmptyState icon={IcoInbox} title="No subscriptions" description="Subscription data will appear here" />
                  ) : (
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={subsByPlan} margin={{ top: 8, right: 8, left: -24, bottom: 4 }}>
                        <CartesianGrid {...gridProps} />
                        <XAxis dataKey="label" tick={axisTick} tickLine={false} axisLine={axisLine} />
                        <YAxis allowDecimals={false} tick={axisTick} tickLine={false} axisLine={false} />
                        <Tooltip content={<ChartTooltip />} />
                        <Bar dataKey="count" name="Tenants" fill="#b472f1" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </Card>

                {/* Top tenants */}
                <Card title="Top Tenants by Usage">
                  {(usage?.top_tenants?.length ?? 0) === 0 ? (
                    <EmptyState icon={IcoInbox} title="No data" description="Usage data will appear here" />
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {usage.top_tenants.map((t, i) => (
                        <div key={t.company_id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '7px 0', borderBottom: '1px solid var(--border)' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                            <span style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text-muted)', width: 18 }}>{i + 1}</span>
                            <span style={{ fontSize: 13, color: 'var(--text-secondary)', fontWeight: 500 }}>{t.name}</span>
                          </div>
                          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{t.sessions}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </Card>
              </div>
            </div>
          )}

          {/* ── Tenants ── */}
          {tab === 'Tenants' && (
            <Card
              title="Tenant Management"
              subtitle={`${tenants?.tenants?.length ?? 0} tenant${tenants?.tenants?.length !== 1 ? 's' : ''}`}
              right={
                <input
                  className="input-dark"
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  placeholder="Search tenants…"
                  style={{ width: 200 }}
                />
              }
            >
              {(tenants?.tenants?.length ?? 0) === 0 ? (
                <EmptyState icon={IcoBuilding} title="No tenants" description={search ? 'No tenants match your search' : 'No tenants registered yet'} />
              ) : (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border)' }}>
                        {['Company', 'Plan', 'Billing', 'Seats', 'Sessions (mo)', 'Status'].map(h => (
                          <th key={h} style={{ textAlign: 'left', padding: '0 12px 12px 0', fontSize: 11.5, fontWeight: 600, color: 'var(--text-muted)', letterSpacing: '0.04em', whiteSpace: 'nowrap' }}>
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {tenants.tenants.map(t => (
                        <tr
                          key={t.company_id}
                          onClick={() => setSelected(t.company_id)}
                          style={{ cursor: 'pointer', borderBottom: '1px solid var(--border)', transition: 'background 0.12s' }}
                          onMouseEnter={e => e.currentTarget.style.background = 'rgba(222,183,255,0.03)'}
                          onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                        >
                          <td style={{ padding: '12px 12px 12px 0', fontSize: 13.5, fontWeight: 500, color: 'var(--text-secondary)' }}>{t.name}</td>
                          <td style={{ padding: '12px 12px 12px 0', fontSize: 12.5, color: 'var(--text-muted)' }}>{t.plan_name || '—'}</td>
                          <td style={{ padding: '12px 12px 12px 0', fontSize: 12.5, color: 'var(--text-muted)' }}>{t.billing_status || '—'}</td>
                          <td style={{ padding: '12px 12px 12px 0', fontSize: 13, color: 'var(--text-secondary)' }}>{t.seats_used}/{t.seat_limit ?? '—'}</td>
                          <td style={{ padding: '12px 12px 12px 0', fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{t.sessions_this_period}</td>
                          <td style={{ padding: '12px 0' }}>
                            <Badge variant={t.is_active ? 'active' : 'suspended'} label={t.is_active ? 'Active' : 'Suspended'} dot={t.is_active} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          )}

          {/* ── Abuse ── */}
          {tab === 'Abuse' && (
            <Card title="Global Abuse Queue" subtitle="Flagged sessions across all tenants">
              {abuse.length === 0 ? (
                <EmptyState icon={IcoFlag} title="No abuse flags" description="All clear — no flagged sessions detected" />
              ) : (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border)' }}>
                        {['Reason', 'Severity', 'Status', 'Created'].map(h => (
                          <th key={h} style={{ textAlign: 'left', padding: '0 12px 12px 0', fontSize: 11.5, fontWeight: 600, color: 'var(--text-muted)', letterSpacing: '0.04em' }}>
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {abuse.map(f => (
                        <tr key={f.id} style={{ borderBottom: '1px solid var(--border)' }}>
                          <td style={{ padding: '12px 12px 12px 0', fontSize: 13, color: 'var(--text-secondary)' }}>{f.reason}</td>
                          <td style={{ padding: '12px 12px 12px 0' }}>
                            <Badge variant={f.severity === 'high' ? 'suspended' : f.severity === 'medium' ? 'open' : 'reviewed'} label={f.severity} />
                          </td>
                          <td style={{ padding: '12px 12px 12px 0' }}>
                            <Badge variant={f.status === 'open' ? 'open' : f.status === 'reviewed' ? 'reviewed' : 'dismissed'} label={f.status} />
                          </td>
                          <td style={{ padding: '12px 0', fontSize: 12, color: 'var(--text-muted)' }}>
                            {f.created_at ? new Date(f.created_at).toLocaleDateString() : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          )}

          {/* ── Audit ── */}
          {tab === 'Audit' && (
            <Card title="Global Audit Log" subtitle="All platform-level actions">
              {audit.length === 0 ? (
                <EmptyState icon={IcoInbox} title="No audit entries" description="Audit log is empty" />
              ) : (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border)' }}>
                        {['Action', 'Role', 'Target', 'When'].map(h => (
                          <th key={h} style={{ textAlign: 'left', padding: '0 12px 12px 0', fontSize: 11.5, fontWeight: 600, color: 'var(--text-muted)', letterSpacing: '0.04em' }}>
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {audit.map(a => (
                        <tr key={a.id} style={{ borderBottom: '1px solid var(--border)' }}>
                          <td style={{ padding: '12px 12px 12px 0', fontSize: 13, color: 'var(--text-secondary)' }}>{a.action}</td>
                          <td style={{ padding: '12px 12px 12px 0', fontSize: 12.5, color: 'var(--text-muted)' }}>{a.actor_role || '—'}</td>
                          <td style={{ padding: '12px 12px 12px 0', fontSize: 12.5, color: 'var(--text-muted)' }}>{a.target_type || '—'}</td>
                          <td style={{ padding: '12px 0', fontSize: 12, color: 'var(--text-subtle)' }}>
                            {a.created_at ? new Date(a.created_at).toLocaleString() : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          )}

          {/* ── Health ── */}
          {tab === 'Health' && (
            <Card
              title="System Health"
              right={
                <span style={{ fontSize: 12, fontWeight: 600, color: health?.status === 'healthy' ? '#a5d6a7' : '#e9c46a' }}>
                  {health?.status ?? 'unknown'}
                </span>
              }
            >
              {Object.keys(health?.checks ?? {}).length === 0 ? (
                <EmptyState icon={IcoServer} title="No health data" description="Health checks unavailable" />
              ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
                  {Object.entries(health.checks).map(([mod, r]) => (
                    <div
                      key={mod}
                      style={{
                        background: 'var(--bg-card-alt)',
                        border: '1px solid var(--border)',
                        borderRadius: 12,
                        padding: '14px 16px',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                        <span style={{ width: 7, height: 7, borderRadius: '50%', background: HEALTH_COLOR[r.status] || '#988d9d', flexShrink: 0 }} />
                        <span style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                          {mod}
                        </span>
                      </div>
                      <p style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.4 }}>{r.message}</p>
                    </div>
                  ))}
                </div>
              )}
            </Card>
          )}
        </>
      )}

      {selected && (
        <TenantDetail
          companyId={selected}
          onClose={() => setSelected(null)}
          onChanged={loadTenants}
        />
      )}
    </DashboardShell>
  );
}
