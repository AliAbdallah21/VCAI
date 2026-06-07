import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { adminAPI } from '../services/api';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

function Shell({ children, user, logout }) {
  const navigate = useNavigate();
  const initials = user?.full_name?.split(' ').map((n) => n[0]).join('').slice(0, 2).toUpperCase() || 'A';
  return (
    <div className="min-h-screen" style={{ background: '#030712' }}>
      <header
        className="sticky top-0 z-30 flex items-center justify-between px-5 py-3"
        style={{ background: 'rgba(8,14,28,0.96)', borderBottom: '1px solid rgba(255,255,255,0.05)', backdropFilter: 'blur(20px)' }}
      >
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #be123c 0%, #7c3aed 100%)', boxShadow: '0 0 18px rgba(190,18,60,0.35)' }}>
            <span className="heading text-white font-bold text-sm">V</span>
          </div>
          <div>
            <p className="heading font-bold text-white text-sm tracking-wider">VCAI Admin</p>
            <p className="text-xs" style={{ color: 'rgba(148,163,184,0.4)' }}>Platform console</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white" style={{ background: 'linear-gradient(135deg, #be123c, #7c3aed)' }} title={user?.email}>
            {initials}
          </div>
          <button onClick={() => { logout(); navigate('/login'); }} className="px-3 py-1.5 rounded-lg text-xs font-medium text-slate-500 hover:text-red-400" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
            Sign out
          </button>
        </div>
      </header>
      <main className="p-4 md:p-8 max-w-6xl mx-auto">{children}</main>
    </div>
  );
}

function MetricCard({ label, value, sub, color }) {
  return (
    <div className="rounded-2xl p-5 relative overflow-hidden" style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}>
      <p className="text-xs font-medium tracking-wide uppercase mb-3" style={{ color: 'rgba(148,163,184,0.5)' }}>{label}</p>
      <p className="heading text-3xl font-bold" style={{ color }}>{value}</p>
      {sub && <p className="text-xs mt-1" style={{ color: 'rgba(148,163,184,0.4)' }}>{sub}</p>}
      <div className="absolute bottom-0 left-0 right-0 h-px" style={{ background: `linear-gradient(90deg, transparent, ${color}40, transparent)` }} />
    </div>
  );
}

function Card({ title, children, right }) {
  return (
    <div className="rounded-2xl p-5" style={{ background: 'rgba(13,21,38,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-white">{title}</h3>
        {right}
      </div>
      {children}
    </div>
  );
}

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl p-3 text-xs" style={{ background: 'rgba(8,14,28,0.97)', border: '1px solid rgba(255,255,255,0.1)' }}>
      <p className="font-semibold text-white mb-1">{label}</p>
      {payload.map((e) => (
        <div key={e.dataKey} className="flex justify-between gap-4"><span style={{ color: e.color }}>{e.name}</span><span className="font-bold text-white">{e.value}</span></div>
      ))}
    </div>
  );
};

const HEALTH_COLOR = { ok: '#34d399', warn: '#fbbf24', error: '#ef4444' };

function TenantDetail({ companyId, onClose, onChanged }) {
  const [detail, setDetail] = useState(null);
  const [busy, setBusy] = useState(false);

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
    } finally {
      setBusy(false);
    }
  };

  if (!detail) return null;
  const active = detail.company.is_active;

  return (
    <div className="fixed inset-0 z-40 flex justify-end" style={{ background: 'rgba(0,0,0,0.5)' }} onClick={onClose}>
      <div className="w-full max-w-lg h-full overflow-y-auto p-6" style={{ background: '#0b1220', borderLeft: '1px solid rgba(255,255,255,0.08)' }} onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between mb-5">
          <div>
            <h2 className="heading text-xl font-bold text-white">{detail.company.name}</h2>
            <p className="text-xs" style={{ color: 'rgba(148,163,184,0.5)' }}>{detail.company.slug}</p>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-white text-lg">×</button>
        </div>

        <div className="flex items-center gap-2 mb-5">
          <span className="px-2 py-1 rounded-md text-xs font-semibold" style={{ color: active ? '#34d399' : '#ef4444', background: active ? 'rgba(16,185,129,0.12)' : 'rgba(239,68,68,0.12)' }}>
            {active ? 'Active' : 'Suspended'}
          </span>
          <button
            onClick={toggle}
            disabled={busy}
            className="ml-auto px-3 py-1.5 rounded-lg text-xs font-semibold"
            style={active
              ? { color: '#fca5a5', background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.3)' }
              : { color: '#34d399', background: 'rgba(16,185,129,0.12)', border: '1px solid rgba(16,185,129,0.3)' }}
          >
            {active ? 'Suspend tenant' : 'Reactivate tenant'}
          </button>
        </div>

        {detail.subscription && (
          <div className="rounded-xl p-4 mb-4" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
            <p className="text-xs uppercase tracking-wide mb-2" style={{ color: 'rgba(148,163,184,0.5)' }}>Subscription</p>
            <p className="text-sm text-slate-200">{detail.subscription.display_name || detail.subscription.plan_name} · {detail.subscription.billing_cycle} · {detail.subscription.billing_status}</p>
          </div>
        )}

        <p className="text-xs uppercase tracking-wide mb-2" style={{ color: 'rgba(148,163,184,0.5)' }}>Agents ({detail.agents.length})</p>
        <div className="space-y-1 mb-4">
          {detail.agents.map((a) => (
            <div key={a.user_id} className="flex justify-between text-sm py-1">
              <span className="text-slate-300">{a.full_name} <span className="text-xs" style={{ color: 'rgba(148,163,184,0.4)' }}>({a.role})</span></span>
              <span className="text-xs" style={{ color: a.is_active ? '#34d399' : 'rgba(148,163,184,0.5)' }}>{a.is_active ? 'active' : 'inactive'}</span>
            </div>
          ))}
        </div>

        <p className="text-xs uppercase tracking-wide mb-2" style={{ color: 'rgba(148,163,184,0.5)' }}>Usage (last 6 periods)</p>
        <div className="space-y-1 mb-4">
          {detail.usage_history.length === 0 ? <p className="text-xs" style={{ color: 'rgba(148,163,184,0.4)' }}>No usage yet.</p> :
            detail.usage_history.map((u, i) => (
              <div key={i} className="flex justify-between text-sm"><span className="text-slate-400">{u.period_start}</span><span className="text-slate-300">{u.sessions_used} sessions</span></div>
            ))}
        </div>

        <p className="text-xs uppercase tracking-wide mb-2" style={{ color: 'rgba(148,163,184,0.5)' }}>Open abuse flags ({detail.open_abuse_flags.length})</p>
        <div className="space-y-1 mb-4">
          {detail.open_abuse_flags.length === 0 ? <p className="text-xs" style={{ color: 'rgba(148,163,184,0.4)' }}>None.</p> :
            detail.open_abuse_flags.map((f) => (
              <div key={f.id} className="flex justify-between text-sm"><span className="text-slate-300">{f.reason}</span><span className="text-xs" style={{ color: '#fbbf24' }}>{f.severity}</span></div>
            ))}
        </div>

        <p className="text-xs uppercase tracking-wide mb-2" style={{ color: 'rgba(148,163,184,0.5)' }}>Recent audit</p>
        <div className="space-y-1">
          {detail.recent_audit.map((a) => (
            <div key={a.id} className="flex justify-between text-xs"><span className="text-slate-400">{a.action}</span><span style={{ color: 'rgba(148,163,184,0.4)' }}>{a.created_at ? new Date(a.created_at).toLocaleDateString() : ''}</span></div>
          ))}
        </div>
      </div>
    </div>
  );
}

const TABS = ['Overview', 'Tenants', 'Abuse', 'Audit', 'Health'];

export default function AdminDashboard() {
  const { user, logout } = useAuth();
  const [tab, setTab] = useState('Overview');
  const [usage, setUsage] = useState(null);
  const [tenants, setTenants] = useState(null);
  const [search, setSearch] = useState('');
  const [abuse, setAbuse] = useState([]);
  const [audit, setAudit] = useState([]);
  const [health, setHealth] = useState(null);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);

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

  const subsByPlan = (usage?.active_subs_by_plan ?? []).map((p) => ({ label: p.plan, count: p.count }));

  return (
    <Shell user={user} logout={logout}>
      <div className="flex gap-2 mb-6 flex-wrap">
        {TABS.map((t) => {
          const active = tab === t;
          return (
            <button key={t} onClick={() => setTab(t)} className="px-3.5 py-1.5 rounded-lg text-xs font-semibold"
              style={{ background: active ? 'rgba(190,18,60,0.15)' : 'rgba(255,255,255,0.03)', border: `1px solid ${active ? 'rgba(190,18,60,0.45)' : 'rgba(255,255,255,0.07)'}`, color: active ? '#fb7185' : 'rgba(148,163,184,0.55)' }}>
              {t}
            </button>
          );
        })}
      </div>

      {loading ? <p className="text-sm py-16 text-center" style={{ color: 'rgba(148,163,184,0.5)' }}>Loading...</p> : (
        <>
          {tab === 'Overview' && (
            <div className="space-y-5">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard label="Total tenants" value={usage?.total_companies ?? 0} sub={`${usage?.active_companies ?? 0} active`} color="#fb7185" />
                <MetricCard label="Active subscriptions" value={usage?.active_subscriptions ?? 0} color="#60a5fa" />
                <MetricCard label="Sessions this period" value={usage?.sessions_this_period ?? 0} sub={`${usage?.total_sessions ?? 0} all-time`} color="#34d399" />
                <MetricCard label="Open abuse flags" value={usage?.open_abuse_flags ?? 0} color={(usage?.open_abuse_flags ?? 0) > 0 ? '#ef4444' : '#34d399'} />
              </div>
              <Card title="Platform sessions per day (last 30 days)">
                {(usage?.sessions_per_day?.length ?? 0) === 0 ? <p className="text-xs py-8 text-center" style={{ color: 'rgba(148,163,184,0.4)' }}>No sessions yet.</p> : (
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={usage.sessions_per_day} margin={{ top: 8, right: 16, left: -24, bottom: 4 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                      <XAxis dataKey="date" tick={{ fill: 'rgba(148,163,184,0.45)', fontSize: 11 }} tickLine={false} axisLine={{ stroke: 'rgba(255,255,255,0.06)' }} />
                      <YAxis allowDecimals={false} tick={{ fill: 'rgba(148,163,184,0.45)', fontSize: 11 }} tickLine={false} axisLine={false} />
                      <Tooltip content={<ChartTooltip />} />
                      <Line type="monotone" dataKey="count" name="Sessions" stroke="#fb7185" strokeWidth={2} dot={{ r: 2.5, fill: '#fb7185' }} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </Card>
              <div className="grid md:grid-cols-2 gap-4">
                <Card title="Subscriptions by plan">
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={subsByPlan} margin={{ top: 8, right: 16, left: -24, bottom: 4 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                      <XAxis dataKey="label" tick={{ fill: 'rgba(148,163,184,0.45)', fontSize: 11 }} tickLine={false} axisLine={{ stroke: 'rgba(255,255,255,0.06)' }} />
                      <YAxis allowDecimals={false} tick={{ fill: 'rgba(148,163,184,0.45)', fontSize: 11 }} tickLine={false} axisLine={false} />
                      <Tooltip content={<ChartTooltip />} />
                      <Bar dataKey="count" name="Tenants" fill="#7c3aed" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
                <Card title="Top tenants by usage">
                  {(usage?.top_tenants?.length ?? 0) === 0 ? <p className="text-xs py-8 text-center" style={{ color: 'rgba(148,163,184,0.4)' }}>No data.</p> : (
                    <div className="space-y-2">
                      {usage.top_tenants.map((t) => (
                        <div key={t.company_id} className="flex justify-between text-sm"><span className="text-slate-300">{t.name}</span><span className="text-slate-400">{t.sessions}</span></div>
                      ))}
                    </div>
                  )}
                </Card>
              </div>
            </div>
          )}

          {tab === 'Tenants' && (
            <Card title="Tenants" right={
              <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search name..." className="px-3 py-1.5 rounded-lg text-xs text-white" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }} />
            }>
              {(tenants?.tenants?.length ?? 0) === 0 ? <p className="text-xs py-8 text-center" style={{ color: 'rgba(148,163,184,0.4)' }}>No tenants.</p> : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left" style={{ color: 'rgba(148,163,184,0.5)' }}>
                        <th className="font-medium pb-2 pr-4">Company</th>
                        <th className="font-medium pb-2 pr-4">Plan</th>
                        <th className="font-medium pb-2 pr-4">Billing</th>
                        <th className="font-medium pb-2 pr-4">Seats</th>
                        <th className="font-medium pb-2 pr-4">Sessions (mo)</th>
                        <th className="font-medium pb-2">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {tenants.tenants.map((t) => (
                        <tr key={t.company_id} onClick={() => setSelected(t.company_id)} className="cursor-pointer hover:bg-white/[0.02]" style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                          <td className="py-2.5 pr-4 text-slate-200">{t.name}</td>
                          <td className="py-2.5 pr-4" style={{ color: 'rgba(148,163,184,0.7)' }}>{t.plan_name || '—'}</td>
                          <td className="py-2.5 pr-4" style={{ color: 'rgba(148,163,184,0.7)' }}>{t.billing_status || '—'}</td>
                          <td className="py-2.5 pr-4 text-slate-300">{t.seats_used}/{t.seat_limit ?? '—'}</td>
                          <td className="py-2.5 pr-4 text-slate-300">{t.sessions_this_period}</td>
                          <td className="py-2.5"><span className="text-xs font-semibold" style={{ color: t.is_active ? '#34d399' : '#ef4444' }}>{t.is_active ? 'Active' : 'Suspended'}</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          )}

          {tab === 'Abuse' && (
            <Card title="Global abuse queue">
              {abuse.length === 0 ? <p className="text-xs py-8 text-center" style={{ color: 'rgba(148,163,184,0.4)' }}>No abuse flags.</p> : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead><tr className="text-left" style={{ color: 'rgba(148,163,184,0.5)' }}>
                      <th className="font-medium pb-2 pr-4">Reason</th><th className="font-medium pb-2 pr-4">Severity</th><th className="font-medium pb-2 pr-4">Status</th><th className="font-medium pb-2">Created</th>
                    </tr></thead>
                    <tbody>
                      {abuse.map((f) => (
                        <tr key={f.id} style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                          <td className="py-2.5 pr-4 text-slate-200">{f.reason}</td>
                          <td className="py-2.5 pr-4" style={{ color: '#fbbf24' }}>{f.severity}</td>
                          <td className="py-2.5 pr-4" style={{ color: 'rgba(148,163,184,0.7)' }}>{f.status}</td>
                          <td className="py-2.5" style={{ color: 'rgba(148,163,184,0.5)' }}>{f.created_at ? new Date(f.created_at).toLocaleDateString() : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          )}

          {tab === 'Audit' && (
            <Card title="Global audit log">
              {audit.length === 0 ? <p className="text-xs py-8 text-center" style={{ color: 'rgba(148,163,184,0.4)' }}>No audit entries.</p> : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead><tr className="text-left" style={{ color: 'rgba(148,163,184,0.5)' }}>
                      <th className="font-medium pb-2 pr-4">Action</th><th className="font-medium pb-2 pr-4">Role</th><th className="font-medium pb-2 pr-4">Target</th><th className="font-medium pb-2">When</th>
                    </tr></thead>
                    <tbody>
                      {audit.map((a) => (
                        <tr key={a.id} style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                          <td className="py-2.5 pr-4 text-slate-200">{a.action}</td>
                          <td className="py-2.5 pr-4" style={{ color: 'rgba(148,163,184,0.7)' }}>{a.actor_role || '—'}</td>
                          <td className="py-2.5 pr-4" style={{ color: 'rgba(148,163,184,0.6)' }}>{a.target_type || '—'}</td>
                          <td className="py-2.5" style={{ color: 'rgba(148,163,184,0.5)' }}>{a.created_at ? new Date(a.created_at).toLocaleString() : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          )}

          {tab === 'Health' && (
            <Card title="System health" right={<span className="text-xs font-semibold" style={{ color: health?.status === 'healthy' ? '#34d399' : '#fbbf24' }}>{health?.status ?? 'unknown'}</span>}>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {Object.entries(health?.checks ?? {}).map(([mod, r]) => (
                  <div key={mod} className="rounded-xl p-3" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="w-2 h-2 rounded-full" style={{ background: HEALTH_COLOR[r.status] || '#94a3b8' }} />
                      <span className="text-sm font-semibold text-white uppercase">{mod}</span>
                    </div>
                    <p className="text-xs" style={{ color: 'rgba(148,163,184,0.55)' }}>{r.message}</p>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}

      {selected && <TenantDetail companyId={selected} onClose={() => setSelected(null)} onChanged={loadTenants} />}
    </Shell>
  );
}
