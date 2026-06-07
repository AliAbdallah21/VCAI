import { useState } from 'react';
import { managerAPI } from '../services/api';

const SEVERITY_COLOR = {
  high: '#ffb4ab',
  medium: '#e9c46a',
  low: '#deb7ff',
};

const STATUS_COLOR = {
  open: '#e9c46a',
  reviewed: '#a5d6a7',
  dismissed: 'var(--text-muted)',
};

function Badge({ text, color }) {
  return (
    <span
      className="px-2 py-0.5 rounded-md text-xs font-semibold"
      style={{ color, background: `${color}1a`, border: `1px solid ${color}33` }}
    >
      {text}
    </span>
  );
}

export default function AbuseQueue({ flags, onResolved }) {
  const [statusFilter, setStatusFilter] = useState('');
  const [busyId, setBusyId] = useState(null);
  const [error, setError] = useState('');
  const flagRows = Array.isArray(flags) ? flags : [];

  const visible = statusFilter
    ? flagRows.filter((f) => f.status === statusFilter)
    : flagRows;

  const resolve = async (flagId, status) => {
    setBusyId(flagId);
    setError('');
    try {
      await managerAPI.resolveAbuse(flagId, status);
      onResolved?.();
    } catch (e) {
      setError(e?.response?.data?.detail || 'Could not update flag');
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div
      className="rounded-2xl p-5"
      style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Abuse flags</h3>
        <div className="flex gap-2">
          {['', 'open', 'reviewed', 'dismissed'].map((s) => {
            const active = statusFilter === s;
            return (
              <button
                key={s || 'all'}
                onClick={() => setStatusFilter(s)}
                className="px-2.5 py-1 rounded-lg text-xs font-semibold"
                style={{
                  background: active ? 'rgba(222,183,255,0.15)' : 'rgba(255,255,255,0.03)',
                  border: `1px solid ${active ? 'rgba(222,183,255,0.45)' : 'rgba(255,255,255,0.07)'}`,
                  color: active ? 'var(--primary)' : 'var(--text-muted)',
                }}
              >
                {s ? s[0].toUpperCase() + s.slice(1) : 'All'}
              </button>
            );
          })}
        </div>
      </div>

      {error && <p className="text-xs mb-3" style={{ color: 'var(--error)' }}>{error}</p>}

      {visible.length === 0 ? (
        <p className="text-xs py-8 text-center" style={{ color: 'var(--text-subtle)' }}>
          No abuse flags{statusFilter ? ` with status "${statusFilter}"` : ''}.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left" style={{ color: 'var(--text-muted)' }}>
                <th className="font-medium pb-2 pr-4">Reason</th>
                <th className="font-medium pb-2 pr-4">Severity</th>
                <th className="font-medium pb-2 pr-4">Detail</th>
                <th className="font-medium pb-2 pr-4">Status</th>
                <th className="font-medium pb-2 pr-4">Created</th>
                <th className="font-medium pb-2">Actions</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((f) => (
                <tr key={f.id} style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                  <td className="py-2.5 pr-4" style={{ color: 'var(--text-primary)' }}>{f.reason}</td>
                  <td className="py-2.5 pr-4">
                    <Badge text={f.severity} color={SEVERITY_COLOR[f.severity] || 'var(--text-muted)'} />
                  </td>
                  <td className="py-2.5 pr-4 max-w-xs truncate" style={{ color: 'var(--text-secondary)' }}>
                    {f.detail ? JSON.stringify(f.detail) : '—'}
                  </td>
                  <td className="py-2.5 pr-4">
                    <Badge text={f.status} color={STATUS_COLOR[f.status] || 'var(--text-muted)'} />
                  </td>
                  <td className="py-2.5 pr-4" style={{ color: 'var(--text-muted)' }}>
                    {f.created_at ? new Date(f.created_at).toLocaleDateString() : '—'}
                  </td>
                  <td className="py-2.5">
                    {f.status === 'open' ? (
                      <div className="flex gap-2">
                        <button
                          disabled={busyId === f.id}
                          onClick={() => resolve(f.id, 'reviewed')}
                          className="px-2.5 py-1 rounded-lg text-xs font-semibold"
                          style={{ color: '#a5d6a7', background: 'rgba(165,214,167,0.12)', border: '1px solid rgba(165,214,167,0.3)' }}
                        >
                          Review
                        </button>
                        <button
                          disabled={busyId === f.id}
                          onClick={() => resolve(f.id, 'dismissed')}
                          className="px-2.5 py-1 rounded-lg text-xs font-semibold"
                          style={{ color: 'var(--text-secondary)', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)' }}
                        >
                          Dismiss
                        </button>
                      </div>
                    ) : (
                      <span className="text-xs" style={{ color: 'var(--text-subtle)' }}>resolved</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
