import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { seatsAPI } from '../services/api';

const UNLIMITED = 1_000_000;
const fmtLimit = (n) => (n >= UNLIMITED ? 'Unlimited' : String(n));

export default function SeatManagement() {
  const [roster, setRoster] = useState(null);
  const [error, setError] = useState('');
  const [email, setEmail] = useState('');
  const [inviting, setInviting] = useState(false);
  const [copiedId, setCopiedId] = useState(null);

  const load = async () => {
    try {
      setRoster(await seatsAPI.getRoster());
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not load seats');
    }
  };

  useEffect(() => {
    load();
  }, []);

  const atLimit = roster && roster.used >= roster.limit;

  const handleInvite = async (e) => {
    e.preventDefault();
    setError('');
    setInviting(true);
    try {
      await seatsAPI.invite(email);
      setEmail('');
      await load();
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not send invite');
    } finally {
      setInviting(false);
    }
  };

  const handleRevoke = async (id) => {
    setError('');
    try {
      await seatsAPI.revoke(id);
      await load();
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not revoke invite');
    }
  };

  const handleDeactivate = async (id) => {
    setError('');
    try {
      await seatsAPI.deactivate(id);
      await load();
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not deactivate user');
    }
  };

  const copyLink = async (invite) => {
    try {
      await navigator.clipboard.writeText(invite.invite_link);
      setCopiedId(invite.id);
      setTimeout(() => setCopiedId(null), 1500);
    } catch {
      setError('Could not copy link');
    }
  };

  return (
    <div className="min-h-screen py-10 px-4" style={{ background: 'var(--bg-app)' }}>
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>Seat management</h1>
            <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>Invite and manage the agents in your workspace.</p>
          </div>
          <Link to="/dashboard" className="text-sm font-medium hover:underline" style={{ color: 'var(--primary)' }}>
            Back to dashboard
          </Link>
        </div>

        {roster && (
          <div className="ds-card p-5 mb-6 flex items-center justify-between">
            <span className="font-medium" style={{ color: 'var(--text-secondary)' }}>
              Seats used: {roster.used} / {fmtLimit(roster.limit)}
            </span>
            <div className="w-1/2 rounded-full h-2 overflow-hidden" style={{ background: 'var(--surface-container-highest)' }}>
              <div
                className="h-2"
                style={{
                  width: `${Math.min(100, roster.limit >= UNLIMITED ? 0 : (roster.used / roster.limit) * 100)}%`,
                  background: '#b472f1',
                }}
              />
            </div>
          </div>
        )}

        {error && <div className="px-4 py-3 rounded-xl text-sm mb-4" style={{ color: 'var(--error)', background: 'rgba(255,180,171,0.08)', border: '1px solid rgba(255,180,171,0.25)' }}>{error}</div>}

        {/* Invite form */}
        <div className="ds-card p-6 mb-6">
          <h2 className="font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Invite an agent</h2>
          {atLimit ? (
            <p className="text-sm px-4 py-3 rounded-xl" style={{ color: 'var(--warning)', background: 'rgba(233,196,106,0.08)', border: '1px solid rgba(233,196,106,0.25)' }}>
              Seat limit reached - upgrade or free a seat to invite more agents.
            </p>
          ) : (
            <form onSubmit={handleInvite} className="flex gap-3">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="input-dark flex-1 px-4 py-3"
                placeholder="agent@company.com"
                required
              />
              <button
                type="submit"
                disabled={inviting}
                className="btn-primary"
                style={{ padding: '12px 20px' }}
              >
                {inviting ? 'Sending...' : 'Send invite'}
              </button>
            </form>
          )}
          <p className="text-xs mt-3" style={{ color: 'var(--text-subtle)' }}>
            Email delivery is stubbed in this build - copy the generated link and share it manually.
          </p>
        </div>

        {/* Pending invites */}
        {roster && roster.pending_invites.length > 0 && (
          <div className="ds-card p-6 mb-6">
            <h2 className="font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Pending invites</h2>
            <div className="divide-y" style={{ borderColor: 'var(--border)' }}>
              {roster.pending_invites.map((inv) => (
                <div key={inv.id} className="py-3 flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-sm font-medium truncate" style={{ color: 'var(--text-primary)' }}>{inv.email}</p>
                    <p className="text-xs truncate" style={{ color: 'var(--text-subtle)' }}>{inv.invite_link}</p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      onClick={() => copyLink(inv)}
                      className="btn-secondary text-sm"
                      style={{ padding: '6px 12px' }}
                    >
                      {copiedId === inv.id ? 'Copied' : 'Copy link'}
                    </button>
                    <button
                      onClick={() => handleRevoke(inv.id)}
                      className="btn-danger text-sm"
                      style={{ padding: '6px 12px' }}
                    >
                      Revoke
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Active agents */}
        <div className="ds-card p-6">
          <h2 className="font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Active members</h2>
          {!roster ? (
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading...</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left" style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border)' }}>
                    <th className="py-2 font-medium">Name</th>
                    <th className="py-2 font-medium">Email</th>
                    <th className="py-2 font-medium">Role</th>
                    <th className="py-2 font-medium text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y" style={{ borderColor: 'var(--border)' }}>
                  {roster.users.map((u) => (
                    <tr key={u.id}>
                      <td className="py-3" style={{ color: 'var(--text-primary)' }}>{u.full_name}</td>
                      <td className="py-3" style={{ color: 'var(--text-secondary)' }}>{u.email}</td>
                      <td className="py-3">
                        <span className="text-xs px-2 py-1 rounded-full" style={{ background: 'var(--surface-container-highest)', color: 'var(--text-secondary)' }}>{u.role}</span>
                      </td>
                      <td className="py-3 text-right">
                        {u.role === 'salesperson' ? (
                          <button
                            onClick={() => handleDeactivate(u.id)}
                            className="btn-danger text-sm"
                            style={{ padding: '6px 12px' }}
                          >
                            Deactivate
                          </button>
                        ) : (
                          <span className="text-xs" style={{ color: 'var(--text-subtle)' }}>-</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
