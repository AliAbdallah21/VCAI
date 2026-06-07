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
    <div className="min-h-screen bg-slate-50 py-10 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold text-slate-800">Seat management</h1>
            <p className="text-slate-500 text-sm mt-1">Invite and manage the agents in your workspace.</p>
          </div>
          <Link to="/dashboard" className="text-sm text-blue-600 font-medium hover:underline">
            Back to dashboard
          </Link>
        </div>

        {roster && (
          <div className="bg-white rounded-2xl shadow-sm p-5 mb-6 flex items-center justify-between">
            <span className="text-slate-700 font-medium">
              Seats used: {roster.used} / {fmtLimit(roster.limit)}
            </span>
            <div className="w-1/2 bg-slate-100 rounded-full h-2 overflow-hidden">
              <div
                className="bg-blue-600 h-2"
                style={{
                  width: `${Math.min(100, roster.limit >= UNLIMITED ? 0 : (roster.used / roster.limit) * 100)}%`,
                }}
              />
            </div>
          </div>
        )}

        {error && <div className="bg-red-50 text-red-600 px-4 py-3 rounded-xl text-sm mb-4">{error}</div>}

        {/* Invite form */}
        <div className="bg-white rounded-2xl shadow-sm p-6 mb-6">
          <h2 className="font-semibold text-slate-800 mb-4">Invite an agent</h2>
          {atLimit ? (
            <p className="text-sm text-amber-700 bg-amber-50 border border-amber-200 px-4 py-3 rounded-xl">
              Seat limit reached - upgrade or free a seat to invite more agents.
            </p>
          ) : (
            <form onSubmit={handleInvite} className="flex gap-3">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="flex-1 px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl outline-none focus:ring-2 focus:ring-blue-500 text-slate-900 placeholder:text-slate-400"
                placeholder="agent@company.com"
                required
              />
              <button
                type="submit"
                disabled={inviting}
                className="px-5 py-3 rounded-xl font-medium bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:opacity-90 disabled:opacity-50"
              >
                {inviting ? 'Sending...' : 'Send invite'}
              </button>
            </form>
          )}
          <p className="text-xs text-slate-400 mt-3">
            Email delivery is stubbed in this build - copy the generated link and share it manually.
          </p>
        </div>

        {/* Pending invites */}
        {roster && roster.pending_invites.length > 0 && (
          <div className="bg-white rounded-2xl shadow-sm p-6 mb-6">
            <h2 className="font-semibold text-slate-800 mb-4">Pending invites</h2>
            <div className="divide-y divide-slate-100">
              {roster.pending_invites.map((inv) => (
                <div key={inv.id} className="py-3 flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-slate-800 truncate">{inv.email}</p>
                    <p className="text-xs text-slate-400 truncate">{inv.invite_link}</p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      onClick={() => copyLink(inv)}
                      className="px-3 py-1.5 text-sm rounded-lg bg-slate-100 text-slate-700 hover:bg-slate-200"
                    >
                      {copiedId === inv.id ? 'Copied' : 'Copy link'}
                    </button>
                    <button
                      onClick={() => handleRevoke(inv.id)}
                      className="px-3 py-1.5 text-sm rounded-lg bg-red-50 text-red-600 hover:bg-red-100"
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
        <div className="bg-white rounded-2xl shadow-sm p-6">
          <h2 className="font-semibold text-slate-800 mb-4">Active members</h2>
          {!roster ? (
            <p className="text-slate-500 text-sm">Loading...</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-slate-400 border-b border-slate-100">
                    <th className="py-2 font-medium">Name</th>
                    <th className="py-2 font-medium">Email</th>
                    <th className="py-2 font-medium">Role</th>
                    <th className="py-2 font-medium text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-50">
                  {roster.users.map((u) => (
                    <tr key={u.id}>
                      <td className="py-3 text-slate-800">{u.full_name}</td>
                      <td className="py-3 text-slate-600">{u.email}</td>
                      <td className="py-3">
                        <span className="text-xs px-2 py-1 rounded-full bg-slate-100 text-slate-600">{u.role}</span>
                      </td>
                      <td className="py-3 text-right">
                        {u.role === 'salesperson' ? (
                          <button
                            onClick={() => handleDeactivate(u.id)}
                            className="px-3 py-1.5 text-sm rounded-lg bg-red-50 text-red-600 hover:bg-red-100"
                          >
                            Deactivate
                          </button>
                        ) : (
                          <span className="text-xs text-slate-400">-</span>
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
