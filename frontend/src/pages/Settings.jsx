import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Layout from '../components/Layout';
import { useAuth } from '../context/AuthContext';
import { seatsAPI } from '../services/api';

export default function Settings() {
  const { user, refreshUser } = useAuth();
  const navigate = useNavigate();

  const [code, setCode] = useState('');
  const [joining, setJoining] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const inCompany = !!user?.company_id;

  const handleJoin = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setJoining(true);
    try {
      await seatsAPI.join(code.trim().toUpperCase());
      // Account role/company changed server-side — refresh so routing reflects it.
      await refreshUser();
      setSuccess('You have joined the team. Redirecting…');
      setTimeout(() => navigate('/dashboard'), 900);
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not join with that code.');
    } finally {
      setJoining(false);
    }
  };

  return (
    <Layout>
      <div className="p-4 md:p-8 max-w-2xl mx-auto">
        <div className="mb-8">
          <h1 className="heading text-2xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>Settings</h1>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Manage your account and team membership.
          </p>
        </div>

        {/* Account card */}
        <div className="ds-card p-6 mb-6">
          <h2 className="font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Account</h2>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span style={{ color: 'var(--text-muted)' }}>Name</span>
              <span style={{ color: 'var(--text-secondary)' }}>{user?.full_name}</span>
            </div>
            <div className="flex justify-between">
              <span style={{ color: 'var(--text-muted)' }}>Email</span>
              <span style={{ color: 'var(--text-secondary)' }}>{user?.email}</span>
            </div>
            <div className="flex justify-between">
              <span style={{ color: 'var(--text-muted)' }}>Role</span>
              <span style={{ color: 'var(--text-secondary)' }} className="capitalize">{user?.role}</span>
            </div>
            {inCompany && (
              <div className="flex justify-between">
                <span style={{ color: 'var(--text-muted)' }}>Company</span>
                <span style={{ color: 'var(--text-secondary)' }}>{user?.company || '—'}</span>
              </div>
            )}
          </div>
        </div>

        {/* Join a company — only for users not yet in a company */}
        {inCompany ? (
          <div className="ds-card p-6">
            <h2 className="font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>Team</h2>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              You're a member of <span style={{ color: 'var(--text-secondary)' }} className="font-medium">{user?.company || 'your team'}</span>.
              To switch teams, ask your new manager for an invite.
            </p>
          </div>
        ) : (
          <div className="ds-card p-6">
            <h2 className="font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>Join a company</h2>
            <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
              Got a 6-character invite code from a manager? Paste it here to join their team.
            </p>
            <form onSubmit={handleJoin} className="flex flex-col sm:flex-row gap-3">
              <input
                value={code}
                onChange={(e) => setCode(e.target.value)}
                className="input-dark flex-1 px-4 py-3 font-mono tracking-wider uppercase"
                placeholder="ABC123"
                maxLength={12}
                autoCapitalize="characters"
                autoCorrect="off"
                spellCheck="false"
                required
              />
              <button
                type="submit"
                disabled={joining || !code.trim()}
                className="btn-primary"
                style={{ padding: '12px 20px' }}
              >
                {joining ? 'Joining…' : 'Join team'}
              </button>
            </form>
            {error && (
              <div className="mt-4 px-4 py-3 rounded-xl text-sm" style={{ color: 'var(--error)', background: 'rgba(255,180,171,0.08)', border: '1px solid rgba(255,180,171,0.25)' }}>
                {error}
              </div>
            )}
            {success && (
              <div className="mt-4 px-4 py-3 rounded-xl text-sm" style={{ color: 'var(--success-green, #a5d6a7)', background: 'rgba(165,214,167,0.08)', border: '1px solid rgba(165,214,167,0.25)' }}>
                {success}
              </div>
            )}
          </div>
        )}
      </div>
    </Layout>
  );
}
