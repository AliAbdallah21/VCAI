import { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { onboardingAPI } from '../services/api';

export default function AcceptInvite() {
  const { token } = useParams();
  const navigate = useNavigate();
  const { setAuth } = useAuth();

  const [invite, setInvite] = useState(null);
  const [loadError, setLoadError] = useState('');
  const [fullName, setFullName] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    let mounted = true;
    onboardingAPI
      .getInvite(token)
      .then((data) => mounted && setInvite(data))
      .catch((err) =>
        mounted &&
        setLoadError(err.response?.data?.detail || 'This invite is not valid or has expired.')
      );
    return () => {
      mounted = false;
    };
  }, [token]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSubmitting(true);
    try {
      const result = await onboardingAPI.accept({ token, full_name: fullName, password });
      setAuth(result);
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not accept the invite');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4" style={{ background: 'var(--bg-app)' }}>
      <div className="w-full max-w-md">
        <div className="flex items-center justify-center gap-2 mb-6">
          <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #b472f1, #deb7ff)' }}>
            <span className="font-bold text-xl" style={{ color: '#4a007f' }}>V</span>
          </div>
          <span className="font-bold text-2xl" style={{ color: 'var(--text-primary)' }}>VCAI</span>
        </div>

        <div className="ds-card p-8">
          {loadError ? (
            <div className="text-center">
              <h2 className="text-xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>Invite unavailable</h2>
              <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>{loadError}</p>
              <Link to="/login" className="font-medium hover:underline" style={{ color: 'var(--primary)' }}>
                Go to sign in
              </Link>
            </div>
          ) : !invite ? (
            <p className="text-center" style={{ color: 'var(--text-muted)' }}>Loading invite...</p>
          ) : (
            <>
              <div className="text-center mb-6">
                <h2 className="text-xl font-bold" style={{ color: 'var(--text-primary)' }}>Join {invite.company_name}</h2>
                <p className="text-sm mt-2" style={{ color: 'var(--text-muted)' }}>
                  You were invited as a {invite.role} ({invite.email}). Set your name and password to continue.
                </p>
              </div>
              <form onSubmit={handleSubmit} className="space-y-5">
                <div>
                  <label className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Full name</label>
                  <input
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    className="input-dark w-full px-4 py-3"
                    placeholder="Your name"
                    required
                    minLength={2}
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
                {error && <div className="px-4 py-3 rounded-xl text-sm" style={{ color: 'var(--error)', background: 'rgba(255,180,171,0.08)', border: '1px solid rgba(255,180,171,0.25)' }}>{error}</div>}
                <button
                  type="submit"
                  disabled={submitting}
                  className="btn-primary w-full"
                  style={{ padding: '12px 0' }}
                >
                  {submitting ? 'Joining...' : 'Accept invite'}
                </button>
              </form>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
