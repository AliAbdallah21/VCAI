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
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="flex items-center justify-center gap-2 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-xl">V</span>
          </div>
          <span className="font-bold text-2xl text-slate-800">VCAI</span>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-8">
          {loadError ? (
            <div className="text-center">
              <h2 className="text-xl font-bold text-slate-800 mb-2">Invite unavailable</h2>
              <p className="text-slate-500 text-sm mb-6">{loadError}</p>
              <Link to="/login" className="text-blue-600 font-medium hover:underline">
                Go to sign in
              </Link>
            </div>
          ) : !invite ? (
            <p className="text-center text-slate-500">Loading invite...</p>
          ) : (
            <>
              <div className="text-center mb-6">
                <h2 className="text-xl font-bold text-slate-800">Join {invite.company_name}</h2>
                <p className="text-slate-500 text-sm mt-2">
                  You were invited as a {invite.role} ({invite.email}). Set your name and password to continue.
                </p>
              </div>
              <form onSubmit={handleSubmit} className="space-y-5">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">Full name</label>
                  <input
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl outline-none focus:ring-2 focus:ring-blue-500 text-slate-900 placeholder:text-slate-400"
                    placeholder="Your name"
                    required
                    minLength={2}
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
                {error && <div className="bg-red-50 text-red-600 px-4 py-3 rounded-xl text-sm">{error}</div>}
                <button
                  type="submit"
                  disabled={submitting}
                  className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-xl font-medium hover:opacity-90 disabled:opacity-50"
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
