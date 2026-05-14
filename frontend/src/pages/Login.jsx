import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Login() {
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [error, setError]       = useState('');
  const [loading, setLoading]   = useState(false);
  const { login }  = useAuth();
  const navigate   = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(email, password);
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.detail || 'Invalid credentials');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center relative overflow-hidden p-4"
      style={{ background: 'radial-gradient(ellipse 80% 60% at 50% -10%, rgba(37,99,235,0.15) 0%, #030712 60%)' }}
    >
      {/* Background orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div
          className="absolute rounded-full blur-3xl opacity-20"
          style={{
            width: 600, height: 600,
            top: '-20%', left: '-10%',
            background: 'radial-gradient(circle, #2563eb 0%, transparent 70%)',
          }}
        />
        <div
          className="absolute rounded-full blur-3xl opacity-15"
          style={{
            width: 500, height: 500,
            bottom: '-15%', right: '-10%',
            background: 'radial-gradient(circle, #7c3aed 0%, transparent 70%)',
          }}
        />
        {/* Grid texture */}
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: `
              linear-gradient(rgba(255,255,255,0.5) 1px, transparent 1px),
              linear-gradient(90deg, rgba(255,255,255,0.5) 1px, transparent 1px)
            `,
            backgroundSize: '60px 60px',
          }}
        />
      </div>

      <div className="w-full max-w-sm relative z-10">
        {/* Logo mark */}
        <div className="text-center mb-10 fade-in">
          <div
            className="inline-flex w-14 h-14 rounded-2xl items-center justify-center mb-5"
            style={{
              background: 'linear-gradient(135deg, #2563eb, #7c3aed)',
              boxShadow: '0 0 30px rgba(37,99,235,0.4)',
            }}
          >
            <span className="heading text-white font-bold text-2xl">V</span>
          </div>
          <h1 className="heading text-2xl font-bold text-white mb-1">Welcome back</h1>
          <p className="text-sm" style={{ color: 'rgba(148,163,184,0.6)' }}>
            Sign in to continue your training
          </p>
        </div>

        {/* Card */}
        <div className="glass rounded-2xl p-7 slide-up">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold mb-2 tracking-wide uppercase" style={{ color: 'rgba(148,163,184,0.7)' }}>
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                className="input-dark w-full px-4 py-3 rounded-xl text-sm"
                placeholder="you@company.com"
                required
              />
            </div>

            <div>
              <label className="block text-xs font-semibold mb-2 tracking-wide uppercase" style={{ color: 'rgba(148,163,184,0.7)' }}>
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                className="input-dark w-full px-4 py-3 rounded-xl text-sm"
                placeholder="••••••••"
                required
              />
            </div>

            {error && (
              <div
                className="px-4 py-3 rounded-xl text-sm"
                style={{
                  background: 'rgba(239,68,68,0.08)',
                  border: '1px solid rgba(239,68,68,0.2)',
                  color: '#fca5a5',
                }}
              >
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full py-3 rounded-xl text-sm font-semibold text-white"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4 spin-ring" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.25)" strokeWidth="3"/>
                    <path d="M12 2a10 10 0 0110 10" stroke="white" strokeWidth="3" strokeLinecap="round"/>
                  </svg>
                  Signing in…
                </span>
              ) : 'Sign In'}
            </button>
          </form>

          <p className="text-center mt-6 text-xs" style={{ color: 'rgba(148,163,184,0.45)' }}>
            No account?{' '}
            <Link to="/register" className="text-blue-400 font-medium hover:text-blue-300 transition-colors">
              Create one
            </Link>
          </p>
        </div>

        <p className="text-center mt-6 text-xs" style={{ color: 'rgba(148,163,184,0.2)' }}>
          © 2024 VCAI · MIU Thesis Project
        </p>
      </div>
    </div>
  );
}
