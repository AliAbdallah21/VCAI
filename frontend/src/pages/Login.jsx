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
      style={{ background: 'radial-gradient(ellipse 80% 60% at 50% -10%, var(--primary-soft-hover) 0%, var(--bg-app) 60%)' }}
    >
      {/* Background orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div
          className="absolute rounded-full blur-3xl opacity-20"
          style={{
            width: 600, height: 600,
            top: '-20%', left: '-10%',
            background: 'radial-gradient(circle, #b472f1 0%, transparent 70%)',
          }}
        />
        <div
          className="absolute rounded-full blur-3xl opacity-15"
          style={{
            width: 500, height: 500,
            bottom: '-15%', right: '-10%',
            background: 'radial-gradient(circle, #deb7ff 0%, transparent 70%)',
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
              background: 'linear-gradient(135deg, #b472f1, #deb7ff)',
              boxShadow: '0 0 30px rgba(180,114,241,0.4)',
            }}
          >
            <span className="heading font-bold text-2xl" style={{ color: '#4a007f' }}>V</span>
          </div>
          <h1 className="headline-md mb-1" style={{ color: 'var(--text-primary)' }}>Welcome back</h1>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Sign in to continue your training
          </p>
        </div>

        {/* Card */}
        <div className="glass rounded-2xl p-7 slide-up">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="label-sm block mb-2" style={{ color: 'var(--text-secondary)' }}>
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
              <label className="label-sm block mb-2" style={{ color: 'var(--text-secondary)' }}>
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
                  background: 'rgba(255,180,171,0.08)',
                  border: '1px solid rgba(255,180,171,0.25)',
                  color: 'var(--error)',
                }}
              >
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full py-3 rounded-xl text-sm font-semibold"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4 spin-ring" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="rgba(74,0,127,0.25)" strokeWidth="3"/>
                    <path d="M12 2a10 10 0 0110 10" stroke="#4a007f" strokeWidth="3" strokeLinecap="round"/>
                  </svg>
                  Signing in…
                </span>
              ) : 'Sign In'}
            </button>
          </form>

          <p className="text-center mt-6 text-xs" style={{ color: 'var(--text-muted)' }}>
            No account?{' '}
            <Link to="/register" className="font-medium transition-colors" style={{ color: 'var(--primary)' }}>
              Create one
            </Link>
          </p>
        </div>

        <p className="text-center mt-6 text-xs" style={{ color: 'var(--text-subtle)' }}>
          © 2025 VCAI · MIU Thesis Project
        </p>
      </div>
    </div>
  );
}
