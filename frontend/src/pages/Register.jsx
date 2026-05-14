import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Register() {
  const [form, setForm] = useState({ full_name: '', email: '', company: '', password: '', confirmPassword: '' });
  const [error, setError]   = useState('');
  const [loading, setLoading] = useState(false);
  const { register } = useAuth();
  const navigate = useNavigate();

  const handleChange = e => setForm({ ...form, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (form.password !== form.confirmPassword) { setError('Passwords do not match'); return; }
    setError('');
    setLoading(true);
    try {
      const { confirmPassword, ...data } = form;
      await register(data);
      navigate('/dashboard');
    } catch (err) {
      const detail = err.response?.data?.detail;
      if (Array.isArray(detail)) setError(detail.map(e => e.msg).join(', '));
      else if (typeof detail === 'string') setError(detail);
      else setError(err.message || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  const inputClass = 'input-dark w-full px-4 py-2.5 rounded-xl text-sm';
  const labelClass = 'block text-xs font-semibold mb-1.5 tracking-wide uppercase';

  return (
    <div
      className="min-h-screen flex items-center justify-center relative overflow-hidden p-4"
      style={{ background: 'radial-gradient(ellipse 80% 60% at 50% -10%, rgba(5,150,105,0.12) 0%, #030712 60%)' }}
    >
      {/* Background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div
          className="absolute rounded-full blur-3xl opacity-15"
          style={{
            width: 550, height: 550,
            top: '-20%', right: '-10%',
            background: 'radial-gradient(circle, #059669 0%, transparent 70%)',
          }}
        />
        <div
          className="absolute rounded-full blur-3xl opacity-10"
          style={{
            width: 450, height: 450,
            bottom: '-15%', left: '-5%',
            background: 'radial-gradient(circle, #2563eb 0%, transparent 70%)',
          }}
        />
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
        {/* Logo */}
        <div className="text-center mb-8 fade-in">
          <div
            className="inline-flex w-14 h-14 rounded-2xl items-center justify-center mb-4"
            style={{
              background: 'linear-gradient(135deg, #059669, #0284c7)',
              boxShadow: '0 0 28px rgba(5,150,105,0.35)',
            }}
          >
            <span className="heading text-white font-bold text-2xl">V</span>
          </div>
          <h1 className="heading text-2xl font-bold text-white mb-1">Create account</h1>
          <p className="text-sm" style={{ color: 'rgba(148,163,184,0.55)' }}>
            Start your sales training journey
          </p>
        </div>

        {/* Card */}
        <div className="glass rounded-2xl p-7 slide-up">
          <form onSubmit={handleSubmit} className="space-y-3.5">
            <div>
              <label className={labelClass} style={{ color: 'rgba(148,163,184,0.6)' }}>Full Name</label>
              <input type="text" name="full_name" value={form.full_name} onChange={handleChange}
                className={inputClass} placeholder="John Doe" required />
            </div>
            <div>
              <label className={labelClass} style={{ color: 'rgba(148,163,184,0.6)' }}>Email</label>
              <input type="email" name="email" value={form.email} onChange={handleChange}
                className={inputClass} placeholder="you@company.com" required />
            </div>
            <div>
              <label className={labelClass} style={{ color: 'rgba(148,163,184,0.6)' }}>
                Company <span style={{ color: 'rgba(148,163,184,0.35)' }}>(optional)</span>
              </label>
              <input type="text" name="company" value={form.company} onChange={handleChange}
                className={inputClass} placeholder="Your company" />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className={labelClass} style={{ color: 'rgba(148,163,184,0.6)' }}>Password</label>
                <input type="password" name="password" value={form.password} onChange={handleChange}
                  className={inputClass} placeholder="••••••" required />
              </div>
              <div>
                <label className={labelClass} style={{ color: 'rgba(148,163,184,0.6)' }}>Confirm</label>
                <input type="password" name="confirmPassword" value={form.confirmPassword} onChange={handleChange}
                  className={inputClass} placeholder="••••••" required />
              </div>
            </div>

            {error && (
              <div
                className="px-4 py-3 rounded-xl text-sm"
                style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', color: '#fca5a5' }}
              >
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 rounded-xl text-sm font-semibold text-white transition-all"
              style={{
                background: 'linear-gradient(135deg, #059669, #0284c7)',
                boxShadow: '0 4px 24px rgba(5,150,105,0.25)',
                opacity: loading ? 0.5 : 1,
              }}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4 spin-ring" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.25)" strokeWidth="3"/>
                    <path d="M12 2a10 10 0 0110 10" stroke="white" strokeWidth="3" strokeLinecap="round"/>
                  </svg>
                  Creating…
                </span>
              ) : 'Create Account'}
            </button>
          </form>

          <p className="text-center mt-5 text-xs" style={{ color: 'rgba(148,163,184,0.4)' }}>
            Already have an account?{' '}
            <Link to="/login" className="text-emerald-400 font-medium hover:text-emerald-300 transition-colors">
              Sign in
            </Link>
          </p>
        </div>

        <p className="text-center mt-5 text-xs" style={{ color: 'rgba(148,163,184,0.18)' }}>
          © 2024 VCAI · MIU Thesis Project
        </p>
      </div>
    </div>
  );
}
