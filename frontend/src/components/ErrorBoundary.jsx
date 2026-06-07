import { Component } from 'react';

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    // Surface the crash to the console for debugging — also a good place to
    // hook into Sentry / a remote logger later.
    console.error('[ErrorBoundary]', error, info?.componentStack);
  }

  handleReload = () => {
    window.location.reload();
  };

  handleGoHome = () => {
    window.location.href = '/dashboard';
  };

  render() {
    if (!this.state.hasError) return this.props.children;

    const msg = this.state.error?.message || 'Unknown error';

    return (
      <div
        className="min-h-screen flex items-center justify-center px-6"
        style={{ background: '#030712' }}
      >
        <div
          className="max-w-md w-full rounded-2xl p-8 text-center"
          style={{
            background: 'rgba(13,21,38,0.7)',
            border: '1px solid rgba(239,68,68,0.2)',
            backdropFilter: 'blur(20px)',
          }}
        >
          <div
            className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-5"
            style={{ background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.2)' }}
          >
            <svg width="24" height="24" fill="none" stroke="#f87171" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          <h2 className="text-lg font-semibold text-white mb-2">Something went wrong</h2>
          <p className="text-sm mb-1" style={{ color: 'rgba(148,163,184,0.7)' }}>
            The page hit an unexpected error. You can reload or go back to the dashboard.
          </p>
          <p
            className="text-xs mb-6 font-mono truncate"
            style={{ color: 'rgba(239,68,68,0.6)' }}
            title={msg}
          >
            {msg}
          </p>
          <div className="flex gap-3 justify-center">
            <button
              onClick={this.handleReload}
              className="px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200"
              style={{
                background: 'linear-gradient(135deg, #2563eb, #7c3aed)',
                color: 'white',
                boxShadow: '0 0 20px rgba(59,130,246,0.25)',
              }}
            >
              Reload page
            </button>
            <button
              onClick={this.handleGoHome}
              className="px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200"
              style={{
                background: 'rgba(255,255,255,0.06)',
                color: '#cbd5e1',
                border: '1px solid rgba(255,255,255,0.08)',
              }}
            >
              Go to dashboard
            </button>
          </div>
        </div>
      </div>
    );
  }
}
