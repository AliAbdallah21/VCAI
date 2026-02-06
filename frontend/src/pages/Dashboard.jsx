import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { sessionsAPI } from '../services/api';
import Layout from '../components/Layout';

export default function Dashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({ total: 0, avgScore: 0, totalMinutes: 0 });

  useEffect(() => {
    sessionsAPI.getAll(5)
      .then(data => {
        setSessions(data.sessions);
        const completed = data.sessions.filter(s => s.overall_score);
        setStats({
          total: data.total,
          avgScore: completed.length ? Math.round(completed.reduce((a, b) => a + b.overall_score, 0) / completed.length) : 0,
          totalMinutes: Math.round(data.sessions.reduce((a, b) => a + (b.duration_seconds || 0), 0) / 60)
        });
      })
      .finally(() => setLoading(false));
  }, []);

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  };

  return (
    <Layout>
      <div className="p-8">
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-slate-800">Welcome back, {user?.full_name?.split(' ')[0]}! 👋</h1>
          <p className="text-slate-500">Here is your training overview</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-500 text-sm">Total Sessions</p>
                <p className="text-3xl font-bold text-slate-800 mt-1">{stats.total}</p>
              </div>
              <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center text-2xl">📊</div>
            </div>
          </div>
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-500 text-sm">Average Score</p>
                <p className="text-3xl font-bold text-emerald-600 mt-1">{stats.avgScore || '—'}</p>
              </div>
              <div className="w-12 h-12 bg-emerald-100 rounded-xl flex items-center justify-center text-2xl">🎯</div>
            </div>
          </div>
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-500 text-sm">Training Time</p>
                <p className="text-3xl font-bold text-slate-800 mt-1">{stats.totalMinutes}m</p>
              </div>
              <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center text-2xl">⏱️</div>
            </div>
          </div>
        </div>

        <Link
          to="/setup"
          className="block bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-8 text-white mb-8 hover:opacity-95 transition"
        >
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-xl font-bold mb-2">Start New Training Session</h3>
              <p className="text-blue-100">Practice with AI customers and improve your skills</p>
            </div>
            <div className="text-4xl">→</div>
          </div>
        </Link>

        <div className="bg-white rounded-2xl shadow-sm border border-slate-100">
          <div className="px-6 py-4 border-b border-slate-100">
            <h2 className="font-semibold text-slate-800">Recent Sessions</h2>
          </div>
          {loading ? (
            <div className="p-8 text-center text-slate-500">Loading...</div>
          ) : sessions.length === 0 ? (
            <div className="p-12 text-center">
              <div className="text-5xl mb-4">🎯</div>
              <h3 className="font-medium text-slate-800 mb-2">No sessions yet</h3>
              <p className="text-slate-500">Start your first training session!</p>
            </div>
          ) : (
            <div className="divide-y divide-slate-100">
              {sessions.map((s) => (
                <div
                  key={s.id}
                  className="px-6 py-4 flex items-center justify-between hover:bg-slate-50 cursor-pointer transition"
                  onClick={() => navigate(`/evaluation/${s.id}`)}
                >
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 bg-slate-100 rounded-full flex items-center justify-center text-lg">👤</div>
                    <div>
                      <p className="font-medium text-slate-800">{s.persona_name || s.persona_id}</p>
                      <p className="text-sm text-slate-500">{formatDate(s.started_at)}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${s.difficulty === 'easy' ? 'bg-emerald-100 text-emerald-700' :
                        s.difficulty === 'hard' ? 'bg-red-100 text-red-700' :
                          'bg-amber-100 text-amber-700'
                      }`}>
                      {s.difficulty}
                    </span>
                    <div className="text-xl font-bold text-emerald-600">{s.overall_score || '—'}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
