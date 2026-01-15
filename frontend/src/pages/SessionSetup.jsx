import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { personasAPI, sessionsAPI } from '../services/api';
import Layout from '../components/Layout';

export default function SessionSetup() {
  const navigate = useNavigate();
  const [personas, setPersonas] = useState([]);
  const [selected, setSelected] = useState(null);
  const [difficulty, setDifficulty] = useState('medium');
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    personasAPI.getAll()
      .then(data => setPersonas(data.personas))
      .finally(() => setLoading(false));
  }, []);

  const handleStart = async () => {
    if (!selected) return;
    setCreating(true);
    try {
      const session = await sessionsAPI.create(selected.id, difficulty);
      navigate(`/session/${session.id}`);
    } catch (e) {
      alert('Failed to create session');
      setCreating(false);
    }
  };

  const filtered = personas.filter(p => p.difficulty === difficulty);

  const difficulties = [
    { id: 'easy', label: 'Easy', icon: '😊', color: 'emerald' },
    { id: 'medium', label: 'Medium', icon: '😐', color: 'amber' },
    { id: 'hard', label: 'Hard', icon: '😤', color: 'red' },
  ];

  return (
    <Layout>
      <div className="p-8 max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-slate-800">New Training Session</h1>
          <p className="text-slate-500">Choose your difficulty and customer persona</p>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-slate-100 p-6 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-8 h-8 bg-blue-600 text-white rounded-lg flex items-center justify-center font-bold">1</div>
            <h2 className="font-semibold text-slate-800">Select Difficulty</h2>
          </div>
          <div className="grid grid-cols-3 gap-4">
            {difficulties.map(d => (
              <button 
                key={d.id} 
                onClick={() => { setDifficulty(d.id); setSelected(null); }} 
                className={`p-4 rounded-xl border-2 transition ${
                  difficulty === d.id 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-slate-200 hover:border-slate-300'
                }`}
              >
                <div className="text-3xl mb-2">{d.icon}</div>
                <div className="font-semibold text-slate-800 capitalize">{d.label}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-slate-100 p-6 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-8 h-8 bg-blue-600 text-white rounded-lg flex items-center justify-center font-bold">2</div>
            <h2 className="font-semibold text-slate-800">Choose Customer</h2>
          </div>
          {loading ? (
            <div className="text-center py-8 text-slate-500">Loading personas...</div>
          ) : filtered.length === 0 ? (
            <div className="text-center py-8 text-slate-500">No personas for this difficulty</div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filtered.map(p => (
                <button 
                  key={p.id} 
                  onClick={() => setSelected(p)} 
                  className={`p-4 rounded-xl border-2 text-left transition ${
                    selected?.id === p.id 
                      ? 'border-blue-500 bg-blue-50' 
                      : 'border-slate-200 hover:border-slate-300'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className="w-12 h-12 bg-slate-200 rounded-full flex items-center justify-center text-2xl flex-shrink-0">👤</div>
                    <div>
                      <div className="font-semibold text-slate-800">{p.name_en}</div>
                      <div className="text-sm text-slate-500 mt-1">{p.description_en}</div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {selected && (
          <div className="bg-blue-50 rounded-2xl p-6 mb-6 border border-blue-200">
            <h3 className="font-semibold text-slate-800 mb-2">Selected:</h3>
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 bg-white rounded-full flex items-center justify-center shadow-sm text-3xl">👤</div>
              <div>
                <div className="font-bold text-slate-800">{selected.name_en}</div>
                <div className="text-slate-600">{difficulty} difficulty</div>
              </div>
            </div>
          </div>
        )}

        <button 
          onClick={handleStart} 
          disabled={!selected || creating} 
          className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 rounded-xl font-semibold text-lg hover:opacity-95 transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {creating ? 'Creating Session...' : '🎤 Start Training Session'}
        </button>
      </div>
    </Layout>
  );
}
