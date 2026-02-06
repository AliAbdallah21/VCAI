import axios from 'axios';

const API_URL = 'http://localhost:8000/api';
const WS_URL = 'ws://localhost:8000';

const api = axios.create({ 
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export const authAPI = {
  register: (data) => {
    console.log('Sending registration:', data);
    return api.post('/auth/register', data).then(r => r.data);
  },
  login: (email, password) => {
    const form = new URLSearchParams();
    form.append('username', email);
    form.append('password', password);
    return api.post('/auth/login', form, { 
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    }).then(r => r.data);
  },
  getMe: () => api.get('/auth/me').then(r => r.data),
};

export const personasAPI = {
  getAll: () => api.get('/personas').then(r => r.data),
};

export const sessionsAPI = {
  create: (personaId, difficulty) => api.post('/sessions', { persona_id: personaId, difficulty }).then(r => r.data),
  getAll: (limit = 20) => api.get(`/sessions?limit=${limit}`).then(r => r.data),
  getById: (id) => api.get(`/sessions/${id}`).then(r => r.data),
  end: (id) => api.post(`/sessions/${id}/end`).then(r => r.data),
};

export const evaluationAPI = {
  triggerEvaluation: (sessionId, mode = 'training') => 
    api.post(`/evaluation/${sessionId}/trigger?mode=${mode}`).then(r => r.data),
  getStatus: (sessionId) => 
    api.get(`/evaluation/${sessionId}/status`).then(r => r.data),
  getQuickStats: (sessionId) => 
    api.get(`/evaluation/${sessionId}/quick-stats`).then(r => r.data),
  getReport: (sessionId) => 
    api.get(`/evaluation/${sessionId}/report`).then(r => r.data),
};

export const createWebSocket = (sessionId) => {
  const token = localStorage.getItem('token');
  return new WebSocket(`${WS_URL}/ws/${sessionId}?token=${token}`);
};

export default api;
