import axios from 'axios';

// Same-origin URLs so the same build works locally (port 8000) and through
// a Cloudflare Tunnel (https://*.trycloudflare.com). Override with the
// VITE_API_URL build-time env if you ever split frontend/backend hosts.
const _origin = (typeof window !== 'undefined' ? window.location.origin : '').replace(/\/$/, '');
const API_URL = import.meta.env.VITE_API_URL || `${_origin}/api`;
const WS_URL = import.meta.env.VITE_WS_URL || _origin.replace(/^http/, 'ws');

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

// Public — no token required.
export const getPlans = () => api.get('/plans').then(r => r.data);

export const plansAPI = {
  getAll: getPlans,
};

// A token-free axios instance for genuinely public calls. The shared `api`
// instance attaches localStorage.token to every request via its interceptor,
// which breaks public endpoints when a stale/wrong-user token is present (an
// invite-accept page must work for a logged-out or different visitor).
const publicApi = axios.create({
  baseURL: API_URL,
  headers: { 'Content-Type': 'application/json' },
});

// Onboarding (mostly public — no token required).
export const onboardingAPI = {
  signup: (data) => api.post('/onboarding/signup', data).then(r => r.data),
  // Public invite lookup — must NOT send an Authorization header.
  getInvite: (token) => publicApi.get(`/onboarding/invite/${token}`).then(r => r.data),
  accept: (data) => publicApi.post('/onboarding/accept', data).then(r => r.data),
};

// Seats (manager-only — token required).
export const seatsAPI = {
  getRoster: () => api.get('/seats').then(r => r.data),
  invite: (email, role = 'salesperson') => api.post('/seats/invite', { email, role }).then(r => r.data),
  revoke: (inviteId) => api.delete(`/seats/invite/${inviteId}`).then(r => r.data),
  deactivate: (userId) => api.post(`/seats/${userId}/deactivate`).then(r => r.data),
  // Existing logged-in user joins a company by pasting a 6-char invite code.
  join: (code) => api.post('/seats/join', { code }).then(r => r.data),
};

// Subscriptions (manager-only — token required).
export const subscriptionsAPI = {
  getMine: () => api.get('/subscriptions/me').then(r => r.data),
  changePlan: (planName, billingCycle = 'monthly') =>
    api.post('/subscriptions/change-plan', { plan_name: planName, billing_cycle: billingCycle }).then(r => r.data),
};
  export const learningAPI = {
  getProfile: (lastN = 5) =>
    api.get(`/learning/profile?last_n=${lastN}`).then(r => r.data),
  getProgress: (groupBy = 'month') =>
    api.get(`/learning/progress?group_by=${groupBy}`).then(r => r.data),
  getInsights: () =>
    api.get('/learning/insights').then(r => r.data),
};

// Manager dashboard (manager/superadmin — token required).
export const managerAPI = {
  getAgents: () => api.get('/manager/agents').then(r => r.data),
  getAgentProgress: (userId) => api.get(`/manager/agents/${userId}/progress`).then(r => r.data),
  getAnalytics: () => api.get('/manager/analytics').then(r => r.data),
  getEmotionTrends: () => api.get('/manager/emotion-trends').then(r => r.data),
  getAbuse: (status = null) =>
    api.get('/manager/abuse', { params: status ? { status } : {} }).then(r => r.data),
  resolveAbuse: (flagId, status, note = null) =>
    api.post(`/manager/abuse/${flagId}/resolve`, { status, note }).then(r => r.data),
};

// Super-admin (platform owner — superadmin token required).
export const adminAPI = {
  getTenants: (params = {}) => api.get('/admin/tenants', { params }).then(r => r.data),
  getTenant: (companyId) => api.get(`/admin/tenants/${companyId}`).then(r => r.data),
  suspendTenant: (companyId) => api.post(`/admin/tenants/${companyId}/suspend`).then(r => r.data),
  reactivateTenant: (companyId) => api.post(`/admin/tenants/${companyId}/reactivate`).then(r => r.data),
  getUsage: () => api.get('/admin/usage').then(r => r.data),
  getAbuse: (params = {}) => api.get('/admin/abuse', { params }).then(r => r.data),
  getAudit: (params = {}) => api.get('/admin/audit', { params }).then(r => r.data),
  getHealth: () => api.get('/admin/health').then(r => r.data),
};

export const sessionsAPI = {
  create: (personaId, difficulty, scenario = null, trainingFocus = null) =>
    api.post('/sessions', {
      persona_id: personaId,
      difficulty,
      scenario,
      training_focus: trainingFocus,
    }).then(r => r.data),
  getScenarioPresets: () => api.get('/sessions/scenario-presets').then(r => r.data),
  getAll: (limit = 20, offset = 0) => api.get(`/sessions?limit=${limit}&offset=${offset}`).then(r => r.data),
  getById: (id) => api.get(`/sessions/${id}`).then(r => r.data),
  getMessages: (id) => api.get(`/sessions/${id}/messages`).then(r => r.data),
  end: (id) => api.post(`/sessions/${id}/end`).then(r => r.data),
  reactivate: (id) => api.post(`/sessions/${id}/reactivate`).then(r => r.data),
  // Build a fully-qualified audio URL with the JWT in the query string —
  // HTML5 <audio> tags can't send Authorization headers.
  messageAudioUrl: (sessionId, messageId) => {
    const token = localStorage.getItem('token') || '';
    return `${API_URL}/sessions/${sessionId}/messages/${messageId}/audio?token=${encodeURIComponent(token)}`;
  },
};

export const evaluationAPI = {
  triggerEvaluation: (sessionId, mode = 'training', force = false) =>
    api.post(`/sessions/${sessionId}/evaluate?mode=${mode}&force=${force}`).then(r => r.data),
  getStatus: (sessionId) =>
    api.get(`/sessions/${sessionId}/eval-status`).then(r => r.data),
  getQuickStats: (sessionId) =>
    api.get(`/sessions/${sessionId}/quick-stats`).then(r => r.data),
  getReport: (sessionId) =>
    api.get(`/sessions/${sessionId}/report`).then(r => r.data),
};

export const createWebSocket = (sessionId) => {
  const token = localStorage.getItem('token');
  return new WebSocket(`${WS_URL}/ws/${sessionId}?token=${token}`);
};

export default api;
