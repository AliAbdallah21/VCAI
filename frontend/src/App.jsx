import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ThemeProvider } from './context/ThemeContext';
import ErrorBoundary from './components/ErrorBoundary';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import SessionSetup from './pages/SessionSetup';
import TrainingSession from './pages/TrainingSession';
import EvaluationReport from './pages/EvaluationReport';
import Landing from './pages/Landing';
import About from './pages/About';
import Privacy from './pages/Privacy';
import Contact from './pages/Contact';
import ChatWidget from './components/ChatWidget';
import Onboarding from './pages/Onboarding';
import AcceptInvite from './pages/AcceptInvite';
import SeatManagement from './pages/SeatManagement';
import SessionsPage from './pages/SessionsPage';
import EvaluatePage from './pages/EvaluatePage';
import ComparePage from './pages/ComparePage';
import ProgressPage from './pages/ProgressPage';
import Settings from './pages/Settings';
import ManagerDashboard from './pages/ManagerDashboard';
import AgentProgress from './pages/AgentProgress';
import AdminDashboard from './pages/AdminDashboard';

const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();
  if (loading) return <div className="min-h-screen flex items-center justify-center">Loading...</div>;
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return children;
};

const PublicRoute = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();
  if (loading) return <div className="min-h-screen flex items-center justify-center">Loading...</div>;
  if (isAuthenticated) return <Navigate to="/dashboard" replace />;
  return children;
};

const RoleRoute = ({ roles, children }) => {
  const { isAuthenticated, loading, user } = useAuth();
  if (loading) return <div className="min-h-screen flex items-center justify-center">Loading...</div>;
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  if (!roles.includes(user?.role)) return <Navigate to="/dashboard" replace />;
  return children;
};

// Branch the landing dashboard by role: managers/superadmins get the team
// dashboard; salespeople get the trainee Dashboard.
const DashboardRouter = () => {
  const { user } = useAuth();
  if (user?.role === 'superadmin') return <AdminDashboard />;
  if (user?.role === 'manager') return <ManagerDashboard />;
  return <Dashboard />;
};

const CatchAll = () => {
  const { isAuthenticated, loading } = useAuth();
  if (loading) return <div className="min-h-screen flex items-center justify-center">Loading...</div>;
  return <Navigate to={isAuthenticated ? '/dashboard' : '/'} replace />;
};

const PUBLIC_CHAT_PATHS = ['/', '/about', '/privacy', '/contact'];

function PublicChatWrapper() {
  const { pathname } = useLocation();
  if (!PUBLIC_CHAT_PATHS.includes(pathname)) return null;
  return <ChatWidget />;
}

function AppRoutes() {
  return (
    <>
    <PublicChatWrapper />
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/about" element={<About />} />
      <Route path="/privacy" element={<Privacy />} />
      <Route path="/contact" element={<Contact />} />
      <Route path="/onboarding" element={<Onboarding />} />
      <Route path="/invite/:token" element={<AcceptInvite />} />
      <Route path="/login" element={<PublicRoute><Login /></PublicRoute>} />
      <Route path="/register" element={<PublicRoute><Register /></PublicRoute>} />
      <Route path="/dashboard" element={<ProtectedRoute><DashboardRouter /></ProtectedRoute>} />
      <Route path="/manager/agents/:id" element={<RoleRoute roles={['manager', 'superadmin']}><AgentProgress /></RoleRoute>} />
      <Route path="/admin" element={<RoleRoute roles={['superadmin']}><AdminDashboard /></RoleRoute>} />
      <Route path="/seats" element={<RoleRoute roles={['manager', 'superadmin']}><SeatManagement /></RoleRoute>} />
      <Route path="/setup" element={<ProtectedRoute><SessionSetup /></ProtectedRoute>} />
      <Route path="/session/:sessionId" element={<ProtectedRoute><TrainingSession /></ProtectedRoute>} />
      <Route path="/evaluation/:sessionId" element={<ProtectedRoute><EvaluationReport /></ProtectedRoute>} />
      <Route path="*" element={<CatchAll />} />
      <Route path="/sessions" element={<ProtectedRoute><SessionsPage /></ProtectedRoute>} />
      <Route path="/evaluate" element={<ProtectedRoute><EvaluatePage /></ProtectedRoute>} />
      <Route path="/compare" element={<ProtectedRoute><ComparePage /></ProtectedRoute>} />
      <Route path="/progress" element={<ProtectedRoute><ProgressPage /></ProtectedRoute>} />
      <Route path="/settings" element={<ProtectedRoute><Settings /></ProtectedRoute>} />
      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
    </>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <BrowserRouter>
          <AuthProvider>
            <AppRoutes />
          </AuthProvider>
        </BrowserRouter>
      </ThemeProvider>
    </ErrorBoundary>
  );
}
