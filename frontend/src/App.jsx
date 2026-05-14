import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import ErrorBoundary from './components/ErrorBoundary';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import SessionSetup from './pages/SessionSetup';
import TrainingSession from './pages/TrainingSession';
import EvaluationReport from './pages/EvaluationReport';
import SessionsPage from './pages/SessionsPage';
import EvaluatePage from './pages/EvaluatePage';
import ComparePage from './pages/ComparePage';

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

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<PublicRoute><Login /></PublicRoute>} />
      <Route path="/register" element={<PublicRoute><Register /></PublicRoute>} />
      <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
      <Route path="/setup" element={<ProtectedRoute><SessionSetup /></ProtectedRoute>} />
      <Route path="/session/:sessionId" element={<ProtectedRoute><TrainingSession /></ProtectedRoute>} />
      <Route path="/evaluation/:sessionId" element={<ProtectedRoute><EvaluationReport /></ProtectedRoute>} />
      <Route path="/sessions" element={<ProtectedRoute><SessionsPage /></ProtectedRoute>} />
      <Route path="/evaluate" element={<ProtectedRoute><EvaluatePage /></ProtectedRoute>} />
      <Route path="/compare" element={<ProtectedRoute><ComparePage /></ProtectedRoute>} />
      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <AuthProvider>
          <AppRoutes />
        </AuthProvider>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
