import { createContext, useContext, useState, useEffect } from 'react';
import { authAPI } from '../services/api';

const AuthContext = createContext(null);

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);
  // const [loading, setLoading] = useState(false);

  // useEffect(() => {
  //   const init = async () => {
  //     const savedToken = localStorage.getItem('token');
  //     const savedUser = localStorage.getItem('user');
  //     if (savedToken && savedUser) {
  //       setToken(savedToken);
  //       setUser(JSON.parse(savedUser));
  //       try {
  //         const userData = await authAPI.getMe();
  //         setUser(userData);
  //         localStorage.setItem('user', JSON.stringify(userData));
  //       } catch { 
  //         logout(); 
  //       }
  //     }
  //     setLoading(false);
  //   };
  //   init();
  // }, []);

  useEffect(() => {
  let mounted = true;

  const init = async () => {
    try {
      const savedToken = localStorage.getItem('token');
      if (!savedToken) return;

      setToken(savedToken);
      const userData = await authAPI.getMe();
      if (mounted) {
        setUser(userData);
        localStorage.setItem('user', JSON.stringify(userData));
      }
    } catch (err) {
      logout();
    } finally {
      if (mounted) setLoading(false);
    }
  };

  init();
  return () => { mounted = false; };
  }, []);










  
  const login = async (email, password) => {
    const data = await authAPI.login(email, password);
    setToken(data.access_token);
    setUser(data.user);
    localStorage.setItem('token', data.access_token);
    localStorage.setItem('user', JSON.stringify(data.user));
    return data;
  };

  const register = async (userData) => {
    const data = await authAPI.register(userData);
    setToken(data.access_token);
    setUser(data.user);
    localStorage.setItem('token', data.access_token);
    localStorage.setItem('user', JSON.stringify(data.user));
    return data;
  };

  // Establish a session from an already-issued token + user (onboarding/invite).
  const setAuth = ({ access_token, user: authUser }) => {
    setToken(access_token);
    setUser(authUser);
    localStorage.setItem('token', access_token);
    localStorage.setItem('user', JSON.stringify(authUser));
  };

  // Re-fetch the current user from the server and update context + storage.
  // Used after an action changes the account server-side (e.g. joining a company
  // by code changes company_id + role) so routing reflects the new state.
  const refreshUser = async () => {
    const userData = await authAPI.getMe();
    setUser(userData);
    localStorage.setItem('user', JSON.stringify(userData));
    return userData;
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    // Clear all auth keys, including legacy ones from earlier builds, so a stale
    // token can never linger and silently re-authenticate the next visitor.
    ['token', 'user', 'fitai_access_token', 'fitai_refresh_token'].forEach((k) =>
      localStorage.removeItem(k)
    );
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, isAuthenticated: !!token, login, register, setAuth, refreshUser, logout }}>
      {children}
    </AuthContext.Provider>
  );
};
