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

  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, isAuthenticated: !!token, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};
