import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  ReactNode,
} from "react";
import { APIError, fetchAPI } from "../lib/api";

export type VerificationState = "checking" | "authenticated" | "anonymous" | "unverified";
export type CloudStatus = "connecting" | "waking" | "ready" | "unavailable";

interface User {
  id: number;
  username: string;
  display_name: string;
  role: "admin" | "teacher" | "student";
  semester: string;
  must_change_password?: boolean;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  loading: boolean;
  verification: VerificationState;
  cloudStatus: CloudStatus;
  retryVerification: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  token: null,
  login: async () => {},
  logout: () => {},
  loading: true,
  verification: "anonymous",
  cloudStatus: "ready",
  retryVerification: async () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(
    () => localStorage.getItem("auth_token")
  );
  const [loading, setLoading] = useState(true);
  const [verification, setVerification] = useState<VerificationState>(token ? "checking" : "anonymous");
  const [cloudStatus, setCloudStatus] = useState<CloudStatus>(token ? "connecting" : "ready");

  const verifyToken = useCallback(async () => {
    if (!token) {
      setVerification("anonymous");
      setCloudStatus("ready");
      setLoading(false);
      return;
    }
    setVerification("checking");
    setCloudStatus("connecting");
    setLoading(true);
    const wakingTimer = window.setTimeout(() => setCloudStatus("waking"), 3000);
    try {
      const verifiedUser = await fetchAPI<User>("/api/auth/me", undefined, token, { timeoutMs: 8000 });
      setUser(verifiedUser);
      setVerification("authenticated");
      setCloudStatus("ready");
    } catch (error) {
      if (error instanceof APIError && error.kind === "unauthorized") {
        localStorage.removeItem("auth_token");
        setToken(null);
        setVerification("anonymous");
        setCloudStatus("ready");
      } else {
        setVerification("unverified");
        setCloudStatus("unavailable");
      }
    } finally {
      window.clearTimeout(wakingTimer);
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    void verifyToken();
  }, [verifyToken]);

  const login = useCallback(async (username: string, password: string) => {
    const data = await fetchAPI<{ access_token: string; user: User }>(
      "/api/auth/login",
      { username, password }
    );
    localStorage.setItem("auth_token", data.access_token);
    setToken(data.access_token);
    setUser(data.user);
    setVerification("authenticated");
    setCloudStatus("ready");
  }, []);

  const logout = useCallback(() => {
    if (token) {
      fetchAPI("/api/auth/logout", {}, token).catch(() => {});
    }
    localStorage.removeItem("auth_token");
    setToken(null);
    setUser(null);
    setVerification("anonymous");
    setCloudStatus("ready");
  }, [token]);

  const contextValue = useMemo(
    () => ({ user, token, login, logout, loading, verification, cloudStatus, retryVerification: verifyToken }),
    [user, token, login, logout, loading, verification, cloudStatus, verifyToken]
  );

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
}

// This module intentionally exposes the provider and its paired consumer hook.
// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
  return useContext(AuthContext);
}
