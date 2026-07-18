import {
  createContext,
  useContext,
  useState,
  useEffect,
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

  const verifyToken = async () => {
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
  };

  useEffect(() => {
    void verifyToken();
    // verifyToken intentionally follows the current token lifecycle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const login = async (username: string, password: string) => {
    const data = await fetchAPI<{ access_token: string; user: User }>(
      "/api/auth/login",
      { username, password }
    );
    localStorage.setItem("auth_token", data.access_token);
    setToken(data.access_token);
    setUser(data.user);
    setVerification("authenticated");
    setCloudStatus("ready");
  };

  const logout = () => {
    localStorage.removeItem("auth_token");
    setToken(null);
    setUser(null);
    setVerification("anonymous");
    setCloudStatus("ready");
  };

  return (
    <AuthContext.Provider value={{ user, token, login, logout, loading, verification, cloudStatus, retryVerification: verifyToken }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
