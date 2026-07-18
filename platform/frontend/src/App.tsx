import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { useState } from "react";
import ErrorBoundary from "./components/ErrorBoundary";
import { AuthProvider, useAuth } from "./hooks/useAuth";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import WeekPage from "./pages/WeekPage";
import Dashboard from "./pages/Dashboard";
import AdminSettings from "./pages/AdminSettings";
import UserManagement from "./pages/UserManagement";
import QuizManagement from "./pages/QuizManagement";
import Login from "./pages/Login";
import NotFound from "./pages/NotFound";
import CloudLoadingState from "./components/CloudLoadingState";
import ChangePasswordDialog from "./components/auth/ChangePasswordDialog";

function ForcedPasswordGate() {
  const { user } = useAuth();
  const [dismissed, setDismissed] = useState(false);
  if (!user?.must_change_password || dismissed) return null;
  return <ChangePasswordDialog forced onClose={() => setDismissed(true)} />;
}

function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user, loading, verification, cloudStatus, retryVerification, logout } = useAuth();
  const [offlineEntry, setOfflineEntry] = useState(false);
  if (loading && !offlineEntry) {
    return <CloudLoadingState phase={cloudStatus === "waking" ? "waking" : "connecting"} onEnterOffline={() => setOfflineEntry(true)} onRetry={() => void retryVerification()} onLogout={logout} />;
  }
  if (verification === "unverified" && !offlineEntry) return <CloudLoadingState phase="unavailable" onEnterOffline={() => setOfflineEntry(true)} onRetry={() => void retryVerification()} onLogout={logout} />;
  if (offlineEntry || verification === "unverified") return <>{children}</>;
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function RequireVerified({ children }: { children: React.ReactNode }) {
  const { user, verification } = useAuth();
  if (!user || verification !== "authenticated") return <Navigate to="/" replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <ErrorBoundary>
      <AuthProvider>
        <BrowserRouter basename={import.meta.env.BASE_URL}>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route
              path="/"
              element={
                <RequireAuth>
                  <Layout />
                  <ForcedPasswordGate />
                </RequireAuth>
              }
            >
              <Route index element={<Home />} />
              <Route path="week/:weekId" element={<WeekPage />} />
              <Route path="dashboard" element={<RequireVerified><Dashboard /></RequireVerified>} />
              <Route path="admin" element={<RequireVerified><AdminSettings /></RequireVerified>} />
              <Route path="admin/users" element={<RequireVerified><UserManagement /></RequireVerified>} />
              <Route path="admin/quiz" element={<RequireVerified><QuizManagement /></RequireVerified>} />
              <Route path="*" element={<NotFound />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </ErrorBoundary>
  );
}
