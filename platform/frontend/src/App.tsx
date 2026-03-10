import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import ErrorBoundary from "./components/ErrorBoundary";
import { AuthProvider, useAuth } from "./hooks/useAuth";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import WeekPage from "./pages/WeekPage";
import Dashboard from "./pages/Dashboard";
import AdminSettings from "./pages/AdminSettings";
import UserManagement from "./pages/UserManagement";
import Login from "./pages/Login";
import NotFound from "./pages/NotFound";

function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <p className="text-gray-400">Loading...</p>
      </div>
    );
  }
  if (!user) return <Navigate to="/login" replace />;
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
                </RequireAuth>
              }
            >
              <Route index element={<Home />} />
              <Route path="week/:weekId" element={<WeekPage />} />
              <Route path="dashboard" element={<Dashboard />} />
              <Route path="admin" element={<AdminSettings />} />
              <Route path="admin/users" element={<UserManagement />} />
              <Route path="*" element={<NotFound />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </ErrorBoundary>
  );
}
