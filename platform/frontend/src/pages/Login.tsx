import { useState } from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import { APIError, fetchAPI } from "../lib/api";
import ThemeToggle from "../components/ThemeToggle";

const INPUT =
  "w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500";

function loginErrorMessage(err: unknown): string {
  // 403 carries a specific reason (pending approval, rejected, disabled);
  // anything else stays generic so usernames can't be probed.
  if (err instanceof APIError && err.status === 403 && err.detail) return err.detail;
  if (err instanceof APIError && (err.kind === "timeout" || err.kind === "offline")) {
    return "無法連線到伺服器，請稍後再試（雲端服務可能正在喚醒）";
  }
  return "帳號或密碼錯誤";
}

function SignupForm({ onDone, onCancel }: { onDone: (username: string) => void; onCancel: () => void }) {
  const [form, setForm] = useState({ username: "", display_name: "", class_name: "", email: "", password: "", confirm: "" });
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) => setForm({ ...form, [k]: e.target.value });

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (!/^[A-Za-z0-9_.-]{3,32}$/.test(form.username.trim())) {
      setError("學號（帳號）需為 3–32 個英數字，可含 _ . -");
      return;
    }
    if (form.password.length < 8) { setError("密碼至少 8 個字元"); return; }
    if (form.password !== form.confirm) { setError("兩次輸入的密碼不一致"); return; }
    setSubmitting(true);
    try {
      await fetchAPI("/api/auth/signup", {
        username: form.username.trim(),
        display_name: form.display_name.trim(),
        class_name: form.class_name.trim(),
        email: form.email.trim(),
        password: form.password,
      });
      onDone(form.username.trim());
    } catch (err) {
      if (err instanceof APIError && err.status === 409) setError(err.detail || "此帳號已存在或已申請過");
      else if (err instanceof APIError && err.status === 422) setError("欄位格式不正確，請檢查後再送出");
      else setError("申請送出失敗，請稍後再試");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={submit} className="space-y-3" aria-label="申請帳號">
      <label className="block text-sm font-medium text-gray-700">學號（作為登入帳號）*
        <input required value={form.username} onChange={set("username")} className={`mt-1 ${INPUT}`} autoFocus placeholder="例如 s1150001" />
      </label>
      <label className="block text-sm font-medium text-gray-700">姓名 *
        <input required value={form.display_name} onChange={set("display_name")} className={`mt-1 ${INPUT}`} />
      </label>
      <label className="block text-sm font-medium text-gray-700">班級
        <input value={form.class_name} onChange={set("class_name")} className={`mt-1 ${INPUT}`} placeholder="例如 護理一甲" />
      </label>
      <label className="block text-sm font-medium text-gray-700">Email
        <input type="email" value={form.email} onChange={set("email")} className={`mt-1 ${INPUT}`} />
      </label>
      <label className="block text-sm font-medium text-gray-700">密碼（至少 8 字元）*
        <input required type="password" value={form.password} onChange={set("password")} className={`mt-1 ${INPUT}`} autoComplete="new-password" />
      </label>
      <label className="block text-sm font-medium text-gray-700">確認密碼 *
        <input required type="password" value={form.confirm} onChange={set("confirm")} className={`mt-1 ${INPUT}`} autoComplete="new-password" />
      </label>
      {error && <p className="text-red-500 text-sm" role="alert">{error}</p>}
      <button type="submit" disabled={submitting}
        className="w-full py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50">
        {submitting ? "送出中..." : "送出申請"}
      </button>
      <button type="button" onClick={onCancel} className="w-full py-2 text-sm text-gray-500 hover:text-gray-800">
        返回登入
      </button>
    </form>
  );
}

export default function Login() {
  const { user, login, loading } = useAuth();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <p className="text-gray-400">Loading...</p>
      </div>
    );
  }

  if (user) {
    return <Navigate to="/" replace />;
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password.trim()) return;
    setError("");
    setNotice("");
    setSubmitting(true);
    try {
      await login(username.trim(), password.trim());
    } catch (err) {
      setError(loginErrorMessage(err));
    }
    setSubmitting(false);
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-50 py-8">
      <div className="absolute top-4 right-4"><ThemeToggle /></div>
      <div className="w-full max-w-sm p-8 bg-white rounded-2xl shadow-sm border border-gray-200">
        <h1 className="text-2xl font-bold text-gray-900 mb-1 text-center">
          ML/DL 視覺化教學平台
        </h1>
        <p className="text-sm text-gray-500 text-center mb-6">
          {mode === "login" ? "請登入以開始學習" : "申請帳號（需經老師核准後才能登入）"}
        </p>

        {mode === "signup" ? (
          <SignupForm
            onCancel={() => setMode("login")}
            onDone={(u) => {
              setMode("login");
              setUsername(u);
              setPassword("");
              setNotice("申請已送出！老師核准後即可用此學號與密碼登入。");
            }}
          />
        ) : (
          <>
            {notice && <p className="mb-4 px-3 py-2 rounded-lg bg-green-50 text-green-700 text-sm" role="status">{notice}</p>}
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  帳號
                </label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className={INPUT}
                  placeholder="請輸入帳號（學號）"
                  autoFocus
                  aria-label="Username"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  密碼
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className={INPUT}
                  placeholder="請輸入密碼"
                  aria-label="Password"
                />
              </div>

              {error && (
                <p className="text-red-500 text-sm" role="alert">{error}</p>
              )}

              <button
                type="submit"
                disabled={submitting}
                className="w-full py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
              >
                {submitting ? "登入中..." : "登入"}
              </button>
            </form>
            <p className="text-sm text-gray-500 text-center mt-4">
              還沒有帳號？
              <button type="button" onClick={() => { setMode("signup"); setError(""); setNotice(""); }}
                className="text-blue-600 hover:underline ml-1">
                申請帳號
              </button>
            </p>
          </>
        )}

        <p className="text-xs text-gray-400 text-center mt-6">
          18 週機器學習與深度學習視覺化互動課程
        </p>
      </div>
    </div>
  );
}
