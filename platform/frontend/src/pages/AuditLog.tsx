import { useCallback, useEffect, useState } from "react";
import { fetchAPI, API_BASE } from "../lib/api";
import { useAuth } from "../hooks/useAuth";

interface AuditItem {
  id: number;
  timestamp: string;
  actor_username: string;
  actor_role: string;
  action: string;
  target_type: string;
  target_id: string;
  detail: string;
  ip: string;
}

interface AuditResponse {
  items: AuditItem[];
  total: number;
  page: number;
  page_size: number;
}

type Tab = "admin" | "login" | "learning";
const PAGE_SIZE = 50;

export default function AuditLog() {
  const { token } = useAuth();
  const [tab, setTab] = useState<Tab>("admin");
  const [page, setPage] = useState(1);
  const [actionFilter, setActionFilter] = useState("");
  const [data, setData] = useState<AuditResponse | null>(null);
  const [error, setError] = useState("");

  const buildQuery = useCallback(
    (forExport = false) => {
      const params = new URLSearchParams();
      if (tab === "login") params.set("action_prefix", "login");
      else if (tab === "admin" && actionFilter) params.set("action_prefix", actionFilter);
      if (!forExport) {
        params.set("page", String(page));
        params.set("page_size", String(PAGE_SIZE));
      }
      return params.toString();
    },
    [tab, actionFilter, page],
  );

  useEffect(() => {
    if (tab === "learning") return;
    let cancelled = false;
    setError("");
    fetchAPI<AuditResponse>(`/api/admin/audit-logs?${buildQuery()}`, undefined, token ?? undefined)
      .then((res) => { if (!cancelled) setData(res); })
      .catch(() => { if (!cancelled) setError("載入稽核紀錄失敗"); });
    return () => { cancelled = true; };
  }, [tab, page, buildQuery, token]);

  const totalPages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  const exportCsv = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/audit-logs/export?${buildQuery(true)}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error(String(res.status));
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "audit-logs.csv";
      a.click();
      URL.revokeObjectURL(a.href);
    } catch {
      setError("匯出失敗，請稍後再試");
    }
  };

  const tabButton = (key: Tab, label: string) => (
    <button
      onClick={() => { setTab(key); setPage(1); }}
      className={`px-4 py-2 text-sm rounded-lg ${tab === key ? "bg-blue-600 text-white" : "bg-white text-gray-600 border border-gray-200"}`}
    >
      {label}
    </button>
  );

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-4">
      <h1 className="text-2xl font-bold text-gray-800">稽核紀錄</h1>
      <div className="flex gap-2">
        {tabButton("admin", "管理動作")}
        {tabButton("login", "登入歷程")}
        {tabButton("learning", "學習行為")}
      </div>

      {tab === "learning" ? (
        <p className="text-sm text-gray-500">
          學習行為統計請見 <a href="/dashboard" className="text-blue-600 underline">學習儀表板</a>（沿用既有分析資料）。
        </p>
      ) : (
        <>
          <div className="flex items-center gap-3">
            {tab === "admin" && (
              <select
                value={actionFilter}
                onChange={(e) => { setActionFilter(e.target.value); setPage(1); }}
                className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm"
                aria-label="動作類型篩選"
              >
                <option value="">全部動作</option>
                <option value="user">帳號管理</option>
                <option value="teacher_student">師生指派</option>
                <option value="settings">系統設定</option>
                <option value="quiz">題庫</option>
                <option value="semester">學期封存</option>
                <option value="nlp">NLP 訓練</option>
              </select>
            )}
            <span className="text-sm text-gray-500">{data ? `共 ${data.total} 筆` : ""}</span>
            <button
              onClick={exportCsv}
              className="ml-auto px-3 py-1.5 text-sm border border-gray-300 rounded-lg text-gray-600 hover:bg-gray-50"
            >
              匯出 CSV
            </button>
          </div>

          {error && <p className="text-sm text-red-600">{error}</p>}

          <div className="overflow-x-auto border border-gray-200 rounded-lg">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50 text-left text-gray-600">
                <tr>
                  <th className="px-3 py-2">時間</th>
                  <th className="px-3 py-2">操作者</th>
                  <th className="px-3 py-2">動作</th>
                  <th className="px-3 py-2">對象</th>
                  <th className="px-3 py-2">詳情</th>
                  <th className="px-3 py-2">IP</th>
                </tr>
              </thead>
              <tbody>
                {(data?.items ?? []).map((item) => (
                  <tr key={item.id} className="border-t border-gray-100">
                    <td className="px-3 py-2 whitespace-nowrap text-gray-500">{item.timestamp}</td>
                    <td className="px-3 py-2">{item.actor_username || "—"}</td>
                    <td className="px-3 py-2 font-mono text-xs">{item.action}</td>
                    <td className="px-3 py-2 text-gray-500">
                      {item.target_type ? `${item.target_type}:${item.target_id}` : "—"}
                    </td>
                    <td className="px-3 py-2 text-gray-500 max-w-xs truncate" title={item.detail}>
                      {item.detail === "{}" ? "—" : item.detail}
                    </td>
                    <td className="px-3 py-2 text-gray-400">{item.ip || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex items-center gap-2 text-sm">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="px-3 py-1 border border-gray-300 rounded disabled:opacity-40"
            >
              上一頁
            </button>
            <span className="text-gray-500">{page} / {totalPages}</span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              className="px-3 py-1 border border-gray-300 rounded disabled:opacity-40"
            >
              下一頁
            </button>
          </div>
        </>
      )}
    </div>
  );
}
