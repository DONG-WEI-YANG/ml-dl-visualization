import { useState, useEffect, useCallback } from "react";
import { useAuth } from "../hooks/useAuth";
import { API_BASE } from "../lib/api";

interface ProviderInfo {
  id: string;
  name: string;
  models: string[];
}

interface SettingsData {
  settings: Record<string, string>;
  available_providers: ProviderInfo[];
}

interface LLMDiagnostics {
  configured_provider: string;
  configured_model: string;
  effective_provider: string;
  effective_model: string;
  status: "ready" | "configured" | "degraded" | "error";
  reason: string | null;
  runtime_warnings?: string[];
  probe: {
    attempted: boolean;
    ok?: boolean;
    reason?: string;
    latency_ms?: number;
  };
}

export default function AdminSettings() {
  const { user, token, logout } = useAuth();
  const [data, setData] = useState<SettingsData | null>(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [ingestMsg, setIngestMsg] = useState("");
  const [trainMsg, setTrainMsg] = useState("");
  const [diagnostics, setDiagnostics] = useState<LLMDiagnostics | null>(null);
  const [diagnosticsError, setDiagnosticsError] = useState(false);
  const [probing, setProbing] = useState(false);
  const [ragStats, setRagStats] = useState<{
    total_chunks: number;
    curriculum_chunks: number;
    web_chunks: number;
    by_week: { week: number; count: number }[];
  } | null>(null);

  const authFetch = useCallback(async <T,>(path: string, body?: unknown): Promise<T> => {
    const res = await fetch(`${API_BASE}${path}`, {
      method: body ? "PUT" : "GET",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (res.status === 401 || res.status === 403) {
      logout();
      throw new Error("未授權");
    }
    if (!res.ok) throw new Error(`Error ${res.status}`);
    return res.json();
  }, [logout, token]);

  const loadRagStats = useCallback(() => {
    fetch(`${API_BASE}/api/rag/stats`)
      .then((r) => r.json())
      .then(setRagStats)
      .catch(() => {});
  }, []);

  const loadDiagnostics = useCallback(async (probe = false) => {
    setDiagnosticsError(false);
    if (probe) setProbing(true);
    try {
      const suffix = probe ? "?probe=true" : "";
      const result = await authFetch<LLMDiagnostics>(`/api/llm/diagnostics${suffix}`);
      setDiagnostics(result);
    } catch {
      setDiagnosticsError(true);
    } finally {
      if (probe) setProbing(false);
    }
  }, [authFetch]);

  useEffect(() => {
    if (!token) return;
    authFetch<SettingsData>("/api/admin/settings")
      .then(setData)
      .catch(() => setMessage("無法載入設定"));
    loadRagStats();
    void loadDiagnostics();
  }, [authFetch, loadDiagnostics, loadRagStats, token]);

  if (user?.role !== "admin") {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8">
        <h1 className="text-xl font-bold text-gray-900 mb-2">權限不足</h1>
        <p className="text-gray-500">僅管理員可存取此頁面</p>
      </div>
    );
  }

  const saveSettings = async (updates: Record<string, string>) => {
    setSaving(true);
    setMessage("");
    try {
      const res = await authFetch<{ settings: Record<string, string> }>("/api/admin/settings", updates);
      setData((prev) => prev ? { ...prev, settings: res.settings } : prev);
      setMessage("設定已儲存");
      void loadDiagnostics();
    } catch {
      setMessage("儲存失敗");
    }
    setSaving(false);
  };

  const ingestRAG = async () => {
    if (ragStats && ragStats.total_chunks > 0) {
      const ok = window.confirm(
        `目前有 ${ragStats.curriculum_chunks} 個教材片段與 ${ragStats.web_chunks} 個網路知識片段。\n` +
        "重新索引將替換教材片段，網路知識不受影響。\n確定繼續？"
      );
      if (!ok) return;
    }
    setIngestMsg("索引中...");
    try {
      const res = await fetch(`${API_BASE}/api/rag/ingest`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
      const json = await res.json();
      const count = json.chunks_indexed;
      if (count === 0) {
        setIngestMsg("本機無教材檔案。請用 scripts/ingest_to_cloud.py 從本機上傳教材至雲端");
      } else {
        setIngestMsg(`索引完成：${count} 個教材片段（網路知識已保留）`);
      }
      loadRagStats();
    } catch {
      setIngestMsg("索引失敗");
    }
  };

  const trainNLP = async () => {
    setTrainMsg("訓練中...");
    try {
      const res = await fetch(`${API_BASE}/api/admin/train-nlp`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
      const json = await res.json();
      const r = json.results;
      setTrainMsg(
        `訓練完成 — 意圖分類: ${r.intent?.cv_accuracy ? (r.intent.cv_accuracy * 100).toFixed(1) : "?"}% (${r.intent?.samples || 0} 樣本), ` +
        `情緒偵測: ${r.emotion?.cv_accuracy ? (r.emotion.cv_accuracy * 100).toFixed(1) : "?"}% (${r.emotion?.samples || 0} 樣本)` +
        (r.corpus?.chunks ? `, 語料庫: ${r.corpus.chunks} 片段` : "")
      );
    } catch {
      setTrainMsg("訓練失敗");
    }
  };

  if (!data) return <div className="p-8 text-gray-400">載入中...</div>;

  const { settings, available_providers } = data;
  const currentProvider = available_providers.find((p) => p.id === settings.llm_provider);

  return (
    <div className="max-w-3xl mx-auto p-8 space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">系統管理</h1>
        <span className="text-sm text-gray-500">
          {user.display_name} ({user.role})
        </span>
      </div>

      {/* LLM Settings */}
      <div className="border border-gray-200 rounded-xl p-6 space-y-4">
        <h2 className="text-lg font-semibold text-gray-900">AI 助教模型設定</h2>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">LLM Provider</label>
          <select
            value={settings.llm_provider}
            onChange={(e) => {
              const p = available_providers.find((pr) => pr.id === e.target.value);
              const newModel = p?.models[0] || "";
              saveSettings({ llm_provider: e.target.value, llm_model: newModel });
            }}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {available_providers.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">模型</label>
          <select
            value={settings.llm_model}
            onChange={(e) => saveSettings({ llm_model: e.target.value })}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {currentProvider?.models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        {message && (
          <p className={`text-sm ${message.includes("失敗") ? "text-red-500" : "text-green-600"}`}>
            {message}
          </p>
        )}
      </div>

      {/* Truthful configured → effective provider status */}
      <div className="border border-slate-200 rounded-xl p-6 space-y-4" aria-labelledby="ai-runtime-status">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 id="ai-runtime-status" className="text-lg font-semibold text-gray-900">AI 運作狀態</h2>
            <p className="mt-1 text-xs text-gray-500">設定模型與實際執行模型分開顯示，降級時不會假裝使用外部 AI。</p>
          </div>
          {diagnostics && (
            <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
              diagnostics.status === "ready" ? "bg-emerald-100 text-emerald-700" :
              diagnostics.status === "degraded" ? "bg-amber-100 text-amber-800" :
              diagnostics.status === "error" ? "bg-red-100 text-red-700" :
              "bg-blue-100 text-blue-700"
            }`} role="status">
              {diagnostics.status === "ready" && "正常運作"}
              {diagnostics.status === "degraded" && "已降級"}
              {diagnostics.status === "error" && "探測失敗"}
              {diagnostics.status === "configured" && "已設定，尚未探測"}
            </span>
          )}
        </div>

        {diagnostics ? (
          <div className="grid gap-2 sm:grid-cols-[1fr_auto_1fr] sm:items-center" aria-label="AI provider resolution">
            <div className="rounded-lg bg-slate-50 p-3">
              <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">設定</p>
              <p className="mt-1 break-all text-sm font-semibold text-slate-800">
                {diagnostics.configured_provider} / {diagnostics.configured_model}
              </p>
            </div>
            <span className="text-center text-slate-400" aria-hidden="true">→</span>
            <div className="rounded-lg bg-blue-50 p-3">
              <p className="text-[11px] font-medium uppercase tracking-wide text-blue-600">實際執行</p>
              <p className="mt-1 break-all text-sm font-semibold text-blue-900">
                {diagnostics.effective_provider} / {diagnostics.effective_model}
              </p>
            </div>
          </div>
        ) : (
          <p className="text-sm text-gray-500">正在讀取 AI 狀態…</p>
        )}

        {diagnostics?.reason === "missing_api_key" && (
          <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
            外部 AI 金鑰未設定，已改用本地 NLP。
          </p>
        )}
        {diagnostics?.reason === "unknown_provider" && (
          <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
            設定的 AI provider 無法識別，已改用本地 NLP。
          </p>
        )}
        {diagnostics?.runtime_warnings && diagnostics.runtime_warnings.length > 0 && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
            <p className="font-medium">本地 AI 可回覆，但部分能力正在使用備援：</p>
            <ul className="mt-1 list-disc space-y-1 pl-5">
              {diagnostics.runtime_warnings.map((warning) => (
                <li key={warning}>
                  {warning === "sklearn_model_version_mismatch" && "分類模型與目前 scikit-learn 版本不一致。"}
                  {warning === "semantic_model_uncached" && "語意向量模型尚未快取，改用關鍵字相似度。"}
                  {warning === "semantic_model_unavailable" && "語意向量套件不可用，改用關鍵字相似度。"}
                  {warning === "model_artifacts_missing" && "部分本地分類模型檔案缺失。"}
                  {warning === "model_metadata_unavailable" && "無法驗證本地模型訓練版本。"}
                </li>
              ))}
            </ul>
          </div>
        )}
        {diagnosticsError && (
          <p role="alert" className="text-sm text-red-700">無法讀取 AI 運作狀態，請稍後重試。</p>
        )}
        {diagnostics?.probe.attempted && diagnostics.probe.ok && (
          <p className="text-xs text-emerald-700">探測成功 · {diagnostics.probe.latency_ms ?? 0} ms</p>
        )}

        <button
          type="button"
          onClick={() => void loadDiagnostics(true)}
          disabled={probing}
          className="rounded-lg border border-blue-200 bg-white px-3 py-2 text-sm font-medium text-blue-700 hover:bg-blue-50 disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {probing ? "探測中…" : "執行 AI 探測"}
        </button>
      </div>

      {/* RAG Settings */}
      <div className="border border-gray-200 rounded-xl p-6 space-y-4">
        <h2 className="text-lg font-semibold text-gray-900">RAG 教材檢索設定</h2>

        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-gray-700">啟用 RAG</p>
            <p className="text-xs text-gray-500">讓 AI 助教根據課程教材內容回答問題</p>
          </div>
          <button
            onClick={() =>
              saveSettings({
                rag_enabled: settings.rag_enabled === "true" ? "false" : "true",
              })
            }
            className={`relative w-11 h-6 rounded-full transition-colors ${
              settings.rag_enabled === "true" ? "bg-blue-500" : "bg-gray-300"
            }`}
          >
            <span
              className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform ${
                settings.rag_enabled === "true" ? "translate-x-5" : ""
              }`}
            />
          </button>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            檢索數量 (top_k)
          </label>
          <select
            value={settings.rag_top_k}
            onChange={(e) => saveSettings({ rag_top_k: e.target.value })}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {[3, 5, 8, 10].map((k) => (
              <option key={k} value={String(k)}>
                {k} 個片段
              </option>
            ))}
          </select>
        </div>

        {ragStats && (
          <div className="bg-gray-50 rounded-lg p-3 space-y-1">
            <p className="text-sm font-medium text-gray-700">
              目前索引：{ragStats.total_chunks} 個片段
              {ragStats.by_week.length > 0 && (
                <span className="text-gray-500 font-normal">
                  （涵蓋 {ragStats.by_week.filter((w) => w.week > 0).length} 週）
                </span>
              )}
            </p>
            <div className="flex gap-4 text-xs text-gray-500">
              <span>教材：{ragStats.curriculum_chunks}</span>
              <span>網路知識：{ragStats.web_chunks}</span>
            </div>
          </div>
        )}

        <div className="flex items-center gap-3 pt-2">
          <button
            onClick={ingestRAG}
            disabled={saving}
            className="px-4 py-2 bg-purple-600 text-white text-sm rounded-lg hover:bg-purple-700 disabled:opacity-50"
          >
            重新索引教材
          </button>
          {ingestMsg && (
            <span className={`text-sm ${ingestMsg.includes("失敗") || ingestMsg.includes("無教材") ? "text-amber-600" : "text-green-600"}`}>
              {ingestMsg}
            </span>
          )}
        </div>
      </div>

      {/* NLP Model Training */}
      <div className="border border-gray-200 rounded-xl p-6 space-y-4">
        <h2 className="text-lg font-semibold text-gray-900">NLP 模型訓練</h2>
        <p className="text-sm text-gray-500">
          訓練意圖分類 (TF-IDF + LinearSVC) 及情緒偵測 (TF-IDF + LogisticRegression) 模型。
          建議先完成教材索引再訓練，以同時建立語料庫 TF-IDF 向量。
        </p>
        <div className="flex items-center gap-3">
          <button
            onClick={trainNLP}
            disabled={saving}
            className="px-4 py-2 bg-emerald-600 text-white text-sm rounded-lg hover:bg-emerald-700 disabled:opacity-50"
          >
            訓練 NLP 模型
          </button>
          {trainMsg && <span className="text-sm text-gray-600">{trainMsg}</span>}
        </div>
      </div>
    </div>
  );
}
