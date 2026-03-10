import { useState, useEffect } from "react";
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

export default function AdminSettings() {
  const { user, token, logout } = useAuth();
  const [data, setData] = useState<SettingsData | null>(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [ingestMsg, setIngestMsg] = useState("");
  const [trainMsg, setTrainMsg] = useState("");

  const authFetch = async <T,>(path: string, body?: unknown): Promise<T> => {
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
  };

  useEffect(() => {
    if (!token) return;
    authFetch<SettingsData>("/api/admin/settings")
      .then(setData)
      .catch(() => setMessage("無法載入設定"));
  }, [token]);

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
    } catch {
      setMessage("儲存失敗");
    }
    setSaving(false);
  };

  const ingestRAG = async () => {
    setIngestMsg("索引中...");
    try {
      const res = await fetch(`${API_BASE}/api/rag/ingest`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
      const json = await res.json();
      setIngestMsg(`索引完成：${json.chunks_indexed} 個教材片段`);
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

        <div className="flex items-center gap-3 pt-2">
          <button
            onClick={ingestRAG}
            disabled={saving}
            className="px-4 py-2 bg-purple-600 text-white text-sm rounded-lg hover:bg-purple-700 disabled:opacity-50"
          >
            重新索引教材
          </button>
          {ingestMsg && <span className="text-sm text-gray-600">{ingestMsg}</span>}
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
