import { useState, useEffect } from "react";
import { useAuth } from "../hooks/useAuth";
import { API_BASE } from "../lib/api";

interface Question {
  id: string;
  week: number;
  question: string;
  options: string[];
  answer: number;
  explanation: string;
  category: string;
}

const WEEKS = Array.from({ length: 18 }, (_, i) => i + 1);

const CATEGORY_LABELS: Record<string, string> = {
  concept: "觀念",
  application: "應用",
  code: "程式",
};

const CATEGORY_COLORS: Record<string, string> = {
  concept: "bg-blue-100 text-blue-700",
  application: "bg-green-100 text-green-700",
  code: "bg-purple-100 text-purple-700",
};

const EMPTY_FORM = {
  week: 1,
  question: "",
  options: ["", "", "", ""],
  answer: 0,
  explanation: "",
  category: "concept",
};

export default function QuizManagement() {
  const { user: me, token, logout } = useAuth();
  const [questions, setQuestions] = useState<Question[]>([]);
  const [filterWeek, setFilterWeek] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState<{ text: string; type: "ok" | "err" }>({ text: "", type: "ok" });

  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState(EMPTY_FORM);

  const authFetch = async (path: string, method = "GET", body?: unknown) => {
    const res = await fetch(`${API_BASE}${path}`, {
      method,
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (res.status === 401 || res.status === 403) { logout(); throw new Error("未授權"); }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Error ${res.status}`);
    }
    return res.json();
  };

  const flash = (text: string, type: "ok" | "err" = "ok") => {
    setMessage({ text, type });
    setTimeout(() => setMessage({ text: "", type: "ok" }), 3000);
  };

  const fetchQuestions = async () => {
    try {
      const params = new URLSearchParams();
      if (filterWeek) params.set("week", filterWeek);
      const qs = params.toString();
      const url = `/api/admin/quiz/questions${qs ? `?${qs}` : ""}`;
      const data = await authFetch(url);
      setQuestions(data);
    } catch { flash("無法載入題目列表", "err"); }
    setLoading(false);
  };

  useEffect(() => { if (token) fetchQuestions(); }, [token, filterWeek]);

  if (me?.role !== "admin") {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8">
        <h1 className="text-xl font-bold text-gray-900 mb-2">權限不足</h1>
        <p className="text-gray-500">僅管理員可存取此頁面</p>
      </div>
    );
  }

  const generateId = (week: number) => {
    const weekQuestions = questions.filter((q) => q.week === week);
    const maxNum = weekQuestions.reduce((max, q) => {
      const match = q.id.match(/q(\d+)$/);
      return match ? Math.max(max, parseInt(match[1])) : max;
    }, 0);
    const wStr = String(week).padStart(2, "0");
    const qStr = String(maxNum + 1).padStart(2, "0");
    return `w${wStr}q${qStr}`;
  };

  const openCreate = () => {
    setEditingId(null);
    setForm({ ...EMPTY_FORM, options: ["", "", "", ""] });
    setShowForm(true);
  };

  const openEdit = (q: Question) => {
    setEditingId(q.id);
    setForm({
      week: q.week,
      question: q.question,
      options: [...q.options],
      answer: q.answer,
      explanation: q.explanation || "",
      category: q.category || "concept",
    });
    setShowForm(true);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (form.options.some((o) => !o.trim())) { flash("請填寫所有選項", "err"); return; }
    try {
      if (editingId) {
        await authFetch(`/api/admin/quiz/questions/${editingId}`, "PUT", {
          week: form.week,
          question: form.question,
          options: form.options,
          answer: form.answer,
          explanation: form.explanation || undefined,
          category: form.category,
        });
        flash("題目已更新");
      } else {
        const id = generateId(form.week);
        await authFetch("/api/admin/quiz/questions", "POST", {
          id,
          week: form.week,
          question: form.question,
          options: form.options,
          answer: form.answer,
          explanation: form.explanation || undefined,
          category: form.category,
        });
        flash("題目已新增");
      }
      setShowForm(false);
      fetchQuestions();
    } catch (err: unknown) {
      flash((err as Error).message || "儲存失敗", "err");
    }
  };

  const handleDelete = async (q: Question) => {
    if (!confirm(`確定要刪除題目 ${q.id}？此操作無法復原。`)) return;
    try {
      await authFetch(`/api/admin/quiz/questions/${q.id}`, "DELETE");
      flash(`已刪除 ${q.id}`);
      fetchQuestions();
    } catch (err: unknown) {
      flash((err as Error).message || "刪除失敗", "err");
    }
  };

  const setOption = (index: number, value: string) => {
    const newOptions = [...form.options];
    newOptions[index] = value;
    setForm({ ...form, options: newOptions });
  };

  const weekCounts = WEEKS.map((w) => ({
    week: w,
    count: questions.filter((q) => q.week === w).length,
  }));
  const totalCount = questions.length;

  return (
    <div className="max-w-5xl mx-auto p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">題庫管理</h1>
        <button
          onClick={openCreate}
          className="px-4 py-2 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600 transition-colors"
        >
          + 新增題目
        </button>
      </div>

      {/* Message */}
      {message.text && (
        <div className={`px-4 py-2 rounded-lg text-sm ${message.type === "err" ? "bg-red-50 text-red-700" : "bg-green-50 text-green-700"}`}>
          {message.text}
        </div>
      )}

      {/* Stats */}
      <div className="flex items-center gap-3">
        <div className="bg-gray-50 text-gray-700 rounded-xl px-4 py-2 text-center">
          <div className="text-2xl font-bold">{totalCount}</div>
          <div className="text-xs mt-0.5">全部題目</div>
        </div>
        <div className="flex-1 flex gap-1 flex-wrap">
          {weekCounts.filter((wc) => wc.count > 0).map((wc) => (
            <button
              key={wc.week}
              onClick={() => setFilterWeek(filterWeek === String(wc.week) ? "" : String(wc.week))}
              className={`px-2 py-1 text-xs rounded-lg transition-colors ${
                filterWeek === String(wc.week)
                  ? "bg-gray-900 text-white"
                  : "bg-white text-gray-600 border border-gray-200 hover:bg-gray-50"
              }`}
            >
              W{wc.week}: {wc.count}
            </button>
          ))}
        </div>
      </div>

      {/* Week Filter */}
      <div className="flex items-center gap-2">
        <select
          value={filterWeek}
          onChange={(e) => setFilterWeek(e.target.value)}
          className="px-3 py-1.5 text-xs border border-gray-200 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-blue-300"
        >
          <option value="">全部週次</option>
          {WEEKS.map((w) => (
            <option key={w} value={String(w)}>第 {w} 週</option>
          ))}
        </select>
        {filterWeek && (
          <button
            onClick={() => setFilterWeek("")}
            className="text-xs text-gray-400 hover:text-gray-600"
          >
            清除篩選
          </button>
        )}
      </div>

      {/* Questions Table */}
      {loading ? (
        <div className="text-gray-400 text-center py-8">載入中...</div>
      ) : (
        <div className="border border-gray-200 rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-gray-600">
              <tr>
                <th className="text-left px-4 py-3 font-medium">ID</th>
                <th className="text-left px-4 py-3 font-medium">週次</th>
                <th className="text-left px-4 py-3 font-medium">題目</th>
                <th className="text-left px-4 py-3 font-medium">類型</th>
                <th className="text-left px-4 py-3 font-medium">答案</th>
                <th className="text-right px-4 py-3 font-medium">操作</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {questions.map((q) => (
                <tr key={q.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 font-mono text-gray-900">{q.id}</td>
                  <td className="px-4 py-3 text-gray-700">{q.week}</td>
                  <td className="px-4 py-3 text-gray-700 max-w-xs truncate">{q.question}</td>
                  <td className="px-4 py-3">
                    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${CATEGORY_COLORS[q.category] || "bg-gray-100 text-gray-700"}`}>
                      {CATEGORY_LABELS[q.category] || q.category}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-500 text-xs">
                    {q.options[q.answer] ? `${String.fromCharCode(65 + q.answer)}. ${q.options[q.answer]}` : "-"}
                  </td>
                  <td className="px-4 py-3 text-right space-x-1">
                    <button
                      onClick={() => openEdit(q)}
                      className="px-2 py-1 text-xs text-blue-600 hover:bg-blue-50 rounded"
                    >
                      編輯
                    </button>
                    <button
                      onClick={() => handleDelete(q)}
                      className="px-2 py-1 text-xs text-red-600 hover:bg-red-50 rounded"
                    >
                      刪除
                    </button>
                  </td>
                </tr>
              ))}
              {questions.length === 0 && (
                <tr><td colSpan={6} className="text-center py-8 text-gray-400">沒有題目</td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Create/Edit Modal */}
      {showForm && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <form onSubmit={handleSubmit} className="bg-white rounded-2xl shadow-xl p-6 w-full max-w-lg space-y-4 max-h-[90vh] overflow-y-auto">
            <h2 className="text-lg font-bold text-gray-900">
              {editingId ? `編輯題目 ${editingId}` : "新增題目"}
            </h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">週次 *</label>
                <select
                  value={form.week}
                  onChange={(e) => setForm({ ...form, week: parseInt(e.target.value) })}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {WEEKS.map((w) => (
                    <option key={w} value={w}>第 {w} 週</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">類型 *</label>
                <select
                  value={form.category}
                  onChange={(e) => setForm({ ...form, category: e.target.value })}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="concept">觀念</option>
                  <option value="application">應用</option>
                  <option value="code">程式</option>
                </select>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">題目 *</label>
              <textarea
                required
                rows={3}
                value={form.question}
                onChange={(e) => setForm({ ...form, question: e.target.value })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-vertical"
              />
            </div>
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">選項 *</label>
              {form.options.map((opt, i) => (
                <div key={i} className="flex items-center gap-2">
                  <input
                    type="radio"
                    name="answer"
                    checked={form.answer === i}
                    onChange={() => setForm({ ...form, answer: i })}
                    className="text-blue-500 focus:ring-blue-500"
                  />
                  <span className="text-sm font-medium text-gray-500 w-5">{String.fromCharCode(65 + i)}.</span>
                  <input
                    required
                    value={opt}
                    onChange={(e) => setOption(i, e.target.value)}
                    placeholder={`選項 ${String.fromCharCode(65 + i)}`}
                    className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              ))}
              <p className="text-xs text-gray-400">選取圓鈕以標記正確答案</p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">解析</label>
              <textarea
                rows={2}
                value={form.explanation}
                onChange={(e) => setForm({ ...form, explanation: e.target.value })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-vertical"
                placeholder="選填，答案說明..."
              />
            </div>
            <div className="flex gap-2 justify-end pt-2">
              <button type="button" onClick={() => setShowForm(false)} className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg">取消</button>
              <button type="submit" className="px-4 py-2 text-sm bg-blue-500 text-white rounded-lg hover:bg-blue-600">
                {editingId ? "儲存" : "建立"}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
