import { useState, useEffect, useRef } from "react";
import { fetchAPI } from "../lib/api";
import { useAuth } from "../hooks/useAuth";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
} from "recharts";

interface ClassSummary {
  total_students: number;
  total_events: number;
  average_score: number;
  popular_llm_topics: { topic: string; count: number }[];
}

interface RosterStudent {
  id: number; username: string; display_name: string; semester: string; class_name: string;
  is_active: boolean; total_events: number; total_weeks_completed: number; quiz_weeks: number;
  average_score: number | null; last_activity: string | null; total_time_minutes: number;
}

interface WeekProgress {
  week: number;
  completed: boolean;
  quiz_score: number | null;
  assignment_score: number | null;
  llm_interactions: number;
  time_spent_minutes: number;
}

interface StudentAnalytics {
  student_id: string;
  total_weeks_completed: number;
  total_time_minutes: number;
  average_score: number;
  weekly_progress: WeekProgress[];
  llm_topics: { topic: string; count: number }[];
  error_patterns?: { type: string; count: number }[];
}

export default function Dashboard() {
  const { token } = useAuth();
  const [roster, setRoster] = useState<RosterStudent[]>([]);
  const [filters, setFilters] = useState({ academic_year: "", semester: "", class_name: "" });
  const [query, setQuery] = useState("");
  const [selectedLabel, setSelectedLabel] = useState("");
  const studentRequest = useRef(0);
  const [summary, setSummary] = useState<ClassSummary | null>(null);
  const [studentId, setStudentId] = useState("");
  const [studentData, setStudentData] = useState<StudentAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [studentLoading, setStudentLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    setStudentData(null);
    studentRequest.current += 1;
    setStudentLoading(false);
    Promise.all([
      fetchAPI<ClassSummary>(`/api/analytics/summary${query}`, undefined, token ?? undefined),
      fetchAPI<RosterStudent[]>(`/api/analytics/roster${query}`, undefined, token ?? undefined),
    ]).then(([totals, students]) => {
      if (!cancelled) { setSummary(totals); setRoster(students); }
    }).catch(() => {
      if (!cancelled) { setSummary(null); setRoster([]); setError("無法載入班級總覽資料"); }
    }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; studentRequest.current += 1; };
  }, [query, token]);

  const lookupStudent = (id = studentId.trim(), semester?: string, label = id) => {
    if (!id) return;
    setStudentLoading(true);
    setError("");
    setStudentData(null);
    const request = ++studentRequest.current;
    setSelectedLabel(label);
    const suffix = semester !== undefined ? `?semester=${encodeURIComponent(semester)}` : "";
    fetchAPI<StudentAnalytics>(`/api/analytics/students/${encodeURIComponent(id)}${suffix}`, undefined, token ?? undefined)
      .then((data) => { if (request === studentRequest.current) setStudentData(data); })
      .catch(() => { if (request === studentRequest.current) setError("找不到該學生資料"); })
      .finally(() => { if (request === studentRequest.current) setStudentLoading(false); });
  };

  const stats = summary
    ? [
        { label: "學生人數", value: summary.total_students, color: "text-blue-600" },
        { label: "學習事件總數", value: summary.total_events, color: "text-green-600" },
        { label: "平均分數", value: summary.average_score, color: "text-purple-600" },
      ]
    : [];

  return (
    <div className="max-w-6xl mx-auto p-8 space-y-8">
      <h1 className="text-2xl font-bold text-gray-900">學習分析儀表板</h1>
      <form className="flex flex-wrap gap-3 items-end" onSubmit={(e) => {
        e.preventDefault();
        const params = new URLSearchParams();
        Object.entries(filters).forEach(([key, value]) => { if (value.trim()) params.set(key, value.trim()); });
        setQuery(params.size ? `?${params}` : "");
      }}>
        <label>學年<input aria-label="學年" placeholder="115" value={filters.academic_year} onChange={(e) => setFilters({ ...filters, academic_year: e.target.value })} className="block border rounded px-3 py-2 w-24" /></label>
        <label>學期<input aria-label="學期" placeholder="115-1" value={filters.semester} onChange={(e) => setFilters({ ...filters, semester: e.target.value })} className="block border rounded px-3 py-2 w-32" /></label>
        <label>班級<input aria-label="班級" placeholder="全部班級" value={filters.class_name} onChange={(e) => setFilters({ ...filters, class_name: e.target.value })} className="block border rounded px-3 py-2" /></label>
        <button className="bg-blue-600 text-white rounded px-4 py-2">套用篩選</button>
      </form>
      {!loading && <section className="border rounded-xl p-4 overflow-x-auto">
        <h2 className="text-lg font-semibold mb-2">學生進度名單</h2>
        <p className="text-sm text-gray-500 mb-3">包含尚未開始的學生；教師只會看到已指派名單。點選學號查看該學期詳情。完成週次以已評分作業計算。</p>
        <table className="w-full min-w-[960px] whitespace-nowrap text-sm text-left"><thead><tr>{["學號", "姓名", "學期", "班級", "帳號", "學習狀態", "測驗週次", "作業完成週次", "平均分數", "最後活動"].map((h) => <th key={h} className="p-2">{h}</th>)}</tr></thead>
          <tbody>{roster.map((r) => <tr key={`${r.id}:${r.semester}`} className="border-t">
            <td className="p-2"><button disabled={studentLoading} className="text-blue-600 underline" onClick={() => lookupStudent(String(r.id), r.semester, `${r.display_name || r.username} · ${r.semester || "未分類"}`)}>{r.username}</button></td>
            <td className="p-2">{r.display_name}</td><td className="p-2">{r.semester || "未分類"}</td><td className="p-2">{r.class_name || "未分班"}</td>
            <td className="p-2">{r.is_active ? "已開通" : "停用"}</td><td className="p-2">{r.total_events ? "學習中" : "尚未開始"}</td>
            <td className="p-2">{r.quiz_weeks ?? 0} / 18</td><td className="p-2">{r.total_weeks_completed} / 18</td><td className="p-2">{r.average_score == null ? "—" : r.average_score.toFixed(1)}</td><td className="p-2">{r.last_activity ? r.last_activity.replace("T", " ").slice(0, 16) : "—"}</td>
          </tr>)}</tbody></table>
        {roster.length === 0 && <p className="p-3 text-gray-500">沒有符合條件的學生；請確認班級、學期與師生指派。</p>}
      </section>}

      {/* Summary stat cards */}
      {loading ? (
        <p className="text-gray-400 text-sm">載入中...</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {stats.map((s) => (
            <div key={s.label} className="border border-gray-200 rounded-xl p-4">
              <p className="text-sm text-gray-500">{s.label}</p>
              <p className={`text-2xl font-bold mt-1 ${s.color}`}>{s.value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Popular LLM Topics */}
      {summary && summary.popular_llm_topics.length > 0 && (
        <div className="border border-gray-200 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            AI 助教熱門提問主題
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={summary.popular_llm_topics}
              layout="vertical"
              margin={{ left: 80 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="topic" width={70} />
              <Tooltip />
              <Bar dataKey="count" fill="#8b5cf6" name="提問次數" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Student Lookup */}
      <div className="border border-gray-200 rounded-xl p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">學生個人分析</h2>
        <div className="flex gap-2 mb-6">
          <input
            type="text"
            value={studentId}
            onChange={(e) => setStudentId(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && lookupStudent()}
            placeholder="輸入學生 ID"
            aria-label="Student ID"
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm flex-1 max-w-xs focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={() => lookupStudent()}
            disabled={studentLoading}
            className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {studentLoading ? "查詢中..." : "查詢"}
          </button>
        </div>

        {error && <p className="text-red-500 text-sm mb-4">{error}</p>}

        {studentData && (
          <div className="space-y-6">
            <p className="font-medium">{selectedLabel}</p>
            {/* Student summary cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { label: "完成週次", value: `${studentData.total_weeks_completed} / 18` },
                { label: "總學習時間", value: `${studentData.total_time_minutes} 分鐘` },
                { label: "作業平均分數", value: studentData.average_score.toFixed(1) },
                { label: "LLM 對話主題數", value: studentData.llm_topics.length },
              ].map((s) => (
                <div key={s.label} className="bg-gray-50 rounded-lg p-3">
                  <p className="text-xs text-gray-500">{s.label}</p>
                  <p className="text-lg font-semibold text-gray-800 mt-0.5">{s.value}</p>
                </div>
              ))}
            </div>

            {/* Weekly progress chart */}
            {studentData.weekly_progress.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-3">每週成績與時間</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={studentData.weekly_progress}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="week"
                      tickFormatter={(w: number) => `W${w}`}
                    />
                    <YAxis yAxisId="score" domain={[0, 100]} />
                    <YAxis yAxisId="time" orientation="right" />
                    <Tooltip
                      labelFormatter={(w: number) => `第 ${w} 週`}
                    />
                    <Legend />
                    <Line
                      yAxisId="score"
                      type="monotone"
                      dataKey="quiz_score"
                      stroke="#3b82f6"
                      name="測驗分數"
                      connectNulls
                      dot={{ r: 3 }}
                    />
                    <Line
                      yAxisId="score"
                      type="monotone"
                      dataKey="assignment_score"
                      stroke="#10b981"
                      name="作業分數"
                      connectNulls
                      dot={{ r: 3 }}
                    />
                    <Line
                      yAxisId="time"
                      type="monotone"
                      dataKey="time_spent_minutes"
                      stroke="#f59e0b"
                      name="學習時間(分)"
                      dot={{ r: 3 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* LLM interaction bar chart */}
            {studentData.weekly_progress.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-3">每週 AI 助教互動次數</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={studentData.weekly_progress}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="week"
                      tickFormatter={(w: number) => `W${w}`}
                    />
                    <YAxis />
                    <Tooltip labelFormatter={(w: number) => `第 ${w} 週`} />
                    <Bar dataKey="llm_interactions" fill="#8b5cf6" name="互動次數" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Student LLM topics */}
            {studentData.llm_topics.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-3">常見提問主題</h3>
                <div className="flex flex-wrap gap-2">
                  {studentData.llm_topics.map((t) => (
                    <span
                      key={t.topic}
                      className="inline-flex items-center gap-1 px-3 py-1 bg-purple-50 text-purple-700 rounded-full text-sm"
                    >
                      {t.topic}
                      <span className="text-purple-400 text-xs">({t.count})</span>
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Error patterns */}
            {studentData.error_patterns && studentData.error_patterns.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-3">錯誤型態分類</h3>
                <div className="space-y-2">
                  {studentData.error_patterns.map((ep) => (
                    <div key={ep.type} className="flex items-center gap-3">
                      <span className="text-sm text-gray-600 w-32 truncate">{ep.type}</span>
                      <div className="flex-1 h-4 bg-gray-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-red-400 rounded-full"
                          style={{ width: `${Math.min(ep.count * 10, 100)}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-500 w-8 text-right">{ep.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {!studentData && !error && (
          <p className="text-gray-400 text-sm">輸入學生 ID 後即可查看個人學習分析</p>
        )}
      </div>
    </div>
  );
}
