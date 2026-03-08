import { useState, useEffect } from "react";
import { fetchAPI } from "../lib/api";
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
  const [summary, setSummary] = useState<ClassSummary | null>(null);
  const [studentId, setStudentId] = useState("");
  const [studentData, setStudentData] = useState<StudentAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [studentLoading, setStudentLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchAPI<ClassSummary>("/api/analytics/summary")
      .then(setSummary)
      .catch(() => setError("無法載入班級總覽資料"))
      .finally(() => setLoading(false));
  }, []);

  const lookupStudent = () => {
    if (!studentId.trim()) return;
    setStudentLoading(true);
    setError("");
    fetchAPI<StudentAnalytics>(`/api/analytics/students/${studentId.trim()}`)
      .then(setStudentData)
      .catch(() => setError("找不到該學生資料"))
      .finally(() => setStudentLoading(false));
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
            onClick={lookupStudent}
            disabled={studentLoading}
            className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {studentLoading ? "查詢中..." : "查詢"}
          </button>
        </div>

        {error && <p className="text-red-500 text-sm mb-4">{error}</p>}

        {studentData && (
          <div className="space-y-6">
            {/* Student summary cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { label: "完成週次", value: `${studentData.total_weeks_completed} / 18` },
                { label: "總學習時間", value: `${studentData.total_time_minutes} 分鐘` },
                { label: "平均分數", value: studentData.average_score.toFixed(1) },
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
