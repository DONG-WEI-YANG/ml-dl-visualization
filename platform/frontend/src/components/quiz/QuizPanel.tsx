import { useState, useEffect } from "react";
import { fetchAPI } from "../../lib/api";

interface Question {
  id: string;
  question: string;
  options: string[];
  category: string;
}

interface GradeResult {
  score: number;
  total: number;
  percentage: number;
  results: { id: string; correct: boolean; correct_answer: number; user_answer: number | null; explanation: string }[];
}

export default function QuizPanel({ week }: { week: number }) {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [result, setResult] = useState<GradeResult | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchAPI<{ questions: Question[] }>(`/api/quiz/week/${week}`)
      .then((data) => {
        setQuestions(data.questions);
        setAnswers({});
        setResult(null);
      })
      .catch(() => setQuestions([]));
  }, [week]);

  const submit = async () => {
    setLoading(true);
    try {
      const data = await fetchAPI<GradeResult>("/api/quiz/submit", { week, answers });
      setResult(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const reset = () => {
    setAnswers({});
    setResult(null);
  };

  const getCategorySummary = () => {
    if (!result) return null;
    const categoryStats: Record<string, { correct: number; total: number }> = {};
    for (const r of result.results) {
      const q = questions.find((qq) => qq.id === r.id);
      const cat = q?.category || "other";
      if (!categoryStats[cat]) categoryStats[cat] = { correct: 0, total: 0 };
      categoryStats[cat].total++;
      if (r.correct) categoryStats[cat].correct++;
    }
    return categoryStats;
  };

  const categoryLabel = (cat: string) =>
    cat === "concept" ? "概念" : cat === "application" ? "應用" : "程式";

  if (questions.length === 0) return null;

  const categorySummary = getCategorySummary();

  return (
    <div className="border border-gray-200 rounded-xl p-6" role="region" aria-label="Weekly quiz">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900">即時測驗</h2>
        {result && (
          <span className={`text-sm font-bold px-3 py-1 rounded-full ${
            result.percentage >= 80 ? "bg-green-100 text-green-700" :
            result.percentage >= 60 ? "bg-yellow-100 text-yellow-700" :
            "bg-red-100 text-red-700"
          }`}>
            {result.score}/{result.total} ({result.percentage}%)
          </span>
        )}
      </div>

      <div className="space-y-4">
        {questions.map((q, qi) => {
          const qResult = result?.results.find((r) => r.id === q.id);
          return (
            <div key={q.id} className={`p-3 rounded-lg ${
              qResult ? (qResult.correct ? "bg-green-50" : "bg-red-50") : "bg-gray-50"
            }`}>
              <p className="text-sm font-medium mb-2 flex items-center gap-2">
                <span>{qi + 1}. {q.question}</span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                  q.category === "concept" ? "bg-blue-100 text-blue-600" :
                  q.category === "application" ? "bg-green-100 text-green-600" :
                  "bg-purple-100 text-purple-600"
                }`}>
                  {q.category === "concept" ? "概念" : q.category === "application" ? "應用" : "程式"}
                </span>
              </p>
              <div className="space-y-1" role="radiogroup" aria-label={`Question ${qi + 1} options`}>
                {q.options.map((opt, oi) => {
                  const selected = answers[q.id] === oi;
                  const isCorrect = qResult?.correct_answer === oi;
                  const isWrong = qResult && !qResult.correct && qResult.user_answer === oi;

                  return (
                    <button
                      key={oi}
                      role="radio"
                      aria-checked={selected}
                      disabled={!!result}
                      onClick={() => setAnswers({ ...answers, [q.id]: oi })}
                      className={`w-full text-left px-3 py-2 rounded text-sm border transition ${
                        result
                          ? isCorrect
                            ? "border-green-400 bg-green-100 text-green-800"
                            : isWrong
                            ? "border-red-400 bg-red-100 text-red-800"
                            : "border-gray-200 text-gray-500"
                          : selected
                          ? "border-blue-400 bg-blue-50 text-blue-700"
                          : "border-gray-200 hover:border-gray-300"
                      }`}
                    >
                      <span className="font-mono mr-2">{String.fromCharCode(65 + oi)}.</span>
                      {opt}
                    </button>
                  );
                })}
              </div>
              {qResult && !qResult.correct && qResult.explanation && (
                <p className="mt-2 text-xs text-blue-700 bg-blue-50 border border-blue-200 rounded px-3 py-2">
                  💡 {qResult.explanation}
                </p>
              )}
            </div>
          );
        })}
      </div>

      {result && categorySummary && (
        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
          <p className="text-xs font-semibold text-gray-700 mb-2">各類別表現</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(categorySummary).map(([cat, stats]) => {
              const pct = Math.round((stats.correct / stats.total) * 100);
              const isStrong = pct >= 80;
              const isWeak = pct < 60;
              return (
                <span
                  key={cat}
                  className={`text-xs px-2 py-1 rounded-full font-medium ${
                    isStrong
                      ? "bg-green-100 text-green-700"
                      : isWeak
                      ? "bg-red-100 text-red-700"
                      : "bg-yellow-100 text-yellow-700"
                  }`}
                >
                  {categoryLabel(cat)} {stats.correct}/{stats.total}
                  {isStrong ? " ✓" : isWeak ? " ✗" : ""}
                </span>
              );
            })}
          </div>
          <p className="text-[10px] text-gray-400 mt-1.5">
            ✓ 表現優良（≥80%） ✗ 需加強（&lt;60%）
          </p>
        </div>
      )}

      <div className="flex gap-2 mt-4">
        {!result ? (
          <button
            onClick={submit}
            disabled={loading || Object.keys(answers).length < questions.length}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600 disabled:opacity-50"
          >
            {loading ? "批改中..." : "提交答案"}
          </button>
        ) : (
          <button
            onClick={reset}
            className="px-4 py-2 bg-gray-500 text-white rounded-lg text-sm font-medium hover:bg-gray-600"
          >
            重新作答
          </button>
        )}
        <span className="text-xs text-gray-400 self-center">
          {!result && `已作答 ${Object.keys(answers).length}/${questions.length}`}
        </span>
      </div>
    </div>
  );
}
