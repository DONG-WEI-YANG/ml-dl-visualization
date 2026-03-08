import { useState, useEffect } from "react";
import { fetchAPI } from "../../lib/api";

interface Question {
  id: string;
  question: string;
  options: string[];
}

interface GradeResult {
  score: number;
  total: number;
  percentage: number;
  results: { id: string; correct: boolean; correct_answer: number; user_answer: number | null }[];
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

  if (questions.length === 0) return null;

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
              <p className="text-sm font-medium mb-2">
                {qi + 1}. {q.question}
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
            </div>
          );
        })}
      </div>

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
