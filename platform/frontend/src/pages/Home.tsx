import { Link } from "react-router-dom";
import { WEEKS } from "../types";

export default function Home() {
  return (
    <div className="max-w-5xl mx-auto p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          ML/DL 視覺化互動教學平台
        </h1>
        <p className="text-gray-600 text-lg">
          18 週機器學習 (Machine Learning) 與深度學習 (Deep Learning) 視覺化互動課程
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" role="list" aria-label="18-week curriculum">
        {WEEKS.map((w) => (
          <Link
            key={w.id}
            to={`/week/${w.id}`}
            className="block p-4 border border-gray-200 rounded-xl hover:border-blue-300 hover:shadow-md transition-all"
          >
            <div className="flex items-center gap-3 mb-2">
              <span className={`inline-flex items-center justify-center w-8 h-8 rounded-lg text-sm font-bold ${
                w.level === "advanced"
                  ? "bg-purple-100 text-purple-700"
                  : "bg-blue-100 text-blue-700"
              }`}>
                {w.id}
              </span>
              <span className={`text-xs px-2 py-0.5 rounded-full ${
                w.level === "advanced"
                  ? "bg-purple-50 text-purple-600"
                  : "bg-blue-50 text-blue-600"
              }`}>
                {w.level === "advanced" ? "進階" : "核心"}
              </span>
            </div>
            <h3 className="font-medium text-gray-900">{w.title}</h3>
            <p className="text-sm text-gray-500 mt-1 line-clamp-2">{w.topic}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
