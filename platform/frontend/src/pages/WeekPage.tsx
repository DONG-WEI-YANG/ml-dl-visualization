import { useParams } from "react-router-dom";
import { lazy, Suspense, useState, useCallback } from "react";
import { WEEKS } from "../types";
import { API_BASE } from "../lib/api";
import ChatPanel from "../components/llm/ChatPanel";
import QuizPanel from "../components/quiz/QuizPanel";
import ConceptCards from "../components/concepts/ConceptCards";

const weekComponents: Record<number, React.LazyExoticComponent<React.ComponentType>> = {
  1: lazy(() => import("../components/viz/EnvironmentSetupViz")),
  2: lazy(() => import("../components/viz/EDAViz")),
  3: lazy(() => import("../components/viz/DataSplitViz")),
  4: lazy(() => import("../components/viz/GradientDescentViz")),
  5: lazy(() => import("../components/viz/DecisionBoundaryViz")),
  6: lazy(() => import("../components/viz/DecisionBoundaryViz")),
  7: lazy(() => import("../components/viz/TreeGrowthViz")),
  8: lazy(() => import("../components/viz/FeatureImportanceViz")),
  9: lazy(() => import("../components/viz/PipelineFlowViz")),
  10: lazy(() => import("../components/viz/LearningCurveViz")),
  11: lazy(() => import("../components/viz/Week11Compound")),
  12: lazy(() => import("../components/viz/CNNLayerViz")),
  13: lazy(() => import("../components/viz/AttentionViz")),
  14: lazy(() => import("../components/viz/TrainingComparisonViz")),
  15: lazy(() => import("../components/viz/FairnessViz")),
  16: lazy(() => import("../components/viz/MLOpsFlowViz")),
  17: lazy(() => import("../components/viz/EmbeddingSpaceViz")),
  18: lazy(() => import("../components/viz/ProjectShowcaseViz")),
};

export default function WeekPage() {
  const { weekId } = useParams();
  const week = Number(weekId);
  const weekInfo = WEEKS.find((w) => w.id === week);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatPinned, setChatPinned] = useState(false);

  const openChat = useCallback(() => setChatOpen(true), []);
  const closeChat = useCallback(() => { setChatOpen(false); setChatPinned(false); }, []);
  const togglePin = useCallback(() => setChatPinned((p) => !p), []);

  if (!weekInfo) {
    return (
      <div className="p-8 text-center text-gray-500">
        找不到第 {weekId} 週的內容
      </div>
    );
  }

  const VizComponent = weekComponents[week];
  const showInGrid = chatOpen && chatPinned;

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div>
        <div className="flex items-center gap-3 mb-2">
          <span
            className={`inline-flex items-center justify-center w-10 h-10 rounded-xl text-lg font-bold ${
              weekInfo.level === "advanced"
                ? "bg-purple-100 text-purple-700"
                : "bg-blue-100 text-blue-700"
            }`}
          >
            {week}
          </span>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              第 {week} 週：{weekInfo.title}
            </h1>
            <p className="text-gray-500 text-sm">{weekInfo.topic}</p>
          </div>
        </div>
      </div>

      <div className={`grid gap-6 ${showInGrid ? "grid-cols-1 lg:grid-cols-[1fr_380px]" : "grid-cols-1"}`}>
        <div className="space-y-4 min-w-0">
          <div className="border border-gray-200 rounded-xl p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              視覺化互動區
            </h2>
            {VizComponent ? (
              <Suspense fallback={<div className="text-gray-400 text-sm p-4">載入視覺化元件中...</div>}>
                <VizComponent />
              </Suspense>
            ) : (
              <div className="bg-gray-50 rounded-lg p-8 text-center text-gray-400">
                <p className="text-4xl mb-3">📊</p>
                <p>第 {week} 週視覺化模組</p>
                <p className="text-xs mt-1">本週使用 Jupyter Notebook 進行互動實作</p>
              </div>
            )}
          </div>

          <ConceptCards week={week} />

          <QuizPanel week={week} />

          <div className="border border-gray-200 rounded-xl p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-3">
              本週教材
            </h2>
            <div className="grid grid-cols-2 gap-2">
              {[
                { name: "講義", icon: "📄", type: "lecture", format: "PDF/HTML" },
                { name: "投影片", icon: "📊", type: "slides", format: "PDF/HTML" },
                { name: "Notebook", icon: "📓", type: "notebook", format: "ipynb" },
                { name: "作業", icon: "✏️", type: "assignment", format: "PDF/HTML" },
              ].map((item) => (
                <a
                  key={item.name}
                  href={`${API_BASE}/api/curriculum/week/${week}/${item.type}`}
                  download
                  className="flex items-center justify-between px-3 py-2 border border-gray-200 rounded-lg text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                >
                  <span className="flex items-center gap-2">
                    <span>{item.icon}</span>
                    {item.name}
                  </span>
                  <span className="text-[10px] text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded">{item.format}</span>
                </a>
              ))}
            </div>
          </div>
        </div>

        {/* Pinned: in-grid chat panel */}
        {showInGrid && (
          <div className="lg:sticky lg:top-6 lg:self-start">
            <ChatPanel
              week={week}
              topic={weekInfo.topic}
              pinned={chatPinned}
              onClose={closeChat}
              onTogglePin={togglePin}
              onAutoOpen={openChat}
            />
          </div>
        )}
      </div>

      {/* Floating toggle button (when chat is closed) */}
      {!chatOpen && (
        <button
          onClick={openChat}
          className="fixed right-5 bottom-6 z-40 flex items-center gap-2 px-4 py-2.5 bg-blue-500 text-white rounded-full shadow-lg hover:bg-blue-600 transition-all hover:shadow-xl cursor-pointer"
        >
          <span className="text-sm font-bold">AI</span>
          <span className="text-sm">助教</span>
        </button>
      )}

      {/* Slide-over panel (when open but not pinned) */}
      {chatOpen && !chatPinned && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-black/20 z-40 lg:bg-transparent lg:pointer-events-none"
            onClick={closeChat}
          />
          {/* Panel */}
          <div className="fixed right-0 top-0 bottom-0 z-50 w-[380px] max-w-[90vw] shadow-2xl">
            <ChatPanel
              week={week}
              topic={weekInfo.topic}
              pinned={chatPinned}
              onClose={closeChat}
              onTogglePin={togglePin}
              onAutoOpen={openChat}
            />
          </div>
        </>
      )}
    </div>
  );
}
