import { useParams } from "react-router-dom";
import { lazy, Suspense } from "react";
import { WEEKS } from "../types";
import ChatPanel from "../components/llm/ChatPanel";
import QuizPanel from "../components/quiz/QuizPanel";

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
  11: lazy(() => import("../components/viz/ActivationFunctionViz")),
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

  if (!weekInfo) {
    return (
      <div className="p-8 text-center text-gray-500">
        找不到第 {weekId} 週的內容
      </div>
    );
  }

  const VizComponent = weekComponents[week];

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

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
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

          <QuizPanel week={week} />

          <div className="border border-gray-200 rounded-xl p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-3">
              本週教材
            </h2>
            <div className="grid grid-cols-2 gap-2">
              {[
                { name: "講義", icon: "📄", type: "lecture" },
                { name: "投影片", icon: "📊", type: "slides" },
                { name: "Notebook", icon: "📓", type: "notebook" },
                { name: "作業", icon: "✏️", type: "assignment" },
              ].map((item) => (
                <a
                  key={item.name}
                  href={`/api/curriculum/week/${week}/${item.type}`}
                  download
                  className="flex items-center gap-2 px-3 py-2 border border-gray-200 rounded-lg text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                >
                  <span>{item.icon}</span>
                  {item.name}
                </a>
              ))}
            </div>
          </div>
        </div>

        <ChatPanel week={week} topic={weekInfo.topic} />
      </div>
    </div>
  );
}
