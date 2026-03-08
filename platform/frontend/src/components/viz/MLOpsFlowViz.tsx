import { useState } from "react";

const STAGES = [
  {
    id: "data", label: "資料管理", icon: "📦",
    items: ["版本控制 (DVC)", "資料品質檢查", "特徵商店 (Feature Store)"],
    color: "#3b82f6",
  },
  {
    id: "train", label: "模型訓練", icon: "🔧",
    items: ["實驗追蹤 (MLflow)", "超參數搜尋", "分散式訓練"],
    color: "#8b5cf6",
  },
  {
    id: "eval", label: "模型評估", icon: "📊",
    items: ["A/B 測試", "效能基準", "公平性檢測"],
    color: "#10b981",
  },
  {
    id: "deploy", label: "模型部署", icon: "🚀",
    items: ["容器化 (Docker)", "API 服務 (FastAPI)", "批次推論"],
    color: "#f59e0b",
  },
  {
    id: "monitor", label: "監測維運", icon: "👁️",
    items: ["資料漂移偵測", "效能監控", "自動重訓練"],
    color: "#ef4444",
  },
];

export default function MLOpsFlowViz() {
  const [active, setActive] = useState<string | null>(null);

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">MLOps 流程 Pipeline</h3>

      <div className="relative">
        <div className="flex items-center justify-between">
          {STAGES.map((stage, i) => (
            <div key={stage.id} className="flex items-center flex-1">
              <button
                onClick={() => setActive(active === stage.id ? null : stage.id)}
                className={`relative z-10 flex flex-col items-center gap-1 p-2 rounded-xl transition-all w-full ${
                  active === stage.id ? "shadow-lg scale-105" : "hover:scale-105"
                }`}
                style={{
                  backgroundColor: active === stage.id ? stage.color + "20" : undefined,
                  borderColor: stage.color,
                  border: active === stage.id ? `2px solid ${stage.color}` : "2px solid transparent",
                }}
              >
                <span className="text-2xl">{stage.icon}</span>
                <span className="text-xs font-medium" style={{ color: stage.color }}>{stage.label}</span>
              </button>
              {i < STAGES.length - 1 && (
                <div className="flex-shrink-0 w-4">
                  <svg viewBox="0 0 16 12" className="w-4 h-3">
                    <path d="M0 6 L12 6 M8 2 L12 6 L8 10" fill="none" stroke="#9ca3af" strokeWidth={2} />
                  </svg>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Feedback loop arrow */}
        <svg viewBox="0 0 400 30" className="w-full h-6 mt-1">
          <path d="M360 5 C380 5 390 15 390 25 L390 28 L10 28 C5 28 2 25 2 20 L2 15"
            fill="none" stroke="#9ca3af" strokeWidth={1.5} strokeDasharray="4 4" />
          <path d="M2 20 L-2 15 L6 15" fill="#9ca3af" />
          <text x="200" y="22" textAnchor="middle" fontSize="8" fill="#9ca3af">持續迭代 CI/CD</text>
        </svg>
      </div>

      {active && (
        <div className="border rounded-lg p-4" style={{ borderColor: STAGES.find((s) => s.id === active)!.color + "60" }}>
          <h4 className="text-sm font-semibold mb-2">
            {STAGES.find((s) => s.id === active)!.icon} {STAGES.find((s) => s.id === active)!.label}
          </h4>
          <ul className="space-y-1">
            {STAGES.find((s) => s.id === active)!.items.map((item) => (
              <li key={item} className="text-sm text-gray-600 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: STAGES.find((s) => s.id === active)!.color }} />
                {item}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="bg-gray-50 rounded-lg p-3 text-xs text-gray-600">
        <p><strong>MLOps = ML + DevOps</strong></p>
        <p>自動化 ML 生命週期：從資料準備 → 訓練 → 部署 → 監控 → 回饋重訓</p>
      </div>
    </div>
  );
}
