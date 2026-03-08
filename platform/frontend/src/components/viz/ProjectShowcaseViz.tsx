import { useState, useMemo } from "react";
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

const RUBRIC_DIMS = [
  { key: "problem", label: "問題定義", weight: 15 },
  { key: "data", label: "資料處理", weight: 20 },
  { key: "model", label: "模型實驗", weight: 25 },
  { key: "analysis", label: "結果分析", weight: 20 },
  { key: "report", label: "報告展示", weight: 10 },
  { key: "reflection", label: "反思改進", weight: 10 },
];

const SAMPLE_PROJECTS = [
  {
    title: "醫療影像分類器",
    team: "第 1 組",
    desc: "使用 CNN 對 X 光影像進行肺炎檢測",
    tags: ["CNN", "影像分類", "醫療"],
    color: "#3b82f6",
    scores: { problem: 90, data: 85, model: 92, analysis: 88, report: 82, reflection: 85 },
  },
  {
    title: "股票趨勢預測",
    team: "第 2 組",
    desc: "LSTM 預測台股大盤走勢，結合技術指標",
    tags: ["LSTM", "時序預測", "金融"],
    color: "#10b981",
    scores: { problem: 88, data: 82, model: 85, analysis: 80, report: 90, reflection: 78 },
  },
  {
    title: "客戶流失預測",
    team: "第 3 組",
    desc: "XGBoost + SHAP 解釋電信客戶流失因素",
    tags: ["XGBoost", "SHAP", "分類"],
    color: "#f59e0b",
    scores: { problem: 92, data: 90, model: 88, analysis: 95, report: 85, reflection: 92 },
  },
  {
    title: "文字情感分析",
    team: "第 4 組",
    desc: "Transformer 模型分析 PTT 評論情感",
    tags: ["Transformer", "NLP", "情感"],
    color: "#ef4444",
    scores: { problem: 85, data: 78, model: 90, analysis: 82, report: 88, reflection: 80 },
  },
];

const PIPELINE_STEPS = [
  { id: "problem", label: "問題定義", icon: "🎯", detail: "確定任務類型、評估指標、成功標準" },
  { id: "data", label: "資料收集", icon: "📦", detail: "資料來源、清洗、EDA 探索分析" },
  { id: "feature", label: "特徵工程", icon: "🔧", detail: "編碼、縮放、選擇、Pipeline 建構" },
  { id: "model", label: "模型訓練", icon: "🧠", detail: "多模型比較、超參數調校、交叉驗證" },
  { id: "eval", label: "模型評估", icon: "📊", detail: "混淆矩陣、ROC/PR、公平性檢測" },
  { id: "deploy", label: "部署監測", icon: "🚀", detail: "模型序列化、API 服務、drift 監測" },
];

type ViewType = "radar" | "pipeline" | "scorer";

export default function ProjectShowcaseViz() {
  const [view, setView] = useState<ViewType>("radar");
  const [selectedProjects, setSelectedProjects] = useState<number[]>([0, 2]);
  const [activeStep, setActiveStep] = useState<number | null>(null);
  const [userScores, setUserScores] = useState<Record<string, number>>(
    Object.fromEntries(RUBRIC_DIMS.map((d) => [d.key, 75]))
  );

  const toggleProject = (idx: number) => {
    setSelectedProjects((prev) =>
      prev.includes(idx) ? prev.filter((i) => i !== idx) : [...prev, idx]
    );
  };

  const radarData = useMemo(
    () =>
      RUBRIC_DIMS.map((dim) => {
        const entry: Record<string, string | number> = { dimension: dim.label };
        selectedProjects.forEach((idx) => {
          entry[SAMPLE_PROJECTS[idx].title] =
            SAMPLE_PROJECTS[idx].scores[dim.key as keyof typeof SAMPLE_PROJECTS[0]["scores"]];
        });
        return entry;
      }),
    [selectedProjects]
  );

  const weightedTotal = useMemo(() => {
    let total = 0;
    RUBRIC_DIMS.forEach((dim) => {
      total += (userScores[dim.key] * dim.weight) / 100;
    });
    return Math.round(total);
  }, [userScores]);

  const barData = useMemo(
    () =>
      SAMPLE_PROJECTS.map((p) => {
        let total = 0;
        RUBRIC_DIMS.forEach((dim) => {
          total +=
            (p.scores[dim.key as keyof typeof p.scores] * dim.weight) / 100;
        });
        return { name: p.team, score: Math.round(total), fill: p.color };
      }),
    []
  );

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">綜合專題展示 Project Showcase</h3>

      <div className="flex gap-2">
        {(
          [
            { key: "radar", label: "專題比較" },
            { key: "pipeline", label: "ML Pipeline" },
            { key: "scorer", label: "評分模擬" },
          ] as { key: ViewType; label: string }[]
        ).map((tab) => (
          <button
            key={tab.key}
            onClick={() => setView(tab.key)}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              view === tab.key
                ? "bg-blue-500 text-white"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {view === "radar" && (
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {SAMPLE_PROJECTS.map((proj, idx) => (
              <button
                key={proj.title}
                onClick={() => toggleProject(idx)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border transition-all ${
                  selectedProjects.includes(idx)
                    ? "border-current shadow-sm"
                    : "border-gray-200 text-gray-400"
                }`}
                style={
                  selectedProjects.includes(idx)
                    ? { color: proj.color, borderColor: proj.color, backgroundColor: proj.color + "10" }
                    : {}
                }
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{
                    backgroundColor: selectedProjects.includes(idx) ? proj.color : "#d1d5db",
                  }}
                />
                {proj.team} {proj.title}
              </button>
            ))}
          </div>

          {selectedProjects.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={radarData}>
                <PolarGrid strokeDasharray="3 3" />
                <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                {selectedProjects.map((idx) => (
                  <Radar
                    key={idx}
                    name={SAMPLE_PROJECTS[idx].title}
                    dataKey={SAMPLE_PROJECTS[idx].title}
                    stroke={SAMPLE_PROJECTS[idx].color}
                    fill={SAMPLE_PROJECTS[idx].color}
                    fillOpacity={0.15}
                    strokeWidth={2}
                  />
                ))}
                <Legend wrapperStyle={{ fontSize: 11 }} />
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center text-gray-400 text-sm py-10">
              請點選上方專題以顯示比較圖
            </div>
          )}

          <ResponsiveContainer width="100%" height={140}>
            <BarChart data={barData} layout="vertical" margin={{ left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={50} />
              <Tooltip formatter={(v: number) => [`${v} 分`, "加權總分"]} />
              <Bar dataKey="score" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {view === "pipeline" && (
        <div className="space-y-3">
          <p className="text-xs text-gray-500">
            點擊各步驟查看說明。期末專題需完成完整 ML Pipeline。
          </p>
          <div className="flex items-center gap-1">
            {PIPELINE_STEPS.map((step, idx) => (
              <div key={step.id} className="flex items-center">
                <button
                  onClick={() => setActiveStep(activeStep === idx ? null : idx)}
                  className={`flex flex-col items-center gap-1 px-2 py-3 rounded-lg text-xs transition-all min-w-[64px] ${
                    activeStep === idx
                      ? "bg-blue-50 border-2 border-blue-400 shadow-sm"
                      : "bg-gray-50 border border-gray-200 hover:bg-gray-100"
                  }`}
                >
                  <span className="text-xl">{step.icon}</span>
                  <span className="font-medium leading-tight text-center">{step.label}</span>
                </button>
                {idx < PIPELINE_STEPS.length - 1 && (
                  <span className="text-gray-300 text-lg mx-0.5">→</span>
                )}
              </div>
            ))}
          </div>

          {activeStep !== null && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl">{PIPELINE_STEPS[activeStep].icon}</span>
                <span className="font-semibold text-blue-800">
                  Step {activeStep + 1}: {PIPELINE_STEPS[activeStep].label}
                </span>
              </div>
              <p className="text-blue-700">{PIPELINE_STEPS[activeStep].detail}</p>
              <div className="mt-2 text-xs text-blue-500">
                相關週次：
                {activeStep === 0 && " Week 1-3（問題定義、資料科學環境）"}
                {activeStep === 1 && " Week 2-3（EDA、資料分割）"}
                {activeStep === 2 && " Week 8-9（特徵重要度、特徵工程）"}
                {activeStep === 3 && " Week 4-7, 11-13（回歸、分類、樹模型、神經網路）"}
                {activeStep === 4 && " Week 10, 15（超參數調校、模型評估與公平性）"}
                {activeStep === 5 && " Week 16（MLOps 入門）"}
              </div>
            </div>
          )}

          <div className="border border-gray-200 rounded-lg p-3">
            <h4 className="text-sm font-semibold mb-2">範例專題對照</h4>
            <div className="grid grid-cols-2 gap-2">
              {SAMPLE_PROJECTS.map((proj) => (
                <div key={proj.title} className="bg-gray-50 rounded-lg p-2.5">
                  <div className="flex items-center gap-1.5 mb-1">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: proj.color }}
                    />
                    <span className="text-xs font-medium">{proj.title}</span>
                  </div>
                  <p className="text-xs text-gray-500">{proj.desc}</p>
                  <div className="flex gap-1 mt-1.5">
                    {proj.tags.map((tag) => (
                      <span
                        key={tag}
                        className="text-[10px] bg-white text-gray-500 px-1.5 py-0.5 rounded border border-gray-200"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {view === "scorer" && (
        <div className="space-y-3">
          <p className="text-xs text-gray-500">
            拖動滑桿模擬各面向評分，觀察加權總分變化。
          </p>

          <div className="space-y-3">
            {RUBRIC_DIMS.map((dim) => (
              <div key={dim.key}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-medium">
                    {dim.label}
                    <span className="text-gray-400 ml-1">({dim.weight}%)</span>
                  </span>
                  <span
                    className={`font-bold ${
                      userScores[dim.key] >= 90
                        ? "text-green-600"
                        : userScores[dim.key] >= 70
                        ? "text-blue-600"
                        : "text-orange-500"
                    }`}
                  >
                    {userScores[dim.key]}
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={userScores[dim.key]}
                  onChange={(e) =>
                    setUserScores((prev) => ({
                      ...prev,
                      [dim.key]: Number(e.target.value),
                    }))
                  }
                  className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
              </div>
            ))}
          </div>

          <div
            className={`text-center py-4 rounded-xl border-2 ${
              weightedTotal >= 90
                ? "bg-green-50 border-green-300"
                : weightedTotal >= 80
                ? "bg-blue-50 border-blue-300"
                : weightedTotal >= 70
                ? "bg-yellow-50 border-yellow-300"
                : "bg-red-50 border-red-300"
            }`}
          >
            <p className="text-xs text-gray-500 mb-1">加權總分</p>
            <p className="text-3xl font-bold">
              {weightedTotal}
              <span className="text-sm font-normal text-gray-400 ml-1">/ 100</span>
            </p>
            <p className="text-xs mt-1">
              {weightedTotal >= 90
                ? "🏆 優秀 Excellent"
                : weightedTotal >= 80
                ? "👍 良好 Good"
                : weightedTotal >= 70
                ? "📝 及格 Pass"
                : "⚠️ 待改進 Needs Improvement"}
            </p>
          </div>

          <div className="bg-gray-50 rounded-lg p-3 text-xs text-gray-600">
            <p className="font-medium mb-1">評分公式：</p>
            <p>
              總分 ={" "}
              {RUBRIC_DIMS.map((d, i) => (
                <span key={d.key}>
                  {i > 0 && " + "}
                  {d.label}×{d.weight}%
                </span>
              ))}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
