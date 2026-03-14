import { useState, useMemo, lazy, Suspense } from "react";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const EmbeddingSpace3D = lazy(() => import("../viz3d/EmbeddingSpace3D"));

const CONCEPTS: { word: string; category: string; x: number; y: number }[] = [
  // ML algorithms cluster
  { word: "Linear Regression", category: "回歸", x: 1.2, y: 3.5 },
  { word: "Ridge", category: "回歸", x: 1.5, y: 3.8 },
  { word: "Lasso", category: "回歸", x: 1.0, y: 4.0 },
  { word: "Logistic Regression", category: "分類", x: 3.2, y: 3.0 },
  { word: "SVM", category: "分類", x: 3.8, y: 2.8 },
  { word: "Decision Tree", category: "樹模型", x: 5.5, y: 2.0 },
  { word: "Random Forest", category: "樹模型", x: 5.8, y: 1.8 },
  { word: "XGBoost", category: "樹模型", x: 6.2, y: 2.3 },
  // DL cluster
  { word: "CNN", category: "深度學習", x: 8.0, y: 5.5 },
  { word: "RNN", category: "深度學習", x: 8.5, y: 6.0 },
  { word: "LSTM", category: "深度學習", x: 8.8, y: 6.3 },
  { word: "Transformer", category: "深度學習", x: 9.0, y: 5.8 },
  { word: "Attention", category: "深度學習", x: 9.2, y: 5.5 },
  // Evaluation cluster
  { word: "Accuracy", category: "評估", x: 4.5, y: 7.0 },
  { word: "Precision", category: "評估", x: 4.8, y: 7.3 },
  { word: "Recall", category: "評估", x: 4.3, y: 7.5 },
  { word: "F1 Score", category: "評估", x: 4.6, y: 7.8 },
  { word: "AUC", category: "評估", x: 5.0, y: 7.1 },
];

const CATEGORY_COLORS: Record<string, string> = {
  "回歸": "#3b82f6",
  "分類": "#10b981",
  "樹模型": "#f59e0b",
  "深度學習": "#ef4444",
  "評估": "#8b5cf6",
};

function addNoise(concepts: typeof CONCEPTS, noise: number) {
  const rng = (s: number) => { let v = s; return () => { v = (v * 16807) % 2147483647; return ((v - 1) / 2147483646 - 0.5) * 2; }; };
  const r = rng(42);
  return concepts.map((c) => ({
    ...c,
    x: +(c.x + r() * noise).toFixed(2),
    y: +(c.y + r() * noise).toFixed(2),
  }));
}

export default function EmbeddingSpaceViz() {
  const [viewMode, setViewMode] = useState<"2d" | "3d">("2d");
  const [noise, setNoise] = useState(0.3);
  const [highlight, setHighlight] = useState<string | null>(null);

  const data = useMemo(() => addNoise(CONCEPTS, noise), [noise]);
  const categories = [...new Set(CONCEPTS.map((c) => c.category))];

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">嵌入空間 Embedding Space</h3>
        <div className="inline-flex rounded-lg border border-gray-200 overflow-hidden text-sm">
          <button onClick={() => setViewMode("2d")}
            className={`px-3 py-1 ${viewMode === "2d" ? "bg-blue-500 text-white" : "bg-white text-gray-600"}`}>2D</button>
          <button onClick={() => setViewMode("3d")}
            className={`px-3 py-1 ${viewMode === "3d" ? "bg-blue-500 text-white" : "bg-white text-gray-600"}`}>3D</button>
        </div>
      </div>

      {viewMode === "2d" && (
        <>
          <div>
            <label className="text-sm text-gray-600">
              擾動程度 Noise: <strong>{noise.toFixed(1)}</strong>
            </label>
            <input type="range" min={0} max={1.5} step={0.1} value={noise}
              onChange={(e) => setNoise(+e.target.value)} className="w-full" />
          </div>

          <div className="flex gap-2 flex-wrap">
            {categories.map((cat) => (
              <button key={cat}
                onMouseEnter={() => setHighlight(cat)}
                onMouseLeave={() => setHighlight(null)}
                className="flex items-center gap-1 px-2 py-1 rounded text-xs"
                style={{
                  backgroundColor: CATEGORY_COLORS[cat] + "20",
                  color: CATEGORY_COLORS[cat],
                  opacity: highlight && highlight !== cat ? 0.3 : 1,
                }}
              >
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: CATEGORY_COLORS[cat] }} />
                {cat}
              </button>
            ))}
          </div>

          <ResponsiveContainer width="100%" height={280}>
            <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" type="number" name="Dim 1" fontSize={11} domain={[0, 10]} />
              <YAxis dataKey="y" type="number" name="Dim 2" fontSize={11} domain={[0, 9]} />
              <Tooltip content={({ payload }) => {
                if (!payload?.[0]) return null;
                const d = payload[0].payload;
                return (
                  <div className="bg-white border rounded px-2 py-1 text-xs shadow">
                    <p className="font-medium">{d.word}</p>
                    <p className="text-gray-500">{d.category}</p>
                  </div>
                );
              }} />
              {categories.map((cat) => (
                <Scatter key={cat} data={data.filter((d) => d.category === cat)}
                  fill={CATEGORY_COLORS[cat]}
                  opacity={highlight && highlight !== cat ? 0.15 : 0.8}
                  r={6} name={cat} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>

          <div className="bg-gray-50 rounded-lg p-3 text-xs text-gray-600">
            <p><strong>嵌入 (Embedding)：</strong>將文字/概念映射到連續向量空間</p>
            <p>語義相似的概念 → 空間中距離較近 | 經 PCA/t-SNE 降至 2D 顯示</p>
          </div>
        </>
      )}

      {viewMode === "3d" && (
        <Suspense fallback={<div className="text-gray-400 text-sm p-4">載入 3D 視覺化...</div>}>
          <EmbeddingSpace3D noise={noise} onNoiseChange={setNoise} />
        </Suspense>
      )}
    </div>
  );
}
