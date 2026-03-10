import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { fetchAPI } from "../../lib/api";

// Simplified iris-like data for demo
function generateIrisLikeData() {
  const X: number[][] = [];
  const y: number[] = [];
  const featureNames = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
  ];
  for (let i = 0; i < 150; i++) {
    const cls = i < 50 ? 0 : i < 100 ? 1 : 2;
    X.push([
      5 + cls * 0.5 + (Math.random() - 0.5),
      3 + (Math.random() - 0.5) * 0.8,
      1.5 + cls * 2 + (Math.random() - 0.5),
      0.3 + cls * 0.8 + (Math.random() - 0.5) * 0.3,
    ]);
    y.push(cls);
  }
  return { X, y, featureNames };
}

export default function FeatureImportanceViz() {
  const [modelType, setModelType] = useState("random_forest");
  const [maxDepth, setMaxDepth] = useState(5);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true);
    try {
      const { X, y, featureNames } = generateIrisLikeData();
      const data = await fetchAPI<Record<string, unknown>>("/api/models/tree", {
        X,
        y,
        model_type: modelType,
        max_depth: maxDepth,
        feature_names: featureNames,
      });
      setResult({ ...data, featureNames });
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const chartData = result
    ? result.featureNames
        .map((name: string, i: number) => ({
          name,
          importance: +(result.feature_importances[i] * 100).toFixed(1),
        }))
        .sort((a: any, b: any) => b.importance - a.importance)
    : [];

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">
        特徵重要度 Feature Importance
      </h3>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-sm text-gray-600 mb-1">模型</label>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm"
          >
            <option value="decision_tree">Decision Tree</option>
            <option value="random_forest">Random Forest</option>
            <option value="gradient_boosting">GBDT</option>
          </select>
        </div>
        <div>
          <label className="block text-sm text-gray-600 mb-1">
            max_depth = {maxDepth}
          </label>
          <input
            type="range"
            min="1"
            max="15"
            value={maxDepth}
            onChange={(e) => setMaxDepth(+e.target.value)}
            className="w-full"
          />
        </div>
      </div>
      <button
        onClick={run}
        disabled={loading}
        className="px-4 py-2 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600 disabled:opacity-50"
      >
        {loading ? "計算中..." : "計算特徵重要度"}
      </button>
      {result && (
        <div className="border border-gray-200 rounded-lg p-4">
          <p className="text-sm text-gray-600 mb-2">
            準確率: <strong>{(result.accuracy * 100).toFixed(1)}%</strong>
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" unit="%" />
              <YAxis dataKey="name" type="category" width={100} />
              <Tooltip />
              <Bar
                dataKey="importance"
                fill="#7c3aed"
                radius={[0, 4, 4, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
