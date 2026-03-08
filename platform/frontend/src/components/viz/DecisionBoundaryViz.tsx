import { useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { fetchAPI } from "../../lib/api";

function generateClassificationData(n: number = 100) {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const cls = Math.random() > 0.5 ? 1 : 0;
    const x1 = (Math.random() - 0.5) * 4 + (cls === 1 ? 1 : -1);
    const x2 = (Math.random() - 0.5) * 4 + (cls === 1 ? 1 : -1);
    X.push([x1, x2]);
    y.push(cls);
  }
  return { X, y };
}

export default function DecisionBoundaryViz() {
  const [modelType, setModelType] = useState("logistic");
  const [C, setC] = useState(1.0);
  const [kernel, setKernel] = useState("rbf");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true);
    try {
      const { X, y } = generateClassificationData();
      const data = await fetchAPI("/api/models/decision-boundary", {
        X,
        y,
        model_type: modelType,
        C,
        kernel,
      });
      setResult(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const class0 = result
    ? result.X.filter((_: any, i: number) => result.y[i] === 0).map(
        (p: number[]) => ({ x: p[0], y: p[1] })
      )
    : [];
  const class1 = result
    ? result.X.filter((_: any, i: number) => result.y[i] === 1).map(
        (p: number[]) => ({ x: p[0], y: p[1] })
      )
    : [];

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">
        決策邊界視覺化 Decision Boundary
      </h3>

      <div className="grid grid-cols-3 gap-3">
        <div>
          <label className="block text-sm text-gray-600 mb-1">模型</label>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm"
          >
            <option value="logistic">Logistic Regression</option>
            <option value="svm">SVM</option>
          </select>
        </div>
        <div>
          <label className="block text-sm text-gray-600 mb-1">C = {C}</label>
          <input
            type="range"
            min="0.01"
            max="100"
            step="0.1"
            value={C}
            onChange={(e) => setC(+e.target.value)}
            className="w-full"
          />
        </div>
        {modelType === "svm" && (
          <div>
            <label className="block text-sm text-gray-600 mb-1">
              核函數 Kernel
            </label>
            <select
              value={kernel}
              onChange={(e) => setKernel(e.target.value)}
              className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm"
            >
              <option value="linear">Linear</option>
              <option value="rbf">RBF</option>
              <option value="poly">Polynomial</option>
            </select>
          </div>
        )}
      </div>

      <button
        onClick={run}
        disabled={loading}
        className="px-4 py-2 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600 disabled:opacity-50"
      >
        {loading ? "訓練中..." : "訓練模型"}
      </button>

      {result && (
        <div className="border border-gray-200 rounded-lg p-4">
          <p className="text-sm text-gray-600 mb-2">
            準確率 Accuracy:{" "}
            <strong>{(result.accuracy * 100).toFixed(1)}%</strong>
          </p>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" type="number" name="X1" />
              <YAxis dataKey="y" type="number" name="X2" />
              <Tooltip />
              <Scatter
                name="Class 0"
                data={class0}
                fill="#2563eb"
                opacity={0.7}
              />
              <Scatter
                name="Class 1"
                data={class1}
                fill="#dc2626"
                opacity={0.7}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
