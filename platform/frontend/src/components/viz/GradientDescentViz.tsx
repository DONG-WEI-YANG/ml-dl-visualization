import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { fetchAPI } from "../../lib/api";

interface GDResult {
  loss_history: number[];
  weights_history: number[][];
  final_weights: number[];
  final_loss: number;
}

// 生成範例資料
function generateData(n: number = 50) {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = (Math.random() - 0.5) * 10;
    X.push([x]);
    y.push(2 * x + 3 + (Math.random() - 0.5) * 2);
  }
  return { X, y };
}

export default function GradientDescentViz() {
  const [lr, setLr] = useState(0.01);
  const [epochs, setEpochs] = useState(100);
  const [result, setResult] = useState<GDResult | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true);
    try {
      const { X, y } = generateData();
      const data = await fetchAPI<GDResult>("/api/models/gradient-descent", {
        X,
        y,
        learning_rate: lr,
        epochs,
      });
      setResult(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const lossData =
    result?.loss_history.map((loss, i) => ({
      epoch: i + 1,
      loss: +loss.toFixed(4),
    })) ?? [];

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">
        梯度下降視覺化 Gradient Descent
      </h3>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-gray-600 mb-1">
            學習率 Learning Rate: <strong>{lr}</strong>
          </label>
          <input
            type="range"
            min="0.001"
            max="1"
            step="0.001"
            value={lr}
            onChange={(e) => setLr(+e.target.value)}
            className="w-full"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-600 mb-1">
            迭代次數 Epochs: <strong>{epochs}</strong>
          </label>
          <input
            type="range"
            min="10"
            max="500"
            step="10"
            value={epochs}
            onChange={(e) => setEpochs(+e.target.value)}
            className="w-full"
          />
        </div>
      </div>

      <button
        onClick={run}
        disabled={loading}
        className="px-4 py-2 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600 disabled:opacity-50"
      >
        {loading ? "訓練中..." : "執行梯度下降"}
      </button>

      {result && (
        <div className="space-y-4">
          <div className="border border-gray-200 rounded-lg p-4">
            <p className="text-sm text-gray-600 mb-2">
              損失曲線 Loss Curve | 最終損失:{" "}
              <strong>{result.final_loss.toFixed(4)}</strong>
            </p>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="epoch"
                  label={{ value: "Epoch", position: "bottom" }}
                />
                <YAxis
                  label={{ value: "Loss", angle: -90, position: "insideLeft" }}
                />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#2563eb"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="bg-gray-50 rounded-lg p-3 text-sm">
            <p>
              最終權重 Final Weights: [
              {result.final_weights.map((w) => w.toFixed(4)).join(", ")}]
            </p>
            <p className="text-gray-500 text-xs mt-1">
              真實值: w=2.0, b=3.0 | 預期收斂至接近此值
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
