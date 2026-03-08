import { useState, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export default function LearningCurveViz() {
  const [complexity, setComplexity] = useState(3);
  const [noise, setNoise] = useState(0.3);

  const data = useMemo(() => {
    const points = [];
    for (let n = 10; n <= 200; n += 10) {
      const trainError = Math.max(
        0,
        0.01 * complexity + noise * 0.1 - 0.001 * n * complexity
      );
      const valError = Math.max(
        trainError + 0.02,
        0.5 / (1 + 0.01 * n) + 0.02 * complexity + noise * 0.2
      );
      points.push({
        samples: n,
        train: +trainError.toFixed(4),
        validation: +valError.toFixed(4),
      });
    }
    return points;
  }, [complexity, noise]);

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">學習曲線 Learning Curve</h3>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-sm text-gray-600 mb-1">
            模型複雜度: {complexity}
          </label>
          <input
            type="range"
            min="1"
            max="10"
            value={complexity}
            onChange={(e) => setComplexity(+e.target.value)}
            className="w-full"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-600 mb-1">
            資料噪聲: {noise.toFixed(1)}
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={noise}
            onChange={(e) => setNoise(+e.target.value)}
            className="w-full"
          />
        </div>
      </div>
      <div className="border border-gray-200 rounded-lg p-4">
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="samples"
              label={{ value: "訓練樣本數", position: "bottom" }}
            />
            <YAxis
              label={{ value: "Error", angle: -90, position: "insideLeft" }}
            />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="train"
              name="訓練誤差"
              stroke="#2563eb"
              dot={false}
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="validation"
              name="驗證誤差"
              stroke="#dc2626"
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-500 mt-2">
          {complexity > 6
            ? "!! 高複雜度：訓練/驗證差距大 -> 過擬合 (Overfitting)"
            : complexity < 3
              ? "!! 低複雜度：兩條線都高 -> 欠擬合 (Underfitting)"
              : "OK 適當複雜度：兩條線接近且都低"}
        </p>
      </div>
    </div>
  );
}
