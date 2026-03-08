import { useState, useEffect } from "react";
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
import { fetchAPI } from "../../lib/api";

const COLORS: Record<string, string> = {
  sigmoid: "#2563eb",
  tanh: "#dc2626",
  relu: "#059669",
  leaky_relu: "#7c3aed",
  gelu: "#d97706",
};

const LABELS: Record<string, string> = {
  sigmoid: "Sigmoid",
  tanh: "Tanh",
  relu: "ReLU",
  leaky_relu: "Leaky ReLU",
  gelu: "GELU",
};

export default function ActivationFunctionViz() {
  const [data, setData] = useState<any>(null);
  const [selected, setSelected] = useState<string[]>([
    "sigmoid",
    "relu",
    "tanh",
  ]);
  const [showDerivative, setShowDerivative] = useState(false);

  useEffect(() => {
    fetchAPI("/api/models/activation-functions")
      .then(setData)
      .catch(console.error);
  }, []);

  if (!data)
    return <div className="text-gray-400 text-sm p-4">載入中...</div>;

  const chartData = data.x.map((x: number, i: number) => {
    const point: any = { x: +x.toFixed(2) };
    for (const fn of selected) {
      if (data[fn]) {
        point[fn] = showDerivative ? data[fn].dy?.[i] : data[fn].y[i];
      }
    }
    return point;
  });

  const toggleFn = (fn: string) => {
    setSelected((prev) =>
      prev.includes(fn) ? prev.filter((f) => f !== fn) : [...prev, fn]
    );
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">
        激活函數視覺化 Activation Functions
      </h3>

      <div className="flex flex-wrap gap-2">
        {Object.keys(LABELS).map((fn) => (
          <button
            key={fn}
            onClick={() => toggleFn(fn)}
            className={`px-3 py-1 text-xs rounded-full border transition-colors ${
              selected.includes(fn)
                ? "text-white border-transparent"
                : "text-gray-600 border-gray-200 bg-white"
            }`}
            style={
              selected.includes(fn) ? { backgroundColor: COLORS[fn] } : {}
            }
          >
            {LABELS[fn]}
          </button>
        ))}
        <button
          onClick={() => setShowDerivative(!showDerivative)}
          className={`px-3 py-1 text-xs rounded-full border ${
            showDerivative
              ? "bg-gray-800 text-white"
              : "bg-white text-gray-600 border-gray-200"
          }`}
        >
          {showDerivative ? "導數 Derivative" : "函數 Function"}
        </button>
      </div>

      <div className="border border-gray-200 rounded-lg p-4">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="x" />
            <YAxis />
            <Tooltip />
            <Legend />
            {selected.map((fn) => (
              <Line
                key={fn}
                type="monotone"
                dataKey={fn}
                name={LABELS[fn]}
                stroke={COLORS[fn]}
                dot={false}
                strokeWidth={2}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
