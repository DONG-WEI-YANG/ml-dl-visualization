import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from "recharts";

const SCENARIOS: Record<string, { groups: { name: string; tp: number; fp: number; fn: number; tn: number }[] }> = {
  balanced: {
    groups: [
      { name: "A 組", tp: 80, fp: 10, fn: 10, tn: 100 },
      { name: "B 組", tp: 75, fp: 12, fn: 13, tn: 100 },
    ],
  },
  biased: {
    groups: [
      { name: "A 組", tp: 85, fp: 5, fn: 5, tn: 105 },
      { name: "B 組", tp: 50, fp: 30, fn: 40, tn: 80 },
    ],
  },
  mitigated: {
    groups: [
      { name: "A 組", tp: 78, fp: 12, fn: 10, tn: 100 },
      { name: "B 組", tp: 70, fp: 15, fn: 15, tn: 100 },
    ],
  },
};

function computeMetrics(g: { tp: number; fp: number; fn: number; tn: number }) {
  const acc = (g.tp + g.tn) / (g.tp + g.fp + g.fn + g.tn);
  const precision = g.tp / (g.tp + g.fp);
  const recall = g.tp / (g.tp + g.fn);
  const fpr = g.fp / (g.fp + g.tn);
  return {
    accuracy: +(acc * 100).toFixed(1),
    precision: +(precision * 100).toFixed(1),
    recall: +(recall * 100).toFixed(1),
    fpr: +(fpr * 100).toFixed(1),
  };
}

export default function FairnessViz() {
  const [scenario, setScenario] = useState("biased");
  const data = SCENARIOS[scenario];

  const chartData = data.groups.map((g) => {
    const m = computeMetrics(g);
    return { name: g.name, ...m };
  });

  const gap = Math.abs(chartData[0].accuracy - chartData[1].accuracy);
  const recallGap = Math.abs(chartData[0].recall - chartData[1].recall);

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">公平性指標 Fairness Metrics</h3>

      <div className="flex gap-2">
        {[
          { id: "balanced", label: "公平模型" },
          { id: "biased", label: "有偏差模型" },
          { id: "mitigated", label: "修正後模型" },
        ].map((s) => (
          <button key={s.id} onClick={() => setScenario(s.id)}
            className={`px-3 py-1 rounded text-sm ${scenario === s.id ? "bg-blue-500 text-white" : "bg-gray-100"}`}
          >{s.label}</button>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" fontSize={12} />
          <YAxis domain={[0, 100]} fontSize={11} />
          <Tooltip formatter={(v: number) => `${v}%`} />
          <Legend />
          <Bar dataKey="accuracy" name="Accuracy" fill="#3b82f6" />
          <Bar dataKey="precision" name="Precision" fill="#10b981" />
          <Bar dataKey="recall" name="Recall" fill="#f59e0b" />
          <Bar dataKey="fpr" name="FPR" fill="#ef4444" />
        </BarChart>
      </ResponsiveContainer>

      <div className="grid grid-cols-2 gap-2 text-center">
        <div className={`rounded-lg p-2 ${gap > 10 ? "bg-red-50" : "bg-green-50"}`}>
          <p className="text-xs text-gray-500">Accuracy Gap</p>
          <p className={`text-lg font-bold ${gap > 10 ? "text-red-600" : "text-green-600"}`}>{gap.toFixed(1)}%</p>
        </div>
        <div className={`rounded-lg p-2 ${recallGap > 10 ? "bg-red-50" : "bg-green-50"}`}>
          <p className="text-xs text-gray-500">Recall Gap</p>
          <p className={`text-lg font-bold ${recallGap > 10 ? "text-red-600" : "text-green-600"}`}>{recallGap.toFixed(1)}%</p>
        </div>
      </div>

      <div className="bg-gray-50 rounded-lg p-3 text-xs text-gray-600">
        <p><strong>公平性原則：</strong>不同群體應有相近的預測表現</p>
        <p>Gap &gt; 10% → 可能存在偏差，需要檢視訓練資料或模型</p>
      </div>
    </div>
  );
}
