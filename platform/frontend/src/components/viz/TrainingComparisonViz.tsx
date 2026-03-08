import { useState, useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from "recharts";

function simulateTraining(lr: number, epochs: number, earlyStop: boolean, augment: boolean) {
  const data: { epoch: number; train: number; val: number }[] = [];
  let trainLoss = 2.5, valLoss = 2.8;
  const noise = augment ? 0.6 : 0.8;
  const overfit = augment ? 0.003 : 0.008;
  let bestVal = Infinity, patience = 0;

  for (let e = 1; e <= epochs; e++) {
    trainLoss *= (1 - lr * 0.3);
    trainLoss = Math.max(0.01, trainLoss + (Math.sin(e * 0.3) * 0.02));
    valLoss = trainLoss * noise + overfit * e + Math.sin(e * 0.5) * 0.03;
    valLoss = Math.max(0.05, valLoss);

    if (earlyStop) {
      if (valLoss < bestVal) { bestVal = valLoss; patience = 0; }
      else { patience++; }
      if (patience >= 10) {
        data.push({ epoch: e, train: +trainLoss.toFixed(4), val: +valLoss.toFixed(4) });
        break;
      }
    }
    data.push({ epoch: e, train: +trainLoss.toFixed(4), val: +valLoss.toFixed(4) });
  }
  return data;
}

const PRESETS = [
  { label: "LR=0.01", lr: 0.01, earlyStop: false, augment: false, color: "#3b82f6" },
  { label: "LR=0.1", lr: 0.1, earlyStop: false, augment: false, color: "#ef4444" },
  { label: "LR=0.01+早停", lr: 0.01, earlyStop: true, augment: false, color: "#10b981" },
  { label: "LR=0.01+增強", lr: 0.01, earlyStop: false, augment: true, color: "#f59e0b" },
];

export default function TrainingComparisonViz() {
  const [selected, setSelected] = useState([0, 1]);
  const epochs = 80;

  const allData = useMemo(() =>
    PRESETS.map((p) => simulateTraining(p.lr, epochs, p.earlyStop, p.augment)),
    []
  );

  const mergedData = useMemo(() => {
    const maxLen = Math.max(...selected.map((i) => allData[i].length));
    return Array.from({ length: maxLen }, (_, e) => {
      const entry: Record<string, number> = { epoch: e + 1 };
      selected.forEach((i) => {
        const d = allData[i][e];
        if (d) {
          entry[`train_${i}`] = d.train;
          entry[`val_${i}`] = d.val;
        }
      });
      return entry;
    });
  }, [selected, allData]);

  const toggle = (idx: number) =>
    setSelected((prev) =>
      prev.includes(idx) ? prev.filter((i) => i !== idx) : [...prev, idx]
    );

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">訓練曲線比較 Training Comparison</h3>

      <div className="flex flex-wrap gap-2">
        {PRESETS.map((p, i) => (
          <button key={i} onClick={() => toggle(i)}
            className={`px-3 py-1 rounded-lg text-sm border-2 transition ${
              selected.includes(i) ? "border-current" : "border-gray-200 opacity-50"
            }`}
            style={{ color: p.color }}
          >{p.label}</button>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={mergedData} margin={{ top: 5, right: 10, bottom: 20, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="epoch" label={{ value: "Epoch", position: "bottom", offset: 0 }} fontSize={11} />
          <YAxis label={{ value: "Loss", angle: -90, position: "insideLeft" }} fontSize={11} />
          <Tooltip />
          <Legend />
          {selected.map((i) => (
            <Line key={`t${i}`} type="monotone" dataKey={`train_${i}`} name={`${PRESETS[i].label} Train`}
              stroke={PRESETS[i].color} dot={false} strokeWidth={2} />
          ))}
          {selected.map((i) => (
            <Line key={`v${i}`} type="monotone" dataKey={`val_${i}`} name={`${PRESETS[i].label} Val`}
              stroke={PRESETS[i].color} dot={false} strokeWidth={2} strokeDasharray="5 5" />
          ))}
        </LineChart>
      </ResponsiveContainer>

      <div className="bg-gray-50 rounded-lg p-3 text-xs text-gray-600">
        <p><strong>實線</strong> = 訓練損失 | <strong>虛線</strong> = 驗證損失</p>
        <p>Train↓ Val↑ → 過擬合 | 早停法在驗證損失不再下降時停止訓練</p>
      </div>
    </div>
  );
}
