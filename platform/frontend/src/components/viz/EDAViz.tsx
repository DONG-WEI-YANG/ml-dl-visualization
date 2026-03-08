import { useState, useMemo } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar,
} from "recharts";

function generateDataset(type: string) {
  const rng = (seed: number) => {
    let s = seed;
    return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
  };
  const r = rng(42);
  const n = 100;
  const data: { x: number; y: number; label: number }[] = [];

  if (type === "linear") {
    for (let i = 0; i < n; i++) {
      const x = r() * 10;
      data.push({ x: +x.toFixed(2), y: +(2 * x + 3 + (r() - 0.5) * 4).toFixed(2), label: 0 });
    }
  } else if (type === "clusters") {
    for (let i = 0; i < n; i++) {
      const c = i < n / 2 ? 0 : 1;
      const cx = c === 0 ? 3 : 7, cy = c === 0 ? 3 : 7;
      data.push({
        x: +(cx + (r() - 0.5) * 3).toFixed(2),
        y: +(cy + (r() - 0.5) * 3).toFixed(2),
        label: c,
      });
    }
  } else {
    for (let i = 0; i < n; i++) {
      const x = r() * 10;
      data.push({ x: +x.toFixed(2), y: +(Math.sin(x) * 3 + (r() - 0.5) * 2).toFixed(2), label: 0 });
    }
  }
  return data;
}

function computeStats(data: { x: number; y: number }[]) {
  const xs = data.map((d) => d.x), ys = data.map((d) => d.y);
  const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
  const std = (a: number[], m: number) => Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length);
  const mx = mean(xs), my = mean(ys);
  const corr = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0) /
    (Math.sqrt(xs.reduce((s, x) => s + (x - mx) ** 2, 0)) * Math.sqrt(ys.reduce((s, y) => s + (y - my) ** 2, 0)));
  return {
    n: data.length, meanX: mx.toFixed(2), meanY: my.toFixed(2),
    stdX: std(xs, mx).toFixed(2), stdY: std(ys, my).toFixed(2),
    corr: corr.toFixed(3),
  };
}

export default function EDAViz() {
  const [dataType, setDataType] = useState("linear");
  const data = useMemo(() => generateDataset(dataType), [dataType]);
  const stats = useMemo(() => computeStats(data), [data]);

  const histBins = useMemo(() => {
    const bins = Array.from({ length: 10 }, (_, i) => ({ range: `${i}-${i + 1}`, count: 0 }));
    data.forEach((d) => { const b = Math.min(Math.floor(d.x), 9); bins[b].count++; });
    return bins;
  }, [data]);

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">探索式資料分析 EDA</h3>

      <div className="flex gap-2">
        {[
          { id: "linear", label: "線性" },
          { id: "clusters", label: "群集" },
          { id: "nonlinear", label: "非線性" },
        ].map((t) => (
          <button key={t.id} onClick={() => setDataType(t.id)}
            className={`px-3 py-1 rounded-lg text-sm ${dataType === t.id ? "bg-blue-500 text-white" : "bg-gray-100 text-gray-700"}`}
          >{t.label}</button>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-sm text-gray-600 mb-1">散佈圖 Scatter Plot</p>
          <ResponsiveContainer width="100%" height={200}>
            <ScatterChart margin={{ top: 5, right: 5, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" type="number" name="X" fontSize={11} />
              <YAxis dataKey="y" type="number" name="Y" fontSize={11} />
              <Tooltip />
              <Scatter data={data.filter((d) => d.label === 0)} fill="#3b82f6" r={3} />
              {dataType === "clusters" && <Scatter data={data.filter((d) => d.label === 1)} fill="#ef4444" r={3} />}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div>
          <p className="text-sm text-gray-600 mb-1">直方圖 Histogram (X)</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={histBins} margin={{ top: 5, right: 5, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="range" fontSize={10} />
              <YAxis fontSize={11} />
              <Tooltip />
              <Bar dataKey="count" fill="#8b5cf6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 text-center">
        {[
          { label: "樣本數 N", value: stats.n },
          { label: "Mean(X)", value: stats.meanX },
          { label: "Mean(Y)", value: stats.meanY },
          { label: "Std(X)", value: stats.stdX },
          { label: "Std(Y)", value: stats.stdY },
          { label: "Corr(X,Y)", value: stats.corr },
        ].map((s) => (
          <div key={s.label} className="bg-gray-50 rounded-lg p-2">
            <p className="text-xs text-gray-500">{s.label}</p>
            <p className="text-sm font-bold">{s.value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
