import { useState, useMemo } from "react";

const COLORS = { train: "#3b82f6", val: "#f59e0b", test: "#ef4444" };
const TOTAL = 100;

export default function DataSplitViz() {
  const [trainPct, setTrainPct] = useState(70);
  const [valPct, setValPct] = useState(15);
  const testPct = TOTAL - trainPct - valPct;

  const dots = useMemo(() => {
    const rng = (s: number) => { let v = s; return () => { v = (v * 16807) % 2147483647; return (v - 1) / 2147483646; }; };
    const r = rng(123);
    return Array.from({ length: TOTAL }, (_, i) => ({
      id: i, x: r() * 90 + 5, y: r() * 90 + 5,
      set: i < trainPct ? "train" : i < trainPct + valPct ? "val" : "test",
    }));
  }, [trainPct, valPct]);

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">資料分割視覺化 Data Split</h3>

      <div className="space-y-2">
        <div>
          <label className="text-sm text-gray-600">
            訓練集 Train: <strong className="text-blue-600">{trainPct}%</strong>
          </label>
          <input type="range" min={50} max={90} value={trainPct}
            onChange={(e) => { const v = +e.target.value; setTrainPct(v); if (v + valPct > 95) setValPct(95 - v); }}
            className="w-full" />
        </div>
        <div>
          <label className="text-sm text-gray-600">
            驗證集 Validation: <strong className="text-amber-500">{valPct}%</strong>
          </label>
          <input type="range" min={0} max={Math.min(30, TOTAL - trainPct - 5)} value={valPct}
            onChange={(e) => setValPct(+e.target.value)} className="w-full" />
        </div>
        <p className="text-sm text-gray-600">
          測試集 Test: <strong className="text-red-500">{testPct}%</strong>
        </p>
      </div>

      <div className="flex gap-1 h-6 rounded-lg overflow-hidden">
        <div className="transition-all" style={{ width: `${trainPct}%`, backgroundColor: COLORS.train }} />
        <div className="transition-all" style={{ width: `${valPct}%`, backgroundColor: COLORS.val }} />
        <div className="transition-all" style={{ width: `${testPct}%`, backgroundColor: COLORS.test }} />
      </div>

      <svg viewBox="0 0 100 100" className="w-full h-48 border border-gray-200 rounded-lg bg-white">
        {dots.map((d) => (
          <circle key={d.id} cx={d.x} cy={d.y} r={2.5}
            fill={COLORS[d.set as keyof typeof COLORS]} opacity={0.7}
            className="transition-all duration-500" />
        ))}
      </svg>

      <div className="flex gap-4 justify-center text-xs">
        {[
          { label: `訓練集 (${trainPct})`, color: COLORS.train },
          { label: `驗證集 (${valPct})`, color: COLORS.val },
          { label: `測試集 (${testPct})`, color: COLORS.test },
        ].map((l) => (
          <span key={l.label} className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-full" style={{ backgroundColor: l.color }} />
            {l.label}
          </span>
        ))}
      </div>

      <div className="bg-gray-50 rounded-lg p-3 text-xs text-gray-600">
        <p><strong>常見比例：</strong>70/15/15 或 80/10/10</p>
        <p>訓練集越大 → 模型學到越多 | 測試集越大 → 評估越可靠</p>
      </div>
    </div>
  );
}
