import { useState } from "react";

const PACKAGES = [
  { name: "Python", version: "3.10+", desc: "程式語言" },
  { name: "NumPy", version: "1.24+", desc: "數值計算" },
  { name: "Pandas", version: "2.0+", desc: "資料處理" },
  { name: "Matplotlib", version: "3.7+", desc: "繪圖" },
  { name: "Seaborn", version: "0.12+", desc: "統計繪圖" },
  { name: "Scikit-learn", version: "1.3+", desc: "機器學習" },
  { name: "Jupyter", version: "4.0+", desc: "互動筆記本" },
  { name: "PyTorch", version: "2.0+", desc: "深度學習 (W11+)" },
];

export default function EnvironmentSetupViz() {
  const [checked, setChecked] = useState<Record<string, boolean>>({});

  const toggle = (name: string) =>
    setChecked((prev) => ({ ...prev, [name]: !prev[name] }));

  const done = Object.values(checked).filter(Boolean).length;
  const pct = Math.round((done / PACKAGES.length) * 100);

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">環境設置檢查 Environment Setup</h3>

      <div className="relative h-3 bg-gray-200 rounded-full overflow-hidden">
        <div
          className="h-full bg-green-500 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="text-sm text-gray-500">
        已完成 {done}/{PACKAGES.length} ({pct}%)
      </p>

      <div className="space-y-2">
        {PACKAGES.map((pkg) => (
          <button
            key={pkg.name}
            onClick={() => toggle(pkg.name)}
            className={`w-full flex items-center gap-3 p-3 rounded-lg border text-left transition-colors ${
              checked[pkg.name]
                ? "border-green-300 bg-green-50"
                : "border-gray-200 hover:bg-gray-50"
            }`}
          >
            <span
              className={`w-5 h-5 rounded flex items-center justify-center text-xs ${
                checked[pkg.name]
                  ? "bg-green-500 text-white"
                  : "border border-gray-300"
              }`}
            >
              {checked[pkg.name] && "✓"}
            </span>
            <div className="flex-1">
              <span className="font-medium text-sm">{pkg.name}</span>
              <span className="text-xs text-gray-400 ml-2">{pkg.version}</span>
            </div>
            <span className="text-xs text-gray-500">{pkg.desc}</span>
          </button>
        ))}
      </div>

      <div className="bg-gray-50 rounded-lg p-3 text-xs text-gray-600">
        <p className="font-medium mb-1">安裝指令：</p>
        <code className="block bg-gray-900 text-green-400 p-2 rounded text-xs">
          pip install numpy pandas matplotlib seaborn scikit-learn jupyter torch
        </code>
      </div>
    </div>
  );
}
