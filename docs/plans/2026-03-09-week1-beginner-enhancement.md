# Week 1 初學者友善增強 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 增強 EnvironmentSetupViz.tsx 讓 Week 1 更適合初學者，加入醫療情境引導、手動擬合、過擬合互動體驗

**Architecture:** 單一檔案修改 `platform/frontend/src/components/viz/EnvironmentSetupViz.tsx`，新增 1 個 tab（過擬合體驗）、增強 3 個現有 tab（認識AI、動手預測、名詞對照），保留 2 個不動（ML工作流程、環境建置）。全部純前端計算，不需後端。

**Tech Stack:** React 19 + TypeScript + Recharts + Tailwind CSS 4

---

### Task 1: Tab 結構更新 + 新增過擬合資料與數學工具

**Files:**
- Modify: `platform/frontend/src/components/viz/EnvironmentSetupViz.tsx:14-25`（TabId type + TABS array）
- Modify: `platform/frontend/src/components/viz/EnvironmentSetupViz.tsx:1`（加入 LineChart import）
- Modify: `platform/frontend/src/components/viz/EnvironmentSetupViz.tsx:162-193`（main component routing）

**Step 1: Update TabId and TABS**

```typescript
type TabId = "landscape" | "workflow" | "tryit" | "overfit" | "glossary" | "setup";

const TABS: { id: TabId; label: string }[] = [
  { id: "landscape", label: "認識 AI" },
  { id: "workflow", label: "ML 工作流程" },
  { id: "tryit", label: "動手預測" },
  { id: "overfit", label: "過擬合體驗" },
  { id: "glossary", label: "名詞對照" },
  { id: "setup", label: "環境建置" },
];
```

**Step 2: Add LineChart to Recharts import**

```typescript
import {
  ScatterChart, Scatter, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
```

**Step 3: Add overfit tab routing in main component**

```tsx
{tab === "overfit" && <OverfittingView />}
```

**Step 4: Add polynomial fitting helpers after leastSquares function**

```typescript
/* ──────────────────── Polynomial Fitting ──────────────────── */
const OVERFIT_TRAIN: Point[] = [
  { x: 0.5, y: 2.3 }, { x: 1.2, y: 3.8 }, { x: 1.8, y: 5.1 }, { x: 2.5, y: 4.5 },
  { x: 3.0, y: 6.8 }, { x: 3.5, y: 7.2 }, { x: 4.2, y: 6.9 }, { x: 4.8, y: 8.5 },
  { x: 5.5, y: 9.1 }, { x: 6.0, y: 8.8 }, { x: 6.5, y: 10.2 }, { x: 7.0, y: 10.8 },
];
const OVERFIT_TEST: Point[] = [
  { x: 0.8, y: 2.9 }, { x: 2.0, y: 5.5 }, { x: 3.2, y: 6.5 }, { x: 4.5, y: 7.8 },
  { x: 5.8, y: 9.5 }, { x: 6.8, y: 10.5 },
];

function polyFit(points: Point[], degree: number): number[] {
  const n = points.length;
  const d = Math.min(degree, n - 1);
  // Build normal equations: (V^T V) c = V^T y
  const m = d + 1;
  const A: number[][] = Array.from({ length: m }, () => Array(m + 1).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      for (let k = 0; k < n; k++) A[i][j] += Math.pow(points[k].x, i + j);
    }
    for (let k = 0; k < n; k++) A[i][m] += points[k].y * Math.pow(points[k].x, i);
  }
  // Gaussian elimination with partial pivoting
  for (let col = 0; col < m; col++) {
    let maxRow = col;
    for (let row = col + 1; row < m; row++) {
      if (Math.abs(A[row][col]) > Math.abs(A[maxRow][col])) maxRow = row;
    }
    [A[col], A[maxRow]] = [A[maxRow], A[col]];
    if (Math.abs(A[col][col]) < 1e-12) continue;
    for (let row = col + 1; row < m; row++) {
      const f = A[row][col] / A[col][col];
      for (let j = col; j <= m; j++) A[row][j] -= f * A[col][j];
    }
  }
  // Back substitution
  const c = Array(m).fill(0);
  for (let i = m - 1; i >= 0; i--) {
    c[i] = A[i][m];
    for (let j = i + 1; j < m; j++) c[i] -= A[i][j] * c[j];
    c[i] /= A[i][i] || 1;
  }
  return c;
}

function polyEval(coeffs: number[], x: number): number {
  return coeffs.reduce((sum, c, i) => sum + c * Math.pow(x, i), 0);
}

function calcMSE(coeffs: number[], points: Point[]): number {
  const sum = points.reduce((s, p) => s + Math.pow(p.y - polyEval(coeffs, p.x), 2), 0);
  return sum / points.length;
}
```

**Step 5: Verify** — Run `npm run build` (or `npx tsc --noEmit`) from `platform/frontend/` to confirm no type errors.

---

### Task 2: 增強 LandscapeView — 醫療情境引導 + 逐步揭露

**Files:**
- Modify: `platform/frontend/src/components/viz/EnvironmentSetupViz.tsx` — LandscapeView function (lines 197-248)

**Step 1: Replace entire LandscapeView function**

Key changes:
- Add `revealLevel` state (0→1→2→3) instead of `active` toggle
- Add medical scenario intro box above SVG
- SVG ellipses use CSS opacity transitions; only visible when `revealLevel >= layerIndex`
- "下一層" button to advance reveal
- Info panel auto-shows for current revealed layer

```tsx
function LandscapeView() {
  const [revealLevel, setRevealLevel] = useState(0); // 0=none, 1=AI, 2=ML, 3=DL

  const advance = () => setRevealLevel((l) => Math.min(l + 1, 3));
  const reset = () => setRevealLevel(0);
  const currentInfo = revealLevel > 0 ? LAYERS[revealLevel - 1] : null;

  return (
    <div className="space-y-4">
      {/* Medical scenario intro */}
      <div className="bg-gradient-to-r from-teal-50 to-cyan-50 border border-teal-200 rounded-xl p-4">
        <p className="text-sm text-teal-900 leading-relaxed">
          <span className="font-bold">情境：</span>你是護理師，每天記錄病人的體溫、血壓、心率、血氧…
          這些數據累積下來，<strong>能不能讓電腦自動預測病人的恢復狀況？</strong>
          這就是「機器學習」的核心想法 — 讓機器從資料中學會預測。
        </p>
        <p className="text-xs text-teal-600 mt-2">
          點擊下方按鈕，逐層認識 AI → ML → DL 的關係：
        </p>
      </div>

      {/* Progressive reveal SVG */}
      <div className="flex justify-center">
        <svg viewBox="0 0 400 320" className="w-full max-w-md">
          {/* AI outer */}
          <ellipse cx="200" cy="160" rx="190" ry="150"
            fill={revealLevel >= 1 ? "#e0e7ff" : "#f8fafc"} stroke={revealLevel >= 1 ? "#6366f1" : "#e2e8f0"}
            strokeWidth="2.5" className="transition-all duration-700" />
          <text x="200" y="38" textAnchor="middle" fontSize="15"
            className={`font-bold transition-all duration-700 ${revealLevel >= 1 ? "fill-indigo-700" : "fill-gray-300"}`}
          >人工智慧 AI</text>

          {/* ML middle */}
          <ellipse cx="200" cy="175" rx="140" ry="110"
            fill={revealLevel >= 2 ? "#ede9fe" : "transparent"} stroke={revealLevel >= 2 ? "#8b5cf6" : "transparent"}
            strokeWidth="2.5" className="transition-all duration-700" />
          {revealLevel >= 2 && (
            <text x="200" y="88" textAnchor="middle" fontSize="14"
              className="font-bold fill-violet-700">機器學習 ML</text>
          )}

          {/* DL inner */}
          <ellipse cx="200" cy="195" rx="85" ry="70"
            fill={revealLevel >= 3 ? "#f3e8ff" : "transparent"} stroke={revealLevel >= 3 ? "#a855f7" : "transparent"}
            strokeWidth="2.5" className="transition-all duration-700" />
          {revealLevel >= 3 && (<>
            <text x="200" y="185" textAnchor="middle" fontSize="13"
              className="font-bold fill-purple-700">深度學習 DL</text>
            <text x="200" y="205" textAnchor="middle" fontSize="11"
              className="fill-purple-500">Deep Learning</text>
          </>)}
        </svg>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-3">
        {revealLevel < 3 ? (
          <button onClick={advance}
            className="px-5 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 transition-colors">
            {revealLevel === 0 ? "開始探索" : `展開下一層：${LAYERS[revealLevel].label}`}
          </button>
        ) : (
          <button onClick={reset}
            className="px-4 py-2 bg-gray-100 text-gray-600 rounded-lg text-sm hover:bg-gray-200 transition-colors">
            重新開始
          </button>
        )}
      </div>

      {/* Auto-show info panel */}
      {currentInfo && (
        <div className={`p-4 rounded-xl border-2 ${currentInfo.bg} ${currentInfo.border} transition-all`}>
          <div className="flex items-center gap-2 mb-1">
            <span className={`w-3 h-3 rounded-full`} style={{ backgroundColor: currentInfo.color }} />
            <h4 className="font-bold text-base">{currentInfo.label}</h4>
            <span className="text-xs text-gray-400">第 {revealLevel}/3 層</span>
          </div>
          <p className="text-sm text-gray-700 mb-2">{currentInfo.desc}</p>
          <p className="text-xs font-semibold text-gray-500 mb-1">實際應用：</p>
          <ul className="list-disc list-inside text-sm text-gray-600 space-y-0.5">
            {currentInfo.examples.map((ex) => <li key={ex}>{ex}</li>)}
          </ul>
        </div>
      )}

      {revealLevel === 0 && (
        <p className="text-center text-sm text-gray-400 py-2">-- 點擊「開始探索」逐步了解 AI 的三層結構 --</p>
      )}
    </div>
  );
}
```

**Step 2: Verify** — Run dev server, navigate to Week 1, click through 3 layers.

---

### Task 3: 增強 TryItView — 手動擬合模式

**Files:**
- Modify: `platform/frontend/src/components/viz/EnvironmentSetupViz.tsx` — TryItView function (lines 299-387)

**Step 1: Replace TryItView function**

Key changes:
- Add `manualW` and `manualB` slider states
- Add `mode` state: "manual" | "auto"
- In manual mode: sliders control the line, show MSE
- "顯示最佳解" button transitions to auto mode
- In auto mode: show original least squares + comparison
- Keep click-to-add-points

```tsx
function TryItView() {
  const [points, setPoints] = useState<Point[]>([...INITIAL_POINTS]);
  const [mode, setMode] = useState<"manual" | "auto">("manual");
  const [manualW, setManualW] = useState(5);
  const [manualB, setManualB] = useState(25);

  const { w: bestW, b: bestB } = useMemo(() => leastSquares(points), [points]);

  const activeW = mode === "manual" ? manualW : bestW;
  const activeB = mode === "manual" ? manualB : bestB;

  const lineData = useMemo(() => {
    if (points.length < 2) return [];
    const xs = points.map((p) => p.x);
    const minX = Math.min(...xs, 0);
    const maxX = Math.max(...xs, 12);
    return [
      { x: minX, y: Math.round((activeW * minX + activeB) * 100) / 100 },
      { x: maxX, y: Math.round((activeW * maxX + activeB) * 100) / 100 },
    ];
  }, [points, activeW, activeB]);

  const manualMSE = useMemo(() => {
    if (points.length === 0) return 0;
    return points.reduce((s, p) => s + Math.pow(p.y - (manualW * p.x + manualB), 2), 0) / points.length;
  }, [points, manualW, manualB]);

  const bestMSE = useMemo(() => {
    if (points.length === 0) return 0;
    return points.reduce((s, p) => s + Math.pow(p.y - (bestW * p.x + bestB), 2), 0) / points.length;
  }, [points, bestW, bestB]);

  const handleChartClick = useCallback(
    (e: { xValue?: number; yValue?: number } | null | undefined) => {
      if (!e || e.xValue == null || e.yValue == null) return;
      const x = Math.round(e.xValue * 10) / 10;
      const y = Math.round(e.yValue * 10) / 10;
      if (x >= 0 && x <= 14 && y >= 0 && y <= 110) {
        setPoints((prev) => [...prev, { x, y }]);
        setMode("manual");
      }
    }, []);

  const removeLastPoint = () => { setPoints((prev) => prev.slice(0, -1)); setMode("manual"); };
  const resetAll = () => { setPoints([...INITIAL_POINTS]); setMode("manual"); setManualW(5); setManualB(25); };

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">
        {mode === "manual"
          ? "用下方滑桿調整直線的斜率和截距，試著讓直線盡可能接近所有資料點！"
          : "這是數學最佳解（最小平方法）。比較看看你的手動結果！"}
      </p>

      {/* Chart */}
      <div className="bg-white border border-gray-200 rounded-xl p-2">
        <ResponsiveContainer width="100%" height={260}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}
            onClick={(state) => {
              if (state && state.xValue !== undefined && state.yValue !== undefined)
                handleChartClick({ xValue: state.xValue as number, yValue: state.yValue as number });
            }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" dataKey="x" domain={[0, 14]} name="讀書時間 (hr)"
              label={{ value: "讀書時間 (hr)", position: "insideBottom", offset: -5, fontSize: 12 }} />
            <YAxis type="number" dataKey="y" domain={[0, 110]} name="考試分數"
              label={{ value: "分數", angle: -90, position: "insideLeft", fontSize: 12 }} />
            <Tooltip formatter={(val: number) => Math.round(val * 10) / 10} labelFormatter={() => ""} />
            <Scatter data={points} fill="#6366f1" r={5} name="資料點" />
            {lineData.length === 2 && (
              <Scatter data={lineData} fill="none"
                line={{ stroke: mode === "manual" ? "#f59e0b" : "#ef4444", strokeWidth: 2.5 }}
                lineType="joint" shape={() => null} name="擬合線" legendType="none" />
            )}
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Manual sliders */}
      {mode === "manual" && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 space-y-3">
          <div className="flex items-center gap-3">
            <label className="text-sm font-medium text-amber-800 w-20">斜率 (w)</label>
            <input type="range" min={-5} max={15} step={0.5} value={manualW}
              onChange={(e) => setManualW(Number(e.target.value))}
              className="flex-1 accent-amber-500" />
            <span className="text-sm font-mono w-10 text-right text-amber-700">{manualW}</span>
          </div>
          <div className="flex items-center gap-3">
            <label className="text-sm font-medium text-amber-800 w-20">截距 (b)</label>
            <input type="range" min={-20} max={60} step={1} value={manualB}
              onChange={(e) => setManualB(Number(e.target.value))}
              className="flex-1 accent-amber-500" />
            <span className="text-sm font-mono w-10 text-right text-amber-700">{manualB}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm font-mono text-amber-800">
              y = {manualW}x {manualB >= 0 ? "+" : "−"} {Math.abs(manualB)}
              <span className="ml-3 text-amber-600">MSE = {manualMSE.toFixed(1)}</span>
            </span>
            <button onClick={() => setMode("auto")}
              className="px-4 py-1.5 bg-amber-500 text-white rounded-lg text-sm font-medium hover:bg-amber-600 transition-colors">
              顯示最佳解
            </button>
          </div>
        </div>
      )}

      {/* Auto result comparison */}
      {mode === "auto" && (
        <div className="bg-green-50 border border-green-200 rounded-xl p-4 space-y-2">
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="bg-amber-50 rounded-lg p-2 text-center">
              <p className="text-xs text-amber-600 mb-0.5">你的手動結果</p>
              <p className="font-mono font-semibold text-amber-800">y = {manualW}x {manualB >= 0 ? "+" : "−"} {Math.abs(manualB)}</p>
              <p className="font-mono text-amber-700">MSE = {manualMSE.toFixed(1)}</p>
            </div>
            <div className="bg-green-100 rounded-lg p-2 text-center">
              <p className="text-xs text-green-600 mb-0.5">最小平方法最佳解</p>
              <p className="font-mono font-semibold text-green-800">y = {bestW}x {bestB >= 0 ? "+" : "−"} {Math.abs(bestB)}</p>
              <p className="font-mono text-green-700">MSE = {bestMSE.toFixed(1)}</p>
            </div>
          </div>
          <button onClick={() => setMode("manual")}
            className="w-full text-center text-xs text-green-600 hover:text-green-800 transition-colors mt-1">
            ← 回到手動模式再試試
          </button>
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-xs text-gray-500">({points.length} 個資料點，點擊圖表可新增)</span>
        <div className="flex gap-2 ml-auto">
          <button onClick={removeLastPoint} disabled={points.length === 0}
            className="px-3 py-1.5 text-xs rounded-lg bg-gray-100 hover:bg-gray-200 disabled:opacity-40 transition-colors">
            移除最後一點
          </button>
          <button onClick={resetAll}
            className="px-3 py-1.5 text-xs rounded-lg bg-indigo-100 hover:bg-indigo-200 text-indigo-700 transition-colors">
            全部重置
          </button>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-100 rounded-lg p-3 text-xs text-blue-800">
        <strong>想一想：</strong> 你的手動 MSE 和最佳解差多少？試著加入一個離群點（如 x=2, y=100），
        看最佳解怎麼被拉偏 — 這就是「異常值 (outlier)」的影響。
      </div>
    </div>
  );
}
```

**Step 2: Verify** — Dev server, try sliders, click "顯示最佳解", compare MSE.

---

### Task 4: 新增 OverfittingView — 過擬合體驗

**Files:**
- Modify: `platform/frontend/src/components/viz/EnvironmentSetupViz.tsx` — Add new function before GlossaryView

**Step 1: Add OverfittingView function**

```tsx
/* ══════════════════ View 4: Overfitting Experience ══════════════════ */
function OverfittingView() {
  const [degree, setDegree] = useState(1);

  const coeffs = useMemo(() => polyFit(OVERFIT_TRAIN, degree), [degree]);

  const curveData = useMemo(() => {
    const pts: { x: number; y: number }[] = [];
    for (let x = 0; x <= 7.5; x += 0.1) {
      pts.push({ x: Math.round(x * 10) / 10, y: Math.round(polyEval(coeffs, x) * 100) / 100 });
    }
    return pts;
  }, [coeffs]);

  const trainMSE = useMemo(() => calcMSE(coeffs, OVERFIT_TRAIN), [coeffs]);
  const testMSE = useMemo(() => calcMSE(coeffs, OVERFIT_TEST), [coeffs]);

  const errorCurve = useMemo(() => {
    return Array.from({ length: 10 }, (_, i) => {
      const d = i + 1;
      const c = polyFit(OVERFIT_TRAIN, d);
      return { degree: d, train: Math.round(calcMSE(c, OVERFIT_TRAIN) * 100) / 100,
               test: Math.round(calcMSE(c, OVERFIT_TEST) * 100) / 100 };
    });
  }, []);

  const status = degree <= 2 ? "underfitting" : degree <= 4 ? "justright" : "overfitting";
  const statusLabel = { underfitting: "欠擬合", justright: "適當擬合", overfitting: "過擬合" }[status];
  const statusColor = { underfitting: "text-blue-600 bg-blue-50 border-blue-200",
    justright: "text-green-600 bg-green-50 border-green-200",
    overfitting: "text-red-600 bg-red-50 border-red-200" }[status];

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">
        拖動滑桿調整多項式次數，觀察模型從「太簡單」到「太複雜」的變化。
        <span className="text-gray-400">（藍點 = 訓練資料，橘點 = 測試資料）</span>
      </p>

      {/* Scatter + curve */}
      <div className="bg-white border border-gray-200 rounded-xl p-2">
        <ResponsiveContainer width="100%" height={260}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" dataKey="x" domain={[0, 7.5]}
              label={{ value: "特徵 x", position: "insideBottom", offset: -5, fontSize: 12 }} />
            <YAxis type="number" dataKey="y" domain={[-2, 16]}
              label={{ value: "目標 y", angle: -90, position: "insideLeft", fontSize: 12 }} />
            <Tooltip formatter={(val: number) => Math.round(val * 100) / 100} />
            <Scatter data={OVERFIT_TRAIN} fill="#6366f1" r={5} name="訓練資料" />
            <Scatter data={OVERFIT_TEST} fill="#f59e0b" r={5} name="測試資料" shape="diamond" />
            <Scatter data={curveData} fill="none"
              line={{ stroke: "#ef4444", strokeWidth: 2 }}
              lineType="joint" shape={() => null} name="擬合曲線" legendType="none" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Degree slider + status */}
      <div className="flex items-center gap-4">
        <label className="text-sm font-medium text-gray-700 whitespace-nowrap">多項式次數</label>
        <input type="range" min={1} max={10} step={1} value={degree}
          onChange={(e) => setDegree(Number(e.target.value))}
          className="flex-1 accent-indigo-500" />
        <span className="text-lg font-bold font-mono text-indigo-700 w-8 text-center">{degree}</span>
        <span className={`text-xs font-bold px-2.5 py-1 rounded-full border ${statusColor}`}>
          {statusLabel}
        </span>
      </div>

      {/* MSE comparison */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-indigo-50 rounded-lg p-3 text-center">
          <p className="text-xs text-indigo-500 mb-0.5">訓練誤差 (Train MSE)</p>
          <p className="text-xl font-bold font-mono text-indigo-700">{trainMSE.toFixed(2)}</p>
        </div>
        <div className="bg-amber-50 rounded-lg p-3 text-center">
          <p className="text-xs text-amber-500 mb-0.5">測試誤差 (Test MSE)</p>
          <p className="text-xl font-bold font-mono text-amber-700">{testMSE.toFixed(2)}</p>
        </div>
      </div>

      {/* Error curves */}
      <div className="bg-white border border-gray-200 rounded-xl p-2">
        <p className="text-xs text-gray-500 ml-2 mb-1">誤差隨模型複雜度變化：</p>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={errorCurve} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="degree" type="number" domain={[1, 10]}
              label={{ value: "多項式次數", position: "insideBottom", offset: -5, fontSize: 12 }} />
            <YAxis label={{ value: "MSE", angle: -90, position: "insideLeft", fontSize: 12 }} />
            <Tooltip />
            <Legend verticalAlign="top" height={28} />
            <Line type="monotone" dataKey="train" name="訓練誤差" stroke="#6366f1" strokeWidth={2} dot={{ r: 3 }} />
            <Line type="monotone" dataKey="test" name="測試誤差" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Explanation */}
      <div className={`rounded-xl p-4 border text-sm ${
        status === "underfitting" ? "bg-blue-50 border-blue-200 text-blue-800" :
        status === "justright" ? "bg-green-50 border-green-200 text-green-800" :
        "bg-red-50 border-red-200 text-red-800"
      }`}>
        {status === "underfitting" && (<>
          <strong>欠擬合 (Underfitting)：</strong>模型太簡單，連訓練資料的趨勢都抓不到。
          就像只用一條直線去描述曲折的關係 — 訓練和測試誤差都很高。
        </>)}
        {status === "justright" && (<>
          <strong>適當擬合：</strong>模型複雜度剛好！能夠抓住資料的真正規律，
          在訓練和測試資料上都有不錯的表現。這是我們追求的目標。
        </>)}
        {status === "overfitting" && (<>
          <strong>過擬合 (Overfitting)：</strong>模型太複雜，把訓練資料的雜訊也「背」了下來。
          訓練誤差很低但測試誤差反而升高 — 就像死記考古題卻不會解新題目。
          <span className="block mt-1">
            <strong>正則化 (Regularization)</strong> 是解決辦法：對模型加上「懲罰」，
            限制它的複雜度，讓它學到通用規律而非記住雜訊。
          </span>
        </>)}
      </div>
    </div>
  );
}
```

**Step 2: Verify** — Slider from 1→10, observe curve behavior and MSE changes.

---

### Task 5: 增強 GlossaryView — 預設初學者分類

**Files:**
- Modify: `platform/frontend/src/components/viz/EnvironmentSetupViz.tsx` — GlossaryView function (lines 390-449)

**Step 1: Change default catFilter to beginner categories + add toggle**

Key changes:
- Default `catFilter` to `"基礎"` (a special value showing 基礎概念 + 資料處理)
- Add `showAdvanced` toggle state
- When `showAdvanced` is false, filter out 深度學習 category

```tsx
function GlossaryView() {
  const [search, setSearch] = useState("");
  const [catFilter, setCatFilter] = useState<string>("基礎");
  const [showAdvanced, setShowAdvanced] = useState(false);

  const BEGINNER_CATS = ["基礎概念", "資料處理"];

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return TERMS.filter((t) => {
      // Category filter
      if (catFilter === "基礎") {
        if (!BEGINNER_CATS.includes(t.cat)) return false;
      } else if (catFilter !== "全部") {
        if (t.cat !== catFilter) return false;
      }
      // Hide advanced unless toggled
      if (!showAdvanced && catFilter === "全部" && t.cat === "深度學習") return false;
      // Text search
      if (!q) return true;
      return t.zh.includes(q) || t.en.toLowerCase().includes(q) || t.desc.includes(q);
    });
  }, [search, catFilter, showAdvanced]);

  return (
    <div className="space-y-3">
      {/* Search & filter */}
      <div className="flex flex-wrap gap-2">
        <input type="text" value={search} onChange={(e) => setSearch(e.target.value)}
          placeholder="搜尋名詞 (中/英文)..."
          className="flex-1 min-w-[180px] px-3 py-1.5 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-300" />
        <select value={catFilter} onChange={(e) => setCatFilter(e.target.value)}
          className="px-2 py-1.5 text-sm border border-gray-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-indigo-300">
          <option value="基礎">初學必備</option>
          <option value="全部">全部分類</option>
          {CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>

      <div className="flex items-center justify-between">
        <p className="text-xs text-gray-400">共 {filtered.length} 筆</p>
        {(catFilter === "全部") && (
          <label className="flex items-center gap-1.5 text-xs text-gray-500 cursor-pointer">
            <input type="checkbox" checked={showAdvanced} onChange={(e) => setShowAdvanced(e.target.checked)}
              className="accent-indigo-500 w-3.5 h-3.5" />
            顯示進階名詞（CNN、RNN 等）
          </label>
        )}
      </div>

      {/* Table — same as before */}
      <div className="overflow-auto max-h-[420px] border border-gray-200 rounded-xl">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              <th className="px-3 py-2 text-left font-semibold text-gray-600">中文</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-600">English</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-600">說明</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-600 hidden sm:table-cell">分類</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((t, i) => (
              <tr key={t.zh} className={i % 2 === 0 ? "bg-white" : "bg-gray-50/60"}>
                <td className="px-3 py-2 font-medium text-gray-800 whitespace-nowrap">{t.zh}</td>
                <td className="px-3 py-2 text-gray-600 whitespace-nowrap">{t.en}</td>
                <td className="px-3 py-2 text-gray-500">{t.desc}</td>
                <td className="px-3 py-2 hidden sm:table-cell">
                  <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-600 whitespace-nowrap">{t.cat}</span>
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr><td colSpan={4} className="px-3 py-6 text-center text-gray-400">找不到符合的名詞</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

**Step 2: Verify** — Default shows ~16 beginner terms, select "全部" shows more, toggle shows DL terms.

---

### Task 6: Visual verification

**Step 1:** Run `cd platform/frontend && npm run build` to confirm no compile errors.

**Step 2:** Start dev server and verify all 6 tabs:
- 認識 AI: Medical intro visible, progressive reveal works
- ML 工作流程: Unchanged, still works
- 動手預測: Manual sliders → show best → comparison
- 過擬合體驗: Degree slider 1→10, curves + MSE update
- 名詞對照: Default "初學必備" shows ~16 terms
- 環境建置: Unchanged, still works
