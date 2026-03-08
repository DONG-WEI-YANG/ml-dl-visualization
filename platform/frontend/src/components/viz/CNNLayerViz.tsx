import { useState, useMemo } from "react";

const KERNELS: Record<string, number[][]> = {
  edge_h: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
  edge_v: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
  sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
  blur: [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
};

function generateImage(seed: number): number[][] {
  const img: number[][] = [];
  for (let i = 0; i < 8; i++) {
    const row: number[] = [];
    for (let j = 0; j < 8; j++) {
      // Simple pattern: diagonal stripe
      row.push(Math.abs(i - j) < 2 ? 200 : (i + j + seed) % 3 === 0 ? 100 : 50);
    }
    img.push(row);
  }
  return img;
}

function convolve(img: number[][], kernel: number[][]): number[][] {
  const h = img.length - 2, w = img[0].length - 2;
  const kSum = kernel.flat().reduce((a, b) => a + Math.abs(b), 0) || 1;
  const out: number[][] = [];
  for (let i = 0; i < h; i++) {
    const row: number[] = [];
    for (let j = 0; j < w; j++) {
      let sum = 0;
      for (let ki = 0; ki < 3; ki++)
        for (let kj = 0; kj < 3; kj++)
          sum += img[i + ki][j + kj] * kernel[ki][kj];
      row.push(Math.max(0, Math.min(255, Math.round(sum / (kSum === 9 ? 9 : 1) + 128))));
    }
    out.push(row);
  }
  return out;
}

function maxPool(img: number[][]): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < img.length - 1; i += 2) {
    const row: number[] = [];
    for (let j = 0; j < img[0].length - 1; j += 2) {
      row.push(Math.max(img[i][j], img[i][j + 1], img[i + 1][j], img[i + 1][j + 1]));
    }
    out.push(row);
  }
  return out;
}

function Grid({ data, size, label }: { data: number[][]; size: number; label: string }) {
  const cellSize = size / data.length;
  return (
    <div className="text-center">
      <svg width={size} height={size} className="border border-gray-300 rounded">
        {data.map((row, i) => row.map((v, j) => (
          <rect key={`${i}-${j}`} x={j * cellSize} y={i * cellSize}
            width={cellSize} height={cellSize}
            fill={`rgb(${v},${v},${v})`} stroke="#e5e7eb" strokeWidth={0.5} />
        )))}
      </svg>
      <p className="text-xs text-gray-500 mt-1">{label}</p>
      <p className="text-xs text-gray-400">{data.length}x{data[0].length}</p>
    </div>
  );
}

export default function CNNLayerViz() {
  const [kernelName, setKernelName] = useState("edge_h");
  const [showPool, setShowPool] = useState(false);

  const image = useMemo(() => generateImage(42), []);
  const kernel = KERNELS[kernelName];
  const convolved = useMemo(() => convolve(image, kernel), [image, kernel]);
  const pooled = useMemo(() => maxPool(convolved), [convolved]);

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">CNN 卷積層視覺化</h3>

      <div className="flex gap-2 flex-wrap">
        {Object.entries(KERNELS).map(([name]) => (
          <button key={name} onClick={() => setKernelName(name)}
            className={`px-3 py-1 rounded text-sm ${kernelName === name ? "bg-blue-500 text-white" : "bg-gray-100"}`}
          >{name === "edge_h" ? "水平邊緣" : name === "edge_v" ? "垂直邊緣" : name === "sharpen" ? "銳化" : "模糊"}</button>
        ))}
      </div>

      <div className="flex items-center gap-2 justify-center flex-wrap">
        <Grid data={image} size={120} label="輸入 Input" />
        <span className="text-gray-400 text-lg">*</span>
        <div className="text-center">
          <div className="grid grid-cols-3 gap-0.5 mx-auto w-fit">
            {kernel.map((row, i) => row.map((v, j) => (
              <div key={`${i}-${j}`}
                className={`w-7 h-7 flex items-center justify-center text-xs font-mono rounded ${
                  v > 0 ? "bg-blue-100 text-blue-700" : v < 0 ? "bg-red-100 text-red-700" : "bg-gray-100 text-gray-500"
                }`}>{v}</div>
            )))}
          </div>
          <p className="text-xs text-gray-500 mt-1">Kernel 3x3</p>
        </div>
        <span className="text-gray-400 text-lg">=</span>
        <Grid data={convolved} size={100} label="卷積結果" />
        {showPool && (
          <>
            <span className="text-gray-400 text-lg">→</span>
            <Grid data={pooled} size={80} label="Max Pool" />
          </>
        )}
      </div>

      <label className="flex items-center gap-2 text-sm text-gray-600">
        <input type="checkbox" checked={showPool} onChange={(e) => setShowPool(e.target.checked)} />
        顯示 Max Pooling (2x2)
      </label>

      <div className="bg-gray-50 rounded-lg p-3 text-xs text-gray-600">
        <p><strong>卷積操作：</strong>Kernel 滑過輸入矩陣，計算逐元素相乘之和</p>
        <p><strong>Max Pooling：</strong>取 2x2 區域最大值，降低空間維度</p>
      </div>
    </div>
  );
}
