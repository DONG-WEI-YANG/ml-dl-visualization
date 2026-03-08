import { useState, useMemo } from "react";

const EXAMPLES: Record<string, { src: string[]; tgt: string[]; weights: number[][] }> = {
  translation: {
    src: ["我", "喜歡", "機器", "學習"],
    tgt: ["I", "like", "machine", "learning"],
    weights: [
      [0.85, 0.05, 0.05, 0.05],
      [0.05, 0.80, 0.10, 0.05],
      [0.05, 0.05, 0.75, 0.15],
      [0.03, 0.07, 0.15, 0.75],
    ],
  },
  self_attention: {
    src: ["The", "cat", "sat", "on", "the", "mat"],
    tgt: ["The", "cat", "sat", "on", "the", "mat"],
    weights: [
      [0.10, 0.15, 0.10, 0.10, 0.45, 0.10],
      [0.05, 0.30, 0.20, 0.05, 0.05, 0.35],
      [0.05, 0.25, 0.30, 0.20, 0.05, 0.15],
      [0.05, 0.05, 0.10, 0.30, 0.10, 0.40],
      [0.40, 0.10, 0.05, 0.10, 0.15, 0.20],
      [0.05, 0.30, 0.15, 0.15, 0.10, 0.25],
    ],
  },
};

function HeatCell({ value, row, col }: { value: number; row: number; col: number }) {
  const intensity = Math.round(value * 255);
  const bg = `rgb(${255 - intensity}, ${255 - intensity * 0.3}, ${255 - intensity})`;
  return (
    <div
      className="flex items-center justify-center text-xs font-mono border border-gray-100"
      style={{
        backgroundColor: bg,
        color: value > 0.5 ? "white" : "#374151",
        width: 44, height: 36,
        gridRow: row + 2, gridColumn: col + 2,
      }}
    >
      {value.toFixed(2)}
    </div>
  );
}

export default function AttentionViz() {
  const [example, setExample] = useState("translation");
  const [highlighted, setHighlighted] = useState<number | null>(null);

  const data = useMemo(() => EXAMPLES[example], [example]);

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">注意力機制 Attention</h3>

      <div className="flex gap-2">
        <button onClick={() => setExample("translation")}
          className={`px-3 py-1 rounded text-sm ${example === "translation" ? "bg-blue-500 text-white" : "bg-gray-100"}`}
        >翻譯 (Cross)</button>
        <button onClick={() => setExample("self_attention")}
          className={`px-3 py-1 rounded text-sm ${example === "self_attention" ? "bg-blue-500 text-white" : "bg-gray-100"}`}
        >Self-Attention</button>
      </div>

      <div className="overflow-x-auto">
        <div className="inline-grid gap-0" style={{
          gridTemplateColumns: `80px repeat(${data.src.length}, 44px)`,
          gridTemplateRows: `28px repeat(${data.tgt.length}, 36px)`,
        }}>
          <div />
          {data.src.map((s, i) => (
            <div key={i} className="flex items-center justify-center text-xs font-medium text-blue-600 cursor-pointer hover:bg-blue-50 rounded"
              onMouseEnter={() => setHighlighted(i)} onMouseLeave={() => setHighlighted(null)}
            >{s}</div>
          ))}
          {data.tgt.map((t, i) => (
            <div key={`row-${i}`} className="contents">
              <div className="flex items-center justify-end pr-2 text-xs font-medium text-purple-600">{t}</div>
              {data.weights[i].map((w, j) => (
                <div key={j} className={`transition-opacity ${highlighted !== null && highlighted !== j ? "opacity-30" : ""}`}>
                  <HeatCell value={w} row={i} col={j} />
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-2 text-xs text-gray-500">
        <span>低</span>
        <div className="flex h-3 flex-1 rounded overflow-hidden">
          {Array.from({ length: 20 }, (_, i) => {
            const v = i / 19;
            return <div key={i} className="flex-1" style={{ backgroundColor: `rgb(${255 - v * 255},${255 - v * 76},${255 - v * 255})` }} />;
          })}
        </div>
        <span>高</span>
        <span className="ml-2">← 注意力權重</span>
      </div>

      <div className="bg-gray-50 rounded-lg p-3 text-xs text-gray-600">
        <p><strong>Attention(Q,K,V) = softmax(QK<sup>T</sup>/√d)V</strong></p>
        <p>每一行的權重加總為 1（softmax），顯示目標 token 對各源 token 的關注程度</p>
      </div>
    </div>
  );
}
