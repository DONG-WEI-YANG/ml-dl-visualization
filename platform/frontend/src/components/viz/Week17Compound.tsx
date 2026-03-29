import { useState, lazy, Suspense } from "react";
import TokenizationViz from "./TokenizationViz";

const EmbeddingSpaceViz = lazy(() => import("./EmbeddingSpaceViz"));

type Mode = "tokenization" | "embedding";

export default function Week17Compound() {
  const [mode, setMode] = useState<Mode>("tokenization");

  return (
    <div className="space-y-4">
      <div className="inline-flex rounded-lg border border-gray-200 overflow-hidden text-sm">
        <button
          onClick={() => setMode("tokenization")}
          className={`px-4 py-1.5 font-medium transition-colors ${
            mode === "tokenization"
              ? "bg-blue-500 text-white"
              : "bg-white text-gray-600 hover:bg-gray-50"
          }`}
        >
          Token 分詞
        </button>
        <button
          onClick={() => setMode("embedding")}
          className={`px-4 py-1.5 font-medium transition-colors ${
            mode === "embedding"
              ? "bg-blue-500 text-white"
              : "bg-white text-gray-600 hover:bg-gray-50"
          }`}
        >
          嵌入空間
        </button>
      </div>

      {mode === "tokenization" && <TokenizationViz />}
      {mode === "embedding" && (
        <Suspense
          fallback={
            <div className="text-gray-400 text-sm p-4">
              載入嵌入空間視覺化...
            </div>
          }
        >
          <EmbeddingSpaceViz />
        </Suspense>
      )}
    </div>
  );
}
