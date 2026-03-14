import { useState, useMemo, useCallback } from "react";
import { Line } from "@react-three/drei";
import Scene3D from "./Scene3D";
import Axis3D from "./Axis3D";
import Tooltip3D from "./Tooltip3D";
import embeddingData from "../data/embedding-3d.json";
import type { Vec3 } from "./types";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface EmbeddingSpace3DProps {
  noise: number;
  onNoiseChange: (noise: number) => void;
}

type Method = "pca" | "tsne" | "umap";

interface WordEntry {
  text: string;
  category: string;
  pca: [number, number, number];
  tsne: [number, number, number];
  umap: [number, number, number];
}

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const CATEGORY_COLORS: Record<string, string> = {
  "動物": "#3b82f6",
  "交通": "#10b981",
  "食物": "#f59e0b",
  "科技": "#ef4444",
  "自然": "#8b5cf6",
};

const METHOD_LABELS: Record<Method, string> = {
  pca: "PCA",
  tsne: "t-SNE",
  umap: "UMAP",
};

const words = embeddingData.words as WordEntry[];

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function dist3(a: Vec3, b: Vec3): number {
  return Math.sqrt(
    (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2,
  );
}

/** Deterministic noise based on word index + axis, so it doesn't change every render. */
function seededNoise(seed: number): number {
  // Simple LCG
  const x = Math.sin(seed * 9301 + 49297) * 233280;
  return x - Math.floor(x);
}

function addNoise(coord: Vec3, noiseLevel: number, wordIdx: number): Vec3 {
  return [
    coord[0] + (seededNoise(wordIdx * 3 + 0) - 0.5) * 2 * noiseLevel,
    coord[1] + (seededNoise(wordIdx * 3 + 1) - 0.5) * 2 * noiseLevel,
    coord[2] + (seededNoise(wordIdx * 3 + 2) - 0.5) * 2 * noiseLevel,
  ];
}

/* ------------------------------------------------------------------ */
/*  Word point (inside Canvas)                                         */
/* ------------------------------------------------------------------ */

function WordPoint({
  position,
  color,
  highlighted,
  onHover,
  onLeave,
  onClick,
}: {
  position: Vec3;
  color: string;
  highlighted: boolean;
  onHover: () => void;
  onLeave: () => void;
  onClick: () => void;
}) {
  return (
    <mesh
      position={position}
      onPointerOver={onHover}
      onPointerOut={onLeave}
      onClick={onClick}
    >
      <sphereGeometry args={[highlighted ? 0.15 : 0.1, 12, 12]} />
      <meshStandardMaterial
        color={color}
        emissive={highlighted ? color : "#000000"}
        emissiveIntensity={highlighted ? 0.4 : 0}
      />
    </mesh>
  );
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

export default function EmbeddingSpace3D({
  noise,
  onNoiseChange,
}: EmbeddingSpace3DProps) {
  const [method, setMethod] = useState<Method>("pca");
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  /** Compute positions with noise applied. */
  const positions: Vec3[] = useMemo(() => {
    return words.map((w, i) => addNoise(w[method], noise, i));
  }, [method, noise]);

  /** Find k=3 nearest neighbours of the selected point. */
  const neighbors: number[] = useMemo(() => {
    if (selectedIdx == null) return [];
    const origin = positions[selectedIdx];
    const dists = positions
      .map((p, i) => ({ i, d: dist3(origin, p) }))
      .filter(({ i }) => i !== selectedIdx)
      .sort((a, b) => a.d - b.d);
    return dists.slice(0, 3).map(({ i }) => i);
  }, [selectedIdx, positions]);

  const highlightedSet = useMemo(() => {
    const s = new Set<number>(neighbors);
    if (selectedIdx != null) s.add(selectedIdx);
    return s;
  }, [selectedIdx, neighbors]);

  const handleClick = useCallback(
    (idx: number) => {
      setSelectedIdx((prev) => (prev === idx ? null : idx));
    },
    [],
  );

  const activeIdx = hoveredIdx ?? selectedIdx;
  const tooltipPos: Vec3 =
    activeIdx != null ? positions[activeIdx] : [0, 0, 0];

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">
        3D 詞嵌入空間 Embedding Space
      </h3>

      {/* Method selector */}
      <div className="flex gap-2">
        {(["pca", "tsne", "umap"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMethod(m)}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              method === m
                ? "bg-blue-500 text-white"
                : "bg-gray-100 text-gray-700 hover:bg-gray-200"
            }`}
          >
            {METHOD_LABELS[m]}
          </button>
        ))}
      </div>

      {/* Noise slider */}
      <div>
        <label className="block text-sm text-gray-600 mb-1">
          雜訊 Noise: <strong>{noise.toFixed(2)}</strong>
        </label>
        <input
          type="range"
          min="0"
          max="2"
          step="0.05"
          value={noise}
          onChange={(e) => onNoiseChange(+e.target.value)}
          className="w-full"
        />
      </div>

      {/* 3D canvas */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <Scene3D cameraPosition={[5, 4, 5]} showGrid>
          <Axis3D
            labels={{ x: "Dim 1", y: "Dim 2", z: "Dim 3" }}
            range={{ x: [-5, 5], y: [-5, 5], z: [-5, 5] }}
            tickCount={5}
          />

          {/* Word points */}
          {words.map((w, i) => (
            <WordPoint
              key={`${w.text}-${method}`}
              position={positions[i]}
              color={CATEGORY_COLORS[w.category] ?? "#94a3b8"}
              highlighted={highlightedSet.has(i)}
              onHover={() => setHoveredIdx(i)}
              onLeave={() => setHoveredIdx(null)}
              onClick={() => handleClick(i)}
            />
          ))}

          {/* Neighbor connecting lines */}
          {selectedIdx != null &&
            neighbors.map((ni) => (
              <Line
                key={`line-${selectedIdx}-${ni}`}
                points={[positions[selectedIdx], positions[ni]]}
                color="#ffffff"
                lineWidth={1.5}
                dashed
                dashSize={0.15}
                gapSize={0.1}
              />
            ))}

          {/* Tooltip */}
          <Tooltip3D position={tooltipPos} visible={activeIdx != null}>
            {activeIdx != null && (
              <div>
                <p className="font-bold">{words[activeIdx].text}</p>
                <p style={{ color: CATEGORY_COLORS[words[activeIdx].category] }}>
                  {words[activeIdx].category}
                </p>
                <p className="text-gray-400 text-[10px]">
                  ({positions[activeIdx][0].toFixed(1)},{" "}
                  {positions[activeIdx][1].toFixed(1)},{" "}
                  {positions[activeIdx][2].toFixed(1)})
                </p>
              </div>
            )}
          </Tooltip3D>
        </Scene3D>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 text-sm text-gray-600">
        {Object.entries(CATEGORY_COLORS).map(([cat, color]) => (
          <span key={cat} className="flex items-center gap-1">
            <span
              className="inline-block w-3 h-3 rounded-full"
              style={{ backgroundColor: color }}
            />
            {cat}
          </span>
        ))}
      </div>
      <p className="text-xs text-gray-400">
        點擊任一詞彙顯示 k=3 最近鄰 | Click a word to show 3 nearest neighbors
      </p>
    </div>
  );
}
