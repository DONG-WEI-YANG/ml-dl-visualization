import { useState, useEffect, useMemo, useCallback } from "react";
import * as THREE from "three";
import Scene3D from "./Scene3D";
import Axis3D from "./Axis3D";
import Tooltip3D from "./Tooltip3D";
import { fetchAPI } from "../../lib/api";
import type { Vec3 } from "./types";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface DecisionBoundary3DProps {
  modelType: string;
  C: number;
  kernel: string;
  onModelTypeChange: (type: string) => void;
  onCChange: (c: number) => void;
  onKernelChange: (k: string) => void;
}

interface DBResponse {
  X: number[][];
  y: number[];
  accuracy: number;
  mesh_vertices: number[][];
  mesh_faces: number[][];
}

/* ------------------------------------------------------------------ */
/*  Data generation (3 features)                                       */
/* ------------------------------------------------------------------ */

function generateClassification3D(n: number = 120) {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const cls = Math.random() > 0.5 ? 1 : 0;
    const x1 = (Math.random() - 0.5) * 4 + (cls === 1 ? 1 : -1);
    const x2 = (Math.random() - 0.5) * 4 + (cls === 1 ? 1 : -1);
    const x3 = (Math.random() - 0.5) * 4 + (cls === 1 ? 0.8 : -0.8);
    X.push([x1, x2, x3]);
    y.push(cls);
  }
  return { X, y };
}

/* ------------------------------------------------------------------ */
/*  Scatter points (inside Canvas)                                     */
/* ------------------------------------------------------------------ */

function ScatterPoint({
  position,
  cls,
  onHover,
  onLeave,
}: {
  position: Vec3;
  cls: number;
  onHover: () => void;
  onLeave: () => void;
}) {
  const color = cls === 0 ? "#3b82f6" : "#ef4444";
  return (
    <mesh
      position={position}
      onPointerOver={onHover}
      onPointerOut={onLeave}
    >
      {cls === 0 ? (
        <sphereGeometry args={[0.08, 12, 12]} />
      ) : (
        <icosahedronGeometry args={[0.1, 0]} />
      )}
      <meshStandardMaterial color={color} />
    </mesh>
  );
}

/* ------------------------------------------------------------------ */
/*  Decision surface mesh (inside Canvas)                              */
/* ------------------------------------------------------------------ */

function DecisionSurface({
  vertices,
  faces,
}: {
  vertices: number[][];
  faces: number[][];
}) {
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(vertices.flat());
    const indices: number[] = [];
    for (const f of faces) {
      if (f.length === 3) {
        indices.push(f[0], f[1], f[2]);
      } else if (f.length === 4) {
        indices.push(f[0], f[1], f[2], f[0], f[2], f[3]);
      }
    }
    geo.setIndex(indices);
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geo.computeVertexNormals();
    return geo;
  }, [vertices, faces]);

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial
        color="#8b5cf6"
        transparent
        opacity={0.35}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

export default function DecisionBoundary3D({
  modelType,
  C,
  kernel,
  onModelTypeChange,
  onCChange,
  onKernelChange,
}: DecisionBoundary3DProps) {
  const [result, setResult] = useState<DBResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    try {
      const { X, y } = generateClassification3D();
      const data = await fetchAPI<DBResponse>("/api/models/decision-boundary", {
        X,
        y,
        model_type: modelType,
        C,
        kernel,
        n_features: 3,
      });
      setResult(data);
    } catch (err) {
      console.error("Failed to fetch decision boundary:", err);
    }
    setLoading(false);
  }, [modelType, C, kernel]);

  useEffect(() => {
    run();
  }, [run]);

  const tooltipPos: Vec3 = useMemo(() => {
    if (hoveredIdx == null || !result) return [0, 0, 0];
    const p = result.X[hoveredIdx];
    return [p[0], p[1], p[2]];
  }, [hoveredIdx, result]);

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">
        3D 決策邊界 Decision Boundary
      </h3>

      {/* Controls */}
      <div className="grid grid-cols-3 gap-3">
        <div>
          <label className="block text-sm text-gray-600 mb-1">模型</label>
          <select
            value={modelType}
            onChange={(e) => onModelTypeChange(e.target.value)}
            className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm"
          >
            <option value="logistic">Logistic Regression</option>
            <option value="svm">SVM</option>
          </select>
        </div>
        <div>
          <label className="block text-sm text-gray-600 mb-1">C = {C.toFixed(2)}</label>
          <input
            type="range"
            min="0.01"
            max="100"
            step="0.1"
            value={C}
            onChange={(e) => onCChange(+e.target.value)}
            className="w-full"
          />
        </div>
        {modelType === "svm" && (
          <div>
            <label className="block text-sm text-gray-600 mb-1">
              核函數 Kernel
            </label>
            <select
              value={kernel}
              onChange={(e) => onKernelChange(e.target.value)}
              className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm"
            >
              <option value="linear">Linear</option>
              <option value="rbf">RBF</option>
              <option value="poly">Polynomial</option>
            </select>
          </div>
        )}
      </div>

      <button
        onClick={run}
        disabled={loading}
        className="px-4 py-2 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600 disabled:opacity-50"
      >
        {loading ? "訓練中..." : "訓練模型"}
      </button>

      {result && (
        <p className="text-sm text-gray-600">
          準確率 Accuracy: <strong>{(result.accuracy * 100).toFixed(1)}%</strong>
        </p>
      )}

      {/* 3D canvas */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        {loading && (
          <div className="flex items-center justify-center h-[400px] bg-gray-50">
            <p className="text-gray-500 text-sm">訓練模型中...</p>
          </div>
        )}
        {!loading && result && (
          <Scene3D cameraPosition={[5, 5, 5]} showGrid>
            <Axis3D
              labels={{ x: "X₁", y: "X₂", z: "X₃" }}
              range={{ x: [-4, 4], y: [-4, 4], z: [-4, 4] }}
            />

            {/* Scatter points */}
            {result.X.map((p, i) => (
              <ScatterPoint
                key={i}
                position={[p[0], p[1], p[2]]}
                cls={result.y[i]}
                onHover={() => setHoveredIdx(i)}
                onLeave={() => setHoveredIdx(null)}
              />
            ))}

            {/* Decision surface */}
            {result.mesh_vertices && result.mesh_faces && (
              <DecisionSurface
                vertices={result.mesh_vertices}
                faces={result.mesh_faces}
              />
            )}

            {/* Tooltip */}
            <Tooltip3D position={tooltipPos} visible={hoveredIdx != null}>
              {hoveredIdx != null && result && (
                <div>
                  <p>
                    座標: ({result.X[hoveredIdx][0].toFixed(2)},{" "}
                    {result.X[hoveredIdx][1].toFixed(2)},{" "}
                    {result.X[hoveredIdx][2].toFixed(2)})
                  </p>
                  <p>
                    類別 Class:{" "}
                    <span
                      style={{
                        color: result.y[hoveredIdx] === 0 ? "#3b82f6" : "#ef4444",
                        fontWeight: "bold",
                      }}
                    >
                      {result.y[hoveredIdx]}
                    </span>
                  </p>
                </div>
              )}
            </Tooltip3D>
          </Scene3D>
        )}
      </div>

      {/* Legend */}
      <div className="flex gap-4 text-sm text-gray-600">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-full bg-blue-500" />
          Class 0 (球形 Sphere)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 bg-red-500" style={{ clipPath: "polygon(50% 0%, 0% 100%, 100% 100%)" }} />
          Class 1 (多面體 Icosahedron)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded bg-purple-400 opacity-60" />
          決策面 Decision Surface
        </span>
      </div>
    </div>
  );
}
