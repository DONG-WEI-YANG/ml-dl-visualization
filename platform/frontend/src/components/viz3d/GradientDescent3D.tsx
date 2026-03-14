import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import Scene3D from "./Scene3D";
import Axis3D from "./Axis3D";
import { fetchAPI } from "../../lib/api";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface GradientDescent3DProps {
  lr: number;
  onLrChange: (lr: number) => void;
}

interface LossLandscapeResponse {
  w0: number[];
  w1: number[];
  loss: number[][];
}

type SurfaceType = "bowl" | "saddle" | "local_minima";

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/** Map a 0-1 value to a colour gradient: green/blue (low) -> yellow -> red (high) */
function lossColor(t: number): THREE.Color {
  if (t < 0.5) {
    const s = t * 2;
    // #10b981 -> #f59e0b
    return new THREE.Color().setRGB(
      0x10 / 255 + s * (0xf5 / 255 - 0x10 / 255),
      0xb9 / 255 + s * (0x9e / 255 - 0xb9 / 255),
      0x81 / 255 + s * (0x0b / 255 - 0x81 / 255),
    );
  }
  const s = (t - 0.5) * 2;
  // #f59e0b -> #ef4444
  return new THREE.Color().setRGB(
    0xf5 / 255 + s * (0xef / 255 - 0xf5 / 255),
    0x9e / 255 + s * (0x44 / 255 - 0x9e / 255),
    0x0b / 255 + s * (0x44 / 255 - 0x0b / 255),
  );
}

/** Bi-linear interpolation of the loss grid at (wx, wy). Returns [loss, dL/dw0, dL/dw1]. */
function sampleSurface(
  w0: number[],
  w1: number[],
  loss: number[][],
  wx: number,
  wy: number,
): [number, number, number] {
  const nx = w0.length;
  const ny = w1.length;
  // Map world coords to grid indices
  const fi = ((wx - w0[0]) / (w0[nx - 1] - w0[0])) * (nx - 1);
  const fj = ((wy - w1[0]) / (w1[ny - 1] - w1[0])) * (ny - 1);
  const i = Math.max(0, Math.min(nx - 2, Math.floor(fi)));
  const j = Math.max(0, Math.min(ny - 2, Math.floor(fj)));
  const u = fi - i;
  const v = fj - j;

  const z00 = loss[j][i];
  const z10 = loss[j][i + 1];
  const z01 = loss[j + 1][i];
  const z11 = loss[j + 1][i + 1];

  const z = (1 - u) * (1 - v) * z00 + u * (1 - v) * z10 + (1 - u) * v * z01 + u * v * z11;

  // Numerical gradient
  const eps = (w0[1] - w0[0]) * 0.5;
  const [zxp] = sampleLoss(w0, w1, loss, wx + eps, wy);
  const [zxm] = sampleLoss(w0, w1, loss, wx - eps, wy);
  const [zyp] = sampleLoss(w0, w1, loss, wx, wy + eps);
  const [zym] = sampleLoss(w0, w1, loss, wx, wy - eps);

  return [z, (zxp - zxm) / (2 * eps), (zyp - zym) / (2 * eps)];
}

/** Just the loss value (no gradient) — avoids recursion in gradient computation. */
function sampleLoss(
  w0: number[],
  w1: number[],
  loss: number[][],
  wx: number,
  wy: number,
): [number] {
  const nx = w0.length;
  const ny = w1.length;
  const fi = ((wx - w0[0]) / (w0[nx - 1] - w0[0])) * (nx - 1);
  const fj = ((wy - w1[0]) / (w1[ny - 1] - w1[0])) * (ny - 1);
  const i = Math.max(0, Math.min(nx - 2, Math.floor(fi)));
  const j = Math.max(0, Math.min(ny - 2, Math.floor(fj)));
  const u = fi - i;
  const v = fj - j;
  const z =
    (1 - u) * (1 - v) * loss[j][i] +
    u * (1 - v) * loss[j][i + 1] +
    (1 - u) * v * loss[j + 1][i] +
    u * v * loss[j + 1][i + 1];
  return [z];
}

/* ------------------------------------------------------------------ */
/*  Surface mesh (runs inside Canvas)                                  */
/* ------------------------------------------------------------------ */

function SurfaceMesh({
  w0,
  w1,
  loss,
}: {
  w0: number[];
  w1: number[];
  loss: number[][];
}) {
  const geometry = useMemo(() => {
    const nx = w0.length;
    const ny = w1.length;
    const geo = new THREE.BufferGeometry();

    const positions: number[] = [];
    const colors: number[] = [];
    const indices: number[] = [];

    // Find min/max for normalisation
    let minL = Infinity;
    let maxL = -Infinity;
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        if (loss[j][i] < minL) minL = loss[j][i];
        if (loss[j][i] > maxL) maxL = loss[j][i];
      }
    const range = maxL - minL || 1;

    // Scale factors to fit nicely in [-4,4]
    const xMin = w0[0],
      xMax = w0[nx - 1];
    const yMin = w1[0],
      yMax = w1[ny - 1];
    const sx = 8 / (xMax - xMin);
    const sy = 8 / (yMax - yMin);
    const sz = 4 / range;

    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const x = (w0[i] - (xMin + xMax) / 2) * sx;
        const z = (w1[j] - (yMin + yMax) / 2) * sy;
        const y = (loss[j][i] - minL) * sz - 2; // shift down a bit
        positions.push(x, y, z);

        const t = (loss[j][i] - minL) / range;
        const c = lossColor(t);
        colors.push(c.r, c.g, c.b);
      }
    }

    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const a = j * nx + i;
        const b = a + 1;
        const c = a + nx;
        const d = c + 1;
        indices.push(a, b, d, a, d, c);
      }
    }

    geo.setIndex(indices);
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    return geo;
  }, [w0, w1, loss]);

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial
        vertexColors
        transparent
        opacity={0.8}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/* ------------------------------------------------------------------ */
/*  Animated ball (runs inside Canvas)                                 */
/* ------------------------------------------------------------------ */

function GDBall({
  w0,
  w1,
  loss,
  lr,
  playing,
  resetToken,
}: {
  w0: number[];
  w1: number[];
  loss: number[][];
  lr: number;
  playing: boolean;
  resetToken: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const posRef = useRef<[number, number]>([0, 0]);

  // Scale factors (must match SurfaceMesh)
  const nx = w0.length;
  const ny = w1.length;
  const xMin = w0[0], xMax = w0[nx - 1];
  const yMin = w1[0], yMax = w1[ny - 1];

  let minL = Infinity, maxL = -Infinity;
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      if (loss[j][i] < minL) minL = loss[j][i];
      if (loss[j][i] > maxL) maxL = loss[j][i];
    }
  const range = maxL - minL || 1;
  const sx = 8 / (xMax - xMin);
  const sy = 8 / (yMax - yMin);
  const sz = 4 / range;
  const cx = (xMin + xMax) / 2;
  const cy = (yMin + yMax) / 2;

  // Reset position when resetToken changes
  useEffect(() => {
    const rx = xMin + Math.random() * (xMax - xMin) * 0.6 + (xMax - xMin) * 0.2;
    const ry = yMin + Math.random() * (yMax - yMin) * 0.6 + (yMax - yMin) * 0.2;
    posRef.current = [rx, ry];
  }, [resetToken, xMin, xMax, yMin, yMax]);

  useFrame(() => {
    if (!meshRef.current) return;

    if (playing) {
      const [wx, wy] = posRef.current;
      const [, gx, gy] = sampleSurface(w0, w1, loss, wx, wy);
      const newWx = Math.max(xMin, Math.min(xMax, wx - lr * gx));
      const newWy = Math.max(yMin, Math.min(yMax, wy - lr * gy));
      posRef.current = [newWx, newWy];
    }

    const [wx, wy] = posRef.current;
    const [lossVal] = sampleLoss(w0, w1, loss, wx, wy);
    meshRef.current.position.set(
      (wx - cx) * sx,
      (lossVal - minL) * sz - 2 + 0.15,
      (wy - cy) * sy,
    );
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.15, 16, 16]} />
      <meshStandardMaterial color="#f59e0b" />
    </mesh>
  );
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

export default function GradientDescent3D({ lr, onLrChange }: GradientDescent3DProps) {
  const [surfaceType, setSurfaceType] = useState<SurfaceType>("bowl");
  const [data, setData] = useState<LossLandscapeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [resetToken, setResetToken] = useState(0);

  const fetchSurface = useCallback(async (type: SurfaceType) => {
    setLoading(true);
    setPlaying(false);
    try {
      const resp = await fetchAPI<LossLandscapeResponse>(
        "/api/models/loss-landscape",
        { surface_type: type, resolution: 40 },
      );
      setData(resp);
      setResetToken((t) => t + 1);
    } catch (err) {
      console.error("Failed to fetch loss landscape:", err);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchSurface(surfaceType);
  }, [surfaceType, fetchSurface]);

  const handleReset = () => {
    setPlaying(false);
    setResetToken((t) => t + 1);
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">
        3D 損失曲面 Loss Landscape
      </h3>

      {/* Surface type selector */}
      <div className="flex gap-2">
        {(["bowl", "saddle", "local_minima"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setSurfaceType(t)}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              surfaceType === t
                ? "bg-blue-500 text-white"
                : "bg-gray-100 text-gray-700 hover:bg-gray-200"
            }`}
          >
            {t === "bowl" ? "碗形 Bowl" : t === "saddle" ? "鞍點 Saddle" : "局部最小 Local Minima"}
          </button>
        ))}
      </div>

      {/* Learning rate slider */}
      <div>
        <label className="block text-sm text-gray-600 mb-1">
          學習率 Learning Rate: <strong>{lr.toFixed(3)}</strong>
        </label>
        <input
          type="range"
          min="0.001"
          max="1"
          step="0.001"
          value={lr}
          onChange={(e) => onLrChange(+e.target.value)}
          className="w-full"
        />
      </div>

      {/* Controls */}
      <div className="flex gap-2">
        <button
          onClick={() => setPlaying((p) => !p)}
          disabled={!data || loading}
          className="px-4 py-2 bg-green-500 text-white rounded-lg text-sm font-medium hover:bg-green-600 disabled:opacity-50"
        >
          {playing ? "暫停 Pause" : "播放 Play"}
        </button>
        <button
          onClick={handleReset}
          disabled={!data || loading}
          className="px-4 py-2 bg-gray-500 text-white rounded-lg text-sm font-medium hover:bg-gray-600 disabled:opacity-50"
        >
          重置 Reset
        </button>
      </div>

      {/* 3D canvas */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        {loading && (
          <div className="flex items-center justify-center h-[400px] bg-gray-50">
            <p className="text-gray-500 text-sm">載入曲面中...</p>
          </div>
        )}
        {!loading && data && (
          <Scene3D cameraPosition={[6, 6, 6]} showGrid>
            <Axis3D
              labels={{ x: "w₀", y: "Loss", z: "w₁" }}
              range={{ x: [-4, 4], y: [-4, 4], z: [-4, 4] }}
            />
            <SurfaceMesh w0={data.w0} w1={data.w1} loss={data.loss} />
            <GDBall
              w0={data.w0}
              w1={data.w1}
              loss={data.loss}
              lr={lr}
              playing={playing}
              resetToken={resetToken}
            />
          </Scene3D>
        )}
      </div>
    </div>
  );
}
