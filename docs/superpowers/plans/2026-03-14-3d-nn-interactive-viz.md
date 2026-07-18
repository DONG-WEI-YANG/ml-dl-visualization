# 3D Visualizations & NN Interactive Diagram Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 3 × 3D scene visualizations (Weeks 4, 5/6, 17) and 1 × NN interactive diagram (Week 11) to the ML/DL teaching platform.

**Architecture:** Shared 3D base layer (`viz3d/`) using react-three-fiber for all 3D scenes. Separate `nn/` directory for the NN interactive diagram using Canvas 2D rendering with pure-function math layer. Existing viz components gain internal 2D/3D tab toggles. Week 11 gets a compound wrapper linking ActivationFunctionViz with NeuralNetworkViz.

**Tech Stack:** React 19, TypeScript, Three.js (react-three-fiber + drei), Recharts, Canvas 2D API, Vitest, FastAPI + scikit-learn (backend)

**Spec:** `docs/superpowers/specs/2026-03-14-3d-nn-interactive-viz-design.md`

---

## Chunk 1: Infrastructure & Shared 3D Components

### Task 1: Install dependencies and clean up

**Files:**
- Modify: `platform/frontend/package.json`

- [ ] **Step 1: Install Three.js dependencies**

Run:
```bash
cd platform/frontend && npm install three @react-three/fiber @react-three/drei && npm install -D @types/three
```

- [ ] **Step 2: Remove unused D3**

Run:
```bash
cd platform/frontend && npm uninstall d3 @types/d3
```

- [ ] **Step 3: Verify build still works**

Run:
```bash
cd platform/frontend && npm run build
```
Expected: Build succeeds with no errors. No existing code imports D3.

- [ ] **Step 4: Commit**

```bash
git add platform/frontend/package.json platform/frontend/package-lock.json
git commit -m "chore: add three.js/r3f, remove unused d3"
```

---

### Task 2: Create viz3d type definitions

**Files:**
- Create: `platform/frontend/src/components/viz3d/types.ts`

- [ ] **Step 1: Create the types file**

```typescript
// platform/frontend/src/components/viz3d/types.ts
import type { CSSProperties, ReactNode } from "react";

export type Vec3 = [number, number, number];
export type Vec3Pair = [number, number];

export interface Scene3DProps {
  children: ReactNode;
  cameraPosition?: Vec3;
  showGrid?: boolean;
  backgroundColor?: string;
  enableDamping?: boolean;
  style?: CSSProperties;
  className?: string;
  fallback?: ReactNode;
}

export interface Axis3DProps {
  labels?: { x: string; y: string; z: string };
  range?: { x: Vec3Pair; y: Vec3Pair; z: Vec3Pair };
  showTicks?: boolean;
  showGrid?: boolean;
  tickCount?: number;
}

export interface Tooltip3DProps {
  position: Vec3;
  visible: boolean;
  children: ReactNode;
}
```

- [ ] **Step 2: Commit**

```bash
git add platform/frontend/src/components/viz3d/types.ts
git commit -m "feat: add viz3d TypeScript type definitions"
```

---

### Task 3: Create Scene3D shared component

**Files:**
- Create: `platform/frontend/src/components/viz3d/Scene3D.tsx`
- Test: `platform/frontend/src/components/viz3d/__tests__/Scene3D.test.tsx`

- [ ] **Step 1: Write the smoke test**

```typescript
// platform/frontend/src/components/viz3d/__tests__/Scene3D.test.tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

// Mock react-three-fiber since WebGL isn't available in jsdom
vi.mock("@react-three/fiber", () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="r3f-canvas">{children}</div>
  ),
}));
vi.mock("@react-three/drei", () => ({
  OrbitControls: () => null,
}));

import Scene3D from "../Scene3D";

describe("Scene3D", () => {
  it("renders canvas when WebGL is available", () => {
    // jsdom doesn't have real WebGL, but our mock bypasses that
    render(
      <Scene3D>
        <mesh />
      </Scene3D>
    );
    expect(screen.getByTestId("r3f-canvas")).toBeDefined();
  });

  it("renders fallback when WebGL check fails", () => {
    // Override the WebGL check
    const origCreate = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
      if (tag === "canvas") {
        const el = origCreate(tag);
        el.getContext = () => null;
        return el;
      }
      return origCreate(tag);
    });

    render(
      <Scene3D fallback={<div data-testid="fallback">No WebGL</div>}>
        <mesh />
      </Scene3D>
    );
    expect(screen.getByTestId("fallback")).toBeDefined();

    vi.restoreAllMocks();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/frontend && npx vitest run src/components/viz3d/__tests__/Scene3D.test.tsx`
Expected: FAIL — `Scene3D` module not found

- [ ] **Step 3: Implement Scene3D**

```typescript
// platform/frontend/src/components/viz3d/Scene3D.tsx
import { useState, useEffect, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { Scene3DProps } from "./types";

function checkWebGL(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return !!(canvas.getContext("webgl") || canvas.getContext("webgl2"));
  } catch {
    return false;
  }
}

const DEFAULT_FALLBACK = (
  <div className="flex items-center justify-center h-full min-h-[300px] bg-gray-50 rounded-lg">
    <div className="text-center text-gray-500 text-sm">
      <p className="text-2xl mb-2">⚠️</p>
      <p>您的瀏覽器不支援 WebGL，請改用 2D 模式</p>
    </div>
  </div>
);

export default function Scene3D({
  children,
  cameraPosition = [5, 5, 5],
  showGrid = false,
  backgroundColor = "#0f172a",
  enableDamping = true,
  style,
  className,
  fallback = DEFAULT_FALLBACK,
  ariaLabel = "3D 視覺化場景",
}: Scene3DProps & { ariaLabel?: string }) {
  const [webglOk, setWebglOk] = useState(true);
  const controlsRef = useRef<any>(null);

  useEffect(() => {
    setWebglOk(checkWebGL());
  }, []);

  // Keyboard shortcuts: arrows=rotate, +/-=zoom
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const ctrl = controlsRef.current;
      if (!ctrl) return;
      const step = 0.1;
      switch (e.key) {
        case "ArrowLeft": ctrl.setAzimuthalAngle(ctrl.getAzimuthalAngle() - step); break;
        case "ArrowRight": ctrl.setAzimuthalAngle(ctrl.getAzimuthalAngle() + step); break;
        case "ArrowUp": ctrl.setPolarAngle(Math.max(0.1, ctrl.getPolarAngle() - step)); break;
        case "ArrowDown": ctrl.setPolarAngle(Math.min(Math.PI - 0.1, ctrl.getPolarAngle() + step)); break;
        case "+": case "=": ctrl.object.zoom = Math.min(10, ctrl.object.zoom * 1.1); ctrl.object.updateProjectionMatrix(); break;
        case "-": ctrl.object.zoom = Math.max(0.1, ctrl.object.zoom / 1.1); ctrl.object.updateProjectionMatrix(); break;
      }
      ctrl.update();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  if (!webglOk) return <>{fallback}</>;

  return (
    <div className={className} style={{ minHeight: 400, ...style }} aria-label={ariaLabel} role="img" tabIndex={0}>
      <Canvas
        camera={{ position: cameraPosition, fov: 50 }}
        style={{ background: backgroundColor }}
      >
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, 10, 5]} intensity={0.8} />
        {showGrid && <gridHelper args={[20, 20, "#334155", "#1e293b"]} />}
        <OrbitControls
          ref={controlsRef}
          enableDamping={enableDamping}
          dampingFactor={0.05}
          enablePan
          enableZoom
          enableRotate
        />
        {children}
      </Canvas>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd platform/frontend && npx vitest run src/components/viz3d/__tests__/Scene3D.test.tsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add platform/frontend/src/components/viz3d/Scene3D.tsx platform/frontend/src/components/viz3d/__tests__/Scene3D.test.tsx
git commit -m "feat: add Scene3D shared component with WebGL fallback"
```

---

### Task 4: Create Axis3D shared component

**Files:**
- Create: `platform/frontend/src/components/viz3d/Axis3D.tsx`

- [ ] **Step 1: Implement Axis3D**

```typescript
// platform/frontend/src/components/viz3d/Axis3D.tsx
import { useMemo } from "react";
import { Text } from "@react-three/drei";
import type { Axis3DProps, Vec3Pair } from "./types";

const DEFAULT_RANGE: Vec3Pair = [-5, 5];
const AXIS_COLOR = "#64748b";
const TICK_COLOR = "#475569";

export default function Axis3D({
  labels = { x: "X", y: "Y", z: "Z" },
  range = { x: DEFAULT_RANGE, y: DEFAULT_RANGE, z: DEFAULT_RANGE },
  showTicks = true,
  showGrid = false,
  tickCount = 5,
}: Axis3DProps) {
  const ticks = useMemo(() => {
    const makeTicks = (r: Vec3Pair) => {
      const step = (r[1] - r[0]) / tickCount;
      return Array.from({ length: tickCount + 1 }, (_, i) =>
        +(r[0] + i * step).toFixed(2)
      );
    };
    return { x: makeTicks(range.x), y: makeTicks(range.y), z: makeTicks(range.z) };
  }, [range, tickCount]);

  return (
    <group>
      {/* X axis */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([range.x[0], 0, 0, range.x[1], 0, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color={AXIS_COLOR} />
      </line>
      <Text position={[range.x[1] + 0.5, 0, 0]} fontSize={0.3} color={AXIS_COLOR}>
        {labels.x}
      </Text>

      {/* Y axis */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, range.y[0], 0, 0, range.y[1], 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color={AXIS_COLOR} />
      </line>
      <Text position={[0, range.y[1] + 0.5, 0]} fontSize={0.3} color={AXIS_COLOR}>
        {labels.y}
      </Text>

      {/* Z axis */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, range.z[0], 0, 0, range.z[1]])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color={AXIS_COLOR} />
      </line>
      <Text position={[0, 0, range.z[1] + 0.5]} fontSize={0.3} color={AXIS_COLOR}>
        {labels.z}
      </Text>

      {/* Ticks */}
      {showTicks &&
        ticks.x.map((v) => (
          <Text key={`tx-${v}`} position={[v, -0.3, 0]} fontSize={0.15} color={TICK_COLOR}>
            {v}
          </Text>
        ))}
      {showTicks &&
        ticks.y.map((v) => (
          <Text key={`ty-${v}`} position={[-0.3, v, 0]} fontSize={0.15} color={TICK_COLOR}>
            {v}
          </Text>
        ))}

      {/* Grid planes */}
      {showGrid && (
        <>
          <gridHelper args={[range.x[1] - range.x[0], tickCount, "#1e293b", "#1e293b"]} />
        </>
      )}
    </group>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add platform/frontend/src/components/viz3d/Axis3D.tsx
git commit -m "feat: add Axis3D shared component"
```

---

### Task 5: Create Tooltip3D shared component

**Files:**
- Create: `platform/frontend/src/components/viz3d/Tooltip3D.tsx`

- [ ] **Step 1: Implement Tooltip3D**

```typescript
// platform/frontend/src/components/viz3d/Tooltip3D.tsx
import { Html } from "@react-three/drei";
import type { Tooltip3DProps } from "./types";

export default function Tooltip3D({ position, visible, children }: Tooltip3DProps) {
  if (!visible) return null;

  return (
    <Html position={position} center>
      <div className="bg-white border border-gray-200 rounded-lg shadow-lg px-3 py-2 text-xs text-gray-700 pointer-events-none whitespace-nowrap">
        {children}
      </div>
    </Html>
  );
}
```

- [ ] **Step 2: Create index barrel export**

```typescript
// platform/frontend/src/components/viz3d/index.ts
export { default as Scene3D } from "./Scene3D";
export { default as Axis3D } from "./Axis3D";
export { default as Tooltip3D } from "./Tooltip3D";
export * from "./types";
```

- [ ] **Step 3: Commit**

```bash
git add platform/frontend/src/components/viz3d/Tooltip3D.tsx platform/frontend/src/components/viz3d/index.ts
git commit -m "feat: add Tooltip3D, create viz3d barrel export"
```

---

## Chunk 2: NN Math Layer (TDD)

### Task 6: Create NN type definitions

**Files:**
- Create: `platform/frontend/src/components/nn/nn-types.ts`

- [ ] **Step 1: Create the types file**

```typescript
// platform/frontend/src/components/nn/nn-types.ts
export type ActivationType = "sigmoid" | "tanh" | "relu" | "leaky_relu" | "gelu";

export type Matrix = number[][];

export interface NetworkConfig {
  inputSize: number;
  hiddenLayers: number;      // 1~4
  neuronsPerLayer: number;   // 2~8
  outputSize: number;
  activation: ActivationType;
}

export interface Weights {
  layers: Matrix[];          // layers[i] = weight matrix between layer i and i+1
  biases: number[][];        // biases[i] = bias vector for layer i+1
}

export interface ForwardResult {
  activations: number[][];   // activations[i] = output of layer i (after activation)
  preActivations: number[][]; // before activation (for gradient computation)
}

export interface GradientResult {
  weightGrads: Matrix[];
  biasGrads: number[][];
  layerGrads: number[][];    // gradient at each layer (for viz)
}

export interface TrainingState {
  epoch: number;
  lossHistory: number[];
  isTraining: boolean;
  weights: Weights;
}

// Datasets for training
export interface Dataset {
  name: string;
  inputs: number[][];
  targets: number[][];
}
```

- [ ] **Step 2: Commit**

```bash
git add platform/frontend/src/components/nn/nn-types.ts
git commit -m "feat: add NN type definitions"
```

---

### Task 7: Implement nn-math.ts with TDD — activation functions

**Files:**
- Create: `platform/frontend/src/components/nn/nn-math.ts`
- Test: `platform/frontend/src/components/nn/__tests__/nn-math.test.ts`

- [ ] **Step 1: Write activation function tests**

```typescript
// platform/frontend/src/components/nn/__tests__/nn-math.test.ts
import { describe, it, expect } from "vitest";
import { activate, activateDerivative } from "../nn-math";

describe("activate", () => {
  it("sigmoid(0) = 0.5", () => {
    expect(activate(0, "sigmoid")).toBeCloseTo(0.5, 5);
  });

  it("sigmoid(large positive) ≈ 1", () => {
    expect(activate(10, "sigmoid")).toBeCloseTo(1, 2);
  });

  it("tanh(0) = 0", () => {
    expect(activate(0, "tanh")).toBeCloseTo(0, 5);
  });

  it("relu(negative) = 0", () => {
    expect(activate(-5, "relu")).toBe(0);
  });

  it("relu(positive) = identity", () => {
    expect(activate(3, "relu")).toBe(3);
  });

  it("leaky_relu(negative) = 0.01 * x", () => {
    expect(activate(-5, "leaky_relu")).toBeCloseTo(-0.05, 5);
  });

  it("gelu(0) = 0", () => {
    expect(activate(0, "gelu")).toBeCloseTo(0, 2);
  });
});

describe("activateDerivative", () => {
  it("sigmoid'(0) = 0.25", () => {
    expect(activateDerivative(0, "sigmoid")).toBeCloseTo(0.25, 5);
  });

  it("relu'(negative) = 0", () => {
    expect(activateDerivative(-1, "relu")).toBe(0);
  });

  it("relu'(positive) = 1", () => {
    expect(activateDerivative(1, "relu")).toBe(1);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/frontend && npx vitest run src/components/nn/__tests__/nn-math.test.ts`
Expected: FAIL — module not found

- [ ] **Step 3: Implement activation functions**

```typescript
// platform/frontend/src/components/nn/nn-math.ts
import type { ActivationType, NetworkConfig, Weights, Matrix, ForwardResult, GradientResult } from "./nn-types";

// ─── Activation Functions ───

export function activate(x: number, type: ActivationType): number {
  switch (type) {
    case "sigmoid": return 1 / (1 + Math.exp(-x));
    case "tanh": return Math.tanh(x);
    case "relu": return Math.max(0, x);
    case "leaky_relu": return x > 0 ? x : 0.01 * x;
    case "gelu": {
      // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
      const c = Math.sqrt(2 / Math.PI);
      return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
    }
  }
}

export function activateDerivative(x: number, type: ActivationType): number {
  switch (type) {
    case "sigmoid": {
      const s = activate(x, "sigmoid");
      return s * (1 - s);
    }
    case "tanh": {
      const t = Math.tanh(x);
      return 1 - t * t;
    }
    case "relu": return x > 0 ? 1 : 0;
    case "leaky_relu": return x > 0 ? 1 : 0.01;
    case "gelu": {
      const c = Math.sqrt(2 / Math.PI);
      const inner = c * (x + 0.044715 * x * x * x);
      const tanhInner = Math.tanh(inner);
      const sech2 = 1 - tanhInner * tanhInner;
      return 0.5 * (1 + tanhInner) + 0.5 * x * sech2 * c * (1 + 3 * 0.044715 * x * x);
    }
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd platform/frontend && npx vitest run src/components/nn/__tests__/nn-math.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add platform/frontend/src/components/nn/nn-math.ts platform/frontend/src/components/nn/__tests__/nn-math.test.ts
git commit -m "feat: add NN activation functions with tests"
```

---

### Task 8: nn-math.ts — weight initialization

**Files:**
- Modify: `platform/frontend/src/components/nn/nn-math.ts`
- Modify: `platform/frontend/src/components/nn/__tests__/nn-math.test.ts`

- [ ] **Step 1: Write initWeights tests**

Append to test file:

```typescript
import { initWeights } from "../nn-math";
import type { NetworkConfig } from "../nn-types";

describe("initWeights", () => {
  const config: NetworkConfig = {
    inputSize: 2,
    hiddenLayers: 2,
    neuronsPerLayer: 3,
    outputSize: 1,
    activation: "relu",
  };

  it("creates correct number of weight matrices", () => {
    const w = initWeights(config);
    // layers: input(2) -> h1(3) -> h2(3) -> output(1) = 3 matrices
    expect(w.layers).toHaveLength(3);
    expect(w.biases).toHaveLength(3);
  });

  it("weight matrix dimensions match layer sizes", () => {
    const w = initWeights(config);
    // input(2) -> h1(3): 2×3
    expect(w.layers[0]).toHaveLength(2);
    expect(w.layers[0][0]).toHaveLength(3);
    // h1(3) -> h2(3): 3×3
    expect(w.layers[1]).toHaveLength(3);
    expect(w.layers[1][0]).toHaveLength(3);
    // h2(3) -> output(1): 3×1
    expect(w.layers[2]).toHaveLength(3);
    expect(w.layers[2][0]).toHaveLength(1);
  });

  it("bias vectors match layer sizes", () => {
    const w = initWeights(config);
    expect(w.biases[0]).toHaveLength(3);
    expect(w.biases[1]).toHaveLength(3);
    expect(w.biases[2]).toHaveLength(1);
  });

  it("weights are non-zero (Xavier init)", () => {
    const w = initWeights(config);
    const allZero = w.layers.every((m) => m.every((r) => r.every((v) => v === 0)));
    expect(allZero).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/frontend && npx vitest run src/components/nn/__tests__/nn-math.test.ts`
Expected: FAIL — `initWeights` not exported

- [ ] **Step 3: Implement initWeights**

Append to `nn-math.ts`:

```typescript
// ─── Weight Initialization ───

function xavierRandom(fanIn: number, fanOut: number): number {
  // Xavier uniform initialization
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  return (Math.random() * 2 - 1) * limit;
}

export function initWeights(config: NetworkConfig): Weights {
  const sizes = [
    config.inputSize,
    ...Array(config.hiddenLayers).fill(config.neuronsPerLayer),
    config.outputSize,
  ];

  const layers: Matrix[] = [];
  const biases: number[][] = [];

  for (let i = 0; i < sizes.length - 1; i++) {
    const fanIn = sizes[i];
    const fanOut = sizes[i + 1];
    const matrix: number[][] = [];
    for (let r = 0; r < fanIn; r++) {
      const row: number[] = [];
      for (let c = 0; c < fanOut; c++) {
        row.push(xavierRandom(fanIn, fanOut));
      }
      matrix.push(row);
    }
    layers.push(matrix);
    biases.push(Array(fanOut).fill(0));
  }

  return { layers, biases };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd platform/frontend && npx vitest run src/components/nn/__tests__/nn-math.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add platform/frontend/src/components/nn/nn-math.ts platform/frontend/src/components/nn/__tests__/nn-math.test.ts
git commit -m "feat: add weight initialization with Xavier init"
```

---

### Task 9: nn-math.ts — forward pass

**Files:**
- Modify: `platform/frontend/src/components/nn/nn-math.ts`
- Modify: `platform/frontend/src/components/nn/__tests__/nn-math.test.ts`

- [ ] **Step 1: Write forward pass tests**

Append to test file:

```typescript
import { forward } from "../nn-math";

describe("forward", () => {
  it("produces correct output for known weights (identity-like)", () => {
    // Simple 2-input, 1 hidden (2 neurons), 1-output network
    const weights: Weights = {
      layers: [
        [[1, 0], [0, 1]],   // input(2) -> hidden(2): identity
        [[1], [1]],          // hidden(2) -> output(1): sum
      ],
      biases: [
        [0, 0],              // hidden bias
        [0],                 // output bias
      ],
    };

    const result = forward([1, 2], weights, "relu");
    // hidden = relu([1*1+2*0, 1*0+2*1]) = relu([1, 2]) = [1, 2]
    // output = relu([1*1+2*1]) = relu([3]) = [3]
    expect(result.activations).toHaveLength(3); // input, hidden, output
    expect(result.activations[0]).toEqual([1, 2]); // input passthrough
    expect(result.activations[1]).toEqual([1, 2]); // hidden after relu
    expect(result.activations[2][0]).toBeCloseTo(3, 5); // output
  });

  it("applies sigmoid correctly", () => {
    const weights: Weights = {
      layers: [[[1]], [[1]]],
      biases: [[0], [0]],
    };
    const result = forward([0], weights, "sigmoid");
    // hidden = sigmoid(0) = 0.5, output = sigmoid(0.5) ≈ 0.622
    expect(result.activations[1][0]).toBeCloseTo(0.5, 5);
    expect(result.activations[2][0]).toBeCloseTo(0.622, 2);
  });

  it("stores pre-activations for backprop", () => {
    const weights: Weights = {
      layers: [[[2]], [[3]]],
      biases: [[1], [0]],
    };
    const result = forward([1], weights, "relu");
    // pre-activation h = 2*1 + 1 = 3, activation = relu(3) = 3
    expect(result.preActivations[0][0]).toBeCloseTo(3, 5);
    expect(result.activations[1][0]).toBeCloseTo(3, 5);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/frontend && npx vitest run src/components/nn/__tests__/nn-math.test.ts`
Expected: FAIL — `forward` not exported

- [ ] **Step 3: Implement forward pass**

Append to `nn-math.ts`:

```typescript
// ─── Forward Pass ───

export function forward(
  input: number[],
  weights: Weights,
  activation: ActivationType,
): ForwardResult {
  const activations: number[][] = [input];
  const preActivations: number[][] = [];

  let current = input;

  for (let l = 0; l < weights.layers.length; l++) {
    const W = weights.layers[l];
    const b = weights.biases[l];
    const nextSize = W[0].length;
    const pre: number[] = [];

    for (let j = 0; j < nextSize; j++) {
      let sum = b[j];
      for (let i = 0; i < current.length; i++) {
        sum += current[i] * W[i][j];
      }
      pre.push(sum);
    }

    preActivations.push(pre);

    // Apply activation to all layers
    const activated = pre.map((v) => activate(v, activation));

    activations.push(activated);
    current = activated;
  }

  return { activations, preActivations };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd platform/frontend && npx vitest run src/components/nn/__tests__/nn-math.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add platform/frontend/src/components/nn/nn-math.ts platform/frontend/src/components/nn/__tests__/nn-math.test.ts
git commit -m "feat: add NN forward pass"
```

---

### Task 10: nn-math.ts — backward pass and weight update

**Files:**
- Modify: `platform/frontend/src/components/nn/nn-math.ts`
- Modify: `platform/frontend/src/components/nn/__tests__/nn-math.test.ts`

- [ ] **Step 1: Write backward pass tests**

Append to test file:

```typescript
import { backward, updateWeights, computeLoss } from "../nn-math";

describe("computeLoss", () => {
  it("MSE of [1] vs [0] = 0.5", () => {
    expect(computeLoss([1], [0])).toBeCloseTo(0.5, 5);
  });

  it("MSE of identical = 0", () => {
    expect(computeLoss([1, 2], [1, 2])).toBeCloseTo(0, 5);
  });
});

describe("backward", () => {
  it("produces gradients with correct shapes", () => {
    const weights: Weights = {
      layers: [[[1, 0], [0, 1]], [[1], [1]]],
      biases: [[0, 0], [0]],
    };
    const fwd = forward([1, 2], weights, "relu");
    const grad = backward(fwd, weights, [2], "relu");

    expect(grad.weightGrads).toHaveLength(2);
    expect(grad.biasGrads).toHaveLength(2);
    // Same shapes as weight matrices
    expect(grad.weightGrads[0]).toHaveLength(2);
    expect(grad.weightGrads[0][0]).toHaveLength(2);
    expect(grad.weightGrads[1]).toHaveLength(2);
    expect(grad.weightGrads[1][0]).toHaveLength(1);
  });
});

describe("updateWeights", () => {
  it("reduces loss after one step on simple case", () => {
    const weights: Weights = {
      layers: [[[0.5]], [[0.5]]],
      biases: [[0], [0]],
    };
    const input = [1];
    const target = [1];

    const fwd1 = forward(input, weights, "relu");
    const loss1 = computeLoss(fwd1.activations[fwd1.activations.length - 1], target);

    const grad = backward(fwd1, weights, target, "relu");
    const newWeights = updateWeights(weights, grad, 0.1);

    const fwd2 = forward(input, newWeights, "relu");
    const loss2 = computeLoss(fwd2.activations[fwd2.activations.length - 1], target);

    expect(loss2).toBeLessThan(loss1);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/frontend && npx vitest run src/components/nn/__tests__/nn-math.test.ts`
Expected: FAIL — `backward`, `updateWeights`, `computeLoss` not exported

- [ ] **Step 3: Implement backward pass, loss, and weight update**

Append to `nn-math.ts`:

```typescript
// ─── Loss ───

export function computeLoss(predicted: number[], target: number[]): number {
  let sum = 0;
  for (let i = 0; i < predicted.length; i++) {
    const diff = predicted[i] - target[i];
    sum += diff * diff;
  }
  return sum / (2 * predicted.length);
}

// ─── Backward Pass ───

export function backward(
  fwd: ForwardResult,
  weights: Weights,
  target: number[],
  activation: ActivationType,
): GradientResult {
  const numLayers = weights.layers.length;
  const weightGrads: Matrix[] = [];
  const biasGrads: number[][] = [];
  const layerGrads: number[][] = [];

  // Output layer error: dL/da = (a - target) / n
  const output = fwd.activations[fwd.activations.length - 1];
  let delta: number[] = output.map((o, i) => {
    const dLoss = (o - target[i]) / target.length;
    const dAct = activateDerivative(fwd.preActivations[numLayers - 1][i], activation);
    return dLoss * dAct;
  });
  layerGrads.unshift(delta);

  // Backpropagate through layers
  for (let l = numLayers - 1; l >= 0; l--) {
    const prevActivations = fwd.activations[l]; // activations of layer feeding into this weight matrix
    const W = weights.layers[l];

    // Weight gradients: dW[i][j] = prevAct[i] * delta[j]
    const wGrad: number[][] = [];
    for (let i = 0; i < prevActivations.length; i++) {
      const row: number[] = [];
      for (let j = 0; j < delta.length; j++) {
        row.push(prevActivations[i] * delta[j]);
      }
      wGrad.push(row);
    }
    weightGrads.unshift(wGrad);

    // Bias gradients: dB[j] = delta[j]
    biasGrads.unshift([...delta]);

    // Propagate delta to previous layer (if not input layer)
    if (l > 0) {
      const newDelta: number[] = [];
      for (let i = 0; i < W.length; i++) {
        let sum = 0;
        for (let j = 0; j < delta.length; j++) {
          sum += W[i][j] * delta[j];
        }
        sum *= activateDerivative(fwd.preActivations[l - 1][i], activation);
        newDelta.push(sum);
      }
      delta = newDelta;
      layerGrads.unshift(delta);
    }
  }

  return { weightGrads, biasGrads, layerGrads };
}

// ─── Weight Update ───

export function updateWeights(
  weights: Weights,
  gradients: GradientResult,
  lr: number,
): Weights {
  const newLayers: Matrix[] = weights.layers.map((layer, l) =>
    layer.map((row, i) =>
      row.map((w, j) => w - lr * gradients.weightGrads[l][i][j])
    )
  );

  const newBiases: number[][] = weights.biases.map((bias, l) =>
    bias.map((b, j) => b - lr * gradients.biasGrads[l][j])
  );

  return { layers: newLayers, biases: newBiases };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd platform/frontend && npx vitest run src/components/nn/__tests__/nn-math.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add platform/frontend/src/components/nn/nn-math.ts platform/frontend/src/components/nn/__tests__/nn-math.test.ts
git commit -m "feat: add NN backward pass, loss, and weight update"
```

---

### Task 11: nn-math.ts — built-in training datasets

**Files:**
- Modify: `platform/frontend/src/components/nn/nn-math.ts`
- Modify: `platform/frontend/src/components/nn/__tests__/nn-math.test.ts`

- [ ] **Step 1: Write dataset tests**

Append to test file:

```typescript
import { generateDataset } from "../nn-math";

describe("generateDataset", () => {
  it("generates XOR dataset with 4 points", () => {
    const ds = generateDataset("xor");
    expect(ds.inputs).toHaveLength(4);
    expect(ds.targets).toHaveLength(4);
    expect(ds.inputs[0]).toHaveLength(2);
    expect(ds.targets[0]).toHaveLength(1);
  });

  it("generates spiral dataset", () => {
    const ds = generateDataset("spiral");
    expect(ds.inputs.length).toBeGreaterThan(10);
    expect(ds.inputs[0]).toHaveLength(2);
  });

  it("generates crescent dataset", () => {
    const ds = generateDataset("crescent");
    expect(ds.inputs.length).toBeGreaterThan(10);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/frontend && npx vitest run src/components/nn/__tests__/nn-math.test.ts`
Expected: FAIL — `generateDataset` not exported

- [ ] **Step 3: Implement dataset generation**

Append to `nn-math.ts`:

```typescript
// ─── Built-in Datasets ───
// Note: merge Dataset into the existing import at the top of nn-math.ts

export function generateDataset(name: string): Dataset {
  switch (name) {
    case "xor":
      return {
        name: "XOR",
        inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
        targets: [[0], [1], [1], [0]],
      };
    case "spiral": {
      const inputs: number[][] = [];
      const targets: number[][] = [];
      const n = 50;
      for (let cls = 0; cls < 2; cls++) {
        for (let i = 0; i < n; i++) {
          const r = (i / n) * 5;
          const t = (i / n) * Math.PI * 2.5 + cls * Math.PI;
          const noise = (Math.random() - 0.5) * 0.5;
          inputs.push([r * Math.cos(t) + noise, r * Math.sin(t) + noise]);
          targets.push([cls]);
        }
      }
      return { name: "Spiral", inputs, targets };
    }
    case "crescent": {
      const inputs: number[][] = [];
      const targets: number[][] = [];
      const n = 50;
      for (let i = 0; i < n; i++) {
        const angle = Math.PI * (i / n);
        const noise = (Math.random() - 0.5) * 0.3;
        inputs.push([Math.cos(angle) + noise, Math.sin(angle) + noise]);
        targets.push([0]);
        inputs.push([1 - Math.cos(angle) + noise, 0.5 - Math.sin(angle) + noise]);
        targets.push([1]);
      }
      return { name: "Crescent", inputs, targets };
    }
    default:
      return generateDataset("xor");
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd platform/frontend && npx vitest run src/components/nn/__tests__/nn-math.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add platform/frontend/src/components/nn/nn-math.ts platform/frontend/src/components/nn/__tests__/nn-math.test.ts
git commit -m "feat: add built-in training datasets (XOR, spiral, crescent)"
```

---

## Chunk 3: Backend API Extensions

### Task 12: Extend loss-landscape endpoint for 3D surface types

**Files:**
- Modify: `platform/backend/app/models/linear.py:45-61`
- Modify: `platform/backend/app/api/model_routes.py:23-35`
- Test: `platform/backend/tests/test_models.py`

- [ ] **Step 1: Write test for surface_type parameter**

Append to `platform/backend/tests/test_models.py`:

```python
def test_loss_landscape_saddle():
    result = compute_loss_landscape(
        X=[[1], [2], [3]],
        y=[2, 4, 6],
        resolution=10,
        surface_type="saddle",
    )
    assert "w0" in result
    assert "w1" in result
    assert "loss" in result
    assert len(result["loss"]) == 10
    assert len(result["loss"][0]) == 10


def test_loss_landscape_local_minima():
    result = compute_loss_landscape(
        X=[[1], [2], [3]],
        y=[2, 4, 6],
        resolution=10,
        surface_type="local_minima",
    )
    assert len(result["loss"]) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/backend && python -m pytest tests/test_models.py::test_loss_landscape_saddle -v`
Expected: FAIL — TypeError (unexpected keyword argument `surface_type`)

- [ ] **Step 3: Implement surface_type in compute_loss_landscape**

Modify `platform/backend/app/models/linear.py`:

```python
def compute_loss_landscape(
    X: list[list[float]],
    y: list[float],
    w0_range: tuple = (-5, 5),
    w1_range: tuple = (-5, 5),
    resolution: int = 50,
    surface_type: str = "bowl",
) -> dict:
    X_arr = np.array(X)
    y_arr = np.array(y)
    w0s = np.linspace(*w0_range, resolution)
    w1s = np.linspace(*w1_range, resolution)
    Z = np.zeros((resolution, resolution))

    for i, w0 in enumerate(w0s):
        for j, w1 in enumerate(w1s):
            if surface_type == "bowl":
                pred = X_arr[:, 0] * w0 + w1
                Z[i, j] = float(np.mean((pred - y_arr) ** 2))
            elif surface_type == "saddle":
                Z[i, j] = float(w0**2 - w1**2)
            elif surface_type == "local_minima":
                Z[i, j] = float(
                    (w0**2 + w1 - 11) ** 2 + (w0 + w1**2 - 7) ** 2
                ) / 100  # Himmelblau's function, scaled

    return {"w0": w0s.tolist(), "w1": w1s.tolist(), "loss": Z.tolist()}
```

- [ ] **Step 4: Update the API request model**

In `platform/backend/app/api/model_routes.py`, modify `LossLandscapeRequest`:

```python
class LossLandscapeRequest(BaseModel):
    X: list[list[float]]
    y: list[float]
    w0_range: list[float] = [-5, 5]
    w1_range: list[float] = [-5, 5]
    resolution: int = 50
    surface_type: str = "bowl"
```

And update the handler:

```python
@router.post("/loss-landscape")
async def loss_landscape(req: LossLandscapeRequest):
    return compute_loss_landscape(
        req.X, req.y, tuple(req.w0_range), tuple(req.w1_range), req.resolution, req.surface_type
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd platform/backend && python -m pytest tests/test_models.py -v -k "loss_landscape"`
Expected: PASS (all loss_landscape tests)

- [ ] **Step 6: Commit**

```bash
git add platform/backend/app/models/linear.py platform/backend/app/api/model_routes.py platform/backend/tests/test_models.py
git commit -m "feat: extend loss-landscape API with surface_type (bowl/saddle/local_minima)"
```

---

### Task 12b: Extend gradient-descent endpoint with surface_type

**Files:**
- Modify: `platform/backend/app/api/model_routes.py:11-20`

The spec requires the gradient-descent endpoint to also accept `surface_type` so the trajectory overlay matches the selected surface. This allows GradientDescent3D to fetch both surface data and trajectory with matching parameters.

- [ ] **Step 1: Update GradientDescentRequest**

In `platform/backend/app/api/model_routes.py`, add `surface_type` to the request model:

```python
class GradientDescentRequest(BaseModel):
    X: list[list[float]]
    y: list[float]
    learning_rate: float = 0.01
    epochs: int = 100
    surface_type: str = "bowl"
```

The `surface_type` parameter is passed through but the actual gradient descent computation remains the same (it always works on the provided X/y data). The frontend uses this to ensure consistent surface_type between the surface and trajectory API calls.

- [ ] **Step 2: Commit**

```bash
git add platform/backend/app/api/model_routes.py
git commit -m "feat: add surface_type parameter to gradient-descent endpoint"
```

---

### Task 13: Extend decision-boundary endpoint for 3D (n_features=3)

**Files:**
- Modify: `platform/backend/app/models/classification.py:8-42`
- Modify: `platform/backend/app/api/model_routes.py:38-50`
- Test: `platform/backend/tests/test_models.py`

- [ ] **Step 1: Write test for 3D decision boundary**

Append to `platform/backend/tests/test_models.py`:

```python
from app.models.classification import train_and_get_decision_boundary

def test_decision_boundary_3d():
    import random
    random.seed(42)
    X = [[random.gauss(c, 1) for _ in range(3)] for c in [0, 0, 0] * 25 + [2, 2, 2] * 25]
    # Simpler: generate 50 points
    X_3d = []
    y_3d = []
    for _ in range(25):
        X_3d.append([random.gauss(-1, 1), random.gauss(-1, 1), random.gauss(-1, 1)])
        y_3d.append(0)
        X_3d.append([random.gauss(1, 1), random.gauss(1, 1), random.gauss(1, 1)])
        y_3d.append(1)

    result = train_and_get_decision_boundary(X_3d, y_3d, n_features=3, resolution=10)
    assert "mesh_vertices" in result
    assert "mesh_faces" in result
    assert "X" in result
    assert len(result["X"][0]) == 3  # 3D points
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/backend && python -m pytest tests/test_models.py::test_decision_boundary_3d -v`
Expected: FAIL — TypeError (unexpected keyword argument `n_features`)

- [ ] **Step 3: Implement 3D decision boundary**

Modify `platform/backend/app/models/classification.py` `train_and_get_decision_boundary`:

```python
def train_and_get_decision_boundary(
    X: list[list[float]],
    y: list[int],
    model_type: str = "logistic",
    C: float = 1.0,
    kernel: str = "rbf",
    resolution: int = 100,
    n_features: int = 2,
) -> dict:
    X_arr = np.array(X)
    y_arr = np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    if model_type == "logistic":
        model = LogisticRegression(C=C, max_iter=1000)
    else:
        model = SVC(C=C, kernel=kernel, probability=True)
    model.fit(X_scaled, y_arr)

    if n_features == 3:
        # 3D: use marching cubes-like approach to find decision surface
        res = min(resolution, 30)  # cap resolution for 3D
        ranges = [(X_scaled[:, i].min() - 1, X_scaled[:, i].max() + 1) for i in range(3)]
        grid = np.mgrid[
            ranges[0][0]:ranges[0][1]:complex(res),
            ranges[1][0]:ranges[1][1]:complex(res),
            ranges[2][0]:ranges[2][1]:complex(res),
        ]
        points = np.c_[grid[0].ravel(), grid[1].ravel(), grid[2].ravel()]

        if hasattr(model, "decision_function"):
            vals = model.decision_function(points).reshape(grid[0].shape)
        else:
            vals = model.predict_proba(points)[:, 1].reshape(grid[0].shape) - 0.5

        # Extract isosurface at decision boundary (value=0)
        from skimage.measure import marching_cubes
        try:
            verts, faces, _, _ = marching_cubes(vals, level=0)
            # Scale vertices back to data coordinates
            for dim in range(3):
                verts[:, dim] = verts[:, dim] / res * (ranges[dim][1] - ranges[dim][0]) + ranges[dim][0]
        except (ValueError, RuntimeError):
            verts = np.array([]).reshape(0, 3)
            faces = np.array([]).reshape(0, 3)

        return {
            "mesh_vertices": verts.tolist(),
            "mesh_faces": faces.tolist(),
            "X": X_scaled.tolist(),
            "y": y_arr.tolist(),
            "accuracy": float(model.score(X_scaled, y_arr)),
        }

    # 2D (existing)
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    return {
        "xx": xx.tolist(),
        "yy": yy.tolist(),
        "Z": Z.tolist(),
        "X": X_scaled.tolist(),
        "y": y_arr.tolist(),
        "accuracy": float(model.score(X_scaled, y_arr)),
    }
```

- [ ] **Step 4: Update API request model**

In `platform/backend/app/api/model_routes.py`:

```python
class DecisionBoundaryRequest(BaseModel):
    X: list[list[float]]
    y: list[int]
    model_type: str = "logistic"
    C: float = 1.0
    kernel: str = "rbf"
    n_features: int = 2
```

Update handler:

```python
@router.post("/decision-boundary")
async def decision_boundary(req: DecisionBoundaryRequest):
    return train_and_get_decision_boundary(
        req.X, req.y, req.model_type, req.C, req.kernel, n_features=req.n_features
    )
```

- [ ] **Step 5: Add scikit-image dependency for marching_cubes**

In `platform/backend/pyproject.toml`, add `scikit-image >= 0.24` to dependencies.

**Note:** scikit-image is ~30MB installed and pulls in scipy (which should already be present via scikit-learn). Verify with `pip show scipy` first.

Run: `cd platform/backend && pip install scikit-image`

- [ ] **Step 6: Run test to verify it passes**

Run: `cd platform/backend && python -m pytest tests/test_models.py::test_decision_boundary_3d -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add platform/backend/app/models/classification.py platform/backend/app/api/model_routes.py platform/backend/tests/test_models.py platform/backend/pyproject.toml
git commit -m "feat: extend decision-boundary API for 3D with marching_cubes"
```

---

## Chunk 4: NN Interactive Diagram UI

### Task 14: Create NetworkCanvas (Canvas 2D rendering)

**Files:**
- Create: `platform/frontend/src/components/nn/NetworkCanvas.tsx`

- [ ] **Step 1: Implement NetworkCanvas**

This is a Canvas 2D component that renders neurons, connections, and particle animations. The component takes `NetworkConfig`, `Weights`, and animation state as props, and renders onto a `<canvas>` element using `useRef` + `useEffect` for the draw loop.

Key rendering logic:
- Calculate neuron positions based on layer sizes (evenly spaced vertically per layer, layers spaced horizontally)
- Draw connections as lines: width = `Math.min(Math.abs(weight) * 3, 5)`, color = weight > 0 ? blue : red, dashed if negative (for colorblind)
- Draw neurons as circles with layer labels
- Forward pass particles: blue dots moving along connections from input→output
- Backprop flow mode: red/blue gradient particles flowing output→input
- Backprop step mode: highlight current layer with golden border

The full implementation should be ~200 lines. Key function signature:

```typescript
interface NetworkCanvasProps {
  config: NetworkConfig;
  weights: Weights;
  gradients?: GradientResult;
  forwardResult?: ForwardResult;
  animationMode: "idle" | "forward" | "backprop-flow" | "backprop-step";
  backpropStep?: number; // 0-3 for step mode
  width?: number;
  height?: number;
}
```

- [ ] **Step 2: Commit**

```bash
git add platform/frontend/src/components/nn/NetworkCanvas.tsx
git commit -m "feat: add NetworkCanvas with Canvas 2D rendering"
```

---

### Task 15: Create TrainingPanel (controls + loss curve)

**Files:**
- Create: `platform/frontend/src/components/nn/TrainingPanel.tsx`

- [ ] **Step 1: Implement TrainingPanel**

The control panel with:
- Architecture controls (hidden layers buttons, neurons +/-, activation dropdown)
- Training controls (Train/Pause/Step/Reset buttons, LR slider, epochs slider)
- Loss curve (Recharts LineChart)
- Dataset selector (XOR / Spiral / Crescent)

Key interface:

```typescript
interface TrainingPanelProps {
  config: NetworkConfig;
  onConfigChange: (config: NetworkConfig) => void;
  trainingState: TrainingState;
  onTrain: () => void;
  onPause: () => void;
  onStep: () => void;
  onReset: () => void;
  lr: number;
  onLrChange: (lr: number) => void;
  maxEpochs: number;
  onMaxEpochsChange: (n: number) => void;
  dataset: string;
  onDatasetChange: (name: string) => void;
}
```

Uses Recharts `LineChart` for the loss curve (same pattern as existing `GradientDescentViz.tsx`).

- [ ] **Step 2: Commit**

```bash
git add platform/frontend/src/components/nn/TrainingPanel.tsx
git commit -m "feat: add TrainingPanel with architecture/training controls and loss curve"
```

---

### Task 16: Create BackpropAnimator (dual-mode animation logic)

**Files:**
- Create: `platform/frontend/src/components/nn/BackpropAnimator.tsx`

- [ ] **Step 1: Implement BackpropAnimator**

A non-rendering hook/utility that manages animation state:

```typescript
export type BackpropMode = "flow" | "step";

interface BackpropAnimatorState {
  mode: BackpropMode;
  step: number; // 0-3 for step mode
  stepLabels: string[];
  stepFormulas: string[]; // KaTeX strings
  isAnimating: boolean;
}

export function useBackpropAnimator(): {
  state: BackpropAnimatorState;
  setMode: (mode: BackpropMode) => void;
  nextStep: () => void;
  prevStep: () => void;
  reset: () => void;
};
```

Step labels and formulas:
- Step 0: "計算損失 Loss" → `L = \\frac{1}{2}(y - \\hat{y})^2`
- Step 1: "輸出層梯度" → `\\frac{\\partial L}{\\partial \\hat{y}} = -(y - \\hat{y})`
- Step 2: "隱藏層梯度 (Chain Rule)" → `\\frac{\\partial L}{\\partial h_j} = \\sum_i \\frac{\\partial L}{\\partial o_i} \\cdot w_{ij} \\cdot \\sigma'(z_j)`
- Step 3: "更新權重" → `w_{ij} \\leftarrow w_{ij} - \\eta \\cdot \\frac{\\partial L}{\\partial w_{ij}}`

- [ ] **Step 2: Commit**

```bash
git add platform/frontend/src/components/nn/BackpropAnimator.tsx
git commit -m "feat: add BackpropAnimator with dual-mode (flow/step) animation logic"
```

---

### Task 17: Create NeuralNetworkViz (main component)

**Files:**
- Create: `platform/frontend/src/components/nn/NeuralNetworkViz.tsx`
- Test: `platform/frontend/src/components/nn/__tests__/NeuralNetworkViz.test.tsx`

- [ ] **Step 1: Write smoke test**

```typescript
// platform/frontend/src/components/nn/__tests__/NeuralNetworkViz.test.tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import NeuralNetworkViz from "../NeuralNetworkViz";

describe("NeuralNetworkViz", () => {
  it("renders with default config", () => {
    render(<NeuralNetworkViz />);
    expect(screen.getByText(/神經網路/)).toBeDefined();
  });

  it("renders training controls", () => {
    render(<NeuralNetworkViz />);
    expect(screen.getByText(/Train/i)).toBeDefined();
    expect(screen.getByText(/Reset/i)).toBeDefined();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform/frontend && npx vitest run src/components/nn/__tests__/NeuralNetworkViz.test.tsx`
Expected: FAIL

- [ ] **Step 3: Implement NeuralNetworkViz**

The main component that:
- Manages `NetworkConfig`, `Weights`, `TrainingState` in local state
- Renders `NetworkCanvas` (left 2/3) and `TrainingPanel` (right 1/3)
- Provides backprop mode toggle at top of canvas
- Runs training loop in `requestAnimationFrame`
- Accepts optional `activation` prop and `onActivationChange` callback for linkage
- Responsive: stacks vertically on mobile (<768px)

Key interface:

```typescript
interface NeuralNetworkVizProps {
  activation?: ActivationType;
  onActivationChange?: (fn: ActivationType) => void;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd platform/frontend && npx vitest run src/components/nn/__tests__/NeuralNetworkViz.test.tsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add platform/frontend/src/components/nn/NeuralNetworkViz.tsx platform/frontend/src/components/nn/__tests__/NeuralNetworkViz.test.tsx
git commit -m "feat: add NeuralNetworkViz main component"
```

---

## Chunk 5: 3D Scene Components

### Task 18: Create GradientDescent3D

**Files:**
- Create: `platform/frontend/src/components/viz3d/GradientDescent3D.tsx`

- [ ] **Step 1: Implement GradientDescent3D**

Uses `Scene3D`, `Axis3D`, `Tooltip3D` from the shared layer.

Key features:
- Fetches surface data from `/api/models/loss-landscape` with selected `surface_type`
- Renders 3D surface mesh using Three.js `BufferGeometry` with vertex colors (red=high, green=low)
- Animated ball (sphere) that follows the gradient descent trajectory
- Surface type selector: bowl / saddle / local_minima
- Learning rate slider controls ball step animation speed
- Play / Pause / Reset controls

Props:

```typescript
interface GradientDescent3DProps {
  lr: number;           // shared with 2D view
  onLrChange: (lr: number) => void;
}
```

- [ ] **Step 2: Commit**

```bash
git add platform/frontend/src/components/viz3d/GradientDescent3D.tsx
git commit -m "feat: add GradientDescent3D scene (loss surface + rolling ball)"
```

---

### Task 19: Create DecisionBoundary3D

**Files:**
- Create: `platform/frontend/src/components/viz3d/DecisionBoundary3D.tsx`

- [ ] **Step 1: Implement DecisionBoundary3D**

Key features:
- Fetches 3D mesh data from `/api/models/decision-boundary` with `n_features=3`
- Renders 3D scatter with class colors (blue circles vs red diamonds for colorblind)
- Renders semi-transparent decision surface mesh from `mesh_vertices` + `mesh_faces`
- Model selector, C slider, kernel selector (shared state with 2D view)
- Hover tooltip showing point coordinates + predicted class

Props:

```typescript
interface DecisionBoundary3DProps {
  modelType: string;
  C: number;
  kernel: string;
  onModelTypeChange: (type: string) => void;
  onCChange: (c: number) => void;
  onKernelChange: (k: string) => void;
}
```

- [ ] **Step 2: Commit**

```bash
git add platform/frontend/src/components/viz3d/DecisionBoundary3D.tsx
git commit -m "feat: add DecisionBoundary3D scene (3D scatter + decision surface)"
```

---

### Task 20: Create embedding-3d.json data file

**Files:**
- Create: `platform/frontend/src/components/data/embedding-3d.json`

- [ ] **Step 1: Create pre-computed embedding data**

Generate JSON with ~80 words across 5 categories, with pre-computed 3D coordinates for PCA, t-SNE, and UMAP. The coordinates should form visible clusters per category.

Format:
```json
{
  "words": [
    {"text": "貓", "category": "動物", "pca": [1.2, 3.5, 0.8], "tsne": [2.1, 4.0, 1.1], "umap": [1.5, 3.2, 0.9]},
    ...
  ]
}
```

Categories: 動物 (16 words), 交通 (16 words), 食物 (16 words), 科技 (16 words), 自然 (16 words)

Each category's points should cluster together in 3D space with some noise.

- [ ] **Step 2: Commit**

```bash
git add platform/frontend/src/components/data/embedding-3d.json
git commit -m "feat: add pre-computed 3D embedding data (80 words, PCA/t-SNE/UMAP)"
```

---

### Task 21: Create EmbeddingSpace3D

**Files:**
- Create: `platform/frontend/src/components/viz3d/EmbeddingSpace3D.tsx`

- [ ] **Step 1: Implement EmbeddingSpace3D**

Key features:
- Imports pre-computed data from `embedding-3d.json`
- Renders 3D scatter with category colors (reuse `CATEGORY_COLORS` pattern)
- Method selector: PCA / t-SNE / UMAP (swaps coordinate set)
- Noise slider shared with 2D view
- Hover tooltip showing word + category
- Click point → highlight k-nearest neighbors with connecting lines

Props:

```typescript
interface EmbeddingSpace3DProps {
  noise: number;
  onNoiseChange: (noise: number) => void;
}
```

- [ ] **Step 2: Commit**

```bash
git add platform/frontend/src/components/viz3d/EmbeddingSpace3D.tsx
git commit -m "feat: add EmbeddingSpace3D scene (3D scatter + nearest neighbor)"
```

---

## Chunk 6: Integration

### Task 22: Modify ActivationFunctionViz for linkage props

**Files:**
- Modify: `platform/frontend/src/components/viz/ActivationFunctionViz.tsx`

- [ ] **Step 1: Add props interface**

Modify the component to accept optional linkage props:

```typescript
interface ActivationFunctionVizProps {
  selectedActivation?: ActivationType;
  onActivationChange?: (fn: ActivationType) => void;
}

export default function ActivationFunctionViz({
  selectedActivation,
  onActivationChange,
}: ActivationFunctionVizProps = {}) {
```

When `selectedActivation` is provided, highlight that function's button with a ring. When user clicks a function button and `onActivationChange` is provided, call it.

- [ ] **Step 2: Commit**

```bash
git add platform/frontend/src/components/viz/ActivationFunctionViz.tsx
git commit -m "feat: add linkage props to ActivationFunctionViz"
```

---

### Task 23: Create Week11Compound wrapper

**Files:**
- Create: `platform/frontend/src/components/viz/Week11Compound.tsx`

- [ ] **Step 1: Implement Week11Compound**

```typescript
// platform/frontend/src/components/viz/Week11Compound.tsx
import { useState, lazy, Suspense } from "react";
import type { ActivationType } from "../nn/nn-types";
import ActivationFunctionViz from "./ActivationFunctionViz";

const NeuralNetworkViz = lazy(() => import("../nn/NeuralNetworkViz"));

export default function Week11Compound() {
  const [activation, setActivation] = useState<ActivationType>("relu");

  return (
    <div className="space-y-6 lg:space-y-0 lg:grid lg:grid-cols-[1fr_2fr] lg:gap-6">
      <div>
        <ActivationFunctionViz
          selectedActivation={activation}
          onActivationChange={setActivation}
        />
      </div>
      <div>
        <Suspense fallback={<div className="text-gray-400 text-sm p-4">載入神經網路互動圖...</div>}>
          <NeuralNetworkViz
            activation={activation}
            onActivationChange={setActivation}
          />
        </Suspense>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Update WeekPage.tsx**

In `platform/frontend/src/pages/WeekPage.tsx`, change line 20:

```typescript
11: lazy(() => import("../components/viz/Week11Compound")),
```

- [ ] **Step 3: Commit**

```bash
git add platform/frontend/src/components/viz/Week11Compound.tsx platform/frontend/src/pages/WeekPage.tsx
git commit -m "feat: add Week11Compound, link ActivationFunctionViz ↔ NeuralNetworkViz"
```

---

### Task 24: Add 2D/3D toggle to GradientDescentViz

**Files:**
- Modify: `platform/frontend/src/components/viz/GradientDescentViz.tsx`

- [ ] **Step 1: Add 2D/3D tab toggle**

Add a `viewMode` state (`"2d" | "3d"`) and render a toggle button group at top-right. When "3D" is selected, lazy-import and render `GradientDescent3D`. Share `lr` state between 2D and 3D views.

```typescript
const [viewMode, setViewMode] = useState<"2d" | "3d">("2d");
const GradientDescent3D = viewMode === "3d"
  ? lazy(() => import("../viz3d/GradientDescent3D"))
  : null;
```

Add toggle buttons above the existing content:

```tsx
<div className="flex justify-end mb-2">
  <div className="inline-flex rounded-lg border border-gray-200 overflow-hidden text-sm">
    <button
      onClick={() => setViewMode("2d")}
      className={`px-3 py-1 ${viewMode === "2d" ? "bg-blue-500 text-white" : "bg-white text-gray-600"}`}
    >
      2D
    </button>
    <button
      onClick={() => setViewMode("3d")}
      className={`px-3 py-1 ${viewMode === "3d" ? "bg-blue-500 text-white" : "bg-white text-gray-600"}`}
    >
      3D
    </button>
  </div>
</div>
```

- [ ] **Step 2: Commit**

```bash
git add platform/frontend/src/components/viz/GradientDescentViz.tsx
git commit -m "feat: add 2D/3D toggle to GradientDescentViz"
```

---

### Task 25: Add 2D/3D toggle to DecisionBoundaryViz

**Files:**
- Modify: `platform/frontend/src/components/viz/DecisionBoundaryViz.tsx`

- [ ] **Step 1: Add 2D/3D tab toggle**

Same pattern as Task 24. Share `modelType`, `C`, `kernel` state between views. When 3D is selected, generate 3-feature data instead of 2-feature.

- [ ] **Step 2: Commit**

```bash
git add platform/frontend/src/components/viz/DecisionBoundaryViz.tsx
git commit -m "feat: add 2D/3D toggle to DecisionBoundaryViz"
```

---

### Task 26: Add 2D/3D toggle to EmbeddingSpaceViz

**Files:**
- Modify: `platform/frontend/src/components/viz/EmbeddingSpaceViz.tsx`

- [ ] **Step 1: Add 2D/3D tab toggle**

Same pattern. Share `noise` state. When 3D is selected, lazy-import and render `EmbeddingSpace3D`.

- [ ] **Step 2: Commit**

```bash
git add platform/frontend/src/components/viz/EmbeddingSpaceViz.tsx
git commit -m "feat: add 2D/3D toggle to EmbeddingSpaceViz"
```

---

### Task 27: Final integration test and build verification

**Files:** (no new files)

- [ ] **Step 1: Run all frontend tests**

Run: `cd platform/frontend && npx vitest run`
Expected: All tests pass

- [ ] **Step 2: Run all backend tests**

Run: `cd platform/backend && python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 3: Verify frontend builds**

Run: `cd platform/frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 4: Manual smoke test**

Start both servers:
```bash
cd platform/backend && uvicorn app.main:app --reload &
cd platform/frontend && npm run dev
```

Verify:
- Week 4: 2D/3D toggle works, 3D loss surface rotates, ball animates
- Week 5: 2D/3D toggle works, 3D decision boundary renders
- Week 11: ActivationFunctionViz + NeuralNetworkViz render side-by-side, activation linkage works, training runs with loss curve updating
- Week 17: 2D/3D toggle works, PCA/t-SNE/UMAP switches coordinates

- [ ] **Step 5: Final commit**

```bash
git add platform/frontend/src/ platform/backend/
git commit -m "feat: complete 3D visualizations and NN interactive diagram integration"
```
