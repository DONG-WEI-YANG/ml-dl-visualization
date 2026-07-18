# 3D Visualizations & NN Interactive Diagram Design

**Date**: 2026-03-14
**Author**: Claude (brainstorming with 楊東偉)
**Status**: Approved

## Summary

Add 4 new visualization components to the ML/DL teaching platform:
- 3 × 3D scene visualizations (Weeks 4, 5/6, 17)
- 1 × Neural Network interactive diagram (Week 11)

All 3D rendering uses **Three.js via react-three-fiber** as the unified 3D engine.

## Decisions Log

| Decision | Choice | Alternatives Considered |
|----------|--------|------------------------|
| Which weeks get 3D | W4, W5/6, W17 | All weeks considered |
| NN interaction level | B+C (adjustable + training viz) | A (static + animation only) |
| Backprop animation | Dual mode (flow + step-by-step toggle) | Flow only, Step only |
| 3D library | Three.js (react-three-fiber) for all | Plotly, Mixed strategy |
| Architecture | Shared 3D base layer (viz3d/) | Flat independent, Plugin system |
| NN placement | Separate component in W11, linked to ActivationFunctionViz | Integrated into existing viz |

## New Dependencies

```
@react-three/fiber    — React renderer for Three.js
@react-three/drei     — Helpers (OrbitControls, Text, Html, etc.)
three                 — Three.js core
@types/three          — TypeScript definitions
```

## Dependency Cleanup

Remove unused `d3` and `@types/d3` from package.json (added early but never used; Three.js/R3F replaces the need).

## Bundle Size Estimate

- `three`: ~150KB gzip
- `@react-three/fiber`: ~30KB gzip
- `@react-three/drei` (tree-shaken, only OrbitControls/Text/Html): ~20KB gzip
- **Total**: ~200KB gzip, fully mitigated by `React.lazy` dynamic imports (3D components only load when their tab/week is visited)

## File Structure

```
platform/frontend/src/components/
├── viz3d/                          ← NEW directory
│   ├── Scene3D.tsx                 ← Shared: Canvas + Camera + Lights + OrbitControls + WebGL fallback
│   ├── Axis3D.tsx                  ← Shared: 3D axes + ticks + labels (drei Text)
│   ├── Tooltip3D.tsx               ← Shared: hover HTML overlay (drei Html)
│   ├── GradientDescent3D.tsx       ← Scene: loss surface + rolling ball animation
│   ├── DecisionBoundary3D.tsx      ← Scene: 3D scatter + classification surface
│   └── EmbeddingSpace3D.tsx        ← Scene: 3D scatter clusters + nearest neighbor
├── nn/                             ← NEW directory
│   ├── NeuralNetworkViz.tsx        ← Main: layout + shared state management
│   ├── NetworkCanvas.tsx           ← Canvas 2D: neurons + connections + particle animation
│   ├── TrainingPanel.tsx           ← Right panel: architecture + training controls + loss curve
│   ├── BackpropAnimator.tsx        ← Dual-mode backprop animation logic
│   ├── nn-math.ts                  ← Pure functions: forward/backward pass, weight update
│   └── nn-types.ts                 ← TypeScript interfaces for NN components
├── viz/
│   ├── ActivationFunctionViz.tsx   ← MODIFY: add props for bidirectional linkage
│   ├── GradientDescentViz.tsx      ← MODIFY: add 2D/3D tab toggle
│   ├── DecisionBoundaryViz.tsx     ← MODIFY: add 2D/3D tab toggle
│   ├── EmbeddingSpaceViz.tsx       ← MODIFY: add 2D/3D tab toggle
│   └── Week11Compound.tsx          ← NEW: compound wrapper for ActivationFunctionViz + NeuralNetworkViz
├── data/
│   └── embedding-3d.json           ← NEW: pre-computed 3D coords (PCA/t-SNE/UMAP × 80 words)
```

## TypeScript Interfaces

```typescript
// === viz3d/types.ts ===
type Vec3 = [number, number, number];

interface Scene3DProps {
  children: React.ReactNode;
  cameraPosition?: Vec3;          // default: [5, 5, 5]
  showGrid?: boolean;             // default: false
  backgroundColor?: string;       // default: app dark theme (#0f172a)
  enableDamping?: boolean;        // default: true
  style?: React.CSSProperties;
  className?: string;
  fallback?: React.ReactNode;     // shown when WebGL unavailable
}

interface Axis3DProps {
  labels?: { x: string; y: string; z: string };
  range?: { x: Vec3Pair; y: Vec3Pair; z: Vec3Pair }; // default: [-5, 5] per axis
  showTicks?: boolean;            // default: true
  showGrid?: boolean;             // default: false
  tickCount?: number;             // default: 5
}
type Vec3Pair = [number, number];

interface Tooltip3DProps {
  position: Vec3;
  visible: boolean;
  children: React.ReactNode;
}

// === nn/nn-types.ts ===
type ActivationType = 'sigmoid' | 'tanh' | 'relu' | 'leaky_relu' | 'gelu';

interface NetworkConfig {
  inputSize: number;              // fixed: matches dataset features
  hiddenLayers: number;           // 1~4
  neuronsPerLayer: number;        // 2~8
  outputSize: number;             // fixed: 1 (regression) or n (classification)
  activation: ActivationType;
}

interface Weights {
  layers: Matrix[];               // layers[i] = weight matrix between layer i and i+1
  biases: number[][];             // biases[i] = bias vector for layer i+1
}
type Matrix = number[][];

interface ForwardResult {
  activations: number[][];        // activations[i] = output of layer i (after activation)
  preActivations: number[][];     // before activation (for gradient computation)
}

interface GradientResult {
  weightGrads: Matrix[];          // gradient for each weight matrix
  biasGrads: number[][];          // gradient for each bias vector
  layerGrads: number[][];         // gradient at each layer (for visualization)
}

interface TrainingState {
  epoch: number;
  lossHistory: number[];
  isTraining: boolean;
  weights: Weights;
}

// === ActivationFunctionViz linkage ===
interface ActivationFunctionVizProps {
  selectedActivation?: ActivationType;           // controlled: highlight this function
  onActivationChange?: (fn: ActivationType) => void;  // callback when user clicks a function
}
```

## Component 1: Scene3D (Shared Base)

Wraps `@react-three/fiber` Canvas with consistent defaults for all 3D scenes.

**Props:** See `Scene3DProps` above.

**Provides:**
- PerspectiveCamera with configurable initial position
- OrbitControls: mouse drag=rotate, scroll=zoom, right-click=pan; touch: 1-finger=rotate, 2-finger=zoom, 3-finger=pan
- Ambient light (intensity 0.4) + Directional light (intensity 0.8)
- Consistent background color matching the app theme

**WebGL Fallback:** On mount, checks `document.createElement('canvas').getContext('webgl')`. If unavailable, renders `fallback` prop (defaults to a message "您的瀏覽器不支援 WebGL，請改用 2D 模式" with a link to switch to the 2D tab). This is built into Scene3D so all 3D scenes get it for free.

## Component 2: Axis3D (Shared)

**Props:** See `Axis3DProps` above.

Default range `[-5, 5]` per axis. Each 3D scene passes data-appropriate ranges.

## Component 3: Tooltip3D (Shared)

**Props:** See `Tooltip3DProps` above.

Uses `drei Html` for HTML overlay anchored to 3D world position.

## Component 4: GradientDescent3D (Week 4)

**Teaching goal:** Students "see" the loss function terrain and understand gradient descent as finding the lowest point on a mountain.

**Features:**
- 3D surface mesh colored by height (red=high loss, blue/green=low loss)
- Animated ball rolling down the surface following gradient
- Real-time 2D loss curve panel (Recharts LineChart) alongside the 3D view

**Interactive controls:**
- Mouse drag to rotate 3D surface (OrbitControls)
- Learning rate slider → ball step size
- Play / Pause / Reset animation
- Surface type selector: bowl, saddle point, local minima

**Data source:**
- Extend existing `/api/models/loss-landscape` endpoint (NOT `/gradient-descent` — the codebase already has a separate `compute_loss_landscape` function)
- Add `surface_type` parameter: "bowl" | "saddle" | "local_minima"
- Add `resolution` parameter (default 50)
- Response: `{ x: number[], y: number[], z: number[][], trajectory: {x: number, y: number, z: number}[] }`
- Trajectory overlay computed by `/api/models/gradient-descent` with matching surface type

## Component 5: DecisionBoundary3D (Week 5/6)

**Teaching goal:** Understand that the decision boundary in 3-feature space is a surface (not a line).

**Features:**
- 3D scatter plot with color-coded classes
- Semi-transparent decision surface mesh
- Two-tone surface coloring (one color per side/class)

**Interactive controls:**
- Mouse drag to rotate 3D space
- Model selector: Logistic Regression / SVM-linear / SVM-RBF
- SVM kernel parameter sliders (C, gamma)
- Hover data point → show coordinates + predicted class

**Data source:**
- Extend existing `/api/models/decision-boundary` endpoint
- Add `n_features=3` parameter
- Return 3D mesh vertices for the decision surface

## Component 6: EmbeddingSpace3D (Week 17)

**Teaching goal:** Students see that semantically similar words/sentences are close in embedding space.

**Features:**
- 3D scatter plot with cluster coloring
- Hover to display word/sentence and cluster label
- Click point → highlight k-nearest neighbors with connecting lines
- Dimensionality reduction method toggle: PCA / t-SNE / UMAP

**Interactive controls:**
- Mouse drag to rotate/zoom
- Noise slider (same as existing 2D version)
- Click point → nearest neighbor highlighting
- Method selector: PCA / t-SNE / UMAP

**Data source:**
- Pre-computed embedding data in `components/data/embedding-3d.json`
- Contains ~80 words across 5 categories (動物, 交通, 食物, 科技, 自然)
- 3 sets of 3D coordinates pre-reduced: PCA, t-SNE, UMAP (avoids needing a client-side DR library)
- Format: `{ words: [{text, category, pca: [x,y,z], tsne: [x,y,z], umap: [x,y,z]}] }`
- No backend API needed; switching method just swaps which coordinate set to render

## Component 7: NeuralNetworkViz (Week 11)

**Teaching goal:** Students build intuition for neural network architecture and training by interactively adjusting and observing.

### Layout
- **Left (2/3 width):** Network canvas — neurons, weighted connections, particle animations
- **Right (1/3 width):** Control panel — architecture settings, training controls, loss curve

### Architecture Controls (B features)
- Hidden layers: 1~4 layers (button toggle)
- Neurons per layer: 2~8 (increment/decrement buttons)
- Activation function: dropdown, bidirectionally linked with ActivationFunctionViz
- Weight visualization: line thickness + color (blue=positive, red=negative)
- Forward pass animation: blue particles flowing along connections
- Random initialization button

### Training Controls (C features)
- Train / Pause / Step (single epoch) / Reset buttons
- Learning rate slider
- Max epochs slider
- Real-time loss curve (Recharts LineChart)
- Weights update dynamically during training (connection lines animate)

### Backpropagation Dual Mode
- **Flow mode (🌊):** Reverse gradient particles + color gradient (red=large gradient, blue=small). Provides intuitive "whole picture" understanding.
- **Step-by-step mode (📝):** Sequential walkthrough: ① Compute loss → ② Output layer gradients → ③ Hidden layer gradients (chain rule) → ④ Weight update. Each step pauses with explanation panel showing the math.
- Toggle switch between modes at top of canvas

### Linkage with ActivationFunctionViz

**Mechanism:** A new `Week11Compound.tsx` component wraps both `ActivationFunctionViz` and `NeuralNetworkViz`. It owns the shared state (`selectedActivation`) and passes it down as props:
- `ActivationFunctionViz` receives `selectedActivation` and `onActivationChange` props
- `NeuralNetworkViz` receives `activation` from its `NetworkConfig` and updates via its own controls
- `Week11Compound` keeps both in sync

In `WeekPage.tsx`, Week 11's entry in `weekComponents` changes from `ActivationFunctionViz` to `Week11Compound`:
```
11: lazy(() => import("../components/viz/Week11Compound")),
```

**Behavior:**
- NN selects activation → ActivationFunctionViz highlights the corresponding function curve
- ActivationFunctionViz clicks a function → NN switches its activation
- Neurons display small activation icon inside when activation changes

### Training Data
- Built-in datasets: XOR, spiral, crescent moon shapes
- Pure frontend JS matrix operations for forward/backward pass
- Max network size: 4 hidden layers × 8 neurons (manageable for browser computation)

## Component 8: NetworkCanvas.tsx — Canvas 2D Rendering

Uses HTML `<canvas>` element with 2D rendering context (NOT Three.js Canvas, NOT SVG).

**Why Canvas 2D over SVG:** Particle animations for forward pass and backprop require many small moving elements per frame. Canvas 2D is more performant for this use case (no DOM node per particle). The rest of the codebase uses SVG via Recharts, but NetworkCanvas is a special case with continuous animation.

**Renders:**
- Neuron circles with layer labels
- Weighted connection lines (thickness = |weight|, color = sign)
- Forward pass: blue particles flowing input→output
- Backprop flow mode: red/blue gradient particles flowing output→input
- Backprop step mode: highlighted layer + explanation overlay (HTML positioned over canvas)

## Component 9: nn-math.ts (Pure Computation)

Pure functions, no UI. Types defined in `nn-types.ts` (see TypeScript Interfaces section).

- `initWeights(config: NetworkConfig): Weights` — Xavier/He initialization
- `forward(input: number[], weights: Weights, activation: ActivationType): ForwardResult`
- `backward(forwardResult: ForwardResult, target: number[], lr: number): GradientResult`
- `updateWeights(weights: Weights, gradients: GradientResult): Weights`
- Matrix multiplication, activation computation, gradient computation

## Backend Changes

### Extended endpoints

**`/api/models/loss-landscape`** — add parameters:
- `surface_type: str` — "bowl" | "saddle" | "local_minima" (default: "bowl")
- `resolution: int` — grid resolution (default 50)
- Response: `{ x: list[float], y: list[float], z: list[list[float]] }`

**`/api/models/gradient-descent`** — add parameter:
- `surface_type: str` — must match the loss-landscape surface_type for trajectory overlay

**`/api/models/decision-boundary`** — add parameters:
- `n_features: int` — 2 (existing) or 3 (new)
- Response adds for 3D: `mesh_vertices: list`, `mesh_faces: list`

## Integration with Existing Pages

### 2D/3D Tab Toggle (Weeks 4, 5/6, 17)

Each viz component that gains a 3D mode adds an internal tab toggle:
- **UI pattern:** Two-button group at top-right of the viz component: `[2D] [3D]`
- **Default tab:** 2D (familiar, works everywhere)
- **Lazy loading:** 3D component is `React.lazy` imported only when 3D tab is selected
- **Shared state:** 2D and 3D views share control state (learning rate, model type, etc.) via the parent viz component's state
- **WebGL fallback:** If Scene3D detects no WebGL, 3D tab shows fallback message

### WeekPage.tsx changes:
- Week 4: `GradientDescentViz` internally manages 2D/3D tab, lazy-imports `GradientDescent3D`
- Week 5/6: `DecisionBoundaryViz` internally manages 2D/3D tab, lazy-imports `DecisionBoundary3D`
- Week 11: Change `weekComponents[11]` to `Week11Compound` (ActivationFunctionViz + NeuralNetworkViz)
- Week 17: `EmbeddingSpaceViz` internally manages 2D/3D tab, lazy-imports `EmbeddingSpace3D`

### Responsive Layout

- **3D scenes (Weeks 4, 5/6, 17):** Full width, minimum height 400px. OrbitControls touch events work natively.
- **Week 11 NN diagram:** Desktop (≥1024px): side-by-side ActivationFunctionViz + NeuralNetworkViz. Tablet/Mobile (<1024px): stacked vertically, NeuralNetworkViz control panel collapses to horizontal row above canvas.
- **NeuralNetworkViz internal layout:** Desktop (≥768px): 2/3 canvas + 1/3 panel. Mobile (<768px): full-width canvas, panel below as collapsible accordion.

### Backprop Step-by-Step Math Display

Uses KaTeX (already in project as `rehype-katex`) for rendering formulas. Each step shows:
1. Loss: `L = ½(y - ŷ)²`
2. Output gradient: `∂L/∂ŷ = -(y - ŷ)`
3. Hidden gradient: `∂L/∂hⱼ = Σᵢ (∂L/∂oᵢ × wᵢⱼ) × σ'(zⱼ)`
4. Weight update: `wᵢⱼ ← wᵢⱼ - η × ∂L/∂wᵢⱼ`

Explanation panel positioned as a floating card below the current highlighted layer.

### Accessibility

- All 3D canvases include `aria-label` describing the visualization content
- Weight colors (blue=positive, red=negative) also use line style (solid=positive, dashed=negative) for colorblind users
- Class scatter colors use both color and shape (circle vs diamond) for colorblind friendliness
- 3D scenes have keyboard shortcuts: arrow keys for rotation, +/- for zoom

## Testing Strategy

- **viz3d/ components:** Snapshot tests + visual regression (Chromatic/Storybook optional)
- **nn-math.ts:** Unit tests with known inputs/outputs for forward pass, backward pass, weight update
- **NeuralNetworkViz:** Integration test — render, change architecture, verify neurons update
- **3D scenes:** Smoke test — render without crash, verify OrbitControls present

## Performance Considerations

- Three.js total bundle ~200KB gzip. Mitigated by `React.lazy` — 3D components only load when 3D tab is selected
- NN training runs in `requestAnimationFrame` loop, yields to main thread between epochs
- Max 8 neurons × 4 layers = 32 neurons, ~200 connections — trivial for Canvas 2D rendering
- 3D surface grids capped at 50×50 = 2500 vertices — smooth on any modern browser
- If NN training causes frame drops on lower-end devices (nursing college PCs), future optimization: move `nn-math.ts` computation to a Web Worker. Not needed for initial implementation given the small max network size.
