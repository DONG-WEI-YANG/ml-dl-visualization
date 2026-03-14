import { useRef, useEffect, useCallback } from "react";
import type {
  NetworkConfig,
  Weights,
  GradientResult,
  ForwardResult,
} from "./nn-types";

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

interface NeuronPos {
  x: number;
  y: number;
  layer: number;
  index: number;
}

interface Particle {
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  progress: number;
  speed: number;
}

/** Build the layer sizes array from a NetworkConfig. */
function getLayerSizes(config: NetworkConfig): number[] {
  const sizes: number[] = [config.inputSize];
  for (let i = 0; i < config.hiddenLayers; i++) {
    sizes.push(config.neuronsPerLayer);
  }
  sizes.push(config.outputSize);
  return sizes;
}

/** Get label for a layer index. */
function getLayerLabel(idx: number, total: number): string {
  if (idx === 0) return "Input";
  if (idx === total - 1) return "Output";
  return `Hidden ${idx}`;
}

export default function NetworkCanvas({
  config,
  weights,
  gradients,
  forwardResult,
  animationMode,
  backpropStep,
  width,
  height = 400,
}: NetworkCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<number>(0);
  const particlesRef = useRef<Particle[]>([]);

  // Calculate neuron positions
  const computePositions = useCallback(
    (w: number, h: number): NeuronPos[] => {
      const sizes = getLayerSizes(config);
      const positions: NeuronPos[] = [];
      const numLayers = sizes.length;
      const padX = 60;
      const padY = 40;
      const usableW = w - padX * 2;
      const usableH = h - padY * 2;

      for (let l = 0; l < numLayers; l++) {
        const x = padX + (usableW * l) / (numLayers - 1 || 1);
        const count = sizes[l];
        for (let n = 0; n < count; n++) {
          const y =
            padY + (usableH * (n + 0.5)) / count;
          positions.push({ x, y, layer: l, index: n });
        }
      }
      return positions;
    },
    [config]
  );

  // Get positions for a specific layer
  const getLayerPositions = useCallback(
    (positions: NeuronPos[], layerIdx: number): NeuronPos[] => {
      return positions.filter((p) => p.layer === layerIdx);
    },
    []
  );

  // Initialize particles for animation
  const initParticles = useCallback(
    (positions: NeuronPos[], reverse: boolean) => {
      const sizes = getLayerSizes(config);
      const particles: Particle[] = [];
      for (let l = 0; l < sizes.length - 1; l++) {
        const srcLayer = reverse ? l + 1 : l;
        const dstLayer = reverse ? l : l + 1;
        const srcPositions = getLayerPositions(positions, srcLayer);
        const dstPositions = getLayerPositions(positions, dstLayer);
        for (const src of srcPositions) {
          for (const dst of dstPositions) {
            if (Math.random() < 0.3) {
              particles.push({
                fromX: src.x,
                fromY: src.y,
                toX: dst.x,
                toY: dst.y,
                progress: Math.random(),
                speed: 0.005 + Math.random() * 0.01,
              });
            }
          }
        }
      }
      return particles;
    },
    [config, getLayerPositions]
  );

  // Main draw loop
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const effectiveWidth = width ?? container.clientWidth ?? 600;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = effectiveWidth * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${effectiveWidth}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const w = effectiveWidth;
    const h = height;
    const positions = computePositions(w, h);
    const sizes = getLayerSizes(config);

    // Initialize particles when animation starts
    if (
      animationMode === "forward" ||
      animationMode === "backprop-flow"
    ) {
      if (particlesRef.current.length === 0) {
        particlesRef.current = initParticles(
          positions,
          animationMode === "backprop-flow"
        );
      }
    } else {
      particlesRef.current = [];
    }

    const draw = () => {
      ctx.clearRect(0, 0, w, h);

      // Determine if we should dim layers (backprop-step mode)
      const stepMode = animationMode === "backprop-step";
      const highlightLayer =
        stepMode && backpropStep !== undefined
          ? mapStepToLayer(backpropStep, sizes.length)
          : -1;

      // Draw connections
      for (let l = 0; l < sizes.length - 1; l++) {
        const srcPositions = getLayerPositions(positions, l);
        const dstPositions = getLayerPositions(positions, l + 1);

        for (let i = 0; i < srcPositions.length; i++) {
          for (let j = 0; j < dstPositions.length; j++) {
            const src = srcPositions[i];
            const dst = dstPositions[j];

            // Get weight value if available
            let weight = 0;
            if (weights.layers[l] && weights.layers[l][i]) {
              weight = weights.layers[l][i][j] ?? 0;
            }

            const absWeight = Math.abs(weight);
            const lineWidth = Math.min(absWeight * 3, 5);
            const isNegative = weight < 0;

            // Dim connection in step mode if neither layer is highlighted
            let alpha = 1;
            if (stepMode && highlightLayer >= 0) {
              if (l !== highlightLayer && l + 1 !== highlightLayer) {
                alpha = 0.15;
              }
            }

            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.beginPath();
            ctx.moveTo(src.x, src.y);
            ctx.lineTo(dst.x, dst.y);
            ctx.lineWidth = Math.max(lineWidth, 0.5);
            ctx.strokeStyle = isNegative ? "#ef4444" : "#3b82f6";
            if (isNegative) {
              ctx.setLineDash([4, 4]);
            }
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();
          }
        }
      }

      // Draw particles (forward/backprop-flow animation)
      if (
        animationMode === "forward" ||
        animationMode === "backprop-flow"
      ) {
        const color = animationMode === "forward" ? "#3b82f6" : "#ef4444";
        for (const p of particlesRef.current) {
          p.progress += p.speed;
          if (p.progress > 1) p.progress -= 1;

          const px = p.fromX + (p.toX - p.fromX) * p.progress;
          const py = p.fromY + (p.toY - p.fromY) * p.progress;

          ctx.beginPath();
          ctx.arc(px, py, 3, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
        }
      }

      // Draw neurons
      const neuronRadius = 16;
      for (let l = 0; l < sizes.length; l++) {
        const layerPositions = getLayerPositions(positions, l);

        for (const pos of layerPositions) {
          // Dim neuron in step mode if not highlighted layer
          let alpha = 1;
          if (stepMode && highlightLayer >= 0 && l !== highlightLayer) {
            alpha = 0.25;
          }

          ctx.save();
          ctx.globalAlpha = alpha;

          // Neuron fill
          let fillColor = "#ffffff";
          if (forwardResult && l > 0) {
            const actIdx = l - 1;
            if (
              forwardResult.activations[actIdx] &&
              forwardResult.activations[actIdx][pos.index] !== undefined
            ) {
              const val = forwardResult.activations[actIdx][pos.index];
              const intensity = Math.min(Math.max(val, 0), 1);
              const g = Math.round(200 - intensity * 150);
              const b = Math.round(220 + intensity * 35);
              fillColor = `rgb(${g}, ${g}, ${b})`;
            }
          }

          ctx.beginPath();
          ctx.arc(pos.x, pos.y, neuronRadius, 0, Math.PI * 2);
          ctx.fillStyle = fillColor;
          ctx.fill();

          // Neuron border
          if (stepMode && highlightLayer >= 0 && l === highlightLayer) {
            ctx.lineWidth = 3;
            ctx.strokeStyle = "#f59e0b"; // golden highlight
          } else {
            ctx.lineWidth = 2;
            ctx.strokeStyle = "#6b7280";
          }
          ctx.stroke();

          // Gradient magnitude indicator
          if (gradients && l > 0) {
            const gradIdx = l - 1;
            if (
              gradients.layerGrads[gradIdx] &&
              gradients.layerGrads[gradIdx][pos.index] !== undefined
            ) {
              const grad = Math.abs(
                gradients.layerGrads[gradIdx][pos.index]
              );
              if (grad > 0.001) {
                const r = Math.min(grad * 8, neuronRadius - 2);
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
                ctx.fillStyle = "rgba(239, 68, 68, 0.3)";
                ctx.fill();
              }
            }
          }

          ctx.restore();
        }
      }

      // Draw layer labels
      ctx.save();
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillStyle = "#6b7280";
      for (let l = 0; l < sizes.length; l++) {
        const layerPositions = getLayerPositions(positions, l);
        if (layerPositions.length > 0) {
          const x = layerPositions[0].x;
          ctx.fillText(
            getLayerLabel(l, sizes.length),
            x,
            h - 8
          );
        }
      }
      ctx.restore();

      // Continue animation loop
      if (
        animationMode === "forward" ||
        animationMode === "backprop-flow"
      ) {
        animRef.current = requestAnimationFrame(draw);
      }
    };

    draw();

    return () => {
      if (animRef.current) {
        cancelAnimationFrame(animRef.current);
        animRef.current = 0;
      }
    };
  }, [
    config,
    weights,
    gradients,
    forwardResult,
    animationMode,
    backpropStep,
    width,
    height,
    computePositions,
    getLayerPositions,
    initParticles,
  ]);

  return (
    <div ref={containerRef} className="w-full">
      <canvas
        ref={canvasRef}
        className="w-full rounded-lg border border-gray-200 bg-white"
        style={{ height: `${height}px` }}
      />
    </div>
  );
}

/**
 * Map backprop step (0-3) to a layer index for highlighting.
 * Step 0: output layer (loss computation)
 * Step 1: output layer (output gradients)
 * Step 2: last hidden layer (chain rule)
 * Step 3: all layers (weight update) -> highlight first hidden
 */
function mapStepToLayer(step: number, numLayers: number): number {
  switch (step) {
    case 0:
      return numLayers - 1; // output
    case 1:
      return numLayers - 1; // output gradients
    case 2:
      return Math.max(1, numLayers - 2); // last hidden
    case 3:
      return 1; // first hidden (weight update)
    default:
      return -1;
  }
}
