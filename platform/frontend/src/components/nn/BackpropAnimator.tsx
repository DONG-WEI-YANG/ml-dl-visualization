import { useState, useCallback } from "react";

// ─── Types & Constants ──────────────────────────────────────────────────────

export type BackpropMode = "flow" | "step";

export interface BackpropState {
  mode: BackpropMode;
  step: number; // 0-3
  isAnimating: boolean;
}

const STEP_LABELS = [
  "計算損失 Loss",
  "輸出層梯度 Output Gradients",
  "隱藏層梯度 Hidden Gradients (Chain Rule)",
  "更新權重 Weight Update",
];

const STEP_FORMULAS = [
  "L = \\frac{1}{2}(y - \\hat{y})^2",
  "\\frac{\\partial L}{\\partial \\hat{y}} = -(y - \\hat{y})",
  "\\frac{\\partial L}{\\partial h_j} = \\sum_i \\frac{\\partial L}{\\partial o_i} \\cdot w_{ij} \\cdot \\sigma'(z_j)",
  "w_{ij} \\leftarrow w_{ij} - \\eta \\cdot \\frac{\\partial L}{\\partial w_{ij}}",
];

// ─── Hook ───────────────────────────────────────────────────────────────────

export function useBackpropAnimator() {
  const [state, setState] = useState<BackpropState>({
    mode: "flow",
    step: 0,
    isAnimating: false,
  });

  const setMode = useCallback((mode: BackpropMode) => {
    setState((prev) => ({
      ...prev,
      mode,
      step: 0,
      isAnimating: mode === "flow",
    }));
  }, []);

  const nextStep = useCallback(() => {
    setState((prev) => {
      if (prev.mode !== "step") return prev;
      const next = Math.min(prev.step + 1, STEP_LABELS.length - 1);
      return { ...prev, step: next };
    });
  }, []);

  const prevStep = useCallback(() => {
    setState((prev) => {
      if (prev.mode !== "step") return prev;
      const next = Math.max(prev.step - 1, 0);
      return { ...prev, step: next };
    });
  }, []);

  const reset = useCallback(() => {
    setState({ mode: "flow", step: 0, isAnimating: false });
  }, []);

  return {
    state,
    setMode,
    nextStep,
    prevStep,
    reset,
    stepLabels: STEP_LABELS,
    stepFormulas: STEP_FORMULAS,
  };
}

// ─── BackpropOverlay Component ──────────────────────────────────────────────

interface BackpropOverlayProps {
  state: BackpropState;
  onModeChange: (mode: BackpropMode) => void;
  onNextStep: () => void;
  onPrevStep: () => void;
}

export default function BackpropOverlay({
  state,
  onModeChange,
  onNextStep,
  onPrevStep,
}: BackpropOverlayProps) {
  return (
    <div className="absolute top-2 left-2 right-2 z-10 pointer-events-none">
      {/* Mode Toggle */}
      <div className="flex gap-2 mb-2 pointer-events-auto">
        <button
          onClick={() => onModeChange("flow")}
          className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
            state.mode === "flow"
              ? "bg-blue-500 text-white shadow-sm"
              : "bg-white/90 border border-gray-300 text-gray-600 hover:bg-gray-100"
          }`}
        >
          🌊 流動模式
        </button>
        <button
          onClick={() => onModeChange("step")}
          className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
            state.mode === "step"
              ? "bg-blue-500 text-white shadow-sm"
              : "bg-white/90 border border-gray-300 text-gray-600 hover:bg-gray-100"
          }`}
        >
          📝 逐步模式
        </button>
      </div>

      {/* Step Mode UI */}
      {state.mode === "step" && (
        <div className="bg-white/95 border border-gray-200 rounded-lg p-3 shadow-sm pointer-events-auto">
          {/* Step progress bar */}
          <div className="flex gap-1 mb-2">
            {STEP_LABELS.map((_, i) => (
              <div
                key={i}
                className={`flex-1 h-1.5 rounded-full transition-colors ${
                  i <= state.step ? "bg-amber-500" : "bg-gray-200"
                }`}
              />
            ))}
          </div>

          {/* Current step label */}
          <p className="text-sm font-medium text-gray-800 mb-1">
            步驟 {state.step + 1}/{STEP_LABELS.length}:{" "}
            {STEP_LABELS[state.step]}
          </p>

          {/* Formula (rendered as text; parent can use KaTeX) */}
          <p className="text-xs text-gray-500 font-mono bg-gray-50 rounded px-2 py-1 mb-2 overflow-x-auto">
            {STEP_FORMULAS[state.step]}
          </p>

          {/* Navigation buttons */}
          <div className="flex gap-2">
            <button
              onClick={onPrevStep}
              disabled={state.step === 0}
              className="px-3 py-1 text-xs font-medium rounded bg-gray-200 text-gray-700 hover:bg-gray-300 disabled:opacity-40"
            >
              ← 上一步
            </button>
            <button
              onClick={onNextStep}
              disabled={state.step === STEP_LABELS.length - 1}
              className="px-3 py-1 text-xs font-medium rounded bg-amber-500 text-white hover:bg-amber-600 disabled:opacity-40"
            >
              下一步 →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
