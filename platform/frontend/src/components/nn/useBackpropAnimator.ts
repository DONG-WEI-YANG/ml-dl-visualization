import { useCallback, useState } from "react";
import type { BackpropMode, BackpropState } from "./BackpropAnimator";

const STEP_COUNT = 4;

export function useBackpropAnimator() {
  const [state, setState] = useState<BackpropState>({
    mode: "flow",
    step: 0,
    isAnimating: false,
  });

  const setMode = useCallback((mode: BackpropMode) => {
    setState((previous) => ({
      ...previous,
      mode,
      step: 0,
      isAnimating: mode === "flow",
    }));
  }, []);

  const nextStep = useCallback(() => {
    setState((previous) => {
      if (previous.mode !== "step") return previous;
      return { ...previous, step: Math.min(previous.step + 1, STEP_COUNT - 1) };
    });
  }, []);

  const prevStep = useCallback(() => {
    setState((previous) => {
      if (previous.mode !== "step") return previous;
      return { ...previous, step: Math.max(previous.step - 1, 0) };
    });
  }, []);

  const reset = useCallback(() => {
    setState({ mode: "flow", step: 0, isAnimating: false });
  }, []);

  return { state, setMode, nextStep, prevStep, reset };
}
