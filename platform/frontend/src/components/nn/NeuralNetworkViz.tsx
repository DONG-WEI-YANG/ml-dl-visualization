import { useState, useRef, useCallback, useEffect } from "react";
import type {
  ActivationType,
  NetworkConfig,
  Weights,
  TrainingState,
  ForwardResult,
  GradientResult,
} from "./nn-types";
import {
  initWeights,
  forward,
  computeLoss,
  backward,
  updateWeights,
  generateDataset,
} from "./nn-math";
import NetworkCanvas from "./NetworkCanvas";
import TrainingPanel from "./TrainingPanel";
import BackpropOverlay, { useBackpropAnimator } from "./BackpropAnimator";

// ─── Props ──────────────────────────────────────────────────────────────────

interface NeuralNetworkVizProps {
  activation?: ActivationType;
  onActivationChange?: (fn: ActivationType) => void;
}

// ─── Component ──────────────────────────────────────────────────────────────

export default function NeuralNetworkViz({
  activation,
  onActivationChange,
}: NeuralNetworkVizProps) {
  // Network config
  const [config, setConfig] = useState<NetworkConfig>({
    inputSize: 2,
    hiddenLayers: 2,
    neuronsPerLayer: 4,
    outputSize: 1,
    activation: activation ?? "relu",
  });

  // Sync activation prop -> config
  useEffect(() => {
    if (activation && activation !== config.activation) {
      setConfig((prev) => ({ ...prev, activation }));
    }
    // Only react to external prop changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activation]);

  // Weights
  const [weights, setWeights] = useState<Weights>(() => initWeights(config));

  // Training state
  const [trainingState, setTrainingState] = useState<TrainingState>({
    epoch: 0,
    lossHistory: [],
    isTraining: false,
    weights,
  });

  // Training parameters
  const [lr, setLr] = useState(0.1);
  const [maxEpochs, setMaxEpochs] = useState(100);
  const [dataset, setDataset] = useState("xor");

  // Forward/gradient results for visualization
  const [forwardResult, setForwardResult] = useState<ForwardResult | undefined>();
  const [gradients, setGradients] = useState<GradientResult | undefined>();

  // Animation references
  const rafRef = useRef<number>(0);
  const isTrainingRef = useRef(false);
  const weightsRef = useRef(weights);
  const epochRef = useRef(0);
  const lossHistoryRef = useRef<number[]>([]);

  // Backprop animator
  const backprop = useBackpropAnimator();

  // Keep weights ref in sync
  useEffect(() => {
    weightsRef.current = weights;
  }, [weights]);

  // Handle config change: re-init weights
  const handleConfigChange = useCallback(
    (newConfig: NetworkConfig) => {
      setConfig(newConfig);

      // Sync activation prop
      if (onActivationChange && newConfig.activation !== config.activation) {
        onActivationChange(newConfig.activation);
      }

      // Re-init weights for the new architecture
      const newWeights = initWeights(newConfig);
      setWeights(newWeights);
      weightsRef.current = newWeights;
      epochRef.current = 0;
      lossHistoryRef.current = [];
      setTrainingState({
        epoch: 0,
        lossHistory: [],
        isTraining: false,
        weights: newWeights,
      });
      setForwardResult(undefined);
      setGradients(undefined);
    },
    [config.activation, onActivationChange]
  );

  // Run one training epoch over the full dataset
  const runEpoch = useCallback(() => {
    const ds = generateDataset(dataset);
    let currentWeights = weightsRef.current;
    let totalLoss = 0;

    for (let i = 0; i < ds.inputs.length; i++) {
      const input = ds.inputs[i];
      const target = ds.targets[i];

      const fwd = forward(input, currentWeights, config.activation);
      const predicted = fwd.activations[fwd.activations.length - 1];
      totalLoss += computeLoss(predicted, target);

      const grads = backward(fwd, currentWeights, target, config.activation, input);
      currentWeights = updateWeights(currentWeights, grads, lr);

      // Save last forward/gradient for visualization
      if (i === ds.inputs.length - 1) {
        setForwardResult(fwd);
        setGradients(grads);
      }
    }

    const avgLoss = totalLoss / ds.inputs.length;
    weightsRef.current = currentWeights;
    epochRef.current += 1;
    lossHistoryRef.current = [...lossHistoryRef.current, avgLoss];

    setWeights(currentWeights);
    setTrainingState({
      epoch: epochRef.current,
      lossHistory: lossHistoryRef.current,
      isTraining: isTrainingRef.current,
      weights: currentWeights,
    });
  }, [config.activation, dataset, lr]);

  // Training loop with rAF
  const trainingLoop = useCallback(() => {
    if (!isTrainingRef.current) return;
    if (epochRef.current >= maxEpochs) {
      isTrainingRef.current = false;
      setTrainingState((prev) => ({ ...prev, isTraining: false }));
      return;
    }

    runEpoch();
    rafRef.current = requestAnimationFrame(trainingLoop);
  }, [maxEpochs, runEpoch]);

  // Train button
  const handleTrain = useCallback(() => {
    isTrainingRef.current = true;
    setTrainingState((prev) => ({ ...prev, isTraining: true }));
    rafRef.current = requestAnimationFrame(trainingLoop);
  }, [trainingLoop]);

  // Pause button
  const handlePause = useCallback(() => {
    isTrainingRef.current = false;
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    }
    setTrainingState((prev) => ({ ...prev, isTraining: false }));
  }, []);

  // Step button
  const handleStep = useCallback(() => {
    runEpoch();
  }, [runEpoch]);

  // Reset button
  const handleReset = useCallback(() => {
    isTrainingRef.current = false;
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    }

    const newWeights = initWeights(config);
    setWeights(newWeights);
    weightsRef.current = newWeights;
    epochRef.current = 0;
    lossHistoryRef.current = [];
    setTrainingState({
      epoch: 0,
      lossHistory: [],
      isTraining: false,
      weights: newWeights,
    });
    setForwardResult(undefined);
    setGradients(undefined);
    backprop.reset();
  }, [config, backprop]);

  // Dataset change: reset training
  const handleDatasetChange = useCallback(
    (name: string) => {
      setDataset(name);
      // Reset training when dataset changes
      handleReset();
    },
    [handleReset]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, []);

  // Determine animation mode for canvas
  const getAnimationMode = (): "idle" | "forward" | "backprop-flow" | "backprop-step" => {
    if (trainingState.isTraining) return "forward";
    if (backprop.state.mode === "flow" && backprop.state.isAnimating)
      return "backprop-flow";
    if (backprop.state.mode === "step") return "backprop-step";
    return "idle";
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">
        互動式神經網路 Interactive Neural Network
      </h3>

      <div className="flex flex-col lg:flex-row gap-4">
        {/* Left: Canvas + Backprop overlay */}
        <div className="flex-1 lg:w-2/3 relative">
          <NetworkCanvas
            config={config}
            weights={weights}
            gradients={gradients}
            forwardResult={forwardResult}
            animationMode={getAnimationMode()}
            backpropStep={backprop.state.step}
            height={400}
          />
          {/* Backprop overlay on top of canvas */}
          {!trainingState.isTraining && (
            <BackpropOverlay
              state={backprop.state}
              onModeChange={backprop.setMode}
              onNextStep={backprop.nextStep}
              onPrevStep={backprop.prevStep}
            />
          )}
        </div>

        {/* Right: Training panel */}
        <div className="lg:w-1/3">
          <TrainingPanel
            config={config}
            onConfigChange={handleConfigChange}
            trainingState={trainingState}
            onTrain={handleTrain}
            onPause={handlePause}
            onStep={handleStep}
            onReset={handleReset}
            lr={lr}
            onLrChange={setLr}
            maxEpochs={maxEpochs}
            onMaxEpochsChange={setMaxEpochs}
            dataset={dataset}
            onDatasetChange={handleDatasetChange}
          />
        </div>
      </div>
    </div>
  );
}
