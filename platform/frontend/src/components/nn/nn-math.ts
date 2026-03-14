import type {
  ActivationType,
  Matrix,
  NetworkConfig,
  Weights,
  ForwardResult,
  GradientResult,
  Dataset,
} from "./nn-types";

// ─── Activation Functions ───────────────────────────────────────────────────

/**
 * Apply activation function to a single value.
 */
export function activate(x: number, type: ActivationType): number {
  switch (type) {
    case "sigmoid":
      return 1 / (1 + Math.exp(-x));
    case "tanh":
      return Math.tanh(x);
    case "relu":
      return Math.max(0, x);
    case "leaky_relu":
      return x >= 0 ? x : 0.01 * x;
    case "gelu": {
      // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      const c = Math.sqrt(2 / Math.PI);
      return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
    }
  }
}

/**
 * Derivative of the activation function applied to the pre-activation value.
 */
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
    case "relu":
      return x > 0 ? 1 : 0;
    case "leaky_relu":
      return x >= 0 ? 1 : 0.01;
    case "gelu": {
      // Numerical derivative of GELU approximation
      const c = Math.sqrt(2 / Math.PI);
      const inner = c * (x + 0.044715 * x * x * x);
      const tanhVal = Math.tanh(inner);
      const sech2 = 1 - tanhVal * tanhVal;
      const innerDeriv = c * (1 + 3 * 0.044715 * x * x);
      return 0.5 * (1 + tanhVal) + 0.5 * x * sech2 * innerDeriv;
    }
  }
}

// ─── Weight Initialization ──────────────────────────────────────────────────

/**
 * Xavier uniform initialization.
 * For a layer with fanIn inputs and fanOut outputs:
 *   limit = sqrt(6 / (fanIn + fanOut))
 *   weights ~ Uniform(-limit, limit)
 */
function xavierUniform(rows: number, cols: number): Matrix {
  const limit = Math.sqrt(6 / (rows + cols));
  const matrix: Matrix = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      row.push((Math.random() * 2 - 1) * limit);
    }
    matrix.push(row);
  }
  return matrix;
}

/**
 * Initialize network weights using Xavier uniform initialization.
 */
export function initWeights(config: NetworkConfig): Weights {
  const layers: Matrix[] = [];
  const biases: number[][] = [];

  // Build layer sizes: [inputSize, neurons, neurons, ..., outputSize]
  const sizes: number[] = [config.inputSize];
  for (let i = 0; i < config.hiddenLayers; i++) {
    sizes.push(config.neuronsPerLayer);
  }
  sizes.push(config.outputSize);

  // Create weight matrices and bias vectors between consecutive layers
  for (let i = 0; i < sizes.length - 1; i++) {
    const fanIn = sizes[i];
    const fanOut = sizes[i + 1];
    layers.push(xavierUniform(fanIn, fanOut));
    biases.push(new Array(fanOut).fill(0));
  }

  return { layers, biases };
}

// ─── Forward Pass ───────────────────────────────────────────────────────────

/**
 * Forward pass through the network.
 * Returns activations and pre-activations for every layer (hidden + output).
 */
export function forward(
  input: number[],
  weights: Weights,
  activation: ActivationType
): ForwardResult {
  const activations: number[][] = [];
  const preActivations: number[][] = [];

  let current = input;

  for (let l = 0; l < weights.layers.length; l++) {
    const W = weights.layers[l];
    const b = weights.biases[l];
    const outSize = b.length;
    const pre: number[] = new Array(outSize).fill(0);

    // z = current * W + b
    for (let j = 0; j < outSize; j++) {
      let sum = b[j];
      for (let i = 0; i < current.length; i++) {
        sum += current[i] * W[i][j];
      }
      pre[j] = sum;
    }

    preActivations.push(pre);

    // Apply activation; for output layer use sigmoid for bounded output
    const isOutputLayer = l === weights.layers.length - 1;
    const actType = isOutputLayer ? "sigmoid" : activation;
    const act = pre.map((z) => activate(z, actType));
    activations.push(act);

    current = act;
  }

  return { activations, preActivations };
}

// ─── Loss ───────────────────────────────────────────────────────────────────

/**
 * Mean squared error: sum((predicted - target)^2) / (2 * n)
 */
export function computeLoss(predicted: number[], target: number[]): number {
  const n = predicted.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const diff = predicted[i] - target[i];
    sum += diff * diff;
  }
  return sum / (2 * n);
}

// ─── Backward Pass ──────────────────────────────────────────────────────────

/**
 * Backpropagation through the network.
 * Computes gradients for all weights, biases, and layer deltas.
 *
 * @param fwd       - ForwardResult from forward()
 * @param weights   - Current network weights
 * @param target    - Target output
 * @param activation - Activation function type used in hidden layers
 * @param input     - Original network input
 */
export function backward(
  fwd: ForwardResult,
  weights: Weights,
  target: number[],
  activation: ActivationType,
  input: number[]
): GradientResult {
  const numLayers = weights.layers.length;
  const weightGrads: Matrix[] = [];
  const biasGrads: number[][] = [];
  const layerGrads: number[][] = [];

  // Initialize arrays
  for (let l = 0; l < numLayers; l++) {
    const W = weights.layers[l];
    weightGrads.push(W.map((row) => row.map(() => 0)));
    biasGrads.push(weights.biases[l].map(() => 0));
    layerGrads.push([]);
  }

  // Compute output layer delta
  // For output layer with sigmoid: delta = (a - t) * sigmoid'(z)
  const outputIdx = numLayers - 1;
  const outputAct = fwd.activations[outputIdx];
  const outputPre = fwd.preActivations[outputIdx];
  const outputDelta: number[] = [];

  for (let j = 0; j < outputAct.length; j++) {
    const error = outputAct[j] - target[j];
    const deriv = activateDerivative(outputPre[j], "sigmoid");
    outputDelta.push(error * deriv);
  }
  layerGrads[outputIdx] = outputDelta;

  // Backpropagate through hidden layers
  for (let l = outputIdx - 1; l >= 0; l--) {
    const pre = fwd.preActivations[l];
    const nextDelta = layerGrads[l + 1];
    const W_next = weights.layers[l + 1]; // shape: [thisLayerSize][nextLayerSize]
    const delta: number[] = [];

    for (let j = 0; j < pre.length; j++) {
      let sum = 0;
      for (let k = 0; k < nextDelta.length; k++) {
        sum += W_next[j][k] * nextDelta[k];
      }
      delta.push(sum * activateDerivative(pre[j], activation));
    }
    layerGrads[l] = delta;
  }

  // Compute weight and bias gradients
  for (let l = 0; l < numLayers; l++) {
    const prevAct = l === 0 ? input : fwd.activations[l - 1];
    const delta = layerGrads[l];

    for (let i = 0; i < prevAct.length; i++) {
      for (let j = 0; j < delta.length; j++) {
        weightGrads[l][i][j] = prevAct[i] * delta[j];
      }
    }

    for (let j = 0; j < delta.length; j++) {
      biasGrads[l][j] = delta[j];
    }
  }

  return { weightGrads, biasGrads, layerGrads };
}

// ─── Weight Update ──────────────────────────────────────────────────────────

/**
 * SGD weight update: w_new = w_old - lr * gradient
 * Returns new weights without mutating the original.
 */
export function updateWeights(
  weights: Weights,
  gradients: GradientResult,
  lr: number
): Weights {
  const newLayers: Matrix[] = weights.layers.map((W, l) =>
    W.map((row, i) =>
      row.map((w, j) => w - lr * gradients.weightGrads[l][i][j])
    )
  );

  const newBiases: number[][] = weights.biases.map((b, l) =>
    b.map((bv, j) => bv - lr * gradients.biasGrads[l][j])
  );

  return { layers: newLayers, biases: newBiases };
}

// ─── Dataset Generation ─────────────────────────────────────────────────────

/**
 * Generate named datasets for the NN interactive demo.
 */
export function generateDataset(name: string): Dataset {
  switch (name) {
    case "xor":
      return {
        name: "xor",
        inputs: [
          [0, 0],
          [0, 1],
          [1, 0],
          [1, 1],
        ],
        targets: [[0], [1], [1], [0]],
      };

    case "spiral":
      return generateSpiral(100);

    case "crescent":
      return generateCrescent(100);

    default:
      throw new Error(`Unknown dataset: ${name}`);
  }
}

function generateSpiral(n: number): Dataset {
  const inputs: number[][] = [];
  const targets: number[][] = [];
  const pointsPerClass = Math.floor(n / 2);

  for (let cls = 0; cls < 2; cls++) {
    for (let i = 0; i < pointsPerClass; i++) {
      const r = (i / pointsPerClass) * 5;
      const t =
        (i / pointsPerClass) * Math.PI * 2.5 +
        cls * Math.PI +
        (Math.random() - 0.5) * 0.3;
      const x = r * Math.cos(t);
      const y = r * Math.sin(t);
      inputs.push([x, y]);
      targets.push([cls]);
    }
  }

  return { name: "spiral", inputs, targets };
}

function generateCrescent(n: number): Dataset {
  const inputs: number[][] = [];
  const targets: number[][] = [];
  const pointsPerClass = Math.floor(n / 2);

  // Upper crescent (class 0)
  for (let i = 0; i < pointsPerClass; i++) {
    const angle = (Math.PI * i) / pointsPerClass;
    const r = 4 + (Math.random() - 0.5) * 0.8;
    const x = r * Math.cos(angle);
    const y = r * Math.sin(angle);
    inputs.push([x, y]);
    targets.push([0]);
  }

  // Lower crescent (class 1)
  for (let i = 0; i < pointsPerClass; i++) {
    const angle = Math.PI + (Math.PI * i) / pointsPerClass;
    const r = 4 + (Math.random() - 0.5) * 0.8;
    const x = r * Math.cos(angle) + 4;
    const y = r * Math.sin(angle) + 0.5;
    inputs.push([x, y]);
    targets.push([1]);
  }

  return { name: "crescent", inputs, targets };
}
