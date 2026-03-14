import { describe, it, expect } from "vitest";
import {
  activate,
  activateDerivative,
  initWeights,
  forward,
  computeLoss,
  backward,
  updateWeights,
  generateDataset,
} from "../nn-math";
import type { NetworkConfig, Weights } from "../nn-types";

// ─── activate ───────────────────────────────────────────────────────────────

describe("activate", () => {
  // sigmoid
  it("sigmoid(0) = 0.5", () => {
    expect(activate(0, "sigmoid")).toBeCloseTo(0.5, 5);
  });
  it("sigmoid(large positive) ≈ 1", () => {
    expect(activate(10, "sigmoid")).toBeCloseTo(1, 3);
  });
  it("sigmoid(large negative) ≈ 0", () => {
    expect(activate(-10, "sigmoid")).toBeCloseTo(0, 3);
  });

  // tanh
  it("tanh(0) = 0", () => {
    expect(activate(0, "tanh")).toBeCloseTo(0, 5);
  });
  it("tanh(large positive) ≈ 1", () => {
    expect(activate(5, "tanh")).toBeCloseTo(1, 3);
  });
  it("tanh(large negative) ≈ -1", () => {
    expect(activate(-5, "tanh")).toBeCloseTo(-1, 3);
  });

  // relu
  it("relu(negative) = 0", () => {
    expect(activate(-5, "relu")).toBe(0);
  });
  it("relu(positive) = identity", () => {
    expect(activate(3.7, "relu")).toBe(3.7);
  });
  it("relu(0) = 0", () => {
    expect(activate(0, "relu")).toBe(0);
  });

  // leaky_relu
  it("leaky_relu(negative) = 0.01 * x", () => {
    expect(activate(-2, "leaky_relu")).toBeCloseTo(-0.02, 5);
  });
  it("leaky_relu(positive) = identity", () => {
    expect(activate(3, "leaky_relu")).toBe(3);
  });

  // gelu
  it("gelu(0) ≈ 0", () => {
    expect(activate(0, "gelu")).toBeCloseTo(0, 3);
  });
  it("gelu(large positive) ≈ x", () => {
    expect(activate(3, "gelu")).toBeCloseTo(3, 1);
  });
  it("gelu(large negative) ≈ 0", () => {
    expect(activate(-3, "gelu")).toBeCloseTo(0, 1);
  });
});

// ─── activateDerivative ─────────────────────────────────────────────────────

describe("activateDerivative", () => {
  it("sigmoid'(0) = 0.25", () => {
    expect(activateDerivative(0, "sigmoid")).toBeCloseTo(0.25, 5);
  });
  it("sigmoid'(large) ≈ 0", () => {
    expect(activateDerivative(10, "sigmoid")).toBeCloseTo(0, 3);
  });

  it("tanh'(0) = 1", () => {
    expect(activateDerivative(0, "tanh")).toBeCloseTo(1, 5);
  });

  it("relu'(negative) = 0", () => {
    expect(activateDerivative(-2, "relu")).toBe(0);
  });
  it("relu'(positive) = 1", () => {
    expect(activateDerivative(3, "relu")).toBe(1);
  });

  it("leaky_relu'(negative) = 0.01", () => {
    expect(activateDerivative(-2, "leaky_relu")).toBeCloseTo(0.01, 5);
  });
  it("leaky_relu'(positive) = 1", () => {
    expect(activateDerivative(3, "leaky_relu")).toBe(1);
  });

  it("gelu'(0) ≈ 0.5", () => {
    expect(activateDerivative(0, "gelu")).toBeCloseTo(0.5, 1);
  });
});

// ─── initWeights ────────────────────────────────────────────────────────────

describe("initWeights", () => {
  const cfg: NetworkConfig = {
    inputSize: 2,
    hiddenLayers: 2,
    neuronsPerLayer: 4,
    outputSize: 1,
    activation: "relu",
  };

  it("produces correct number of weight matrices (hiddenLayers + 1)", () => {
    const w = initWeights(cfg);
    // input→hidden1, hidden1→hidden2, hidden2→output = 3
    expect(w.layers).toHaveLength(3);
    expect(w.biases).toHaveLength(3);
  });

  it("first matrix is inputSize x neuronsPerLayer", () => {
    const w = initWeights(cfg);
    // layers[0]: shape [inputSize][neuronsPerLayer] = [2][4]
    expect(w.layers[0]).toHaveLength(cfg.inputSize);
    expect(w.layers[0][0]).toHaveLength(cfg.neuronsPerLayer);
  });

  it("hidden matrix is neuronsPerLayer x neuronsPerLayer", () => {
    const w = initWeights(cfg);
    expect(w.layers[1]).toHaveLength(cfg.neuronsPerLayer);
    expect(w.layers[1][0]).toHaveLength(cfg.neuronsPerLayer);
  });

  it("last matrix is neuronsPerLayer x outputSize", () => {
    const w = initWeights(cfg);
    expect(w.layers[2]).toHaveLength(cfg.neuronsPerLayer);
    expect(w.layers[2][0]).toHaveLength(cfg.outputSize);
  });

  it("bias dimensions match layer outputs", () => {
    const w = initWeights(cfg);
    expect(w.biases[0]).toHaveLength(cfg.neuronsPerLayer);
    expect(w.biases[1]).toHaveLength(cfg.neuronsPerLayer);
    expect(w.biases[2]).toHaveLength(cfg.outputSize);
  });

  it("weights contain non-zero values", () => {
    const w = initWeights(cfg);
    const allZero = w.layers.every((m) =>
      m.every((row) => row.every((v) => v === 0))
    );
    expect(allZero).toBe(false);
  });

  it("single hidden layer config produces 2 weight matrices", () => {
    const cfg1: NetworkConfig = {
      inputSize: 3,
      hiddenLayers: 1,
      neuronsPerLayer: 5,
      outputSize: 2,
      activation: "sigmoid",
    };
    const w = initWeights(cfg1);
    expect(w.layers).toHaveLength(2);
    expect(w.layers[0]).toHaveLength(3);
    expect(w.layers[0][0]).toHaveLength(5);
    expect(w.layers[1]).toHaveLength(5);
    expect(w.layers[1][0]).toHaveLength(2);
  });
});

// ─── forward ────────────────────────────────────────────────────────────────

describe("forward", () => {
  it("produces output with known weights (single hidden layer, sigmoid)", () => {
    // simple 2-input, 2-hidden, 1-output
    const weights: Weights = {
      layers: [
        // input(2) -> hidden(2)
        [
          [0.5, -0.5],
          [0.3, 0.7],
        ],
        // hidden(2) -> output(1)
        [[0.6], [-0.4]],
      ],
      biases: [
        [0, 0],
        [0],
      ],
    };

    const input = [1, 0];
    const result = forward(input, weights, "sigmoid");

    // Layer 0 pre-activation: [1*0.5+0*0.3, 1*-0.5+0*0.7] = [0.5, -0.5]
    expect(result.preActivations[0][0]).toBeCloseTo(0.5, 5);
    expect(result.preActivations[0][1]).toBeCloseTo(-0.5, 5);

    // Layer 0 activation: sigmoid([0.5, -0.5])
    const s1 = 1 / (1 + Math.exp(-0.5));
    const s2 = 1 / (1 + Math.exp(0.5));
    expect(result.activations[0][0]).toBeCloseTo(s1, 5);
    expect(result.activations[0][1]).toBeCloseTo(s2, 5);

    // Layer 1 pre-activation: s1*0.6 + s2*(-0.4)
    const pre = s1 * 0.6 + s2 * -0.4;
    expect(result.preActivations[1][0]).toBeCloseTo(pre, 5);

    expect(result.activations).toHaveLength(2);
    expect(result.preActivations).toHaveLength(2);
  });

  it("stores pre-activations for every layer", () => {
    const cfg: NetworkConfig = {
      inputSize: 2,
      hiddenLayers: 3,
      neuronsPerLayer: 3,
      outputSize: 1,
      activation: "relu",
    };
    const w = initWeights(cfg);
    const result = forward([0.5, -0.5], w, "relu");
    // 3 hidden + 1 output = 4 layers total
    expect(result.preActivations).toHaveLength(4);
    expect(result.activations).toHaveLength(4);
  });

  it("output values are finite numbers", () => {
    const cfg: NetworkConfig = {
      inputSize: 2,
      hiddenLayers: 2,
      neuronsPerLayer: 4,
      outputSize: 1,
      activation: "sigmoid",
    };
    const w = initWeights(cfg);
    const result = forward([1, 0], w, "sigmoid");
    const output = result.activations[result.activations.length - 1];
    output.forEach((v) => {
      expect(Number.isFinite(v)).toBe(true);
    });
  });
});

// ─── computeLoss ────────────────────────────────────────────────────────────

describe("computeLoss", () => {
  it("MSE([1],[0]) = 0.5", () => {
    // sum((1-0)^2) / (2*1) = 1/2 = 0.5
    expect(computeLoss([1], [0])).toBeCloseTo(0.5, 5);
  });

  it("identical predictions give 0 loss", () => {
    expect(computeLoss([0.5, 0.3], [0.5, 0.3])).toBeCloseTo(0, 5);
  });

  it("MSE([1,0],[0,1]) = 1.0", () => {
    // sum((1-0)^2 + (0-1)^2) / (2*2) = 2/4 = 0.5
    expect(computeLoss([1, 0], [0, 1])).toBeCloseTo(0.5, 5);
  });

  it("is non-negative", () => {
    expect(computeLoss([0.3], [0.9])).toBeGreaterThanOrEqual(0);
  });
});

// ─── backward ───────────────────────────────────────────────────────────────

describe("backward", () => {
  const weights: Weights = {
    layers: [
      [
        [0.5, -0.5],
        [0.3, 0.7],
      ],
      [[0.6], [-0.4]],
    ],
    biases: [
      [0, 0],
      [0],
    ],
  };

  it("returns gradient shapes matching weight shapes", () => {
    const input = [1, 0];
    const fwd = forward(input, weights, "sigmoid");
    const grads = backward(fwd, weights, [1], "sigmoid", input);

    expect(grads.weightGrads).toHaveLength(weights.layers.length);
    for (let i = 0; i < weights.layers.length; i++) {
      expect(grads.weightGrads[i]).toHaveLength(weights.layers[i].length);
      expect(grads.weightGrads[i][0]).toHaveLength(
        weights.layers[i][0].length
      );
    }
  });

  it("returns bias gradients matching bias shapes", () => {
    const input = [1, 0];
    const fwd = forward(input, weights, "sigmoid");
    const grads = backward(fwd, weights, [1], "sigmoid", input);

    expect(grads.biasGrads).toHaveLength(weights.biases.length);
    for (let i = 0; i < weights.biases.length; i++) {
      expect(grads.biasGrads[i]).toHaveLength(weights.biases[i].length);
    }
  });

  it("returns layer gradients for each layer", () => {
    const input = [1, 0];
    const fwd = forward(input, weights, "sigmoid");
    const grads = backward(fwd, weights, [1], "sigmoid", input);

    expect(grads.layerGrads).toHaveLength(weights.layers.length);
  });

  it("gradients are finite numbers", () => {
    const input = [1, 0];
    const fwd = forward(input, weights, "sigmoid");
    const grads = backward(fwd, weights, [1], "sigmoid", input);

    grads.weightGrads.forEach((m) =>
      m.forEach((row) =>
        row.forEach((v) => expect(Number.isFinite(v)).toBe(true))
      )
    );
    grads.biasGrads.forEach((b) =>
      b.forEach((v) => expect(Number.isFinite(v)).toBe(true))
    );
  });
});

// ─── updateWeights ──────────────────────────────────────────────────────────

describe("updateWeights", () => {
  it("loss decreases after one SGD step on a known example", () => {
    const weights: Weights = {
      layers: [
        [
          [0.5, -0.5],
          [0.3, 0.7],
        ],
        [[0.6], [-0.4]],
      ],
      biases: [
        [0, 0],
        [0],
      ],
    };

    const input = [1, 0];
    const target = [1];

    // forward before
    const fwdBefore = forward(input, weights, "sigmoid");
    const predBefore = fwdBefore.activations[fwdBefore.activations.length - 1];
    const lossBefore = computeLoss(predBefore, target);

    // backward
    const grads = backward(fwdBefore, weights, target, "sigmoid", input);

    // update
    const newWeights = updateWeights(weights, grads, 0.5);

    // forward after
    const fwdAfter = forward(input, newWeights, "sigmoid");
    const predAfter = fwdAfter.activations[fwdAfter.activations.length - 1];
    const lossAfter = computeLoss(predAfter, target);

    expect(lossAfter).toBeLessThan(lossBefore);
  });

  it("does not mutate original weights", () => {
    const weights: Weights = {
      layers: [[[0.5]], [[0.3]]],
      biases: [[0.1], [0.2]],
    };
    const originalValue = weights.layers[0][0][0];

    const fwd = forward([1], weights, "relu");
    const grads = backward(fwd, weights, [1], "relu", [1]);
    updateWeights(weights, grads, 0.1);

    expect(weights.layers[0][0][0]).toBe(originalValue);
  });
});

// ─── generateDataset ────────────────────────────────────────────────────────

describe("generateDataset", () => {
  it("XOR dataset has 4 points", () => {
    const ds = generateDataset("xor");
    expect(ds.inputs).toHaveLength(4);
    expect(ds.targets).toHaveLength(4);
    expect(ds.name).toBe("xor");
  });

  it("XOR inputs are 2D", () => {
    const ds = generateDataset("xor");
    ds.inputs.forEach((inp) => expect(inp).toHaveLength(2));
  });

  it("XOR targets are 1D", () => {
    const ds = generateDataset("xor");
    ds.targets.forEach((t) => expect(t).toHaveLength(1));
  });

  it("spiral dataset has > 10 points", () => {
    const ds = generateDataset("spiral");
    expect(ds.inputs.length).toBeGreaterThan(10);
    expect(ds.targets.length).toBe(ds.inputs.length);
    expect(ds.name).toBe("spiral");
  });

  it("spiral inputs are 2D", () => {
    const ds = generateDataset("spiral");
    ds.inputs.forEach((inp) => expect(inp).toHaveLength(2));
  });

  it("crescent dataset has > 10 points", () => {
    const ds = generateDataset("crescent");
    expect(ds.inputs.length).toBeGreaterThan(10);
    expect(ds.targets.length).toBe(ds.inputs.length);
    expect(ds.name).toBe("crescent");
  });

  it("crescent inputs are 2D", () => {
    const ds = generateDataset("crescent");
    ds.inputs.forEach((inp) => expect(inp).toHaveLength(2));
  });
});
