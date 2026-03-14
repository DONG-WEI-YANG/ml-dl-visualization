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
  layers: Matrix[];
  biases: number[][];
}

export interface ForwardResult {
  activations: number[][];
  preActivations: number[][];
}

export interface GradientResult {
  weightGrads: Matrix[];
  biasGrads: number[][];
  layerGrads: number[][];
}

export interface TrainingState {
  epoch: number;
  lossHistory: number[];
  isTraining: boolean;
  weights: Weights;
}

export interface Dataset {
  name: string;
  inputs: number[][];
  targets: number[][];
}
