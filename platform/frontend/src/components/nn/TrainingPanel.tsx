import type { NetworkConfig, TrainingState, ActivationType } from "./nn-types";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
} from "recharts";

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

const ACTIVATIONS: { value: ActivationType; label: string }[] = [
  { value: "sigmoid", label: "Sigmoid" },
  { value: "tanh", label: "Tanh" },
  { value: "relu", label: "ReLU" },
  { value: "leaky_relu", label: "Leaky ReLU" },
  { value: "gelu", label: "GELU" },
];

const DATASETS = [
  { value: "xor", label: "XOR" },
  { value: "spiral", label: "Spiral" },
  { value: "crescent", label: "Crescent" },
];

export default function TrainingPanel({
  config,
  onConfigChange,
  trainingState,
  onTrain,
  onPause,
  onStep,
  onReset,
  lr,
  onLrChange,
  maxEpochs,
  onMaxEpochsChange,
  dataset,
  onDatasetChange,
}: TrainingPanelProps) {
  const { epoch, lossHistory, isTraining } = trainingState;

  const lossData = lossHistory.map((loss, i) => ({
    epoch: i + 1,
    loss: +loss.toFixed(6),
  }));

  const currentLoss = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1] : null;

  return (
    <div className="space-y-4 p-4 bg-gray-50 border border-gray-200 rounded-lg">
      {/* Architecture Controls */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-2">
          🏗️ 網路架構 Architecture
        </h4>

        {/* Hidden Layers */}
        <div className="mb-3">
          <label className="block text-xs text-gray-500 mb-1">
            隱藏層數 Hidden Layers
          </label>
          <div className="flex gap-1">
            {[1, 2, 3, 4].map((n) => (
              <button
                key={n}
                onClick={() =>
                  onConfigChange({ ...config, hiddenLayers: n })
                }
                disabled={isTraining}
                className={`px-3 py-1 text-sm rounded font-medium transition-colors ${
                  config.hiddenLayers === n
                    ? "bg-blue-500 text-white"
                    : "bg-white border border-gray-300 text-gray-600 hover:bg-gray-100"
                } disabled:opacity-50`}
              >
                {n}
              </button>
            ))}
          </div>
        </div>

        {/* Neurons per Layer */}
        <div className="mb-3">
          <label className="block text-xs text-gray-500 mb-1">
            每層神經元 Neurons/Layer: <strong>{config.neuronsPerLayer}</strong>
          </label>
          <div className="flex items-center gap-2">
            <button
              onClick={() =>
                onConfigChange({
                  ...config,
                  neuronsPerLayer: Math.max(2, config.neuronsPerLayer - 1),
                })
              }
              disabled={isTraining || config.neuronsPerLayer <= 2}
              className="px-2 py-1 text-sm bg-white border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50"
            >
              −
            </button>
            <div className="flex-1 bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all"
                style={{
                  width: `${((config.neuronsPerLayer - 2) / 6) * 100}%`,
                }}
              />
            </div>
            <button
              onClick={() =>
                onConfigChange({
                  ...config,
                  neuronsPerLayer: Math.min(8, config.neuronsPerLayer + 1),
                })
              }
              disabled={isTraining || config.neuronsPerLayer >= 8}
              className="px-2 py-1 text-sm bg-white border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50"
            >
              +
            </button>
          </div>
        </div>

        {/* Activation */}
        <div className="mb-3">
          <label className="block text-xs text-gray-500 mb-1">
            激活函數 Activation
          </label>
          <select
            value={config.activation}
            onChange={(e) =>
              onConfigChange({
                ...config,
                activation: e.target.value as ActivationType,
              })
            }
            disabled={isTraining}
            className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded bg-white disabled:opacity-50"
          >
            {ACTIVATIONS.map((a) => (
              <option key={a.value} value={a.value}>
                {a.label}
              </option>
            ))}
          </select>
        </div>

        {/* Dataset */}
        <div className="mb-3">
          <label className="block text-xs text-gray-500 mb-1">
            資料集 Dataset
          </label>
          <select
            value={dataset}
            onChange={(e) => onDatasetChange(e.target.value)}
            disabled={isTraining}
            className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded bg-white disabled:opacity-50"
          >
            {DATASETS.map((d) => (
              <option key={d.value} value={d.value}>
                {d.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Training Controls */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-2">
          ⚡ 訓練控制 Training
        </h4>

        {/* Learning Rate */}
        <div className="mb-3">
          <label className="block text-xs text-gray-500 mb-1">
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

        {/* Max Epochs */}
        <div className="mb-3">
          <label className="block text-xs text-gray-500 mb-1">
            迭代次數 Epochs: <strong>{maxEpochs}</strong>
          </label>
          <input
            type="range"
            min="10"
            max="500"
            step="10"
            value={maxEpochs}
            onChange={(e) => onMaxEpochsChange(+e.target.value)}
            className="w-full"
          />
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-2">
          <button
            onClick={isTraining ? onPause : onTrain}
            className={`px-3 py-1.5 text-sm font-medium rounded transition-colors ${
              isTraining
                ? "bg-amber-500 text-white hover:bg-amber-600"
                : "bg-blue-500 text-white hover:bg-blue-600"
            }`}
          >
            {isTraining ? "⏸ Pause" : "▶ Train"}
          </button>
          <button
            onClick={onStep}
            disabled={isTraining}
            className="px-3 py-1.5 text-sm font-medium rounded bg-green-500 text-white hover:bg-green-600 disabled:opacity-50"
          >
            ⏭ Step
          </button>
          <button
            onClick={onReset}
            className="px-3 py-1.5 text-sm font-medium rounded bg-gray-500 text-white hover:bg-gray-600"
          >
            ↺ Reset
          </button>
        </div>

        {/* Epoch/Loss info */}
        <div className="mt-2 text-xs text-gray-500">
          Epoch: <strong>{epoch}</strong>
          {currentLoss !== null && (
            <span className="ml-3">
              Loss: <strong>{currentLoss.toFixed(6)}</strong>
            </span>
          )}
        </div>
      </div>

      {/* Loss Curve */}
      {lossData.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-gray-700 mb-2">
            損失曲線 Loss Curve
          </h4>
          <div className="bg-white border border-gray-200 rounded-lg p-2">
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="epoch"
                  tick={{ fontSize: 10 }}
                  label={{
                    value: "Epoch",
                    position: "bottom",
                    style: { fontSize: 10 },
                  }}
                />
                <YAxis
                  tick={{ fontSize: 10 }}
                  label={{
                    value: "Loss",
                    angle: -90,
                    position: "insideLeft",
                    style: { fontSize: 10 },
                  }}
                />
                <Tooltip
                  contentStyle={{ fontSize: 11 }}
                  formatter={(value: number) => [value.toFixed(6), "Loss"]}
                />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#2563eb"
                  dot={false}
                  strokeWidth={2}
                  isAnimationActive={false}
                />
                {/* Current epoch marker */}
                {lossData.length > 0 && (
                  <ReferenceDot
                    x={lossData[lossData.length - 1].epoch}
                    y={lossData[lossData.length - 1].loss}
                    r={4}
                    fill="#ef4444"
                    stroke="none"
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
