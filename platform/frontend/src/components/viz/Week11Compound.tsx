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
