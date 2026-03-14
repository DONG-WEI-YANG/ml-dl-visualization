import { Html } from "@react-three/drei";
import type { Tooltip3DProps } from "./types";

export default function Tooltip3D({ position, visible, children }: Tooltip3DProps) {
  if (!visible) return null;
  return (
    <Html position={position} center>
      <div className="bg-white border border-gray-200 rounded-lg shadow-lg px-3 py-2 text-xs text-gray-700 pointer-events-none whitespace-nowrap">
        {children}
      </div>
    </Html>
  );
}
