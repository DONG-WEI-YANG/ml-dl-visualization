import { useState, useEffect, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { Scene3DProps } from "./types";

function checkWebGL(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return !!(canvas.getContext("webgl") || canvas.getContext("webgl2"));
  } catch {
    return false;
  }
}

const DEFAULT_FALLBACK = (
  <div className="flex items-center justify-center h-full min-h-[300px] bg-gray-50 rounded-lg">
    <div className="text-center text-gray-500 text-sm">
      <p className="text-2xl mb-2">⚠️</p>
      <p>您的瀏覽器不支援 WebGL，請改用 2D 模式</p>
    </div>
  </div>
);

export default function Scene3D({
  children,
  cameraPosition = [5, 5, 5],
  showGrid = false,
  backgroundColor = "#0f172a",
  enableDamping = true,
  style,
  className,
  fallback = DEFAULT_FALLBACK,
  ariaLabel = "3D 視覺化場景",
}: Scene3DProps & { ariaLabel?: string }) {
  const [webglOk, setWebglOk] = useState(true);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const controlsRef = useRef<any>(null);

  useEffect(() => {
    setWebglOk(checkWebGL());
  }, []);

  // Keyboard shortcuts: arrows=rotate, +/-=zoom
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const ctrl = controlsRef.current;
      if (!ctrl) return;
      const step = 0.1;
      switch (e.key) {
        case "ArrowLeft": ctrl.setAzimuthalAngle(ctrl.getAzimuthalAngle() - step); break;
        case "ArrowRight": ctrl.setAzimuthalAngle(ctrl.getAzimuthalAngle() + step); break;
        case "ArrowUp": ctrl.setPolarAngle(Math.max(0.1, ctrl.getPolarAngle() - step)); break;
        case "ArrowDown": ctrl.setPolarAngle(Math.min(Math.PI - 0.1, ctrl.getPolarAngle() + step)); break;
        case "+": case "=": ctrl.object.zoom = Math.min(10, ctrl.object.zoom * 1.1); ctrl.object.updateProjectionMatrix(); break;
        case "-": ctrl.object.zoom = Math.max(0.1, ctrl.object.zoom / 1.1); ctrl.object.updateProjectionMatrix(); break;
      }
      ctrl.update();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  if (!webglOk) return <>{fallback}</>;

  return (
    <div className={className} style={{ height: 500, ...style }} aria-label={ariaLabel} role="img" tabIndex={0}>
      <Canvas
        camera={{ position: cameraPosition, fov: 60 }}
        style={{ background: backgroundColor, width: "100%", height: "100%" }}
      >
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, 10, 5]} intensity={0.8} />
        {showGrid && <gridHelper args={[10, 10, "#334155", "#1e293b"]} />}
        <OrbitControls
          ref={controlsRef}
          enableDamping={enableDamping}
          dampingFactor={0.05}
          enablePan
          enableZoom
          enableRotate
        />
        {children}
      </Canvas>
    </div>
  );
}
