import type { CSSProperties, ReactNode } from "react";

export type Vec3 = [number, number, number];
export type Vec3Pair = [number, number];

export interface Scene3DProps {
  children: ReactNode;
  cameraPosition?: Vec3;
  showGrid?: boolean;
  backgroundColor?: string;
  enableDamping?: boolean;
  style?: CSSProperties;
  className?: string;
  fallback?: ReactNode;
}

export interface Axis3DProps {
  labels?: { x: string; y: string; z: string };
  range?: { x: Vec3Pair; y: Vec3Pair; z: Vec3Pair };
  showTicks?: boolean;
  showGrid?: boolean;
  tickCount?: number;
}

export interface Tooltip3DProps {
  position: Vec3;
  visible: boolean;
  children: ReactNode;
}
