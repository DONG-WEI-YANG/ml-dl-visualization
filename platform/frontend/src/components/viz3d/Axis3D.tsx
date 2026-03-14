import { useMemo } from "react";
import { Line, Text } from "@react-three/drei";
import type { Axis3DProps, Vec3Pair, Vec3 } from "./types";

const DEFAULT_RANGE: Vec3Pair = [-5, 5];
const AXIS_COLOR = "#64748b";
const TICK_COLOR = "#475569";

export default function Axis3D({
  labels = { x: "X", y: "Y", z: "Z" },
  range = { x: DEFAULT_RANGE, y: DEFAULT_RANGE, z: DEFAULT_RANGE },
  showTicks = true,
  showGrid = false,
  tickCount = 5,
}: Axis3DProps) {
  const ticks = useMemo(() => {
    const makeTicks = (r: Vec3Pair) => {
      const step = (r[1] - r[0]) / tickCount;
      return Array.from({ length: tickCount + 1 }, (_, i) =>
        +(r[0] + i * step).toFixed(2)
      );
    };
    return { x: makeTicks(range.x), y: makeTicks(range.y), z: makeTicks(range.z) };
  }, [range, tickCount]);

  const xLinePoints: Vec3[] = [[range.x[0], 0, 0], [range.x[1], 0, 0]];
  const yLinePoints: Vec3[] = [[0, range.y[0], 0], [0, range.y[1], 0]];
  const zLinePoints: Vec3[] = [[0, 0, range.z[0]], [0, 0, range.z[1]]];

  return (
    <group>
      {/* X axis line */}
      <Line points={xLinePoints} color={AXIS_COLOR} lineWidth={1} />
      <Text position={[range.x[1] + 0.5, 0, 0]} fontSize={0.3} color={AXIS_COLOR}>{labels.x}</Text>

      {/* Y axis line */}
      <Line points={yLinePoints} color={AXIS_COLOR} lineWidth={1} />
      <Text position={[0, range.y[1] + 0.5, 0]} fontSize={0.3} color={AXIS_COLOR}>{labels.y}</Text>

      {/* Z axis line */}
      <Line points={zLinePoints} color={AXIS_COLOR} lineWidth={1} />
      <Text position={[0, 0, range.z[1] + 0.5]} fontSize={0.3} color={AXIS_COLOR}>{labels.z}</Text>

      {/* X ticks */}
      {showTicks && ticks.x.map((v) => (
        <Text key={`tx-${v}`} position={[v, -0.3, 0]} fontSize={0.15} color={TICK_COLOR}>{v}</Text>
      ))}
      {/* Y ticks */}
      {showTicks && ticks.y.map((v) => (
        <Text key={`ty-${v}`} position={[-0.3, v, 0]} fontSize={0.15} color={TICK_COLOR}>{v}</Text>
      ))}

      {/* Grid */}
      {showGrid && <gridHelper args={[range.x[1] - range.x[0], tickCount, "#1e293b", "#1e293b"]} />}
    </group>
  );
}
