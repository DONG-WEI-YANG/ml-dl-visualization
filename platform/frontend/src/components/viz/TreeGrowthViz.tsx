import { useState } from "react";

interface TreeNode {
  feature: string;
  threshold: number;
  left?: TreeNode | string;
  right?: TreeNode | string;
  depth: number;
}

const FULL_TREE: TreeNode = {
  feature: "花瓣長度", threshold: 2.5, depth: 0,
  left: "Setosa",
  right: {
    feature: "花瓣寬度", threshold: 1.75, depth: 1,
    left: {
      feature: "花瓣長度", threshold: 4.95, depth: 2,
      left: "Versicolor",
      right: "Virginica",
    },
    right: {
      feature: "花瓣寬度", threshold: 1.65, depth: 2,
      left: "Virginica",
      right: "Virginica",
    },
  },
};

function getVisibleTree(tree: TreeNode, maxDepth: number): TreeNode | string {
  if (tree.depth >= maxDepth) return "?";
  return {
    ...tree,
    left: typeof tree.left === "string" ? tree.left : tree.left ? getVisibleTree(tree.left, maxDepth) : "?",
    right: typeof tree.right === "string" ? tree.right : tree.right ? getVisibleTree(tree.right, maxDepth) : "?",
  };
}

function RenderNode({ node, x, y, dx }: { node: TreeNode | string; x: number; y: number; dx: number }) {
  if (typeof node === "string") {
    return (
      <g>
        <rect x={x - 30} y={y - 12} width={60} height={24} rx={12} fill={
          node === "Setosa" ? "#bbf7d0" : node === "Versicolor" ? "#bfdbfe" :
          node === "Virginica" ? "#fecaca" : "#e5e7eb"
        } stroke="#9ca3af" />
        <text x={x} y={y + 4} textAnchor="middle" fontSize={9} fill="#374151">{node}</text>
      </g>
    );
  }

  const childY = y + 60;
  const childDx = dx * 0.55;

  return (
    <g>
      <line x1={x} y1={y + 12} x2={x - dx} y2={childY - 12} stroke="#9ca3af" strokeWidth={1.5} />
      <line x1={x} y1={y + 12} x2={x + dx} y2={childY - 12} stroke="#9ca3af" strokeWidth={1.5} />
      <text x={x - dx / 2 - 5} y={y + 30} textAnchor="middle" fontSize={8} fill="#6b7280">{"≤"}</text>
      <text x={x + dx / 2 + 5} y={y + 30} textAnchor="middle" fontSize={8} fill="#6b7280">{">"}</text>

      <rect x={x - 42} y={y - 14} width={84} height={28} rx={6} fill="#eff6ff" stroke="#3b82f6" strokeWidth={1.5} />
      <text x={x} y={y - 1} textAnchor="middle" fontSize={8} fill="#1e40af">{node.feature}</text>
      <text x={x} y={y + 10} textAnchor="middle" fontSize={8} fill="#3b82f6">{"≤ "}{node.threshold}</text>

      {node.left && <RenderNode node={node.left} x={x - dx} y={childY} dx={childDx} />}
      {node.right && <RenderNode node={node.right} x={x + dx} y={childY} dx={childDx} />}
    </g>
  );
}

export default function TreeGrowthViz() {
  const [depth, setDepth] = useState(1);
  const tree = getVisibleTree(FULL_TREE, depth);

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">決策樹生長 Decision Tree Growth</h3>

      <div>
        <label className="text-sm text-gray-600">
          樹的深度 Depth: <strong>{depth}</strong>
        </label>
        <input type="range" min={1} max={4} value={depth}
          onChange={(e) => setDepth(+e.target.value)} className="w-full" />
      </div>

      <div className="flex gap-2 justify-center">
        {[1, 2, 3, 4].map((d) => (
          <button key={d} onClick={() => setDepth(d)}
            className={`px-3 py-1 rounded text-sm ${depth === d ? "bg-blue-500 text-white" : "bg-gray-100"}`}
          >Depth {d - 1}</button>
        ))}
      </div>

      <svg viewBox="0 0 400 250" className="w-full border border-gray-200 rounded-lg bg-white">
        {typeof tree !== "string" && <RenderNode node={tree} x={200} y={30} dx={120} />}
      </svg>

      <div className="bg-gray-50 rounded-lg p-3 text-xs text-gray-600">
        <p><strong>Iris 資料集：</strong>根據花瓣長度/寬度分類三種鳶尾花</p>
        <p>深度越深 → 分類越精確 → 但可能過擬合</p>
      </div>
    </div>
  );
}
