import type { Vec } from "../secondOrder";
import { fitIsoBounds, projectIso } from "./isometric";

interface SurfacePathChartProps {
  objective: (x: Vec) => number;
  path: Vec[];
  xRange?: [number, number];
  yRange?: [number, number];
  height?: number;
}

export function SurfacePathChart({
  objective,
  path,
  xRange = [-1, 4],
  yRange = [-1, 4],
  height = 240,
}: SurfacePathChartProps) {
  const width = 480;
  const pad = 24;
  const grid = 20;
  const [xLo, xHi] = xRange;
  const [yLo, yHi] = yRange;

  type Node = { i: number; j: number; x: number; y: number; z: number };
  const nodes: Node[] = [];
  let zMin = Infinity;
  let zMax = -Infinity;
  for (let i = 0; i <= grid; i += 1) {
    for (let j = 0; j <= grid; j += 1) {
      const x = xLo + ((xHi - xLo) * i) / grid;
      const y = yLo + ((yHi - yLo) * j) / grid;
      const z = objective([x, y]);
      zMin = Math.min(zMin, z);
      zMax = Math.max(zMax, z);
      nodes.push({ i, j, x, y, z });
    }
  }
  const zSpan = Math.max(zMax - zMin, 1e-6);
  const xySpan = Math.max(xHi - xLo, yHi - yLo, 1e-6);
  const zScale = (xySpan / zSpan) * 0.35;
  const scale = 1;

  const nodeAt = (i: number, j: number) => nodes[i * (grid + 1) + j]!;

  const isoPoints: { px: number; py: number }[] = [];
  for (const n of nodes) {
    isoPoints.push(projectIso(n.x, n.y, n.z * zScale, scale));
  }
  for (const pt of path) {
    const z = objective(pt);
    isoPoints.push(projectIso(pt[0]!, pt[1]!, z * zScale, scale));
  }
  const toSvg = fitIsoBounds(isoPoints, width, height, pad);

  const quads: {
    key: string;
    depth: number;
    points: string;
    fill: string;
  }[] = [];
  for (let i = 0; i < grid; i += 1) {
    for (let j = 0; j < grid; j += 1) {
      const a = nodeAt(i, j);
      const b = nodeAt(i + 1, j);
      const c = nodeAt(i + 1, j + 1);
      const d = nodeAt(i, j + 1);
      const depth = a.x + a.y + a.z;
      const corners = [a, b, c, d];
      const svgPts = corners.map((n) =>
        toSvg(projectIso(n.x, n.y, n.z * zScale, scale)),
      );
      const avgZ = corners.reduce((s, n) => s + n.z, 0) / 4;
      const t = (avgZ - zMin) / zSpan;
      const hue = 160 - t * 100;
      quads.push({
        key: `${i}-${j}`,
        depth,
        points: svgPts.map((p) => `${p.sx},${p.sy}`).join(" "),
        fill: `hsl(${hue} 42% ${16 + t * 20}%)`,
      });
    }
  }
  quads.sort((a, b) => a.depth - b.depth);

  const pathD = path
    .map((pt, idx) => {
      const z = objective(pt);
      const { sx, sy } = toSvg(projectIso(pt[0]!, pt[1]!, z * zScale, scale));
      return `${idx === 0 ? "M" : "L"} ${sx} ${sy}`;
    })
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full max-w-xl rounded-lg border border-slate-800 bg-slate-950"
      role="img"
      aria-label="Optimization path on 3D surface plot of the objective"
    >
      {quads.map((q) => (
        <polygon
          key={q.key}
          points={q.points}
          fill={q.fill}
          stroke="#0f172a"
          strokeWidth={0.25}
        />
      ))}
      {path.length > 1 ? (
        <path
          d={pathD}
          fill="none"
          stroke="#34d399"
          strokeWidth={2}
          strokeLinejoin="round"
        />
      ) : null}
      {path.map((pt, idx) => {
        const z = objective(pt);
        const { sx, sy } = toSvg(projectIso(pt[0]!, pt[1]!, z * zScale, scale));
        const isStart = idx === 0;
        const isEnd = idx === path.length - 1;
        return (
          <circle
            key={idx}
            cx={sx}
            cy={sy}
            r={isStart || isEnd ? 5 : 3}
            fill={isStart ? "#fbbf24" : isEnd ? "#34d399" : "#818cf8"}
          />
        );
      })}
    </svg>
  );
}
