import type { Vec } from "../secondOrder";
import { fitIsoBounds, projectIso } from "./isometric";

interface Path3DChartProps {
  path: Vec[];
  /** Per-axis plot ranges [lo, hi] for x₁, x₂, x₃. */
  axisRanges?: [[number, number], [number, number], [number, number]];
  height?: number;
}

export function Path3DChart({
  path,
  axisRanges = [
    [-1, 4],
    [-1, 4],
    [-1, 4],
  ],
  height = 260,
}: Path3DChartProps) {
  const width = 480;
  const pad = 28;
  const scale = 1;

  const isoPoints = path.flatMap((pt) => [
    projectIso(pt[0]!, pt[1]!, pt[2]!, scale),
  ]);
  for (const [axis, range] of axisRanges.entries()) {
    const [lo, hi] = range;
    const mid = (lo + hi) / 2;
    const tip = mid + (hi - lo) * 0.45;
    if (axis === 0) isoPoints.push(projectIso(tip, lo, lo, scale));
    if (axis === 1) isoPoints.push(projectIso(lo, tip, lo, scale));
    if (axis === 2) isoPoints.push(projectIso(lo, lo, tip, scale));
  }
  const toSvg = fitIsoBounds(isoPoints, width, height, pad);

  const pathD = path
    .map((pt, idx) => {
      const { sx, sy } = toSvg(projectIso(pt[0]!, pt[1]!, pt[2]!, scale));
      return `${idx === 0 ? "M" : "L"} ${sx} ${sy}`;
    })
    .join(" ");

  const axisLines = [
    {
      label: "x₁",
      from: [axisRanges[0]![0], axisRanges[1]![0], axisRanges[2]![0]] as Vec,
      to: [axisRanges[0]![1], axisRanges[1]![0], axisRanges[2]![0]] as Vec,
    },
    {
      label: "x₂",
      from: [axisRanges[0]![0], axisRanges[1]![0], axisRanges[2]![0]] as Vec,
      to: [axisRanges[0]![0], axisRanges[1]![1], axisRanges[2]![0]] as Vec,
    },
    {
      label: "x₃",
      from: [axisRanges[0]![0], axisRanges[1]![0], axisRanges[2]![0]] as Vec,
      to: [axisRanges[0]![0], axisRanges[1]![0], axisRanges[2]![1]] as Vec,
    },
  ];

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full max-w-xl rounded-lg border border-slate-800 bg-slate-950"
      role="img"
      aria-label="Optimization path in three-dimensional parameter space"
    >
      {axisLines.map(({ label, from, to }) => {
        const a = toSvg(projectIso(from[0]!, from[1]!, from[2]!, scale));
        const b = toSvg(projectIso(to[0]!, to[1]!, to[2]!, scale));
        return (
          <g key={label}>
            <line
              x1={a.sx}
              y1={a.sy}
              x2={b.sx}
              y2={b.sy}
              stroke="#475569"
              strokeWidth={1}
            />
            <text
              x={b.sx}
              y={b.sy}
              className="fill-slate-500 text-[10px]"
              textAnchor="middle"
            >
              {label}
            </text>
          </g>
        );
      })}
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
        const { sx, sy } = toSvg(projectIso(pt[0]!, pt[1]!, pt[2]!, scale));
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
