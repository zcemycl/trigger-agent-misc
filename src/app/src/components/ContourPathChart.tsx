import type { Vec } from "../secondOrder";

interface ContourPathChartProps {
  objective: (x: Vec) => number;
  path: Vec[];
  xRange?: [number, number];
  yRange?: [number, number];
  height?: number;
}

export function ContourPathChart({
  objective,
  path,
  xRange = [-1, 4],
  yRange = [-1, 4],
  height = 240,
}: ContourPathChartProps) {
  const width = 480;
  const pad = { top: 16, right: 16, bottom: 32, left: 48 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const grid = 32;
  let zMin = Infinity;
  let zMax = -Infinity;
  const cells: { cx: number; cy: number; z: number }[] = [];
  const [xLo, xHi] = xRange;
  const [yLo, yHi] = yRange;

  for (let i = 0; i < grid; i += 1) {
    for (let j = 0; j < grid; j += 1) {
      const x = xLo + ((xHi - xLo) * (i + 0.5)) / grid;
      const y = yLo + ((yHi - yLo) * (j + 0.5)) / grid;
      const z = objective([x, y]);
      zMin = Math.min(zMin, z);
      zMax = Math.max(zMax, z);
      cells.push({ cx: x, cy: y, z });
    }
  }

  const toSvg = (x: number, y: number) => ({
    sx: pad.left + ((x - xLo) / (xHi - xLo)) * innerW,
    sy: pad.top + innerH - ((y - yLo) / (yHi - yLo)) * innerH,
  });

  const cellW = innerW / grid;
  const cellH = innerH / grid;

  const pathD = path
    .map((pt, idx) => {
      const { sx, sy } = toSvg(pt[0]!, pt[1]!);
      return `${idx === 0 ? "M" : "L"} ${sx} ${sy}`;
    })
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full max-w-xl rounded-lg border border-slate-800 bg-slate-950"
      role="img"
      aria-label="Optimization path on objective contour"
    >
      {cells.map((cell, idx) => {
        const t = (cell.z - zMin) / Math.max(zMax - zMin, 1e-12);
        const hue = 160 - t * 100;
        const { sx, sy } = toSvg(cell.cx, cell.cy);
        return (
          <rect
            key={idx}
            x={sx - cellW / 2}
            y={sy - cellH / 2}
            width={cellW + 1}
            height={cellH + 1}
            fill={`hsl(${hue} 45% ${18 + t * 22}%)`}
          />
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
        const { sx, sy } = toSvg(pt[0]!, pt[1]!);
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
      <text
        x={pad.left + innerW / 2}
        y={height - 8}
        textAnchor="middle"
        className="fill-slate-500 text-[10px]"
      >
        x₁
      </text>
      <text
        x={12}
        y={pad.top + innerH / 2}
        textAnchor="middle"
        transform={`rotate(-90 12 ${pad.top + innerH / 2})`}
        className="fill-slate-500 text-[10px]"
      >
        x₂
      </text>
    </svg>
  );
}
