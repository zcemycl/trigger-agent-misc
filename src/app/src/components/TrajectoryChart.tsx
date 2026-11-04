interface TrajectoryChartProps {
  objective: (x: number) => number;
  history: number[];
  xMin?: number;
  xMax?: number;
  height?: number;
}

export function TrajectoryChart({
  objective,
  history,
  xMin = -3,
  xMax = 4,
  height = 200,
}: TrajectoryChartProps) {
  const width = 480;
  const pad = { top: 16, right: 16, bottom: 32, left: 48 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const samples = 80;
  const points: { x: number; y: number }[] = [];
  let yMin = 0;
  let yMax = objective(history[0] ?? 0);
  for (let i = 0; i <= samples; i += 1) {
    const x = xMin + ((xMax - xMin) * i) / samples;
    const y = objective(x);
    points.push({ x, y });
    yMin = Math.min(yMin, y);
    yMax = Math.max(yMax, y);
  }
  for (const x of history) {
    yMin = Math.min(yMin, objective(x));
    yMax = Math.max(yMax, objective(x));
  }
  const yPad = (yMax - yMin) * 0.08 + 0.05;
  yMin -= yPad;
  yMax += yPad;

  const sx = (x: number) => pad.left + ((x - xMin) / (xMax - xMin)) * innerW;
  const sy = (y: number) =>
    pad.top + innerH - ((y - yMin) / (yMax - yMin)) * innerH;

  const pathD = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.x)} ${sy(p.y)}`)
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="h-auto w-full max-w-xl text-slate-400"
      role="img"
      aria-label="Objective curve with optimizer iterates"
    >
      <path
        d={pathD}
        fill="none"
        stroke="currentColor"
        strokeWidth={2}
        className="text-slate-500"
      />
      {history.map((x, i) => (
        <circle
          key={`${x}-${i}`}
          cx={sx(x)}
          cy={sy(objective(x))}
          r={i === history.length - 1 ? 5 : 3.5}
          className={
            i === history.length - 1 ? "fill-emerald-400" : "fill-indigo-400"
          }
        />
      ))}
      <line
        x1={pad.left}
        y1={pad.top + innerH}
        x2={pad.left + innerW}
        y2={pad.top + innerH}
        stroke="currentColor"
        strokeWidth={1}
      />
      <line
        x1={pad.left}
        y1={pad.top}
        x2={pad.left}
        y2={pad.top + innerH}
        stroke="currentColor"
        strokeWidth={1}
      />
      <text
        x={pad.left + innerW / 2}
        y={height - 6}
        textAnchor="middle"
        fontSize={11}
      >
        x
      </text>
      <text
        x={14}
        y={pad.top + innerH / 2}
        textAnchor="middle"
        fontSize={11}
        transform={`rotate(-90 14 ${pad.top + innerH / 2})`}
      >
        f(x)
      </text>
    </svg>
  );
}
