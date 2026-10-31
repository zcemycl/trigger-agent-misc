import type { Phi } from "../lineSearch";

export interface ChartMarker {
  alpha: number;
  label?: string;
  color?: string;
}

export interface ChartBand {
  alpha: number;
  yLower: number;
  yUpper: number;
  label?: string;
}

interface PhiChartProps {
  phi: Phi;
  alphaMax?: number;
  height?: number;
  markers?: ChartMarker[];
  armijoLineAtAlpha?: number;
  phi0?: number;
  phiPrime0?: number;
  c1?: number;
  band?: ChartBand;
  bracket?: { a: number; b: number };
}

export function PhiChart({
  phi,
  alphaMax = 2,
  height = 200,
  markers = [],
  armijoLineAtAlpha,
  phi0 = phi(0),
  phiPrime0,
  c1 = 1e-4,
  band,
  bracket,
}: PhiChartProps) {
  const width = 480;
  const pad = { top: 16, right: 16, bottom: 32, left: 48 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const samples = 80;
  const points: { x: number; y: number }[] = [];
  let yMin = 0;
  let yMax = phi0;
  for (let i = 0; i <= samples; i += 1) {
    const a = (alphaMax * i) / samples;
    const y = phi(a);
    points.push({ x: a, y });
    yMin = Math.min(yMin, y);
    yMax = Math.max(yMax, y);
  }
  if (band) {
    yMin = Math.min(yMin, band.yLower, band.yUpper);
    yMax = Math.max(yMax, band.yLower, band.yUpper);
  }
  if (armijoLineAtAlpha != null && phiPrime0 != null) {
    const armijoY = phi0 + c1 * armijoLineAtAlpha * phiPrime0;
    yMin = Math.min(yMin, armijoY);
    yMax = Math.max(yMax, armijoY);
  }
  const yPad = (yMax - yMin) * 0.08 + 0.05;
  yMin -= yPad;
  yMax += yPad;

  const sx = (a: number) => pad.left + (a / alphaMax) * innerW;
  const sy = (y: number) =>
    pad.top + innerH - ((y - yMin) / (yMax - yMin)) * innerH;

  const pathD = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.x)} ${sy(p.y)}`)
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full max-w-xl rounded-lg border border-slate-700 bg-slate-900/60"
      role="img"
      aria-label="Plot of phi(alpha) along the search ray"
    >
      <line
        x1={pad.left}
        y1={pad.top + innerH}
        x2={pad.left + innerW}
        y2={pad.top + innerH}
        stroke="#475569"
        strokeWidth={1}
      />
      <line
        x1={pad.left}
        y1={pad.top}
        x2={pad.left}
        y2={pad.top + innerH}
        stroke="#475569"
        strokeWidth={1}
      />
      <text
        x={pad.left + innerW / 2}
        y={height - 6}
        textAnchor="middle"
        className="fill-slate-400 text-[10px]"
      >
        α
      </text>
      <text
        x={12}
        y={pad.top + innerH / 2}
        textAnchor="middle"
        transform={`rotate(-90 12 ${pad.top + innerH / 2})`}
        className="fill-slate-400 text-[10px]"
      >
        φ(α)
      </text>
      <path d={pathD} fill="none" stroke="#34d399" strokeWidth={2} />
      {bracket && (
        <rect
          x={sx(bracket.a)}
          y={pad.top}
          width={Math.max(sx(bracket.b) - sx(bracket.a), 2)}
          height={innerH}
          fill="#6366f1"
          fillOpacity={0.12}
          stroke="#818cf8"
          strokeWidth={1}
          strokeDasharray="4 2"
        />
      )}
      {band && (
        <>
          <line
            x1={sx(band.alpha)}
            y1={sy(band.yLower)}
            x2={sx(band.alpha)}
            y2={sy(band.yUpper)}
            stroke="#fbbf24"
            strokeWidth={3}
            strokeLinecap="round"
          />
          {band.label && (
            <text
              x={sx(band.alpha) + 6}
              y={sy((band.yLower + band.yUpper) / 2)}
              className="fill-amber-200 text-[9px]"
            >
              {band.label}
            </text>
          )}
        </>
      )}
      {armijoLineAtAlpha != null && phiPrime0 != null && (
        <>
          <circle
            cx={sx(armijoLineAtAlpha)}
            cy={sy(phi0 + c1 * armijoLineAtAlpha * phiPrime0)}
            r={4}
            fill="#f472b6"
          />
          <text
            x={sx(armijoLineAtAlpha) + 6}
            y={sy(phi0 + c1 * armijoLineAtAlpha * phiPrime0) - 4}
            className="fill-pink-300 text-[9px]"
          >
            Armijo bound
          </text>
        </>
      )}
      {markers.map((m) => (
        <g key={`${m.alpha}-${m.label ?? ""}`}>
          <line
            x1={sx(m.alpha)}
            y1={pad.top}
            x2={sx(m.alpha)}
            y2={pad.top + innerH}
            stroke={m.color ?? "#a78bfa"}
            strokeWidth={1.5}
            strokeDasharray="3 3"
            opacity={0.85}
          />
          <circle
            cx={sx(m.alpha)}
            cy={sy(phi(m.alpha))}
            r={5}
            fill={m.color ?? "#a78bfa"}
          />
          {m.label && (
            <text
              x={sx(m.alpha) + 4}
              y={sy(phi(m.alpha)) - 8}
              className="fill-slate-200 text-[9px]"
            >
              {m.label}
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}
