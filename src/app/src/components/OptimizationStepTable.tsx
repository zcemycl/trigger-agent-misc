import { formatVector, type Vector } from "../gradientMethod";

export interface OptimizationStepRow {
  iteration: number;
  x: Vector;
  delta: Vector;
  xNext: Vector;
  objective: number;
  detail: string;
}

interface OptimizationStepTableProps {
  steps: OptimizationStepRow[];
  maxRows?: number;
  caption?: string;
}

export function OptimizationStepTable({
  steps,
  maxRows = 12,
  caption = "Step-by-step iterations",
}: OptimizationStepTableProps) {
  if (steps.length === 0) {
    return null;
  }

  const visible = steps.slice(0, maxRows);
  const hidden = steps.length - visible.length;

  return (
    <div className="overflow-x-auto rounded-lg border border-slate-800">
      <table className="w-full min-w-[40rem] text-left text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead className="bg-slate-800/60 text-xs uppercase tracking-wide text-slate-400">
          <tr>
            <th scope="col" className="px-3 py-2 font-medium">
              Step k
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              x<sub>k</sub> (vector)
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              f(x<sub>k</sub>)
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              Vector update at k
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              Δx<sub>k</sub> (vector)
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              x<sub>k+1</sub> (vector)
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800 text-slate-300">
          {visible.map((row) => (
            <tr key={row.iteration} className="hover:bg-slate-800/40">
              <td className="px-3 py-2 font-mono text-slate-400">
                {row.iteration}
              </td>
              <td className="px-3 py-2 font-mono">{formatVector(row.x)}</td>
              <td className="px-3 py-2 font-mono">
                {row.objective.toExponential(3)}
              </td>
              <td className="px-3 py-2 text-xs leading-snug text-slate-400">
                {row.detail}
              </td>
              <td className="px-3 py-2 font-mono text-amber-200/90">
                {formatVector(row.delta)}
              </td>
              <td className="px-3 py-2 font-mono text-emerald-300/90">
                {formatVector(row.xNext)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {hidden > 0 ? (
        <p className="border-t border-slate-800 px-3 py-2 font-mono text-xs text-slate-500">
          +{hidden} / {steps.length}
        </p>
      ) : null}
    </div>
  );
}
