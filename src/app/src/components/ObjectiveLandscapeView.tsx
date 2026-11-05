import type { Vec } from "../secondOrder";
import { ContourPathChart } from "./ContourPathChart";
import type { PathHighlightRange } from "./pathHighlight";
import { SurfacePathChart } from "./SurfacePathChart";

interface ObjectiveLandscapeViewProps {
  objective: (x: Vec) => number;
  path: Vec[];
  xRange?: [number, number];
  yRange?: [number, number];
  highlightPathRange?: PathHighlightRange | null;
}

export function ObjectiveLandscapeView({
  objective,
  path,
  xRange,
  yRange,
  highlightPathRange = null,
}: ObjectiveLandscapeViewProps) {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <ContourPathChart
        objective={objective}
        path={path}
        xRange={xRange}
        yRange={yRange}
        highlightPathRange={highlightPathRange}
      />
      <SurfacePathChart
        objective={objective}
        path={path}
        xRange={xRange}
        yRange={yRange}
        highlightPathRange={highlightPathRange}
      />
    </div>
  );
}
