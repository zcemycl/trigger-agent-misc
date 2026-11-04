import type { Vec } from "../secondOrder";
import { ContourPathChart } from "./ContourPathChart";
import { SurfacePathChart } from "./SurfacePathChart";

interface ObjectiveLandscapeViewProps {
  objective: (x: Vec) => number;
  path: Vec[];
  xRange?: [number, number];
  yRange?: [number, number];
}

export function ObjectiveLandscapeView({
  objective,
  path,
  xRange,
  yRange,
}: ObjectiveLandscapeViewProps) {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <ContourPathChart
        objective={objective}
        path={path}
        xRange={xRange}
        yRange={yRange}
      />
      <SurfacePathChart
        objective={objective}
        path={path}
        xRange={xRange}
        yRange={yRange}
      />
    </div>
  );
}
