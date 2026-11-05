import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  getLeastSquaresProblem,
  getSmoothProblem,
  leastSquaresExampleById,
  smoothExampleById,
  type LeastSquaresExampleId,
  type SmoothOptimizationExampleId,
} from "../secondOrderExampleCatalog";
import {
  wrapGradientVec,
  wrapObjectiveVec,
  wrapResidualVec,
} from "../demoNoise";
import { leastSquaresValue } from "../secondOrderProblems";
import { useDemoNoise } from "./DemoNoise";
import { DemoExamplePicker } from "./DemoExamplePicker";
import { MathBlock } from "./MathBlock";

export function useSmoothExampleProblem(
  exampleId: SmoothOptimizationExampleId,
) {
  const noise = useDemoNoise();
  return useMemo(() => {
    const base = getSmoothProblem(exampleId);
    return {
      objective: wrapObjectiveVec(base.objective, noise),
      gradient: wrapGradientVec(base.gradient, noise),
      hessian: base.hessian,
    };
  }, [exampleId, noise.enabled, noise.seed]);
}

export function useLeastSquaresExampleProblem(
  exampleId: LeastSquaresExampleId,
) {
  const noise = useDemoNoise();
  return useMemo(() => {
    const base = getLeastSquaresProblem(exampleId);
    const residual = wrapResidualVec(base.residual, noise);
    return {
      residual,
      jacobian: base.jacobian,
      objective: (x: number[]) => leastSquaresValue(residual, x),
    };
  }, [exampleId, noise.enabled, noise.seed]);
}

export function useSmoothExampleStart(exampleId: SmoothOptimizationExampleId) {
  const meta = smoothExampleById(exampleId);
  const [start, setStart] = useState(meta.defaultStart);
  useEffect(() => {
    setStart(meta.defaultStart);
  }, [exampleId, meta.defaultStart]);
  return { start, setStart, meta };
}

export function useLeastSquaresExampleStart(exampleId: LeastSquaresExampleId) {
  const meta = leastSquaresExampleById(exampleId);
  const [start, setStart] = useState(meta.defaultStart);
  useEffect(() => {
    setStart(meta.defaultStart);
  }, [exampleId, meta.defaultStart]);
  return { start, setStart, meta };
}

export function SmoothExampleControls({
  exampleId,
  onExampleIdChange,
  examples,
  start,
  onStartChange,
  hideStartSliders = false,
  children,
}: {
  exampleId: SmoothOptimizationExampleId;
  onExampleIdChange: (id: SmoothOptimizationExampleId) => void;
  examples: readonly { id: SmoothOptimizationExampleId; label: string }[];
  start: [number, number];
  onStartChange: (x: number, y: number) => void;
  hideStartSliders?: boolean;
  children?: ReactNode;
}) {
  const meta = smoothExampleById(exampleId);
  return (
    <div className="space-y-3">
      <DemoExamplePicker
        label="Example objective"
        examples={examples}
        value={exampleId}
        onChange={onExampleIdChange}
      />
      <MathBlock tex={meta.objectiveTex} />
      {children}
      {hideStartSliders ? null : (
        <div className="flex flex-col gap-3 sm:flex-row">
          <label className="flex flex-1 flex-col gap-1 text-sm text-slate-300">
            Start x₁ = {start[0].toFixed(2)}
            <input
              type="range"
              min={meta.xRange[0]}
              max={meta.xRange[1]}
              step={0.1}
              value={start[0]}
              onChange={(e) => onStartChange(Number(e.target.value), start[1])}
              className="accent-emerald-500"
            />
          </label>
          <label className="flex flex-1 flex-col gap-1 text-sm text-slate-300">
            Start x₂ = {start[1].toFixed(2)}
            <input
              type="range"
              min={meta.yRange[0]}
              max={meta.yRange[1]}
              step={0.1}
              value={start[1]}
              onChange={(e) => onStartChange(start[0], Number(e.target.value))}
              className="accent-emerald-500"
            />
          </label>
        </div>
      )}
    </div>
  );
}

export function LeastSquaresExampleControls({
  exampleId,
  onExampleIdChange,
  examples,
  start,
  onStartChange,
}: {
  exampleId: LeastSquaresExampleId;
  onExampleIdChange: (id: LeastSquaresExampleId) => void;
  examples: readonly { id: LeastSquaresExampleId; label: string }[];
  start: [number, number];
  onStartChange: (x: number, y: number) => void;
}) {
  const meta = leastSquaresExampleById(exampleId);
  return (
    <div className="space-y-3">
      <DemoExamplePicker
        label="Example least-squares problem"
        examples={examples}
        value={exampleId}
        onChange={onExampleIdChange}
      />
      <MathBlock tex={meta.residualTex} />
      <div className="flex flex-col gap-3 sm:flex-row">
        <label className="flex flex-1 flex-col gap-1 text-sm text-slate-300">
          Start x₁ = {start[0].toFixed(2)}
          <input
            type="range"
            min={meta.xRange[0]}
            max={meta.xRange[1]}
            step={0.1}
            value={start[0]}
            onChange={(e) => onStartChange(Number(e.target.value), start[1])}
            className="accent-emerald-500"
          />
        </label>
        <label className="flex flex-1 flex-col gap-1 text-sm text-slate-300">
          Start x₂ = {start[1].toFixed(2)}
          <input
            type="range"
            min={meta.yRange[0]}
            max={meta.yRange[1]}
            step={0.1}
            value={start[1]}
            onChange={(e) => onStartChange(start[0], Number(e.target.value))}
            className="accent-emerald-500"
          />
        </label>
      </div>
    </div>
  );
}
