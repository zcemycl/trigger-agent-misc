/** Demo example metadata and problem lookup for second-order React demos. */

import type {
  Gradient,
  Hessian,
  Jacobian,
  Objective,
  Residual,
} from "./secondOrder";
import {
  bealeObjective,
  bealeGradient,
  bealeHessian,
  boothObjective,
  boothGradient,
  boothHessian,
  exponentialFitJacobian,
  exponentialFitResidual,
  freudensteinRothJacobian,
  freudensteinRothResidual,
  illConditionedQuadraticGradient,
  illConditionedQuadraticHessian,
  illConditionedQuadraticObjective,
  rosenbrockObjective,
  rosenbrockObjectiveGradient,
  rosenbrockObjectiveHessian,
  rosenbrockJacobian,
  rosenbrockResidual,
  shiftedQuadratic,
  shiftedQuadratic3D,
  shiftedQuadraticGrad,
  shiftedQuadraticHess,
} from "./secondOrderProblems";

export interface PlotRange {
  xRange: [number, number];
  yRange: [number, number];
}

export interface SmoothOptimizationExample extends PlotRange {
  id: string;
  label: string;
  /** LaTeX for the objective (display only). */
  objectiveTex: string;
  defaultStart: [number, number];
  supports3d?: boolean;
}

export interface LeastSquaresExample extends PlotRange {
  id: string;
  label: string;
  residualTex: string;
  defaultStart: [number, number];
}

export const SMOOTH_OPTIMIZATION_EXAMPLES: readonly SmoothOptimizationExample[] =
  [
    {
      id: "shifted-quadratic",
      label: "Shifted quadratic",
      objectiveTex:
        "f(\\mathbf{x}) = \\tfrac12\\bigl((x_1-1)^2 + (x_2-2)^2\\bigr)",
      defaultStart: [0, 0],
      xRange: [-1, 4],
      yRange: [-1, 4],
      supports3d: true,
    },
    {
      id: "ill-conditioned",
      label: "Ill-conditioned quadratic",
      objectiveTex:
        "f(\\mathbf{x}) = \\tfrac12\\bigl(50(x_1-1)^2 + 0.5(x_2-2)^2\\bigr)",
      defaultStart: [0, 0],
      xRange: [-1, 4],
      yRange: [-1, 4],
    },
    {
      id: "booth",
      label: "Booth",
      objectiveTex: "f(\\mathbf{x}) = (x_1 + 2x_2 - 7)^2 + (2x_1 + x_2 - 5)^2",
      defaultStart: [0, 0],
      xRange: [-1, 4],
      yRange: [-1, 4],
    },
    {
      id: "rosenbrock",
      label: "Rosenbrock",
      objectiveTex: "f(\\mathbf{x}) = (1-x_1)^2 + 100(x_2-x_1^2)^2",
      defaultStart: [0, 0],
      xRange: [-1, 2],
      yRange: [-1, 2],
    },
    {
      id: "beale",
      label: "Beale",
      objectiveTex:
        "f(\\mathbf{x}) = \\sum_{i=1}^3 \\bigl(a_i - x_1 + x_1 x_2^{\\,i}\\bigr)^2",
      defaultStart: [1, 1],
      xRange: [-1, 4],
      yRange: [-1, 3],
    },
  ] as const;

export const LEAST_SQUARES_EXAMPLES: readonly LeastSquaresExample[] = [
  {
    id: "rosenbrock",
    label: "Rosenbrock residual",
    residualTex:
      "\\mathbf{r}(\\mathbf{x}) = \\begin{pmatrix}x_1-1\\\\10(x_2-x_1^2)\\end{pmatrix}",
    defaultStart: [0, 0],
    xRange: [-1, 2],
    yRange: [-1, 2],
  },
  {
    id: "exponential-fit",
    label: "Exponential fit",
    residualTex: "r_i = a e^{-b t_i} - y_i,\\quad \\mathbf{x}=(a,b)",
    defaultStart: [1, 0.3],
    xRange: [0, 2],
    yRange: [0, 1.5],
  },
  {
    id: "freudenstein-roth",
    label: "Freudenstein–Roth",
    residualTex:
      "\\mathbf{r}(\\mathbf{x}) \\text{ (two nonlinear equations in } x_1,x_2\\text{)}",
    defaultStart: [0.5, 2],
    xRange: [-2, 6],
    yRange: [-1, 5],
  },
] as const;

export type SmoothOptimizationExampleId =
  (typeof SMOOTH_OPTIMIZATION_EXAMPLES)[number]["id"];

export type LeastSquaresExampleId =
  (typeof LEAST_SQUARES_EXAMPLES)[number]["id"];

export function smoothExampleById(
  id: SmoothOptimizationExampleId,
): SmoothOptimizationExample {
  const found = SMOOTH_OPTIMIZATION_EXAMPLES.find((e) => e.id === id);
  if (!found) {
    return SMOOTH_OPTIMIZATION_EXAMPLES[0]!;
  }
  return found;
}

export function leastSquaresExampleById(
  id: LeastSquaresExampleId,
): LeastSquaresExample {
  const found = LEAST_SQUARES_EXAMPLES.find((e) => e.id === id);
  if (!found) {
    return LEAST_SQUARES_EXAMPLES[0]!;
  }
  return found;
}

export function getSmoothProblem(id: SmoothOptimizationExampleId): {
  objective: Objective;
  gradient: Gradient;
  hessian: Hessian;
} {
  switch (id) {
    case "shifted-quadratic":
      return {
        objective: shiftedQuadratic,
        gradient: shiftedQuadraticGrad,
        hessian: shiftedQuadraticHess,
      };
    case "ill-conditioned":
      return {
        objective: illConditionedQuadraticObjective,
        gradient: illConditionedQuadraticGradient,
        hessian: illConditionedQuadraticHessian,
      };
    case "booth":
      return {
        objective: boothObjective,
        gradient: boothGradient,
        hessian: boothHessian,
      };
    case "rosenbrock":
      return {
        objective: rosenbrockObjective,
        gradient: rosenbrockObjectiveGradient,
        hessian: rosenbrockObjectiveHessian,
      };
    case "beale":
      return {
        objective: bealeObjective,
        gradient: bealeGradient,
        hessian: bealeHessian,
      };
    default:
      return getSmoothProblem("shifted-quadratic");
  }
}

export function getShiftedQuadratic3dProblem(): {
  objective: Objective;
  gradient: Gradient;
  hessian: Hessian;
} {
  return {
    objective: shiftedQuadratic3D.objective,
    gradient: shiftedQuadratic3D.gradient,
    hessian: shiftedQuadratic3D.hessian,
  };
}

export function getLeastSquaresProblem(id: LeastSquaresExampleId): {
  residual: Residual;
  jacobian: Jacobian;
} {
  switch (id) {
    case "rosenbrock":
      return { residual: rosenbrockResidual, jacobian: rosenbrockJacobian };
    case "exponential-fit":
      return {
        residual: exponentialFitResidual,
        jacobian: exponentialFitJacobian,
      };
    case "freudenstein-roth":
      return {
        residual: freudensteinRothResidual,
        jacobian: freudensteinRothJacobian,
      };
    default:
      return getLeastSquaresProblem("rosenbrock");
  }
}
