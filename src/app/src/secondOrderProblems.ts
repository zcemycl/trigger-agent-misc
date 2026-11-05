/** Benchmark objectives mirrored from Python second-order evals. */

export type Vec = number[];

export const SHIFTED_QUADRATIC_TARGETS_2D = [1, 2] as const;
export const SHIFTED_QUADRATIC_TARGETS_3D = [1, 2, 3] as const;

export function makeShiftedQuadratic(targets: readonly number[]) {
  const objective = (x: Vec): number => {
    let sum = 0;
    for (let i = 0; i < x.length; i += 1) {
      const t = targets[i] ?? 0;
      sum += 0.5 * (x[i]! - t) ** 2;
    }
    return sum;
  };
  const gradient = (x: Vec): Vec => x.map((xi, i) => xi - (targets[i] ?? 0));
  const hessian = (_x: Vec): number[][] => {
    const n = targets.length;
    return Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
    );
  };
  return { objective, gradient, hessian, targets: [...targets] };
}

const shifted2d = makeShiftedQuadratic(SHIFTED_QUADRATIC_TARGETS_2D);

export function shiftedQuadratic(x: Vec): number {
  return shifted2d.objective(x);
}

export function shiftedQuadraticGrad(x: Vec): Vec {
  return shifted2d.gradient(x);
}

export function shiftedQuadraticHess(_x: Vec): number[][] {
  return shifted2d.hessian(_x);
}

export const shiftedQuadratic3D = makeShiftedQuadratic(
  SHIFTED_QUADRATIC_TARGETS_3D,
);

export function rosenbrockResidual(x: Vec): Vec {
  return [x[0]! - 1, 10 * (x[1]! - x[0]! ** 2)];
}

export function rosenbrockJacobian(x: Vec): number[][] {
  return [
    [1, 0],
    [-20 * x[0]!, 10],
  ];
}

export function leastSquaresValue(residual: (x: Vec) => Vec, x: Vec): number {
  const r = residual(x);
  return 0.5 * r.reduce((acc, ri) => acc + ri * ri, 0);
}

export function illConditionedQuadraticObjective(x: Vec): number {
  return 0.5 * (50 * (x[0]! - 1) ** 2 + 0.5 * (x[1]! - 2) ** 2);
}

export function illConditionedQuadraticGradient(x: Vec): Vec {
  return [50 * (x[0]! - 1), 0.5 * (x[1]! - 2)];
}

export function illConditionedQuadraticHessian(_x: Vec): number[][] {
  return [
    [50, 0],
    [0, 0.5],
  ];
}

export function boothObjective(x: Vec): number {
  const u = x[0]! + 2 * x[1]! - 7;
  const v = 2 * x[0]! + x[1]! - 5;
  return u * u + v * v;
}

export function boothGradient(x: Vec): Vec {
  const u = x[0]! + 2 * x[1]! - 7;
  const v = 2 * x[0]! + x[1]! - 5;
  return [2 * u + 4 * v, 4 * u + 2 * v];
}

export function boothHessian(_x: Vec): number[][] {
  return [
    [10, 8],
    [8, 10],
  ];
}

export function rosenbrockObjective(x: Vec): number {
  const r = rosenbrockResidual(x);
  return 0.5 * (r[0]! ** 2 + r[1]! ** 2);
}

export function rosenbrockObjectiveGradient(x: Vec): Vec {
  const x0 = x[0]!;
  const x1 = x[1]!;
  const u = x1 - x0 ** 2;
  return [x0 - 1 - 200 * x0 * u, 100 * u];
}

export function rosenbrockObjectiveHessian(x: Vec): number[][] {
  const x0 = x[0]!;
  const x1 = x[1]!;
  const u = x1 - x0 ** 2;
  return [
    [1 - 200 * u + 400 * x0 ** 2, -200 * x0],
    [-200 * x0, 100],
  ];
}

export function bealeObjective(x: Vec): number {
  const x1 = x[0]!;
  const x2 = x[1]!;
  const t1 = 1.5 - x1 + x1 * x2;
  const t2 = 2.25 - x1 + x1 * x2 ** 2;
  const t3 = 2.625 - x1 + x1 * x2 ** 3;
  return t1 * t1 + t2 * t2 + t3 * t3;
}

export function bealeGradient(x: Vec): Vec {
  const x1 = x[0]!;
  const x2 = x[1]!;
  const t1 = 1.5 - x1 + x1 * x2;
  const t2 = 2.25 - x1 + x1 * x2 ** 2;
  const t3 = 2.625 - x1 + x1 * x2 ** 3;
  const g1 =
    2 * t1 * (-1 + x2) + 2 * t2 * (-1 + x2 ** 2) + 2 * t3 * (-1 + x2 ** 3);
  const g2 = 2 * t1 * x1 + 2 * t2 * (2 * x1 * x2) + 2 * t3 * (3 * x1 * x2 ** 2);
  return [g1, g2];
}

export function bealeHessian(x: Vec): number[][] {
  const x1 = x[0]!;
  const x2 = x[1]!;
  const t1 = 1.5 - x1 + x1 * x2;
  const t2 = 2.25 - x1 + x1 * x2 ** 2;
  const t3 = 2.625 - x1 + x1 * x2 ** 3;
  const dt1dx1 = -1 + x2;
  const dt1dx2 = x1;
  const dt2dx1 = -1 + x2 ** 2;
  const dt2dx2 = 2 * x1 * x2;
  const dt3dx1 = -1 + x2 ** 3;
  const dt3dx2 = 3 * x1 * x2 ** 2;
  const h11 = 2 * (dt1dx1 ** 2 + dt2dx1 ** 2 + dt3dx1 ** 2);
  const h22 =
    2 *
    (dt1dx2 ** 2 +
      dt2dx2 ** 2 +
      dt3dx2 ** 2 +
      2 * t2 * x1 +
      2 * t2 * x2 +
      2 * t3 * (3 * x1 * x2));
  const h12 =
    2 *
    (dt1dx1 * dt1dx2 +
      t1 +
      dt2dx1 * dt2dx2 +
      2 * t2 * x2 +
      dt3dx1 * dt3dx2 +
      3 * t3 * x2 ** 2);
  return [
    [h11, h12],
    [h12, h22],
  ];
}

const EXPONENTIAL_FIT_T = [0, 0.5, 1, 1.5, 2] as const;
const EXPONENTIAL_FIT_Y = [1, 0.72, 0.55, 0.38, 0.28] as const;

export function exponentialFitResidual(x: Vec): Vec {
  const a = x[0]!;
  const b = x[1]!;
  return EXPONENTIAL_FIT_T.map(
    (t, i) => a * Math.exp(-b * t) - EXPONENTIAL_FIT_Y[i]!,
  );
}

export function exponentialFitJacobian(x: Vec): number[][] {
  const a = x[0]!;
  const b = x[1]!;
  return EXPONENTIAL_FIT_T.map((t) => {
    const e = Math.exp(-b * t);
    return [e, -a * t * e];
  });
}

export function freudensteinRothResidual(x: Vec): Vec {
  const x1 = x[0]!;
  const x2 = x[1]!;
  const r1 = -13 + x1 + ((5 - x2) * x2 - 2) * x2;
  const r2 = -29 + x1 + ((1 + x2) * x2 - 14) * x2;
  return [r1, r2];
}

export function freudensteinRothJacobian(x: Vec): number[][] {
  const x2 = x[1]!;
  const dr1dx2 = 10 * x2 - 3 * x2 ** 2 - 2;
  const dr2dx2 = 3 * x2 ** 2 + 2 * x2 - 14;
  return [
    [1, dr1dx2],
    [1, dr2dx2],
  ];
}
