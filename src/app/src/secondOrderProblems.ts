/** Benchmark objectives mirrored from Python second-order evals. */

export type Vec = number[];

export function shiftedQuadratic(x: Vec): number {
  const targets = [1, 2];
  let sum = 0;
  for (let i = 0; i < x.length; i += 1) {
    const t = targets[i] ?? 0;
    sum += 0.5 * (x[i]! - t) ** 2;
  }
  return sum;
}

export function shiftedQuadraticGrad(x: Vec): Vec {
  return [x[0]! - 1, x[1]! - 2];
}

export function shiftedQuadraticHess(_x: Vec): number[][] {
  return [
    [1, 0],
    [0, 1],
  ];
}

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
