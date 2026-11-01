/** Multivariate second-order optimization (mirrors second_order.py). */

import { backtrackingArmijo } from "./lineSearch";

export type Vec = number[];
export type Mat = number[][];
export type Objective = (x: Vec) => number;
export type Gradient = (x: Vec) => Vec;
export type Hessian = (x: Vec) => Mat;
export type Residual = (x: Vec) => Vec;
export type Jacobian = (x: Vec) => Mat;

export interface OptimizeResult {
  x: Vec;
  value: number;
  iterations: number;
  converged: boolean;
  path: Vec[];
}

export function vecAdd(a: Vec, b: Vec): Vec {
  return a.map((v, i) => v + b[i]!);
}

export function vecSub(a: Vec, b: Vec): Vec {
  return a.map((v, i) => v - b[i]!);
}

export function vecScale(a: Vec, s: number): Vec {
  return a.map((v) => s * v);
}

export function dot(a: Vec, b: Vec): number {
  return a.reduce((acc, v, i) => acc + v * b[i]!, 0);
}

export function norm2(a: Vec): number {
  return Math.sqrt(dot(a, a));
}

export function matVec(a: Mat, x: Vec): Vec {
  return a.map((row) => dot(row, x));
}

export function matTranspose(a: Mat): Mat {
  const n = a[0]?.length ?? 0;
  return Array.from({ length: n }, (_, j) => a.map((row) => row[j]!));
}

export function matAdd(a: Mat, b: Mat): Mat {
  return a.map((row, i) => row.map((v, j) => v + b[i]![j]!));
}

export function matScale(a: Mat, s: number): Mat {
  return a.map((row) => row.map((v) => s * v));
}

export function identity(n: number): Mat {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );
}

export function matMul(a: Mat, b: Mat): Mat {
  const bt = matTranspose(b);
  return a.map((row) => bt.map((col) => dot(row, col)));
}

export function solveLinear(a: Mat, b: Vec, tol = 1e-14): Vec {
  const n = b.length;
  const aug = a.map((row, i) => [...row, b[i]!]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let r = col + 1; r < n; r += 1) {
      if (Math.abs(aug[r]![col]!) > Math.abs(aug[pivot]![col]!)) {
        pivot = r;
      }
    }
    if (Math.abs(aug[pivot]![col]!) < tol) {
      throw new Error("singular matrix");
    }
    if (pivot !== col) {
      [aug[col], aug[pivot]] = [aug[pivot]!, aug[col]!];
    }
    const div = aug[col]![col]!;
    for (let j = col; j <= n; j += 1) {
      aug[col]![j]! /= div;
    }
    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = aug[row]![col]!;
      if (Math.abs(factor) < tol) continue;
      for (let j = col; j <= n; j += 1) {
        aug[row]![j]! -= factor * aug[col]![j]!;
      }
    }
  }
  return aug.map((row) => row[n]!);
}

function checkConvergence(
  grad: Vec,
  gradTol: number,
  x: Vec,
  xPrev: Vec | null,
  xTol: number,
): boolean {
  if (norm2(grad) <= gradTol) return true;
  if (xPrev && norm2(vecSub(x, xPrev)) <= xTol) return true;
  return false;
}

function armijoStep(
  objective: Objective,
  x: Vec,
  direction: Vec,
  g: Vec,
  alpha0: number,
  c1: number,
  rho: number,
): number {
  let dir = direction;
  let phiPrime0 = dot(g, dir);
  if (phiPrime0 >= 0) {
    dir = vecScale(g, -1);
    phiPrime0 = dot(g, dir);
  }
  const ls = backtrackingArmijo(
    (alpha) => objective(vecAdd(x, vecScale(dir, alpha))),
    phiPrime0,
    { alpha0, c1, rho },
  );
  return ls.alpha;
}

export function newtonsMethod(
  objective: Objective,
  gradient: Gradient,
  hessian: Hessian,
  x0: Vec,
  options: {
    gradTol?: number;
    xTol?: number;
    maxIter?: number;
  } = {},
): OptimizeResult {
  const gradTol = options.gradTol ?? 1e-8;
  const xTol = options.xTol ?? 1e-10;
  const maxIter = options.maxIter ?? 100;
  let x = [...x0];
  const path: Vec[] = [x];
  let iterations = 0;
  let xPrev: Vec | null = null;

  while (iterations < maxIter) {
    const g = gradient(x);
    if (checkConvergence(g, gradTol, x, xPrev, xTol)) {
      return {
        x,
        value: objective(x),
        iterations,
        converged: true,
        path,
      };
    }
    try {
      const step = vecScale(solveLinear(hessian(x), g), -1);
      xPrev = x;
      x = vecAdd(x, step);
      path.push([...x]);
    } catch {
      return {
        x,
        value: objective(x),
        iterations,
        converged: false,
        path,
      };
    }
    iterations += 1;
  }
  const g = gradient(x);
  return {
    x,
    value: objective(x),
    iterations,
    converged: norm2(g) <= gradTol,
    path,
  };
}

export function dampedNewtonMethod(
  objective: Objective,
  gradient: Gradient,
  hessian: Hessian,
  x0: Vec,
  options: {
    gradTol?: number;
    xTol?: number;
    maxIter?: number;
    alpha0?: number;
    c1?: number;
    rho?: number;
  } = {},
): OptimizeResult {
  const gradTol = options.gradTol ?? 1e-8;
  const xTol = options.xTol ?? 1e-10;
  const maxIter = options.maxIter ?? 100;
  const alpha0 = options.alpha0 ?? 1;
  const c1 = options.c1 ?? 1e-4;
  const rho = options.rho ?? 0.5;
  let x = [...x0];
  const path: Vec[] = [x];
  let fx = objective(x);
  let iterations = 0;
  let xPrev: Vec | null = null;

  while (iterations < maxIter) {
    const g = gradient(x);
    if (checkConvergence(g, gradTol, x, xPrev, xTol)) {
      return { x, value: fx, iterations, converged: true, path };
    }
    let direction: Vec;
    try {
      direction = vecScale(solveLinear(hessian(x), g), -1);
    } catch {
      return { x, value: fx, iterations, converged: false, path };
    }
    const alpha = armijoStep(objective, x, direction, g, alpha0, c1, rho);
    xPrev = x;
    x = vecAdd(x, vecScale(direction, alpha));
    path.push([...x]);
    fx = objective(x);
    iterations += 1;
  }
  return {
    x,
    value: fx,
    iterations,
    converged: norm2(gradient(x)) <= gradTol,
    path,
  };
}

export function conjugateGradient(
  objective: Objective,
  gradient: Gradient,
  x0: Vec,
  options: {
    gradTol?: number;
    maxIter?: number;
    alpha0?: number;
    c1?: number;
    rho?: number;
    restartEvery?: number;
  } = {},
): OptimizeResult {
  const gradTol = options.gradTol ?? 1e-8;
  const maxIter = options.maxIter ?? 1000;
  const alpha0 = options.alpha0 ?? 1;
  const c1 = options.c1 ?? 1e-4;
  const rho = options.rho ?? 0.5;
  const restartEvery = options.restartEvery ?? x0.length;
  let x = [...x0];
  const path: Vec[] = [x];
  let g = gradient(x);
  let direction = vecScale(g, -1);
  let fx = objective(x);
  let iterations = 0;

  while (iterations < maxIter) {
    if (norm2(g) <= gradTol) {
      return { x, value: fx, iterations, converged: true, path };
    }
    const alpha = armijoStep(objective, x, direction, g, alpha0, c1, rho);
    x = vecAdd(x, vecScale(direction, alpha));
    path.push([...x]);
    const gNew = gradient(x);
    fx = objective(x);
    iterations += 1;
    if (norm2(gNew) <= gradTol) {
      return { x, value: fx, iterations, converged: true, path };
    }
    if (iterations % restartEvery === 0) {
      direction = vecScale(gNew, -1);
    } else {
      const beta = dot(gNew, gNew) / Math.max(dot(g, g), 1e-30);
      direction = vecAdd(vecScale(gNew, -1), vecScale(direction, beta));
    }
    g = gNew;
  }
  return {
    x,
    value: fx,
    iterations,
    converged: norm2(g) <= gradTol,
    path,
  };
}

export function bfgs(
  objective: Objective,
  gradient: Gradient,
  x0: Vec,
  options: {
    gradTol?: number;
    maxIter?: number;
    alpha0?: number;
    c1?: number;
    rho?: number;
  } = {},
): OptimizeResult {
  const gradTol = options.gradTol ?? 1e-8;
  const maxIter = options.maxIter ?? 200;
  const alpha0 = options.alpha0 ?? 1;
  const c1 = options.c1 ?? 1e-4;
  const rho = options.rho ?? 0.5;
  let x = [...x0];
  const path: Vec[] = [x];
  const n = x.length;
  let g = gradient(x);
  let b: Mat = identity(n);
  let fx = objective(x);
  let iterations = 0;

  while (iterations < maxIter) {
    if (norm2(g) <= gradTol) {
      return { x, value: fx, iterations, converged: true, path };
    }
    let direction = matVec(b, vecScale(g, -1));
    const alpha = armijoStep(objective, x, direction, g, alpha0, c1, rho);
    const s = vecScale(direction, alpha);
    const xNew = vecAdd(x, s);
    const gNew = gradient(xNew);
    const y = vecSub(gNew, g);
    const sy = dot(s, y);
    if (sy > 1e-12) {
      const rhoK = 1 / sy;
      const iMat = identity(n);
      const left = iMat.map((row, i) =>
        row.map((_, j) => iMat[i]![j]! - rhoK * y[i]! * s[j]!),
      );
      const right = iMat.map((row, i) =>
        row.map((_, j) => iMat[i]![j]! - rhoK * s[i]! * y[j]!),
      );
      const outerS = s.map((_, i) => s.map((_, j) => rhoK * s[i]! * s[j]!));
      b = matAdd(matMul(matMul(left, b), right), outerS);
    }
    x = xNew;
    path.push([...x]);
    g = gNew;
    fx = objective(x);
    iterations += 1;
  }
  return {
    x,
    value: fx,
    iterations,
    converged: norm2(g) <= gradTol,
    path,
  };
}

export const quasiNewtonBfgs = bfgs;

function lbfgsDirection(
  g: Vec,
  sHist: Vec[],
  yHist: Vec[],
  rhoHist: number[],
): Vec {
  let q = vecScale(g, -1);
  const alphas: number[] = [];
  for (let i = sHist.length - 1; i >= 0; i -= 1) {
    const s = sHist[i]!;
    const rho = rhoHist[i]!;
    const alpha = rho * dot(s, q);
    alphas.push(alpha);
    q = vecSub(q, vecScale(s, alpha));
  }
  let r: Vec;
  if (sHist.length > 0) {
    const sLast = sHist[sHist.length - 1]!;
    const yLast = yHist[yHist.length - 1]!;
    const gamma = dot(sLast, yLast) / Math.max(dot(yLast, yLast), 1e-30);
    r = vecScale(q, gamma);
  } else {
    r = q;
  }
  for (let i = 0; i < sHist.length; i += 1) {
    const s = sHist[i]!;
    const y = yHist[i]!;
    const rho = rhoHist[i]!;
    const alpha = alphas[sHist.length - 1 - i]!;
    const beta = rho * dot(y, r);
    r = vecAdd(r, vecScale(s, alpha - beta));
  }
  return r;
}

export function lbfgs(
  objective: Objective,
  gradient: Gradient,
  x0: Vec,
  options: {
    memory?: number;
    gradTol?: number;
    maxIter?: number;
    alpha0?: number;
    c1?: number;
    rho?: number;
  } = {},
): OptimizeResult {
  const memory = options.memory ?? 10;
  const gradTol = options.gradTol ?? 1e-8;
  const maxIter = options.maxIter ?? 200;
  const alpha0 = options.alpha0 ?? 1;
  const c1 = options.c1 ?? 1e-4;
  const rho = options.rho ?? 0.5;
  let x = [...x0];
  const path: Vec[] = [x];
  let g = gradient(x);
  let fx = objective(x);
  const sHist: Vec[] = [];
  const yHist: Vec[] = [];
  const rhoHist: number[] = [];
  let iterations = 0;

  while (iterations < maxIter) {
    if (norm2(g) <= gradTol) {
      return { x, value: fx, iterations, converged: true, path };
    }
    const direction = lbfgsDirection(g, sHist, yHist, rhoHist);
    const alpha = armijoStep(objective, x, direction, g, alpha0, c1, rho);
    const s = vecScale(direction, alpha);
    const xNew = vecAdd(x, s);
    const gNew = gradient(xNew);
    const y = vecSub(gNew, g);
    const sy = dot(s, y);
    if (sy > 1e-12) {
      sHist.push(s);
      yHist.push(y);
      rhoHist.push(1 / sy);
      if (sHist.length > memory) {
        sHist.shift();
        yHist.shift();
        rhoHist.shift();
      }
    }
    x = xNew;
    path.push([...x]);
    g = gNew;
    fx = objective(x);
    iterations += 1;
  }
  return {
    x,
    value: fx,
    iterations,
    converged: norm2(g) <= gradTol,
    path,
  };
}

export function gaussNewton(
  residual: Residual,
  jacobian: Jacobian,
  x0: Vec,
  options: {
    gradTol?: number;
    maxIter?: number;
    damping?: number;
  } = {},
): OptimizeResult {
  const gradTol = options.gradTol ?? 1e-8;
  const maxIter = options.maxIter ?? 100;
  const damping = options.damping ?? 1e-8;
  let x = [...x0];
  const path: Vec[] = [x];
  let iterations = 0;

  while (iterations < maxIter) {
    const r = residual(x);
    const j = jacobian(x);
    const jt = matTranspose(j);
    const jtj = matAdd(matMul(jt, j), matScale(identity(x.length), damping));
    const g = matVec(jt, r);
    if (norm2(g) <= gradTol) {
      return {
        x,
        value: 0.5 * dot(r, r),
        iterations,
        converged: true,
        path,
      };
    }
    try {
      const step = vecScale(solveLinear(jtj, g), -1);
      x = vecAdd(x, step);
      path.push([...x]);
    } catch {
      return {
        x,
        value: 0.5 * dot(r, r),
        iterations,
        converged: false,
        path,
      };
    }
    iterations += 1;
  }
  const r = residual(x);
  const g = matVec(matTranspose(jacobian(x)), r);
  return {
    x,
    value: 0.5 * dot(r, r),
    iterations,
    converged: norm2(g) <= gradTol,
    path,
  };
}

export function levenbergMarquardt(
  residual: Residual,
  jacobian: Jacobian,
  x0: Vec,
  options: {
    gradTol?: number;
    maxIter?: number;
    lambda0?: number;
    lambdaUp?: number;
    lambdaDown?: number;
  } = {},
): OptimizeResult {
  const gradTol = options.gradTol ?? 1e-8;
  const maxIter = options.maxIter ?? 100;
  let lam = options.lambda0 ?? 1e-2;
  const lambdaUp = options.lambdaUp ?? 10;
  const lambdaDown = options.lambdaDown ?? 0.1;
  let x = [...x0];
  const path: Vec[] = [x];
  let fx = 0.5 * dot(residual(x), residual(x));
  let iterations = 0;

  while (iterations < maxIter) {
    const r = residual(x);
    const j = jacobian(x);
    const jt = matTranspose(j);
    const g = matVec(jt, r);
    if (norm2(g) <= gradTol) {
      return { x, value: fx, iterations, converged: true, path };
    }
    const jtj = matMul(jt, j);
    let accepted = false;
    for (let attempt = 0; attempt < 12; attempt += 1) {
      try {
        const a = matAdd(jtj, matScale(identity(x.length), lam));
        const step = vecScale(solveLinear(a, g), -1);
        const xTrial = vecAdd(x, step);
        const fTrial = 0.5 * dot(residual(xTrial), residual(xTrial));
        if (fTrial < fx) {
          x = xTrial;
          path.push([...x]);
          fx = fTrial;
          lam = Math.max(lam * lambdaDown, 1e-16);
          accepted = true;
          break;
        }
        lam *= lambdaUp;
      } catch {
        lam *= lambdaUp;
      }
    }
    if (!accepted) {
      return { x, value: fx, iterations, converged: false, path };
    }
    iterations += 1;
  }
  const g = matVec(matTranspose(jacobian(x)), residual(x));
  return {
    x,
    value: fx,
    iterations,
    converged: norm2(g) <= gradTol,
    path,
  };
}
