/** One-dimensional line search (mirrors trigger_agent_misc.optimization.line_search). */

export type Phi = (alpha: number) => number;

export interface LineSearchResult {
  alpha: number;
  value: number;
  iterations: number;
}

export function armijoSufficientDecrease(
  phi0: number,
  phiAlpha: number,
  alpha: number,
  phiPrimeAtZero: number,
  c1 = 1e-4,
): boolean {
  return phiAlpha <= phi0 + c1 * alpha * phiPrimeAtZero;
}

export function wolfeCurvature(
  phiPrimeAlpha: number,
  phiPrimeAtZero: number,
  c2 = 0.9,
): boolean {
  return phiPrimeAlpha >= c2 * phiPrimeAtZero;
}

export function strongWolfeCurvature(
  phiPrimeAlpha: number,
  phiPrimeAtZero: number,
  c2 = 0.9,
): boolean {
  return Math.abs(phiPrimeAlpha) <= c2 * Math.abs(phiPrimeAtZero);
}

export function goldsteinConditions(
  phi0: number,
  phiAlpha: number,
  alpha: number,
  phiPrimeAtZero: number,
  c = 0.1,
): boolean {
  if (!(c > 0 && c < 0.5)) {
    throw new Error("c must be in (0, 0.5)");
  }
  const lower = phi0 + (1 - c) * alpha * phiPrimeAtZero;
  const upper = phi0 + c * alpha * phiPrimeAtZero;
  return lower <= phiAlpha && phiAlpha <= upper;
}

export function exactLineSearch(
  phi: Phi,
  options: { a?: number; b?: number; tol?: number; maxIter?: number } = {},
): LineSearchResult {
  const a = options.a ?? 0;
  const b = options.b ?? 1;
  const tol = options.tol ?? 1e-10;
  const maxIter = options.maxIter ?? 200;
  if (a >= b) {
    throw new Error("require a < b");
  }
  const gr = (Math.sqrt(5) - 1) / 2;
  let left = a;
  let right = b;
  let iterations = 0;
  let c = right - gr * (right - left);
  let d = left + gr * (right - left);
  let fc = phi(c);
  let fd = phi(d);

  while (right - left > tol && iterations < maxIter) {
    iterations += 1;
    if (fc < fd) {
      right = d;
      d = c;
      fd = fc;
      c = right - gr * (right - left);
      fc = phi(c);
    } else {
      left = c;
      c = d;
      fc = fd;
      d = left + gr * (right - left);
      fd = phi(d);
    }
  }
  const alpha = 0.5 * (left + right);
  return { alpha, value: phi(alpha), iterations };
}

export function backtrackingArmijo(
  phi: Phi,
  phiPrimeAtZero: number,
  options: {
    alpha0?: number;
    c1?: number;
    rho?: number;
    alphaMin?: number;
    maxIter?: number;
  } = {},
): LineSearchResult {
  const alpha0 = options.alpha0 ?? 1;
  const c1 = options.c1 ?? 1e-4;
  const rho = options.rho ?? 0.5;
  const alphaMin = options.alphaMin ?? 1e-14;
  const maxIter = options.maxIter ?? 100;
  if (phiPrimeAtZero >= 0) {
    throw new Error(
      "phi_prime_at_zero must be negative for a descent direction",
    );
  }
  const phi0 = phi(0);
  let alpha = alpha0;
  let iterations = 0;

  while (
    !armijoSufficientDecrease(phi0, phi(alpha), alpha, phiPrimeAtZero, c1)
  ) {
    iterations += 1;
    if (iterations >= maxIter) {
      break;
    }
    const nextAlpha = alpha * rho;
    if (nextAlpha < alphaMin) {
      alpha = nextAlpha;
      break;
    }
    alpha = nextAlpha;
  }

  return { alpha, value: phi(alpha), iterations };
}

/** Trial alphas visited when backtracking from ``alpha0`` (for visualization). */
export function backtrackingTrialAlphas(
  phi: Phi,
  phiPrimeAtZero: number,
  alpha0: number,
  c1 = 1e-4,
  rho = 0.5,
  maxIter = 100,
): number[] {
  const phi0 = phi(0);
  const trials: number[] = [alpha0];
  let alpha = alpha0;
  let iterations = 0;
  while (
    !armijoSufficientDecrease(phi0, phi(alpha), alpha, phiPrimeAtZero, c1)
  ) {
    iterations += 1;
    if (iterations >= maxIter) {
      break;
    }
    alpha *= rho;
    trials.push(alpha);
  }
  return trials;
}

export function backtrackingLineSearch(
  phi: Phi,
  phiPrimeAtZero: number,
  options?: Parameters<typeof backtrackingArmijo>[2],
): LineSearchResult {
  return backtrackingArmijo(phi, phiPrimeAtZero, options);
}

function zoomWolfe(
  phi: Phi,
  phiPrime: Phi,
  phi0: number,
  phiPrimeAtZero: number,
  alphaLo: number,
  alphaHi: number,
  c1: number,
  c2: number,
  strong: boolean,
  maxIter: number,
): number {
  let lo = alphaLo;
  let hi = alphaHi;
  for (let i = 0; i < maxIter; i += 1) {
    const alpha = 0.5 * (lo + hi);
    const phiA = phi(alpha);
    if (phiA > phi0 + c1 * alpha * phiPrimeAtZero || phi(lo) <= phiA) {
      hi = alpha;
      continue;
    }
    const phiP = phiPrime(alpha);
    if (strong) {
      if (strongWolfeCurvature(phiP, phiPrimeAtZero, c2)) {
        return alpha;
      }
    } else if (wolfeCurvature(phiP, phiPrimeAtZero, c2)) {
      return alpha;
    }
    if (phiP * (hi - lo) >= 0) {
      hi = lo;
    }
    lo = alpha;
  }
  return 0.5 * (lo + hi);
}

function lineSearchWolfeImpl(
  phi: Phi,
  phiPrime: Phi,
  phiPrimeAtZero: number,
  alpha0: number,
  c1: number,
  c2: number,
  alphaMax: number,
  maxIter: number,
  strong: boolean,
): LineSearchResult {
  const phi0 = phi(0);
  let alphaPrev = 0;
  let phiPrev = phi0;
  let alpha = alpha0;
  let iterations = 0;

  while (true) {
    iterations += 1;
    if (iterations > maxIter) {
      break;
    }
    const phiA = phi(alpha);
    if (
      phiA > phi0 + c1 * alpha * phiPrimeAtZero ||
      (phiA >= phiPrev && alpha > alphaPrev)
    ) {
      const alphaStar = zoomWolfe(
        phi,
        phiPrime,
        phi0,
        phiPrimeAtZero,
        alphaPrev,
        alpha,
        c1,
        c2,
        strong,
        maxIter,
      );
      return {
        alpha: alphaStar,
        value: phi(alphaStar),
        iterations,
      };
    }
    const phiP = phiPrime(alpha);
    if (strong) {
      if (strongWolfeCurvature(phiP, phiPrimeAtZero, c2)) {
        return { alpha, value: phiA, iterations };
      }
    } else if (wolfeCurvature(phiP, phiPrimeAtZero, c2)) {
      return { alpha, value: phiA, iterations };
    }
    if (alpha >= alphaMax) {
      return { alpha, value: phiA, iterations };
    }
    alphaPrev = alpha;
    phiPrev = phiA;
    alpha = Math.min(2 * alpha, alphaMax);
  }
  return { alpha, value: phi(alpha), iterations };
}

export function lineSearchWolfe(
  phi: Phi,
  phiPrime: Phi,
  phiPrimeAtZero: number,
  options: {
    alpha0?: number;
    c1?: number;
    c2?: number;
    alphaMax?: number;
    maxIter?: number;
  } = {},
): LineSearchResult {
  return lineSearchWolfeImpl(
    phi,
    phiPrime,
    phiPrimeAtZero,
    options.alpha0 ?? 1,
    options.c1 ?? 1e-4,
    options.c2 ?? 0.9,
    options.alphaMax ?? 10,
    options.maxIter ?? 100,
    false,
  );
}

export function lineSearchStrongWolfe(
  phi: Phi,
  phiPrime: Phi,
  phiPrimeAtZero: number,
  options: {
    alpha0?: number;
    c1?: number;
    c2?: number;
    alphaMax?: number;
    maxIter?: number;
  } = {},
): LineSearchResult {
  return lineSearchWolfeImpl(
    phi,
    phiPrime,
    phiPrimeAtZero,
    options.alpha0 ?? 1,
    options.c1 ?? 1e-4,
    options.c2 ?? 0.9,
    options.alphaMax ?? 10,
    options.maxIter ?? 100,
    true,
  );
}

export function lineSearchGoldstein(
  phi: Phi,
  phiPrimeAtZero: number,
  options: {
    alpha0?: number;
    c?: number;
    rho?: number;
    alphaMax?: number;
    maxIter?: number;
  } = {},
): LineSearchResult {
  const alpha0 = options.alpha0 ?? 1;
  const c = options.c ?? 0.1;
  const rho = options.rho ?? 0.5;
  const alphaMax = options.alphaMax ?? 10;
  const maxIter = options.maxIter ?? 100;
  const phi0 = phi(0);
  let alpha = alpha0;
  let iterations = 0;

  while (iterations < maxIter) {
    const phiA = phi(alpha);
    if (goldsteinConditions(phi0, phiA, alpha, phiPrimeAtZero, c)) {
      return { alpha, value: phiA, iterations };
    }
    iterations += 1;
    if (phiA > phi0 + c * alpha * phiPrimeAtZero) {
      alpha *= rho;
    } else if (phiA < phi0 + (1 - c) * alpha * phiPrimeAtZero) {
      alpha = Math.min(alpha / rho, alphaMax);
    } else {
      break;
    }
  }
  return { alpha, value: phi(alpha), iterations };
}
