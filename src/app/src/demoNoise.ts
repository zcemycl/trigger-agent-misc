/** Deterministic pseudo-random perturbations for interactive optimizer demos. */

export type DemoNoiseSettings = {
  enabled: boolean;
  seed: number;
};

export function noiseOffset(
  coords: readonly number[],
  salt: number,
  seed: number,
  amplitude: number,
): number {
  let h = seed ^ Math.imul(salt, 0x9e3779b9);
  for (const c of coords) {
    h = Math.imul(h ^ Math.floor(c * 1000), 0x85ebca6b);
  }
  const u = ((h >>> 0) % 10_000) / 10_000;
  return amplitude * (2 * u - 1);
}

export function numericDerivative(
  f: (x: number) => number,
  x: number,
  h = 1e-5,
): number {
  return (f(x + h) - f(x - h)) / (2 * h);
}

export function wrapObjectiveVec(
  objective: (x: number[]) => number,
  settings: DemoNoiseSettings,
  amplitude = 0.12,
): (x: number[]) => number {
  if (!settings.enabled) return objective;
  return (x) => objective(x) + noiseOffset(x, 0, settings.seed, amplitude);
}

export function wrapGradientVec(
  gradient: (x: number[]) => number[],
  settings: DemoNoiseSettings,
  amplitude = 0.06,
): (x: number[]) => number[] {
  if (!settings.enabled) return gradient;
  return (x) => {
    const g = gradient(x);
    return g.map(
      (gi, i) => gi + noiseOffset(x, i + 1, settings.seed, amplitude),
    );
  };
}

export function wrapGradAtIndex(
  gradAtIndex: (index: number, x: number[]) => number[],
  settings: DemoNoiseSettings,
  amplitude = 0.06,
): (index: number, x: number[]) => number[] {
  if (!settings.enabled) return gradAtIndex;
  return (index, x) => {
    const g = gradAtIndex(index, x);
    return g.map(
      (gi, i) =>
        gi + noiseOffset([index, ...x], i + 1, settings.seed, amplitude),
    );
  };
}

export function wrapPhi(
  phi: (alpha: number) => number,
  settings: DemoNoiseSettings,
  amplitude = 0.08,
): (alpha: number) => number {
  if (!settings.enabled) return phi;
  return (alpha) =>
    phi(alpha) + noiseOffset([alpha], 0, settings.seed, amplitude);
}

export function wrapResidualVec(
  residual: (x: number[]) => number[],
  settings: DemoNoiseSettings,
  amplitude = 0.04,
): (x: number[]) => number[] {
  if (!settings.enabled) return residual;
  return (x) => {
    const r = residual(x);
    return r.map(
      (ri, i) => ri + noiseOffset(x, i + 1, settings.seed, amplitude),
    );
  };
}
