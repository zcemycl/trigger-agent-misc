/** Shared 1D finite-sum quadratic used in gradient method demos and evals. */

export const SAMPLE_CENTERS = [-2, 0.5, 1.5] as const;

export function finiteSumObjective(x: number[]): number {
  const v = x[0];
  return (
    SAMPLE_CENTERS.reduce((acc, c) => acc + 0.5 * (v - c) ** 2, 0) /
    SAMPLE_CENTERS.length
  );
}

export function finiteSumGrad(x: number[]): number[] {
  const v = x[0];
  const mean = SAMPLE_CENTERS.reduce((acc, c) => acc + (v - c), 0);
  return [mean / SAMPLE_CENTERS.length];
}

export function finiteSumGradAtIndex(index: number, x: number[]): number[] {
  return [x[0] - SAMPLE_CENTERS[index]];
}

export const FINITE_SUM_OPTIMUM =
  SAMPLE_CENTERS.reduce((a, b) => a + b, 0) / SAMPLE_CENTERS.length;
