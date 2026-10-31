/** Example ray: φ(α) = (1 − α)² with φ′(0) = −2. */

export function quadraticPhi(alpha: number): number {
  return (1 - alpha) ** 2;
}

export function quadraticPhiPrime(alpha: number): number {
  return -2 * (1 - alpha);
}

export const QUADRATIC_PHI0 = quadraticPhi(0);
export const QUADRATIC_PHI_PRIME0 = -2;
