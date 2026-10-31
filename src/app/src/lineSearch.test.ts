import { describe, expect, it } from "vitest";
import {
  armijoSufficientDecrease,
  backtrackingArmijo,
  exactLineSearch,
  goldsteinConditions,
  lineSearchGoldstein,
  lineSearchStrongWolfe,
  lineSearchWolfe,
  strongWolfeCurvature,
  wolfeCurvature,
} from "./lineSearch";
import { quadraticPhi, quadraticPhiPrime, QUADRATIC_PHI0 } from "./quadratic";

describe("lineSearch", () => {
  it("finds unit step with backtracking on quadratic", () => {
    const result = backtrackingArmijo(quadraticPhi, -2, { alpha0: 1 });
    expect(result.alpha).toBeCloseTo(1, 9);
    expect(result.value).toBeLessThan(1e-12);
    expect(result.iterations).toBe(0);
  });

  it("exact search finds quadratic minimizer", () => {
    const result = exactLineSearch(quadraticPhi, { a: 0, b: 2 });
    expect(result.alpha).toBeCloseTo(1, 5);
    expect(result.value).toBeLessThan(1e-10);
  });

  it("evaluates Armijo, Wolfe, and Goldstein conditions", () => {
    expect(
      armijoSufficientDecrease(QUADRATIC_PHI0, quadraticPhi(0.5), 0.5, -2),
    ).toBe(true);
    expect(wolfeCurvature(0, -2)).toBe(true);
    expect(strongWolfeCurvature(0, -2)).toBe(true);
    expect(goldsteinConditions(1, 1 + 0.1 * 0.25 * -2, 0.25, -2, 0.1)).toBe(
      true,
    );
  });

  it("runs Wolfe-family line searches on quadratic", () => {
    const wolfe = lineSearchWolfe(quadraticPhi, quadraticPhiPrime, -2);
    expect(wolfe.alpha).toBeCloseTo(1, 2);
    const strong = lineSearchStrongWolfe(quadraticPhi, quadraticPhiPrime, -2);
    expect(strong.value).toBeLessThan(0.5);
    const gold = lineSearchGoldstein(quadraticPhi, -2);
    expect(
      goldsteinConditions(QUADRATIC_PHI0, gold.value, gold.alpha, -2, 0.1),
    ).toBe(true);
  });
});
