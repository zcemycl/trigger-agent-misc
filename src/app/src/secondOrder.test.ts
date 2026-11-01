import { describe, expect, it } from "vitest";
import {
  bfgs,
  conjugateGradient,
  dampedNewtonMethod,
  gaussNewton,
  lbfgs,
  levenbergMarquardt,
  newtonsMethod,
  quasiNewtonBfgs,
} from "./secondOrder";
import {
  rosenbrockJacobian,
  rosenbrockResidual,
  shiftedQuadratic,
  shiftedQuadraticGrad,
  shiftedQuadraticHess,
} from "./secondOrderProblems";

describe("secondOrder", () => {
  it("Newton solves quadratic in one step", () => {
    const result = newtonsMethod(
      shiftedQuadratic,
      shiftedQuadraticGrad,
      shiftedQuadraticHess,
      [0, 0],
    );
    expect(result.converged).toBe(true);
    expect(result.iterations).toBe(1);
    expect(result.x[0]).toBeCloseTo(1, 8);
    expect(result.x[1]).toBeCloseTo(2, 8);
  });

  it("damped Newton and CG reach the quadratic minimizer", () => {
    const damped = dampedNewtonMethod(
      shiftedQuadratic,
      shiftedQuadraticGrad,
      shiftedQuadraticHess,
      [4, -3],
    );
    expect(damped.converged).toBe(true);
    const cg = conjugateGradient(
      shiftedQuadratic,
      shiftedQuadraticGrad,
      [0, 0],
    );
    expect(cg.converged).toBe(true);
    expect(cg.x[0]).toBeCloseTo(1, 3);
  });

  it("BFGS, quasi-Newton alias, and L-BFGS converge", () => {
    for (const solver of [bfgs, quasiNewtonBfgs, lbfgs]) {
      const result = solver(shiftedQuadratic, shiftedQuadraticGrad, [0, 0]);
      expect(result.converged).toBe(true);
      expect(result.x[1]).toBeCloseTo(2, 2);
    }
  });

  it("Gauss–Newton and LM reduce Rosenbrock residual", () => {
    const gn = gaussNewton(rosenbrockResidual, rosenbrockJacobian, [0, 0]);
    expect(gn.converged).toBe(true);
    expect(gn.value).toBeLessThan(1e-5);
    const lm = levenbergMarquardt(
      rosenbrockResidual,
      rosenbrockJacobian,
      [0, 0],
    );
    expect(lm.value).toBeLessThan(0.5);
  });
});
