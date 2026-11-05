import { describe, expect, it } from "vitest";
import { numericDerivative } from "./demoNoise";
import {
  boothGradient,
  boothObjective,
  exponentialFitJacobian,
  exponentialFitResidual,
  freudensteinRothJacobian,
  freudensteinRothResidual,
  makeShiftedQuadratic,
  rosenbrockObjective,
  rosenbrockObjectiveGradient,
  shiftedQuadratic3D,
  shiftedQuadraticGrad,
} from "./secondOrderProblems";

describe("secondOrderProblems", () => {
  it("shifted quadratic gradient matches dimension", () => {
    expect(shiftedQuadraticGrad([0, 0])).toEqual([-1, -2]);
    const g3 = shiftedQuadratic3D.gradient([0, 0, 0]);
    expect(g3).toEqual([-1, -2, -3]);
  });

  it("makeShiftedQuadratic builds identity Hessian", () => {
    const p = makeShiftedQuadratic([1, 2, 3]);
    expect(p.hessian([0, 0, 0])).toEqual([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);
  });

  it("booth objective is zero at the known minimizer", () => {
    expect(boothObjective([1, 3])).toBeCloseTo(0, 10);
    expect(boothGradient([1, 3])).toEqual([0, 0]);
  });

  it("rosenbrock objective gradient matches partial derivatives", () => {
    const x = [0.3, 0.5];
    const g = rosenbrockObjectiveGradient(x);
    expect(g[0]).toBeCloseTo(
      numericDerivative((v) => rosenbrockObjective([v, x[1]!]), x[0]!),
      4,
    );
    expect(g[1]).toBeCloseTo(
      numericDerivative((v) => rosenbrockObjective([x[0]!, v]), x[1]!),
      4,
    );
  });

  it("least-squares Jacobians have correct shape", () => {
    const x = [1, 0.4];
    expect(exponentialFitResidual(x)).toHaveLength(5);
    expect(exponentialFitJacobian(x)).toHaveLength(5);
    expect(freudensteinRothResidual(x)).toHaveLength(2);
    expect(freudensteinRothJacobian(x)).toHaveLength(2);
  });
});
