import { describe, expect, it } from "vitest";
import {
  FINITE_SUM_OPTIMUM,
  finiteSumGrad,
  finiteSumGradAtIndex,
  finiteSumObjective,
  SAMPLE_CENTERS,
} from "./finiteSumQuadratic";
import {
  formatVector,
  gradientDescent,
  minibatchSgd,
  momentum,
} from "./gradientMethod";

describe("gradientDescent", () => {
  it("finds the mean of sample centers", () => {
    const result = gradientDescent([3], finiteSumObjective, finiteSumGrad, {
      learningRate: 0.2,
      maxIter: 500,
      tol: 1e-8,
    });
    expect(Math.abs(result.x[0] - FINITE_SUM_OPTIMUM)).toBeLessThan(1e-3);
    expect(result.steps.length).toBe(result.iterations);
    const step0 = result.steps[0];
    expect(step0?.detail).toContain("∇f(x_k)");
    expect(step0?.detail).toContain("Δx_k");
    expect(step0?.detail).toContain("x_{k+1}");
    expect(formatVector(step0!.x)).toMatch(/^\[/);
    expect(formatVector(step0!.xNext)).toMatch(/^\[/);
    expect(formatVector(step0!.delta)).toMatch(/^\[/);
  });
});

describe("momentum", () => {
  it("records velocity and position vectors in step details", () => {
    const result = momentum([3], finiteSumObjective, finiteSumGrad, {
      learningRate: 0.05,
      beta: 0.9,
      maxIter: 5,
      tol: 1e-8,
    });
    const step0 = result.steps[0];
    expect(step0?.detail).toContain("v_{k+1}");
    expect(step0?.detail).toContain("∇f(x_k) = [");
  });
});

describe("formatVector", () => {
  it("uses bracket notation", () => {
    expect(formatVector([1.234567])).toBe("[1.2346]");
    expect(formatVector([1, 2])).toBe("[1.0000, 2.0000]");
  });
});

describe("minibatchSgd", () => {
  it("runs with fixed seed", () => {
    const result = minibatchSgd(
      [3],
      finiteSumObjective,
      finiteSumGradAtIndex,
      SAMPLE_CENTERS.length,
      { batchSize: 2, learningRate: 0.1, maxIter: 400, seed: 1 },
    );
    expect(result.iterations).toBeGreaterThan(0);
    expect(result.value).toBeLessThan(finiteSumObjective([3]));
  });
});
