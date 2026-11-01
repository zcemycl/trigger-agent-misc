import { describe, expect, it } from "vitest";
import {
  FINITE_SUM_OPTIMUM,
  finiteSumGrad,
  finiteSumGradAtIndex,
  finiteSumObjective,
  SAMPLE_CENTERS,
} from "./finiteSumQuadratic";
import { gradientDescent, minibatchSgd } from "./gradientMethod";

describe("gradientDescent", () => {
  it("finds the mean of sample centers", () => {
    const result = gradientDescent([3], finiteSumObjective, finiteSumGrad, {
      learningRate: 0.2,
      maxIter: 500,
      tol: 1e-8,
    });
    expect(Math.abs(result.x[0] - FINITE_SUM_OPTIMUM)).toBeLessThan(1e-3);
    expect(result.steps.length).toBe(result.iterations);
    expect(result.steps[0]?.detail).toContain("∇f");
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
