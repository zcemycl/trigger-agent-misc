import { describe, expect, it } from "vitest";
import {
  noiseOffset,
  wrapGradientVec,
  wrapObjectiveVec,
  wrapPhi,
} from "./demoNoise";

describe("demoNoise", () => {
  it("is deterministic for fixed seed and coords", () => {
    const a = noiseOffset([1, 2], 0, 42, 0.1);
    const b = noiseOffset([1, 2], 0, 42, 0.1);
    expect(a).toBe(b);
    expect(Math.abs(a)).toBeLessThanOrEqual(0.1);
  });

  it("wraps leave functions unchanged when disabled", () => {
    const off = { enabled: false, seed: 0 };
    const f = (x: number[]) => x[0]! + x[1]!;
    expect(wrapObjectiveVec(f, off)([1, 2])).toBe(3);
    const g = (x: number[]) => [x[0]!, x[1]!];
    expect(wrapGradientVec(g, off)([1, 2])).toEqual([1, 2]);
    expect(wrapPhi((a) => a * a, off)(2)).toBe(4);
  });

  it("perturbs values when enabled", () => {
    const on = { enabled: true, seed: 7 };
    const f = (x: number[]) => x[0]! ** 2;
    expect(wrapObjectiveVec(f, on)([1])).not.toBe(1);
  });
});
