import { describe, expect, it } from "vitest";
import {
  makeShiftedQuadratic,
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
});
