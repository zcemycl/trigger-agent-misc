import { describe, expect, it } from "vitest";
import { pathHighlightRangeForStep } from "./pathHighlight";

describe("pathHighlightRangeForStep", () => {
  it("returns inclusive path indices for a table step", () => {
    const range = pathHighlightRangeForStep(
      [
        {
          iteration: 0,
          pathStartIndex: 0,
          pathEndIndex: 2,
        },
        {
          iteration: 1,
          pathStartIndex: 2,
          pathEndIndex: 3,
        },
      ],
      0,
    );
    expect(range).toEqual([0, 2]);
  });

  it("returns null when step is unknown or indices missing", () => {
    expect(pathHighlightRangeForStep([], 0)).toBeNull();
    expect(pathHighlightRangeForStep([{ iteration: 0 }], 0)).toBeNull();
    expect(pathHighlightRangeForStep([], null)).toBeNull();
  });
});
