import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { OptimizationStepTable } from "./OptimizationStepTable";

const SAMPLE_STEPS = [
  {
    iteration: 0,
    x: [0],
    delta: [1],
    xNext: [1],
    objective: 1,
    detail: "step 0",
  },
  {
    iteration: 1,
    x: [1],
    delta: [0.5],
    xNext: [1.5],
    objective: 0.5,
    detail: "step 1",
  },
];

describe("OptimizationStepTable", () => {
  it("notifies hover changes for step highlighting", () => {
    const onHighlight = vi.fn();
    render(
      <OptimizationStepTable
        steps={SAMPLE_STEPS}
        onHighlightedStepChange={onHighlight}
      />,
    );
    const rows = screen.getAllByRole("row");
    fireEvent.mouseEnter(rows[1]!);
    expect(onHighlight).toHaveBeenLastCalledWith(0);
    fireEvent.mouseLeave(rows[1]!);
    expect(onHighlight).toHaveBeenLastCalledWith(null);
  });
});
