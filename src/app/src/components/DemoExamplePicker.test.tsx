import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DemoExamplePicker } from "./DemoExamplePicker";

const EXAMPLES = [
  { id: "a" as const, label: "Alpha" },
  { id: "b" as const, label: "Beta" },
] as const;

describe("DemoExamplePicker", () => {
  it("selects an example and marks it pressed", () => {
    const onChange = vi.fn();
    render(
      <DemoExamplePicker
        label="Demo examples"
        examples={EXAMPLES}
        value="a"
        onChange={onChange}
      />,
    );
    expect(screen.getByRole("button", { name: "Alpha" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    fireEvent.click(screen.getByRole("button", { name: "Beta" }));
    expect(onChange).toHaveBeenCalledWith("b");
  });
});
