import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DemoNoiseControls, DemoNoiseProvider } from "./DemoNoise";

describe("DemoNoiseControls", () => {
  it("toggles noise mode and shows reseed when enabled", () => {
    render(
      <DemoNoiseProvider>
        <DemoNoiseControls />
      </DemoNoiseProvider>,
    );
    const toggle = screen.getByRole("checkbox", { name: /noise mode/i });
    expect(toggle).not.toBeChecked();
    expect(screen.queryByRole("button", { name: /reseed noise/i })).toBeNull();
    fireEvent.click(toggle);
    expect(toggle).toBeChecked();
    expect(
      screen.getByRole("button", { name: /reseed noise/i }),
    ).toBeInTheDocument();
  });
});
