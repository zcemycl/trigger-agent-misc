import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  ConceptPageOverview,
  type ConceptOverview,
} from "./ConceptPageOverview";

const SAMPLE: ConceptOverview = {
  purposeTex: "\\min f(\\mathbf{x})",
  explanationTex:
    "\\mathbf{x}_{k+1} = \\mathbf{x}_k - \\eta\\nabla f(\\mathbf{x}_k)",
  references: [
    {
      href: "https://example.com/paper",
      label: "Example reference",
    },
  ],
};

describe("ConceptPageOverview", () => {
  it("renders LaTeX blocks and linked references", () => {
    render(
      <ConceptPageOverview title="Gradient descent (GD)" overview={SAMPLE} />,
    );
    expect(
      screen.getByLabelText("Gradient descent (GD) overview"),
    ).toBeInTheDocument();
    expect(screen.getAllByTestId("math-block")).toHaveLength(2);
    const link = screen.getByRole("link", { name: "Example reference" });
    expect(link).toHaveAttribute("href", "https://example.com/paper");
    expect(link).toHaveAttribute("target", "_blank");
  });
});
