import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MathBlock } from "./MathBlock";

describe("MathBlock", () => {
  it("renders KaTeX output for a display equation", () => {
    const { container } = render(
      <MathBlock
        tex={"\\alpha^* = \\arg\\min_{\\alpha \\in [a,b]} \\varphi(\\alpha)"}
      />,
    );
    expect(screen.getByTestId("math-block")).toBeInTheDocument();
    expect(container.querySelector(".katex")).toBeTruthy();
  });
});
