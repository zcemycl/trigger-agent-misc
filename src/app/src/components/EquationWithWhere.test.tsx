import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MathBlock } from "./MathBlock";
import { EquationWithWhere } from "./EquationWithWhere";

describe("EquationWithWhere", () => {
  it("renders where clauses with inline math", () => {
    render(
      <EquationWithWhere
        where={[
          { tex: "\\eta", description: "learning rate" },
          { tex: "\\mathbf{x}_k", description: "current iterate" },
        ]}
      >
        <MathBlock tex="\\mathbf{x}_{k+1} = \\mathbf{x}_k - \\eta \\nabla f(\\mathbf{x}_k)" />
      </EquationWithWhere>,
    );
    expect(screen.getByTestId("math-where")).toHaveTextContent("where");
    expect(screen.getByTestId("math-where")).toHaveTextContent("learning rate");
    expect(screen.getByTestId("math-block")).toBeInTheDocument();
  });
});
