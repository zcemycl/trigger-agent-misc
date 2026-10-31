import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("App", () => {
  it("renders interactive line search concept sections", () => {
    render(<App />);
    expect(
      screen.getByRole("heading", { name: "Line Search Concepts" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Exact line search" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Backtracking line search" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Armijo condition" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Wolfe conditions" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Strong Wolfe conditions" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Goldstein conditions" }),
    ).toBeInTheDocument();
    expect(screen.getAllByRole("slider").length).toBeGreaterThanOrEqual(6);
  });
});
