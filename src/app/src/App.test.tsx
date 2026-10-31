import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";
import { LINE_SEARCH_PAGES } from "./components/LineSearchConcepts";

describe("App", () => {
  it("renders line search sidebar chapter and one concept page at a time", () => {
    render(<App />);
    expect(
      screen.getByRole("heading", { name: "Line Search Concepts" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Line search")).toBeInTheDocument();
    expect(
      screen.getByRole("navigation", { name: "Line search concepts" }),
    ).toBeInTheDocument();

    for (const { title } of LINE_SEARCH_PAGES) {
      expect(screen.getByRole("button", { name: title })).toBeInTheDocument();
    }

    expect(
      screen.getByRole("heading", { name: "Exact line search" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Armijo condition" }),
    ).not.toBeInTheDocument();
    expect(screen.getAllByRole("slider").length).toBe(1);
  });

  it("switches concept page when sidebar item is selected", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Armijo condition" }));
    expect(
      screen.getByRole("heading", { name: "Armijo condition" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Exact line search" }),
    ).not.toBeInTheDocument();
    expect(screen.getAllByTestId("math-block").length).toBeGreaterThanOrEqual(
      2,
    );
  });
});
