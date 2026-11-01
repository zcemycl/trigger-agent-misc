import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";
import { LINE_SEARCH_PAGES } from "./components/LineSearchConcepts";
import { SECOND_ORDER_PAGES } from "./components/SecondOrderConcepts";

describe("App", () => {
  it("renders second-order chapter by default with sidebar pages", () => {
    render(<App />);
    expect(
      screen.getByRole("heading", { name: "Optimization Concepts" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("navigation", {
        name: "Second-order optimization concepts",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Newton's method" }),
    ).toBeInTheDocument();

    for (const { title } of SECOND_ORDER_PAGES) {
      expect(screen.getByRole("button", { name: title })).toBeInTheDocument();
    }
  });

  it("switches to line search chapter", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Line search" }));
    expect(
      screen.getByRole("navigation", { name: "Line search concepts" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Exact line search" }),
    ).toBeInTheDocument();

    for (const { title } of LINE_SEARCH_PAGES) {
      expect(screen.getByRole("button", { name: title })).toBeInTheDocument();
    }
  });

  it("switches second-order concept page from sidebar", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "BFGS" }));
    expect(screen.getByRole("heading", { name: "BFGS" })).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Newton's method" }),
    ).not.toBeInTheDocument();
  });
});
