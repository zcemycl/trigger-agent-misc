import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";
import { GRADIENT_METHOD_PAGES } from "./components/GradientMethodConcepts";
import { LINE_SEARCH_PAGES } from "./components/LineSearchConcepts";

describe("App", () => {
  it("renders gradient method sidebar by default", () => {
    render(<App />);
    expect(
      screen.getByRole("heading", { name: "Optimization Concepts" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("navigation", { name: "Optimization chapters" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("navigation", { name: "Gradient methods concepts" }),
    ).toBeInTheDocument();

    for (const { title } of GRADIENT_METHOD_PAGES) {
      expect(screen.getByRole("button", { name: title })).toBeInTheDocument();
    }

    expect(
      screen.getByRole("heading", { name: "Gradient descent (GD)" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Step-by-step example")).toBeInTheDocument();
    expect(
      screen.getByRole("table", { name: /gradient descent step-by-step/i }),
    ).toBeInTheDocument();
  });

  it("switches gradient concept page from the sidebar", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Mini-batch SGD" }));
    expect(
      screen.getByRole("heading", { name: "Mini-batch SGD" }),
    ).toBeInTheDocument();
  });

  it("shows line search chapter when selected", () => {
    render(<App />);
    fireEvent.click(
      within(
        screen.getByRole("navigation", { name: "Optimization chapters" }),
      ).getByRole("button", { name: "Line search" }),
    );
    expect(
      screen.getByRole("navigation", { name: "Line search concepts" }),
    ).toBeInTheDocument();
    for (const { title } of LINE_SEARCH_PAGES) {
      expect(screen.getByRole("button", { name: title })).toBeInTheDocument();
    }
    expect(
      screen.getByRole("heading", { name: "Exact line search" }),
    ).toBeInTheDocument();
  });
});
