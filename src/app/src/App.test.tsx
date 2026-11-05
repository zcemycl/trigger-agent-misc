import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";
import { GRADIENT_METHOD_PAGES } from "./components/GradientMethodConcepts";
import { LINE_SEARCH_PAGES } from "./components/LineSearchConcepts";
import { SECOND_ORDER_PAGES } from "./components/SecondOrderConcepts";

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
    const conceptNav = screen.getByRole("navigation", {
      name: "Gradient methods concepts",
    });
    expect(within(conceptNav).getAllByRole("button")[0]).toHaveAccessibleName(
      "Overview",
    );

    for (const { title } of GRADIENT_METHOD_PAGES) {
      expect(screen.getByRole("button", { name: title })).toBeInTheDocument();
    }

    expect(
      screen.getByRole("heading", { name: "Gradient descent (GD)" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("table", { name: /gradient descent step-by-step/i }),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Overview" }));
    expect(
      screen.getByLabelText("Gradient descent (GD) overview"),
    ).toBeInTheDocument();
    expect(screen.getAllByTestId("math-block").length).toBeGreaterThanOrEqual(
      2,
    );
    expect(screen.getAllByRole("link").length).toBeGreaterThanOrEqual(1);
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
    fireEvent.click(screen.getByRole("button", { name: "Overview" }));
    expect(
      screen.getByLabelText("Exact line search overview"),
    ).toBeInTheDocument();
  });

  it("shows second-order chapter when selected", () => {
    render(<App />);
    fireEvent.click(
      within(
        screen.getByRole("navigation", { name: "Optimization chapters" }),
      ).getByRole("button", { name: "Second-order" }),
    );
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
    expect(screen.getByRole("button", { name: "Booth" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Ill-conditioned quadratic" }),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Overview" }));
    expect(
      screen.getByLabelText("Newton's method overview"),
    ).toBeInTheDocument();
  });

  it("switches second-order concept page from sidebar", () => {
    render(<App />);
    fireEvent.click(
      within(
        screen.getByRole("navigation", { name: "Optimization chapters" }),
      ).getByRole("button", { name: "Second-order" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "BFGS" }));
    expect(screen.getByRole("heading", { name: "BFGS" })).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Newton's method" }),
    ).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Overview" }));
    expect(screen.getByLabelText("BFGS overview")).toBeInTheDocument();
  });
});
