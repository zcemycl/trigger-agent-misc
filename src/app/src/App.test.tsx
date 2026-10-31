import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("App", () => {
  it("renders the readings page title and entry links", () => {
    render(<App />);
    expect(
      screen.getByRole("heading", { name: "Readings" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /Line Search Methods/i }),
    ).toHaveAttribute(
      "href",
      "https://github.com/zcemycl/trigger-agent-misc/blob/main/notebooks/line_search_demo.ipynb",
    );
  });
});
