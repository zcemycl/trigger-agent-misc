import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ContourPathChart } from "./ContourPathChart";
import { ObjectiveLandscapeView } from "./ObjectiveLandscapeView";
import { Path3DChart } from "./Path3DChart";
import { SurfacePathChart } from "./SurfacePathChart";

const quad2 = (x: number[]) => 0.5 * ((x[0]! - 1) ** 2 + (x[1]! - 2) ** 2);

describe("OptimizationCharts", () => {
  it("renders contour and surface landscape views", () => {
    render(
      <ObjectiveLandscapeView
        objective={quad2}
        path={[
          [0, 0],
          [1, 2],
        ]}
      />,
    );
    expect(
      screen.getByRole("img", {
        name: /optimization path on objective contour/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /3d surface plot/i }),
    ).toBeInTheDocument();
  });

  it("renders 3D parameter path chart", () => {
    render(
      <Path3DChart
        path={[
          [0, 0, 0],
          [1, 2, 3],
        ]}
      />,
    );
    expect(
      screen.getByRole("img", {
        name: /three-dimensional parameter space/i,
      }),
    ).toBeInTheDocument();
  });

  it("marks highlighted path points in contour chart", () => {
    const { container } = render(
      <ContourPathChart
        objective={quad2}
        path={[
          [0, 0],
          [0.5, 1],
          [1, 2],
        ]}
        highlightPathRange={[1, 2]}
      />,
    );
    const circles = container.querySelectorAll("circle");
    expect(circles[1]?.getAttribute("r")).toBe("7");
    expect(circles[0]?.getAttribute("fill-opacity")).toBe("0.35");
  });

  it("renders surface path chart alone", () => {
    render(
      <SurfacePathChart
        objective={quad2}
        path={[
          [0, 0],
          [0.5, 1],
        ]}
      />,
    );
    expect(
      screen.getByRole("img", { name: /3d surface plot/i }),
    ).toBeInTheDocument();
  });
});
