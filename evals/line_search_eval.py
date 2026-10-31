"""Synthetic benchmarks for backtracking line search."""

from __future__ import annotations

from trigger_agent_misc.optimization.line_search import backtracking_armijo


def eval_quadratic_along_gradient() -> dict[str, float | int | bool]:
    """Line search on ``phi(alpha) = (1 - alpha)^2`` with exact step ``alpha=1``."""

    def phi(alpha: float) -> float:
        return (1.0 - alpha) ** 2

    phi_prime0 = -2.0
    result = backtracking_armijo(phi, phi_prime0, alpha0=1.0)

    return {
        "alpha": result.alpha,
        "value": result.value,
        "iterations": result.iterations,
        "alpha_near_optimum": abs(result.alpha - 1.0) < 1e-6,
        "value_near_optimum": result.value < 1e-10,
    }


def eval_rosenbrock_along_steepest_descent_at_origin() -> dict[str, float | int | bool]:
    """One step along ``-grad f`` from the Rosenbrock start ``(0, 0)``."""

    def rosenbrock(x: float, y: float) -> float:
        return (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2

    x0, y0 = 0.0, 0.0
    grad_x = -2.0 * (1.0 - x0) - 400.0 * x0 * (y0 - x0**2)
    grad_y = 200.0 * (y0 - x0**2)
    px, py = -grad_x, -grad_y
    phi_prime0 = grad_x * px + grad_y * py

    def phi(alpha: float) -> float:
        return rosenbrock(x0 + alpha * px, y0 + alpha * py)

    result = backtracking_armijo(phi, phi_prime0, alpha0=1.0)

    return {
        "alpha": result.alpha,
        "value": result.value,
        "iterations": result.iterations,
        "decreased_objective": result.value < rosenbrock(x0, y0),
    }


def run_all() -> dict[str, dict[str, float | int | bool]]:
    return {
        "quadratic": eval_quadratic_along_gradient(),
        "rosenbrock_origin": eval_rosenbrock_along_steepest_descent_at_origin(),
    }


if __name__ == "__main__":
    for name, metrics in run_all().items():
        print(f"{name}: {metrics}")
