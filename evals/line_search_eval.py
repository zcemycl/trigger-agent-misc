"""Synthetic benchmarks for line search methods and conditions."""

from __future__ import annotations

from trigger_agent_misc.optimization.line_search import (
    armijo_sufficient_decrease,
    backtracking_armijo,
    exact_line_search,
    goldstein_conditions,
    line_search_goldstein,
    line_search_strong_wolfe,
    line_search_wolfe,
    strong_wolfe_curvature,
    wolfe_curvature,
)


def _quadratic_phi(alpha: float) -> float:
    return (1.0 - alpha) ** 2


def _quadratic_phi_prime(alpha: float) -> float:
    return -2.0 * (1.0 - alpha)


def eval_exact_line_search_quadratic() -> dict[str, float | int | bool]:
    """Exact search on ``phi(alpha) = (1 - alpha)^2`` over ``[0, 2]``."""
    result = exact_line_search(_quadratic_phi, a=0.0, b=2.0)
    return {
        "alpha": result.alpha,
        "value": result.value,
        "iterations": result.iterations,
        "alpha_near_optimum": abs(result.alpha - 1.0) < 1e-6,
    }


def eval_backtracking_armijo_quadratic() -> dict[str, float | int | bool]:
    """Backtracking with Armijo on the same quadratic ray."""
    phi_prime0 = -2.0
    result = backtracking_armijo(_quadratic_phi, phi_prime0, alpha0=1.0)
    phi0 = _quadratic_phi(0.0)
    return {
        "alpha": result.alpha,
        "value": result.value,
        "iterations": result.iterations,
        "armijo_holds": armijo_sufficient_decrease(
            phi0, result.value, result.alpha, phi_prime0
        ),
    }


def eval_armijo_condition_example() -> dict[str, float | bool]:
    """Demonstrate Armijo pass/fail at two trial steps."""
    phi0 = _quadratic_phi(0.0)
    phi_prime0 = -2.0
    alpha_good = 1.0
    alpha_bad = 2.0
    return {
        "alpha_good": alpha_good,
        "armijo_at_good": armijo_sufficient_decrease(
            phi0, _quadratic_phi(alpha_good), alpha_good, phi_prime0
        ),
        "armijo_at_bad": armijo_sufficient_decrease(
            phi0, _quadratic_phi(alpha_bad), alpha_bad, phi_prime0
        ),
    }


def eval_wolfe_line_search_quadratic() -> dict[str, float | int | bool]:
    phi_prime0 = -2.0
    result = line_search_wolfe(
        _quadratic_phi, _quadratic_phi_prime, phi_prime0, alpha0=1.0
    )
    return {
        "alpha": result.alpha,
        "value": result.value,
        "iterations": result.iterations,
        "wolfe_curvature_holds": wolfe_curvature(
            _quadratic_phi_prime(result.alpha), phi_prime0
        ),
    }


def eval_strong_wolfe_line_search_quadratic() -> dict[str, float | int | bool]:
    phi_prime0 = -2.0
    result = line_search_strong_wolfe(
        _quadratic_phi, _quadratic_phi_prime, phi_prime0, alpha0=1.0
    )
    return {
        "alpha": result.alpha,
        "value": result.value,
        "iterations": result.iterations,
        "strong_wolfe_holds": strong_wolfe_curvature(
            _quadratic_phi_prime(result.alpha), phi_prime0
        ),
    }


def eval_goldstein_line_search_quadratic() -> dict[str, float | int | bool]:
    phi_prime0 = -2.0
    result = line_search_goldstein(_quadratic_phi, phi_prime0, alpha0=1.0)
    phi0 = _quadratic_phi(0.0)
    return {
        "alpha": result.alpha,
        "value": result.value,
        "iterations": result.iterations,
        "goldstein_holds": goldstein_conditions(
            phi0, result.value, result.alpha, phi_prime0, c=0.1
        ),
    }


def eval_quadratic_along_gradient() -> dict[str, float | int | bool]:
    """Line search on ``phi(alpha) = (1 - alpha)^2`` with exact step ``alpha=1``."""

    phi_prime0 = -2.0
    result = backtracking_armijo(_quadratic_phi, phi_prime0, alpha0=1.0)

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
        "exact_quadratic": eval_exact_line_search_quadratic(),
        "backtracking_armijo": eval_backtracking_armijo_quadratic(),
        "armijo_condition": eval_armijo_condition_example(),
        "wolfe": eval_wolfe_line_search_quadratic(),
        "strong_wolfe": eval_strong_wolfe_line_search_quadratic(),
        "goldstein": eval_goldstein_line_search_quadratic(),
        "quadratic": eval_quadratic_along_gradient(),
        "rosenbrock_origin": eval_rosenbrock_along_steepest_descent_at_origin(),
    }


if __name__ == "__main__":
    for name, metrics in run_all().items():
        print(f"{name}: {metrics}")
