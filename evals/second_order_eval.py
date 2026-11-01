"""Synthetic benchmarks for second-order optimization methods."""

from __future__ import annotations

from trigger_agent_misc.optimization.second_order import (
    bfgs,
    conjugate_gradient,
    damped_newton_method,
    dot,
    gauss_newton,
    lbfgs,
    levenberg_marquardt,
    newtons_method,
    norm2,
    quasi_newton_bfgs,
)


def _shifted_quadratic(x: tuple[float, ...]) -> float:
    return 0.5 * sum((xi - t) ** 2 for xi, t in zip(x, (1.0, 2.0), strict=True))


def _shifted_quadratic_grad(x: tuple[float, ...]) -> tuple[float, float]:
    return (x[0] - 1.0, x[1] - 2.0)


def _shifted_quadratic_hess(
    _x: tuple[float, ...]
) -> tuple[tuple[float, float], tuple[float, float]]:
    return ((1.0, 0.0), (0.0, 1.0))


def _rosenbrock_residual(x: tuple[float, ...]) -> tuple[float, float]:
    return (x[0] - 1.0, 10.0 * (x[1] - x[0] ** 2))


def _rosenbrock_jacobian(
    x: tuple[float, ...]
) -> tuple[tuple[float, float], tuple[float, float]]:
    return ((1.0, 0.0), (-20.0 * x[0], 10.0))


def eval_newton_quadratic() -> dict[str, float | int | bool]:
    result = newtons_method(
        _shifted_quadratic,
        _shifted_quadratic_grad,
        _shifted_quadratic_hess,
        (0.0, 0.0),
    )
    return {
        "iterations": result.iterations,
        "converged": result.converged,
        "x0_near_one": abs(result.x[0] - 1.0) < 1e-8,
        "x1_near_two": abs(result.x[1] - 2.0) < 1e-8,
    }


def eval_damped_newton_quadratic() -> dict[str, float | int | bool]:
    result = damped_newton_method(
        _shifted_quadratic,
        _shifted_quadratic_grad,
        _shifted_quadratic_hess,
        (4.0, -3.0),
    )
    return {
        "converged": result.converged,
        "grad_small": norm2(_shifted_quadratic_grad(result.x)) < 1e-6,
    }


def eval_conjugate_gradient_quadratic() -> dict[str, float | int | bool]:
    result = conjugate_gradient(_shifted_quadratic, _shifted_quadratic_grad, (0.0, 0.0))
    return {
        "converged": result.converged,
        "x0_near_one": abs(result.x[0] - 1.0) < 1e-3,
        "x1_near_two": abs(result.x[1] - 2.0) < 1e-3,
    }


def eval_bfgs_quadratic() -> dict[str, float | int | bool]:
    result = bfgs(_shifted_quadratic, _shifted_quadratic_grad, (0.0, 0.0))
    return {
        "converged": result.converged,
        "x0_near_one": abs(result.x[0] - 1.0) < 1e-3,
    }


def eval_quasi_newton_bfgs_quadratic() -> dict[str, float | int | bool]:
    result = quasi_newton_bfgs(_shifted_quadratic, _shifted_quadratic_grad, (0.0, 0.0))
    return {"converged": result.converged, "value_small": result.value < 1e-4}


def eval_lbfgs_quadratic() -> dict[str, float | int | bool]:
    result = lbfgs(_shifted_quadratic, _shifted_quadratic_grad, (0.0, 0.0))
    return {
        "converged": result.converged,
        "x1_near_two": abs(result.x[1] - 2.0) < 1e-3,
    }


def eval_gauss_newton_rosenbrock() -> dict[str, float | int | bool]:
    result = gauss_newton(_rosenbrock_residual, _rosenbrock_jacobian, (0.0, 0.0))
    return {"converged": result.converged, "value_small": result.value < 1e-5}


def eval_levenberg_marquardt_rosenbrock() -> dict[str, float | int | bool]:
    start = (0.0, 0.0)
    start_val = 0.5 * dot(_rosenbrock_residual(start), _rosenbrock_residual(start))
    result = levenberg_marquardt(_rosenbrock_residual, _rosenbrock_jacobian, start)
    return {
        "reduced_objective": result.value < start_val,
        "value_below_half": result.value < 0.5,
    }


def run_all() -> dict[str, dict[str, float | int | bool]]:
    return {
        "newton": eval_newton_quadratic(),
        "damped_newton": eval_damped_newton_quadratic(),
        "conjugate_gradient": eval_conjugate_gradient_quadratic(),
        "bfgs": eval_bfgs_quadratic(),
        "quasi_newton_bfgs": eval_quasi_newton_bfgs_quadratic(),
        "lbfgs": eval_lbfgs_quadratic(),
        "gauss_newton": eval_gauss_newton_rosenbrock(),
        "levenberg_marquardt": eval_levenberg_marquardt_rosenbrock(),
    }


if __name__ == "__main__":
    for name, metrics in run_all().items():
        print(f"{name}: {metrics}")
