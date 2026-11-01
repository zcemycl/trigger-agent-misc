"""Tests for second-order optimization methods."""

import math

from trigger_agent_misc.optimization.second_order import (
    OptimizeResult,
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
    return 0.5 * sum(
        (xi - target) ** 2 for xi, target in zip(x, (1.0, 2.0), strict=True)
    )


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


def test_newton_solves_quadratic_in_one_step() -> None:
    result = newtons_method(
        _shifted_quadratic,
        _shifted_quadratic_grad,
        _shifted_quadratic_hess,
        (0.0, 0.0),
    )
    assert isinstance(result, OptimizeResult)
    assert result.converged
    assert result.iterations == 1
    assert math.isclose(result.x[0], 1.0, abs_tol=1e-9)
    assert math.isclose(result.x[1], 2.0, abs_tol=1e-9)


def test_damped_newton_reaches_quadratic_min() -> None:
    result = damped_newton_method(
        _shifted_quadratic,
        _shifted_quadratic_grad,
        _shifted_quadratic_hess,
        (4.0, -3.0),
    )
    assert result.converged
    assert math.isclose(norm2(_shifted_quadratic_grad(result.x)), 0.0, abs_tol=1e-6)


def test_conjugate_gradient_finds_minimizer() -> None:
    result = conjugate_gradient(_shifted_quadratic, _shifted_quadratic_grad, (0.0, 0.0))
    assert result.converged
    assert math.isclose(result.x[0], 1.0, abs_tol=1e-4)
    assert math.isclose(result.x[1], 2.0, abs_tol=1e-4)


def test_bfgs_and_quasi_newton_alias() -> None:
    for solver in (bfgs, quasi_newton_bfgs):
        result = solver(_shifted_quadratic, _shifted_quadratic_grad, (0.0, 0.0))
        assert result.converged
        assert math.isclose(result.x[0], 1.0, abs_tol=1e-3)
        assert math.isclose(result.x[1], 2.0, abs_tol=1e-3)


def test_lbfgs_finds_minimizer() -> None:
    result = lbfgs(_shifted_quadratic, _shifted_quadratic_grad, (0.0, 0.0))
    assert result.converged
    assert math.isclose(result.x[0], 1.0, abs_tol=1e-3)
    assert math.isclose(result.x[1], 2.0, abs_tol=1e-3)


def test_gauss_newton_on_linear_least_squares() -> None:
    result = gauss_newton(_rosenbrock_residual, _rosenbrock_jacobian, (0.0, 0.0))
    assert result.converged
    assert result.value < 1e-6


def test_levenberg_marquardt_reduces_rosenbrock_residual() -> None:
    start = (0.0, 0.0)
    start_val = 0.5 * dot(_rosenbrock_residual(start), _rosenbrock_residual(start))
    result = levenberg_marquardt(_rosenbrock_residual, _rosenbrock_jacobian, start)
    assert result.value < start_val
    assert result.value < 0.5
