"""Tests for line search methods and conditions."""

import math

import pytest

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


def test_quadratic_finds_unit_step() -> None:
    result = backtracking_armijo(_quadratic_phi, phi_prime_at_zero=-2.0, alpha0=1.0)
    assert math.isclose(result.alpha, 1.0, rel_tol=0.0, abs_tol=1e-9)
    assert result.value < 1e-12
    assert result.iterations == 0


def test_backtracks_when_alpha0_is_too_large() -> None:
    def phi(alpha: float) -> float:
        return alpha**2

    result = backtracking_armijo(phi, phi_prime_at_zero=-2.0, alpha0=10.0, c1=0.1)
    assert result.alpha < 10.0
    assert result.iterations >= 1
    assert result.value <= phi(0.0) + 0.1 * result.alpha * (-2.0) + 1e-12


def test_rejects_non_descent_direction() -> None:
    with pytest.raises(ValueError, match="descent"):
        backtracking_armijo(lambda a: a**2, phi_prime_at_zero=1.0)


def test_armijo_condition_on_quadratic_example() -> None:
    phi0 = _quadratic_phi(0.0)
    alpha = 0.5
    assert armijo_sufficient_decrease(phi0, _quadratic_phi(alpha), alpha, -2.0, c1=1e-4)
    assert not armijo_sufficient_decrease(phi0, phi0, 1.0, -2.0, c1=1e-4)


def test_exact_line_search_finds_quadratic_minimizer() -> None:
    result = exact_line_search(_quadratic_phi, a=0.0, b=2.0)
    assert math.isclose(result.alpha, 1.0, abs_tol=1e-6)
    assert result.value < 1e-10


def test_wolfe_curvature_at_minimizer() -> None:
    assert wolfe_curvature(0.0, -2.0, c2=0.9)
    assert not wolfe_curvature(-2.0, -2.0, c2=0.9)


def test_strong_wolfe_curvature() -> None:
    assert strong_wolfe_curvature(0.0, -2.0, c2=0.9)
    assert strong_wolfe_curvature(1.0, -2.0, c2=0.9)
    assert not strong_wolfe_curvature(-2.0, -2.0, c2=0.9)


def test_goldstein_conditions_example() -> None:
    phi0 = 1.0
    phi_prime0 = -2.0
    alpha = 0.25
    phi_alpha = phi0 + 0.1 * alpha * phi_prime0
    assert goldstein_conditions(phi0, phi_alpha, alpha, phi_prime0, c=0.1)


def test_line_search_wolfe_on_quadratic() -> None:
    result = line_search_wolfe(
        _quadratic_phi, _quadratic_phi_prime, phi_prime_at_zero=-2.0, alpha0=1.0
    )
    assert armijo_sufficient_decrease(
        _quadratic_phi(0.0), result.value, result.alpha, -2.0
    )
    assert wolfe_curvature(_quadratic_phi_prime(result.alpha), -2.0)
    assert math.isclose(result.alpha, 1.0, abs_tol=1e-2)


def test_line_search_strong_wolfe_on_quadratic() -> None:
    result = line_search_strong_wolfe(
        _quadratic_phi, _quadratic_phi_prime, phi_prime_at_zero=-2.0, alpha0=1.0
    )
    assert strong_wolfe_curvature(_quadratic_phi_prime(result.alpha), -2.0)
    assert result.value < 0.5


def test_line_search_goldstein_on_quadratic() -> None:
    result = line_search_goldstein(_quadratic_phi, phi_prime_at_zero=-2.0, alpha0=1.0)
    assert goldstein_conditions(
        _quadratic_phi(0.0), result.value, result.alpha, -2.0, c=0.1
    )
