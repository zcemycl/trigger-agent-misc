"""Tests for backtracking Armijo line search."""

import math

import pytest

from trigger_agent_misc.optimization.line_search import backtracking_armijo


def test_quadratic_finds_unit_step() -> None:
    def phi(alpha: float) -> float:
        return (1.0 - alpha) ** 2

    result = backtracking_armijo(phi, phi_prime_at_zero=-2.0, alpha0=1.0)
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
