"""Tests for gradient-based optimization methods."""

import math
import random

import pytest

from trigger_agent_misc.optimization.gradient_method import (
    gradient_descent,
    heavy_ball_method,
    minibatch_sgd,
    momentum,
    nesterov_accelerated_gradient,
    stochastic_gradient_descent,
)

CENTERS = (-2.0, 0.5, 1.5)


def _objective(x: tuple[float, ...]) -> float:
    return sum(0.5 * (x[0] - c) ** 2 for c in CENTERS) / len(CENTERS)


def _grad(x: tuple[float, ...]) -> tuple[float, ...]:
    mean = sum(x[0] - c for c in CENTERS) / len(CENTERS)
    return (mean,)


def _grad_at_index(i: int, x: tuple[float, ...]) -> tuple[float, ...]:
    return (x[0] - CENTERS[i],)


def test_gradient_descent_finds_quadratic_mean() -> None:
    result = gradient_descent(
        (3.0,),
        _objective,
        _grad,
        learning_rate=0.2,
        max_iter=500,
        tol=1e-8,
    )
    expected = sum(CENTERS) / len(CENTERS)
    assert math.isclose(result.x[0], expected, abs_tol=1e-4)
    assert math.isclose(result.value, _objective((expected,)), abs_tol=1e-4)
    assert len(result.history) >= 2


def test_stochastic_gradient_descent_runs_with_seed() -> None:
    rng = random.Random(0)
    result = stochastic_gradient_descent(
        (3.0,),
        _objective,
        _grad_at_index,
        num_samples=len(CENTERS),
        learning_rate=0.05,
        max_iter=800,
        tol=1e-8,
        rng=rng,
    )
    assert result.iterations >= 1
    assert result.value < _objective((3.0,))


def test_minibatch_sgd_uses_batch() -> None:
    rng = random.Random(1)
    result = minibatch_sgd(
        (3.0,),
        _objective,
        _grad_at_index,
        num_samples=len(CENTERS),
        batch_size=2,
        learning_rate=0.15,
        max_iter=2000,
        tol=1e-8,
        rng=rng,
    )
    expected = sum(CENTERS) / len(CENTERS)
    assert math.isclose(result.x[0], expected, abs_tol=5e-2)
    assert result.value < _objective((3.0,))


def test_momentum_reaches_minimum() -> None:
    result = momentum(
        (3.0,),
        _objective,
        _grad,
        learning_rate=0.05,
        beta=0.9,
        max_iter=400,
        tol=1e-8,
    )
    expected = sum(CENTERS) / len(CENTERS)
    assert math.isclose(result.x[0], expected, abs_tol=1e-3)


def test_nesterov_reaches_minimum() -> None:
    result = nesterov_accelerated_gradient(
        (3.0,),
        _objective,
        _grad,
        learning_rate=0.05,
        beta=0.9,
        max_iter=400,
        tol=1e-8,
    )
    expected = sum(CENTERS) / len(CENTERS)
    assert math.isclose(result.x[0], expected, abs_tol=1e-3)


def test_heavy_ball_reaches_minimum() -> None:
    result = heavy_ball_method(
        (3.0,),
        _objective,
        _grad,
        learning_rate=0.05,
        beta=0.85,
        max_iter=400,
        tol=1e-8,
    )
    expected = sum(CENTERS) / len(CENTERS)
    assert math.isclose(result.x[0], expected, abs_tol=1e-3)


def test_rejects_invalid_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        gradient_descent((0.0,), _objective, _grad, learning_rate=0.0)
