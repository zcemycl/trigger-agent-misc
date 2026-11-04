"""Synthetic benchmarks for gradient-based optimizers."""

from __future__ import annotations

import random

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


def _metrics(
    result_x: float, value: float, iterations: int
) -> dict[str, float | int | bool]:
    target = sum(CENTERS) / len(CENTERS)
    return {
        "x": result_x,
        "value": value,
        "iterations": iterations,
        "near_optimum": abs(result_x - target) < 1e-2,
    }


def eval_gradient_descent() -> dict[str, float | int | bool]:
    result = gradient_descent(
        (3.0,),
        _objective,
        _grad,
        learning_rate=0.2,
        max_iter=500,
        tol=1e-8,
    )
    return _metrics(result.x[0], result.value, result.iterations)


def eval_stochastic_gradient_descent() -> dict[str, float | int | bool]:
    result = stochastic_gradient_descent(
        (3.0,),
        _objective,
        _grad_at_index,
        num_samples=len(CENTERS),
        learning_rate=0.05,
        max_iter=800,
        tol=1e-8,
        rng=random.Random(0),
    )
    return _metrics(result.x[0], result.value, result.iterations)


def eval_minibatch_sgd() -> dict[str, float | int | bool]:
    result = minibatch_sgd(
        (3.0,),
        _objective,
        _grad_at_index,
        num_samples=len(CENTERS),
        batch_size=2,
        learning_rate=0.1,
        max_iter=400,
        tol=1e-8,
        rng=random.Random(1),
    )
    return _metrics(result.x[0], result.value, result.iterations)


def eval_momentum() -> dict[str, float | int | bool]:
    result = momentum(
        (3.0,),
        _objective,
        _grad,
        learning_rate=0.05,
        beta=0.9,
        max_iter=400,
        tol=1e-8,
    )
    return _metrics(result.x[0], result.value, result.iterations)


def eval_nesterov() -> dict[str, float | int | bool]:
    result = nesterov_accelerated_gradient(
        (3.0,),
        _objective,
        _grad,
        learning_rate=0.05,
        beta=0.9,
        max_iter=400,
        tol=1e-8,
    )
    return _metrics(result.x[0], result.value, result.iterations)


def eval_heavy_ball() -> dict[str, float | int | bool]:
    result = heavy_ball_method(
        (3.0,),
        _objective,
        _grad,
        learning_rate=0.05,
        beta=0.85,
        max_iter=400,
        tol=1e-8,
    )
    return _metrics(result.x[0], result.value, result.iterations)


def run_all() -> dict[str, dict[str, float | int | bool]]:
    return {
        "gradient_descent": eval_gradient_descent(),
        "stochastic_gradient_descent": eval_stochastic_gradient_descent(),
        "minibatch_sgd": eval_minibatch_sgd(),
        "momentum": eval_momentum(),
        "nesterov": eval_nesterov(),
        "heavy_ball": eval_heavy_ball(),
    }


if __name__ == "__main__":
    for name, metrics in run_all().items():
        print(name, metrics)
