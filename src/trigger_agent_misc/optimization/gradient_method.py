"""First-order gradient-based optimization methods."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

Vector = tuple[float, ...]
Objective = Callable[[Vector], float]
GradFn = Callable[[Vector], Vector]
SampleGradFn = Callable[[int, Vector], Vector]


@dataclass(frozen=True)
class GradientMethodResult:
    """Outcome of running a gradient-based optimizer."""

    x: Vector
    value: float
    iterations: int
    history: tuple[Vector, ...]


def _validate_vector(x: Sequence[float]) -> Vector:
    if not x:
        raise ValueError("x must be non-empty")
    return tuple(float(v) for v in x)


def _add(a: Vector, b: Vector) -> Vector:
    return tuple(x + y for x, y in zip(a, b))


def _sub(a: Vector, b: Vector) -> Vector:
    return tuple(x - y for x, y in zip(a, b))


def _scale(s: float, v: Vector) -> Vector:
    return tuple(s * x for x in v)


def _norm(v: Vector) -> float:
    return sum(x * x for x in v) ** 0.5


def _mean(vectors: Sequence[Vector]) -> Vector:
    if not vectors:
        raise ValueError("cannot average empty batch")
    dim = len(vectors[0])
    acc = [0.0] * dim
    for vec in vectors:
        if len(vec) != dim:
            raise ValueError("batch vectors must share dimension")
        for i, val in enumerate(vec):
            acc[i] += val
    n = float(len(vectors))
    return tuple(v / n for v in acc)


def _run_first_order(
    x0: Sequence[float],
    objective: Objective,
    step: Callable[[Vector, int], Vector | None],
    *,
    max_iter: int,
    tol: float,
) -> GradientMethodResult:
    x = _validate_vector(x0)
    history: list[Vector] = [x]
    for k in range(max_iter):
        delta = step(x, k)
        if delta is None:
            break
        x = _add(x, delta)
        history.append(x)
        if _norm(delta) <= tol:
            break
    return GradientMethodResult(
        x=x,
        value=objective(x),
        iterations=len(history) - 1,
        history=tuple(history),
    )


def gradient_descent(
    x0: Sequence[float],
    objective: Objective,
    grad: GradFn,
    *,
    learning_rate: float = 0.1,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> GradientMethodResult:
    """Full-batch gradient descent: ``x <- x - lr * grad f(x)``."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    def step(x: Vector, _k: int) -> Vector | None:
        g = _validate_vector(grad(x))
        return _scale(-learning_rate, g)

    return _run_first_order(x0, objective, step, max_iter=max_iter, tol=tol)


def stochastic_gradient_descent(
    x0: Sequence[float],
    objective: Objective,
    grad_at_index: SampleGradFn,
    num_samples: int,
    *,
    learning_rate: float = 0.1,
    max_iter: int = 200,
    tol: float = 1e-9,
    rng: random.Random | None = None,
) -> GradientMethodResult:
    """SGD: one sampled gradient per step."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    random_source = rng or random.Random()

    def step(x: Vector, _k: int) -> Vector | None:
        idx = random_source.randrange(num_samples)
        g = _validate_vector(grad_at_index(idx, x))
        return _scale(-learning_rate, g)

    return _run_first_order(x0, objective, step, max_iter=max_iter, tol=tol)


def minibatch_sgd(
    x0: Sequence[float],
    objective: Objective,
    grad_at_index: SampleGradFn,
    num_samples: int,
    *,
    batch_size: int = 4,
    learning_rate: float = 0.1,
    max_iter: int = 200,
    tol: float = 1e-9,
    rng: random.Random | None = None,
) -> GradientMethodResult:
    """Mini-batch SGD: average gradient over a sampled batch each step."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    random_source = rng or random.Random()
    size = min(batch_size, num_samples)

    def step(x: Vector, _k: int) -> Vector | None:
        batch = random_source.sample(range(num_samples), k=size)
        grads = [_validate_vector(grad_at_index(i, x)) for i in batch]
        g = _mean(grads)
        return _scale(-learning_rate, g)

    return _run_first_order(x0, objective, step, max_iter=max_iter, tol=tol)


def momentum(
    x0: Sequence[float],
    objective: Objective,
    grad: GradFn,
    *,
    learning_rate: float = 0.1,
    beta: float = 0.9,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> GradientMethodResult:
    """Polyak momentum: accumulate velocity ``v <- beta * v + grad f(x)``."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not (0.0 <= beta < 1.0):
        raise ValueError("beta must be in [0, 1)")
    velocity = _scale(0.0, _validate_vector(x0))

    def step(x: Vector, _k: int) -> Vector | None:
        nonlocal velocity
        g = _validate_vector(grad(x))
        velocity = _add(_scale(beta, velocity), g)
        return _scale(-learning_rate, velocity)

    return _run_first_order(x0, objective, step, max_iter=max_iter, tol=tol)


def nesterov_accelerated_gradient(
    x0: Sequence[float],
    objective: Objective,
    grad: GradFn,
    *,
    learning_rate: float = 0.1,
    beta: float = 0.9,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> GradientMethodResult:
    """Nesterov AG: evaluate the gradient at a look-ahead point."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not (0.0 <= beta < 1.0):
        raise ValueError("beta must be in [0, 1)")
    velocity = _scale(0.0, _validate_vector(x0))

    def step(x: Vector, _k: int) -> Vector | None:
        nonlocal velocity
        look_ahead = _sub(x, _scale(beta, velocity))
        g = _validate_vector(grad(look_ahead))
        velocity = _add(_scale(beta, velocity), g)
        return _scale(-learning_rate, velocity)

    return _run_first_order(x0, objective, step, max_iter=max_iter, tol=tol)


def heavy_ball_method(
    x0: Sequence[float],
    objective: Objective,
    grad: GradFn,
    *,
    learning_rate: float = 0.1,
    beta: float = 0.9,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> GradientMethodResult:
    """Heavy-ball: ``x <- x - lr * grad f(x) + beta * (x - x_prev)``."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not (0.0 <= beta < 1.0):
        raise ValueError("beta must be in [0, 1)")
    x_prev = _validate_vector(x0)

    def step(x: Vector, k: int) -> Vector | None:
        nonlocal x_prev
        g = _validate_vector(grad(x))
        if k == 0:
            delta = _scale(-learning_rate, g)
        else:
            delta = _add(_scale(-learning_rate, g), _scale(beta, _sub(x, x_prev)))
        x_prev = x
        return delta

    return _run_first_order(x0, objective, step, max_iter=max_iter, tol=tol)
