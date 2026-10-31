"""One-dimensional line search along a fixed direction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class LineSearchResult:
    """Outcome of a line search along ``x + alpha * direction``."""

    alpha: float
    value: float
    iterations: int


def backtracking_armijo(
    phi: Callable[[float], float],
    phi_prime_at_zero: float,
    *,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    alpha_min: float = 1e-14,
    max_iter: int = 100,
) -> LineSearchResult:
    """Backtracking line search with the Armijo sufficient-decrease condition.

    ``phi`` is the restriction of the objective to the search ray,
    ``phi(alpha) = f(x + alpha * p)``. ``phi_prime_at_zero`` is ``phi'(0)`` (often
    ``grad f(x) · p``) and must be negative for a descent direction.
    """
    if phi_prime_at_zero >= 0:
        raise ValueError("phi_prime_at_zero must be negative for a descent direction")
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0, 1)")
    if not (0.0 < c1 < 1.0):
        raise ValueError("c1 must be in (0, 1)")

    phi0 = phi(0.0)
    alpha = alpha0
    iterations = 0

    while phi(alpha) > phi0 + c1 * alpha * phi_prime_at_zero:
        iterations += 1
        if iterations >= max_iter:
            break
        next_alpha = alpha * rho
        if next_alpha < alpha_min:
            alpha = next_alpha
            break
        alpha = next_alpha

    return LineSearchResult(alpha=alpha, value=phi(alpha), iterations=iterations)
