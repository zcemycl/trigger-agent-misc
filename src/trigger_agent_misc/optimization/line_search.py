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


def armijo_sufficient_decrease(
    phi0: float,
    phi_alpha: float,
    alpha: float,
    phi_prime_at_zero: float,
    *,
    c1: float = 1e-4,
) -> bool:
    """Armijo condition: ``phi(alpha) <= phi(0) + c1 * alpha * phi'(0)``."""
    return phi_alpha <= phi0 + c1 * alpha * phi_prime_at_zero


def wolfe_curvature(
    phi_prime_alpha: float,
    phi_prime_at_zero: float,
    *,
    c2: float = 0.9,
) -> bool:
    """Weak Wolfe curvature: ``phi'(alpha) >= c2 * phi'(0)``."""
    return phi_prime_alpha >= c2 * phi_prime_at_zero


def strong_wolfe_curvature(
    phi_prime_alpha: float,
    phi_prime_at_zero: float,
    *,
    c2: float = 0.9,
) -> bool:
    """Strong Wolfe curvature: ``|phi'(alpha)| <= c2 * |phi'(0)|``."""
    return abs(phi_prime_alpha) <= c2 * abs(phi_prime_at_zero)


def goldstein_conditions(
    phi0: float,
    phi_alpha: float,
    alpha: float,
    phi_prime_at_zero: float,
    *,
    c: float = 0.1,
) -> bool:
    """Goldstein interval with ``0 < c < 1/2``.

    ``phi(0) + (1 - c) * alpha * phi'(0) <= phi(alpha) <= phi(0) + c * alpha * phi'(0)``
    """
    if not (0.0 < c < 0.5):
        raise ValueError("c must be in (0, 0.5)")
    lower = phi0 + (1.0 - c) * alpha * phi_prime_at_zero
    upper = phi0 + c * alpha * phi_prime_at_zero
    return lower <= phi_alpha <= upper


def exact_line_search(
    phi: Callable[[float], float],
    *,
    a: float = 0.0,
    b: float = 1.0,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> LineSearchResult:
    """Golden-section exact line search for unimodal ``phi`` on ``[a, b]``."""
    if a >= b:
        raise ValueError("require a < b")
    gr = (5**0.5 - 1.0) / 2.0
    iterations = 0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = phi(c)
    fd = phi(d)

    while b - a > tol and iterations < max_iter:
        iterations += 1
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = phi(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = phi(d)

    alpha = 0.5 * (a + b)
    return LineSearchResult(alpha=alpha, value=phi(alpha), iterations=iterations)


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

    while not armijo_sufficient_decrease(
        phi0, phi(alpha), alpha, phi_prime_at_zero, c1=c1
    ):
        iterations += 1
        if iterations >= max_iter:
            break
        next_alpha = alpha * rho
        if next_alpha < alpha_min:
            alpha = next_alpha
            break
        alpha = next_alpha

    return LineSearchResult(alpha=alpha, value=phi(alpha), iterations=iterations)


backtracking_line_search = backtracking_armijo


def _zoom_wolfe(
    phi: Callable[[float], float],
    phi_prime: Callable[[float], float],
    phi0: float,
    phi_prime_at_zero: float,
    alpha_lo: float,
    alpha_hi: float,
    *,
    c1: float,
    c2: float,
    strong: bool,
    max_iter: int,
) -> float:
    for _ in range(max_iter):
        alpha = 0.5 * (alpha_lo + alpha_hi)
        phi_a = phi(alpha)
        if phi_a > phi0 + c1 * alpha * phi_prime_at_zero or phi(alpha_lo) <= phi_a:
            alpha_hi = alpha
            continue
        phi_p = phi_prime(alpha)
        if strong:
            if strong_wolfe_curvature(phi_p, phi_prime_at_zero, c2=c2):
                return alpha
        elif wolfe_curvature(phi_p, phi_prime_at_zero, c2=c2):
            return alpha
        if phi_p * (alpha_hi - alpha_lo) >= 0.0:
            alpha_hi = alpha_lo
        alpha_lo = alpha
    return 0.5 * (alpha_lo + alpha_hi)


def _line_search_wolfe_impl(
    phi: Callable[[float], float],
    phi_prime: Callable[[float], float],
    phi_prime_at_zero: float,
    *,
    alpha0: float,
    c1: float,
    c2: float,
    alpha_max: float,
    max_iter: int,
    strong: bool,
) -> LineSearchResult:
    if phi_prime_at_zero >= 0:
        raise ValueError("phi_prime_at_zero must be negative for a descent direction")
    if not (0.0 < c1 < c2 < 1.0):
        raise ValueError("require 0 < c1 < c2 < 1")

    phi0 = phi(0.0)
    alpha_prev = 0.0
    phi_prev = phi0
    alpha = alpha0
    iterations = 0

    while True:
        iterations += 1
        if iterations > max_iter:
            break
        phi_a = phi(alpha)
        if phi_a > phi0 + c1 * alpha * phi_prime_at_zero or (
            phi_a >= phi_prev and alpha > alpha_prev
        ):
            alpha_star = _zoom_wolfe(
                phi,
                phi_prime,
                phi0,
                phi_prime_at_zero,
                alpha_prev,
                alpha,
                c1=c1,
                c2=c2,
                strong=strong,
                max_iter=max_iter,
            )
            return LineSearchResult(
                alpha=alpha_star, value=phi(alpha_star), iterations=iterations
            )
        phi_p = phi_prime(alpha)
        if strong:
            if strong_wolfe_curvature(phi_p, phi_prime_at_zero, c2=c2):
                return LineSearchResult(alpha=alpha, value=phi_a, iterations=iterations)
        elif wolfe_curvature(phi_p, phi_prime_at_zero, c2=c2):
            return LineSearchResult(alpha=alpha, value=phi_a, iterations=iterations)
        if alpha >= alpha_max:
            return LineSearchResult(alpha=alpha, value=phi_a, iterations=iterations)
        alpha_prev, phi_prev = alpha, phi_a
        alpha = min(2.0 * alpha, alpha_max)

    return LineSearchResult(alpha=alpha, value=phi(alpha), iterations=iterations)


def line_search_wolfe(
    phi: Callable[[float], float],
    phi_prime: Callable[[float], float],
    phi_prime_at_zero: float,
    *,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    alpha_max: float = 10.0,
    max_iter: int = 100,
) -> LineSearchResult:
    """Line search satisfying Armijo and weak Wolfe curvature conditions."""
    return _line_search_wolfe_impl(
        phi,
        phi_prime,
        phi_prime_at_zero,
        alpha0=alpha0,
        c1=c1,
        c2=c2,
        alpha_max=alpha_max,
        max_iter=max_iter,
        strong=False,
    )


def line_search_strong_wolfe(
    phi: Callable[[float], float],
    phi_prime: Callable[[float], float],
    phi_prime_at_zero: float,
    *,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    alpha_max: float = 10.0,
    max_iter: int = 100,
) -> LineSearchResult:
    """Line search satisfying Armijo and strong Wolfe curvature conditions."""
    return _line_search_wolfe_impl(
        phi,
        phi_prime,
        phi_prime_at_zero,
        alpha0=alpha0,
        c1=c1,
        c2=c2,
        alpha_max=alpha_max,
        max_iter=max_iter,
        strong=True,
    )


def line_search_goldstein(
    phi: Callable[[float], float],
    phi_prime_at_zero: float,
    *,
    alpha0: float = 1.0,
    c: float = 0.1,
    rho: float = 0.5,
    alpha_max: float = 10.0,
    max_iter: int = 100,
) -> LineSearchResult:
    """Backtracking / expansion until Goldstein conditions hold."""
    if phi_prime_at_zero >= 0:
        raise ValueError("phi_prime_at_zero must be negative for a descent direction")
    if not (0.0 < c < 0.5):
        raise ValueError("c must be in (0, 0.5)")
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0, 1)")

    phi0 = phi(0.0)
    alpha = alpha0
    iterations = 0

    while iterations < max_iter:
        phi_a = phi(alpha)
        if goldstein_conditions(phi0, phi_a, alpha, phi_prime_at_zero, c=c):
            return LineSearchResult(alpha=alpha, value=phi_a, iterations=iterations)
        iterations += 1
        if phi_a > phi0 + c * alpha * phi_prime_at_zero:
            alpha *= rho
        elif phi_a < phi0 + (1.0 - c) * alpha * phi_prime_at_zero:
            alpha = min(alpha / rho, alpha_max)
        else:
            break

    return LineSearchResult(alpha=alpha, value=phi(alpha), iterations=iterations)
