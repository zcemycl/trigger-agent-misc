"""Multivariate second-order optimization methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from trigger_agent_misc.optimization.line_search import backtracking_armijo

Vec = tuple[float, ...]
Mat = tuple[tuple[float, ...], ...]
Objective = Callable[[Vec], float]
Gradient = Callable[[Vec], Vec]
Hessian = Callable[[Vec], Mat]
Residual = Callable[[Vec], Vec]
Jacobian = Callable[[Vec], Mat]


@dataclass(frozen=True)
class OptimizeResult:
    """Outcome of a multivariate optimization run."""

    x: Vec
    value: float
    iterations: int
    converged: bool


def _as_vec(x: Sequence[float]) -> Vec:
    return tuple(float(v) for v in x)


def vec_add(a: Sequence[float], b: Sequence[float]) -> Vec:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def vec_sub(a: Sequence[float], b: Sequence[float]) -> Vec:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def vec_scale(a: Sequence[float], s: float) -> Vec:
    return tuple(s * x for x in a)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def norm2(a: Sequence[float]) -> float:
    return dot(a, a) ** 0.5


def mat_vec(a: Mat, x: Sequence[float]) -> Vec:
    return tuple(dot(row, x) for row in a)


def mat_transpose(a: Mat) -> Mat:
    if not a:
        return ()
    n = len(a[0])
    return tuple(tuple(a[i][j] for i in range(len(a))) for j in range(n))


def mat_add(a: Mat, b: Mat) -> Mat:
    return tuple(
        tuple(x + y for x, y in zip(ra, rb, strict=True))
        for ra, rb in zip(a, b, strict=True)
    )


def mat_scale(a: Mat, s: float) -> Mat:
    return tuple(tuple(s * x for x in row) for row in a)


def identity(n: int) -> Mat:
    return tuple(tuple(1.0 if i == j else 0.0 for j in range(n)) for i in range(n))


def mat_mul(a: Mat, b: Mat) -> Mat:
    bt = mat_transpose(b)
    return tuple(tuple(dot(row, col) for col in bt) for row in a)


def solve_linear(a: Mat, b: Sequence[float], *, tol: float = 1e-14) -> Vec:
    """Solve ``A x = b`` with partial pivoting (small dense systems)."""
    n = len(b)
    if len(a) != n or any(len(row) != n for row in a):
        raise ValueError("A must be square and match b")
    aug = [list(row) + [float(bi)] for row, bi in zip(a, b, strict=True)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < tol:
            raise ValueError("singular matrix")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= div
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) < tol:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    return tuple(aug[i][n] for i in range(n))


def _check_convergence(
    grad: Vec, *, grad_tol: float, x: Vec, x_prev: Vec | None, x_tol: float
) -> bool:
    if norm2(grad) <= grad_tol:
        return True
    if x_prev is not None and norm2(vec_sub(x, x_prev)) <= x_tol:
        return True
    return False


def newtons_method(
    objective: Objective,
    gradient: Gradient,
    hessian: Hessian,
    x0: Sequence[float],
    *,
    grad_tol: float = 1e-8,
    x_tol: float = 1e-10,
    max_iter: int = 100,
) -> OptimizeResult:
    """Pure Newton steps: ``x_{k+1} = x_k - H(x_k)^{-1} \\nabla f(x_k)``."""
    x = _as_vec(x0)
    iterations = 0
    x_prev: Vec | None = None
    while iterations < max_iter:
        g = gradient(x)
        if _check_convergence(g, grad_tol=grad_tol, x=x, x_prev=x_prev, x_tol=x_tol):
            return OptimizeResult(
                x=x, value=objective(x), iterations=iterations, converged=True
            )
        h = hessian(x)
        try:
            step = vec_scale(solve_linear(h, g), -1.0)
        except ValueError:
            return OptimizeResult(
                x=x, value=objective(x), iterations=iterations, converged=False
            )
        x_prev = x
        x = vec_add(x, step)
        iterations += 1
    g = gradient(x)
    converged = norm2(g) <= grad_tol
    return OptimizeResult(
        x=x, value=objective(x), iterations=iterations, converged=converged
    )


def damped_newton_method(
    objective: Objective,
    gradient: Gradient,
    hessian: Hessian,
    x0: Sequence[float],
    *,
    grad_tol: float = 1e-8,
    x_tol: float = 1e-10,
    max_iter: int = 100,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
) -> OptimizeResult:
    """Newton direction with backtracking Armijo line search."""
    x = _as_vec(x0)
    iterations = 0
    x_prev: Vec | None = None
    fx = objective(x)
    while iterations < max_iter:
        g = gradient(x)
        if _check_convergence(g, grad_tol=grad_tol, x=x, x_prev=x_prev, x_tol=x_tol):
            return OptimizeResult(x=x, value=fx, iterations=iterations, converged=True)
        h = hessian(x)
        try:
            direction = vec_scale(solve_linear(h, g), -1.0)
        except ValueError:
            return OptimizeResult(x=x, value=fx, iterations=iterations, converged=False)
        phi_prime0 = dot(g, direction)
        if phi_prime0 >= 0:
            direction = vec_scale(g, -1.0)
            phi_prime0 = dot(g, direction)

        def phi(alpha: float) -> float:
            return objective(vec_add(x, vec_scale(direction, alpha)))

        ls = backtracking_armijo(phi, phi_prime0, alpha0=alpha0, c1=c1, rho=rho)
        x_prev = x
        x = vec_add(x, vec_scale(direction, ls.alpha))
        fx = objective(x)
        iterations += 1
    g = gradient(x)
    converged = norm2(g) <= grad_tol
    return OptimizeResult(x=x, value=fx, iterations=iterations, converged=converged)


def conjugate_gradient(
    objective: Objective,
    gradient: Gradient,
    x0: Sequence[float],
    *,
    grad_tol: float = 1e-8,
    max_iter: int = 1000,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    restart_every: int | None = None,
) -> OptimizeResult:
    """Nonlinear Fletcher–Reeves conjugate gradient with Armijo line search."""
    x = _as_vec(x0)
    g = gradient(x)
    direction = vec_scale(g, -1.0)
    fx = objective(x)
    n = len(x)
    restart = restart_every if restart_every is not None else n
    iterations = 0

    while iterations < max_iter:
        if norm2(g) <= grad_tol:
            return OptimizeResult(x=x, value=fx, iterations=iterations, converged=True)
        phi_prime0 = dot(g, direction)
        if phi_prime0 >= 0:
            direction = vec_scale(g, -1.0)
            phi_prime0 = dot(g, direction)

        def phi(alpha: float) -> float:
            return objective(vec_add(x, vec_scale(direction, alpha)))

        ls = backtracking_armijo(phi, phi_prime0, alpha0=alpha0, c1=c1, rho=rho)
        x = vec_add(x, vec_scale(direction, ls.alpha))
        g_new = gradient(x)
        fx = objective(x)
        iterations += 1
        if norm2(g_new) <= grad_tol:
            return OptimizeResult(x=x, value=fx, iterations=iterations, converged=True)
        if iterations % restart == 0:
            direction = vec_scale(g_new, -1.0)
        else:
            beta = dot(g_new, g_new) / max(dot(g, g), 1e-30)
            direction = vec_add(vec_scale(g_new, -1.0), vec_scale(direction, beta))
        g = g_new

    converged = norm2(g) <= grad_tol
    return OptimizeResult(x=x, value=fx, iterations=iterations, converged=converged)


def bfgs(
    objective: Objective,
    gradient: Gradient,
    x0: Sequence[float],
    *,
    grad_tol: float = 1e-8,
    max_iter: int = 200,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
) -> OptimizeResult:
    """BFGS quasi-Newton method with inverse Hessian approximation."""
    x = _as_vec(x0)
    n = len(x)
    g = gradient(x)
    b = identity(n)
    fx = objective(x)
    iterations = 0

    while iterations < max_iter:
        if norm2(g) <= grad_tol:
            return OptimizeResult(x=x, value=fx, iterations=iterations, converged=True)
        direction = vec_scale(mat_vec(b, vec_scale(g, -1.0)), 1.0)
        phi_prime0 = dot(g, direction)
        if phi_prime0 >= 0:
            direction = vec_scale(g, -1.0)
            phi_prime0 = dot(g, direction)

        def phi(alpha: float) -> float:
            return objective(vec_add(x, vec_scale(direction, alpha)))

        ls = backtracking_armijo(phi, phi_prime0, alpha0=alpha0, c1=c1, rho=rho)
        s = vec_scale(direction, ls.alpha)
        x_new = vec_add(x, s)
        g_new = gradient(x_new)
        y = vec_sub(g_new, g)
        sy = dot(s, y)
        if sy > 1e-12:
            rho_k = 1.0 / sy
            i_mat = identity(n)
            left = tuple(
                tuple(i_mat[i][j] - rho_k * y[i] * s[j] for j in range(n))
                for i in range(n)
            )
            right = tuple(
                tuple(i_mat[i][j] - rho_k * s[i] * y[j] for j in range(n))
                for i in range(n)
            )
            outer_s = tuple(
                tuple(rho_k * s[i] * s[j] for j in range(n)) for i in range(n)
            )
            b = mat_add(mat_mul(mat_mul(left, b), right), outer_s)
        x, g, fx = x_new, g_new, objective(x_new)
        iterations += 1

    converged = norm2(g) <= grad_tol
    return OptimizeResult(x=x, value=fx, iterations=iterations, converged=converged)


def quasi_newton_bfgs(
    objective: Objective,
    gradient: Gradient,
    x0: Sequence[float],
    *,
    grad_tol: float = 1e-8,
    max_iter: int = 200,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
) -> OptimizeResult:
    """Alias for :func:`bfgs` (BFGS is a quasi-Newton method)."""
    return bfgs(
        objective,
        gradient,
        x0,
        grad_tol=grad_tol,
        max_iter=max_iter,
        alpha0=alpha0,
        c1=c1,
        rho=rho,
    )


def _lbfgs_direction(
    g: Vec,
    s_hist: list[Vec],
    y_hist: list[Vec],
    rho_hist: list[float],
) -> Vec:
    q = vec_scale(g, -1.0)
    alphas: list[float] = []
    for s, y, rho in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
        alpha = rho * dot(s, q)
        alphas.append(alpha)
        q = vec_sub(q, vec_scale(s, alpha))
    if s_hist:
        gamma = dot(s_hist[-1], y_hist[-1]) / max(dot(y_hist[-1], y_hist[-1]), 1e-30)
        r = vec_scale(q, gamma)
    else:
        r = q
    for s, y, rho, alpha in zip(s_hist, y_hist, rho_hist, reversed(alphas)):
        beta = rho * dot(y, r)
        r = vec_add(r, vec_scale(s, alpha - beta))
    return r


def lbfgs(
    objective: Objective,
    gradient: Gradient,
    x0: Sequence[float],
    *,
    memory: int = 10,
    grad_tol: float = 1e-8,
    max_iter: int = 200,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
) -> OptimizeResult:
    """Limited-memory BFGS (L-BFGS)."""
    x = _as_vec(x0)
    g = gradient(x)
    fx = objective(x)
    s_hist: list[Vec] = []
    y_hist: list[Vec] = []
    rho_hist: list[float] = []
    iterations = 0

    while iterations < max_iter:
        if norm2(g) <= grad_tol:
            return OptimizeResult(x=x, value=fx, iterations=iterations, converged=True)
        direction = _lbfgs_direction(g, s_hist, y_hist, rho_hist)
        phi_prime0 = dot(g, direction)
        if phi_prime0 >= 0:
            direction = vec_scale(g, -1.0)
            phi_prime0 = dot(g, direction)

        def phi(alpha: float) -> float:
            return objective(vec_add(x, vec_scale(direction, alpha)))

        ls = backtracking_armijo(phi, phi_prime0, alpha0=alpha0, c1=c1, rho=rho)
        s = vec_scale(direction, ls.alpha)
        x_new = vec_add(x, s)
        g_new = gradient(x_new)
        y = vec_sub(g_new, g)
        sy = dot(s, y)
        if sy > 1e-12:
            s_hist.append(s)
            y_hist.append(y)
            rho_hist.append(1.0 / sy)
            if len(s_hist) > memory:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
        x, g, fx = x_new, g_new, objective(x_new)
        iterations += 1

    converged = norm2(g) <= grad_tol
    return OptimizeResult(x=x, value=fx, iterations=iterations, converged=converged)


def gauss_newton(
    residual: Residual,
    jacobian: Jacobian,
    x0: Sequence[float],
    *,
    grad_tol: float = 1e-8,
    max_iter: int = 100,
    damping: float = 1e-8,
) -> OptimizeResult:
    """Gauss–Newton for ``\\min_x \\tfrac12 \\|r(x)\\|^2``."""
    x = _as_vec(x0)
    iterations = 0
    while iterations < max_iter:
        r = residual(x)
        j = jacobian(x)
        jt = mat_transpose(j)
        jtj = mat_mul(jt, j)
        jtj = mat_add(jtj, mat_scale(identity(len(x)), damping))
        g = mat_vec(jt, r)
        if norm2(g) <= grad_tol:
            return OptimizeResult(
                x=x,
                value=0.5 * dot(r, r),
                iterations=iterations,
                converged=True,
            )
        try:
            step = vec_scale(solve_linear(jtj, g), -1.0)
        except ValueError:
            return OptimizeResult(
                x=x,
                value=0.5 * dot(r, r),
                iterations=iterations,
                converged=False,
            )
        x = vec_add(x, step)
        iterations += 1
    r = residual(x)
    g = mat_vec(mat_transpose(jacobian(x)), r)
    converged = norm2(g) <= grad_tol
    return OptimizeResult(
        x=x,
        value=0.5 * dot(r, r),
        iterations=iterations,
        converged=converged,
    )


def levenberg_marquardt(
    residual: Residual,
    jacobian: Jacobian,
    x0: Sequence[float],
    *,
    grad_tol: float = 1e-8,
    max_iter: int = 100,
    lambda0: float = 1e-2,
    lambda_up: float = 10.0,
    lambda_down: float = 0.1,
) -> OptimizeResult:
    """Levenberg–Marquardt for nonlinear least squares."""
    x = _as_vec(x0)
    lam = lambda0
    fx = 0.5 * dot(residual(x), residual(x))
    iterations = 0

    while iterations < max_iter:
        r = residual(x)
        j = jacobian(x)
        jt = mat_transpose(j)
        g = mat_vec(jt, r)
        if norm2(g) <= grad_tol:
            return OptimizeResult(x=x, value=fx, iterations=iterations, converged=True)
        jtj = mat_mul(jt, j)
        accepted = False
        for _ in range(12):
            a = mat_add(jtj, mat_scale(identity(len(x)), lam))
            try:
                step = vec_scale(solve_linear(a, g), -1.0)
            except ValueError:
                lam *= lambda_up
                continue
            x_trial = vec_add(x, step)
            f_trial = 0.5 * dot(residual(x_trial), residual(x_trial))
            if f_trial < fx:
                x = x_trial
                fx = f_trial
                lam = max(lam * lambda_down, 1e-16)
                accepted = True
                break
            lam *= lambda_up
        if not accepted:
            return OptimizeResult(x=x, value=fx, iterations=iterations, converged=False)
        iterations += 1

    r = residual(x)
    g = mat_vec(mat_transpose(jacobian(x)), r)
    converged = norm2(g) <= grad_tol
    return OptimizeResult(x=x, value=fx, iterations=iterations, converged=converged)
