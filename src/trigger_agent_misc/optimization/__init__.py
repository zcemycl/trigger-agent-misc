"""Optimization utilities for ML experiments."""

from trigger_agent_misc.optimization.line_search import (
    LineSearchResult,
    armijo_sufficient_decrease,
    backtracking_armijo,
    backtracking_line_search,
    exact_line_search,
    goldstein_conditions,
    line_search_goldstein,
    line_search_strong_wolfe,
    line_search_wolfe,
    strong_wolfe_curvature,
    wolfe_curvature,
)
from trigger_agent_misc.optimization.second_order import (
    OptimizeResult,
    bfgs,
    conjugate_gradient,
    damped_newton_method,
    gauss_newton,
    lbfgs,
    levenberg_marquardt,
    newtons_method,
    quasi_newton_bfgs,
)

__all__ = [
    "LineSearchResult",
    "OptimizeResult",
    "armijo_sufficient_decrease",
    "backtracking_armijo",
    "backtracking_line_search",
    "bfgs",
    "conjugate_gradient",
    "damped_newton_method",
    "exact_line_search",
    "gauss_newton",
    "goldstein_conditions",
    "lbfgs",
    "levenberg_marquardt",
    "line_search_goldstein",
    "line_search_strong_wolfe",
    "line_search_wolfe",
    "newtons_method",
    "quasi_newton_bfgs",
    "strong_wolfe_curvature",
    "wolfe_curvature",
]
