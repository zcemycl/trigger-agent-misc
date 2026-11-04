"""Optimization utilities for ML experiments."""

from trigger_agent_misc.optimization.gradient_method import (
    GradientMethodResult,
    gradient_descent,
    heavy_ball_method,
    minibatch_sgd,
    momentum,
    nesterov_accelerated_gradient,
    stochastic_gradient_descent,
)
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

__all__ = [
    "GradientMethodResult",
    "gradient_descent",
    "heavy_ball_method",
    "minibatch_sgd",
    "momentum",
    "nesterov_accelerated_gradient",
    "stochastic_gradient_descent",
    "LineSearchResult",
    "armijo_sufficient_decrease",
    "backtracking_armijo",
    "backtracking_line_search",
    "exact_line_search",
    "goldstein_conditions",
    "line_search_goldstein",
    "line_search_strong_wolfe",
    "line_search_wolfe",
    "strong_wolfe_curvature",
    "wolfe_curvature",
]
