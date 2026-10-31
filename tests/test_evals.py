"""Smoke tests for evaluation harnesses under ``evals/``."""

from evals.line_search_eval import eval_quadratic_along_gradient, run_all


def test_quadratic_eval_passes() -> None:
    metrics = eval_quadratic_along_gradient()
    assert metrics["alpha_near_optimum"] is True
    assert metrics["value_near_optimum"] is True


def test_run_all_includes_benchmarks() -> None:
    results = run_all()
    expected = {
        "exact_quadratic",
        "backtracking_armijo",
        "armijo_condition",
        "wolfe",
        "strong_wolfe",
        "goldstein",
        "quadratic",
        "rosenbrock_origin",
    }
    assert set(results) == expected
    assert results["rosenbrock_origin"]["decreased_objective"] is True
    assert results["exact_quadratic"]["alpha_near_optimum"] is True
    assert results["armijo_condition"]["armijo_at_good"] is True
    assert results["wolfe"]["wolfe_curvature_holds"] is True
    assert results["strong_wolfe"]["strong_wolfe_holds"] is True
    assert results["goldstein"]["goldstein_holds"] is True
