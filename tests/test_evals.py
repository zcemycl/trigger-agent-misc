"""Smoke tests for evaluation harnesses under ``evals/``."""

from evals.line_search_eval import eval_quadratic_along_gradient, run_all


def test_quadratic_eval_passes() -> None:
    metrics = eval_quadratic_along_gradient()
    assert metrics["alpha_near_optimum"] is True
    assert metrics["value_near_optimum"] is True


def test_run_all_includes_benchmarks() -> None:
    results = run_all()
    assert set(results) == {"quadratic", "rosenbrock_origin"}
    assert results["rosenbrock_origin"]["decreased_objective"] is True
