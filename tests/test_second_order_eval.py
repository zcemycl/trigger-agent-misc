"""Smoke tests for second-order evaluation harness."""

from evals.second_order_eval import eval_newton_quadratic, run_all


def test_newton_eval_passes() -> None:
    metrics = eval_newton_quadratic()
    assert metrics["converged"] is True
    assert metrics["x0_near_one"] is True
    assert metrics["x1_near_two"] is True


def test_run_all_includes_second_order_benchmarks() -> None:
    results = run_all()
    expected = {
        "newton",
        "damped_newton",
        "conjugate_gradient",
        "bfgs",
        "quasi_newton_bfgs",
        "lbfgs",
        "gauss_newton",
        "levenberg_marquardt",
    }
    assert set(results) == expected
    assert results["gauss_newton"]["converged"] is True
    assert results["levenberg_marquardt"]["reduced_objective"] is True
