"""Special functions and bands behind the uncertainty validation operation."""

import math

import numpy as np
import pytest

from ciw import consistency_math as consistency


def _closed_form_cdf(x, dof):
    """Independent oracle: erf for one degree of freedom, the Poisson sum for even ones."""
    if dof == 1:
        return math.erf(math.sqrt(x / 2))
    half = dof // 2
    return 1 - math.exp(-x / 2) * sum((x / 2) ** i / math.factorial(i) for i in range(half))


@pytest.mark.parametrize("p,dof,expected", [
    (0.95, 1, 3.841458820694124),
    (0.95, 2, -2 * math.log(0.05)),
    (0.05, 10, 3.940299136370042),
    (0.95, 10, 18.307038053275146),
    (0.975, 10, 20.483177350807657),
])
def test_chi_square_quantiles_match_reference_values(p, dof, expected):
    assert consistency.chi_square_quantile(p, dof) == pytest.approx(expected, rel=1e-9)
    assert consistency.chi_square_cdf(expected, dof) == pytest.approx(p, abs=1e-9)


@pytest.mark.parametrize("dof", [1, 2, 4, 10, 40, 128])
def test_chi_square_distribution_matches_closed_forms(dof):
    for x in (0.05, 0.5, 2.0, 7.5, 20.0, 60.0, 150.0):
        assert consistency.chi_square_cdf(x, dof) == pytest.approx(_closed_form_cdf(x, dof), abs=1e-12)
    for p in (0.001, 0.025, 0.5, 0.975, 0.999):
        assert _closed_form_cdf(consistency.chi_square_quantile(p, dof), dof) == pytest.approx(p, abs=1e-10)


@pytest.mark.parametrize("dof", [1, 3, 30, 128, 2048])
@pytest.mark.parametrize("p", [0.001, 0.025, 0.5, 0.975, 0.999])
def test_chi_square_quantile_inverts_the_distribution(dof, p):
    assert consistency.chi_square_cdf(consistency.chi_square_quantile(p, dof), dof) == pytest.approx(p, abs=1e-10)


def test_incomplete_gamma_and_beta_agree_with_closed_forms():
    for x in (0.1, 1.0, 3.5, 20.0):
        assert consistency.regularized_gamma_p(1, x) == pytest.approx(1 - math.exp(-x), rel=1e-12)
    for a in (0.5, 1.0, 4.0, 25.0):
        assert consistency.regularized_beta(a, a, 0.5) == pytest.approx(0.5, rel=1e-12)
    assert consistency.regularized_beta(2, 3, 0.4) == pytest.approx(0.5248, abs=1e-4)
    assert consistency.regularized_beta(1, 1, 0.3) == pytest.approx(0.3, rel=1e-12)


def test_normal_quantile_is_the_two_sided_standard_normal_bound():
    assert consistency.normal_quantile(0.95) == pytest.approx(1.959963984540054, rel=1e-9)
    assert consistency.normal_quantile(0.99) == pytest.approx(2.5758293035489004, rel=1e-9)


@pytest.mark.parametrize("count,total,lower,upper", [
    (0, 10, 0.0, 0.3085),
    (10, 10, 0.6915, 1.0),
    (5, 10, 0.1871, 0.8129),
    (95, 100, 0.8872, 0.9836),
])
def test_clopper_pearson_intervals_match_reference_values(count, total, lower, upper):
    low, high = consistency.binomial_interval(count, total, 0.95)
    assert low == pytest.approx(lower, abs=1e-4)
    assert high == pytest.approx(upper, abs=1e-4)


def test_domain_errors_are_refused_not_repaired():
    with pytest.raises(ValueError):
        consistency.chi_square_quantile(1.0, 3)
    with pytest.raises(ValueError):
        consistency.chi_square_quantile(0.5, 0)
    with pytest.raises(ValueError):
        consistency.binomial_interval(11, 10, 0.95)
    with pytest.raises(ValueError):
        consistency.regularized_beta(1, 1, 1.5)


def test_statuses_separate_too_small_too_large_and_bias():
    # Unit errors with alternating signs: every normalized square is exactly n,
    # the mean error is exactly zero, so each classifier's answer is known.
    samples = 200
    errors = np.array([[1.0, -1.0] if index % 2 else [-1.0, 1.0] for index in range(samples)])
    identity = [[[1.0, 0.0], [0.0, 1.0]]] * samples
    quarter = [[[0.25, 0.0], [0.0, 0.25]]] * samples
    quadruple = [[[4.0, 0.0], [0.0, 4.0]]] * samples
    consistent = consistency.mean_square_status(consistency.normalized_squares(errors, identity), 2, 0.95)
    assert consistent["status"] == consistency.CONSISTENT and consistent["dof"] == 400
    assert consistent["mean"] == pytest.approx(2.0)
    small = consistency.mean_square_status(consistency.normalized_squares(errors, quarter), 2, 0.95)
    large = consistency.mean_square_status(consistency.normalized_squares(errors, quadruple), 2, 0.95)
    assert small["status"] == consistency.TOO_SMALL and small["mean"] == pytest.approx(8.0)
    assert large["status"] == consistency.TOO_LARGE and large["mean"] == pytest.approx(0.5)
    assert all(item["status"] == consistency.UNDER for item in consistency.coverage_status(errors, quarter, 0.95)["components"])
    # Every unit error lies inside 1.96 sigma, and 200 of 200 excludes the nominal 0.95.
    assert all(item["status"] == consistency.OVER for item in consistency.coverage_status(errors, identity, 0.95)["components"])
    assert all(item["status"] == consistency.UNBIASED for item in consistency.bias_status(errors, identity, 0.95)["components"])
    assert all(item["status"] == consistency.BIASED for item in consistency.bias_status(errors + 1.0, identity, 0.95)["components"])


def test_overflowing_normalized_squares_are_refused_rather_than_propagated():
    huge = np.array([[1e200, 1e200]])
    tiny = [[[1e-200, 0.0], [0.0, 1e-200]]]
    with pytest.raises(ValueError, match="overflow"):
        consistency.normalized_squares(huge, tiny)
    assert consistency.normalized_squares(np.array([[1.0, 0.0]]), [[[1.0, 0.0], [0.0, 1.0]]]) == [1.0]
