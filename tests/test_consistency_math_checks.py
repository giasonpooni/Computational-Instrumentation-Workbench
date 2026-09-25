"""Every argument check of the consistency special functions refuses the argument it names.

An AST mutation probe dropped each ``if ...: raise`` of ``consistency_math``
in turn; 14 of 16 survived because the suite only called the functions with
valid arguments.
"""
import pytest

from ciw import consistency_math as consistency

CASES = {
    "gamma shape given as text": (lambda: consistency.regularized_gamma_p("1", 1.0), "shape must be a finite positive number"),
    "gamma shape infinite": (lambda: consistency.regularized_gamma_p(float("inf"), 1.0), "shape must be a finite positive number"),
    "gamma shape zero": (lambda: consistency.regularized_gamma_p(0, 1.0), "shape must be a finite positive number"),
    "gamma argument given as text": (lambda: consistency.regularized_gamma_p(1.0, "1"), "finite nonnegative number"),
    "gamma argument infinite": (lambda: consistency.regularized_gamma_p(1.0, float("inf")), "finite nonnegative number"),
    "gamma argument negative": (lambda: consistency.regularized_gamma_p(1.0, -1.0), "finite nonnegative number"),
    "beta argument given as text": (lambda: consistency.regularized_beta(1.0, 1.0, "0.5"), "unit interval"),
    "beta argument infinite": (lambda: consistency.regularized_beta(1.0, 1.0, float("inf")), "unit interval"),
    "beta argument above one": (lambda: consistency.regularized_beta(1.0, 1.0, 1.5), "unit interval"),
    "quantile probability given as text": (lambda: consistency.chi_square_quantile("0.5", 2.0), "strictly inside the unit interval"),
    "quantile probability at one": (lambda: consistency.chi_square_quantile(1.0, 2.0), "strictly inside the unit interval"),
    "binomial count given as text": (lambda: consistency.binomial_interval("1", 3, 0.95), "count and total must be integers"),
    "binomial total given as text": (lambda: consistency.binomial_interval(1, "3", 0.95), "count and total must be integers"),
    "binomial count above total": (lambda: consistency.binomial_interval(5, 3, 0.95), "count and total must be integers"),
    "binomial total zero": (lambda: consistency.binomial_interval(0, 0, 0.95), "count and total must be integers"),
}


@pytest.mark.parametrize("call, message", CASES.values(), ids=list(CASES))
def test_an_invalid_argument_is_refused_by_its_own_check(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_valid_arguments_still_evaluate():
    assert consistency.regularized_gamma_p(1.0, 0.0) == 0.0
    assert consistency.regularized_beta(1.0, 1.0, 0.0) == 0.0
    assert 0 < consistency.chi_square_quantile(0.5, 2.0) < 2
    lower, upper = consistency.binomial_interval(1, 3, 0.95)
    assert 0 <= lower < upper <= 1
