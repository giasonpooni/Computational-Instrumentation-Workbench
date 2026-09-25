"""Every argument check of the free-energy mathematics refuses the argument it names.

An AST mutation probe dropped each ``if ...: raise`` of ``free_energy_math``
in turn; 27 of 38 survived. Each case here calls the public function with
exactly one invalid argument and expects the refusal that names it. The
guards on computed matrices and results, which no admitted input reaches,
are listed in the mutation gate rather than exercised here.
"""
from copy import deepcopy

import pytest

from ciw import free_energy_math as fem

PROBLEM = {"coordinate_system": "normalized_dimensionless", "prior_mean": [0.0, 0.0],
           "prior_covariance": [[1.0, 0.0], [0.0, 1.0]], "observation_matrix": [[1.0, 0.2], [0.0, 1.0]],
           "observations": [1.0, 2.0], "noise_covariance": [[1.0, 0.1], [0.1, 1.0]]}
HELD_OUT = {"coordinate_system": "normalized_dimensionless", "observation_matrix": [[1.0, 1.0], [1.0, -1.0]],
            "observations": [0.5, 1.5], "noise_covariance": [[0.25, 0.0], [0.0, 0.25]]}
RAW = dict(prior_mean=[0.0, 0.0], prior_covariance=[[1.0, 0.0], [0.0, 1.0]], observation_matrix=[[1.0, 0.0], [0.0, 1.0]],
           observations=[1.0, 2.0], noise_covariance=[[1.0, 0.0], [0.0, 1.0]], latent_scales=[1.0, 1.0],
           observation_scales=[1.0, 1.0])


def problem(**changes):
    value = deepcopy(PROBLEM)
    value.update(changes)
    return value


def normalize(**changes):
    return fem.normalize_problem(**{**deepcopy(RAW), **changes})


CASES = {
    "problem given as a list of its names": (lambda: fem.gaussian_reference(list(PROBLEM)), "declared normalized Gaussian fields"),
    "problem with an extra field": (lambda: fem.gaussian_reference(problem(extra=1)), "declared normalized Gaussian fields"),
    "alpha given as text": (lambda: fem.mean_iteration_matrix(PROBLEM, "1"), "alpha requires a finite real number"),
    "alpha infinite": (lambda: fem.mean_iteration_matrix(PROBLEM, float("inf")), "alpha requires a finite real number"),
    "alpha zero": (lambda: fem.mean_iteration_matrix(PROBLEM, 0), r"alpha must lie in \(0,1e6\]"),
    "prior mean with three entries": (lambda: fem.gaussian_reference(problem(prior_mean=[0.0, 0.0, 0.0])), "prior_mean has an unsupported shape or magnitude"),
    "prior mean with a NaN": (lambda: fem.gaussian_reference(problem(prior_mean=[float("nan"), 0.0])), "prior_mean requires a finite real number"),
    "prior mean beyond its bound": (lambda: fem.gaussian_reference(problem(prior_mean=[1e7, 0.0])), "prior_mean has an unsupported shape or magnitude"),
    "prior covariance not exactly symmetric": (lambda: fem.gaussian_reference(problem(prior_covariance=[[1.0, 0.1], [0.0, 1.0]])), "prior_covariance must already be exactly symmetric"),
    "prior covariance indefinite": (lambda: fem.gaussian_reference(problem(prior_covariance=[[1.0, 0.0], [0.0, -1.0]])), "positive-definite or conditioning bound"),
    "prior covariance ill conditioned": (lambda: fem.gaussian_reference(problem(prior_covariance=[[1e6, 0.0], [0.0, 1e-3]])), "positive-definite or conditioning bound"),
    "latent scale below its floor": (lambda: normalize(latent_scales=[1e-13, 1.0]), r"Normalization scales must lie in \[1e-12,1e12\]"),
    "observation scale below its floor": (lambda: normalize(observation_scales=[1e-13, 1.0]), r"Normalization scales must lie in \[1e-12,1e12\]"),
    "raw prior covariance not symmetric": (lambda: normalize(prior_covariance=[[1.0, 0.1], [0.0, 1.0]]), "Raw covariance declarations must already be exactly symmetric"),
    "raw noise covariance not symmetric": (lambda: normalize(noise_covariance=[[1.0, 0.1], [0.0, 1.0]]), "Raw covariance declarations must already be exactly symmetric"),
    "coverage below one half": (lambda: fem.predict_held_out(HELD_OUT, [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], coverage=0.3), r"Nominal coverage must lie in \[0.5,0.999\]"),
    "replicates given as a float": (lambda: fem.simulate_ensemble(PROBLEM, HELD_OUT, seed=0, replicates=2.0), r"replicates must lie in \[1,512\]"),
    "no retained replicates": (lambda: fem.evaluate_ensemble(PROBLEM, [], [], HELD_OUT, []), "1..512 retained prior-predictive replicates"),
}


@pytest.mark.parametrize("call, message", CASES.values(), ids=list(CASES))
def test_an_invalid_argument_is_refused_by_its_own_check(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_valid_arguments_still_evaluate():
    assert fem.gaussian_reference(PROBLEM) is not None
    assert fem.normalize_problem(**RAW) is not None
    assert fem.mean_iteration_matrix(PROBLEM, 0.5) is not None
