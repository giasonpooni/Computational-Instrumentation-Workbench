"""Scientific anchors for normalized Gaussian VI and prior-predictive checks."""
from copy import deepcopy
import math

import numpy as np
import pytest

from ciw import free_energy_math as fem


def problem():
    return {"coordinate_system":"normalized_dimensionless", "prior_mean":[0,0],
            "prior_covariance":[[1,0],[0,1]], "observation_matrix":[[1,0],[0,1]],
            "observations":[2,-1], "noise_covariance":[[1,0],[0,1]]}


def held_out():
    return {"coordinate_system":"normalized_dimensionless", "observation_matrix":[[1,1],[1,-1]],
            "observations":[0.5,1.5], "noise_covariance":[[0.25,0],[0,0.25]]}


def fit(value=None, **changes):
    settings = dict(initial_mean=[0,0], initial_covariance=[[1,0],[0,1]], alpha=0.2, beta=0.2,
                    max_iterations=200, gradient_tolerance=1e-9, precision_tolerance=1e-9)
    settings.update(changes)
    return fem.variational_fit(problem() if value is None else value, **settings)


def test_observation_reference_matches_exact_scalar_conditionals(monkeypatch):
    monkeypatch.setattr(fem, "_information", lambda *_: pytest.fail("Reference reused information solver"))
    reference = fem.gaussian_reference(problem())
    assert reference["method"] == "observation_space_conditioning"
    np.testing.assert_allclose(reference["mean"], [1,-0.5], rtol=0, atol=1e-15)
    np.testing.assert_allclose(reference["covariance"], 0.5*np.eye(2), rtol=0, atol=1e-15)
    np.testing.assert_allclose(reference["innovation_covariance"], 2*np.eye(2))
    assert reference["log_evidence"] == pytest.approx(-math.log(4*math.pi)-1.25, abs=1e-14)


def test_correlated_prior_and_noise_observation_conditioning_matches_joint_gaussian():
    value = problem()
    value.update(prior_mean=[0.3,-0.2], prior_covariance=[[2,0.5],[0.5,1]],
                 observation_matrix=[[1,0.4],[-0.2,0.8]], noise_covariance=[[0.6,0.15],[0.15,0.4]])
    reference = fem.gaussian_reference(value)
    m0,c0,g,sigma = (np.array(value[key]) for key in ("prior_mean","prior_covariance","observation_matrix","noise_covariance"))
    joint = np.block([[c0,c0@g.T],[g@c0,g@c0@g.T+sigma]])
    gain = np.linalg.solve(joint[2:,2:],joint[2:,:2]).T
    expected_mean = m0 + gain @ (np.array(value["observations"])-g@m0)
    expected_cov = joint[:2,:2]-gain@joint[2:,:2]
    np.testing.assert_allclose(reference["mean"], expected_mean, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(reference["covariance"], expected_cov, rtol=1e-13, atol=1e-14)
    information = fem.information_system(value)
    np.testing.assert_allclose(np.linalg.solve(information["precision"],information["information_vector"]), reference["mean"], rtol=1e-13)


def test_free_energy_keeps_gaussian_normalizations_and_evidence_identity():
    value = problem()
    initial = fem.free_energy(value,[0,0],np.eye(2))
    assert initial["free_energy"] == pytest.approx(math.log(2*math.pi)+3.5, abs=1e-14)
    terms = initial["terms"]
    assert terms["likelihood_normalization"] == pytest.approx(math.log(2*math.pi))
    assert terms["prior_normalization"] == pytest.approx(math.log(2*math.pi))
    assert terms["variational_entropy"] == pytest.approx(math.log(2*math.pi)+1)
    assert terms["expected_negative_log_likelihood"] + terms["prior_cross_entropy"] - terms["variational_entropy"] == initial["free_energy"]
    reference = fem.gaussian_reference(value)
    optimum = fem.free_energy(value,reference["mean"],reference["covariance"])
    assert optimum["free_energy"] == pytest.approx(-reference["log_evidence"], abs=1e-14)
    kl = fem.gaussian_kl([0,0],np.eye(2),reference["mean"],reference["covariance"])
    assert kl == pytest.approx(2.25-math.log(2), abs=1e-14)
    assert initial["free_energy"] + reference["log_evidence"] == pytest.approx(kl, abs=1e-14)


def test_stable_iteration_changes_both_mean_and_covariance_and_converges():
    result = fit()
    assert result["status"] == "converged" and result["converged"]
    assert 2 < result["iterations"] < 200
    first = result["trace"][1]
    np.testing.assert_allclose(first["mean"],[0.4,-0.2], atol=1e-15)
    np.testing.assert_allclose(first["precision"],1.2*np.eye(2), atol=1e-15)
    np.testing.assert_allclose(first["covariance"],(5/6)*np.eye(2), atol=1e-15)
    assert first["mean"] != result["reference"]["mean"]
    assert first["covariance"] != result["reference"]["covariance"]
    np.testing.assert_allclose(result["mean"],[1,-0.5], atol=1e-9)
    np.testing.assert_allclose(result["covariance"],0.5*np.eye(2), atol=1e-9)
    assert result["stability"]["asymptotically_stable"]
    assert result["stability"]["spectral_radius"] == pytest.approx(0.6)
    assert result["stability"]["stable_step_size_upper_bound"] == pytest.approx(1)
    assert len(result["trace"]) == result["iterations"]+1
    energies = [row["free_energy"] for row in result["trace"]]
    assert all(b <= a+2e-14 for a,b in zip(energies,energies[1:]))
    assert all(row["kl_to_reference"] >= 0 for row in result["trace"])
    assert max(abs(row["free_energy_identity_residual"]) for row in result["trace"]) < 1e-13


def test_full_precision_relaxation_matches_closed_recurrence_and_natural_gradient():
    value = problem()
    value.update(observation_matrix=[[1,0.4],[0.2,1.2]], noise_covariance=[[1,0.2],[0.2,0.8]])
    initial = np.array([[2,0.7],[0.7,1]])
    result = fit(value, initial_covariance=initial.tolist(), max_iterations=7, beta=0.3, alpha=0.1)
    target = np.array(result["information_system"]["precision"])
    q0 = np.linalg.solve(initial,np.eye(2))
    for row in result["trace"]:
        expected = target + 0.7**row["iteration"]*(q0-target)
        np.testing.assert_allclose(row["precision"],expected,rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(np.array(row["precision"])@np.array(row["covariance"]),np.eye(2),atol=1e-14)
        assert np.linalg.eigvalsh(row["covariance"])[0] > 0
    direction = np.array([[0.2,0.1],[0.1,-0.1]])
    epsilon = 1e-5
    plus = fem.free_energy(value,[0.3,-0.1],np.linalg.solve(q0+epsilon*direction,np.eye(2)))["free_energy"]
    minus = fem.free_energy(value,[0.3,-0.1],np.linalg.solve(q0-epsilon*direction,np.eye(2)))["free_energy"]
    fisher_pairing = 0.5*np.trace(initial@(q0-target)@initial@direction)
    assert (plus-minus)/(2*epsilon) == pytest.approx(fisher_pairing, rel=1e-8, abs=1e-8)


def test_iteration_matrix_describes_actual_mean_error_and_plsr_stability_object():
    result = fit(max_iterations=3)
    matrix = np.array(fem.mean_iteration_matrix(problem(),0.2))
    np.testing.assert_allclose(matrix,result["stability"]["iteration_matrix"],atol=0)
    target = np.array(result["reference"]["mean"])
    for before,after in zip(result["trace"],result["trace"][1:]):
        np.testing.assert_allclose(np.array(after["mean"])-target,matrix@(np.array(before["mean"])-target),atol=1e-15)
    assert result["status"] == "iteration_limit" and not result["converged"]


def test_unstable_run_retains_capped_divergence_and_valid_numerical_limit_prefix():
    capped = fit(alpha=1.1,max_iterations=12)
    assert capped["status"] == "iteration_limit" and not capped["converged"]
    assert not capped["stability"]["asymptotically_stable"]
    assert capped["trace"][-1]["kl_to_reference"] > capped["trace"][0]["kl_to_reference"]
    limited = fit(alpha=2,max_iterations=512)
    assert limited["status"] == "numerical_limit" and not limited["converged"]
    assert limited["iterations"] < 512
    assert limited["termination"]["candidate_retained"] is False
    assert limited["termination"]["attempted_iteration"] == limited["iterations"]+1
    assert limited["mean"] == limited["trace"][-1]["mean"]
    assert all(math.isfinite(row["free_energy"]) for row in limited["trace"])
    assert max(abs(v) for row in limited["trace"] for v in row["mean"]) <= fem.MAX_ITERATE_MEAN


def test_held_out_prediction_has_full_uncertainty_and_distinct_joint_coverage():
    reference = fem.gaussian_reference(problem())
    prediction = fem.predict_held_out(held_out(),reference["mean"],reference["covariance"])
    np.testing.assert_allclose(prediction["mean"],[0.5,1.5],atol=1e-15)
    np.testing.assert_allclose(prediction["covariance"],1.25*np.eye(2),atol=1e-15)
    assert prediction["log_predictive_density"] == pytest.approx(-math.log(2*math.pi)-math.log(1.25),abs=1e-14)
    assert prediction["marginal_z"] == 1.959963984540054
    assert prediction["joint_chi2_threshold"] == pytest.approx(-2*math.log(0.05))
    assert prediction["joint_chi2_degrees_of_freedom"] == 2
    assert prediction["joint_covered"] and all(prediction["marginal_covered"])
    changed = held_out()
    changed["observations"][0] += 3*math.sqrt(1.25)
    outside = fem.predict_held_out(changed,reference["mean"],reference["covariance"])
    assert outside["marginal_covered"] == [False,True] and not outside["joint_covered"]
    assert outside["mahalanobis_squared"] == pytest.approx(9)


def normalized_with_units(latent_scale,observation_scale):
    value = problem()
    latent,observed = np.array(latent_scale),np.array(observation_scale)
    return fem.normalize_problem(prior_mean=np.array(value["prior_mean"])*latent,
        prior_covariance=np.array(value["prior_covariance"])*np.outer(latent,latent),
        observation_matrix=np.array(value["observation_matrix"])*observed[:,None]/latent[None,:],
        observations=np.array(value["observations"])*observed,
        noise_covariance=np.array(value["noise_covariance"])*np.outer(observed,observed),
        latent_scales=latent,observation_scales=observed)


def test_unit_changes_normalize_to_same_inference_and_density_jacobian_is_explicit():
    first = normalized_with_units([2,0.1],[3,4])
    changed = normalized_with_units([200,0.1],[3000,0.04])
    a,b = fem.gaussian_reference(first["problem"]),fem.gaussian_reference(changed["problem"])
    np.testing.assert_allclose(a["mean"],b["mean"],rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(a["covariance"],b["covariance"],rtol=1e-14,atol=1e-14)
    assert a["log_evidence"] == pytest.approx(b["log_evidence"],abs=1e-14)
    raw_a = a["log_evidence"]-first["normalization"]["log_observation_jacobian"]
    raw_b = b["log_evidence"]-changed["normalization"]["log_observation_jacobian"]
    assert raw_b-raw_a == pytest.approx(-math.log(1000*0.01),abs=1e-13)
    f_a = fem.free_energy(first["problem"],[0,0],np.eye(2))["free_energy"]
    f_b = fem.free_energy(changed["problem"],[0,0],np.eye(2))["free_energy"]
    assert f_a == pytest.approx(f_b,abs=1e-14)
    assert f_a+first["normalization"]["log_observation_jacobian"] != pytest.approx(f_b+changed["normalization"]["log_observation_jacobian"])
    assert "kl_relation" in first["normalization"]


def test_kl_is_invariant_under_common_invertible_coordinate_changes():
    mean,cov = np.array([0.3,-0.1]),np.array([[1,0.2],[0.2,2]])
    ref_mean,ref_cov = np.array([0.2,0.4]),np.array([[0.7,-0.1],[-0.1,0.5]])
    transform = np.array([[2,0.4],[-0.3,0.7]])
    before = fem.gaussian_kl(mean,cov,ref_mean,ref_cov)
    after = fem.gaussian_kl(transform@mean,transform@cov@transform.T,transform@ref_mean,transform@ref_cov@transform.T)
    assert before == pytest.approx(after,rel=1e-13,abs=1e-13)
    assert fem.gaussian_kl(mean,cov,mean,cov) < 1e-28
    assert math.isfinite(fem.gaussian_kl([0,0],1e-20*np.eye(2),[0,0],np.eye(2)))


def test_seeded_ensemble_retains_draws_and_reference_coverage_counts():
    simulated = fem.simulate_ensemble(problem(),held_out(),seed=20260923,replicates=512)
    assert simulated == fem.simulate_ensemble(problem(),held_out(),seed=20260923,replicates=512)
    assert simulated["rng"]["bit_generator"] == "PCG64"
    np.testing.assert_allclose(simulated["training_observations"],np.array(simulated["truths"]) + simulated["training_noise"],atol=1e-14)
    np.testing.assert_allclose(simulated["heldout_observations"],np.array(simulated["truths"])@np.array(held_out()["observation_matrix"]).T + simulated["heldout_noise"],atol=1e-14)
    summary = fem.evaluate_ensemble(problem(),simulated["truths"],simulated["training_observations"],held_out(),simulated["heldout_observations"])
    assert "prior_predictive" in summary["scope"] and "not_fixed_truth" in summary["scope"]
    assert summary["reference_method"] == "observation_space_conditioning"
    assert len(summary["records"]) == 512
    for domain in ("latent","heldout"):
        assert 0.88 < summary[domain+"_joint_coverage"] < 1
        assert summary[domain+"_joint_count"] / 512 == summary[domain+"_joint_coverage"]
        for count,proportion,interval in zip(summary[domain+"_marginal_counts"],summary[domain+"_marginal_coverage"],summary[domain+"_marginal_wilson_95"]):
            assert count/512 == proportion
            assert interval["lower"] <= proportion <= interval["upper"]
            assert interval["successes"] == count and interval["trials"] == 512
    assert summary["nominal_monte_carlo_standard_error"] == pytest.approx(math.sqrt(0.95*0.05/512))
    for row in summary["records"][:3]:
        declared = problem()
        declared["observations"] = row["training_observations"]
        reference = fem.gaussian_reference(declared)
        np.testing.assert_allclose(row["posterior_mean"],reference["mean"],atol=1e-14)
        assert row["log_evidence"] == pytest.approx(reference["log_evidence"],abs=1e-14)


def test_wilson_intervals_handle_zero_and_full_counts():
    z = 1.959963984540054
    zero,full = fem._wilson95(0,10),fem._wilson95(10,10)
    assert zero["lower"] == pytest.approx(0,abs=1e-15)
    assert zero["upper"] == pytest.approx(z*z/(10+z*z))
    assert full["lower"] == pytest.approx(10/(10+z*z))
    assert full["upper"] == pytest.approx(1,abs=1e-15)


@pytest.mark.parametrize("field,value", [
    ("coordinate_system","raw"), ("prior_mean",[0]), ("prior_mean",[True,0]),
    ("prior_covariance",[[1,0.1],[0,1]]), ("prior_covariance",[[1,0],[0,0]]),
    ("noise_covariance",[[1,2],[2,1]]), ("observation_matrix",[[1,0,0],[0,1,0]]),
    ("observations",[float("nan"),1]), ("observations",[float("inf"),1]),
])
def test_invalid_or_unnormalized_problems_fail_closed(field,value):
    declared = problem()
    declared[field] = value
    with pytest.raises(ValueError):
        fem.gaussian_reference(declared)


@pytest.mark.parametrize("settings", [
    {"alpha":0}, {"alpha":True}, {"beta":0}, {"beta":1}, {"beta":-0.1},
    {"max_iterations":0}, {"max_iterations":513}, {"max_iterations":True},
    {"gradient_tolerance":0}, {"precision_tolerance":1},
])
def test_iteration_settings_are_bounded_and_precision_replacement_is_not_allowed(settings):
    with pytest.raises(ValueError):
        fit(**settings)


def test_bounds_and_no_mutation():
    declared,held = problem(),held_out()
    before = deepcopy((declared,held))
    fit(declared)
    fem.simulate_ensemble(declared,held,seed=1,replicates=2)
    assert (declared,held) == before
    for seed,replicates in ((True,1),(-1,1),(2**64,1),(1,0),(1,513)):
        with pytest.raises(ValueError):
            fem.simulate_ensemble(declared,held,seed=seed,replicates=replicates)
    with pytest.raises(ValueError):
        fem.evaluate_ensemble(declared,[],[],held,[])
    with pytest.raises(ValueError):
        fem.predict_held_out(held,[0,0],np.eye(2),coverage=1)
