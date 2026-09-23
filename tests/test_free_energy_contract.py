"""Offline algebra tests; kernel records are not claimed as native evidence."""
from copy import deepcopy

import numpy as np
import pytest

from ciw import free_energy_math as mathematics
from ciw.free_energy_contract import validate_fit, validate_ensemble


def problem():
    return {"coordinate_system":"normalized_dimensionless","prior_mean":[0,0],
        "prior_covariance":[[1,.2],[.2,1]],"observation_matrix":[[1,.3],[-.2,1]],
        "observations":[2,-1],"noise_covariance":[[1,.1],[.1,.7]]}


def held():
    return {"coordinate_system":"normalized_dimensionless","observation_matrix":[[1,1],[1,-1]],
        "observations":[.5,1.5],"noise_covariance":[[.25,.05],[.05,.25]]}


def settings(**changes):
    result={"initial_mean":[0,0],"initial_covariance":[[1,.1],[.1,1]],"alpha":.2,"beta":.2,
        "max_iterations":200,"gradient_tolerance":1e-9,"precision_tolerance":1e-9}
    result.update(changes)
    return result


def record(**changes):
    solver=settings(**changes)
    return solver,mathematics.variational_fit(problem(),**solver)


def set_path(value,path,replacement):
    for key in path[:-1]:
        value=value[key]
    value[path[-1]]=replacement


@pytest.mark.parametrize("changes", [{},{"max_iterations":2},{"alpha":2.0}])
def test_complete_limited_and_unstable_retained_iteration_checks(changes,monkeypatch):
    solver,data=record(**changes)
    monkeypatch.setattr(mathematics,"variational_fit",lambda *_a,**_k: pytest.fail("Offline validation reran VI"))
    monkeypatch.setattr(mathematics,"gaussian_reference",lambda *_a,**_k: pytest.fail("Offline validation reran conditioning"))
    before=deepcopy(data)
    validate_fit(problem(),solver,data)
    assert data==before


@pytest.mark.parametrize("path,replacement", [
    (("reference","mean",0),1.1),
    (("reference","covariance",0,0),.7),
    (("reference","log_evidence"),0),
    (("reference","method"),"claimed_exact"),
    (("information_system","precision",0,0),2),
    (("settings","alpha"),.1),
    (("settings","covariance_update"),"direct_replacement"),
    (("stability","iteration_matrix",0,0),1),
    (("stability","asymptotically_stable"),False),
    (("trace",0,"mean",0),.01),
    (("trace",0,"free_energy"),0),
    (("trace",0,"terms","likelihood_normalization"),0),
    (("trace",0,"kl_to_reference"),0),
    (("trace",0,"free_energy_identity_residual"),.01),
    (("trace",0,"gradient_norm"),0),
    (("trace",0,"precision_relative_residual"),0),
    (("trace",1,"iteration"),True),
    (("trace",1,"mean",0),.5),
    (("trace",1,"covariance",0,0),.5),
    (("trace",1,"precision",0,0),1),
    (("trace",1,"gradient",0),0),
    (("status",),"iteration_limit"),
    (("converged",),1),
    (("termination","candidate_retained"),False),
    (("mean",0),0),
])
def test_tampered_retained_fit_is_refused(path,replacement):
    solver,data=record()
    set_path(data,path,replacement)
    with pytest.raises(ValueError):
        validate_fit(problem(),solver,data)


def test_truncated_valid_trace_cannot_claim_numerical_exit():
    solver,data=record()
    data["trace"]=data["trace"][:2]
    data.update(status="numerical_limit",converged=False,iterations=1,
                termination={"reason":"nonfinite_or_out_of_bound_candidate","attempted_iteration":2,"candidate_retained":False})
    for key in ("mean","covariance","precision"):
        data[key]=deepcopy(data["trace"][-1][key])
    with pytest.raises(ValueError,match="admissible"):
        validate_fit(problem(),solver,data)


def ensemble_record():
    sampled=mathematics.simulate_ensemble(problem(),held(),seed=9281,replicates=12)
    source={"coordinates":{"scales":[1,1]},"sensors":[{"scale":1},{"scale":1}],
        "assumed_model":{"training_bias":[0,0],"heldout_bias":[0,0]},"samples":[]}
    for index in range(12):
        source["samples"].append({"truth":sampled["truths"][index],"training":sampled["training_observations"][index],
                                  "heldout":sampled["heldout_observations"][index]})
    value={"basis":"exact_reference_posterior_on_retained_ensemble",
        "generator_relationship":"operator_declared_iid_prior_predictive_not_authenticated",
        "metrics":mathematics.evaluate_ensemble(problem(),sampled["truths"],sampled["training_observations"],held(),sampled["heldout_observations"])}
    return source,mathematics.gaussian_reference(problem()),value


def test_ensemble_check_uses_retained_rows_without_refitting(monkeypatch):
    source,reference,data=ensemble_record()
    monkeypatch.setattr(mathematics,"evaluate_ensemble",lambda *_a,**_k: pytest.fail("Offline validation reran ensemble estimation"))
    monkeypatch.setattr(mathematics,"gaussian_reference",lambda *_a,**_k: pytest.fail("Offline validation reran conditioning"))
    saved=deepcopy(data)
    validate_ensemble(source,problem(),held(),reference,data)
    assert saved==data


@pytest.mark.parametrize("path,replacement", [
    (("basis",),"variational_fit_on_every_trial"),
    (("generator_relationship",),"authenticated_iid"),
    (("metrics","coverage"),.99),
    (("metrics","replicates"),True),
    (("metrics","latent_marginal_counts",0),0),
    (("metrics","heldout_joint_count"),0),
    (("metrics","heldout_joint_wilson_95","upper"),.1),
    (("metrics","nominal_monte_carlo_standard_error"),0),
    (("metrics","records",0,"replicate"),1),
    (("metrics","records",0,"truth",0),999),
    (("metrics","records",0,"training_observations",0),999),
    (("metrics","records",0,"posterior_mean",0),999),
    (("metrics","records",0,"posterior_covariance",0,0),999),
    (("metrics","records",0,"latent_mahalanobis_squared"),999),
    (("metrics","records",0,"latent_marginal_covered",0),1),
    (("metrics","records",0,"held_out","covariance",0,0),999),
    (("metrics","records",0,"held_out","noise_relation"),"unknown"),
    (("metrics","records",0,"held_out","joint_covered"),1),
])
def test_ensemble_prediction_counts_and_authority_bindings(path,replacement):
    source,reference,data=ensemble_record()
    set_path(data,path,replacement)
    with pytest.raises(ValueError):
        validate_ensemble(source,problem(),held(),reference,data)


def test_heldout_values_cannot_modify_reference_posterior_rows():
    source,reference,data=ensemble_record()
    changed=deepcopy(source)
    changed["samples"][0]["heldout"][0]+=1
    with pytest.raises(ValueError):
        validate_ensemble(changed,problem(),held(),reference,data)


def test_bad_covariance_not_accepted_as_close_to_valid_matrix():
    solver,data=record()
    data["trace"][0]["covariance"]=[[1,.1],[.2,1]]
    with pytest.raises(ValueError,match="symmetric"):
        validate_fit(problem(),solver,data)


def test_conditioned_valid_posterior_uses_factor_scaled_product_residual():
    value=problem()
    value.update(prior_covariance=[[100,999.99],[999.99,10000]],
        observation_matrix=[[.001,0],[0,.0001]],observations=[.01,-.01],noise_covariance=[[1,0],[0,1]])
    solver=settings(alpha=.0001,max_iterations=2)
    data=mathematics.variational_fit(value,**solver)
    validate_fit(value,solver,data)


# The workflow fixture runs real pinned providers, then shares their retained
# records. These aggregate checks bypass outer envelopes deliberately, testing
# semantic relationships rather than merely detecting an unrepaired hash.
from test_free_energy_workflow import native_experiments


def native_data(native_experiments):
    from ciw.telemetry import byte_digest
    case=native_experiments["cases"]["baseline"]
    return case["source"],case["bundle"]["steps"][0]["result"]["data"],byte_digest(case["raw"])


def test_actual_native_aggregate_validation_is_offline(native_experiments,monkeypatch):
    from ciw import free_energy_native as native
    from ciw.free_energy_contract import validate_data
    source,data,evidence=native_data(native_experiments)
    def forbidden(*_args,**_kwargs):
        pytest.fail("Offline validation invoked an estimator or native provider")
    for name in ("variational_fit","gaussian_reference","evaluate_ensemble"):
        monkeypatch.setattr(mathematics,name,forbidden)
    for name in ("invoke","bind","_dispatch"):
        monkeypatch.setattr(native,name,forbidden)
    saved=deepcopy(data)
    validate_data(source,data,evidence)
    assert data==saved


@pytest.mark.parametrize("path,replacement", [
    (("claim_scope",),"physical_sensor_fusion"),
    (("normalization","log_observation_jacobian"),0),
    (("problem","observations",0),999),
    (("held_out_problem","observations",0),999),
    (("gsie_agreement","passed"),False),
    (("gsie_agreement","tolerance"),1),
    (("physical_posterior","mean",0),999),
    (("physical_posterior","units"),["1","1"]),
    (("physical_posterior","uncertainty_scope"),"unconditional_physical_truth"),
    (("held_out","noise_relation"),"unmodelled"),
    (("held_out","covariance",0,0),999),
    (("objective_units","free_energy_physical"),0),
    (("objective_units","log_evidence_physical"),0),
    (("objective_units","kl_gap"),1),
    (("ensemble","basis"),"variational_fit_on_every_trial"),
    (("ensemble","metrics","latent_joint_count"),0),
    (("ensemble","generator_relationship"),"authenticated"),
    (("fit","trace",1,"gradient",0),999),
    (("stages",2,"request","alpha"),1),
])
def test_native_aggregate_semantic_tampering_is_refused(native_experiments,path,replacement):
    from ciw.free_energy_contract import validate_data
    source,data,evidence=native_data(native_experiments)
    changed=deepcopy(data)
    set_path(changed,path,replacement)
    with pytest.raises(ValueError):
        validate_data(source,changed,evidence)
