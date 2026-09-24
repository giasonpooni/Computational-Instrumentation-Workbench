"""Offline algebra and semantic checks for retained Gaussian bench records.

No provider is imported or executed, and no optimizer or ensemble estimator is
rerun. Checks bind supplied matrices, iterates, objectives and prediction/count
evidence. They do not authenticate an execution or a declared sampling law.
"""
import math

import numpy as np

from . import free_energy_math as mathematics
from .free_energy_profile import csg_request, problems
from .telemetry import canonical

_I = np.eye(2)
_ROUND = 512 * np.finfo(float).eps
_FIT = {"status", "converged", "iterations", "mean", "covariance", "precision", "reference",
        "information_system", "settings", "stability", "termination", "trace"}
_TRACE = {"iteration", "mean", "covariance", "precision", "free_energy", "terms", "kl_to_reference",
          "negative_log_evidence", "free_energy_identity_residual", "gradient", "gradient_norm", "precision_relative_residual"}


def _keys(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError("Free-energy contract has missing or unexpected fields")


def _same(actual, expected):
    if canonical(actual) != canonical(expected):
        raise ValueError("Free-energy retained declaration differs")


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Free-energy evidence requires finite real numbers")
    return float(value)


def _array(value, shape, *, positive=False, bound=1e100):
    def visit(item):
        if type(item) is list:
            for child in item:
                visit(child)
        else:
            _number(item)
    visit(value)
    result = np.asarray(value, dtype=float)
    if result.shape != shape or np.any(np.abs(result) > bound):
        raise ValueError("Free-energy evidence exceeds its shape or magnitude bounds")
    if positive:
        if not np.array_equal(result, result.T):
            raise ValueError("Retained Gaussian matrix must be symmetric")
        values = np.linalg.eigvalsh(result)
        if values[0] <= 0 or values[-1] / values[0] > 1e12:
            raise ValueError("Retained Gaussian matrix must be positive and conditioned")
        np.linalg.cholesky(result)
    return result


def _close(actual, expected):
    """Scale-aware binary64 consistency, not a statistical uncertainty bound."""
    if isinstance(actual, np.ndarray):
        actual = actual.tolist()
    if isinstance(expected, np.ndarray):
        expected = expected.tolist()
    if isinstance(expected, np.generic):
        expected = expected.item()
    if type(expected) is dict:
        _keys(actual, expected)
        for key in expected:
            _close(actual[key], expected[key])
    elif type(expected) is list:
        if type(actual) is not list or len(actual) != len(expected):
            raise ValueError("Free-energy evidence list shape differs")
        for a, b in zip(actual, expected):
            _close(a, b)
    elif type(expected) in (int, float):
        a, b = _number(actual), _number(expected)
        if abs(a-b) > _ROUND * max(1.0, abs(a), abs(b)):
            raise ValueError("Free-energy numerical relationship differs")
    else:
        _same(actual, expected)


def _logdet(matrix):
    return float(2 * np.log(np.diag(np.linalg.cholesky(matrix))).sum())


def _product(left, right, expected):
    # A nearly cancelling matrix product must be bounded by the magnitudes of
    # its factors, not just by a small resulting entry or the identity matrix.
    # This is a backward-residual check, not a forward accuracy certificate.
    value = left @ right
    scale = max(1.0, float(np.linalg.norm(left, ord=np.inf)) * float(np.linalg.norm(right, ord=np.inf)),
                float(np.linalg.norm(expected, ord=np.inf)))
    if float(np.linalg.norm(value-expected, ord=np.inf)) > _ROUND * scale:
        raise ValueError("Retained Gaussian equations exceed their product roundoff budget")


def _reference(problem, reference, precision, information):
    _keys(reference, {"method", "covariance_method", "mean", "covariance", "innovation", "innovation_covariance",
                      "log_evidence", "evidence_normalization", "innovation_quadratic"})
    _same(reference["method"], "observation_space_conditioning")
    _same(reference["covariance_method"], "joseph_conditional_covariance")
    mean = _array(reference["mean"], (2,))
    covariance = _array(reference["covariance"], (2,2), positive=True)
    _product(precision, mean, information)
    _product(precision, covariance, _I)
    m0, c0, g, y, noise = mathematics._problem(problem)
    innovation = y-g@m0
    s = noise+g@c0@g.T
    s = (s+s.T)*.5
    quadratic = float(innovation @ np.linalg.solve(s,innovation))
    normalization = .5*(2*math.log(2*math.pi)+_logdet(s))
    _close(reference["innovation"], innovation)
    _close(reference["innovation_covariance"], s)
    _close(reference["innovation_quadratic"], quadratic)
    _close(reference["evidence_normalization"], normalization)
    _close(reference["log_evidence"], -normalization-.5*quadratic)
    return mean,covariance


def _valid_candidate(problem, mean, precision, reference):
    """Check one rejected candidate's numerical domain, not another iteration."""
    try:
        if not np.all(np.isfinite(mean)) or np.max(np.abs(mean)) > mathematics.MAX_ITERATE_MEAN:
            return False
        precision = mathematics._positive(precision, "candidate precision")
        covariance = mathematics._positive(np.linalg.solve(precision,_I), "candidate covariance")
        mathematics.free_energy(problem,mean.tolist(),covariance.tolist())
        mathematics.gaussian_kl(mean.tolist(),covariance.tolist(),reference["mean"],reference["covariance"])
        return True
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return False


def validate_fit(problem, solver, fit):
    """Check supplied posterior equations and each retained update recurrence."""
    _keys(fit,_FIT)
    information = mathematics.information_system(problem)
    _close(fit["information_system"],information)
    precision = np.asarray(information["precision"])
    eta = np.asarray(information["information_vector"])
    _reference(problem,fit["reference"],precision,eta)
    settings = {key:float(solver[key]) for key in ("alpha","beta","gradient_tolerance","precision_tolerance")}
    settings.update(max_iterations=solver["max_iterations"],mean_update="euclidean_gradient_in_normalized_coordinates",
                    covariance_update="fisher_natural_gradient_euler_in_precision_coordinates")
    _same(fit["settings"],settings)
    alpha,beta = settings["alpha"],settings["beta"]
    a = _I-alpha*precision
    eig = np.linalg.eigvalsh(a)
    radius = float(np.max(np.abs(eig)))
    _close(fit["stability"],{"iteration_matrix":a.tolist(),"eigenvalues":eig.tolist(),"spectral_radius":radius,
        "asymptotically_stable":radius<1,"stable_step_size_upper_bound":float(2/np.linalg.eigvalsh(precision)[-1]),
        "criterion":"spectral_radius(I-alpha*Lambda)<1; normalized_mean_error_iteration"})
    trace = fit["trace"]
    if type(trace) is not list or not 1 <= len(trace) <= solver["max_iterations"]+1:
        raise ValueError("Free-energy trace violates the declared iteration budget")
    previous = None
    for index,row in enumerate(trace):
        _keys(row,_TRACE)
        if type(row["iteration"]) is not int or row["iteration"] != index:
            raise ValueError("Free-energy iteration sequence differs")
        mean = _array(row["mean"],(2,),bound=mathematics.MAX_ITERATE_MEAN)
        covariance = _array(row["covariance"],(2,2),positive=True)
        q = _array(row["precision"],(2,2),positive=True)
        _product(q,covariance,_I)
        if previous is None:
            _close(row["mean"],np.asarray(solver["initial_mean"],dtype=float))
            _close(row["covariance"],np.asarray(solver["initial_covariance"],dtype=float))
        else:
            old_mean,old_q,old_gradient,old_converged = previous
            if old_converged:
                raise ValueError("Trace continued after its convergence condition")
            _close(mean,old_mean-alpha*old_gradient)
            _close(q,(1-beta)*old_q+beta*precision)
        gradient = precision@mean-eta
        norm = float(np.linalg.norm(gradient))
        residual = float(np.linalg.norm(q-precision)/max(1.0,float(np.linalg.norm(precision))))
        energy = mathematics.free_energy(problem,row["mean"],row["covariance"])
        kl = mathematics.gaussian_kl(row["mean"],row["covariance"],fit["reference"]["mean"],fit["reference"]["covariance"])
        _close(row["free_energy"],energy["free_energy"])
        _close(row["terms"],energy["terms"])
        _close(row["kl_to_reference"],kl)
        _close(row["negative_log_evidence"],-fit["reference"]["log_evidence"])
        identity = energy["free_energy"]+fit["reference"]["log_evidence"]-kl
        _close(row["free_energy_identity_residual"],identity)
        if abs(identity) > 1e-9*max(1.0,abs(energy["free_energy"]),abs(kl),abs(fit["reference"]["log_evidence"])):
            raise ValueError("Free-energy normalization violates the Gaussian KL identity")
        _close(row["gradient"],gradient)
        _close(row["gradient_norm"],norm)
        _close(row["precision_relative_residual"],residual)
        converged = norm<=settings["gradient_tolerance"] and residual<=settings["precision_tolerance"]
        previous = mean,q,gradient,converged
    last = trace[-1]
    if type(fit["iterations"]) is not int or fit["iterations"] != len(trace)-1:
        raise ValueError("Free-energy final iteration count differs")
    for key in ("mean","covariance","precision"):
        _same(fit[key],last[key])
    status = fit["status"]
    if status == "converged":
        if not converged:
            raise ValueError("Free-energy convergence claim contradicts its residuals")
        termination = {"reason":"convergence_tolerances_met","attempted_iteration":len(trace)-1,"candidate_retained":True}
    elif status == "iteration_limit":
        if converged or len(trace)-1 != settings["max_iterations"]:
            raise ValueError("Free-energy iteration-limit status contradicts the retained prefix")
        termination = {"reason":"maximum_iterations_reached","attempted_iteration":len(trace)-1,"candidate_retained":True}
    elif status == "numerical_limit":
        if converged or len(trace)-1 >= settings["max_iterations"]:
            raise ValueError("Free-energy numerical exit occurs outside an attempted update")
        with np.errstate(over="ignore",invalid="ignore"):
            if _valid_candidate(problem,mean-alpha*gradient,(1-beta)*q+beta*precision,fit["reference"]):
                raise ValueError("Free-energy numerical-limit claim omits an admissible next iterate")
        termination = {"reason":"nonfinite_or_out_of_bound_candidate","attempted_iteration":len(trace),"candidate_retained":False}
    else:
        raise ValueError("Unknown free-energy termination status")
    _same(fit["converged"],status=="converged")
    _same(fit["termination"],termination)


def _wilson(successes,trials):
    z=1.959963984540054
    p=successes/trials
    denominator=1+z*z/trials
    center=(p+z*z/(2*trials))/denominator
    half=z*math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/denominator
    return {"lower":max(0.0,center-half),"upper":min(1.0,center+half),"confidence":.95,
            "method":"wilson_score","successes":successes,"trials":trials}


def validate_ensemble(source, problem, held, reference, ensemble):
    """Check supplied per-trial posterior equations, predictions and counts."""
    _keys(ensemble,{"basis","generator_relationship","metrics"})
    _same(ensemble["basis"],"exact_reference_posterior_on_retained_ensemble")
    _same(ensemble["generator_relationship"],"operator_declared_iid_prior_predictive_not_authenticated")
    data=ensemble["metrics"]
    keys={"scope","reference_method","noise_relation","replicates","coverage","marginal_z","joint_chi2_degrees_of_freedom",
        "joint_chi2_threshold","latent_marginal_counts","latent_marginal_coverage","latent_joint_count","latent_joint_coverage",
        "heldout_marginal_counts","heldout_marginal_coverage","heldout_joint_count","heldout_joint_coverage",
        "latent_marginal_wilson_95","latent_joint_wilson_95","heldout_marginal_wilson_95","heldout_joint_wilson_95",
        "nominal_monte_carlo_standard_error","records"}
    _keys(data,keys)
    count=len(source["samples"])
    z=1.959963984540054
    threshold=-2*math.log1p(-.95)
    expected={"scope":"empirical_iid_prior_predictive_coverage_not_fixed_truth_frequentist_calibration",
        "reference_method":"observation_space_conditioning","noise_relation":mathematics.NOISE_RELATION,
        "replicates":count,"coverage":.95,"marginal_z":z,"joint_chi2_degrees_of_freedom":2,"joint_chi2_threshold":threshold}
    for key,value in expected.items():
        _same(data[key],value)
    rows=data["records"]
    if type(rows) is not list or len(rows)!=count:
        raise ValueError("Free-energy coverage requires one record per retained trial")
    scale=np.asarray(source["coordinates"]["scales"])
    obs_scale=np.asarray([s["scale"] for s in source["sensors"]])
    model=source["assumed_model"]
    m0,c0,g,_,noise=mathematics._problem(problem)
    precision=np.asarray(mathematics.information_system(problem)["precision"])
    covariance=np.asarray(reference["covariance"])
    s=noise+g@c0@g.T
    s=(s+s.T)*.5
    std=np.sqrt(np.diag(covariance))
    latent_counts=np.zeros(2,dtype=int);held_counts=np.zeros(2,dtype=int)
    latent_joint=held_joint=0
    row_keys={"replicate","truth","training_observations","posterior_mean","posterior_covariance","log_evidence",
        "latent_error","latent_mahalanobis_squared","latent_marginal_intervals","latent_marginal_covered","latent_joint_covered","held_out"}
    for index,(sample,row) in enumerate(zip(source["samples"],rows)):
        _keys(row,row_keys)
        if type(row["replicate"]) is not int or row["replicate"]!=index:
            raise ValueError("Free-energy trial order differs")
        truth=np.asarray(sample["truth"])/scale
        training=(np.asarray(sample["training"])-model["training_bias"])/obs_scale
        observed=(np.asarray(sample["heldout"])-model["heldout_bias"])/obs_scale
        _close(row["truth"],truth);_close(row["training_observations"],training)
        mean=_array(row["posterior_mean"],(2,))
        eta=np.linalg.solve(c0,m0)+g.T@np.linalg.solve(noise,training)
        _product(precision,mean,eta)
        _close(row["posterior_covariance"],covariance)
        error=truth-mean
        quadratic=float(error@np.linalg.solve(covariance,error))
        marginal=np.abs(error)<=z*std
        joint=bool(quadratic<=threshold)
        innovation=training-g@m0
        logp=-.5*(2*math.log(2*math.pi)+_logdet(s)+float(innovation@np.linalg.solve(s,innovation)))
        _close(row["log_evidence"],logp)
        _close(row["latent_error"],error)
        _close(row["latent_mahalanobis_squared"],quadratic)
        _close(row["latent_marginal_intervals"],np.column_stack((mean-z*std,mean+z*std)))
        _same(row["latent_marginal_covered"],marginal.tolist());_same(row["latent_joint_covered"],joint)
        held_problem={**held,"observations":observed.tolist()}
        prediction=mathematics.predict_held_out(held_problem,row["posterior_mean"],reference["covariance"])
        _close(row["held_out"],prediction)
        latent_counts+=marginal;held_counts+=np.asarray(prediction["marginal_covered"],dtype=int)
        latent_joint+=joint;held_joint+=prediction["joint_covered"]
    for group,counts,joint in (("latent",latent_counts,latent_joint),("heldout",held_counts,held_joint)):
        _same(data[group+"_marginal_counts"],counts.tolist())
        _close(data[group+"_marginal_coverage"],counts/count)
        _same(data[group+"_joint_count"],int(joint));_close(data[group+"_joint_coverage"],joint/count)
        _close(data[group+"_marginal_wilson_95"],[_wilson(int(v),count) for v in counts])
        _close(data[group+"_joint_wilson_95"],_wilson(int(joint),count))
    _close(data["nominal_monte_carlo_standard_error"],math.sqrt(.95*.05/count))


def validate_data(source,data,evidence_id):
    """Validate an aggregate after its outer workflow checked source/envelopes."""
    try:
        _validate_data(source,data,evidence_id)
    except (KeyError,TypeError,IndexError,AttributeError,OverflowError,RecursionError,UnicodeError,np.linalg.LinAlgError) as exc:
        raise ValueError("Malformed retained free-energy aggregate") from exc


def _validate_data(source,data,evidence_id):
    from . import free_energy_native as native
    _keys(data,{"schema","claim_scope","normalization","problem","held_out_problem","fit","gsie_agreement",
        "physical_posterior","held_out","ensemble","objective_units","stages"})
    _same(data["schema"],"ciw.variational-free-energy-result.v1")
    _same(data["claim_scope"],"synthetic_static_linear_gaussian_inference")
    canonical(data)
    stages=data["stages"]
    if type(stages) is not list or len(stages)!=3 or [s["runtime_ref"] for s in stages]!=["csg","gsie","plsr"]:
        raise ValueError("Free-energy aggregate requires all three ordered native stages")
    csg=stages[0]["result"]["data"]
    native.validate_response("csg",csg_request(source),csg)
    problem,held,normalization=problems(source,csg["Phi"])
    _same(data["problem"],problem);_same(data["held_out_problem"],held);_same(data["normalization"],normalization)
    fit=data["fit"]
    validate_fit(problem,source["solver"],fit)
    reference=fit["reference"]
    requests={"csg":csg_request(source),"gsie":{"G":problem["observation_matrix"],"y":problem["observations"],
        "Sigma":problem["noise_covariance"],"prior":{"mean":problem["prior_mean"],"covariance":problem["prior_covariance"]},"evidence_id":evidence_id},
        "plsr":{"Lambda":fit["information_system"]["precision"],"alpha":source["solver"]["alpha"],
            "error":(np.asarray(source["solver"]["initial_mean"])-reference["mean"]).tolist(),"evidence_id":evidence_id,"required_margin":0.0}}
    for role,stage in zip(("csg","gsie","plsr"),stages):
        _same(stage["request"],requests[role])
        if role!="csg":
            native.validate_response(role,requests[role],stage["result"]["data"])
    estimate=stages[1]["result"]["data"]["estimate"]
    mean_error=float(np.max(np.abs(np.asarray(estimate["mean"])-reference["mean"])))
    covariance_error=float(np.max(np.abs(np.asarray(estimate["covariance"])-reference["covariance"])))
    _close(data["gsie_agreement"],{"mean_max_abs_error":mean_error,"covariance_max_abs_error":covariance_error,"tolerance":1e-9,
                                "passed":max(mean_error,covariance_error)<=1e-9})
    if max(mean_error,covariance_error)>1e-9:
        raise ValueError("Native Gaussian comparison did not meet its declared tolerance")
    scale=np.asarray(normalization["latent_scales"])
    _close(data["physical_posterior"],{"mean":(scale*np.asarray(fit["mean"])).tolist(),
        "covariance":(np.asarray(fit["covariance"])*np.outer(scale,scale)).tolist(),"units":source["coordinates"]["units"],
        "frame":source["coordinates"]["frame"],"uncertainty_scope":"conditional_on_declared_model"})
    _close(data["held_out"],mathematics.predict_held_out(held,fit["mean"],fit["covariance"]))
    validate_ensemble(source,problem,held,reference,data["ensemble"])
    final=fit["trace"][-1];jacobian=normalization["log_observation_jacobian"]
    _close(data["objective_units"],{"free_energy_normalized":final["free_energy"],"free_energy_physical":final["free_energy"]+jacobian,
        "log_evidence_normalized":reference["log_evidence"],"log_evidence_physical":reference["log_evidence"]-jacobian,
        "kl_gap":final["kl_to_reference"]})
