"""Two-latent linear-Gaussian reference and genuinely iterative Gaussian VI.

All inference is in explicitly normalized, dimensionless coordinates. Training
and held-out noise groups are independent given the latent; each group may have
full within-group covariance. Nothing here authenticates a physical model.

For q=N(m,C), F = E_q[-log p(y|x)] + KL(q||p(x)). Every Gaussian normalization
constant is retained. The independent reference uses observation-space Gaussian
conditioning, while the optimizer uses information precision Lambda and b.

The covariance Fisher metric is .5 tr(C^-1 U C^-1 V). Since dF/dC is
.5(Lambda-C^-1), its natural gradient in C is C(Lambda-Q)C. Under Q=C^-1 the
natural gradient is Q-Lambda. Euler descent in precision coordinates therefore
gives Q_next=(1-beta)Q+beta*Lambda. This is not an exact covariance replacement
when 0<beta<1. Mean descent is ordinary Euclidean descent in the declared
normalized coordinates: m_next=m-alpha*(Lambda*m-b).
"""
from __future__ import annotations

from copy import deepcopy
import math
from statistics import NormalDist

import numpy as np

COORDINATE_SYSTEM = "normalized_dimensionless"
DIMENSION = 2
MAX_ITERATIONS = 512
MAX_REPLICATES = 512
MAX_ITERATE_MEAN = 1e12
LOG_2PI = math.log(2 * math.pi)
NOISE_RELATION = "held_out_noise_independent_of_training_noise_given_latent"
_I = np.eye(2)


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " requires a finite real number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(name + " exceeds binary64") from exc
    if not math.isfinite(number):
        raise ValueError(name + " requires a finite real number")
    return number


def _array(value, shape, name, *, bound=1e6):
    def leaves(item):
        if isinstance(item, (list, tuple, np.ndarray)):
            for child in item:
                yield from leaves(child)
        else:
            yield _scalar(item, name)
    try:
        list(leaves(value))
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError(name + " has invalid numerical entries") from exc
    if result.shape != shape or not np.all(np.isfinite(result)) or np.max(np.abs(result)) > bound:
        raise ValueError(name + " has an unsupported shape or magnitude")
    return result.copy()


def _symmetric(matrix):
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Nonfinite computed matrix")
    if np.max(np.abs(matrix - matrix.T)) > 128 * np.finfo(float).eps * max(1.0, float(np.linalg.norm(matrix))):
        raise ValueError("Computed matrix has more than roundoff asymmetry")
    return (matrix + matrix.T) * 0.5


def _positive(matrix, name, *, input_matrix=False):
    if input_matrix and not np.array_equal(matrix, matrix.T):
        raise ValueError(name + " must already be exactly symmetric")
    matrix = matrix.copy() if input_matrix else _symmetric(matrix)
    eigenvalues = np.linalg.eigvalsh(matrix)
    floor = 1e-8 if input_matrix else 0.0
    limit = 1e8 if input_matrix else 1e12
    if (not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= floor or
            eigenvalues[-1] / eigenvalues[0] > limit):
        raise ValueError(name + " violates the positive-definite or conditioning bound")
    np.linalg.cholesky(matrix)
    return matrix


def _covariance(value, name, *, input_matrix=False):
    return _positive(_array(value, (2, 2), name, bound=1e6 if input_matrix else 1e20),
                     name, input_matrix=input_matrix)


def _logdet(matrix):
    factor = np.linalg.cholesky(matrix)
    return float(2 * np.log(np.diag(factor)).sum())


def _mapping(value, required):
    if type(value) is not dict or set(value) != set(required):
        raise ValueError("Require exactly the declared normalized Gaussian fields")
    if value["coordinate_system"] != COORDINATE_SYSTEM:
        raise ValueError("Inference requires explicitly normalized dimensionless coordinates")


def _problem(problem):
    _mapping(problem, {"coordinate_system", "prior_mean", "prior_covariance", "observation_matrix", "observations", "noise_covariance"})
    return (_array(problem["prior_mean"], (2,), "prior_mean"),
            _covariance(problem["prior_covariance"], "prior_covariance", input_matrix=True),
            _array(problem["observation_matrix"], (2, 2), "observation_matrix", bound=1e3),
            _array(problem["observations"], (2,), "observations"),
            _covariance(problem["noise_covariance"], "noise_covariance", input_matrix=True))


def _held_out(problem):
    _mapping(problem, {"coordinate_system", "observation_matrix", "observations", "noise_covariance"})
    return (_array(problem["observation_matrix"], (2, 2), "held_out.observation_matrix", bound=1e3),
            _array(problem["observations"], (2,), "held_out.observations"),
            _covariance(problem["noise_covariance"], "held_out.noise_covariance", input_matrix=True))


def normalize_problem(*, prior_mean, prior_covariance, observation_matrix, observations,
                      noise_covariance, latent_scales, observation_scales):
    """x_norm=D^-1*x_raw, y_norm=T^-1*y_raw, G_norm=T^-1*G_raw*D.

    The returned density Jacobian is essential: log p(y_raw) equals normalized
    log evidence minus sum(log observation_scales). F_raw adds that same sum;
    KL is unchanged. Absolute free energies in different units cannot be compared.
    """
    latent = _array(latent_scales, (2,), "latent_scales", bound=1e12)
    observed = _array(observation_scales, (2,), "observation_scales", bound=1e12)
    if np.any(latent < 1e-12) or np.any(observed < 1e-12):
        raise ValueError("Normalization scales must lie in [1e-12,1e12]")
    raw = lambda value, shape, name: _array(value, shape, name, bound=1e100)
    c0 = raw(prior_covariance, (2, 2), "prior_covariance")
    sigma = raw(noise_covariance, (2, 2), "noise_covariance")
    if not np.array_equal(c0, c0.T) or not np.array_equal(sigma, sigma.T):
        raise ValueError("Raw covariance declarations must already be exactly symmetric")
    normalized = {"coordinate_system": COORDINATE_SYSTEM,
        "prior_mean": (raw(prior_mean, (2,), "prior_mean") / latent).tolist(),
        "prior_covariance": (c0 / np.outer(latent, latent)).tolist(),
        "observation_matrix": (raw(observation_matrix, (2, 2), "observation_matrix") * latent[None, :] / observed[:, None]).tolist(),
        "observations": (raw(observations, (2,), "observations") / observed).tolist(),
        "noise_covariance": (sigma / np.outer(observed, observed)).tolist()}
    _problem(normalized)
    return {"problem": normalized, "normalization": {
        "latent_scales": latent.tolist(), "observation_scales": observed.tolist(),
        "log_observation_jacobian": float(np.log(observed).sum()),
        "coordinate_relation": "normalized=raw/declared_scale",
        "evidence_relation": "log_evidence_raw=log_evidence_normalized-log_observation_jacobian",
        "free_energy_relation": "free_energy_raw=free_energy_normalized+log_observation_jacobian",
        "kl_relation": "invariant_under_invertible_coordinate_normalization"}}


def _conditioning(m0, c0, g, sigma):
    # Independent reference: no information precision or information vector.
    innovation_covariance = _positive(sigma + g @ c0 @ g.T, "innovation_covariance")
    gain = np.linalg.solve(innovation_covariance, g @ c0).T
    remainder = _I - gain @ g
    covariance = _positive(remainder @ c0 @ remainder.T + gain @ sigma @ gain.T, "conditional_covariance")
    return innovation_covariance, gain, covariance


def gaussian_reference(problem):
    """Observation-space Gaussian conditioning, independent of VI information form."""
    m0, c0, g, y, sigma = _problem(problem)
    s, gain, covariance = _conditioning(m0, c0, g, sigma)
    innovation = y - g @ m0
    mean = m0 + gain @ innovation
    normalization = 0.5 * (2 * LOG_2PI + _logdet(s))
    quadratic = float(innovation @ np.linalg.solve(s, innovation))
    return {"method": "observation_space_conditioning", "covariance_method": "joseph_conditional_covariance",
        "mean": mean.tolist(), "covariance": covariance.tolist(),
        "innovation": innovation.tolist(), "innovation_covariance": s.tolist(),
        "log_evidence": float(-normalization - 0.5 * quadratic),
        "evidence_normalization": float(normalization), "innovation_quadratic": quadratic}


def _information(m0, c0, g, y, sigma):
    prior_precision = np.linalg.solve(c0, _I)
    precision = _positive(prior_precision + g.T @ np.linalg.solve(sigma, g), "information_precision")
    information = prior_precision @ m0 + g.T @ np.linalg.solve(sigma, y)
    return precision, information


def information_system(problem):
    m0, c0, g, y, sigma = _problem(problem)
    precision, information = _information(m0, c0, g, y, sigma)
    return {"precision": precision.tolist(), "information_vector": information.tolist()}


def mean_iteration_matrix(problem, alpha):
    alpha = _scalar(alpha, "alpha")
    if not 0 < alpha <= 1e6:
        raise ValueError("alpha must lie in (0,1e6]")
    precision = np.asarray(information_system(problem)["precision"])
    return (_I - alpha * precision).tolist()


def _free_energy(m0, c0, g, y, sigma, mean, covariance):
    residual, prior_delta = y - g @ mean, mean - m0
    likelihood_normalization = 0.5 * (2 * LOG_2PI + _logdet(sigma))
    likelihood_quadratic = float(residual @ np.linalg.solve(sigma, residual))
    likelihood_trace = float(np.trace(np.linalg.solve(sigma, g @ covariance @ g.T)))
    prior_normalization = 0.5 * (2 * LOG_2PI + _logdet(c0))
    prior_quadratic = float(prior_delta @ np.linalg.solve(c0, prior_delta))
    prior_trace = float(np.trace(np.linalg.solve(c0, covariance)))
    nll = likelihood_normalization + 0.5 * (likelihood_quadratic + likelihood_trace)
    cross_entropy = prior_normalization + 0.5 * (prior_quadratic + prior_trace)
    entropy = 0.5 * (2 * (1 + LOG_2PI) + _logdet(covariance))
    value = nll + cross_entropy - entropy
    terms = {"likelihood_normalization": float(likelihood_normalization),
        "likelihood_mean_quadratic": likelihood_quadratic, "likelihood_covariance_trace": likelihood_trace,
        "expected_negative_log_likelihood": float(nll), "prior_normalization": float(prior_normalization),
        "prior_mean_quadratic": prior_quadratic, "prior_covariance_trace": prior_trace,
        "prior_cross_entropy": float(cross_entropy), "variational_entropy": float(entropy)}
    if not all(math.isfinite(number) for number in [value, *terms.values()]):
        raise ValueError("Free-energy calculation left the finite numerical domain")
    return {"free_energy": float(value), "terms": terms}


def free_energy(problem, mean, covariance):
    args = _problem(problem)
    mean = _array(mean, (2,), "variational mean", bound=MAX_ITERATE_MEAN)
    covariance = _covariance(covariance, "variational covariance")
    return _free_energy(*args, mean, covariance)


def _lambda_minus_log(value):
    # Avoid cancellation in lambda-1-log(lambda) near the posterior optimum.
    delta = value - 1
    if abs(delta) < 1e-4:
        return math.fsum(((-1) ** order) * delta ** order / order for order in range(2, 10))
    return delta - math.log(value)


def _kl(mean, covariance, reference_mean, reference_covariance):
    factor = np.linalg.cholesky(reference_covariance)
    whitened_mean = np.linalg.solve(factor, mean - reference_mean)
    relative_factor = np.linalg.solve(factor, np.linalg.cholesky(covariance))
    eigenvalues = np.linalg.svd(relative_factor, compute_uv=False) ** 2
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0):
        raise ValueError("KL relative covariance left its finite positive domain")
    shape = math.fsum(_lambda_minus_log(float(value)) for value in eigenvalues)
    result = 0.5 * (float(whitened_mean @ whitened_mean) + shape)
    if not math.isfinite(result) or result < 0:
        raise ValueError("KL calculation left its finite nonnegative domain")
    return result


def gaussian_kl(mean, covariance, reference_mean, reference_covariance):
    return _kl(_array(mean, (2,), "mean", bound=MAX_ITERATE_MEAN), _covariance(covariance, "covariance"),
               _array(reference_mean, (2,), "reference_mean", bound=MAX_ITERATE_MEAN),
               _covariance(reference_covariance, "reference_covariance"))


def variational_fit(problem, *, initial_mean, initial_covariance, alpha, beta,
                    max_iterations, gradient_tolerance=1e-9, precision_tolerance=1e-9):
    """Iterate both mean and full precision; keep a valid prefix on numerical exit."""
    args = _problem(problem)
    precision, information = _information(*args)
    mean = _array(initial_mean, (2,), "initial_mean")
    covariance = _covariance(initial_covariance, "initial_covariance", input_matrix=True)
    q_precision = _positive(np.linalg.solve(covariance, _I), "initial_precision")
    alpha, beta = _scalar(alpha, "alpha"), _scalar(beta, "beta")
    if not 0 < alpha <= 1e6 or not 0 < beta < 1:
        raise ValueError("Require alpha in (0,1e6] and genuinely iterative beta in (0,1)")
    if type(max_iterations) is not int or not 1 <= max_iterations <= MAX_ITERATIONS:
        raise ValueError("max_iterations must lie in [1,512]")
    gradient_tolerance = _scalar(gradient_tolerance, "gradient_tolerance")
    precision_tolerance = _scalar(precision_tolerance, "precision_tolerance")
    if not all(1e-12 <= value <= 1e-3 for value in (gradient_tolerance, precision_tolerance)):
        raise ValueError("Convergence tolerances must lie in [1e-12,1e-3]")
    reference = gaussian_reference(problem)
    reference_mean, reference_covariance = np.asarray(reference["mean"]), np.asarray(reference["covariance"])
    iteration_matrix = _I - alpha * precision
    iteration_eigenvalues = np.linalg.eigvalsh(iteration_matrix)
    spectral_radius = float(np.max(np.abs(iteration_eigenvalues)))
    stability = {"iteration_matrix": iteration_matrix.tolist(), "eigenvalues": iteration_eigenvalues.tolist(),
        "spectral_radius": spectral_radius, "asymptotically_stable": spectral_radius < 1,
        "stable_step_size_upper_bound": float(2 / np.linalg.eigvalsh(precision)[-1]),
        "criterion": "spectral_radius(I-alpha*Lambda)<1; normalized_mean_error_iteration"}
    precision_scale = max(1.0, float(np.linalg.norm(precision)))
    trace = []
    status, reason, attempted, candidate_retained = "iteration_limit", "maximum_iterations_reached", 0, True
    for iteration in range(max_iterations + 1):
        gradient = precision @ mean - information
        gradient_norm = float(np.linalg.norm(gradient))
        precision_residual = float(np.linalg.norm(q_precision - precision) / precision_scale)
        energy = _free_energy(*args, mean, covariance)
        kl = _kl(mean, covariance, reference_mean, reference_covariance)
        trace.append({"iteration": iteration, "mean": mean.tolist(), "covariance": covariance.tolist(),
            "precision": q_precision.tolist(), **energy, "kl_to_reference": kl,
            "negative_log_evidence": -reference["log_evidence"],
            "free_energy_identity_residual": float(energy["free_energy"] + reference["log_evidence"] - kl),
            "gradient": gradient.tolist(), "gradient_norm": gradient_norm,
            "precision_relative_residual": precision_residual})
        attempted = iteration
        if gradient_norm <= gradient_tolerance and precision_residual <= precision_tolerance:
            status, reason = "converged", "convergence_tolerances_met"
            break
        if iteration == max_iterations:
            break
        attempted = iteration + 1
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                proposed_mean = mean - alpha * gradient
                proposed_precision = (1 - beta) * q_precision + beta * precision
                if not np.all(np.isfinite(proposed_mean)) or np.max(np.abs(proposed_mean)) > MAX_ITERATE_MEAN:
                    raise ValueError("candidate mean exceeded its finite bound")
                proposed_precision = _positive(proposed_precision, "iterated_precision")
                proposed_covariance = _positive(np.linalg.solve(proposed_precision, _I), "iterated_covariance")
                # Only commit a candidate that can produce all required evidence.
                _free_energy(*args, proposed_mean, proposed_covariance)
                _kl(proposed_mean, proposed_covariance, reference_mean, reference_covariance)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            status, reason, candidate_retained = "numerical_limit", "nonfinite_or_out_of_bound_candidate", False
            break
        mean, q_precision, covariance = proposed_mean, proposed_precision, proposed_covariance
    last = trace[-1]
    return {"status": status, "converged": status == "converged", "iterations": last["iteration"],
        "mean": deepcopy(last["mean"]), "covariance": deepcopy(last["covariance"]), "precision": deepcopy(last["precision"]),
        "reference": reference, "information_system": {"precision": precision.tolist(), "information_vector": information.tolist()},
        "settings": {"alpha": alpha, "beta": beta, "max_iterations": max_iterations,
            "gradient_tolerance": gradient_tolerance, "precision_tolerance": precision_tolerance,
            "mean_update": "euclidean_gradient_in_normalized_coordinates",
            "covariance_update": "fisher_natural_gradient_euler_in_precision_coordinates"},
        "stability": stability, "termination": {"reason": reason, "attempted_iteration": attempted,
            "candidate_retained": candidate_retained}, "trace": trace}


def _coverage_parameters(coverage):
    coverage = _scalar(coverage, "coverage")
    if not 0.5 <= coverage <= 0.999:
        raise ValueError("Nominal coverage must lie in [0.5,0.999]")
    z = 1.959963984540054 if coverage == 0.95 else NormalDist().inv_cdf((1 + coverage) / 2)
    return coverage, z, -2 * math.log1p(-coverage)


def _prediction(g, y, sigma, mean, covariance, coverage):
    nominal, z, threshold = _coverage_parameters(coverage)
    prediction = g @ mean
    predictive_covariance = _positive(sigma + g @ covariance @ g.T, "predictive_covariance")
    residual = y - prediction
    std = np.sqrt(np.diag(predictive_covariance))
    quadratic = float(residual @ np.linalg.solve(predictive_covariance, residual))
    normalization = 0.5 * (2 * LOG_2PI + _logdet(predictive_covariance))
    return {"mean": prediction.tolist(), "covariance": predictive_covariance.tolist(),
        "observations": y.tolist(), "residual": residual.tolist(), "standard_deviation": std.tolist(),
        "log_predictive_density": float(-normalization - 0.5 * quadratic),
        "density_normalization": float(normalization), "mahalanobis_squared": quadratic,
        "coverage": nominal, "marginal_z": z, "marginal_intervals": np.column_stack((prediction-z*std, prediction+z*std)).tolist(),
        "marginal_covered": (np.abs(residual) <= z * std).tolist(),
        "joint_chi2_degrees_of_freedom": 2, "joint_chi2_threshold": threshold,
        "joint_covered": bool(quadratic <= threshold), "noise_relation": NOISE_RELATION}


def predict_held_out(held_out_problem, mean, covariance, *, coverage=0.95):
    return _prediction(*_held_out(held_out_problem), _array(mean, (2,), "mean", bound=MAX_ITERATE_MEAN),
                       _covariance(covariance, "covariance"), coverage)


def simulate_ensemble(problem, held_out_problem, *, seed, replicates):
    """IID prior-predictive simulation; retain returned draws, not just the seed."""
    m0, c0, g, _, sigma = _problem(problem)
    h, _, held_noise = _held_out(held_out_problem)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if type(replicates) is not int or not 1 <= replicates <= MAX_REPLICATES:
        raise ValueError("replicates must lie in [1,512]")
    generator = np.random.Generator(np.random.PCG64(seed))
    truths = m0 + generator.standard_normal((replicates, 2)) @ np.linalg.cholesky(c0).T
    training_noise = generator.standard_normal((replicates, 2)) @ np.linalg.cholesky(sigma).T
    heldout_noise = generator.standard_normal((replicates, 2)) @ np.linalg.cholesky(held_noise).T
    return {"sampling_scope": "iid_prior_predictive", "noise_relation": NOISE_RELATION,
        "rng": {"bit_generator": "PCG64", "seed": seed, "numpy_version": np.__version__,
            "replay_policy": "retain_actual_truth_and_noise_arrays"},
        "truths": truths.tolist(), "training_noise": training_noise.tolist(), "heldout_noise": heldout_noise.tolist(),
        "training_observations": (truths @ g.T + training_noise).tolist(),
        "heldout_observations": (truths @ h.T + heldout_noise).tolist()}


def _wilson95(successes, trials):
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z*z/trials
    center = (proportion + z*z/(2*trials)) / denominator
    radius = z * math.sqrt(proportion*(1-proportion)/trials + z*z/(4*trials*trials)) / denominator
    return {"lower": max(0.0, center-radius), "upper": min(1.0, center+radius),
            "confidence": 0.95, "method": "wilson_score", "successes": int(successes), "trials": trials}


def evaluate_ensemble(problem, truths, training_observations, held_out_problem, heldout_observations, *, coverage=0.95):
    """Empirical prior-predictive coverage; not fixed-truth frequentist calibration.

    Shared matrices are factored once. Each retained training vector is conditioned
    by the observation-space reference; VI is not silently substituted for it.
    """
    m0, c0, g, _, sigma = _problem(problem)
    h, _, held_noise = _held_out(held_out_problem)
    count = len(truths) if isinstance(truths, (list, tuple, np.ndarray)) else 0
    if not 1 <= count <= MAX_REPLICATES:
        raise ValueError("Require 1..512 retained prior-predictive replicates")
    truths = _array(truths, (count, 2), "truths")
    training = _array(training_observations, (count, 2), "training_observations")
    heldout = _array(heldout_observations, (count, 2), "heldout_observations")
    nominal, z, threshold = _coverage_parameters(coverage)
    s, gain, covariance = _conditioning(m0, c0, g, sigma)
    std = np.sqrt(np.diag(covariance))
    records, latent_marginal, held_marginal = [], np.zeros(2, dtype=int), np.zeros(2, dtype=int)
    latent_joint, held_joint = 0, 0
    for index in range(count):
        innovation = training[index] - g @ m0
        mean = m0 + gain @ innovation
        error = truths[index] - mean
        quadratic = float(error @ np.linalg.solve(covariance, error))
        marginal = np.abs(error) <= z * std
        joint = bool(quadratic <= threshold)
        prediction = _prediction(h, heldout[index], held_noise, mean, covariance, nominal)
        log_evidence = -0.5 * (2 * LOG_2PI + _logdet(s) + float(innovation @ np.linalg.solve(s, innovation)))
        latent_marginal += marginal
        held_marginal += np.array(prediction["marginal_covered"], dtype=int)
        latent_joint += joint
        held_joint += prediction["joint_covered"]
        records.append({"replicate": index, "truth": truths[index].tolist(), "training_observations": training[index].tolist(),
            "posterior_mean": mean.tolist(), "posterior_covariance": covariance.tolist(), "log_evidence": float(log_evidence),
            "latent_error": error.tolist(), "latent_mahalanobis_squared": quadratic,
            "latent_marginal_intervals": np.column_stack((mean-z*std, mean+z*std)).tolist(),
            "latent_marginal_covered": marginal.tolist(), "latent_joint_covered": joint, "held_out": prediction})
    return {"scope": "empirical_iid_prior_predictive_coverage_not_fixed_truth_frequentist_calibration",
        "reference_method": "observation_space_conditioning", "noise_relation": NOISE_RELATION,
        "replicates": count, "coverage": nominal, "marginal_z": z,
        "joint_chi2_degrees_of_freedom": 2, "joint_chi2_threshold": threshold,
        "latent_marginal_counts": latent_marginal.tolist(), "latent_marginal_coverage": (latent_marginal/count).tolist(),
        "latent_joint_count": int(latent_joint), "latent_joint_coverage": latent_joint/count,
        "heldout_marginal_counts": held_marginal.tolist(), "heldout_marginal_coverage": (held_marginal/count).tolist(),
        "heldout_joint_count": int(held_joint), "heldout_joint_coverage": held_joint/count,
        "latent_marginal_wilson_95": [_wilson95(int(v), count) for v in latent_marginal],
        "latent_joint_wilson_95": _wilson95(latent_joint, count),
        "heldout_marginal_wilson_95": [_wilson95(int(v), count) for v in held_marginal],
        "heldout_joint_wilson_95": _wilson95(held_joint, count),
        "nominal_monte_carlo_standard_error": math.sqrt(nominal*(1-nominal)/count), "records": records}
