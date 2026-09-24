"""Independent two-capacity thermal and Gaussian reference, without Julia calls.

The operational Julia model is generated symbolically. This reference instead
constructs the physical conductance matrix directly and exponentiates its
capacity-scaled symmetric eigensystem. It does not consume Julia matrices.
"""
from __future__ import annotations

import math

import numpy as np


def model_matrices(model, sample_interval_s):
    """Analytical RC matrices and exact constant-input ZOH in declared order."""
    capacities = np.asarray(model["capacities_j_per_k"], dtype=float)
    g, ga = model["conductances_w_per_k"]
    conductance = np.array([[g, -g], [-g, g + ga]], dtype=float)
    root = np.sqrt(capacities)
    symmetric = -conductance / np.outer(root, root)
    rates, vectors = np.linalg.eigh(symmetric)
    if np.any(rates >= 0):
        raise ValueError("Thermal reference requires a dissipative two-capacity model")
    dt = float(sample_interval_s)
    transition = (vectors * np.exp(rates * dt)) @ vectors.T
    integral = (vectors * (np.expm1(rates * dt) / rates)) @ vectors.T
    ad = transition * root[None, :] / root[:, None]
    integral = integral * root[None, :] / root[:, None]
    a = -conductance / capacities[:, None]
    b = np.array([[1 / capacities[0], 0], [0, ga / capacities[1]]])
    return {"A": a.tolist(), "B": b.tolist(), "C": model["observation_matrix"],
            "D": [[0.0, 0.0], [0.0, 0.0]], "Ad": ad.tolist(),
            "Bd": (integral @ b).tolist(), "sample_interval_s": dt}


def _symmetric(value):
    return (value + value.T) * .5


def correction(mean, covariance, observation, h, r):
    """Observation-space conditioning with Joseph covariance, including dropout."""
    indices = [i for i, value in enumerate(observation) if value is not None]
    mask = sum(1 << i for i in indices)
    if not indices:
        return {"available_mask": mask, "innovation": [], "innovation_covariance": [],
                "gain": [[], []], "posterior_mean": mean.tolist(),
                "posterior_covariance": covariance.tolist(), "nis": None,
                "observation_rank": 0, "innovation_condition": None}
    selected = h[indices, :]
    noise = r[np.ix_(indices, indices)]
    innovation = np.asarray([observation[i] for i in indices]) - selected @ mean
    innovation_cov = _symmetric(selected @ covariance @ selected.T + noise)
    gain = np.linalg.solve(innovation_cov, selected @ covariance).T
    posterior_mean = mean + gain @ innovation
    residual = np.eye(2) - gain @ selected
    posterior_cov = _symmetric(residual @ covariance @ residual.T + gain @ noise @ gain.T)
    return {"available_mask": mask, "innovation": innovation.tolist(),
            "innovation_covariance": innovation_cov.tolist(), "gain": gain.tolist(),
            "posterior_mean": posterior_mean.tolist(),
            "posterior_covariance": posterior_cov.tolist(),
            "nis": float(innovation @ np.linalg.solve(innovation_cov, innovation)),
            "observation_rank": int(np.linalg.matrix_rank(selected)),
            "innovation_condition": float(np.linalg.cond(innovation_cov))}


def selection_reference(prior_covariance, h, r, selection, sensor_order):
    """Four-subset exhaustive optimum; utilities precede future observations."""
    candidates = []
    logdet_prior = np.linalg.slogdet(prior_covariance)[1]
    for mask in range(4):
        indices = [i for i in range(2) if mask & (1 << i)]
        cost = sum(selection["costs"][i] for i in indices)
        observation = [0.0 if i in indices else None for i in range(2)]
        cov = np.asarray(correction(np.zeros(2), prior_covariance, observation, h, r)["posterior_covariance"])
        gain = .5 * (logdet_prior - np.linalg.slogdet(cov)[1])
        if mask == 0:
            gain = 0.0
        candidates.append({"mask": mask, "sensors": [sensor_order[i] for i in indices],
                           "cost": cost, "feasible": cost <= selection["budget"] and len(indices) >= selection["minimum_sensors"],
                           "posterior_covariance": cov.tolist(), "information_gain_nats": float(gain)})
    feasible = [row for row in candidates if row["feasible"]]
    maximum = max((row["information_gain_nats"] for row in feasible), default=None)
    selected = next((row for row in feasible if row["information_gain_nats"] >= maximum - selection["tie_tolerance_nats"]), None)
    return {"basis": "next_predicted_covariance_without_future_observations",
            "prior_covariance": prior_covariance.tolist(), "candidates": candidates,
            "status": "optimal" if selected is not None else "infeasible",
            "selected_mask": None if selected is None else selected["mask"],
            "objective_nats": None if selected is None else selected["information_gain_nats"]}


def reference(request):
    """Replay only the declared inference inputs; no source truth is accepted."""
    from .thermal_contract import validate_request
    validate_request(request)
    matrices = model_matrices(request["model"], request["sample_interval_s"])
    ad, bd = np.asarray(matrices["Ad"]), np.asarray(matrices["Bd"])
    h = np.asarray(request["model"]["observation_matrix"], dtype=float)
    q, r = np.asarray(request["process_noise_covariance"]), np.asarray(request["observation_noise_covariance"])
    mean = np.asarray(request["prior"]["mean"], dtype=float)
    covariance = np.asarray(request["prior"]["covariance"], dtype=float)
    trace = []
    for index, (inputs, observation) in enumerate(zip(request["inputs"], request["observations"])):
        mean = ad @ mean + bd @ np.asarray(inputs)
        covariance = _symmetric(ad @ covariance @ ad.T + q)
        row = {"index": index, "predicted_mean": mean.tolist(), "predicted_covariance": covariance.tolist()}
        row.update(correction(mean, covariance, observation, h, r))
        trace.append(row)
        mean, covariance = np.asarray(row["posterior_mean"]), np.asarray(row["posterior_covariance"])
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("Thermal reference exceeded finite arithmetic")
    predicted_covariance = _symmetric(ad @ covariance @ ad.T + q)
    return {"model": matrices,
            "observer": {"ordering": "predict_then_correct",
                         "noise_assumption": "process_and_measurement_noise_independent_across_steps_and_of_prior",
                         "trace": trace, "final_mean": mean.tolist(), "final_covariance": covariance.tolist()},
            "selection": selection_reference(predicted_covariance, h, r, request["selection"], request["model"]["sensor_order"])}


def evaluate(source, observer):
    """Synthetic diagnostic errors, kept separate from inference and planning."""
    request, evidence = source["request"], source["evaluation"]
    truth = np.asarray(evidence["states"])
    estimated = np.asarray([row["posterior_mean"] for row in observer["trace"]])
    held = evidence["held_out"]
    matrices = model_matrices(request["model"], request["sample_interval_s"])
    ad, bd = np.asarray(matrices["Ad"]), np.asarray(matrices["Bd"])
    mean = ad @ np.asarray(observer["final_mean"]) + bd @ np.asarray(held["input"])
    covariance = _symmetric(ad @ np.asarray(observer["final_covariance"]) @ ad.T + np.asarray(request["process_noise_covariance"]))
    h, r = np.asarray(request["model"]["observation_matrix"]), np.asarray(request["observation_noise_covariance"])
    innovation = np.asarray(held["observations"]) - h @ mean
    predictive_covariance = _symmetric(h @ covariance @ h.T + r)
    return {"basis": "retained_synthetic_truth_and_independent_held_out_observation",
            "physical_validation": "not_established", "trajectory_rmse_k": np.sqrt(np.mean((estimated-truth)**2, axis=0)).tolist(),
            "final_error_k": (estimated[-1]-truth[-1]).tolist(),
            "held_out": {"predicted_mean": (h @ mean).tolist(), "predictive_covariance": predictive_covariance.tolist(),
                         "residual_k": innovation.tolist(), "nis": float(innovation @ np.linalg.solve(predictive_covariance, innovation))},
            "coverage_claim": "not_estimated_from_one_trajectory"}
