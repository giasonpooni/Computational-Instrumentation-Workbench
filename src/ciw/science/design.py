"""Active experiment design: choose the next experiment by expected information gain.

A design problem (``ciw.design-problem.v1``) declares competing hypotheses
about what explains a discrepancy, candidate experiments (coupon geometry,
cost, risk, calibration burden, availability) and a sampling plan of fiducial
pairs. For every candidate the geometry kernel predicts each hypothesis's
observations; the expected information gain ``I(H; Y)`` between the
hypothesis and the noisy observations is estimated by seeded Monte Carlo
(log-sum-exp over the Gaussian mixture), and pairwise separations in units of
the per-pair measurement standard deviation show which hypotheses a candidate
cannot tell apart.

Hypotheses are (observable, curvature scale) pairs: an instrument reporting
``camera_chord`` or ``intrinsic_distance`` on a surface whose Gaussian
curvature is the declared one times a factor. Curvature scaling is defined for
constant-curvature surfaces only (a sphere's radius scales by ``1/sqrt(f)``;
planes and cylinders have ``K = 0`` and are unchanged).

The ranking is a computation over declared inputs. Costs, risks and priors are
the author's declarations, not discovered facts; the output lists them next to
the recommendation so a reviewer can disagree with them explicitly.
"""
from __future__ import annotations

from copy import deepcopy
import itertools
import math
from typing import Any

import numpy as np

from ._common import Refusal, content_identity, finite, integer, mapping, plain, require_keys, text
from .geometry import Surface, chord_distance, geodesic, surface_from_json, unit_direction
from .observation import intrinsic_distance
from .units import Quantity
from .vocabulary import OBSERVABLES

SCHEMA = "ciw.design-problem.v1"
MAX_CANDIDATES = 32
MAX_HYPOTHESES = 8
CENTERS = {"plane": [0.0, 0.0], "cylinder": [0.0, 0.0], "sphere": [math.pi / 2, 0.0]}
CONFOUNDED_SIGMA = 3.0


def validate_problem(problem: Any) -> dict:
    problem = require_keys(mapping(problem, "design problem"), "design problem",
                           {"schema", "question", "hypotheses", "candidates", "sampling", "noise_std", "repeats",
                            "length_unit"}, {"prior", "target_pair", "monte_carlo", "risk_weight"})
    if problem["schema"] != SCHEMA:
        raise Refusal("unsupported_schema", f"Expected {SCHEMA}")
    text(problem["question"], "question")
    hypotheses = problem["hypotheses"]
    if not isinstance(hypotheses, list) or not 2 <= len(hypotheses) <= MAX_HYPOTHESES:
        raise Refusal("malformed_record", f"Declare 2..{MAX_HYPOTHESES} hypotheses")
    for item in hypotheses:
        require_keys(item, "hypothesis", {"id", "description", "observable", "curvature_scale"})
        if item["observable"] not in {"camera_chord", "intrinsic_distance"}:
            raise Refusal("observable_model_unbound", "Design hypotheses use camera_chord or intrinsic_distance",
                          allowed_in_vocabulary=sorted(OBSERVABLES))
        finite(item["curvature_scale"], "curvature_scale", minimum=0.0, exclusive_minimum=True)
    ids = [item["id"] for item in hypotheses]
    if len(set(ids)) != len(ids):
        raise Refusal("malformed_record", "Hypothesis identifiers must be unique")
    if not isinstance(problem["candidates"], list) or not 1 <= len(problem["candidates"]) <= MAX_CANDIDATES:
        raise Refusal("malformed_record", f"Declare 1..{MAX_CANDIDATES} candidates")
    for item in problem["candidates"]:
        require_keys(item, "candidate", {"candidate_id", "surface", "cost", "risk", "calibration_burden", "available"},
                     {"description"})
        if item["surface"].get("type") not in CENTERS:
            raise Refusal("unsupported_surface", "Design candidates are plane, cylinder or sphere coupons")
        for key in ("cost", "risk", "calibration_burden"):
            finite(item[key], key, minimum=0.0)
    sampling = require_keys(problem["sampling"], "sampling", {"separation", "headings"})
    Quantity.from_json(sampling["separation"], "sampling.separation")
    headings = require_keys(sampling["headings"], "sampling.headings", {"values", "unit"})
    if not isinstance(headings["values"], list) or not 1 <= len(headings["values"]) <= 64:
        raise Refusal("malformed_record", "sampling.headings needs 1..64 values")
    integer(problem["repeats"], "repeats", minimum=1, maximum=1000)
    Quantity.from_json(problem["noise_std"], "noise_std")
    if "prior" in problem:
        prior = mapping(problem["prior"], "prior")
        if set(prior) != set(ids) or abs(sum(finite(v, "prior") for v in prior.values()) - 1.0) > 1e-9:
            raise Refusal("malformed_record", "prior must assign probabilities summing to 1 to every hypothesis")
    if "target_pair" in problem and (len(problem["target_pair"]) != 2 or set(problem["target_pair"]) - set(ids)):
        raise Refusal("malformed_record", "target_pair must name two declared hypotheses")
    return problem


def _markers(surface: Surface, separation: float, headings: list[float]) -> tuple[np.ndarray, list[np.ndarray]]:
    center = np.array(CENTERS[surface.kind])
    markers = []
    for heading in headings:
        velocity = unit_direction(surface, center, heading)
        markers.append(geodesic(surface, center, velocity, separation, 200, transitions=False).u[-1])
    return center, markers


def _true_surface(surface: Surface, record: dict, scale: float, length_unit: str) -> Surface:
    if scale == 1.0 or surface.kind in {"plane", "cylinder"}:
        return surface
    if surface.kind == "sphere":
        radius = Quantity.from_json(record["radius"], "radius")
        scaled = dict(record, radius={"value": radius.value / math.sqrt(scale), "unit": radius.unit})
        return surface_from_json(scaled, length_unit)
    raise Refusal("curvature_scaling_undefined", f"Curvature scaling is not defined for {surface.kind}")


def predictions(problem: dict, candidate: dict) -> dict[str, np.ndarray]:
    unit = problem["length_unit"]
    surface = surface_from_json(candidate["surface"], unit)
    separation = Quantity.from_json(problem["sampling"]["separation"], "separation").magnitude(unit)
    headings = [Quantity(value, problem["sampling"]["headings"]["unit"]).magnitude("rad")
                for value in problem["sampling"]["headings"]["values"]]
    center, markers = _markers(surface, separation, headings)
    result = {}
    for hypothesis in problem["hypotheses"]:
        truth = _true_surface(surface, candidate["surface"], float(hypothesis["curvature_scale"]), unit)
        if hypothesis["observable"] == "camera_chord":
            values = [chord_distance(truth, center, marker) for marker in markers]
        else:
            values = [intrinsic_distance(truth, center, marker) for marker in markers]
        result[hypothesis["id"]] = np.array(values)
    return result


def expected_information_gain(means: dict[str, np.ndarray], prior: dict[str, float], sigma: float,
                              samples: int, seed: int) -> float:
    """Monte Carlo estimate of I(H; Y) in bits for a Gaussian mixture with shared isotropic noise."""
    ids = list(means)
    matrix = np.array([means[name] for name in ids])
    weights = np.array([prior[name] for name in ids])
    rng = np.random.default_rng(seed)
    total = 0.0
    for index, name in enumerate(ids):
        draws = matrix[index] + sigma * rng.standard_normal((samples, matrix.shape[1]))
        # log p(y|h') up to a shared constant
        squared = ((draws[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2) / (2 * sigma ** 2)
        log_likelihood = -squared
        mixture = np.log(weights)[None, :] + log_likelihood
        peak = mixture.max(axis=1, keepdims=True)
        log_evidence = (peak + np.log(np.exp(mixture - peak).sum(axis=1, keepdims=True)))[:, 0]
        total += weights[index] * float(np.mean(log_likelihood[:, index] - log_evidence))
    return max(0.0, total / math.log(2))


def rank(problem: dict) -> dict:
    """Rank candidates by information gain per unit of declared cost, risk and calibration burden."""
    problem = validate_problem(deepcopy(problem))
    unit = problem["length_unit"]
    ids = [item["id"] for item in problem["hypotheses"]]
    prior = problem.get("prior") or {name: 1.0 / len(ids) for name in ids}
    sigma = Quantity.from_json(problem["noise_std"], "noise_std").magnitude(unit) / math.sqrt(problem["repeats"])
    settings = problem.get("monte_carlo", {"samples": 4000, "seed": 1})
    risk_weight = float(problem.get("risk_weight", 1.0))
    target = problem.get("target_pair")
    rows = []
    for candidate in problem["candidates"]:
        means = predictions(problem, candidate)
        eig = expected_information_gain(means, prior, sigma, integer(settings["samples"], "samples", minimum=100,
                                                                     maximum=100_000), integer(settings["seed"], "seed"))
        separation = {f"{a}|{b}": float(np.linalg.norm(means[a] - means[b]) / sigma)
                      for a, b in itertools.combinations(ids, 2)}
        confounded = [pair.split("|") for pair, value in separation.items() if value < CONFOUNDED_SIGMA]
        burden = float(candidate["cost"]) + float(candidate["calibration_burden"]) + risk_weight * float(candidate["risk"])
        target_sep = None
        if target:
            key = f"{target[0]}|{target[1]}" if f"{target[0]}|{target[1]}" in separation else f"{target[1]}|{target[0]}"
            target_sep = separation[key]
        rows.append({
            "candidate_id": candidate["candidate_id"], "available": bool(candidate["available"]),
            "surface": candidate["surface"], "eig_bits": eig, "max_eig_bits": -sum(p * math.log2(p) for p in prior.values() if p > 0),
            "separation_sigma": separation, "confounded_pairs": confounded, "target_separation_sigma": target_sep,
            "burden": burden, "utility_bits_per_burden": eig / burden if burden > 0 else math.inf,
            "declared": {key: candidate[key] for key in ("cost", "risk", "calibration_burden")},
            "predictions": {name: values.tolist() for name, values in means.items()}})
    eligible = [row for row in rows if row["available"] and (target is None or row["target_separation_sigma"] >= 2 * CONFOUNDED_SIGMA)]
    rows.sort(key=lambda row: (-row["utility_bits_per_burden"], row["candidate_id"]))
    best = max(eligible, key=lambda row: (row["utility_bits_per_burden"], row["candidate_id"]), default=None)
    reasons = []
    for row in rows:
        if row is best:
            continue
        why = []
        if not row["available"]:
            why.append("unavailable")
        if target and row["target_separation_sigma"] < 2 * CONFOUNDED_SIGMA:
            why.append(f"cannot separate {target[0]} from {target[1]} ({row['target_separation_sigma']:.2f} sigma)")
        if best is not None and row["utility_bits_per_burden"] < best["utility_bits_per_burden"]:
            why.append(f"lower information per burden ({row['utility_bits_per_burden']:.3f} < "
                       f"{best['utility_bits_per_burden']:.3f} bits)")
        reasons.append({"candidate_id": row["candidate_id"], "not_selected_because": why})
    summary = None
    if best is not None:
        summary = (f"{best['candidate_id']}: {best['eig_bits']:.3f} of {best['max_eig_bits']:.3f} bits for burden "
                   f"{best['burden']:.2f}" + (f"; separates {target[0]} from {target[1]} by "
                                              f"{best['target_separation_sigma']:.0f} sigma" if target else "")
                   + (f"; still confounds {', '.join('/'.join(p) for p in best['confounded_pairs'])}"
                      if best["confounded_pairs"] else ""))
    return plain({"schema": "ciw.design-ranking.v1", "question": problem["question"],
                  "problem_identity": content_identity(problem), "prior": prior, "per_pair_sigma": sigma,
                  "unit": unit, "ranking": rows, "recommendation": best["candidate_id"] if best else None,
                  "summary": summary, "not_selected": reasons,
                  "method": "Monte Carlo mutual information (log-sum-exp), isotropic Gaussian noise, declared costs"})


def next_experiment(problem: dict, ranking: dict) -> dict | None:
    """An experiment specification for the recommended candidate (a proposal, not an instruction)."""
    if ranking["recommendation"] is None:
        return None
    candidate = next(item for item in problem["candidates"] if item["candidate_id"] == ranking["recommendation"])
    kind = candidate["surface"]["type"]
    center = CENTERS[kind]
    units = {"plane": ["m", "m"], "cylinder": ["rad", "m"], "sphere": ["rad", "rad"]}[kind]
    return {
        "schema": "ciw.experiment-spec.v1",
        "experiment_id": f"design-{candidate['candidate_id']}",
        "title": f"Discriminating experiment on {candidate['candidate_id']}",
        "hypothesis": {"statement": problem["question"],
                       "competing": [{"id": item["id"], "statement": item["description"]} for item in problem["hypotheses"]]},
        "model": {"kind": "surface-geodesic", "surface": candidate["surface"], "length_unit": problem["length_unit"]},
        "sweep": {"mode": "product", "parameters": {"heading": problem["sampling"]["headings"]}},
        "initial_conditions": {"point": {"value": center, "unit": units}, "heading": "$heading"},
        "observation": {"observable": "camera_chord", "noise_std": problem["noise_std"], "acquisition": "planned",
                        "instrument": "declared-by-protocol"},
        "solver": {"solver_id": "geometry.geodesic-rk4.v1", "steps": 400},
        "tolerances": {"abs": 1e-12, "rel": 1e-9},
        "invariants": ["closed-form-endpoint", "speed-conservation", "extrinsic-alternate"],
        "stopping": {"arclength": problem["sampling"]["separation"], "max_steps": 20000},
        "evidence": {"physical_measurements": "planned", "claims_sought": ["computed", "predicted", "verified"]},
    }


def retain(ledger, problem: dict, ranking: dict, refs: list[str] | None = None) -> dict:
    return ledger.append("ciw.science.design-recommendation.v1", {
        "ranking": ranking, "problem": problem, "recommendation": ranking["recommendation"],
        "summary": ranking["summary"]}, refs=refs or [])
