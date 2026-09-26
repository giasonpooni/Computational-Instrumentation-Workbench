"""Exact scalar-square reference checks, separate from SP1 and float previews.

Explicit checking evaluates three claims. Reopening only inspects the retained
report's structure and bindings; it cannot authenticate a historical outcome.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path

from .telemetry import canonical, digest

SCHEMA = "ciw.exact-response-check.v1"
MAX_BYTES = 64 * 1024
INPUT_LIMB_LIMIT = 1_000_000
OUTPUT_LIMB_LIMIT = 2**255 - 1
AUTHORITY = {
    "kind": "local_exact_reference_check",
    "execution_id": None, "result_id": None, "verification_id": None,
    "proof_id": None, "sp1_verification": "not_performed",
    "formal_verification": "not_performed", "physical_validation": "not_established",
    "measurement_uncertainty": "not_declared", "state_admission": "not_performed",
    "hardware_actuation": "not_performed",
    "retained_trust": "historical_report_requires_explicit_recheck",
}
_INPUTS = ("x", "delta", "radius")
_OUTPUTS = ("baseline_output", "local_multiplier", "predicted_output",
            "model_output", "residual", "error_bound")
_CLAIMS = ("arithmetic", "meaning", "approximation")


def specification():
    """Code-owned mathematical and encoding contract, never executable input."""
    return {
        "profile": "scalar-square-rational.v1",
        "checker": "ciw.exact_response/reference-v1",
        "model": "f(x)=x^2", "derivative": "f'(x)=2*x",
        "identity": "(x+u)^2=x^2+2*x*u+u^2",
        "bound": "abs(u)<=radius implies abs((x+u)^2-(x^2+2*x*u))<=radius^2",
        "justification": "documented_polynomial_identity_not_formally_verified",
        "arithmetic": "exact_rational_python_integers",
        "rational_encoding": "reduced integer numerator and positive denominator; zero is 0/1",
        "input_limb_limit": INPUT_LIMB_LIMIT,
        "output_limb_limit": OUTPUT_LIMB_LIMIT,
        "limb_limit_scope": "wire_size_guard_not_fixed_width_machine_arithmetic",
        "domain": {"coordinate_min": -100, "coordinate_max": 100,
                   "radius_min": 0, "radius_max": 100,
                   "require_entire_neighborhood_in_domain": True},
        "input_order": ["x"], "output_order": ["f"],
        "input_units": ["1"], "output_units": ["1"],
        "frame": "dimensionless-cartesian", "clock": "not_applicable",
        "policy": "all_three_claims_required; exact_equality; no_numeric_tolerance",
        "float_preview_relation": "same_mathematical_model; no_float_output_or_rounding_claim",
    }


def _keys(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("Exact response requires exactly the declared fields")


def _bounded(value):
    try:
        raw = canonical(value)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Exact response requires bounded data-only JSON") from exc
    if not 1 <= len(raw) <= MAX_BYTES:
        raise ValueError("Exact response exceeds its byte budget")
    return raw


def _rational(value, limit):
    _keys(value, {"numerator", "denominator"})
    n, d = value["numerator"], value["denominator"]
    if (type(n) is not int or type(d) is not int or abs(n) > limit
            or not 1 <= d <= limit):
        raise ValueError("Require bounded integer numerator and positive denominator")
    if math.gcd(n, d) != 1:
        raise ValueError("Rationals must be reduced; zero must be 0/1")
    return Fraction(n, d)


def _pair(value):
    result = {"numerator": value.numerator, "denominator": value.denominator}
    _rational(result, OUTPUT_LIMB_LIMIT)
    return result


def _inputs(values):
    x, d, r = (_rational(values[name], INPUT_LIMB_LIMIT) for name in _INPUTS)
    if (not all(abs(v) <= 100 for v in (x, d, r)) or r < 0
            or x-r < -100 or x+r > 100 or not -100 <= x+d <= 100):
        raise ValueError("Base, changed point and entire nonnegative-radius neighborhood must be in [-100,100]")
    return x, d, r


def _candidate(candidate):
    _bounded(candidate)
    _keys(candidate, {"specification_digest", *_INPUTS, *_OUTPUTS})
    if candidate["specification_digest"] != digest(specification()):
        raise ValueError("Exact response specification binding differs")
    _inputs(candidate)
    for name in _OUTPUTS:
        _rational(candidate[name], OUTPUT_LIMB_LIMIT)
    return deepcopy(candidate)


def make_candidate(x, delta, radius):
    """Propose exact values explicitly; this is not a checker or a proof."""
    candidate = {"specification_digest": digest(specification()),
                 "x": deepcopy(x), "delta": deepcopy(delta), "radius": deepcopy(radius)}
    a, d, r = _inputs(candidate)
    start = a**2
    multiplier = 2*a
    predicted = start + multiplier*d
    actual = (a+d)**2
    values = (start, multiplier, predicted, actual, actual-predicted, r**2)
    candidate.update({name: _pair(value) for name, value in zip(_OUTPUTS, values)})
    return _candidate(candidate)


def _bindings(candidate):
    spec_id, candidate_id = digest(specification()), digest(candidate)
    return {"specification_digest": spec_id, "candidate_digest": candidate_id,
            "statement_digest": digest({"specification_digest": spec_id,
                                        "candidate_digest": candidate_id,
                                        "claims": list(_CLAIMS)})}


def check_candidate(candidate):
    """Fresh local exact check. Well-formed false claims produce rejection.

    This uses the fixed square-model identity, not sampled residuals as evidence
    of a general bound. It produces no cryptographic or formal proof artifact.
    """
    candidate = _candidate(candidate)
    x, d, r = _inputs(candidate)
    y0, j, predicted, actual, residual, bound = (
        _rational(candidate[name], OUTPUT_LIMB_LIMIT) for name in _OUTPUTS)
    checks = {
        "arithmetic": predicted == y0+j*d and residual == actual-predicted,
        "meaning": y0 == x*x and j == x+x and actual == (x+d)*(x+d),
        "approximation": abs(d) <= r and residual == d*d and bound == r*r,
    }
    report = {"schema": SCHEMA, "specification": specification(), "candidate": candidate,
              "bindings": _bindings(candidate), "checks": checks,
              "status": "accepted" if all(checks.values()) else "rejected",
              "authority": deepcopy(AUTHORITY)}
    report["report_id"] = digest(report)
    return inspect_check(report)


def inspect_check(report):
    """Check saved structure/content only, without re-running the three claims.

    Anyone able to rewrite the entire report can reseal its hashes. This function
    therefore must not be presented as fresh acceptance or authentication.
    """
    _bounded(report)
    _keys(report, {"schema", "specification", "candidate", "bindings", "checks",
                   "status", "authority", "report_id"})
    if report["schema"] != SCHEMA or report["report_id"] != digest(
            {k: v for k, v in report.items() if k != "report_id"}):
        raise ValueError("Exact response report identity differs")
    candidate = _candidate(report["candidate"])
    if (canonical(report["specification"]) != canonical(specification())
            or canonical(report["bindings"]) != canonical(_bindings(candidate))
            or canonical(report["authority"]) != canonical(AUTHORITY)):
        raise ValueError("Exact response specification, statement or authority binding differs")
    _keys(report["checks"], _CLAIMS)
    if any(type(value) is not bool for value in report["checks"].values()):
        raise ValueError("Retained claim outcomes must be booleans")
    expected_status = "accepted" if all(report["checks"].values()) else "rejected"
    if report["status"] != expected_status:
        raise ValueError("Retained aggregate differs from retained claim outcomes")
    return deepcopy(report)


def _load_json(path):
    with Path(path).open("rb") as handle:
        raw = handle.read(MAX_BYTES+1)
    if not 1 <= len(raw) <= MAX_BYTES:
        raise ValueError("Exact response exceeds its byte budget")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("Exact response requires bounded, unique-key UTF-8 JSON") from exc
    _bounded(value)
    return raw, value


def load_candidate(path):
    return _candidate(_load_json(path)[1])


def save_check(path, report):
    raw = canonical(inspect_check(report))
    with Path(path).open("xb") as handle:
        handle.write(raw)


def load_check(path):
    raw, report = _load_json(path)
    if raw != canonical(report):
        raise ValueError("Saved exact response report bytes must be canonical")
    return inspect_check(report)


def render_check(report, details=False):
    report = inspect_check(report)
    c = report["candidate"]

    def show(name):
        pair = c[name]
        return f"{pair['numerator']}/{pair['denominator']}"

    lines = ["Retained exact square-model checker report (not freshly rechecked)",
             f"Reported outcome: {report['status']}",
             f"State: {show('x')} | variation: {show('delta')} | allowed radius: {show('radius')}",
             f"Multiply change by {show('local_multiplier')}, then add baseline {show('baseline_output')}.",
             f"Prediction: {show('predicted_output')} | model-evaluated: {show('model_output')}",
             f"Residual: {show('residual')} | requested radius bound: {show('error_bound')}"]
    lines.extend(f"{name}: {'passed' if report['checks'][name] else 'failed'} (retained outcome)"
                 for name in _CLAIMS)
    lines.extend(["Exact rational model only; no floating-point rounding or physical claim.",
                  "SP1 verification: not performed. Formal verification: not performed.",
                  f"Statement binding: {report['bindings']['statement_digest']}",
                  f"Report content identity: {report['report_id']}"])
    if details:
        lines.extend(["", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)])
    return "\n".join(lines)
