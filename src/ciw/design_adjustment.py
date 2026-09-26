"""Exact certificates for one bounded scalar linearized design problem.

This reference checker is not a registered SCR operation, formal proof, or SP1
guest. Its optimality claims concern the declared local quadratic objective.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path

from . import exact_response as exact
from .telemetry import canonical, digest

SCHEMA = "ciw.design-adjustment-check.v1"
_NAMES = ("x", "target", "regularization", "lower", "upper", "epsilon")
_VALUES = ("delta", "lower_multiplier", "upper_multiplier", "predicted_output",
           "nonlinear_output", "objective", "lower_bound", "gap")
_CLAIMS = ("evaluation", "feasibility", "dual_feasibility", "bounded_suboptimality", "exact_optimality")
AUTHORITY = {**exact.AUTHORITY, "kind": "local_exact_design_check",
             "claim_scope": "fixed_linearized_scalar_quadratic_problem",
             "nonlinear_optimality": "not_established"}


def specification():
    return {
        "profile": "scalar-square-design.v1", "checker": "ciw.design_adjustment/reference-v1",
        "model": "f(x)=x^2", "response": "y0=x^2; j=2*x",
        "objective": "(y0+j*delta-target)^2+regularization*delta^2",
        "constraints": "lower<=delta<=upper; regularization>0",
        "dual": "L=c+mu_lower*lower-mu_upper*upper-(2*b-mu_lower+mu_upper)^2/(4*H)",
        "coefficients": "H=j^2+regularization; b=j*(y0-target); c=(y0-target)^2",
        "policy": "exact_evaluation; feasible_primal; nonnegative_multipliers; 0<=U-L<=epsilon",
        "arithmetic": "exact_rational_python_integers", "input_limb_limit": exact.INPUT_LIMB_LIMIT,
        "output_limb_limit": exact.OUTPUT_LIMB_LIMIT,
        "domain": "abs(x)<=10; abs(target)<=100; 1/1000000<=regularization<=100; -1<=lower<=upper<=1; 0<=epsilon<=1",
        "input_order": ["x"], "output_order": ["f"], "input_units": ["1"], "output_units": ["1"],
        "normalization": "dimensionless_example_coordinates", "frame": "dimensionless-cartesian",
        "clock": "not_applicable", "formal_soundness": "documented_convex_duality_not_formally_verified",
        "nonlinear_check": "report_(x+delta)^2_separately; no_nonlinear_optimality_claim",
    }


def _ratio(value):
    return exact._rational(value, exact.OUTPUT_LIMB_LIMIT)


def validate_problem(problem):
    exact._bounded(problem)
    exact._keys(problem, {"specification_digest", *_NAMES})
    if problem["specification_digest"] != digest(specification()):
        raise ValueError("Design specification binding differs")
    x, target, reg, lower, upper, epsilon = (
        exact._rational(problem[name], exact.INPUT_LIMB_LIMIT) for name in _NAMES)
    if (abs(x) > 10 or abs(target) > 100 or not Fraction(1, 1_000_000) <= reg <= 100
            or not -1 <= lower <= upper <= 1 or not 0 <= epsilon <= 1):
        raise ValueError("Design problem is outside the declared normalized scalar domain")
    return deepcopy(problem)


def make_problem(x, target, regularization, lower, upper, epsilon):
    return validate_problem({"specification_digest": digest(specification()),
                             **dict(zip(_NAMES, (x, target, regularization, lower, upper, epsilon)))})


def _terms(problem):
    x, target, reg, lower, upper, eps = (_ratio(problem[name]) for name in _NAMES)
    y0, j = x*x, 2*x
    return x, target, reg, lower, upper, eps, y0, j, j*j+reg, j*(y0-target), (y0-target)**2


def _candidate(candidate):
    exact._bounded(candidate)
    exact._keys(candidate, {"problem", *_VALUES})
    validate_problem(candidate["problem"])
    for key in _VALUES:
        _ratio(candidate[key])
    if abs(_ratio(candidate["delta"])) > 100 or any(
            abs(_ratio(candidate[key])) > 1_000_000 for key in ("lower_multiplier", "upper_multiplier")):
        raise ValueError("Design candidate exceeds its value budget")
    return deepcopy(candidate)


def make_candidate(problem, delta, lower_multiplier=None, upper_multiplier=None):
    """Explicit certificate preparation; it does not establish acceptance."""
    problem = validate_problem(problem)
    x, target, reg, lo, hi, _, y0, j, h, b, c = _terms(problem)
    d = _ratio(delta)
    if (lower_multiplier is None) != (upper_multiplier is None):
        raise ValueError("Supply both multipliers or neither")
    if lower_multiplier is None:
        gradient = 2*(h*d+b)
        ml = gradient if d == lo and gradient >= 0 else Fraction(0)
        mu = -gradient if d == hi and gradient < 0 else Fraction(0)
    else:
        ml, mu = _ratio(lower_multiplier), _ratio(upper_multiplier)
    predicted = y0+j*d
    objective = (predicted-target)**2+reg*d*d
    lower_bound = c+ml*lo-mu*hi-(2*b-ml+mu)**2/(4*h)
    values = (d, ml, mu, predicted, (x+d)**2, objective, lower_bound, objective-lower_bound)
    return _candidate({"problem": problem,
                       **{name: exact._pair(value) for name, value in zip(_VALUES, values)}})


def _bindings(candidate):
    ids = {"specification_digest": digest(specification()), "candidate_digest": digest(candidate)}
    return {**ids, "statement_digest": digest({**ids, "claims": list(_CLAIMS)})}


def check_candidate(candidate):
    candidate = _candidate(candidate)
    x, target, reg, lo, hi, eps, y0, j, h, b, c = _terms(candidate["problem"])
    d, ml, mu, predicted, nonlinear, reported_u, reported_l, reported_gap = (
        _ratio(candidate[name]) for name in _VALUES)
    # Expanded objective independently checks the producer's residual-squared form.
    upper = h*d*d+2*b*d+c
    s = 2*b-ml+mu
    lower = c+ml*lo-mu*hi-s*s/(4*h)
    gap = upper-lower
    evaluation = (predicted == y0+j*d and nonlinear == x*x+2*x*d+d*d
                  and reported_u == upper and reported_l == lower and reported_gap == gap)
    feasible, dual_feasible = lo <= d <= hi, ml >= 0 and mu >= 0
    common = evaluation and feasible and dual_feasible
    checks = {"evaluation": evaluation, "feasibility": feasible, "dual_feasibility": dual_feasible,
              "bounded_suboptimality": common and 0 <= gap <= eps,
              "exact_optimality": common and 2*(h*d+b)-ml+mu == 0
              and ml*(d-lo) == 0 and mu*(hi-d) == 0}
    report = {"schema": SCHEMA, "specification": specification(), "candidate": candidate,
              "bindings": _bindings(candidate), "checks": checks,
              "status": "accepted" if checks["bounded_suboptimality"] else "rejected",
              "authority": deepcopy(AUTHORITY)}
    report["report_id"] = digest(report)
    return inspect_check(report)


def inspect_check(report):
    """Structural retained-report inspection, never fresh certificate checking."""
    exact._bounded(report)
    exact._keys(report, {"schema", "specification", "candidate", "bindings", "checks",
                        "status", "authority", "report_id"})
    if report["schema"] != SCHEMA or report["report_id"] != digest(
            {k: v for k, v in report.items() if k != "report_id"}):
        raise ValueError("Design report content identity differs")
    candidate = _candidate(report["candidate"])
    if (canonical(report["specification"]) != canonical(specification())
            or canonical(report["bindings"]) != canonical(_bindings(candidate))
            or canonical(report["authority"]) != canonical(AUTHORITY)):
        raise ValueError("Design report specification, statement or authority differs")
    checks = report["checks"]
    exact._keys(checks, _CLAIMS)
    if any(type(v) is not bool for v in checks.values()):
        raise ValueError("Retained design check outcomes must be booleans")
    if ((checks["bounded_suboptimality"] and not all(checks[k] for k in _CLAIMS[:3]))
            or (checks["exact_optimality"] and not checks["bounded_suboptimality"])
            or report["status"] != ("accepted" if checks["bounded_suboptimality"] else "rejected")):
        raise ValueError("Retained design outcome dependencies differ")
    return deepcopy(report)


def load_candidate(path):
    return _candidate(exact._load_json(path)[1])


def load_problem(path):
    return validate_problem(exact._load_json(path)[1])


def save_check(path, report):
    raw = canonical(inspect_check(report))
    with Path(path).open("xb") as handle:
        handle.write(raw)


def load_check(path):
    raw, value = exact._load_json(path)
    if raw != canonical(value):
        raise ValueError("Design report bytes must be canonical")
    return inspect_check(value)


def render_check(report, details=False):
    report = inspect_check(report)
    c = report["candidate"]
    lines = ["Retained scalar design checker report (not freshly rechecked)",
             f"Reported outcome: {report['status']}"]
    for name in ("delta", "predicted_output", "nonlinear_output", "objective", "lower_bound", "gap"):
        v = c[name]
        lines.append(f"{name}: {v['numerator']}/{v['denominator']}")
    lines.extend(f"{k}: {'passed' if v else 'failed'} (retained outcome)" for k, v in report["checks"].items())
    lines.extend(["Scope: exact rational linearized quadratic objective; nonlinear output is evaluated separately.",
                  "No SP1 proof, formal verification, nonlinear optimality or actuation authority.",
                  f"Statement binding: {report['bindings']['statement_digest']}"])
    if details:
        lines.extend(["", json.dumps(report, indent=2, allow_nan=False)])
    return "\n".join(lines)
