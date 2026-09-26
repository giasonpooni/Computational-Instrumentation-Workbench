"""Bounded analytic response previews; no solver, retained operation or admission.

Creation evaluates fixed models. Inspection validates saved bindings and
arithmetic only; it never evaluates a model or its derivative. Content identity
is an integrity check, not independent mathematical verification.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from . import julia_oscillator as oscillator
from .telemetry import canonical, digest

SCHEMA = "ciw.linear-response-preview.v1"
MAX_BYTES = 512 * 1024
MAX_DIMENSION = 16
PROFILES = ("scalar-square.v1", "affine.v1", "nonlinear-vector.v1",
            "composed-vector.v1", "absolute.v1", "oscillator-rate.v1", "oscillator-energy.v1")
AUTHORITY = {
    "kind": "analytic_educational_preview", "execution_id": None,
    "result_id": None, "verification_id": None, "provider_execution": "not_performed",
    "physical_validation": "not_established", "measurement_uncertainty": "not_declared",
    "state_admission": "not_performed", "hardware_actuation": "not_performed",
}
_CONVENTION = "Column vectors; Jacobian rows follow outputs and columns follow inputs; scalar derivative is 1 x 1."


def _keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("Linear response requires exactly the declared fields")


def _finite(value, bound=1e100):
    try:
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > bound:
            raise ValueError("Linear response values must be finite and within the declared domain")
        return float(value)
    except OverflowError as exc:
        raise ValueError("Linear response number exceeds its finite domain") from exc


def _vector(value, size=None, bound=1e100):
    if type(value) is not list or not 1 <= len(value) <= MAX_DIMENSION or (size is not None and len(value) != size):
        raise ValueError("Linear response vector shape differs from coordinate order")
    return [_finite(item, bound) for item in value]


def _bounded(value):
    try:
        raw = canonical(value)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Linear response must be bounded finite JSON") from exc
    if len(raw) > MAX_BYTES:
        raise ValueError("Linear response exceeds the byte budget")
    return raw


def _descriptor(profile):
    if type(profile) is not str or profile not in PROFILES:
        raise ValueError("Unsupported linear response profile")
    scalar = profile in {"scalar-square.v1", "absolute.v1"}
    physical = profile.startswith("oscillator-")
    energy = profile == "oscillator-energy.v1"
    inputs = ["q", "v"] if physical else (["x"] if scalar else ["x1", "x2"])
    outputs = (["E"] if energy else ["q_dot", "v_dot"]) if physical else (["f"] if scalar else ["y1", "y2"])
    in_units = ["m", "m/s"] if physical else ["1"] * len(inputs)
    out_units = (["J"] if energy else ["m/s", "m/s^2"]) if physical else ["1"] * len(outputs)
    equations = {
        "scalar-square.v1": ["f(x) = x^2", "J(x) = [2*x]"],
        "affine.v1": ["F(x) = [[2,1],[-1,3]] @ x + [5,-2]", "J = [[2,1],[-1,3]]"],
        "nonlinear-vector.v1": ["F(x1,x2) = [x1^2+x2, x1*x2]", "J(x) = [[2*x1,1],[x2,x1]]"],
        "composed-vector.v1": ["F(x) = [x1^2+x2, x1*x2]; G(u) = [u1^2,u2]",
                               "J_(G o F)(x) = J_G(F(x)) @ J_F(x)"],
        "absolute.v1": ["f(x) = abs(x)", "J(x) = [sign(x)] for x != 0; undefined at zero"],
        "oscillator-rate.v1": ["z_dot = A @ z; A = [[0,1],[-omega_0^2,-2*gamma]]",
                                "delta_z_dot = A @ delta_z at fixed model parameters",
                                "z(t1) = z(t0) + integral(z_dot(t), t0, t1); integration requires a declared method"],
        "oscillator-energy.v1": ["E = 0.5*m*(v^2+omega_0^2*q^2)", "J_E = [[m*omega_0^2*q, m*v]]",
                                  "remainder = 0.5*m*(delta_v^2+omega_0^2*delta_q^2)"],
    }
    return {
        "revision": profile, "input_order": inputs, "output_order": outputs,
        "input_units": in_units, "output_units": out_units,
        "jacobian_units": [[f"({out})/({inp})" for inp in in_units] for out in out_units],
        "frame": "oscillator-state" if physical else "dimensionless-cartesian",
        "time_basis": "declared_simulation_time_s_at_initial_state" if physical else "not_applicable",
        "derivative_method": "analytic_code_owned", "derivative_source": "ciw.linear_response:" + profile,
        "scope": "exact_affine_change" if profile in {"affine.v1", "oscillator-rate.v1"} else "local_first_order",
        "map_class": "affine_nonzero_offset" if profile == "affine.v1" else (
            "linear_state_to_rate" if profile == "oscillator-rate.v1" else "nonlinear"),
        "convention": _CONVENTION, "equations": equations[profile],
        "limits": [
            "Fixed analytic profile; content integrity does not establish numerical or physical correctness.",
            "Componentwise output tolerances describe this comparison; no mixed-unit aggregate norm.",
            "No inverse recovery or global validity radius is inferred from a local response.",
            ("Baseline and changed initial states must satisfy ciw.julia-oscillator-source.v1; parameters remain fixed."
             if physical else "Each base and changed coordinate lies in [-100,100]; each change lies in [-200,200]."),
            ("State-to-rate A is not a finite-time state transition; this preview performs no integration."
             if physical else "Only the declared polynomial fixtures have the illustrated polynomial remainder rates."),
        ],
    }


def make_request(profile, base_point, delta, *, source=None):
    """Build an explicit ordered request; supplied JSON uses these same fields."""
    meta = _descriptor(profile)
    request = {
        "profile": profile, "base_point": deepcopy(base_point), "delta": deepcopy(delta),
        "input_order": meta["input_order"], "input_units": meta["input_units"],
        "frame": meta["frame"], "anchor": "initial_state" if profile.startswith("oscillator-") else "declared_point",
        "source": deepcopy(source),
    }
    return _validate_request(request)[0]


def _validate_request(request):
    _bounded(request)
    _keys(request, {"profile", "base_point", "delta", "input_order", "input_units", "frame", "anchor", "source"})
    meta = _descriptor(request["profile"])
    for key in ("input_order", "input_units", "frame"):
        if request[key] != meta[key]:
            raise ValueError("Input coordinate order, units or frame differ from the profile")
    physical = request["profile"].startswith("oscillator-")
    expected_anchor = "initial_state" if physical else "declared_point"
    if request["anchor"] != expected_anchor:
        raise ValueError("Unsupported linear response anchor")
    base = _vector(request["base_point"], len(meta["input_order"]), 100)
    delta = _vector(request["delta"], len(base), 200)
    changed = [a + b for a, b in zip(base, delta)]
    _vector(changed, len(base), 100)
    if request["profile"] == "absolute.v1" and base == [0.0]:
        raise ValueError("abs(x) is not differentiable at zero; no derivative claim is available")
    if physical:
        source = oscillator.validate_source(canonical(request["source"]))
        if base != [source["initial_state"]["q0_m"], source["initial_state"]["v0_m_s"]]:
            raise ValueError("Base point differs from the bound initial_state anchor")
        altered = deepcopy(source)
        altered["initial_state"] = dict(zip(("q0_m", "v0_m_s"), changed))
        oscillator.validate_source(canonical(altered))
        binding = {"source_digest": digest(source), "changed_source_digest": digest(altered),
                   "fixed_parameters": deepcopy(source["model"]), "anchor": expected_anchor,
                   "source_class": "declared_simulation_source", "source_operation_id": source["operation_id"],
                   "integration": {"performed": False, "declared_solver": deepcopy(source["solver"]),
                                   "declared_time_s": deepcopy(source["time_s"]),
                                   "trajectory_execution_ref": None}}
    else:
        if request["source"] is not None:
            raise ValueError("Elementary profiles cannot import a source or executable")
        binding = {"source_digest": digest(meta), "changed_source_digest": None,
                   "fixed_parameters": {"A": [[2, 1], [-1, 3]], "b": [5, -2]} if request["profile"] == "affine.v1" else {},
                   "anchor": expected_anchor, "source_class": "code_owned_analytic_profile",
                   "source_operation_id": None, "integration": None}
    return deepcopy(request), meta, binding


def response_arithmetic(baseline_output, jacobian, delta, model_output):
    """Scale and sum caller-supplied numbers; this does not validate a derivative.

    Rectangular and singular matrices are allowed. Coordinate/unit semantics are
    supplied by the fixed-profile layer, not inferred by this arithmetic helper.
    """
    y0, dx = _vector(baseline_output), _vector(delta)
    y1 = _vector(model_output, len(y0))
    if type(jacobian) is not list or len(jacobian) != len(y0):
        raise ValueError("Jacobian row shape differs from outputs")
    matrix = [_vector(row, len(dx)) for row in jacobian]
    contributions = [[a * b for a, b in zip(row, dx)] for row in matrix]
    predicted = [math.fsum(row) for row in contributions]
    output = [a + b for a, b in zip(y0, predicted)]
    actual = [a - b for a, b in zip(y1, y0)]
    residual = [a - b for a, b in zip(y1, output)]
    for row in contributions:
        _vector(row)
    for values in (predicted, output, actual, residual):
        _vector(values)
    return {"baseline_output": y0, "jacobian": matrix, "delta": dx,
            "contributions": contributions, "predicted_delta": predicted,
            "predicted_output": output, "model_output": y1,
            "actual_delta": actual, "residual": residual}


def _evaluate(profile, point, parameters):
    if profile == "scalar-square.v1":
        return [point[0] ** 2]
    if profile == "absolute.v1":
        return [abs(point[0])]
    x, y = point
    if profile == "affine.v1":
        return [2*x + y + 5, -x + 3*y - 2]
    if profile == "nonlinear-vector.v1":
        return [x*x + y, x*y]
    if profile == "composed-vector.v1":
        return [(x*x + y)**2, x*y]
    omega, gamma, mass = (parameters[key] for key in ("omega_0_rad_s", "gamma_s_inv", "mass_kg"))
    if profile == "oscillator-rate.v1":
        return [y, -omega**2*x - 2*gamma*y]
    return [0.5*mass*(y*y + omega**2*x*x)]


def _jacobian(profile, point, parameters):
    if profile == "scalar-square.v1":
        return [[2*point[0]]]
    if profile == "absolute.v1":
        if point[0] == 0:
            raise ValueError("abs(x) is not differentiable at zero")
        return [[1.0 if point[0] > 0 else -1.0]]
    x, y = point
    if profile == "affine.v1":
        return [[2, 1], [-1, 3]]
    if profile == "nonlinear-vector.v1":
        return [[2*x, 1], [y, x]]
    if profile == "composed-vector.v1":
        # J_G is evaluated at F(x), with multiplication J_G(F(x)) @ J_F(x).
        u1 = x*x + y
        return [[2*u1*2*x, 2*u1], [y, x]]
    omega, gamma, mass = (parameters[key] for key in ("omega_0_rad_s", "gamma_s_inv", "mass_kg"))
    return [[0, 1], [-omega**2, -2*gamma]] if profile == "oscillator-rate.v1" else [[mass*omega**2*x, mass*y]]


def _comparison(calculation):
    tolerances = [1e-12 + 1e-12*abs(y) for y in calculation["model_output"]]
    dx = calculation["delta"]
    quotient = None
    if len(dx) == len(calculation["baseline_output"]) == 1 and dx[0] != 0:
        quotient = [calculation["actual_delta"][0] / dx[0]]
        _vector(quotient)
    return {"componentwise_tolerance": tolerances,
            "within_tolerance": [abs(r) <= t for r, t in zip(calculation["residual"], tolerances)],
            "finite_change_ratio": quotient}


def preview(request):
    """Explicitly evaluate an analytic preview; no provider is invoked."""
    request, meta, binding = _validate_request(request)
    x, dx = request["base_point"], request["delta"]
    changed = [a + b for a, b in zip(x, dx)]
    profile, params = request["profile"], binding["fixed_parameters"]
    calculation = response_arithmetic(_evaluate(profile, x, params), _jacobian(profile, x, params), dx,
                                      _evaluate(profile, changed, params))
    calculation.update(base_point=deepcopy(x), changed_point=changed)
    calculation.update(_comparison(calculation))
    record = {"schema": SCHEMA, "request": request, "request_digest": digest(request),
              "profile": meta, "binding": binding, "calculation": calculation,
              "authority": deepcopy(AUTHORITY)}
    record["preview_id"] = digest(record)
    return inspect_preview(record)


def inspect_preview(record):
    """Check content, declarations and accounting without model evaluation.

    This detects inconsistent bindings and arithmetic, not an adversary who
    rewrites all content and digests consistently. Recompute explicitly to
    compare stored model outputs or derivatives against the code-owned profile.
    """
    _bounded(record)
    _keys(record, {"schema", "request", "request_digest", "profile", "binding", "calculation", "authority", "preview_id"})
    if record["schema"] != SCHEMA or record["preview_id"] != digest({k: v for k, v in record.items() if k != "preview_id"}):
        raise ValueError("Linear response preview identity differs")
    request, meta, binding = _validate_request(record["request"])
    if (record["request_digest"] != digest(request) or canonical(record["profile"]) != canonical(meta)
            or canonical(record["binding"]) != canonical(binding)):
        raise ValueError("Linear response source, profile or request binding differs")
    if canonical(record["authority"]) != canonical(AUTHORITY):
        raise ValueError("Linear response preview authority differs")
    calc = record["calculation"]
    _keys(calc, {"baseline_output", "jacobian", "delta", "contributions", "predicted_delta", "predicted_output",
                 "model_output", "actual_delta", "residual", "base_point", "changed_point",
                 "componentwise_tolerance", "within_tolerance", "finite_change_ratio"})
    _vector(calc["baseline_output"], len(meta["output_order"]))
    if calc["base_point"] != request["base_point"] or calc["delta"] != request["delta"]:
        raise ValueError("Linear response calculation anchor differs")
    expected = response_arithmetic(calc["baseline_output"], calc["jacobian"], request["delta"], calc["model_output"])
    expected.update(base_point=request["base_point"], changed_point=[a+b for a, b in zip(request["base_point"], request["delta"])])
    expected.update(_comparison(expected))
    # Canonical equality also distinguishes booleans from numeric 0/1.
    if canonical(calc) != canonical(expected):
        raise ValueError("Linear response contribution or comparison accounting differs")
    return deepcopy(record)


def save_preview(path, record):
    """Save new canonical bytes, refusing to overwrite an earlier preview."""
    raw = canonical(inspect_preview(record))
    with Path(path).open("xb") as handle:
        handle.write(raw)


def _load_json(path):
    with Path(path).open("rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    if not 1 <= len(raw) <= MAX_BYTES:
        raise ValueError("Linear response JSON exceeds its byte budget")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Linear response input must be bounded UTF-8 JSON") from exc
    _bounded(value)
    return raw, value


def load_request(path):
    return _validate_request(_load_json(path)[1])[0]


def load_preview(path):
    raw, record = _load_json(path)
    if raw != canonical(record):
        raise ValueError("Saved linear response preview bytes must be canonical")
    return inspect_preview(record)


def render_preview(record, details=False):
    """Compact and expanded views of the same saved calculation."""
    record = inspect_preview(record)
    p, c = record["profile"], record["calculation"]
    lines = [f"Profile: {p['revision']} | scope: {p['scope']}",
             f"State: {c['base_point']} | variation: {c['delta']}",
             "Scale each change and add its contributions by output row.",
             "Input change | output contributions"]
    for j, name in enumerate(p["input_order"]):
        contributions = "; ".join(f"{out}: {c['contributions'][i][j]:.9g} {p['output_units'][i]}"
                                  for i, out in enumerate(p["output_order"]))
        lines.append(f"{name}: {c['delta'][j]:.9g} {p['input_units'][j]} | {contributions}")
    lines.append("Output | baseline | predicted | model-evaluated | residual | comparison tolerance")
    for i, name in enumerate(p["output_order"]):
        values = " | ".join(f"{c[key][i]:.9g}" for key in ("baseline_output", "predicted_output", "model_output", "residual", "componentwise_tolerance"))
        lines.append(f"{name} ({p['output_units'][i]}) | {values}")
    lines.extend([f"Method: {p['derivative_method']} | evidence: analytic educational preview",
                  f"Model binding: {record['binding']['source_digest']}",
                  "Limits: declared profile domain; no provider execution or physical validation."])
    if p["revision"] == "oscillator-rate.v1":
        lines.append("A maps state to rate, not state at a later time. No integration performed.")
    if details:
        lines.extend(["", json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)])
    return "\n".join(lines)
