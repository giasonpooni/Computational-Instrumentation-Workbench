"""Operation contracts: request validation, committed bytes, references and comparisons.

For each worker operation this module builds the exact CIWB input and
configuration bytes from a validated model and request, decodes the committed
output, computes an independent Python reference and compares both under
declared thresholds. A comparison records actual errors; it is not a proof
of accuracy and never becomes a verification identity.
"""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np

from . import codec
from .simulation import (CompiledModel, REQUEST_SCHEMA, simulate_reference, validate_request)
from .spec import Model, SpecificationError, _inside, _keys, lower

LINEARIZATION_SCHEMA = "ciw.model-linearization-request.v1"
DESIGN_SCHEMA = "ciw.model-design-request.v1"
REQUEST_SCHEMAS = {
    "ciw.model.simulate.v1": REQUEST_SCHEMA,
    "ciw.model.linearize.v1": LINEARIZATION_SCHEMA,
    "ciw.model.symbolic.v1": LINEARIZATION_SCHEMA,
    "ciw.model.measurement-selection.v1": DESIGN_SCHEMA,
}
MAX_CANDIDATES = 64
ENUMERATION_LIMIT = 20


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise SpecificationError(f"{name} must be a finite JSON number, not a boolean")
    return float(value)


def _state_vector(model: Model, values, name):
    states = model.state_order
    if not isinstance(values, list) or len(values) != len(states):
        raise SpecificationError(f"{name} must list {len(states)} values in state order {states}")
    bounds = model.spec["validity_domain"]["bounds"]
    for symbol, value in zip(states, values):
        _finite(value, f"{name}.{symbol}")
        if symbol in bounds and not _inside(value, bounds[symbol]):
            raise SpecificationError(f"{name}.{symbol} lies outside the declared validity domain")
    return [float(value) for value in values]


def _inputs(model: Model, values):
    declared = [item["symbol"] for item in model.spec["inputs"]]
    if not isinstance(values, dict) or set(values) != set(declared):
        raise SpecificationError(f"inputs must give a constant value for exactly {declared}")
    bounds = model.spec["validity_domain"]["bounds"]
    for symbol, value in values.items():
        _finite(value, f"inputs.{symbol}")
        if symbol in bounds and not _inside(value, bounds[symbol]):
            raise SpecificationError(f"Input {symbol} lies outside the declared validity domain")
    return {key: float(value) for key, value in values.items()}


def validate_linearization(model: Model, request) -> dict:
    _keys(request, {"schema", "spec_digest", "state", "inputs", "time", "rank_rtol", "reference"},
          name="linearization request")
    if request["schema"] != LINEARIZATION_SCHEMA or request["spec_digest"] != model.digest:
        raise SpecificationError("Linearization request must use its schema and bind this model")
    _state_vector(model, request["state"], "state")
    _inputs(model, request["inputs"])
    _finite(request["time"], "time")
    if not 0 < _finite(request["rank_rtol"], "rank_rtol") < 1:
        raise SpecificationError("rank_rtol must lie in (0, 1)")
    reference = request["reference"]
    _keys(reference, {"fd_relative_step", "jacobian_rtol", "jacobian_atol"}, name="linearization reference")
    for key in reference:
        if not 0 < _finite(reference[key], f"reference.{key}") < 1:
            raise SpecificationError(f"reference.{key} must lie in (0, 1)")
    return request


def validate_design(model: Model, request) -> dict:
    _keys(request, {"schema", "spec_digest", "initial_state", "inputs", "start_time", "candidates",
                    "design_parameters", "budget", "max_count", "solver", "reference"}, name="design request")
    if request["schema"] != DESIGN_SCHEMA or request["spec_digest"] != model.digest:
        raise SpecificationError("Design request must use its schema and bind this model")
    _state_vector(model, request["initial_state"], "initial_state")
    _inputs(model, request["inputs"])
    start = _finite(request["start_time"], "start_time")
    observations = {item["symbol"] for item in model.spec["observations"]}
    candidates = request["candidates"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise SpecificationError(f"candidates must list 1..{MAX_CANDIDATES} measurements")
    seen = set()
    for index, candidate in enumerate(candidates):
        _keys(candidate, {"observation", "time", "cost"}, name=f"candidate {index}")
        if candidate["observation"] not in observations:
            raise SpecificationError(f"candidate {index} names an undeclared observation")
        if _finite(candidate["time"], "candidate time") <= start or _finite(candidate["cost"], "candidate cost") < 0:
            raise SpecificationError(f"candidate {index} needs a time after start_time and a nonnegative cost")
        key = (candidate["observation"], candidate["time"])
        if key in seen:
            raise SpecificationError("candidates must be distinct observation/time pairs")
        seen.add(key)
    domain = model.spec["validity_domain"]["independent_variable"]
    if domain is not None and not all(_inside(value, domain) for value in
                                      [start, *(item["time"] for item in candidates)]):
        raise SpecificationError("candidate times lie outside the declared validity domain")
    design = request["design_parameters"]
    parameters = {item["symbol"] for item in model.spec["parameters"]}
    if (not isinstance(design, list) or not design or len(set(design)) != len(design)
            or any(item not in parameters for item in design)):
        raise SpecificationError("design_parameters must name distinct declared parameters")
    _finite(request["budget"], "budget")
    if type(request["max_count"]) is not int or not 1 <= request["max_count"] <= len(candidates):
        raise SpecificationError("max_count must be an integer in 1..len(candidates)")
    solver = request["solver"]
    _keys(solver, {"reltol", "abstol", "maxiters", "time_limit_s"}, name="design solver")
    if not 1e-14 <= _finite(solver["reltol"], "solver.reltol") <= 1e-2:
        raise SpecificationError("solver.reltol must lie in [1e-14, 1e-2]")
    _state_vector_positive(model, solver["abstol"])
    if type(solver["maxiters"]) is not int or not 1 <= solver["maxiters"] <= 10_000_000:
        raise SpecificationError("solver.maxiters must be an integer in 1..10000000")
    if not 0 < _finite(solver["time_limit_s"], "solver.time_limit_s") <= 600:
        raise SpecificationError("solver.time_limit_s must lie in (0, 600]")
    reference = request["reference"]
    _keys(reference, {"substeps", "max_step", "fd_relative_step", "information_rtol"}, name="design reference")
    if type(reference["substeps"]) is not int or not 1 <= reference["substeps"] <= 4096:
        raise SpecificationError("reference.substeps must be 1..4096")
    span = max(float(c["time"]) for c in candidates) - start
    if not 0 < _finite(reference["max_step"], "reference.max_step") or span / reference["max_step"] > 4000:
        raise SpecificationError("reference.max_step must be positive with at most 4000 grid intervals")
    for key in ("fd_relative_step", "information_rtol"):
        if not 0 < _finite(reference[key], f"reference.{key}") < 1:
            raise SpecificationError(f"reference.{key} must lie in (0, 1)")
    design_variances(model, request)
    return request


def _state_vector_positive(model, values):
    if (not isinstance(values, list) or len(values) != len(model.state_order)
            or any(_finite(value, "abstol") <= 0 for value in values)):
        raise SpecificationError("abstol must give one positive tolerance per state")


def design_variances(model: Model, request) -> tuple[dict, dict]:
    """Prior and noise variances from declared covariances; correlations are refused, not dropped."""
    uncertainty = model.spec["uncertainty"]

    def marginal(block, symbols, what):
        if block is None:
            raise SpecificationError(f"The design objective needs a declared {what} covariance")
        index = {symbol: position for position, symbol in enumerate(block["order"])}
        missing = [symbol for symbol in symbols if symbol not in index]
        if missing:
            raise SpecificationError(f"The declared {what} covariance does not cover {missing}")
        for a in symbols:
            for b in symbols:
                if a != b and block["matrix"][index[a]][index[b]] != 0:
                    raise SpecificationError(f"The design objective uses marginal {what} variances; "
                                             f"correlated {a}/{b} would be discarded, so it is refused")
        variances = {symbol: float(block["matrix"][index[symbol]][index[symbol]]) for symbol in symbols}
        if any(value <= 0 for value in variances.values()):
            raise SpecificationError(f"The design objective needs positive {what} variances")
        return variances

    prior = marginal(uncertainty["parameters"], request["design_parameters"], "parameter")
    noise = marginal(uncertainty["measurement_noise"],
                     sorted({item["observation"] for item in request["candidates"]}), "measurement noise")
    return prior, noise


def validate_operation_request(operation: str, model: Model, request) -> dict:
    if operation == "ciw.model.simulate.v1":
        return validate_request(model, request)
    if operation in ("ciw.model.linearize.v1", "ciw.model.symbolic.v1"):
        return validate_linearization(model, request)
    if operation == "ciw.model.measurement-selection.v1":
        return validate_design(model, request)
    raise SpecificationError(f"Unsupported model operation: {operation}")


def committed_bytes(operation: str, model: Model, request: dict) -> tuple[bytes, bytes]:
    """Return (configuration, input) CIWB bytes; every result-affecting setting is explicit."""
    validate_operation_request(operation, model, request)
    lowered = lower(model)
    if operation == "ciw.model.simulate.v1":
        solver = request["solver"]
        configuration = {"algorithm": "Tsit5", "abstol": codec.F64Array(solver["abstol"]),
                         "controller": "algorithm_default", "dtmax": solver["dtmax"],
                         "initial_dt": solver["initial_dt"], "internalnorm": "ODE_DEFAULT_NORM",
                         "maxiters": solver["maxiters"], "reltol": float(solver["reltol"]),
                         "save_policy": solver["save_policy"], "span": "first_to_last_sample"}
        payload = {"lowered": lowered, "initial_state": codec.F64Array(request["initial_state"]),
                   "inputs": {k: float(v) for k, v in request["inputs"].items()},
                   "sample_times": codec.F64Array(request["sample_times"])}
    elif operation in ("ciw.model.linearize.v1", "ciw.model.symbolic.v1"):
        configuration = ({"analysis": "ControlSystemsBase ss/poles/ctrb/obsv", "differentiation": "ForwardDiff.jacobian",
                          "rank_rtol": float(request["rank_rtol"])} if operation == "ciw.model.linearize.v1" else
                         {"compile": "ModelingToolkit.mtkcompile", "jacobian": "Symbolics.jacobian",
                          "latex": "Latexify.latexify"})
        payload = {"lowered": lowered, "state": codec.F64Array(request["state"]),
                   "inputs": {k: float(v) for k, v in request["inputs"].items()}, "time": float(request["time"])}
    else:
        solver = request["solver"]
        prior, noise = design_variances(model, request)
        configuration = {"abstol": codec.F64Array(solver["abstol"]), "algorithm": "Tsit5",
                         "maxiters": solver["maxiters"], "mip_rel_gap": 0.0,
                         "objective": "maximin_prior_normalized_fisher_diagonal", "random_seed": 0,
                         "reltol": float(solver["reltol"]), "sensitivity": "ForwardDiff_through_Tsit5",
                         "solver": "HiGHS", "threads": 1, "time_limit_s": float(solver["time_limit_s"])}
        payload = {"lowered": lowered, "initial_state": codec.F64Array(request["initial_state"]),
                   "inputs": {k: float(v) for k, v in request["inputs"].items()},
                   "start_time": float(request["start_time"]),
                   "candidates": [{"cost": float(c["cost"]), "observation": c["observation"], "time": float(c["time"])}
                                  for c in request["candidates"]],
                   "design_parameters": list(request["design_parameters"]), "budget": float(request["budget"]),
                   "max_count": request["max_count"], "prior_variance": prior, "noise_variance": noise}
    return codec.encode(configuration), codec.encode(payload)


# -- independent Python references ------------------------------------------------------------

def _max_errors(actual, expected, atol, rtol):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    error = np.abs(actual - expected)
    allowed = np.asarray(atol) + rtol * np.abs(expected)
    return float(np.max(error)) if error.size else 0.0, bool(np.all(error <= allowed))


def reference_and_compare(operation: str, model: Model, request: dict, view: dict) -> dict:
    if operation == "ciw.model.simulate.v1":
        return _compare_simulation(model, request, view)
    if operation == "ciw.model.linearize.v1":
        return _compare_linearization(model, request, view)
    if operation == "ciw.model.symbolic.v1":
        return _compare_symbolic(model, request, view)
    return _compare_design(model, request, view)


def _compare_simulation(model, request, view):
    reference = simulate_reference(model, request)
    if reference["status"] != "completed":
        return {"reference": {"method": "rk4", "status": reference["status"], "detail": reference["detail"]},
                "passed": False}
    acceptance = request["acceptance"]
    states = np.asarray(view["x"], dtype=np.float64)
    ref_states = np.asarray(reference["states"], dtype=np.float64)
    atol = np.asarray(acceptance["state_atol"], dtype=np.float64)
    rtol = float(acceptance["state_rtol"])
    per_state = {}
    passed = view["t"] == request["sample_times"]
    for index, symbol in enumerate(model.state_order):
        error, ok = _max_errors(states[:, index], ref_states[:, index], atol[index], rtol)
        per_state[symbol] = {"max_abs_error": error, "unit": model.entries[symbol]["unit"], "within_threshold": ok}
        passed = passed and ok
    return {"reference": {"method": "rk4", "substeps": request["reference"]["substeps"],
                          "rhs_evaluations": reference["rhs_evaluations"]},
            "thresholds": {"state_atol": atol.tolist(), "state_rtol": rtol,
                           "rule": "|julia - rk4| <= atol + rtol*|rk4| componentwise"},
            "sample_times_exact": view["t"] == request["sample_times"],
            "states": per_state, "passed": bool(passed),
            "solver_stats": view["stats"], "retcode": view["retcode"]}


def finite_difference_linearization(model: Model, request: dict) -> dict:
    compiled = CompiledModel(model)
    x = np.asarray(request["state"], dtype=np.float64)
    inputs = {k: float(v) for k, v in request["inputs"].items()}
    t = float(request["time"])
    names = [item["symbol"] for item in model.spec["inputs"]]
    step = request["reference"]["fd_relative_step"]

    def observe(state, values):
        return np.array(list(compiled.outputs(t, state, values)[1].values()), dtype=np.float64)

    def column(f, base, index):
        h = step * max(1.0, abs(base[index]))
        up, down = base.copy(), base.copy()
        up[index] += h
        down[index] -= h
        return (f(up) - f(down)) / (2 * h)

    n, ny = len(x), len(model.spec["observations"])
    A = np.column_stack([column(lambda s: compiled.rhs(t, s, inputs), x, i) for i in range(n)])
    C = (np.column_stack([column(lambda s: observe(s, inputs), x, i) for i in range(n)])
         if ny else np.zeros((0, n)))
    u = np.array([inputs[name] for name in names], dtype=np.float64)
    as_inputs = lambda vector: dict(zip(names, vector))  # noqa: E731
    B = (np.column_stack([column(lambda w: compiled.rhs(t, x, as_inputs(w)), u, j) for j in range(len(u))])
         if len(u) else np.zeros((n, 0)))
    D = (np.column_stack([column(lambda w: observe(x, as_inputs(w)), u, j) for j in range(len(u))])
         if len(u) and ny else np.zeros((ny, len(u))))
    return {"A": A, "B": B, "C": C, "D": D, "rhs": compiled.rhs(t, x, inputs)}


def _rank(matrix, rtol):
    if matrix.size == 0:
        return 0
    singular = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular > rtol * np.max(singular)))


def _compare_linearization(model, request, view):
    reference = finite_difference_linearization(model, request)
    rtol, atol = request["reference"]["jacobian_rtol"], request["reference"]["jacobian_atol"]
    result, passed = {}, True
    for key in ("A", "B", "C", "D", "rhs"):
        actual = np.asarray(view[key], dtype=np.float64).reshape(np.shape(reference[key]))
        error, ok = _max_errors(actual, reference[key], atol, rtol)
        result[key] = {"max_abs_error": error, "within_threshold": ok}
        passed = passed and ok
    A = np.asarray(view["A"], dtype=np.float64).reshape(reference["A"].shape)
    B = np.asarray(view["B"], dtype=np.float64).reshape(reference["B"].shape)
    C = np.asarray(view["C"], dtype=np.float64).reshape(reference["C"].shape)
    eig = np.linalg.eigvals(A)
    eig = eig[np.lexsort((eig.imag, eig.real))]
    poles = np.array(view["poles_real"]) + 1j * np.array(view["poles_imag"])
    pole_error = float(np.max(np.abs(poles - eig))) if len(eig) else 0.0
    n = A.shape[0]
    controllability = _rank(np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)]), request["rank_rtol"]) if B.size else 0
    observability = _rank(np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)]), request["rank_rtol"]) if C.size else 0
    ranks_ok = (controllability == view["controllability_rank"] and observability == view["observability_rank"])
    pole_ok = pole_error <= atol + rtol * max(1.0, float(np.max(np.abs(eig)))) if len(eig) else True
    return {"reference": {"method": "central_finite_difference", "relative_step": request["reference"]["fd_relative_step"],
                          "eigen": "numpy.linalg.eigvals of the Julia A", "rank": "numpy SVD with rank_rtol"},
            "thresholds": {"atol": atol, "rtol": rtol}, "matrices": result,
            "poles": {"max_abs_difference": pole_error, "within_threshold": bool(pole_ok)},
            "ranks": {"controllability": controllability, "observability": observability, "agree": bool(ranks_ok)},
            "passed": bool(passed and pole_ok and ranks_ok)}


def _compare_symbolic(model, request, view):
    reference = finite_difference_linearization(model, request)
    rtol, atol = request["reference"]["jacobian_rtol"], request["reference"]["jacobian_atol"]
    declared = np.asarray(view["jacobian_declared"], dtype=np.float64)
    mapped = np.asarray(view["jacobian_compiled_to_declared"], dtype=np.float64)
    d_error, d_ok = _max_errors(declared, reference["A"], atol, rtol)
    m_error, m_ok = _max_errors(mapped, reference["A"], atol, rtol)
    order_ok = view["declared_unknowns"] == [symbol.replace(".", "__") for symbol in model.state_order]
    return {"reference": {"method": "central_finite_difference",
                          "relative_step": request["reference"]["fd_relative_step"]},
            "thresholds": {"atol": atol, "rtol": rtol},
            "state_order": {"declared": model.state_order, "compiled": view["compiled_unknowns"],
                            "permutation": view["permutation"],
                            "reordered_by_compiler": view["compiled_unknowns"] != view["declared_unknowns"],
                            "declared_order_preserved_in_results": bool(order_ok)},
            "jacobian_declared": {"max_abs_error": d_error, "within_threshold": d_ok},
            "jacobian_compiled_to_declared": {"max_abs_error": m_error, "within_threshold": m_ok},
            "passed": bool(d_ok and m_ok and order_ok)}


def reference_sensitivities(model: Model, request: dict) -> np.ndarray:
    """Central finite differences of observation values with the RK4 reference."""
    from copy import deepcopy
    from .spec import validate_spec
    candidates = request["candidates"]
    times = sorted({float(c["time"]) for c in candidates})
    start, max_step = float(request["start_time"]), float(request["reference"]["max_step"])
    # The RK4 grid joins candidate times to a uniform grid of at most max_step,
    # so truncation error (and its parameter derivative) stays below the check.
    uniform = [start + k * max_step for k in range(int(math.ceil((times[-1] - start) / max_step)) + 1)]
    grid = sorted({start, *times, *(t for t in uniform if t < times[-1])})
    step = request["reference"]["fd_relative_step"]
    spec = model.spec
    positions = {item["symbol"]: index for index, item in enumerate(spec["parameters"])}

    def observe(values):
        # A perturbed copy of the model: bounds on the perturbed parameters and
        # the independent interval are lifted so a central difference at a
        # domain edge is not refused; the copy is never retained.
        changed = deepcopy(spec)
        for symbol, value in values.items():
            changed["parameters"][positions[symbol]]["value"] = value
        changed["validity_domain"]["bounds"] = {k: v for k, v in changed["validity_domain"]["bounds"].items()
                                                if k not in values}
        changed["validity_domain"]["independent_variable"] = None
        variant = validate_spec(changed)
        simulation = {"schema": REQUEST_SCHEMA, "spec_digest": variant.digest,
                      "initial_state": list(request["initial_state"]), "inputs": dict(request["inputs"]),
                      "sample_times": grid, "interval": [grid[0], grid[-1] + 1.0],
                      "solver": {"algorithm": "Tsit5", "reltol": 1e-10, "abstol": [1e-12] * len(model.state_order),
                                 "maxiters": 1000, "dtmax": None, "initial_dt": "solver_automatic",
                                 "save_policy": "interpolated_saveat"},
                      "reference": {"method": "rk4", "substeps": request["reference"]["substeps"]},
                      "acceptance": {"state_atol": [1.0] * len(model.state_order), "state_rtol": 0.0}}
        trajectory = simulate_reference(variant, simulation)
        if trajectory["status"] != "completed":
            raise SpecificationError("Reference sensitivity simulation halted")
        index = {t: i for i, t in enumerate(trajectory["sample_times"])}
        return np.array([trajectory["observations"][c["observation"]][index[float(c["time"])]] for c in candidates])

    base = {item["symbol"]: float(item["value"]) for item in spec["parameters"]}
    columns = []
    for symbol in request["design_parameters"]:
        h = step * max(abs(base[symbol]), 1e-12)
        columns.append((observe({symbol: base[symbol] + h}) - observe({symbol: base[symbol] - h})) / (2 * h))
    return np.column_stack(columns)


def enumerate_design(information: np.ndarray, costs, budget, max_count):
    """Exhaustive maximin check of the MILP optimum (small candidate sets only)."""
    best, best_sets = 0.0, [()]
    count = len(costs)
    for size in range(1, max_count + 1):
        for subset in combinations(range(count), size):
            if sum(costs[i] for i in subset) > budget:
                continue
            value = float(np.min(information[list(subset)].sum(axis=0)))
            if value > best:
                best, best_sets = value, [subset]
            elif value == best:
                best_sets.append(subset)
    return best, best_sets


def _compare_design(model, request, view):
    information = np.asarray(view["information"], dtype=np.float64)
    sensitivities = np.asarray(view["sensitivities"], dtype=np.float64)
    prior, noise = design_variances(model, request)
    candidates = request["candidates"]
    costs = [float(c["cost"]) for c in candidates]
    selected = list(view["selected"])
    recomputed = np.array([[sensitivities[i, j] ** 2 * prior[p] / noise[c["observation"]]
                            for j, p in enumerate(request["design_parameters"])] for i, c in enumerate(candidates)])
    info_consistent = bool(np.allclose(recomputed, information, rtol=1e-12, atol=0.0))
    feasible = (sum(costs[i] for i in selected) <= request["budget"] + 1e-9
                and len(selected) <= request["max_count"] and len(set(selected)) == len(selected))
    achieved = float(np.min(information[selected].sum(axis=0))) if selected else 0.0
    rtol = request["reference"]["information_rtol"]
    enumeration = None
    optimal = None
    if len(candidates) <= ENUMERATION_LIMIT:
        best, best_sets = enumerate_design(information, costs, request["budget"], request["max_count"])
        optimal = bool(abs(achieved - best) <= rtol * max(best, 1e-300) and abs(view["objective"] - best) <= rtol * max(best, 1e-300))
        enumeration = {"best_objective": best, "optimal_sets": [list(s) for s in best_sets[:16]],
                       "optimal_set_count": len(best_sets), "selected_is_optimal": optimal}
    fd = reference_sensitivities(model, request)
    scale = float(np.max(np.abs(fd))) if fd.size else 0.0
    error, sens_ok = _max_errors(sensitivities, fd, rtol * scale, rtol)
    passed = info_consistent and feasible and sens_ok and (optimal is not False)
    return {"reference": {"sensitivity": "central_finite_difference_with_rk4",
                          "relative_step": request["reference"]["fd_relative_step"],
                          "rk4_max_step": request["reference"]["max_step"],
                          "rk4_substeps": request["reference"]["substeps"],
                          "optimality": "exhaustive_enumeration" if enumeration else "not_enumerated_above_limit"},
            "information_matches_sensitivities": info_consistent, "selection_feasible": bool(feasible),
            "achieved_objective": achieved, "enumeration": enumeration,
            "sensitivities": {"max_abs_difference": error, "rtol": rtol, "atol": rtol * scale,
                              "within_threshold": sens_ok},
            "objective_units": "dimensionless (prior-normalized Fisher information)",
            "passed": bool(passed)}
