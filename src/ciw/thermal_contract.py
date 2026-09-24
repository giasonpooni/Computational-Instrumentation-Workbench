"""Strict retained thermal experiment contract and independent numerical checks.

Validation runs bounded Python algebra, never Julia or another provider. The
simulation declaration is an operator assertion, not hardware authentication.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .telemetry import canonical

SOURCE_SCHEMA = "ciw.thermal-observer-source.v1"
REQUEST_SCHEMA = "ciw.thermal-observer-request.v1"
RESULT_SCHEMA = "ciw.thermal-observer-result.v1"
OPERATION_ID = "julia.thermal-design.v1"
CLAIM_SCOPE = "simulated_linear_thermal_observer_and_prospective_sensor_selection"
SOURCE_LIMIT = 256 * 1024
RESULT_LIMIT = 2 * 1024 * 1024
STATE_ORDER = ["core_temperature", "shell_temperature"]
INPUT_ORDER = ["heat_power", "ambient_temperature"]
SENSOR_ORDER = ["core", "shell"]
POLICY = {"plant": "two_capacity_linear_thermal",
          "inference": "julia_time_varying_kalman_filter",
          "reference": "independent_python_analytical_rc_and_joseph_conditioning",
          "selection": "prospective_four_subset_information_gain_under_declared_cost_budget",
          "observations": "retained_synthetic_values_with_explicit_dropout",
          "physical_validation": "not_established", "state_admission": "not_performed",
          "proof": "not_requested"}
TOLERANCES = {"absolute": 1e-7, "relative": 1e-8}


def _keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("Thermal object has missing or unexpected fields")


def _number(value, lower=-1e12, upper=1e12):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError("Thermal scalar must be finite and inside its declared profile")
    return float(value)


def _integer(value, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("Thermal integer is outside its declared profile")
    return value


def _vector(value, lower=-1e12, upper=1e12):
    if type(value) is not list or len(value) != 2:
        raise ValueError("Require two ordered thermal coordinates")
    return np.array([_number(x, lower, upper) for x in value])


def _covariance(value):
    if type(value) is not list or len(value) != 2:
        raise ValueError("Require a two by two covariance")
    array = np.array([_vector(row, -1e4, 1e4) for row in value])
    if not np.array_equal(array, array.T):
        raise ValueError("Covariance must be exactly symmetric, without repair")
    eigenvalues = np.linalg.eigvalsh(array)
    if eigenvalues[0] < 1e-10 or eigenvalues[-1] > 1e4 or eigenvalues[-1] / eigenvalues[0] > 1e8:
        raise ValueError("Covariance violates the positive or conditioning budget")
    return array


def _text(value, limit=8192):
    if type(value) is not str or not 1 <= len(value) <= limit or "\x00" in value:
        raise ValueError("Require bounded nonempty text")


def validate_model(model):
    _keys(model, {"state_order", "state_units", "input_order", "input_units", "sensor_order", "sensor_units",
                  "capacities_j_per_k", "conductances_w_per_k", "observation_matrix"})
    fixed = {"state_order": STATE_ORDER, "state_units": ["K", "K"], "input_order": INPUT_ORDER,
             "input_units": ["W", "K"], "sensor_order": SENSOR_ORDER, "sensor_units": ["K", "K"]}
    for key, expected in fixed.items():
        if model[key] != expected:
            raise ValueError("Thermal coordinate order and units must be explicit")
    capacities = _vector(model["capacities_j_per_k"], 1, 1e6)
    conductances = _vector(model["conductances_w_per_k"], 1e-3, 1e3)
    if max(capacities)/min(capacities) > 1e4 or max(conductances)/min(conductances) > 1e4:
        raise ValueError("Thermal parameter ratios exceed the conditioning profile")
    h = model["observation_matrix"]
    if type(h) is not list or len(h) != 2 or not np.array_equal(np.array([_vector(row) for row in h]), np.eye(2)):
        raise ValueError("This profile has separate core and shell thermometers")
    return deepcopy(model)


def _input(value, model):
    power, ambient = _vector(value)
    g, ga = model["conductances_w_per_k"]
    if not 0 <= power <= 1e4 or not 100 <= ambient <= 1000 or ambient + power/ga + power/g > 1000:
        raise ValueError("Inputs must keep the declared deterministic equilibrium within 100..1000 K")


def validate_request(request):
    _keys(request, {"schema", "operation_id", "model", "sample_interval_s", "prior", "process_noise_covariance",
                    "observation_noise_covariance", "inputs", "observations", "selection"})
    if request["schema"] != REQUEST_SCHEMA or request["operation_id"] != OPERATION_ID:
        raise ValueError("Unsupported thermal request schema or operation")
    model = request["model"]
    validate_model(model)
    dt = _number(request["sample_interval_s"], .001, 3600)
    _keys(request["prior"], {"mean", "covariance"})
    _vector(request["prior"]["mean"], 100, 1000)
    for value in (request["prior"]["covariance"], request["process_noise_covariance"], request["observation_noise_covariance"]):
        _covariance(value)
    inputs, observations = request["inputs"], request["observations"]
    if type(inputs) is not list or not 1 <= len(inputs) <= 128 or dt * len(inputs) > 86400:
        raise ValueError("Require 1..128 steps spanning at most one day")
    if type(observations) is not list or len(observations) != len(inputs):
        raise ValueError("Each input step needs an explicit possibly missing observation row")
    for row, measured in zip(inputs, observations):
        _input(row, model)
        if type(measured) is not list or len(measured) != 2:
            raise ValueError("Observations must retain both ordered channels")
        for value in measured:
            if value is not None:
                _number(value, 0, 2000)
    choice = request["selection"]
    _keys(choice, {"costs", "budget", "minimum_sensors", "tie_tolerance_nats", "tie_policy"})
    _vector(choice["costs"], 0, 1e6)
    _number(choice["budget"], 0, 2e6)
    _integer(choice["minimum_sensors"], 0, 2)
    if type(choice["tie_tolerance_nats"]) not in (int, float) or choice["tie_tolerance_nats"] != 1e-10 or choice["tie_policy"] != "lowest_mask_within_tolerance":
        raise ValueError("Require the fixed lowest-mask tie policy")
    if len(canonical(request)) > SOURCE_LIMIT:
        raise ValueError("Thermal request exceeds the source budget")
    return deepcopy(request)


def _close(received, expected, path="result"):
    """Strict JSON shape/type and bounded binary64 agreement, without coercion."""
    if isinstance(expected, dict):
        _keys(received, expected)
        for key in expected:
            _close(received[key], expected[key], path + "." + key)
    elif isinstance(expected, list):
        if type(received) is not list or len(received) != len(expected):
            raise ValueError(path + " shape differs")
        for i, (actual, wanted) in enumerate(zip(received, expected)):
            _close(actual, wanted, path + "[" + str(i) + "]")
    elif type(expected) is float:
        value = _number(received)
        if not math.isclose(value, expected, rel_tol=TOLERANCES["relative"], abs_tol=TOLERANCES["absolute"]):
            raise ValueError(path + " differs from the independent analytical reference")
    elif type(received) is not type(expected) or received != expected:
        raise ValueError(path + " differs")


def validate_source(raw):
    from .reference_workflow import parse_json
    from .thermal_reference import model_matrices
    if type(raw) is not bytes or not 1 <= len(raw) <= SOURCE_LIMIT:
        raise ValueError("Thermal source requires bounded exact JSON bytes")
    source = parse_json(raw, "Thermal source")
    _keys(source, {"schema", "experiment_id", "configuration", "request", "evaluation"})
    if source["schema"] != SOURCE_SCHEMA or canonical(source["configuration"]) != canonical(POLICY):
        raise ValueError("Unsupported thermal source or authority policy")
    _text(source["experiment_id"], 128)
    request = source["request"]
    validate_request(request)
    evaluation = source["evaluation"]
    _keys(evaluation, {"origin", "generator_model", "initial_state", "process_noise", "measurement_noise", "states", "held_out", "noise_law"})
    if evaluation["origin"] != "synthetic_fixture" or evaluation["noise_law"] != "operator_declared_not_authenticated":
        raise ValueError("Thermal truth is explicitly synthetic; no empirical noise-law claim")
    validate_model(evaluation["generator_model"])
    model = evaluation["generator_model"]
    matrices = model_matrices(model, request["sample_interval_s"])
    ad, bd = np.array(matrices["Ad"]), np.array(matrices["Bd"])
    state = _vector(evaluation["initial_state"], 100, 1000)
    for key in ("process_noise", "measurement_noise", "states"):
        if type(evaluation[key]) is not list or len(evaluation[key]) != len(request["inputs"]):
            raise ValueError("Retain every simulated noise and state row")
    for index, inputs in enumerate(request["inputs"]):
        _input(inputs, model)
        state = ad @ state + bd @ np.array(inputs) + _vector(evaluation["process_noise"][index], -100, 100)
        observed = state + _vector(evaluation["measurement_noise"][index], -100, 100)
        _vector(evaluation["states"][index], 100, 1000)
        _close(evaluation["states"][index], state.tolist(), "evaluation.states")
        for channel, value in enumerate(request["observations"][index]):
            if value is not None:
                _close(value, float(observed[channel]), "evaluation.observations")
    held = evaluation["held_out"]
    _keys(held, {"input", "process_noise", "state", "measurement_noise", "observations"})
    _input(held["input"], model)
    _input(held["input"], request["model"])
    state = ad @ state + bd @ np.array(held["input"]) + _vector(held["process_noise"], -100, 100)
    observed = state + _vector(held["measurement_noise"], -100, 100)
    _vector(held["state"], 100, 1000)
    _vector(held["observations"], 0, 2000)
    _close(held["state"], state.tolist(), "evaluation.held_out.state")
    _close(held["observations"], observed.tolist(), "evaluation.held_out.observations")
    return source


def validate_result(request, result):
    """Compare native artifact against independently derived matrices and replay."""
    from .thermal_reference import reference
    validate_request(request)
    _keys(result, {"schema", "request", "claim_scope", "model", "observer", "selection"})
    if result["schema"] != RESULT_SCHEMA or result["claim_scope"] != CLAIM_SCOPE or canonical(result["request"]) != canonical(request):
        raise ValueError("Thermal result request or authority binding differs")
    if len(canonical(result)) > RESULT_LIMIT:
        raise ValueError("Thermal result exceeds the profile budget")
    expected = reference(request)
    model = result["model"]
    _keys(model, set(expected["model"]) | {"state_order", "input_order", "sensor_order", "symbolic"})
    for key in ("state_order", "input_order", "sensor_order"):
        if model[key] != request["model"][key]:
            raise ValueError("Generated model changed declared coordinate order")
    for key, value in expected["model"].items():
        _close(model[key], value, "model." + key)
    symbolic = model["symbolic"]
    _keys(symbolic, {"equations", "latex", "native_state_order", "state_permutation", "rendering"})
    if symbolic["rendering"] not in {"Symbolics/Latexify", "python-reference"}:
        raise ValueError("Require explicit symbolic rendering provenance")
    for key in ("equations", "latex"):
        if type(symbolic[key]) is not list or not 2 <= len(symbolic[key]) <= 16:
            raise ValueError("Retain bounded native model equations and rendering")
        for value in symbolic[key]:
            _text(value)
    order, permutation = symbolic["native_state_order"], symbolic["state_permutation"]
    if type(order) is not list or len(order) != 2 or len(set(order)) != 2:
        raise ValueError("Retain the two distinct native state names")
    for name in order:
        _text(name, 256)
    if type(permutation) is not list or len(permutation) != 2 or any(type(x) is not int for x in permutation) or sorted(permutation) != [1, 2]:
        raise ValueError("Retain a bijection from native to declared state order")
    _close(result["observer"], expected["observer"], "observer")
    selected = result["selection"]
    _keys(selected, set(expected["selection"]) | {"solver"})
    for key, value in expected["selection"].items():
        _close(selected[key], value, "selection." + key)
    solver = selected["solver"]
    _keys(solver, {"name", "termination_status", "primal_status", "objective_value", "objective_bound", "relative_gap",
                   "primal_feasibility_tolerance", "dual_feasibility_tolerance", "mip_feasibility_tolerance", "mip_relative_gap_tolerance"})
    if solver["name"] not in {"HiGHS", "python-reference-enumeration"}:
        raise ValueError("Require the declared constrained optimizer or bounded reference enumeration")
    for key in ("primal_feasibility_tolerance", "dual_feasibility_tolerance", "mip_feasibility_tolerance"):
        if type(solver[key]) not in (int, float) or solver[key] != 1e-9:
            raise ValueError("Unexpected HiGHS feasibility tolerance")
    if type(solver["mip_relative_gap_tolerance"]) not in (int, float) or solver["mip_relative_gap_tolerance"] != 0:
        raise ValueError("Require zero relative gap target for the bounded selection problem")
    if selected["status"] == "infeasible":
        if solver["termination_status"] != "INFEASIBLE" or solver["primal_status"] != "NO_SOLUTION" or any(solver[key] is not None for key in ("objective_value", "objective_bound", "relative_gap")):
            raise ValueError("Infeasible selection must not invent a solver solution")
    else:
        if solver["termination_status"] != "OPTIMAL" or solver["primal_status"] != "FEASIBLE_POINT":
            raise ValueError("Do not promote an unchecked incumbent to an optimum")
        maximum = max(row["information_gain_nats"] for row in expected["selection"]["candidates"] if row["feasible"])
        _close(solver["objective_value"], maximum, "solver.objective_value")
        _close(solver["objective_bound"], maximum, "solver.objective_bound")
        if _number(solver["relative_gap"], 0, 1) > 1e-9:
            raise ValueError("Native selection retains a nonzero optimality gap")


def validate_data(source, data, evidence_id=None):
    """Optional aggregate checker used by the workbench's retained artifact."""
    from .thermal_reference import evaluate
    checked = validate_source(canonical(source))
    _keys(data, {"native", "evaluation"})
    validate_result(checked["request"], data["native"])
    _close(data["evaluation"], evaluate(checked, data["native"]["observer"]), "evaluation")
