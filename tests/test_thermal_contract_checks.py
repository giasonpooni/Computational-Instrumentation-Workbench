"""Every check of the thermal contract refuses a source or result that is otherwise the valid example.

An AST mutation probe dropped each ``if ...: raise`` of ``thermal_contract``
in turn; 78 of 79 survived because the suite only exercised valid sources.
Each case starts from the retained example source, or the result the Python
reference derives from it, changes exactly one thing, and expects the refusal
that names it, so a dropped check either accepts the input or fails with a
different message.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from ciw import thermal_contract as contract
from ciw import thermal_workflow

ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "examples" / "thermal-observer" / "source.json").read_bytes()
SOURCE = json.loads(RAW)
REQUEST = SOURCE["request"]
RESULT = thermal_workflow._native_result(contract.validate_source(RAW))


def encoded(source):
    return json.dumps(source, allow_nan=False).encode("utf-8")


def infeasible_source():
    source = deepcopy(SOURCE)
    source["request"]["selection"]["budget"] = 0.0
    return source


INFEASIBLE_RESULT = thermal_workflow._native_result(contract.validate_source(encoded(infeasible_source())))


def assign(*path, value):
    def apply(root):
        target = root
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return apply


def no_steps(source):
    source["request"]["inputs"] = []
    source["request"]["observations"] = []
    for key in ("process_noise", "measurement_noise", "states"):
        source["evaluation"][key] = []


def nudge(*path, by):
    def apply(root):
        target = root
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] += by
    return apply


def extra_row(*path, row):
    def apply(root):
        target = root
        for key in path:
            target = target[key]
        target.append(row)
    return apply


SOURCE_CASES = {
    "request schema renamed": (assign("request", "schema", value="x"), "Unsupported thermal request schema"),
    "request operation renamed": (assign("request", "operation_id", value="x"), "Unsupported thermal request schema"),
    "inputs given as text": (assign("request", "inputs", value="abc"), "1..128 steps"),
    "no steps at all": (no_steps, "1..128 steps"),
    "observations given as text": (assign("request", "observations", value="abc"), "possibly missing observation row"),
    "one observation row too many": (extra_row("request", "observations", row=[None, None]), "possibly missing observation row"),
    "observation row given as text": (assign("request", "observations", 0, value="ab"), "both ordered channels"),
    "observation row with a third channel": (assign("request", "observations", 0, value=[None, None, None]), "both ordered channels"),
    "tie tolerance changed": (assign("request", "selection", "tie_tolerance_nats", value=1e-9), "fixed lowest-mask tie policy"),
    "tie policy renamed": (assign("request", "selection", "tie_policy", value="highest"), "fixed lowest-mask tie policy"),
    "source schema renamed": (assign("schema", value="x"), "Unsupported thermal source or authority policy"),
    "authority policy changed": (assign("configuration", "proof", value="requested"), "Unsupported thermal source or authority policy"),
    "truth origin not synthetic": (assign("evaluation", "origin", value="physical_measurement"), "explicitly synthetic"),
    "noise law claimed": (assign("evaluation", "noise_law", value="gaussian"), "explicitly synthetic"),
    "process noise rows given as text": (assign("evaluation", "process_noise", value="abc"), "every simulated noise and state row"),
    "one process noise row too many": (extra_row("evaluation", "process_noise", row=[0.0, 0.0]), "every simulated noise and state row"),
    "object given as a list of its names": (assign("request", "prior", value=["mean", "covariance"]), "missing or unexpected fields"),
    "object with an extra field": (assign("request", "prior", "extra", value=1), "missing or unexpected fields"),
    "scalar given as text": (assign("request", "sample_interval_s", value="1"), "Thermal scalar"),
    "scalar above its profile": (assign("request", "sample_interval_s", value=4000), "Thermal scalar"),
    "integer given as a float": (assign("request", "selection", "minimum_sensors", value=1.0), "Thermal integer"),
    "integer above its profile": (assign("request", "selection", "minimum_sensors", value=3), "Thermal integer"),
    "vector given as text": (assign("request", "prior", "mean", value="ab"), "two ordered thermal coordinates"),
    "vector with three entries": (assign("request", "prior", "mean", value=[300.0, 300.0, 300.0]), "two ordered thermal coordinates"),
    "covariance given as text": (assign("request", "prior", "covariance", value="ab"), "two by two covariance"),
    "covariance with three rows": (assign("request", "prior", "covariance", value=[[4.0, 0.0], [0.0, 4.0], [0.0, 0.0]]), "two by two covariance"),
    "covariance asymmetric": (assign("request", "prior", "covariance", value=[[4.0, 0.1], [0.0, 4.0]]), "exactly symmetric"),
    "covariance singular": (assign("request", "prior", "covariance", value=[[0.0, 0.0], [0.0, 0.0]]), "positive or conditioning budget"),
    "covariance eigenvalue above budget": (assign("request", "prior", "covariance", value=[[9000.0, 5000.0], [5000.0, 9000.0]]), "positive or conditioning budget"),
    "covariance ill conditioned": (assign("request", "prior", "covariance", value=[[1e4, 0.0], [0.0, 1e-5]]), "positive or conditioning budget"),
    "experiment id given a number": (assign("experiment_id", value=5), "bounded nonempty text"),
    "experiment id empty": (assign("experiment_id", value=""), "bounded nonempty text"),
    "experiment id with a NUL": (assign("experiment_id", value="a\u0000b"), "bounded nonempty text"),
    "model units changed": (assign("request", "model", "state_units", value=["C", "C"]), "coordinate order and units"),
    "capacity ratio beyond profile": (assign("request", "model", "capacities_j_per_k", value=[1.0, 1e5]), "parameter ratios"),
    "conductance ratio beyond profile": (assign("request", "model", "conductances_w_per_k", value=[1e-3, 1e2]), "parameter ratios"),
    "observation matrix given as text": (assign("request", "model", "observation_matrix", value="ab"), "separate core and shell thermometers"),
    "observation matrix not identity": (assign("request", "model", "observation_matrix", value=[[1, 0], [0, 2]]), "separate core and shell thermometers"),
    "negative heat power": (assign("request", "inputs", 0, value=[-1.0, 300.0]), "Inputs must keep"),
    "ambient below profile": (assign("request", "inputs", 0, value=[100.0, 50.0]), "Inputs must keep"),
    "equilibrium above profile": (assign("request", "inputs", 0, value=[5000.0, 900.0]), "Inputs must keep"),
    "retained state differs from the simulation": (nudge("evaluation", "states", 0, 0, by=1e-3), r"evaluation\.states\[0\] differs from the independent analytical reference"),
}

RESULT_CASES = {
    "result schema renamed": (assign("schema", value="x"), "request or authority binding"),
    "claim scope renamed": (assign("claim_scope", value="x"), "request or authority binding"),
    "coordinate order changed": (assign("model", "state_order", value=list(reversed(REQUEST["model"]["state_order"]))), "changed declared coordinate order"),
    "matrix given as text": (assign("model", "Ad", value="ab"), r"model\.Ad shape differs"),
    "matrix with a third row": (extra_row("model", "Ad", row=[0.0, 0.0]), r"model\.Ad shape differs"),
    "matrix entry differs": (nudge("model", "Ad", 0, 0, by=1e-3), r"model\.Ad\[0\]\[0\] differs from the independent analytical reference"),
    "boolean given as an integer": (assign("selection", "candidates", 1, "feasible", value=1), r"feasible differs"),
    "text field differs": (assign("observer", "ordering", value="other"), r"observer\.ordering differs"),
    "rendering provenance unknown": (assign("model", "symbolic", "rendering", value="other"), "symbolic rendering provenance"),
    "equations given as text": (assign("model", "symbolic", "equations", value="abc"), "bounded native model equations"),
    "seventeen equations": (assign("model", "symbolic", "equations", value=["x"] * 17), "bounded native model equations"),
    "native state order given as text": (assign("model", "symbolic", "native_state_order", value="ab"), "two distinct native state names"),
    "native state order with a repeated third name": (assign("model", "symbolic", "native_state_order", value=["a", "a", "b"]), "two distinct native state names"),
    "native state order repeated": (assign("model", "symbolic", "native_state_order", value=["a", "a"]), "two distinct native state names"),
    "permutation given as a tuple": (assign("model", "symbolic", "state_permutation", value=(1, 2)), "bijection"),
    "permutation given as floats": (assign("model", "symbolic", "state_permutation", value=[1.0, 2.0]), "bijection"),
    "permutation repeating an index": (assign("model", "symbolic", "state_permutation", value=[1, 1]), "bijection"),
    "solver renamed": (assign("selection", "solver", "name", value="CPLEX"), "declared constrained optimizer"),
    "feasibility tolerance as a numpy scalar": (assign("selection", "solver", "primal_feasibility_tolerance", value=np.float64(1e-9)), "HiGHS feasibility tolerance"),
    "feasibility tolerance changed": (assign("selection", "solver", "primal_feasibility_tolerance", value=1e-8), "HiGHS feasibility tolerance"),
    "gap tolerance as a numpy scalar": (assign("selection", "solver", "mip_relative_gap_tolerance", value=np.float64(0)), "zero relative gap target"),
    "gap tolerance changed": (assign("selection", "solver", "mip_relative_gap_tolerance", value=0.5), "zero relative gap target"),
    "optimal solution reported infeasible": (assign("selection", "solver", "termination_status", value="INFEASIBLE"), "Do not promote"),
    "optimal solution without a feasible point": (assign("selection", "solver", "primal_status", value="NO_SOLUTION"), "Do not promote"),
    "nonzero optimality gap": (assign("selection", "solver", "relative_gap", value=1e-6), "nonzero optimality gap"),
}

INFEASIBLE_CASES = {
    "infeasible selection reported optimal": (assign("selection", "solver", "termination_status", value="OPTIMAL"), "must not invent"),
    "infeasible selection with a feasible point": (assign("selection", "solver", "primal_status", value="FEASIBLE_POINT"), "must not invent"),
    "infeasible selection with an objective": (assign("selection", "solver", "objective_value", value=0.0), "must not invent"),
}


def test_the_example_source_and_its_reference_result_are_accepted():
    assert contract.validate_source(RAW)["experiment_id"] == SOURCE["experiment_id"]
    contract.validate_result(REQUEST, deepcopy(RESULT))
    assert INFEASIBLE_RESULT["selection"]["status"] == "infeasible"
    contract.validate_result(infeasible_source()["request"], deepcopy(INFEASIBLE_RESULT))


@pytest.mark.parametrize("mutate, message", SOURCE_CASES.values(), ids=list(SOURCE_CASES))
def test_one_change_to_the_example_source_is_refused_by_its_own_check(mutate, message):
    source = deepcopy(SOURCE)
    mutate(source)
    with pytest.raises(ValueError, match=message):
        contract.validate_source(encoded(source))


@pytest.mark.parametrize("mutate, message", RESULT_CASES.values(), ids=list(RESULT_CASES))
def test_one_change_to_the_reference_result_is_refused_by_its_own_check(mutate, message):
    result = deepcopy(RESULT)
    mutate(result)
    with pytest.raises(ValueError, match=message):
        contract.validate_result(REQUEST, result)


@pytest.mark.parametrize("mutate, message", INFEASIBLE_CASES.values(), ids=list(INFEASIBLE_CASES))
def test_an_infeasible_selection_cannot_carry_a_solution(mutate, message):
    result = deepcopy(INFEASIBLE_RESULT)
    mutate(result)
    with pytest.raises(ValueError, match=message):
        contract.validate_result(infeasible_source()["request"], result)


@pytest.mark.parametrize("raw", ["text", b""], ids=["text", "empty"])
def test_a_source_must_be_bounded_exact_bytes(raw):
    with pytest.raises(ValueError, match="bounded exact JSON bytes"):
        contract.validate_source(raw)


def test_the_byte_budgets_are_enforced(monkeypatch):
    monkeypatch.setattr(contract, "SOURCE_LIMIT", 100)
    with pytest.raises(ValueError, match="exceeds the source budget"):
        contract.validate_request(deepcopy(REQUEST))
    monkeypatch.undo()
    monkeypatch.setattr(contract, "RESULT_LIMIT", 100)
    with pytest.raises(ValueError, match="exceeds the profile budget"):
        contract.validate_result(REQUEST, deepcopy(RESULT))


def test_a_request_may_not_span_more_than_one_day():
    request = deepcopy(REQUEST)
    request["sample_interval_s"] = 3600
    request["inputs"] = [[100.0, 300.0]] * 25
    request["observations"] = [[None, None]] * 25
    with pytest.raises(ValueError, match="1..128 steps spanning at most one day"):
        contract.validate_request(request)
