"""Typed-port composition: meaning-preserving connections, refusals and composition laws."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from ciw.model.compose import compose, semantic_view
from ciw.model.simulation import simulate_reference, uniform_request
from ciw.model.spec import SpecificationError, validate_spec
from ciw.model.transform import rescale

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "model-core"
S = lambda name: {"sym": name}  # noqa: E731
N = lambda value: {"num": value}  # noqa: E731


def op(name, *args):
    return {"op": name, "args": list(args)}


def load(name):
    return validate_spec(json.loads((EXAMPLES / name).read_text()))


LOOP = [{"from": "plant.position", "to": "ctrl.measurement"}, {"from": "ctrl.command", "to": "plant.force"}]


def closed_loop(plant=None, controller=None):
    return compose("closed-loop", "Oscillator under filtered PD control",
                   [{"name": "plant", "model": plant or load("damped-oscillator.json")},
                    {"name": "ctrl", "model": controller or load("pd-controller.json")}], LOOP)


def monolithic_si():
    """The same closed loop written by hand, entirely in SI units."""
    plant = deepcopy(load("damped-oscillator.json").spec)
    plant["states"].append({"symbol": "z", "quantity": "displacement", "unit": "m", "frame": "oscillator-axis"})
    plant["inputs"] = []
    plant["parameters"] += [{"symbol": "kp", "quantity": "stiffness_gain", "unit": "N/m", "value": 20.0},
                            {"symbol": "kd", "quantity": "damping_gain", "unit": "N*s/m", "value": 4.0},
                            {"symbol": "tau_f", "quantity": "time_constant", "unit": "s", "value": 0.02},
                            {"symbol": "r", "quantity": "displacement", "unit": "m", "value": 0.0}]
    plant["derived"].append({"symbol": "F", "quantity": "force", "unit": "N", "role": "derived",
                             "expression": op("sub", op("mul", S("kp"), op("sub", S("r"), S("q"))),
                                              op("mul", S("kd"), op("div", op("sub", S("q"), S("z")), S("tau_f"))))})
    plant["dynamics"].append({"state": "z", "rhs": op("div", op("sub", S("q"), S("z")), S("tau_f"))})
    plant["ports"] = []
    plant["validity_domain"]["bounds"].pop("F")
    return validate_spec(plant)


def test_closed_loop_composition_preserves_meaning_across_units():
    loop = closed_loop()
    assert loop.state_order == ["plant.q", "plant.v", "ctrl.z"]
    assert loop.spec["inputs"] == []
    derived = {entry["symbol"]: entry for entry in loop.spec["derived"]}
    assert derived["ctrl.y_meas"]["unit"] == "mm" and derived["ctrl.y_meas"]["expression"] == S("plant.q")
    assert derived["plant.F"]["expression"] == S("ctrl.u") and derived["plant.F"]["role"] == "commanded"
    connection = loop.spec["provenance"]["derived_from"]["connections"][0]
    assert connection["substitution"] == "noise_free_predicted_observation"
    assert connection["calibration"]["calibration_id"] == "cal-displacement-probe-1-2026-09"
    assert "independent_component_uncertainty" in loop.spec["uncertainty"]["assumptions"]

    request = uniform_request(loop, initial_state=[0.3, 0.0, 300.0], inputs={})
    reference = monolithic_si()
    reference_request = uniform_request(reference, initial_state=[0.3, 0.0, 0.3], inputs={})
    a, b = simulate_reference(loop, request), simulate_reference(reference, reference_request)
    states = np.asarray(a["states"]) * [1.0, 1.0, 1e-3]
    assert np.allclose(states, b["states"], rtol=1e-10, atol=1e-13)
    assert np.allclose(a["derived"]["plant.F"], b["derived"]["F"], rtol=1e-10, atol=1e-12)


def actuator():
    return validate_spec({
        "schema": "ciw.model-spec.v1", "model_id": "force-actuator", "title": "First-order force actuator",
        "independent_variable": {"symbol": "t", "kind": "time", "unit": "s", "origin": "seconds since run start"},
        "states": [{"symbol": "Fa", "quantity": "force", "unit": "N", "frame": "oscillator-axis"}],
        "inputs": [{"symbol": "cmd", "quantity": "force", "unit": "N", "frame": "oscillator-axis", "role": "commanded"}],
        "parameters": [{"symbol": "tau_a", "quantity": "time_constant", "unit": "ms", "value": 5.0}],
        "dynamics": [{"state": "Fa", "rhs": op("div", op("sub", S("cmd"), S("Fa")), S("tau_a"))}],
        "derived": [{"symbol": "out", "quantity": "force", "unit": "N", "frame": "oscillator-axis",
                     "role": "commanded", "expression": S("Fa")}],
        "observations": [],
        "ports": [{"name": "cmd", "direction": "in", "symbol": "cmd"}, {"name": "force", "direction": "out", "symbol": "out"}],
        "uncertainty": {"parameters": {"order": ["tau_a"], "matrix": [[0.25]]}, "initial_state": None,
                        "measurement_noise": None, "process_noise": None,
                        "assumptions": ["no_process_noise", "gaussian_parameters"]},
        "validity_domain": {"independent_variable": [0.0, 30.0], "bounds": {}, "notes": ["No saturation"]},
        "provenance": {"kind": "authored", "derived_from": None, "notes": []}})


def test_composition_is_associative_up_to_its_derivation_record():
    plant, act, ctrl = load("damped-oscillator.json"), actuator(), load("pd-controller.json")
    c1 = {"from": "act.force", "to": "plant.force"}
    c2 = {"from": "ctrl.command", "to": "act.cmd"}
    c3 = {"from": "plant.position", "to": "ctrl.measurement"}
    left_inner = compose("pa", "plant+actuator", [{"name": "plant", "model": plant}, {"name": "act", "model": act}], [c1])
    left = compose("all", "all", [{"name": None, "model": left_inner}, {"name": "ctrl", "model": ctrl}], [c2, c3])
    right_inner = compose("ac", "actuator+controller", [{"name": "act", "model": act}, {"name": "ctrl", "model": ctrl}], [c2])
    right = compose("all", "all", [{"name": "plant", "model": plant}, {"name": None, "model": right_inner}], [c1, c3])
    assert semantic_view(left) == semantic_view(right)
    assert left.state_order == right.state_order == ["plant.q", "plant.v", "act.Fa", "ctrl.z"]
    a = simulate_reference(left, uniform_request(left, initial_state=[0.3, 0.0, 0.0, 300.0], inputs={}, count=256))
    b = simulate_reference(right, uniform_request(right, initial_state=[0.3, 0.0, 0.0, 300.0], inputs={}, count=256))
    assert a["states"] == b["states"]


def test_rescaling_commutes_with_composition():
    plant, controller = load("damped-oscillator.json"), load("pd-controller.json")
    original = closed_loop(plant, controller)
    scaled = closed_loop(rescale(plant, {"q": "mm", "v": "mm/s", "y": "mm"}), rescale(controller, {"z": "m", "y_meas": "m"}))
    a = simulate_reference(original, uniform_request(original, initial_state=[0.3, 0.0, 300.0], inputs={}, count=256))
    b = simulate_reference(scaled, uniform_request(scaled, initial_state=[300.0, 0.0, 0.3], inputs={}, count=256))
    assert np.allclose(np.asarray(b["states"]) * [1e-3, 1e-3, 1e3], a["states"], rtol=1e-10, atol=1e-12)


def refused(connections, plant=None, controller=None, message=""):
    with pytest.raises(SpecificationError, match=message):
        compose("x", "x", [{"name": "plant", "model": plant or load("damped-oscillator.json")},
                           {"name": "ctrl", "model": controller or load("pd-controller.json")}], connections)


def test_role_mismatch_command_is_not_a_measurement():
    controller = deepcopy(load("pd-controller.json").spec)
    controller["derived"][0].update(quantity="displacement", unit="mm")
    controller["derived"][0]["expression"] = {"sym": "z"}
    refused([{"from": "ctrl.command", "to": "ctrl.measurement"}], controller=validate_spec(controller),
            message="Role mismatch")


def test_frame_quantity_dimension_and_calibration_are_checked():
    controller = deepcopy(load("pd-controller.json").spec)
    controller["inputs"][0]["frame"] = "machine-x"
    refused(LOOP[:1], controller=validate_spec(controller), message="Frame mismatch")
    controller = deepcopy(load("pd-controller.json").spec)
    controller["inputs"][0]["calibration"]["calibration_id"] = "cal-other-probe"
    refused(LOOP[:1], controller=validate_spec(controller), message="Calibration identity mismatch")
    controller = deepcopy(load("pd-controller.json").spec)
    controller["inputs"][0]["quantity"] = "position_error"
    refused(LOOP[:1], controller=validate_spec(controller), message="Quantity mismatch")
    source = deepcopy(actuator().spec)
    source["states"][0]["unit"] = source["inputs"][0]["unit"] = source["derived"][0]["unit"] = "N*m"
    with pytest.raises(SpecificationError, match="Dimension mismatch"):
        compose("x", "x", [{"name": "plant", "model": load("damped-oscillator.json")},
                           {"name": "act", "model": validate_spec(source)}], [{"from": "act.force", "to": "plant.force"}])


def test_time_and_path_length_models_do_not_compose():
    with pytest.raises(SpecificationError, match="time model with a path_length model"):
        compose("x", "x", [{"name": "plant", "model": load("damped-oscillator.json")},
                           {"name": "road", "model": load("path-heading.json")}], [])


def test_independent_variable_origins_must_agree():
    controller = deepcopy(load("pd-controller.json").spec)
    controller["independent_variable"]["origin"] = "seconds since trigger"
    refused([], controller=validate_spec(controller), message="different independent-variable origins")


def test_double_connection_and_direction_are_refused():
    refused(LOOP + [{"from": "plant.position", "to": "ctrl.measurement"}], message="already connected")
    refused([{"from": "ctrl.measurement", "to": "plant.force"}], message="out-port to an in-port")


def test_algebraic_loops_need_explicit_semantics():
    def feedthrough(name):
        return validate_spec({
            "schema": "ciw.model-spec.v1", "model_id": name, "title": name,
            "independent_variable": {"symbol": "t", "kind": "time", "unit": "s", "origin": "o"},
            "states": [{"symbol": "x", "quantity": "q", "unit": "1", "frame": "f"}],
            "inputs": [{"symbol": "u", "quantity": "force", "unit": "N", "frame": "f", "role": "commanded"}],
            "parameters": [], "dynamics": [{"state": "x", "rhs": op("div", N(0), S("t"))}],
            "derived": [{"symbol": "w", "quantity": "force", "unit": "N", "frame": "f", "role": "commanded",
                         "expression": S("u")}],
            "observations": [],
            "ports": [{"name": "in", "direction": "in", "symbol": "u"}, {"name": "out", "direction": "out", "symbol": "w"}],
            "uncertainty": {"parameters": None, "initial_state": None, "measurement_noise": None,
                            "process_noise": None, "assumptions": ["no_process_noise"]},
            "validity_domain": {"independent_variable": None, "bounds": {}, "notes": []},
            "provenance": {"kind": "authored", "derived_from": None, "notes": []}})
    with pytest.raises(SpecificationError, match="Algebraic loop"):
        compose("loop", "loop", [{"name": "a", "model": feedthrough("a")}, {"name": "b", "model": feedthrough("b")}],
                [{"from": "a.out", "to": "b.in"}, {"from": "b.out", "to": "a.in"}])
