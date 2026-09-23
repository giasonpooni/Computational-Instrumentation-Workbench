"""Language-neutral model specification: units, expressions, validation and the RK4 reference."""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ciw.adapters.oscillator import make_demo_run
from ciw.model import expression
from ciw.model.simulation import simulate_reference, uniform_request, validate_request
from ciw.model.spec import SpecificationError, default_latex_name, lower, summary, validate_spec
from ciw.model.units import parse_unit

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "model-core"


def load(name):
    return json.loads((EXAMPLES / name).read_text())


def oscillator():
    return load("damped-oscillator.json")


def test_units_parse_scale_and_dimension():
    assert parse_unit("mm").scale == pytest.approx(1e-3)
    assert parse_unit("kg*m^2/s^2").dimension == parse_unit("J").dimension
    assert parse_unit("N*s/m").same_dimension(parse_unit("kg/s"))
    assert parse_unit("rad/s").dimension == parse_unit("Hz").dimension
    assert parse_unit("m/s/s").dimension == parse_unit("m*s^-2").dimension
    assert parse_unit("deg").factor_to(parse_unit("rad")) == pytest.approx(math.pi / 180)
    with pytest.raises(ValueError, match="dimensions differ"):
        parse_unit("m").factor_to(parse_unit("s"))


@pytest.mark.parametrize("text", ["degC", "furlong", "m^0", "m^^2", "", "m/", "(m", "km^99", "hmin"])
def test_units_refuse_affine_unknown_and_malformed(text):
    with pytest.raises(ValueError):
        parse_unit(text)


def test_examples_validate_and_keep_declared_meaning():
    model = validate_spec(oscillator())
    table = summary(model)
    assert table["state_order"] == ["q", "v"]
    assert table["independent_variable"] == "time"
    rows = {row["symbol"]: row for row in table["symbols"]}
    assert rows["F"]["role"] == "commanded" and rows["y"]["role"] == "measured"
    assert rows["y"]["calibration"]["calibration_id"] == "cal-displacement-probe-1-2026-09"
    assert validate_spec(load("pd-controller.json")).units["tau_f"].scale == pytest.approx(1e-3)
    path = validate_spec(load("path-heading.json"))
    assert path.spec["independent_variable"]["kind"] == "path_length"
    assert model.digest == validate_spec(oscillator()).digest


def test_lowering_makes_every_scale_explicit():
    lowered = lower(validate_spec(load("pd-controller.json")))
    assert lowered["schema"] == "ciw.model-lowered.v1"
    assert {item["symbol"]: item["scale"] for item in lowered["parameters"]}["tau_f"] == pytest.approx(1e-3)
    assert lowered["states"] == [{"symbol": "z", "scale": 1e-3}]


def mutate(path, value):
    spec = oscillator()
    target = spec
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return spec


@pytest.mark.parametrize("path, value, message", [
    (["dynamics", 1, "rhs"], {"op": "add", "args": [{"sym": "v"}, {"sym": "q"}]}, "Dimensionally inconsistent"),
    (["dynamics", 0, "rhs"], {"sym": "q"}, "declared m\\*s\\^-1"),
    (["dynamics", 0, "rhs"], {"sym": "undeclared"}, "undeclared symbol"),
    (["dynamics", 0, "rhs"], {"op": "sin", "args": [{"sym": "q"}]}, "dimensionless argument"),
    (["dynamics"], [], "one equation per state"),
    (["independent_variable", "unit"], "m", "time independent variable"),
    (["inputs", 0, "role"], "measured", "calibration status"),
    (["parameters", 0, "value"], 99.0, "outside its declared validity domain"),
    (["parameters", 0, "value"], True, "finite JSON number"),
    (["parameters", 0, "latex"], "\\input{secrets}", "unsupported LaTeX command"),
    (["parameters", 0, "latex"], "$x$", "restricted LaTeX name set"),
    (["uncertainty", "assumptions"], ["gaussian_parameters"], "no_process_noise"),
    (["uncertainty", "measurement_noise"], {"order": ["y"], "matrix": [[-1.0]]}, "nonnegative"),
    (["uncertainty", "parameters"], {"order": ["omega_0", "q"], "matrix": [[1.0, 0.0], [0.0, 1.0]]}, "distinct declared"),
    (["observations", 0, "calibration"], {"status": "uncalibrated", "calibration_id": "x"}, "cannot carry"),
    (["ports", 0, "symbol"], "y", "must name a declared input"),
    (["provenance", "kind"], "handwritten", "provenance.kind"),
])
def test_specification_refusals(path, value, message):
    with pytest.raises(SpecificationError, match=message):
        validate_spec(mutate(path, value))


def test_text_and_exploratory_latex_are_not_models():
    with pytest.raises(SpecificationError, match="latex_is_not_a_model"):
        validate_spec(r"\frac{dq}{dt} = v")
    with pytest.raises(SpecificationError, match="latex_is_not_a_model"):
        validate_spec({"schema": "ciw.exploratory-derivation.v1", "latex": "x"})


def test_commanded_input_cannot_claim_a_calibration():
    spec = oscillator()
    spec["inputs"][0]["calibration"] = {"status": "calibrated", "calibration_id": "cal-1"}
    with pytest.raises(SpecificationError, match="carries no measurement calibration"):
        validate_spec(spec)


def test_path_length_model_rejects_time_dimensioned_derivative():
    spec = load("path-heading.json")
    spec["parameters"][0]["unit"] = "s^-1"
    with pytest.raises(SpecificationError, match="declared m\\^-1"):
        validate_spec(spec)


def test_expression_latex_is_generated_from_structure():
    names = validate_spec(oscillator()).names()
    rhs = oscillator()["dynamics"][1]["rhs"]
    rendered = expression.to_latex(rhs, names)
    assert rendered == r"-2 \, \gamma \, v - {\omega_0}^{2} \, q + \frac{F}{m}"
    assert default_latex_name("omega_0") == r"\omega_{0}"
    assert default_latex_name("plant.y_meas") == r"y_{\mathrm{meas}}^{\mathrm{plant}}"


def test_rk4_reference_matches_the_analytic_oscillator_on_every_sample():
    model = validate_spec(oscillator())
    request = uniform_request(model, initial_state=[1.0, 0.0], inputs={"F": 0.0})
    reference = simulate_reference(model, request)
    run = make_demo_run()
    states = np.asarray(reference["states"])
    assert reference["sample_times"] == run["time_s"] and len(run["time_s"]) == 768
    assert np.max(np.abs(states[:, 0] - run["channels"]["q"]["values"])) < 1e-11
    assert np.max(np.abs(states[:, 1] - run["channels"]["v"]["values"])) < 1e-10
    assert np.max(np.abs(np.asarray(reference["derived"]["E"]) - run["channels"]["energy"]["values"])) < 1e-10
    assert reference["observations"]["y"] == [row[0] for row in reference["states"]]


@pytest.mark.parametrize("field, value, message", [
    ("sample_times", [0.0, 0.0], "strictly increasing"),
    ("interval", [0.0, 0.015625], "half-open end is excluded"),
    ("initial_state", [1.0], "in state order"),
    ("initial_state", [11.0, 0.0], "validity domain"),
    ("inputs", {}, "exactly"),
    ("spec_digest", "sha256:" + "0" * 64, "different model"),
])
def test_simulation_request_refusals(field, value, message):
    model = validate_spec(oscillator())
    request = uniform_request(model, initial_state=[1.0, 0.0], inputs={"F": 0.0}, count=4)
    request[field] = value
    with pytest.raises(SpecificationError, match=message):
        validate_request(model, request)


def test_solver_settings_are_explicit_and_bounded():
    model = validate_spec(oscillator())
    request = uniform_request(model, initial_state=[1.0, 0.0], inputs={"F": 0.0}, count=4)
    for key, value in (("reltol", 1.0), ("abstol", [1e-9]), ("maxiters", True), ("algorithm", "Rodas5")):
        broken = deepcopy(request)
        broken["solver"][key] = value
        with pytest.raises(SpecificationError):
            validate_request(model, broken)


def test_reference_halts_on_nonfinite_arithmetic_instead_of_raising():
    spec = oscillator()
    spec["parameters"][2]["value"] = 0.0
    del spec["validity_domain"]["bounds"]["mass"]
    model = validate_spec(spec)
    reference = simulate_reference(model, uniform_request(model, initial_state=[1.0, 0.0], inputs={"F": 0.0}, count=4))
    assert reference["status"] == "halted" and "nonfinite" in reference["detail"]
