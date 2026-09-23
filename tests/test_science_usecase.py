"""Use-case compiler: deterministic rule tables, 10:1 and 4:1 rules, capabilities and refusals."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import pytest

from ciw.science._common import Refusal, canonical_json
from ciw.science.ledger import KINDS, Ledger
from ciw.science.usecase import compile_use_case, retain_requirements, validate_use_case
from ciw.science.vocabulary import CAPABILITIES, CLAIM_CLASSES

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "science" / "usecases"
NAMES = sorted(path.name for path in EXAMPLES.glob("*.json"))


def load(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def refused(code, function, *args, **kwargs):
    with pytest.raises(Refusal) as caught:
        function(*args, **kwargs)
    assert caught.value.code == code, caught.value.to_dict()
    return caught.value


def names(items):
    return [item["name"] for item in items]


def test_examples_cover_several_domains():
    assert len({load(name)["domain"] for name in NAMES}) >= 6


@pytest.mark.parametrize("name", NAMES)
def test_compilation_is_deterministic_and_complete(name):
    record = load(name)
    first = compile_use_case(record, {"uncertainty_propagation"})
    second = compile_use_case(copy.deepcopy(record), ["uncertainty_propagation"])
    assert canonical_json(first) == canonical_json(second)
    assert validate_use_case(validate_use_case(record)) == validate_use_case(record)
    assert first["authorizes_actuation"] is False
    assert "do not authorize actuation" in first["operational_limitations"][0]
    items = (first["geometry_requirements"] + first["sensor_requirements"]["observables"]
             + first["sensor_requirements"]["existing_sensors"] + first["calibration_requirements"]
             + first["path_constraints"] + first["uncertainty_budget"]["allocations"] + first["acceptance_tests"]
             + first["required_evidence"] + first["required_capabilities"] + [first["fusion"]])
    assert all(isinstance(item["rationale"], str) and item["rationale"] for item in items)
    assert {item["kind"] for item in first["required_evidence"]} <= KINDS
    assert {item["claim_class"] for item in first["required_evidence"]} - {None} <= CLAIM_CLASSES
    assert {cap for item in first["required_capabilities"] for cap in item["any_of"]} <= CAPABILITIES
    assert "test-uncertainty ratio on a reference artefact" in names(first["acceptance_tests"])


@pytest.mark.parametrize("name", NAMES)
def test_budget_is_tolerance_over_four_split_by_rss(name):
    compiled = compile_use_case(load(name))
    tolerance = compiled["tolerance"]["value"]
    budget = compiled["uncertainty_budget"]
    assert budget["allowed_expanded_uncertainty"]["value"] == pytest.approx(tolerance / 4)
    assert budget["rss_check"]["value"] == pytest.approx(tolerance / 4)
    assert sum(item["variance_share"] for item in budget["allocations"]) == pytest.approx(1.0)
    first = budget["allocations"][0]
    assert first["allowed_expanded_uncertainty"]["value"] == pytest.approx(tolerance / 4 * math.sqrt(
        first["weight"] / sum(item["weight"] for item in budget["allocations"])))
    assert compiled["sensor_requirements"]["max_resolution"]["value"] == pytest.approx(tolerance / 10)
    assert compiled["missing_capabilities"] is None and compiled["available_capabilities"] is None


def test_curved_inspection_distinguishes_chord_and_intrinsic():
    compiled = compile_use_case(load("robotic-inspection-cylinder.json"))
    geometry = {item["name"]: item for item in compiled["geometry_requirements"]}
    assert {"tolerance_semantics", "chord_intrinsic_divergence", "intrinsically_flat"} <= geometry.keys()
    span = geometry["chord_intrinsic_divergence"]["critical_arc_span"]
    radius = geometry["chord_intrinsic_divergence"]["radius"]
    assert radius == {"value": pytest.approx(300.0), "unit": "mm"} and span["unit"] == "mm"
    s = span["value"]
    assert s - 2 * 300 * math.sin(s / 600) == pytest.approx(0.2, rel=1e-9)
    assert s == pytest.approx((24 * 300 ** 2 * 0.2) ** (1 / 3), rel=1e-3)  # s^3/(24 R^2) series
    assert "cylinder chord-vs-intrinsic discrimination" in names(compiled["acceptance_tests"])
    assert "clock-aligned multi-sensor fusion consistency (NIS)" in names(compiled["acceptance_tests"])
    assert "intrinsic_distance" in [item["observable"] for item in compiled["sensor_requirements"]["observables"]]
    groups = [item["any_of"] for item in compiled["required_capabilities"]]
    assert ["constant_curvature", "arbitrary_metric"] in groups and ["embedded_surfaces"] in groups
    assert ["asynchronous_streams"] in groups and ["clock_alignment"] in groups
    calibration = [(item["kind"], item.get("source"), item.get("target")) for item in compiled["calibration_requirements"]]
    assert ("transform", "tool", "machine") in calibration and ("transform", "camera", "tool") in calibration
    assert ("transform", "part_datum", "machine") in calibration and calibration[-2][0] == "clock_alignment"
    assert calibration[-1][0] == "reference_temperature"
    cycle = next(item for item in compiled["path_constraints"] if item["name"] == "cycle_time")
    assert cycle["max_cycle_time"] == {"value": 300.0, "unit": "s"}


def test_existing_sensor_failing_ten_to_one_is_flagged():
    compiled = compile_use_case(load("robotic-inspection-cylinder.json"))
    sensors = {item["sensor_id"]: item for item in compiled["sensor_requirements"]["existing_sensors"]}
    assert sensors["sl-scanner"]["verdict"] == "meets_10_to_1"
    assert sensors["robot-encoders"]["verdict"] == "fails_10_to_1"  # 0.05 mm > 0.2 mm / 10
    assert "encoder_displacement" in compiled["sensor_requirements"]["uncovered_observables"]
    assert any("robot-encoders" in line and "10:1" in line for line in compiled["operational_limitations"])
    record = load("robotic-inspection-cylinder.json")
    record["existing_sensors"][1]["resolution"] = {"value": 20, "unit": "um"}  # exactly tolerance / 10
    sensors = compile_use_case(record)["sensor_requirements"]["existing_sensors"]
    assert sensors[1]["verdict"] == "meets_10_to_1"


def test_freeform_needs_mesh_surfaces_when_not_available():
    record = load("injection-moulding-freeform.json")
    compiled = compile_use_case(record, {"uncertainty_propagation", "constant_curvature"})
    assert compiled["missing_capabilities"] == [{"any_of": ["mesh_surfaces"], "rationale": compiled[
        "required_capabilities"][1]["rationale"]}]
    assert compile_use_case(record, {"uncertainty_propagation", "mesh_surfaces"})["missing_capabilities"] == []
    mesh = next(item for item in compiled["geometry_requirements"] if item["name"] == "mesh_resolution")
    assert mesh["max_edge_length"]["value"] == pytest.approx(math.sqrt(8 * 5 * 0.01))
    assert compiled["fusion"]["used"] is False
    assert compiled["sensor_requirements"]["existing_sensors"][0]["verdict"] == "fails_10_to_1"


def test_fibre_placement_on_sphere_has_steering_limit_and_conjugate_points():
    compiled = compile_use_case(load("fibre-placement-sphere.json"), set())
    path = {item["name"]: item for item in compiled["path_constraints"]}
    steering = path["geodesic_or_steering_limit"]
    assert steering["max_geodesic_curvature"] == {"value": 0.5, "unit": "1/m"}
    assert steering["max_latitude_circle"]["value"] == pytest.approx(math.degrees(math.atan(0.8 / 2.0)))
    assert "great_circles_and_clairaut" in path
    geometry = {item["name"]: item for item in compiled["geometry_requirements"]}
    assert geometry["conjugate_points"]["antipodal_distance"]["value"] == pytest.approx(800 * math.pi)
    assert "geodesic oracle checks" in names(compiled["acceptance_tests"])
    assert ["conjugate_detection"] in [item["any_of"] for item in compiled["missing_capabilities"]]
    evidence = [(item["kind"], item["claim_class"]) for item in compiled["required_evidence"]]
    assert ("numerical_result", "computed") in evidence and evidence[-1] == ("authority_decision", "authorized")


def test_filament_winding_on_cylinder_uses_helices_and_winding_classes():
    compiled = compile_use_case(load("filament-winding-cylinder.json"))
    path = names(compiled["path_constraints"])
    assert "helices_are_geodesics" in path and "geodesic_or_slip_limit" in path
    assert ["winding_classes"] in [item["any_of"] for item in compiled["required_capabilities"]]
    record = load("filament-winding-cylinder.json")
    record["part"] = {"geometry": "sphere", "dimensions": {"radius": {"value": 0.5, "unit": "m"}}}
    slip = next(item for item in compile_use_case(record)["path_constraints"] if item["name"] == "geodesic_or_slip_limit")
    assert slip["max_geodesic_curvature"] == {"value": pytest.approx(0.2 / 500), "unit": "1/mm"}


def test_servo_stability_needs_a_lyapunov_certificate_and_accepts_angular_tolerances():
    compiled = compile_use_case(load("servo-stability-axis.json"), {"uncertainty_propagation"})
    assert "servo stability margin/Lyapunov certificate" in names(compiled["acceptance_tests"])
    assert compiled["fusion"]["used"] is False and compiled["missing_capabilities"] == []
    assert compiled["sensor_requirements"]["uncovered_observables"] == []
    record = load("servo-stability-axis.json")
    record["feature_tolerance"] = {"value": 5, "unit": "arcsec"}
    record["existing_sensors"][0].update(kind="rotary_encoder", resolution={"value": 1e-6, "unit": "rad"})
    sensor = compile_use_case(record)["sensor_requirements"]["existing_sensors"][0]
    assert sensor["verdict"] == "meets_10_to_1"  # 1 urad = 0.206 arcsec <= 0.5 arcsec
    record["existing_sensors"][0]["resolution"] = {"value": 1, "unit": "um"}
    assert compile_use_case(record)["sensor_requirements"]["existing_sensors"][0]["verdict"] == "not_checkable"


@pytest.mark.parametrize("mutate, code", [
    (lambda r: r.__setitem__("domain", "cold-fusion"), "unknown_domain"),
    (lambda r: r["part"].__setitem__("geometry", "klein_bottle"), "unknown_geometry"),
    (lambda r: r["part"].__setitem__("geometry", "axis"), "geometry_not_applicable"),
    (lambda r: r.__setitem__("feature_tolerance", {"value": 1, "unit": "s"}), "dimension_mismatch"),
    (lambda r: r.__setitem__("feature_tolerance", {"value": 1, "unit": "arcsec"}), "dimension_mismatch"),
    (lambda r: r["part"]["dimensions"].__setitem__("radius", {"value": 3, "unit": "kg"}), "dimension_mismatch"),
    (lambda r: r["part"]["dimensions"].pop("radius"), "malformed_record"),
    (lambda r: r["existing_sensors"][0].__setitem__("resolution", {"value": 0.5, "unit": "px"}), "dimension_mismatch"),
    (lambda r: r["existing_sensors"][0].__setitem__("observable", "filtered_state"), "not_a_raw_observable"),
    (lambda r: r["existing_sensors"][0].__setitem__("observable", "telepathy"), "unknown_observable"),
    (lambda r: r.__setitem__("process", {"friction_coefficient": 0.2}), "malformed_record"),
    (lambda r: r.__setitem__("feature_tolerance", {"value": 0, "unit": "mm"}), "out_of_domain"),
    (lambda r: r.__setitem__("schema", "ciw.use-case.v0"), "unsupported_schema"),
    (lambda r: r.__setitem__("sensors", []), "malformed_record"),
])
def test_refusals(mutate, code):
    record = load("robotic-inspection-cylinder.json")
    mutate(record)
    refused(code, compile_use_case, record)


def test_unknown_capability_is_refused():
    refused("unknown_capability", compile_use_case, load("servo-stability-axis.json"), {"telekinesis"})


def test_requirements_are_retained_only_when_they_match_a_fresh_compilation(tmp_path):
    ledger = Ledger.create(tmp_path / "ledger", now=lambda: datetime(2026, 9, 1, tzinfo=timezone.utc))
    record = load("bim-construction-state.json")
    compiled = compile_use_case(record, {"mesh_surfaces"})
    entry = retain_requirements(ledger, record, compiled)
    assert entry["kind"] == "use_case_requirements" and entry["body"]["requirements"] == compiled
    assert entry["body"]["use_case"] == validate_use_case(record)
    tampered = copy.deepcopy(compiled)
    tampered["uncertainty_budget"]["allowed_expanded_uncertainty"]["value"] *= 2
    refused("requirements_mismatch", retain_requirements, ledger, record, tampered)
    assert len(ledger) == 1 and ledger.verify().ok
