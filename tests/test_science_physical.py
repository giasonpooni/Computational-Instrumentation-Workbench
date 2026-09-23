"""Physical protocols: GUM budget arithmetic, En decisions, exclusions, synthetic flagging and retention."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import pytest

from ciw.science._common import Refusal
from ciw.science.ledger import Ledger
from ciw.science.physical import (coverage_factor, evaluate, protocol_identity, retain_measurements, retain_protocol,
                                  sampling_order, uncertainty_budget, validate_protocol)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "science" / "physical"


def load(name: str):
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


@pytest.fixture
def protocol():
    return load("cylinder-chord-protocol.json")


@pytest.fixture
def measurements():
    """The shipped example is labelled synthetic. These unit tests exercise the physical
    acceptance path, so they relabel an in-memory copy; no test data is presented as real."""
    records = load("cylinder-chord-measurements.json")
    assert {record["acquisition"] for record in records} == {"synthetic"}
    for record in records:
        record["acquisition"] = "physical"
    return records


def test_shipped_example_measurements_are_synthetic_only(protocol, predictions):
    result = evaluate(protocol, load("cylinder-chord-measurements.json"), predictions)
    assert result["physical_evidence"] is False
    assert result["summary"]["accepted"] == 0 and result["summary"]["overall"] == "accepted_synthetic_only"


@pytest.fixture
def predictions():
    return load("cylinder-chord-predictions.json")


def with_components(protocol, components, level=0.95):
    result = copy.deepcopy(protocol)
    result["uncertainty_budget"]["components"] = components
    result["coverage"]["level"] = level
    return result


def component(name, kind, distribution, value, unit="mm", dof=None, **extra):
    return {"name": name, "type": kind, "distribution": distribution, "value": {"value": value, "unit": unit},
            "sensitivity": extra.pop("sensitivity", 1.0), "dof": dof, **extra}


def refused(code, function, *args, **kwargs):
    with pytest.raises(Refusal) as caught:
        function(*args, **kwargs)
    assert caught.value.code == code, caught.value.to_dict()
    return caught.value


def by_point(result):
    return {point["point_id"]: point for point in result["points"]}


# ---------------------------------------------------------------------- protocol
def test_protocol_normalizes_idempotently_with_stable_identity(protocol):
    normalized = validate_protocol(protocol)
    assert validate_protocol(normalized) == normalized
    assert protocol_identity(protocol) == protocol_identity(normalized)
    changed = copy.deepcopy(protocol)
    changed["sampling_plan"]["seed"] += 1
    assert protocol_identity(changed) != protocol_identity(protocol)


def test_sampling_order_is_seeded_complete_and_deterministic(protocol):
    first, second = sampling_order(protocol), sampling_order(copy.deepcopy(protocol))
    assert first == second
    pairs = {(run["point_id"], run["repeat"]) for run in first}
    assert pairs == {(p, r) for p in ("arc-30", "arc-60", "arc-90") for r in (1, 2, 3)} and len(first) == 9
    reseeded = copy.deepcopy(protocol)
    reseeded["sampling_plan"]["seed"] = 7
    assert [run["point_id"] for run in sampling_order(reseeded)] != [run["point_id"] for run in first]
    ordered = copy.deepcopy(protocol)
    ordered["sampling_plan"]["randomize_order"] = False
    assert [(run["point_id"], run["repeat"]) for run in sampling_order(ordered)][:3] == [
        ("arc-30", 1), ("arc-30", 2), ("arc-30", 3)]


@pytest.mark.parametrize("mutate, code", [
    (lambda p: p["instrument"].__setitem__("observable", "filtered_state"), "not_a_raw_observable"),
    (lambda p: p["instrument"].__setitem__("observable", "vibes"), "unknown_observable"),
    (lambda p: p["instrument"].__setitem__("unit", "px"), "dimension_mismatch"),
    (lambda p: p["uncertainty_budget"]["components"][0].__setitem__("distribution", "cauchy"), "unknown_distribution"),
    (lambda p: p["uncertainty_budget"]["components"][1].__setitem__("type", "A"), "malformed_record"),
    (lambda p: p["acceptance"].__setitem__("rule", "abs_residual_le_limit"), "malformed_record"),
    (lambda p: p["coverage"].__setitem__("level", 0.9), "unsupported_coverage"),
    (lambda p: p["sampling_plan"].pop("seed"), "malformed_record"),
    (lambda p: p["sampling_plan"].__setitem__("repeats", 1), "out_of_domain"),
    (lambda p: p["object"]["geometry"]["dimensions"].pop("radius"), "malformed_record"),
    (lambda p: p["object"]["geometry"].__setitem__("type", "torus"), "unknown_geometry"),
    (lambda p: p["instrument"]["calibration"].__setitem__("valid_until", "2026-01-01T00:00:00+00:00"),
     "malformed_interval"),
    (lambda p: p["instrument"]["calibration"].__setitem__("valid_from", "2026-08-15T00:00:00"), "clock_unspecified"),
    (lambda p: p.__setitem__("extra", 1), "malformed_record"),
    (lambda p: p["uncertainty_budget"]["components"].append(p["uncertainty_budget"]["components"][0]),
     "duplicate_component"),
])
def test_protocol_refusals(protocol, mutate, code):
    mutate(protocol)
    refused(code, validate_protocol, protocol)


# ---------------------------------------------------------------------- budget
def test_budget_divisors_match_hand_calculation(protocol):
    budget = uncertainty_budget(with_components(protocol, [
        component("rect", "B", "rectangular", 0.3),
        component("tri", "B", "triangular", 0.6),
        component("u", "B", "u_shaped", 0.2),
        component("cert", "B", "normal", 0.1, coverage_factor=2.0),
        component("plain", "B", "normal", 0.04),
    ]))
    expected = {"rect": 0.3 / math.sqrt(3), "tri": 0.6 / math.sqrt(6), "u": 0.2 / math.sqrt(2), "cert": 0.05,
                "plain": 0.04}
    rows = {row["name"]: row for row in budget["components"]}
    for name, value in expected.items():
        assert rows[name]["standard_uncertainty"]["value"] == pytest.approx(value, rel=1e-15)
        assert rows[name]["contribution"] == {"value": pytest.approx(value, rel=1e-15), "unit": "mm"}
    combined = math.sqrt(sum(value ** 2 for value in expected.values()))
    assert budget["combined_standard_uncertainty"]["value"] == pytest.approx(combined, rel=1e-15)
    assert sum(row["percent_of_variance"] for row in budget["components"]) == pytest.approx(100.0)
    assert rows["rect"]["percent_of_variance"] == pytest.approx(100 * 0.03 / combined ** 2)
    assert budget["effective_dof"] is None and budget["coverage_factor"] == 1.96
    assert budget["expanded_uncertainty"]["value"] == pytest.approx(1.96 * combined)


def test_components_convert_into_the_measurand_unit(protocol):
    budget = uncertainty_budget(with_components(protocol, [
        component("in metres", "B", "rectangular", 0.0001, unit="m"),
        component("in micrometres", "B", "normal", 3.0, unit="um"),
        component("thermal", "B", "rectangular", 2.0, unit="K", sensitivity=-0.00115, sensitivity_unit="mm/K"),
        component("thermal in m/K", "B", "normal", 0.5, unit="degC", sensitivity=1e-6, sensitivity_unit="m/K"),
    ]))
    rows = {row["name"]: row["contribution"]["value"] for row in budget["components"]}
    assert rows["in metres"] == pytest.approx(0.1 / math.sqrt(3))
    assert rows["in micrometres"] == pytest.approx(0.003)
    assert rows["thermal"] == pytest.approx(0.00115 * 2.0 / math.sqrt(3))  # sign of c_i does not enter c_i u_i
    assert rows["thermal in m/K"] == pytest.approx(0.5 * 1e-6 * 1e3)  # a degC uncertainty is a K difference
    assert all(row["contribution"]["unit"] == "mm" for row in budget["components"])


@pytest.mark.parametrize("bad, code", [
    (component("time", "B", "normal", 1.0, unit="s"), "dimension_mismatch"),
    (component("undeclared sensitivity unit", "B", "normal", 1.0, unit="K"), "dimension_mismatch"),
    (component("wrong sensitivity", "B", "normal", 1.0, unit="mm", sensitivity_unit="mm/K"), "dimension_mismatch"),
    (component("type A without dof", "A", "normal", 1.0), "malformed_record"),
    (component("negative", "B", "normal", -1.0), "out_of_domain"),
    (component("k on rectangle", "B", "rectangular", 1.0, coverage_factor=2.0), "malformed_record"),
])
def test_incommensurable_or_malformed_components_are_refused(protocol, bad, code):
    refused(code, uncertainty_budget, with_components(protocol, [bad]))


def test_welch_satterthwaite_and_student_t_lookup(protocol):
    # u1 = 0.02 (dof 4) and u2 = 0.02 (infinite): nu = (2 u^2)^2 / (u^4 / 4) = 16 exactly.
    budget = uncertainty_budget(with_components(protocol, [
        component("repeat", "A", "normal", 0.02, dof=4),
        component("bound", "B", "rectangular", 0.02 * math.sqrt(3)),
    ]))
    assert budget["effective_dof"] == pytest.approx(16.0)
    assert budget["coverage_factor"] == pytest.approx(2.1199)
    assert budget["expanded_uncertainty"]["value"] == pytest.approx(2.1199 * math.sqrt(0.0008))
    at_99 = uncertainty_budget(with_components(protocol, budget_inputs := [
        component("repeat", "A", "normal", 0.02, dof=4), component("bound", "B", "rectangular", 0.02 * math.sqrt(3))],
        level=0.99))
    assert at_99["coverage_factor"] == pytest.approx(2.9208) and len(budget_inputs) == 2
    extra = uncertainty_budget(protocol, type_a=[component("run", "A", "normal", 0.004, dof=2)])
    assert extra["components"][-1]["name"] == "run" and extra["effective_dof"] < 30


def test_coverage_factor_table_and_interpolation_in_inverse_dof():
    assert coverage_factor(None) == 1.96 and coverage_factor(None, 0.99) == 2.5758
    assert coverage_factor(1) == 12.7062 and coverage_factor(30) == 2.0423 and coverage_factor(120) == 1.9799
    weight = (1 / 35 - 1 / 40) / (1 / 30 - 1 / 40)
    assert coverage_factor(35) == pytest.approx(2.0211 + weight * (2.0423 - 2.0211))
    assert coverage_factor(1000) == pytest.approx(1.96 + (120 / 1000) * (1.9799 - 1.96))
    assert coverage_factor(4, 0.99) == 4.6041
    refused("unsupported_coverage", coverage_factor, 5, 0.9)
    refused("out_of_domain", coverage_factor, 0.5)


def test_zero_budget_is_refused(protocol):
    refused("degenerate_budget", uncertainty_budget, with_components(protocol, [component("zero", "B", "normal", 0.0)]))


# ---------------------------------------------------------------------- evaluation
def test_example_is_accepted_with_hand_checked_en(protocol, measurements, predictions):
    result = evaluate(protocol, measurements, predictions)
    assert result["summary"] == {"points": 3, "accepted": 3, "accepted_synthetic_only": 0, "rejected": 0,
                                 "excluded": 0, "overall": "accepted"}
    assert result["physical_evidence"] is True and result["claim_class"] == "measured"
    point = by_point(result)["arc-60"]
    values = [50.0031, 49.9978, 50.0012]
    mean = sum(values) / 3
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / 2)
    assert point["mean"]["value"] == pytest.approx(mean) and point["std"]["value"] == pytest.approx(std)
    budget = uncertainty_budget(protocol, type_a=[component("repeatability:arc-60", "A", "normal",
                                                            std / math.sqrt(3), dof=2)])
    assert point["expanded_uncertainty"]["value"] == pytest.approx(budget["expanded_uncertainty"]["value"])
    assert point["budget"]["components"][-1]["dof"] == 2
    en = abs(mean - 50.0) / math.hypot(budget["expanded_uncertainty"]["value"], 1.96 * 0.0005)
    assert point["en"] == pytest.approx(en)
    assert point["decision"] == "accepted" and point["reasons"][0]["code"] == "en_within_1"


def test_shifted_point_is_rejected(protocol, measurements, predictions):
    for item in measurements:
        if item["point_id"] == "arc-90":
            item["value"]["value"] += 0.05
    result = evaluate(protocol, measurements, predictions)
    point = by_point(result)["arc-90"]
    assert point["decision"] == "rejected" and point["reasons"][0]["code"] == "en_exceeds_1" and point["en"] > 1
    assert result["summary"]["overall"] == "rejected"


def test_residual_rules(protocol, measurements, predictions):
    limit = copy.deepcopy(protocol)
    limit["acceptance"] = {"rule": "abs_residual_le_limit", "limit": {"value": 0.5, "unit": "um"}}
    points = by_point(evaluate(limit, measurements, predictions))
    assert points["arc-60"]["decision"] == "rejected"  # |residual| = 0.7 um > 0.5 um
    assert points["arc-60"]["reasons"][0]["code"] == "residual_exceeds_limit"
    assert points["arc-60"]["residual"]["value"] == pytest.approx(0.0007)
    assert points["arc-90"]["decision"] == "accepted"  # |residual| ~ 0.05 um
    within_u = copy.deepcopy(protocol)
    within_u["acceptance"] = {"rule": "abs_residual_le_U"}
    points = by_point(evaluate(within_u, measurements, predictions))
    assert all(point["reasons"][0]["code"] == "residual_within_U" for point in points.values())


def test_expired_and_not_yet_valid_calibration_exclude(protocol, measurements, predictions):
    expired = copy.deepcopy(protocol)
    expired["instrument"]["calibration"]["valid_until"] = "2026-09-01T10:10:00+00:00"
    result = evaluate(expired, measurements, predictions)
    codes = {point["point_id"]: {reason["code"] for reason in point["reasons"]} for point in result["points"]}
    assert any("calibration_expired" in item for item in codes.values())
    assert result["summary"]["overall"] == "inconclusive"
    assert all(point["decision"] == "excluded" for point in result["points"] if "calibration_expired" in
               codes[point["point_id"]])
    early = copy.deepcopy(protocol)
    early["instrument"]["calibration"]["valid_from"] = "2026-09-02T00:00:00+02:00"
    result = evaluate(early, measurements, predictions)
    assert all(point["reasons"][0]["code"] == "calibration_not_yet_valid" for point in result["points"])
    lenient = copy.deepcopy(early)
    lenient["exclusion"]["require_valid_calibration"] = False
    result = evaluate(lenient, measurements, predictions)
    assert result["summary"]["accepted"] == 3 and not by_point(result)["arc-30"]["checks"]["calibration_valid"]


def test_datum_not_established_excludes(protocol, measurements, predictions):
    protocol["datum"]["established"] = False
    result = evaluate(protocol, measurements, predictions)
    assert result["summary"]["excluded"] == 3
    assert all(point["reasons"][0]["code"] == "datum_not_established" for point in result["points"])
    protocol["exclusion"]["require_datum"] = False
    assert evaluate(protocol, measurements, predictions)["summary"]["overall"] == "accepted"


def test_environment_out_of_limits_or_unrecorded_excludes(protocol, measurements, predictions):
    hot = next(item for item in measurements if item["point_id"] == "arc-30" and item["repeat"] == 2)
    hot["environment"] = {"temperature": {"value": 298.15, "unit": "K"}, "humidity": {"value": 45, "unit": "percent"}}
    bare = next(item for item in measurements if item["point_id"] == "arc-60" and item["repeat"] == 1)
    bare.pop("environment")
    points = by_point(evaluate(protocol, measurements, predictions))
    assert points["arc-30"]["decision"] == "excluded"
    assert points["arc-30"]["reasons"] == [{"code": "environment_out_of_limits",
                                            "message": "Environment outside the declared limits", "repeats": [2]}]
    assert points["arc-60"]["reasons"][0]["code"] == "environment_unrecorded"
    assert points["arc-90"]["decision"] == "accepted"
    hot["environment"]["temperature"] = {"value": 21.9, "unit": "degC"}  # inside 18-22 degC
    bare["environment"] = {"temperature": {"value": 293.15, "unit": "K"}, "humidity": {"value": 50, "unit": "percent"}}
    assert evaluate(protocol, measurements, predictions)["summary"]["overall"] == "accepted"


def test_repeatability_above_limit_excludes(protocol, measurements, predictions):
    for item in measurements:
        if item["point_id"] == "arc-30" and item["repeat"] == 3:
            item["value"]["value"] += 0.1
    point = by_point(evaluate(protocol, measurements, predictions))["arc-30"]
    assert point["decision"] == "excluded"
    assert point["reasons"][0]["code"] == "repeatability_exceeded" and point["reasons"][0]["std"] > 0.02


def test_incomplete_sampling_and_missing_prediction_exclude(protocol, measurements, predictions):
    measurements = [item for item in measurements if not (item["point_id"] == "arc-30" and item["repeat"] == 3)]
    predictions.pop("arc-90")
    points = by_point(evaluate(protocol, measurements, predictions))
    assert points["arc-30"]["reasons"][0]["code"] == "incomplete_sampling"
    assert points["arc-90"]["reasons"][0]["code"] == "prediction_missing" and "en" not in points["arc-90"]
    assert points["arc-60"]["decision"] == "accepted"


def test_any_synthetic_measurement_denies_physical_acceptance(protocol, measurements, predictions):
    measurements[4]["acquisition"] = "synthetic"
    result = evaluate(protocol, measurements, predictions)
    assert result["physical_evidence"] is False and result["claim_class"] == "computed"
    assert {point["decision"] for point in result["points"]} == {"accepted_synthetic_only"}
    assert result["summary"]["accepted"] == 0 and result["summary"]["overall"] == "accepted_synthetic_only"
    assert all(point["reasons"][-1]["code"] == "synthetic_only" for point in result["points"])


@pytest.mark.parametrize("mutate, code", [
    (lambda m: m[0].pop("clock"), "clock_unspecified"),
    (lambda m: m[0].__setitem__("clock", None), "clock_unspecified"),
    (lambda m: m[0].__setitem__("clock", " "), "clock_unspecified"),
    (lambda m: m[0].__setitem__("acquired_at", "2026-09-01T10:00:00"), "clock_unspecified"),
    (lambda m: m[1].__setitem__("clock", "robot-controller"), "clock_mismatch"),
    (lambda m: m[0].__setitem__("point_id", "arc-120"), "unknown_point"),
    (lambda m: m[1].update(point_id=m[0]["point_id"], repeat=m[0]["repeat"]), "duplicate_measurement"),
    (lambda m: m[0].__setitem__("repeat", 4), "out_of_domain"),
    (lambda m: m[0].__setitem__("acquisition", "planned"), "not_a_measurement"),
    (lambda m: m[0].__setitem__("acquisition", "imagined"), "unknown_acquisition"),
    (lambda m: m[0].__setitem__("value", {"value": 1.0, "unit": "s"}), "dimension_mismatch"),
])
def test_measurement_refusals(protocol, measurements, predictions, mutate, code):
    mutate(measurements)
    refused(code, evaluate, protocol, measurements, predictions)


def test_predictions_are_validated(protocol, measurements, predictions):
    predictions["arc-45"] = predictions["arc-30"]
    refused("unknown_point", evaluate, protocol, measurements, predictions)
    predictions.pop("arc-45")
    predictions["arc-30"]["standard_uncertainty"] = {"value": 1, "unit": "K"}
    refused("dimension_mismatch", evaluate, protocol, measurements, predictions)


def test_measurements_in_metres_are_converted(protocol, measurements, predictions):
    for item in measurements:
        item["value"] = {"value": item["value"]["value"] / 1000.0, "unit": "m"}
    result = evaluate(protocol, measurements, predictions)
    assert result["summary"]["overall"] == "accepted"
    assert by_point(result)["arc-60"]["mean"]["value"] == pytest.approx((50.0031 + 49.9978 + 50.0012) / 3)


# ---------------------------------------------------------------------- ledger
def test_protocol_and_measurements_are_retained(tmp_path, protocol, measurements):
    ledger = Ledger.create(tmp_path / "ledger", now=lambda: datetime(2026, 9, 1, 12, tzinfo=timezone.utc))
    measurements[0]["acquisition"] = "synthetic"
    entry = retain_protocol(ledger, protocol)
    assert entry["kind"] == "physical_protocol"
    assert entry["body"]["protocol_identity"] == protocol_identity(protocol)
    observations = retain_measurements(ledger, entry["entry_id"], measurements)
    assert len(observations) == 9 and all(item["kind"] == "observation" for item in observations)
    assert all(item["refs"] == [entry["entry_id"]] for item in observations)
    assert [item["body"]["acquisition"] for item in observations] == [m["acquisition"] for m in measurements]
    assert observations[0]["body"]["observable"] == "camera_chord" and observations[0]["body"]["unit"] == "mm"
    assert Ledger.open(tmp_path / "ledger").verify().ok
    refused("wrong_entry_kind", retain_measurements, ledger, observations[0]["entry_id"], measurements)
    refused("clock_unspecified", retain_measurements, ledger, entry["entry_id"],
            [{**measurements[0], "clock": None}])
    assert len(ledger) == 10
