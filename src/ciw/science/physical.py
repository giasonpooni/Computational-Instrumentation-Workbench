"""Physical experiment protocols, GUM uncertainty budgets and En acceptance decisions.

Scope: a ``ciw.physical-protocol.v1`` record pre-registers one measurement
experiment on a physical coupon: the object and its geometry, the instrument
and its calibration validity window, the datum, a sampling plan with repeats
and a seeded run order, environment limits, a declared uncertainty budget, a
coverage level, an acceptance rule and exclusion criteria. ``evaluate`` compares
repeated measurements with model predictions point by point, adds each point's
repeatability as a Type A component, and returns accepted, rejected or excluded
with explicit reasons. Protocols and measurements can be retained in the ledger.

Honest limits: the budget is the first-order GUM law of propagation for
*uncorrelated* inputs; correlations, non-linear models and Monte Carlo (GUM
Supplement 1) are out of scope. Sensitivity coefficients are declared, not
derived. The coverage factor comes from a tabulated two-sided Student t
(95 % and 99 % only) interpolated linearly in 1/nu, so it is approximate between
table rows. Instrument resolution is not added automatically; declare it as a
component. Acceptance is simple acceptance without guard bands. Timestamps on
one declared clock are compared as civil time with offsets; no clock model is
applied. Synthetic data can pass a check but never yields physical evidence.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from ._common import Refusal, content_identity, finite, integer, iso_time, mapping, require_keys, text
from .units import Quantity, convert, dimension_text, parse_unit, require_dimension
from .vocabulary import ACQUISITION_KINDS, OBSERVABLES

SCHEMA = "ciw.physical-protocol.v1"
EVALUATION_SCHEMA = "ciw.physical-evaluation.v1"
PROTOCOL_BODY = "ciw.science.physical-protocol.v1"
OBSERVATION_BODY = "ciw.science.observation.v1"
GEOMETRY_TYPES = {"plate": set(), "cylinder": {"radius"}, "sphere": {"radius"}, "curved": set()}
# Raw observables and the dimensions a physical instrument may report them in.
# ``filtered_state`` is an estimator output and never a physical instrument reading.
OBSERVABLE_UNITS = {"intrinsic_distance": ("m",), "camera_chord": ("m",), "image_residual": ("px",),
                    "encoder_displacement": ("m", "rad"), "tracker_position": ("m",), "imu_orientation": ("rad",),
                    "reconstructed_geometry": ("m",)}
DIVISORS = {"normal": 1.0, "rectangular": math.sqrt(3.0), "triangular": math.sqrt(6.0), "u_shaped": math.sqrt(2.0)}
RULES = frozenset({"en_le_1", "abs_residual_le_U", "abs_residual_le_limit"})
MEASURED = frozenset({"physical", "synthetic"})
MAX_POINTS, MAX_REPEATS, MAX_RUNS, MAX_COMPONENTS, MAX_ITEMS = 1024, 1000, 100_000, 64, 64

# Two-sided Student t quantiles t_{(1+p)/2}(nu); the last entry is nu = infinity.
_T_NU = tuple(range(1, 31)) + (40, 60, 120)
_T_TABLE = {
    0.95: (12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622, 2.2281, 2.2010, 2.1788,
           2.1604, 2.1448, 2.1314, 2.1199, 2.1098, 2.1009, 2.0930, 2.0860, 2.0796, 2.0739, 2.0687, 2.0639,
           2.0595, 2.0555, 2.0518, 2.0484, 2.0452, 2.0423, 2.0211, 2.0003, 1.9799, 1.9600),
    0.99: (63.6567, 9.9248, 5.8409, 4.6041, 4.0321, 3.7074, 3.4995, 3.3554, 3.2498, 3.1693, 3.1058, 3.0545,
           3.0123, 2.9768, 2.9467, 2.9208, 2.8982, 2.8784, 2.8609, 2.8453, 2.8314, 2.8188, 2.8073, 2.7969,
           2.7874, 2.7787, 2.7707, 2.7633, 2.7564, 2.7500, 2.7045, 2.6603, 2.6174, 2.5758),
}

__all__ = ["SCHEMA", "validate_protocol", "protocol_identity", "sampling_order", "coverage_factor",
           "uncertainty_budget", "evaluate", "retain_protocol", "retain_measurements"]


# ---------------------------------------------------------------------- helpers
def _quantity(value: Any, name: str, like: str | tuple[str, ...] | None = None, *,
              positive: bool = False, nonnegative: bool = False) -> dict:
    """A scalar ``{value, unit}`` quantity, optionally required to share a dimension with ``like``."""
    require_keys(value, name, {"value", "unit"})
    quantity = Quantity.from_json(value, name)
    if isinstance(quantity.value, list):
        raise Refusal("malformed_record", f"{name} must be a scalar quantity")
    if like is not None:
        options = (like,) if isinstance(like, str) else like
        dimension = parse_unit(quantity.unit).dimension
        if all(parse_unit(option).dimension != dimension for option in options):
            require_dimension(quantity.unit, options[0], name)
    if positive and quantity.value <= 0:
        raise Refusal("out_of_domain", f"{name} must be > 0")
    if nonnegative and quantity.value < 0:
        raise Refusal("out_of_domain", f"{name} must be >= 0")
    return {"value": float(quantity.value), "unit": quantity.unit}


def _in(value: dict, unit: str) -> float:
    """Absolute magnitude of a quantity in ``unit`` (offsets applied for degC and the like)."""
    return float(convert(value["value"], value["unit"], unit))


def _span(value: dict, unit: str) -> float:
    """Magnitude of a difference or uncertainty in ``unit``; offsets never enter."""
    return value["value"] * parse_unit(value["unit"]).scale / parse_unit(unit).scale


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise Refusal("malformed_record", f"{name} must be a boolean")
    return value


def _items(value: Any, name: str, limit: int, minimum: int = 1) -> list:
    if not isinstance(value, list) or not minimum <= len(value):
        raise Refusal("malformed_record", f"{name} must be an array with at least {minimum} item(s)")
    if len(value) > limit:
        raise Refusal("oversized_input", f"{name} exceeds {limit} items")
    return value


def _reason(code: str, message: str, **detail: Any) -> dict:
    return {"code": code, "message": message, **detail}


def _choice(value: Any, allowed: Any, name: str, code: str = "malformed_record") -> str:
    if not isinstance(value, str) or value not in allowed:
        raise Refusal(code, f"{name} {value!r} is not one of the declared terms", allowed=sorted(allowed))
    return value


def _level(value: Any) -> float:
    level = finite(value, "coverage level")
    if level not in _T_TABLE:
        raise Refusal("unsupported_coverage", "Coverage level must be 0.95 or 0.99", allowed=sorted(_T_TABLE))
    return level


# ---------------------------------------------------------------------- protocol
def _instrument(value: Any) -> dict:
    value = require_keys(value, "instrument", {"instrument_id", "kind", "observable", "unit", "resolution",
                                               "calibration"})
    observable = _choice(value["observable"], OBSERVABLES, "instrument.observable", "unknown_observable")
    if observable not in OBSERVABLE_UNITS:
        raise Refusal("not_a_raw_observable", f"{observable!r} is an estimator output, not an instrument reading")
    unit = parse_unit(value["unit"]).expression
    if all(parse_unit(unit).dimension != parse_unit(option).dimension for option in OBSERVABLE_UNITS[observable]):
        raise Refusal("dimension_mismatch", f"instrument.unit {unit!r} cannot report {observable!r}",
                      expected=list(OBSERVABLE_UNITS[observable]))
    calibration = require_keys(value["calibration"], "instrument.calibration",
                               {"calibration_id", "version", "valid_from", "valid_until"}, {"digest"})
    if iso_time(calibration["valid_from"], "calibration.valid_from") >= iso_time(
            calibration["valid_until"], "calibration.valid_until"):
        raise Refusal("malformed_interval", "calibration validity must be a nonempty interval")
    normalized = {"calibration_id": text(calibration["calibration_id"], "calibration.calibration_id", 256),
                  "version": text(calibration["version"], "calibration.version", 64),
                  "valid_from": calibration["valid_from"], "valid_until": calibration["valid_until"]}
    if "digest" in calibration:
        digest = calibration["digest"]
        if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71:
            raise Refusal("malformed_record", "calibration.digest must be a sha256: content identity")
        normalized["digest"] = digest
    return {"instrument_id": text(value["instrument_id"], "instrument.instrument_id", 256),
            "kind": text(value["kind"], "instrument.kind", 128), "observable": observable, "unit": unit,
            "resolution": _quantity(value["resolution"], "instrument.resolution", unit, positive=True),
            "calibration": normalized}


def _object(value: Any, unit: str) -> dict:
    value = require_keys(value, "object", {"coupon_id", "geometry", "material", "nominal_tolerance"})
    geometry = require_keys(value["geometry"], "object.geometry", {"type", "dimensions"})
    kind = _choice(geometry["type"], GEOMETRY_TYPES, "object.geometry.type", "unknown_geometry")
    dimensions = mapping(geometry["dimensions"], "object.geometry.dimensions")
    if len(dimensions) > 16:
        raise Refusal("oversized_input", "object.geometry.dimensions exceeds 16 entries")
    missing = GEOMETRY_TYPES[kind] - dimensions.keys()
    if missing:
        raise Refusal("malformed_record", f"A {kind} geometry must declare {sorted(missing)}")
    normalized = {text(name, "dimension name", 64): _quantity(item, f"dimensions.{name}", "m", positive=True)
                  for name, item in sorted(dimensions.items())}
    return {"coupon_id": text(value["coupon_id"], "object.coupon_id", 256),
            "geometry": {"type": kind, "dimensions": normalized},
            "material": text(value["material"], "object.material", 256),
            "nominal_tolerance": _quantity(value["nominal_tolerance"], "object.nominal_tolerance", unit, positive=True)}


def _component(value: Any, measurand: str, name: str) -> dict:
    value = require_keys(value, name, {"name", "type", "distribution", "value", "sensitivity", "dof"},
                         {"sensitivity_unit", "coverage_factor"})
    kind = _choice(value["type"], {"A", "B"}, f"{name}.type")
    distribution = _choice(value["distribution"], DIVISORS, f"{name}.distribution", "unknown_distribution")
    if kind == "A" and distribution != "normal":
        raise Refusal("malformed_record", f"{name}: a Type A component is a normal standard uncertainty")
    dof = value["dof"]
    if dof is not None or kind == "A":
        dof = integer(dof, f"{name}.dof", minimum=1)
    quantity = _quantity(value["value"], f"{name}.value", nonnegative=True)
    sensitivity_unit = parse_unit(value.get("sensitivity_unit", "1")).expression
    product = tuple(a + b for a, b in zip(parse_unit(quantity["unit"]).dimension,
                                          parse_unit(sensitivity_unit).dimension))
    if product != parse_unit(measurand).dimension:
        raise Refusal("dimension_mismatch", f"{name}: sensitivity x value is not commensurable with {measurand!r}",
                      found=dimension_text(product), expected=dimension_text(parse_unit(measurand).dimension))
    result = {"name": text(value["name"], f"{name}.name", 128), "type": kind, "distribution": distribution,
              "value": quantity, "sensitivity": finite(value["sensitivity"], f"{name}.sensitivity"), "dof": dof}
    if "sensitivity_unit" in value:
        result["sensitivity_unit"] = sensitivity_unit
    if "coverage_factor" in value:
        if kind != "B" or distribution != "normal":
            raise Refusal("malformed_record", f"{name}: coverage_factor applies only to a Type B normal component")
        result["coverage_factor"] = finite(value["coverage_factor"], f"{name}.coverage_factor", minimum=0.0,
                                           exclusive_minimum=True)
    return result


def _limits(value: Any, name: str, like: str) -> dict:
    value = require_keys(value, name, {"min", "max"})
    low, high = _quantity(value["min"], f"{name}.min", like), _quantity(value["max"], f"{name}.max", like)
    if _in(low, like) >= _in(high, like):
        raise Refusal("malformed_interval", f"{name}.min must be below {name}.max")
    return {"min": low, "max": high}


def validate_protocol(record: Any) -> dict:
    """Validate a ``ciw.physical-protocol.v1`` record and return its normalized form (idempotent)."""
    record = require_keys(record, "protocol", {"schema", "protocol_id", "object", "instrument", "datum",
                                               "sampling_plan", "environment_limits", "uncertainty_budget",
                                               "coverage", "acceptance", "exclusion"}, {"description"})
    if record["schema"] != SCHEMA:
        raise Refusal("unsupported_schema", f"Expected {SCHEMA}")
    instrument = _instrument(record["instrument"])
    unit = instrument["unit"]
    datum = require_keys(record["datum"], "datum", {"frame_id", "features", "established"})
    features = [text(item, "datum.features[]", 128) for item in _items(datum["features"], "datum.features",
                                                                       MAX_ITEMS, minimum=0)]
    if datum["established"] is True and not features:
        raise Refusal("malformed_record", "An established datum must list the features that establish it")
    plan = require_keys(record["sampling_plan"], "sampling_plan", {"points", "repeats", "randomize_order"}, {"seed"})
    points, seen = [], set()
    for index, point in enumerate(_items(plan["points"], "sampling_plan.points", MAX_POINTS)):
        point = require_keys(point, f"points[{index}]", {"point_id", "description"}, {"nominal"})
        point_id = text(point["point_id"], f"points[{index}].point_id", 128)
        if point_id in seen:
            raise Refusal("duplicate_point", f"Point {point_id!r} is declared twice")
        seen.add(point_id)
        item = {"point_id": point_id, "description": text(point["description"], f"points[{index}].description", 512)}
        if "nominal" in point:
            item["nominal"] = _quantity(point["nominal"], f"points[{index}].nominal", unit)
        points.append(item)
    repeats = integer(plan["repeats"], "sampling_plan.repeats", minimum=2, maximum=MAX_REPEATS)
    if len(points) * repeats > MAX_RUNS:
        raise Refusal("oversized_input", f"The sampling plan exceeds {MAX_RUNS} runs")
    sampling = {"points": points, "repeats": repeats,
                "randomize_order": _boolean(plan["randomize_order"], "sampling_plan.randomize_order")}
    if "seed" in plan:
        sampling["seed"] = integer(plan["seed"], "sampling_plan.seed", minimum=0, maximum=2 ** 63 - 1)
    elif sampling["randomize_order"]:
        raise Refusal("malformed_record", "A randomized sampling order must declare its seed")
    environment = require_keys(record["environment_limits"], "environment_limits", {"temperature"}, {"humidity"})
    limits = {"temperature": _limits(environment["temperature"], "environment_limits.temperature", "K")}
    if "humidity" in environment:
        limits["humidity"] = _limits(environment["humidity"], "environment_limits.humidity", "percent")
    budget = require_keys(record["uncertainty_budget"], "uncertainty_budget", {"components"})
    components = [_component(item, unit, f"uncertainty_budget.components[{index}]")
                  for index, item in enumerate(_items(budget["components"], "uncertainty_budget.components",
                                                      MAX_COMPONENTS))]
    if len({item["name"] for item in components}) != len(components):
        raise Refusal("duplicate_component", "Uncertainty component names must be unique")
    coverage = require_keys(record["coverage"], "coverage", {"level"})
    acceptance = require_keys(record["acceptance"], "acceptance", {"rule"}, {"limit"})
    _choice(acceptance["rule"], RULES, "acceptance.rule", "unknown_rule")
    if (acceptance["rule"] == "abs_residual_le_limit") != ("limit" in acceptance):
        raise Refusal("malformed_record", "acceptance.limit is required by, and only by, abs_residual_le_limit")
    rule = {"rule": acceptance["rule"]}
    if "limit" in acceptance:
        rule["limit"] = _quantity(acceptance["limit"], "acceptance.limit", unit, positive=True)
    exclusion = require_keys(record["exclusion"], "exclusion", {"max_repeatability_std", "require_datum",
                                                                "require_valid_calibration"})
    result = {
        "schema": SCHEMA, "protocol_id": text(record["protocol_id"], "protocol_id", 256),
        "object": _object(record["object"], unit), "instrument": instrument,
        "datum": {"frame_id": text(datum["frame_id"], "datum.frame_id", 128), "features": features,
                  "established": _boolean(datum["established"], "datum.established")},
        "sampling_plan": sampling, "environment_limits": limits, "uncertainty_budget": {"components": components},
        "coverage": {"level": _level(coverage["level"])}, "acceptance": rule,
        "exclusion": {"max_repeatability_std": _quantity(exclusion["max_repeatability_std"],
                                                         "exclusion.max_repeatability_std", unit, positive=True),
                      "require_datum": _boolean(exclusion["require_datum"], "exclusion.require_datum"),
                      "require_valid_calibration": _boolean(exclusion["require_valid_calibration"],
                                                            "exclusion.require_valid_calibration")},
    }
    if "description" in record:
        result["description"] = text(record["description"], "description", 4096)
    return result


def protocol_identity(record: Any) -> str:
    return content_identity(validate_protocol(record))


def sampling_order(protocol: Any) -> list[dict]:
    """Run order: every (point, repeat) pair, shuffled by a seeded SHA-256 Fisher-Yates when randomized.

    The generator is defined here rather than borrowed from numpy so the order
    is identical across platforms and library versions for the same seed.
    """
    protocol = validate_protocol(protocol)
    plan = protocol["sampling_plan"]
    runs = [(point["point_id"], repeat) for point in plan["points"] for repeat in range(1, plan["repeats"] + 1)]
    if plan["randomize_order"]:
        for index in range(len(runs) - 1, 0, -1):
            digest = hashlib.sha256(f"ciw.sampling-order.v1:{plan['seed']}:{index}".encode()).digest()
            other = int.from_bytes(digest, "big") % (index + 1)
            runs[index], runs[other] = runs[other], runs[index]
    return [{"order": order, "point_id": point_id, "repeat": repeat} for order, (point_id, repeat) in enumerate(runs)]


# ---------------------------------------------------------------------- uncertainty
def coverage_factor(dof: float | None, level: float = 0.95) -> float:
    """Two-sided Student t coverage factor; ``dof=None`` means infinite degrees of freedom."""
    table = _T_TABLE[_level(level)]
    if dof is None:
        return table[-1]
    dof = finite(dof, "degrees of freedom")
    if dof < 1.0 - 1e-9:
        raise Refusal("out_of_domain", "Degrees of freedom must be >= 1")
    dof = max(dof, 1.0)
    nodes = [1.0 / nu for nu in _T_NU] + [0.0]
    x = 1.0 / dof
    for index in range(len(nodes) - 1):
        high, low = nodes[index], nodes[index + 1]
        if low <= x <= high:
            weight = (x - low) / (high - low)
            return table[index + 1] + weight * (table[index] - table[index + 1])
    return table[0]


def _contribution(component: dict, measurand: str) -> dict:
    value = component["value"]
    divisor = (component.get("coverage_factor", 1.0) if component["distribution"] == "normal"
               else DIVISORS[component["distribution"]])
    standard = value["value"] / divisor
    sensitivity_unit = component.get("sensitivity_unit", "1")
    scale = parse_unit(value["unit"]).scale * parse_unit(sensitivity_unit).scale / parse_unit(measurand).scale
    return {"name": component["name"], "type": component["type"], "distribution": component["distribution"],
            "value": value, "divisor": divisor, "standard_uncertainty": {"value": standard, "unit": value["unit"]},
            "sensitivity": component["sensitivity"], "sensitivity_unit": sensitivity_unit,
            "contribution": {"value": abs(component["sensitivity"]) * standard * scale, "unit": measurand},
            "dof": component["dof"]}


def _budget(protocol: dict, type_a: Any) -> dict:
    unit, level = protocol["instrument"]["unit"], protocol["coverage"]["level"]
    components = list(protocol["uncertainty_budget"]["components"])
    for index, item in enumerate(_items(type_a, "type_a", MAX_COMPONENTS, minimum=0) if type_a is not None else []):
        component = _component(item, unit, f"type_a[{index}]")
        if component["type"] != "A":
            raise Refusal("malformed_record", f"type_a[{index}] must be a Type A component")
        components.append(component)
    if len({item["name"] for item in components}) != len(components):
        raise Refusal("duplicate_component", "Uncertainty component names must be unique")
    rows = [_contribution(item, unit) for item in components]
    variance = sum(row["contribution"]["value"] ** 2 for row in rows)
    combined = math.sqrt(variance)
    denominator = sum(row["contribution"]["value"] ** 4 / row["dof"] for row in rows if row["dof"] is not None)
    effective = combined ** 4 / denominator if variance > 0 and denominator > 0 else None
    for row in rows:
        row["percent_of_variance"] = 100.0 * row["contribution"]["value"] ** 2 / variance if variance > 0 else 0.0
    k = coverage_factor(effective, level)
    return {"measurand_unit": unit, "coverage_level": level, "components": rows,
            "combined_standard_uncertainty": {"value": combined, "unit": unit},
            "effective_dof": effective, "coverage_factor": k,
            "expanded_uncertainty": {"value": k * combined, "unit": unit},
            "method": "GUM first-order propagation of uncorrelated inputs; Welch-Satterthwaite effective dof "
                      "(null = infinite); two-sided Student t interpolated linearly in 1/nu"}


def uncertainty_budget(protocol: Any, *, type_a: list[dict] | None = None) -> dict:
    """Combined, effective-dof and expanded uncertainty in the instrument (measurand) unit.

    ``type_a`` adds Type A components (normal standard uncertainties with finite dof).
    """
    budget = _budget(validate_protocol(protocol), type_a)
    if budget["combined_standard_uncertainty"]["value"] == 0.0:
        raise Refusal("degenerate_budget", "The combined standard uncertainty is zero")
    return budget


# ---------------------------------------------------------------------- measurements
def _environment(value: Any, name: str) -> dict:
    value = require_keys(value, name, set(), {"temperature", "humidity"})
    result = {}
    if "temperature" in value:
        result["temperature"] = _quantity(value["temperature"], f"{name}.temperature", "K")
    if "humidity" in value:
        result["humidity"] = _quantity(value["humidity"], f"{name}.humidity", "percent")
    return result


def _measurements(protocol: dict, measurements: Any) -> list[dict]:
    unit, plan = protocol["instrument"]["unit"], protocol["sampling_plan"]
    known = {point["point_id"] for point in plan["points"]}
    result, seen, clocks = [], set(), set()
    for index, item in enumerate(_items(measurements, "measurements", MAX_RUNS)):
        name = f"measurements[{index}]"
        mapping(item, name)
        if not isinstance(item.get("clock"), str) or not item["clock"].strip():
            raise Refusal("clock_unspecified", f"{name} must name the clock of its timestamp")
        item = require_keys(item, name, {"point_id", "repeat", "value", "acquired_at", "clock", "acquisition"},
                            {"environment"})
        iso_time(item["acquired_at"], f"{name}.acquired_at")
        if item["point_id"] not in known:
            raise Refusal("unknown_point", f"{name} names undeclared point {item['point_id']!r}")
        repeat = integer(item["repeat"], f"{name}.repeat", minimum=1, maximum=plan["repeats"])
        if (item["point_id"], repeat) in seen:
            raise Refusal("duplicate_measurement", f"{name} repeats point {item['point_id']!r} repeat {repeat}")
        seen.add((item["point_id"], repeat))
        acquisition = _choice(item["acquisition"], ACQUISITION_KINDS, f"{name}.acquisition", "unknown_acquisition")
        if acquisition not in MEASURED:
            raise Refusal("not_a_measurement", f"{name}: a {acquisition!r} record is not a measurement")
        clocks.add(text(item["clock"], f"{name}.clock", 128))
        result.append({"point_id": item["point_id"], "repeat": repeat,
                       "value": _quantity(item["value"], f"{name}.value", unit),
                       "acquired_at": item["acquired_at"], "clock": item["clock"],
                       "environment": _environment(item.get("environment", {}), f"{name}.environment"),
                       "acquisition": acquisition})
    if len(clocks) > 1:
        raise Refusal("clock_mismatch", "Measurements on different clocks are not compared without a clock mapping",
                      clocks=sorted(clocks))
    return result


def _predictions(protocol: dict, predictions: Any) -> dict:
    unit = protocol["instrument"]["unit"]
    known = {point["point_id"] for point in protocol["sampling_plan"]["points"]}
    predictions = mapping(predictions, "predictions")
    if len(predictions) > MAX_POINTS:
        raise Refusal("oversized_input", f"predictions exceed {MAX_POINTS} points")
    result = {}
    for point_id, item in predictions.items():
        if point_id not in known:
            raise Refusal("unknown_point", f"Prediction names undeclared point {point_id!r}")
        item = require_keys(item, f"predictions[{point_id}]", {"value", "standard_uncertainty"}, {"dof"})
        dof = item.get("dof")
        result[point_id] = {
            "value": _quantity(item["value"], f"predictions[{point_id}].value", unit),
            "standard_uncertainty": _quantity(item["standard_uncertainty"],
                                              f"predictions[{point_id}].standard_uncertainty", unit, nonnegative=True),
            "dof": None if dof is None else integer(dof, f"predictions[{point_id}].dof", minimum=1)}
    return result


def _outside(value: dict, limits: dict, unit: str) -> bool:
    return not _in(limits["min"], unit) <= _in(value, unit) <= _in(limits["max"], unit)


def evaluate(protocol: Any, measurements: Any, predictions: Any) -> dict:
    """Decide each sampling point as accepted, rejected or excluded, with explicit reasons.

    Exclusions (checked before any acceptance rule): datum not established,
    incomplete repeats, calibration not valid at an acquisition time, environment
    unrecorded or outside limits, repeatability above the declared maximum,
    missing prediction or zero combined uncertainty. If any measurement is
    synthetic the result carries ``physical_evidence: false`` and accepted points
    become ``accepted_synthetic_only``.
    """
    protocol = validate_protocol(protocol)
    runs, expected = _measurements(protocol, measurements), _predictions(protocol, predictions)
    unit, level = protocol["instrument"]["unit"], protocol["coverage"]["level"]
    plan, exclusion, limits = protocol["sampling_plan"], protocol["exclusion"], protocol["environment_limits"]
    calibration = protocol["instrument"]["calibration"]
    start, end = iso_time(calibration["valid_from"], "valid_from"), iso_time(calibration["valid_until"], "valid_until")
    synthetic = any(run["acquisition"] == "synthetic" for run in runs)
    max_std = _span(exclusion["max_repeatability_std"], unit)
    rule = protocol["acceptance"]
    grouped: dict[str, list[dict]] = {point["point_id"]: [] for point in plan["points"]}
    for run in runs:
        grouped[run["point_id"]].append(run)
    points = []
    for point in plan["points"]:
        point_id = point["point_id"]
        items = sorted(grouped[point_id], key=lambda run: run["repeat"])
        values = np.array([_in(run["value"], unit) for run in items], dtype=float)
        reasons, checks = [], {}
        checks["datum_established"] = protocol["datum"]["established"]
        if exclusion["require_datum"] and not checks["datum_established"]:
            reasons.append(_reason("datum_not_established", f"Datum {protocol['datum']['frame_id']!r} is not established"))
        checks["complete"] = len(items) == plan["repeats"]
        if not checks["complete"]:
            reasons.append(_reason("incomplete_sampling", f"{len(items)} of {plan['repeats']} planned repeats acquired",
                                   acquired=len(items), planned=plan["repeats"]))
        early = [run["repeat"] for run in items if iso_time(run["acquired_at"], "acquired_at") < start]
        late = [run["repeat"] for run in items if iso_time(run["acquired_at"], "acquired_at") >= end]
        checks["calibration_valid"] = not early and not late
        if exclusion["require_valid_calibration"]:
            if early:
                reasons.append(_reason("calibration_not_yet_valid", f"Calibration {calibration['calibration_id']!r} "
                                       f"is valid from {calibration['valid_from']}", repeats=early))
            if late:
                reasons.append(_reason("calibration_expired", f"Calibration {calibration['calibration_id']!r} "
                                       f"expired at {calibration['valid_until']}", repeats=late))
        unrecorded = [run["repeat"] for run in items if any(key not in run["environment"] for key in limits)]
        outside = [run["repeat"] for run in items if any(key in run["environment"] and _outside(
            run["environment"][key], limits[key], "K" if key == "temperature" else "percent") for key in limits)]
        checks["environment_within_limits"] = not unrecorded and not outside
        if unrecorded:
            reasons.append(_reason("environment_unrecorded", f"Environment {sorted(limits)} not recorded",
                                   repeats=unrecorded))
        if outside:
            reasons.append(_reason("environment_out_of_limits", "Environment outside the declared limits",
                                   repeats=outside))
        record: dict[str, Any] = {"point_id": point_id, "n": len(items)}
        if len(values) >= 1:
            record["mean"] = {"value": float(values.mean()), "unit": unit}
        if len(values) >= 2:
            std = float(values.std(ddof=1))
            record["std"] = {"value": std, "unit": unit}
            checks["repeatability_within_limit"] = std <= max_std
            if std > max_std:
                reasons.append(_reason("repeatability_exceeded", f"Sample std {std:.6g} {unit} exceeds "
                                       f"{max_std:.6g} {unit}", std=std, limit=max_std))
        prediction = expected.get(point_id)
        if prediction is None:
            reasons.append(_reason("prediction_missing", "No prediction for this point"))
        if len(values) >= 2 and prediction is not None:
            type_a = {"name": f"repeatability:{point_id}", "type": "A", "distribution": "normal",
                      "value": {"value": record["std"]["value"] / math.sqrt(len(values)), "unit": unit},
                      "sensitivity": 1.0, "dof": len(values) - 1}
            budget = _budget(protocol, [type_a])
            measured_u = budget["expanded_uncertainty"]["value"]
            predicted_k = coverage_factor(prediction["dof"], level)
            predicted_u = predicted_k * _span(prediction["standard_uncertainty"], unit)
            residual = record["mean"]["value"] - _in(prediction["value"], unit)
            record.update({"prediction": prediction["value"], "residual": {"value": residual, "unit": unit},
                           "budget": budget, "expanded_uncertainty": budget["expanded_uncertainty"],
                           "prediction_coverage_factor": predicted_k,
                           "prediction_expanded_uncertainty": {"value": predicted_u, "unit": unit}})
            if budget["combined_standard_uncertainty"]["value"] == 0.0:
                reasons.append(_reason("zero_combined_uncertainty", "The combined standard uncertainty is zero"))
            else:
                record["en"] = abs(residual) / math.hypot(measured_u, predicted_u)
        if reasons:
            decision = "excluded"
        else:
            decision, reason = _decide(rule, record, unit)
            reasons.append(reason)
            if synthetic and decision == "accepted":
                decision = "accepted_synthetic_only"
                reasons.append(_reason("synthetic_only", "Synthetic data never produces a physical acceptance"))
        record.update({"decision": decision, "reasons": reasons, "checks": checks})
        points.append(record)
    counts = {name: sum(point["decision"] == name for point in points)
              for name in ("accepted", "accepted_synthetic_only", "rejected", "excluded")}
    if counts["rejected"]:
        overall = "rejected"
    elif counts["excluded"]:
        overall = "inconclusive"
    else:
        overall = "accepted_synthetic_only" if synthetic else "accepted"
    return {"schema": EVALUATION_SCHEMA, "protocol_id": protocol["protocol_id"],
            "protocol_identity": content_identity(protocol), "observable": protocol["instrument"]["observable"],
            "measurand_unit": unit, "acceptance_rule": rule["rule"], "coverage_level": level,
            "clock": runs[0]["clock"], "acquisitions": sorted({run["acquisition"] for run in runs}),
            "physical_evidence": not synthetic, "claim_class": "computed" if synthetic else "measured",
            "points": points, "summary": {"points": len(points), **counts, "overall": overall}}


def _decide(rule: dict, record: dict, unit: str) -> tuple[str, dict]:
    residual = abs(record["residual"]["value"])
    if rule["rule"] == "en_le_1":
        ok = record["en"] <= 1.0
        return ("accepted" if ok else "rejected"), _reason(
            "en_within_1" if ok else "en_exceeds_1", f"En = {record['en']:.4g} {'<=' if ok else '>'} 1", en=record["en"])
    if rule["rule"] == "abs_residual_le_U":
        bound = record["expanded_uncertainty"]["value"]
        code = "residual_within_U" if residual <= bound else "residual_exceeds_U"
    else:
        bound = _span(rule["limit"], unit)
        code = "residual_within_limit" if residual <= bound else "residual_exceeds_limit"
    ok = residual <= bound
    return ("accepted" if ok else "rejected"), _reason(
        code, f"|residual| = {residual:.6g} {unit} {'<=' if ok else '>'} {bound:.6g} {unit}", abs_residual=residual,
        bound=bound)


# ---------------------------------------------------------------------- ledger
def retain_protocol(ledger: Any, record: Any) -> dict:
    """Append the normalized protocol and its identity as a ``physical_protocol`` entry."""
    protocol = validate_protocol(record)
    return ledger.append(PROTOCOL_BODY, {"protocol": protocol, "protocol_identity": content_identity(protocol)})


def retain_measurements(ledger: Any, protocol_entry_id: str, measurements: Any) -> list[dict]:
    """Append each measurement as an ``observation`` entry referencing the protocol entry.

    The acquisition kind is preserved exactly; retention records what was
    acquired and never decides acceptance.
    """
    entry = ledger.get(protocol_entry_id)
    if entry["kind"] != "physical_protocol":
        raise Refusal("wrong_entry_kind", f"{protocol_entry_id} is a {entry['kind']!r} entry, not a physical protocol")
    protocol = validate_protocol(entry["body"]["protocol"])
    identity = content_identity(protocol)
    if identity != entry["body"]["protocol_identity"]:
        raise Refusal("protocol_identity_mismatch", "The retained protocol does not match its recorded identity")
    instrument = protocol["instrument"]
    entries = []
    for run in _measurements(protocol, measurements):
        body = {"observable": instrument["observable"], "acquisition": run["acquisition"],
                "value": run["value"]["value"], "unit": run["value"]["unit"], "point_id": run["point_id"],
                "repeat": run["repeat"], "acquired_at": run["acquired_at"], "clock": run["clock"],
                "environment": run["environment"], "protocol_id": protocol["protocol_id"],
                "protocol_identity": identity, "instrument_id": instrument["instrument_id"],
                "calibration": {"calibration_id": instrument["calibration"]["calibration_id"],
                                "version": instrument["calibration"]["version"]}}
        entries.append(ledger.append(OBSERVATION_BODY, body, refs=[protocol_entry_id]))
    return entries
