"""Compile an industrial use case into explicit, reviewable measurement requirements.

Scope: a ``ciw.use-case.v1`` record names a domain, a part geometry with unit
dimensions, a feature tolerance and optionally throughput, environment, process
parameters and the sensors already installed. ``compile_use_case`` applies
fixed per-domain rule tables and returns geometry, sensor, calibration, path,
uncertainty-budget, acceptance-test, evidence and capability requirements, each
with a short rationale, plus operational limitations. The same input always
compiles to byte-identical output.

Honest limits: the rules are conventional engineering heuristics (10:1
resolution, 4:1 test-uncertainty ratio, RSS allocation with fixed weights), not
a substitute for a metrologist's budget or a process qualification. Derived
numbers (chord/intrinsic divergence span, mesh edge bound, steering and slip
limits) use closed forms for circles, spheres and helices and the declared
minimum radius as a worst case for freeform shapes. Reachability, collision,
cycle-time feasibility, materials and thermal models are not evaluated.
Compiled requirements never authorize actuation.
"""
from __future__ import annotations

import math
from typing import Any

from ._common import Refusal, canonical_json, content_identity, finite, mapping, require_keys, text
from .physical import OBSERVABLE_UNITS, _choice, _items, _limits, _quantity
from .units import conversion_factor, parse_unit
from .vocabulary import CAPABILITIES, OBSERVABLES

SCHEMA = "ciw.use-case.v1"
REQUIREMENTS_SCHEMA = "ciw.use-case-requirements.v1"
BODY = "ciw.science.use-case-requirements.v1"
RESOLUTION_RATIO, TEST_UNCERTAINTY_RATIO = 10.0, 4.0
GEOMETRIES = {"plate": set(), "cylinder": {"radius"}, "sphere": {"radius"}, "freeform": {"min_radius"},
              "building": set(), "mould": {"min_radius"}, "axis": set()}
CURVED = frozenset({"cylinder", "sphere", "freeform", "mould"})
PATH_DOMAINS = frozenset({"fibre-placement", "filament-winding"})
MAX_SENSORS = 32

# Per domain: admissible geometries, required observables, calibrated transforms (source, target),
# RSS contributors with integer variance weights, path constraints, extra acceptance tests, limitations.
DOMAINS: dict[str, dict] = {
    "robotic-inspection": {
        "geometries": ("plate", "cylinder", "sphere", "freeform", "mould"),
        "observables": (("camera_chord", "surface points reconstructed by the robot-carried camera or scanner"),
                        ("encoder_displacement", "joint encoders place every view in the machine frame")),
        "frames": (("tool", "machine", "flange/TCP pose from kinematic calibration carries each view"),
                   ("camera", "tool", "hand-eye calibration; its error biases every reconstructed point"),
                   ("part_datum", "machine", "tolerances are defined in the part datum, so the part is registered")),
        "contributors": (("sensor", 5, "camera or scanner noise and calibration"),
                         ("robot_pose", 3, "robot positioning and kinematic calibration"),
                         ("part_registration", 2, "datum fitting and fixture repeatability")),
        "paths": (("standoff_and_incidence", "each view stays inside the sensor's calibrated working volume and "
                   "incidence range; outside it the calibration does not apply"),
                  ("view_redundancy", "each toleranced feature is seen from at least two poses so pose error is "
                   "observable")),
        "tests": (),
        "limitations": ("Robot reachability and collision are not checked by this compiler.",),
    },
    "fibre-placement": {
        "geometries": ("plate", "cylinder", "sphere", "freeform", "mould"),
        "observables": (("encoder_displacement", "head position along the programmed tow path"),
                        ("camera_chord", "tow edge positions for gap and overlap inspection")),
        "frames": (("tool", "machine", "compaction roller TCP on the placement head"),
                   ("part_datum", "machine", "layup tool registration; paths are defined on the tool surface"),
                   ("camera", "tool", "inspection camera mounted on the head")),
        "contributors": (("path_following", 4, "head path following and machine kinematics"),
                         ("tool_registration", 3, "layup tool surface registration"),
                         ("inspection_sensor", 3, "gap/overlap inspection sensor")),
        "paths": (("tow_gap_overlap", "gaps and overlaps between courses are measured along the surface, not as "
                   "chords"),),
        "tests": (),
        "limitations": ("The steering limit depends on tow width, material and compaction and must be established "
                        "by trials; the compiler only applies the declared value.",
                        "Wrinkling and puckering on doubly curved surfaces are not predicted."),
    },
    "filament-winding": {
        "geometries": ("cylinder", "sphere", "freeform"),
        "observables": (("encoder_displacement", "mandrel rotation and carriage position define the winding angle"),
                        ("reconstructed_geometry", "laid band position and winding angle on the mandrel")),
        "frames": (("part_datum", "machine", "mandrel axis alignment to the spindle"),
                   ("tool", "machine", "payout eye position relative to the carriage axis")),
        "contributors": (("winding_kinematics", 4, "mandrel and carriage axis coordination"),
                         ("mandrel_alignment", 3, "mandrel run-out and axis alignment"),
                         ("inspection_sensor", 3, "band position inspection")),
        "paths": (("pattern_closure", "the winding pattern must close after an integral number of circuits; "
                   "circuits are counted as winding classes"),),
        "tests": (),
        "limitations": ("Fibre-mandrel friction depends on resin, fibre and tension; slip limits need trials.",),
    },
    "coating": {
        "geometries": ("plate", "cylinder", "sphere", "freeform", "mould"),
        "observables": (("encoder_displacement", "spray gun path from the robot encoders"),
                        ("reconstructed_geometry", "coating thickness from before/after surface reconstruction")),
        "frames": (("tool", "machine", "spray gun TCP"),
                   ("part_datum", "machine", "part registration for the programmed path"),
                   ("sensor", "machine", "thickness scanner pose for before/after scans")),
        "contributors": (("thickness_sensor", 5, "scanner noise on both scans"),
                         ("before_after_registration", 3, "registration between the two scans"),
                         ("gun_path", 2, "standoff and path deviation")),
        "paths": (("standoff_and_incidence", "constant standoff and near-normal incidence keep deposition "
                   "within the calibrated spray model"),
                  ("pass_spacing", "pass spacing is set along the surface; on curved parts chords understate it")),
        "tests": (("witness-coupon thickness against a reference gauge", "coat witness coupons with the production "
                   "path and measure them with a calibrated reference gauge",
                   "scan-derived thickness agrees with the gauge within the allowed expanded uncertainty",
                   "anchors the before/after difference to a direct thickness measurement"),),
        "limitations": ("Thickness from before/after scans inherits both scans' registration error.",
                        "Cure shrinkage changes thickness after measurement."),
    },
    "welding": {
        "geometries": ("plate", "cylinder", "freeform"),
        "observables": (("camera_chord", "seam position from the laser-line seam sensor"),
                        ("encoder_displacement", "torch pose from the robot encoders")),
        "frames": (("tool", "machine", "torch TCP"),
                   ("camera", "tool", "seam sensor to torch offset; it sets the lateral tracking error"),
                   ("part_datum", "machine", "part registration for the nominal seam")),
        "contributors": (("seam_sensor", 4, "seam sensor noise and calibration"),
                         ("torch_pose", 3, "robot positioning of the torch"),
                         ("thermal_distortion", 3, "part movement during welding")),
        "paths": (("seam_following", "the torch follows the sensed seam, not the nominal CAD seam"),),
        "tests": (("seam-tracking accuracy on a reference seam", "track a machined reference seam with the arc off "
                   "and on", "lateral tracking error within the allowed expanded uncertainty",
                   "separates sensor/robot error from arc-induced disturbance"),),
        "limitations": ("Thermal distortion invalidates pre-weld registration; re-measure after cooling.",
                        "Arc light and spatter degrade optical seam sensing near the arc."),
    },
    "bim-construction-state": {
        "geometries": ("building",),
        "observables": (("reconstructed_geometry", "as-built surfaces from laser scanning or photogrammetry"),
                        ("tracker_position", "total-station coordinates of site control points")),
        "frames": (("sensor", "site_control", "each scan station is registered to the site control network"),
                   ("bim_model", "site_control", "georeference of the pinned BIM model version"),
                   ("tracker", "site_control", "total station set-up on control")),
        "contributors": (("scan_noise", 3, "scanner range noise"),
                         ("registration_to_control", 5, "station registration to the control network"),
                         ("model_georeference", 2, "BIM model placement on site")),
        "paths": (("scan_station_network", "overlapping stations with control targets visible from each "
                   "station"),),
        "tests": (("control-point registration check", "measure independent check points not used in "
                   "registration", "check-point residuals within the allowed expanded uncertainty",
                   "registration residuals on the fitted targets understate the true error"),),
        "limitations": ("Deviations are relative to one pinned BIM model version; a revision changes every "
                        "deviation.", "Occluded elements are unobserved, not conforming."),
    },
    "injection-moulding": {
        "geometries": ("mould", "freeform", "plate"),
        "observables": (("reconstructed_geometry", "part surface from CT, CMM or optical scan"),),
        "frames": (("sensor", "fixture", "scanner or CMM registered to the measuring fixture"),
                   ("part_datum", "fixture", "part located on its functional datums, not the parting line")),
        "contributors": (("scanner", 5, "measuring instrument"),
                         ("fixture_and_datum", 3, "fixturing and datum fitting"),
                         ("conditioning", 2, "residual thermal and moisture state of the part")),
        "paths": (("conditioning_before_measurement", "parts are measured only after the declared conditioning "
                   "time and state"),),
        "tests": (("post-conditioning dimensional stability", "measure the same parts after conditioning and again "
                   "after a further interval", "change between the two within the allowed expanded uncertainty",
                   "shrinkage and warpage continue after ejection"),),
        "limitations": ("Results hold only for the declared conditioning time and state.",
                        "One cavity does not represent a multi-cavity mould."),
    },
    "servo-stability": {
        "geometries": ("axis",),
        "observables": (("encoder_displacement", "axis position feedback"),),
        "frames": (("sensor", "machine", "encoder scale and mounting against a reference such as a laser "
                    "interferometer"),),
        "contributors": (("encoder", 4, "encoder resolution and scale error"),
                         ("identified_plant_model", 4, "identification error of the plant model"),
                         ("disturbance", 2, "load and friction disturbance")),
        "paths": (("certified_region", "commanded trajectories and gains stay inside the region and parameter "
                   "range covered by the stability certificate"),),
        "tests": (),
        "limitations": ("A stability certificate holds only for the identified plant model and parameter range.",
                        "Margins from linear identification do not cover saturation or backlash."),
    },
}


def _geometry_capabilities(geometry: str) -> list[tuple[tuple[str, ...], str]]:
    return {
        "cylinder": [(("constant_curvature", "arbitrary_metric"), "a cylinder has constant (zero) Gaussian curvature")],
        "sphere": [(("constant_curvature", "arbitrary_metric"), "a sphere has constant positive Gaussian curvature")],
        "freeform": [(("mesh_surfaces",), "a freeform surface has no closed form and is represented as a mesh")],
        "mould": [(("mesh_surfaces",), "mould cavities are represented as meshes")],
        "building": [(("mesh_surfaces",), "BIM element geometry is compared as meshes")],
    }.get(geometry, [])


# ---------------------------------------------------------------------- validation
def validate_use_case(record: Any) -> dict:
    """Validate a ``ciw.use-case.v1`` record and return its normalized form (idempotent)."""
    record = require_keys(record, "use_case", {"schema", "use_case_id", "domain", "part", "feature_tolerance"},
                          {"throughput_per_hour", "environment", "existing_sensors", "process", "description"})
    if record["schema"] != SCHEMA:
        raise Refusal("unsupported_schema", f"Expected {SCHEMA}")
    domain = _choice(record["domain"], DOMAINS, "domain", "unknown_domain")
    part = require_keys(record["part"], "part", {"geometry", "dimensions"}, {"material"})
    geometry = _choice(part["geometry"], GEOMETRIES, "part.geometry", "unknown_geometry")
    if geometry not in DOMAINS[domain]["geometries"]:
        raise Refusal("geometry_not_applicable", f"{domain} does not apply to {geometry!r} parts",
                      allowed=list(DOMAINS[domain]["geometries"]))
    lengths = ("m", "rad") if geometry == "axis" else ("m",)
    dimensions = mapping(part["dimensions"], "part.dimensions")
    if len(dimensions) > 16:
        raise Refusal("oversized_input", "part.dimensions exceeds 16 entries")
    missing = GEOMETRIES[geometry] - dimensions.keys()
    if missing:
        raise Refusal("malformed_record", f"A {geometry} part must declare {sorted(missing)}")
    normalized_part = {"geometry": geometry, "dimensions": {
        text(name, "dimension name", 64): _quantity(value, f"part.dimensions.{name}", lengths, positive=True)
        for name, value in sorted(dimensions.items())}}
    if "material" in part:
        normalized_part["material"] = text(part["material"], "part.material", 256)
    result = {"schema": SCHEMA, "use_case_id": text(record["use_case_id"], "use_case_id", 256), "domain": domain,
              "part": normalized_part,
              "feature_tolerance": _quantity(record["feature_tolerance"], "feature_tolerance", lengths, positive=True)}
    if "description" in record:
        result["description"] = text(record["description"], "description", 4096)
    if "throughput_per_hour" in record:
        result["throughput_per_hour"] = finite(record["throughput_per_hour"], "throughput_per_hour", minimum=0.0,
                                               maximum=1e6, exclusive_minimum=True)
    if "environment" in record:
        environment = require_keys(record["environment"], "environment", set(), {"temperature", "humidity"})
        result["environment"] = {key: _limits(value, f"environment.{key}", "K" if key == "temperature" else "percent")
                                 for key, value in sorted(environment.items())}
    if "process" in record:
        if domain not in PATH_DOMAINS:
            raise Refusal("malformed_record", "process parameters apply only to fibre-placement and filament-winding")
        process = require_keys(record["process"], "process", set(), {"min_steering_radius", "friction_coefficient"})
        result["process"] = {}
        if "min_steering_radius" in process:
            result["process"]["min_steering_radius"] = _quantity(process["min_steering_radius"],
                                                                 "process.min_steering_radius", "m", positive=True)
        if "friction_coefficient" in process:
            result["process"]["friction_coefficient"] = finite(process["friction_coefficient"],
                                                               "process.friction_coefficient", minimum=0.0,
                                                               maximum=2.0, exclusive_minimum=True)
    if "existing_sensors" in record:
        sensors = []
        for index, sensor in enumerate(_items(record["existing_sensors"], "existing_sensors", MAX_SENSORS, 0)):
            name = f"existing_sensors[{index}]"
            sensor = require_keys(sensor, name, {"kind", "observable", "resolution"}, {"sensor_id"})
            observable = _choice(sensor["observable"], OBSERVABLES, f"{name}.observable", "unknown_observable")
            if observable not in OBSERVABLE_UNITS:
                raise Refusal("not_a_raw_observable", f"{name}: {observable!r} is an estimator output, not a sensor")
            item = {"kind": text(sensor["kind"], f"{name}.kind", 128), "observable": observable,
                    "resolution": _quantity(sensor["resolution"], f"{name}.resolution", OBSERVABLE_UNITS[observable],
                                            positive=True)}
            if "sensor_id" in sensor:
                item["sensor_id"] = text(sensor["sensor_id"], f"{name}.sensor_id", 128)
            sensors.append(item)
        result["existing_sensors"] = sensors
    return result


def _capabilities(value: Any) -> frozenset[str] | None:
    if value is None:
        return None
    if not isinstance(value, (set, frozenset, list, tuple)) or len(value) > len(CAPABILITIES):
        raise Refusal("malformed_record", "available_capabilities must be a set of capability names")
    unknown = {item for item in value if not isinstance(item, str) or item not in CAPABILITIES}
    if unknown:
        raise Refusal("unknown_capability", "Undeclared capabilities", unknown=sorted(map(str, unknown)),
                      allowed=sorted(CAPABILITIES))
    return frozenset(value)


# ---------------------------------------------------------------------- geometry
def _q(value: float, unit: str) -> dict:
    return {"value": float(value), "unit": unit}


def _inverse(unit: str) -> str:
    return f"1/{unit}" if unit.isalnum() else f"1/({unit})"


def _divergence_span(radius: float, tolerance: float) -> float | None:
    """Arc length s at which s - 2R sin(s/2R) equals the tolerance (None if never within a half turn)."""
    gap = lambda s: s - 2.0 * radius * math.sin(s / (2.0 * radius)) - tolerance  # noqa: E731
    low, high = 0.0, math.pi * radius
    if gap(high) < 0:
        return None
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if gap(middle) < 0 else (low, middle)
    return 0.5 * (low + high)


def _geometry_requirements(case: dict, unit: str, tolerance: float) -> list[dict]:
    geometry, dims = case["part"]["geometry"], case["part"]["dimensions"]
    radius_key = "radius" if geometry in {"cylinder", "sphere"} else "min_radius"
    items = []
    if geometry == "plate":
        items.append({"name": "flatness_assumption", "rationale": "chord equals intrinsic distance only while the "
                      "plate is flat; a warped plate is a curved surface and its flatness must be verified"})
    if geometry in CURVED:
        radius = _q(dims[radius_key]["value"] * conversion_factor(dims[radius_key]["unit"], unit), unit)
        items.append({"name": "tolerance_semantics", "rationale": "on a curved part the tolerance must declare "
                      "whether it is intrinsic (along the surface, geodesic) or a chord (straight line)"})
        span = _divergence_span(radius["value"], tolerance)
        items.append({"name": "chord_intrinsic_divergence", "radius": radius,
                      "critical_arc_span": None if span is None else _q(span, unit),
                      "rationale": "arc s and chord 2R sin(s/2R) differ by ~s^3/(24R^2); beyond this span along a "
                      "direction of curvature radius R the difference exceeds the tolerance"
                      + ("" if geometry in {"cylinder", "sphere"} else "; min_radius is used as the worst case")})
        if geometry == "cylinder":
            items.append({"name": "intrinsically_flat", "rationale": "Gaussian curvature is zero: unrolling preserves "
                          "intrinsic distance s^2 = (R dtheta)^2 + dz^2, and helices are geodesics; axial chords "
                          "equal intrinsic distances, circumferential ones do not"})
        elif geometry == "sphere":
            items.append({"name": "conjugate_points", "antipodal_distance": _q(math.pi * radius["value"], unit),
                          "rationale": "geodesics are great circles; beyond the antipode (pi R) a geodesic is no "
                          "longer the shortest path, so paths must stay below it or be flagged"})
        else:
            error = tolerance / RESOLUTION_RATIO
            items.append({"name": "mesh_resolution", "allowed_mesh_error": _q(error, unit),
                          "max_edge_length": _q(math.sqrt(8.0 * radius["value"] * error), unit),
                          "rationale": "chordal sagitta of an edge h on radius R is h^2/(8R); keeping it below "
                          "tolerance/10 bounds the mesh discretization error"})
    if geometry == "building":
        items.append({"name": "site_frame", "rationale": "as-built geometry is compared with one pinned BIM model "
                      "version in the site control frame"})
    if geometry == "axis":
        items.append({"name": "one_dimensional_axis", "rationale": "the measurand is axis position; stroke, scale "
                      "and encoder resolution are declared in the axis frame"})
    return items


# ---------------------------------------------------------------------- compile
def compile_use_case(record: Any, available_capabilities: set[str] | None = None) -> dict:
    """Deterministically compile a use case into requirements (see module docstring)."""
    case = validate_use_case(record)
    available = _capabilities(available_capabilities)
    domain, geometry = case["domain"], case["part"]["geometry"]
    rules = DOMAINS[domain]
    tolerance_q = case["feature_tolerance"]
    unit, tolerance = tolerance_q["unit"], tolerance_q["value"]
    tolerance_dimension = parse_unit(unit).dimension
    curved = geometry in CURVED
    observables = list(rules["observables"])
    if curved and domain in {"robotic-inspection", "fibre-placement", "coating"}:
        observables.append(("intrinsic_distance", "a tolerance along a curved surface is intrinsic; derive it from "
                            "reconstructed geometry or contact traces, never from chords"))
    sensors = case.get("existing_sensors", [])
    fusion = len(observables) > 1 or len(sensors) > 1
    max_resolution, allowed_u = tolerance / RESOLUTION_RATIO, tolerance / TEST_UNCERTAINTY_RATIO

    def commensurable(observable: str) -> bool:
        return any(parse_unit(option).dimension == tolerance_dimension for option in OBSERVABLE_UNITS[observable])

    sensor_items = []
    for observable, purpose in observables:
        item = {"observable": observable, "purpose": purpose}
        if commensurable(observable):
            item.update({"max_resolution": _q(max_resolution, unit), "rationale": "10:1 rule: resolution at most "
                         "a tenth of the tolerance"})
        else:
            item.update({"max_resolution": None, "rationale": "not commensurable with the tolerance; the 10:1 rule "
                         "applies only after a calibrated model maps it into tolerance units"})
        sensor_items.append(item)
    assessments, covered = [], set()
    required = {observable for observable, _ in observables}
    for sensor in sensors:
        assessment = {key: sensor[key] for key in ("sensor_id", "kind", "observable", "resolution") if key in sensor}
        if parse_unit(sensor["resolution"]["unit"]).dimension != tolerance_dimension:
            assessment.update({"verdict": "not_checkable", "rationale": "resolution unit is not commensurable with "
                               "the tolerance without a calibrated model"})
        else:
            resolution = sensor["resolution"]["value"] * conversion_factor(sensor["resolution"]["unit"], unit)
            meets = resolution <= max_resolution * (1.0 + 1e-12)
            assessment.update({"resolution_in_tolerance_unit": _q(resolution, unit),
                               "verdict": "meets_10_to_1" if meets else "fails_10_to_1",
                               "rationale": f"resolution {resolution:.6g} {unit} vs required <= "
                                            f"{max_resolution:.6g} {unit}"})
            if meets:
                covered.add(sensor["observable"])
        assessment["required_observable"] = sensor["observable"] in required
        assessments.append(assessment)
    uncovered = [item["observable"] for item in sensor_items
                 if item["max_resolution"] is not None and item["observable"] not in covered]
    sensor_requirements = {
        "tolerance": tolerance_q,
        "max_resolution": {**_q(max_resolution, unit), "rationale": "10:1 rule: resolution <= tolerance/10"},
        "max_expanded_uncertainty": {**_q(allowed_u, unit), "rationale": "test-uncertainty ratio 4:1: expanded "
                                     "uncertainty (k~2) <= tolerance/4"},
        "observables": sensor_items, "existing_sensors": assessments, "uncovered_observables": uncovered,
    }

    calibration = [{"kind": "transform", "source": source, "target": target, "rationale": why}
                   for source, target, why in rules["frames"]]
    if fusion:
        calibration.append({"kind": "clock_alignment", "source": "each sensor clock", "target": "machine clock",
                            "rationale": "multi-sensor data are fused only after timestamps are mapped to one clock "
                            "with a stated uncertainty"})
    if "environment" in case:
        calibration.append({"kind": "reference_temperature", "reference": _q(20.0, "degC"),
                            "rationale": "dimensional results refer to 20 degC (ISO 1); measurement temperatures "
                            "must be recorded and compensated within the declared environment"})

    paths = [{"name": name, "rationale": why} for name, why in rules["paths"]]
    paths.extend(_path_rules(case, unit))
    if "throughput_per_hour" in case:
        paths.append({"name": "cycle_time", "max_cycle_time": _q(3600.0 / case["throughput_per_hour"], "s"),
                      "rationale": "throughput bounds the time per part available for measurement and motion"})

    contributors = list(rules["contributors"]) + ([("clock_alignment", 1, "residual timestamp misalignment "
                                                    "between fused streams")] if fusion else [])
    total = sum(weight for _, weight, _ in contributors)
    allocations = [{"contributor": name, "weight": weight, "variance_share": weight / total,
                    "allowed_expanded_uncertainty": _q(allowed_u * math.sqrt(weight / total), unit), "rationale": why}
                   for name, weight, why in contributors]
    budget = {"allowed_expanded_uncertainty": _q(allowed_u, unit), "coverage": "k~2 (about 95 %)",
              "method": "RSS split: U_i = U * sqrt(w_i / sum w), so sqrt(sum U_i^2) = U",
              "allocations": allocations,
              "rss_check": _q(math.sqrt(sum(item["allowed_expanded_uncertainty"]["value"] ** 2
                                            for item in allocations)), unit),
              "rationale": "the allowed expanded uncertainty is tolerance/4 and is shared by declared contributors"}

    tests = [{"name": "test-uncertainty ratio on a reference artefact", "procedure": "measure a calibrated reference "
              "artefact with the full measurement chain under the physical protocol",
              "pass_criterion": "En <= 1 against the reference and expanded U <= tolerance/4",
              "rationale": "demonstrates the 4:1 ratio with the installed chain rather than datasheets"}]
    if domain == "robotic-inspection" and curved:
        tests.append({"name": f"{geometry} chord-vs-intrinsic discrimination", "procedure": "measure marked pairs "
                      "with spans below and above the critical arc span as chords and as intrinsic distances",
                      "pass_criterion": "the chain resolves the predicted chord/intrinsic difference with En <= 1 "
                      "at each pair", "rationale": "proves the system reports the declared observable, not the other"})
    if domain in PATH_DOMAINS:
        tests.append({"name": "geodesic oracle checks", "procedure": "compare planned paths with closed-form "
                      "geodesics (cylinder helices, sphere great circles) and replay the path computation",
                      "pass_criterion": "path deviation and geodesic curvature within the declared limits",
                      "rationale": "path planning is verified against references with known answers first"})
    if domain == "servo-stability":
        tests.append({"name": "servo stability margin/Lyapunov certificate", "procedure": "identify the axis "
                      "plant, compute gain/phase margins and a Lyapunov certificate over the parameter range, then "
                      "confirm with bounded step and frequency-response tests",
                      "pass_criterion": "certificate verifies over the declared range and measured margins agree "
                      "with the model within uncertainty",
                      "rationale": "stability is a verified property of a model, confirmed but not proven by tests"})
    if fusion:
        tests.append({"name": "clock-aligned multi-sensor fusion consistency (NIS)", "procedure": "fuse clock-aligned "
                      "streams and accumulate normalized innovation squared over a reference trajectory",
                      "pass_criterion": "average NIS inside the chi-square acceptance interval for its dof",
                      "rationale": "an inconsistent filter under- or over-states its uncertainty"})
    tests.extend({"name": name, "procedure": procedure, "pass_criterion": criterion, "rationale": why}
                 for name, procedure, criterion, why in rules["tests"])

    evidence = [
        ("physical_protocol", None, "the experiment is pre-registered before data are acquired"),
        ("calibration", "measured", "every calibrated transform and instrument named above"),
        ("frame_registry", "measured", "transforms are composed from one registry with validity intervals"),
        ("observation", "measured", "physically acquired observations; synthetic data cannot stand in"),
        ("verification", "verified", "each acceptance test above has a passed verification"),
    ]
    if domain in PATH_DOMAINS or domain == "servo-stability":
        evidence += [("numerical_result", "computed", "planned paths or stability certificates are computed results"),
                     ("replay_receipt", "verified", "computed results reproduce on replay")]
    if fusion:
        evidence.append(("fusion_state", "estimated", "fused state with its consistency statistics"))
    evidence.append(("authority_decision", "authorized", "any actuation or release needs a separate authority "
                     "decision"))
    required_evidence = [{"kind": kind, "claim_class": claim, "rationale": why} for kind, claim, why in evidence]

    limitations = ["Compiled requirements do not authorize actuation; they state what must be evidenced first."]
    if curved and "camera_chord" in required:
        limitations.append("A camera reports chords; on curved parts it cannot confirm an intrinsic tolerance "
                            "without the surface model.")
    limitations.extend(rules["limitations"])
    failing = [item.get("sensor_id", item["kind"]) for item in assessments if item["verdict"] == "fails_10_to_1"]
    if failing:
        limitations.append(f"Existing sensors {failing} fail the 10:1 rule and cannot supply acceptance evidence.")

    groups = [(("uncertainty_propagation",), "budget allocations propagate through calibrated transform chains")]
    groups += _geometry_capabilities(geometry)
    if domain == "robotic-inspection" and curved:
        groups.append((("embedded_surfaces",), "chords are distances in the embedding; comparing them with intrinsic "
                       "distances needs the embedded surface"))
    if domain in PATH_DOMAINS and geometry == "sphere":
        groups.append((("conjugate_detection",), "sphere geodesics stop minimizing at conjugate (antipodal) points"))
    if domain == "filament-winding" and geometry == "cylinder":
        groups.append((("winding_classes",), "helical circuits on a cylinder are distinguished by winding class"))
    if fusion:
        groups += [(("asynchronous_streams",), "sensors sample on their own clocks and rates"),
                   (("clock_alignment",), "streams are mapped to one clock before fusion")]
    capabilities, seen = [], set()
    for any_of, why in groups:
        if any_of not in seen:
            seen.add(any_of)
            capabilities.append({"any_of": list(any_of), "rationale": why})
    missing = None if available is None else [item for item in capabilities if not set(item["any_of"]) & available]

    result = {
        "schema": REQUIREMENTS_SCHEMA, "use_case_id": case["use_case_id"], "use_case_identity": content_identity(case),
        "domain": domain, "geometry": geometry, "tolerance": tolerance_q,
        "fusion": {"used": fusion, "rationale": "more than one sensor or observable contributes, so streams are "
                   "clock-aligned before fusion" if fusion else "a single observable from a single sensor"},
        "geometry_requirements": _geometry_requirements(case, unit, tolerance),
        "sensor_requirements": sensor_requirements, "calibration_requirements": calibration,
        "path_constraints": paths, "uncertainty_budget": budget, "acceptance_tests": tests,
        "required_evidence": required_evidence, "operational_limitations": limitations,
        "required_capabilities": capabilities, "missing_capabilities": missing,
        "available_capabilities": None if available is None else sorted(available),
        "authorizes_actuation": False,
    }
    return result


def _path_rules(case: dict, unit: str) -> list[dict]:
    domain, geometry = case["domain"], case["part"]["geometry"]
    if domain not in PATH_DOMAINS:
        return []
    dims, process = case["part"]["dimensions"], case.get("process", {})
    items = []
    radius = None
    if geometry in {"cylinder", "sphere"}:
        radius = dims["radius"]["value"] * conversion_factor(dims["radius"]["unit"], unit)
    if domain == "fibre-placement":
        item = {"name": "geodesic_or_steering_limit", "rationale": "tows follow geodesics (zero geodesic curvature) "
                "or are steered with geodesic curvature <= 1/R_min; tighter steering buckles the inner tow edge"}
        if "min_steering_radius" in process:
            steering = process["min_steering_radius"]
            item["max_geodesic_curvature"] = _q(1.0 / steering["value"], _inverse(steering["unit"]))
            if geometry == "sphere":
                r_min = steering["value"] * conversion_factor(steering["unit"], unit)
                item["max_latitude_circle"] = _q(math.degrees(math.atan(radius / r_min)), "deg")
                item["rationale"] += "; a latitude circle at phi has k_g = tan(phi)/R, so |phi| <= atan(R/R_min)"
        else:
            item["max_geodesic_curvature"] = None
            item["rationale"] += "; no min_steering_radius declared, so only geodesic paths are admissible"
        items.append(item)
    else:
        item = {"name": "geodesic_or_slip_limit", "rationale": "fibres wound along geodesics do not slip; a "
                "non-geodesic path needs |k_g| <= mu |k_n| (slippage coefficient mu)"}
        if "friction_coefficient" in process:
            item["friction_coefficient"] = process["friction_coefficient"]
            if geometry == "sphere":
                item["max_geodesic_curvature"] = _q(process["friction_coefficient"] / radius, _inverse(unit))
                item["rationale"] += "; on a sphere k_n = 1/R"
            elif geometry == "cylinder":
                item["rationale"] += "; on a cylinder k_n = sin^2(alpha)/R for winding angle alpha to the axis"
        else:
            item["rationale"] += "; no friction coefficient declared, so only geodesic winding is admissible"
        items.append(item)
    if geometry == "plate":
        items.append({"name": "straight_lines_are_geodesics", "rationale": "on a flat plate geodesics are straight "
                      "lines; a steered course has geodesic curvature equal to its in-plane curvature"})
    elif geometry == "cylinder":
        items.append({"name": "helices_are_geodesics", "rationale": "on a cylinder every constant-angle helix is a "
                      "geodesic (it unrolls to a straight line), so constant winding or placement angles need no "
                      "steering or friction"})
    elif geometry == "sphere":
        items.append({"name": "great_circles_and_clairaut", "rationale": "sphere geodesics are great circles; on "
                      "surfaces of revolution geodesics obey Clairaut's relation r sin(alpha) = constant"})
    else:
        items.append({"name": "numerical_geodesics", "rationale": "no closed-form geodesics on a freeform surface; "
                      "paths come from a mesh geodesic solver and are checked against the oracle cases"})
    return items


# ---------------------------------------------------------------------- ledger
def retain_requirements(ledger: Any, record: Any, compiled: Any) -> dict:
    """Append a ``use_case_requirements`` entry after re-compiling and matching ``compiled`` exactly."""
    compiled = mapping(compiled, "compiled requirements")
    if "available_capabilities" not in compiled:
        raise Refusal("malformed_record", "compiled requirements must carry available_capabilities")
    available = compiled["available_capabilities"]
    expected = compile_use_case(record, None if available is None else set(available))
    if canonical_json(expected) != canonical_json(compiled):
        raise Refusal("requirements_mismatch", "compiled requirements do not match a fresh compilation of the record")
    case = validate_use_case(record)
    return ledger.append(BODY, {"use_case": case, "use_case_identity": content_identity(case), "requirements": expected,
                                "requirements_identity": content_identity(expected)})
