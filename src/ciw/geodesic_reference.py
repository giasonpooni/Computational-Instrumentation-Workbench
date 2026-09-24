"""Native flat geometry and declared-curvature reference objects in one workbench.

The providers own geometry, integration and covariance propagation. These
independent workflows retain complete native responses; neither is a measured
state, an embedded physical surface, or a sensor-fusion operation.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import math
import re
import uuid

from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import _json
from .declared_workload import AUTHORITY, DeclaredWorkflow, RESULT_SCHEMA, VERIFY_SCHEMA, _text, _verification
from .exchange import _identity
from .pipelines import provider_pin
from .telemetry import canonical, digest, _keys

KINDS = frozenset({"flat-torus-reference", "curved-path-transfer"})
MAX_SAMPLES = 128
SOURCE_LIMIT = 32 * 1024
FRAME = "transverse-to-gamma, parallel-transported"
# The pipeline descriptors are the pin definition; this module executes them.
PINS = {kind: provider_pin(kind) for kind in sorted(KINDS)}
POLICIES = {
    "flat-torus-reference": {"geometry_basis": "area_one_flat_quotient", "length_unit": "normalized_length",
        "uncertainty": "not_applicable", "physical_geometry": "not_established"},
    "curved-path-transfer": {"geometry_basis": "declared_constant_curvature", "solver": "native_rk4",
        "observation_mode": "intrinsic-surface-distance", "uncertainty": "conditional_on_declared_starting_covariance",
        "physical_geometry": "not_established"},
}


def _number(value, name, limit=None):
    if type(value) not in (int, float):
        raise ValueError(name + " must be a finite JSON number, not a Boolean")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(name + " exceeds the binary64 domain") from exc
    if not math.isfinite(result) or Fraction(value) != Fraction(result):
        raise ValueError(name + " must be exactly representable as finite binary64")
    if limit is not None and abs(result) > limit:
        raise ValueError(name + " exceeds the reference operation bound")
    return result


def _vector(value, size, name, limit=None):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(name + " has the wrong shape")
    return [_number(item, name, limit) for item in value]


def _matrix(value, rows, columns, name):
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(name + " has the wrong shape")
    return [_vector(row, columns, name) for row in value]


def _covariance(value, name):
    result = _matrix(value, 2, 2, name)
    a, b, c, d = (Fraction(x) for row in result for x in row)
    if b != c or a < 0 or d < 0 or a * d < b * b:
        raise ValueError(name + " must be an exactly symmetric positive-semidefinite matrix")
    return result


def _same(actual, expected, message):
    if canonical(actual) != canonical(expected):
        raise ValueError(message)


def _source(kind, raw):
    try:
        if not isinstance(raw, bytes) or not 1 <= len(raw) <= SOURCE_LIMIT:
            raise ValueError("Geodesic reference source must contain 1..32768 exact bytes")
        source = _json(raw)
        canonical(source)
        fields = {"tau", "winding", "start", "samples"} if kind == "flat-torus-reference" else {
            "arclength", "gaussian_curvature", "units", "initial_perturbation", "starting_covariance", "relative_tolerance"}
        _keys(source, {"schema", "experiment_id", "configuration"} | fields)
        if source["schema"] != "ciw." + kind + "-source.v1":
            raise ValueError("Unsupported geodesic reference source")
        _text(source["experiment_id"])
        _same(source["configuration"], POLICIES[kind], "Require the explicit reference operation scope")
        if kind == "flat-torus-reference":
            tau = _vector(source["tau"], 2, "tau", 8)
            if abs(tau[0]) > 4 or not .125 <= tau[1] <= 8:
                raise ValueError("Flat shape requires |Re(tau)|<=4 and .125<=Im(tau)<=8")
            _vector(source["start"], 2, "start", 8)
            winding = source["winding"]
            if (not isinstance(winding, list) or len(winding) != 2 or
                    any(type(v) is not int or abs(v) > 16 for v in winding) or winding == [0, 0]):
                raise ValueError("Winding must be two integers in [-16,16], with a nonzero pair")
            if type(source["samples"]) is not int or not 2 <= source["samples"] <= MAX_SAMPLES:
                raise ValueError("Flat trajectory requires 2..128 samples")
        else:
            grid = source["arclength"]
            if not isinstance(grid, list) or not 2 <= len(grid) <= MAX_SAMPLES:
                raise ValueError("Curvature profile requires 2..128 arclength samples")
            grid = _vector(grid, len(grid), "arclength", 8)
            if grid[0] != 0 or any(a >= b for a, b in zip(grid, grid[1:])):
                raise ValueError("Arclength must begin at zero and strictly increase")
            curvature = _number(source["gaussian_curvature"], "Gaussian curvature", 1)
            if any(abs(curvature) * (b - a) ** 2 > .01000000000001 for a, b in zip(grid, grid[1:])):
                raise ValueError("Refine the grid: each step times sqrt(abs(curvature)) must be at most 0.1")
            _keys(source["units"], {"length", "angle"})
            if (source["units"]["length"] not in ("m", "mm", "normalized_length") or
                    source["units"]["angle"] != "radian"):
                raise ValueError("Declare m, mm or normalized_length and radian units")
            _vector(source["initial_perturbation"], 2, "initial perturbation", .1)
            covariance = source["starting_covariance"]
            _keys(covariance, {"matrix", "basis", "note"})
            if covariance["basis"] != "assumed":
                raise ValueError("The reference requires explicitly assumed starting covariance")
            _text(covariance["note"])
            matrix = _covariance(covariance["matrix"], "starting covariance")
            if any(abs(x) > 1 for row in matrix for x in row):
                raise ValueError("Starting covariance entries must have magnitude at most one")
            if not 0 < _number(source["relative_tolerance"], "linearization relative tolerance") <= .01:
                raise ValueError("Declare a linearization tolerance in (0,.01]")
        return source
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed geodesic reference source") from exc


_BOOTSTRAP = r'''
import dataclasses, json, sys
import numpy as np
role, root = sys.argv[1:3]
sys.path.insert(0, root)
source = json.loads(sys.stdin.buffer.read())
def native(value):
    if dataclasses.is_dataclass(value): return native(dataclasses.asdict(value))
    if isinstance(value, dict): return {key: native(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)): return [native(item) for item in value]
    if isinstance(value, np.ndarray): return native(value.tolist())
    if isinstance(value, np.generic): return native(value.item())
    if isinstance(value, complex): return {'re':value.real, 'im':value.imag}
    return value
if role == 'ftr':
    from flat_torus.companion import flat_geodesic_reference
    from flat_torus.trajectories import trace_closed_geodesic
    tau, start = complex(*source['tau']), complex(*source['start'])
    m,n = source['winding']
    reference = flat_geodesic_reference(tau,m,n,start=start)
    trajectory = trace_closed_geodesic(tau,m,n,start=start,samples=source['samples'])
    data = {'schema':'ciw.flat-torus-reference-native.v1','reference':reference.as_dict(),'trajectory':native(trajectory)}
elif role == 'csg':
    from geodesic_testbed.jacobi import integrate_jacobi
    from geodesic_testbed.boundary import Units, StartingCovariance
    units = Units(**source['units'])
    covariance = StartingCovariance(units=units,**source['starting_covariance'])
    trace = integrate_jacobi(source['arclength'],source['gaussian_curvature'])
    record = trace.as_transfer_record(units=units,covariance=covariance,
        relative_tolerance=source['relative_tolerance'],observation_mode=source['configuration']['observation_mode'])
    lateral,heading = source['initial_perturbation']
    data = {'schema':'ciw.curved-path-transfer-native.v1','record':record.to_dict(include_samples=True),
        'propagated_covariance':record.propagate_declared_covariance(),
        'separation':record.separation(lateral,heading),'heading_change':record.heading_change(lateral,heading),
        'determinant':record.determinant}
else:
    raise ValueError('Unsupported fixed geodesic reference provider')
print(json.dumps(native(data),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False))
'''


def _pair(value, name):
    _keys(value, {"re", "im"})
    return {key: _number(value[key], name) for key in ("re", "im")}


def _check_flat(source, data):
    _keys(data, {"schema", "reference", "trajectory"})
    if data["schema"] != "ciw.flat-torus-reference-native.v1":
        raise ValueError("Unsupported native flat reference")
    reference, trajectory = data["reference"], data["trajectory"]
    _keys(reference, {"schema", "claim_scope", "source_repository", "consumer_repository", "geometry", "geometry_digest", "handoff", "limitations"})
    if (reference["schema"] != "flat-geodesic-reference-v1" or reference["claim_scope"] != "exact-flat-geodesic-reference" or
            reference["source_repository"] != "giasonpooni/Flat-Torus-Geodesic-Reference" or
            reference["consumer_repository"] != "giasonpooni/Curved-Surface-Geodesic-Sensitivity-Runtime"):
        raise ValueError("Flat reference owner or claim scope mismatch")
    geometry = reference["geometry"]
    _keys(geometry, {"tau", "omega1", "omega2", "area", "winding", "start", "cover_vector", "length", "unit_tangent", "closed", "gaussian_curvature", "arclength_interval"})
    for field in ("tau", "omega1", "omega2", "start", "cover_vector", "unit_tangent"):
        _pair(geometry[field], field)
    _same(geometry["tau"], dict(zip(("re", "im"), map(float, source["tau"]))), "Flat shape differs from source")
    _same(geometry["winding"], source["winding"], "Flat winding differs from source")
    # A native start is reduced into the quotient cell, so equality to the
    # submitted cover point is modulo the retained lattice. The absolute
    # 1e-12 allowance covers binary64 basis conversion at these source bounds.
    first, second, start = (geometry[k] for k in ("omega1", "omega2", "start"))
    if first["im"] != 0 or first["re"] <= 0 or second["im"] <= 0:
        raise ValueError("Invalid normalized flat lattice orientation")
    beta = (source["start"][1] - start["im"]) / second["im"]
    alpha = (source["start"][0] - start["re"] - beta * second["re"]) / first["re"]
    reduced_beta = start["im"] / second["im"]
    reduced_alpha = (start["re"] - reduced_beta * second["re"]) / first["re"]
    if (any(abs(v - round(v)) > 1e-12 for v in (alpha, beta)) or
            any(not -1e-12 <= v < 1 + 1e-12 for v in (reduced_alpha, reduced_beta))):
        raise ValueError("Native flat start is not the declared point modulo the lattice")
    if (geometry["closed"] is not True or _number(geometry["gaussian_curvature"], "flat curvature") != 0 or
            _number(geometry["area"], "flat area") <= 0 or _number(geometry["length"], "loop length") <= 0):
        raise ValueError("Invalid native flat geometry")
    _same(geometry["arclength_interval"], [0.0, geometry["length"]], "Flat arclength interval mismatch")
    if reference["geometry_digest"] != sha256(canonical(geometry)).hexdigest():
        raise ValueError("Native flat geometry digest mismatch")
    _same(reference["handoff"], {"quantity": "declared-curvature-profile", "curvature_model": "constant",
        "gaussian_curvature": 0.0, "arclength_interval": geometry["arclength_interval"]}, "Flat handoff mismatch")
    _same(reference["limitations"], ["Zero local curvature does not encode quotient edge identifications.",
        "This reference does not propagate Jacobi fields or manufacturing tolerances.",
        "The exact closed-loop claim applies to the declared integer winding."], "Native flat limitations changed")
    _keys(trajectory, {"lattice", "winding", "start", "length", "times", "parallelogram_points", "cover_points", "crossings", "closed", "notes"})
    _same(trajectory["lattice"], {"shape": dict(zip(("x", "y"), map(float, source["tau"]))),
        "omega1": geometry["omega1"], "omega2": geometry["omega2"]}, "Trajectory lattice differs from reference")
    _same(trajectory["winding"], dict(zip(("m", "n"), source["winding"])), "Trajectory winding differs from reference")
    for field in ("start", "length", "closed"):
        _same(trajectory[field], geometry[field], "Trajectory differs from native reference")
    count = source["samples"]
    times = _vector(trajectory["times"], count, "path parameter")
    if times[0] != 0 or times[-1] != 1 or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("Invalid native path parameter grid")
    for field in ("parallelogram_points", "cover_points"):
        points = trajectory[field]
        if not isinstance(points, list) or len(points) != count:
            raise ValueError("Invalid native flat path shape")
        for point in points:
            _pair(point, field)
        if any(abs(points[0][axis] - geometry["start"][axis]) > 1e-12 for axis in ("re", "im")):
            raise ValueError("Native trajectory initial point differs from reference")
    _text(trajectory["notes"])
    crossings = trajectory["crossings"]
    if not isinstance(crossings, list) or len(crossings) > sum(abs(v) for v in source["winding"]):
        raise ValueError("Invalid native quotient crossing count")
    previous = 0.0
    for crossing in crossings:
        _keys(crossing, {"parameter", "edge", "incoming", "outgoing", "lattice_step"})
        value = _number(crossing["parameter"], "crossing parameter")
        if not previous < value <= 1:
            raise ValueError("Crossings must follow the native parameter order")
        previous = value
        _text(crossing["edge"])
        for field in ("incoming", "outgoing"):
            _pair(crossing[field], field)
        step = crossing["lattice_step"]
        if not isinstance(step, list) or len(step) != 2 or any(type(v) is not int or abs(v) > 1 for v in step):
            raise ValueError("Invalid native quotient lattice step")


def _check_curved(source, data):
    _keys(data, {"schema", "record", "propagated_covariance", "separation", "heading_change", "determinant"})
    if data["schema"] != "ciw.curved-path-transfer-native.v1":
        raise ValueError("Unsupported native path transfer")
    record = data["record"]
    _keys(record, {"schema", "contract", "frame", "units", "source_digest", "resolution", "validity", "observation_mode", "domain",
        "samples", "grid", "covariance", "provenance", "calibration", "path_type", "path_type_basis", "chart", "geometry",
        "arclength", "gaussian_curvature", "a", "a_rate", "b", "b_rate"})
    if (record["schema"] != "path-transfer-record-v2" or record["contract"] != "path-sensitivity-boundary-v1" or
            record["frame"] != FRAME or record["domain"] != "constant-curvature" or record["path_type"] != "geodesic" or
            record["observation_mode"] != source["configuration"]["observation_mode"] or record["geometry"] is not None or record["chart"] is not None):
        raise ValueError("Native transfer scope differs from the declared curvature profile")
    _same(record["units"], source["units"], "Transfer units differ from source")
    count = len(source["arclength"])
    if type(record["samples"]) is not int or record["samples"] != count:
        raise ValueError("Native transfer sample count mismatch")
    for field in ("arclength", "gaussian_curvature", "a", "a_rate", "b", "b_rate"):
        _vector(record[field], count, field)
    _same(record["arclength"], list(map(float, source["arclength"])), "Native arclength differs from declared grid")
    _same(record["gaussian_curvature"], [float(source["gaussian_curvature"])] * count, "Native curvature differs from declaration")
    _same([record[field][0] for field in ("a", "a_rate", "b", "b_rate")], [1.0, 0.0, 0.0, 1.0], "Native transfer initial condition mismatch")
    description = {"curvature": [float(source["gaussian_curvature"])], "constant": True,
                   "span": [0.0, float(source["arclength"][-1])]}
    if record["source_digest"] != "sha256:" + sha256(canonical(description)).hexdigest()[:32]:
        raise ValueError("Native curvature description digest mismatch")
    declared = source["starting_covariance"]
    _same(record["covariance"], {"declared": True, "matrix": [[float(x) for x in row] for row in declared["matrix"]],
        "frame": FRAME, "units": source["units"], "basis": "assumed", "coverage_factor": None, "note": declared["note"]},
        "Native covariance differs from the declared starting distribution")
    _same(record["calibration"], {"bound": False, "calibration_ids": [], "registration_id": "", "reconstruction_version": "",
        "instrument_id": "", "note": "no instrument took part in this computation"}, "Reference cannot claim calibration")
    _same(record["provenance"], {"producer": "curved-surface-geodesic-sensitivity-runtime", "producer_version": "0.3.0",
        "upstream": [], "note": "Jacobi transfer from a declared curvature profile", "extra": {}}, "Native transfer provenance differs")
    if record["path_type_basis"] != "declared: K(s) is given as the curvature along a geodesic, which is what makes j'' + K j = 0 the right equation":
        raise ValueError("Native geodesic assumption changed")
    grid, resolution = record["grid"], record["resolution"]
    _keys(grid, {"start", "end", "samples", "min_step", "max_step", "uniform"})
    _keys(resolution, {"method", "samples", "max_step", "uniform", "convergence"})
    for item in (grid, resolution):
        if type(item["samples"]) is not int or item["samples"] != count or type(item["uniform"]) is not bool:
            raise ValueError("Native grid metadata types or sample count differ")
    deltas = [b - a for a, b in zip(record["arclength"], record["arclength"][1:])]
    if (grid["start"] != 0.0 or grid["end"] != record["arclength"][-1] or
            _number(grid["min_step"], "minimum step") != min(deltas) or _number(grid["max_step"], "maximum step") != max(deltas) or
            _number(resolution["max_step"], "resolution step") != max(deltas) or resolution["method"] != "rk4"):
        raise ValueError("Native solver resolution differs from its retained grid")
    validity = record["validity"]
    _keys(validity, {"basis", "established", "observation_mode", "relative_tolerance", "tolerance_basis", "max_lateral", "max_heading",
        "directions", "probe_magnitudes", "pointwise_error", "route_error", "probe_limited", "probe_limited_directions", "fits", "reference",
        "reference_method", "reference_samples", "reference_digest", "convergence", "note"})
    if (validity["established"] is not True or validity["max_lateral"] is not None or validity["directions"] != ["heading"] or
            validity["observation_mode"] != "intrinsic-surface-distance" or validity["reference"] != "closed-form" or
            validity["relative_tolerance"] != source["relative_tolerance"] or validity["pointwise_error"] != source["relative_tolerance"] or
            _number(validity["max_heading"], "heading validity bound") <= 0):
        raise ValueError("Native validity cannot expand beyond its declared heading-only scope")
    for convergence in (resolution["convergence"], validity["convergence"]):
        _keys(convergence, {"basis", "order", "refinement", "position", "transfer", "curvature", "focus", "covariance", "note", "established", "worst"})
        if (convergence["established"] is not False or not isinstance(convergence["basis"], str) or
                not convergence["basis"].startswith("not-established:") or
                any(convergence[k] is not None for k in ("order", "refinement", "position", "transfer", "curvature", "focus", "covariance", "worst"))):
            raise ValueError("A single native solve cannot claim step-doubling convergence")
    for field in ("separation", "heading_change", "determinant"):
        _vector(data[field], count, field)
    _same([data["separation"][0], data["heading_change"][0]], list(map(float, source["initial_perturbation"])),
          "Native initial perturbation differs from source")
    covariance = data["propagated_covariance"]
    if not isinstance(covariance, list) or len(covariance) != count:
        raise ValueError("Native propagated covariance shape mismatch")
    for matrix in covariance:
        _matrix(matrix, 2, 2, "propagated covariance")
        if matrix[0][0] < 0 or matrix[1][1] < 0:
            raise ValueError("Native covariance contains a negative variance")
    _same(covariance[0], record["covariance"]["matrix"], "Initial propagated covariance must retain C0")


def _check_data(kind, source, data):
    try:
        canonical(data)
        (_check_flat if kind == "flat-torus-reference" else _check_curved)(source, data)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed native geodesic reference response") from exc


class GeodesicReferenceWorkflow(DeclaredWorkflow):
    def __init__(self, kind):
        if not isinstance(kind, str) or kind not in KINDS:
            raise ValueError("Unsupported geodesic reference kind")
        self.kind, self.pin = kind, PINS[kind]
        self.role = self.pin["role"]
        self.ROLES = {self.role}
        self.SOURCE_SCHEMA = "ciw." + kind + "-source.v1"
        self.schema, self.operation = "ciw." + kind + "-session.v1", "ciw." + kind + ".v1"

    def _source(self, raw):
        return _source(self.kind, raw)

    def _adapters(self, repositories, expected=None):
        bound = super()._adapters(repositories, expected)
        self._check_runtime(bound[1])
        return bound

    def _check_runtime(self, runtime):
        if runtime["source_tree"] != self.pin["source_tree"] or runtime["adapter_version"] != "ciw-pinned-subprocess-v1":
            raise ValueError("Geodesic provider source tree or adapter differs from the approved pin")
        version = runtime["python_version"]
        if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
            raise ValueError("Invalid geodesic runtime Python version")
        minimum = (3, 12) if self.role == "ftr" else (3, 11)
        if tuple(map(int, version.split(".")[:2])) < minimum:
            raise ValueError("Native reference provider requires Python " + ".".join(map(str, minimum)) + " or newer")
        _keys(runtime["dependencies"], {"numpy", "scipy"})
        for name, value in runtime["dependencies"].items():
            if value is not None:
                _text(value)
        if runtime["dependencies"]["numpy"] is None:
            raise ValueError("Native reference provider requires NumPy")

    def _validate(self, bundle):
        raw = super()._validate(bundle)
        self._check_runtime(bundle["runtimes"][self.role])
        try:
            receipts = bundle.get("replay_receipts", [])
            if not isinstance(receipts, list) or len(receipts) > 1:
                raise ValueError("A geodesic occurrence retains at most one replay receipt")
            for receipt in receipts:
                _keys(receipt, {"schema", "source_bundle_digest", "replayed_bundle_digest", "numerical_match",
                                "verification", "admission", "replay_id"})
                source_id = receipt["source_bundle_digest"]
                if (receipt["schema"] != "ciw." + self.kind + "-replay.v1" or
                        not isinstance(source_id, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", source_id) or
                        source_id == bundle["bundle_digest"] or receipt["replayed_bundle_digest"] != bundle["bundle_digest"] or
                        receipt["numerical_match"] is not True or receipt["admission"] != "not_performed"):
                    raise ValueError("Invalid geodesic replay receipt binding or authority")
                verification = receipt["verification"]
                _keys(verification, {"schema", "subject_ref", "outcome", "independent", "method", "runtime_digest",
                                     "reproduction", "authority", "verification_id"})
                if (verification["schema"] != VERIFY_SCHEMA or verification["subject_ref"] != source_id or
                        verification["outcome"] != "passed" or verification["independent"] is not False or
                        verification["method"] != "same_runtime_fresh_occurrence_reproduction" or
                        not isinstance(verification["runtime_digest"], str) or
                        not re.fullmatch(r"sha256:[a-f0-9]{64}", verification["runtime_digest"])):
                    raise ValueError("Invalid geodesic replay verification scope")
                _same(verification["authority"], AUTHORITY, "Replay cannot confer authority")
                _same(verification["reproduction"], bundle["steps"][0], "Replay must bind this exact fresh step")
                _identity(verification, "verification_id")
                if receipt["replay_id"] != digest({k: v for k, v in receipt.items() if k != "replay_id"}):
                    raise ValueError("Geodesic replay receipt content identity mismatch")
                # The historical runtime digest includes its original paths.
                # Standalone inspection cannot resolve that source bundle;
                # Workbench._validate_links checks it against retained history.
        except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
            raise ValueError("Malformed geodesic replay receipt") from exc
        return raw

    def _step(self, source, evidence_id, bound):
        adapter, runtime, _ = bound
        _same(self._runtime_projection(adapter.runtime_identity()), self._runtime_projection(runtime), "Geodesic provider changed before execution")
        code, raw = adapter._run(_BOOTSTRAP, [self.role, str(adapter.source_root)], canonical(source))
        _same(self._runtime_projection(adapter.runtime_identity()), self._runtime_projection(runtime), "Geodesic provider changed during execution")
        if code:
            raise AdapterRefusal("GEODESIC_REFERENCE_REFUSED", "Pinned " + self.role + " refused the declared reference")
        data = _json(raw)
        _check_data(self.kind, source, data)
        occurrence = "execution-" + uuid.uuid4().hex
        result = {"schema": RESULT_SCHEMA, "operation_id": self.operation, "execution_ref": occurrence,
                  "input_refs": [evidence_id], "data": data, "authority": deepcopy(AUTHORITY)}
        result["result_id"] = digest(result)
        numerical = {"operation_id": self.operation, "data": deepcopy(data)}
        return {"runtime_ref": self.role, "operation_id": self.operation, "execution_id": occurrence,
            "input_refs": [evidence_id], "request": deepcopy(source), "request_sha256": digest(source),
            "result": result, "result_sha256": digest(result), "result_id": result["result_id"],
            "numerical_result": numerical, "numerical_result_id": digest(numerical)}

    def _validate_step(self, step, source, evidence_id):
        _keys(step, {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256",
            "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"})
        if (step["runtime_ref"] != self.role or step["operation_id"] != self.operation or step["input_refs"] != [evidence_id] or
                not isinstance(step["execution_id"], str) or not re.fullmatch(r"execution-[a-f0-9]{32}", step["execution_id"])):
            raise ValueError("Invalid geodesic operation, evidence or execution identity")
        _same(step["request"], source, "Geodesic request differs from retained source")
        result = step["result"]
        _keys(result, {"schema", "operation_id", "execution_ref", "input_refs", "data", "authority", "result_id"})
        _check_data(self.kind, source, result["data"])
        _same(result["authority"], AUTHORITY, "Geodesic reference cannot confer authority")
        if (result["schema"] != RESULT_SCHEMA or result["operation_id"] != self.operation or result["execution_ref"] != step["execution_id"] or
                result["input_refs"] != [evidence_id] or result["result_id"] != step["result_id"] or
                result["result_id"] != digest({k: v for k, v in result.items() if k != "result_id"})):
            raise ValueError("Geodesic result binding mismatch")
        _same(step["numerical_result"], {"operation_id": self.operation, "data": result["data"]}, "Geodesic numerical projection mismatch")
        for key, content in (("request_sha256", source), ("result_sha256", result), ("numerical_result_id", step["numerical_result"])):
            if step[key] != digest(content):
                raise ValueError("Geodesic step content binding mismatch")

    def _check_verification(self, bundle, verification, source, evidence):
        _keys(verification, {"schema", "subject_ref", "outcome", "independent", "method", "runtime_digest", "reproduction", "authority", "verification_id"})
        self._validate_step(verification["reproduction"], source, evidence)
        old, new = bundle["steps"][0], verification["reproduction"]
        if old["execution_id"] == new["execution_id"] or old["result_id"] == new["result_id"]:
            raise ValueError("Geodesic verification requires a fresh native occurrence")
        _same(verification, _verification(bundle, new), "Geodesic verification binding or scope mismatch")
        _identity(verification, "verification_id")
