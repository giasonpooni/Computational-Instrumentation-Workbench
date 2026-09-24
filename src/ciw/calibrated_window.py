"""Shared affine calibration, nominal clock mapping and window-state workflow.

Native TBRT/MCUR Jacobians compose a full declared joint covariance. This is
first-order uncertainty conditional on the declared nominal sample grid; time
uncertainty does not silently become measurement noise or exact membership.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from fractions import Fraction
from importlib import resources
import json
import math
import re
import uuid

from . import telemetry
from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import PinnedSubprocessAdapter, _json
from .exchange import _identity
from .calibrated_observable import _exact_timestamp
from .telemetry import canonical, digest, byte_digest, _bundle_digest, _instant, _now, _keys

SCHEMA = "ciw.calibrated-window-session.v1"
SOURCE_SCHEMA = "ciw.calibrated-window-source.v1"
RESULT_SCHEMA = "ciw.calibrated-operation-result.v1"
MAX_BYTES = 4 * 1024 * 1024
OPERATIONS = (("tbrt", "ciw.tbrt-window.v1"), ("mcur", "ciw.mcur-window.v1"),
              ("stfe", "stfe.window-mean.v1"), ("gsie", "ciw.gsie-predict-update.v1"))
ROLES = frozenset(role for role, _ in OPERATIONS) | {"set"}
COMPOSITION = {"transformation": "affine", "order": "calibrate_then_mean",
               "calibration_operation_id": "mcur.affine-first-order.v1",
               "feature_operation_id": "stfe.window-mean.v1",
               "time_policy": "nominal_grid_with_retained_joint_time_uncertainty"}


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Require finite real numbers")
    if Fraction(value) != Fraction(float(value)):
        raise ValueError("Declared numbers must be exactly representable as binary64")
    return value


def _text(value):
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 512:
        raise ValueError("Require bounded unpadded identities")


def _refs(value):
    if not isinstance(value, list) or not value or len(value) > 32 or len(set(value)) != len(value):
        raise ValueError("Require distinct evidence references")
    for item in value:
        _text(item)


def covariance_order(source):
    ids = [row["observation_id"] for row in source["samples"]]
    return (["device_time:" + name for name in ids] + ["clock_skew", "clock_offset"] +
            ["indicated_value:" + name for name in ids] + ["calibration_gain", "calibration_offset"])


def _covariance(matrix, size):
    if not isinstance(matrix, list) or len(matrix) != size or any(not isinstance(r, list) or len(r) != size for r in matrix):
        raise ValueError("Full ordered joint covariance is required")
    exact = [[Fraction(_number(v)) for v in row] for row in matrix]
    if any(exact[i][j] != exact[j][i] for i in range(size) for j in range(size)):
        raise ValueError("Joint covariance must be exactly symmetric")
    # Exact Schur complements certify PSD, including singular shared effects.
    # No tolerance, jitter, triangle averaging or diagonal substitution.
    reduced = deepcopy(exact)
    for k in range(size):
        pivot = reduced[k][k]
        if pivot < 0 or (pivot == 0 and any(reduced[i][k] for i in range(k + 1, size))):
            raise ValueError("Joint covariance must be positive semidefinite")
        if pivot:
            for i in range(k + 1, size):
                for j in range(i, size):
                    reduced[i][j] -= reduced[i][k] * reduced[j][k] / pivot
                    reduced[j][i] = reduced[i][j]
    return exact


def _project(jacobian, matrix):
    exact = [[Fraction(v) for v in row] for row in matrix]
    rows = [[Fraction(v) for v in row] for row in jacobian]
    products = [[sum((a * b for a, b in zip(row, column)), Fraction())
                 for column in zip(*exact)] for row in rows]
    result = []
    for left in products:
        output = []
        for right in rows:
            value = sum((a * b for a, b in zip(left, right)), Fraction())
            rounded = float(value)
            if not math.isfinite(rounded) or (value and rounded == 0):
                raise ValueError("Covariance propagation exceeds binary64; rescale inputs")
            output.append(rounded)
        result.append(output)
    return result


def _source(raw):
    try:
        if not isinstance(raw, bytes) or len(raw) > MAX_BYTES:
            raise ValueError("Source must be bounded exact bytes")
        source = _json(raw)
        canonical(source)
        _keys(source, {"schema", "experiment_id", "epoch", "channel_id", "frame", "frame_mapping",
                       "device_clock", "receipt_clock", "clock_model", "calibration_profile", "samples",
                       "joint_covariance", "configuration"})
        if source["schema"] != SOURCE_SCHEMA:
            raise ValueError("Unsupported calibrated window source")
        for key in ("experiment_id", "channel_id"):
            _text(source[key])
        _exact_timestamp(source["epoch"], "Window epoch", require_utc=True)
        samples = source["samples"]
        if not isinstance(samples, list) or not 1 <= len(samples) <= 16:
            raise ValueError("A calibrated window requires 1 to 16 scalar samples")
        for sample in samples:
            _keys(sample, {"observation_id", "artifact_id", "sensor_id", "quantity_id", "unit",
                           "indicated_value", "raw_value", "device_time", "received_at"})
            for key in ("observation_id", "artifact_id", "sensor_id", "quantity_id", "unit"):
                _text(sample[key])
            for key in ("indicated_value", "raw_value", "device_time", "received_at"):
                _number(sample[key])
            _instant(source["epoch"], sample["received_at"])
        if len({row["observation_id"] for row in samples}) != len(samples):
            raise ValueError("Duplicate sample identity")
        model = source["clock_model"]
        _keys(model, {"model_id", "source_frame", "reference_frame", "device_origin", "reference_origin",
                      "skew", "offset", "valid_device_interval", "synchronization_evidence_ids"})
        if model["source_frame"] != source["device_clock"] or model["reference_frame"] != source["receipt_clock"]:
            raise ValueError("Device and receipt clocks must match the declared map")
        if source["receipt_clock"].get("time_scale") != "UTC" or source["receipt_clock"].get("unit") != "s":
            raise ValueError("Mapped coordinates are UTC seconds relative to the retained epoch")
        _refs(model["synchronization_evidence_ids"])
        for key in ("device_origin", "reference_origin", "skew", "offset"):
            _number(model[key])
        for value in model["valid_device_interval"]:
            _number(value)
        _keys(source["frame_mapping"], {"ref", "kind", "source", "target"})
        if (source["frame_mapping"]["kind"] != "identity" or
                source["frame_mapping"]["source"] != source["frame"]["id"] or
                source["frame_mapping"]["target"] != source["frame"]["id"]):
            raise ValueError("This window operation requires an explicit identity quantity frame map")
        profile = source["calibration_profile"]
        _keys(profile, {"profile_id", "artifact_id", "sensor_id", "quantity_id", "input_unit", "output_unit",
                        "gain", "offset", "coefficient_covariance", "valid_from", "valid_until", "reference_ids", "input_range"})
        for key in ("valid_from", "valid_until"):
            _exact_timestamp(profile[key], "calibration " + key)
        _refs(profile["reference_ids"])
        for key in ("gain", "offset"):
            _number(profile[key])
        for value in profile["input_range"]:
            _number(value)
        for sample in samples:
            if any(sample[a] != profile[b] for a, b in (("sensor_id", "sensor_id"), ("quantity_id", "quantity_id"), ("unit", "input_unit"))):
                raise ValueError("Every sample must share the declared affine profile applicability")
        joint = source["joint_covariance"]
        _keys(joint, {"order", "matrix", "cross_covariance_policy", "evidence_ids"})
        _refs(joint["evidence_ids"])
        if joint["order"] != covariance_order(source):
            raise ValueError("Joint covariance order differs from raw times, clock parameters, values and calibration parameters")
        if joint["cross_covariance_policy"] not in {"declared", "declared_zero"}:
            raise ValueError("Unknown cross-covariance refuses calibrated window estimation")
        size = 2 * len(samples) + 4
        _covariance(joint["matrix"], size)
        if joint["cross_covariance_policy"] == "declared_zero" and any(joint["matrix"][i][j] != 0 for i in range(size) for j in range(size) if i != j):
            raise ValueError("Declared-zero cross-covariance contradicts nonzero off-diagonal terms")
        if [row[-2:] for row in joint["matrix"][-2:]] != profile["coefficient_covariance"]:
            raise ValueError("Joint calibration parameter block differs from the retained profile")
        configuration = source["configuration"]
        _keys(configuration, {"window", "gsie", "composition"})
        window = configuration["window"]
        _keys(window, {"start", "end", "received_by", "decision_time", "sample_period", "max_lateness"})
        for key in ("start", "end", "received_by", "decision_time", "sample_period", "max_lateness"):
            _number(window[key])
        if not window["start"] < window["end"] or window["sample_period"] <= 0 or window["max_lateness"] < 0:
            raise ValueError("Window bounds, sample period and lateness must describe a forward window")
        prior = configuration["gsie"]["prior"]
        _number(prior["time"])
        if not isinstance(prior["mean"], list) or not prior["mean"]:
            raise ValueError("Prior mean must be a nonempty list of numbers")
        for value in prior["mean"]:
            _number(value)
        _covariance(prior["covariance"], len(prior["mean"]))
        if configuration["composition"] != COMPOSITION:
            raise ValueError("Only declared affine calibration before a window mean on the nominal time grid is implemented")
        gsie = configuration["gsie"]
        _keys(gsie, {"prior", "dynamics", "observation_model", "target_time", "variables", "frame",
                     "prior_measurement_crosscov_policy", "feature_observation_semantics"})
        if (gsie["prior_measurement_crosscov_policy"] != "declared_zero" or
                gsie["feature_observation_semantics"] != "window_mean_observes_declared_state_at_window_end" or
                gsie["target_time"] != configuration["window"]["end"]):
            raise ValueError("Declare independent prior and window-end feature observation semantics")
        if (gsie["dynamics"]["matrix"] != [[1]] or gsie["dynamics"]["process_covariance"] != [[0]] or
                gsie["observation_model"]["matrix"] != [[1]]):
            raise ValueError("Nominal time uncertainty currently requires a scalar stationary hold model")
        return source
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed calibrated window source") from exc


_BOOTSTRAP = r'''
import dataclasses, datetime, enum, json, sys
from fractions import Fraction
role, root = sys.argv[1:3]
sys.path.insert(0, root)
request = json.loads(sys.stdin.buffer.read())
source = request['source']
def native(value):
    if dataclasses.is_dataclass(value): return native(dataclasses.asdict(value))
    if isinstance(value, dict): return {k: native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [native(v) for v in value]
    if isinstance(value, datetime.datetime): return value.isoformat().replace('+00:00', 'Z')
    if isinstance(value, enum.Enum): return value.value
    return value
n = len(source['samples'])
matrix = source['joint_covariance']['matrix']
if role == 'tbrt':
    from tbrt import ClockFrame, TimePoint, TimestampObservation, AffineClockModel, reconcile_time
    model = dict(source['clock_model'])
    model['source_frame'] = ClockFrame(**model['source_frame'])
    model['reference_frame'] = ClockFrame(**model['reference_frame'])
    model = AffineClockModel(**model)
    result = []
    for i, sample in enumerate(source['samples']):
        observation = TimestampObservation(sample['device_time'], ClockFrame(**source['device_clock']),
            sample['artifact_id'], received_at=TimePoint(sample['received_at'], ClockFrame(**source['receipt_clock'])))
        indices = [i, n, n + 1]
        mapped = reconcile_time(observation, model, [[matrix[a][b] for b in indices] for a in indices],
            expected_reference=ClockFrame(**source['receipt_clock']), require_synchronization_evidence=True)
        if Fraction(mapped.reference_origin) + Fraction(mapped.event_time_delta) != Fraction(mapped.event_time):
            raise ValueError('Nominal event time loses retained origin/delta precision')
        result.append({'observation_id': sample['observation_id'], 'event_time': mapped.event_time,
                       'reconciliation': native(mapped)})
elif role == 'mcur':
    from mcur import Observation, CalibrationProfile, Interval, JointCovariance, CrossCovariancePolicy, calibrate, assess_feature_compatibility
    profile = dict(source['calibration_profile'])
    for key in ('valid_from', 'valid_until'):
        profile[key] = datetime.datetime.fromisoformat(profile[key].replace('Z', '+00:00'))
    profile['input_range'] = Interval(*profile['input_range'])
    profile = CalibrationProfile(**profile)
    epoch = datetime.datetime.fromisoformat(source['epoch'].replace('Z', '+00:00'))
    rows = []
    for i, (sample, mapped) in enumerate(zip(source['samples'], request['alignment']['samples'])):
        acquired = epoch + datetime.timedelta(seconds=mapped['event_time'])
        if (acquired - epoch).total_seconds() != mapped['event_time']:
            raise ValueError('Calibration event time loses microsecond precision')
        observation = {k: v for k, v in sample.items() if k not in ('device_time', 'received_at')}
        observation = Observation(**observation, acquired_at=acquired)
        indices = [n + 2 + i, 2*n + 2, 2*n + 3]
        joint = JointCovariance(values=[[matrix[a][b] for b in indices] for a in indices],
            cross_covariance_policy=CrossCovariancePolicy(source['joint_covariance']['cross_covariance_policy']),
            evidence_ids=source['joint_covariance']['evidence_ids'])
        rows.append(native(calibrate(observation, profile, joint)))
    compatibility = assess_feature_compatibility(profile, 'stfe.window-mean.v1',
        same_profile_for_all_samples=True, joint_temporal_covariance_declared=True)
    if compatibility.status.value != 'compatible': raise ValueError('Incompatible calibration and feature')
    result = {'samples': rows, 'compatibility': native(compatibility)}
else:
    raise ValueError('Unsupported fixed calibrated window provider')
print(json.dumps(result, allow_nan=False, ensure_ascii=False))
'''


def _pins():
    return json.loads(resources.files("ciw").joinpath("calibrated-window-runtimes.json").read_text())


def _adapters(repositories, expected=None):
    if set(repositories) != ROLES:
        raise ValueError("Bind exactly the five calibrated window providers")
    adapters = {}
    for role, pin in _pins().items():
        retained = expected[role] if expected else {}
        adapter = PinnedSubprocessAdapter(repositories[role], pin["revision"], pin["module"], source_root=pin["source_root"],
            expected_python_sha256=retained.get("python_sha256"), expected_python_version=retained.get("python_version"),
            expected_dependencies=retained.get("dependencies"))
        identity = adapter.runtime_identity()
        if retained and any(identity[k] != retained[k] for k in ("schema", "adapter_version", "revision", "source_tree", "module", "source_root", "python_sha256", "python_version", "dependencies")):
            raise ValueError("Calibrated window runtime identity mismatch")
        adapters[role] = adapter
    return adapters


def _jacobians(source, times, calibration=None):
    n = len(source["samples"])
    rows = []
    for i, mapped in enumerate(times["samples"]):
        row = [0] * (2*n + 4)
        for index, value in zip((i, n, n + 1), mapped["reconciliation"]["jacobian"]):
            row[index] = value
        rows.append(row)
    if calibration:
        for i, calibrated in enumerate(calibration["samples"]):
            row = [0] * (2*n + 4)
            for index, value in zip((n + 2 + i, 2*n + 2, 2*n + 3), calibrated["jacobian"]):
                row[index] = value
            rows.append(row)
    return rows


def _invoke(role, adapters, request):
    if role not in {"tbrt", "mcur"}:
        return telemetry._invoke(role, adapters, request)
    adapter = adapters[role]
    adapter.runtime_identity()
    encoded = canonical(request)
    if len(encoded) > MAX_BYTES:
        raise ValueError("Calibrated window request exceeds budget")
    code, raw = adapter._run(_BOOTSTRAP, [role, str(adapter.source_root)], encoded)
    if code:
        raise AdapterRefusal("CALIBRATED_WINDOW_" + role.upper() + "_REFUSED", "Pinned " + role + " refused calibrated window inputs")
    adapter.runtime_identity()
    source = request["source"]
    result = _json(raw)
    if role == "tbrt":
        result = {"samples": result, "time_policy": COMPOSITION["time_policy"]}
        result["temporal_covariance"] = _project(_jacobians(source, result), source["joint_covariance"]["matrix"])
    else:
        n = len(source["samples"])
        result["joint_time_value_covariance"] = _project(_jacobians(source, request["alignment"], result), source["joint_covariance"]["matrix"])
        result["temporal_covariance"] = [row[n:] for row in result["joint_time_value_covariance"][n:]]
        if any(result["temporal_covariance"][i][i] != row["variance"] for i, row in enumerate(result["samples"])):
            raise ValueError("Composed covariance differs from native MCUR marginal propagation")
    return result


def _request(index, source, steps, execution_id, created_at, runtimes, evidence_ref):
    if index == 0:
        return {"source": deepcopy(source)}, [evidence_ref]
    if index == 1:
        return {"source": deepcopy(source), "alignment": deepcopy(steps[0]["result"]["data"])}, [evidence_ref, steps[0]["result_id"]]
    if index == 2:
        times, calibrated = (steps[i]["result"]["data"] for i in (0, 1))
        n = len(source["samples"])
        profile = source["calibration_profile"]
        refs = ["calibrated:" + digest({"source": evidence_ref, "observation": row["observation_id"]}) for row in source["samples"]]
        request = {"operation_id": "stfe.window-mean.v1", "source_batch_ref": steps[1]["result_id"],
            "source_batch_digest": digest(steps[1]["result"]),
            "samples": [{"observation_ref": refs[i], "event_time": times["samples"][i]["event_time"],
                         "received_at": source["samples"][i]["received_at"], "value": calibrated["samples"][i]["corrected_value"],
                         "missing": False} for i in range(n)],
            "window": deepcopy(source["configuration"]["window"]), "channel_id": source["channel_id"],
            "value_unit": profile["output_unit"], "frame": source["frame"]["id"],
            "clock_basis": source["receipt_clock"]["clock_id"], "clock_mapping_ref": source["clock_model"]["model_id"],
            "frame_mapping_ref": source["frame_mapping"]["ref"], "calibration_refs": [profile["artifact_id"], *profile["reference_ids"]],
            "uncertainty": {"status": "known", "crosscov_policy": "declared", "matrix": deepcopy(calibrated["temporal_covariance"]),
                            "observation_refs": refs},
            "execution_id": execution_id, "created_at": created_at, "implementation_revision": runtimes["stfe"]["revision"]}
        # All supplied samples belong to this one declared window. STFE refuses
        # duplicates, irregular grid, lateness and missing entries in that order.
        if any(not request["window"]["start"] <= row["event_time"] < request["window"]["end"] for row in request["samples"]):
            raise ValueError("Mapped sample falls outside the declared window; no silent selection")
        return request, [steps[0]["result_id"], steps[1]["result_id"]]
    feature_source = {"epoch": source["epoch"], "clock_basis": source["receipt_clock"]["clock_id"]}
    return {"declaration": deepcopy(source["configuration"]["gsie"]), "epoch": source["epoch"],
            "observation_batch": telemetry._feature_batch(steps[2]["result"], feature_source, source["configuration"]),
            "execution_id": execution_id, "created_at": created_at}, [steps[2]["result_id"]]


def _numerical(role, result):
    if role in {"tbrt", "mcur"}:
        return {"operation_id": result["operation_id"], "data": deepcopy(result["data"])}
    return telemetry._numerical(role, result)


def _execute(raw, adapters, template=None):
    source = _source(raw)
    created_at = template["created_at"] if template else _now()
    runtimes = {role: adapter.runtime_identity() for role, adapter in adapters.items()}
    steps = []
    for i, (role, operation) in enumerate(OPERATIONS):
        execution = template["steps"][i]["execution_id"] if template else "execution-" + uuid.uuid4().hex
        request, refs = _request(i, source, steps, execution, created_at, runtimes, byte_digest(raw))
        result = _invoke(role, adapters, request)
        if i < 2:
            result = {"schema": RESULT_SCHEMA, "operation_id": operation, "execution_ref": execution, "input_refs": refs, "data": result}
            result["result_id"] = digest(result)
        numerical = _numerical(role, result)
        artifact = result.get("result_artifact", result)
        steps.append({"runtime_ref": role, "operation_id": operation, "execution_id": execution, "input_refs": refs,
                      "request": request, "request_sha256": digest(request), "result": result, "result_sha256": digest(result),
                      "result_id": artifact["result_id"], "numerical_result": numerical, "numerical_result_id": digest(numerical)})
    bundle = {"schema": SCHEMA, "session_id": "session-" + uuid.uuid4().hex, "created_at": created_at,
              "source": {"experiment_id": source["experiment_id"], "experiment_digest": digest(source),
                         "evidence": [{"artifact_ref": byte_digest(raw), "sha256": byte_digest(raw), "bytes_b64": base64.b64encode(raw).decode()}]},
              "configuration": deepcopy(source["configuration"]), "runtimes": runtimes, "steps": steps}
    bundle["bundle_digest"] = _bundle_digest(bundle)
    return bundle


def _validate(bundle):
    try:
        _keys(bundle, {"schema", "session_id", "created_at", "source", "configuration", "runtimes", "steps", "bundle_digest"}, {"verification", "replay_receipts"})
        if len(canonical(bundle)) > MAX_BYTES or bundle["schema"] != SCHEMA or bundle["bundle_digest"] != _bundle_digest(bundle):
            raise ValueError("Calibrated window content binding mismatch")
        evidence, = bundle["source"]["evidence"]
        raw = base64.b64decode(evidence["bytes_b64"], validate=True)
        if evidence != {"artifact_ref": byte_digest(raw), "sha256": byte_digest(raw), "bytes_b64": base64.b64encode(raw).decode()}:
            raise ValueError("Exact source byte binding mismatch")
        source = _source(raw)
        if bundle["source"] != {"experiment_id": source["experiment_id"], "experiment_digest": digest(source), "evidence": [evidence]} or canonical(bundle["configuration"]) != canonical(source["configuration"]):
            raise ValueError("Source/configuration binding mismatch")
        if [(s["runtime_ref"], s["operation_id"]) for s in bundle["steps"]] != list(OPERATIONS) or set(bundle["runtimes"]) != ROLES:
            raise ValueError("Calibrated window graph/runtime mismatch")
        for role, pin in _pins().items():
            runtime = bundle["runtimes"][role]
            if (runtime.get("schema") != "ciw.subprocess-runtime.v1" or any(runtime.get(k) != pin[k] for k in ("revision", "module", "source_root")) or
                    not re.fullmatch("[a-f0-9]{40}", runtime.get("source_tree", "")) or
                    not re.fullmatch("[a-f0-9]{64}", runtime.get("python_sha256", "")) or
                    not isinstance(runtime.get("dependencies"), dict) or not runtime.get("python_version")):
                raise ValueError("Retained runtime differs from approved pin")
        identities = {bundle["session_id"], evidence["artifact_ref"], *(op for _, op in OPERATIONS)}
        for i, step in enumerate(bundle["steps"]):
            _keys(step, {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256", "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"})
            request, refs = _request(i, source, bundle["steps"][:i], step["execution_id"], bundle["created_at"], bundle["runtimes"], evidence["artifact_ref"])
            if canonical(step["request"]) != canonical(request) or step["input_refs"] != refs:
                raise ValueError("Request differs from retained clock/calibration/window lineage")
            result = step["result"]
            if canonical(step["numerical_result"]) != canonical(_numerical(step["runtime_ref"], result)):
                raise ValueError("Numerical projection mismatch")
            for key, content in (("request_sha256", step["request"]), ("result_sha256", result), ("numerical_result_id", step["numerical_result"])):
                if step[key] != digest(content):
                    raise ValueError("Step content binding mismatch")
            artifact = result.get("result_artifact", result)
            if artifact["result_id"] != step["result_id"] or artifact["execution_ref"] != step["execution_id"] or result["operation_id"] != step["operation_id"]:
                raise ValueError("Native result/operation/occurrence mismatch")
            if i < 2:
                _keys(result, {"schema", "operation_id", "execution_ref", "input_refs", "data", "result_id"})
                if result["schema"] != RESULT_SCHEMA or result["input_refs"] != refs or result["result_id"] != digest({k: v for k, v in result.items() if k != "result_id"}):
                    raise ValueError("Clock/calibration native wrapper identity mismatch")
            else:
                _identity(artifact, "result_id")
            if step["execution_id"] in identities or step["result_id"] in identities or step["execution_id"] == step["result_id"]:
                raise ValueError("Evidence, operation, execution and result identities must differ")
            identities.update((step["execution_id"], step["result_id"]))
        if "verification" in bundle:
            _identity(bundle["verification"], "verification_id")
            if bundle["verification"]["subject_ref"] != bundle["bundle_digest"] or bundle["verification"].get("independent") is not False:
                raise ValueError("Verification subject or authority mismatch")
        return raw
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed calibrated window session") from exc


def _verify(bundle, fresh, adapters):
    values = {old["execution_id"]: new["numerical_result"] for old, new in zip(bundle["steps"], fresh["steps"])}
    receipt = _invoke("set", adapters, {"bundle": bundle, "replay_results": values, "created_at": _now()})
    if receipt.get("outcome") != "passed":
        raise ValueError("SET did not verify calibrated window replay")
    return receipt, values


def create_session(raw, repositories):
    _source(raw)
    adapters = _adapters(repositories)
    bundle = _execute(raw, adapters)
    _validate(bundle)
    reproduced = _execute(raw, adapters, bundle)
    telemetry._compare_reproduction(bundle, reproduced)
    bundle["verification"], _ = _verify(bundle, reproduced, adapters)
    return bundle


def replay_session(bundle, repositories):
    raw = _validate(bundle)
    adapters = _adapters(repositories, bundle["runtimes"])
    reproduced = _execute(raw, adapters, bundle)
    telemetry._compare_reproduction(bundle, reproduced)
    fresh = _execute(raw, adapters)
    verification, values = _verify(bundle, fresh, adapters)
    fresh["verification"], _ = _verify(fresh, reproduced, adapters)
    receipt = {"schema": "ciw.calibrated-window-replay.v1", "source_bundle_digest": bundle["bundle_digest"],
               "replayed_bundle_digest": fresh["bundle_digest"], "numerical_match": True,
               "verification": verification, "admission": "not_performed"}
    receipt["replay_id"] = digest(receipt)
    fresh["replay_receipts"] = [receipt]
    return {"session": fresh, "replay_receipt": receipt, "replay_results": values}
