"""Pinned two-channel process experiment with retained evidence and exact replay.

The fixed script delegates scientific operations to their owning repositories.
Repository paths are operator bindings, never executable instructions in artifacts.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timedelta
from importlib import resources
import json
import math
from pathlib import Path
import re
import uuid

from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import PinnedSubprocessAdapter, _json
from .exchange import _read, _identity
from .session import write_json
from .telemetry import canonical, digest, byte_digest, _now, _bundle_digest

SCHEMA = "ciw.calibrated-observable-session.v1"
RESULT_SCHEMA = "ciw.calibrated-operation-result.v1"
MAX_BYTES = 4 * 1024 * 1024
OPERATIONS = (
    ("fsrt", "fsrt.declare-calibrated-two-channel.v1"),
    ("tbrt", "ciw.tbrt-two-channel.v1"),
    ("mcur", "ciw.mcur-two-channel.v1"),
    ("oit", "oit.finite-horizon-linear.v1"),
    ("gsie", "ciw.gsie-observable-predict-update.v1"),
    ("cbsr", "cbsr.affine-exact.v1"),
    ("fdir", "fdir.residual-isolability.v1"),
)
ROLES = frozenset(role for role, _ in OPERATIONS) | {"set"}

def _exact_timestamp(value, name, *, require_utc=False):
    """Reject precision loss before Python's microsecond datetime conversion."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a timezone-aware ISO timestamp")
    # Keep the previously supported ISO spellings (including legacy space
    # separators and offset seconds), while checking fractions before parsing.
    if any(match.group(1)[6:].strip("0") for match in re.finditer(r"[.,](\d+)", value)):
        raise ValueError(f"{name} cannot be represented at microsecond precision")
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware ISO timestamp")
    offset_fraction = re.search(r"[+-]\d{2}(?::?\d{2}){0,2}[.,](\d+)$", value)
    if (offset_fraction and offset_fraction.group(1).strip("0")
            and instant.utcoffset().microseconds == 0):
        # fromisoformat also drops fractional offsets below one whole second.
        raise ValueError(f"{name} timezone offset cannot be represented without precision loss")
    if require_utc and instant.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must declare UTC")
    return instant


# Source-controlled adapter code only; no artifact can select code or imports.
_BOOTSTRAP = r'''
import dataclasses, datetime, enum, json, math, sys
from fractions import Fraction
import numpy as np
role, root = sys.argv[1:3]
sys.path.insert(0, root)
request = json.loads(sys.stdin.buffer.read())
def native(value):
    if dataclasses.is_dataclass(value):
        return {key: native(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, datetime.datetime):
        return value.isoformat().replace('+00:00', 'Z')
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
if role == 'set':
    from state_estimation_testbed.replay import verify_replay_bundle
    result = verify_replay_bundle(request['bundle'], replay_results=request.get('replay_results'),
                                  created_at=request['created_at'])
elif role == 'fsrt':
    from set_lcm.bridge.calibrated_observable import declare
    result = declare(**request['inputs'])
else:
    d = request['inputs']
    if role == 'tbrt':
        from tbrt import ClockFrame, TimestampObservation, AffineClockModel, reconcile_time
        rows = []
        for channel in d['channels']:
            model = dict(channel['clock_model'])
            model['source_frame'] = ClockFrame(**model['source_frame'])
            model['reference_frame'] = ClockFrame(**model['reference_frame'])
            model = AffineClockModel(**model)
            observation = TimestampObservation(channel['observation']['device_time'],
                                               ClockFrame(**channel['observation']['clock_frame']),
                                               channel['observation']['artifact_id'])
            mapped = reconcile_time(observation, model, channel['clock_joint_covariance'],
                                    require_synchronization_evidence=True)
            if Fraction(mapped.reference_origin) + Fraction(mapped.event_time_delta) != Fraction(mapped.event_time):
                raise ValueError('nominal alignment cannot collapse retained clock origin/delta precision')
            rows.append({'channel_id': channel['channel_id'], 'event_time': mapped.event_time,
                         'reconciliation': native(mapped)})
        times = [row['event_time'] for row in rows]
        limit = d['alignment']['maximum_nominal_separation_s']
        if type(limit) not in (int, float) or not math.isfinite(limit) or limit < 0:
            raise ValueError('invalid nominal alignment tolerance')
        if max(times) - min(times) > limit:
            raise ValueError('nominal channel times exceed declared alignment tolerance')
        result = {'channels': rows, 'target_time': max(times), 'nominal_separation_s': max(times)-min(times),
                  'time_uncertainty_policy': d['alignment']['measurement_time_policy']}
    elif role == 'mcur':
        from mcur import Observation, CalibrationProfile, Interval, JointCovariance, CrossCovariancePolicy, calibrate
        epoch = datetime.datetime.fromisoformat(d['epoch_utc'].replace('Z', '+00:00'))
        if epoch.tzinfo is None or epoch.utcoffset() != datetime.timedelta(0):
            raise ValueError('epoch_utc must declare UTC')
        rows = []
        for channel, aligned in zip(d['channels'], d['alignment']['channels']):
            raw = dict(channel['observation'])
            del raw['device_time']
            del raw['clock_frame']
            elapsed = aligned['event_time']
            acquired = epoch + datetime.timedelta(seconds=elapsed)
            if (acquired - epoch).total_seconds() != elapsed:
                raise ValueError('calibration event time would lose precision')
            observation = Observation(**raw, acquired_at=acquired)
            profile = dict(channel['calibration_profile'])
            for key in ('valid_from', 'valid_until'):
                profile[key] = datetime.datetime.fromisoformat(profile[key].replace('Z', '+00:00'))
            profile['input_range'] = Interval(*profile['input_range'])
            references = profile['reference_ids']
            if (not isinstance(references, list) or not references or
                any(not isinstance(ref, str) or not ref.strip() for ref in references) or
                len(set(references)) != len(references)):
                raise ValueError('calibration evidence is required')
            joint = dict(channel['calibration_joint_covariance'])
            joint['cross_covariance_policy'] = CrossCovariancePolicy(joint['cross_covariance_policy'])
            if not joint['evidence_ids']:
                raise ValueError('calibration covariance evidence is required')
            result = calibrate(observation, CalibrationProfile(**profile), JointCovariance(**joint))
            rows.append({'channel_id': channel['channel_id'], 'calibration': native(result)})
        declared = d['calibrated_covariance']
        if declared['cross_covariance_policy'] == 'unknown' or not declared['evidence_ids']:
            raise ValueError('unknown calibrated cross-covariance cannot enter estimation')
        covariance = np.asarray(declared['matrix'], dtype=float)
        if any(covariance[i, i] != row['calibration']['variance'] for i, row in enumerate(rows)):
            raise ValueError('combined covariance diagonal differs from calibration propagation')
        if declared['cross_covariance_policy'] == 'declared_zero' and covariance[0, 1] != 0:
            raise ValueError('declared_zero contradicts combined covariance')
        result = {'channels': rows, 'values': [row['calibration']['corrected_value'] for row in rows],
                  'units': [row['calibration']['output_unit'] for row in rows],
                  'covariance': declared}
    elif role == 'oit':
        from oit import lti_observability
        declaration = dict(d['declaration'])
        model_id = declaration.pop('model_id')
        assessed = lti_observability(**declaration)
        result = native(assessed)
        result.update(model_id=model_id, rank=assessed.rank, condition_number=native(assessed.condition_number),
                      state_dimension=assessed.diagnostics.input_dimension)
        result['observation_matrix'] = declaration['observation']
        result['declaration'] = d['declaration']
    elif role == 'gsie':
        from geometric_state_inference import (StatePrior, LinearDynamics, LinearObservation, Observation,
                                               ObservabilityAssessment, predict, update_observable)
        declaration = d['declaration']
        prior = predict(StatePrior(**declaration['prior']), LinearDynamics(**declaration['dynamics']), d['time'])
        observation = Observation(time=d['time'], values=d['values'], covariance=d['covariance'],
            frame_id=declaration['observation_model']['measurement_frame_id'], units=d['units'],
            observation_id=d['observation_id'], evidence_refs=d['evidence_refs'])
        estimate = update_observable(prior, observation, LinearObservation(**declaration['observation_model']),
                                     ObservabilityAssessment(**d['observability']))
        result = {key: native(getattr(estimate, key)) for key in ('time', 'mean', 'covariance', 'frame_id',
            'units', 'innovation', 'innovation_covariance', 'residual', 'nis', 'state_id', 'observability_assessment_id')}
        result['replay_snapshot'] = estimate.replay_snapshot
        result['component_operations'] = ['geometric-state-inference.predict.v1', 'geometric-state-inference.update.v1']
    elif role == 'cbsr':
        from cbsr import reconcile_affine_exact
        result = reconcile_affine_exact(d)
    elif role == 'fdir':
        from fdir import evaluate_residual, assess_isolability
        diagnostics = evaluate_residual(d['residual'], d['residual_covariance'], threshold=d['detection_threshold'],
            variable_order=d['variable_order'], source_ids=d['source_ids'])
        isolation = assess_isolability(diagnostics, d['fault_signatures'],
            cross_covariance_policy=d['cross_covariance_policy'], max_unexplained_nis=d['max_unexplained_nis'])
        result = {'detection': native(diagnostics), 'isolability': native(isolation),
                  'cbsr_status': d['cbsr_status'], 'residual_basis': 'retained_gsie_prior_innovation'}
    else:
        raise ValueError('unsupported fixed adapter')
print(json.dumps(result, allow_nan=False, ensure_ascii=False))
'''


def _adapters(repositories, expected=None):
    if set(repositories) != ROLES:
        raise ValueError("Bind exactly the eight required repositories")
    pins = json.loads(resources.files("ciw").joinpath("calibrated-observable-runtimes.json").read_text())
    adapters = {}
    for role in sorted(ROLES):
        pin = pins[role]
        retained = expected[role] if expected else None
        adapter = PinnedSubprocessAdapter(repositories[role], pin["revision"], pin["module"],
            source_root=pin["source_root"],
            expected_python_sha256=retained["python_sha256"] if retained else None,
            expected_python_version=retained["python_version"] if retained else None,
            expected_dependencies=retained["dependencies"] if retained else None)
        identity = adapter.runtime_identity()
        if retained and any(identity[key] != retained[key] for key in (
            "schema", "adapter_version", "revision", "source_tree", "module", "source_root",
            "python_sha256", "python_version", "dependencies",
        )):
            raise ValueError("Calibrated-observable runtime identity mismatch")
        adapters[role] = adapter
    return adapters


def _invoke(role, adapters, request):
    encoded = canonical(request)
    if len(encoded) > MAX_BYTES:
        raise ValueError("Calibrated operation exceeds byte budget")
    adapter = adapters[role]
    adapter.runtime_identity()
    code, raw = adapter._run(_BOOTSTRAP, [role, str(adapter.source_root)], encoded)
    if code:
        raise AdapterRefusal("CALIBRATED_" + role.upper() + "_REFUSED", "Pinned " + role + " operation refused declared inputs")
    adapter.runtime_identity()
    result = _json(raw)
    if not isinstance(result, dict):
        raise ValueError("Calibrated operation result must be an object")
    return result


def _source(raw):
    try:
        return _source_inner(raw)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed calibrated experiment") from exc


def _source_inner(raw):
    if not isinstance(raw, bytes) or len(raw) > MAX_BYTES:
        raise ValueError("Experiment must be bounded exact bytes")
    value = _json(raw)
    if not isinstance(value, dict) or value.get("schema") != "fsrt.calibrated-observable-two-channel.v1":
        raise ValueError("Unsupported calibrated experiment")
    if len(value.get("channels", [])) != 2:
        raise ValueError("Exactly two channels are required")
    _exact_timestamp(value["epoch_utc"], "epoch_utc", require_utc=True)
    for channel in value["channels"]:
        for key in ("valid_from", "valid_until"):
            _exact_timestamp(channel["calibration_profile"][key], "calibration " + key)
    configuration = value["configuration"]
    gsie = configuration["gsie"]
    oit = configuration["observability"]
    if gsie.get("prior_measurement_crosscov_policy") != "declared_zero":
        raise ValueError("Prior-measurement independence must be declared")
    if (oit["transition"] != gsie["dynamics"]["matrix"] or
        oit["observation"] != gsie["observation_model"]["matrix"] or
        oit["model_id"] != gsie["observation_model"]["model_id"]):
        raise ValueError("Observability must assess the exact estimator model")
    if (gsie["dynamics"]["matrix"] != [[1, 0], [0, 1]] or
        gsie["dynamics"]["process_covariance"] != [[0, 0], [0, 0]]):
        raise ValueError("This bounded timing policy requires a stationary hold model")
    if configuration["alignment"].get("measurement_time_policy") != "nominal_alignment_with_retained_time_uncertainty":
        raise ValueError("Timing uncertainty policy must be explicit")
    _numeric_fields(value)
    canonical(value)
    return value


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _numbers(values, name, count=None):
    if not isinstance(values, list) or (count is not None and len(values) != count):
        raise ValueError(f"{name} must be a list of {count if count is not None else 'finite'} numbers")
    for value in values:
        _finite(value, name + " entry")


def _matrix(rows, name, shape=None):
    if not isinstance(rows, list) or not rows or (shape is not None and len(rows) != shape[0]):
        raise ValueError(f"{name} must be a numeric matrix")
    width = len(rows[0]) if isinstance(rows[0], list) else None
    for row in rows:
        _numbers(row, name + " row", shape[1] if shape is not None else width)


def _numeric_fields(value):
    """Every quantity the pinned providers will read must already be a finite number at retention."""
    for channel in value["channels"]:
        observation, clock, profile = channel["observation"], channel["clock_model"], channel["calibration_profile"]
        for key in ("device_time", "indicated_value", "raw_value"):
            _finite(observation[key], "observation." + key)
        for key in ("device_origin", "reference_origin", "skew", "offset"):
            _finite(clock[key], "clock_model." + key)
        _numbers(clock["valid_device_interval"], "clock_model.valid_device_interval", 2)
        _matrix(channel["clock_joint_covariance"], "clock_joint_covariance", (3, 3))
        for key in ("gain", "offset"):
            _finite(profile[key], "calibration_profile." + key)
        _numbers(profile["input_range"], "calibration_profile.input_range", 2)
        _matrix(profile["coefficient_covariance"], "calibration_profile.coefficient_covariance", (2, 2))
        _matrix(channel["calibration_joint_covariance"]["values"], "calibration_joint_covariance.values", (3, 3))
    _matrix(value["calibrated_covariance"]["matrix"], "calibrated_covariance.matrix", (2, 2))
    configuration = value["configuration"]
    _finite(configuration["alignment"]["maximum_nominal_separation_s"], "alignment.maximum_nominal_separation_s")
    observability = configuration["observability"]
    _matrix(observability["transition"], "observability.transition", (2, 2))
    _matrix(observability["observation"], "observability.observation", (2, 2))
    _numbers(observability["state_scales"], "observability.state_scales", 2)
    _finite(observability["condition_limit"], "observability.condition_limit")
    if type(observability["horizon"]) is not int or observability["horizon"] < 1:
        raise ValueError("observability.horizon must be a positive integer")
    gsie = configuration["gsie"]
    _finite(gsie["prior"]["time"], "gsie.prior.time")
    _numbers(gsie["prior"]["mean"], "gsie.prior.mean", 2)
    _matrix(gsie["prior"]["covariance"], "gsie.prior.covariance", (2, 2))
    _matrix(gsie["dynamics"]["matrix"], "gsie.dynamics.matrix", (2, 2))
    _matrix(gsie["dynamics"]["process_covariance"], "gsie.dynamics.process_covariance", (2, 2))
    _matrix(gsie["observation_model"]["matrix"], "gsie.observation_model.matrix", (2, 2))
    constraints = configuration["cbsr"]["constraints"]
    _matrix(constraints["coefficients"], "cbsr.constraints.coefficients")
    _numbers(constraints["rhs"], "cbsr.constraints.rhs", len(constraints["coefficients"]))
    _finite(configuration["cbsr"]["max_normalized_residual"], "cbsr.max_normalized_residual")
    fdir = configuration["fdir"]
    for key in ("detection_threshold", "max_unexplained_nis"):
        _finite(fdir[key], "fdir." + key)
    for name, signature in fdir["fault_signatures"].items():
        _numbers(signature, "fdir.fault_signatures." + str(name))


def _data(step):
    return step["result"]["data"]


def _request(index, experiment, steps, execution_id, evidence_ref):
    role, operation = OPERATIONS[index]
    c = experiment["configuration"]
    if role == "fsrt":
        inputs = {"experiment": deepcopy(experiment), "execution_id": execution_id}
        refs = [evidence_ref]
    elif role == "tbrt":
        inputs = {"channels": experiment["channels"], "alignment": c["alignment"]}
        refs = [steps[0]["result_id"]]
    elif role == "mcur":
        inputs = {"channels": experiment["channels"], "epoch_utc": experiment["epoch_utc"],
                  "alignment": _data(steps[1]), "calibrated_covariance": experiment["calibrated_covariance"]}
        refs = [steps[i]["result_id"] for i in (0, 1)]
    elif role == "oit":
        inputs = {"declaration": c["observability"]}
        refs = [steps[i]["result_id"] for i in (0, 2)]
    elif role == "gsie":
        calibrated, assessed = _data(steps[2]), _data(steps[3])
        inputs = {"declaration": c["gsie"], "time": _data(steps[1])["target_time"],
                  "values": calibrated["values"], "covariance": calibrated["covariance"]["matrix"],
                  "units": calibrated["units"], "observation_id": steps[2]["numerical_result_id"],
                  "evidence_refs": [steps[i]["numerical_result_id"] for i in (1, 2)],
                  "observability": {key: assessed[key] for key in
                      ("model_id", "status", "rank", "state_dimension", "condition_number", "condition_limit", "observation_matrix")}}
        inputs["observability"].update(assessment_id=steps[3]["numerical_result_id"],
            evidence_refs=[steps[3]["numerical_result_id"]],
            transition_matrix=assessed["declaration"]["transition"],
            dynamics_model_id=c["gsie"]["dynamics"]["model_id"])
        refs = [steps[i]["result_id"] for i in (1, 2, 3)]
    elif role == "cbsr":
        candidate = _data(steps[4])
        inputs = deepcopy(c["cbsr"])
        if inputs["state_units"] != candidate["units"] or inputs["frame_ref"] != candidate["frame_id"]:
            raise ValueError("Reconciliation coordinates differ from retained estimate")
        inputs.update(estimate=candidate["mean"], covariance=candidate["covariance"],
            state_id=candidate["state_id"], execution_id=execution_id, evidence_refs=[steps[4]["result_id"]],
            source_result_id=steps[4]["result_id"], source_result_digest=digest(steps[4]["result"]))
        refs = [steps[4]["result_id"]]
    else:
        candidate = _data(steps[4])
        inputs = deepcopy(c["fdir"])
        inputs.update(residual=candidate["innovation"], residual_covariance=candidate["innovation_covariance"],
            source_ids=[steps[i]["result_id"] for i in (4, 5)], cbsr_status=_data(steps[5])["status"])
        refs = list(inputs["source_ids"])
    return {"schema": "ciw.adapter-request.v1", "operation_id": operation, "inputs": deepcopy(inputs)}, refs


def _numerical(role, result):
    data = deepcopy(result if role == "fsrt" else result["data"])
    if role == "fsrt":
        for key in ("schema", "result_id", "execution_id", "operation_id"):
            data.pop(key, None)
    elif role == "cbsr":
        for key in ("schema", "execution_id", "input_state_id", "output_state_id", "constraint_id", "request_digest",
                    "request", "evidence_refs", "result_id", "source_binding_verification"):
            data.pop(key, None)
    elif role == "fdir":
        for key in ("detection", "isolability"):
            data[key].pop("source_ids", None)
    return {"operation_id": result["operation_id"], "data": data}


def _execute(raw, adapters, template=None):
    experiment = _source(raw)
    evidence_ref = byte_digest(raw)
    steps = []
    for index, (role, operation) in enumerate(OPERATIONS):
        execution_id = template["steps"][index]["execution_id"] if template else "execution-" + uuid.uuid4().hex
        request, refs = _request(index, experiment, steps, execution_id, evidence_ref)
        output = _invoke(role, adapters, request)
        if role == "fsrt":
            result = output
        else:
            result = {"schema": RESULT_SCHEMA, "operation_id": operation,
                "execution_ref": execution_id, "input_refs": refs, "data": output}
            result["result_id"] = digest(result)
        numerical = _numerical(role, result)
        steps.append({"runtime_ref": role, "operation_id": operation, "execution_id": execution_id,
            "input_refs": refs, "request": request, "request_sha256": digest(request),
            "result": result, "result_sha256": digest(result), "result_id": result["result_id"],
            "numerical_result": numerical, "numerical_result_id": digest(numerical)})
    bundle = {"schema": SCHEMA, "session_id": "session-" + uuid.uuid4().hex,
        "created_at": template["created_at"] if template else _now(),
        "source": {"experiment_id": experiment["experiment_id"], "experiment_digest": digest(experiment),
            "evidence": [{"artifact_ref": evidence_ref, "sha256": evidence_ref,
                          "bytes_b64": base64.b64encode(raw).decode("ascii")}]},
        "configuration": deepcopy(experiment["configuration"]),
        "runtimes": {role: adapter.runtime_identity() for role, adapter in adapters.items()}, "steps": steps}
    bundle["bundle_digest"] = _bundle_digest(bundle)
    return bundle


def _validate(bundle):
    try:
        if len(canonical(bundle)) > MAX_BYTES or bundle["schema"] != SCHEMA or bundle["bundle_digest"] != _bundle_digest(bundle):
            raise ValueError("Calibrated session content binding mismatch")
        evidence, = bundle["source"]["evidence"]
        raw = base64.b64decode(evidence["bytes_b64"], validate=True)
        if evidence["artifact_ref"] != byte_digest(raw) or evidence["sha256"] != byte_digest(raw):
            raise ValueError("Experiment evidence byte binding mismatch")
        experiment = _source(raw)
        if bundle["source"]["experiment_id"] != experiment["experiment_id"] or bundle["source"]["experiment_digest"] != digest(experiment):
            raise ValueError("Experiment identity binding mismatch")
        if canonical(bundle["configuration"]) != canonical(experiment["configuration"]):
            raise ValueError("Configuration differs from retained experiment")
        steps = bundle["steps"]
        if [(s["runtime_ref"], s["operation_id"]) for s in steps] != list(OPERATIONS) or set(bundle["runtimes"]) != ROLES:
            raise ValueError("Calibrated operation order/runtime mismatch")
        pins = json.loads(resources.files("ciw").joinpath("calibrated-observable-runtimes.json").read_text())
        for role, runtime in bundle["runtimes"].items():
            if runtime.get("schema") != "ciw.subprocess-runtime.v1" or any(
                runtime.get(key) != pins[role][key] for key in ("revision", "module", "source_root")
            ):
                raise ValueError("Retained runtime differs from approved source pin")
            if (not re.fullmatch(r"[0-9a-f]{40}", runtime.get("source_tree", "")) or
                not re.fullmatch(r"[0-9a-f]{64}", runtime.get("python_sha256", "")) or
                not isinstance(runtime.get("dependencies"), dict) or not runtime.get("python_version")):
                raise ValueError("Retained runtime identity is incomplete")
        identities = {bundle["session_id"], evidence["artifact_ref"], *(op for _, op in OPERATIONS)}
        for index, step in enumerate(steps):
            request, refs = _request(index, experiment, steps[:index], step["execution_id"], evidence["artifact_ref"])
            if canonical(step["request"]) != canonical(request) or step["input_refs"] != refs:
                raise ValueError("Request differs from exact retained operation graph")
            result = step["result"]
            for key, content in (("request_sha256", step["request"]), ("result_sha256", result),
                                 ("numerical_result_id", step["numerical_result"])):
                if step[key] != digest(content):
                    raise ValueError("Calibrated step content binding mismatch")
            if canonical(step["numerical_result"]) != canonical(_numerical(step["runtime_ref"], result)):
                raise ValueError("Calibrated numerical projection mismatch")
            if result["operation_id"] != step["operation_id"] or step["result_id"] != result["result_id"]:
                raise ValueError("Calibrated operation/result identity mismatch")
            if index:
                if (result.get("schema") != RESULT_SCHEMA or result.get("execution_ref") != step["execution_id"] or
                    result.get("input_refs") != refs or result["result_id"] != digest({k:v for k,v in result.items() if k != "result_id"})):
                    raise ValueError("Calibrated result content identity mismatch")
            else:
                from hashlib import sha256
                expected = "sha256:" + sha256(result["schema"].encode()+b"\0"+canonical({k:v for k,v in result.items() if k != "result_id"})).hexdigest()
                experiment_hash = "sha256:" + sha256(experiment["schema"].encode()+b"\0"+canonical(experiment)).hexdigest()
                if (result.get("schema") != "fsrt.calibrated-observable-declaration.v1" or
                    result["result_id"] != expected or result["execution_id"] != step["execution_id"] or
                    result.get("experiment_id") != experiment["experiment_id"] or
                    result.get("experiment_digest") != experiment_hash or
                    result.get("channel_order") != [channel["channel_id"] for channel in experiment["channels"]] or
                    result.get("claim_scope") != experiment["claim_scope"]):
                    raise ValueError("FSRT declaration identity mismatch")
            if step["execution_id"] in identities or step["result_id"] in identities or step["execution_id"] == step["result_id"]:
                raise ValueError("Occurrence, evidence and result identities must differ")
            identities.update((step["execution_id"], step["result_id"]))
        if "verification" in bundle:
            if bundle["verification"]["subject_ref"] != bundle["bundle_digest"]:
                raise ValueError("Verification subject mismatch")
            _identity(bundle["verification"], "verification_id")
        return raw
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed calibrated-observable session") from exc


def inspect_session(bundle):
    _validate(bundle)
    return {"schema": "ciw.calibrated-observable-inspection.v1", "session_id": bundle["session_id"],
        "bundle_digest": bundle["bundle_digest"], "status": "content_consistent", "numerical_replay": "not_performed",
        "operation_ids": [operation for _, operation in OPERATIONS], "admission": "not_performed"}


def _compare(original, reproduced):
    if any(any(old[key] != new[key] for key in ("request_sha256", "result_sha256", "numerical_result_id"))
           for old, new in zip(original["steps"], reproduced["steps"])):
        raise ValueError("Retained result differs from exact pinned recomputation")


def _verify(bundle, reproduced, adapters):
    results = {old["execution_id"]: new["numerical_result"] for old,new in zip(bundle["steps"], reproduced["steps"])}
    verification = _invoke("set", adapters, {"bundle": bundle, "replay_results": results, "created_at": _now()})
    if verification.get("outcome") != "passed":
        raise ValueError("SET did not verify calibrated-observable numerical replay")
    return verification, results


def create_session(source_bytes, repositories):
    _source(source_bytes)
    adapters = _adapters(repositories)
    bundle = _execute(source_bytes, adapters)
    _validate(bundle)
    reproduced = _execute(source_bytes, adapters, bundle)
    _compare(bundle, reproduced)
    bundle["verification"], _ = _verify(bundle, reproduced, adapters)
    return bundle


def replay_session(bundle, repositories):
    raw = _validate(bundle)
    adapters = _adapters(repositories, bundle["runtimes"])
    reproduced = _execute(raw, adapters, bundle)
    _compare(bundle, reproduced)
    fresh = _execute(raw, adapters)
    verification, results = _verify(bundle, fresh, adapters)
    fresh["verification"], _ = _verify(fresh, reproduced, adapters)
    receipt = {"schema": "ciw.calibrated-observable-replay.v1", "source_bundle_digest": bundle["bundle_digest"],
        "replayed_bundle_digest": fresh["bundle_digest"], "numerical_match": True,
        "verification": verification, "admission": "not_performed"}
    receipt["replay_id"] = digest(receipt)
    fresh["replay_receipts"] = [receipt]
    return {"session": fresh, "replay_receipt": receipt, "replay_results": results}


def read_session(path):
    return _json(_read(Path(path), MAX_BYTES))


def save_session(bundle, output_dir):
    _validate(bundle)
    path = Path(output_dir) / "calibrated-observable-session.json"
    if path.exists():
        raise ValueError("Refusing to overwrite an existing calibrated session")
    return write_json(path, bundle)
