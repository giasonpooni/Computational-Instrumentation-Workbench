"""Native residual monitoring over retained scalar calibrated windows.

FDIR consumes the original GSIE innovation and its covariance. A separate OIT
assessment gates interpretation; neither provider executes an estimator. CUSUM
is a declared recurrence with unknown temporal dependence, not a false-alarm
probability or a diagnosis of physical sensor drift.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import math
import re
import uuid

from . import calibrated_window
from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import PinnedSubprocessAdapter, _json
from .declared_workload import (AUTHORITY, DeclaredWorkflow, RESULT_SCHEMA,
                               SOURCE_LIMIT, _text, _verification)
from .telemetry import canonical, digest, byte_digest, _bundle_digest, _keys, _now, _instant

KIND = "residual-monitor"
SOURCE_SCHEMA = "ciw.residual-monitor-source.v1"
SCHEMA = "ciw.residual-monitor-session.v1"
OPERATION = "ciw.residual-monitor.v1"
MAX_BYTES = 4 * 1024 * 1024
ROLES = frozenset({"fdir", "oit"})
PINS = {
    "fdir": {"revision": "29e4b306492b793487d47a96b97a56548217aa12", "module": "fdir.diagnostics", "source_root": "src"},
    "oit": {"revision": "db4c564bddbe1911f96585bd18f58659a0026fb7", "module": "oit.diagnostics", "source_root": "src"},
}
SOURCE_TREES = {"fdir": "c7dc21c64377368341a2029ecd4f3e2a70d31646",
                "oit": "3ba42834535b396a13395e5053c6fd5b4f1302c9"}
POLICY = {
    "stream": "scalar_marginal_normalized_innovation",
    "temporal_covariance_policy": "unknown",
    "cross_covariance_policy": "unknown",
    "prior_feedback": "not_performed",
    "window_order": "strictly_increasing_declared_target_time",
    "false_alarm_probability": "not_established",
    "physical_drift": "not_established",
    "observability_hold": "does_not_advance_cusum",
}
DEFAULT_CONFIGURATION = {
    **POLICY,
    "detection_threshold": 9.0,
    "cusum": {"drift": 0.5, "threshold": 3.0, "direction": "two_sided", "reset_on_alarm": False},
    "observability": {"horizon": 1, "rank_rtol": 1e-12, "rank_atol": 0.0,
                      "weak_rtol": 1e-6, "condition_limit": 1e8},
    "fault_signatures": {"sensor.bias": [1.0], "process.bias": [1.0], "calibration.bias": [1.0]},
    "max_unexplained_nis": 1.0,
}


def _finite(value, *, minimum=0, strict=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= minimum and not (strict and value == minimum)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("Require a finite declared numerical threshold")


def _source(raw):
    if not isinstance(raw, bytes) or len(raw) > SOURCE_LIMIT:
        raise ValueError("Residual monitor source exceeds byte budget")
    source = _json(raw)
    canonical(source)
    _keys(source, {"schema", "experiment_id", "window_bundle_ids", "configuration"})
    if source["schema"] != SOURCE_SCHEMA:
        raise ValueError("Unsupported residual monitor source")
    _text(source["experiment_id"])
    ids = source["window_bundle_ids"]
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 16
            or any(not isinstance(v, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", v) for v in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Select 1..16 distinct retained window bundle identities in declared order")
    config = source["configuration"]
    _keys(config, set(DEFAULT_CONFIGURATION))
    if any(config[k] != v for k, v in POLICY.items()):
        raise ValueError("Residual monitoring requires explicit unknown temporal dependence and limited authority")
    _finite(config["detection_threshold"], strict=True)
    _finite(config["max_unexplained_nis"])
    if canonical(config["fault_signatures"]) != canonical(DEFAULT_CONFIGURATION["fault_signatures"]):
        raise ValueError("Retain competing scalar sensor, process and calibration bias signatures")
    cusum = config["cusum"]
    _keys(cusum, {"drift", "threshold", "direction", "reset_on_alarm"})
    _finite(cusum["drift"])
    _finite(cusum["threshold"], strict=True)
    if cusum["direction"] not in {"positive", "negative", "two_sided"} or type(cusum["reset_on_alarm"]) is not bool:
        raise ValueError("Declare the CUSUM direction and reset policy")
    observability = config["observability"]
    _keys(observability, set(DEFAULT_CONFIGURATION["observability"]))
    if any(type(observability[k]) is bool or observability[k] != v for k, v in DEFAULT_CONFIGURATION["observability"].items() if k != "condition_limit"):
        raise ValueError("The scalar stationary model has a fixed one-step observability policy")
    if observability["condition_limit"] is not None:
        _finite(observability["condition_limit"], minimum=1)
    return source


def requested_upstream_ids(raw):
    return list(_source(raw)["window_bundle_ids"])


def _unwrap(bundle):
    from .acquired_window import _check_child_receipt
    if bundle.get("schema") == calibrated_window.SCHEMA:
        raw = calibrated_window._validate(bundle)
        child = bundle
    elif bundle.get("schema") == "ciw.acquired-calibrated-window-session.v1":
        from . import acquired_window
        acquired_window._validate(bundle)
        child = bundle["child_window"]
        raw = calibrated_window._validate(child)
    else:
        raise ValueError("Residual monitoring requires retained calibrated window results")
    _check_child_receipt(child)
    return child, calibrated_window._source(raw)


def _request(source, upstreams):
    ids = source["window_bundle_ids"]
    windows, scope, overlaps, previous = [], None, [], None
    seen_results, seen_windows, seen_sample_sets, seen_observations = set(), set(), set(), {}
    for bundle_id in ids:
        if bundle_id not in upstreams:
            raise ValueError("Requested window is absent from the retained catalog")
        upstream = upstreams[bundle_id]
        if upstream["bundle_digest"] != bundle_id:
            raise ValueError("Selected window identity differs from retained content")
        child, declaration = _unwrap(upstream)
        gsie = child["steps"][3]
        artifact = gsie["result"]["result_artifact"]
        diagnostics = artifact["diagnostics"]
        model = declaration["configuration"]["gsie"]
        window = declaration["configuration"]["window"]
        target = model["target_time"]
        if previous is not None and target <= previous:
            raise ValueError("Window targets must increase strictly; replay is not another observation")
        previous = target
        current_scope = {key: deepcopy(declaration[key]) for key in
                         ("epoch", "channel_id", "frame", "frame_mapping", "device_clock", "receipt_clock", "clock_model", "calibration_profile")}
        current_scope.update(model={key: deepcopy(value) for key, value in model.items() if key != "target_time"},
                             measurement_variables=deepcopy(diagnostics["measurement_variables"]),
                             measurement_units=deepcopy(diagnostics["measurement_units"]),
                             measurement_frame=deepcopy(diagnostics["measurement_frame"]))
        if scope is None:
            scope = current_scope
        elif canonical(scope) != canonical(current_scope):
            raise ValueError("One residual sequence requires the same channel, frame, units, clock, calibration, model and reference prior")
        residual, covariance = diagnostics["innovation"], diagnostics["innovation_covariance"]
        if (not isinstance(residual, list) or len(residual) != 1 or type(residual[0]) not in (int, float)
                or not math.isfinite(residual[0]) or len(diagnostics["measurement_variables"]) != 1
                or not isinstance(covariance, list) or len(covariance) != 1 or not isinstance(covariance[0], list) or len(covariance[0]) != 1):
            raise ValueError("Monitor consumes the unchanged scalar GSIE innovation and covariance")
        _finite(covariance[0][0], strict=True)
        samples = declaration["samples"]
        observation_ids = [sample["observation_id"] for sample in samples]
        for sample in samples:
            old = seen_observations.setdefault(sample["observation_id"], canonical(sample))
            if old != canonical(sample):
                raise ValueError("A repeated observation identity cannot change its raw content")
        window_identity = digest({"scope": scope, "samples": samples, "window": window})
        sample_set = tuple(sorted(observation_ids))
        if gsie["result_id"] in seen_results or window_identity in seen_windows or sample_set in seen_sample_sets:
            raise ValueError("Duplicate GSIE result or replayed physical window cannot count twice")
        seen_results.add(gsie["result_id"])
        seen_windows.add(window_identity)
        seen_sample_sets.add(sample_set)
        binding = {"bundle_id": bundle_id, "child_bundle_id": child["bundle_digest"],
                   "gsie_result_id": gsie["result_id"], "gsie_execution_id": gsie["execution_id"],
                   "gsie_numerical_result_id": gsie["numerical_result_id"],
                   "window_evidence_id": child["source"]["evidence"][0]["artifact_ref"],
                   "source_observation_ids": observation_ids,
                   "source_artifact_ids": [sample["artifact_id"] for sample in samples],
                   "epoch": declaration["epoch"], "target_time": target,
                   "target_at": _instant(declaration["epoch"], target),
                   "device_times": [sample["device_time"] for sample in samples],
                   "mapped_event_times": [sample["event_time"] for sample in child["steps"][0]["result"]["data"]["samples"]],
                   "window_interval": [window["start"], window["end"]]}
        for prior in windows:
            old = prior["binding"]
            overlaps.append({"left_bundle_id": old["bundle_id"], "right_bundle_id": bundle_id,
                             "interval_overlap": max(old["window_interval"][0], window["start"]) < min(old["window_interval"][1], window["end"]),
                             "shared_observation_ids": sorted(set(old["source_observation_ids"]) & set(observation_ids)),
                             "shared_artifact_ids": sorted(set(old["source_artifact_ids"]) & set(binding["source_artifact_ids"])),
                             "shared_calibration_profile": declaration["calibration_profile"]["profile_id"],
                             "shared_reference_prior": model["prior"]["state_id"],
                             "residual_cross_covariance": "unknown"})
        oit = dict(deepcopy(source["configuration"]["observability"]),
                   transition=deepcopy(model["dynamics"]["matrix"]), observation=deepcopy(model["observation_model"]["matrix"]),
                   state_names=deepcopy(model["variables"]))
        windows.append({"binding": binding, "residual": deepcopy(residual), "innovation_covariance": deepcopy(covariance),
                        "variable_order": deepcopy(diagnostics["measurement_variables"]),
                        "source_ids": [gsie["result_id"]], "observability_declaration": oit})
    return {"source": deepcopy(source), "windows": windows, "scope": scope, "overlaps": overlaps}


def validate_upstreams(bundle, upstreams):
    source = _source(base64.b64decode(bundle["source"]["evidence"][0]["bytes_b64"], validate=True))
    ids = source["window_bundle_ids"]
    if not isinstance(upstreams, dict) or any(key not in upstreams for key in ids):
        raise ValueError("Monitor dependencies are absent from the retained catalog")
    selected = {key: upstreams[key] for key in ids}
    if canonical(bundle["upstream_windows"]) != canonical(selected):
        raise ValueError("Monitor dependencies differ from the selected retained windows")
    request = _request(source, selected)
    if canonical(bundle["steps"][0]["request"]) != canonical(request):
        raise ValueError("Residual request differs from retained native GSIE innovations")
    return request


_BOOTSTRAP = r'''
import dataclasses, json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
from fdir import CusumState, evaluate_residual, assess_isolability, cusum_step
from oit import lti_observability
request = json.loads(sys.stdin.buffer.read())
config = request['source']['configuration']
def native(value):
    if dataclasses.is_dataclass(value): return native(dataclasses.asdict(value))
    if isinstance(value, dict): return {k: native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [native(v) for v in value]
    if isinstance(value, np.ndarray): return native(value.tolist())
    if isinstance(value, np.generic): return value.item()
    return value
rows, state = [], CusumState()
for window in request['windows']:
    observable = lti_observability(**window['observability_declaration'])
    detection = evaluate_residual(window['residual'], window['innovation_covariance'],
        threshold=config['detection_threshold'], variable_order=window['variable_order'], source_ids=window['source_ids'])
    isolation = assess_isolability(detection, config['fault_signatures'],
        cross_covariance_policy='unknown', max_unexplained_nis=config['max_unexplained_nis'])
    before = state
    recurrence = None
    if observable.status == 'observable':
        recurrence = cusum_step(detection.marginal_normalized_residual[0], state,
            source_ids=window['source_ids'], **config['cusum'])
        state = recurrence.next_state
    rows.append({'binding': window['binding'], 'observability': native(observable),
        'detection': native(detection), 'isolability': native(isolation), 'cusum': native(recurrence),
        'monitor_state_before': native(before), 'monitor_state_after': native(state),
        'interpretation': {'observability_gate': 'passed' if observable.status == 'observable' else 'held',
            'diagnostic_drift_candidate': recurrence is not None and recurrence.status == 'statistical_anomaly',
            'alarm_authority': 'held_unknown_temporal_dependence', 'physical_drift': 'not_established',
            'unique_sensor_fault': None}})
print(json.dumps({'schema':'ciw.residual-monitor-data.v1', 'scope':request['scope'],
    'policy':config, 'overlaps':request['overlaps'], 'rows':rows, 'final_state':native(state)}, allow_nan=False))
'''


def _assert_numerical(actual, expected):
    """Tight scalar analytic oracle; every nonnumerical binding is exact."""
    if type(expected) is int:
        if type(actual) is not int or actual != expected:
            raise ValueError("Retained native ranks and counts require exact integer values")
    elif type(expected) is float:
        if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=2e-12, abs_tol=1e-14):
            raise ValueError("Retained native residual/CUSUM/observability arithmetic mismatch")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("Retained native diagnostic fields mismatch")
        for key in expected:
            _assert_numerical(actual[key], expected[key])
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError("Retained native diagnostic shape mismatch")
        for a, e in zip(actual, expected):
            _assert_numerical(a, e)
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError("Retained native diagnostic semantics mismatch")


def _oracle(request):
    config, rows = request["source"]["configuration"], []
    state = {"positive": 0.0, "negative": 0.0, "sample_count": 0}
    for window in request["windows"]:
        r, variance = window["residual"][0], window["innovation_covariance"][0][0]
        normalized = r / math.sqrt(variance)
        nis = normalized * normalized
        if not math.isfinite(nis):
            raise ValueError("Scalar residual diagnostic exceeds binary64")
        anomalous = nis >= config["detection_threshold"]
        status = "statistical_anomaly" if anomalous else "nominal"
        detection = {"raw_residual": window["residual"], "innovation_covariance": window["innovation_covariance"],
                     "variable_order": window["variable_order"], "source_ids": window["source_ids"],
                     "marginal_normalized_residual": [normalized], "whitened_residual": [normalized],
                     "nis": nis, "threshold": float(config["detection_threshold"]), "status": status}
        isolation = {"detection_status": status, "status": "ambiguous" if anomalous else "not_detected",
                     "cross_covariance_policy": "unknown", "max_unexplained_nis": float(config["max_unexplained_nis"]),
                     "candidates": list(config["fault_signatures"]), "isolated_fault": None,
                     "fits": [{"fault_id": key, "fitted_amplitude": float(r), "unexplained_nis": 0.0, "compatible": True} for key in config["fault_signatures"]],
                     "source_ids": window["source_ids"], "reason": "cross-covariance is unknown; unique fault nomination is refused" if anomalous else "the declared residual detector did not cross its threshold"}
        cusum = config["cusum"]
        observed = {"positive": max(0.0, state["positive"] + normalized - cusum["drift"]) if cusum["direction"] != "negative" else 0.0,
                    "negative": max(0.0, state["negative"] - normalized - cusum["drift"]) if cusum["direction"] != "positive" else 0.0,
                    "sample_count": state["sample_count"] + 1}
        sides = [name for name in ("positive", "negative") if observed[name] >= cusum["threshold"]]
        next_state = {"positive": 0.0, "negative": 0.0, "sample_count": observed["sample_count"]} if sides and cusum["reset_on_alarm"] else deepcopy(observed)
        recurrence = {"raw_value": normalized, "source_ids": window["source_ids"], **cusum,
                      "prior_state": deepcopy(state), "observed_state": observed, "next_state": next_state,
                      "alarm_sides": sides, "status": "statistical_anomaly" if sides else "nominal"}
        recurrence.update(drift=float(cusum["drift"]), threshold=float(cusum["threshold"]))
        oit = config["observability"]
        passed = oit["condition_limit"] is not None
        before = deepcopy(state)
        if passed:
            state = next_state
        else:
            recurrence = None
        observable = {"state_names": window["observability_declaration"]["state_names"], "state_scales": None,
                      "coordinate_mode": "raw model coordinates and units", "horizon": 1,
                      "observability_matrix": [[1.0]], "analyzed_matrix": [[1.0]],
                      "diagnostics": {"rank": 1, "input_dimension": 1, "full_column_rank": True,
                          "singular_values": [1.0], "rank_rtol": float(oit["rank_rtol"]), "rank_atol": float(oit["rank_atol"]),
                          "rank_threshold": float(oit["rank_rtol"]), "weak_rtol": float(oit["weak_rtol"]), "condition_number": 1.0,
                          "retained_condition_number": 1.0, "numerical_nullspace": [[]], "weak_directions": [[]]},
                      "status": "observable" if passed else "unresolved", "condition_limit": float(oit["condition_limit"]) if passed else None,
                      "classification_reason": "full rank and conditioning satisfy the declared finite-horizon policy" if passed else "full rank was found, but no conditioning acceptance limit was declared"}
        rows.append({"binding": window["binding"], "observability": observable, "detection": detection,
                     "isolability": isolation, "cusum": recurrence,
                     "monitor_state_before": before, "monitor_state_after": deepcopy(state),
                     "interpretation": {"observability_gate": "passed" if passed else "held",
                         "diagnostic_drift_candidate": bool(passed and sides),
                         "alarm_authority": "held_unknown_temporal_dependence", "physical_drift": "not_established", "unique_sensor_fault": None}})
    return {"schema": "ciw.residual-monitor-data.v1", "scope": request["scope"], "policy": config,
            "overlaps": request["overlaps"], "rows": rows, "final_state": state}


def _check_data(request, data):
    _assert_numerical(data, _oracle(request))
    for key, expected in (("scope", request["scope"]), ("policy", request["source"]["configuration"]), ("overlaps", request["overlaps"])):
        if canonical(data[key]) != canonical(expected):
            raise ValueError("Residual monitor changed a retained scientific declaration")
    for window, row in zip(request["windows"], data["rows"]):
        if canonical(row["binding"]) != canonical(window["binding"]):
            raise ValueError("Residual monitor changed retained window lineage")
        for key, expected in (("raw_residual", window["residual"]), ("innovation_covariance", window["innovation_covariance"]),
                              ("source_ids", window["source_ids"]), ("variable_order", window["variable_order"])):
            if canonical(row["detection"][key]) != canonical(expected):
                raise ValueError("FDIR must retain the exact GSIE residual, covariance and coordinate order")


class ResidualMonitorWorkflow(DeclaredWorkflow):
    MAX_BYTES, ROLES, SOURCE_SCHEMA = MAX_BYTES, ROLES, SOURCE_SCHEMA

    def __init__(self):
        self.kind, self.role, self.pin = KIND, "fdir", PINS["fdir"]
        self.schema, self.operation = SCHEMA, OPERATION

    def _source(self, raw): return _source(raw)
    def requested_upstream_ids(self, raw): return requested_upstream_ids(raw)
    def validate_upstreams(self, bundle, upstreams): return validate_upstreams(bundle, upstreams)

    @staticmethod
    def _runtime_projection(runtime):
        value = DeclaredWorkflow._runtime_projection(runtime)
        value["companions"] = {"oit": DeclaredWorkflow._runtime_projection(runtime["companions"]["oit"])}
        return value

    def _adapters(self, repositories, expected=None):
        if set(repositories) != ROLES:
            raise ValueError("Bind exactly the native FDIR and OIT repositories")
        old = expected["fdir"] if expected else None
        adapters, runtimes = {}, {}
        for role, pin in PINS.items():
            retained = (old if role == "fdir" else old["companions"]["oit"]) if old else {}
            adapters[role] = PinnedSubprocessAdapter(repositories[role], pin["revision"], pin["module"], source_root=pin["source_root"],
                expected_python_sha256=retained.get("python_sha256"), expected_python_version=retained.get("python_version"), expected_dependencies=retained.get("dependencies"))
            runtimes[role] = adapters[role].runtime_identity()
        runtime = {**runtimes["fdir"], "companions": {"oit": runtimes["oit"]}}
        if old and self._runtime_projection(runtime) != self._runtime_projection(old):
            raise ValueError("Residual monitor runtime differs from retained pins")
        return adapters, runtime, None

    def _step(self, request, evidence, bound):
        adapters, runtime, _ = bound
        for role in ROLES:
            retained = runtime if role == "fdir" else runtime["companions"]["oit"]
            current = adapters[role].runtime_identity()
            if DeclaredWorkflow._runtime_projection(current) != DeclaredWorkflow._runtime_projection({k: v for k, v in retained.items() if k != "companions"}):
                raise ValueError("Native monitor provider changed before execution")
        code, raw = adapters["fdir"]._run(_BOOTSTRAP, [str(adapters["fdir"].source_root), str(adapters["oit"].source_root)], canonical(request))
        for adapter in adapters.values(): adapter.runtime_identity()
        if code:
            raise AdapterRefusal("RESIDUAL_MONITOR_REFUSED", "Pinned FDIR/OIT refused retained residual monitoring")
        data = _json(raw)
        _check_data(request, data)
        refs = [evidence, *[window["source_ids"][0] for window in request["windows"]]]
        occurrence = "execution-" + uuid.uuid4().hex
        result = {"schema": RESULT_SCHEMA, "operation_id": OPERATION, "execution_ref": occurrence,
                  "input_refs": refs, "data": data, "authority": AUTHORITY}
        result["result_id"] = digest(result)
        numerical = {"operation_id": OPERATION, "data": data}
        return {"runtime_ref": "fdir", "operation_id": OPERATION, "execution_id": occurrence, "input_refs": refs,
                "request": request, "request_sha256": digest(request), "result": result, "result_sha256": digest(result),
                "result_id": result["result_id"], "numerical_result": numerical, "numerical_result_id": digest(numerical)}

    def _validate_step(self, step, source, evidence):
        _keys(step, {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256", "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"})
        request = step["request"]
        _keys(request, {"source", "windows", "scope", "overlaps"})
        if canonical(request["source"]) != canonical(source):
            raise ValueError("Monitor step source mismatch")
        refs = [evidence, *[window["source_ids"][0] for window in request["windows"]]]
        if (step["runtime_ref"] != "fdir" or step["operation_id"] != OPERATION or step["input_refs"] != refs
                or not isinstance(step["execution_id"], str) or not re.fullmatch(r"execution-[a-f0-9]{32}", step["execution_id"])):
            raise ValueError("Residual execution and upstream result binding mismatch")
        result = step["result"]
        _keys(result, {"schema", "operation_id", "execution_ref", "input_refs", "data", "authority", "result_id"})
        _check_data(request, result["data"])
        if (result != {"schema": RESULT_SCHEMA, "operation_id": OPERATION, "execution_ref": step["execution_id"],
                "input_refs": refs, "data": result["data"], "authority": AUTHORITY,
                "result_id": digest({k: v for k, v in result.items() if k != "result_id"})}
                or result["result_id"] != step["result_id"]
                or canonical(step["numerical_result"]) != canonical({"operation_id": OPERATION, "data": result["data"]})):
            raise ValueError("Residual result content or authority mismatch")
        for key, content in (("request_sha256", request), ("result_sha256", result), ("numerical_result_id", step["numerical_result"])):
            if step[key] != digest(content): raise ValueError("Residual step content binding mismatch")

    def _check_verification(self, bundle, verification, source, evidence):
        super()._check_verification(bundle, verification, source, evidence)
        if canonical(verification["reproduction"]["request"]) != canonical(bundle["steps"][0]["request"]):
            raise ValueError("Residual reproduction must consume the same retained innovations")

    def _validate(self, bundle):
        try:
            _keys(bundle, {"schema", "session_id", "created_at", "source", "configuration", "runtimes", "steps", "bundle_digest", "verification", "upstream_windows"}, {"replay_receipts"})
            if len(canonical(bundle)) > MAX_BYTES or bundle["schema"] != SCHEMA or bundle["bundle_digest"] != _bundle_digest(bundle) or not re.fullmatch(r"session-[a-f0-9]{32}", bundle["session_id"]):
                raise ValueError("Residual monitor bundle identity mismatch")
            _text(bundle["created_at"])
            evidence, = bundle["source"]["evidence"]
            raw = base64.b64decode(evidence["bytes_b64"], validate=True)
            source = _source(raw)
            if (evidence != {"artifact_ref": byte_digest(raw), "sha256": byte_digest(raw), "bytes_b64": base64.b64encode(raw).decode()}
                    or bundle["source"] != {"experiment_id": source["experiment_id"], "experiment_digest": digest(source), "evidence": [evidence]}
                    or canonical(bundle["configuration"]) != canonical(source["configuration"])):
                raise ValueError("Residual source/configuration exact byte binding mismatch")
            if set(bundle["upstream_windows"]) != set(source["window_bundle_ids"]) or set(bundle["runtimes"]) != {"fdir"}:
                raise ValueError("Unexpected residual monitor dependencies or runtimes")
            runtime = bundle["runtimes"]["fdir"]
            if set(runtime["companions"]) != {"oit"}:
                raise ValueError("Unexpected monitor companion")
            for role, value in (("fdir", runtime), ("oit", runtime["companions"]["oit"])):
                _keys(value, {"schema", "adapter_version", "repository_root", "revision", "source_tree", "module", "source_root", "python_executable", "python_sha256", "python_version", "dependencies"} | ({"companions"} if role == "fdir" else set()))
                if (value["schema"] != "ciw.subprocess-runtime.v1" or any(value[k] != v for k, v in PINS[role].items())
                        or value["source_tree"] != SOURCE_TREES[role]
                        or not re.fullmatch(r"[a-f0-9]{64}", value["python_sha256"]) or not isinstance(value["dependencies"], dict)):
                    raise ValueError("Unapproved monitor runtime pin")
                for key in ("adapter_version", "repository_root", "python_executable", "python_version"): _text(value[key])
            step, = bundle["steps"]
            validate_upstreams(bundle, bundle["upstream_windows"])
            self._validate_step(step, source, evidence["artifact_ref"])
            self._check_verification(bundle, bundle["verification"], source, evidence["artifact_ref"])
            return raw
        except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
            raise ValueError("Malformed residual monitor bundle") from exc

    def _execute_selected(self, raw, upstreams, bound):
        source, evidence = _source(raw), byte_digest(raw)
        request = _request(source, upstreams)
        bundle = {"schema": SCHEMA, "session_id": "session-" + uuid.uuid4().hex, "created_at": _now(),
                  "source": {"experiment_id": source["experiment_id"], "experiment_digest": digest(source),
                      "evidence": [{"artifact_ref": evidence, "sha256": evidence, "bytes_b64": base64.b64encode(raw).decode()}]},
                  "configuration": deepcopy(source["configuration"]), "runtimes": {"fdir": bound[1]},
                  "upstream_windows": {key: deepcopy(upstreams[key]) for key in source["window_bundle_ids"]},
                  "steps": [self._step(request, evidence, bound)]}
        bundle["bundle_digest"] = _bundle_digest(bundle)
        bundle["verification"] = _verification(bundle, self._step(request, evidence, bound))
        self._validate(bundle)
        return bundle

    def create_session(self, raw, upstreams, repositories):
        source = _source(raw)
        _request(source, upstreams)
        return self._execute_selected(raw, upstreams, self._adapters(repositories))

    def replay_session(self, bundle, repositories):
        raw = self._validate(bundle)
        fresh = self._execute_selected(raw, bundle["upstream_windows"], self._adapters(repositories, bundle["runtimes"]))
        receipt = {"schema": "ciw.residual-monitor-replay.v1", "source_bundle_digest": bundle["bundle_digest"],
                   "replayed_bundle_digest": fresh["bundle_digest"], "numerical_match": True,
                   "verification": _verification(bundle, fresh["steps"][0]), "admission": "not_performed"}
        receipt["replay_id"] = digest(receipt)
        fresh["replay_receipts"] = [receipt]
        return {"session": fresh, "replay_receipt": receipt}


workflow = ResidualMonitorWorkflow()
_validate = workflow._validate
_adapters = workflow._adapters
create_session = workflow.create_session
replay_session = workflow.replay_session
