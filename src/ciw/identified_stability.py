"""Retained identified dynamics and predicted state → native PLSR verdict.

The selected discrete point model is never converted to continuous time. The
caller supplies a sealed quadratic certificate; state and parameter uncertainty
are retained as context and do not become a robust or physical stability claim.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
import uuid

import numpy as np

from . import identified_design, plsr_engine
from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import _json
from .declared_workload import (AUTHORITY, DeclaredWorkflow, MAX_BYTES, RESULT_SCHEMA,
                               SOURCE_LIMIT, _text, _verification)
from .core.canonical import canonical, digest, byte_digest, bundle_digest, exact_keys, utc_now

KIND = "identified-stability"
SOURCE_SCHEMA = "ciw.identified-stability-source.v1"
OPERATION = "ciw.identified-stability.v1"
ROLES = frozenset({"plsr"})
PIN = {"revision": "19ea6967060166ba09db6cd4563bd87bd6b3d196",
       "source_root": "src", "module": "lyapunov.model_artifact"}
SOURCE_TREE = "e261315f46851d99053e305ef2c03d4160f02a92"
POLICY = {
    "model_semantics": "discrete_linear_zero_input_zero_equilibrium",
    "state_selection": "retained_gsie_conditional_prediction_mean",
    "uncertainty_scope": "conditional_on_identified_point_model",
    "parameter_covariance": "unknown",
    "state_covariance": "retained_context_not_propagated_through_certificate",
    "certificate": "caller_supplied_fixed_quadratic",
    "coordinate_metric": "euclidean_in_declared_equal_units",
    "certificate_value_unit": "1",
    "equilibrium_scope": "declared_model_only_not_physical_equilibrium",
    "physical_stability": "not_established",
    "certificate_authentication": "not_performed",
    "state_freshness": "not_assessed",
    "proof_status": "NOT_CHECKED",
}
MATURITY = {"kernel_api": "changing", "numerical_validation": "passed",
            "physical_validation": "not_started", "deployment_authorization": "prohibited"}
SELECTION_FIELDS = {"upstream_bundle_id", "model_result_id", "model_execution_id",
                    "model_numerical_result_id", "state_result_id", "state_execution_id",
                    "state_numerical_result_id", "state_id"}


def _native_digest(value, omitted):
    return sha256(json.dumps({k: v for k, v in value.items() if k not in omitted},
                            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _number(value, limit=1e12):
    plsr_engine._number(value, "bounded stability coordinate")
    if abs(value) > limit:
        raise ValueError("Stability declaration exceeds its numeric bound")


def _matrix(value, count):
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("Stability matrix must match declared state order")
    for row in value:
        if not isinstance(row, list) or len(row) != count:
            raise ValueError("Stability matrix must match declared state order")
        for item in row:
            _number(item)


def _model(model):
    exact_keys(model, {"artifact_schema", "model_id", "model_version", "state", "time", "plant", "certificate",
                  "policy", "estimator", "provenance", "claim_scope", "may_authorize", "artifact_digest"})
    if (model["artifact_schema"] != "model-artifact-v1" or model["claim_scope"] != "computational-integrity-only"
            or model["may_authorize"] is not False or model["artifact_digest"] != _native_digest(model, {"artifact_digest"})):
        raise ValueError("Require a received sealed native computational model artifact")
    for key in ("model_id", "model_version"):
        _text(model[key])
    exact_keys(model["state"], {"definition", "coordinates"})
    _text(model["state"]["definition"])
    coordinates = model["state"]["coordinates"]
    if not isinstance(coordinates, list) or not 1 <= len(coordinates) <= 8:
        raise ValueError("Stability workload supports one to eight declared coordinates")
    for item in coordinates:
        exact_keys(item, {"name", "unit"})
        _text(item["name"])
        _text(item["unit"])
    if (len({c["name"] for c in coordinates}) != len(coordinates)
            or len({c["unit"] for c in coordinates}) != 1):
        raise ValueError("Declare unique state names in common units; no implicit coordinate rescaling")
    exact_keys(model["time"], {"convention", "sample_period_s"})
    _number(model["time"]["sample_period_s"])
    if model["time"]["convention"] != "discrete" or model["time"]["sample_period_s"] <= 0:
        raise ValueError("Retained discrete dynamics require an explicit positive sample period")
    exact_keys(model["plant"], {"kind", "A"})
    exact_keys(model["certificate"], {"kind", "P"})
    if model["plant"]["kind"] != "linear" or model["certificate"]["kind"] != "quadratic":
        raise ValueError("Require a declared discrete linear plant and fixed quadratic certificate")
    count = len(coordinates)
    for matrix in (model["plant"]["A"], model["certificate"]["P"]):
        _matrix(matrix, count)
    p = np.asarray(model["certificate"]["P"], dtype=float)
    if not np.array_equal(p, p.T) or np.linalg.eigvalsh(p)[0] <= 0:
        raise ValueError("Certificate must be exactly symmetric positive definite; no repair")
    exact = [[Fraction(x) for x in row] for row in model["certificate"]["P"]]
    for k in range(count):
        pivot = exact[k][k]
        if pivot <= 0:
            raise ValueError("Certificate is not positive definite in exact declared arithmetic")
        for i in range(k + 1, count):
            for j in range(k + 1, count):
                exact[i][j] -= exact[i][k] * exact[k][j] / pivot
    policy = model["policy"]
    exact_keys(policy, {"required_margin", "level", "margin_derivation", "numerical_policy", "runtime_status_schema", "claim_codes_schema"})
    _number(policy["required_margin"])
    if policy["required_margin"] < 0:
        raise ValueError("A supplied margin may only tighten the numerical test")
    if policy["level"] is not None:
        _number(policy["level"])
    if (policy["numerical_policy"] != "float64-decrease-v1" or policy["runtime_status_schema"] != "runtime-status-v1"
            or policy["claim_codes_schema"] != "claim-codes-v1"):
        raise ValueError("Unsupported native numerical policy")
    derivation = policy["margin_derivation"]
    exact_keys(derivation, {"method", "description", "evidence_digest", "quantity"})
    for key in ("method", "description"):
        _text(derivation[key])
    if derivation["quantity"] != "negative-largest-eigenvalue-of-decrease-matrix":
        raise ValueError("Margin must name its native decrease-matrix quantity")
    if derivation["evidence_digest"] is not None and not re.fullmatch(r"[a-f0-9]{64}", derivation["evidence_digest"]):
        raise ValueError("Malformed margin evidence digest")
    exact_keys(model["estimator"], {"identity", "version", "configuration_digest", "state_compatibility"})
    exact_keys(model["provenance"], {"producer", "producer_version", "model_data_digest", "construction_report_digest"})
    for group in (model["estimator"], model["provenance"]):
        for key, value in group.items():
            _text(value)
            if key.endswith("digest") and not re.fullmatch(r"[a-f0-9]{64}", value):
                raise ValueError("Malformed native model digest")


def _source(raw):
    try:
        if not isinstance(raw, bytes) or len(raw) > SOURCE_LIMIT:
            raise ValueError("Require bounded exact stability source bytes")
        source = _json(raw)
        exact_keys(source, {"schema", "experiment_id", "configuration", "selection", "equilibrium", "certificate_unit", "model_artifact"})
        if source["schema"] != SOURCE_SCHEMA or source["configuration"] != POLICY:
            raise ValueError("Require the explicit bounded stability policy")
        _text(source["experiment_id"])
        exact_keys(source["selection"], SELECTION_FIELDS)
        for key, value in source["selection"].items():
            _text(value)
            if key != "state_id" and not re.fullmatch(r"execution-[a-f0-9]{32}" if key.endswith("execution_id") else r"sha256:[a-f0-9]{64}", value):
                raise ValueError("Malformed selected native identity")
        _model(source["model_artifact"])
        coordinates = source["model_artifact"]["state"]["coordinates"]
        equilibrium = source["equilibrium"]
        exact_keys(equilibrium, {"coordinates", "units", "frame_id", "value"})
        _text(equilibrium["frame_id"])
        if (equilibrium["coordinates"] != [c["name"] for c in coordinates] or equilibrium["units"] != [c["unit"] for c in coordinates]
                or not isinstance(equilibrium["value"], list) or len(equilibrium["value"]) != len(coordinates)
                or any(type(v) not in (int, float) or v != 0 for v in equilibrium["value"])):
            raise ValueError("Declare the zero equilibrium in exact native coordinate order")
        unit = coordinates[0]["unit"]
        if source["certificate_unit"] != "1/(" + unit + "*" + unit + ")":
            raise ValueError("Declare certificate and margin units in the native common-unit basis")
        return source
    except (KeyError, TypeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed stability source") from exc


def _binding(upstream):
    identified_design._validate(upstream)
    model, state = upstream["steps"][0], upstream["steps"][2]
    ident = upstream["configuration"]["identification"]
    numerical, prediction = model["result"]["data"]["numerical_result"], state["result"]["data"]
    if numerical["status"] != "identified" or prediction["parameter_covariance_status"] != "unknown":
        raise ValueError("Stability requires an identified point model with retained unknown parameter covariance")
    return {"upstream_bundle_id": upstream["bundle_digest"], "model_result_id": model["result_id"],
            "model_execution_id": model["execution_id"], "model_numerical_result_id": model["numerical_result_id"],
            "state_result_id": state["result_id"], "state_execution_id": state["execution_id"],
            "state_numerical_result_id": state["numerical_result_id"], "state_id": prediction["state_id"],
            "state_names": ident["state_names"], "state_units": ident["state_units"], "frame_id": prediction["frame_id"],
            "clock_frame": ident["clock_frame"], "time": prediction["time"], "sample_interval": ident["sample_interval"],
            "A": numerical["candidate"]["A"], "mean": prediction["mean"], "covariance": prediction["covariance"],
            "parameter_covariance_status": "unknown", "prediction_request_digest": state["request_sha256"],
            "uncertainty_scope": prediction["uncertainty_scope"]}


def _selected(source, upstream):
    binding = _binding(upstream)
    model = source["model_artifact"]
    if source["selection"] != {k: binding[k] for k in SELECTION_FIELDS}:
        raise ValueError("Stability selection must bind the exact retained model and state occurrences")
    expected_coordinates = [{"name": name, "unit": unit} for name, unit in zip(binding["state_names"], binding["state_units"])]
    if (canonical(model["plant"]["A"]) != canonical(binding["A"])
            or model["state"]["coordinates"] != expected_coordinates
            or model["time"] != {"convention": "discrete", "sample_period_s": binding["sample_interval"]}
            or source["equilibrium"]["frame_id"] != binding["frame_id"]
            or model["estimator"]["identity"] != "gsie:retained-conditional-prediction"
            or model["estimator"]["configuration_digest"] != binding["prediction_request_digest"].removeprefix("sha256:")
            or model["provenance"]["model_data_digest"] != binding["model_numerical_result_id"].removeprefix("sha256:")):
        raise ValueError("PLSR model convention, coordinates, frame or provenance differs from retained inputs")
    if binding["uncertainty_scope"] != POLICY["uncertainty_scope"]:
        raise ValueError("Unknown identified model uncertainty cannot become a robust stability claim")
    for value in binding["mean"]:
        _number(value)
    return deepcopy(binding)


_BOOTSTRAP = r'''
import importlib.metadata, json, pathlib, sys
sys.path.insert(0, sys.argv[2])
sys.path.insert(0, sys.argv[1])
import lyapunov
from ciw import plsr_engine as bridge
if not pathlib.Path(lyapunov.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve()):
    raise ValueError('PLSR escaped the bound source tree')
request = json.loads(sys.stdin.buffer.read())
model = lyapunov.model_artifact_from_dict(request['model_artifact'])
sample = request['sample']
verdict = model.verdict(sample['x'], theta=None, theta_dot=None)
body = {'adapter_schema': bridge.EVALUATION_SCHEMA, 'model_artifact_schema': 'model-artifact-v1',
        'model_artifact_digest': model.artifact_digest, 'sample': sample,
        'runtime_status_schema': lyapunov.RUNTIME_STATUS_VERSION, 'code': verdict.code,
        'presentation_category': bridge._CATEGORIES[verdict.code],
        'inequality_certified': bool(verdict.inequality_certified),
        'meets_required_margin': bool(verdict.meets_required_margin),
        'operationally_acceptable': bool(verdict.operationally_acceptable),
        'required_margin': float(verdict.required_margin), 'level': model.level, 'details': verdict.details,
        'proof_status': 'NOT_CHECKED', 'diagnostics': bridge._diagnostics(verdict)}
record = lyapunov.as_companion_record(kind='ciw-plsr-evaluation', body=body)
print(json.dumps(record, allow_nan=False))
'''


def _record(source, binding, record):
    exact_keys(record, plsr_engine._RECORD_KEYS)
    model = source["model_artifact"]
    sample = {"sample_schema": "plsr-sample-v1", "x": binding["mean"], "theta": None, "theta_dot": None}
    expected = {"record_schema": "companion-record-v1", "record_kind": "ciw-plsr-evaluation",
                "claim_scope": "computational-integrity-only", "may_authorize": False,
                "confirmed_out_of_development": False, "maturity": MATURITY,
                "adapter_schema": plsr_engine.EVALUATION_SCHEMA, "model_artifact_schema": "model-artifact-v1",
                "model_artifact_digest": model["artifact_digest"], "sample": sample,
                "runtime_status_schema": "runtime-status-v1", "proof_status": "NOT_CHECKED",
                "required_margin": float(model["policy"]["required_margin"]), "level": model["policy"]["level"]}
    if (any(canonical(record[k]) != canonical(v) for k, v in expected.items())
            or record["record_digest"] != _native_digest(record, {"generated_at", "record_digest", "cargo_prove_available", "guest_manifest"})):
        raise ValueError("Native PLSR record authority, sample or content binding mismatch")
    _text(record["details"])
    code = record["code"]
    if (code not in plsr_engine._CATEGORIES or code in plsr_engine._SAMPLELESS
            or record["presentation_category"] != plsr_engine._CATEGORIES[code]):
        raise ValueError("Unsupported native status for the bounded discrete workload")
    for key in ("inequality_certified", "meets_required_margin", "operationally_acceptable"):
        if type(record[key]) is not bool:
            raise ValueError("Native PLSR outcome flags must remain booleans")
    if record["operationally_acceptable"] != (code == "CERTIFIED_WITH_MARGIN"):
        raise ValueError("Native status and operational flag disagree")
    proxy = SimpleNamespace(to_dict=lambda: model, required_margin=float(model["policy"]["required_margin"]), level=model["policy"]["level"])
    plsr_engine._validate_diagnostics(proxy, sample, record)
    d = record["diagnostics"]
    a, p = np.asarray(model["plant"]["A"], dtype=float), np.asarray(model["certificate"]["P"], dtype=float)
    form = a.T @ p @ a - p
    form = 0.5 * form + 0.5 * form.T
    if d["A"] != a.tolist() or d["P"] != p.tolist() or d["decrease_matrix"] != form.tolist():
        raise ValueError("Native decrease form must equal A.T P A - P in the retained basis")
    state = np.asarray(sample["x"]) / d["state_scale"]
    n, u = len(state), np.finfo(float).eps / 2
    gamma = (n + 3) * u / (1 - (n + 3) * u)
    resolution = n * (2 * gamma * n * n * float(np.max(np.abs(a))) ** 2 * float(np.max(np.abs(p))) + u * float(np.max(np.abs(p)))) + n**3 * u * float(np.max(np.abs(form)))
    numbers = {"scaled_value": float(state @ p @ state), "scaled_decrease": float(state @ form @ state),
               "min_P": float(np.linalg.eigvalsh(p)[0]), "max_decrease": float(np.linalg.eigvalsh(form)[-1]), "resolution": resolution}
    for key, value in numbers.items():
        if d[key] is None or not math.isclose(d[key], value, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError("Native PLSR diagnostic contradicts the declared quadratic form: " + key)
    if code not in {"CERTIFICATE_NOT_POSITIVE", "OUTSIDE_LEVEL_SET"}:
        if d["scaled_decrease"] > d["resolution"] * float(state @ state): expected_code = "NOT_CERTIFIED"
        elif d["margin"] > d["resolution"]: expected_code = "CERTIFIED_WITH_MARGIN" if d["margin"] > proxy.required_margin else "MARGIN_LOW"
        elif d["max_decrease"] > d["resolution"]: expected_code = "DECREASE_NOT_DEFINITE"
        else: expected_code = "NUMERICAL_INCONCLUSIVE"
        if code != expected_code:
            raise ValueError("Native PLSR status differs from its retained numerical diagnostics")


def validate_upstream(bundle, upstream):
    if canonical(bundle["upstream_design"]) != canonical(upstream):
        raise ValueError("Stability must retain the exact selected design bundle")
    source = _source(base64.b64decode(bundle["source"]["evidence"][0]["bytes_b64"], validate=True))
    binding = _selected(source, upstream)
    if canonical(bundle["upstream_binding"]) != canonical(binding):
        raise ValueError("Retained stability input binding differs")
    return binding


class IdentifiedStabilityWorkflow(DeclaredWorkflow):
    ROLES, SOURCE_SCHEMA = ROLES, SOURCE_SCHEMA

    def __init__(self):
        self.kind, self.role, self.pin = KIND, "plsr", PIN
        self.schema, self.operation = "ciw.identified-stability-session.v1", OPERATION

    def _source(self, raw): return _source(raw)
    def validate_upstream(self, bundle, upstream): return validate_upstream(bundle, upstream)

    def _adapters(self, repositories, expected=None):
        # The standard adapter records NumPy/SciPy. ModelArtifact additionally
        # depends on jsonschema, so bind that dependency explicitly as well.
        expected_base = None if expected is None else {"plsr": {k: v for k, v in expected["plsr"].items() if k != "model_schema_dependency"}}
        adapter, runtime, _ = super()._adapters(repositories, expected_base)
        code, raw = adapter._run("import importlib.metadata,json;print(json.dumps(importlib.metadata.version('jsonschema')))", [], b"")
        if code: raise AdapterRefusal("STABILITY_SCHEMA_UNAVAILABLE", "The pinned native model loader requires jsonschema")
        runtime["model_schema_dependency"] = {"name": "jsonschema", "version": _json(raw)}
        if runtime["source_tree"] != SOURCE_TREE or expected and self._runtime_projection(runtime) != self._runtime_projection(expected["plsr"]):
            raise ValueError("Native stability runtime source or schema dependency differs")
        return adapter, runtime, None

    def _step(self, source, evidence, bound, binding):
        adapter, runtime, _ = bound
        adapter.runtime_identity()
        sample = {"sample_schema": "plsr-sample-v1", "x": binding["mean"], "theta": None, "theta_dot": None}
        code, raw = adapter._run(_BOOTSTRAP, [str(adapter.source_root), str(Path(__file__).resolve().parent.parent)],
                                 canonical({"model_artifact": source["model_artifact"], "sample": sample}))
        adapter.runtime_identity()
        if code: raise AdapterRefusal("IDENTIFIED_STABILITY_REFUSED", "Pinned PLSR refused the declared model and retained state")
        record = _json(raw)
        _record(source, binding, record)
        data = {"binding": deepcopy(binding), "model_artifact": deepcopy(source["model_artifact"]),
                "sample": deepcopy(sample), "record": record, "policy": deepcopy(POLICY)}
        execution = "execution-" + uuid.uuid4().hex
        refs = [evidence, binding["model_result_id"], binding["state_result_id"]]
        result = {"schema": RESULT_SCHEMA, "operation_id": OPERATION, "execution_ref": execution, "input_refs": refs, "data": data, "authority": AUTHORITY}
        result["result_id"] = digest(result)
        numerical = {"operation_id": OPERATION, "data": data}
        return {"runtime_ref": "plsr", "operation_id": OPERATION, "execution_id": execution, "input_refs": refs,
                "request": source, "request_sha256": digest(source), "result": result, "result_sha256": digest(result),
                "result_id": result["result_id"], "numerical_result": numerical, "numerical_result_id": digest(numerical)}

    def _validate_step(self, step, source, evidence):
        exact_keys(step, {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256", "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"})
        selection = source["selection"]
        refs = [evidence, selection["model_result_id"], selection["state_result_id"]]
        if (step["runtime_ref"] != "plsr" or step["operation_id"] != OPERATION or step["input_refs"] != refs
                or canonical(step["request"]) != canonical(source) or not re.fullmatch(r"execution-[a-f0-9]{32}", step["execution_id"])):
            raise ValueError("Stability request or occurrence binding mismatch")
        result = step["result"]
        exact_keys(result, {"schema", "operation_id", "execution_ref", "input_refs", "data", "authority", "result_id"})
        data = result["data"]
        exact_keys(data, {"binding", "model_artifact", "sample", "record", "policy"})
        if (data["policy"] != POLICY or canonical(data["model_artifact"]) != canonical(source["model_artifact"])
                or data["sample"] != {"sample_schema": "plsr-sample-v1", "x": data["binding"]["mean"], "theta": None, "theta_dot": None}):
            raise ValueError("Stability result must retain the exact model, sample and uncertainty scope")
        _record(source, data["binding"], data["record"])
        if (result["schema"] != RESULT_SCHEMA or result["operation_id"] != OPERATION or result["execution_ref"] != step["execution_id"]
                or result["input_refs"] != refs or result["authority"] != AUTHORITY
                or result["result_id"] != digest({k: v for k, v in result.items() if k != "result_id"}) or result["result_id"] != step["result_id"]
                or step["numerical_result"] != {"operation_id": OPERATION, "data": data}):
            raise ValueError("Stability result identity or authority mismatch")
        for key, value in (("request_sha256", source), ("result_sha256", result), ("numerical_result_id", step["numerical_result"])):
            if step[key] != digest(value): raise ValueError("Stability content digest mismatch")

    def _validate(self, bundle):
        try:
            exact_keys(bundle, {"schema", "session_id", "created_at", "source", "configuration", "runtimes", "steps", "bundle_digest", "verification", "upstream_design", "upstream_binding"}, {"replay_receipts"})
            if len(canonical(bundle)) > MAX_BYTES or bundle["schema"] != self.schema or bundle["bundle_digest"] != bundle_digest(bundle) or not re.fullmatch(r"session-[a-f0-9]{32}", bundle["session_id"]):
                raise ValueError("Stability bundle content binding mismatch")
            _text(bundle["created_at"])
            evidence, = bundle["source"]["evidence"]
            raw = base64.b64decode(evidence["bytes_b64"], validate=True)
            source = _source(raw)
            if (evidence != {"artifact_ref": byte_digest(raw), "sha256": byte_digest(raw), "bytes_b64": base64.b64encode(raw).decode()}
                    or bundle["source"] != {"experiment_id": source["experiment_id"], "experiment_digest": digest(source), "evidence": [evidence]}
                    or bundle["configuration"] != POLICY or set(bundle["runtimes"]) != {"plsr"}):
                raise ValueError("Stability source/configuration binding mismatch")
            binding = validate_upstream(bundle, bundle["upstream_design"])
            runtime = bundle["runtimes"]["plsr"]
            exact_keys(runtime, {"schema", "adapter_version", "repository_root", "revision", "source_tree", "module", "source_root", "python_executable", "python_sha256", "python_version", "dependencies", "model_schema_dependency"})
            if (runtime["schema"] != "ciw.subprocess-runtime.v1" or any(runtime[k] != PIN[k] for k in PIN)
                    or runtime["source_tree"] != SOURCE_TREE or not re.fullmatch(r"[a-f0-9]{64}", runtime["python_sha256"])
                    or not isinstance(runtime["dependencies"], dict)):
                raise ValueError("Unapproved native stability runtime pin")
            for key in ("adapter_version", "repository_root", "python_executable", "python_version"): _text(runtime[key])
            exact_keys(runtime["model_schema_dependency"], {"name", "version"})
            if runtime["model_schema_dependency"]["name"] != "jsonschema": raise ValueError("Missing native schema dependency")
            _text(runtime["model_schema_dependency"]["version"])
            step, = bundle["steps"]
            for retained in (step, bundle["verification"]["reproduction"]):
                if canonical(retained["result"]["data"]["binding"]) != canonical(binding):
                    raise ValueError("Native stability result refers to another retained model or state")
            self._validate_step(step, source, evidence["artifact_ref"])
            self._check_verification(bundle, bundle["verification"], source, evidence["artifact_ref"])
            if "replay_receipts" in bundle:
                receipt, = bundle["replay_receipts"]
                exact_keys(receipt, {"schema", "source_bundle_digest", "replayed_bundle_digest", "numerical_match", "verification", "admission", "replay_id"})
                if (receipt["schema"] != "ciw.identified-stability-replay.v1" or receipt["replayed_bundle_digest"] != bundle["bundle_digest"]
                        or receipt["source_bundle_digest"] == bundle["bundle_digest"] or receipt["numerical_match"] is not True
                        or receipt["admission"] != "not_performed" or receipt["replay_id"] != digest({k:v for k,v in receipt.items() if k != "replay_id"})
                        or receipt["verification"]["subject_ref"] != receipt["source_bundle_digest"]):
                    raise ValueError("Stability replay receipt binding mismatch")
                proof = receipt["verification"]
                exact_keys(proof, {"schema", "subject_ref", "outcome", "independent", "method", "runtime_digest", "reproduction", "authority", "verification_id"})
                # A receipt names the earlier subject, but its native fresh
                # reproduction is exactly the retained primary of this replay.
                expected_proof = _verification(bundle, step)
                expected_proof["subject_ref"] = receipt["source_bundle_digest"]
                expected_proof["verification_id"] = byte_digest(expected_proof["schema"].encode() + b"\0" + canonical({k:v for k,v in expected_proof.items() if k != "verification_id"}))
                if canonical(proof) != canonical(expected_proof):
                    raise ValueError("Stability replay proof must bind this exact fresh native occurrence")
            return raw
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
            raise ValueError("Malformed stability session") from exc

    def _execute_selected(self, raw, upstream, bound):
        source = _source(raw)
        binding = _selected(source, upstream)
        evidence = byte_digest(raw)
        step = self._step(source, evidence, bound, binding)
        bundle = {"schema": self.schema, "session_id": "session-" + uuid.uuid4().hex, "created_at": utc_now(),
                  "source": {"experiment_id": source["experiment_id"], "experiment_digest": digest(source),
                             "evidence": [{"artifact_ref": evidence, "sha256": evidence, "bytes_b64": base64.b64encode(raw).decode()}]},
                  "configuration": deepcopy(POLICY), "runtimes": {"plsr": bound[1]}, "steps": [step],
                  "upstream_design": deepcopy(upstream), "upstream_binding": binding}
        bundle["bundle_digest"] = bundle_digest(bundle)
        bundle["verification"] = _verification(bundle, self._step(source, evidence, bound, binding))
        self._validate(bundle)
        return bundle

    def create_session(self, raw, upstream, repositories):
        _selected(_source(raw), upstream)
        return self._execute_selected(raw, upstream, self._adapters(repositories))

    def replay_session(self, bundle, repositories):
        raw = self._validate(bundle)
        fresh = self._execute_selected(raw, bundle["upstream_design"], self._adapters(repositories, bundle["runtimes"]))
        receipt = {"schema": "ciw.identified-stability-replay.v1", "source_bundle_digest": bundle["bundle_digest"],
                   "replayed_bundle_digest": fresh["bundle_digest"], "numerical_match": True,
                   "verification": _verification(bundle, fresh["steps"][0]), "admission": "not_performed"}
        receipt["replay_id"] = digest(receipt)
        fresh["replay_receipts"] = [receipt]
        self._check_verification(bundle, receipt["verification"], _source(raw), byte_digest(raw))
        self._validate(fresh)
        return {"session": fresh, "replay_receipt": receipt}


workflow = IdentifiedStabilityWorkflow()
create_session = workflow.create_session
replay_session = workflow.replay_session
_validate = workflow._validate
