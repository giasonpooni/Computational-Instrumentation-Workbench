"""Offline analysis of an exact retained energy log; never a measurement runner.

A fresh execution or replay recomputes derived quantities from the same sealed
counter/output evidence. It never imports a CUDA/NVML worker, samples hardware,
or runs variational iterations. Historical capture provenance is not attested.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import platform
import re
import uuid

import numpy as np

from . import energy_records
from .adapters.subprocess import _json
from .declared_workload import DeclaredWorkflow, RESULT_SCHEMA, VERIFY_SCHEMA
from .exchange import _identity
from .telemetry import canonical, digest, byte_digest, _bundle_digest, _now, _keys

KIND = "energy-accuracy"
OPERATION = "ciw.energy-accuracy.v1"
SOURCE_LIMIT = 4 * 1024 * 1024
MAX_BYTES = 8 * 1024 * 1024
POLICY = {"operation": "offline_analysis_of_retained_energy_log", "physical_measurement": "not_performed",
          "replay": "fresh_analysis_of_same_retained_measurement", "hardware_provenance": "not_authenticated",
          "state_admission": "not_performed"}
AUTHORITY = {"state_admission": "not_performed", "sensor_fusion": "not_performed",
             "physical_measurement": "not_performed_by_analysis", "hardware_provenance": "not_authenticated"}
METHOD = "fresh_analysis_of_same_retained_measurement"


def _same(actual, expected):
    if canonical(actual) != canonical(expected):
        raise ValueError("Retained energy analysis binding differs")


def analysis_identity():
    files = ("energy_records.py", "free_energy_math.py", "energy_workflow.py")
    code = b"\0".join(name.encode() + b"\0" + Path(__file__).with_name(name).read_text(encoding="utf-8").encode()
                       for name in files)
    return {"schema": "ciw.energy-analysis-runtime.v1", "profile": "ciw.energy-accuracy-analysis.v1",
            "execution_scope": "cpu_offline_retained_log_analysis", "code_sha256": sha256(code).hexdigest(),
            "source_normalization": "utf8_lf", "python_version": platform.python_version(), "numpy_version": np.__version__}


def _check_runtime(runtime):
    _keys(runtime, {"schema", "profile", "execution_scope", "code_sha256", "source_normalization", "python_version", "numpy_version"})
    fixed = {"schema": "ciw.energy-analysis-runtime.v1", "profile": "ciw.energy-accuracy-analysis.v1",
             "execution_scope": "cpu_offline_retained_log_analysis", "source_normalization": "utf8_lf"}
    if any(runtime[key] != value for key, value in fixed.items()):
        raise ValueError("Unsupported retained energy analysis runtime")
    if type(runtime["code_sha256"]) is not str or not re.fullmatch(r"[a-f0-9]{64}", runtime["code_sha256"]):
        raise ValueError("Invalid retained analysis implementation identity")
    for key in ("python_version", "numpy_version"):
        if type(runtime[key]) is not str or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", runtime[key]):
            raise ValueError("Invalid retained analysis dependency identity")


def _request(source, evidence):
    return {"schema": "ciw.energy-analysis-request.v1", "evidence_id": evidence,
            "log_digest": source["log_digest"], "measurement_run_id": source["run_id"],
            "analysis_profile": "ciw.energy-accuracy-analysis.v1"}


def _verification(bundle, reproduction):
    _same(bundle["steps"][0]["numerical_result"], reproduction["numerical_result"])
    value = {"schema": VERIFY_SCHEMA, "subject_ref": bundle["bundle_digest"], "outcome": "passed",
             "independent": False, "method": METHOD, "runtime_digest": digest(bundle["runtimes"]),
             "reproduction": reproduction, "authority": deepcopy(AUTHORITY)}
    value["verification_id"] = byte_digest(VERIFY_SCHEMA.encode() + b"\0" + canonical(value))
    return value


class EnergyAccuracyWorkflow(DeclaredWorkflow):
    MAX_BYTES = MAX_BYTES

    def __init__(self):
        self.kind, self.role, self.ROLES = KIND, "energy", frozenset()
        self.SOURCE_SCHEMA = energy_records.SCHEMA
        self.schema, self.operation = "ciw.energy-accuracy-session.v1", OPERATION

    def _source(self, raw):
        if type(raw) is not bytes or not 1 <= len(raw) <= SOURCE_LIMIT:
            raise ValueError("Energy analysis source must contain 1..4194304 exact retained bytes")
        source = _json(raw)
        energy_records.validate_log(source)
        return source

    @staticmethod
    def _runtime_projection(runtime):
        return deepcopy(runtime)

    def _adapters(self, repositories, expected=None):
        if type(repositories) is not dict or repositories:
            raise ValueError("Offline energy analysis takes no repository or hardware bindings")
        runtime = analysis_identity()
        if expected is not None:
            _same(expected, {"energy": runtime})
        return None, runtime, None

    def _step(self, source, evidence, bound):
        _same(analysis_identity(), bound[1])
        data = energy_records.analyze(source)
        _same(analysis_identity(), bound[1])
        occurrence = "execution-" + uuid.uuid4().hex
        request = _request(source, evidence)
        result = {"schema": RESULT_SCHEMA, "operation_id": OPERATION, "execution_ref": occurrence,
                  "input_refs": [evidence], "data": data, "authority": deepcopy(AUTHORITY)}
        result["result_id"] = digest(result)
        numerical = {"operation_id": OPERATION, "data": deepcopy(data)}
        return {"runtime_ref": self.role, "operation_id": OPERATION, "execution_id": occurrence,
                "input_refs": [evidence], "request": request, "request_sha256": digest(request),
                "result": result, "result_sha256": digest(result), "result_id": result["result_id"],
                "numerical_result": numerical, "numerical_result_id": digest(numerical)}

    def _validate_step(self, step, source, evidence):
        _keys(step, {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256",
                     "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"})
        if (step["runtime_ref"] != self.role or step["operation_id"] != OPERATION or
                type(step["execution_id"]) is not str or not re.fullmatch(r"execution-[a-f0-9]{32}", step["execution_id"])):
            raise ValueError("Invalid energy analysis occurrence or operation")
        request = _request(source, evidence)
        _same(step["request"], request)
        _same(step["input_refs"], [evidence])
        result = step["result"]
        _keys(result, {"schema", "operation_id", "execution_ref", "input_refs", "data", "authority", "result_id"})
        _same(result["data"], energy_records.analyze(source))
        unsigned = {"schema": RESULT_SCHEMA, "operation_id": OPERATION, "execution_ref": step["execution_id"],
                    "input_refs": [evidence], "data": result["data"], "authority": AUTHORITY}
        _same(result, {**unsigned, "result_id": digest(unsigned)})
        if step["result_id"] != result["result_id"]:
            raise ValueError("Energy result identity differs")
        _same(step["numerical_result"], {"operation_id": OPERATION, "data": result["data"]})
        for key, value in (("request_sha256", request), ("result_sha256", result), ("numerical_result_id", step["numerical_result"])):
            if step[key] != digest(value):
                raise ValueError("Energy analysis commitment differs")

    def _check_verification(self, bundle, verification, source, evidence):
        _keys(verification, {"schema", "subject_ref", "outcome", "independent", "method", "runtime_digest",
                             "reproduction", "authority", "verification_id"})
        reproduction = verification["reproduction"]
        self._validate_step(reproduction, source, evidence)
        original = bundle["steps"][0]
        if original["execution_id"] == reproduction["execution_id"] or original["result_id"] == reproduction["result_id"]:
            raise ValueError("Energy replay requires a fresh analysis occurrence")
        _same(verification, _verification(bundle, reproduction))
        _identity(verification, "verification_id")

    def _validate(self, bundle):
        try:
            _keys(bundle, {"schema", "session_id", "created_at", "source", "configuration", "runtimes", "steps", "bundle_digest", "verification"}, {"replay_receipts"})
            if (len(canonical(bundle)) > MAX_BYTES or bundle["schema"] != self.schema or
                    bundle["bundle_digest"] != _bundle_digest(bundle) or
                    type(bundle["session_id"]) is not str or not re.fullmatch(r"session-[a-f0-9]{32}", bundle["session_id"])):
                raise ValueError("Energy analysis bundle identity, schema or size differs")
            if type(bundle["created_at"]) is not str or not 1 <= len(bundle["created_at"]) <= 128:
                raise ValueError("Missing bounded analysis creation metadata")
            evidence, = bundle["source"]["evidence"]
            raw = base64.b64decode(evidence["bytes_b64"], validate=True)
            source = self._source(raw)
            _same(evidence, {"artifact_ref": byte_digest(raw), "sha256": byte_digest(raw), "bytes_b64": base64.b64encode(raw).decode()})
            _same(bundle["source"], {"experiment_id": source["run_id"], "experiment_digest": digest(source), "evidence": [evidence]})
            _same(bundle["configuration"], POLICY)
            _keys(bundle["runtimes"], {"energy"})
            _check_runtime(bundle["runtimes"]["energy"])
            step, = bundle["steps"]
            self._validate_step(step, source, evidence["artifact_ref"])
            self._check_verification(bundle, bundle["verification"], source, evidence["artifact_ref"])
            receipts = bundle.get("replay_receipts", [])
            if type(receipts) is not list or len(receipts) > 1:
                raise ValueError("At most one replay receipt belongs to an energy analysis")
            for receipt in receipts:
                _keys(receipt, {"schema", "source_bundle_digest", "replayed_bundle_digest", "numerical_match", "verification", "admission", "replay_id"})
                if (receipt["schema"] != "ciw.energy-accuracy-replay.v1" or receipt["replayed_bundle_digest"] != bundle["bundle_digest"] or
                        receipt["source_bundle_digest"] == bundle["bundle_digest"] or type(receipt["source_bundle_digest"]) is not str or
                        not re.fullmatch(r"sha256:[a-f0-9]{64}", receipt["source_bundle_digest"]) or receipt["numerical_match"] is not True or
                        receipt["admission"] != "not_performed" or receipt["replay_id"] != digest({k:v for k,v in receipt.items() if k != "replay_id"})):
                    raise ValueError("Invalid retained energy replay receipt")
                verification = receipt["verification"]
                _keys(verification, {"schema", "subject_ref", "outcome", "independent", "method", "runtime_digest", "reproduction", "authority", "verification_id"})
                _same(verification["reproduction"], step)
                expected = {"schema": VERIFY_SCHEMA, "subject_ref": receipt["source_bundle_digest"], "outcome": "passed",
                            "independent": False, "method": METHOD, "runtime_digest": digest(bundle["runtimes"]),
                            "reproduction": step, "authority": AUTHORITY}
                expected["verification_id"] = byte_digest(VERIFY_SCHEMA.encode() + b"\0" + canonical(expected))
                _same(verification, expected)
            return raw
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError, UnicodeError) as exc:
            raise ValueError("Malformed retained energy analysis session") from exc

    def _execute(self, raw, bound):
        source = self._source(raw)
        evidence = byte_digest(raw)
        bundle = {"schema": self.schema, "session_id": "session-" + uuid.uuid4().hex, "created_at": _now(),
            "source": {"experiment_id": source["run_id"], "experiment_digest": digest(source),
                       "evidence": [{"artifact_ref": evidence, "sha256": evidence, "bytes_b64": base64.b64encode(raw).decode()}]},
            "configuration": deepcopy(POLICY), "runtimes": {"energy": bound[1]}, "steps": [self._step(source, evidence, bound)]}
        bundle["bundle_digest"] = _bundle_digest(bundle)
        bundle["verification"] = _verification(bundle, self._step(source, evidence, bound))
        self._validate(bundle)
        return bundle

    def replay_session(self, bundle, repositories):
        raw = self._validate(bundle)
        fresh = self._execute(raw, self._adapters(repositories, bundle["runtimes"]))
        receipt = {"schema": "ciw.energy-accuracy-replay.v1", "source_bundle_digest": bundle["bundle_digest"],
                   "replayed_bundle_digest": fresh["bundle_digest"], "numerical_match": True,
                   "verification": _verification(bundle, fresh["steps"][0]), "admission": "not_performed"}
        receipt["replay_id"] = digest(receipt)
        fresh["replay_receipts"] = [receipt]
        self._validate(fresh)
        return {"session": fresh, "replay_receipt": receipt}
