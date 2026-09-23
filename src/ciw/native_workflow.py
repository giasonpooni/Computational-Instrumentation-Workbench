"""Shared lifecycle for provider-free native reference operations.

A subclass declares one exact source contract and one deterministic evaluation.
The retained envelope matches the machine-manifest and thermal-observer
workflows: exact source bytes, one execution occurrence, a fresh reproduction by
the same reference, and at most one replay receipt. The base never retrieves
documents, opens devices, admits state or authorizes equipment.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import re
import uuid

import numpy as np

from .adapters.subprocess import _json
from .exchange import _identity
from .telemetry import _bundle_digest, _now, byte_digest, canonical, digest, _keys

REFERENCE_RUNTIME_SCHEMA = "ciw.python-reference-runtime.v1"
REPRODUCTION_METHOD = "same_python_reference_fresh_occurrence_reproduction"
_STEP_KEYS = {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256",
              "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"}
_VERIFY_KEYS = {"schema", "subject_ref", "outcome", "independent", "method", "runtime_digest",
                "reproduction", "authority", "verification_id"}
_RECEIPT_KEYS = {"schema", "source_bundle_digest", "replayed_bundle_digest", "numerical_match",
                 "verification", "admission", "replay_id"}


def text(value, limit=512):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("Require bounded nonempty text")


def parse_source(raw, limit, name):
    """Strict JSON bytes: bounded, no duplicate keys and no nonfinite numbers."""
    if type(raw) is not bytes or not 1 <= len(raw) <= limit:
        raise ValueError(name + " source requires bounded exact JSON bytes")
    return _json(raw)


def code_digest(files):
    content = b"\0".join(
        Path(path).name.encode("utf-8") + b"\0" +
        Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        for path in files
    )
    return sha256(content).hexdigest()


class NativeReferenceWorkflow:
    """Template for one deterministic provider-free CIW operation.

    Subclasses set the class constants and implement ``validate_source`` and
    ``evaluate``. ``evaluate`` must not read clocks, randomness or host state;
    replay compares its numerical identity exactly.
    """

    kind = None
    schema = None
    operation = None
    role = None
    SOURCE_SCHEMA = None
    RESULT_SCHEMA = None
    VERIFY_SCHEMA = None
    PROFILE = None
    MAX_BYTES = 2 * 1024 * 1024
    SOURCE_LIMIT = 256 * 1024
    AUTHORITY = {"physical_validation": "not_performed", "state_admission": "not_performed",
                 "hardware_actuation": "not_performed"}
    ROLES = frozenset()
    FILES = ()
    LABEL = "native reference"

    # -- subclass contract -------------------------------------------------
    def validate_source(self, raw):
        raise NotImplementedError

    def evaluate(self, source):
        raise NotImplementedError

    # -- runtime identity --------------------------------------------------
    def algorithm_identity(self):
        return {"profile": self.PROFILE, "code_sha256": code_digest([*self.FILES, __file__]),
                "source_normalization": "utf8_lf", "numpy_version": np.__version__}

    def runtime_identity(self):
        return {"schema": REFERENCE_RUNTIME_SCHEMA, "role": self.role, "profile": self.PROFILE,
                "algorithm": self.algorithm_identity(),
                "execution_scope": "independent_python_reference_only", **deepcopy(self.AUTHORITY)}

    @staticmethod
    def _runtime_projection(value):
        return deepcopy(value)

    def _adapters(self, repositories, expected=None):
        if not isinstance(repositories, dict) or repositories:
            raise ValueError("The provider-free " + self.LABEL + " accepts no repository bindings")
        runtime = self.runtime_identity()
        if expected is not None:
            expected_runtime = expected.get(self.role, expected) if isinstance(expected, dict) else expected
            if self._runtime_projection(runtime) != self._runtime_projection(expected_runtime):
                raise ValueError(self.LABEL.capitalize() + " runtime identity differs from the retained execution")
        return None, runtime

    def _check_runtime(self, runtime):
        expected = self.runtime_identity()
        if runtime == expected:
            return
        # Retained identities stay strict in shape and scope; replay compares
        # the complete current identity, including the code commitment.
        _keys(runtime, set(expected))
        for key in set(expected) - {"algorithm"}:
            if runtime[key] != expected[key]:
                raise ValueError("Unapproved " + self.LABEL + " runtime")
        algorithm = runtime["algorithm"]
        _keys(algorithm, {"profile", "code_sha256", "source_normalization", "numpy_version"})
        if (algorithm["profile"] != self.PROFILE or algorithm["source_normalization"] != "utf8_lf" or
                not re.fullmatch(r"[a-f0-9]{64}", algorithm["code_sha256"]) or
                not isinstance(algorithm["numpy_version"], str)):
            raise ValueError("Malformed " + self.LABEL + " algorithm identity")

    # -- occurrence records ------------------------------------------------
    def _source(self, raw):
        return self.validate_source(raw)

    def _native_data(self, source):
        return self.evaluate(source)

    def _make_step(self, source, evidence_id, runtime, execution_id=None):
        occurrence = execution_id or "execution-" + uuid.uuid4().hex
        data = self._native_data(source)
        result = {"schema": self.RESULT_SCHEMA, "operation_id": self.operation, "execution_ref": occurrence,
                  "input_refs": [evidence_id], "data": data, "authority": deepcopy(self.AUTHORITY)}
        result["result_id"] = digest(result)
        numerical = {"operation_id": self.operation, "data": deepcopy(data)}
        return {"runtime_ref": self.role, "operation_id": self.operation, "execution_id": occurrence,
                "input_refs": [evidence_id], "request": deepcopy(source["request"]),
                "request_sha256": digest(source["request"]), "result": result, "result_sha256": digest(result),
                "result_id": result["result_id"], "numerical_result": numerical,
                "numerical_result_id": digest(numerical)}

    def _step(self, source, evidence_id, bound):
        _, runtime = bound
        return self._make_step(source, evidence_id, runtime)

    def _validate_step(self, step, source, evidence_id):
        _keys(step, _STEP_KEYS)
        if (step["runtime_ref"] != self.role or step["operation_id"] != self.operation or
                not isinstance(step["execution_id"], str) or
                not re.fullmatch(r"execution-[a-f0-9]{32}", step["execution_id"]) or
                step["input_refs"] != [evidence_id] or
                canonical(step["request"]) != canonical(source["request"]) or
                step["request_sha256"] != digest(source["request"])):
            raise ValueError(self.LABEL.capitalize() + " step request or occurrence binding differs")
        result = step["result"]
        _keys(result, {"schema", "operation_id", "execution_ref", "input_refs", "data", "authority", "result_id"})
        if (result["schema"] != self.RESULT_SCHEMA or result["operation_id"] != self.operation or
                result["execution_ref"] != step["execution_id"] or result["input_refs"] != [evidence_id] or
                result["authority"] != self.AUTHORITY or
                result["result_id"] != digest({k: v for k, v in result.items() if k != "result_id"})):
            raise ValueError(self.LABEL.capitalize() + " native result envelope differs")
        if canonical(result["data"]) != canonical(self._native_data(source)):
            raise ValueError(self.LABEL.capitalize() + " result differs from deterministic evaluation")
        numerical = {"operation_id": self.operation, "data": result["data"]}
        if step["numerical_result"] != numerical or step["numerical_result_id"] != digest(numerical):
            raise ValueError(self.LABEL.capitalize() + " numerical result identity differs")
        if step["result_id"] != result["result_id"] or step["result_sha256"] != digest(result):
            raise ValueError(self.LABEL.capitalize() + " step result commitment differs")

    def _verification(self, bundle, reproduced):
        if canonical(bundle["steps"][0]["numerical_result"]) != canonical(reproduced["numerical_result"]):
            raise ValueError(self.LABEL.capitalize() + " replay numerical result differs")
        value = {"schema": self.VERIFY_SCHEMA, "subject_ref": bundle["bundle_digest"], "outcome": "passed",
                 "independent": False, "method": REPRODUCTION_METHOD, "runtime_digest": digest(bundle["runtimes"]),
                 "reproduction": deepcopy(reproduced), "authority": deepcopy(self.AUTHORITY)}
        value["verification_id"] = byte_digest(self.VERIFY_SCHEMA.encode() + b"\0" + canonical(value))
        return value

    def _check_scope(self, verification, subject):
        _keys(verification, _VERIFY_KEYS)
        if (verification["schema"] != self.VERIFY_SCHEMA or verification["subject_ref"] != subject or
                verification["outcome"] != "passed" or verification["independent"] is not False or
                verification["method"] != REPRODUCTION_METHOD or verification["authority"] != self.AUTHORITY):
            raise ValueError(self.LABEL.capitalize() + " verification scope differs")

    def _check_verification(self, bundle, verification, source, evidence):
        self._check_scope(verification, bundle["bundle_digest"])
        self._validate_step(verification["reproduction"], source, evidence)
        old, new = bundle["steps"][0], verification["reproduction"]
        if (old["execution_id"] == new["execution_id"] or old["result_id"] == new["result_id"] or
                verification != self._verification(bundle, new)):
            raise ValueError(self.LABEL.capitalize() + " verification must bind a fresh occurrence")
        _identity(verification, "verification_id")

    # -- retained bundles --------------------------------------------------
    def _validate(self, bundle):
        try:
            _keys(bundle, {"schema", "session_id", "created_at", "source", "configuration", "runtimes",
                           "steps", "bundle_digest", "verification"}, {"replay_receipts"})
            if (len(canonical(bundle)) > self.MAX_BYTES or bundle["schema"] != self.schema or
                    bundle["bundle_digest"] != _bundle_digest(bundle) or
                    not isinstance(bundle["session_id"], str) or
                    not re.fullmatch(r"session-[a-f0-9]{32}", bundle["session_id"])):
                raise ValueError(self.LABEL.capitalize() + " bundle identity or size differs")
            text(bundle["created_at"])
            evidence, = bundle["source"]["evidence"]
            raw = base64.b64decode(evidence["bytes_b64"], validate=True)
            source = self._source(raw)
            expected_evidence = {"artifact_ref": byte_digest(raw), "sha256": byte_digest(raw),
                                 "bytes_b64": base64.b64encode(raw).decode()}
            if (evidence != expected_evidence or bundle["source"] != {
                    "experiment_id": source["experiment_id"], "experiment_digest": digest(source),
                    "evidence": [evidence]} or
                    canonical(bundle["configuration"]) != canonical(source["configuration"])):
                raise ValueError(self.LABEL.capitalize() + " source or configuration binding differs")
            _keys(bundle["runtimes"], {self.role})
            self._check_runtime(bundle["runtimes"][self.role])
            step, = bundle["steps"]
            self._validate_step(step, source, evidence["artifact_ref"])
            self._check_verification(bundle, bundle["verification"], source, evidence["artifact_ref"])
            receipts = bundle.get("replay_receipts", [])
            if not isinstance(receipts, list) or len(receipts) > 1:
                raise ValueError("At most one " + self.LABEL + " replay receipt belongs to an occurrence")
            for receipt in receipts:
                _keys(receipt, _RECEIPT_KEYS)
                if (receipt["schema"] != "ciw." + self.kind + "-replay.v1" or
                        receipt["replayed_bundle_digest"] != bundle["bundle_digest"] or
                        receipt["source_bundle_digest"] == bundle["bundle_digest"] or
                        receipt["numerical_match"] is not True or receipt["admission"] != "not_performed" or
                        receipt["replay_id"] != digest({k: v for k, v in receipt.items() if k != "replay_id"})):
                    raise ValueError("Invalid " + self.LABEL + " replay receipt")
                verification = receipt["verification"]
                self._check_scope(verification, receipt["source_bundle_digest"])
                self._validate_step(verification["reproduction"], source, evidence["artifact_ref"])
                _identity(verification, "verification_id")
            return raw
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
            raise ValueError("Malformed retained " + self.LABEL + " session") from exc

    def _execute(self, raw, runtime):
        source = self._source(raw)
        evidence = byte_digest(raw)
        step = self._make_step(source, evidence, runtime)
        bundle = {"schema": self.schema, "session_id": "session-" + uuid.uuid4().hex, "created_at": _now(),
                  "source": {"experiment_id": source["experiment_id"], "experiment_digest": digest(source),
                             "evidence": [{"artifact_ref": evidence, "sha256": evidence,
                                           "bytes_b64": base64.b64encode(raw).decode()}]},
                  "configuration": deepcopy(source["configuration"]),
                  "runtimes": {self.role: deepcopy(runtime)}, "steps": [step]}
        bundle["bundle_digest"] = _bundle_digest(bundle)
        bundle["verification"] = self._verification(bundle, self._make_step(source, evidence, runtime))
        self._validate(bundle)
        return bundle

    def create_session(self, raw, repositories):
        self._source(raw)
        return self._execute(raw, self._adapters(repositories)[1])

    def replay_session(self, bundle, repositories):
        raw = self._validate(bundle)
        runtime = self._adapters(repositories, bundle["runtimes"])[1]
        fresh = self._execute(raw, runtime)
        receipt = {"schema": "ciw." + self.kind + "-replay.v1", "source_bundle_digest": bundle["bundle_digest"],
                   "replayed_bundle_digest": fresh["bundle_digest"], "numerical_match": True,
                   "verification": self._verification(bundle, fresh["steps"][0]), "admission": "not_performed"}
        receipt["replay_id"] = digest(receipt)
        fresh["replay_receipts"] = [receipt]
        self._validate(fresh)
        return {"session": fresh, "replay_receipt": receipt}
