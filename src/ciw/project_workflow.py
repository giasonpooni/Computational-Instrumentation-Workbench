"""Provider-free CIW adapter for the versioned project graph.

The project graph is an authoring contract: typed components, signals,
computations, results and evidence with separate physical, computation and
evidence edges in an append-only, content-addressed history.  This operation
retains one exact project artifact, replays its history through the
independent Python reference and records the resulting inspection as a native
result with its own operation, execution and result identities.  It never
executes a declared computation, fetches evidence, admits state or authorizes
an action.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import platform
import re
import uuid

from . import project_model as project
from .adapters.subprocess import _json
from .exchange import _identity
from .telemetry import _bundle_digest, _now, byte_digest, canonical, digest, _keys

KIND = "project-graph"
SCHEMA = "ciw.project-graph-session.v1"
SOURCE_SCHEMA = "ciw.project-graph-source.v1"
DATA_SCHEMA = "ciw.project-graph-workbench-data.v1"
RESULT_SCHEMA = "ciw.project-graph-workbench-result.v1"
VERIFY_SCHEMA = "ciw.project-graph-verification.v1"
OPERATION = "ciw.project-graph.v1"
ROLE = "project"
ROLES = set()
# A retained bundle holds the source bytes, the inspection, its numerical copy
# and one reproduction; the source limit keeps the whole bundle inside MAX_BYTES.
SOURCE_LIMIT = 512 * 1024
MAX_BYTES = 4 * 1024 * 1024
CONFIGURATION = {
    "profile": "versioned_project_graph",
    "activation": "read_only",
    "declared_computation_execution": "not_performed",
    "physical_validation": "not_performed",
    "state_admission": "not_performed",
}
AUTHORITY = {
    "declared_computation_execution": "not_performed",
    "physical_validation": "not_performed",
    "state_admission": "not_performed",
}
CLAIM_SCOPE = "declared_graph_consistency_and_result_staleness_under_retained_history"
_REVISION = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _text(value, limit=512):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("Require bounded nonempty text")


def _algorithm_identity():
    files = [Path(project.__file__), Path(__file__)]
    content = b"\0".join(
        path.name.encode("utf-8") + b"\0" +
        path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        for path in files
    )
    return {
        "profile": "ciw.project-graph.python-reference.v1",
        "code_sha256": sha256(content).hexdigest(),
        "source_normalization": "utf8_lf",
        "python_version": platform.python_version(),
    }


def runtime_identity():
    return {
        "schema": "ciw.python-reference-runtime.v1",
        "role": ROLE,
        "profile": "ciw.project-graph.python-reference.v1",
        "algorithm": _algorithm_identity(),
        "execution_scope": "independent_python_reference_only",
        "declared_computation_execution": "not_performed",
        "physical_validation": "not_performed",
        "state_admission": "not_performed",
    }


def _runtime_projection(value):
    return deepcopy(value)


def _adapters(repositories, expected=None):
    if not isinstance(repositories, dict) or repositories:
        raise ValueError("The provider-free project graph reference accepts no repository bindings")
    runtime = runtime_identity()
    if expected is not None:
        expected_runtime = expected.get(ROLE, expected) if isinstance(expected, dict) else expected
        if _runtime_projection(runtime) != _runtime_projection(expected_runtime):
            raise ValueError("Project graph reference runtime identity differs from the retained execution")
    return None, runtime


def _request(value, artifact):
    _keys(value, {"expected_revision"})
    if not isinstance(value["expected_revision"], str) or not _REVISION.fullmatch(value["expected_revision"]):
        raise ValueError("Requested project revision must be a canonical SHA256 digest")
    if value["expected_revision"] != artifact["revision"]:
        raise ValueError("Project revision differs from the requested revision; reselect the project")
    return deepcopy(value)


def validate_source(raw):
    """Validate exact source bytes and replay the project history without executing it."""
    if type(raw) is not bytes or not 1 <= len(raw) <= SOURCE_LIMIT:
        raise ValueError("Project graph source requires bounded exact JSON bytes")
    source = _json(raw)
    _keys(source, {"schema", "experiment_id", "configuration", "project", "request"})
    if source["schema"] != SOURCE_SCHEMA or canonical(source["configuration"]) != canonical(CONFIGURATION):
        raise ValueError("Unsupported project graph source or authority policy")
    _text(source["experiment_id"], 128)
    artifact = project.validate(source["project"])
    _request(source["request"], artifact)
    if len(canonical(source)) > SOURCE_LIMIT:
        raise ValueError("Project graph source exceeds the byte budget")
    return deepcopy(source)


def _native_data(source):
    inspection = project.inspect(source["project"])
    objects = inspection["objects"]
    counts = {kind: sum(1 for item in objects if item["kind"] == kind) for kind in sorted(project.KINDS)}
    edges = {relation: sum(1 for edge in inspection["edges"] if edge["relation"] == relation)
             for relation in ("physical", "computation", "evidence")}
    statuses = {status: sum(1 for item in objects if item["kind"] == "result" and item["result_status"] == status)
                for status in ("current_for_declared_inputs", "needs_reevaluation")}
    return {
        "schema": DATA_SCHEMA,
        "operation_id": OPERATION,
        "project_id": inspection["project_id"],
        "project_revision": inspection["revision"],
        "history_length": inspection["history_length"],
        "request": deepcopy(source["request"]),
        "inspection": inspection,
        "summary": {
            "status": inspection["status"],
            "object_counts": counts,
            "edge_counts": edges,
            "result_statuses": statuses,
            "unresolved_physical_edges": len(inspection["unresolved_physical_edges"]),
            "context_status": deepcopy(inspection["context_status"]),
        },
        "claim_scope": CLAIM_SCOPE,
        "authority": deepcopy(AUTHORITY),
    }


def _step(source, evidence_id, runtime, execution_id=None):
    occurrence = execution_id or "execution-" + uuid.uuid4().hex
    data = _native_data(source)
    result = {
        "schema": RESULT_SCHEMA,
        "operation_id": OPERATION,
        "execution_ref": occurrence,
        "input_refs": [evidence_id],
        "data": data,
        "authority": deepcopy(AUTHORITY),
    }
    result["result_id"] = digest(result)
    numerical = {"operation_id": OPERATION, "data": deepcopy(data)}
    return {
        "runtime_ref": ROLE,
        "operation_id": OPERATION,
        "execution_id": occurrence,
        "input_refs": [evidence_id],
        "request": deepcopy(source["request"]),
        "request_sha256": digest(source["request"]),
        "result": result,
        "result_sha256": digest(result),
        "result_id": result["result_id"],
        "numerical_result": numerical,
        "numerical_result_id": digest(numerical),
    }


def _validate_step(step, source, evidence_id):
    _keys(step, {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256",
                 "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"})
    if (step["runtime_ref"] != ROLE or step["operation_id"] != OPERATION or
            not re.fullmatch(r"execution-[a-f0-9]{32}", step["execution_id"]) or
            step["input_refs"] != [evidence_id] or
            canonical(step["request"]) != canonical(source["request"]) or
            step["request_sha256"] != digest(source["request"])):
        raise ValueError("Project graph step request or occurrence binding differs")
    result = step["result"]
    _keys(result, {"schema", "operation_id", "execution_ref", "input_refs", "data", "authority", "result_id"})
    if (result["schema"] != RESULT_SCHEMA or result["operation_id"] != OPERATION or
            result["execution_ref"] != step["execution_id"] or result["input_refs"] != [evidence_id] or
            result["authority"] != AUTHORITY or
            result["result_id"] != digest({k: v for k, v in result.items() if k != "result_id"})):
        raise ValueError("Project graph native result envelope differs")
    expected_data = _native_data(source)
    if canonical(result["data"]) != canonical(expected_data):
        raise ValueError("Project graph native result differs from deterministic inspection")
    numerical = {"operation_id": OPERATION, "data": result["data"]}
    if step["numerical_result"] != numerical or step["numerical_result_id"] != digest(numerical):
        raise ValueError("Project graph numerical result identity differs")
    if step["result_id"] != result["result_id"] or step["result_sha256"] != digest(result):
        raise ValueError("Project graph step result commitment differs")


def _verification(bundle, reproduced):
    if canonical(bundle["steps"][0]["numerical_result"]) != canonical(reproduced["numerical_result"]):
        raise ValueError("Project graph replay numerical result differs")
    value = {
        "schema": VERIFY_SCHEMA,
        "subject_ref": bundle["bundle_digest"],
        "outcome": "passed",
        "independent": False,
        "method": "same_python_reference_fresh_occurrence_reproduction",
        "runtime_digest": digest(bundle["runtimes"]),
        "reproduction": deepcopy(reproduced),
        "authority": deepcopy(AUTHORITY),
    }
    value["verification_id"] = byte_digest(VERIFY_SCHEMA.encode() + b"\0" + canonical(value))
    return value


class ProjectGraphWorkflow:
    MAX_BYTES = MAX_BYTES
    kind = KIND
    role = ROLE
    ROLES = ROLES
    SOURCE_SCHEMA = SOURCE_SCHEMA
    schema = SCHEMA
    operation = OPERATION

    def _source(self, raw):
        return validate_source(raw)

    def _adapters(self, repositories, expected=None):
        return _adapters(repositories, expected)

    @staticmethod
    def _runtime_projection(value):
        return _runtime_projection(value)

    def _step(self, source, evidence_id, bound):
        _, runtime = bound
        return _step(source, evidence_id, runtime)

    @staticmethod
    def _validate_step(step, source, evidence_id):
        return _validate_step(step, source, evidence_id)

    def _check_verification(self, bundle, verification, source, evidence):
        _keys(verification, {"schema", "subject_ref", "outcome", "independent", "method", "runtime_digest",
                             "reproduction", "authority", "verification_id"})
        if (verification["schema"] != VERIFY_SCHEMA or verification["subject_ref"] != bundle["bundle_digest"] or
                verification["outcome"] != "passed" or verification["independent"] is not False or
                verification["method"] != "same_python_reference_fresh_occurrence_reproduction" or
                verification["authority"] != AUTHORITY):
            raise ValueError("Project graph verification scope differs")
        _validate_step(verification["reproduction"], source, evidence)
        old, new = bundle["steps"][0], verification["reproduction"]
        if (old["execution_id"] == new["execution_id"] or old["result_id"] == new["result_id"] or
                verification != _verification(bundle, new)):
            raise ValueError("Project graph verification must bind a fresh occurrence")
        _identity(verification, "verification_id")

    def _validate(self, bundle):
        try:
            _keys(bundle, {"schema", "session_id", "created_at", "source", "configuration", "runtimes",
                           "steps", "bundle_digest", "verification"}, {"replay_receipts"})
            if (len(canonical(bundle)) > MAX_BYTES or bundle["schema"] != SCHEMA or
                    bundle["bundle_digest"] != _bundle_digest(bundle) or
                    not re.fullmatch(r"session-[a-f0-9]{32}", bundle["session_id"])):
                raise ValueError("Project graph bundle identity or size differs")
            _text(bundle["created_at"])
            evidence, = bundle["source"]["evidence"]
            raw = base64.b64decode(evidence["bytes_b64"], validate=True)
            source = self._source(raw)
            expected_evidence = {"artifact_ref": byte_digest(raw), "sha256": byte_digest(raw),
                                 "bytes_b64": base64.b64encode(raw).decode()}
            if (evidence != expected_evidence or bundle["source"] != {
                    "experiment_id": source["experiment_id"], "experiment_digest": digest(source),
                    "evidence": [evidence]} or canonical(bundle["configuration"]) != canonical(CONFIGURATION)):
                raise ValueError("Project graph source or configuration binding differs")
            _keys(bundle["runtimes"], {ROLE})
            runtime = bundle["runtimes"][ROLE]
            expected_runtime = runtime_identity()
            if runtime != expected_runtime:
                # A reopened workspace keeps the retained identity strict in
                # shape and code commitment; replay compares the complete
                # current identity before executing again.
                _keys(runtime, set(expected_runtime))
                if any(runtime[key] != expected_runtime[key] for key in expected_runtime if key != "algorithm"):
                    raise ValueError("Unapproved project graph reference runtime")
                algorithm = runtime["algorithm"]
                _keys(algorithm, {"profile", "code_sha256", "source_normalization", "python_version"})
                if (algorithm["profile"] != expected_runtime["algorithm"]["profile"] or
                        algorithm["source_normalization"] != "utf8_lf" or
                        not re.fullmatch(r"[a-f0-9]{64}", algorithm["code_sha256"]) or
                        not isinstance(algorithm["python_version"], str) or
                        not 1 <= len(algorithm["python_version"]) <= 64):
                    raise ValueError("Malformed project graph reference algorithm identity")
            step, = bundle["steps"]
            self._validate_step(step, source, evidence["artifact_ref"])
            self._check_verification(bundle, bundle["verification"], source, evidence["artifact_ref"])
            receipts = bundle.get("replay_receipts", [])
            if not isinstance(receipts, list) or len(receipts) > 1:
                raise ValueError("At most one project graph replay receipt belongs to an occurrence")
            for receipt in receipts:
                _keys(receipt, {"schema", "source_bundle_digest", "replayed_bundle_digest", "numerical_match",
                                "verification", "admission", "replay_id"})
                if (receipt["schema"] != "ciw." + KIND + "-replay.v1" or
                        receipt["replayed_bundle_digest"] != bundle["bundle_digest"] or
                        receipt["source_bundle_digest"] == bundle["bundle_digest"] or
                        receipt["numerical_match"] is not True or receipt["admission"] != "not_performed" or
                        receipt["replay_id"] != digest({k: v for k, v in receipt.items() if k != "replay_id"})):
                    raise ValueError("Invalid project graph replay receipt")
                verification = receipt["verification"]
                _keys(verification, {"schema", "subject_ref", "outcome", "independent", "method", "runtime_digest",
                                     "reproduction", "authority", "verification_id"})
                if (verification["schema"] != VERIFY_SCHEMA or
                        verification["subject_ref"] != receipt["source_bundle_digest"] or
                        verification["outcome"] != "passed" or verification["independent"] is not False or
                        verification["method"] != "same_python_reference_fresh_occurrence_reproduction" or
                        verification["authority"] != AUTHORITY):
                    raise ValueError("Invalid project graph replay verification scope")
                _validate_step(verification["reproduction"], source, evidence["artifact_ref"])
                _identity(verification, "verification_id")
            return raw
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
            raise ValueError("Malformed retained project graph session") from exc

    def _execute(self, raw, runtime):
        source = self._source(raw)
        evidence = byte_digest(raw)
        step = _step(source, evidence, runtime)
        bundle = {
            "schema": SCHEMA,
            "session_id": "session-" + uuid.uuid4().hex,
            "created_at": _now(),
            "source": {"experiment_id": source["experiment_id"], "experiment_digest": digest(source),
                       "evidence": [{"artifact_ref": evidence, "sha256": evidence,
                                     "bytes_b64": base64.b64encode(raw).decode()}]},
            "configuration": deepcopy(CONFIGURATION),
            "runtimes": {ROLE: deepcopy(runtime)},
            "steps": [step],
        }
        bundle["bundle_digest"] = _bundle_digest(bundle)
        bundle["verification"] = _verification(bundle, _step(source, evidence, runtime))
        self._validate(bundle)
        return bundle

    def create_session(self, raw, repositories):
        self._source(raw)
        return self._execute(raw, self._adapters(repositories)[1])

    def replay_session(self, bundle, repositories):
        raw = self._validate(bundle)
        runtime = self._adapters(repositories, bundle["runtimes"])[1]
        fresh = self._execute(raw, runtime)
        receipt = {
            "schema": "ciw." + KIND + "-replay.v1",
            "source_bundle_digest": bundle["bundle_digest"],
            "replayed_bundle_digest": fresh["bundle_digest"],
            "numerical_match": True,
            "verification": _verification(bundle, fresh["steps"][0]),
            "admission": "not_performed",
        }
        receipt["replay_id"] = digest(receipt)
        fresh["replay_receipts"] = [receipt]
        self._validate(fresh)
        return {"session": fresh, "replay_receipt": receipt}
