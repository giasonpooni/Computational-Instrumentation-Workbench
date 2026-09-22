"""Shared retained sources and native instrument results for one CIW session.

This catalog does not estimate state. GSIE's retained posterior and conditional
prediction remain separate, explicitly selected contexts. Repository bindings
are trusted process configuration and never enter the saved workspace.
"""
from __future__ import annotations

import base64
import binascii
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from threading import RLock

from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import _json

SCHEMA = "ciw.retained-workbench.v1"
SOURCE_SCHEMA = "ciw.workbench-source.v1"
OPERATIONS = {
    "calibrated-observable": "ciw.calibrated-observable.v1",
    "identified-design": "ciw.identified-design.v1",
}
WORKFLOW_OPERATION_IDS = frozenset(OPERATIONS.values())
from .candidate_evidence import OPERATIONS as CANDIDATE_OPERATIONS
WORKBENCH_OPERATION_IDS = WORKFLOW_OPERATION_IDS | CANDIDATE_OPERATIONS.keys()
MAX_SOURCES = 64
MAX_BUNDLES = 128
MAX_BYTES = 64 * 1024 * 1024
_OVERHEAD = 4096


def _workflow(kind):
    # Lazy imports avoid the existing workflows' Session persistence dependency.
    if kind == "calibrated-observable":
        from . import calibrated_observable
        return calibrated_observable
    if kind == "identified-design":
        from . import identified_design
        return identified_design
    raise ValueError("Unknown workbench source kind")


def _canonical(value):
    from .telemetry import canonical
    return canonical(value)


def _digest(value):
    return "sha256:" + sha256(_canonical(value)).hexdigest()


def _keys(value, required, optional=()):
    if (not isinstance(value, dict) or
            not set(required) <= value.keys() <= set(required) | set(optional)):
        raise ValueError("Unexpected or missing workbench fields")


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"{name} must be a nonempty bounded string")


def _descriptor(source):
    return {k: v for k, v in source.items() if k != "bytes_b64"}


def _source(payload):
    _keys(payload, {"kind", "label", "bytes_b64"})
    kind, label, encoded = payload["kind"], payload["label"], payload["bytes_b64"]
    _text(kind, "Source kind")
    _text(label, "Source label")
    workflow = _workflow(kind)
    if not isinstance(encoded, str) or len(encoded) > 4 * ((workflow.MAX_BYTES + 2) // 3):
        raise ValueError("Source bytes exceed the workflow byte budget")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Source bytes must use canonical base64") from exc
    if len(raw) > workflow.MAX_BYTES or base64.b64encode(raw).decode("ascii") != encoded:
        raise ValueError("Source bytes must use bounded canonical base64")
    if kind == "calibrated-observable":
        declaration = workflow._source(raw)
    else:
        # The full design validator needs an explicitly selected retained prior.
        declaration = _json(raw)
        if not isinstance(declaration, dict) or declaration.get("schema") != workflow.SOURCE_SCHEMA:
            raise ValueError("Unsupported identified-design source")
        _canonical(declaration)
    descriptor = {"schema": SOURCE_SCHEMA, "kind": kind, "label": label,
        "source_schema": declaration["schema"],
        "evidence_id": "sha256:" + sha256(raw).hexdigest(), "byte_count": len(raw)}
    descriptor["source_id"] = "source:" + _digest(descriptor)
    return {**descriptor, "bytes_b64": encoded}


def _summary(record):
    native = record["native"]
    verification = native.get("verification", {})
    return {"bundle_id": record["bundle_id"], "kind": record["kind"],
        "source_id": record["source_id"], "upstream_bundle_id": record["upstream_bundle_id"],
        "session_id": native["session_id"], "operation_id": OPERATIONS[record["kind"]],
        "result_ids": [s["result_id"] for s in native["steps"]],
        "execution_ids": [s["execution_id"] for s in native["steps"]],
        "verification_id": verification.get("verification_id"),
        "retained_verification_outcome": verification.get("outcome"),
        "validation": "content_consistent", "numerical_replay": "not_performed_by_inspection",
        "state_admission": "not_performed"}


def _claims(record):
    """Reject identity aliasing while allowing exact embedded upstream reuse."""
    claims = {}

    def claim(identity, role, value):
        _text(identity, "Retained identity")
        content = (role, _digest(value))
        if identity in claims and claims[identity] != content:
            raise ValueError("Retained identity collision")
        claims[identity] = content

    def verification(value):
        claim(value["verification_id"], "verification", value)

    def bundle(value):
        owner = value["bundle_digest"]
        body = {k: v for k, v in value.items() if k not in {"verification", "replay_receipts"}}
        claim(owner, "bundle", body)
        claim(value["session_id"], "session", owner)
        for evidence in value["source"]["evidence"]:
            claim(evidence["artifact_ref"], "evidence", evidence["bytes_b64"])
        for step in value["steps"]:
            claim(step["operation_id"], "operation", step["operation_id"])
            claim(step["execution_id"], "execution", {"bundle_id": owner, "step": step})
            claim(step["result_id"], "result", {"bundle_id": owner, "result": step["result"]})
            claim(step["numerical_result_id"], "numerical_result", step["numerical_result"])
        if "verification" in value:
            verification(value["verification"])
        for receipt in value.get("replay_receipts", []):
            claim(receipt["replay_id"], "replay", receipt)
            verification(receipt["verification"])
        if "upstream" in value:
            bundle(value["upstream"])
            bundle(value["upstream_replay"]["session"])
            receipt = value["upstream_replay"]["replay_receipt"]
            claim(receipt["replay_id"], "replay", receipt)
            verification(receipt["verification"])

    bundle(record["native"])
    return claims


def _source_claims(source):
    return {source["source_id"]: ("source", _digest(_descriptor(source))),
        source["evidence_id"]: ("evidence", _digest(source["bytes_b64"]))}


def _validate_record(record, sources):
    _keys(record, {"kind", "source_id", "upstream_bundle_id", "bundle_id", "native"})
    _text(record["kind"], "Bundle kind")
    _text(record["source_id"], "Source identity")
    _text(record["bundle_id"], "Bundle identity")
    if record["upstream_bundle_id"] is not None:
        _text(record["upstream_bundle_id"], "Upstream bundle identity")
    workflow = _workflow(record["kind"])
    source = sources.get(record["source_id"])
    if source is None or source["kind"] != record["kind"]:
        raise ValueError("Bundle must name its retained source of the same kind")
    native = record["native"]
    if not isinstance(native, dict):
        raise ValueError("Native bundle must be an object")
    raw = workflow._validate(native)
    if (native["bundle_digest"] != record["bundle_id"] or
            raw != base64.b64decode(source["bytes_b64"], validate=True)):
        raise ValueError("Bundle differs from its exact retained source bytes")
    if "verification" not in native:
        raise ValueError("A retained workflow must preserve its native verification artifact")
    if record["kind"] == "calibrated-observable":
        if record["upstream_bundle_id"] is not None:
            raise ValueError("Calibrated workflow has no implicit upstream bundle")
    elif record["upstream_bundle_id"] is None:
        raise ValueError("Identified design needs an explicitly selected upstream bundle")
    _validate_receipts(native, record["kind"])
    return _claims(record)


def _validate_receipts(native, kind):
    receipts = native.get("replay_receipts", [])
    if not isinstance(receipts, list) or len(receipts) > 1:
        raise ValueError("A native replay retains one receipt")
    for receipt in receipts:
        _keys(receipt, {"schema", "source_bundle_digest", "replayed_bundle_digest", "numerical_match",
                        "verification", "admission", "replay_id"})
        if (receipt["schema"] != "ciw." + kind + "-replay.v1" or
                receipt["replayed_bundle_digest"] != native["bundle_digest"] or
                receipt["source_bundle_digest"] == native["bundle_digest"] or
                receipt["numerical_match"] is not True or receipt["admission"] != "not_performed" or
                receipt["replay_id"] != _digest({k: v for k, v in receipt.items() if k != "replay_id"})):
            raise ValueError("Replay receipt does not bind the retained fresh bundle")
        from .exchange import _identity
        _identity(receipt["verification"], "verification_id")
        if receipt["verification"]["subject_ref"] != receipt["source_bundle_digest"]:
            raise ValueError("Replay verification subject differs from replay source")


def _validate_links(record, bundles):
    native = record["native"]
    upstream_id = record["upstream_bundle_id"]
    if upstream_id is not None:
        upstream = bundles.get(upstream_id)
        if (upstream is None or upstream["kind"] != "calibrated-observable" or
                _canonical(native["upstream"]) != _canonical(upstream["native"])):
            raise ValueError("Design upstream must exactly match a retained calibrated bundle")
    for receipt in native.get("replay_receipts", []):
        original = bundles.get(receipt["source_bundle_digest"])
        if (original is None or original["kind"] != record["kind"] or
                original["source_id"] != record["source_id"] or
                original["upstream_bundle_id"] != upstream_id):
            raise ValueError("Replay source must already belong to this workbench")
        old_steps, new_steps = original["native"]["steps"], native["steps"]
        if any(old["numerical_result_id"] != new["numerical_result_id"] or
               old["execution_id"] == new["execution_id"] or old["result_id"] == new["result_id"]
               for old, new in zip(old_steps, new_steps)):
            raise ValueError("Replay must preserve numerical identity and create fresh occurrences")


def _context(record, sources):
    native = record["native"]
    calibrated = record["kind"] == "calibrated-observable"
    step = native["steps"][4 if calibrated else 2]
    state = step["result"]["data"]
    configuration = native["configuration"]
    source = sources[record["source_id"]]
    common = {"context_id": "context:" + _digest({"bundle_id": record["bundle_id"],
                                                   "result_id": step["result_id"]}),
        "bundle_id": record["bundle_id"], "source_id": record["source_id"],
        "evidence_id": source["evidence_id"], "upstream_bundle_id": record["upstream_bundle_id"],
        "state_kind": "posterior" if calibrated else "conditional_prediction",
        "owner": "gsie", "state_id": state["state_id"], "result_id": step["result_id"],
        "execution_id": step["execution_id"], "event_time": state["time"],
        "frame_id": state["frame_id"], "units": state["units"],
        "mean": state["mean"], "covariance": state["covariance"],
        "validation": "content_consistent", "numerical_replay": "not_performed_by_inspection",
        "state_admission": "not_performed",
        "verification_id": native["verification"]["verification_id"],
        "retained_verification_outcome": native["verification"].get("outcome")}
    if calibrated:
        experiment = _json(base64.b64decode(source["bytes_b64"], validate=True))
        observable = native["steps"][3]["result"]["data"]
        reconciled = native["steps"][5]["result"]["data"]
        faults = native["steps"][6]["result"]["data"]
        common.update(state_names=configuration["observability"]["state_names"],
            model_id=configuration["gsie"]["observation_model"]["model_id"],
            dynamics_model_id=configuration["gsie"]["dynamics"]["model_id"],
            clock_frames=[c["clock_model"]["reference_frame"] for c in experiment["channels"]],
            channel_ids=[c["channel_id"] for c in experiment["channels"]],
            claim_scope=experiment["claim_scope"],
            observability={k: observable[k] for k in ("status", "rank", "condition_number", "condition_limit")},
            reconciliation={"result_id": native["steps"][5]["result_id"],
                            "status": reconciled["status"], "reason": reconciled["reason"]},
            fault_assessment={"result_id": native["steps"][6]["result_id"],
                "detection": faults["detection"]["status"], "isolability": faults["isolability"]["status"],
                "isolated_fault": faults["isolability"].get("isolated_fault"),
                "cross_covariance_policy": faults["isolability"]["cross_covariance_policy"]})
    else:
        previous = native["upstream"]
        previous_faults = previous["steps"][6]["result"]["data"]
        common.update(state_names=configuration["identification"]["state_names"],
            model_id=state["model_result_id"],
            declared_model_id=configuration["identification"]["model_id"],
            clock_frame=configuration["identification"]["clock_frame"],
            predecessor_state_id=state["predecessor_state_id"],
            uncertainty_scope=state["uncertainty_scope"],
            parameter_covariance_status=state["parameter_covariance_status"],
            upstream_assessment={"bundle_id": record["upstream_bundle_id"],
                "reconciliation_status": previous["steps"][5]["result"]["data"]["status"],
                "detection_status": previous_faults["detection"]["status"],
                "isolability_status": previous_faults["isolability"]["status"]},
            decision=native["decision"])
    return common


class Workbench:
    """Thread-safe append-only shared catalog with explicit trusted bindings."""

    def __init__(self):
        self._lock = RLock()
        self._sources = {}
        self._bundles = {}
        self._bindings = {}
        self._candidate_adapter = None
        self._candidates = {}
        self._identities = {operation: ("operation", _digest(operation)) for operation in WORKFLOW_OPERATION_IDS}
        self._used_bytes = 0
        self._pending = 0
        self._reserved_bytes = 0
        self._revision = 0

    def bind_workflow(self, kind, repositories):
        """Bind host-selected repositories; no saved/client value calls this."""
        workflow = _workflow(kind)
        if not isinstance(repositories, dict) or set(repositories) != workflow.ROLES:
            raise ValueError("Bind exactly the repositories declared by the workflow")
        if any(not isinstance(path, (str, Path)) or not str(path).strip() for path in repositories.values()):
            raise ValueError("Workflow repository bindings must be explicit host paths")
        bindings = {role: Path(path).resolve() for role, path in repositories.items()}
        # Validate trusted provider identities before advertising availability.
        # The workflows check them again at each execution and replay.
        workflow._adapters(bindings)
        with self._lock:
            self._bindings[kind] = bindings

    @property
    def pending_operations(self):
        with self._lock:
            return self._pending

    def describe_operations(self):
        with self._lock:
            return [{"operation_id": operation, "role": "state_estimator" if kind == "calibrated-observable" else "decision",
                     "source_kind": kind, "available": kind in self._bindings,
                     "requires_upstream_bundle": kind == "identified-design"}
                    for kind, operation in OPERATIONS.items()] + [
                        {"operation_id": operation, "role": "candidate_evidence", "requires_bundle": "calibrated-observable",
                         "available": self._candidate_adapter is not None and
                            (action == "inspect" or self._candidate_adapter.capture_available),
                         "canonical_admission": "not_performed"}
                        for operation, action in CANDIDATE_OPERATIONS.items()]

    def bind_candidate_adapter(self, configuration):
        """Trusted host binding only; never recovered from workspace records."""
        from .candidate_evidence import CandidateAdapter
        adapter = CandidateAdapter(configuration)
        with self._lock:
            self._candidate_adapter = adapter

    def instrument_views(self):
        with self._lock:
            return deepcopy([{ "bundle_id": record["bundle_id"], "source_id": record["source_id"],
                "instrument": step["runtime_ref"], "operation_id": step["operation_id"],
                "result_id": step["result_id"], "execution_id": step["execution_id"],
                "view": "retained_native_result", "state_admission": "not_performed"}
                for record in self._bundles.values() for step in record["native"]["steps"]
                if step["runtime_ref"] in {"gsie", "cbsr", "fdir"}])

    def inspect_instrument(self, payload):
        _keys(payload, {"bundle_id", "instrument"})
        _text(payload["bundle_id"], "Bundle identity")
        if payload["instrument"] not in ("gsie", "cbsr", "fdir"):
            raise ValueError("Choose gsie, cbsr or fdir")
        with self._lock:
            native = self.get_bundle(payload["bundle_id"])
            steps = {step["runtime_ref"]: step for step in native["steps"]}
            if payload["instrument"] not in steps:
                raise ValueError("Instrument did not execute in this bundle; explicitly select its upstream bundle")
            record = self._bundles[payload["bundle_id"]]
            return deepcopy({"bundle_id": record["bundle_id"], "source_id": record["source_id"],
                "instrument": payload["instrument"], "step": steps[payload["instrument"]],
                "fusion_context": _context(record, self._sources),
                "linked_results": {role: steps[role]["result_id"] for role in ("gsie", "cbsr", "fdir", "oit") if role in steps},
                "numerical_replay": "not_performed_by_inspection", "state_admission": "not_performed"})

    def execute_candidate(self, operation_id, parameters):
        from .candidate_evidence import MAX_RESPONSE
        action = CANDIDATE_OPERATIONS[operation_id]
        required = {"bundle_id", "inspected_at"} if action == "inspect" else {"bundle_id", "evidence_id", "workflow_id", "retained_at"}
        _keys(parameters, required)
        for key in required:
            _text(parameters[key], key)
        parameters = deepcopy(parameters)
        reserved = MAX_RESPONSE * 2 + 2 * 1024 * 1024
        with self._lock:
            bundle = self.get_bundle(parameters["bundle_id"])
            if bundle["schema"] != "ciw.calibrated-observable-session.v1":
                raise ValueError("ESM candidate operations currently require an explicitly selected calibrated bundle")
            adapter = self._candidate_adapter
            if adapter is None:
                raise AdapterRefusal("operation_unavailable", "No operator-bound ESM candidate adapter")
            if (len(self._candidates) + self._pending >= MAX_BUNDLES or
                    self._used_bytes + self._reserved_bytes + reserved + _OVERHEAD > MAX_BYTES):
                raise AdapterRefusal("workbench_capacity", "Candidate receipt capacity exceeded")
            self._pending += 1
            self._reserved_bytes += reserved
        try:
            from uuid import uuid4
            evidence = adapter.execute(action, parameters, bundle)
            record = {"schema": "ciw.candidate-action.v1", "execution_id": "candidate-execution:" + str(uuid4()),
                      "operation_id": operation_id, "parameters": parameters, **evidence}
            record["candidate_id"] = "candidate:" + _digest(record)
            self._validate_candidate(record)
            size = len(_canonical(record))
            with self._lock:
                if size > reserved:
                    raise AdapterRefusal("workbench_capacity", "Candidate receipt exceeds reservation; capture may have retained bytes")
                self._candidates[record["candidate_id"]] = record
                self._used_bytes += size
                self._revision += 1
            return self.get_candidate(record["candidate_id"])
        finally:
            with self._lock:
                self._pending -= 1
                self._reserved_bytes -= reserved

    def _validate_candidate(self, record):
        from .candidate_evidence import MAX_RESPONSE, validate_response
        _keys(record, {"schema", "candidate_id", "execution_id", "operation_id", "parameters", "response_bytes_b64", "operator_policy", "adapter_identity"})
        if record["schema"] != "ciw.candidate-action.v1" or record["operation_id"] not in CANDIDATE_OPERATIONS:
            raise ValueError("Unsupported candidate action record")
        _text(record["execution_id"], "Candidate execution identity")
        if record["candidate_id"] != "candidate:" + _digest({k: v for k, v in record.items() if k != "candidate_id"}):
            raise ValueError("Candidate receipt content mismatch")
        action = CANDIDATE_OPERATIONS[record["operation_id"]]
        _keys(record["operator_policy"], {"review_context"} | ({"capture_registration"} if action == "capture" else set()))
        if any(not isinstance(value, dict) for value in record["operator_policy"].values()):
            raise ValueError("Candidate receipt needs explicit operator policy records")
        _keys(record["adapter_identity"], {"repository", "revision", "artifact_sha256", "helper_sha256", "replay_ciw_revision", "node_sha256"})
        from re import fullmatch
        if (record["adapter_identity"]["repository"] != "giasonpooni/Evidence-and-State-Management" or
                any(not isinstance(value, str) or not fullmatch("[a-f0-9]{40}" if key.endswith("revision") else "[a-f0-9]{64}", value)
                    for key, value in record["adapter_identity"].items() if key != "repository")):
            raise ValueError("Candidate adapter identity is malformed")
        _keys(record["parameters"], {"bundle_id", "inspected_at"} if action == "inspect" else
              {"bundle_id", "evidence_id", "workflow_id", "retained_at"})
        for key, value in record["parameters"].items():
            _text(value, key)
        encoded = record["response_bytes_b64"]
        if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_RESPONSE + 2) // 3):
            raise ValueError("Candidate receipt byte limit")
        output = base64.b64decode(encoded, validate=True)
        if base64.b64encode(output).decode("ascii") != encoded:
            raise ValueError("Candidate receipt must retain canonical base64 bytes")
        bundle = self.get_bundle(record["parameters"]["bundle_id"])
        if bundle["schema"] != "ciw.calibrated-observable-session.v1":
            raise ValueError("Candidate receipt requires a retained calibrated bundle")
        validate_response(_json(output), action, bundle, _canonical(bundle), record["parameters"], record["operator_policy"])

    def list_candidates(self):
        with self._lock:
            return [{"candidate_id": record["candidate_id"], "bundle_id": record["parameters"]["bundle_id"],
                     "operation_id": record["operation_id"], "execution_id": record["execution_id"],
                     "state": _json(base64.b64decode(record["response_bytes_b64"]))["state"],
                     "eligibility": "historical_receipt_only", "state_admission": "not_performed"}
                    for record in self._candidates.values()]

    def get_candidate(self, candidate_id):
        _text(candidate_id, "Candidate action identity")
        with self._lock:
            if candidate_id not in self._candidates:
                raise ValueError("Unknown candidate action")
            record = deepcopy(self._candidates[candidate_id])
            return {**record, "native_response": _json(base64.b64decode(record["response_bytes_b64"])),
                    "eligibility": "historical_receipt_only", "state_admission": "not_performed"}

    def candidate_executions(self):
        with self._lock:
            return [{"execution_id": record["execution_id"], "operation_id": record["operation_id"],
                     "runtime_ref": "esm", "bundle_id": record["parameters"]["bundle_id"],
                     "candidate_id": record["candidate_id"], "status": "completed",
                     "outcome": _json(base64.b64decode(record["response_bytes_b64"]))["state"],
                     "state_admission": "not_performed"} for record in self._candidates.values()]

    def _check_claims(self, claims):
        for identity, claim in claims.items():
            if identity in self._identities and self._identities[identity] != claim:
                raise ValueError("Identity collision across retained workbench artifacts")

    def add_source(self, payload):
        try:
            source = _source(payload)
        except (RecursionError, OverflowError) as exc:
            raise ValueError("Source exceeds the JSON nesting or numeric budget") from exc
        size, claims = len(_canonical(source)), _source_claims(source)
        with self._lock:
            existing = self._sources.get(source["source_id"])
            if existing is not None:
                if _canonical(existing) != _canonical(source):
                    raise ValueError("Source identity collision")
                return deepcopy(_descriptor(existing))
            if len(self._sources) >= MAX_SOURCES or self._used_bytes + self._reserved_bytes + size + _OVERHEAD > MAX_BYTES:
                raise AdapterRefusal("workbench_capacity", "Retained source capacity exceeded")
            self._check_claims(claims)
            self._sources[source["source_id"]] = source
            self._identities.update(claims)
            self._used_bytes += size
            self._revision += 1
            return deepcopy(_descriptor(source))

    def list_sources(self):
        with self._lock:
            return deepcopy([_descriptor(value) for value in self._sources.values()])

    def get_source(self, source_id):
        _text(source_id, "Source identity")
        with self._lock:
            if source_id not in self._sources:
                raise ValueError("Unknown retained workbench source")
            return deepcopy(self._sources[source_id])

    def list_bundles(self):
        with self._lock:
            return deepcopy([_summary(value) for value in self._bundles.values()])

    def get_bundle(self, bundle_id):
        _text(bundle_id, "Bundle identity")
        with self._lock:
            if bundle_id not in self._bundles:
                raise ValueError("Unknown retained workbench bundle")
            return deepcopy(self._bundles[bundle_id]["native"])

    def _reserve(self, kind):
        if kind not in self._bindings:
            raise AdapterRefusal("operation_unavailable", "No trusted repositories bound for this workbench workflow")
        reserved = _workflow(kind).MAX_BYTES + _OVERHEAD
        if (len(self._bundles) + self._pending >= MAX_BUNDLES or
                self._used_bytes + self._reserved_bytes + reserved + _OVERHEAD > MAX_BYTES):
            raise AdapterRefusal("workbench_capacity", "Retained bundle capacity exceeded")
        self._pending += 1
        self._reserved_bytes += reserved
        return dict(self._bindings[kind]), reserved

    def _retain(self, kind, source, upstream_id, native):
        native = deepcopy(native)
        record = {"kind": kind, "source_id": source["source_id"],
            "upstream_bundle_id": upstream_id, "bundle_id": native.get("bundle_digest"), "native": native}
        claims = _validate_record(record, {source["source_id"]: source})
        size = len(_canonical(record))
        with self._lock:
            if record["bundle_id"] in self._bundles:
                raise ValueError("Workflow must return a fresh native bundle identity")
            _validate_links(record, self._bundles)
            self._check_claims(claims)
            if self._used_bytes + size + _OVERHEAD > MAX_BYTES:
                raise AdapterRefusal("workbench_capacity", "Retained bundle byte capacity exceeded")
            self._bundles[record["bundle_id"]] = record
            self._identities.update(claims)
            self._used_bytes += size
            self._revision += 1
            return deepcopy(_summary(record))

    def execute(self, payload):
        _keys(payload, {"operation_id", "source_id"}, {"upstream_bundle_id"})
        _text(payload["operation_id"], "Operation identity")
        _text(payload["source_id"], "Source identity")
        kind = next((kind for kind, operation in OPERATIONS.items() if operation == payload["operation_id"]), None)
        if kind is None:
            raise AdapterRefusal("operation_unavailable", "Unknown shared workbench operation")
        upstream_id = payload.get("upstream_bundle_id")
        if kind == "identified-design":
            _text(upstream_id, "Explicit upstream bundle identity")
        elif upstream_id is not None:
            raise ValueError("Calibrated workflow does not accept an upstream bundle")
        with self._lock:
            source = self.get_source(payload["source_id"])
            if source["kind"] != kind:
                raise ValueError("Operation source kind mismatch")
            upstream = None
            if upstream_id is not None:
                if upstream_id not in self._bundles or self._bundles[upstream_id]["kind"] != "calibrated-observable":
                    raise ValueError("Select a calibrated bundle already retained in this workbench")
                upstream = deepcopy(self._bundles[upstream_id]["native"])
            bindings, reserved = self._reserve(kind)
        try:
            raw = base64.b64decode(source["bytes_b64"], validate=True)
            workflow = _workflow(kind)
            native = workflow.create_session(raw, bindings) if upstream is None else workflow.create_session(raw, upstream, bindings)
            return self._retain(kind, source, upstream_id, native)
        finally:
            with self._lock:
                self._pending -= 1
                self._reserved_bytes -= reserved

    def replay(self, payload):
        _keys(payload, {"bundle_id"})
        _text(payload["bundle_id"], "Bundle identity")
        with self._lock:
            if payload["bundle_id"] not in self._bundles:
                raise ValueError("Unknown retained workbench bundle")
            record = deepcopy(self._bundles[payload["bundle_id"]])
            source = deepcopy(self._sources[record["source_id"]])
            bindings, reserved = self._reserve(record["kind"])
        try:
            replayed = _workflow(record["kind"]).replay_session(record["native"], bindings)
            native, receipt = replayed["session"], replayed["replay_receipt"]
            if (receipt["source_bundle_digest"] != record["bundle_id"] or
                    _canonical(native.get("replay_receipts")) != _canonical([receipt])):
                raise ValueError("Replay receipt differs from the returned native bundle")
            summary = self._retain(record["kind"], source, record["upstream_bundle_id"], native)
            return {"bundle": summary, "replay_receipt": deepcopy(receipt)}
        finally:
            with self._lock:
                self._pending -= 1
                self._reserved_bytes -= reserved

    def fusion_contexts(self):
        with self._lock:
            return deepcopy([_context(record, self._sources) for record in self._bundles.values()])

    def _native_steps(self):
        """Expose embedded replay occurrences without counting reused upstream twice."""
        seen = set()

        def visit(record, native):
            for step in native["steps"]:
                if step["result_id"] not in seen:
                    seen.add(step["result_id"])
                    yield record, native, step
            if "upstream" in native:
                yield from visit(record, native["upstream"])
                yield from visit(record, native["upstream_replay"]["session"])

        # Prefer the explicit catalog owner for original upstream occurrences.
        for record in self._bundles.values():
            for step in record["native"]["steps"]:
                if step["result_id"] not in seen:
                    seen.add(step["result_id"])
                    yield record, record["native"], step
        for record in self._bundles.values():
            yield from visit(record, record["native"])

    def native_results(self):
        with self._lock:
            return deepcopy([step["result"] for _, _, step in self._native_steps()])

    def native_executions(self):
        with self._lock:
            return deepcopy([{**{k: step[k] for k in ("execution_id", "operation_id", "runtime_ref", "input_refs", "result_id")},
                "bundle_id": record["bundle_id"], "native_bundle_id": native["bundle_digest"],
                "source_id": (record["source_id"] if native["bundle_digest"] == record["bundle_id"]
                              else self._bundles[record["upstream_bundle_id"]]["source_id"]),
                "status": "completed"}
                for record, native, step in self._native_steps()])

    def snapshot(self):
        with self._lock:
            return {"schema": SCHEMA, "revision": self._revision, "sources": self.list_sources(),
                "bundles": self.list_bundles(), "fusion_contexts": self.fusion_contexts(),
                "instruments": self.instrument_views(), "candidates": self.list_candidates(),
                "operations": self.describe_operations()}

    def serialize(self):
        with self._lock:
            return deepcopy({"schema": "ciw.retained-workbench.v2" if self._candidates else SCHEMA, "revision": self._revision,
                "sources": list(self._sources.values()), "bundles": list(self._bundles.values()),
                **({"candidates": list(self._candidates.values())} if self._candidates else {})})

    @classmethod
    def restore(cls, value):
        """Validate all saved data before returning an unbound workbench."""
        try:
            if len(_canonical(value)) > MAX_BYTES:
                raise ValueError("Retained workbench exceeds the byte budget")
            value = deepcopy(value)
            new = value.get("schema") == "ciw.retained-workbench.v2"
            _keys(value, {"schema", "revision", "sources", "bundles"} | ({"candidates"} if new else set()))
            candidates = value.get("candidates", [])
            if not isinstance(candidates, list) or len(candidates) > MAX_BUNDLES:
                raise ValueError("Malformed candidate receipt catalog")
            if (value["schema"] not in {SCHEMA, "ciw.retained-workbench.v2"} or type(value["revision"]) is not int or
                    not isinstance(value["sources"], list) or not isinstance(value["bundles"], list) or
                    len(value["sources"]) > MAX_SOURCES or len(value["bundles"]) > MAX_BUNDLES or
                    value["revision"] != len(value["sources"]) + len(value["bundles"]) + len(candidates)):
                raise ValueError("Malformed retained workbench catalog")
            restored = cls()
            for retained in value["sources"]:
                _keys(retained, {"schema", "kind", "label", "source_schema", "evidence_id", "byte_count", "source_id", "bytes_b64"})
                source = _source({k: retained[k] for k in ("kind", "label", "bytes_b64")})
                if _canonical(source) != _canonical(retained) or source["source_id"] in restored._sources:
                    raise ValueError("Retained source identity/content mismatch")
                claims = _source_claims(source)
                restored._check_claims(claims)
                restored._sources[source["source_id"]] = source
                restored._identities.update(claims)
                restored._used_bytes += len(_canonical(source))
            for record in value["bundles"]:
                claims = _validate_record(record, restored._sources)
                if record["bundle_id"] in restored._bundles:
                    raise ValueError("Duplicate retained bundle identity")
                restored._check_claims(claims)
                restored._bundles[record["bundle_id"]] = record
                restored._identities.update(claims)
                restored._used_bytes += len(_canonical(record))
            for record in restored._bundles.values():
                _validate_links(record, restored._bundles)
            occurrences = {entry["execution_id"] for entry in restored.native_executions()}
            for record in candidates:
                restored._validate_candidate(record)
                if record["candidate_id"] in restored._candidates or record["execution_id"] in occurrences:
                    raise ValueError("Duplicate candidate action identity")
                occurrences.add(record["execution_id"])
                restored._candidates[record["candidate_id"]] = record
                restored._used_bytes += len(_canonical(record))
            if restored._used_bytes + _OVERHEAD > MAX_BYTES:
                raise ValueError("Retained workbench exceeds the storage byte budget")
            restored._revision = value["revision"]
            # Check projections now, so a restored catalog can always be read.
            restored.fusion_contexts()
            return restored
        except (KeyError, TypeError, IndexError, OverflowError, RecursionError) as exc:
            raise ValueError("Malformed retained workbench") from exc
