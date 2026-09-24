"""Bounded offline PPDA acquisition through native registries and checkpoints.

The source contains exact append-only dataset snapshots. Native acquisition
produces records and observations in a temporary durable store; the complete
evidence graph and checkpoint history enter the existing workbench bundle.
Sequence order is a declared acquisition cursor, never inferred event time.
"""
from __future__ import annotations

import base64
from datetime import datetime
from hashlib import sha256
import json
import re
import uuid

from .adapters.ppda_acquisition import AcquisitionAdapter, PPDA_REVISION, VENDOR_REVISION
from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import _json
from .declared_workload import DeclaredWorkflow, RESULT_SCHEMA, AUTHORITY, _text
from .core.canonical import canonical, digest, byte_digest, bundle_digest, exact_keys

MAX_BYTES = 4 * 1024 * 1024
ROLES = {"ppda"}
SOURCE_SCHEMA = "ciw.acquired-dataset-source.v1"
POLICY = {"mode": "incremental", "source_format": "json_records_with_declared_sequence",
          "replay_scope": "isolated_offline_reacquisition", "event_time_order": "not_inferred",
          "sensor_fusion": "not_performed"}
DATA_SCHEMA = "ciw.ppda-acquisition.v1"
ADAPTER_VERSION = "352e01125ba1fe5c751ab78bd70740f8ab999590c8ef253068d07213d7a7cdf7"
SOURCE_TREES = {PPDA_REVISION: "5d7515101f00763cff7165aa4c61c0b4ae69e152",
                VENDOR_REVISION: "145f0617b2da7e4d391985f0a03fb58f408992ee"}


def _source(raw):
    if not isinstance(raw, bytes) or len(raw) > 262144:
        raise ValueError("Acquisition source exceeds the 256 KiB budget")
    value = _json(raw)
    canonical(value)
    exact_keys(value, {"schema", "experiment_id", "source", "plan_id", "snapshots", "configuration"})
    if value["schema"] != SOURCE_SCHEMA or value["configuration"] != POLICY:
        raise ValueError("Require the explicit bounded acquisition policy")
    _text(value["experiment_id"])
    exact_keys(value["source"], {"source_id", "name", "domain"})
    for item in value["source"].values():
        _text(item)
    if not isinstance(value["plan_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value["plan_id"]):
        raise ValueError("Require a bounded safe plan identifier")
    if not isinstance(value["snapshots"], list) or not 1 <= len(value["snapshots"]) <= 8:
        raise ValueError("Require 1..8 explicitly retained snapshots")
    previous, previous_time = [], None
    for snapshot in value["snapshots"]:
        exact_keys(snapshot, {"requested_at", "bytes_b64"})
        try:
            at = datetime.fromisoformat(snapshot["requested_at"].replace("Z", "+00:00"))
            if at.utcoffset() is None or previous_time is not None and at <= previous_time:
                raise ValueError("Acquisition occurrences require increasing timezone-aware request times")
            encoded = snapshot["bytes_b64"]
            data = base64.b64decode(encoded, validate=True)
            if base64.b64encode(data).decode() != encoded or len(data) > 65536:
                raise ValueError("Require canonical base64 and bounded exact snapshot bytes")
        except (TypeError, AttributeError) as exc:
            raise ValueError("Malformed acquisition snapshot") from exc
        records = _json(data)
        canonical(records)
        if not isinstance(records, list) or not 1 <= len(records) <= 64:
            raise ValueError("Require 1..64 retained records in each snapshot")
        last = -1
        for record in records:
            if not isinstance(record, dict) or type(record.get("sequence")) is not int or not last < record["sequence"] <= 999999999999:
                raise ValueError("Records require explicit distinct increasing nonnegative sequence numbers")
            last = record["sequence"]
        if canonical(records[:len(previous)]) != canonical(previous):
            raise ValueError("This incremental lane accepts append-only snapshots; revisions require another plan")
        previous, previous_time = records, at
    return value


def _filename(source):
    return "dataset-" + sha256(canonical(source["source"])).hexdigest() + ".json"


_BOOTSTRAP = r'''
import base64, dataclasses, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import daf
from daf.catalog.checkpoint import CheckpointStore
from daf.catalog.plan import AcquisitionPlan
from daf.orchestration.adapter_registry import AdapterRegistry
from daf.orchestration.bindings import incremental_dataset_binding
from daf.orchestration.source_registry import SourceDefinition, SourceRegistry
from daf.scheduling.runner import execute_plan
from daf.storage.durable_pool import DurablePool
from daf.storage.filesystem_store import FilesystemEvidenceStore
from daf.storage import serialization
source = json.loads(sys.stdin.buffer.read())
definition = SourceDefinition(**source['source'], adapter_id='incremental-dataset',
    capabilities=('incremental',), required_parameters=('path',))
sources, adapters = SourceRegistry(), AdapterRegistry()
sources.register(definition)
binding = incremental_dataset_binding()
adapters.register(binding)
plan = AcquisitionPlan(plan_id=source['plan_id'], source_id=definition.source_id,
    parameters={'path': sys.argv[2]}, mode='incremental')
store = FilesystemEvidenceStore(Path('evidence'))
pool = DurablePool.restore(store)
checkpoints = CheckpointStore(Path('checkpoints'))
runs = []
for index, snapshot in enumerate(source['snapshots']):
    raw = base64.b64decode(snapshot['bytes_b64'], validate=True)
    Path(sys.argv[2]).write_bytes(raw)
    before = checkpoints.get(plan.plan_id)
    result = execute_plan(plan, sources, adapters, pool, checkpoints, snapshot['requested_at'])
    if not result.succeeded or result.admission_failures:
        raise ValueError('Native acquisition did not retain every declared record')
    after = checkpoints.get(plan.plan_id)
    from hashlib import sha256
    runs.append({'snapshot_index': index, 'snapshot_sha256': 'sha256:' + sha256(raw).hexdigest(),
        'requested_at': snapshot['requested_at'], 'result': dataclasses.asdict(result),
        'checkpoint_before': dataclasses.asdict(before) if before else None,
        'checkpoint_after': dataclasses.asdict(after), 'pool_fingerprint': pool.fingerprint()})
    pool = DurablePool.restore(store)
evidence = {}
for category, singular in [('sources','source'),('documents','document'),('records','record'),('observations','observation')]:
    evidence[category] = sorted([getattr(serialization, singular+'_to_dict')(v)
        for v in getattr(store, 'all_'+category)()], key=lambda v: v['id'])
data = {'schema': 'ciw.ppda-acquisition.v1', 'scope': source['configuration'],
    'source_definition': dict(source['source'], adapter_id='incremental-dataset', configuration={},
        capabilities=['incremental'], required_parameters=['path'], enabled=True),
    'plan': {'plan_id': plan.plan_id, 'source_id': plan.source_id, 'parameters': dict(plan.parameters),
        'enabled': True, 'schedule': None, 'mode': 'incremental', 'interval_seconds': None},
    'adapter_version': binding.version, 'runs': runs, 'evidence': evidence,
    'checkpoint': dataclasses.asdict(checkpoints.get(plan.plan_id)),
    'restored_pool_fingerprint': pool.fingerprint()}
print(json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False))
'''


def _native_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _check_data(source, data):
    """Check native identity formulas and lineage from the retained snapshot bytes.

This is an integrity check. Only the separately retained native reproduction
claims that the acquisition code actually reproduced the graph.
"""
    exact_keys(data, {"schema", "scope", "source_definition", "plan", "adapter_version", "runs", "evidence", "checkpoint", "restored_pool_fingerprint"})
    if data["schema"] != DATA_SCHEMA or data["scope"] != POLICY or data["adapter_version"] != ADAPTER_VERSION:
        raise ValueError("Invalid native acquisition scope or binding version")
    definition = dict(source["source"], adapter_id="incremental-dataset", configuration={}, capabilities=["incremental"], required_parameters=["path"], enabled=True)
    plan = {"plan_id": source["plan_id"], "source_id": source["source"]["source_id"], "parameters": {"path": _filename(source)}, "enabled": True, "schedule": None, "mode": "incremental", "interval_seconds": None}
    if data["source_definition"] != definition or data["plan"] != plan:
        raise ValueError("Native source registry/plan differs from retained declaration")
    native_source = {"kind": "incremental-dataset", "name": source["source"]["name"]}
    native_source["id"] = _native_hash(native_source)
    evidence = {"sources": [native_source], "documents": [], "records": [], "observations": []}
    checkpoint, seen, runs = None, -1, []
    for index, snapshot in enumerate(source["snapshots"]):
        raw = base64.b64decode(snapshot["bytes_b64"], validate=True)
        artifacts = []
        for row in _json(raw):
            if row["sequence"] <= seen:
                continue
            content = json.dumps(row, sort_keys=True, allow_nan=False)
            locator = _filename(source) + "#" + str(row["sequence"]).zfill(12)
            document = {"source_id": native_source["id"], "raw_content": content, "retrieval_method": "file:incremental_json_v1", "retrieved_at": snapshot["requested_at"]}
            document["id"] = _native_hash({"source_id": document["source_id"], "content_hash": _native_hash(content), "retrieval_method": document["retrieval_method"]})
            record = {"document_id": document["id"], "locator": locator, "raw_content": content}
            record["id"] = _native_hash(record)
            observation = {"record_ids": [record["id"]], "extraction_method": "json:local_dataset_v1", "content": row}
            observation["id"] = _native_hash(observation)
            observation.update(confidence=1.0, extracted_at=snapshot["requested_at"])
            artifacts.append({"artifact_id": _native_hash({"source_id": native_source["id"], "locator": locator}), "version_id": document["id"], "is_new": True, "locator": locator, "raw_content": content})
            evidence["documents"].append(document)
            evidence["records"].append(record)
            evidence["observations"].append(observation)
            seen = row["sequence"]
        fingerprints = {key: sorted(v["id"] for v in values) for key, values in evidence.items()}
        fingerprints.update(referents=[], claimed_relationships=[], derived_values=[], derived_groundings=[])
        next_checkpoint = {"plan_id": source["plan_id"], "source_id": source["source"]["source_id"], "position": str(seen).zfill(12), "updated_at": snapshot["requested_at"]}
        runs.append({"snapshot_index": index, "snapshot_sha256": byte_digest(raw), "requested_at": snapshot["requested_at"],
            "result": {"source_id": source["source"]["source_id"], "outcome": "acquired", "artifacts": artifacts, "admission_failures": [], "error": None},
            "checkpoint_before": checkpoint, "checkpoint_after": next_checkpoint, "pool_fingerprint": _native_hash(fingerprints)})
        checkpoint = next_checkpoint
    evidence = {key: sorted(values, key=lambda v: v["id"]) for key, values in evidence.items()}
    if canonical(data["runs"]) != canonical(runs) or canonical(data["evidence"]) != canonical(evidence) or data["checkpoint"] != checkpoint or data["restored_pool_fingerprint"] != runs[-1]["pool_fingerprint"]:
        raise ValueError("Native acquisition lineage/checkpoint/content binding mismatch")


class AcquisitionWorkflow(DeclaredWorkflow):
    def __init__(self):
        self.kind, self.role = "acquired-dataset", "ppda"
        self.ROLES = ROLES
        self.SOURCE_SCHEMA = SOURCE_SCHEMA
        self.schema, self.operation = "ciw.acquired-dataset-session.v1", "ciw.acquired-dataset.v1"
        self.pin = {"revision": PPDA_REVISION, "module": "daf.scheduling.runner", "source_root": "."}

    def _source(self, raw):
        return _source(raw)

    @staticmethod
    def _runtime_projection(runtime):
        value = {k: v for k, v in runtime.items() if k not in {"repository_root", "python_executable"}}
        if "vendor" in value:
            value["vendor"] = AcquisitionWorkflow._runtime_projection(value["vendor"])
        return value

    def _adapters(self, repositories, expected=None):
        if set(repositories) != ROLES:
            raise ValueError("Bind exactly the native PPDA checkout with its pinned vendor")
        retained = expected["ppda"] if expected else {}
        adapter = AcquisitionAdapter(repositories["ppda"], expected_python_sha256=retained.get("python_sha256"), expected_python_version=retained.get("python_version"), expected_dependencies=retained.get("dependencies"))
        runtime = adapter.runtime_identity()
        if retained and self._runtime_projection(runtime) != self._runtime_projection(retained):
            raise ValueError("Native acquisition runtime differs from retained pins")
        return adapter, runtime, None

    def _step(self, source, evidence_id, bound):
        adapter, runtime, _ = bound
        if self._runtime_projection(adapter.runtime_identity()) != self._runtime_projection(runtime):
            raise ValueError("Native acquisition runtime changed")
        code, raw = adapter._run(_BOOTSTRAP, [str(adapter.source_root), _filename(source)], canonical(source))
        adapter.runtime_identity()
        if code:
            raise AdapterRefusal("ACQUISITION_REFUSED", "Pinned PPDA refused the declared offline acquisition")
        data = _json(raw)
        _check_data(source, data)
        occurrence = "execution-" + uuid.uuid4().hex
        result = {"schema": RESULT_SCHEMA, "operation_id": self.operation, "execution_ref": occurrence, "input_refs": [evidence_id], "data": data, "authority": AUTHORITY}
        result["result_id"] = digest(result)
        numerical = {"operation_id": self.operation, "data": data}
        return {"runtime_ref": self.role, "operation_id": self.operation, "execution_id": occurrence, "input_refs": [evidence_id], "request": source, "request_sha256": digest(source), "result": result, "result_sha256": digest(result), "result_id": result["result_id"], "numerical_result": numerical, "numerical_result_id": digest(numerical)}

    def _validate_step(self, step, source, evidence_id):
        exact_keys(step, {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256", "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"})
        if step["runtime_ref"] != self.role or step["operation_id"] != self.operation or step["input_refs"] != [evidence_id] or canonical(step["request"]) != canonical(source) or not isinstance(step["execution_id"], str) or not re.fullmatch(r"execution-[a-f0-9]{32}", step["execution_id"]):
            raise ValueError("Acquisition request/execution/evidence mismatch")
        result = step["result"]
        exact_keys(result, {"schema", "operation_id", "execution_ref", "input_refs", "data", "authority", "result_id"})
        _check_data(source, result["data"])
        if result != {"schema": RESULT_SCHEMA, "operation_id": self.operation, "execution_ref": step["execution_id"], "input_refs": [evidence_id], "data": result["data"], "authority": AUTHORITY, "result_id": digest({k: v for k, v in result.items() if k != "result_id"})} or result["result_id"] != step["result_id"] or canonical(step["numerical_result"]) != canonical({"operation_id": self.operation, "data": result["data"]}):
            raise ValueError("Acquisition result/authority binding mismatch")
        for key, content in (("request_sha256", source), ("result_sha256", result), ("numerical_result_id", step["numerical_result"])):
            if step[key] != digest(content):
                raise ValueError("Acquisition step content mismatch")

    def _validate(self, bundle):
        try:
            exact_keys(bundle, {"schema", "session_id", "created_at", "source", "configuration", "runtimes", "steps", "bundle_digest", "verification"}, {"replay_receipts"})
            if len(canonical(bundle)) > MAX_BYTES or bundle["schema"] != self.schema or bundle["bundle_digest"] != bundle_digest(bundle) or not re.fullmatch(r"session-[a-f0-9]{32}", bundle["session_id"]):
                raise ValueError("Acquisition bundle identity mismatch")
            _text(bundle["created_at"])
            evidence, = bundle["source"]["evidence"]
            raw = base64.b64decode(evidence["bytes_b64"], validate=True)
            source = _source(raw)
            if evidence != {"artifact_ref": byte_digest(raw), "sha256": byte_digest(raw), "bytes_b64": base64.b64encode(raw).decode()} or bundle["source"] != {"experiment_id": source["experiment_id"], "experiment_digest": digest(source), "evidence": [evidence]} or bundle["configuration"] != POLICY:
                raise ValueError("Acquisition retained source mismatch")
            if set(bundle["runtimes"]) != ROLES:
                raise ValueError("Unexpected acquisition runtime")
            runtime = bundle["runtimes"]["ppda"]
            for value, pin, vendor in ((runtime, self.pin, True), (runtime["vendor"], {"revision": VENDOR_REVISION, "module": "evidence.types", "source_root": "."}, False)):
                exact_keys(value, {"schema", "adapter_version", "repository_root", "revision", "source_tree", "module", "source_root", "python_executable", "python_sha256", "python_version", "dependencies"} | ({"vendor"} if vendor else set()))
                if value["schema"] != "ciw.subprocess-runtime.v1" or any(value[k] != pin[k] for k in pin) or value["source_tree"] != SOURCE_TREES[pin["revision"]] or not re.fullmatch("[a-f0-9]{64}", value["python_sha256"]) or not isinstance(value["dependencies"], dict):
                    raise ValueError("Unapproved acquisition provider/vendor pin")
                for field in ("adapter_version", "repository_root", "python_executable", "python_version"):
                    _text(value[field])
            step, = bundle["steps"]
            self._validate_step(step, source, evidence["artifact_ref"])
            self._check_verification(bundle, bundle["verification"], source, evidence["artifact_ref"])
            return raw
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
            raise ValueError("Malformed acquisition session") from exc


_workflow = AcquisitionWorkflow()
workflow = _workflow
_validate = _workflow._validate
_adapters = _workflow._adapters
create_session = _workflow.create_session
replay_session = _workflow.replay_session
