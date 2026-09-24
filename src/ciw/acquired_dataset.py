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

from .adapters.ppda_acquisition import AcquisitionAdapter, VENDOR_REVISION
from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import _json
from .pipelines import provider_pin
from .pipelines.runner import RUNTIME_FIELDS, PipelineRunner, text as _text
from .core.canonical import canonical, byte_digest, exact_keys

SOURCE_SCHEMA = "ciw.acquired-dataset-source.v1"
POLICY = {"mode": "incremental", "source_format": "json_records_with_declared_sequence",
          "replay_scope": "isolated_offline_reacquisition", "event_time_order": "not_inferred",
          "sensor_fusion": "not_performed"}
DATA_SCHEMA = "ciw.ppda-acquisition.v1"
ADAPTER_VERSION = "352e01125ba1fe5c751ab78bd70740f8ab999590c8ef253068d07213d7a7cdf7"
# The SCOUT vendor is a gitlink inside the PPDA checkout, pinned by that commit.
VENDOR_SOURCE_TREE = "145f0617b2da7e4d391985f0a03fb58f408992ee"
VENDOR_PIN = {"revision": VENDOR_REVISION, "source_tree": VENDOR_SOURCE_TREE,
              "module": "evidence.types", "source_root": "."}


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


class AcquisitionWorkflow(PipelineRunner):
    LABEL = "Acquisition"
    RUNTIME_EXTRA = frozenset({"vendor"})

    def __init__(self):
        super().__init__("acquired-dataset", provider_pin("acquired-dataset"))

    def parse_source(self, raw):
        return _source(raw)

    def make_adapter(self, repository, retained):
        return AcquisitionAdapter(repository, expected_python_sha256=retained.get("python_sha256"),
                                  expected_python_version=retained.get("python_version"),
                                  expected_dependencies=retained.get("dependencies"))

    def check_runtime(self, runtime):
        vendor = runtime["vendor"]
        exact_keys(vendor, RUNTIME_FIELDS)
        if (vendor["schema"] != "ciw.subprocess-runtime.v1" or
                any(vendor[key] != value for key, value in VENDOR_PIN.items()) or
                not isinstance(vendor["python_sha256"], str) or not re.fullmatch("[a-f0-9]{64}", vendor["python_sha256"]) or
                not isinstance(vendor["dependencies"], dict)):
            raise ValueError("Unapproved acquisition provider/vendor pin")
        for field in ("adapter_version", "repository_root", "python_executable", "python_version"):
            _text(vendor[field])

    def invoke(self, source, bound):
        adapter = bound[0]
        code, raw = self.run_provider(bound, _BOOTSTRAP, [str(adapter.source_root), _filename(source)], canonical(source))
        if code:
            raise AdapterRefusal("ACQUISITION_REFUSED", "Pinned PPDA refused the declared offline acquisition")
        return _json(raw)

    def check_data(self, source, data):
        _check_data(source, data)


_workflow = AcquisitionWorkflow()
workflow = _workflow
_validate = _workflow._validate
_adapters = _workflow._adapters
create_session = _workflow.create_session
replay_session = _workflow.replay_session
