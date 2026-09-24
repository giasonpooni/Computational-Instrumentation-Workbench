"""Retained workspaces from earlier code must reopen without executing a provider.

``tests/fixtures/retained/workbench.json`` is a committed snapshot of one
executed bundle per provider-free kind, built from the committed example inputs
by ``tests/fixtures/retained/generate.py``. These tests are the compatibility
gate for identity-critical changes: the current code must validate every
retained bundle against its deterministic reference, keep every retained
identity, and replay each bundle either as a fresh occurrence with the same
numerical identity or refuse solely because the reference runtime identity
changed. Regenerating the snapshot is a deliberate, stated format change.
"""

import base64
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from ciw.instruments import make_demo_run
from ciw.session import Session
from ciw.telemetry import canonical
from ciw.workbench import Workbench, _workflow

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "retained"
KINDS = {"energy-accuracy", "thermal-observer", "machine-manifest", "project-graph", "uncertainty-validation"}


def _load(name):
    return json.loads((FIXTURE / name).read_text(encoding="utf-8"))


def _call(session, kind, payload):
    response = session.handle({"protocol_version": 1, "request_id": "compat-" + kind, "type": kind, "payload": payload})
    assert response["type"] == "response", response
    return response["payload"]


def _first_number(node, path=()):
    """Path to the first numeric leaf of a retained data object."""
    if isinstance(node, dict):
        for key in sorted(node):
            found = _first_number(node[key], path + (key,))
            if found is not None:
                return found
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found = _first_number(item, path + (index,))
            if found is not None:
                return found
    elif type(node) in (int, float):
        return path
    return None


@pytest.fixture(scope="module")
def retained():
    return _load("workbench.json")


@pytest.fixture(scope="module")
def manifest():
    return _load("manifest.json")


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("retained_fixture_generator", FIXTURE / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_pins_every_retained_identity_once(retained, manifest):
    assert manifest["schema"] == "ciw.retained-compatibility-fixture.v1"
    entries = {entry["kind"]: entry for entry in manifest["bundles"]}
    records = {record["kind"]: record for record in retained["bundles"]}
    assert set(entries) == set(records) == KINDS
    assert len(manifest["bundles"]) == len(retained["bundles"]) == len(retained["sources"]) == len(KINDS)
    for kind, entry in entries.items():
        record = records[kind]
        step = record["native"]["steps"][0]
        assert record["bundle_id"] == entry["bundle_id"] == record["native"]["bundle_digest"]
        assert record["source_id"] == entry["source_id"]
        assert step["execution_id"] == entry["execution_id"]
        assert step["result_id"] == entry["result_id"]
        assert step["numerical_result_id"] == entry["numerical_result_id"]
        runtime = record["native"]["runtimes"][step["runtime_ref"]]
        assert runtime.get("algorithm", runtime)["code_sha256"] == entry["code_sha256"]


def test_retained_sources_are_the_committed_example_inputs(retained, manifest, generator):
    sources = {source["kind"]: source for source in retained["sources"]}
    for entry in manifest["bundles"]:
        assert generator.INPUTS[entry["kind"]] == entry["input"]
        assert base64.b64decode(sources[entry["kind"]]["bytes_b64"], validate=True) == generator.input_bytes(entry["kind"])


def test_retained_workbench_reopens_without_a_provider_or_a_process(retained, manifest, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("Reopening started a process"))
    restored = Workbench.restore(retained)
    assert restored.serialize() == retained
    assert restored.pending_operations == 0
    available = {row["operation_id"] for row in restored.describe_operations() if row["available"]}
    for entry in manifest["bundles"]:
        assert entry["operation_id"] in available
        native = restored.get_bundle(entry["bundle_id"])
        assert native["steps"][0]["numerical_result_id"] == entry["numerical_result_id"]
        view = restored.inspect_experiment({"bundle_id": entry["bundle_id"]})
        assert view["panels"] and view["object_context"], entry["kind"]
    assert restored.serialize() == retained


def test_every_retained_bundle_validates_against_the_current_reference(retained):
    sources = {source["source_id"]: source for source in retained["sources"]}
    for record in retained["bundles"]:
        raw = _workflow(record["kind"])._validate(record["native"])
        assert raw == base64.b64decode(sources[record["source_id"]]["bytes_b64"], validate=True), record["kind"]


def test_fresh_execution_of_the_retained_sources_reproduces_the_numerical_identities(retained, manifest):
    sources = {source["source_id"]: source for source in retained["sources"]}
    for entry in manifest["bundles"]:
        raw = base64.b64decode(sources[entry["source_id"]]["bytes_b64"], validate=True)
        fresh = _workflow(entry["kind"]).create_session(raw, {})
        assert fresh["steps"][0]["numerical_result_id"] == entry["numerical_result_id"], entry["kind"]
        assert fresh["steps"][0]["execution_id"] != entry["execution_id"]
        assert fresh["bundle_digest"] != entry["bundle_id"]


def test_replay_is_a_fresh_matching_occurrence_or_refused_on_runtime_identity_alone(retained, manifest):
    restored = Workbench.restore(retained)
    outcomes = {}
    for entry in manifest["bundles"]:
        native = restored.get_bundle(entry["bundle_id"])
        step = native["steps"][0]
        current = _workflow(entry["kind"])._runtime_identity()
        if canonical(current) == canonical(native["runtimes"][step["runtime_ref"]]):
            replayed = restored.replay({"bundle_id": entry["bundle_id"]})
            fresh = restored.get_bundle(replayed["bundle"]["bundle_id"])
            assert fresh["steps"][0]["numerical_result_id"] == entry["numerical_result_id"]
            assert fresh["steps"][0]["execution_id"] != entry["execution_id"]
            assert replayed["replay_receipt"]["source_bundle_digest"] == entry["bundle_id"]
            assert replayed["replay_receipt"]["numerical_match"] is True
            outcomes[entry["kind"]] = "replayed"
        else:
            with pytest.raises(ValueError, match="runtime identity differs"):
                restored.replay({"bundle_id": entry["bundle_id"]})
            assert restored.get_bundle(entry["bundle_id"]) == native
            outcomes[entry["kind"]] = "refused"
    assert set(outcomes) == KINDS
    assert restored.pending_operations == 0


def test_version_3_workspace_reopens_and_serves_the_retained_bundles(retained, manifest, tmp_path):
    seed = Session(make_demo_run(), tmp_path / "seed")
    workspace = json.loads(seed.save_workspace(tmp_path / "seed.json").read_text(encoding="utf-8"))
    workspace.update({"workspace_version": 3, "workbench": retained})
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps(workspace, allow_nan=False), encoding="utf-8")
    reopened = Session.from_workspace(path, tmp_path / "reopened")
    assert reopened.workbench.serialize() == retained
    listed = {row["bundle_id"]: row for row in _call(reopened, "bundle.list", {})["bundles"]}
    assert set(listed) == {entry["bundle_id"] for entry in manifest["bundles"]}
    for entry in manifest["bundles"]:
        assert listed[entry["bundle_id"]]["kind"] == entry["kind"]
        assert _call(reopened, "bundle.get", {"bundle_id": entry["bundle_id"]})["bundle_digest"] == entry["bundle_id"]
        assert _call(reopened, "experiment.inspect", {"bundle_id": entry["bundle_id"]})["panels"]
    saved = json.loads(reopened.save_workspace(tmp_path / "again.json").read_text(encoding="utf-8"))
    assert saved["workspace_version"] == 3 and saved["workbench"] == retained


def test_a_changed_retained_number_or_source_byte_refuses_the_whole_reopen(retained):
    for record in retained["bundles"]:
        changed = deepcopy(retained)
        target = next(item for item in changed["bundles"] if item["bundle_id"] == record["bundle_id"])
        data = target["native"]["steps"][0]["result"]["data"]
        path = _first_number(data)
        assert path is not None, record["kind"]
        parent = data
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = parent[path[-1]] + 1
        with pytest.raises(ValueError):
            Workbench.restore(changed)
    changed = deepcopy(retained)
    raw = bytearray(base64.b64decode(changed["sources"][0]["bytes_b64"], validate=True))
    raw[-2] ^= 0x01
    changed["sources"][0]["bytes_b64"] = base64.b64encode(bytes(raw)).decode("ascii")
    with pytest.raises(ValueError):
        Workbench.restore(changed)
