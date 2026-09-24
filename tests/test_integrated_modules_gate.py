"""One installed session must retain the real native modules and their replay."""
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy

import pytest

from ciw.instruments import make_demo_run
from ciw.session import Session
from ciw.core.canonical import canonical

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("schematic-companions", "bim-quantity", "acquired-dataset")


def call(session, kind, payload=None, *, error=False):
    result = session.handle({"protocol_version": 1, "request_id": "integrated-gate", "type": kind, "payload": payload or {}})
    assert result["type"] == ("error" if error else "response"), result
    return result["payload"]


def retain(session, kind, raw, *, upstream=None):
    source = call(session, "source.add", {"kind": kind, "label": kind, "bytes_b64": base64.b64encode(raw).decode()})
    assert base64.b64decode(call(session, "source.get", {"source_id": source["source_id"]})["bytes_b64"]) == raw
    parameters = {"source_id": source["source_id"]}
    if upstream is not None:
        parameters["upstream_bundle_id"] = upstream
    result = call(session, "operation.execute", {"operation_id": "ciw." + kind + ".v1", "parameters": parameters})
    return call(session, "bundle.get", {"bundle_id": result["bundle_id"]})


def output(kind, name, value):
    directory = os.environ.get("CIW_INTEGRATED_FIXTURE_DIR")
    if directory:
        target = Path(directory) / kind
        target.mkdir(parents=True, exist_ok=True)
        (target / (name + ".json")).write_bytes(value if isinstance(value, bytes) else canonical(value))


@pytest.fixture(scope="module")
def retained(tmp_path_factory):
    stack = os.environ.get("CIW_INTEGRATED_STACK_ROOT")
    if not stack:
        pytest.skip("set CIW_INTEGRATED_STACK_ROOT for exact native integrated-module execution")
    providers = Path(stack)
    directory = tmp_path_factory.mktemp("integrated-modules")
    session = Session(make_demo_run(), directory)
    session.workbench.bind_workflow("schematic-assessment", {"sra": providers / "sra"})
    for kind, roles in ((KINDS[0], ("sra", "jspt", "plsr")), (KINDS[1], ("cse",)), (KINDS[2], ("ppda",))):
        session.workbench.bind_workflow(kind, {role: providers / role for role in roles})

    companion = json.loads((ROOT / "examples/declared-workloads/schematic-companions.json").read_bytes())
    assessment = json.loads((ROOT / "examples/declared-workloads/schematic-assessment.json").read_bytes())
    assessment["schematic"] = deepcopy(companion["schematic"])
    assessment_raw = canonical(assessment)
    upstream = retain(session, "schematic-assessment", assessment_raw)
    companion["schematic"] = deepcopy(upstream["steps"][0]["result"]["data"]["schematic"])
    output(KINDS[0], "upstream-source", assessment_raw)
    output(KINDS[0], "upstream", upstream)
    bim = runpy.run_path(str(ROOT / "examples/bim-quantity/make_source.py"))["source"]()
    sources = {KINDS[0]: b"\n" + canonical(companion) + b"\n ", KINDS[1]: canonical(bim),
               KINDS[2]: (ROOT / "examples/acquired-dataset/source.json").read_bytes()}
    bundles = {}
    for kind, raw in sources.items():
        original = retain(session, kind, raw, upstream=upstream["bundle_digest"] if kind == KINDS[0] else None)
        replay = call(session, "bundle.replay", {"bundle_id": original["bundle_digest"]})
        fresh = call(session, "bundle.get", {"bundle_id": replay["bundle"]["bundle_id"]})
        bundles[kind] = original, fresh
        output(kind, "source", raw)
        output(kind, "original", original)
        output(kind, "replay", fresh)

    alternatives = {}
    for name, change in (("held", {"cross_covariance_policy": "unknown"}), ("refused", {"value": -1.0, "variance": 1e-8})):
        declaration = deepcopy(bim)
        observation = json.loads(base64.b64decode(declaration["observation_bytes_b64"]))
        observation.update(change)
        declaration["observation_bytes_b64"] = base64.b64encode(canonical(observation)).decode()
        raw = canonical(declaration)
        alternatives[name] = retain(session, "bim-quantity", raw)
        output("bim-quantity", name + "-source", raw)
        output("bim-quantity", name, alternatives[name])

    geographic_raw = (ROOT / "examples/workbench/geographic-context.json").read_bytes()
    geographic = call(session, "source.add", {"kind": "geographic-context", "label": "Declared geographic context", "bytes_b64": base64.b64encode(geographic_raw).decode()})
    spatial = call(session, "experiment.inspect", {"view": "spatial", "source_id": geographic["source_id"]})
    output("geographic-context", "source", geographic_raw)
    output("geographic-context", "spatial-view", spatial)
    workspace = session.save_workspace(directory / "workspace.json")
    return session, bundles, alternatives, geographic, spatial, workspace


def test_native_modules_share_one_catalog_and_replay(retained):
    session, bundles, _, _, _, _ = retained
    assert len(call(session, "bundle.list")["bundles"]) == 9
    assert session.workbench.fusion_contexts() == []
    for kind, (original, fresh) in bundles.items():
        assert original["bundle_digest"] != fresh["bundle_digest"]
        assert original["source"] == fresh["source"]
        assert original["configuration"] == fresh["configuration"]
        for before, after in zip(original["steps"], fresh["steps"], strict=True):
            assert before["numerical_result_id"] == after["numerical_result_id"]
            assert before["execution_id"] != after["execution_id"]
            assert before["result_id"] != after["result_id"]
            assert call(session, "result.get", {"result_id": before["result_id"]}) == before["result"]
        view = call(session, "experiment.inspect", {"bundle_id": original["bundle_digest"]})
        assert view["fusion_context"] is None
        assert view["object_context"]["sensor_fusion"] == "not_performed"
        assert original["verification"]["independent"] is False
    companion = bundles["schematic-companions"][0]
    assert companion["upstream_assessment"]["bundle_digest"] in {row["bundle_id"] for row in call(session, "bundle.list")["bundles"]}


def test_bim_unknown_covariance_and_native_invariant_rejection_preserve_world(retained):
    _, bundles, alternatives, _, _, _ = retained
    accepted = bundles["bim-quantity"][0]["steps"][0]["result"]["data"]
    assert accepted["status"] == "accepted" and accepted["reason"] == "conditioned"
    for label, bundle in alternatives.items():
        data = bundle["steps"][0]["result"]["data"]
        assert data["status"] == label
        assert data["reason"] == ("unknown_cross_covariance" if label == "held" else "invariant_rejection")
        assert data["prior"] == data["posterior"]
        assert data["ledger_replay"]["world_digest"] == data["posterior"]["world_digest"]


def test_geographic_source_is_read_only_declared_context(retained):
    session, _, _, geographic, spatial, _ = retained
    assert spatial["schema"] == "ciw.spatial-view.v1"
    assert spatial["source"] == call(session, "source.get", {"source_id": geographic["source_id"]})
    assert spatial["coordinate_frame"]["id"] == "OGC:CRS84"
    assert spatial["authority"]["read_only"] is True
    assert spatial["authority"]["execution"] == "not_performed"
    assert spatial["authority"]["state_admission"] == "not_performed"
    assert not any(row["operation_id"] == "ciw.geographic-context.v1" for row in call(session, "operation.list")["operations"])


def test_restore_retains_modules_without_executing_or_binding_providers(retained, tmp_path):
    session, bundles, _, geographic, spatial, workspace = retained
    restored = Session.from_workspace(workspace, tmp_path)
    assert restored.workbench.serialize() == session.workbench.serialize()
    assert {o["operation_id"] for o in restored.workbench.describe_operations() if o["available"]} == {"ciw.energy-accuracy.v1", "ciw.encoder-position.v1", "ciw.thermal-observer.v1"}
    assert call(restored, "experiment.inspect", {"view": "spatial", "source_id": geographic["source_id"]}) == spatial
    for original, _ in bundles.values():
        assert call(restored, "bundle.replay", {"bundle_id": original["bundle_digest"]})["status"] == "refused"
    assert restored.workbench.pending_operations == 0
