"""Typed exchange producer -> native CIW adapter and workspace replay."""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

import pytest

from ciw import exchange_adapter as adapter
from ciw.cli import parser
from ciw.instruments import make_demo_run
from ciw.session import Session
from ciw.workbench import Workbench


ROOT = Path(__file__).resolve().parents[1]


def _source():
    artifact = json.loads((ROOT / "examples/exchange/observation.json").read_text(encoding="utf-8"))
    return {"schema": adapter.SOURCE_SCHEMA,
            "producer": {"name": "synthetic-producer", "revision": "fixture-1",
                          "operation_ids": ["bridge.instrumentation.observation_batch_v1"]},
            "artifacts": [artifact]}


def _runtime_fixture(monkeypatch):
    identity = {"repository": adapter.PIN["repository"], "revision": adapter.PIN["revision"],
                "source_tree": "fixture-tree", "module": adapter.PIN["path"],
                "source_sha256": adapter.PIN["sha256"],
                "execution_scope": "standalone_checked_source_only"}
    monkeypatch.setattr(adapter, "_runtime", lambda repo: dict(identity))
    path = Path(os.environ.get("CIW_SET_REPO", ROOT))
    return path


def _bind_and_add(workbench, monkeypatch):
    if not os.environ.get("CIW_SET_REPO"):
        pytest.skip("set CIW_SET_REPO to the pinned State Estimation Evaluation Testbed checkout")
    provider = _runtime_fixture(monkeypatch)
    workbench.bind_workflow("instrument-exchange", {"set": provider})
    raw = adapter.canonical(_source())
    descriptor = workbench.add_source({"kind": "instrument-exchange", "label": "typed exchange fixture",
                                       "bytes_b64": base64.b64encode(raw).decode("ascii")})
    return provider, raw, descriptor


def test_structural_source_rejects_duplicate_producer_identity():
    source = _source()
    source["artifacts"] = [source["artifacts"][0], source["artifacts"][0]]
    with pytest.raises(ValueError, match="duplicate"):
        adapter._source(adapter.canonical(source))


def test_native_result_keeps_operation_execution_and_producer_identities(monkeypatch):
    workbench = Workbench()
    provider, raw, descriptor = _bind_and_add(workbench, monkeypatch)
    summary = workbench.execute({"operation_id": adapter.OPERATION, "source_id": descriptor["source_id"]})
    native = workbench.get_bundle(summary["bundle_id"])
    step = native["steps"][0]
    producer_id = step["result"]["producer_artifact_ids"][0]
    assert step["operation_id"] == adapter.OPERATION
    assert step["execution_id"] != producer_id
    assert step["result_id"] != producer_id
    assert step["numerical_result_id"].startswith("sha256:")
    assert native["authority"] == {"may_authorize": False, "state_admission": "not_performed",
                                   "physical_validation": "not_established"}
    assert adapter._validate(native) == raw
    replay = workbench.replay({"bundle_id": summary["bundle_id"]})
    fresh = workbench.get_bundle(replay["bundle"]["bundle_id"])
    assert replay["replay_receipt"]["numerical_match"] is True
    assert fresh["steps"][0]["execution_id"] != step["execution_id"]
    assert fresh["steps"][0]["result_id"] != step["result_id"]
    assert fresh["steps"][0]["numerical_result_id"] == step["numerical_result_id"]


def test_native_exchange_survives_save_reopen_and_replay(monkeypatch, tmp_path):
    if not os.environ.get("CIW_SET_REPO"):
        pytest.skip("set CIW_SET_REPO to the pinned State Estimation Evaluation Testbed checkout")
    # Keep the session files one level above the per-test directory.  This
    # avoids Windows pytest cleanup races with the atomic hidden-file writer.
    session_root = tmp_path.parent.parent / "exchange-session"
    session = Session(make_demo_run(), session_root / "original")
    provider, raw, descriptor = _bind_and_add(session.workbench, monkeypatch)
    summary = session.workbench.execute({"operation_id": adapter.OPERATION, "source_id": descriptor["source_id"]})
    workspace_path = session.save_workspace(session_root / "workspace.json")
    reopened = Session.from_workspace(workspace_path, session_root / "reopened")
    retained = reopened.workbench.get_bundle(summary["bundle_id"])
    assert adapter._validate(retained) == raw
    monkeypatch.setattr(adapter, "_runtime", lambda repo: {
        "repository": adapter.PIN["repository"], "revision": adapter.PIN["revision"],
        "source_tree": "fixture-tree", "module": adapter.PIN["path"],
        "source_sha256": adapter.PIN["sha256"], "execution_scope": "standalone_checked_source_only"})
    reopened.workbench.bind_workflow("instrument-exchange", {"set": provider})
    replay = reopened.workbench.replay({"bundle_id": summary["bundle_id"]})
    assert replay["replay_receipt"]["numerical_match"] is True
    assert replay["bundle"]["bundle_id"] != summary["bundle_id"]


def test_provider_revision_mismatch_is_refused(tmp_path):
    # A standalone checkout at another commit, so the refusal cannot depend on
    # whether the temporary directory happens to sit inside a Git work tree.
    fake = tmp_path / "provider"
    fake.mkdir()
    git = ["git", "-C", str(fake), "-c", "user.name=CIW", "-c", "user.email=ciw@example.invalid",
           "-c", "commit.gpgsign=false"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "unpinned"], check=True)
    with pytest.raises(ValueError, match="revision"):
        adapter._runtime(fake)


def test_terminal_parser_exposes_the_exact_exchange_binding():
    args = parser().parse_args(["serve", "--exchange-set-repo", "trusted/set"])
    assert args.exchange_set_repo == Path("trusted/set")


def test_operator_binding_reports_the_exact_set_pin():
    # Keep the local fixture checkout useful for adapter tests while reserving
    # this assertion for a checkout that was explicitly prepared at the
    # published revision.  CI sets this second variable to its exact checkout.
    provider = os.environ.get("CIW_SET_PINNED_REPO")
    if not provider:
        pytest.skip("set CIW_SET_PINNED_REPO to an exact SET pin checkout")
    identity = adapter._runtime(Path(provider))
    assert identity["repository"] == adapter.PIN["repository"]
    assert identity["revision"] == adapter.PIN["revision"]
    assert identity["module"] == adapter.PIN["path"]
    assert identity["source_sha256"] == adapter.PIN["sha256"]
