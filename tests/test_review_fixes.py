"""Findings from the branch review: malformed inputs report instead of crash, bounds and exactness hold."""

import base64
import json
import math
from pathlib import Path

import pytest

from ciw import reference_workflow as base
from ciw import thermal_contract, thermal_workflow, workspace_verify
from ciw.adapters.protocol import AdapterRefusal
from ciw.cli import main
from ciw.workbench import Workbench

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "retained" / "workbench.json"
EXAMPLES = {
    "uncertainty-validation": ROOT / "examples" / "uncertainty-validation" / "consistent.json",
    "project-graph": None,
    "machine-manifest": ROOT / "examples" / "machine-manifest" / "source.json",
    "energy-accuracy": ROOT / "examples" / "energy-accuracy" / "baseline.json",
    "thermal-observer": ROOT / "examples" / "thermal-observer" / "source.json",
}


@pytest.mark.parametrize("retained", [
    {"schema": "ciw.retained-workbench.v1", "revision": 1, "sources": [], "bundles": 5},
    {"schema": "ciw.retained-workbench.v1", "revision": 0, "sources": None, "bundles": None},
    {"schema": "ciw.retained-workbench.v1", "revision": 2, "sources": [1], "bundles": ["x", {"kind": 3}]},
])
def test_malformed_retained_catalogs_yield_an_invalid_report_not_a_traceback(tmp_path, retained, capsys):
    path = tmp_path / "retained.json"
    path.write_text(json.dumps(retained), encoding="utf-8")
    report = workspace_verify.verify(path)
    assert report["status"] == "invalid" and report["refusal"]
    assert all(entry["validation"] == "invalid" for entry in report["bundles"])
    assert main(["workspace", "verify", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"


def test_a_workspace_whose_workbench_is_not_an_object_is_refused_as_malformed(tmp_path):
    from ciw.instruments import make_demo_run
    from ciw.session import Session
    seed = Session(make_demo_run(), tmp_path / "session")
    saved = json.loads(seed.save_workspace(tmp_path / "seed.json").read_text(encoding="utf-8"))
    for bad in ([1, 2], "workbench", None):
        saved.update({"workspace_version": 3, "workbench": bad})
        path = tmp_path / "workspace.json"
        path.write_text(json.dumps(saved), encoding="utf-8")
        with pytest.raises(ValueError):
            Session.from_workspace(path, tmp_path / "reopened")
        report = workspace_verify.verify(path)
        assert report["status"] == "invalid" and report["bundles"] == []
    for bad in (None, [1], "x"):
        with pytest.raises(ValueError, match="Malformed retained workbench"):
            Workbench.restore(bad)


@pytest.mark.parametrize("kind", sorted(EXAMPLES))
def test_bad_json_uploads_are_source_errors_not_provider_runtime_refusals(kind):
    workbench = Workbench()
    for raw in (b'{"a": 1, "a": 2}', b'{"value": NaN}', b"\xff\xfe", b"[1, 2"):
        payload = {"kind": kind, "label": "bad upload", "bytes_b64": base64.b64encode(raw).decode("ascii")}
        with pytest.raises(ValueError):
            workbench.add_source(payload)
        try:
            workbench.add_source(payload)
        except AdapterRefusal as exc:  # pragma: no cover - the defect this test guards against
            pytest.fail(f"{kind}: bad upload reported as runtime refusal {exc.code}")
        except ValueError:
            pass
    assert workbench.list_sources() == []


def test_close_data_keeps_integers_exact_and_floats_tolerant():
    base.close_data({"count": 2_000_000_000, "value": 1.0}, {"count": 2_000_000_000, "value": 1.0 + 1e-12})
    base.close_data([0, 0.0], [0.0, 0])
    with pytest.raises(ValueError, match="differs numerically"):
        base.close_data({"count": 2_000_000_000}, {"count": 2_000_000_001})
    with pytest.raises(ValueError, match="differs numerically"):
        base.close_data(1.0, 1.0 + 1e-6)
    with pytest.raises(ValueError, match="differs from"):
        base.close_data(True, 1)


def test_kernel_probe_covers_the_c_math_library(monkeypatch):
    probe = base._kernel_probe()
    lgamma = math.lgamma
    monkeypatch.setattr(math, "lgamma", lambda value: lgamma(value) * (1 + 2 ** -52))
    assert base._kernel_probe() != probe
    monkeypatch.undo()
    erf = math.erf
    monkeypatch.setattr(math, "erf", lambda value: erf(value) + 2 ** -60)
    assert base._kernel_probe() != probe


def test_thermal_bundle_budget_holds_a_contract_sized_result():
    assert thermal_workflow.MAX_BYTES >= 4 * thermal_contract.RESULT_LIMIT
    assert thermal_workflow.ThermalWorkflow.MAX_BYTES == thermal_workflow.MAX_BYTES


def test_source_add_refuses_an_oversized_file_before_contacting_the_session(tmp_path, capsys):
    from ciw.cli import SOURCE_FRAME_LIMIT
    huge = tmp_path / "huge.json"
    with huge.open("wb") as stream:
        stream.seek(SOURCE_FRAME_LIMIT)
        stream.write(b"}")
    assert main(["source", "add", "--kind", "energy-accuracy", "--file", str(huge), "--url", "ws://127.0.0.1:9"]) == 2
    assert "8 MiB" in capsys.readouterr().err


def test_execution_runs_the_reference_once_per_occurrence(monkeypatch):
    from ciw import project_workflow
    calls = []
    original = project_workflow.ProjectGraphWorkflow._native_data
    monkeypatch.setattr(project_workflow.ProjectGraphWorkflow, "_native_data",
                        lambda self, source: calls.append(1) or original(self, source))
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    source = next(s for s in raw["sources"] if s["kind"] == "project-graph")
    workflow = project_workflow.ProjectGraphWorkflow()
    bundle = workflow.create_session(base64.b64decode(source["bytes_b64"]), {})
    assert len(calls) == 2, "one reference computation for the occurrences, one independent check"
    calls.clear()
    workflow._validate(bundle)
    assert len(calls) == 1
