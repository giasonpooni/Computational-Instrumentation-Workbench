"""Exchange bundles, providers and visibility tasks T091-T100 against the real CIW code.

Provider-backed tests are opt-in: CIW_LAB_SCR_REPO (clean SCR checkout at the
declared-workload pin) with CIW_LAB_SCR_ENGINE or cargo on PATH;
CIW_LAB_SET_REPO, CIW_LAB_PPDA_REPO and CIW_LAB_SCR_EXCHANGE_REPO add the
exchange integrations. Everything else runs offline with NumPy only.
"""
import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ciw.lab import exchange_provenance_bundles as section
from ciw.lab import exchange_provenance_bundles_fixtures as fixtures
from ciw.lab import exchange_provenance_bundles_providers as providers
from ciw.lab import registry, runner
from ciw.lab.evidence import validate_finding
from ciw.lab.report import build_report, render_markdown

ROOT = Path(__file__).resolve().parents[1]
QUEUE = {item["id"]: item for item in registry.load_queue()["tasks"]}


def run(task_id, directory, providers_bound=None):
    report = runner.run_task(QUEUE[task_id], registry._REGISTRY[task_id], runner.Context(directory, providers_bound), {})
    (directory / "reports").mkdir(parents=True, exist_ok=True)
    (directory / "reports" / f"{task_id}.json").write_text(runner.dumps(report), encoding="utf-8")
    return report


def claim(report, prefix):
    return next(f for f in report["findings"] if f["claim"].startswith(prefix))


def checks_pass(record):
    return all(check["passed"] for check in record["basis"].get("checks", []))


def test_every_section_task_is_registered_with_tests_in_this_file():
    for number in range(91, 101):
        implementation = registry._REGISTRY[f"T{number:03d}"]
        assert implementation.regression_tests
        for node in implementation.regression_tests:
            path, name = node.split("::")
            assert path == "tests/test_lab_exchange_provenance_bundles.py" and name in globals(), node


def test_embedded_examples_are_the_repository_examples():
    examples = ROOT / "examples"
    if not examples.is_dir():
        pytest.skip("examples/ is not beside the tests")
    for name, digest in fixtures.EXAMPLE_SHA256.items():
        assert fixtures.example_bytes(name) == (examples / name).read_bytes()
        assert sha256(fixtures.example_bytes(name)).hexdigest() == digest


def test_execution_guard_restores_every_patched_attribute():
    from importlib import import_module
    before = {}
    for module, qualname, attribute in fixtures.EXECUTION_PATHS + fixtures.RECOMPUTATION_PATHS:
        target = import_module(module) if qualname is None else getattr(import_module(module), qualname)
        before[(module, qualname, attribute)] = (target, attribute in vars(target), getattr(target, attribute))
    with pytest.raises(RuntimeError, match="boom"):
        with fixtures.execution_guard() as guard:
            with pytest.raises(fixtures.ExecutionForbidden):
                subprocess.Popen([sys.executable, "-c", "pass"])
            assert guard["attempts"] == ["subprocess.Popen"]
            raise RuntimeError("boom")
    for key, (target, own, value) in before.items():
        assert (key[2] in vars(target)) == own and getattr(target, key[2]) is value, key


def test_t091_reopen_needs_no_provider_and_reaches_no_execution_path(tmp_path):
    report = run("T091", tmp_path)
    assert report["state"] == "completed"
    primary = report["findings"][0]
    assert primary["evidence_status"] == "numerically_verified" and checks_pass(primary)
    assert primary["value"]["execution_path_calls"] == 0 and primary["value"]["provider_bindings"] == []
    assert primary["value"]["available_operations"] == ["ciw.energy-accuracy.v1"]
    assert primary["value"]["bundles"] == 2 and primary["value"]["results"] == 3
    assert claim(report, "The execution guard intercepts")["value"] == "intercepted"
    recomputed = claim(report, "Reopening recomputes")
    assert recomputed["value"] >= 1 and recomputed["counterexample"]["statement"].startswith("Reopening a saved")
    assert claim(report, "After reopen")["value"] is True


def test_t092_unbound_replay_and_execution_are_refused(tmp_path):
    report = run("T092", tmp_path)
    assert report["state"] == "completed"
    primary = report["findings"][0]
    assert primary["evidence_status"] == "numerically_verified"
    assert primary["value"]["requests"] == primary["value"]["refused_as_expected"] == 13
    assert "operation_unavailable" in primary["value"]["codes"]
    cases = json.loads((tmp_path / "artifacts" / "T092" / "refusals.json").read_text(encoding="utf-8"))
    assert cases["provider_paths_reached"] == [] and cases["reopen_attempts"] == []
    assert claim(report, "A content-consistent")["evidence_status"] == "not_established"


def test_fabricated_heat_bundle_is_content_consistent_but_wrong():
    from ciw.workbench import Workbench
    catalog = fixtures.fabricated_heat_catalog([0, 1, 2, 3, 0])
    assert fixtures.fabricated_heat_catalog([0, 1, 2, 3, 0]) == catalog  # deterministic
    restored = Workbench.restore(catalog)
    native = restored.get_bundle(catalog["bundles"][0]["bundle_id"])
    assert native["steps"][0]["result"]["data"]["values"] == [0, 1, 2, 3, 0]
    assert fixtures.heat_reference([0, 0, 64, 0, 0], 2) == [0, 16, 24, 16, 0]
    # A non-resealed edit is still caught: validation checks consistency, not correctness.
    catalog["bundles"][0]["native"]["steps"][0]["result"]["data"]["values"][1] = 7
    with pytest.raises(ValueError):
        Workbench.restore(catalog)


@pytest.mark.parametrize("values,steps,expected", [
    ([0, 0, 1000, 0, 0], 3, [0, 219, 313, 219, 0]), ([0, -3, 0], 1, [0, -2, 0]), ([0, 3, 0], 1, [0, 2, 0]),
    ([7, -3, 5], 0, [7, -3, 5]), ([7, -3, 5], 1, [7, 1, 5])])
def test_heat_reference_truncates_toward_zero_with_fixed_ends(values, steps, expected):
    assert fixtures.heat_reference(values, steps) == expected


def test_t093_refusals_leave_workspace_and_state_unchanged(tmp_path):
    report = run("T093", tmp_path)
    assert report["state"] == "completed"
    primary = report["findings"][0]
    assert checks_pass(primary) and primary["value"]["state_parts_changed"] == []
    assert primary["value"]["refused_requests"] == primary["value"]["requests"] == 16
    assert primary["value"]["workspace_changed"] is False and primary["value"]["directory_changed"] is False
    reopen = claim(report, "A refused reopen")
    assert reopen["value"] == {"refusal": "Saved result does not refer to this evidence", "output_directory_created": False}
    counter = claim(report, "A refused recording operation")
    assert "executions" in counter["value"]["state_parts_changed"] and counter["value"]["saved_workspace_changed"] is False


def test_golden_manifest_matches_committed_fixtures():
    root = fixtures.fixture_root()
    if root is None:
        pytest.skip("tests/fixtures/lab is not reachable")
    for name, digest in fixtures.GOLDEN_MANIFEST.items():
        assert sha256((root / name).read_bytes()).hexdigest() == digest, name


def test_t094_golden_workspaces_reopen_with_current_code(tmp_path, monkeypatch):
    if fixtures.fixture_root() is None:
        pytest.skip("tests/fixtures/lab is not reachable")
    report = run("T094", tmp_path / "run")
    assert report["state"] == "completed"
    primary = report["findings"][0]
    assert primary["evidence_status"] == "numerically_verified" and checks_pass(primary)
    assert primary["value"]["golden/energy-accuracy-workspace.json"]["bundles"] == 2
    assert primary["value"]["golden/oscillator-workspace.json"]["results"] == 3
    assert claim(report, "Golden fixture SHA-256")["value"] == fixtures.GOLDEN_MANIFEST
    heat = claim(report, "Retained golden SCR heat values")
    assert heat["evidence_status"] == "independently_verified" and heat["value"] == [0, 16, 24, 16, 0]
    assert claim(report, "Replay of the golden SCR bundle")["evidence_status"] == "numerically_verified"
    # Without a reachable fixture directory the task reports blocked, not a silent pass.
    monkeypatch.setattr(runner, "repository_path", lambda *parts: None)
    blocked = run("T094", tmp_path / "blocked")
    assert blocked["state"] == "blocked" and blocked["findings"] == []


def test_committed_malformed_fixtures_match_their_generator():
    root = fixtures.fixture_root()
    if root is None:
        pytest.skip("tests/fixtures/lab is not reachable")
    committed = sorted(path.name for path in (root / "malformed").iterdir())
    assert committed == sorted(fixtures.malformed_fixtures())
    for name, raw in fixtures.malformed_fixtures().items():
        assert (root / "malformed" / name).read_bytes() == raw, name


def test_t095_malformed_fixtures_are_refused_with_retained_text(tmp_path):
    report = run("T095", tmp_path)
    assert report["state"] == "completed"
    primary = report["findings"][0]
    assert primary["evidence_status"] == "numerically_verified"
    assert primary["value"] == {"fixtures": 19, "refused": 19}
    retained = json.loads((tmp_path / "artifacts" / "T095" / "malformed-refusals.json").read_text(encoding="utf-8"))
    assert retained["fixtures"]["duplicate-key.json"]["session_read_json"]["message"] == "Duplicate JSON key: schema"
    assert retained["fixtures"]["nan.json"]["exchange_inspect"]["message"] == "nonfinite JSON number: NaN"
    overflow = claim(report, "session.read_json accepts an overflowing number")
    assert overflow["value"] == {"session_read_json_refused": False, "exchange_refused": True, "workbench_refused": True}
    assert claim(report, "Session.from_workspace accepts an unknown")["value"] == {"accepted": True, "dropped_on_resave": True}
    assert all(f.get("counterexample") for f in report["findings"][1:4])


def test_t096_provider_free_conformance(tmp_path):
    report = run("T096", tmp_path)
    assert report["state"] == "completed"
    identity = report["findings"][0]
    assert identity["evidence_status"] == "numerically_verified"
    assert identity["value"]["accepted"] == identity["value"]["valid"] == 24
    assert identity["value"]["refused"] == identity["value"]["mutations"] > 24
    candidate = claim(report, "candidate_evidence.validate_response")
    assert candidate["value"]["valid"] == candidate["value"]["valid_accepted"] == 6
    assert candidate["value"]["mutations"] == candidate["value"]["mutations_refused"] >= 60
    assert claim(report, "Observation-batch identities")["counterexample"]
    assert claim(report, "Passing provider-free conformance")["evidence_status"] == "not_established"


def test_t097_is_blocked_without_scr(tmp_path):
    report = run("T097", tmp_path)
    assert report["state"] == "blocked"
    assert [f["domain"] for f in report["findings"]] == ["physical"]
    assert report["findings"][0]["evidence_status"] == "not_established"
    t099 = run("T099", tmp_path)
    assert t099["state"] == "blocked" and t099["findings"][0]["domain"] == "production_acceptance"


def _scr_bindings():
    scr = os.environ.get("CIW_LAB_SCR_REPO")
    if not scr:
        pytest.skip("set CIW_LAB_SCR_REPO to a clean SCR checkout at the declared-workload pin")
    bound = {"scr": scr}
    if os.environ.get("CIW_LAB_SCR_ENGINE"):
        bound["scr-engine"] = os.environ["CIW_LAB_SCR_ENGINE"]
    elif shutil.which("cargo") is None:
        pytest.skip("set CIW_LAB_SCR_ENGINE or put cargo on PATH")
    for role, name in (("set", "CIW_LAB_SET_REPO"), ("ppda", "CIW_LAB_PPDA_REPO"),
                       ("scr-exchange", "CIW_LAB_SCR_EXCHANGE_REPO")):
        if os.environ.get(name):
            bound[role] = os.environ[name]
    return bound


def test_t097_scr_numerical_heat_integration(tmp_path):
    bound = _scr_bindings()
    report = run("T097", tmp_path, bound)
    assert report["state"] in ("completed", "partial")
    primary = report["findings"][0]
    assert primary["evidence_status"] == "provider_backed"
    assert primary["value"]["workbench_example"] == [0, 16, 24, 16, 0]
    assert primary["value"]["api_cases"][0] == [0, 219, 313, 219, 0]
    assert claim(report, "SCR heat outputs")["evidence_status"] == "independently_verified"
    assert claim(report, "The reopened SCR workspace")["evidence_status"] == "numerically_verified"
    if {"set", "ppda", "scr-exchange"} <= bound.keys():
        assert report["state"] == "completed"
        assert claim(report, "PPDA and SCR exchange producers")["evidence_status"] == "provider_backed"


def test_t098_provider_identities(tmp_path):
    blocked = run("T098", tmp_path / "none")
    assert blocked["state"] == "blocked" and blocked["findings"] == []
    table = json.loads((tmp_path / "none" / "artifacts" / "T098" / "provider-identities.json").read_text(encoding="utf-8"))
    assert {pin["revision"] for pin in table["ciw_pins"]["scr"]} == {"a59aba283b0304faeeb3e5d305087e7709e171ca"}
    if not os.environ.get("CIW_LAB_SCR_REPO"):
        pytest.skip("set CIW_LAB_SCR_REPO for provider identities")
    report = run("T098", tmp_path / "scr", {"scr": os.environ["CIW_LAB_SCR_REPO"]})
    assert report["state"] == "completed"
    pins = claim(report, "CIW declares one SCR revision")
    assert pins["evidence_status"] == "numerically_verified" and len(pins["value"]["set_revisions"]) == 4
    accepted = claim(report, "Every accepted provider checkout")
    assert accepted["value"]["rejected"] == []
    assert accepted["value"]["accepted"]["scr"]["head"] == "a59aba283b0304faeeb3e5d305087e7709e171ca"
    assert accepted["value"]["accepted"]["scr"]["tree"] == "4068a711534932e8d89bb0d87d373376dafdf6cd"
    assert claim(report, "The working bytes")["evidence_status"] == "independently_verified"
    digests = claim(report, "Tracked-source and Cargo.lock")["value"]["scr"]
    assert digests["cargo_locks"]["crates/Cargo.lock"] == \
        "24b1db68d762d3457eb40465dae6c3edc8a2da2925d42f3f794d090d18a5f2bc"


def _git(repository, *arguments):
    subprocess.run(["git", "-C", str(repository), "-c", "user.name=lab", "-c", "user.email=lab@example.invalid",
                    "-c", "core.autocrlf=false", *arguments], check=True, capture_output=True)


def test_tree_recomputation_matches_git_on_a_synthetic_repository(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    repository = tmp_path / "provider"
    (repository / "crates" / "a").mkdir(parents=True)
    (repository / "crates" / "Cargo.lock").write_bytes(b"# lock\n")
    (repository / "crates" / "a" / "lib.rs").write_bytes(b"pub fn f() {}\n")
    (repository / "crates.txt").write_bytes(b"sorted before the crates tree\n")
    (repository / "run.sh").write_bytes(b"#!/bin/sh\n")
    if os.name == "posix":
        (repository / "run.sh").chmod(0o755)
        (repository / "link").symlink_to("run.sh")
    _git(repository, "init", "-q")
    _git(repository, "add", "-A")
    _git(repository, "commit", "-q", "-m", "synthetic provider")
    identity = providers.checkout_identity(repository)
    assert identity["recomputed_tree"] == identity["tree"] and identity["mismatched_files"] == []
    assert identity["cargo_locks"] == {"crates/Cargo.lock": sha256(b"# lock\n").hexdigest()}
    assert providers.compare_with_pins("scr", identity, providers.ciw_pins())["accepted"] is False
    (repository / "crates" / "a" / "lib.rs").write_bytes(b"pub fn g() {}\n")
    changed = providers.checkout_identity(repository)
    assert changed["recomputed_tree"] != changed["tree"] and changed["mismatched_files"] == ["crates/a/lib.rs"]
    assert changed["dirty"] is True


def test_exchange_workflow_pins_mirror_the_ci_workflow():
    workflow = ROOT / ".github" / "workflows" / "exchange.yml"
    if not workflow.is_file():
        pytest.skip(".github/workflows is not beside the tests")
    text = workflow.read_text(encoding="utf-8")
    for revision in providers.EXCHANGE_WORKFLOW_PINS.values():
        assert f"ref: {revision}" in text


def test_t099_locked_offline_scr_build(tmp_path):
    scr = os.environ.get("CIW_LAB_SCR_REPO")
    if not scr or shutil.which("cargo") is None:
        pytest.skip("set CIW_LAB_SCR_REPO and put cargo on PATH")
    report = run("T099", tmp_path, {"scr": scr})
    assert report["state"] == "partial"
    assert report["findings"][0]["evidence_status"] == "provider_backed"
    assert report["findings"][0]["value"]["cargo_lock_sha256"] == \
        "24b1db68d762d3457eb40465dae6c3edc8a2da2925d42f3f794d090d18a5f2bc"
    assert claim(report, "The locked build leaves")["value"]["distinct_binary_digests"] == 1
    assert claim(report, "The freshly built engine")["value"] == [0, 219, 313, 219, 0]
    sp1 = claim(report, "The SP1 proved-heat locked build")
    assert sp1["evidence_status"] == "not_established" and sp1["value"]["attempted"] is False


def test_t100_labels_and_origins_stay_distinct(tmp_path):
    run("T096", tmp_path)
    run("T092", tmp_path)
    report = run("T100", tmp_path)
    assert report["state"] == "completed"
    primary = report["findings"][0]
    assert primary["value"] == {"reports_checked": 2, "label_violations": 0}
    rendered = claim(report, "Every retained finding's label is rendered")
    assert rendered["value"] == {"reports_checked": 2, "rendering_violations": 0}
    assert claim(report, "An unescaped pipe")["value"] == {"row_cells": 4, "label_column_shifted": True}
    energy = claim(report, "CIW keeps the synthetic energy fixture")
    assert energy["evidence_status"] == "numerically_verified"
    assert energy["value"]["synthetic"]["classification"] == "synthetic_only"
    relabel = claim(report, "A resealed relabel")
    assert relabel["value"]["resealed_relabel"]["classification"] == "physical_domain_measurement"
    assert relabel["counterexample"]["statement"].startswith("CIW energy records can distinguish")
    assert claim(report, "CIW refuses free-energy sources")["evidence_status"] == "numerically_verified"
    physical = [f for f in report["findings"] if f["domain"] in ("physical", "sensor_performance")]
    assert physical and all(f["evidence_status"] == "not_established" for f in physical)


def test_t100_flags_a_retained_report_whose_label_leaves_its_column(tmp_path):
    from ciw.lab.evidence import finding
    record = finding("pipe | in a claim", "numerical", 1.0, {"generator": {"name": "probe"}})
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "T001.json").write_text(runner.dumps(build_report(QUEUE["T001"], "partial", {}, [record])))
    report = run("T100", tmp_path)
    assert report["findings"][0]["value"] == {"reports_checked": 1, "label_violations": 0}
    rendered = claim(report, "Every retained finding's label is rendered")
    assert rendered["value"] == {"reports_checked": 1, "rendering_violations": 1}
    assert rendered["evidence_status"] == "not_established" and report["state"] == "partial"


def test_render_markdown_claim_pipe_counterexample():
    from ciw.lab.evidence import finding
    record = finding("claim with a | pipe", "numerical", 1.0, {"generator": {"name": "probe"}})
    validate_finding(record)
    row = render_markdown(build_report(QUEUE["T100"], "partial", {}, [record])).splitlines()[-1]
    # GitHub-flavoured Markdown drops cells beyond the header's three: the label cell is lost.
    assert row.endswith("| `synthetic` |") and section._cells(row) == 4


def test_relabelled_energy_origin_collides_within_one_workbench(tmp_path):
    from ciw import energy_records
    from ciw.instruments import make_demo_run
    from ciw.session import Session
    from ciw.telemetry import canonical
    log = json.loads(fixtures.example_bytes("energy-accuracy/baseline.json"))
    client = fixtures.Client(Session(make_demo_run(), tmp_path))
    first = client.ok("source.add", fixtures.source_payload("energy-accuracy", canonical(log), "synthetic"))
    client.ok("operation.execute", {"operation_id": "ciw.energy-accuracy.v1", "parameters": {"source_id": first["source_id"]}})
    relabel = energy_records.seal({k: v for k, v in log.items() if k != "log_digest"} | {"origin": "physical_measurement"})
    second = client.ok("source.add", fixtures.source_payload("energy-accuracy", canonical(relabel), "relabelled"))
    refused = client.call("operation.execute", {"operation_id": "ciw.energy-accuracy.v1",
                                                "parameters": {"source_id": second["source_id"]}})
    assert refused["type"] == "error" and "Identity collision" in refused["payload"]["message"]
    assert base64.b64decode(client.ok("source.get", {"source_id": second["source_id"]})["bytes_b64"]) == canonical(relabel)
