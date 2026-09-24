"""Exchange bundles, providers and visibility tasks T091-T100 against the real CIW code.

Provider-backed tests are opt-in: CIW_LAB_SCR_REPO (clean SCR checkout at the
declared-workload pin) with CIW_LAB_SCR_ENGINE or cargo on PATH;
CIW_LAB_SET_REPO, CIW_LAB_PPDA_REPO and CIW_LAB_SCR_EXCHANGE_REPO add the
exchange integrations. Everything else runs offline with NumPy only (the
synthetic-repository tests also need git).
"""
import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import platform

import numpy as np
import pytest

from ciw.lab import exchange_provenance_bundles as section
from ciw.lab import exchange_provenance_bundles_fixtures as fixtures
from ciw.lab import exchange_provenance_bundles_providers as providers
from ciw.lab import planner, registry, research_portfolio, runner
from ciw.lab.evidence import describe_origin, finding, finding_origin, validate_finding
from ciw.lab.report import FINDINGS_RULE, build_report, finding_row, render_markdown

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


def artifact(directory, task_id, name):
    return json.loads((directory / "artifacts" / task_id / name).read_text(encoding="utf-8"))


def test_every_section_task_is_registered_with_tests_in_this_file():
    for number in range(91, 101):
        implementation = registry._REGISTRY[f"T{number:03d}"]
        assert implementation.regression_tests
        for node in implementation.regression_tests:
            path, name = node.split("::")
            assert path == "tests/test_lab_exchange_provenance_bundles.py" and name in globals(), node


def test_numerical_findings_state_an_uncertainty(tmp_path):
    for task_id in ("T091", "T093", "T096"):
        report = run(task_id, tmp_path)
        for record in report["findings"]:
            if record["basis"].get("checks") or record["basis"].get("independent_check"):
                assert record["uncertainty"] is not None, (task_id, record["claim"])


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
    for module, qualname, attribute in fixtures.execution_paths() + fixtures.RECOMPUTATION_PATHS:
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


def test_execution_guard_covers_every_workbench_workflow_kind():
    from ciw.workbench import OPERATIONS, _workflow
    with fixtures.execution_guard() as guard:
        for kind in sorted(OPERATIONS):
            workflow = _workflow(kind)
            for name in ("create_session", "replay_session"):
                with pytest.raises(fixtures.ExecutionForbidden):
                    getattr(workflow, name)(b"{}", {})
    assert len(guard["attempts"]) == 2 * len(OPERATIONS)


def test_t091_reopen_needs_no_provider_and_reaches_no_execution_path(tmp_path):
    report = run("T091", tmp_path)
    assert report["state"] == "completed"
    primary = report["findings"][0]
    assert primary["evidence_status"] == "numerically_verified" and checks_pass(primary)
    assert primary["value"]["execution_path_calls"] == 0 and primary["value"]["provider_bindings"] == []
    # The saving session held a trusted binding and a provider-kind bundle; neither the binding nor execution survives.
    assert primary["value"]["bindings_before_save"] == ["numerical-heat"]
    assert primary["value"]["provider_kind_bundles"] == ["numerical-heat"]
    assert primary["value"]["available_operations"] == ["ciw.energy-accuracy.v1"]
    assert primary["value"]["bundles"] == 3 and primary["value"]["results"] == 3
    replay = next(c for c in primary["basis"]["checks"] if c["reference_kind"] == "refusal")
    assert replay["observed_refusal"] == f"operation_unavailable: {section.UNBOUND}" and replay["passed"]
    retained = artifact(tmp_path, "T091", "reopen.json")
    assert retained["binding_paths_in_saved_workspace"] == 0 and retained["reopen_attempts"] == []
    assert "ciw.telemetry.create_session" in retained["guarded_paths"]
    guard = claim(report, "The execution guard intercepts")
    assert guard["value"] == "intercepted" and guard["evidence_status"] == "numerically_verified"
    recomputed = claim(report, "Reopening recomputes")
    assert recomputed["evidence_status"] == "numerically_verified" and recomputed["value"] >= 1
    assert recomputed["counterexample"]["statement"].startswith("Reopening a saved")
    after = claim(report, "After reopen")
    assert after["value"] is True and after["evidence_status"] == "numerically_verified"


def test_t091_refutes_its_claim_when_reopen_recovers_a_binding(tmp_path, monkeypatch):
    from ciw.session import Session
    original = Session.from_workspace.__func__

    def recovering(cls, path, output_dir=None):
        reopened = original(cls, path, output_dir)
        reopened.workbench._bindings["numerical-heat"] = {"scr": Path("/recovered")}
        return reopened

    monkeypatch.setattr(Session, "from_workspace", classmethod(recovering))
    report = run("T091", tmp_path)
    primary = report["findings"][0]
    assert primary["evidence_status"] == "not_established" and primary["value"]["provider_bindings"] == ["numerical-heat"]
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "not_established"


def test_t092_unbound_replay_and_execution_are_refused(tmp_path):
    report = run("T092", tmp_path)
    assert report["state"] == "completed"
    primary = report["findings"][0]
    assert primary["evidence_status"] == "numerically_verified" and checks_pass(primary)
    value = primary["value"]
    assert value["requests"] == value["refused_as_expected"]
    assert "operation_unavailable" in value["codes"]
    assert {"numerical-heat", "proved-heat"} <= set(value["kinds_refused_unbound"])
    if (ROOT / "examples").is_dir():
        assert value["requests"] == 25
        assert "variational-free-energy" in value["kinds_refused_unbound"]
        assert value["kinds_refused_at_upstream_selection"] == ["identified-design", "schematic-companions"]
        assert value["unavailable_kinds_not_exercised"] == ["acquired-calibrated-window", "bim-quantity",
                                                            "identified-stability", "residual-monitor"]
    assert value["kinds_with_replay_refusal_observed"] == ["numerical-heat"]
    cases = artifact(tmp_path, "T092", "refusals.json")
    assert cases["provider_paths_reached"] == [] and cases["reopen_attempts"] == []
    assert "ciw.telemetry.create_session" in cases["provider_paths_counted"]
    assert cases["fabricated_bundle_reopen"] == "accepted"
    esm = next(row for row in cases["cases"] if row["case"].startswith("ESM candidate inspection"))
    assert esm["case"] == "ESM candidate inspection of a non-telemetry bundle"
    fabricated = claim(report, "A retained numerical-heat bundle whose values no provider computed")
    assert fabricated["evidence_status"] == "numerically_verified" and fabricated["value"]["reopen_outcome"] == "accepted"
    assert fabricated["counterexample"]["witness"]["retained_values"] == [0, 1, 2, 3, 0]
    assert claim(report, "The binding-free energy-accuracy replay")["evidence_status"] == "numerically_verified"
    assert claim(report, "A content-consistent")["evidence_status"] == "not_established"
    assert any("inferred" in line and "numerical-heat" in line for line in report["unresolved_assumptions"])


def test_t092_keeps_its_refusals_when_reopen_refuses_the_fabricated_bundle(tmp_path, monkeypatch):
    genuine = fixtures.fabricated_heat_catalog

    def unsealed(values, **options):
        catalog = genuine(values, **options)
        catalog["bundles"][0]["native"]["steps"][0]["result"]["data"]["values"][1] = 7  # edit without resealing
        return catalog

    monkeypatch.setattr(section, "fabricated_heat_catalog", unsealed)
    report = run("T092", tmp_path)
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "not_established"
    primary = report["findings"][0]
    assert primary["evidence_status"] == "numerically_verified" and primary["value"]["refused_as_expected"] > 0
    assert primary["value"]["kinds_with_replay_refusal_observed"] == []
    fabricated = claim(report, "A retained numerical-heat bundle whose values no provider computed")
    assert fabricated["evidence_status"] == "not_established" and fabricated["value"]["reopen_outcome"] != "accepted"


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
    assert primary["evidence_status"] == "numerically_verified" and checks_pass(primary)
    assert primary["value"]["state_parts_changed"] == [] and primary["value"]["directory_changed"] is False
    assert primary["value"]["refused_as_expected"] == primary["value"]["requests"] == 16
    assert primary["value"]["resaved_content_equal"] is True
    # Each refusal is compared with its expected text, not merely counted as "not accepted".
    refusals = [check for check in primary["basis"]["checks"] if check["reference_kind"] == "refusal"]
    assert len(refusals) == 16 and all(check["expected_refusal"] != "accepted" for check in refusals)
    assert primary["claim"].startswith("Each of the 16 exercised non-recording request classes")
    reopen = claim(report, "A refused reopen")
    assert reopen["value"] == {"refusal": "Saved result does not refer to this evidence", "output_directory_created": False}
    assert reopen["evidence_status"] == "numerically_verified"
    counter = claim(report, "A refused recording operation")
    assert "executions" in counter["value"]["state_parts_changed"] and counter["value"]["resaved_workspace_changed"]
    assert counter["evidence_status"] == "numerically_verified" and counter["counterexample"]


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
    bit_exact = claim(report, "Reopening recomputes the golden energy analysis bit for bit")
    platform = fixtures.platform_fingerprint()
    if bit_exact["value"]["refused_as_platform_dependent"] and platform != fixtures.GOLDEN_PLATFORM:
        # Reopen compares a LAPACK-backed recomputation bit for bit; T094 records this as its own finding.
        pytest.xfail(f"golden energy analysis is not bit-exact on this platform: {platform}")
    assert report["state"] == "completed"
    assert bit_exact["evidence_status"] == "numerically_verified"
    primary = report["findings"][0]
    assert primary["evidence_status"] == "numerically_verified" and checks_pass(primary)
    assert primary["value"]["golden/energy-accuracy-workspace.json"]["bundles"] == 2
    assert primary["value"]["golden/oscillator-workspace.json"]["results"] == 3
    assert claim(report, "Golden fixture SHA-256")["value"] == fixtures.GOLDEN_MANIFEST
    heat = claim(report, "The retained golden numerical-heat bundle")
    assert heat["evidence_status"] == "numerically_verified" and heat["value"] == [0, 16, 24, 16, 0]
    assert "independent_check" not in heat["basis"] and len(heat["basis"]["checks"]) == 4
    origin = claim(report, "The retained golden numerical-heat values were computed")
    assert origin["evidence_status"] == "not_established" and origin["expected_not_established"] is True
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
    assert primary["evidence_status"] == "numerically_verified" and checks_pass(primary)
    assert primary["value"] == {"inputs": 21, "with_declared_refusal": 19, "refused_with_declared_text": 19,
                                "without_declared_refusal": {
                                    "extra top-level workspace field (Session.from_workspace)": "accepted",
                                    "incomplete-workspace.json": "session_from_workspace: AttributeError"}}
    assert report["numerical_result"].startswith("19/21 ")
    retained = artifact(tmp_path, "T095", "malformed-refusals.json")
    # A crash is retained with its type and never counted as a refusal.
    crashed = retained["fixtures"]["incomplete-workspace.json"]["session_from_workspace"]
    assert crashed["error_type"] == "AttributeError" and crashed["refused"] is False and crashed["crashed"] is True
    deep = retained["fixtures"]["deep-nesting.json"]["session_read_json"]
    assert deep["error_type"] == "RecursionError" and deep["refused"] is False
    if fixtures.fixture_root() is not None:
        assert retained["committed_fixture_comparison"] == {"differing": [], "missing": [], "extra": []}
        assert claim(report, "The committed malformed fixtures")["evidence_status"] == "numerically_verified"
    assert retained["fixtures"]["duplicate-key.json"]["session_read_json"]["message"] == "Duplicate JSON key: schema"
    assert retained["fixtures"]["nan.json"]["exchange_inspect"]["message"] == "nonfinite JSON number: NaN"
    assert retained["runtime"]["wrong-type workspace_version (Session.from_workspace)"]["message"] == \
        "Unsupported workspace format"
    overflow = claim(report, "session.read_json accepts an overflowing number")
    assert overflow["value"] == {"session_read_json_refused": False, "session_read_json_value": "inf",
                                 "exchange_refused": True, "workbench_refused": True}
    assert overflow["counterexample"]["witness"]["parsed_value"] == "inf"
    crash = claim(report, "Session.from_workspace raises AttributeError")
    assert crash["value"] == "AttributeError" and crash["evidence_status"] == "numerically_verified"
    assert claim(report, "Session.from_workspace accepts an unknown")["value"] == {"accepted": True, "dropped_on_resave": True}
    source = claim(report, "Workbench source.add reports")
    assert source["counterexample"]["witness"]["message"] == section.MALFORMED_TEXT
    counterexamples = [f for f in report["findings"] if f.get("counterexample")]
    assert len(counterexamples) == 5 and all(f["evidence_status"] == "numerically_verified" for f in counterexamples)


def test_canonical_encoder_matches_json_dumps_on_seeded_records():
    generator = np.random.Generator(np.random.PCG64(4))
    for _ in range(200):
        value = {"schema": "x", "body": section._random_value(generator), "text": "é\n\t\"\\\x01–Ω"}
        assert fixtures.canonical_text(value) == json.dumps(value, sort_keys=True, separators=(",", ":"),
                                                             ensure_ascii=False, allow_nan=False)
    with pytest.raises(ValueError):
        fixtures.canonical_text({"x": float("nan")})


def test_t096_provider_free_conformance(tmp_path):
    report = run("T096", tmp_path)
    assert report["state"] == "completed"
    identity = report["findings"][0]
    assert identity["evidence_status"] == "numerically_verified" and checks_pass(identity)
    assert identity["value"]["accepted"] == identity["value"]["valid"] == identity["value"]["encoder_agreement"] == 24
    assert identity["value"]["reordered_accepted"] == 24
    assert identity["value"]["refused"] == identity["value"]["mutations"] > 24
    assert "escap" not in identity["claim"].casefold()
    assert "lab-written canonical encoder that agrees byte for byte with json.dumps" in identity["claim"]
    assert "independent" not in identity["claim"]
    candidate = claim(report, "candidate_evidence.validate_response")
    assert candidate["evidence_status"] == "numerically_verified"
    assert candidate["value"]["valid"] == candidate["value"]["valid_accepted"] == 6
    assert candidate["value"]["mutations"] == candidate["value"]["mutations_refused"] >= 60
    batch = claim(report, "Observation-batch identities")
    assert batch["evidence_status"] == "numerically_verified" and batch["counterexample"]
    extra = claim(report, "validate_response accepts unknown extra fields")
    assert extra["evidence_status"] == "numerically_verified" and extra["value"] == {"cases": 2, "accepted": 2}
    assert extra["counterexample"]["witness"]["fields"] == ["lab_admission_override", "candidate.lab_admitted"]
    assert claim(report, "Passing provider-free conformance")["evidence_status"] == "not_established"
    assert set(report["provider_runtime_identity"]["sources"]) == {section.MODULE, section.FIXTURES}
    # Every finding that rests on synthetic inputs declares their generator, so its Basis column names it; the
    # checks still decide the label.
    records = "synthetic inputs (ciw.lab seeded nested JSON records, seed 9601)"
    responses = f"synthetic inputs ({section.CANDIDATE_GENERATOR})"
    for record, shown in ((identity, records), (candidate, responses), (batch, records), (extra, responses)):
        assert record["basis"]["generator"] and shown in finding_row(record), record["claim"]
        assert record["evidence_status"] == "numerically_verified"
    assert candidate["basis"]["generator"]["count"] == 72


def test_t097_is_blocked_without_scr(tmp_path):
    report = run("T097", tmp_path)
    assert report["state"] == "blocked"
    assert [f["domain"] for f in report["findings"]] == ["physical"]
    assert report["findings"][0]["evidence_status"] == "not_established"
    t099 = run("T099", tmp_path)
    assert t099["state"] == "blocked" and t099["findings"][0]["domain"] == "production_acceptance"


def _git(repository, *arguments):
    subprocess.run(["git", "-C", str(repository), "-c", "user.name=lab", "-c", "user.email=lab@example.invalid",
                    "-c", "core.autocrlf=false", *arguments], check=True, capture_output=True)


def _synthetic_repository(path):
    """A small committed Git repository standing in for a provider checkout (at no CIW pin)."""
    (path / "crates" / "a").mkdir(parents=True)
    (path / "crates" / "Cargo.lock").write_bytes(b"# lock\n")
    (path / "uv.lock").write_bytes(b"version = 1\n")
    (path / "requirements.txt").write_bytes(b"numpy==2.4.3 \\\n    --hash=sha256:00\n# pinned\n")
    (path / "requirements-dev.txt").write_bytes(b"pytest>=8\n")
    (path / "crates" / "a" / "lib.rs").write_bytes(b"pub fn f() {}\n")
    (path / "crates.txt").write_bytes(b"sorted before the crates tree\n")
    (path / "run.sh").write_bytes(b"#!/bin/sh\n")
    if os.name == "posix":
        (path / "run.sh").chmod(0o755)
        (path / "link").symlink_to("run.sh")
    _git(path, "init", "-q")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "synthetic provider")
    return path


SET_RESULT = {"status": "conformant", "effective_rank": 2, "identity_status": "caller_declared_reference",
              "validator": {"repository": "https://github.com/giasonpooni/State-Estimation-Evaluation-Testbed",
                            "revision": "542e672be512bf43b61253f2b2a43cd967cb3062",
                            "path": "state_estimation_testbed/contracts.py", "sha256": "0" * 64},
              "authority": {"may_authorize": False}, "unchanged_input": True,
              "indefinite_covariance": "covariance.matrix is not positive-semidefinite"}


@pytest.mark.parametrize("outcome", ["as_expected", "contrary"])
def test_t097_keeps_set_results_when_the_engine_is_missing(tmp_path, monkeypatch, outcome):
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    scr = _synthetic_repository(tmp_path / "scr")
    set_repo = _synthetic_repository(tmp_path / "set")

    def pinned(role, identity, pins):
        matched = {"scr": ["ciw.declared_workload.PINS[numerical-heat]"], "set": ["ciw/exchange-runtime.json"]}
        return {"role": role, "matched": matched.get(role, []), "unmatched": [], "tree_refusals": [],
                "accepted": role in matched, "clean": True}

    result = dict(SET_RESULT)
    if outcome == "contrary":
        # A SET that reports another outcome must refute the finding, not keep its claim.
        result.update(status="nonconformant", effective_rank=1, indefinite_covariance="covariance.matrix is not symmetric")
    monkeypatch.setattr(providers, "compare_with_pins", pinned)
    monkeypatch.setattr(section, "_engine", lambda ctx: None)
    monkeypatch.setattr(providers, "set_exchange_inspection", lambda repository, directory: result)
    report = run("T097", tmp_path / "run", {"scr": str(scr), "set": str(set_repo)})
    assert report["state"] == "partial"
    set_finding = claim(report, "The pinned SET contracts validator")
    assert claim(report, "SCR numerical-heat execution was not performed")["expected_not_established"] is True
    if outcome == "as_expected":
        assert set_finding["evidence_status"] == "numerically_verified" and checks_pass(set_finding)
        assert report["evidence_status"]["primary"] == "numerically_verified"
    else:
        assert set_finding["evidence_status"] == "not_established"
        assert report["evidence_status"]["primary"] == "not_established"
    assert "ciw" in report["provider_runtime_identity"]
    # The executed SET checkout is declared as the finding's provider, so its Basis column names it either way.
    head = report["provider_runtime_identity"]["set"]["head"]
    assert set_finding["basis"]["provider"] == {"repository": providers.REPOSITORIES["set"], "revision": head,
                                                "source_tree": report["provider_runtime_identity"]["set"]["tree"],
                                                "executed": True}
    assert f"pinned provider run ({providers.REPOSITORIES['set']}@{head[:12]})" in finding_row(set_finding)


ROUNDTRIP_RESULT = {"status": "conformant", "links": ["matched_supplied_reference", "matched_supplied_reference"],
                    "verification_outcome": "failed", "authority": {"may_authorize": False},
                    "changed_result_refusal": "result_id does not match the artifact content"}


@pytest.mark.parametrize("outcome", ["as_expected", "contrary"])
def test_t097_roundtrip_finding_is_refuted_by_a_contrary_producer_outcome(tmp_path, monkeypatch, outcome):
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    bound = {role: str(_synthetic_repository(tmp_path / role)) for role in ("scr", "set", "ppda", "scr-exchange")}

    def pinned(role, identity, pins):
        matched = {"scr": ["ciw.declared_workload.PINS[numerical-heat]"], "set": ["ciw/exchange-runtime.json"],
                   "ppda": [".github/workflows/exchange.yml"], "scr-exchange": [".github/workflows/exchange.yml"]}
        return {"role": role, "matched": matched.get(role, []), "unmatched": [], "tree_refusals": [],
                "accepted": role in matched, "clean": True}

    result = dict(ROUNDTRIP_RESULT)
    if outcome == "contrary":
        # Producers or SET reporting another outcome must refute the roundtrip finding, not keep its claim.
        result.update(status="nonconformant", links=["unresolved_reference"], verification_outcome="passed",
                      changed_result_refusal="not_refused", authority={"may_authorize": True})
    calls = []
    monkeypatch.setattr(providers, "compare_with_pins", pinned)
    monkeypatch.setattr(section, "_engine", lambda ctx: None)
    monkeypatch.setattr(providers, "set_exchange_inspection", lambda repository, directory: SET_RESULT)
    monkeypatch.setattr(providers, "exchange_roundtrip",
                        lambda ppda, scr, set_repo, directory: calls.append((ppda, scr, set_repo)) or result)
    report = run("T097", tmp_path / "run", bound)
    assert [tuple(map(str, call)) for call in calls] == [(bound["ppda"], bound["scr-exchange"], bound["set"])]
    assert report["state"] == "partial"  # the SCR engine is missing here
    roundtrip = claim(report, "PPDA and SCR exchange artifacts")
    assert claim(report, "The pinned SET contracts validator")["evidence_status"] == "numerically_verified"
    if outcome == "as_expected":
        assert roundtrip["evidence_status"] == "numerically_verified" and checks_pass(roundtrip)
        assert report["evidence_status"]["primary"] == "numerically_verified"
    else:
        failed = [check["reference"] for check in roundtrip["basis"]["checks"] if not check["passed"]]
        assert len(failed) == 5
        assert roundtrip["evidence_status"] == "not_established"
        assert report["evidence_status"]["primary"] == "not_established"
    for role in ("ppda", "scr-exchange", "set"):
        assert roundtrip["basis"]["notes"]["provider"][role]["executed"] is True
    # Its one provider slot names the SET checker whose verdict the checks read.
    assert roundtrip["basis"]["provider"] == roundtrip["basis"]["notes"]["provider"]["set"]
    assert f"pinned provider run ({providers.REPOSITORIES['set']}@" in finding_row(roundtrip)


def test_t097_records_head_and_tree_of_each_exchange_checkout_as_t098_does(tmp_path, monkeypatch):
    # A release inventory (T165) keys runtimes by (runtime, revision, tree): T097 and T098 must agree on all three.
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    bound = {role: str(_synthetic_repository(tmp_path / role)) for role in ("scr", "set", "ppda", "scr-exchange")}
    matched = {"scr": ["ciw.declared_workload.PINS[numerical-heat]"], "set": ["ciw/exchange-runtime.json"],
               "ppda": [".github/workflows/exchange.yml"], "scr-exchange": [".github/workflows/exchange.yml"]}
    monkeypatch.setattr(providers, "compare_with_pins", lambda role, identity, pins: {
        "role": role, "matched": matched[role], "unmatched": [], "tree_refusals": [], "accepted": True, "clean": True})
    monkeypatch.setattr(section, "_engine", lambda ctx: None)
    monkeypatch.setattr(providers, "set_exchange_inspection", lambda repository, directory: SET_RESULT)
    monkeypatch.setattr(providers, "exchange_roundtrip", lambda ppda, scr, set_repo, directory: ROUNDTRIP_RESULT)
    t097 = run("T097", tmp_path / "run", bound)
    t098 = run("T098", tmp_path / "run", bound)
    exchange = ("set", "ppda", "scr-exchange")
    for role in exchange:
        tree = subprocess.run(["git", "-C", bound[role], "rev-parse", "HEAD^{tree}"], capture_output=True,
                              text=True, check=True).stdout.strip()
        recorded = t097["provider_runtime_identity"][role]
        assert recorded["state"] == "ready" and recorded["tree"] == tree == t098["provider_runtime_identity"][role]["tree"]
    rows = {task["task_id"]: {(entry["runtime"], entry["revision"], entry["tree"])
                              for entry in research_portfolio.runtime_identities(task["provider_runtime_identity"])
                              if entry["runtime"] in exchange}
            for task in (t097, t098)}
    assert len(rows["T097"]) == 3 and rows["T097"] == rows["T098"]
    assert t097["recommended_next_task"] == section.NEXT_STEPS["T097"]
    assert t098["recommended_next_task"] == section.NEXT_STEPS["T098"]


def _mocked_scr_run(monkeypatch):
    """Stand-ins for the SCR engine, its workbench run and its Python API, returning the integer reference."""
    from ciw.declared_workload import HEAT_DESCRIPTOR
    binary = b"lab stand-in engine"
    engine = {"origin": "cargo_build_locked_offline_in_this_run", "binary": binary,
              "sha256": sha256(binary).hexdigest(), "build": {"binary": binary}}
    run_record = {"initial_values": [0, 0, 64, 0, 0], "steps": 2, "values": fixtures.heat_reference([0, 0, 64, 0, 0], 2),
                  "numerical_identity_equal": True, "fresh_occurrences": 4, "replay_numerical_match": True,
                  "verification_independent": False}
    workbench = {"runs": [run_record], "reopened_bindings": [], "reopened_available": [], "reopen_attempts": [],
                 "unbound_replay": {"code": "operation_unavailable", "message": section.UNBOUND}}
    api = {"cases": [{"values": fixtures.heat_reference(values, steps)} for steps, values in section.SURVEY_CASES],
           "descriptor_sha256": sha256(HEAT_DESCRIPTOR).hexdigest()}
    monkeypatch.setattr(section, "_engine", lambda ctx: engine)
    monkeypatch.setattr(providers, "materialize", lambda data, directory: Path(directory) / "execution-cli")
    monkeypatch.setattr(providers, "scr_workbench_integration", lambda scr, path, directory, sources: workbench)
    monkeypatch.setattr(providers, "run_heat_kernel", lambda scr, path, cases: {**api, "cases": api["cases"][:len(cases)]})
    return engine


def test_t097_names_each_executed_provider_beside_its_label(tmp_path, monkeypatch):
    # Every finding resting on a provider's execution declares it, so the Basis column names it; the labels stay.
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    bound = {role: str(_synthetic_repository(tmp_path / role)) for role in ("scr", "set", "ppda", "scr-exchange")}
    matched = {"scr": ["ciw.declared_workload.PINS[numerical-heat]"], "set": ["ciw/exchange-runtime.json"],
               "ppda": [".github/workflows/exchange.yml"], "scr-exchange": [".github/workflows/exchange.yml"]}
    monkeypatch.setattr(providers, "compare_with_pins", lambda role, identity, pins: {
        "role": role, "matched": matched[role], "unmatched": [], "tree_refusals": [], "accepted": True, "clean": True})
    engine = _mocked_scr_run(monkeypatch)
    monkeypatch.setattr(providers, "set_exchange_inspection", lambda repository, directory: SET_RESULT)
    monkeypatch.setattr(providers, "exchange_roundtrip", lambda ppda, scr, set_repo, directory: ROUNDTRIP_RESULT)
    report = run("T097", tmp_path / "run", bound)
    assert report["state"] == "completed"
    identities = report["provider_runtime_identity"]
    expected = {"SCR executed through": ("provider_backed", "scr"),
                "SCR heat outputs": ("independently_verified", "scr"),
                "SCR replay reproduces": ("numerically_verified", "scr"),
                "The pinned SET contracts validator": ("numerically_verified", "set"),
                "PPDA and SCR exchange artifacts": ("numerically_verified", "set")}
    for prefix, (label, role) in expected.items():
        record = claim(report, prefix)
        revision = identities[role]["revision" if role == "scr" else "head"]
        assert record["evidence_status"] == label, prefix
        assert "provider" in finding_origin(record) and record["basis"]["provider"]["revision"] == revision, prefix
        assert f"pinned provider run ({providers.REPOSITORIES[role]}@{revision[:12]})" in finding_row(record), prefix
    assert claim(report, "SCR replay reproduces")["basis"]["provider"]["runtime_digest"] == "sha256:" + engine["sha256"]
    # Reopen executes nothing, so the refusal it records rests on CIW's reopen, not on a provider run.
    assert "provider" not in finding_origin(claim(report, "The reopened SCR workspace"))
    t100 = run("T100", tmp_path / "run")
    sources = claim(t100, "Every finding of those reports whose basis declares a generator")
    assert sources["value"]["provider_findings"] == len(expected) and sources["value"]["not_shown"] == 0


def test_t097_t099_refuse_a_non_repository_scr_binding(tmp_path):
    folder = tmp_path / "not-a-repository"
    folder.mkdir()
    for task_id in ("T097", "T099"):
        report = run(task_id, tmp_path / task_id, {"scr": str(folder)})
        assert report["state"] == "blocked"
        assert "unexpected" not in report["experiment"] + " ".join(report["failure_modes_checked"])
        if runner.Context(tmp_path, {"scr": str(folder)}).available("tool:cargo") or task_id == "T097":
            assert "not a readable Git repository root" in report["numerical_result"]
        assert str(folder) not in json.dumps({k: report[k] for k in runner.PROSE_FIELDS})


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
    replay = claim(report, "SCR replay reproduces")
    assert replay["evidence_status"] == "numerically_verified" and replay["value"]["verification_independent"] is False
    assert f"pinned provider run ({providers.REPOSITORIES['scr']}@a59aba283b03)" in finding_row(replay)
    assert claim(report, "The reopened SCR workspace")["evidence_status"] == "numerically_verified"
    retained = (tmp_path / "artifacts" / "T097" / "integration.json").read_text(encoding="utf-8")
    for role, path in bound.items():
        assert path not in retained and str(Path(path).resolve()) not in retained, role
    assert sys.executable not in retained
    if "set" in bound:
        set_finding = claim(report, "The pinned SET contracts validator")
        assert set_finding["evidence_status"] == "numerically_verified"
        assert set_finding["value"]["indefinite_covariance"] == "covariance.matrix is not positive-semidefinite"
    if {"set", "ppda", "scr-exchange"} <= bound.keys():
        assert report["state"] == "completed"
        roundtrip = claim(report, "PPDA and SCR exchange artifacts")
        assert roundtrip["evidence_status"] == "numerically_verified"
        assert f"pinned provider run ({providers.REPOSITORIES['set']}@542e672be512)" in finding_row(roundtrip)
        assert roundtrip["value"]["changed_result_refusal"] == "result_id does not match the artifact content"


def test_t098_provider_identities(tmp_path):
    blocked = run("T098", tmp_path / "none")
    assert blocked["state"] == "blocked" and blocked["findings"] == []
    table = artifact(tmp_path / "none", "T098", "provider-identities.json")
    assert {pin["revision"] for pin in table["ciw_pins"]["scr"]} == {"a59aba283b0304faeeb3e5d305087e7709e171ca"}
    scr_repository = table["pins_by_repository"]["giasonpooni/Scientific-Computation-Runtime"]
    assert sorted(scr_repository) == ["5f0409743e0098a0691a88302a9b3dcdcbcf25fd",
                                      "a59aba283b0304faeeb3e5d305087e7709e171ca"]
    if not os.environ.get("CIW_LAB_SCR_REPO"):
        pytest.skip("set CIW_LAB_SCR_REPO for provider identities")
    bound = {"scr": os.environ["CIW_LAB_SCR_REPO"]}
    for role, name in (("csg", "CIW_LAB_CSG_REPO"), ("plsr-python", "CIW_LAB_PLSR_PYTHON")):
        if os.environ.get(name):
            bound[role] = os.environ[name]
    report = run("T098", tmp_path / "scr", bound)
    assert report["state"] == "completed"
    pins = claim(report, "The declared-workload and proved-heat workflows pin one SCR revision")
    assert pins["evidence_status"] == "numerically_verified"
    assert len(pins["value"]["set_revisions"]) == 4 and len(pins["value"]["scr_revisions"]) == 2
    accepted = claim(report, "Every bound provider checkout is clean")
    assert accepted["evidence_status"] == "numerically_verified" and accepted["value"]["refused"] == []
    assert accepted["value"]["accepted"]["scr"]["head"] == "a59aba283b0304faeeb3e5d305087e7709e171ca"
    assert accepted["value"]["accepted"]["scr"]["tree"] == "4068a711534932e8d89bb0d87d373376dafdf6cd"
    assert claim(report, "The working bytes of every clean")["evidence_status"] == "independently_verified"
    lockfiles = claim(report, "Tracked-source and lockfile digests")
    # Cross-read from Git's HEAD objects (origin git), not a second call of the same reader.
    assert lockfiles["evidence_status"] == "independently_verified"
    assert lockfiles["basis"]["independent_check"]["checker"]["implementation"].startswith("git cat-file")
    assert lockfiles["value"]["scr"]["lockfiles"]["crates/Cargo.lock"] == \
        "24b1db68d762d3457eb40465dae6c3edc8a2da2925d42f3f794d090d18a5f2bc"
    if "csg" in bound:
        # CSG pins its Python dependencies with uv.lock, not Cargo.lock.
        assert lockfiles["value"]["csg"]["lockfiles"] == {
            "uv.lock": "d552b3f5719f40d1abe6d7aebc6af4d7103dcea2744f2dba18a023e70e451b99"}
    adapter = claim(report, "CIW's pinned subprocess adapter")
    assert adapter["evidence_status"] == "numerically_verified"
    assert adapter["value"]["scr"]["refused_declared_pins"] == [] and len(adapter["value"]["scr"]["at_head"]) == 2
    # The refusal side is exercised by a control revision even though every declared SCR pin is at HEAD.
    assert adapter["value"]["scr"]["control_refused"] is True and adapter["value"]["refusal_cases"] >= 1
    control = [c for c in adapter["basis"]["checks"] if c["reference"].startswith("scr against control:")]
    assert len(control) == 1 and control[0]["observed_refusal"] == section.REVISION_REFUSAL
    if "plsr-python" in bound:
        plsr = claim(report, "The PLSR runtime installed in the bound plsr-python interpreter")
        assert plsr["evidence_status"] == "numerically_verified" and plsr["value"]["differing_source_files"] == []
    assert claim(report, "A matching HEAD, tree and lock digest")["evidence_status"] == "not_established"
    assert all(os.environ["CIW_LAB_SCR_REPO"] not in line for line in report["input_data"])
    retained = (tmp_path / "scr" / "artifacts" / "T098" / "provider-identities.json").read_text(encoding="utf-8")
    for location in {path for value in bound.values() for path in (value, str(Path(value).resolve()))} | {sys.executable}:
        assert location not in retained
    assert '"path": "<scr>"' in retained


def test_t098_refuses_an_unpinned_checkout_and_is_location_independent(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    first = _synthetic_repository(tmp_path / "ciw-lab-providers-a" / "scr")
    second = tmp_path / "ciw-lab-providers-b" / "scr"
    shutil.copytree(first, second, symlinks=True)
    reports = [run("T098", tmp_path / name, {"scr": str(path)}) for name, path in (("one", first), ("two", second))]
    report = reports[0]
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "not_established"
    headline = claim(report, "Every bound provider checkout is clean")
    assert headline["evidence_status"] == "not_established" and headline["value"]["refused"] == ["scr"]
    refusal = claim(report, "The bound scr checkout is refused against CIW's pins")
    assert refusal["evidence_status"] == "numerically_verified" and refusal["value"]["reasons"] == ["no CIW pin matches HEAD"]
    adapter = claim(report, "CIW's pinned subprocess adapter")
    assert adapter["evidence_status"] == "numerically_verified" and adapter["value"]["scr"]["at_head"] == []
    assert adapter["value"]["scr"]["control_refused"] is True
    assert adapter["value"]["refusal_cases"] == len(adapter["basis"]["checks"])
    assert claim(report, "The working bytes of every clean")["evidence_status"] == "independently_verified"
    lockfiles = claim(report, "Tracked-source and lockfile digests")
    assert lockfiles["evidence_status"] == "independently_verified"
    assert sorted(lockfiles["value"]["scr"]["lockfiles"]) == ["crates/Cargo.lock", "requirements.txt", "uv.lock"]
    assert not any(f["claim"].startswith("The SCR engine") for f in report["findings"])
    # The same checkout under another path verifies against the first run: prose names no location.
    assert runner.compare(tmp_path / "one", tmp_path / "two")["problems"] == []


def test_tree_recomputation_matches_git_on_a_synthetic_repository(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    repository = _synthetic_repository(tmp_path / "provider")
    identity = providers.checkout_identity(repository)
    assert identity["recomputed_tree"] == identity["tree"] and identity["mismatched_files"] == []
    # Every recognised lockfile, including a fully pinned requirements file; an unpinned one is not a lock.
    assert identity["lockfiles"] == {
        "crates/Cargo.lock": sha256(b"# lock\n").hexdigest(), "uv.lock": sha256(b"version = 1\n").hexdigest(),
        "requirements.txt": sha256((repository / "requirements.txt").read_bytes()).hexdigest()}
    objects = providers.head_object_digests(repository)
    assert {key: identity[key] for key in objects} == objects
    assert providers.compare_with_pins("scr", identity, providers.ciw_pins())["accepted"] is False
    assert providers.status_entries(repository) == 0
    (repository / "crates" / "a" / "lib.rs").write_bytes(b"pub fn g() {}\n")
    (repository / "uv.lock").write_bytes(b"version = 2\n")
    changed = providers.checkout_identity(repository)
    assert changed["recomputed_tree"] != changed["tree"]
    assert changed["mismatched_files"] == ["crates/a/lib.rs", "uv.lock"]
    assert changed["dirty"] is True and providers.status_entries(repository) == 2
    # The Git-object reading still sees HEAD, so a working-tree change separates the two readers.
    assert providers.head_object_digests(repository) == objects
    assert changed["lockfiles"]["uv.lock"] != objects["lockfiles"]["uv.lock"]
    assert changed["tracked_sha256"] != objects["tracked_sha256"]


@pytest.mark.parametrize("name,data,expected", [
    ("Cargo.lock", b"", True), ("a/b/uv.lock", b"", True), ("poetry.lock", b"", True), ("package-lock.json", b"{}", True),
    ("requirements.txt", b"numpy==2.4.3\nscipy @ https://example.invalid/scipy.whl\n", True),
    ("requirements-dev.txt", b"numpy>=2\n", False), ("requirements.txt", b"# empty\n", False),
    ("Cargo.toml", b"", False), ("uv.lock.orig", b"", False)])
def test_lockfile_recognition(name, data, expected):
    assert providers.is_lockfile(name, data) is expected


def test_interpreter_identity_records_version_and_digest():
    identity = providers.interpreter_identity(sys.executable)
    assert identity["python_version"] == platform.python_version()
    assert identity["sha256"] == sha256(Path(sys.executable).read_bytes()).hexdigest()


def _plsr_record(files=None, version=None):
    manifest = providers.plsr_manifest()
    return {"python_version": "3.12.0", "sha256": "0" * 64,
            "plsr": {"version": version or manifest["package_version"], "install_commit": None, "install_kind": "dir_info",
                     "files": dict(manifest["files"]) if files is None else files}}


def test_plsr_installation_finding_is_refuted_by_drift():
    manifest = providers.plsr_manifest()
    matching = section._plsr_installation_finding(_plsr_record(), None)
    assert matching["evidence_status"] == "numerically_verified" and matching["value"]["differing_source_files"] == []
    drifted = dict(manifest["files"], **{"runtime.py": "1" * 64})
    assert section._plsr_installation_finding(_plsr_record(drifted), None)["evidence_status"] == "not_established"
    assert section._plsr_installation_finding(_plsr_record(version="0.0.1"), None)["evidence_status"] == "not_established"
    assert section._plsr_installation_finding(None, "ValueError")["evidence_status"] == "not_established"


def test_t098_records_bound_interpreters_without_their_location(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    scr = _synthetic_repository(tmp_path / "scr")
    report = run("T098", tmp_path / "run", {"scr": str(scr), "ftr-python": sys.executable})
    retained = artifact(tmp_path / "run", "T098", "provider-identities.json")
    assert retained["interpreters"]["ftr-python"]["python_version"] == platform.python_version()
    assert retained["interpreters"]["ftr-python"]["sha256"] == sha256(Path(sys.executable).read_bytes()).hexdigest()
    assert report["provider_runtime_identity"]["ftr-python"]["python_version"] == platform.python_version()
    assert any(line.startswith("ftr-python: bound interpreter") for line in report["input_data"])
    assert sys.executable not in json.dumps(retained) and str(scr) not in json.dumps(retained)
    assert not any(f["claim"].startswith("The PLSR runtime") for f in report["findings"])


def test_exchange_workflow_pins_mirror_the_ci_workflow():
    workflow = ROOT / ".github" / "workflows" / "exchange.yml"
    if not workflow.is_file():
        pytest.skip(".github/workflows is not beside the tests")
    text = workflow.read_text(encoding="utf-8")
    for revision in providers.EXCHANGE_WORKFLOW_PINS.values():
        assert f"ref: {revision}" in text


def test_sp1_requirements_mirror_the_proved_heat_workflow():
    workflow = ROOT / ".github" / "workflows" / "proved-heat.yml"
    if not workflow.is_file():
        pytest.skip(".github/workflows is not beside the tests")
    text = workflow.read_text(encoding="utf-8")
    requirements = providers.SP1_REQUIREMENTS
    for value in (requirements["sp1_checkout"]["revision"], requirements["succinct_compiler_archive"]["url"],
                  requirements["succinct_compiler_archive"]["sha256"], requirements["guest_recipe"]["identity"],
                  requirements["guest_sha256"], *requirements["system_packages"]):
        assert value in text, value


def test_t099_locked_offline_scr_build(tmp_path):
    scr = os.environ.get("CIW_LAB_SCR_REPO")
    if not scr or shutil.which("cargo") is None:
        pytest.skip("set CIW_LAB_SCR_REPO and put cargo on PATH")
    report = run("T099", tmp_path, {"scr": scr})
    assert report["state"] == "partial"
    build = claim(report, section.BUILD_CLAIM)
    assert build["evidence_status"] == "numerically_verified" and build["value"]["exit_codes"] == [0, 0]
    assert build["value"]["cargo_lock_sha256"] == "24b1db68d762d3457eb40465dae6c3edc8a2da2925d42f3f794d090d18a5f2bc"
    assert claim(report, "The locked build leaves")["value"]["distinct_binary_digests"] == 1
    engine = claim(report, "The freshly built engine")
    assert engine["evidence_status"] == "independently_verified" and engine["value"] == [0, 219, 313, 219, 0]
    sp1 = claim(report, "The SP1 proved-heat locked build")
    assert sp1["evidence_status"] == "not_established" and sp1["value"]["attempted"] is False
    # The observed toolchain and host probes stay out of the compared prose (they are in provider_runtime_identity).
    toolchain = report["provider_runtime_identity"]["scr"]
    prose = json.dumps({name: report[name] for name in runner.PROSE_FIELDS})
    assert toolchain["rustc"] and toolchain["rustc"] not in prose and toolchain["cargo"] not in prose
    assert "probes here" not in prose


def test_t099_names_the_built_checkout_beside_its_label(tmp_path, monkeypatch):
    # The build and the engine run rest on the bound SCR checkout: each declares it, and the labels stay.
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    scr = _synthetic_repository(tmp_path / "scr")
    which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *args, **kwargs:
                        "cargo" if name == "cargo" else which(name, *args, **kwargs))
    monkeypatch.setattr(providers, "compare_with_pins", lambda role, identity, pins: {
        "role": role, "matched": ["ciw.declared_workload.PINS[numerical-heat]"], "unmatched": [], "tree_refusals": [],
        "accepted": True, "clean": True})
    engine = _mocked_scr_run(monkeypatch)
    digest = engine["sha256"]
    build = {"builds": [{"returncode": 0, "binary_sha256": digest}, {"returncode": 0, "binary_sha256": digest}],
             "binary": engine["binary"], "cargo_lock_sha256_before": "0" * 64, "cargo_lock_sha256_after": "0" * 64,
             "cargo": "cargo (stand-in)", "rustc": "rustc (stand-in)"}
    monkeypatch.setattr(section, "_locked_build", lambda ctx: build)
    report = run("T099", tmp_path / "run", {"scr": str(scr)})
    assert report["state"] == "partial"
    head = report["provider_runtime_identity"]["scr"]["head"]
    shown = f"pinned provider run ({providers.REPOSITORIES['scr']}@{head[:12]})"
    for prefix, label in ((section.BUILD_CLAIM, "numerically_verified"), ("The locked build leaves", "numerically_verified"),
                          ("The freshly built engine", "independently_verified")):
        record = claim(report, prefix)
        assert record["evidence_status"] == label and shown in finding_row(record), prefix
    assert claim(report, "The freshly built engine")["basis"]["provider"]["runtime_digest"] == "sha256:" + digest


def test_t100_labels_and_origins_stay_distinct(tmp_path):
    t096 = run("T096", tmp_path)
    t092 = run("T092", tmp_path)
    report = run("T100", tmp_path)
    assert report["state"] == "completed"
    primary = report["findings"][0]
    assert primary["value"] == {"reports_checked": 2, "reports_absent_of_T001_T099": 97, "label_violations": 0}
    rendered = claim(report, "Every finding of those reports shows its label")
    assert rendered["value"]["rendering_violations"] == 0 and rendered["evidence_status"] == "numerically_verified"
    basis = claim(report, "Every finding of those reports shows exactly its declared basis")
    assert basis["value"]["basis_violations"] == 0 and basis["evidence_status"] == "numerically_verified"
    # T096 seeds its records from a declared generator: the audit sees it and finds it named beside the label.
    sources = claim(report, "Every finding of those reports whose basis declares a generator")
    assert sources["evidence_status"] == "numerically_verified"
    assert sources["value"]["generator_findings"] >= 1 and sources["value"]["not_shown"] == 0
    table = artifact(tmp_path, "T100", "label-by-basis.json")
    assert table["findings"] == len(t092["findings"]) + len(t096["findings"])
    assert sum(row["synthetic_inputs"] for row in table["counts"].values()) == sources["value"]["generator_findings"]
    witness, row, kept = section._rendering_probe()
    assert kept is True and row == finding_row(witness)
    probe = claim(report, "render_markdown keeps")
    assert probe["evidence_status"] == "numerically_verified" and probe["value"]["row_cells"] == 4
    energy = claim(report, "CIW keeps the synthetic energy fixture")
    assert energy["evidence_status"] == "numerically_verified"
    assert energy["value"]["synthetic"]["classification"] == "synthetic_only"
    relabel = claim(report, "A resealed relabel")
    assert relabel["evidence_status"] == "numerically_verified"
    assert relabel["value"]["fresh_occurrence_execute"] == "accepted"
    assert relabel["value"]["fresh_bundle"]["classification"] == "physical_domain_measurement"
    assert relabel["counterexample"]["statement"].startswith("CIW energy records can distinguish")
    assert claim(report, "CIW refuses free-energy sources")["evidence_status"] == "numerically_verified"
    # Provider-backed versus fabricated: the classifier refuses the invented tree, not a copied pin.
    invented = claim(report, "The workspace classifier labels a fabricated")
    assert invented["evidence_status"] == "numerically_verified" and invented.get("counterexample") is None
    assert invented["value"]["reopen"] == "accepted" and invented["value"]["numerical_labels"] == ["not_established"]
    assert invented["value"]["reader_view"]["source_tree_is_ciw_pin"] is False
    copied = claim(report, "A fabricated numerical-heat bundle sealed with CIW's pinned")
    assert copied["evidence_status"] == "numerically_verified"
    assert copied["value"]["numerical_labels"] == ["provider_backed"]
    assert copied["value"]["reader_view"]["source_tree_is_ciw_pin"] is True
    assert copied["counterexample"]["statement"].startswith("CIW's retained records and their classification")
    assert copied["counterexample"]["witness"]["matched_pins"] == ["ciw.declared_workload.PINS[numerical-heat]"]
    assert not any(f["claim"].startswith("A fabricated, content-consistent") for f in report["findings"])
    assumptions = " ".join(report["unresolved_assumptions"])
    assert "Session.from_workspace (reopen) does not compare it" in assumptions
    # The invented-tree check reaches only kinds whose pinned revision has a tree CIW records; the others are named.
    from ciw.lab.bridge import pins_without_tree
    kinds = sorted({kind for kind, _, _ in pins_without_tree()["kinds"]})
    assert "telemetry" in kinds and "numerical-heat" not in kinds
    assert f"ciw.lab.bridge.pins_without_tree ({', '.join(kinds)}) any tree is accepted (tree_pinned: false)" in assumptions
    assert "pins_without_tree" in section.NEXT_STEPS["T100"] and "source tree" in section.NEXT_STEPS["T100"]
    doc = (ROOT / "docs" / "lab" / "EXCHANGE_BUNDLES.md").read_text(encoding="utf-8")
    non_claim = doc.split("- **A retained runtime identity is a declaration too.**", 1)[1].split("\n- **", 1)[0]
    assert "`ciw.lab.bridge.pins_without_tree`" in non_claim and "`tree_pinned: false`" in non_claim
    # The audit's reach: declared components only, and one provider slot per finding.
    assert "The audit sees only the components a finding declares" in assumptions
    assert "One provider slot shows one provider of a result several providers produced" in assumptions
    physical = [f for f in report["findings"] if f["domain"] in ("physical", "sensor_performance")]
    assert physical and all(f["evidence_status"] == "not_established" for f in physical)
    for task_report in (t092, t096, report):
        assert task_report["recommended_next_task"] == section.NEXT_STEPS[task_report["task_id"]]


def test_t100_classifier_study_separates_an_invented_tree_from_a_copied_pin():
    from ciw.proved_heat import PIN
    study = section._provider_origin_study()
    invented, copied = study["fabricated"], study["copied_pin"]
    assert invented["reopen"] == copied["reopen"] == "accepted"
    assert invented["numerical_labels"] == ["not_established"]
    row, = invented["runtime_pins"]
    assert row["matched"] is None and f"is not the tree {PIN['source_tree']}" in row["problem"]
    assert copied["numerical_labels"] == ["provider_backed"]
    row, = copied["runtime_pins"]
    assert row["matched"] == ["ciw.declared_workload.PINS[numerical-heat]"] and row["tree_pinned"] is True
    # Only the sealed source tree differs; the reader-visible traces of the fabrication are what this lab wrote.
    assert invented["bundle_id"] != copied["bundle_id"]
    assert {key: value for key, value in invented["reader_view"].items() if key != "source_tree_is_ciw_pin"} == \
        {key: value for key, value in copied["reader_view"].items() if key != "source_tree_is_ciw_pin"}
    catalog = fixtures.fabricated_heat_catalog([0, 1, 2, 3, 0], source_tree=PIN["source_tree"])
    assert catalog["bundles"][0]["native"]["runtimes"]["scr"]["source_tree"] == PIN["source_tree"]


def _report_with(findings):
    return build_report(QUEUE["T001"], "partial", {}, findings)


PROVIDER_FINDING = finding("Provider run of the probe workload", "numerical", 1.0,
                           {"provider": {"repository": "owner/probe-runtime", "revision": "0123456789abcdef0123",
                                         "source_tree": "f" * 40, "executed": True}})
GENERATOR_FINDING = finding("Check on seeded synthetic inputs", "numerical", 0.0,
                            {"generator": {"name": "ciw.lab.probe_bench", "seed": 602026},
                             "checks": [{"reference_kind": "analytic", "reference": "probe", "observed": 0.0,
                                         "tolerance": 0.0, "passed": True}]})


def _origin_words_only(report):
    """A renderer that keeps the label column but names only basis component words, not the generator or provider."""
    lines = render_markdown(report).splitlines()
    start = lines.index(FINDINGS_RULE) + 1
    rows = [finding_row(record).rsplit(" | ", 1)[0] + f" | {describe_origin(finding_origin(record))} |"
            for record in report["findings"]]
    return "\n".join(lines[:start] + rows) + "\n"


def test_t100_flags_a_rendered_row_that_hides_its_generator_or_provider(tmp_path, monkeypatch):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "T001.json").write_text(
        runner.dumps(_report_with([GENERATOR_FINDING, PROVIDER_FINDING])), encoding="utf-8")
    report = run("T100", tmp_path)
    sources = claim(report, "Every finding of those reports whose basis declares a generator")
    assert sources["value"] == {"reports_checked": 1, "reports_absent_of_T001_T099": 98, "generator_findings": 1,
                                "provider_findings": 1, "not_shown": 0}
    assert sources["evidence_status"] == "numerically_verified" and report["state"] == "completed"
    table = artifact(tmp_path, "T100", "label-by-basis.json")["counts"]
    assert table["numerically_verified"]["synthetic_inputs"] == 1 and table["provider_backed"]["provider"] == 1
    monkeypatch.setattr(section, "render_markdown", _origin_words_only)
    hidden = run("T100", tmp_path)
    sources = claim(hidden, "Every finding of those reports whose basis declares a generator")
    assert sources["value"]["not_shown"] == 2 and sources["evidence_status"] == "not_established"
    assert claim(hidden, "Every finding of those reports shows exactly its declared basis")["value"]["basis_violations"] == 2
    assert hidden["state"] == "partial"
    audit = artifact(tmp_path, "T100", "visibility-audit.json")
    assert audit["hidden_generator_or_provider"] == ["T001[0]", "T001[1]"]


ACQUISITION = {"device": "nvml:GPU-0", "raw_sha256": "a" * 64, "acquired_at": "epoch+0ns", "calibration": "declared"}
FAILED_CHECK = {"reference_kind": "analytic", "reference": "probe", "observed": 1.0, "tolerance": 0.0, "passed": False}


def _upgraded_acquisitions(record):
    """A renderer that shows every declared acquisition record as an accepted hardware acquisition."""
    from ciw.lab.evidence import ACCEPTED_ACQUISITION, UNACCEPTED_ACQUISITION, describe_basis
    text = describe_basis(record)
    if "acquisition" not in record["basis"]:
        return text
    return text.replace(UNACCEPTED_ACQUISITION, f"{ACCEPTED_ACQUISITION} ({record['basis']['acquisition']['device']})")


def test_t100_flags_an_unaccepted_acquisition_shown_as_hardware_acquisition(tmp_path, monkeypatch):
    from ciw.lab import report as report_module
    authority = finding("The coupon energy draw is within its declared budget", "production_acceptance", None,
                        {"acquisition": ACQUISITION})
    refuted = finding("The heater reached its set temperature", "physical", None,
                      {"acquisition": ACQUISITION, "checks": [FAILED_CHECK]})
    assert authority["evidence_status"] == refuted["evidence_status"] == "not_established"
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "T001.json").write_text(runner.dumps(_report_with([authority, refuted])), encoding="utf-8")
    honest = run("T100", tmp_path)
    assert honest["state"] == "completed"
    assert claim(honest, "Every finding of those reports shows exactly its declared basis")["value"]["basis_violations"] == 0
    # No audited finding declares a generator or provider: the untested visibility claim is recorded as such.
    untested = claim(honest, "Every finding of those reports whose basis declares a generator")
    assert untested["claim"].endswith("(no audited finding declares one)")
    assert untested["evidence_status"] == "not_established" and untested["expected_not_established"] is True
    assert untested["value"]["generator_findings"] == untested["value"]["provider_findings"] == 0
    assert not any(f["claim"].startswith("No finding of those reports declares") for f in honest["findings"])
    # The shared renderer upgrades the display: finding_row agrees with it, the independently built cell does not.
    monkeypatch.setattr(report_module, "describe_basis", _upgraded_acquisitions)
    upgraded = run("T100", tmp_path)
    assert claim(upgraded, "Every finding of those reports shows its label")["value"]["rendering_violations"] == 0
    basis = claim(upgraded, "Every finding of those reports shows exactly its declared basis")
    assert basis["value"]["basis_violations"] == 2 and basis["evidence_status"] == "not_established"
    assert upgraded["state"] == "partial"
    violations = artifact(tmp_path, "T100", "visibility-audit.json")["basis_violations"]
    assert all(section.UPGRADED_ACQUISITION in violation for violation in violations) and len(violations) == 2


def test_basis_audit_compares_the_whole_cell():
    from ciw.lab.evidence import describe_basis
    # A component word inside a declared identity does not stand in for the component.
    bench = finding("Check on seeded inputs", "numerical", 0.0,
                    {"generator": {"name": "derivation and reference checks bench", "seed": 1}, "derivation": "eq. 3",
                     "checks": [dict(FAILED_CHECK, observed=0.0, passed=True)]})
    assert section._basis_problems(bench, describe_basis(bench)) == []
    dropped = "synthetic inputs (derivation and reference checks bench, seed 1)"
    assert section._basis_problems(bench, dropped) == ["derivation", "reference_checks"]
    reordered = "synthetic inputs (derivation and reference checks bench, seed 1), derivation, reference checks"
    assert section._basis_problems(bench, reordered) == ["order or extra text"]
    # An acquisition reads as hardware acquisition only on a physical finding it establishes.
    measured = finding("The probe run drew the recorded energy", "physical", 1.0, {"acquisition": ACQUISITION})
    assert measured["evidence_status"] == "hardware_measured"
    assert section._basis_problems(measured, "hardware acquisition (nvml:GPU-0)") == []
    assert section._basis_problems(measured, "hardware acquisition (nvml:GPU-1)") == ["acquisition"]
    refuted = finding("The heater reached its set temperature", "physical", None,
                      {"acquisition": ACQUISITION, "checks": [FAILED_CHECK]})
    assert section._basis_problems(refuted, "declared acquisition record (not accepted), reference checks") == []
    assert section._basis_problems(refuted, "hardware acquisition (x), reference checks") == [
        "acquisition", section.UPGRADED_ACQUISITION]
    # Nor anywhere a finding declares no acquisition at all.
    assert section._basis_problems(GENERATOR_FINDING, "hardware acquisition (x), " + describe_basis(GENERATOR_FINDING)) \
        == [section.UPGRADED_ACQUISITION]
    assert section._basis_problems(finding("Nothing declared", "numerical", None, {}), "no declared basis") == []
    assert section._basis_problems(finding("Nothing declared", "numerical", None, {}), "reference checks") == ["none"]


def _unescaped_markdown(report):
    """The renderer before claim escaping: the claim goes into its cell verbatim."""
    lines = render_markdown(report).splitlines()
    start = lines.index(FINDINGS_RULE) + 1
    rows = [finding_row(record).replace(record["claim"].replace("|", "\\|"), record["claim"], 1)
            for record in report["findings"]]
    return "\n".join(lines[:start] + rows) + "\n"


def test_t100_flags_a_retained_report_whose_label_leaves_its_column(tmp_path, monkeypatch):
    record = finding("pipe | in a claim", "numerical", 1.0, {"generator": {"name": "probe"}})
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "T001.json").write_text(runner.dumps(_report_with([record])), encoding="utf-8")
    monkeypatch.setattr(section, "render_markdown", _unescaped_markdown)
    report = run("T100", tmp_path)
    assert report["findings"][0]["value"]["label_violations"] == 0
    rendered = claim(report, "Every finding of those reports shows its label")
    assert rendered["value"]["rendering_violations"] == 1
    assert rendered["evidence_status"] == "not_established" and report["state"] == "partial"
    assert claim(report, "An unescaped pipe")["evidence_status"] == "numerically_verified"


def test_render_markdown_pipe_probe_matches_the_renderer():
    record = finding("claim with a | pipe", "numerical", 1.0, {"generator": {"name": "rendering probe"}})
    validate_finding(record)
    row = render_markdown(build_report(QUEUE["T100"], "partial", {}, [record])).splitlines()[-1]
    escaped = "claim with a \\| pipe" in row
    # GitHub-flavoured Markdown drops cells beyond the header's four: unescaped, the label cell is lost.
    assert section._cells(row) == (4 if escaped else 5)
    witness, probe_row, kept = section._rendering_probe()
    assert probe_row == row == finding_row(witness) and kept is escaped
    assert section.HEADER_CELLS == 4
    probe = section._probe_finding(witness, probe_row, kept)
    assert probe["evidence_status"] == "numerically_verified" and probe.get("counterexample") is None
    assert probe["claim"].startswith("render_markdown keeps" if escaped else "An unescaped pipe")
    unescaped = _unescaped_markdown(build_report(QUEUE["T100"], "partial", {}, [record])).splitlines()[-1]
    assert section._cells(unescaped) == 5
    shifted = section._probe_finding(witness, unescaped, False)
    assert shifted["evidence_status"] == "numerically_verified" and shifted["counterexample"]
    # A row that lost its label column without the extra cell a raw pipe makes is no witness of that counterexample.
    reordered = f"| {record['claim'].replace('|', chr(92) + '|')} | `synthetic` | 1 | synthetic inputs (rendering probe) |"
    assert section._cells(reordered) == 4
    lost = section._probe_finding(witness, reordered, False)
    # It refutes the label-column claim instead, and carries no counterexample for a catalogue (T157) to list.
    assert lost["claim"] == section.KEEPS_LABEL_COLUMN and lost["evidence_status"] == "not_established"
    assert "counterexample" not in lost and lost["value"]["label_column_shifted"] is True
    assert [check["reference"] for check in lost["basis"]["checks"] if not check["passed"]] == [
        "rendered rows whose third cell is not the finding's label", "rendered row differing from report.finding_row"]
    assert research_portfolio._counterexample_statements([{"task_id": "T100", "findings": [lost]}], {"T100"}) == []


def test_t092_next_step_names_the_manifest_that_pins_each_kind():
    # Telemetry and calibrated-observable bundles come from different stacks pinned in different manifests.
    from ciw.lab.bridge import declared_pins
    kinds = declared_pins()["kinds"]
    doc = (ROOT / "docs" / "lab" / "EXCHANGE_BUNDLES.md").read_text(encoding="utf-8")
    open_question = doc.split("\n- T092:", 1)[1].split("\n- T093:", 1)[0]
    for kind, manifest in (("telemetry", "telemetry-runtimes.json"),
                           ("calibrated-observable", "calibrated-observable-runtimes.json")):
        assert all(pin["declared_in"].startswith(f"ciw/{manifest}") for group in kinds[kind].values() for pin in group)
        assert f"src/ciw/{manifest} for {kind} bundles" in section.NEXT_STEPS["T092"], kind
        assert f"`{manifest}`" in open_question, kind


def test_next_steps_name_forward_work():
    # No next step of T091-T100 points at a queue task, since every queue task has run by the time it is read.
    everything = {task["id"] for task in registry.load_queue()["tasks"]}
    assert sorted(section.NEXT_STEPS) == [f"T{number:03d}" for number in range(91, 101)]
    for task_id, text in section.NEXT_STEPS.items():
        kept, stale = planner.next_step_items(text, task_id, everything)
        assert stale == [] and kept and all(kind == "question" for kind, *_ in kept), task_id
        assert not text.startswith("T") and "then T" not in text, task_id


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


def test_retained_artifacts_name_host_locations_by_role(tmp_path):
    checkout = tmp_path / "checkouts" / "scr"
    value = {"repository_root": str(checkout), "nested": [f"{checkout}/crates/Cargo.lock", "no path here"],
             "python_executable": sys.executable}
    located = section._located(value, {"scr": checkout, "python": sys.executable})
    assert located == {"repository_root": "<scr>", "nested": ["<scr>/crates/Cargo.lock", "no path here"],
                       "python_executable": "<python>"}
