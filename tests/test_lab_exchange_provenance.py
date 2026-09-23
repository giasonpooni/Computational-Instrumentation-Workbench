"""Exchange/provenance lab tasks T077-T090 against the real, offline CIW integrity layer.

One module-scoped lab context builds the oscillator session, energy-accuracy
bundles, replays and saved workspaces once; every task report is then checked
for its state, evidence labels, key numbers and exact refusal messages.
"""
import json

import pytest

from ciw.lab import exchange_provenance as ep
from ciw.lab import exchange_provenance_common as common
from ciw.lab import runner
from ciw.lab.registry import _REGISTRY, load_queue
from ciw.lab.report import validate_report

TASKS = [f"T0{number}" for number in range(77, 91)]


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    if not common.fixture_available():
        pytest.skip("bundled energy-accuracy fixture logs are absent")
    ctx = runner.Context(tmp_path_factory.mktemp("exchange-lab"))
    queue = {item["id"]: item for item in load_queue()["tasks"]}
    reports = {}

    def run(task_id):
        if task_id not in reports:
            reports[task_id] = validate_report(runner.run_task(queue[task_id], _REGISTRY[task_id], ctx, {}))
        return reports[task_id]
    run.ctx = ctx
    return run


def _findings(report):
    return {record["claim"]: record for record in report["findings"]}


def _artifact(lab, task_id, name):
    return (lab.ctx.output_dir / "artifacts" / task_id / name).read_text(encoding="utf-8")


def _survivors(report):
    return sorted(record["value"]["mutant"] for record in report["findings"] if record.get("counterexample")
                  and isinstance(record["value"], dict) and "mutant" in record["value"])


def _kill_messages(report):
    """Observed refusal per predicted-refusal mutant, from the task's kill finding."""
    kills = next(record for record in report["findings"] if isinstance(record["value"], list)
                 and all(isinstance(row, dict) and "mutant" in row for row in record["value"]))
    assert kills["evidence_status"] == "numerically_verified"
    return {row["mutant"]: row["observed"] for row in kills["value"]}


def _authentication_is_unestablished(report):
    record = report["findings"][-1]
    return (record["claim"].startswith("Retained workspace records are authenticated")
            and record["domain"] == "provenance" and record["evidence_status"] == "not_established"
            and record["expected_not_established"] is True)


def test_every_section_task_is_registered_with_its_regression_test():
    for task_id in TASKS:
        implementation = _REGISTRY[task_id]
        assert implementation.regression_tests and all(node.startswith("tests/test_lab_exchange_provenance.py::")
                                                       for node in implementation.regression_tests)
        assert ep.MODULE in implementation.changed_files


def test_identity_matrix(lab):
    report = lab("T077")
    assert report["state"] == "completed"
    assert report["evidence_status"]["primary"] == "numerically_verified"
    summary = report["findings"][0]["value"]
    assert summary["rows"] == 28 and summary["exercised_rows"] == 27
    assert summary["properties"] == summary["properties_held"] == 67
    assert summary["derivation_classes"]["fresh_event_uuid"] == 5
    found = _findings(report)
    gap = found["Oscillator operation results carry no replay-stable numerical-result identity"]
    assert gap["evidence_status"] == "numerically_verified" and gap["value"]["data_equal"] is True
    assert gap["counterexample"]["statement"].startswith("Every retained CIW operation result")
    excluded = found["The energy replay bundle identity excludes its replay receipt and verification"]
    assert excluded["value"]["digest_unchanged_without_receipt"] is True
    assert _authentication_is_unestablished(report)
    matrix = {row["identity"]: row for row in json.loads(_artifact(lab, "T077", "identity-matrix.json"))["rows"]}
    assert matrix["energy numerical_result_id"]["across_replay"] == "stable"
    assert matrix["energy step execution_id"]["derivation_class"] == "fresh_event_uuid"
    assert matrix["exchange batch_id"]["derivation_class"] == "caller_declared"
    assert matrix["ESM candidate_id / candidate execution_id"]["exercised"] is False


def test_exact_source_bytes(lab):
    report = lab("T078")
    assert report["state"] == "completed"
    found = _findings(report)
    retained = report["findings"][0]
    assert retained["evidence_status"] == "numerically_verified"
    assert retained["value"]["sources"] == 12
    assert all(retained["value"][key] == 0 for key in ("live_byte_mismatches", "reopened_byte_mismatches",
                                                       "bundle_byte_mismatches", "evidence_id_mismatches",
                                                       "byte_count_mismatches"))
    transports = found["Non-canonical base64 transports are refused rather than normalized into retained bytes"]
    assert transports["value"] == {"base64/line-wrapped": "Source bytes must use canonical base64",
                                   "base64/missing-padding": "Source bytes must use canonical base64",
                                   "base64/noncanonical-trailing-bits": "Source bytes must use bounded canonical base64"}
    physical = found["The retained energy logs are real GPU energy measurements"]
    assert physical["domain"] == "physical" and physical["evidence_status"] == "not_established"
    assert report["physical_validation_status"]["status"] == "not_established"


def test_whitespace_variants(lab):
    report = lab("T079")
    assert report["state"] == "completed"
    assert report["evidence_status"]["counts"]["numerically_verified"] == 4
    distinct = report["findings"][0]["value"]
    assert distinct["variants"] == distinct["distinct_evidence_id"] == distinct["distinct_source_id"] == 8
    assert report["findings"][1]["value"] == {"experiment_digest": 1, "log_digest": 1, "numerical_result_id": 1}
    typed = report["findings"][2]
    assert typed["value"]["python_equal"] is True and typed["value"]["canonical_equal"] is False
    assert typed["value"]["observed"] == "Workload solver differs from plan"
    assert report["findings"][3]["value"]["observed"] == "The bound runtime did not return finite, unambiguous JSON"


def test_identity_separation(lab):
    report = lab("T080")
    assert report["state"] == "completed"
    separation = report["findings"][0]["value"]
    assert separation["cross_role_overlaps"] == 0
    assert separation["energy"] == {"numerical_result_id_unchanged": True, "operation_id_unchanged": True,
                                    "result_id_changed": True}
    assert _kill_messages(report)["alias.execution-result"] == "Execution/result execution_id binding mismatch"
    assert _survivors(report) == ["revision.gap"]
    assert _authentication_is_unestablished(report)


def test_numerical_identity(lab):
    report = lab("T081")
    assert report["state"] == "completed"
    stable = report["findings"][0]["value"]
    assert stable["occurrences"] == stable["distinct_result_ids"] == 12
    assert stable["distinct_numerical_result_ids"] == 1 and stable["recomputation_mismatches"] == 0
    assert _kill_messages(report)["energy-data.reforged"] == "Retained energy analysis binding differs"
    assert _survivors(report) == ["oscillator-stats.legacy", "oscillator-stats.resealed"]


def test_fresh_occurrences(lab):
    report = lab("T082")
    assert report["state"] == "completed"
    table = report["findings"][0]["value"]
    assert all(entry["distinct"] == entry["occurrences"] for entry in table.values())
    assert table["execution"]["occurrences"] == 17 and table["replay"]["occurrences"] == 3
    kills = _kill_messages(report)
    assert kills["fresh.energy-replay-reuse"] == ("Declared workload bundles must have distinct execution and "
                                                  "reproduction occurrences")
    assert kills["fresh.cross-namespace"] == "Identity collision between recording operations and retained workflows"
    assert _survivors(report) == ["fresh.created-at-shift"]


def test_replay_receipt_binding(lab):
    report = lab("T083")
    assert report["state"] == "completed"
    assert report["findings"][0]["value"] == {"binding_properties": 30, "held": 30, "receipts": 3}
    assert _survivors(report) == ["receipt.deleted"]
    independent = _findings(report)["Replay agreement establishes verification by an independent party"]
    assert independent["evidence_status"] == "not_established"
    assert independent["value"]["receipt_independent_flags"] == [False, False, False]


def test_receipt_digest_mutations(lab):
    source, replayed = lab("T084"), lab("T085")
    assert source["state"] == replayed["state"] == "completed"
    assert _kill_messages(source) == {
        "receipt-source.naive": "Invalid retained energy replay receipt",
        "receipt-source.replay-id": "Retained energy analysis binding differs",
        "receipt-source.subject-rebound": "Replay source must already belong to this workbench",
        "receipt-source.self": "Invalid retained energy replay receipt",
        "receipt-source.other-source": "Replay source must already belong to this workbench"}
    assert _survivors(source) == ["receipt-source.sibling-execution"]
    assert set(_kill_messages(replayed).values()) == {"Invalid retained energy replay receipt"}
    assert _survivors(replayed) == ["receipt-replayed.reidentified-bundle"]
    assert _authentication_is_unestablished(source) and _authentication_is_unestablished(replayed)


def test_verification_mutations(lab):
    subject, method, independent = lab("T086"), lab("T087"), lab("T088")
    for report in (subject, method, independent):
        assert report["state"] == "completed" and report["evidence_status"]["primary"] == "numerically_verified"
        assert _authentication_is_unestablished(report)
    assert _kill_messages(subject)["oscillator-verification.resealed"] == (
        "Protocol v1 saved results must remain not_verified with verification_id null")
    assert _kill_messages(subject)["exchange.verification-subject"] == "verification_id does not match the artifact content"
    assert _survivors(subject) == ["receipt-subject.rebound-with-source"]
    assert _kill_messages(method)["receipt-method.resealed"] == "Retained energy analysis binding differs"
    assert _survivors(method) == ["oscillator-method.injected"]
    assert _kill_messages(independent)["esm.inspection-independent"] == "Native ESM inspection binding or scope mismatch"
    assert _survivors(independent) == ["exchange.verification-independent", "oscillator-independent.injected"]


def test_admission_and_runtime_mutations(lab):
    admission, runtime = lab("T089"), lab("T090")
    assert admission["state"] == runtime["state"] == "completed"
    assert _kill_messages(admission)["esm.canonical-admission"] == "ESM may retain candidate evidence only"
    authority = _findings(admission)["A retained replay receipt or verification authorizes admission of the replayed "
                                     "result into canonical state"]
    assert authority["domain"] == "production_acceptance" and authority["evidence_status"] == "not_established"
    assert _survivors(admission) == ["oscillator-admission.injected"]
    kills = _kill_messages(runtime)
    assert kills["oscillator-runtime.execution-only"] == "Execution/result runtime binding mismatch"
    assert kills["energy-runtime.replay-only"] == "Retained energy analysis binding differs"
    assert _survivors(runtime) == ["energy-runtime.all-bundles", "energy-runtime.python-version",
                                   "oscillator-runtime.both"]
    detected = _findings(runtime)["A consistently forged energy runtime identity is detected only when a replay "
                                  "recomputes the current analysis identity"]
    assert detected["value"]["replay_after_reopen"] == {"message": "Retained energy analysis binding differs",
                                                         "outcome": "refused"}
    matrix = json.loads(_artifact(lab, "T090", "mutation-matrix.json"))
    assert matrix["harness"] == {"reforge_reproduces_ciw_digests": True, "unchanged_workspace_reopens": True}
    assert len(matrix["rows"]) == len(ep.MUTANTS) + 7
    assert all(row["matches_prediction"] for row in matrix["rows"])
    assert sum(not row["killed"] for row in matrix["rows"]) == 15


def test_tasks_block_without_the_fixture_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("CIW_LAB_REPOSITORY_ROOT", str(tmp_path / "absent"))
    queue = {item["id"]: item for item in load_queue()["tasks"]}
    ctx = runner.Context(tmp_path / "out")
    for task_id in ("T077", "T078", "T084", "T089"):
        report = validate_report(runner.run_task(queue[task_id], _REGISTRY[task_id], ctx, {}))
        assert report["state"] == "blocked"
        assert report["experiment"].startswith("Blocked: the bundled energy-accuracy fixture logs")
        assert all(record["evidence_status"] == "not_established" for record in report["findings"])
        assert {record["domain"] for record in report["findings"]} <= {"physical", "production_acceptance"}
    assert report["findings"][0]["domain"] == "production_acceptance"
