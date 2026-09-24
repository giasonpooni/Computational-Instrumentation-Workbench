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
from ciw.lab.registry import load_queue, module_implementations
from ciw.lab.report import validate_report

TASKS = [f"T0{number}" for number in range(77, 91)]
IMPLEMENTATIONS = module_implementations("exchange_provenance")
VALIDATOR_ROWS = {"T086": ["esm.bundle-subject", "exchange.verification-subject"],
                  "T088": ["esm.inspection-independent", "esm.candidate-independent",
                           "exchange.verification-independent"],
                  "T089": ["esm.canonical-admission", "esm.candidate-admitted"]}
STEMS = {"T080": "aliasing-mutations", "T081": "numerical-mutations", "T082": "freshness-mutations",
         "T083": "receipt-mutations", **{task_id: spec["stem"] for task_id, spec in ep.MUTATION_TASKS.items()}}


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    if not common.fixture_available():
        pytest.skip("bundled energy-accuracy fixture logs are absent")
    ctx = runner.Context(tmp_path_factory.mktemp("exchange-lab"))
    queue = {item["id"]: item for item in load_queue()["tasks"]}
    reports = {}

    def run(task_id):
        if task_id not in reports:
            reports[task_id] = validate_report(runner.run_task(queue[task_id], IMPLEMENTATIONS[task_id], ctx, {}))
        return reports[task_id]
    run.ctx = ctx
    return run


def _findings(report):
    return {record["claim"]: record for record in report["findings"]}


def _artifact(lab, task_id, name):
    return (lab.ctx.output_dir / "artifacts" / task_id / name).read_text(encoding="utf-8")


def _labels(report, primary, counts, state="completed"):
    assert report["state"] == state
    assert report["evidence_status"]["primary"] == primary
    assert {label: count for label, count in report["evidence_status"]["counts"].items() if count} == counts


def _survivor_findings(report):
    return {record["value"]["mutant"]: record for record in report["findings"] if record.get("counterexample")
            and isinstance(record["value"], dict) and "mutant" in record["value"]}


def _survivors(report):
    """Surviving mutants; each one also checks, beyond its acceptance, what the reopened session shows."""
    found = _survivor_findings(report)
    for record in found.values():
        checks = record["basis"]["checks"]
        assert record["evidence_status"] == "numerically_verified" and len(checks) >= 2
        assert all(check["passed"] for check in checks)
    return sorted(found)


def _kill_messages(report):
    """Observed refusal per predicted-refusal mutant, from the task's kill finding."""
    kills = next(record for record in report["findings"] if isinstance(record["value"], list)
                 and all(isinstance(row, dict) and "mutant" in row for row in record["value"]))
    assert kills["evidence_status"] == "numerically_verified"
    assert kills["uncertainty"]["kind"] == "exact"
    return {row["mutant"]: row["observed"] for row in kills["value"]}


def _authentication_is_unestablished(report):
    record = report["findings"][-1]
    return (record["claim"].startswith("Retained workspace records are authenticated")
            and record["domain"] == "provenance" and record["evidence_status"] == "not_established"
            and record["expected_not_established"] is True)


def _retained_rows(lab, task_id):
    """The task's retained mutation rows equal its catalogue rows plus its pure-validator rows."""
    rows = json.loads(_artifact(lab, task_id, STEMS[task_id] + ".json"))["rows"]
    expected = [mutant.name for mutant in ep.MUTANTS if mutant.task == task_id] + VALIDATOR_ROWS.get(task_id, [])
    assert [row["name"] for row in rows] == expected
    assert all(row["outcome_matches_prediction"] and row["message_matches_pin"] for row in rows)
    return {row["name"]: row for row in rows}


def test_every_section_task_is_registered_with_its_regression_test():
    assert set(IMPLEMENTATIONS) == set(TASKS)
    for task_id in TASKS:
        implementation = IMPLEMENTATIONS[task_id]
        assert implementation.regression_tests
        for node in implementation.regression_tests:
            path, name = node.split("::")
            assert path == "tests/test_lab_exchange_provenance.py" and callable(globals().get(name))
        assert ep.MODULE in implementation.changed_files


def test_identity_matrix(lab):
    report = lab("T077")
    # Partial: two planned rows (ESM candidate, pinned-provider runtime) cannot run offline.
    _labels(report, "numerically_verified", {"numerically_verified": 3, "not_established": 2}, state="partial")
    summary = report["findings"][0]["value"]
    assert summary["rows"] == 30 and summary["exercised_rows"] == 28
    assert summary["unexercised_rows"] == ["ESM candidate_id / candidate execution_id",
                                           "pinned-provider runtime identity (ciw.subprocess-runtime.v1)"]
    assert report["unresolved_assumptions"][0].startswith("Partial: ESM candidate_id")
    assert summary["properties"] == summary["properties_held"] == 97 and summary["rederived_properties"] == 11
    assert summary["reopen_stability_properties"] == 23
    assert summary["derivation_classes"]["fresh_event_uuid"] == 5
    found = _findings(report)
    gap = found["Oscillator operation results carry no replay-stable numerical-result identity"]
    assert gap["evidence_status"] == "numerically_verified" and gap["value"]["data_equal"] is True
    assert gap["counterexample"]["statement"].startswith("Every retained CIW operation result")
    excluded = found["The energy replay bundle identity excludes its replay receipt and verification"]
    assert excluded["value"]["digest_unchanged_without_receipt"] is True
    assert excluded["value"]["digest_unchanged_with_replaced_verification"] is True
    producers = found["The retained log's device and kernel identities identify the producing GPU and code"]
    assert producers["domain"] == "physical" and producers["evidence_status"] == "not_established"
    assert producers["value"]["fixture_hashes_are_placeholders"] is True
    assert _authentication_is_unestablished(report)
    kinds = {check["reference_kind"] for check in report["findings"][0]["basis"]["checks"]}
    assert kinds == {"cross_implementation", "exact_arithmetic"}
    matrix = {row["identity"]: row for row in json.loads(_artifact(lab, "T077", "identity-matrix.json"))["rows"]}
    numerical = matrix["energy numerical_result_id"]
    assert numerical["across_replay"] == "stable for canonically identical logs" and "log_digest" in numerical["binds"]
    assert numerical["properties"]["metadata_only_resealed_edit_changes_it"] is True
    assert matrix["energy step execution_id"]["derivation_class"] == "fresh_event_uuid"
    assert matrix["exchange batch_id"]["derivation_class"] == "caller_declared"
    assert matrix["ESM candidate_id / candidate execution_id"]["exercised"] is False
    assert matrix["pinned-provider runtime identity (ciw.subprocess-runtime.v1)"]["exercised"] is False
    # Every exercised identity predicted stable across reopen is compared before and after reopen, and holds.
    stable = [row for row in matrix.values() if row["exercised"] and row["across_reopen"].startswith("stable")]
    assert len(stable) == 22 and all(row["properties"]["stable_across_reopen"] is True for row in stable)
    assert matrix["oscillator numerical-result identity"]["properties"]["absent_after_reopen"] is True
    assert all("fixture_hashes_are_placeholders" not in row["properties"] for row in matrix.values())
    assert matrix["energy result_id"]["observed_refusals"]["moved_step_stale_result_id"] == ep.BINDING
    assert matrix["ESM bundleBytesDigest"]["observed_refusals"] == {
        "esm_noncanonical_bytes": ep.ESM_SCOPE, "esm_forged_digest": ep.ESM_SCOPE}
    assert matrix["energy log producer identities"]["observed_refusals"] == {
        "producer_edit_unsealed": "Retained log digest differs", "producer_edit_resealed": "accepted"}


def test_exact_source_bytes(lab):
    report = lab("T078")
    _labels(report, "numerically_verified", {"numerically_verified": 4, "not_established": 1})
    found = _findings(report)
    retained = report["findings"][0]
    assert retained["value"]["sources"] == 14 and retained["value"]["resealed_variants"] == 2
    assert all(retained["value"][key] == 0 for key in ("live_byte_mismatches", "reopened_byte_mismatches",
                                                       "bundle_byte_mismatches", "evidence_id_mismatches",
                                                       "byte_count_mismatches"))
    transports = found["Non-canonical base64 transports are refused rather than normalized, and refused submissions "
                       "retain no source"]
    assert transports["value"]["messages"] == {
        "base64/line-wrapped": "Source bytes must use canonical base64",
        "base64/missing-padding": "Source bytes must use canonical base64",
        "base64/noncanonical-trailing-bits": "Source bytes must use bounded canonical base64"}
    assert transports["value"]["refused_submissions"] == 6
    assert transports["value"]["sources_added_by_refused_submissions"] == 0
    physical = found["The retained energy logs are real GPU energy measurements"]
    assert physical["domain"] == "physical" and physical["evidence_status"] == "not_established"
    assert physical["value"] == {"declared_origins": ["synthetic_fixture"]}
    assert report["physical_validation_status"]["status"] == "not_established"


def test_whitespace_variants(lab):
    report = lab("T079")
    _labels(report, "numerically_verified", {"numerically_verified": 5})
    distinct = report["findings"][0]["value"]
    assert distinct["labels"] == 1 and distinct["distinct_input_sha256"] == 8
    assert distinct["variants"] == distinct["distinct_evidence_id"] == distinct["distinct_source_id"] == 8
    # The lab's own inputs are harness preconditions, never checks.
    references = [check["reference"] for check in report["findings"][0]["basis"]["checks"]]
    assert not any("label" in text or "input_sha256" in text for text in references)
    assert report["findings"][1]["value"] == {"experiment_digest": 1, "log_digest": 1, "numerical_result_id": 1}
    typed = report["findings"][2]
    assert typed["value"]["python_equal"] is True and typed["value"]["ciw_canonical_equal"] is False
    assert typed["value"]["unsealed_observed"] == "Retained log digest differs"
    assert typed["value"]["resealed_identities_differ_from_baseline"] == {
        "experiment_digest": True, "log_digest": True, "numerical_result_id": True}
    assert typed["counterexample"]["statement"].startswith("Canonical-content identity treats numerically equal")
    assert report["findings"][3]["value"]["observed"] == "Workload solver differs from plan"
    assert report["findings"][4]["value"]["observed"] == ep.BOM_MESSAGE


def test_identity_separation(lab):
    report = lab("T080")
    _labels(report, "numerically_verified", {"numerically_verified": 4, "not_established": 1})
    separation = report["findings"][0]["value"]
    assert separation["cross_role_overlaps"] == 0
    assert separation["moved_step"] == {"stale_result_id": ep.BINDING, "recomputed_result_id": "accepted",
                                        "result_id_changed": True, "numerical_result_id_unchanged": True,
                                        "operation_id_unchanged": True}
    assert _kill_messages(report)["alias.execution-result"] == "Execution/result execution_id binding mismatch"
    assert _survivors(report) == ["alias.swap-pairing", "revision.gap"]
    assert _authentication_is_unestablished(report)
    rows = _retained_rows(lab, "T080")
    assert rows["alias.swap-pairing"]["post_reopen"] == {"result:R1": "execution:E2", "result:R2": "execution:E1",
                                                         "execution:E1": "result:R2", "execution:E2": "result:R1"}


def test_numerical_identity(lab):
    report = lab("T081")
    _labels(report, "numerically_verified", {"numerically_verified": 9, "not_established": 1})
    stable = report["findings"][0]["value"]
    assert stable["occurrences"] == stable["distinct_result_ids"] == 12
    assert stable["distinct_numerical_result_ids"] == 1 and stable["recomputation_mismatches"] == 0
    metadata = report["findings"][1]["value"]
    assert metadata["numerical_result_id_differs"] is True and metadata["data_keys_differing"] == ["log_digest"]
    statistics = report["findings"][2]
    assert statistics["value"]["max_relative_difference"] <= 1e-12
    assert statistics["basis"]["checks"][0]["reference_kind"] == "cross_implementation"
    assert statistics["value"]["inequalities"] == {"abs_mean_le_rms": True, "rms_le_max_abs": True}
    assert report["findings"][3]["value"]["message"] == "Identity collision across retained workbench artifacts"
    assert _kill_messages(report)["energy-data.reforged"] == ep.BINDING
    assert _survivors(report) == ["energy-source.resealed", "oscillator-stats.impossible-moments",
                                  "oscillator-stats.legacy", "oscillator-stats.resealed"]
    rows = _retained_rows(lab, "T081")
    forged = rows["energy-source.resealed"]["post_reopen"]
    assert forged["gross_energy_j_after_reopen"] == {"B0": 0.25, "B0b": 0.25, "B1": 0.25}
    assert forged["gross_energy_j_computed_from_original_bytes"] == 0.2
    assert forged["original_session_ids_and_created_at_kept"] is True
    moments = rows["oscillator-stats.impossible-moments"]["post_reopen"]["retained_inequalities"]
    assert moments == {"result:R1": {"abs_mean_le_rms": False, "rms_le_max_abs": True},
                       "result:R2": {"abs_mean_le_rms": True, "rms_le_max_abs": False}}
    # Witnesses carrying floats tolerate last-bit platform differences; the others stay exact.
    tolerances = {name: record["regression_tolerance"] for name, record in _survivor_findings(report).items()}
    assert tolerances["oscillator-stats.impossible-moments"] == tolerances["energy-source.resealed"] == {
        "abs": 0.0, "rel": 1e-9}
    assert tolerances["oscillator-stats.resealed"] == {"abs": 0.0, "rel": 0.0}


def test_fresh_occurrences(lab):
    report = lab("T082")
    _labels(report, "numerically_verified", {"numerically_verified": 3, "not_established": 1})
    table = report["findings"][0]["value"]
    assert all(entry["distinct"] == entry["occurrences"] for entry in table.values())
    assert table["execution"]["occurrences"] == 17 and table["replay"]["occurrences"] == 3
    assert report["findings"][0]["uncertainty"]["kind"] == "collision_bound"
    assert report["findings"][0]["uncertainty"]["basis"].startswith("27 uuid4 draws")
    assert 0 < report["findings"][0]["uncertainty"]["value"] == 27 * 26 / 2.0 ** 123
    kills = _kill_messages(report)
    assert kills["fresh.energy-replay-reuse"] == ("Declared workload bundles must have distinct execution and "
                                                  "reproduction occurrences")
    assert kills["fresh.cross-namespace"] == "Identity collision between recording operations and retained workflows"
    assert _survivors(report) == ["fresh.created-at-shift"]
    _retained_rows(lab, "T082")


def test_replay_receipt_binding(lab):
    report = lab("T083")
    _labels(report, "numerically_verified", {"numerically_verified": 5, "not_established": 2})
    assert report["findings"][0]["value"] == {"binding_properties": 30, "held": 30, "receipts": 3}
    assert _kill_messages(report) == {"receipt.numerical-match-false": "Invalid retained energy replay receipt",
                                      "receipt.transplanted": "Invalid retained energy replay receipt",
                                      "receipt.transplanted-resealed": ep.BINDING}
    assert _survivors(report) == ["receipt.deleted", "receipt.fabricated", "receipt.transplanted-full"]
    assert "kills show only that these particular edits are detected" in report["uncertainty"].lower()
    independent = _findings(report)["Replay agreement establishes verification by an independent party"]
    assert independent["evidence_status"] == "not_established"
    assert independent["value"]["receipt_independent_flags"] == [False, False, False]
    rows = _retained_rows(lab, "T083")
    none = {"bundle:B0": [], "bundle:B0b": [], "bundle:Bother": [], "bundle:B1": []}
    assert rows["receipt.deleted"]["post_reopen"] == {"receipt_sources": none}
    assert rows["receipt.transplanted-full"]["post_reopen"] == {"receipt_sources": dict(none, **{
        "bundle:B0b": ["bundle:B0"]})}
    assert rows["receipt.fabricated"]["post_reopen"] == {"receipt_sources": dict(none, **{
        "bundle:B0b": ["bundle:B0"], "bundle:B1": ["bundle:B0"]})}


def test_receipt_digest_mutations(lab):
    source, replayed = lab("T084"), lab("T085")
    for report in (source, replayed):
        _labels(report, "numerically_verified", {"numerically_verified": 2, "not_established": 1})
        assert _authentication_is_unestablished(report)
    assert _kill_messages(source) == {
        "receipt-source.naive": "Invalid retained energy replay receipt",
        "receipt-source.replay-id": ep.BINDING,
        "receipt-source.subject-rebound": "Replay source must already belong to this workbench",
        "receipt-source.self": "Invalid retained energy replay receipt",
        "receipt-source.other-source": "Replay source must already belong to this workbench"}
    assert _survivors(source) == ["receipt-source.sibling-execution"]
    sibling = _retained_rows(lab, "T084")["receipt-source.sibling-execution"]["post_reopen"]
    assert sibling["receipt_source"] == sibling["verification_subject"] == "bundle:B0b"
    assert set(_kill_messages(replayed).values()) == {"Invalid retained energy replay receipt"}
    assert _survivors(replayed) == ["receipt-replayed.reidentified-bundle"]
    witness = _retained_rows(lab, "T085")["receipt-replayed.reidentified-bundle"]["post_reopen"]
    assert witness["replay_predates_source"] is True and witness["receipt_source"] == "bundle:B0"


def test_verification_mutations(lab):
    subject, method, independent = lab("T086"), lab("T087"), lab("T088")
    _labels(subject, "numerically_verified", {"numerically_verified": 3, "not_established": 1})
    _labels(method, "numerically_verified", {"numerically_verified": 2, "not_established": 1})
    _labels(independent, "numerically_verified", {"numerically_verified": 3, "not_established": 1})
    for report in (subject, method, independent):
        assert _authentication_is_unestablished(report)
    assert _kill_messages(subject)["oscillator-verification.resealed"] == (
        "Protocol v1 saved results must remain not_verified with verification_id null")
    assert _kill_messages(subject)["exchange.verification-subject"] == "verification_id does not match the artifact content"
    assert _survivors(subject) == ["oscillator-subject.injected"]
    cited = next(record for record in subject["findings"] if record["claim"].startswith("The verification subject moves"))
    assert cited["value"]["mutant"] == "receipt-source.sibling-execution" and "counterexample" not in cited
    assert _kill_messages(method)["receipt-method.resealed"] == ep.BINDING
    assert _survivors(method) == ["oscillator-method.injected"]
    assert _kill_messages(independent)["esm.inspection-independent"] == ep.ESM_SCOPE
    assert _survivors(independent) == ["oscillator-independent.injected"]
    exchange = next(record for record in independent["findings"] if record["claim"].startswith("exchange._identity"))
    assert exchange["evidence_status"] == "numerically_verified" and "counterexample" not in exchange
    assert exchange["value"]["observed"] == "accepted:content_recomputed_not_authenticated"
    rows = _retained_rows(lab, "T088")
    assert "accepted_by_design" in rows["exchange.verification-independent"]
    injected = {"T086": ("oscillator-subject.injected", "subject_ref", "result:R2"),
                "T087": ("oscillator-method.injected", "verification_method", "independent_reimplementation"),
                "T088": ("oscillator-independent.injected", "independent", True)}
    for task_id, (name, field, value) in injected.items():
        shown = _retained_rows(lab, task_id)[name]["post_reopen"]["result.get"]
        assert shown == {field: value, "verification_status": "not_verified"}


def test_admission_and_runtime_mutations(lab):
    admission, runtime = lab("T089"), lab("T090")
    _labels(admission, "numerically_verified", {"numerically_verified": 2, "not_established": 2})
    _labels(runtime, "numerically_verified", {"numerically_verified": 5, "not_established": 1})
    assert _kill_messages(admission)["esm.canonical-admission"] == "ESM may retain candidate evidence only"
    authority = _findings(admission)["A retained replay receipt or verification authorizes admission of the replayed "
                                     "result into canonical state"]
    assert authority["domain"] == "production_acceptance" and authority["evidence_status"] == "not_established"
    assert authority["value"] == {"receipt_admissions": ["not_performed"] * 3}
    assert _survivors(admission) == ["oscillator-admission.injected"]
    assert _retained_rows(lab, "T089")["oscillator-admission.injected"]["post_reopen"]["result.get"] == {
        "state_admission": "admitted", "verification_status": "not_verified"}
    assert any(text.startswith("Pinned-provider subprocess runtime identities") for text in runtime[
        "unresolved_assumptions"])
    kills = _kill_messages(runtime)
    assert kills["oscillator-runtime.execution-only"] == "Execution/result runtime binding mismatch"
    assert kills["energy-runtime.replay-only"] == ep.BINDING
    assert _survivors(runtime) == ["energy-runtime.all-bundles", "energy-runtime.python-version",
                                   "oscillator-runtime.both"]
    detected = _findings(runtime)["A consistently forged energy runtime identity is detected only when a replay "
                                  "recomputes the current analysis identity"]
    assert detected["value"]["replay_after_reopen"] == {"message": ep.BINDING, "outcome": "refused"}
    matrix = json.loads(_artifact(lab, "T090", "mutation-matrix.json"))
    assert matrix["harness"] == {"reforge_reproduces_ciw_digests": True, "unchanged_workspace_reopens": True}
    rows = matrix["rows"]
    assert len(rows) == len(ep.MUTANTS) + 7 == 72
    assert all(row["outcome_matches_prediction"] and row["message_matches_pin"] for row in rows)
    workspace_survivors = [row["name"] for row in rows if not row["killed"] and row["kind"] == "workspace"]
    assert len(workspace_survivors) == len(set(workspace_survivors)) == 19
    assert set(workspace_survivors) == set(ep.STATEMENTS)
    assert [row["name"] for row in rows if not row["killed"] and row["kind"] == "validator"] == [
        "exchange.verification-independent"]
    _retained_rows(lab, "T090")


def test_next_steps_name_forward_work(lab):
    """Each next step names the task's own open question, never the next queue task (which has already run)."""
    for task_id in TASKS:
        report = lab(task_id)
        text = report["recommended_next_task"]
        assert text == ep.NEXT_STEPS[task_id], task_id
        assert IMPLEMENTATIONS[task_id].run.plan["recommended_next_task"] == text, task_id
        if report["state"] == "completed":
            assert text.startswith("Deferred research question"), (task_id, text)
    assert lab("T077")["state"] == "partial" and ep.NEXT_STEPS["T077"].startswith("Complete T077: ")
    assert "T078" not in ep.NEXT_STEPS["T077"] and "T091" not in ep.NEXT_STEPS["T090"]


def _literal(path, name):
    """A module-level literal assignment read without importing the script."""
    import ast
    from pathlib import Path
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name
                                                                           for t in n.targets))
    return ast.literal_eval(node.value)


def test_shared_deferred_questions_are_recorded_where_they_apply(lab):
    """Key custody and the telemetry stack are recorded as deferred questions in every report that depends on them."""
    key_tasks = {"T077", "T078"} | {f"T0{n}" for n in range(80, 91)}
    telemetry_tasks = {"T077", "T086", "T088", "T089", "T090"}
    for task_id in TASKS:
        report = lab(task_id)
        assumptions = report["unresolved_assumptions"]
        assert assumptions.count(common.KEY_CUSTODY_QUESTION) == (task_id in key_tasks), task_id
        assert assumptions.count(common.TELEMETRY_STACK_QUESTION) == (task_id in telemetry_tasks), task_id
        if any(r["claim"].startswith("Retained workspace records are authenticated") for r in report["findings"]):
            assert task_id in key_tasks
    for question in (common.KEY_CUSTODY_QUESTION, common.TELEMETRY_STACK_QUESTION):
        assert question.startswith("Deferred research question (")
    # The telemetry question's statement about provisioning must match the scripts and pins it names.
    from ciw.lab.runner import PACKAGE_ROOT, repository_path
    root = repository_path()
    if root is None or not (root / "scripts" / "check_lab.py").is_file():
        pytest.skip("repository scripts are not available (the clean room copies no scripts/)")
    # The pins are package data, so the installed package is read, not the repository's source tree.
    pins = json.loads((PACKAGE_ROOT / "telemetry-runtimes.json").read_text(encoding="utf-8"))
    provisioned = set(_literal(root / "scripts" / "check_lab.py", "REPOSITORIES"))
    variables = set(_literal(root / "scripts" / "reproduce_lab.py", "TEST_VARIABLES"))
    assert sorted(pins) == ["cbsr", "gsie", "ppda", "set", "stfe"]
    assert set(pins) & provisioned == {"ppda", "set"} and not {"stfe", "gsie", "cbsr"} & variables


def test_t081_defers_cross_platform_reproduction_as_one_question(lab):
    report = lab("T081")
    assumptions = report["unresolved_assumptions"]
    platform = [a for a in assumptions if "platform" in a.lower()]
    assert platform == [ep.PLATFORM_QUESTION_T081]
    for fragment in ("Windows x86-64", "macOS arm64", "Linux x86-64", "numerical_result_id", "five OpenBLAS kernels",
                     "ciw lab verify", "leaf by leaf within 2e-14", "does not record its kernel"):
        assert fragment in platform[0], fragment
    assert "OpenBLAS kernels SkylakeX, Haswell, Sandybridge, Nehalem and Katmai" in report["uncertainty"]
    # The id hashes floats whose last bits depend on the BLAS kernel, so the regression gate compares the counts the
    # claim is about and the numerical result the id hashes, within the kernel tolerance; the id is retained verbatim
    # (not masked by relabel) in the named artifact, beside that result.
    identity = report["findings"][0]
    assert identity["claim"].startswith("The energy numerical_result_id is identical")
    value = identity["value"]
    assert {key: value[key] for key in value if key != "numerical_result"} == {
        "occurrences": 12, "distinct_numerical_result_ids": 1, "distinct_result_ids": 12, "recomputation_mismatches": 0}
    assert identity["regression_tolerance"] == ep.KERNEL_ROUNDING == {"abs": 2e-14, "rel": 0.0}
    assert identity["uncertainty"]["kind"] == "roundoff" and "about 90 times" in identity["uncertainty"]["basis"]
    artifact = json.loads(_artifact(lab, "T081", "numerical-identity.json"))
    retained = artifact["energy_numerical_result_id"]
    assert retained.startswith("sha256:") and len(retained) == len("sha256:") + 64
    int(retained.removeprefix("sha256:"), 16)
    assert not any(retained in json.dumps(record["value"]) for record in report["findings"])
    assert set(artifact["energy_numerical_result"]) == {"operation_id", "data"}
    assert ep._sha(artifact["energy_numerical_result"]) == retained == ep._sha(value["numerical_result"])
    # The gate passes last-bit kernel differences (at most 2.2e-16 measured) and refuses a changed analysed number.
    for delta, close in ((1e-15, True), (1e-3, False)):
        changed = json.loads(json.dumps(value))
        changed["numerical_result"]["data"]["reference"]["log_evidence"] += delta
        assert runner._close(value, changed, identity["regression_tolerance"]) is close, delta
    # It is the baseline log's identity that T078 retains for its source records.
    lab("T078")
    records = json.loads(_artifact(lab, "T078", "source-retention.json"))["records"]
    assert next(r for r in records if r["name"] == "baseline")["numerical_result_id"] == retained


def test_tasks_block_without_the_fixture_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("CIW_LAB_REPOSITORY_ROOT", str(tmp_path / "absent"))
    queue = {item["id"]: item for item in load_queue()["tasks"]}
    ctx = runner.Context(tmp_path / "out")
    for task_id in ("T077", "T078", "T084", "T089"):
        report = validate_report(runner.run_task(queue[task_id], IMPLEMENTATIONS[task_id], ctx, {}))
        assert report["state"] == "blocked"
        assert report["experiment"].startswith("Blocked: the bundled energy-accuracy fixture logs")
        assert all(record["evidence_status"] == "not_established" for record in report["findings"])
        assert {record["domain"] for record in report["findings"]} <= {"physical", "production_acceptance"}
        # A blocked report never presents unobserved values as observations.
        assert all(record["value"] == {"observed": False} for record in report["findings"])
    assert report["findings"][0]["domain"] == "production_acceptance"
