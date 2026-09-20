"""Reproduce the declared PLSR reference corpus and pin its contract checks."""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


L = pytest.importorskip("lyapunov", reason="install the optional plsr extra")

from ciw import plsr_corpus
from ciw import plsr_engine as engine


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "plsr"
CORPUS = EXAMPLES / "corpus.json"

#: An independent statement of what each case is for, written by hand rather
#: than recorded. The corpus stores the same codes; a case whose recorded
#: expectation drifts away from its intent fails here as well as there.
INTENT = {
    "continuous-certified": "CERTIFIED_WITH_MARGIN",
    "continuous-box-upper-endpoint": "CERTIFIED_WITH_MARGIN",
    "continuous-box-lower-endpoint": "CERTIFIED_WITH_MARGIN",
    "continuous-theta-one-ulp-above-box": "OUTSIDE_PARAMETER_BOX",
    "continuous-rate-one-ulp-above-box": "OUTSIDE_PARAMETER_BOX",
    "continuous-outside-level-set": "OUTSIDE_LEVEL_SET",
    "continuous-origin": "CERTIFIED_WITH_MARGIN",
    "continuous-denormal-state": "CERTIFIED_WITH_MARGIN",
    "continuous-huge-state-outside-level": "OUTSIDE_LEVEL_SET",
    "continuous-huge-state-no-level": "CERTIFIED_WITH_MARGIN",
    "discrete-certified": "CERTIFIED_WITH_MARGIN",
    "continuous-margin-shortfall": "MARGIN_LOW",
    "continuous-sample-violation": "NOT_CERTIFIED",
    "continuous-decrease-not-definite": "DECREASE_NOT_DEFINITE",
    "continuous-numerically-inconclusive": "NUMERICAL_INCONCLUSIVE",
    "continuous-numerical-overflow": "NUMERICAL_OVERFLOW",
    "refusal-affine-certificate-not-positive": "refused",
    "refusal-theta-on-linear-plant": "refused",
    "refusal-rate-on-discrete-plant": "refused",
    "refusal-missing-theta": "refused",
    "refusal-state-dimension": "refused",
    "refusal-boolean-state-entry": "refused",
    "refusal-unknown-sample-schema": "refused",
}

#: Every runtime status this adapter can reach through a declared model.
#: CERTIFICATE_NOT_POSITIVE is absent on purpose: a constant P is checked for
#: positive definiteness when the model loads, and an affine P(theta) that
#: loses it raises from the certificate rather than returning that code, so it
#: reaches the terminal as an input refusal. The corpus covers that refusal.
REACHABLE_CODES = {
    "CERTIFIED_WITH_MARGIN", "MARGIN_LOW", "NOT_CERTIFIED", "DECREASE_NOT_DEFINITE",
    "NUMERICAL_INCONCLUSIVE", "NUMERICAL_OVERFLOW", "OUTSIDE_PARAMETER_BOX",
    "OUTSIDE_LEVEL_SET",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cli(*args, exit_code=0):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "ciw", *map(str, args)], cwd=ROOT,
        env=environment, capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == exit_code, completed.stderr
    if exit_code == 2:
        assert not completed.stdout and completed.stderr
        return completed
    assert not completed.stderr
    return json.loads(completed.stdout)


def seal(document):
    body = {key: value for key, value in document.items() if key != "corpus_digest"}
    body["corpus_digest"] = plsr_corpus.corpus_digest(body)
    return body


@pytest.fixture
def corpus_copy(tmp_path):
    """A writable copy of the corpus and every model it names."""
    directory = tmp_path / "corpus"
    directory.mkdir()
    for path in EXAMPLES.glob("*.json"):
        shutil.copy(path, directory / path.name)
    return directory / "corpus.json"


def edit(path, change):
    document = load(path)
    change(document)
    Path(path).write_text(json.dumps(seal(document), indent=2) + "\n", encoding="utf-8")
    return path


def case_of(document, case_id):
    return next(case for case in document["cases"] if case["case_id"] == case_id)


def report_case(report, case_id):
    return next(case for case in report["cases"] if case["case_id"] == case_id)


def test_every_declared_case_reproduces_its_expectation():
    report = plsr_corpus.check_corpus(CORPUS)
    assert report["runtime_status"] == "supported", report["runtime_differences"]
    assert report["status"] == "reproduced"
    assert report["outcome_counts"] == {
        "reproduced": len(INTENT), "mismatched": 0, "errored": 0, "skipped": 0}
    assert [case["differences"] for case in report["cases"]] == [[]] * len(INTENT)


def test_recorded_expectations_still_state_the_intent_each_case_was_written_for():
    declared = {case["case_id"]: case["expect"] for case in load(CORPUS)["cases"]}
    assert set(declared) == set(INTENT)
    for case_id, intent in INTENT.items():
        expect = declared[case_id]
        if intent == "refused":
            assert expect["outcome"] == "refused" and expect["reason_contains"]
        else:
            assert expect["outcome"] == "evaluated" and expect["code"] == intent


def test_the_corpus_covers_every_reachable_code_and_every_declared_purpose():
    document = load(CORPUS)
    codes = {case["expect"].get("code") for case in document["cases"]} - {None}
    assert codes == REACHABLE_CODES
    purposes = {case["purpose"] for case in document["cases"]}
    assert purposes == plsr_corpus._PURPOSES
    assert "CERTIFICATE_NOT_POSITIVE" not in codes
    refusal = case_of(document, "refusal-affine-certificate-not-positive")
    assert refusal["expect"]["reason_contains"] == "must be positive definite"


def test_each_case_binds_the_model_it_names_by_digest():
    document = load(CORPUS)
    for case in document["cases"]:
        model = engine.load_model(EXAMPLES / case["model_file"])
        assert model.artifact_digest == case["model_artifact_digest"]


def test_a_changed_binding_runtime_identity_fails_as_stale_without_running_a_case(corpus_copy):
    edit(corpus_copy, lambda d: d["runtime"].update(commit="0" * 40))
    report = plsr_corpus.check_corpus(corpus_copy)
    assert report["runtime_status"] == "stale" and report["status"] == "stale"
    assert report["cases"] == []
    assert report["outcome_counts"]["skipped"] == len(INTENT)
    assert [item["field"] for item in report["runtime_differences"] if item["binding"]] == ["commit"]
    cli("plsr", "corpus", "check", corpus_copy, exit_code=5)


def test_a_non_binding_environment_difference_is_reported_but_still_reproduces(corpus_copy):
    edit(corpus_copy, lambda d: d["runtime"].update(python_version="3.12.0"))
    report = plsr_corpus.check_corpus(corpus_copy)
    assert report["runtime_status"] == "supported" and report["status"] == "reproduced"
    assert report["runtime_differences"] == [{
        "field": "python_version", "declared": "3.12.0",
        "observed": engine.runtime_identity()["python_version"], "binding": False}]


def test_binding_every_runtime_field_makes_an_environment_difference_stale(corpus_copy):
    def change(document):
        document["runtime_binding"] = sorted(engine.runtime_identity())
        document["runtime"]["numpy_version"] = "0.0.0"
    edit(corpus_copy, change)
    assert plsr_corpus.check_corpus(corpus_copy)["runtime_status"] == "stale"


def test_an_edited_expectation_is_refused_before_any_case_runs(corpus_copy):
    document = load(corpus_copy)
    case_of(document, "continuous-margin-shortfall")["expect"]["meets_required_margin"] = True
    Path(corpus_copy).write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(plsr_corpus.CorpusError, match="corpus_digest does not match"):
        plsr_corpus.check_corpus(corpus_copy)
    cli("plsr", "corpus", "check", corpus_copy, exit_code=2)


def test_a_resealed_expectation_still_fails_against_the_engine(corpus_copy):
    edit(corpus_copy, lambda d: case_of(d, "continuous-margin-shortfall")["expect"]
         .update(meets_required_margin=True))
    report = plsr_corpus.check_corpus(corpus_copy)
    assert report["status"] == "mismatched"
    assert report["outcome_counts"]["mismatched"] == 1
    failed = next(case for case in report["cases"] if case["differences"])
    assert failed["case_id"] == "continuous-margin-shortfall"
    assert failed["differences"] == [{
        "field": "meets_required_margin", "reason": "value differs",
        "declared": True, "observed": False, "tolerance": None}]
    cli("plsr", "corpus", "check", corpus_copy, exit_code=4)


@pytest.mark.parametrize("field", ["code", "presentation_category", "inequality_certified",
                                   "operationally_acceptable"])
def test_codes_and_booleans_are_compared_exactly(corpus_copy, field):
    def change(document):
        expect = case_of(document, "continuous-certified")["expect"]
        expect[field] = False if isinstance(expect[field], bool) else expect[field] + "x"
    edit(corpus_copy, change)
    report = plsr_corpus.check_corpus(corpus_copy)
    assert report["status"] == "mismatched"
    failed = report_case(report, "continuous-certified")
    assert [item["field"] for item in failed["differences"]] == [field]


def test_a_tolerance_named_for_a_boolean_field_is_itself_a_difference(corpus_copy):
    edit(corpus_copy, lambda d: d["tolerance_policy"]["fields"].update(
        {"inequality_certified": {"relative": 1.0, "absolute": 1.0}}))
    report = plsr_corpus.check_corpus(corpus_copy)
    assert report["status"] == "mismatched"
    reasons = {item["reason"] for case in report["cases"] for item in case["differences"]}
    assert reasons == {"tolerance policy names a boolean field; only floats take a tolerance"}


def test_a_named_numeric_field_absorbs_a_difference_inside_its_declared_tolerance(corpus_copy):
    def change(document):
        expect = case_of(document, "continuous-margin-shortfall")["expect"]
        expect["diagnostics"]["margin"] *= 1 + 5e-13
    edit(corpus_copy, change)
    assert plsr_corpus.check_corpus(corpus_copy)["status"] == "reproduced"


def test_the_same_difference_on_an_unnamed_numeric_field_is_a_mismatch(corpus_copy):
    def change(document):
        expect = case_of(document, "continuous-margin-shortfall")["expect"]
        expect["diagnostics"]["value"] *= 1 + 5e-13
    edit(corpus_copy, change)
    report = plsr_corpus.check_corpus(corpus_copy)
    assert report["status"] == "mismatched"
    failed = report_case(report, "continuous-margin-shortfall")
    assert [item["field"] for item in failed["differences"]] == ["diagnostics.value"]
    assert failed["differences"][0]["reason"] == (
        "float differs and no tolerance is declared for this field")


def test_a_named_numeric_field_still_fails_outside_its_declared_tolerance(corpus_copy):
    def change(document):
        expect = case_of(document, "continuous-margin-shortfall")["expect"]
        expect["diagnostics"]["margin"] *= 1 + 1e-9
    edit(corpus_copy, change)
    report = plsr_corpus.check_corpus(corpus_copy)
    assert report["status"] == "mismatched"
    difference = report_case(report, "continuous-margin-shortfall")["differences"][0]
    assert difference["field"] == "diagnostics.margin"
    assert difference["tolerance"] == {"relative": 1e-12, "absolute": 0.0}


def test_an_unnamed_float_distinguishes_negative_zero_from_zero(corpus_copy):
    document = load(corpus_copy)
    expect = case_of(document, "continuous-numerically-inconclusive")["expect"]
    assert expect["diagnostics"]["decrease"] == 0.0
    assert math.copysign(1.0, expect["diagnostics"]["decrease"]) == 1.0
    edit(corpus_copy, lambda d: case_of(d, "continuous-numerically-inconclusive")["expect"]
         ["diagnostics"].update(decrease=-0.0))
    report = plsr_corpus.check_corpus(corpus_copy)
    assert report["status"] == "mismatched"
    failed = report_case(report, "continuous-numerically-inconclusive")
    assert [item["field"] for item in failed["differences"]] == ["diagnostics.decrease"]


def test_the_declared_matrices_are_compared_entry_by_entry(corpus_copy):
    def change(document):
        expect = case_of(document, "continuous-certified")["expect"]
        expect["diagnostics"]["A"][0][1] = 1.0
    edit(corpus_copy, change)
    report = plsr_corpus.check_corpus(corpus_copy)
    failed = report_case(report, "continuous-certified")
    assert [item["field"] for item in failed["differences"]] == ["diagnostics.A"]


def test_every_evaluated_case_reproduces_its_record_digest_on_this_installation():
    report = plsr_corpus.check_corpus(CORPUS)
    evaluated = [case for case in report["cases"] if case["observed_outcome"] == "evaluated"]
    assert evaluated and all(case["record_digest_matches"] for case in evaluated)


def test_enforcing_the_digest_makes_a_stale_digest_a_mismatch(corpus_copy):
    def change(document):
        document["tolerance_policy"]["digest_enforcement"] = "enforced"
        case_of(document, "continuous-certified")["expect"]["record_digest"] = "0" * 64
    edit(corpus_copy, change)
    report = plsr_corpus.check_corpus(corpus_copy)
    assert report["digest_enforcement"] == "enforced" and report["status"] == "mismatched"
    failed = report_case(report, "continuous-certified")
    assert [item["field"] for item in failed["differences"]] == ["record_digest"]
    assert failed["record_digest_matches"] is False


def test_a_reported_digest_difference_alone_does_not_fail_the_case(corpus_copy):
    edit(corpus_copy, lambda d: case_of(d, "continuous-certified")["expect"]
         .update(record_digest="0" * 64))
    report = plsr_corpus.check_corpus(corpus_copy)
    assert report["status"] == "reproduced"
    assert report_case(report, "continuous-certified")["record_digest_matches"] is False


def test_a_changed_model_file_fails_as_a_binding_error_not_a_numeric_difference(corpus_copy):
    model = corpus_copy.parent / "continuous-shortfall.json"
    declared = load(model)
    declared["policy"]["required_margin"] = 0.2
    model.write_text(json.dumps(declared, indent=2), encoding="utf-8")
    report = plsr_corpus.check_corpus(corpus_copy)
    assert report["outcome_counts"]["errored"] == 1
    failed = next(case for case in report["cases"] if case["outcome"] == "errored")
    assert failed["case_id"] == "continuous-margin-shortfall"
    assert failed["observed_outcome"] == "unreadable_model"
    assert "artifact_digest does not match" in failed["refusal_reason"]


def test_a_resealed_model_file_fails_against_the_digest_the_case_binds(corpus_copy):
    model = corpus_copy.parent / "continuous-shortfall.json"
    declared = load(model)
    declared["policy"]["required_margin"] = 0.2
    model.write_text(json.dumps(L.seal_model_artifact(declared), indent=2), encoding="utf-8")
    report = plsr_corpus.check_corpus(corpus_copy)
    failed = next(case for case in report["cases"] if case["outcome"] == "errored")
    assert failed["observed_outcome"] == "model_digest_mismatch"
    assert failed["differences"][0]["field"] == "model_artifact_digest"


@pytest.mark.parametrize("model_file", ["/etc/passwd", "../outside.json", "a\\b.json"])
def test_a_model_file_outside_the_corpus_directory_is_refused(corpus_copy, model_file):
    edit(corpus_copy, lambda d: case_of(d, "continuous-certified").update(model_file=model_file))
    with pytest.raises(plsr_corpus.CorpusError, match="relative path beside the corpus"):
        plsr_corpus.check_corpus(corpus_copy)


@pytest.mark.parametrize("change,message", [
    (lambda d: d["cases"].append(dict(d["cases"][0])), "duplicate case_id"),
    (lambda d: d["cases"][0].update(purpose="whatever"), "purpose must be one of"),
    (lambda d: d.update(corpus_schema="ciw-plsr-corpus-v2"), "corpus_schema must be"),
    (lambda d: d.update(runtime_binding=["nonsense"]), "runtime_binding names a field"),
    (lambda d: d["tolerance_policy"].update(digest_enforcement="off"), "digest_enforcement"),
    (lambda d: d["tolerance_policy"]["fields"].update(
        {"diagnostics.margin": {"relative": -1.0, "absolute": 0.0}}), "non-negative float"),
    (lambda d: d["cases"][0]["sample"].pop("theta_dot"), "a case sample needs exactly"),
    (lambda d: d["cases"][0]["expect"].update(outcome="maybe"), "outcome must be"),
    (lambda d: d.update(cases=[]), "at least one case"),
])
def test_a_malformed_corpus_is_refused_with_a_named_reason(corpus_copy, change, message):
    edit(corpus_copy, change)
    with pytest.raises(plsr_corpus.CorpusError, match=message):
        plsr_corpus.check_corpus(corpus_copy)


def test_checking_never_writes_to_the_corpus_or_its_models(corpus_copy):
    before = {path.name: path.read_bytes() for path in corpus_copy.parent.iterdir()}
    plsr_corpus.check_corpus(corpus_copy)
    cli("plsr", "corpus", "check", corpus_copy)
    after = {path.name: path.read_bytes() for path in corpus_copy.parent.iterdir()}
    assert after == before


def test_checking_can_retain_one_run_bundle_per_evaluated_case(tmp_path):
    directory = tmp_path / "runs"
    report = plsr_corpus.check_corpus(CORPUS, output_dir=directory,
                                      case_ids=["continuous-certified", "discrete-certified"])
    assert report["status"] == "reproduced"
    assert report["outcome_counts"] == {
        "reproduced": 2, "mismatched": 0, "errored": 0, "skipped": len(INTENT) - 2}
    saved = sorted(path.name for path in directory.iterdir())
    assert len(saved) == 2 and all(name.startswith("run-") for name in saved)
    for case in report["cases"]:
        bundle = load(case["saved_file"])
        assert bundle["bundle_schema"] == "ciw-plsr-run-v1"
        assert bundle["record"]["record_digest"] == case["record_digest"]


def test_selecting_an_unknown_case_is_an_error(corpus_copy):
    with pytest.raises(plsr_corpus.CorpusError, match="no such case: nope"):
        plsr_corpus.check_corpus(corpus_copy, case_ids=["nope"])


def test_recording_reports_every_change_and_refuses_to_replace_the_corpus(corpus_copy):
    edit(corpus_copy, lambda d: case_of(d, "continuous-certified")["sample"].update(x=[0.3, 0.4]))
    output = corpus_copy.parent / "recorded.json"
    report = plsr_corpus.record_corpus(corpus_copy, output)
    assert report["written"] is False and not output.exists()
    changed = [item for item in report["changes"] if item["case_id"] == "continuous-certified"]
    assert changed and changed[0]["change"] == "changed"
    fields = {item["field"] for item in changed[0]["differences"]}
    assert "expect.diagnostics.scaled_value" in fields
    accepted = plsr_corpus.record_corpus(corpus_copy, output, accept_changes=True)
    assert accepted["written"] is True and output.exists()
    assert plsr_corpus.check_corpus(output)["status"] == "reproduced"


def test_recording_an_unchanged_corpus_writes_identical_content(corpus_copy):
    output = corpus_copy.parent / "recorded.json"
    report = plsr_corpus.record_corpus(corpus_copy, output)
    assert report["written"] is True and report["change_count"] == 0
    assert load(output) == load(CORPUS)
    assert report["corpus_digest"] == load(CORPUS)["corpus_digest"]


def test_recording_refuses_to_overwrite_a_different_corpus_in_place(corpus_copy):
    edit(corpus_copy, lambda d: case_of(d, "continuous-certified")["sample"].update(x=[0.3, 0.4]))
    original = corpus_copy.read_bytes()
    report = plsr_corpus.record_corpus(corpus_copy, corpus_copy)
    assert report["written"] is False
    assert corpus_copy.read_bytes() == original
    assert plsr_corpus.record_corpus(corpus_copy, corpus_copy, accept_changes=True)["written"]
    assert corpus_copy.read_bytes() != original
    assert plsr_corpus.check_corpus(corpus_copy)["status"] == "reproduced"


def test_recording_a_removed_case_is_reported_against_the_saved_corpus(corpus_copy, tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(corpus_copy.read_text(encoding="utf-8"), encoding="utf-8")
    edit(plan, lambda d: d["cases"].pop(0))
    for path in EXAMPLES.glob("*.json"):
        shutil.copy(path, tmp_path / path.name)
    report = plsr_corpus.record_corpus(plan, corpus_copy)
    assert report["written"] is False
    assert {item["change"] for item in report["changes"]} == {"removed"}
    assert [item["case_id"] for item in report["changes"]] == ["continuous-certified"]


def test_an_unused_tolerance_entry_is_reported_without_failing(corpus_copy):
    edit(corpus_copy, lambda d: d["tolerance_policy"]["fields"].update(
        {"diagnostics.not_a_field": {"relative": 1e-9, "absolute": 0.0}}))
    report = plsr_corpus.check_corpus(corpus_copy)
    assert report["status"] == "reproduced"
    assert report["unused_tolerances"] == ["diagnostics.not_a_field"]


def test_the_published_corpus_declares_no_unused_tolerance():
    assert plsr_corpus.unused_tolerances(load(CORPUS)) == []


def test_the_terminal_reports_the_whole_corpus_and_exits_zero():
    report = cli("plsr", "corpus", "check", CORPUS)
    assert report["report_schema"] == "ciw-plsr-corpus-report-v1"
    assert report["status"] == "reproduced"
    assert report["corpus_digest"] == load(CORPUS)["corpus_digest"]
    assert report["observed_runtime"] == engine.runtime_identity()
