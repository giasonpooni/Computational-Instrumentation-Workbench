"""Compare saved batches from retained evidence, never by evaluating again."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


L = pytest.importorskip("lyapunov", reason="install the optional plsr extra")

from ciw import plsr
from ciw import plsr_batch
from ciw import plsr_compare
from ciw import plsr_engine as engine


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "plsr"
CORPUS = EXAMPLES / "corpus.json"


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


def sample(x, theta=None, theta_dot=None):
    return {"sample_schema": "plsr-sample-v1", "x": x, "theta": theta, "theta_dot": theta_dot}


SAMPLES = [
    {"sample_id": "certified", "sample": sample([0.1, 0.2], [0.5], [0.1])},
    {"sample_id": "outside-box", "sample": sample([0.1, 0.2], [1.5], [0.1])},
    {"sample_id": "shortfall", "model_file": "continuous-shortfall.json",
     "sample": sample([0.1, 0.2])},
    {"sample_id": "violating", "model_file": "continuous-indefinite.json",
     "sample": sample([1.0, 0.0])},
]


def write_plan(directory, samples=SAMPLES, *, batch_id="compare-batch"):
    plan = {"batch_schema": "ciw-plsr-batch-v1", "batch_id": batch_id,
            "description": "A declared collection written by the test suite.",
            "model_file": "continuous-affine.json", "samples": samples}
    path = Path(directory) / "plan.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def area(tmp_path):
    """Two independent model trees, so one side's models can be changed."""
    for side in ("baseline", "candidate"):
        (tmp_path / side).mkdir()
        for path in EXAMPLES.glob("*.json"):
            shutil.copy(path, tmp_path / side / path.name)
    return tmp_path


def run(area, side, samples=SAMPLES, *, batch_id="compare-batch"):
    plan = write_plan(area / side, samples, batch_id=batch_id)
    return plsr_batch.run_batch(plan, area / side / "out")


def bundle_path(area, side, index, sample_id):
    entry = next(e for e in index["entries"] if e["sample_id"] == sample_id)
    return area / side / "out" / plsr_batch.RUNS_NAME / entry["saved_file"]


def reseal(path, change):
    bundle = load(path)
    change(bundle)
    bundle["record"]["record_digest"] = plsr_compare.record_digest(bundle["record"])
    bundle["bundle_digest"] = plsr_compare.bundle_digest(bundle)
    Path(path).write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle


def reseal_index(directory, sample_id, record_digest):
    """Stand in for a batch produced elsewhere whose last bits came out differently."""
    path = Path(directory) / plsr_batch.INDEX_NAME
    index = load(path)
    for entry in index["entries"]:
        if entry["sample_id"] == sample_id:
            entry["record_digest"] = record_digest
    index.pop("index_digest")
    body = json.dumps(index, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    from hashlib import sha256
    index["index_digest"] = sha256(body.encode("ascii")).hexdigest()
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def test_two_runs_of_the_same_collection_compare_as_identical(area):
    run(area, "baseline")
    run(area, "candidate")
    report = plsr_compare.compare_batches(area / "baseline" / "out", area / "candidate" / "out")
    assert report["status"] == "identical"
    assert report["compatibility"]["incompatible"] is False
    assert report["compatibility"]["reasons"] == []
    assert report["summary"]["compared"] == len(SAMPLES)
    assert report["summary"]["changed"] == 0
    assert report["outcome_transitions"] == {} and report["code_transitions"] == {}
    assert all(s["record_digest_matches"] for s in report["samples"])


def test_a_comparison_never_evaluates_anything(area, monkeypatch):
    run(area, "baseline")
    run(area, "candidate")

    def refuse(*args, **kwargs):
        raise AssertionError("a comparison must not evaluate")

    monkeypatch.setattr(engine, "evaluate", refuse)
    monkeypatch.setattr(L.ModelArtifact, "verdict", refuse)
    report = plsr_compare.compare_batches(area / "baseline" / "out", area / "candidate" / "out")
    assert report["status"] == "identical"


def test_every_compared_sample_names_the_bundle_it_resolved(area):
    baseline = run(area, "baseline")
    run(area, "candidate")
    report = plsr_compare.compare_batches(area / "baseline" / "out", area / "candidate" / "out")
    for entry in report["samples"]:
        for side in ("baseline", "candidate"):
            saved = Path(entry[side]["saved_file"])
            assert saved.is_file() and saved.parent.name == plsr_batch.RUNS_NAME
            assert entry[side]["resolution"] == "validated"
            assert load(saved)["record"]["record_digest"] == entry[side]["record_digest"]
    assert str(bundle_path(area, "baseline", baseline, "certified")) == \
        next(e for e in report["samples"] if e["sample_id"] == "certified")["baseline"]["saved_file"]


def test_a_lowered_required_margin_shows_as_a_status_transition_and_a_margin_change(area):
    run(area, "baseline")
    model = area / "candidate" / "continuous-shortfall.json"
    declared = load(model)
    declared["policy"]["required_margin"] = 0.01
    model.write_text(json.dumps(L.seal_model_artifact(declared), indent=2), encoding="utf-8")
    run(area, "candidate")

    report = plsr_compare.compare_batches(area / "baseline" / "out", area / "candidate" / "out")
    assert report["status"] == "changed"
    assert report["compatibility"]["incompatible"] is False
    assert report["code_transitions"] == {"MARGIN_LOW->CERTIFIED_WITH_MARGIN": 1}
    assert report["summary"]["required_margin_changed"] == 1
    assert report["summary"]["model_changed"] == 1
    assert report["summary"]["certificate_changed"] == 0
    assert report["summary"]["diagnostics_changed"] == 0
    changed = next(s for s in report["samples"] if s["sample_id"] == "shortfall")
    assert changed["model_artifact_digest_changed"] is True
    assert changed["record_digest_matches"] is False
    margin = next(d for d in changed["model_differences"]
                  if d["field"] == "model.policy.required_margin")
    assert (margin["baseline"], margin["candidate"]) == (0.1, 0.01)
    verdict = {d["field"] for d in changed["verdict_differences"]}
    assert {"code", "required_margin", "operationally_acceptable",
            "meets_required_margin"} <= verdict


def test_a_changed_certificate_shows_as_a_certificate_and_diagnostic_difference(area):
    run(area, "baseline")
    model = area / "candidate" / "continuous-indefinite.json"
    declared = load(model)
    declared["certificate"]["P"] = [[2.0, 0.0], [0.0, 2.0]]
    model.write_text(json.dumps(L.seal_model_artifact(declared), indent=2), encoding="utf-8")
    run(area, "candidate")

    report = plsr_compare.compare_batches(area / "baseline" / "out", area / "candidate" / "out")
    assert report["status"] == "changed"
    assert report["summary"]["certificate_changed"] == 1
    assert report["summary"]["plant_changed"] == 0
    assert report["summary"]["diagnostics_changed"] == 1
    assert report["code_transitions"] == {}
    changed = next(s for s in report["samples"] if s["sample_id"] == "violating")
    assert changed["code_transition"] is None and changed["outcome_transition"] is None
    fields = {d["field"] for d in changed["diagnostic_differences"]}
    assert {"diagnostics.P", "diagnostics.decrease", "diagnostics.margin"} <= fields


def test_a_different_declared_collection_is_explicit_and_still_compares_the_overlap(area):
    run(area, "baseline")
    run(area, "candidate", SAMPLES[:2] + [
        {"sample_id": "extra", "sample": sample([0.2, 0.2], [0.0], [0.0])}],
        batch_id="other-batch")
    report = plsr_compare.compare_batches(area / "baseline" / "out", area / "candidate" / "out")
    assert report["status"] == "incompatible"
    compatibility = report["compatibility"]
    assert compatibility["same_collection"] is False
    assert compatibility["only_in_baseline"] == ["shortfall", "violating"]
    assert compatibility["only_in_candidate"] == ["extra"]
    assert any("different declared collections" in reason for reason in compatibility["reasons"])
    assert report["summary"]["compared"] == 2 and report["summary"]["changed"] == 0


def test_two_batches_with_no_common_sample_have_nothing_to_compare(area):
    run(area, "baseline", SAMPLES[:1])
    run(area, "candidate", [{"sample_id": "other", "sample": sample([0.1, 0.2], [0.5], [0.1])}])
    report = plsr_compare.compare_batches(area / "baseline" / "out", area / "candidate" / "out")
    assert report["status"] == "incompatible" and report["samples"] == []
    assert any("share no sample_id" in reason for reason in report["compatibility"]["reasons"])


def test_a_different_runtime_pin_is_explicit_and_resolves_structurally(area, monkeypatch):
    run(area, "baseline")
    identity = dict(engine.runtime_identity())
    identity["commit"] = "0" * 40
    monkeypatch.setattr(engine, "runtime_identity", lambda: identity)
    run(area, "candidate")
    monkeypatch.undo()

    report = plsr_compare.compare_batches(area / "baseline" / "out", area / "candidate" / "out")
    assert report["status"] == "incompatible"
    assert report["compatibility"]["same_runtime_pin"] is False
    assert any("different runtime source pins" in reason
               for reason in report["compatibility"]["reasons"])
    pin = next(item for item in report["compatibility"]["runtime_differences"]
               if item["field"] == "commit")
    assert pin["pin"] is True and pin["candidate"] == "0" * 40
    assert report["summary"]["runtime_changed"] == len(SAMPLES)
    assert report["summary"]["unresolved"] == 0
    for entry in report["samples"]:
        assert entry["baseline"]["resolution"] == "validated"
        assert entry["candidate"]["resolution"] == "structural"
        # The numbers themselves did not move; only the recorded pin did.
        assert entry["diagnostic_differences"] == [] and entry["verdict_differences"] == []
        assert entry["record_digest_matches"] is True


def test_a_missing_bundle_is_reported_rather_than_skipped(area):
    baseline = run(area, "baseline")
    run(area, "candidate")
    bundle_path(area, "baseline", baseline, "certified").unlink()
    report = plsr_compare.compare_batches(area / "baseline" / "out", area / "candidate" / "out")
    assert report["status"] == "incompatible"
    assert report["compatibility"]["unresolved_samples"] == ["certified"]
    entry = next(s for s in report["samples"] if s["sample_id"] == "certified")
    assert entry["baseline"]["resolution"] == "unresolved" and entry["comparable"] is False
    assert any("unreadable" in problem for problem in entry["baseline"]["resolution_problems"])


def test_an_edited_bundle_does_not_pass_as_evidence(area):
    baseline = run(area, "baseline")
    run(area, "candidate")
    path = bundle_path(area, "baseline", baseline, "certified")
    edited = load(path)
    edited["record"]["diagnostics"]["margin"] = 99.0
    path.write_text(json.dumps(edited, indent=2), encoding="utf-8")
    report = plsr_compare.compare_batches(area / "baseline" / "out", area / "candidate" / "out")
    entry = next(s for s in report["samples"] if s["sample_id"] == "certified")
    assert entry["baseline"]["resolution"] == "unresolved"
    assert entry["baseline"]["resolution_problems"][0].startswith("bundle_digest does not match")


def test_a_resealed_bundle_is_still_bound_to_the_batch_index(area):
    baseline = run(area, "baseline")
    run(area, "candidate")
    path = bundle_path(area, "baseline", baseline, "certified")
    reseal(path, lambda b: b["record"]["diagnostics"].update(margin=99.0))
    report = plsr_compare.compare_batches(area / "baseline" / "out", area / "candidate" / "out")
    entry = next(s for s in report["samples"] if s["sample_id"] == "certified")
    assert entry["baseline"]["resolution"] == "unresolved"
    problems = entry["baseline"]["resolution_problems"]
    assert any("record_digest differs from the batch index" in problem for problem in problems)
    assert any("failed full validation" in problem for problem in problems)


def test_the_local_digest_functions_agree_with_the_pinned_runtime(area):
    index = run(area, "baseline")
    for entry in index["entries"]:
        bundle = load(bundle_path(area, "baseline", index, entry["sample_id"]))
        assert plsr_compare.record_digest(bundle["record"]) == L.record_digest(bundle["record"])
        assert L.verify_record(bundle["record"])
        assert plsr_compare.bundle_digest(bundle) == bundle["bundle_digest"]


def perturb(directory, index, sample_id, relative):
    """Move one eigensolver-derived value, keeping the saved batch self-consistent."""
    def change(bundle):
        diagnostics = bundle["record"]["diagnostics"]
        moved = diagnostics["max_decrease"] * (1 + relative)
        diagnostics["max_decrease"] = moved
        diagnostics["margin"] = -moved
        diagnostics["margin_ratio"] = -moved / diagnostics["resolution"]

    entry = next(e for e in index["entries"] if e["sample_id"] == sample_id)
    path = Path(directory) / plsr_batch.RUNS_NAME / entry["saved_file"]
    bundle = reseal(path, change)
    reseal_index(directory, sample_id, bundle["record"]["record_digest"])


def test_a_named_numeric_difference_is_exact_by_default_and_toleranced_on_request(area):
    baseline = run(area, "baseline")
    run(area, "candidate")
    perturb(area / "baseline" / "out", baseline, "certified", 5e-13)

    strict = plsr_compare.compare_batches(area / "baseline" / "out", area / "candidate" / "out")
    entry = next(s for s in strict["samples"] if s["sample_id"] == "certified")
    assert entry["baseline"]["resolution"] == "validated", entry["baseline"]["resolution_problems"]
    assert strict["status"] == "changed"
    assert {d["field"] for d in entry["diagnostic_differences"]} == {
        "diagnostics.max_decrease", "diagnostics.margin", "diagnostics.margin_ratio"}

    policy = plsr_compare.load_tolerance_policy(CORPUS)
    relaxed = plsr_compare.compare_batches(area / "baseline" / "out", area / "candidate" / "out",
                                           tolerance_policy=policy)
    assert relaxed["status"] == "identical"
    assert relaxed["tolerance_policy"]["policy_id"] == "plsr-corpus-tolerance-v1"
    # The digest still records that the retained bytes are not the same.
    assert relaxed["summary"]["record_digest_changed"] == 1


def test_a_difference_beyond_the_declared_tolerance_still_shows(area):
    baseline = run(area, "baseline")
    run(area, "candidate")
    perturb(area / "baseline" / "out", baseline, "certified", 1e-6)
    policy = plsr_compare.load_tolerance_policy(CORPUS)
    report = plsr_compare.compare_batches(area / "baseline" / "out", area / "candidate" / "out",
                                          tolerance_policy=policy)
    assert report["status"] == "changed"
    entry = next(s for s in report["samples"] if s["sample_id"] == "certified")
    assert entry["diagnostic_differences"][0]["tolerance"] == {"relative": 1e-12, "absolute": 0.0}


def test_a_tolerance_policy_may_be_given_directly_or_taken_from_a_corpus(tmp_path):
    from_corpus = plsr_compare.load_tolerance_policy(CORPUS)
    bare = tmp_path / "policy.json"
    bare.write_text(json.dumps(from_corpus), encoding="utf-8")
    assert plsr_compare.load_tolerance_policy(bare) == from_corpus
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"policy_id": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="tolerance_policy fields"):
        plsr_compare.load_tolerance_policy(broken)


def test_an_unfinished_sample_is_a_reported_transition_not_a_silent_omission(area):
    run(area, "baseline")
    plan = write_plan(area / "candidate")
    plsr_batch.run_batch(plan, area / "candidate" / "out", limit=2)
    report = plsr_compare.compare_batches(area / "baseline" / "out", area / "candidate" / "out")
    assert report["status"] == "changed"
    assert report["compatibility"]["incompatible"] is False
    assert report["summary"]["compared"] == len(SAMPLES)
    assert set(report["outcome_transitions"]) == {"completed->unfinished", "violated->unfinished"}
    unfinished = [s for s in report["samples"] if s["candidate"]["outcome"] == "unfinished"]
    assert len(unfinished) == 2
    for entry in unfinished:
        assert entry["comparable"] is False and entry["changed"] is True
        assert entry["candidate"]["saved_file"] is None
        assert entry["candidate"]["resolution"] == "absent"


def test_comparing_a_batch_with_itself_is_identical(area):
    run(area, "baseline")
    directory = area / "baseline" / "out"
    report = plsr_compare.compare_batches(directory, directory)
    assert report["status"] == "identical" and report["summary"]["changed"] == 0


def test_a_directory_that_holds_no_batch_is_an_error(area, tmp_path):
    run(area, "baseline")
    with pytest.raises((OSError, ValueError)):
        plsr_compare.compare_batches(area / "baseline" / "out", tmp_path / "empty")
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / plsr_batch.INDEX_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(plsr_batch.BatchError, match="is not a ciw-plsr-batch-index-v1"):
        plsr_compare.compare_batches(area / "baseline" / "out", tmp_path / "empty")


def test_the_terminal_distinguishes_identical_changed_and_incompatible(area):
    baseline, candidate = area / "baseline" / "out", area / "candidate" / "out"
    run(area, "baseline")
    run(area, "candidate")
    report = cli("plsr", "batch", "compare", baseline, candidate)
    assert report["comparison_schema"] == "ciw-plsr-comparison-v1"
    assert report["status"] == "identical"

    perturb(baseline, plsr_batch.read_index(baseline), "certified", 5e-13)
    changed = cli("plsr", "batch", "compare", baseline, candidate, exit_code=4)
    assert changed["status"] == "changed"
    relaxed = cli("plsr", "batch", "compare", baseline, candidate,
                  "--tolerance-policy", CORPUS)
    assert relaxed["status"] == "identical"

    other = area / "other"
    other.mkdir()
    for path in EXAMPLES.glob("*.json"):
        shutil.copy(path, other / path.name)
    plsr_batch.run_batch(write_plan(other, SAMPLES[:1], batch_id="other"), other / "out")
    incompatible = cli("plsr", "batch", "compare", baseline, other / "out", exit_code=5)
    assert incompatible["status"] == "incompatible"
    assert incompatible["compatibility"]["reasons"]
