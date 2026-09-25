"""Every check of the machine manifest source validator refuses a source that is otherwise the valid example.

An AST mutation probe dropped each ``if ...: raise`` of ``machine_workflow``
in turn; all 15 survived. Artifacts changed here are resealed so that only
the named link is broken.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from ciw import machine_manifest as manifest
from ciw import machine_workflow
from ciw.telemetry import canonical

ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "examples" / "machine-manifest" / "source.json").read_bytes()
SOURCE = json.loads(RAW)
FOREIGN = "sha256:" + "0" * 64


def encoded(source):
    return json.dumps(source, allow_nan=False).encode("utf-8")


def resealed(artifact, **changes):
    unsigned = {key: value for key, value in artifact.items() if key != "artifact_digest"}
    unsigned.update(changes)
    return manifest.seal(unsigned)


def candidate_of_another_machine(source):
    source["candidate_manifest"] = resealed(source["candidate_manifest"], machine_id="machine:other")
    source["challenge_report"] = resealed(source["challenge_report"], candidate_digest=source["candidate_manifest"]["artifact_digest"])


def report_of_another_machine(source):
    source["challenge_report"] = resealed(source["challenge_report"], machine_id="machine:other")


def candidate_over_other_evidence(source):
    source["candidate_manifest"] = resealed(source["candidate_manifest"], evidence_bundle_digest=FOREIGN)
    source["challenge_report"] = resealed(source["challenge_report"], candidate_digest=source["candidate_manifest"]["artifact_digest"])


def report_of_another_candidate(source):
    source["challenge_report"] = resealed(source["challenge_report"], candidate_digest=FOREIGN)


def report_over_other_evidence(source):
    source["challenge_report"] = resealed(source["challenge_report"], evidence_bundle_digest=FOREIGN)


def swap(target, other):
    def apply(source):
        source[target] = deepcopy(source[other])
    return apply


def assign(*path, value):
    def apply(root):
        target = root
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return apply


CASES = {
    "source schema renamed": (assign("schema", value="x"), "Unsupported machine manifest source"),
    "authority policy changed": (assign("configuration", "activation", value="write"), "Unsupported machine manifest source"),
    "candidate in place of the evidence": (swap("evidence_bundle", "candidate_manifest"), "evidence, candidate and challenge artifacts"),
    "evidence in place of the candidate": (swap("candidate_manifest", "evidence_bundle"), "evidence, candidate and challenge artifacts"),
    "candidate in place of the report": (swap("challenge_report", "candidate_manifest"), "evidence, candidate and challenge artifacts"),
    "candidate of another machine": (candidate_of_another_machine, "identities are not linked"),
    "report of another machine": (report_of_another_machine, "identities are not linked"),
    "candidate over other evidence": (candidate_over_other_evidence, "identities are not linked"),
    "report of another candidate": (report_of_another_candidate, "identities are not linked"),
    "report over other evidence": (report_over_other_evidence, "identities are not linked"),
    "count given as a float": (assign("request", "counts", value=300.0), "exactly retained bounded integer"),
    "count beyond binary64 exactness": (assign("request", "counts", value=2 ** 53), "exactly retained bounded integer"),
}


def test_the_example_source_is_accepted():
    assert machine_workflow.validate_source(RAW)["experiment_id"] == SOURCE["experiment_id"]


@pytest.mark.parametrize("mutate, message", CASES.values(), ids=list(CASES))
def test_one_change_to_the_example_source_is_refused_by_its_own_check(mutate, message):
    source = deepcopy(SOURCE)
    mutate(source)
    with pytest.raises(ValueError, match=message):
        machine_workflow.validate_source(encoded(source))


@pytest.mark.parametrize("raw", ["text", b""], ids=["text", "empty"])
def test_a_source_must_be_bounded_exact_bytes(raw):
    with pytest.raises(ValueError, match="bounded exact JSON bytes"):
        machine_workflow.validate_source(raw)


def shorter_spelling(source, before, after):
    """Exact bytes that decode to ``source`` but are shorter than its canonical form."""
    raw = canonical(source).replace(before, after, 1)
    assert json.loads(raw) == source and len(raw) < len(canonical(source))
    return raw


def test_the_canonical_byte_budget_applies_after_decoding(monkeypatch):
    raw = shorter_spelling(SOURCE, b"1e-06", b"1e-6")
    monkeypatch.setattr(machine_workflow, "SOURCE_LIMIT", len(raw))
    with pytest.raises(ValueError, match="exceeds the byte budget"):
        machine_workflow.validate_source(raw)
