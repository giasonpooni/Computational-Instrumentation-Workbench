"""Every check of the uncertainty validation source validator refuses a source that is otherwise the valid example.

An AST mutation probe dropped each ``if ...: raise`` of
``uncertainty_validation`` in turn; 24 of 30 survived. Each case here changes
exactly one thing in the retained consistent example and expects the refusal
that names it.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from ciw import uncertainty_validation as uncertainty
from ciw.telemetry import canonical

ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "examples" / "uncertainty-validation" / "consistent.json").read_bytes()
SOURCE = json.loads(RAW)


def encoded(source):
    return json.dumps(source, allow_nan=False).encode("utf-8")


def assign(*path, value):
    def apply(root):
        target = root
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return apply


def drop(*path):
    def apply(root):
        target = root
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
    return apply


CASES = {
    "source schema renamed": (assign("schema", value="x"), "Unsupported uncertainty validation source"),
    "authority policy changed": (assign("configuration", "activation", value="write"), "Unsupported uncertainty validation source"),
    "confidence given as text": (assign("request", "confidence", value="0.9"), "confidence must be a probability"),
    "state units given as text of the same length": (assign("request", "state_units", value="mm"), "state_units must match state_names"),
    "state units with a third entry": (assign("request", "state_units", value=["m", "m/s", "m"]), "state_units must match state_names"),
    "innovations without measurement units": (drop("request", "measurement_units"), "measurement_names and measurement_units together"),
    "measurement units given as text of the same length": (assign("request", "measurement_units", value="m"), "measurement_units must match measurement_names"),
    "measurement units with a second entry": (assign("request", "measurement_units", value=["m", "m"]), "measurement_units must match measurement_names"),
    "state names given as text": (assign("request", "state_names", value="ab"), "distinct names"),
    "seventeen state names": (assign("request", "state_names", value=[f"s{i}" for i in range(17)]), "distinct names"),
    "repeated state names": (assign("request", "state_names", value=["position", "position"]), "distinct names"),
    "reference vector given as text": (assign("request", "samples", 0, "reference", value="ab"), "must hold exactly 2 ordered numbers"),
    "reference entry given as text": (assign("request", "samples", 0, "reference", 0, value="1"), "must hold finite numbers"),
    "zero variance with a zero row": (assign("request", "samples", 0, "covariance", value=[[0.0, 0.0], [0.0, 0.0025]]), "strictly positive variances"),
    "covariance beyond the condition budget": (assign("request", "samples", 0, "covariance", value=[[1.0, 0.0], [0.0, 1e-13]]), "bounded condition number"),
    "samples given as text": (assign("request", "samples", value="ab"), "must hold between 2 and"),
    "a single sample": (assign("request", "samples", value=SOURCE["request"]["samples"][:1]), "must hold between 2 and"),
    "sample time given as text": (assign("request", "samples", 1, "time", value="1"), "time must be a finite number"),
}


def test_the_example_source_is_accepted():
    assert uncertainty.validate_source(RAW)["experiment_id"] == SOURCE["experiment_id"]


@pytest.mark.parametrize("mutate, message", CASES.values(), ids=list(CASES))
def test_one_change_to_the_example_source_is_refused_by_its_own_check(mutate, message):
    source = deepcopy(SOURCE)
    mutate(source)
    with pytest.raises(ValueError, match=message):
        uncertainty.validate_source(encoded(source))


@pytest.mark.parametrize("raw", ["text", b""], ids=["text", "empty"])
def test_a_source_must_be_bounded_exact_bytes(raw):
    with pytest.raises(ValueError, match="bounded exact JSON bytes"):
        uncertainty.validate_source(raw)


def test_nonfinite_numbers_are_refused_after_decoding():
    with pytest.raises(ValueError, match="must hold finite numbers"):
        uncertainty._vector([float("inf"), 0.0], 2, "reference")
    entry = deepcopy(SOURCE["request"]["samples"][0])
    later = deepcopy(SOURCE["request"]["samples"][1])
    entry["time"] = float("inf")
    with pytest.raises(ValueError, match="time must be a finite number"):
        uncertainty._samples([entry, later], "samples", ("reference", "estimate"), 2)


def shorter_spelling(source, before, after):
    """Exact bytes that decode to ``source`` but are shorter than its canonical form."""
    raw = canonical(source).replace(before, after, 1)
    assert json.loads(raw) == source and len(raw) < len(canonical(source))
    return raw


def test_the_canonical_byte_budget_applies_after_decoding(monkeypatch):
    # A number may be spelled more briefly than its canonical form, so the
    # canonical size is checked again after decoding.
    raw = shorter_spelling(SOURCE, b"10.0", b"1e1")
    monkeypatch.setattr(uncertainty, "SOURCE_LIMIT", len(raw))
    with pytest.raises(ValueError, match="exceeds the byte budget"):
        uncertainty.validate_source(raw)
