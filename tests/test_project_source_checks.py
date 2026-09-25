"""Every check of the project graph source validator refuses a source that is otherwise the valid fixture.

An AST mutation probe dropped each ``if ...: raise`` of ``project_workflow``
in turn; 6 of 8 survived.
"""
from copy import deepcopy
import json

import pytest

from ciw import project_model as project
from ciw import project_workflow
from ciw.telemetry import canonical
from test_project_workflow import _project, _source

SOURCE = _source()


def encoded(source):
    return json.dumps(source, allow_nan=False).encode("utf-8")


def assign(*path, value):
    def apply(root):
        target = root
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return apply


CASES = {
    "source schema renamed": (assign("schema", value="x"), "Unsupported project graph source"),
    "authority policy changed": (assign("configuration", "activation", value="write"), "Unsupported project graph source"),
    "expected revision given a number": (assign("request", "expected_revision", value=5), "canonical SHA256 digest"),
    "expected revision malformed": (assign("request", "expected_revision", value="abc"), "canonical SHA256 digest"),
    "expected revision of another project": (assign("request", "expected_revision", value="sha256:" + "0" * 64), "differs from the requested revision"),
}


def test_the_fixture_source_is_accepted():
    assert project_workflow.validate_source(encoded(SOURCE))["experiment_id"] == SOURCE["experiment_id"]


@pytest.mark.parametrize("mutate, message", CASES.values(), ids=list(CASES))
def test_one_change_to_the_fixture_source_is_refused_by_its_own_check(mutate, message):
    source = deepcopy(SOURCE)
    mutate(source)
    with pytest.raises(ValueError, match=message):
        project_workflow.validate_source(encoded(source))


@pytest.mark.parametrize("raw", ["text", b""], ids=["text", "empty"])
def test_a_source_must_be_bounded_exact_bytes(raw):
    with pytest.raises(ValueError, match="bounded exact JSON bytes"):
        project_workflow.validate_source(raw)


def shorter_spelling(source, before, after):
    """Exact bytes that decode to ``source`` but are shorter than its canonical form."""
    raw = canonical(source).replace(before, after, 1)
    assert json.loads(raw) == source and len(raw) < len(canonical(source))
    return raw


def test_the_canonical_byte_budget_applies_after_decoding(monkeypatch):
    value = _project()
    objects = {item["object_id"]: item for item in project.inspect(value)["objects"]}
    value = project.put(value, {"object_id": "result:position", "kind": "result", "label": "Position estimate", "content": {
        "value": 10.0, "input_revisions": {
            "signal:position": objects["signal:position"]["revision"],
            "computation:position": objects["computation:position"]["revision"]},
        "unit": "m", "frame": "frame:carriage", "time_basis": "clock:utc", "semantics": "estimated"}})
    source = _source(value)
    raw = shorter_spelling(source, b"10.0", b"1e1")
    monkeypatch.setattr(project_workflow, "SOURCE_LIMIT", len(raw))
    with pytest.raises(ValueError, match="exceeds the byte budget"):
        project_workflow.validate_source(raw)
