"""Byte/frame/time retention and explicit refusal for geographic view declarations."""
import base64
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from ciw.spatial_view import _source, project, AUTHORITY
from ciw.telemetry import canonical

EXAMPLE = Path(__file__).parents[1] / "examples/workbench/geographic-context.json"


def retained(raw=None):
    raw = EXAMPLE.read_bytes() if raw is None else raw
    descriptor = {"schema": "ciw.workbench-source.v1", "kind": "geographic-context", "label": "Geography — retained",
                  "source_schema": "ciw.geographic-context.v1", "evidence_id": "sha256:" + sha256(raw).hexdigest(),
                  "byte_count": len(raw)}
    return {**descriptor, "source_id": "source:sha256:" + sha256(canonical(descriptor)).hexdigest(),
            "bytes_b64": base64.b64encode(raw).decode()}


def test_exact_evidence_native_values_and_authority_remain_distinct():
    source = retained()
    original = deepcopy(source)
    view = project(source)
    assert source == original
    assert base64.b64decode(view["source"]["bytes_b64"]) == EXAMPLE.read_bytes()
    assert view["authority"] == AUTHORITY
    assert {"result_id", "execution_id", "verification_id"}.isdisjoint(view)
    assert _source(EXAMPLE.read_bytes())["snapshot"]["nodes"][0]["geometry"]["coordinates"] == [-79.7, 43.65]
    view["coordinate_frame"]["id"] = "corrupt"
    assert project(source)["coordinate_frame"]["id"] == "OGC:CRS84"


@pytest.mark.parametrize("mutation", [
    lambda s: s["coordinate_frame"].update(id="bench-plane", unit="m"),
    lambda s: s["coordinate_frame"].update(axes=["latitude", "longitude"]),
    lambda s: s["snapshot"]["nodes"][0]["geometry"].update(coordinates=[True, 2]),
    lambda s: s["snapshot"]["nodes"][0]["geometry"].update(coordinates=[181, 2]),
    lambda s: s["snapshot"]["nodes"][0]["provenance"].update(evidence=[]),
    lambda s: s["snapshot"]["nodes"][0]["provenance"].update(validTo=s["snapshot"]["timeRange"]["end"]),
    lambda s: s["snapshot"]["timeRange"].update(now="2026-02-30T00:00:00Z"),
    lambda s: s["snapshot"]["nodes"][1].update(id=s["snapshot"]["nodes"][0]["id"]),
    lambda s: s["snapshot"].update(flows=[{}]),
    lambda s: s["states"].pop(),
    lambda s: s["states"][0].pop("utilization"),
    lambda s: s["states"][0].update(entityId="absent"),
    lambda s: s["states"][0].update(status="unknown"),
    lambda s: s["states"][0].update(t="2026-09-22T10:00:00Z"),
    lambda s: s.update(state_policy="interpolate"),
])
def test_no_implicit_geometry_state_time_or_missingness_conversion(mutation):
    source = json.loads(EXAMPLE.read_bytes())
    mutation(source)
    with pytest.raises(ValueError):
        _source(canonical(source))


@pytest.mark.parametrize("field,value", [
    ("evidence_id", "sha256:" + "0" * 64), ("source_id", "source:sha256:" + "0" * 64),
    ("kind", "calibrated-observable"), ("byte_count", 1), ("label", "substituted"),
])
def test_resealed_or_substituted_descriptor_refuses(field, value):
    source = retained()
    source[field] = value
    with pytest.raises(ValueError):
        project(source)


def test_duplicate_source_fields_and_oversized_bytes_refuse():
    with pytest.raises(ValueError):
        _source(b'{"schema":"wrong",' + EXAMPLE.read_bytes().lstrip()[1:])
    with pytest.raises(ValueError):
        _source(b" " * 262145)
