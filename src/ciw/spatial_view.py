"""Read-only geographic declarations for the existing GSV provider interface.

Coordinates and constant states must already be declared in retained evidence.
This module neither projects local geometry to Earth nor constructs dynamics.
GSV performs its native WorldSnapshot/EntityState validation before rendering.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import math
import re

from .adapters.subprocess import _json
from .telemetry import canonical

MAX_BYTES = 262144
SOURCE_SCHEMA = "ciw.geographic-context.v1"
VIEW_SCHEMA = "ciw.spatial-view.v1"
STATE_POLICY = "declared_constant_over_inclusive_time_range"
AUTHORITY = {"read_only": True, "coordinate_transform": "not_performed",
             "sensor_fusion": "not_performed", "state_admission": "not_performed",
             "execution": "not_performed", "verification": "not_performed",
             "physical_authenticity": "not_established", "covariance": "not_supplied"}
_COLLECTIONS = ("routes", "flows", "commodities", "events", "constraints", "assertions", "observations", "cityLights")
_KINDS = {"port", "airport", "rail_terminal", "trucking_hub", "warehouse", "distribution_center",
          "border_crossing", "mine", "oil_field", "gas_field", "agricultural_region", "refinery",
          "smelter", "chemical_plant", "steel_mill", "processing_facility", "factory", "industrial_park",
          "manufacturing_cluster", "consumption_center", "city", "chokepoint"}
_STATUS = {"active", "inactive", "planned", "degraded", "disrupted", "unknown"}


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() <= set(required) | set(optional):
        raise ValueError("Unexpected or missing geographic declaration fields")


def _text(value, maximum=256, identity=False):
    if (not isinstance(value, str) or not value.strip() or len(value) > maximum or
            any(ord(c) < 32 or ord(c) == 127 for c in value) or (identity and any(c.isspace() for c in value))):
        raise ValueError("Require bounded geographic text or identity")


def _time(value):
    _text(value, 24)
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,3})?Z", value):
        raise ValueError("Require explicit UTC time with at most millisecond precision")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid geographic UTC time") from exc


def _number(value, low, high):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError("Geographic number outside declared bounds")


def _source(raw: bytes) -> dict:
    if not isinstance(raw, bytes) or len(raw) > MAX_BYTES:
        raise ValueError("Geographic source exceeds byte budget")
    source = _json(raw)
    _keys(source, {"schema", "coordinate_frame", "state_policy", "snapshot", "states"})
    if source["schema"] != SOURCE_SCHEMA or source["state_policy"] != STATE_POLICY:
        raise ValueError("Require explicit geographic schema and constant-state policy")
    frame = source["coordinate_frame"]
    _keys(frame, {"id", "axes", "unit", "authority_ref"})
    if frame["id"] != "OGC:CRS84" or frame["axes"] != ["longitude", "latitude"] or frame["unit"] != "deg":
        raise ValueError("Geographic view requires declared CRS84 longitude/latitude degrees")
    _text(frame["authority_ref"], identity=True)
    snapshot = source["snapshot"]
    _keys(snapshot, {"nodes", "timeRange", "meta", *_COLLECTIONS})
    if any(snapshot[key] != [] for key in _COLLECTIONS):
        raise ValueError("Geographic v1 accepts declared facility points; dynamic collections must be empty")
    if not isinstance(snapshot["nodes"], list) or not 1 <= len(snapshot["nodes"]) <= 128:
        raise ValueError("Geographic source requires 1..128 facility points")
    time_range = snapshot["timeRange"]
    _keys(time_range, {"start", "end", "now"})
    start, end, now = (_time(time_range[key]) for key in ("start", "end", "now"))
    if not start < end or not start <= now <= end:
        raise ValueError("Invalid geographic time range")
    _keys(snapshot["meta"], {"label", "disclaimer", "generatedAt"})
    for key in ("label", "disclaimer"):
        _text(snapshot["meta"][key], 4096)
    _time(snapshot["meta"]["generatedAt"])
    nodes = {}
    for node in snapshot["nodes"]:
        _keys(node, {"id", "kind", "name", "geometry", "status", "provenance", "importance"}, {"country", "tags"})
        _text(node["id"], identity=True)
        _text(node["name"], 4096)
        if (node["id"] in nodes or not isinstance(node["kind"], str) or node["kind"] not in _KINDS or
                not isinstance(node["status"], str) or node["status"] not in _STATUS):
            raise ValueError("Duplicate geographic identity or unsupported facility/status")
        _number(node["importance"], 0, 1)
        _keys(node["geometry"], {"type", "coordinates"})
        coordinates = node["geometry"]["coordinates"]
        if node["geometry"]["type"] != "Point" or not isinstance(coordinates, list) or len(coordinates) != 2:
            raise ValueError("Require explicit point geometry")
        _number(coordinates[0], -180, 180)
        _number(coordinates[1], -90, 90)
        provenance = node["provenance"]
        _keys(provenance, {"source", "knownAt", "evidence", "validFrom", "validTo"}, {"confidence"})
        _text(provenance["source"], 4096)
        _time(provenance["knownAt"])
        # GSV's native range is inclusive; a half-open fact interval must
        # cover its final selectable instant as well.
        if _time(provenance["validFrom"]) > start or _time(provenance["validTo"]) <= end:
            raise ValueError("Facility validity must cover the complete inclusive view range")
        refs = provenance["evidence"]
        if not isinstance(refs, list) or not 1 <= len(refs) <= 64 or frame["authority_ref"] not in refs:
            raise ValueError("Each location must retain its declared frame authority reference")
        for ref in refs:
            _text(ref, 4096)
        if "confidence" in provenance:
            _number(provenance["confidence"], 0, 1)
        if "country" in node and (not isinstance(node["country"], str) or not re.fullmatch("[A-Z]{2}", node["country"])):
            raise ValueError("Require declared two-letter country code")
        if "tags" in node:
            if not isinstance(node["tags"], list) or len(node["tags"]) > 64:
                raise ValueError("Too many geographic tags")
            for tag in node["tags"]:
                _text(tag, 4096)
        nodes[node["id"]] = node
    states = source["states"]
    if not isinstance(states, list) or len(states) != len(nodes):
        raise ValueError("Each facility requires its own explicitly declared state")
    seen = set()
    for state in states:
        _keys(state, {"entityId", "t", "utilization", "congestion", "status", "activeEventIds"})
        identity = state["entityId"]
        _text(identity, identity=True)
        if identity not in nodes or identity in seen or state["t"] != time_range["now"]:
            raise ValueError("Declared state must bind one facility and the reference instant")
        if state["activeEventIds"] != [] or state["status"] != nodes[identity]["status"]:
            raise ValueError("State status must match its facility; dynamic events are unsupported")
        for key in ("utilization", "congestion"):
            _number(state[key], 0, 1)
        seen.add(identity)
    return source


def project(source: dict) -> dict:
    """Project a retained CIW source descriptor including its exact bytes_b64.

    The descriptor is the ordinary workbench source record. No native result,
    execution or verification identity is invented for a view operation.
    """
    _keys(source, {"schema", "kind", "label", "source_schema", "evidence_id", "byte_count", "source_id", "bytes_b64"})
    encoded = source["bytes_b64"]
    if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_BYTES + 2) // 3):
        raise ValueError("Invalid geographic source byte envelope")
    raw = base64.b64decode(encoded, validate=True)
    if (base64.b64encode(raw).decode("ascii") != encoded or source["evidence_id"] != "sha256:" + sha256(raw).hexdigest() or
            type(source["byte_count"]) is not int or source["byte_count"] != len(raw)):
        raise ValueError("Geographic evidence byte identity mismatch")
    if source["schema"] != "ciw.workbench-source.v1" or source["kind"] != "geographic-context" or source["source_schema"] != SOURCE_SCHEMA:
        raise ValueError("Geographic view requires retained geographic source")
    _text(source["source_id"], identity=True)
    _text(source["label"], 512)
    descriptor = {key: value for key, value in source.items() if key not in {"source_id", "bytes_b64"}}
    if source["source_id"] != "source:sha256:" + sha256(canonical(descriptor)).hexdigest():
        raise ValueError("Geographic source descriptor identity mismatch")
    declaration = _source(raw)
    return deepcopy({"schema": VIEW_SCHEMA, "source": source,
                     "coordinate_frame": declaration["coordinate_frame"],
                     "state_policy": STATE_POLICY, "authority": AUTHORITY})
