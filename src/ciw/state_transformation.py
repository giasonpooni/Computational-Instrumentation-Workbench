"""Provider-free contracts for state-space transformations.

This module describes a transformation and the invariants it is expected to
preserve. It does not execute the transformation, inspect a provider, or
turn a declaration into evidence that the transformation happened.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .telemetry import canonical, digest


SCHEMA = "ciw.state-transformation-contract.v1"
SEALED_SCHEMA = "ciw.state-transformation-record.v1"
MAX_BYTES = 128 * 1024
MAX_COORDINATES = 32
MAX_ITEMS = 64
_AUTHORITY = {
    "execution": "not_performed",
    "verification": "not_performed",
    "state_admission": "not_performed",
    "hardware_actuation": "not_performed",
}


def _keys(value, required):
    if type(value) is not dict or set(value) != set(required):
        raise ValueError("State transformation requires exactly the declared fields")


def _text(value, field, limit=256):
    if type(value) is not str or not value.strip() or len(value) > limit or any(ord(c) < 32 for c in value):
        raise ValueError(f"{field} must be bounded nonempty text")
    return value


def _items(value, field, *, minimum=0):
    if type(value) is not list or not minimum <= len(value) <= MAX_ITEMS:
        raise ValueError(f"{field} must contain {minimum}..{MAX_ITEMS} items")
    return value


def _state_space(value, field):
    _keys(value, {"name", "coordinates", "frame", "clock"})
    _text(value["name"], f"{field}.name")
    _text(value["frame"], f"{field}.frame")
    _text(value["clock"], f"{field}.clock")
    coordinates = value["coordinates"]
    if type(coordinates) is not list or not 1 <= len(coordinates) <= MAX_COORDINATES:
        raise ValueError(f"{field}.coordinates must contain 1..{MAX_COORDINATES} items")
    names = set()
    for coordinate in coordinates:
        _keys(coordinate, {"name", "unit", "role"})
        name = _text(coordinate["name"], f"{field}.coordinates[].name", 128)
        _text(coordinate["unit"], f"{field}.coordinates[].unit", 64)
        _text(coordinate["role"], f"{field}.coordinates[].role", 64)
        if name in names:
            raise ValueError(f"{field}.coordinates names must be distinct")
        names.add(name)


def _declarations(value, field, *, minimum):
    declarations = _items(value, field, minimum=minimum)
    seen = set()
    for declaration in declarations:
        _keys(declaration, {"id", "statement", "check"})
        identifier = _text(declaration["id"], f"{field}[].id", 128)
        _text(declaration["statement"], f"{field}[].statement", 1024)
        _text(declaration["check"], f"{field}[].check", 256)
        if identifier in seen:
            raise ValueError(f"{field} identifiers must be distinct")
        seen.add(identifier)


def _evidence(value):
    records = _items(value, "evidence", minimum=1)
    for record in records:
        _keys(record, {"ref", "kind", "role"})
        _text(record["ref"], "evidence[].ref", 256)
        _text(record["kind"], "evidence[].kind", 64)
        _text(record["role"], "evidence[].role", 128)


def validate_contract(contract):
    """Validate a declaration without executing or authenticating it."""
    try:
        encoded = canonical(contract)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("State transformation must be finite bounded JSON") from exc
    if len(encoded) > MAX_BYTES:
        raise ValueError("State transformation exceeds its byte budget")
    _keys(contract, {"schema", "transformation_id", "input_state_space", "transformation",
                     "constraints", "invariants", "output_state_space", "evidence", "authority"})
    if contract["schema"] != SCHEMA:
        raise ValueError("Unsupported state transformation schema")
    _text(contract["transformation_id"], "transformation_id", 128)
    _state_space(contract["input_state_space"], "input_state_space")
    _state_space(contract["output_state_space"], "output_state_space")
    _keys(contract["transformation"], {"name", "kind", "parameters"})
    _text(contract["transformation"]["name"], "transformation.name")
    _text(contract["transformation"]["kind"], "transformation.kind", 128)
    if type(contract["transformation"]["parameters"]) is not dict:
        raise ValueError("transformation.parameters must be an object")
    canonical(contract["transformation"]["parameters"])
    _declarations(contract["constraints"], "constraints", minimum=0)
    _declarations(contract["invariants"], "invariants", minimum=1)
    _evidence(contract["evidence"])
    if contract["authority"] != _AUTHORITY:
        raise ValueError("State transformation authority must remain read-only and unperformed")
    return deepcopy(contract)


def seal_contract(contract):
    """Create a content-addressed declaration record, without claiming execution."""
    validated = validate_contract(contract)
    record = {"schema": SEALED_SCHEMA, "contract": validated, "contract_digest": digest(validated)}
    record["record_digest"] = digest(record)
    return record


def validate_sealed(record):
    """Reopen a sealed declaration and check both content identities."""
    _keys(record, {"schema", "contract", "contract_digest", "record_digest"})
    if record["schema"] != SEALED_SCHEMA or record["contract_digest"] != digest(record["contract"]):
        raise ValueError("State transformation contract identity differs")
    body = {key: value for key, value in record.items() if key != "record_digest"}
    if record["record_digest"] != digest(body):
        raise ValueError("State transformation record identity differs")
    return validate_contract(record["contract"])

