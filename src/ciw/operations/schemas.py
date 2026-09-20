"""Trusted, read-only payload schemas, independent of executable runtime bindings.

Persisted files cannot register or import validators. New providers register a
schema in trusted process setup, alongside their executable operation binding.
"""
from __future__ import annotations

from collections.abc import Callable

from ..core.records import finite_tree
from .registry import valid_operation_id

_VALIDATORS: dict[str, Callable] = {}


def validate_role(operation_id: str, role: str) -> None:
    expected = {"statistics.v1": "analysis", "spectrum.periodogram.v1": "analysis",
                "fsrt.tank-reconstruct.v1": "state_estimator",
                "fsrt.tank-reconstruct.v2": "state_estimator",
                "jspt.covariance-propagate.v1": "backend"}.get(operation_id)
    if expected is not None and role != expected:
        raise ValueError("Operation role contradicts the declared payload contract")


def register_payload_validator(operation_id: str, validator: Callable) -> None:
    if not valid_operation_id(operation_id) or not callable(validator):
        raise ValueError("A payload schema requires a versioned operation and callable validator")
    if operation_id in _VALIDATORS or operation_id in {
        "statistics.v1", "spectrum.periodogram.v1", "fsrt.tank-reconstruct.v1",
        "fsrt.tank-reconstruct.v2", "jspt.covariance-propagate.v1"
    }:
        raise ValueError("Payload schema already registered")
    _VALIDATORS[operation_id] = validator


def validate_payload(operation_id: str, data: dict, run: dict, parameters: dict, selection: dict) -> None:
    if not isinstance(data, dict):
        raise ValueError("Operation data must be an object")
    finite_tree(data, "operation data")
    if operation_id in {"statistics.v1", "spectrum.periodogram.v1"}:
        from ..adapters.oscillator_records import validate_payload as validator
    elif operation_id == "fsrt.tank-reconstruct.v1":
        from ..adapters.fsrt_records import validate_payload as validator
    elif operation_id == "fsrt.tank-reconstruct.v2":
        from ..adapters.covariance_records import validate_fsrt_payload as validator
    elif operation_id == "jspt.covariance-propagate.v1":
        from ..adapters.covariance_records import validate_jspt_payload as validator
    else:
        validator = _VALIDATORS.get(operation_id)
        if validator is None:
            raise ValueError(f"No trusted saved-payload schema for {operation_id}")
    validator(operation_id, data, run, parameters, selection)
