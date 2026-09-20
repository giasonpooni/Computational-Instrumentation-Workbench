"""Structural run.v1 validation, independent of scientific domain calculations."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np


RUN_SCHEMA = "run.v1"


def mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def finite_tree(value: Any, name: str = "record") -> None:
    """Reject nonfinite and non-JSON leaves anywhere, not just the selected channel."""
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} keys must be strings")
            finite_tree(child, f"{name}.{key}")
    elif isinstance(value, np.ndarray) and value.ndim == 0:
        finite_tree(value.item(), name)
    elif isinstance(value, (list, tuple, np.ndarray)):
        for index, child in enumerate(value):
            finite_tree(child, f"{name}[{index}]")
    elif isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        number(value, name)
    elif value is not None and not isinstance(value, (str, bool, np.bool_)):
        raise ValueError(f"{name} must contain only JSON-compatible values")


def _sequence(value: Any, name: str) -> Any:
    if not isinstance(value, (list, tuple, np.ndarray)) or (
        isinstance(value, np.ndarray) and value.ndim != 1
    ):
        raise ValueError(f"{name} must be a one-dimensional array")
    return value


def validate_run_structure(run: dict) -> None:
    """Validate transport shape, finite data, time order, units, and source lengths.

    Null channel samples represent explicit missing observations, never zeroes.
    Uniform sampling, calibration validity, covariance semantics, and physical
    model checks belong to the selected adapter. Empty render data is valid.
    Legacy oscillator runs need no added schema field and retain their hashes.
    """
    mapping(run, "run")
    if run.get("run_schema", RUN_SCHEMA) != RUN_SCHEMA:
        raise ValueError("Unsupported run schema")
    for name in ("run_id", "evidence_id", "instrument"):
        string(run.get(name), name)
    finite_tree(run, "run")
    metadata = mapping(run.get("metadata"), "metadata")
    duration = number(metadata.get("duration_s"), "duration_s")
    if duration <= 0:
        raise ValueError("duration_s must be positive")
    count = metadata.get("sample_count")
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("sample_count must be a positive integer")
    frame = string(metadata.get("coordinate_frame"), "coordinate_frame")
    mapping(metadata.get("provenance"), "metadata.provenance")
    if "model" in metadata:
        mapping(metadata["model"], "metadata.model")
    if metadata.get("sample_rate_hz") is not None:
        if number(metadata["sample_rate_hz"], "sample_rate_hz") <= 0:
            raise ValueError("sample_rate_hz must be positive when declared")
    time = _sequence(run.get("time_s"), "time_s")
    if len(time) != count:
        raise ValueError("time_s length must match sample_count")
    previous = -math.inf
    for timestamp in time:
        timestamp = number(timestamp, "time_s entry")
        if not 0 <= timestamp < duration or timestamp <= previous:
            raise ValueError("time_s must be nonnegative, strictly increasing, and below duration_s")
        previous = timestamp
    channels = mapping(run.get("channels"), "channels")
    if not channels:
        raise ValueError("channels must contain at least one source channel")
    for name, channel in channels.items():
        string(name, "channel name")
        channel = mapping(channel, f"channels.{name}")
        string(channel.get("unit"), f"channels.{name}.unit")
        values = _sequence(channel.get("values"), f"channels.{name}.values")
        if len(values) != count:
            raise ValueError(f"channel {name} length must match sample_count")
        for value in values:
            if value is not None:
                number(value, f"channels.{name}.values entry")
    render = mapping(run.get("render", {}), "render")
    if render and render.get("coordinate_frame") != frame:
        raise ValueError("render coordinate_frame must match the source frame")
