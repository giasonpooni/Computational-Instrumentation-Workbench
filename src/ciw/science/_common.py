"""Shared refusal type and bounded validation helpers for scientific records.

Validators reject rather than repair: a malformed covariance, a non-finite
number or an undeclared unit is refused, never symmetrized, clipped or guessed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from numbers import Real
from typing import Any

import numpy as np

from ..adapters.protocol import AdapterRefusal
from ..core.identities import canonical_json, content_identity

__all__ = [
    "Refusal", "canonical_json", "content_identity", "mapping", "text", "finite", "integer",
    "vector", "matrix", "covariance", "utc_now", "iso_time", "require_keys", "bounded_json",
]

MAX_DIMENSION = 64


class Refusal(AdapterRefusal):
    """A declared inability to proceed, carrying machine-readable detail."""

    def __init__(self, code: str, message: str, **detail: Any):
        super().__init__(code, message)
        self.detail = detail

    def to_dict(self) -> dict:
        record = super().to_dict()
        if self.detail:
            record["detail"] = _plain(self.detail)
        return record


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_plain(item) for item in value]
        return sorted(items, key=canonical_json) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise Refusal("malformed_record", f"{name} must be an object")
    return value


def require_keys(value: dict, name: str, required: set[str], optional: set[str] = frozenset()) -> dict:
    mapping(value, name)
    missing = required - value.keys()
    unknown = value.keys() - required - set(optional)
    if missing:
        raise Refusal("malformed_record", f"{name} is missing {sorted(missing)}")
    if unknown:
        raise Refusal("malformed_record", f"{name} has undeclared fields {sorted(unknown)}")
    return value


def text(value: Any, name: str, limit: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Refusal("malformed_record", f"{name} must be a nonempty string")
    if len(value) > limit:
        raise Refusal("oversized_input", f"{name} exceeds {limit} characters")
    return value


def finite(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None,
           exclusive_minimum: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise Refusal("malformed_record", f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise Refusal("malformed_record", f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise Refusal("malformed_record", f"{name} must be a finite number")
    if minimum is not None and (result < minimum or (exclusive_minimum and result == minimum)):
        raise Refusal("out_of_domain", f"{name} must be {'>' if exclusive_minimum else '>='} {minimum}")
    if maximum is not None and result > maximum:
        raise Refusal("out_of_domain", f"{name} must be <= {maximum}")
    return result


def integer(value: Any, name: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise Refusal("malformed_record", f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise Refusal("out_of_domain", f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise Refusal("out_of_domain", f"{name} must be <= {maximum}")
    return result


def vector(value: Any, name: str, size: int | None = None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or (size is not None and len(value) != size):
        raise Refusal("malformed_record", f"{name} must be an array" + ("" if size is None else f" of length {size}"))
    if len(value) > MAX_DIMENSION * MAX_DIMENSION:
        raise Refusal("oversized_input", f"{name} is too large")
    return np.array([finite(item, f"{name}[{index}]") for index, item in enumerate(value)], dtype=float)


def matrix(value: Any, name: str, rows: int | None = None, columns: int | None = None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value or (rows is not None and len(value) != rows):
        raise Refusal("malformed_record", f"{name} must be a nonempty matrix" + ("" if rows is None else f" with {rows} rows"))
    if len(value) > MAX_DIMENSION:
        raise Refusal("oversized_input", f"{name} exceeds {MAX_DIMENSION} rows")
    width = columns if columns is not None else (len(value[0]) if isinstance(value[0], (list, tuple)) else -1)
    result = np.array([vector(row, f"{name}[{index}]", width) for index, row in enumerate(value)], dtype=float)
    return result


def covariance(value: Any, name: str, size: int | None = None, *, tolerance: float = 1e-12) -> np.ndarray:
    """Admit a covariance only if it is square, symmetric and positive semidefinite.

    Symmetry is checked to a relative tolerance; the matrix is never repaired.
    """
    result = matrix(value, name, size, size)
    if result.shape[0] != result.shape[1]:
        raise Refusal("malformed_covariance", f"{name} must be square")
    scale = max(1.0, float(np.max(np.abs(result))))
    if not np.allclose(result, result.T, rtol=0.0, atol=tolerance * scale):
        raise Refusal("malformed_covariance", f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(0.5 * (result + result.T))
    if eigenvalues.min() < -tolerance * scale * result.shape[0]:
        raise Refusal("malformed_covariance", f"{name} must be positive semidefinite",
                      minimum_eigenvalue=float(eigenvalues.min()))
    return result


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_time(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise Refusal("malformed_record", f"{name} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise Refusal("malformed_record", f"{name} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Refusal("clock_unspecified", f"{name} must carry an explicit UTC offset")
    return parsed


def bounded_json(value: Any, name: str, limit: int) -> str:
    """Canonical JSON with finite numbers only, refused above a byte bound."""
    try:
        encoded = canonical_json(_plain(value))
    except (TypeError, ValueError) as exc:
        raise Refusal("malformed_record", f"{name} must be finite JSON: {exc}") from exc
    if len(encoded.encode("utf-8")) > limit:
        raise Refusal("oversized_input", f"{name} exceeds {limit} bytes")
    return encoded


def plain(value: Any) -> Any:
    """Convert numpy containers to JSON-native values without changing meaning."""
    return _plain(value)
