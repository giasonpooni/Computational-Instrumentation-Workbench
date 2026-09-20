"""Content-addressed covariance artifacts; CIW validates, domain providers compute.

``quantity_ids`` specifies both matrix axes. ``units[i]`` is the unit of the
corresponding quantity, so entry (i, j) has units ``units[i] * units[j]``. Mixed
units are permitted, but conversion, propagation, coordinate transformations,
and interpretation remain the responsibility of the scientific provider.

Validation never repairs an artifact. In particular it does not symmetrize,
clip eigenvalues, replace missing covariance, or discard off-diagonal entries.
Numerical acceptance tolerances apply to dimensionless correlation coordinates;
all original entries and declarations remain part of the content identity.
"""

from __future__ import annotations

from copy import deepcopy
import math
import re
from typing import Any

import numpy as np

from .identities import content_identity


COVARIANCE_SCHEMA = "covariance-artifact.v1"
SYMMETRY_ATOL = 1e-12
PSD_ATOL = 1e-10
_FIELDS = frozenset({
    "schema", "covariance_id", "quantity_ids", "units", "frame", "reference_values",
    "matrix", "method", "basis", "provenance", "assumptions",
})
_BASIS_KINDS = frozenset({
    "observation", "calibrated_observation", "estimated_state", "parameter", "coordinate", "residual",
})
_CONTENT_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite JSON number, not a boolean")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite JSON number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite JSON number")
    return result


def _json_tree(value: Any, name: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} keys must be strings")
            _json_tree(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _json_tree(item, f"{name}[{index}]")
    elif value is None or type(value) in (str, bool):
        return
    else:
        _number(value, name)


def _strings(value: Any, name: str, *, count: int | None = None) -> list[str]:
    if not isinstance(value, list) or (count is not None and len(value) != count):
        raise ValueError(f"{name} must be an array" + (f" with {count} entries" if count is not None else ""))
    for entry in value:
        _string(entry, f"{name} entry")
    return value


def _content_ids(value: Any, name: str) -> None:
    entries = _strings(value, name)
    if len(set(entries)) != len(entries):
        raise ValueError(f"{name} content identities must be unique")
    for entry in entries:
        if _CONTENT_ID.fullmatch(entry) is None:
            raise ValueError(f"{name} entries must be sha256 content identities")


def _validate_matrix(matrix: Any, size: int) -> None:
    if not isinstance(matrix, list) or len(matrix) != size:
        raise ValueError("matrix shape must match the ordered quantities")
    for row in matrix:
        if not isinstance(row, list) or len(row) != size:
            raise ValueError("matrix must be square in the declared quantity order")
        for value in row:
            _number(value, "matrix entry")
    diagonal = [float(matrix[index][index]) for index in range(size)]
    if any(value < 0 for value in diagonal):
        raise ValueError("Covariance variances must be nonnegative")
    # A zero-variance coordinate cannot have any covariance with another
    # coordinate. Do not erase small supplied cross terms to make it pass.
    for index, variance in enumerate(diagonal):
        if variance == 0 and any(matrix[index][other] != 0 or matrix[other][index] != 0
                                 for other in range(size)):
            raise ValueError("Zero variance requires an exactly zero covariance row and column")
    active = [index for index, variance in enumerate(diagonal) if variance > 0]
    if not active:
        return
    scales = [math.sqrt(diagonal[index]) for index in active]
    correlation = np.empty((len(active), len(active)), dtype=np.float64)
    for row, source_row in enumerate(active):
        for column, source_column in enumerate(active):
            # Sequential division avoids forming variance products that may
            # underflow/overflow for legitimately different physical scales.
            value = (float(matrix[source_row][source_column]) / scales[row]) / scales[column]
            if not math.isfinite(value):
                raise ValueError("Covariance has nonfinite normalized correlation")
            correlation[row, column] = value
    if not np.allclose(correlation, correlation.T, rtol=0.0, atol=SYMMETRY_ATOL):
        raise ValueError("Covariance must be symmetric in correlation coordinates")
    # Inspect both stored triangles rather than silently trusting one side or
    # averaging them. Tolerance permits floating-point roundoff, never repair.
    try:
        for triangle in ("L", "U"):
            eigenvalues = np.linalg.eigvalsh(correlation, UPLO=triangle)
            if not np.all(np.isfinite(eigenvalues)) or np.min(eigenvalues) < -PSD_ATOL:
                raise ValueError("Covariance must be positive semidefinite in correlation coordinates")
    except np.linalg.LinAlgError as exc:
        raise ValueError("Covariance positive-semidefinite validation did not converge") from exc


def covariance_identity(artifact: dict) -> str:
    """Hash every retained declaration except the identity itself, using CIW's encoder."""
    if not isinstance(artifact, dict):
        raise ValueError("Covariance artifact must be an object")
    return content_identity({key: value for key, value in artifact.items() if key != "covariance_id"})


def validate_covariance_artifact(
    artifact: dict, *, expected_quantity_ids: list[str] | None = None,
    expected_units: list[str] | None = None, expected_frame: str | None = None,
) -> None:
    """Validate schema, content commitment, matrix domain, and optional caller bindings.

    The caller must supply expected axis order/units/frame when binding an
    artifact to a particular model or channel vector. Mathematical validity
    alone cannot establish the truth of those domain declarations.
    """
    if not isinstance(artifact, dict) or artifact.keys() != _FIELDS:
        raise ValueError("Covariance artifact must contain exactly the v1 contract fields")
    if artifact["schema"] != COVARIANCE_SCHEMA:
        raise ValueError("Unsupported covariance artifact schema")
    _json_tree(artifact, "covariance artifact")
    quantities = _strings(artifact["quantity_ids"], "quantity_ids")
    if not quantities or len(set(quantities)) != len(quantities):
        raise ValueError("quantity_ids must contain unique ordered quantities")
    size = len(quantities)
    units = _strings(artifact["units"], "units", count=size)
    _string(artifact["frame"], "frame")
    _string(artifact["method"], "method")
    reference = artifact["reference_values"]
    if not isinstance(reference, list) or len(reference) != size:
        raise ValueError("reference_values must match the declared quantity order")
    for value in reference:
        _number(value, "reference_values entry")
    basis = artifact["basis"]
    if not isinstance(basis, dict) or set(basis) != {"kind", "id"}:
        raise ValueError("basis must declare kind and id")
    if not isinstance(basis["kind"], str) or basis["kind"] not in _BASIS_KINDS:
        raise ValueError("Unsupported covariance basis kind")
    _string(basis["id"], "basis.id")
    provenance = artifact["provenance"]
    required = {"provider", "source_evidence_ids", "source_covariance_ids"}
    if (not isinstance(provenance, dict) or not required <= provenance.keys()
            or provenance.keys() - required - {"metadata"}):
        raise ValueError("provenance requires provider and evidence/covariance source identities")
    _string(provenance["provider"], "provenance.provider")
    _content_ids(provenance["source_evidence_ids"], "provenance.source_evidence_ids")
    _content_ids(provenance["source_covariance_ids"], "provenance.source_covariance_ids")
    if "metadata" in provenance and not isinstance(provenance["metadata"], dict):
        raise ValueError("provenance.metadata must be a JSON object")
    _strings(artifact["assumptions"], "assumptions")
    for actual, expected, name in (
        (quantities, expected_quantity_ids, "quantity order"),
        (units, expected_units, "quantity units"),
        (artifact["frame"], expected_frame, "coordinate frame"),
    ):
        if expected is not None and actual != expected:
            raise ValueError(f"Covariance {name} does not match the caller's declared binding")
    _validate_matrix(artifact["matrix"], size)
    if (not isinstance(artifact["covariance_id"], str)
            or _CONTENT_ID.fullmatch(artifact["covariance_id"]) is None
            or artifact["covariance_id"] != covariance_identity(artifact)):
        raise ValueError("Covariance content identity mismatch")


def create_covariance_artifact(
    *, matrix: list[list[float]], quantity_ids: list[str], units: list[str], frame: str,
    reference_values: list[float], method: str, basis: dict, provenance: dict,
    assumptions: list[str],
) -> dict:
    """Retain an already-computed provider covariance as a detached validated artifact."""
    artifact = deepcopy({
        "schema": COVARIANCE_SCHEMA, "quantity_ids": quantity_ids, "units": units,
        "frame": frame, "reference_values": reference_values, "matrix": matrix,
        "method": method, "basis": basis, "provenance": provenance,
        "assumptions": assumptions,
    })
    # Reject hidden non-JSON values before attempting canonical serialization.
    _json_tree(artifact, "covariance artifact")
    artifact["covariance_id"] = covariance_identity(artifact)
    validate_covariance_artifact(artifact)
    return artifact
