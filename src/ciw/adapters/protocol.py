"""Versioned declarative adapter seam; manifests are data, never import instructions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core.records import finite_tree, mapping, string


class AdapterRefusal(ValueError):
    """A declared inability to calculate, distinct from an estimate or verification."""

    def __init__(self, code: str, message: str, *, reason_code: str | None = None):
        super().__init__(message)
        self.code = string(code, "refusal code")
        self.reason_code = None if reason_code is None else string(reason_code, "refusal reason code")

    def to_dict(self) -> dict:
        record = {"code": self.code, "message": str(self)}
        if self.reason_code is not None:
            record["reason_code"] = self.reason_code
        return record


@dataclass(frozen=True)
class InstrumentManifest:
    """Scientific interface declaration for instruments and other adapter roles.

    The historical name does not make calibration adapters, operation providers,
    or record-only readers into instruments: ``role`` declares that distinction.
    """

    instrument_id: str
    version: str = "1"
    role: str = "instrument"
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ("run.v1",)
    units: dict[str, str] = field(default_factory=dict)
    frames: tuple[str, ...] = ()
    sampling: dict[str, Any] = field(default_factory=dict)
    normalization: dict[str, Any] = field(default_factory=dict)
    supported_operations: tuple[str, ...] = ()
    determinism: dict[str, Any] = field(default_factory=dict)
    tolerance_policy: dict[str, Any] = field(default_factory=dict)
    calibration_requirements: dict[str, Any] = field(default_factory=dict)
    manifest_schema: str = "ciw.instrument-manifest.v1"

    def __post_init__(self) -> None:
        if self.manifest_schema != "ciw.instrument-manifest.v1":
            raise ValueError("Unsupported instrument manifest schema")
        for key in ("instrument_id", "version", "role"):
            string(getattr(self, key), f"manifest.{key}")
        for key in ("inputs", "outputs", "frames", "supported_operations"):
            values = getattr(self, key)
            if not isinstance(values, (tuple, list)):
                raise ValueError(f"manifest.{key} must be an array of strings")
            for value in values:
                string(value, f"manifest.{key} entry")
            if len(set(values)) != len(values):
                raise ValueError(f"manifest.{key} entries must be unique")
            object.__setattr__(self, key, tuple(values))
        for key in ("units", "sampling", "normalization", "determinism", "tolerance_policy",
                    "calibration_requirements"):
            value = mapping(getattr(self, key), f"manifest.{key}")
            finite_tree(value, f"manifest.{key}")
            object.__setattr__(self, key, deepcopy(value))
        for channel, unit in self.units.items():
            string(channel, "manifest unit channel")
            string(unit, "manifest unit")

    def to_dict(self) -> dict:
        value = asdict(self)
        for key in ("inputs", "outputs", "frames", "supported_operations"):
            value[key] = list(value[key])
        return value

    @classmethod
    def from_dict(cls, value: dict) -> InstrumentManifest:
        mapping(value, "manifest")
        fields = set(cls.__dataclass_fields__)
        if value.keys() != fields:
            raise ValueError("Saved manifest must declare all v1 contract fields and no unknown fields")
        return cls(**deepcopy(value))


@runtime_checkable
class Adapter(Protocol):
    manifest: InstrumentManifest

    def validate_run(self, run: dict) -> None:
        """Validate domain validity without mutating or executing the recording."""
        ...

    def execute(self, operation_id: str, run: dict, parameters: dict) -> dict:
        """Compute from full-resolution evidence or raise AdapterRefusal."""
        ...
