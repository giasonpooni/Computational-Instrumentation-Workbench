"""Explicit adapter registry and offline structural reader for retained evidence."""

from __future__ import annotations

from copy import deepcopy
import threading

from ..core.records import validate_run_structure
from .protocol import Adapter, AdapterRefusal, InstrumentManifest


class RecordAdapter:
    """Read retained runs from a declared contract without loading a scientific engine.

    This checks recording shape and declared units/frames, not physical validity
    or fresh calibration. It cannot evaluate operations or confer verification.
    """

    def __init__(self, manifest: InstrumentManifest):
        self.manifest = InstrumentManifest.from_dict(manifest.to_dict())

    def validate_run(self, run: dict) -> None:
        validate_run_structure(run)
        if run["instrument"] != self.manifest.instrument_id:
            raise ValueError("Run instrument does not match adapter manifest")
        if self.manifest.frames and run["metadata"]["coordinate_frame"] not in self.manifest.frames:
            raise ValueError("Run frame is not declared by adapter manifest")
        for name, channel in run["channels"].items():
            expected = self.manifest.units.get(name, self.manifest.units.get("*"))
            if expected is None:
                raise ValueError(f"Channel {name} has no declared adapter unit mapping")
            if expected != channel["unit"]:
                raise ValueError(f"Channel {name} unit does not match adapter manifest")

    def execute(self, operation_id: str, run: dict, parameters: dict) -> dict:
        raise AdapterRefusal("adapter_unavailable", "Retained evidence is readable; its execution adapter is not bound")


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, Adapter] = {}
        self._lock = threading.RLock()

    def register(self, adapter: Adapter) -> None:
        if not isinstance(adapter, Adapter) or not isinstance(adapter.manifest, InstrumentManifest):
            raise TypeError("Adapters require a validated manifest, validate_run, and execute")
        instrument_id = adapter.manifest.instrument_id
        with self._lock:
            if instrument_id in self._adapters:
                raise ValueError(f"Adapter already registered: {instrument_id}")
            self._adapters[instrument_id] = adapter

    def get(self, instrument_id: str) -> Adapter:
        with self._lock:
            try:
                return self._adapters[instrument_id]
            except KeyError as exc:
                raise AdapterRefusal("adapter_unavailable", f"No adapter registered for {instrument_id}") from exc

    def manifests(self) -> list[dict]:
        with self._lock:
            return [adapter.manifest.to_dict() for adapter in self._adapters.values()]

    def resolve(self, run: dict) -> Adapter:
        validate_run_structure(run)
        with self._lock:
            adapter = self._adapters.get(run["instrument"])
        declared = run["metadata"].get("manifest")
        if declared is not None:
            manifest = InstrumentManifest.from_dict(declared)
            if manifest.instrument_id != run["instrument"]:
                raise ValueError("Saved manifest instrument does not match run")
            if adapter is not None and manifest.to_dict() != adapter.manifest.to_dict():
                raise ValueError("Saved manifest does not match the registered adapter contract")
        elif adapter is None:
            raise AdapterRefusal("adapter_unavailable", "Unknown recordings require an embedded adapter manifest")
        return adapter if adapter is not None else RecordAdapter(manifest)

    def validate_run(self, run: dict) -> None:
        self.resolve(run).validate_run(run)

    def execute(self, operation_id: str, run: dict, parameters: dict | None = None) -> dict:
        adapter = self.resolve(run)
        adapter.validate_run(run)
        if operation_id not in adapter.manifest.supported_operations:
            raise AdapterRefusal("unsupported_operation", f"Adapter does not support {operation_id}")
        return adapter.execute(operation_id, deepcopy(run), deepcopy(parameters or {}))


_DEFAULT: AdapterRegistry | None = None
_DEFAULT_LOCK = threading.Lock()


def default_registry() -> AdapterRegistry:
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            from .oscillator import OscillatorAdapter
            from .julia_oscillator_run import JuliaOscillatorRunAdapter
            registry = AdapterRegistry()
            registry.register(OscillatorAdapter())
            registry.register(JuliaOscillatorRunAdapter())
            _DEFAULT = registry
        return _DEFAULT
