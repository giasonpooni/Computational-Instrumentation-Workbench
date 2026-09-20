"""Scientific adapters declare their contracts without transferring domain authority."""

from .protocol import Adapter, AdapterRefusal, InstrumentManifest
from .registry import AdapterRegistry, RecordAdapter, default_registry

__all__ = ["Adapter", "AdapterRefusal", "InstrumentManifest", "AdapterRegistry",
           "RecordAdapter", "default_registry"]
