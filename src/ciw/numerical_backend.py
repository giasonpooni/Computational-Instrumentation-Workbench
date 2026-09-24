"""Best-effort, human-readable name of the linear-algebra backend NumPy loaded.

The kernel probe in ``reference_workflow`` is what enters a runtime identity;
this label is informational. It reads the OpenBLAS core name and build line
from the library NumPy's wheel bundles, through the symbols those wheels
export, and reports ``unknown`` when the backend is something else or cannot
be read. It never influences validation, identity or replay, so it lives
outside the files a reference hashes into its identity.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import glob
import os
from pathlib import Path

import numpy as np

UNKNOWN = "unknown"
LIMIT = 64
_CORENAME_SYMBOLS = ("scipy_openblas_get_corename64_", "scipy_openblas_get_corename",
                     "openblas_get_corename64_", "openblas_get_corename")
_CONFIG_SYMBOLS = ("scipy_openblas_get_config64_", "scipy_openblas_get_config",
                   "openblas_get_config64_", "openblas_get_config")


def _candidate_libraries():
    package = Path(np.__file__).resolve().parent
    patterns = [package.parent / "numpy.libs" / "*openblas*", package / ".libs" / "*openblas*",
                package / ".dylibs" / "*openblas*"]
    found = []
    for pattern in patterns:
        found.extend(sorted(glob.glob(str(pattern))))
    maps = Path("/proc/self/maps")
    if maps.exists():
        try:
            for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
                path = line.split(maxsplit=5)[-1] if len(line.split()) >= 6 else ""
                if "openblas" in os.path.basename(path).lower() and path not in found:
                    found.append(path)
        except OSError:
            pass
    return found


def _read(library, symbols):
    for name in symbols:
        function = getattr(library, name, None)
        if function is not None:
            function.restype = ctypes.c_char_p
            value = function()
            if value:
                return value.decode("utf-8", errors="replace").strip()
    return None


def _describe():
    for path in _candidate_libraries():
        try:
            library = ctypes.CDLL(path)
        except OSError:
            continue
        core = _read(library, _CORENAME_SYMBOLS)
        config = _read(library, _CONFIG_SYMBOLS)
        if core or config:
            version = (config or "").split()[1] if config and len(config.split()) > 1 and config.startswith("OpenBLAS") else None
            parts = ["OpenBLAS"] + ([version] if version else []) + ([core] if core else [])
            return " ".join(parts)
    return UNKNOWN


@lru_cache(maxsize=1)
def linear_algebra_backend():
    """``OpenBLAS <version> <core>`` when readable, otherwise ``unknown``; at most 64 characters."""
    try:
        label = _describe()
    except Exception:  # noqa: BLE001 - informational only; never let introspection fail a report
        label = UNKNOWN
    return label[:LIMIT] if isinstance(label, str) and label.strip() else UNKNOWN
