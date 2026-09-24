"""Which OpenBLAS kernel NumPy runs: the one native lookup of the lab outside the energy probes.

Scope: a DYNAMIC_ARCH OpenBLAS, as NumPy bundles it, picks its kernels for the
CPU when it loads, or as ``OPENBLAS_CORETYPE`` forces, and different kernels
round BLAS and LAPACK results differently. NumPy's build configuration names
only the build target, so :func:`openblas_core` asks the loaded library for the
name of the kernel it runs. T094's platform fingerprint and the clean-room gate
(``scripts/reproduce_lab.py --blas-core``) read it here. The module is declared
in the T144 native-loading allowlist and does nothing else.

Non-claims: the name identifies a kernel family, not the CPU or its rounding;
two hosts running the same kernel may still differ in other libraries.
"""
from __future__ import annotations

from pathlib import Path

# Functions through which an OpenBLAS names its kernel: NumPy's scipy-openblas64 (ILP64), scipy-openblas32 and
# an OpenBLAS without the scipy prefix.
OPENBLAS_CORENAME = ("scipy_openblas_get_corename64_", "scipy_openblas_get_corename", "openblas_get_corename64_",
                     "openblas_get_corename")


def openblas_core() -> str | None:
    """The OpenBLAS kernel NumPy's bundled OpenBLAS runs (``SkylakeX``, ``Haswell``, ...), or None.

    The name is read from the library NumPy loaded (``numpy.libs`` on Linux and
    Windows, ``numpy/.dylibs`` on macOS). None where NumPy bundles no OpenBLAS
    or the library does not name its kernel.
    """
    import ctypes
    import numpy as np
    package = Path(np.__file__).parent
    for directory in (package.parent / "numpy.libs", package / ".dylibs"):
        for path in sorted(directory.glob("*openblas*")) if directory.is_dir() else ():
            try:
                library = ctypes.CDLL(str(path))  # the library NumPy loaded: same handle, same kernel
            except OSError:
                continue
            for symbol in OPENBLAS_CORENAME:
                function = getattr(library, symbol, None)
                if function is not None:
                    function.argtypes, function.restype = [], ctypes.c_char_p
                    name = (function() or b"").decode("ascii", "replace").strip()
                    return name or None
    return None
