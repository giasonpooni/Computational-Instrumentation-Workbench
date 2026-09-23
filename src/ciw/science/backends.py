"""Execution backends, runtime identity and per-job execution telemetry.

Numerical results can depend on precision, compiler, library versions, BLAS,
reduction order, random seed and hardware. Every execution therefore records
the backend chosen, why others were not, and a runtime identity that includes
a digest of the solver source files, so a later replay can detect provider
drift instead of assuming it away.

Only ``python-numpy-cpu`` binds solvers in this build. Julia, Rust, C++, CUDA,
FPGA and remote backends are *detected and described*; a backend without bound
solvers is never selected, and detection never executes user-supplied code.
Energy is recorded only when a counter was actually read; otherwise the record
says ``not_measured`` rather than estimating joules from time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np

from ._common import Refusal, plain

NATIVE_BACKEND = "python-numpy-cpu"
_SOURCE_FILES = ("geometry.py", "solvers.py", "oracles.py", "observation.py", "units.py", "frames.py")


@dataclass(frozen=True)
class Backend:
    backend_id: str
    language: str
    device: str
    precision: tuple[str, ...]
    available: bool
    bound_solvers: tuple[str, ...] = ()
    detail: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return plain({"backend_id": self.backend_id, "language": self.language, "device": self.device,
                      "precision": list(self.precision), "available": self.available,
                      "bound_solvers": list(self.bound_solvers), "detail": self.detail})


def source_digest(names: tuple[str, ...] = _SOURCE_FILES) -> str:
    """Digest of the scientific source files that implement native solvers."""
    root = Path(__file__).resolve().parent
    hasher = hashlib.sha256()
    for name in sorted(names):
        path = root / name
        hasher.update(name.encode() + b"\0")
        hasher.update(path.read_bytes() if path.is_file() else b"<absent>")
    return "sha256:" + hasher.hexdigest()


def runtime_identity(solver_id: str | None = None, *, seed: int | None = None) -> dict:
    """The facts a numerical result may depend on, captured before execution."""
    identity = {
        "provider": "ciw.science", "backend": NATIVE_BACKEND, "source_digest": source_digest(),
        "python": platform.python_version(), "python_implementation": platform.python_implementation(),
        "numpy": np.__version__, "machine": platform.machine(), "system": platform.system(),
        "byteorder": sys.byteorder, "float": {"format": "binary64", "epsilon": float(np.finfo(float).eps)},
        "reduction_order": "sequential numpy per-step evaluation; no parallel reductions in native solvers",
        "seed": seed,
    }
    if solver_id is not None:
        identity["solver_id"] = solver_id
    return identity


def drift(original: dict, replay: dict, ignore: tuple[str, ...] = ()) -> list[dict]:
    """Keys whose values differ between two runtime identities."""
    keys = sorted((set(original) | set(replay)) - set(ignore))
    return [{"key": key, "original": original.get(key), "replay": replay.get(key)}
            for key in keys if original.get(key) != replay.get(key)]


def _tool_version(executable: str, arguments: list[str]) -> dict:
    path = shutil.which(executable)
    if path is None:
        return {"available": False, "reason": f"{executable} not found on PATH"}
    try:
        completed = subprocess.run([path, *arguments], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"{executable} could not report a version: {exc}", "path": path}
    output = (completed.stdout or completed.stderr).strip().splitlines()
    return {"available": completed.returncode == 0, "path": path, "version": output[0][:200] if output else None}


def _gpu() -> dict:
    try:
        from ..energy_nvml import NVMLEnergyCounter, NVMLUnavailable
    except ImportError as exc:  # pragma: no cover - module ships with the workbench
        return {"available": False, "reason": str(exc)}
    try:
        with NVMLEnergyCounter(device_index=0) as counter:
            identity = counter.identity()
        return {"available": True, "device": identity.get("name"), "device_uuid": identity.get("device_uuid"),
                "driver_version": identity.get("driver_version"), "energy_counter": "nvml gpu_device mJ"}
    except (NVMLUnavailable, OSError, ValueError, RuntimeError) as exc:
        return {"available": False, "reason": f"NVML unavailable: {exc}"}


def detect(probe_tools: bool = False) -> list[Backend]:
    """Describe candidate backends. ``probe_tools`` runs ``--version`` on toolchains found on PATH."""
    from .solvers import default_registry
    solvers = tuple(sorted(default_registry().ids()))
    backends = [Backend(NATIVE_BACKEND, "python", "cpu", ("binary64",), True, solvers,
                        {"python": platform.python_version(), "numpy": np.__version__, "cpus": os.cpu_count()})]
    toolchains = [("julia-cpu", "julia", "julia", ["--version"]), ("rust-cpu", "rust", "rustc", ["--version"]),
                  ("cxx-cpu", "c++", "c++", ["--version"])]
    for backend_id, language, executable, arguments in toolchains:
        detail = (_tool_version(executable, arguments) if probe_tools
                  else {"available": shutil.which(executable) is not None, "path": shutil.which(executable)})
        backends.append(Backend(backend_id, language, "cpu", ("binary64",), bool(detail.get("available")), (),
                                {**detail, "note": "toolchain detected only; no solver is bound to it in this build"}))
    gpu = _gpu() if probe_tools else {"available": shutil.which("nvidia-smi") is not None,
                                       "reason": "not probed; run with probe to open NVML"}
    backends.append(Backend("cuda-gpu", "cuda", "gpu", ("binary32", "binary64"), bool(gpu.get("available")), (),
                            {**gpu, "note": "GPU energy capture is available through `ciw energy`; no science solver "
                             "is bound to the GPU in this build"}))
    backends.append(Backend("fpga", "hdl", "fpga", (), False, (),
                            {"reason": "no FPGA board is selected; hardware.py is an observation-only boundary"}))
    backends.append(Backend("remote", "any", "remote", (), False, (),
                            {"reason": "no remote execution service is configured; offline operation is the default"}))
    return backends


def select(solver_id: str, preferred: list[str] | None = None, precision: str = "binary64",
           backends: list[Backend] | None = None) -> dict:
    """Choose a backend that is available, supports the precision and binds the solver."""
    candidates = backends if backends is not None else detect()
    order = list(preferred or []) + [item.backend_id for item in candidates if item.backend_id not in (preferred or [])]
    by_id = {item.backend_id: item for item in candidates}
    rejected = []
    for backend_id in order:
        item = by_id.get(backend_id)
        if item is None:
            rejected.append({"backend_id": backend_id, "reason": "not declared"})
        elif not item.available:
            rejected.append({"backend_id": backend_id, "reason": "unavailable"})
        elif precision not in item.precision:
            rejected.append({"backend_id": backend_id, "reason": f"does not provide {precision}"})
        elif solver_id not in item.bound_solvers:
            rejected.append({"backend_id": backend_id, "reason": f"no binding for {solver_id}"})
        else:
            return {"backend_id": backend_id, "precision": precision, "rejected": rejected}
    raise Refusal("no_backend", f"No available backend binds {solver_id} at {precision}", rejected=rejected)


def measured(function: Callable[[], Any]) -> tuple[Any, dict]:
    """Run a callable and record wall and process CPU time; energy is explicitly not measured."""
    wall, cpu = time.perf_counter_ns(), time.process_time_ns()
    value = function()
    return value, {"wall_time_s": (time.perf_counter_ns() - wall) / 1e9,
                   "cpu_time_s": (time.process_time_ns() - cpu) / 1e9,
                   "energy": {"status": "not_measured", "reason": "no energy counter bound to CPU execution"}}
