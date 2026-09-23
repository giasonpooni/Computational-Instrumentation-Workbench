"""Read-only NVML telemetry, with raw counters and explicit host call brackets.

Energy is the vendor-reported GPU-device counter, not process-attributed or
whole-machine energy. This reader neither launches work nor infers joules from
time/utilization. Counter resets, repeated values and decreases remain raw data
for the retained-log validator. Optional context is sampled AFTER the energy
call's brackets; its observations do not share an asserted sensor timestamp.

API semantics: https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html
"""
from __future__ import annotations

import ctypes as ct
from ctypes.util import find_library
from hashlib import sha256
import os
from pathlib import Path
import re
from sys import platform as _PLATFORM
from threading import RLock
from time import perf_counter_ns, time_ns
from types import MappingProxyType

_HANDLE = ct.c_void_p
_UINT = ct.c_uint
_U64 = ct.c_ulonglong
_FUNCTION_NOT_FOUND = 13
_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z")


class NVMLUnavailable(RuntimeError):
    """Unavailable initialization/identity, or an explicit lifecycle failure."""

    def __init__(self, operation: str, error_code: int | None = None, detail: str = ""):
        self.operation, self.error_code = operation, error_code
        message = f"NVML {operation} unavailable"
        if error_code is not None:
            message += f" (code {error_code})"
        if detail:
            message += ": " + detail
        super().__init__(message)


def _function(library, name, arguments, *, required=True):
    try:
        function = getattr(library, name)
    except AttributeError as exc:
        if required:
            raise NVMLUnavailable(name, _FUNCTION_NOT_FOUND) from exc
        return None
    function.argtypes, function.restype = arguments, ct.c_int
    return function


def _windows_library_path():
    # The trusted Windows system directory comes from the OS, not a record,
    # environment path override or the caller's current working directory.
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    get_directory = kernel.GetSystemDirectoryW
    get_directory.argtypes, get_directory.restype = [ct.c_wchar_p, _UINT], _UINT
    buffer = ct.create_unicode_buffer(32768)
    count = get_directory(buffer, len(buffer))
    if count == 0 or count >= len(buffer):
        raise NVMLUnavailable("system_library_path", detail="Cannot resolve the Windows system directory")
    return Path(buffer.value) / "nvml.dll"


def _linux_library_path(library):
    class DlInfo(ct.Structure):
        _fields_ = [("name", ct.c_char_p), ("base", ct.c_void_p),
                    ("symbol_name", ct.c_char_p), ("symbol", ct.c_void_p)]

    # Resolve the already-loaded NVML symbol, so the hash covers the actual
    # loaded system library rather than another file with the same basename.
    dl = ct.CDLL(find_library("dl") or None)
    address = dl.dladdr
    address.argtypes, address.restype = [ct.c_void_p, ct.POINTER(DlInfo)], ct.c_int
    info = DlInfo()
    if not address(ct.cast(library.nvmlInit_v2, ct.c_void_p), ct.byref(info)) or not info.name:
        raise NVMLUnavailable("library_identity", detail="Cannot resolve the loaded NVML library")
    return Path(os.fsdecode(info.name)).resolve(strict=True)


def _load_nvml():
    """Only platform NVML discovery; retained data can never choose a library."""
    try:
        if _PLATFORM == "win32":
            path = _windows_library_path().resolve(strict=True)
            return ct.CDLL(str(path)), path
        if _PLATFORM.startswith("linux"):
            name = find_library("nvidia-ml")
            if not name:
                raise NVMLUnavailable("library_load", 12)
            library = ct.CDLL(name)
            return library, _linux_library_path(library)
        raise NVMLUnavailable("platform", detail="Only Windows system NVML and Linux NVML are supported")
    except (OSError, AttributeError) as exc:
        raise NVMLUnavailable("library_load", 12, str(exc)) from exc


class NVMLEnergyCounter:
    """An opened GPU counter. Use as a context manager or call close().

    identity() returns a detached copy of an immutable opening descriptor.
    read() retains failed energy reads as null with the exact NVML return code.
    The caller owns GPU synchronization, trial boundaries and energy derivation.
    """

    def __init__(self, device_index=0, uuid=None):
        if type(device_index) is not int or not 0 <= device_index < 2**32:
            raise ValueError("device_index must be an unsigned 32-bit integer")
        if uuid is not None and (type(uuid) is not str or not _UUID.fullmatch(uuid)):
            raise ValueError("uuid must be a physical GPU UUID")
        self._lock, self._closed = RLock(), True
        self._library, path = _load_nvml()
        try:
            library_digest = sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise NVMLUnavailable("library_identity", detail="Cannot hash the loaded NVML library") from exc
        initialize = _function(self._library, "nvmlInit_v2", [])
        self._shutdown = _function(self._library, "nvmlShutdown", [])
        self._require(initialize(), "nvmlInit_v2")
        self._closed = False
        try:
            self._handle = _HANDLE()
            if uuid is None:
                select = _function(self._library, "nvmlDeviceGetHandleByIndex_v2", [_UINT, ct.POINTER(_HANDLE)])
                self._require(select(device_index, ct.byref(self._handle)), "nvmlDeviceGetHandleByIndex_v2")
            else:
                select = _function(self._library, "nvmlDeviceGetHandleByUUID", [ct.c_char_p, ct.POINTER(_HANDLE)])
                self._require(select(uuid.encode("ascii"), ct.byref(self._handle)), "nvmlDeviceGetHandleByUUID")
            reported_uuid = self._text("nvmlDeviceGetUUID", device=True)
            if not _UUID.fullmatch(reported_uuid) or (uuid is not None and uuid.lower() != reported_uuid.lower()):
                raise NVMLUnavailable("device_identity", detail="Selected device UUID differs")
            descriptor = {"backend":"nvml", "device_uuid":reported_uuid,
                "name":self._text("nvmlDeviceGetName", device=True),
                "driver_version":self._text("nvmlSystemGetDriverVersion"),
                "nvml_version":self._text("nvmlSystemGetNVMLVersion"),
                "library_sha256":library_digest, "counter_unit":"mJ", "counter_scope":"gpu_device",
                "power_unit":"mW", "timestamp_semantics":"host_call_brackets",
                "update_interval_s":None, "resolution_j":None, "accuracy_j":None,
                "background_inclusive":True}
            self._identity = MappingProxyType(descriptor)
            self._energy = _function(self._library, "nvmlDeviceGetTotalEnergyConsumption", [_HANDLE, ct.POINTER(_U64)])
            self._context = {
                "power_mw":(_function(self._library, "nvmlDeviceGetPowerUsage", [_HANDLE, ct.POINTER(_UINT)], required=False), ()),
                "temperature_c":(_function(self._library, "nvmlDeviceGetTemperature", [_HANDLE, _UINT, ct.POINTER(_UINT)], required=False), (0,)),
                "graphics_clock_mhz":(_function(self._library, "nvmlDeviceGetClockInfo", [_HANDLE, _UINT, ct.POINTER(_UINT)], required=False), (0,)),
            }
        except BaseException:
            # Match only successful initialization. Preserve the primary
            # selection/identity failure if cleanup reports another error.
            try:
                self.close()
            except NVMLUnavailable:
                pass
            raise

    @staticmethod
    def _require(code, operation):
        if code != 0:
            raise NVMLUnavailable(operation, int(code))

    def _text(self, name, *, device=False):
        arguments = ([_HANDLE] if device else []) + [ct.c_char_p, _UINT]
        function = _function(self._library, name, arguments)
        buffer = ct.create_string_buffer(96)
        prefix = (self._handle,) if device else ()
        self._require(function(*prefix, buffer, len(buffer)), name)
        try:
            value = buffer.value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise NVMLUnavailable(name, detail="Invalid NVML identity text") from exc
        if not value or len(value.encode("utf-8")) >= len(buffer):
            raise NVMLUnavailable(name, detail="Invalid NVML identity text")
        return value

    def identity(self):
        return dict(self._identity)

    def read(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("NVML counter is closed")
            energy = _U64()
            pointer = ct.byref(energy)
            utc_start = time_ns()
            start = perf_counter_ns()
            code = int(self._energy(self._handle, pointer))
            end = perf_counter_ns()
            utc_end = time_ns()
            result = {"read_start_ns":start, "read_end_ns":end,
                "utc_start_ns":utc_start, "utc_end_ns":utc_end,
                "status":"ok" if code == 0 else "error", "energy_mj":int(energy.value) if code == 0 else None,
                "error_code":None if code == 0 else code,
                "power_mw":None, "temperature_c":None, "graphics_clock_mhz":None, "context_errors":{}}
            for key, (function, arguments) in self._context.items():
                value = _UINT()
                context_code = _FUNCTION_NOT_FOUND if function is None else int(function(self._handle, *arguments, ct.byref(value)))
                if context_code == 0:
                    result[key] = int(value.value)
                else:
                    result["context_errors"][key] = context_code
            return result

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._require(self._shutdown(), "nvmlShutdown")

    def __enter__(self):
        if self._closed:
            raise RuntimeError("NVML counter is closed")
        return self

    def __exit__(self, exception_type, exception, traceback):
        try:
            self.close()
        except NVMLUnavailable:
            if exception_type is None:
                raise
        return False
