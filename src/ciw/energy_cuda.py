"""Bounded binary64 Gaussian VI on a real CUDA device, without a toolkit.

Construction prepares information parameters on the CPU and JIT-loads fixed PTX;
each solve restarts identical replicas, iterates on the GPU, synchronizes and
copies the final means/covariances to the host. This is neither an energy meter
nor a CPU-energy estimate. Reference conditioning and accuracy scoring belong
outside a measured solve window. Contexts, buffers and modules are worker-owned.

ABI references (explicit legacy _v2 symbols avoid newer context ABI changes):
https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__CTX.html
https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__EXEC.html
https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import ctypes as ct
from hashlib import sha256
import json
import os
import sys
from threading import RLock
import uuid

import numpy as np

from . import free_energy_math as mathematics

MAX_REPLICAS = 65536
MAX_ITERATIONS = 256
THREADS_PER_BLOCK = 128
OUTPUT_LAYOUT = ["mean_0", "mean_1", "covariance_00", "covariance_01", "covariance_10", "covariance_11"]

# Explicit .rn operations prevent a JIT from silently contracting multiply/add
# into FMA. Each thread independently performs every declared update. The final
# 2x2 inverse uses all four iterated precision entries, not a reference answer.
PTX = r""".version 6.0
.target sm_50
.address_size 64
.visible .entry gaussian_vi(
    .param .u64 inputs, .param .u64 outputs,
    .param .u32 iterations, .param .u32 replicas)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<5>;
    .reg .f64 %d<32>;
    ld.param.u64 %rd0, [inputs];
    ld.param.u64 %rd1, [outputs];
    ld.param.u32 %r0, [iterations];
    ld.param.u32 %r1, [replicas];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p0, %r5, %r1;
    @%p0 bra DONE;
    ld.global.f64 %d0, [%rd0+0];
    ld.global.f64 %d1, [%rd0+8];
    ld.global.f64 %d2, [%rd0+16];
    ld.global.f64 %d3, [%rd0+24];
    ld.global.f64 %d4, [%rd0+32];
    ld.global.f64 %d5, [%rd0+40];
    ld.global.f64 %d6, [%rd0+48];
    ld.global.f64 %d7, [%rd0+56];
    ld.global.f64 %d8, [%rd0+64];
    ld.global.f64 %d9, [%rd0+72];
    ld.global.f64 %d10, [%rd0+80];
    ld.global.f64 %d11, [%rd0+88];
    ld.global.f64 %d12, [%rd0+96];
    ld.global.f64 %d13, [%rd0+104];
    ld.global.f64 %d14, [%rd0+112];
    mov.u32 %r6, 0;
LOOP:
    setp.ge.u32 %p1, %r6, %r0;
    @%p1 bra OUTPUT;
    mul.rn.f64 %d15, %d0, %d6;
    mul.rn.f64 %d16, %d1, %d7;
    add.rn.f64 %d15, %d15, %d16;
    sub.rn.f64 %d15, %d15, %d4;
    mul.rn.f64 %d17, %d2, %d6;
    mul.rn.f64 %d18, %d3, %d7;
    add.rn.f64 %d17, %d17, %d18;
    sub.rn.f64 %d17, %d17, %d5;
    mul.rn.f64 %d15, %d12, %d15;
    mul.rn.f64 %d17, %d12, %d17;
    sub.rn.f64 %d6, %d6, %d15;
    sub.rn.f64 %d7, %d7, %d17;
    mul.rn.f64 %d15, %d14, %d8;
    mul.rn.f64 %d16, %d13, %d0;
    add.rn.f64 %d8, %d15, %d16;
    mul.rn.f64 %d15, %d14, %d9;
    mul.rn.f64 %d16, %d13, %d1;
    add.rn.f64 %d9, %d15, %d16;
    mul.rn.f64 %d15, %d14, %d10;
    mul.rn.f64 %d16, %d13, %d2;
    add.rn.f64 %d10, %d15, %d16;
    mul.rn.f64 %d15, %d14, %d11;
    mul.rn.f64 %d16, %d13, %d3;
    add.rn.f64 %d11, %d15, %d16;
    add.u32 %r6, %r6, 1;
    bra LOOP;
OUTPUT:
    mul.rn.f64 %d15, %d8, %d11;
    mul.rn.f64 %d16, %d9, %d10;
    sub.rn.f64 %d15, %d15, %d16;
    div.rn.f64 %d20, %d11, %d15;
    neg.f64 %d21, %d9;
    div.rn.f64 %d21, %d21, %d15;
    neg.f64 %d22, %d10;
    div.rn.f64 %d22, %d22, %d15;
    div.rn.f64 %d23, %d8, %d15;
    mul.wide.u32 %rd2, %r5, 48;
    add.u64 %rd3, %rd1, %rd2;
    st.global.f64 [%rd3+0], %d6;
    st.global.f64 [%rd3+8], %d7;
    st.global.f64 [%rd3+16], %d20;
    st.global.f64 [%rd3+24], %d21;
    st.global.f64 [%rd3+32], %d22;
    st.global.f64 [%rd3+40], %d23;
DONE:
    ret;
}
"""


class CudaError(RuntimeError):
    """A CUDA driver failure; there is no silent CPU fallback."""


class _Driver:
    def __init__(self):
        if ct.sizeof(ct.c_void_p) != 8 or sys.byteorder != "little":
            raise CudaError("The CUDA worker requires a 64-bit little-endian host")
        try:
            self.library = (ct.WinDLL("nvcuda.dll", winmode=0x00000800) if os.name == "nt"
                            else ct.CDLL("libcuda.so.1"))
        except OSError as exc:
            raise CudaError("CUDA driver library is unavailable; no GPU solve was performed") from exc
        pointer, integer, size = ct.c_void_p, ct.c_int, ct.c_size_t
        p_int, p_void, deviceptr = ct.POINTER(integer), ct.POINTER(pointer), ct.c_uint64
        signatures = {
            "cuInit": [ct.c_uint], "cuDriverGetVersion": [p_int], "cuDeviceGetCount": [p_int],
            "cuDeviceGet": [p_int, integer], "cuDeviceGetName": [pointer, integer, integer],
            "cuDeviceGetUuid": [pointer, integer], "cuDeviceGetAttribute": [p_int, integer, integer],
            "cuCtxCreate_v2": [p_void, ct.c_uint, integer], "cuCtxDestroy_v2": [pointer],
            "cuCtxPushCurrent_v2": [pointer], "cuCtxPopCurrent_v2": [p_void], "cuCtxSynchronize": [],
            "cuModuleLoadData": [p_void, pointer], "cuModuleGetFunction": [p_void, pointer, ct.c_char_p],
            "cuMemAlloc_v2": [ct.POINTER(deviceptr), size],
            "cuMemcpyHtoD_v2": [deviceptr, pointer, size], "cuMemcpyDtoH_v2": [pointer, deviceptr, size],
            "cuLaunchKernel": [pointer, *([ct.c_uint] * 7), pointer, p_void, p_void],
            "cuGetErrorName": [integer, ct.POINTER(ct.c_char_p)],
            "cuGetErrorString": [integer, ct.POINTER(ct.c_char_p)],
        }
        for name, arguments in signatures.items():
            try:
                function = getattr(self.library, name)
            except AttributeError as exc:
                raise CudaError("CUDA driver lacks required ABI: " + name) from exc
            function.argtypes, function.restype = arguments, integer
            setattr(self, name, function)
        self.call("cuInit", 0)

    def call(self, name, *arguments):
        code = getattr(self, name)(*arguments)
        if code:
            error, detail = ct.c_char_p(), ct.c_char_p()
            self.cuGetErrorName(code, ct.byref(error))
            self.cuGetErrorString(code, ct.byref(detail))
            message = (detail.value or b"unknown error").decode("utf-8", errors="replace")
            label = (error.value or b"CUDA_ERROR").decode("ascii", errors="replace")
            raise CudaError(f"{name}: {label} ({code}): {message}")


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
    return value


def _prepare(problem, settings, iterations, replicas, device_index):
    _integer(iterations, "iterations", 0, MAX_ITERATIONS)
    _integer(replicas, "replicas", 1, MAX_REPLICAS)
    _integer(device_index, "device_index", 0, 63)
    required = {"initial_mean", "initial_covariance", "alpha", "beta"}
    optional = {"max_iterations", "gradient_tolerance", "precision_tolerance"}
    if type(settings) is not dict or not required <= set(settings) <= required | optional:
        raise ValueError("Require the declared initial state and Gaussian VI step settings")
    if "max_iterations" in settings:
        maximum = _integer(settings["max_iterations"], "max_iterations", 1, mathematics.MAX_ITERATIONS)
        if iterations > maximum:
            raise ValueError("Fixed GPU iterations exceed the declared solver budget")
    for key in ("gradient_tolerance", "precision_tolerance"):
        if key in settings and not 1e-12 <= mathematics._scalar(settings[key], key) <= 1e-3:
            raise ValueError("Unsupported declared convergence tolerance")
    initial = mathematics._array(settings["initial_mean"], (2,), "initial_mean")
    covariance = mathematics._covariance(settings["initial_covariance"], "initial_covariance", input_matrix=True)
    q = mathematics._positive(np.linalg.solve(covariance, np.eye(2)), "initial_precision")
    alpha, beta = (mathematics._scalar(settings[key], key) for key in ("alpha", "beta"))
    if not 0 < alpha <= 1e6 or not 0 < beta < 1:
        raise ValueError("Require alpha in (0,1e6] and beta in (0,1)")
    information = mathematics.information_system(problem)
    values = [*np.asarray(information["precision"]).ravel(), *information["information_vector"],
              *initial, *q.ravel(), alpha, beta, 1 - beta]
    return np.ascontiguousarray(values, dtype=np.float64)


class CudaGaussianWorker:
    """Persistent GPU allocations; each solve produces a fresh fixed-step batch.

    A private CUDA context is pushed only for this worker's operations and then
    popped, preserving the caller's context. Calls on one worker are serialized.
    No reference solve, accuracy metric or driver loading occurs inside solve().
    Explicit close (or the context manager) releases all owned GPU resources.
    """

    def __init__(self, problem, solver_settings, iterations, replicas=4096, device_index=0):
        values = _prepare(problem, solver_settings, iterations, replicas, device_index)
        self._lock, self._closed = RLock(), True
        self._context, self._module, self._function = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
        self._input, self._output = ct.c_uint64(), ct.c_uint64()
        self._iterations, self._replicas = iterations, replicas
        self._driver = driver = _Driver()
        count, device, version = ct.c_int(), ct.c_int(), ct.c_int()
        driver.call("cuDeviceGetCount", ct.byref(count))
        if device_index >= count.value:
            raise CudaError("Declared CUDA device index is unavailable")
        driver.call("cuDeviceGet", ct.byref(device), device_index)
        driver.call("cuDriverGetVersion", ct.byref(version))
        name, raw_uuid = ct.create_string_buffer(256), (ct.c_ubyte * 16)()
        driver.call("cuDeviceGetName", name, len(name), device)
        driver.call("cuDeviceGetUuid", raw_uuid, device)
        major, minor = ct.c_int(), ct.c_int()
        driver.call("cuDeviceGetAttribute", ct.byref(major), 75, device)
        driver.call("cuDeviceGetAttribute", ct.byref(minor), 76, device)
        if major.value < 5:
            raise CudaError("Fixed PTX requires compute capability 5.0 or newer")
        self._identity = {
            "schema": "ciw.cuda-gaussian-worker.v1", "execution_device": "gpu",
            "device_index": device_index, "device_name": name.value.decode("utf-8", errors="strict"),
            "device_uuid": "GPU-" + str(uuid.UUID(bytes=bytes(raw_uuid))),
            "compute_capability": [major.value, minor.value], "cuda_driver_version": version.value,
            "kernel_sha256": sha256(PTX.encode("ascii")).hexdigest(), "ptx_version": "6.0", "ptx_target": "sm_50",
            "arithmetic": "binary64_explicit_round_to_nearest_no_fma_contraction",
            "algorithm": "fixed_step_mean_gradient_and_full_precision_relaxation",
            "replica_semantics": "identical_declared_problem_independent_thread_work",
            "iterations": iterations, "replicas": replicas, "threads_per_block": THREADS_PER_BLOCK,
            "output_layout": list(OUTPUT_LAYOUT), "prepared_input_sha256": sha256(values.tobytes()).hexdigest(),
            "problem_sha256": sha256(json.dumps(problem, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "solver_settings": deepcopy(solver_settings),
            "measurement_boundary": "constructor_preparation_and_jit_excluded;solve_includes_launch_sync_copy_and_output_check",
        }
        driver.call("cuCtxCreate_v2", ct.byref(self._context), 0, device)
        self._closed = False
        try:
            try:
                image = ct.create_string_buffer(PTX.encode("ascii"))
                driver.call("cuModuleLoadData", ct.byref(self._module), image)
                driver.call("cuModuleGetFunction", ct.byref(self._function), self._module, b"gaussian_vi")
                driver.call("cuMemAlloc_v2", ct.byref(self._input), values.nbytes)
                driver.call("cuMemAlloc_v2", ct.byref(self._output), replicas * 6 * 8)
                driver.call("cuMemcpyHtoD_v2", self._input, values.ctypes.data, values.nbytes)
                driver.call("cuCtxSynchronize")
            finally:
                popped = ct.c_void_p()
                driver.call("cuCtxPopCurrent_v2", ct.byref(popped))
        except BaseException:
            self.close()
            raise

    def identity(self):
        return deepcopy(self._identity)

    @contextmanager
    def _current(self):
        if self._closed:
            raise CudaError("CUDA worker is closed")
        self._driver.call("cuCtxPushCurrent_v2", self._context)
        try:
            yield
        finally:
            popped = ct.c_void_p()
            self._driver.call("cuCtxPopCurrent_v2", ct.byref(popped))

    def solve(self):
        with self._lock, self._current():
            output = np.empty((self._replicas, 6), dtype=np.float64)
            arguments = [self._input, self._output, ct.c_uint(self._iterations), ct.c_uint(self._replicas)]
            pointers = (ct.c_void_p * len(arguments))(*(ct.addressof(value) for value in arguments))
            blocks = (self._replicas + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            self._driver.call("cuLaunchKernel", self._function, blocks, 1, 1, THREADS_PER_BLOCK, 1, 1, 0, None, pointers, None)
            self._driver.call("cuCtxSynchronize")
            self._driver.call("cuMemcpyDtoH_v2", output.ctypes.data, self._output, output.nbytes)
        if not np.all(np.isfinite(output)) or np.max(np.abs(output[:, :2])) > mathematics.MAX_ITERATE_MEAN:
            raise ValueError("GPU iteration produced nonfinite or out-of-bound output")
        covariances = output[:, 2:].reshape(-1, 2, 2)
        if not np.array_equal(covariances, covariances.transpose(0, 2, 1)):
            raise ValueError("GPU covariance lost symmetry")
        eigenvalues = np.linalg.eigvalsh(covariances)
        if np.any(eigenvalues[:, 0] <= 0) or np.any(eigenvalues[:, 1] / eigenvalues[:, 0] > 1e12):
            raise ValueError("GPU covariance violates its positive-definite conditioning bound")
        return output

    def close(self):
        with self._lock:
            if not self._closed:
                # Destroying this private context also frees its modules/buffers.
                self._driver.call("cuCtxDestroy_v2", self._context)
                self._closed = True

    def __enter__(self):
        if self._closed:
            raise CudaError("CUDA worker is closed")
        return self

    def __exit__(self, *_exception):
        self.close()
