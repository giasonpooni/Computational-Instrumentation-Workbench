"""Telemetry helpers for the energy and GPU experiments (T115-T125).

Scope: read-only probes of Linux RAPL powercap counters and an operator-run
RAPL capture that executes outside the lab runner (one bracket per declared
workload: the T115 sphere geodesics and the common Gaussian VI workload of
``energy_gpu_workload`` in NumPy float64 and float32 and in the Rust port,
each declaring its work boundary, with the prepared inputs computed once
outside every bracket); loading and tampering the retained synthetic
energy/accuracy fixtures; gating when a retained NVML log (which must name
the common workload) or RAPL capture (whose unit counts must match its own
repeats and batches) may support a physical-domain finding; parsing timestamped
nvidia-smi sidecar rows; and a replay of the offline energy-accuracy
workflow through a ciw Session.

Non-claims: the bundled fixtures are synthetic. Validation of a sealed log
establishes internal integrity only; neither the declared ``origin`` nor the
device identity in a log authenticates that a physical device produced it.
The lab runner calls nothing here that reads an energy counter: counters are
read only by ``python -m ciw.lab.energy_gpu_telemetry rapl-capture`` (an
operator action) and by ``ciw energy record``. The runner's own
``hardware:rapl`` availability probe (``ciw.lab.runner``), made by T115, T117
and T120 only when a capture is supplied, reads one ``energy_uj`` value to
confirm readability and discards it. Nothing here starts GPU work,
changes device settings or estimates energy from time or utilization.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import time

import numpy as np

from . import runner

FIXTURES = ("baseline", "reset", "missing", "under-target")
POWERCAP = Path("/sys/class/powercap")
LOG_ENV = "CIW_LAB_ENERGY_LOG"
SMI_ENV = "CIW_LAB_NVIDIA_SMI_CSV"
SMI_OFFSET_ENV = "CIW_LAB_NVIDIA_SMI_UTC_OFFSET"
RAPL_ENV = "CIW_LAB_RAPL_LOG"
RAPL_SCHEMA = "ciw.lab.rapl-capture.v3"
OPERATOR_NOTE = ("acquired by an operator outside the lab runner; the lab checked the identity binding and "
                 "retained the bytes, it did not authenticate the capture")


# RAPL ---------------------------------------------------------------------
def rapl_domains(root=POWERCAP, separator=":") -> list:
    """Top-level package domains ``intel-rapl:N`` with their name and wrap range.

    ``separator`` exists so tests can mimic the tree on file systems that
    forbid ':' in names.
    """
    domains = []
    for directory in sorted(Path(root).glob(f"intel-rapl{separator}*")):
        if directory.name.count(separator) != 1 or not (directory / "energy_uj").is_file():
            continue  # subzones (intel-rapl:0:0) are contained in their package
        name = (directory / "name").read_text().strip() if (directory / "name").is_file() else directory.name
        limit = directory / "max_energy_range_uj"
        domains.append({"zone": directory.name, "name": name, "path": str(directory / "energy_uj"),
                        "max_energy_range_uj": int(limit.read_text()) if limit.is_file() else None})
    return domains


def rapl_read(domains) -> list:
    """Raw integer counters in microjoules; PermissionError propagates to the caller."""
    return [int(Path(domain["path"]).read_text().strip()) for domain in domains]


def rapl_delta_uj(before: int, after: int, max_range: int | None) -> tuple[int, bool]:
    """Counter difference allowing at most one wrap at ``max_range``; returns (delta, wrapped)."""
    if after >= before:
        return after - before, False
    if max_range is None:
        raise ValueError("RAPL counter decreased without a declared wrap range")
    return after + (max_range - before) + 1, True


def rapl_host_identity(root=POWERCAP, separator=":") -> dict:
    """CPU model, platform and RAPL package zones of this host (reads no counter)."""
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu_model": cpu, "machine": platform.machine(), "system": platform.system(),
            "zones": [f"{d['zone']} {d['name']}" for d in rapl_domains(root, separator)]}


def _bracket(domains, action) -> dict:
    """Counter reads before and after ``action`` with UTC and monotonic brackets."""
    utc_start, mono_start = time.time_ns(), time.monotonic_ns()
    before = rapl_read(domains)
    action()
    after = rapl_read(domains)
    mono_end, utc_end = time.monotonic_ns(), time.time_ns()
    return {"before_uj": before, "after_uj": after, "start_utc_ns": str(utc_start), "end_utc_ns": str(utc_end),
            "elapsed_monotonic_ns": mono_end - mono_start}


# One bracket per workload; the Gaussian VI brackets run the common workload of energy_gpu_workload.
GEODESIC = "geodesic"
VI_NUMPY64, VI_NUMPY32, VI_RUST64 = "gaussian-vi-numpy-float64", "gaussian-vi-numpy-float32", "gaussian-vi-rust-float64"
BRACKETS = (GEODESIC, VI_NUMPY64, VI_NUMPY32, VI_RUST64)
UNIT = {GEODESIC: "trajectory", VI_NUMPY64: "batch", VI_NUMPY32: "batch", VI_RUST64: "batch"}


# What each Gaussian VI bracket contains per batch: the work boundary is part of the declaration, so a capture made
# with other bracket contents does not name the same workload.
NUMPY_BOUNDARY = ("prepared inputs computed once before the bracket (excluded); per batch: the 15 inputs rounded to "
                  "the declared precision and broadcast to the replica columns, K iterations and the output block, "
                  "vectorized across replicas in NumPy")
RUST_BOUNDARY = ("prepared inputs computed once before the bracket (excluded); one process start of the compiled "
                 "port and one JSON exchange included; per batch, inside that process: every replica rounds the 15 "
                 "inputs and runs K iterations and the output block")
# The Rust port refuses repeats outside [1, 1000] (RUST_SOURCE), so a Gaussian VI bracket holds at most 1000 batches.
MAX_BATCHES = 1000
MAX_REPEATS = 1000


def declared_workloads(rust_identity: dict | None = None) -> dict:
    """The workload each bracket declares; a task analyzes a bracket only when the capture names exactly this.

    Each Gaussian VI declaration carries its bracket's work boundary. The Rust
    bracket's declaration includes the compiled port's rustc version and
    binary digest, so it is comparable only with a port built by the same
    rustc (the binary is reproducible for a given rustc and target).
    """
    from . import energy_gpu_kernels as kernels
    from . import energy_gpu_workload as common
    declared = {GEODESIC: dict(kernels.WORKLOAD),
                VI_NUMPY64: dict(common.workload("float64", "numpy"), boundary=NUMPY_BOUNDARY),
                VI_NUMPY32: dict(common.workload("float32", "numpy"), boundary=NUMPY_BOUNDARY)}
    if rust_identity is not None:
        declared[VI_RUST64] = dict(common.workload("float64", "rust"), boundary=RUST_BOUNDARY,
                                   rustc=rust_identity["rustc"], source_sha256=rust_identity["source_sha256"],
                                   binary_sha256=rust_identity["binary_sha256"])
    return declared


def expected_units(record, name) -> int | None:
    """Units a capture's bracket must count: trajectories (repeats x headings) or batches; None when unknowable."""
    from . import energy_gpu_kernels as kernels
    repeats, batches = record.get("repeats"), record.get("batches")
    if name == GEODESIC:
        return repeats * len(kernels.HEADINGS) if type(repeats) is int and 1 <= repeats <= MAX_REPEATS else None
    return batches if type(batches) is int and 1 <= batches <= MAX_BATCHES else None


def capture_rapl(output, repeats=3, batches=500, root=POWERCAP, separator=":", sleep=time.sleep, rust=True) -> dict:
    """Operator-run capture: bracket each declared workload with package counter reads, then an idle interval.

    Brackets: ``geodesic`` runs ``repeats`` batches of the T115 sphere
    geodesics; ``gaussian-vi-numpy-float64`` and ``-float32`` run ``batches``
    batches of the common Gaussian VI workload with the NumPy reference;
    ``gaussian-vi-rust-float64`` runs the same batches in one process of the
    compiled Rust port (process start and JSON exchange included) when rustc
    builds it here, and is otherwise listed under ``skipped`` with the reason.
    The prepared inputs are computed once, before any bracket, so no bracket
    contains them (``NUMPY_BOUNDARY``, ``RUST_BOUNDARY``). Before bracketing,
    the port runs one replica with the real ``repeats`` payload, so a refusal
    surfaces before any counter is read; a port that fails inside its bracket
    is moved to ``skipped`` and the other brackets are kept. The idle bracket
    lasts as long as the longest workload bracket. This is the only function
    in the section that reads energy counters; the lab runner never calls it.
    The record is written once (an existing file is refused) and analyzed
    read-only by T115, T117 and T120 through ``CIW_LAB_RAPL_LOG``
    (``--capture rapl-log=PATH``). It holds no host path.
    """
    from . import energy_gpu_kernels as kernels
    from . import energy_gpu_workload as common
    path = Path(output)
    if path.exists():
        raise ValueError("Refusing to overwrite an existing RAPL capture")
    for name, value, high in (("repeats", repeats, MAX_REPEATS), ("batches", batches, MAX_BATCHES)):
        if type(value) is not int or not 1 <= value <= high:
            raise ValueError(f"{name} must be an integer in [1, {high}]")
    domains = rapl_domains(root, separator)
    if not domains:
        raise ValueError("No intel-rapl package domain with an energy_uj counter")
    values = common.prepared_inputs()  # once, outside every bracket
    common.plan_iterations()
    work = {GEODESIC: lambda: [kernels.fixed_step_batch() for _ in range(repeats)],
            VI_NUMPY64: lambda: [common.run_numpy(np.float64, values=values) for _ in range(batches)],
            VI_NUMPY32: lambda: [common.run_numpy(np.float32, values=values) for _ in range(batches)]}
    counts = {GEODESIC: repeats * len(kernels.HEADINGS), VI_NUMPY64: batches, VI_NUMPY32: batches}
    skipped, rust_identity = {}, None
    with tempfile.TemporaryDirectory() as scratch:
        if rust:
            try:
                rust_identity = common.build_rust_port(scratch)
                # The real repeats payload on one replica, before bracketing: a refusal cannot lose the capture.
                common.run_rust_port(rust_identity["executable"], "float64", replicas=1, repeats=batches, values=values)
            except kernels.NativeKernelUnavailable as exc:
                rust_identity, skipped[VI_RUST64] = None, str(exc)
        else:
            skipped[VI_RUST64] = "not requested (--no-rust)"
        if rust_identity is not None:
            executable = rust_identity["executable"]
            work[VI_RUST64] = lambda: common.run_rust_port(executable, "float64", repeats=batches, values=values)
            counts[VI_RUST64] = batches
        brackets = {}
        for name, action in work.items():
            try:
                brackets[name] = _bracket(domains, action)
            except kernels.NativeKernelUnavailable as exc:
                if name != VI_RUST64:
                    raise
                skipped[name] = f"the Rust port failed inside its bracket: {exc}"
                counts.pop(name)
                rust_identity = None
    longest = max(bracket["elapsed_monotonic_ns"] for bracket in brackets.values())
    idle = _bracket(domains, lambda: sleep(longest / 1e9))
    declared = declared_workloads(rust_identity)
    record = {"schema": RAPL_SCHEMA, "captured_by": "python -m ciw.lab.energy_gpu_telemetry rapl-capture",
              "repeats": repeats, "batches": batches, "host": rapl_host_identity(root, separator),
              "domains": [{key: d[key] for key in ("zone", "name", "max_energy_range_uj")} for d in domains],
              "workloads": {name: declared[name] for name in brackets}, "units": dict(counts),
              "unit_names": {name: UNIT[name] for name in brackets}, "brackets": brackets, "idle_bracket": idle,
              "skipped": skipped}
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _bracket_reasons(name, bracket, domains) -> list:
    if not isinstance(bracket, dict) or len(bracket.get("before_uj") or []) != len(domains) \
            or len(bracket.get("after_uj") or []) != len(domains) or not domains \
            or not isinstance(bracket.get("elapsed_monotonic_ns"), int) or bracket["elapsed_monotonic_ns"] <= 0 \
            or not all(str(bracket.get(key, "")).isdigit() for key in ("start_utc_ns", "end_utc_ns")):
        return [f"capture {name} bracket is malformed"]
    return []


def rapl_capture_reasons(record, host_identity: dict, declared: dict) -> dict:
    """Why each declared bracket cannot support a physical finding here: {bracket: [reasons]} (empty when it can).

    ``declared`` maps bracket names to the workload the analyzing task
    declares for them (:func:`declared_workloads`). A reason common to the
    whole capture (schema, host, idle bracket) is listed under every bracket.
    """
    if not isinstance(record, dict) or record.get("schema") != RAPL_SCHEMA:
        return {name: [f"capture is not a {RAPL_SCHEMA} record"] for name in declared}
    common = []
    host = record.get("host") if isinstance(record.get("host"), dict) else {}
    differing = [key for key in ("cpu_model", "machine", "system", "zones") if host.get(key) != host_identity.get(key)]
    if differing:
        common.append("capture host differs from this host: " + ", ".join(differing))
    domains = record.get("domains") or []
    if not isinstance(domains, list) or not all(isinstance(d, dict) for d in domains):
        return {name: ["capture domains are malformed"] for name in declared}
    common += _bracket_reasons("idle", record.get("idle_bracket"), domains)
    reasons = {}
    workloads, units, unit_names, brackets, skipped = (
        record.get(key) if isinstance(record.get(key), dict) else {}
        for key in ("workloads", "units", "unit_names", "brackets", "skipped"))
    for name, workload in declared.items():
        own = list(common)
        if name not in brackets:
            why = skipped.get(name)
            own.append(f"capture has no {name} bracket" + (f" ({why})" if why else ""))
        else:
            if workloads.get(name) != workload:
                own.append(f"capture names a different {name} workload than this task declares")
            if type(units.get(name)) is not int or units[name] < 1:
                own.append(f"capture {name} unit count is malformed")
            elif units[name] != expected_units(record, name):
                own.append(f"capture {name} unit count disagrees with its repeats/batches")
            if unit_names.get(name) != UNIT.get(name):
                own.append(f"capture {name} unit name is not {UNIT.get(name)}")
            own += _bracket_reasons(name, brackets[name], domains)
        reasons[name] = own
    return reasons


def rapl_capture_energy(record, name) -> dict:
    """Gross and idle-subtracted package energy per unit of bracket ``name`` (one wrap per counter allowed)."""
    domains = record["domains"]

    def total(bracket):
        deltas = [rapl_delta_uj(int(b), int(a), d["max_energy_range_uj"])
                  for b, a, d in zip(bracket["before_uj"], bracket["after_uj"], domains)]
        return sum(delta for delta, _ in deltas), [wrapped for _, wrapped in deltas]

    bracket = record["brackets"][name]
    gross_uj, wrapped = total(bracket)
    idle_uj, idle_wrapped = total(record["idle_bracket"])
    work_s = bracket["elapsed_monotonic_ns"] / 1e9
    idle_s = record["idle_bracket"]["elapsed_monotonic_ns"] / 1e9
    units = record["units"][name]
    # Idle energy is rescaled to the workload interval before subtraction.
    idle_equivalent_uj = idle_uj * work_s / idle_s
    return {"units": units, "unit": record["unit_names"][name], "gross_uj": gross_uj, "idle_uj": idle_uj,
            "workload_s": work_s, "idle_s": idle_s, "wrapped": wrapped + idle_wrapped,
            "gross_j_per_unit": gross_uj * 1e-6 / units,
            "idle_subtracted_j_per_unit": (gross_uj - idle_equivalent_uj) * 1e-6 / units}


# Fixtures -------------------------------------------------------------------
def fixture_dir() -> Path | None:
    """``examples/energy-accuracy`` under the lab repository root (honours CIW_LAB_REPOSITORY_ROOT)."""
    return runner.repository_path("examples", "energy-accuracy")


def fixture_bytes() -> dict | None:
    """Exact retained bytes of the synthetic fixtures, or None when the repository files are unreachable."""
    root = fixture_dir()
    if root is None:
        return None
    paths = {name: root / f"{name}.json" for name in FIXTURES}
    if not all(path.is_file() for path in paths.values()):
        return None
    return {name: path.read_bytes() for name, path in paths.items()}


def reseal(log):
    from ciw import energy_records
    return energy_records.seal({key: deepcopy(value) for key, value in log.items() if key != "log_digest"})


def _set(path, value):
    def change(log):
        node = log
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return change


def _drop(path):
    def change(log):
        node = log
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
    return change


# Each tampering class: (name, mutation, reseal after mutation, expected refusal).
MEASURE = ("phases", 3, "samples", 1)
TAMPERING = (
    ("raw counter reading edited without resealing", _set(MEASURE + ("energy_mj",), "1950"), False,
     "Retained log digest differs"),
    ("sensor device UUID replaced by a non-UUID", _set(("sensor", "device_uuid"), "GPU-unknown"), True,
     "Require a physical GPU UUID"),
    ("calibration metadata invented for the sensor", _set(("sensor", "resolution_j"), 0.001), True,
     "Unsupported sensor scope or invented calibration metadata"),
    ("monotonic read bracket reversed", _set(MEASURE + ("read_start_ns",), 1600000390), True,
     "Reversed monotonic hardware call bracket"),
    ("failed counter read keeps a fabricated energy", _set(MEASURE + ("status",), "error"), True,
     "Failed counter reading cannot contain fabricated energy"),
    ("UTC timestamp of a reading removed", _drop(MEASURE + ("utc_start_ns",)), True,
     "Require exactly the declared energy-log fields"),
    ("runtime executable identity removed", _drop(("runtime", "python", "executable_sha256")), True,
     "Require exactly the declared energy-log fields"),
    ("prepared device input identity replaced", _set(("runtime", "workload", "prepared_input_sha256"), "0" * 64), True,
     "Prepared device input differs from declared initial problem"),
    ("retained batch output edited", lambda log: log["phases"][3]["batches"][0]["result"]["values"].__setitem__(0, 0.35),
     True, "Expanded result bytes differ from their digest"),
)


def tamper_outcome(log, mutation, resealed) -> str | None:
    """Apply one mutation to a copy; return the refusal message or None when accepted."""
    from ciw import energy_records
    changed = deepcopy(log)
    mutation(changed)
    try:
        if resealed:
            changed = reseal(changed)
        energy_records.validate_log(changed)
    except ValueError as exc:
        return str(exc)
    return None


def is_placeholder(value) -> bool:
    """A fixture-style stand-in rather than an identification.

    Digests and UUIDs whose hex digits are one repeated character or the
    ascending run 0123456789abcdef, and text naming a fixture or synthetic
    source, identify nothing.
    """
    if not isinstance(value, str):
        return False
    text = value.lower()
    if "fixture" in text or "synthetic" in text:
        return True
    digits = text.removeprefix("gpu-").removeprefix("sha256:").replace("-", "")
    if len(digits) < 16 or not re.fullmatch(r"[0-9a-f]+", digits):
        return False
    return len(set(digits)) == 1 or digits == "0123456789abcdef" * (len(digits) // 16)


# Identity values that describe the device itself; they identify nothing when the device's own identity
# (its UUID or name) is a placeholder, whatever their type (compute capability is a list of integers).
DEVICE_DERIVED = ("compute_capability",)
DEVICE_IDENTITY = ("sensor_device_uuid", "sensor_name", "workload_device_uuid")


def raw_inventory(log) -> dict:
    """Count retained raw readings, the identity fields a log carries and how many hold placeholders.

    A field is a placeholder when :func:`is_placeholder` says so, or when it is
    a device-derived value (compute capability) of a device whose UUID or name
    is itself a placeholder: such a value was invented for a device that does
    not exist.
    """
    samples = [sample for phase in log["phases"] for sample in phase["samples"]]
    timestamped = [s for s in samples if all(s.get(k) is not None for k in
                                             ("read_start_ns", "read_end_ns", "utc_start_ns", "utc_end_ns"))]
    raw_counters = [s for s in samples if s["status"] == "ok" and isinstance(s["energy_mj"], str)]
    sensor, runtime = log["sensor"], log["runtime"]
    identity = {"sensor_device_uuid": sensor["device_uuid"], "sensor_name": sensor["name"],
                "driver_version": sensor["driver_version"], "nvml_version": sensor["nvml_version"],
                "nvml_library_sha256": sensor["library_sha256"],
                "workload_device_uuid": runtime["workload"]["device_uuid"],
                "compute_capability": runtime["workload"]["compute_capability"],
                "kernel_sha256": runtime["workload"]["kernel_sha256"],
                "prepared_input_sha256": runtime["workload"]["prepared_input_sha256"],
                "python_version": runtime["python"]["version"], "numpy_version": runtime["python"]["numpy_version"],
                "python_executable_sha256": runtime["python"]["executable_sha256"],
                "implementation_code_sha256": runtime["implementation"]["code_sha256"],
                "clock": log["clock"]["implementation"]}
    invented_device = any(is_placeholder(identity[key]) for key in DEVICE_IDENTITY)
    placeholders = sorted(k for k, v in identity.items()
                          if is_placeholder(v) or (k in DEVICE_DERIVED and invented_device and v not in (None, "")))
    return {"samples": len(samples), "timestamped_samples": len(timestamped),
            "raw_counter_readings": len(raw_counters), "batches": sum(len(p["batches"]) for p in log["phases"]),
            "identity_fields": identity, "identity_fields_present": sum(v not in (None, "") for v in identity.values()),
            "placeholder_values": len(placeholders), "placeholder_fields": placeholders,
            "origin": log["origin"], "log_digest": log["log_digest"]}


# Operator-captured NVML logs (GPU host only) --------------------------------
SENSOR_BINDING = ("device_uuid", "name", "driver_version", "nvml_version", "library_sha256")


def host_sensor_identity(device_uuid: str) -> dict | None:
    """NVML identity of the device with this UUID on this host (no counter read), or None."""
    try:
        from ciw.energy_nvml import NVMLEnergyCounter, NVMLUnavailable
    except ImportError:
        return None
    try:
        with NVMLEnergyCounter(uuid=device_uuid) as counter:
            return counter.identity()
    except (NVMLUnavailable, ValueError, OSError):
        return None


def _utc(ns_decimal: str) -> str:
    seconds, remainder = divmod(int(ns_decimal), 10**9)
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{stamp}.{remainder:09d}Z"


# The fields of a log's runtime workload that must equal the common workload's declaration.
WORKLOAD_FIELDS = ("kernel_sha256", "problem_sha256", "prepared_input_sha256", "iterations", "replicas")


def workload_reasons(log) -> list:
    """Why an operator NVML log does not name the common workload; empty when it does.

    The log's runtime workload (PTX kernel digest, problem digest, prepared
    input digest, K and replicas) must equal ``energy_gpu_workload.workload()``
    and its plan's KL target must be the declared one, so an energy per batch
    or per accepted result is for the computation the CPU brackets (T115,
    T117, T120) and the CPU/GPU comparisons (T117, T121, T147) use.
    """
    from . import energy_gpu_workload as common
    declared = common.workload()
    runtime = log.get("runtime") if isinstance(log, dict) else None
    work = runtime.get("workload") if isinstance(runtime, dict) else None
    work = work if isinstance(work, dict) else {}
    plan = log.get("plan") if isinstance(log, dict) and isinstance(log.get("plan"), dict) else {}
    differing = [key for key in WORKLOAD_FIELDS if work.get(key) != declared[key]]
    if plan.get("target_kl_nats") != common.SPEC["target_kl_nats"]:
        differing.append("target_kl_nats")
    return (["log names a different GPU workload than the common workload (examples/energy-accuracy/problem.json, "
             f"K = {declared['iterations']}, {declared['replicas']} replicas, the gaussian_vi PTX kernel): "
             + ", ".join(differing)] if differing else [])


def physical_basis(raw: bytes, log, analysis, host_identity: dict | None,
                   required_name: str | None = None) -> tuple[dict, list]:
    """Acquisition basis for a physical finding, or a notes-only basis and the reasons it is withheld.

    A retained log supports a physical claim here only when it declares a
    physical measurement, its analysis is eligible, it names the common
    workload (:func:`workload_reasons`), and its sensor identity (UUID, name,
    driver, NVML version and NVML library digest) equals the NVML identity of
    that device on this host; optionally the device name must contain
    ``required_name``. This binds the record to hardware and software present
    on this host. It does not authenticate that the operator's capture was
    genuine: no signature ties the readings to the device.
    """
    reasons = []
    if log["origin"] != "physical_measurement":
        reasons.append("log declares a synthetic fixture, not a physical measurement")
    reasons += workload_reasons(log)
    if not analysis["comparison"]["eligible"]:
        reasons.extend(analysis["comparison"]["reasons"] or ["analysis is not eligible for comparison"])
    if host_identity is None:
        reasons.append("no NVML device with the log's sensor UUID is present on this host")
    else:
        differing = [key for key in SENSOR_BINDING if host_identity.get(key) != log["sensor"].get(key)]
        if differing:
            reasons.append("log sensor identity differs from this host's NVML identity: " + ", ".join(differing))
    if required_name and required_name.lower() not in log["sensor"]["name"].lower():
        reasons.append(f"sensor device is not an {required_name}")
    if reasons:
        return {"notes": reasons}, reasons
    samples = log["phases"][3]["samples"]
    acquisition = {"device": f"{log['sensor']['name']} {log['sensor']['device_uuid']} (NVML total energy counter; "
                             "operator-captured log, unauthenticated)",
                   "raw_sha256": hashlib.sha256(raw).hexdigest(), "acquired_at": _utc(samples[0]["utc_start_ns"]),
                   "calibration": "not_applied: vendor counter, resolution and accuracy undeclared"}
    return {"acquisition": acquisition, "notes": [OPERATOR_NOTE]}, []


def read_operator_log(path) -> tuple[bytes, dict, dict]:
    from ciw import energy_records
    raw = Path(path).read_bytes()
    if len(raw) > energy_records.MAX_BYTES:
        raise ValueError("Operator energy log exceeds its byte budget")
    log = json.loads(raw)
    return raw, log, energy_records.analyze(log)


def utc_offset_ns(text: str | None) -> int | None:
    """Declared offset of local time from UTC ("+HH:MM" or "-HH:MM") in nanoseconds, or None."""
    match = re.fullmatch(r"([+-])(\d{2}):(\d{2})", (text or "").strip())
    if not match or int(match.group(2)) > 14 or int(match.group(3)) > 59:
        return None
    sign = 1 if match.group(1) == "+" else -1
    return sign * (int(match.group(2)) * 3600 + int(match.group(3)) * 60) * 10**9


def smi_rows(text: str) -> list:
    """Rows of ``nvidia-smi --query-gpu=... --format=csv,nounits`` text as {header: raw cell} dicts."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    header = [name.strip() for name in lines[0].split(",")]
    return [dict(zip(header, (cell.strip() for cell in line.split(",")))) for line in lines[1:]]


def smi_utc_ns(cell: str, offset_ns: int) -> int | None:
    """nvidia-smi prints local wall time ("YYYY/MM/DD HH:MM:SS.fff"); convert with the declared offset."""
    for pattern in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            local = datetime.strptime(cell.strip(), pattern)
        except ValueError:
            continue
        since = local - datetime(1970, 1, 1)
        return (since.days * 86400 + since.seconds) * 10**9 + since.microseconds * 1000 - offset_ns
    return None


def number(cell) -> float | None:
    """A numeric cell as a float; unavailable readings ("[N/A]", blanks) become None."""
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def smi_window(text: str, start_ns: int, end_ns: int, offset_ns: int | None) -> tuple[list, list]:
    """Sidecar rows whose UTC timestamp lies in [start_ns, end_ns], and the reasons none can be used.

    The sidecar runs longer than the measurement phase (it is started before
    and stopped after the capture), so rows outside the window are idle,
    warmup or startup samples and must not enter a measurement statistic.
    """
    if offset_ns is None:
        return [], [f"no UTC offset declared for the sidecar's local timestamps ({SMI_OFFSET_ENV})"]
    rows = smi_rows(text)
    stamped = [(smi_utc_ns(row.get("timestamp", ""), offset_ns), row) for row in rows]
    if any(stamp is None for stamp, _ in stamped):
        return [], ["sidecar rows carry unparseable timestamps"]
    inside = [row for stamp, row in stamped if start_ns <= stamp <= end_ns]
    return inside, ([] if inside else ["no sidecar row falls inside the measurement window"])


# Offline replay through a ciw Session ----------------------------------------
def session_replay(raws: dict) -> dict:
    """Add each fixture to one Session, execute the offline analysis, replay it and restore the workspace."""
    from ciw import energy_records
    from ciw.energy_workflow import OPERATION, EnergyAccuracyWorkflow
    from ciw.instruments import make_demo_run
    from ciw.session import Session
    from ciw.telemetry import digest

    rows, refusals = {}, {}
    with tempfile.TemporaryDirectory() as scratch:
        session = Session(make_demo_run(), Path(scratch) / "session")

        def call(kind, payload):
            answer = session.handle({"protocol_version": 1, "request_id": kind, "type": kind, "payload": payload})
            if answer["type"] != "response":
                raise ValueError(f"Session refused {kind}: {answer['payload']}")
            return answer["payload"]

        views = {}
        for name, raw in raws.items():
            source = call("source.add", {"kind": "energy-accuracy", "label": f"Synthetic fixture {name}",
                                         "bytes_b64": base64.b64encode(raw).decode("ascii")})
            original = call("operation.execute", {"operation_id": OPERATION,
                                                  "parameters": {"source_id": source["source_id"]}})
            replay = call("bundle.replay", {"bundle_id": original["bundle_id"]})
            first = call("bundle.get", {"bundle_id": original["bundle_id"]})
            second = call("bundle.get", {"bundle_id": replay["bundle"]["bundle_id"]})
            direct = digest({"operation_id": OPERATION, "data": energy_records.analyze(json.loads(raw))})
            executions = [first["steps"][0]["execution_id"], first["verification"]["reproduction"]["execution_id"],
                          second["steps"][0]["execution_id"], second["verification"]["reproduction"]["execution_id"]]
            rows[name] = {
                "numerical_result_ids": [first["steps"][0]["numerical_result_id"],
                                         first["verification"]["reproduction"]["numerical_result_id"],
                                         second["steps"][0]["numerical_result_id"],
                                         second["verification"]["reproduction"]["numerical_result_id"]],
                "direct_numerical_result_id": direct, "execution_ids": executions,
                "result_ids": [first["steps"][0]["result_id"], second["steps"][0]["result_id"]],
                "bundle_digests": [first["bundle_digest"], second["bundle_digest"]],
                "receipt_numerical_match": replay["replay_receipt"]["numerical_match"],
                "receipt_admission": replay["replay_receipt"]["admission"],
                "verification_independent": [first["verification"]["independent"], second["verification"]["independent"]],
                "verification_method": first["verification"]["method"],
                "fresh_hardware_measurement": call("experiment.inspect", {"bundle_id": original["bundle_id"]})
                ["object_context"]["fresh_hardware_measurement"],
            }
            if name == "baseline":
                views[name] = call("experiment.inspect", {"bundle_id": original["bundle_id"]})
                # An edited retained bundle cannot be replayed, whether or not its envelopes were resealed.
                for label, resealed in (("edited_bundle", False), ("edited_and_resealed_bundle", True)):
                    tampered = deepcopy(first)
                    for step in (tampered["steps"][0], tampered["verification"]["reproduction"]):
                        step["result"]["data"]["measurement"]["gross_energy_j"] = 99.0
                    if resealed:
                        reseal_bundle(tampered)
                    try:
                        EnergyAccuracyWorkflow().replay_session(tampered, {})
                        refusals[label] = None
                    except ValueError as exc:
                        refusals[label] = str(exc)
        saved = call("workspace.save", {})
        restored = Session.from_workspace(Path(saved["workspace_file"]), Path(scratch) / "restored")
        restored_equal = restored.workbench.serialize() == session.workbench.serialize()
        bundle_count = len(session.workbench.serialize()["bundles"])
    return {"fixtures": rows, "refusals": refusals, "restored_equal": restored_equal,
            "bundle_count": bundle_count, "baseline_view": views.get("baseline")}


def reseal_bundle(bundle):
    """Recompute every envelope digest so a refusal reflects the edited content, not a stale hash."""
    from ciw.telemetry import _bundle_digest, byte_digest, canonical, digest
    for step in (bundle["steps"][0], bundle["verification"]["reproduction"]):
        result = step["result"]
        result["result_id"] = digest({k: v for k, v in result.items() if k != "result_id"})
        step["result_id"], step["result_sha256"] = result["result_id"], digest(result)
        step["request_sha256"] = digest(step["request"])
        step["numerical_result"] = {"operation_id": step["operation_id"], "data": deepcopy(result["data"])}
        step["numerical_result_id"] = digest(step["numerical_result"])
    bundle["bundle_digest"] = _bundle_digest(bundle)
    verification = bundle["verification"]
    verification["subject_ref"] = bundle["bundle_digest"]
    verification["runtime_digest"] = digest(bundle["runtimes"])
    verification["verification_id"] = byte_digest(verification["schema"].encode() + b"\0" + canonical(
        {k: v for k, v in verification.items() if k != "verification_id"}))
    return bundle


def environment_log_path(variable=LOG_ENV) -> str | None:
    value = os.environ.get(variable, "").strip()
    return value or None


def main(argv=None) -> int:
    """``python -m ciw.lab.energy_gpu_telemetry rapl-capture OUTPUT``: the operator-run RAPL capture."""
    import argparse
    parser = argparse.ArgumentParser(prog="python -m ciw.lab.energy_gpu_telemetry",
                                     description="Operator-run RAPL capture for lab tasks T115, T117 and T120 (outside "
                                                 "the lab runner)")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("rapl-capture", help="bracket the geodesic workload and the common Gaussian VI "
                                                       "workload (NumPy float64 and float32, Rust float64), then an "
                                                       "idle interval")
    capture.add_argument("output", help="new JSON file for the raw capture (an existing file is refused)")
    capture.add_argument("--repeats", type=int, default=3,
                         help=f"geodesic batches (six trajectories each; 1 to {MAX_REPEATS})")
    capture.add_argument("--batches", type=int, default=500,
                         help=f"common-workload batches per Gaussian VI bracket (1 to {MAX_BATCHES}, the Rust port's "
                              "limit)")
    capture.add_argument("--no-rust", action="store_true", help="skip the Rust port bracket")
    args = parser.parse_args(argv)
    try:
        record = capture_rapl(args.output, args.repeats, args.batches, rust=not args.no_rust)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"RAPL capture refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "zones": record["host"]["zones"], "brackets": sorted(record["brackets"]),
                      "skipped": record["skipped"],
                      "then": f"ciw lab run T115 T117 T120 --capture rapl-log={args.output} --output-dir <dir>"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
