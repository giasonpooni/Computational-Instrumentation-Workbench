"""Telemetry helpers for the energy and GPU experiments (T115-T125).

Scope: read-only probes of Linux RAPL powercap counters and an operator-run
RAPL capture that executes outside the lab runner; loading and tampering the
retained synthetic energy/accuracy fixtures; gating when a retained NVML log
or RAPL capture may support a physical-domain finding; parsing timestamped
nvidia-smi sidecar rows; and a replay of the offline energy-accuracy
workflow through a ciw Session.

Non-claims: the bundled fixtures are synthetic. Validation of a sealed log
establishes internal integrity only; neither the declared ``origin`` nor the
device identity in a log authenticates that a physical device produced it.
The lab runner calls nothing here that reads an energy counter: counters are
read only by ``python -m ciw.lab.energy_gpu_telemetry rapl-capture`` (an
operator action) and by ``ciw energy record``. Nothing here starts GPU work,
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

from . import runner

FIXTURES = ("baseline", "reset", "missing", "under-target")
POWERCAP = Path("/sys/class/powercap")
LOG_ENV = "CIW_LAB_ENERGY_LOG"
SMI_ENV = "CIW_LAB_NVIDIA_SMI_CSV"
SMI_OFFSET_ENV = "CIW_LAB_NVIDIA_SMI_UTC_OFFSET"
RAPL_ENV = "CIW_LAB_RAPL_LOG"
RAPL_SCHEMA = "ciw.lab.rapl-capture.v1"
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


def capture_rapl(output, repeats=3, root=POWERCAP, separator=":", sleep=time.sleep) -> dict:
    """Operator-run capture: bracket ``repeats`` batches of the T115 workload, then an idle interval of equal length.

    This is the only function in the section that reads energy counters; the
    lab runner never calls it. The record is written once (an existing file is
    refused) and analyzed read-only by T115 through ``CIW_LAB_RAPL_LOG``.
    """
    from . import energy_gpu_kernels as kernels
    path = Path(output)
    if path.exists():
        raise ValueError("Refusing to overwrite an existing RAPL capture")
    if type(repeats) is not int or not 1 <= repeats <= 1000:
        raise ValueError("repeats must be an integer in [1, 1000]")
    domains = rapl_domains(root, separator)
    if not domains:
        raise ValueError("No intel-rapl package domain with an energy_uj counter")

    def work():
        for _ in range(repeats):
            kernels.fixed_step_batch()

    workload = _bracket(domains, work)
    idle = _bracket(domains, lambda: sleep(workload["elapsed_monotonic_ns"] / 1e9))
    record = {"schema": RAPL_SCHEMA, "captured_by": "python -m ciw.lab.energy_gpu_telemetry rapl-capture",
              "workload": dict(kernels.WORKLOAD), "repeats": repeats,
              "trajectories_per_repeat": len(kernels.HEADINGS), "host": rapl_host_identity(root, separator),
              "domains": domains, "workload_bracket": workload, "idle_bracket": idle}
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return record


def rapl_capture_reasons(record, host_identity: dict, workload: dict) -> list:
    """Why a capture cannot support a physical finding here (empty when it can)."""
    reasons = []
    if not isinstance(record, dict) or record.get("schema") != RAPL_SCHEMA:
        return [f"capture is not a {RAPL_SCHEMA} record"]
    if record.get("workload") != workload:
        reasons.append("capture names a different workload than this task declares")
    if type(record.get("repeats")) is not int or record["repeats"] < 1 or \
            record.get("trajectories_per_repeat") != len(workload["headings_rad"]):
        reasons.append("capture repeat or trajectory counts are malformed")
    host = record.get("host") or {}
    differing = [key for key in ("cpu_model", "machine", "system", "zones") if host.get(key) != host_identity.get(key)]
    if differing:
        reasons.append("capture host differs from this host: " + ", ".join(differing))
    domains = record.get("domains") or []
    for name in ("workload_bracket", "idle_bracket"):
        bracket = record.get(name) or {}
        if len(bracket.get("before_uj") or []) != len(domains) or len(bracket.get("after_uj") or []) != len(domains) \
                or not domains or not isinstance(bracket.get("elapsed_monotonic_ns"), int) \
                or bracket["elapsed_monotonic_ns"] <= 0 \
                or not all(str(bracket.get(key, "")).isdigit() for key in ("start_utc_ns", "end_utc_ns")):
            reasons.append(f"capture {name} is malformed")
    return reasons


def rapl_capture_energy(record) -> dict:
    """Gross and idle-subtracted package energy per trajectory (one wrap per bracket allowed)."""
    domains = record["domains"]

    def total(bracket):
        deltas = [rapl_delta_uj(int(b), int(a), d["max_energy_range_uj"])
                  for b, a, d in zip(bracket["before_uj"], bracket["after_uj"], domains)]
        return sum(delta for delta, _ in deltas), [wrapped for _, wrapped in deltas]

    gross_uj, wrapped = total(record["workload_bracket"])
    idle_uj, idle_wrapped = total(record["idle_bracket"])
    work_s = record["workload_bracket"]["elapsed_monotonic_ns"] / 1e9
    idle_s = record["idle_bracket"]["elapsed_monotonic_ns"] / 1e9
    trajectories = record["repeats"] * record["trajectories_per_repeat"]
    # Idle energy is rescaled to the workload interval before subtraction.
    idle_equivalent_uj = idle_uj * work_s / idle_s
    return {"trajectories": trajectories, "gross_uj": gross_uj, "idle_uj": idle_uj, "workload_s": work_s,
            "idle_s": idle_s, "wrapped": wrapped + idle_wrapped,
            "gross_j_per_trajectory": gross_uj * 1e-6 / trajectories,
            "idle_subtracted_j_per_trajectory": (gross_uj - idle_equivalent_uj) * 1e-6 / trajectories}


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


def raw_inventory(log) -> dict:
    """Count retained raw readings, the identity fields a log carries and how many hold placeholders."""
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
    return {"samples": len(samples), "timestamped_samples": len(timestamped),
            "raw_counter_readings": len(raw_counters), "batches": sum(len(p["batches"]) for p in log["phases"]),
            "identity_fields": identity, "identity_fields_present": sum(v not in (None, "") for v in identity.values()),
            "placeholder_values": sum(is_placeholder(v) for v in identity.values()),
            "placeholder_fields": sorted(k for k, v in identity.items() if is_placeholder(v)),
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


def physical_basis(raw: bytes, log, analysis, host_identity: dict | None,
                   required_name: str | None = None) -> tuple[dict, list]:
    """Acquisition basis for a physical finding, or a notes-only basis and the reasons it is withheld.

    A retained log supports a physical claim here only when it declares a
    physical measurement, its analysis is eligible, and its sensor identity
    (UUID, name, driver, NVML version and NVML library digest) equals the NVML
    identity of that device on this host; optionally the device name must
    contain ``required_name``. This binds the record to hardware and software
    present on this host. It does not authenticate that the operator's capture
    was genuine: no signature ties the readings to the device.
    """
    reasons = []
    if log["origin"] != "physical_measurement":
        reasons.append("log declares a synthetic fixture, not a physical measurement")
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
    """``python -m ciw.lab.energy_gpu_telemetry rapl-capture OUTPUT``: the operator-run T115 capture."""
    import argparse
    parser = argparse.ArgumentParser(prog="python -m ciw.lab.energy_gpu_telemetry",
                                     description="Operator-run RAPL capture for lab task T115 (outside the lab runner)")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("rapl-capture", help="bracket the T115 workload and an equal idle interval")
    capture.add_argument("output", help="new JSON file for the raw capture (an existing file is refused)")
    capture.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        record = capture_rapl(args.output, args.repeats)
    except (OSError, ValueError) as exc:
        print(f"RAPL capture refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "zones": record["host"]["zones"],
                      "then": f"{RAPL_ENV}={args.output} ciw lab run T115 --output-dir <dir>"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
