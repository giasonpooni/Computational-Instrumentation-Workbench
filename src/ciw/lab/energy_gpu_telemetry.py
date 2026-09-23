"""Telemetry helpers for the energy and GPU experiments (T115-T125).

Scope: read-only probes of Linux RAPL powercap counters; loading and
tampering the retained synthetic energy/accuracy fixtures; gating when a
retained NVML log may support a physical-domain finding; and a replay of the
offline energy-accuracy workflow through a ciw Session.

Non-claims: the bundled fixtures are synthetic. Validation of a sealed log
establishes internal integrity only; neither the declared ``origin`` nor the
device identity in a log authenticates that a physical device produced it.
Nothing here starts GPU work, changes device settings or estimates energy
from time or utilization.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "examples" / "energy-accuracy"
FIXTURES = ("baseline", "reset", "missing", "under-target")
POWERCAP = Path("/sys/class/powercap")
LOG_ENV = "CIW_LAB_ENERGY_LOG"
SMI_ENV = "CIW_LAB_NVIDIA_SMI_CSV"


# RAPL ---------------------------------------------------------------------
def rapl_domains(root=POWERCAP) -> list:
    """Top-level package domains ``intel-rapl:N`` with their name and wrap range."""
    domains = []
    for directory in sorted(Path(root).glob("intel-rapl:*")):
        if directory.name.count(":") != 1 or not (directory / "energy_uj").is_file():
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


# Fixtures -------------------------------------------------------------------
def fixture_bytes() -> dict | None:
    """Exact retained bytes of the synthetic fixtures, or None outside a source checkout."""
    paths = {name: FIXTURE_DIR / f"{name}.json" for name in FIXTURES}
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


def raw_inventory(log) -> dict:
    """Count retained raw readings and the identity fields a log carries."""
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
            "origin": log["origin"], "log_digest": log["log_digest"]}


# Operator-captured NVML logs (GPU host only) --------------------------------
def gpu_listing() -> str:
    """``nvidia-smi -L`` output of this host (device names and UUIDs), or empty text."""
    try:
        done = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout if done.returncode == 0 else ""


def _utc(ns_decimal: str) -> str:
    seconds, remainder = divmod(int(ns_decimal), 10**9)
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{stamp}.{remainder:09d}Z"


def physical_basis(raw: bytes, log, analysis, listing: str, required_name: str | None = None) -> tuple[dict, list]:
    """Acquisition basis for a physical finding, or an empty basis and the reasons it is withheld.

    A retained log supports a physical claim here only when it declares a
    physical measurement, its analysis is eligible, its sensor UUID is a device
    present on this host, and (optionally) the device has the required name.
    These checks bind the record to hardware on this host; they do not
    authenticate that the operator's capture was genuine.
    """
    reasons = []
    if log["origin"] != "physical_measurement":
        reasons.append("log declares a synthetic fixture, not a physical measurement")
    if not analysis["comparison"]["eligible"]:
        reasons.extend(analysis["comparison"]["reasons"] or ["analysis is not eligible for comparison"])
    uuid = log["sensor"]["device_uuid"].lower()
    if uuid not in listing.lower():
        reasons.append("sensor device UUID is not present on this host")
    if required_name and required_name.lower() not in log["sensor"]["name"].lower():
        reasons.append(f"sensor device is not an {required_name}")
    if reasons:
        return {"notes": reasons}, reasons
    samples = log["phases"][3]["samples"]
    acquisition = {"device": f"{log['sensor']['name']} {log['sensor']['device_uuid']} (NVML total energy counter)",
                   "raw_sha256": hashlib.sha256(raw).hexdigest(), "acquired_at": _utc(samples[0]["utc_start_ns"]),
                   "calibration": "not_applied: vendor counter, resolution and accuracy undeclared"}
    return {"acquisition": acquisition}, []


def read_operator_log(path) -> tuple[bytes, dict, dict]:
    from ciw import energy_records
    raw = Path(path).read_bytes()
    if len(raw) > energy_records.MAX_BYTES:
        raise ValueError("Operator energy log exceeds its byte budget")
    log = json.loads(raw)
    return raw, log, energy_records.analyze(log)


def smi_columns(text: str) -> dict:
    """Parse ``nvidia-smi --query-gpu=... --format=csv,nounits`` text into numeric columns."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return {}
    header = [name.strip() for name in lines[0].split(",")]
    columns = {name: [] for name in header}
    for line in lines[1:]:
        for name, cell in zip(header, line.split(",")):
            try:
                columns[name].append(float(cell.strip()))
            except ValueError:
                columns[name].append(None)
    return columns


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
                # A retained bundle whose numerical data was edited cannot be replayed.
                tampered = deepcopy(first)
                tampered["steps"][0]["result"]["data"]["measurement"]["gross_energy_j"] = 99.0
                try:
                    EnergyAccuracyWorkflow().replay_session(tampered, {})
                    refusals["edited_bundle"] = None
                except ValueError as exc:
                    refusals["edited_bundle"] = str(exc)
        saved = call("workspace.save", {})
        restored = Session.from_workspace(Path(saved["workspace_file"]), Path(scratch) / "restored")
        restored_equal = restored.workbench.serialize() == session.workbench.serialize()
        bundle_count = len(session.workbench.serialize()["bundles"])
    return {"fixtures": rows, "refusals": refusals, "restored_equal": restored_equal,
            "bundle_count": bundle_count, "baseline_view": views.get("baseline")}


def environment_log_path(variable=LOG_ENV) -> str | None:
    value = os.environ.get(variable, "").strip()
    return value or None
