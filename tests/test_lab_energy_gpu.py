"""Energy and GPU experiments T115-T125: labels, key numbers and refusals.

No test reads hardware counters or starts GPU work. Hardware probes are forced
off (or simulated) so every assertion is deterministic; RAPL counters and NVML
logs are simulated files. The Rust comparison is skipped when rustc cannot
build the kernel.
"""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import platform
import sys

import numpy as np
import pytest

from ciw import energy_records
from ciw.lab import energy_gpu, energy_gpu_kernels as kernels, energy_gpu_telemetry as telemetry, runner
from ciw.lab.evidence import COMPUTATIONAL_DOMAINS, PHYSICAL_DOMAINS, EvidenceRefusal, supported_label
from ciw.lab.registry import load_queue, section_implementations
from ciw.lab.report import validate_report
from ciw.telemetry import canonical

QUEUE = {t["id"]: t for t in load_queue()["tasks"]}
IMPLEMENTATIONS = section_implementations("energy-gpu")
TASK_IDS = [f"T{number}" for number in range(115, 126)]
ENVIRONMENT = (telemetry.LOG_ENV, telemetry.SMI_ENV, telemetry.SMI_OFFSET_ENV, telemetry.RAPL_ENV)
needs_fixtures = pytest.mark.skipif(telemetry.fixture_bytes() is None, reason="repository fixtures unreachable")
# The runner's real probe, captured before the module-scoped fixture forces probes off.
REAL_PROBE_HARDWARE = runner._probe_hardware


def run(task_id, context):
    return validate_report(runner.run_task(QUEUE[task_id], IMPLEMENTATIONS[task_id], context, {}))


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    """Run tasks through the runner with one shared context and hardware probes forced off."""
    context = runner.Context(tmp_path_factory.mktemp("lab-energy-gpu"))
    patch = pytest.MonkeyPatch()
    patch.setattr(runner, "_probe_hardware", lambda name: False)
    for variable in ENVIRONMENT:
        patch.delenv(variable, raising=False)
    reports = {}

    def get(task_id):
        if task_id not in reports:
            reports[task_id] = run(task_id, context)
        return reports[task_id]

    get.context = context
    yield get
    patch.undo()


@pytest.fixture
def clean_environment(monkeypatch):
    for variable in ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)
    return monkeypatch


def by_claim(report, prefix):
    matches = [f for f in report["findings"] if f["claim"].startswith(prefix)]
    assert len(matches) == 1, prefix
    return matches[0]


def physical_labels(report):
    return {f["evidence_status"] for f in report["findings"] if f["domain"] in PHYSICAL_DOMAINS}


def artifact(report, context, name):
    entry = next(a for a in report["generated_artifacts"] if a["path"].endswith("/" + name))
    return entry, (context.output_dir / entry["path"]).read_bytes()


def relabelled_log(name=None):
    """The baseline fixture declared as a physical measurement (optionally renamed) and resealed."""
    log = json.loads(telemetry.fixture_bytes()["baseline"])
    sensor = dict(log["sensor"], name=name) if name else log["sensor"]
    return telemetry.reseal(dict(log, origin="physical_measurement", sensor=sensor))


def simulate_gpu_host(monkeypatch, tmp_path, log, smi=None, offset=None):
    """A host whose nvidia-gpu probe answers and whose NVML identity equals the log's sensor block."""
    monkeypatch.setattr(runner, "_probe_hardware", lambda name: name == "nvidia-gpu")
    monkeypatch.setattr(telemetry, "host_sensor_identity", lambda uuid: dict(log["sensor"]))
    path = tmp_path / "log.json"
    path.write_bytes(canonical(log))
    monkeypatch.setenv(telemetry.LOG_ENV, str(path))
    if smi is not None:
        (tmp_path / "smi.csv").write_bytes(smi)
        monkeypatch.setenv(telemetry.SMI_ENV, str(tmp_path / "smi.csv"))
    if offset is not None:
        monkeypatch.setenv(telemetry.SMI_OFFSET_ENV, offset)
    return runner.Context(tmp_path / "out")


def test_every_section_task_is_registered_with_tests():
    assert sorted(IMPLEMENTATIONS) == TASK_IDS
    for implementation in IMPLEMENTATIONS.values():
        assert implementation.regression_tests
        assert all(node.startswith("tests/test_lab_energy_gpu.py::") for node in implementation.regression_tests)


def test_every_numerical_finding_declares_uncertainty_and_tolerance(lab):
    for task_id in TASK_IDS:
        for record in lab(task_id)["findings"]:
            if record["domain"] in COMPUTATIONAL_DOMAINS and record["evidence_status"] != "not_established":
                uncertainty = record["uncertainty"]
                assert uncertainty is not None, (task_id, record["claim"])
                # A declared uncertainty states a number, not only a kind.
                assert not isinstance(uncertainty, dict) or uncertainty.get("value") is not None, (task_id, record["claim"])
                if record["evidence_status"] != "analytic" and isinstance(record["value"], (int, float, list, dict)):
                    assert "regression_tolerance" in record or record["domain"] != "numerical", (task_id, record["claim"])
                # rel >= 1 would accept any decrease down to zero: a one-sided vacuous regression tolerance.
                assert record.get("regression_tolerance", {}).get("rel", 0.0) < 1.0, (task_id, record["claim"])


def test_cpu_energy_task_counts_work_and_leaves_energy_unestablished(lab):
    report = lab("T115")
    assert report["state"] == "partial"
    work = by_claim(report, "Fixed-step RK4 spends exactly")
    assert work["value"] == [4 * energy_gpu.STEPS] * len(kernels.HEADINGS)
    assert work["evidence_status"] == "numerically_verified"
    accepted = by_claim(report, "The fixed-step trajectories are accepted")
    assert accepted["evidence_status"] == "numerically_verified" and accepted["value"] < 1e-8
    for claim in (energy_gpu.GROSS_CPU, energy_gpu.IDLE_CPU):
        energy = by_claim(report, claim)
        assert energy["domain"] == "physical" and energy["evidence_status"] == "not_established"
        assert energy["value"] is None
    assert report["physical_validation_status"]["status"] == "not_established"
    names = {a["path"].rsplit("/", 1)[-1] for a in report["generated_artifacts"]}
    assert {"work-proxies.json", "timing.json", "rapl-probe.json"} <= names and "rapl-capture.json" not in names
    # Elapsed time is retained as an artifact, never as a finding.
    assert not any("time" in f["claim"].lower() for f in report["findings"])


def test_lab_run_acquires_no_energy_measurement(tmp_path, clean_environment):
    """Even where RAPL answers the probe, T115 reads no counter without an operator capture."""
    def refuse(domains):
        raise AssertionError("the lab runner read an energy counter")

    clean_environment.setattr(runner, "_probe_hardware", lambda name: True)
    clean_environment.setattr(telemetry, "rapl_read", refuse)
    report = run("T115", runner.Context(tmp_path))
    assert report["state"] == "partial" and physical_labels(report) == {"not_established"}
    assert "no RAPL capture was supplied" in by_claim(report, energy_gpu.GROSS_CPU)["basis"]["notes"][0]


def simulated_rapl_capture(tmp_path, monkeypatch, host):
    """An operator capture made with simulated counters (1 -> 10 J over the workload, 0.5 J over the idle bracket)."""
    domain = {"zone": "intel-rapl:0", "name": "package-0", "path": "unused", "max_energy_range_uj": 262143328850}
    readings = iter([[1_000_000], [10_000_000], [10_000_000], [10_500_000]])
    with monkeypatch.context() as patch:
        patch.setattr(telemetry, "rapl_domains", lambda root=None, separator=":": [domain])
        patch.setattr(telemetry, "rapl_read", lambda domains: next(readings))
        patch.setattr(telemetry, "rapl_host_identity", lambda root=None, separator=":": dict(host))
        record = telemetry.capture_rapl(tmp_path / "captured.json", repeats=1, sleep=lambda seconds: None)
    # Fix the brackets' durations so the idle rescaling is deterministic.
    for bracket in ("workload_bracket", "idle_bracket"):
        record[bracket]["elapsed_monotonic_ns"] = 400_000_000
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps(record), encoding="utf-8")
    return record, capture


@pytest.mark.skipif(sys.platform == "win32", reason="powercap zone directories contain ':'")
def test_rapl_probe_is_the_only_counter_read(tmp_path, clean_environment):
    """The runner's real hardware:rapl probe on a simulated powercap tree is T115's only counter access.

    Without a capture nothing reads energy_uj; with one, the probe reads the
    simulated counter to confirm readability and its value enters no result.
    """
    monkeypatch = clean_environment
    zone = tmp_path / "powercap" / "intel-rapl:0"
    zone.mkdir(parents=True)
    (zone / "energy_uj").write_text("987654321\n", encoding="utf-8")
    (zone / "name").write_text("package-0\n", encoding="utf-8")
    monkeypatch.setattr(runner, "POWERCAP", tmp_path / "powercap")
    monkeypatch.setattr(runner, "_probe_hardware", REAL_PROBE_HARDWARE)
    reads = []
    read_text = Path.read_text

    def spy(self, *args, **kwargs):
        if self.name == "energy_uj":
            reads.append(self)
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    report = run("T115", runner.Context(tmp_path / "no-capture"))
    assert reads == [] and physical_labels(report) == {"not_established"}
    host = {"cpu_model": "Simulated CPU", "machine": "x86_64", "system": "Linux", "zones": ["intel-rapl:0 package-0"]}
    record, capture = simulated_rapl_capture(tmp_path, monkeypatch, host)
    monkeypatch.setattr(telemetry, "rapl_host_identity", lambda root=None, separator=":": dict(host))
    monkeypatch.setenv(telemetry.RAPL_ENV, str(capture))
    reads.clear()
    report = run("T115", runner.Context(tmp_path / "capture"))
    assert reads == [zone / "energy_uj"]  # the probe, once
    assert report["provider_runtime_identity"]["requirement_probes"]["hardware:rapl"] is True
    gross = by_claim(report, energy_gpu.GROSS_CPU)
    assert gross["evidence_status"] == "hardware_measured"
    assert gross["value"] == pytest.approx(9.0 / len(kernels.HEADINGS))  # from the capture, not the probed counter
    assert "987654321" not in json.dumps(report)


def test_rapl_helpers_read_counters_and_one_wrap(tmp_path):
    # '_' stands in for ':' so the tree can be built on Windows too.
    for zone, name, value in (("intel-rapl_0", "package-0", 1000), ("intel-rapl_0_0", "core", 5)):
        directory = tmp_path / zone
        directory.mkdir()
        (directory / "energy_uj").write_text(f"{value}\n", encoding="utf-8")
        (directory / "name").write_text(name + "\n", encoding="utf-8")
        (directory / "max_energy_range_uj").write_text("262143328850\n", encoding="utf-8")
    domains = telemetry.rapl_domains(tmp_path, separator="_")
    assert [d["name"] for d in domains] == ["package-0"]  # subzones are inside their package
    assert telemetry.rapl_read(domains) == [1000]
    assert telemetry.rapl_delta_uj(1000, 1500, 262143328850) == (500, False)
    assert telemetry.rapl_delta_uj(262143328800, 49, 262143328850) == (100, True)
    with pytest.raises(ValueError, match="without a declared wrap range"):
        telemetry.rapl_delta_uj(10, 5, None)


def test_rapl_capture_is_analyzed_read_only_and_gated(tmp_path, clean_environment):
    """A simulated operator capture: gross and idle-subtracted energy, retained raw bytes, identity gate."""
    monkeypatch = clean_environment
    host = {"cpu_model": "Simulated CPU", "machine": "x86_64", "system": "Linux", "zones": ["intel-rapl:0 package-0"]}
    record, capture = simulated_rapl_capture(tmp_path, monkeypatch, host)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        telemetry.capture_rapl(tmp_path / "captured.json", repeats=1)
    monkeypatch.setattr(telemetry, "rapl_host_identity", lambda root=None, separator=":": dict(host))
    monkeypatch.setenv(telemetry.RAPL_ENV, str(capture))
    monkeypatch.setattr(runner, "_probe_hardware", lambda name: name == "rapl")
    context = runner.Context(tmp_path / "out")
    report = run("T115", context)
    gross, idle = by_claim(report, energy_gpu.GROSS_CPU), by_claim(report, energy_gpu.IDLE_CPU)
    trajectories = len(kernels.HEADINGS)
    assert gross["value"] == pytest.approx(9.0 / trajectories) and idle["value"] == pytest.approx(8.5 / trajectories)
    assert {gross["evidence_status"], idle["evidence_status"]} == {"hardware_measured"}
    assert report["state"] == "completed"
    acquisition = gross["basis"]["acquisition"]
    assert acquisition["calibration"].startswith("not_applied") and "unauthenticated" in acquisition["device"]
    entry, raw = artifact(report, context, "rapl-capture.json")
    assert raw == capture.read_bytes() and entry["sha256"] == acquisition["raw_sha256"]
    # The gate withholds the label for another host, another workload or a host whose probe fails.
    monkeypatch.setattr(telemetry, "rapl_host_identity", lambda root=None, separator=":": dict(host, cpu_model="Other"))
    other_host = run("T115", runner.Context(tmp_path / "other-host"))
    assert physical_labels(other_host) == {"not_established"} and other_host["state"] == "partial"
    assert "cpu_model" in by_claim(other_host, energy_gpu.GROSS_CPU)["basis"]["notes"][0]
    monkeypatch.setattr(telemetry, "rapl_host_identity", lambda root=None, separator=":": dict(host))
    capture.write_text(json.dumps(dict(record, workload=dict(record["workload"], steps=128))), encoding="utf-8")
    assert physical_labels(run("T115", runner.Context(tmp_path / "other-workload"))) == {"not_established"}
    capture.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(runner, "_probe_hardware", lambda name: False)
    assert physical_labels(run("T115", runner.Context(tmp_path / "no-probe"))) == {"not_established"}


def test_gpu_tasks_are_blocked_with_the_recording_protocol(lab, tmp_path, clean_environment):
    for task_id in ("T116", "T118"):
        report = lab(task_id)
        assert report["state"] == "blocked" and report["findings"]
        assert all(f["domain"] in PHYSICAL_DOMAINS and f["evidence_status"] == "not_established"
                   for f in report["findings"])
        assert "hardware:nvidia-gpu" in report["experiment"]
        assert "ciw energy record --problem examples/energy-accuracy/problem.json" in report["experiment"]
        assert report["physical_validation_status"]["status"] == "not_established"
        assert report["numerical_result"].startswith("none")
    t118 = lab("T118")
    assert "TZ=UTC nvidia-smi --query-gpu" in t118["experiment"] and "nsys" in t118["experiment"]
    assert telemetry.SMI_OFFSET_ENV in t118["experiment"]
    # Kernel-only duration is a named deferred question, not another task's job.
    assert "nsys stats --report cuda_gpu_kern_sum" in t118["recommended_next_task"]
    assert "T147" not in t118["recommended_next_task"]
    # T116's protocol does not run T118 without the sidecar it needs.
    t116_run = lab("T116")["experiment"].split("(4)")[1]
    assert "ciw lab run T116 T119" in t116_run and "T118 additionally needs the nvidia-smi sidecar" in t116_run
    assert {f["claim"] for f in t118["findings"]} == set(energy_gpu.T118_UNITS)
    assert {f["claim"] for f in lab("T116")["findings"]} == {energy_gpu.GPU_ENERGY, energy_gpu.NVML_ACCURACY}
    # A GPU host without an operator log, or with an unreadable one, blocks with the same claims.
    clean_environment.setattr(runner, "_probe_hardware", lambda name: name == "nvidia-gpu")
    no_log = run("T116", runner.Context(tmp_path / "no-log"))
    assert no_log["state"] == "blocked" and "never acquires hardware data" in no_log["experiment"]
    assert [f["claim"] for f in no_log["findings"]] == [f["claim"] for f in lab("T116")["findings"]]
    (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
    clean_environment.setenv(telemetry.LOG_ENV, str(tmp_path / "bad.json"))
    invalid = run("T118", runner.Context(tmp_path / "bad-log"))
    assert invalid["state"] == "blocked" and "unreadable or invalid" in invalid["experiment"]
    assert physical_labels(invalid) == {"not_established"}


@needs_fixtures
def test_operator_log_gate_withholds_physical_labels_from_synthetic_logs(tmp_path):
    raw = telemetry.fixture_bytes()["baseline"]
    log = json.loads(raw)
    context = runner.Context(tmp_path)
    context.begin("T116")
    matching = dict(log["sensor"])
    synthetic = energy_gpu.operator_log_outcome(context, "T116", raw, log, energy_records.analyze(log), matching)
    assert synthetic["state"] == "partial"
    assert physical_labels(synthetic) == {"not_established"}
    assert "synthetic fixture" in " ".join(synthetic["fields"]["unresolved_assumptions"])
    # The operator path emits exactly the physical claims the blocked plan records.
    assert {f["claim"] for f in synthetic["findings"] if f["domain"] in PHYSICAL_DOMAINS} == \
        {f["claim"] for f in energy_gpu.PLAN_T116["findings"]}
    # A relabelled log is still withheld when this host's NVML identity differs or the device is not an RTX 2080.
    relabelled = relabelled_log()
    relabelled_raw = canonical(relabelled)
    analysis = energy_records.analyze(relabelled)
    absent = energy_gpu.operator_log_outcome(context, "T116", relabelled_raw, relabelled, analysis, None)
    assert physical_labels(absent) == {"not_established"}
    other_driver = dict(matching, driver_version="host-driver", library_sha256="e" * 64)
    differing = energy_gpu.operator_log_outcome(context, "T116", relabelled_raw, relabelled, analysis, other_driver)
    assert any(item.endswith("driver_version, library_sha256") for item in differing["fields"]["unresolved_assumptions"])
    assert physical_labels(differing) == {"not_established"}
    context.begin("T118")
    rtx = energy_gpu.operator_log_outcome(context, "T118", relabelled_raw, relabelled, analysis, matching)
    assert physical_labels(rtx) == {"not_established"}
    assert any("not an RTX 2080" in item for item in rtx["fields"]["unresolved_assumptions"])
    assert {f["claim"] for f in rtx["findings"] if f["domain"] in PHYSICAL_DOMAINS} == set(energy_gpu.T118_UNITS)
    kernel = by_claim(rtx, energy_gpu.KERNEL_ONLY)
    assert kernel["value"] is None and kernel["evidence_status"] == "not_established"


@needs_fixtures
def test_operator_log_gate_trust_boundary_is_the_host_identity(tmp_path, clean_environment):
    """Through the runner: the gate binds a declared physical log to this host's NVML identity, nothing more.

    A synthetic fixture relabelled as a physical measurement and paired with a
    host identity equal to its sensor block passes the gate. This is the
    documented trust boundary (see the T124 origin-relabelling counterexample);
    the acquisition record says the log is operator-captured and unauthenticated.
    """
    log = relabelled_log()
    context = simulate_gpu_host(clean_environment, tmp_path, log)
    report = run("T116", context)
    energy = by_claim(report, energy_gpu.GPU_ENERGY)
    assert energy["evidence_status"] == "hardware_measured" and energy["value"] == pytest.approx(0.2)
    acquisition = energy["basis"]["acquisition"]
    assert acquisition["calibration"].startswith("not_applied") and "unauthenticated" in acquisition["device"]
    entry, raw = artifact(report, context, "operator-log.json")
    assert raw == canonical(log) and entry["sha256"] == acquisition["raw_sha256"]
    assert by_claim(report, energy_gpu.NVML_ACCURACY)["evidence_status"] == "not_established"
    assert report["state"] == "completed"
    # The pipeline finding recomputes the counter delta and KL from raw data, not by re-running the analysis.
    pipeline = by_claim(report, "The operator log's measurement counter delta and batch KL values recompute")
    assert pipeline["evidence_status"] == "numerically_verified"
    assert pipeline["value"]["recomputed_delta_j"] == pytest.approx(0.2)
    assert [c["reference_kind"] for c in pipeline["basis"]["checks"]] == ["cross_implementation"] * 2


@needs_fixtures
def test_operator_log_pipeline_check_detects_a_disagreeing_analysis(tmp_path):
    """The recomputation can fail: an analysis whose gross energy differs from the raw readings is refuted."""
    log = relabelled_log()
    raw = canonical(log)
    analysis = energy_records.analyze(log)
    analysis["measurement"]["gross_energy_j"] = 0.3
    context = runner.Context(tmp_path)
    context.begin("T116")
    outcome = energy_gpu.operator_log_outcome(context, "T116", raw, log, analysis, dict(log["sensor"]))
    pipeline = by_claim(outcome, "The operator log's measurement counter delta and batch KL values recompute")
    assert pipeline["evidence_status"] == "not_established" and outcome["state"] == "partial"


def smi_csv(start_utc, rows, offset_hours, newline):
    """nvidia-smi style rows every 100 ms in local time; utilization 97 inside [0.6 s, 1.6 s] of the fixture clock."""
    header = "timestamp, uuid, name, utilization.gpu [%], temperature.gpu, power.draw [W]"
    lines = [header]
    for index in range(rows):
        utc = start_utc + timedelta(milliseconds=100 * index)
        local = utc + timedelta(hours=offset_hours)
        busy = datetime(2001, 9, 9, 1, 46, 40, 600000) <= utc <= datetime(2001, 9, 9, 1, 46, 41, 600000)
        lines.append(f"{local.strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]}, GPU-01234567-89ab-cdef-0123-456789abcdef, "
                     f"NVIDIA GeForce RTX 2080, {97 if busy else 0}, 40, 100.0")
    return (newline.join(lines) + newline).encode("utf-8")


@needs_fixtures
def test_t118_sidecar_rows_are_restricted_to_the_measurement_window(tmp_path, clean_environment):
    log = relabelled_log("NVIDIA GeForce RTX 2080")
    # Fixture UTC clock: measurement readings at 1000000000.6 s and 1000000001.6 s (2001-09-09T01:46:40Z + ...).
    smi = smi_csv(datetime(2001, 9, 9, 1, 46, 40), 25, offset_hours=2, newline="\r\n")
    context = simulate_gpu_host(clean_environment, tmp_path, log, smi=smi, offset="+02:00")
    report = run("T118", context)
    utilization = by_claim(report, energy_gpu.UTILIZATION)
    assert utilization["evidence_status"] == "hardware_measured"
    assert utilization["value"] == {"min": 97.0, "mean": 97.0, "max": 97.0, "count": 10}
    entry, raw = artifact(report, context, "nvidia-smi.csv")
    assert raw == smi and utilization["basis"]["acquisition"]["raw_sha256"] == hashlib.sha256(smi).hexdigest()
    assert entry["sha256"] == hashlib.sha256(smi).hexdigest()  # CRLF bytes retained exactly
    for claim in (energy_gpu.POWER, energy_gpu.POWER_STEADY, energy_gpu.TEMPERATURE_STEADY):
        assert by_claim(report, claim)["evidence_status"] == "hardware_measured"
    assert by_claim(report, energy_gpu.POWER_STEADY)["value"] == 0.0
    assert by_claim(report, energy_gpu.KERNEL_ONLY)["evidence_status"] == "not_established"
    assert report["state"] == "partial"
    # Without a declared UTC offset the sidecar statistic is withheld; the NVML findings are unaffected.
    clean_environment.delenv(telemetry.SMI_OFFSET_ENV)
    undeclared = run("T118", runner.Context(tmp_path / "no-offset"))
    assert by_claim(undeclared, energy_gpu.UTILIZATION)["evidence_status"] == "not_established"
    assert by_claim(undeclared, energy_gpu.POWER)["evidence_status"] == "hardware_measured"
    # A sidecar recorded entirely outside the window yields no in-window rows.
    late = smi_csv(datetime(2001, 9, 9, 1, 47, 0), 5, offset_hours=0, newline="\n")
    assert telemetry.smi_window(late.decode(), 1000000000600000380, 1000000001600000420, 0)[1] == \
        ["no sidecar row falls inside the measurement window"]


def rust_identity(tmp_path):
    try:
        return kernels.build_rust_kernel(tmp_path)
    except kernels.NativeKernelUnavailable as exc:
        pytest.skip(f"Rust kernel unavailable: {exc}")


def test_rust_kernel_matches_python_kernel(lab, tmp_path):
    identity = rust_identity(tmp_path)
    report = lab("T117")
    assert report["state"] == "partial"
    agreement = by_claim(report, energy_gpu.AGREE)
    assert agreement["evidence_status"] == "numerically_verified" and agreement["value"] <= 1e-12
    # Endpoint agreement carries only the endpoint comparison; the call count is its own finding.
    assert [check["reference_kind"] for check in agreement["basis"]["checks"]] == ["cross_implementation"]
    count = by_claim(report, energy_gpu.RUST_COUNT)
    assert count["evidence_status"] == "numerically_verified"
    assert count["value"] == 4 * energy_gpu.STEPS * len(kernels.HEADINGS)
    assert by_claim(report, energy_gpu.RUST_ERROR)["value"] < 1e-8
    refusals = by_claim(report, energy_gpu.RUST_REFUSES)
    assert refusals["evidence_status"] == "numerically_verified"
    assert refusals["value"]["nonfinite"].endswith("nonfinite state")
    assert identity["source_sha256"] == kernels.rust_source_sha256()
    rust = report["provider_runtime_identity"]["rust"]
    # The build is reproducible: the scratch path is remapped, so another directory gives the same binary.
    assert rust["binary_sha256"] == identity["binary_sha256"] and rust["rustc"] == identity["rustc"]
    # The Rust rhs() counts its own calls.
    result = kernels.run_rust_kernel(identity["executable"], kernels.initial_states()[:2], 1.0, 3)
    assert result["evaluations"] == 4 * 3 * 2
    with pytest.raises(kernels.NativeKernelUnavailable, match="steps must be an integer"):
        kernels.run_rust_kernel(identity["executable"], [[1.0, 0.0, 1.0, 0.0]], 1.0, 0)


def test_cross_language_agreement_is_not_independent(lab, tmp_path, clean_environment):
    # Declaring the Rust kernel an independent checker of the Python kernel is refused: both are ciw code.
    check = {"reference_kind": "high_precision", "reference": "Python RK4 endpoint", "observed": 0.0,
             "tolerance": 1e-12, "comparison": "abs_le", "passed": True,
             "producer": {"implementation": "ciw.lab.integrators", "revision": "working-tree"},
             "checker": {"implementation": "ciw.lab.energy_gpu_kernels.RUST_SOURCE",
                         "revision": kernels.rust_source_sha256()}}
    with pytest.raises(EvidenceRefusal, match="share an implementation origin"):
        supported_label({"independent_check": check}, "numerical")
    report = lab("T117")
    assert all(f["evidence_status"] != "independently_verified" for f in report["findings"])
    assert not any(f["claim"].startswith("Declaring the Rust kernel") for f in report["findings"])
    for prefix in ("A Julia implementation", "A GPU implementation"):
        assert by_claim(report, prefix)["evidence_status"] == "not_established"
        assert "was written" in by_claim(report, prefix)["basis"]["notes"][0]
    # The missing Julia and GPU kernels are stated as such, not blamed on this host.
    assert energy_gpu.NOT_WRITTEN in report["unresolved_assumptions"]
    assert "T147" not in report["recommended_next_task"] and "Deferred research question" in report["recommended_next_task"]
    assert "no NVIDIA GPU answered" in by_claim(report, "A GPU implementation")["basis"]["notes"][0]
    # On a GPU host with julia on PATH the notes follow the probes and still say no kernel exists.
    clean_environment.setattr(runner, "_probe_hardware", lambda name: name == "nvidia-gpu")
    clean_environment.setattr(runner.shutil, "which", lambda name: "/usr/bin/julia" if name == "julia" else None)
    gpu_host = run("T117", runner.Context(tmp_path))
    gpu_note = by_claim(gpu_host, "A GPU implementation")["basis"]["notes"][0]
    assert "an NVIDIA GPU answered" in gpu_note and "no GPU implementation of this kernel was written" in gpu_note
    assert "julia is on PATH here" in by_claim(gpu_host, "A Julia implementation")["basis"]["notes"][0]
    assert gpu_host["provider_runtime_identity"]["requirement_probes"] == {"tool:julia": True,
                                                                          "hardware:nvidia-gpu": True}
    generic = by_claim(report, "Generic Christoffel-symbol RK4")
    assert generic["evidence_status"] == "numerically_verified" and generic["value"] <= 1e-12
    assert generic["basis"]["checks"][0]["reference_kind"] == "cross_implementation"
    assert physical_labels(report) == {"not_established"}
    # Units do not depend on whether rustc is present.
    assert by_claim(report, energy_gpu.AGREE)["unit"] == energy_gpu.ANGLE_UNIT


@needs_fixtures
def test_energy_per_accepted_result_on_fixtures(lab):
    report = lab("T119")
    assert report["state"] == "partial"
    metric = by_claim(report, "Energy per accepted replica solve of the synthetic baseline")
    assert metric["value"] == pytest.approx(0.05, abs=1e-15) and metric["evidence_status"] == "numerically_verified"
    assert metric["basis"]["checks"][1]["observed"] <= 1e-12  # textbook KL agrees with energy_records
    distinct = by_claim(report, "Energy per distinct accepted result")
    assert distinct["value"] == pytest.approx(0.2) and distinct["evidence_status"] == "numerically_verified"
    assert distinct["basis"]["checks"][0]["reference"].startswith("distinct binary64 byte strings")
    fixtures = telemetry.fixture_bytes()
    assert distinct["basis"]["generator"]["fixture_sha256"] == {
        name: hashlib.sha256(raw).hexdigest() for name, raw in fixtures.items()}
    withheld = by_claim(report, "The metric is withheld")
    assert sorted(withheld["value"]) == ["missing", "reset", "under-target"]
    assert [c["observed"] for c in withheld["basis"]["checks"]] == [0.0, 0.0]
    naive = by_claim(report, "Dividing gross energy by executed solves")
    assert naive["value"]["accepted_solves"] == 0 and naive["value"]["naive_j_per_solve"] == pytest.approx(0.05)
    assert "counterexample" in naive
    assert by_claim(report, "Widening the boundary")["value"] == pytest.approx(7.0)
    physical = by_claim(report, energy_gpu.T119_PHYSICAL)
    assert physical["evidence_status"] == "not_established" and "no operator NVML log" in physical["basis"]["notes"][0]
    assert "ciw lab run T116 T119" in report["recommended_next_task"]


@needs_fixtures
def test_t119_operator_log_is_gated_like_t116(tmp_path, clean_environment):
    """On a (simulated) GPU host T119 measures energy per accepted solve from the operator log, or withholds it."""
    log = relabelled_log()
    context = simulate_gpu_host(clean_environment, tmp_path, log)
    report = run("T119", context)
    physical = by_claim(report, energy_gpu.T119_PHYSICAL)
    assert physical["evidence_status"] == "hardware_measured" and physical["value"] == pytest.approx(0.05)
    assert [c["reference_kind"] for c in physical["basis"]["checks"]] == ["cross_implementation"] * 2
    entry, raw = artifact(report, context, "operator-log.json")
    assert raw == canonical(log) and entry["sha256"] == physical["basis"]["acquisition"]["raw_sha256"]
    assert report["state"] == "completed"
    # Without a GPU answering the probe, or for a log declaring a synthetic fixture, the value is withheld.
    clean_environment.setattr(runner, "_probe_hardware", lambda name: False)
    no_gpu = run("T119", runner.Context(tmp_path / "no-gpu"))
    assert by_claim(no_gpu, energy_gpu.T119_PHYSICAL)["evidence_status"] == "not_established"
    assert no_gpu["state"] == "partial"
    assert any("no NVIDIA GPU answered" in item for item in no_gpu["unresolved_assumptions"])
    synthetic = json.loads(telemetry.fixture_bytes()["baseline"])
    (tmp_path / "synthetic").mkdir()
    fixture_log = run("T119", simulate_gpu_host(clean_environment, tmp_path / "synthetic", synthetic))
    assert by_claim(fixture_log, energy_gpu.T119_PHYSICAL)["evidence_status"] == "not_established"
    assert any("synthetic fixture" in item for item in fixture_log["unresolved_assumptions"])


def test_textbook_kl_matches_closed_form_values():
    mean, covariance = energy_gpu._textbook_posterior({"prior_mean": [0.0, 0.0], "prior_covariance": np.eye(2),
                                                       "observation_matrix": np.eye(2), "observations": [2.0, 0.0],
                                                       "noise_covariance": np.eye(2)})
    assert mean.tolist() == pytest.approx([1.0, 0.0]) and covariance.ravel().tolist() == pytest.approx([0.5, 0, 0, 0.5])
    # KL(N(0, I) || N(0, 2I)) in two dimensions = ln 2 - 1/2.
    assert energy_gpu._textbook_kl(np.zeros(2), np.eye(2), np.zeros(2), 2 * np.eye(2)) == \
        pytest.approx(np.log(2) - 0.5)


def test_precision_study_float32_floor_and_counterexample(lab):
    report = lab("T120")
    assert report["state"] == "partial"
    order = by_claim(report, "float64 RK4 endpoint error converges at order 4")
    assert order["evidence_status"] == "numerically_verified" and abs(order["value"] - 4) < 0.3
    assert order["uncertainty"]["kind"] == "fit_residual"
    floor = by_claim(report, "float32 RK4 endpoint error stops improving")
    assert floor["evidence_status"] == "numerically_verified"
    value = floor["value"]
    assert value["crossover_steps"] < energy_gpu.PRECISION_GRID[-1]
    assert value["error_at_2048"] >= 2 * value["crossover_min_error"]
    assert value["plateau_median_error_n_ge_128"] > value["crossover_min_error"] >= 1e-7
    counter = by_claim(report, "Lowering precision to float32 cannot reach")
    assert counter["counterexample"]["witness"]["float64_steps"] == 128
    reach = by_claim(report, "Smallest grid N (>= 16) meeting each accuracy target")
    assert reach["value"]["1e-07"] == {"float32": None, "float64": 128}
    assert reach["regression_tolerance"] == {"abs": 0.0, "rel": 0.0}
    ops = by_claim(report, "Per RK4 step")
    assert ops["evidence_status"] == "numerically_verified" and ops["value"]["flops_per_step"] == 80
    for name, itemsize in (("float32", 4), ("float64", 8)):
        row = ops["value"]["instrumented"][name]
        assert (row["flops"], row["transcendentals"], row["results_outside_dtype"]) == (80, 8, 0)
        assert row["result_dtypes"] == [name] and row["state_bytes"] == 4 * itemsize
    assert len(ops["basis"]["checks"]) == 6
    assert physical_labels(report) == {"not_established"}
    energy = by_claim(report, "float32 lowers the energy per accepted trajectory")
    assert "no capture path measures a float32 workload" in energy["basis"]["notes"][0]
    assert report["recommended_next_task"].startswith("Deferred research question: a dtype-parameterized capture")
    # float32 must stay float32 through the whole integration.
    assert kernels.rk4_batch(kernels.initial_states()[:1], 1.0, 4, np.float32).dtype == np.float32


def test_operation_count_detects_precision_promotion(monkeypatch):
    """The per-dtype instrumentation can fail: a float64 step size in the float32 path shows up as promotion."""
    step = kernels.rk4_step
    monkeypatch.setattr(kernels, "rk4_step", lambda y, h, dtype: step(y, np.float64(h), dtype))
    promoted = kernels.counted_operations_per_step(np.float32)
    assert promoted["results_outside_dtype"] > 0 and "float64" in promoted["result_dtypes"]
    assert kernels.counted_operations_per_step(np.float64)["results_outside_dtype"] == 0


def test_reduction_orders_bound_and_sign_counterexample(lab):
    report = lab("T121")
    assert report["state"] == "partial"
    assert report["evidence_status"]["primary"] == "numerically_verified"
    for prefix in ("Every emulated float32 summation order", "Every emulated float64 summation order"):
        bounded = by_claim(report, prefix)
        assert bounded["evidence_status"] == "numerically_verified"
        assert bounded["value"]["worst_fraction_of_order_bound"] <= 1.0
    flip = by_claim(report, "Reduction order alone flips the sign of a float32 sum")
    assert set(flip["value"].values()) == {-1, 1} and flip["value"]["sequential"] == -1
    witness = flip["counterexample"]["witness"]
    assert witness["exact_sum"] == 0.25 and witness["seed"] == 201
    exact64 = by_claim(report, "float64 accumulation of these float32-valued")
    assert exact64["value"]["max_abs_error"] == 0.0 and exact64["evidence_status"] == "numerically_verified"
    native = by_claim(report, "Reduction order alone flips the sign of a float64 sum")
    assert set(native["value"].values()) == {-1, 1}
    assert native["counterexample"]["statement"] == "float64 reductions are order-robust for sign decisions"
    assert native["counterexample"]["witness"]["exact_sum"] == 0.25
    atomic = by_claim(report, "Atomic completion order alone changes a float32 pass/fail test")
    assert set(atomic["value"].values()) == {True, False} and all(k.startswith("atomic-") for k in atomic["value"])
    guard = by_claim(report, "The bound-guarded sign test")
    assert all(row["decided_orders"] == 0 and row["smallest_order_bound"] > row["largest_abs_sum"]
               for row in guard["value"].values())
    assert not any(f["claim"].startswith("Max reductions of this NaN-free") for f in report["findings"])
    select = by_claim(report, "A comparison-select maximum")
    assert select["value"] == [False, True] and select["evidence_status"] == "numerically_verified"
    assert "IEEE" not in select["claim"] and "IEEE 754-2019 maximum" in select["basis"]["notes"][0]
    assert energy_gpu.select_max(0.0, -0.0) == 0.0 and np.signbit(energy_gpu.select_max(-0.0, 0.0))
    # numpy.max's signed-zero result is architecture-specific: recorded with the machine, never asserted.
    _, raw = artifact(report, lab.context, "reductions.json")
    observed = json.loads(raw)["numpy_max_signed_zero_this_build"]
    assert observed["machine"] == platform.machine() and len(observed["signbits"]) == 2
    assert all(isinstance(bit, bool) for bit in observed["signbits"])
    assert by_claim(report, "Emulated atomicAdd completion orders")["value"] >= 2
    assert physical_labels(report) == {"not_established"}
    # Order-specific bounds detect a dropped element that the generic gamma_(n-1) bound would miss.
    rng = np.random.Generator(np.random.PCG64(3))
    x = (rng.random(1 << 14) * 1000).astype(np.float32)
    exact = sum(float(v) for v in x.astype(np.float64))
    dropped = float(kernels.sum_tree_strided(np.concatenate([x[:-1], np.zeros(1, dtype=np.float32)])))
    bounds = kernels.order_bounds(x, np.float32)
    assert abs(dropped - exact) > bounds["tree-strided"] and abs(dropped - exact) < kernels.summation_bound(x, np.float32)
    # Kahan summation is within 2u sum|x| of the exact sum on positive data.
    y = np.arange(1, 4097, dtype=np.float32) / np.float32(7)
    exact = sum(float(v) for v in y.astype(np.float64))
    assert abs(float(kernels.sum_kahan(y)) - exact) <= 2 * kernels.unit_roundoff(np.float32) * exact


def test_bounded_free_energy_identity_and_counterexamples(lab):
    report = lab("T122")
    assert report["state"] == "completed"
    identity = by_claim(report, "The variational free-energy identity")
    assert identity["value"] <= 1e-10 and identity["evidence_status"] == "numerically_verified"
    assert identity["uncertainty"]["kind"] == "roundoff"
    descent = by_claim(report, "KL to the exact posterior decreases")
    assert descent["basis"]["checks"][1]["comparison"] == "signed_le"
    descent = descent["value"]
    assert descent["status"] == "converged" and descent["iterations"] <= 512 and descent["final_kl_nats"] <= 1e-12
    assert by_claim(report, "The raw-unit posterior and log evidence are invariant")["value"] <= 1e-10
    unstable = by_claim(report, "A mean step 1.2 times the stability bound")
    assert unstable["value"]["spectral_radius"] > 1 and unstable["value"]["kl_final"] > 1e3 * unstable["value"]["kl_initial"]
    unit = by_claim(report, "Unit scales leave the same problem unconverged")
    assert unit["value"]["iterations"] == 512 and unit["value"]["final_kl_nats"] > 1e-3
    assert "counterexample" in unstable and "counterexample" in unit
    assert physical_labels(report) == {"not_established"}


def test_typed_quantities_refuse_nats_plus_joules(lab):
    Q = kernels.Quantity
    with pytest.raises(kernels.QuantityRefusal, match="Cannot add information and energy"):
        Q(1.0, "nat") + Q(1.0, "J")
    with pytest.raises(kernels.QuantityRefusal, match="Cannot compare"):
        Q(1.0, "nat") <= Q(1.0, "J")
    with pytest.raises(kernels.QuantityRefusal, match="Cannot compare information and energy"):
        Q(1.0, "nat") == Q(1.0, "J")
    with pytest.raises(kernels.QuantityRefusal, match="Cannot compare"):
        Q(1.0, "nat") > Q(1.0, "J")
    with pytest.raises(kernels.QuantityRefusal, match="untyped number"):
        3.0 + Q(1.0, "nat")
    with pytest.raises(kernels.QuantityRefusal, match="Cannot express energy in nat"):
        Q(1.0, "J").to("nat")
    with pytest.raises(kernels.QuantityRefusal, match="Cannot add energy/time and energy"):
        Q(1.0, "W") + Q(1.0, "J")
    with pytest.raises(TypeError):
        hash(Q(1.0, "J"))
    assert Q(1.0, "J") == Q(1000.0, "mJ") and Q(2.0, "J") > Q(1500.0, "mJ")
    assert (Q(0.25, "J") + Q(250, "mJ")).to("J") == 0.5
    assert Q(1000.0, "mW").to("W") == pytest.approx(1.0)
    assert (Q(1.0, "J") / Q(2.0, "nat")).describe() == "energy/information"
    report = lab("T123")
    refusals = by_claim(report, "Typed arithmetic refuses")
    assert refusals["evidence_status"] == "numerically_verified" and None not in refusals["value"].values()
    assert set(refusals["value"]) >= {"equal", "untyped_left", "power_plus_energy"}
    same = by_claim(report, "Typed arithmetic within one dimension agrees")
    assert same["value"]["one_joule_equals_1000_millijoules"] is True and same["evidence_status"] == "numerically_verified"
    audit = by_claim(report, "CIW energy records and free-energy records keep joules, watts and nats")
    if telemetry.fixture_bytes() is not None:
        assert report["state"] == "completed"
        assert audit["evidence_status"] == "numerically_verified"
        assert "gross_energy_j" in audit["value"]["energy_record_joule_fields"]
        assert audit["value"]["energy_record_power_fields"] == ["power_mw"]
        assert "power_mw" not in audit["value"]["energy_record_joule_fields"]
        assert "free_energy" in audit["value"]["variational_nat_fields"]
        assert audit["value"]["panel_units"]["accuracy"] == ["nat"]
    untyped = by_claim(report, "An untyped sum of free energy")
    assert "counterexample" in untyped and untyped["evidence_status"] == "analytic" and "checks" not in untyped["basis"]
    assert untyped["value"]["sum_with_millijoules"] - untyped["value"]["sum_with_joules"] == pytest.approx(199.8)
    # The weakest established computational label decides the headline.
    assert report["evidence_status"]["primary"] == "analytic"
    assert physical_labels(report) == {"not_established"}


def test_fixture_tasks_without_repository_files(tmp_path, clean_environment):
    """Clean room without examples/: T119, T124 and T125 block informatively; T123 is partial, not failed."""
    clean_environment.setenv("CIW_LAB_REPOSITORY_ROOT", str(tmp_path))
    clean_environment.setattr(runner, "_probe_hardware", lambda name: False)
    assert telemetry.fixture_bytes() is None
    context = runner.Context(tmp_path / "out")
    for task_id in ("T119", "T124", "T125"):
        report = run(task_id, context)
        assert report["state"] == "blocked" and "Not evaluated" not in report["mathematical_model"]
        assert report["experiment"].startswith("Blocked: examples/energy-accuracy is not reachable")
        assert report["findings"] and physical_labels(report) == {"not_established"}
    t123 = run("T123", context)
    assert t123["state"] == "partial" and not t123.get("tests_failed")
    audit = by_claim(t123, "CIW energy records and free-energy records")
    assert audit["evidence_status"] == "not_established" and audit.get("expected_not_established") is True
    assert by_claim(t123, "Typed arithmetic refuses")["evidence_status"] == "numerically_verified"


@needs_fixtures
def test_raw_telemetry_retention_and_tampering(lab):
    report = lab("T124")
    assert report["state"] == "completed"
    retained = by_claim(report, "Every fixture retains raw timestamped counter readings")
    assert retained["evidence_status"] == "numerically_verified" and retained["value"]["baseline"]["samples"] == 12
    # 11 placeholders: ten stand-in strings plus the compute capability given to the placeholder device.
    assert all(row["identity_fields_present"] == 14 and row["placeholder_values"] == 11
               for row in retained["value"].values())
    inventory = json.loads(artifact(report, lab.context, "inventory.json")[1])
    assert "compute_capability" in inventory["baseline"]["placeholder_fields"]
    # The task retains the fixtures' exact bytes, not a re-serialized excerpt.
    for name, raw in telemetry.fixture_bytes().items():
        entry, retained_bytes = artifact(report, lab.context, f"fixture-{name}.json")
        assert retained_bytes == raw and entry["sha256"] == hashlib.sha256(raw).hexdigest()
        assert inventory[name]["fixture_sha256"] == entry["sha256"]
    assert "present in the four synthetic fixtures" in report["numerical_result"]
    tampering = by_claim(report, "The energy-log validator refuses")
    assert tampering["evidence_status"] == "numerically_verified"
    assert len(tampering["basis"]["checks"]) == len(telemetry.TAMPERING)
    assert all(check["passed"] for check in tampering["basis"]["checks"])
    resealed = by_claim(report, "A log whose counter readings were doubled")
    assert resealed["value"]["gross_energy_j_resealed"] == pytest.approx(2 * resealed["value"]["gross_energy_j_original"])
    assert "counterexample" in resealed and resealed["evidence_status"] == "numerically_verified"
    relabel = by_claim(report, "Relabelling a synthetic fixture")
    assert relabel["value"]["hardware_provenance"] == "retained_operator_record_not_authenticated"
    assert by_claim(report, "The fixtures' counter readings were produced")["evidence_status"] == "not_established"
    assert telemetry.is_placeholder("a" * 64) and not telemetry.is_placeholder(hashlib.sha256(b"x").hexdigest())


@needs_fixtures
def test_session_replay_keeps_numerical_result_id(lab):
    report = lab("T125")
    assert report["state"] == "completed"
    stable = by_claim(report, "Session replay keeps numerical_result_id identical")
    assert stable["value"] == {"baseline": 1, "reset": 1, "missing": 1, "under-target": 1}
    fresh = by_claim(report, "Each replay is a fresh analysis occurrence")["value"]
    assert all(row == {"execution_ids": 4, "result_ids": 2, "bundle_digests": 2} for row in fresh.values())
    edited = by_claim(report, "Replaying a retained bundle whose numerical result was edited")
    assert edited["value"]["edited_and_resealed_bundle"] == "Retained energy analysis binding differs"
    assert edited["evidence_status"] == "numerically_verified"
    assert by_claim(report, "Replayed energy values are physically valid")["evidence_status"] == "not_established"
    # reseal_bundle depends on this internal digest; fail loudly if it is renamed.
    from ciw import telemetry as ciw_telemetry
    assert callable(ciw_telemetry._bundle_digest)
