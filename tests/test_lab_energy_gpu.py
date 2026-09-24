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

from ciw import energy_bench, energy_cuda, energy_records
from ciw.lab import energy_gpu, energy_gpu_kernels as kernels, energy_gpu_telemetry as telemetry, runner
from ciw.lab import energy_gpu_workload as common, planner
from ciw.lab.evidence import COMPUTATIONAL_DOMAINS, PHYSICAL_DOMAINS, EvidenceRefusal, supported_label
from ciw.lab.registry import load_queue, section_implementations
from ciw.lab.report import validate_report
from ciw.telemetry import canonical, digest

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


def pointers(report):
    """Task pointers the planner reads in a report's next step: (pointers to other tasks, stale pointers)."""
    kept, stale = planner.next_step_items(report["recommended_next_task"], report["task_id"], set(QUEUE))
    return [item for item in kept if item[0] == "pointer"], stale


def physical_labels(report):
    return {f["evidence_status"] for f in report["findings"] if f["domain"] in PHYSICAL_DOMAINS}


def artifact(report, context, name):
    entry = next(a for a in report["generated_artifacts"] if a["path"].endswith("/" + name))
    return entry, (context.output_dir / entry["path"]).read_bytes()


def common_workload_log(log):
    """A fixture log rewritten to name the common workload: plan, runtime workload and every retained batch output.

    The counter readings and clocks stay the fixture's; the outputs are the
    NumPy reference of the common workload, so the log validates and its
    analysis is eligible.
    """
    log = json.loads(json.dumps(log))
    spec, declared = common.spec(), common.workload()
    problem = spec["problem"]
    log["plan"].update(problem=problem, solver=spec["solver"], iterations=declared["iterations"],
                       replicas=declared["replicas"], target_kl_nats=spec["target_kl_nats"],
                       problem_digest=digest(problem),
                       model_digest=digest({key: value for key, value in problem.items() if key != "observations"}),
                       observations_digest=digest(problem["observations"]))
    log["runtime"]["workload"].update(iterations=declared["iterations"], replicas=declared["replicas"],
                                      solver_settings=spec["solver"], problem_sha256=declared["problem_sha256"],
                                      prepared_input_sha256=declared["prepared_input_sha256"],
                                      kernel_sha256=declared["kernel_sha256"])
    result = energy_records.encode_result(common.run_numpy(np.float64))
    for phase in log["phases"]:
        for batch in phase["batches"]:
            batch["result"] = dict(result)
    return telemetry.reseal(log)


def relabelled_log(name=None, workload=True):
    """The baseline fixture declared as a physical measurement (optionally renamed) and resealed.

    With ``workload`` it names the common workload (``common_workload_log``);
    without, it keeps the fixture's own problem, K = 128 and 4 replicas.
    """
    log = json.loads(telemetry.fixture_bytes()["baseline"])
    sensor = dict(log["sensor"], name=name) if name else log["sensor"]
    log = dict(log, origin="physical_measurement", sensor=sensor)
    return common_workload_log(log) if workload else telemetry.reseal(log)


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
        for node in implementation.regression_tests:
            path, name = node.split("::")
            assert path == "tests/test_lab_energy_gpu.py" and callable(globals()[name]), node


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
    for claim in (energy_gpu.GROSS_CPU, energy_gpu.IDLE_CPU, energy_gpu.GROSS_VI, energy_gpu.IDLE_VI):
        energy = by_claim(report, claim)
        assert energy["domain"] == "physical" and energy["evidence_status"] == "not_established"
        assert energy["value"] is None
    work = by_claim(report, energy_gpu.VI_WORK)
    assert work["evidence_status"] == "numerically_verified"
    assert work["value"]["flops_per_batch"] == common.REPLICAS * (24 * common.plan_iterations() + 7)
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
    assert "no rapl-log capture was bound" in by_claim(report, energy_gpu.GROSS_CPU)["basis"]["notes"][0]
    assert "--capture rapl-log=" in report["recommended_next_task"]
    assert "ciw lab hardware retain" in report["recommended_next_task"]


SIMULATED_BATCHES = 2


def simulated_rapl_capture(tmp_path, monkeypatch, host, rust=False):
    """An operator capture made with simulated counters.

    Brackets in capture order: geodesic 1 -> 10 J, NumPy float64 10 -> 12 J,
    NumPy float32 12 -> 13 J, (Rust float64 13 -> 14 J,) idle + 0.5 J.
    Every bracket lasts 400 ms, so the idle rescaling is deterministic.
    """
    domain = {"zone": "intel-rapl:0", "name": "package-0", "path": "unused", "max_energy_range_uj": 262143328850}
    values = [1_000_000, 10_000_000, 10_000_000, 12_000_000, 12_000_000, 13_000_000]
    values += [13_000_000, 14_000_000, 14_000_000, 14_500_000] if rust else [13_000_000, 13_500_000]
    readings = iter([[value] for value in values])
    with monkeypatch.context() as patch:
        patch.setattr(telemetry, "rapl_domains", lambda root=None, separator=":": [domain])
        patch.setattr(telemetry, "rapl_read", lambda domains: next(readings))
        patch.setattr(telemetry, "rapl_host_identity", lambda root=None, separator=":": dict(host))
        record = telemetry.capture_rapl(tmp_path / "captured.json", repeats=1, batches=SIMULATED_BATCHES,
                                        sleep=lambda seconds: None, rust=rust)
    for bracket in list(record["brackets"].values()) + [record["idle_bracket"]]:
        bracket["elapsed_monotonic_ns"] = 400_000_000
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
    assert by_claim(report, energy_gpu.GROSS_VI)["value"] == pytest.approx(2.0 / SIMULATED_BATCHES)
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
    """A simulated operator capture: gross and idle-subtracted energy per bracket, retained raw bytes, identity gate."""
    monkeypatch = clean_environment
    host = {"cpu_model": "Simulated CPU", "machine": "x86_64", "system": "Linux", "zones": ["intel-rapl:0 package-0"]}
    record, capture = simulated_rapl_capture(tmp_path, monkeypatch, host)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        telemetry.capture_rapl(tmp_path / "captured.json", repeats=1)
    # The capture declares the common workload exactly as the analyzing tasks do, and holds no host path.
    assert sorted(record["brackets"]) == sorted([telemetry.GEODESIC, telemetry.VI_NUMPY64, telemetry.VI_NUMPY32])
    assert record["workloads"][telemetry.VI_NUMPY32] == dict(common.workload("float32", "numpy"),
                                                             boundary=telemetry.NUMPY_BOUNDARY)
    assert record["skipped"] == {telemetry.VI_RUST64: "not requested (--no-rust)"}
    assert all("path" not in domain for domain in record["domains"])
    monkeypatch.setattr(telemetry, "rapl_host_identity", lambda root=None, separator=":": dict(host))
    monkeypatch.setenv(telemetry.RAPL_ENV, str(capture))
    monkeypatch.setattr(runner, "_probe_hardware", lambda name: name == "rapl")
    context = runner.Context(tmp_path / "out")
    report = run("T115", context)
    gross, idle = by_claim(report, energy_gpu.GROSS_CPU), by_claim(report, energy_gpu.IDLE_CPU)
    trajectories = len(kernels.HEADINGS)
    assert gross["value"] == pytest.approx(9.0 / trajectories) and idle["value"] == pytest.approx(8.5 / trajectories)
    gross_vi, idle_vi = by_claim(report, energy_gpu.GROSS_VI), by_claim(report, energy_gpu.IDLE_VI)
    assert gross_vi["value"] == pytest.approx(1.0) and idle_vi["value"] == pytest.approx(0.75)
    assert physical_labels(report) == {"hardware_measured"}
    assert report["state"] == "completed"
    acquisition = gross["basis"]["acquisition"]
    assert acquisition["calibration"].startswith("not_applied") and "unauthenticated" in acquisition["device"]
    entry, raw = artifact(report, context, "rapl-capture.json")
    assert raw == capture.read_bytes() and entry["sha256"] == acquisition["raw_sha256"]
    # T120 reads the float64 and float32 brackets of the same capture: energy per batch by precision.
    t120 = run("T120", runner.Context(tmp_path / "t120"))
    precision = by_claim(t120, energy_gpu.PREC_ENERGY)
    assert precision["evidence_status"] == "hardware_measured" and t120["state"] == "completed"
    assert precision["value"]["float64"] == {"gross_j": pytest.approx(1.0), "idle_subtracted_j": pytest.approx(0.75)}
    assert precision["value"]["float32_over_float64_idle_subtracted"] == pytest.approx(0.25 / 0.75)
    assert by_claim(t120, energy_gpu.GPU_FLOAT32)["evidence_status"] == "not_established"
    # The gate withholds the label for another host, another workload or a host whose probe fails.
    monkeypatch.setattr(telemetry, "rapl_host_identity", lambda root=None, separator=":": dict(host, cpu_model="Other"))
    other_host = run("T115", runner.Context(tmp_path / "other-host"))
    assert physical_labels(other_host) == {"not_established"} and other_host["state"] == "partial"
    assert "cpu_model" in by_claim(other_host, energy_gpu.GROSS_CPU)["basis"]["notes"][0]
    monkeypatch.setattr(telemetry, "rapl_host_identity", lambda root=None, separator=":": dict(host))
    edited = json.loads(json.dumps(record))
    edited["workloads"][telemetry.GEODESIC]["steps"] = 128
    capture.write_text(json.dumps(edited), encoding="utf-8")
    other_workload = run("T115", runner.Context(tmp_path / "other-workload"))
    assert by_claim(other_workload, energy_gpu.GROSS_CPU)["evidence_status"] == "not_established"
    assert "different geodesic workload" in by_claim(other_workload, energy_gpu.GROSS_CPU)["basis"]["notes"][0]
    assert by_claim(other_workload, energy_gpu.GROSS_VI)["evidence_status"] == "hardware_measured"
    assert other_workload["state"] == "partial"
    edited = json.loads(json.dumps(record))
    edited["workloads"][telemetry.VI_NUMPY32]["iterations"] = 37
    capture.write_text(json.dumps(edited), encoding="utf-8")
    assert physical_labels(run("T120", runner.Context(tmp_path / "other-iterations"))) == {"not_established"}
    # Unit counts must match the capture's own repeats and batches, and unit names the declared units: an edited
    # count would rescale the per-unit energy.
    edited = json.loads(json.dumps(record))
    edited["units"][telemetry.GEODESIC] = 1  # repeats = 1 is six trajectories
    edited["units"][telemetry.VI_NUMPY64] = 1000  # batches = 2
    capture.write_text(json.dumps(edited), encoding="utf-8")
    miscounted = run("T115", runner.Context(tmp_path / "miscounted"))
    for claim, name in ((energy_gpu.GROSS_CPU, telemetry.GEODESIC), (energy_gpu.GROSS_VI, telemetry.VI_NUMPY64)):
        withheld = by_claim(miscounted, claim)
        assert withheld["evidence_status"] == "not_established" and miscounted["state"] == "partial"
        assert f"capture {name} unit count disagrees with its repeats/batches" in withheld["basis"]["notes"]
    edited = json.loads(json.dumps(record))
    edited["unit_names"][telemetry.VI_NUMPY32] = "trajectory"
    capture.write_text(json.dumps(edited), encoding="utf-8")
    renamed = by_claim(run("T120", runner.Context(tmp_path / "renamed-unit")), energy_gpu.PREC_ENERGY)
    assert renamed["evidence_status"] == "not_established"
    assert "capture gaussian-vi-numpy-float32 unit name is not batch" in renamed["basis"]["notes"]
    capture.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(runner, "_probe_hardware", lambda name: False)
    assert physical_labels(run("T115", runner.Context(tmp_path / "no-probe"))) == {"not_established"}
    assert physical_labels(run("T120", runner.Context(tmp_path / "no-probe-t120"))) == {"not_established"}


def counting_rapl(monkeypatch, host):
    """Simulated counters that also say whether a bracket is open (reads alternate open/close)."""
    domain = {"zone": "intel-rapl:0", "name": "package-0", "path": "unused", "max_energy_range_uj": 262143328850}
    state = {"inside": False, "reads": 0}

    def read(domains):
        state["inside"] = not state["inside"]
        state["reads"] += 1
        return [state["reads"] * 1000]

    monkeypatch.setattr(telemetry, "rapl_domains", lambda root=None, separator=":": [domain])
    monkeypatch.setattr(telemetry, "rapl_read", read)
    monkeypatch.setattr(telemetry, "rapl_host_identity", lambda root=None, separator=":": dict(host))
    return state


def test_rapl_brackets_exclude_preparation_and_respect_the_port_limit(tmp_path, monkeypatch):
    """No bracket prepares inputs; batches follow the Rust port's repeats limit; a failing port loses only its bracket."""
    host = {"cpu_model": "Simulated CPU", "machine": "x86_64", "system": "Linux", "zones": ["intel-rapl:0 package-0"]}
    state = counting_rapl(monkeypatch, host)
    prepared, calls = common.prepared_inputs, []

    def spy(*args, **kwargs):
        calls.append(state["inside"])
        return prepared(*args, **kwargs)

    monkeypatch.setattr(common, "prepared_inputs", spy)
    record = telemetry.capture_rapl(tmp_path / "numpy.json", repeats=1, batches=3, sleep=lambda seconds: None,
                                    rust=False)
    assert calls and not any(calls), "prepared_inputs ran inside a bracket"
    for name in (telemetry.VI_NUMPY64, telemetry.VI_NUMPY32):
        assert record["workloads"][name]["boundary"] == telemetry.NUMPY_BOUNDARY
        assert record["units"][name] == 3
    assert record["workloads"] == {name: telemetry.declared_workloads()[name] for name in record["brackets"]}
    # The capture limit is the Rust port's repeats limit, refused before any counter is read.
    assert f'integer(text, "repeats", 1.0, {telemetry.MAX_BATCHES}.0)' in common.RUST_SOURCE
    reads = state["reads"]
    with pytest.raises(ValueError, match=r"batches must be an integer in \[1, 1000\]"):
        telemetry.capture_rapl(tmp_path / "too-many.json", repeats=1, batches=telemetry.MAX_BATCHES + 1)
    assert state["reads"] == reads and not (tmp_path / "too-many.json").exists()
    assert telemetry.main(["rapl-capture", str(tmp_path / "cli.json"), "--batches", "1001"]) == 2
    assert state["reads"] == reads and not (tmp_path / "cli.json").exists()
    # A port that fails inside its bracket is recorded under skipped; the other brackets are written.
    smoke = []

    def port(executable, precision="float64", iterations=None, replicas=common.REPLICAS, repeats=1, values=None,
             payload=None):
        smoke.append((replicas, repeats, values is not None))
        if replicas != 1:
            raise kernels.NativeKernelUnavailable("Rust port could not run: TimeoutExpired")
        return {"outputs": np.zeros(6)}

    monkeypatch.setattr(common, "build_rust_port", lambda directory: {"executable": "port", "rustc": "rustc 1.0",
                                                                      "source_sha256": "a" * 64, "binary_sha256": "b" * 64})
    monkeypatch.setattr(common, "run_rust_port", port)
    record = telemetry.capture_rapl(tmp_path / "rust.json", repeats=1, batches=4, sleep=lambda seconds: None)
    # Smoke run first with the real repeats on one replica, then the bracket with every replica.
    assert smoke == [(1, 4, True), (common.REPLICAS, 4, True)]
    assert record["skipped"] == {telemetry.VI_RUST64: "the Rust port failed inside its bracket: Rust port could not "
                                                      "run: TimeoutExpired"}
    assert telemetry.VI_RUST64 not in record["brackets"] and telemetry.VI_RUST64 not in record["units"]
    assert json.loads((tmp_path / "rust.json").read_text(encoding="utf-8"))["brackets"].keys() == record["brackets"].keys()


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
    # One lab run on the GPU host binds both captures by role and is retained with `ciw lab hardware retain`.
    for task_id in ("T116", "T118", "T119"):
        step = lab(task_id)["recommended_next_task"]
        assert energy_gpu.GPU_LAB_RUN in step and energy_gpu.GPU_RETAIN in step
        assert "--capture energy-log=" in step and "--capture nvidia-smi-csv=" in step
    t116 = lab("T116")["experiment"]
    assert energy_gpu.GPU_LAB_RUN in t116 and energy_gpu.GPU_RETAIN in t116
    assert "T118 needs it; T116 and T119 do not" in t116
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
    julia = by_claim(report, energy_gpu.JULIA_AGREE)
    assert julia["evidence_status"] == "not_established" and julia["expected_not_established"] is True
    assert julia["basis"]["notes"][0].startswith("implementation missing: no Julia implementation")
    # The GPU comparison is blamed on the probe, not on a missing implementation: the PTX kernel exists.
    gpu = by_claim(report, energy_gpu.GPU_AGREE)
    assert gpu["evidence_status"] == "not_established" and gpu["basis"]["notes"] == [common.NO_GPU_PROBE]
    assert energy_gpu.NOT_WRITTEN in report["unresolved_assumptions"]
    assert energy_gpu.JULIA_QUESTION in report["unresolved_assumptions"]
    assert energy_gpu.GPU_LAB_RUN in report["recommended_next_task"]
    assert energy_gpu.RAPL_LAB_RUN in report["recommended_next_task"]
    # On a GPU host with julia on PATH whose driver fails, the notes follow the probe and the driver's reason.
    clean_environment.setattr(runner, "_probe_hardware", lambda name: name == "nvidia-gpu")
    clean_environment.setattr(common, "run_gpu", lambda: {"unavailable": "the PTX kernel did not run: CudaError: "
                                                                        "simulated driver failure"})
    which = runner.shutil.which
    clean_environment.setattr(runner.shutil, "which",
                              lambda name: "/usr/bin/julia" if name == "julia" else which(name))
    gpu_host = run("T117", runner.Context(tmp_path))
    gpu_note = by_claim(gpu_host, energy_gpu.GPU_AGREE)["basis"]["notes"][0]
    assert gpu_note.endswith("simulated driver failure")
    assert "julia is on PATH here" in by_claim(gpu_host, energy_gpu.JULIA_AGREE)["basis"]["notes"][0]
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
    assert energy_gpu.GPU_LAB_RUN in report["recommended_next_task"]


@needs_fixtures
def test_t119_operator_log_is_gated_like_t116(tmp_path, clean_environment):
    """On a (simulated) GPU host T119 measures energy per accepted solve from the operator log, or withholds it."""
    log = relabelled_log()
    context = simulate_gpu_host(clean_environment, tmp_path, log)
    report = run("T119", context)
    physical = by_claim(report, energy_gpu.T119_PHYSICAL)
    # 0.2 J over one measured batch of the common workload's 4096 accepted replica solves.
    assert physical["evidence_status"] == "hardware_measured"
    assert physical["value"] == pytest.approx(0.2 / common.REPLICAS)
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
    energy = by_claim(report, energy_gpu.PREC_ENERGY)
    assert energy["basis"]["notes"] == [energy_gpu.NO_RAPL]
    gpu32 = by_claim(report, energy_gpu.GPU_FLOAT32)
    assert gpu32["basis"]["notes"][0].startswith("implementation missing") and common.GPU_QUESTION in gpu32["basis"]["notes"][0]
    assert energy_gpu.RAPL_PROTOCOL in report["recommended_next_task"]
    assert common.GPU_QUESTION in report["recommended_next_task"]
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
    # GPU reductions are computational outputs: no physical claim; the missing device reduction kernel is named.
    assert physical_labels(report) == set()
    device = by_claim(report, energy_gpu.DEVICE_REDUCTION)
    assert device["domain"] == "numerical" and device["expected_not_established"] is True
    assert device["basis"]["notes"] == [energy_gpu.DEVICE_REDUCTION_NOTE]
    assert pointers(report) == ([], [])
    assert common.GPU_QUESTION in report["recommended_next_task"] and energy_gpu.GPU_LAB_RUN in report["recommended_next_task"]
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
    # Only synthetic fixtures were retained: the task says so and stays partial (as T139 does without an acquisition).
    assert report["state"] == "partial"
    assert energy_gpu.NOT_RETAINED_REAL in report["unresolved_assumptions"]
    assert energy_gpu.NOT_RETAINED_REAL in report["numerical_result"]
    retained = by_claim(report, "Every fixture retains raw timestamped counter readings")
    assert retained["evidence_status"] == "numerically_verified" and retained["value"]["baseline"]["samples"] == 12
    # 11 placeholders: ten stand-in strings plus the compute capability given to the placeholder device.
    assert all(row["identity_fields_present"] == 14 and row["placeholder_values"] == 11
               for row in retained["value"].values())
    inventory = json.loads(artifact(report, lab.context, "inventory.json")[1])["fixtures"]
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


# ----------------------------------------------------------------- the common Gaussian VI workload
def test_common_workload_matches_the_ptx_kernel_and_the_energy_problem(lab, monkeypatch):
    """One workload for CPU energy, languages, precision, reductions and the CPU/GPU harness: the PTX kernel's own."""
    problem = runner.repository_path("examples", "energy-accuracy", "problem.json")
    if problem is not None and problem.is_file():
        assert json.loads(problem.read_text(encoding="utf-8")) == common.SPEC
    assert common.plan_iterations() == energy_bench.prepare_plan(common.spec())["iterations"] == 38
    ptx = common.ptx_operation_counts()
    assert ptx == {"iteration_flops": 24, "output_flops": 7, "output_negations": 2, "fused_instructions": 0}
    for dtype in (np.float64, np.float32):
        counted = common.counted_operations(dtype)
        assert (counted["iteration_flops"], counted["output_flops"], counted["output_negations"]) == (24, 7, 2)
        assert counted["results_outside_dtype"] == 0 and counted["result_dtypes"] == [np.dtype(dtype).name]
    reference = common.run_numpy(np.float64)
    assert reference.shape == (common.REPLICAS, 6) and np.all(reference == reference[0])
    # A scalar Python-float evaluation of the declared order is bitwise the NumPy reference.
    assert common.ulp_distance(common.run_variant("kernel"), reference[0]) == 0
    assert common.run_numpy(np.float32, replicas=2).dtype == np.float32
    declaration = common.workload()
    assert declaration["kernel_sha256"] == hashlib.sha256(energy_cuda.PTX.encode("ascii")).hexdigest()
    assert declaration["prepared_input_sha256"] == hashlib.sha256(common.prepared_inputs().tobytes()).hexdigest()
    assert (declaration["iterations"], declaration["replicas"]) == (38, 4096)
    # The exact FMA emulation rounds once: 0.1 * 10 - 1 is 0 in two roundings and 2^-54 in one.
    assert 0.1 * 10.0 - 1.0 == 0.0 and common.fma(0.1, 10.0, -1.0) == 2.0 ** -54
    # The operation-count check can fail: a kernel that fused a multiply-add would be refuted in T115.
    monkeypatch.setattr(energy_cuda, "PTX", energy_cuda.PTX.replace("    mul.rn.f64 %d16, %d1, %d7;\n    add.rn.f64 "
                                                                    "%d15, %d15, %d16;",
                                                                    "    fma.rn.f64 %d15, %d1, %d7, %d15;", 1))
    monkeypatch.setattr(runner, "_probe_hardware", lambda name: False)
    fused = energy_gpu._vi_work_finding(energy_gpu._common_counts())
    assert fused["evidence_status"] == "not_established"
    assert [check["passed"] for check in fused["basis"]["checks"]] == [False, True, False]


def rust_port(tmp_path):
    try:
        return common.build_rust_port(tmp_path)
    except kernels.NativeKernelUnavailable as exc:
        pytest.skip(f"Rust port unavailable: {exc}")


def test_rust_port_of_the_common_workload_is_bitwise(lab, tmp_path):
    identity = rust_port(tmp_path)
    for precision in common.PRECISIONS:
        dtype = np.dtype(precision).type
        run_ = common.run_rust_port(identity["executable"], precision, replicas=8, repeats=2)
        assert run_["outputs"].dtype == dtype and run_["replicas_identical"] is True
        assert run_["iterations_executed"] == 38 * 8 * 2
        assert common.ulp_distance(run_["outputs"], common.run_numpy(dtype, replicas=1)[0], dtype) == 0
    with pytest.raises(kernels.NativeKernelUnavailable, match="precision must be float64 or float32"):
        common.run_rust_port(identity["executable"], "float16")
    report = lab("T117")
    for precision in common.PRECISIONS:
        agreement = by_claim(report, energy_gpu.PORT_AGREE.format(precision=precision))
        assert agreement["evidence_status"] == "numerically_verified" and agreement["value"]["max_ulp"] == 0
        assert agreement["value"]["iterations_executed"] == 38 * common.REPLICAS
        assert agreement["basis"]["checks"][0]["reference_kind"] == "cross_implementation"
    refusals = by_claim(report, energy_gpu.PORT_REFUSES)
    assert refusals["evidence_status"] == "numerically_verified" and set(refusals["value"]) == set(energy_gpu.PORT_REFUSALS)
    port = report["provider_runtime_identity"]["rust_port"]
    assert port["source_sha256"] == common.rust_source_sha256() and port["binary_sha256"] == identity["binary_sha256"]
    # The Rust energy comparison needs a capture with both brackets; without one it is withheld, not estimated.
    energy = by_claim(report, energy_gpu.RUST_ENERGY)
    assert energy["evidence_status"] == "not_established" and energy["basis"]["notes"] == [energy_gpu.NO_RAPL]


def simulated_gpu(monkeypatch, outputs=None, unavailable=None):
    """A host whose nvidia-gpu probe answers and whose PTX kernel returns ``outputs`` (or fails with ``unavailable``)."""
    monkeypatch.setattr(runner, "_probe_hardware", lambda name: name == "nvidia-gpu")
    identity = {"device_name": "NVIDIA GeForce RTX 2080", "device_uuid": "GPU-5d3f9a2c-7e41-4b8a-9c06-1f2e3d4c5b6a",
                "compute_capability": [7, 5], "cuda_driver_version": 12040,
                "kernel_sha256": common.kernel_sha256(), "arithmetic": "binary64_explicit_round_to_nearest",
                "prepared_input_sha256": hashlib.sha256(common.prepared_inputs().tobytes()).hexdigest()}
    result = {"unavailable": unavailable} if unavailable else {"outputs": outputs, "identity": identity}
    monkeypatch.setattr(common, "run_gpu", lambda: result)


def test_gpu_comparisons_follow_the_probe(lab, tmp_path, clean_environment):
    """Without a GPU the comparisons name the probe; with one they are computed, and a mismatch refutes them."""
    for task_id, claim in (("T117", energy_gpu.GPU_AGREE), ("T121", energy_gpu.KERNEL_GPU)):
        record = by_claim(lab(task_id), claim)
        assert record["evidence_status"] == "not_established" and record["expected_not_established"] is True
        assert record["basis"]["notes"] == [common.NO_GPU_PROBE]
    reference = common.run_numpy(np.float64)
    simulated_gpu(clean_environment, outputs=reference.copy())
    agreeing = {task_id: run(task_id, runner.Context(tmp_path / f"agree-{task_id}")) for task_id in ("T117", "T121")}
    for task_id, claim in (("T117", energy_gpu.GPU_AGREE), ("T121", energy_gpu.KERNEL_GPU)):
        record = by_claim(agreeing[task_id], claim)
        assert record["evidence_status"] == "numerically_verified" and record["value"]["max_ulp"] == 0
        assert record["value"]["replicas"] == common.REPLICAS and record["value"]["differing_values"] == 0
        assert [c["reference_kind"] for c in record["basis"]["checks"]] == ["cross_implementation"] * 2 + [
            "exact_arithmetic"]
    assert agreeing["T117"]["provider_runtime_identity"]["gpu"]["device_name"] == "NVIDIA GeForce RTX 2080"
    # A device that contracts multiply-adds is caught: the finding is refuted and the headline drops.
    contracted = np.tile(common.run_variant("fma-first"), (common.REPLICAS, 1))
    simulated_gpu(clean_environment, outputs=contracted)
    for task_id, claim in (("T117", energy_gpu.GPU_AGREE), ("T121", energy_gpu.KERNEL_GPU)):
        report = run(task_id, runner.Context(tmp_path / f"contracted-{task_id}"))
        record = by_claim(report, claim)
        assert record["evidence_status"] == "not_established" and not record.get("expected_not_established")
        assert record["value"]["max_ulp"] >= 1 and report["evidence_status"]["primary"] == "not_established"
        assert report["state"] == "partial"
    # A device that drops the kernel's neg.f64 flips the sign of covariance_01: refuted by the ULP distance across
    # signs and by the bitwise count, never reported as bitwise agreement.
    flipped = reference.copy()
    flipped[:, 3] = -flipped[:, 3]
    simulated_gpu(clean_environment, outputs=flipped)
    for task_id, claim in (("T117", energy_gpu.GPU_AGREE), ("T121", energy_gpu.KERNEL_GPU)):
        report = run(task_id, runner.Context(tmp_path / f"sign-flipped-{task_id}"))
        record = by_claim(report, claim)
        assert record["evidence_status"] == "not_established" and not record.get("expected_not_established")
        assert record["value"]["max_ulp"] > 0 and record["value"]["differing_values"] == common.REPLICAS
        assert [c["passed"] for c in record["basis"]["checks"]] == [False, False, True]
        assert report["evidence_status"]["primary"] == "not_established"
    # Outputs of another shape are a failed comparison, not a crashed (blocked) task.
    simulated_gpu(clean_environment, outputs=reference[:10].copy())
    short = run("T121", runner.Context(tmp_path / "short"))
    assert short["state"] == "partial" and by_claim(short, energy_gpu.KERNEL_GPU)["evidence_status"] == "not_established"
    assert not by_claim(short, energy_gpu.KERNEL_GPU).get("expected_not_established")
    # A driver failure is reported with its reason, never replaced by a CPU result.
    simulated_gpu(clean_environment, unavailable="the PTX kernel did not run: CudaError: simulated")
    failed = by_claim(run("T121", runner.Context(tmp_path / "failed")), energy_gpu.KERNEL_GPU)
    assert failed["expected_not_established"] is True and failed["basis"]["notes"] == [
        "the PTX kernel did not run: CudaError: simulated"]


def test_ulp_distance_counts_across_signs():
    """ULP distance is zero exactly for equal bits: a sign flip is far, and +0.0 against -0.0 is one step."""
    for dtype in (np.float64, np.float32):
        one, two = np.array([1.0], dtype=dtype), np.array([1.0, 2.0], dtype=dtype)
        assert common.ulp_distance(one, -one, dtype) > 0
        assert common.ulp_distance(two, np.array([1.0, -2.0], dtype=dtype), dtype) > 0
        assert common.ulp_distance(np.array([0.0], dtype=dtype), np.array([-0.0], dtype=dtype), dtype) == 1
        tiny = np.finfo(dtype).smallest_subnormal
        assert common.ulp_distance(np.array([tiny], dtype=dtype), np.array([-tiny], dtype=dtype), dtype) == 3
        assert common.ulp_distance(one, np.nextafter(one, dtype(2)), dtype) == 1
        assert common.ulp_distance(-one, np.nextafter(-one, dtype(-2)), dtype) == 1
        assert common.ulp_distance(two, two.copy(), dtype) == 0
    # float64 +1 and -1 are 2 x 0x3FF0000000000000 + 1 steps apart; the largest distance exceeds int64.
    assert common.ulp_distance([1.0], [-1.0]) == 2 * 0x3FF0000000000000 + 1
    big = np.finfo(np.float64).max
    assert common.ulp_distance([big], [-big]) == 2 * 0x7FEFFFFFFFFFFFFF + 1 > 2 ** 63
    assert kernels.ulp_distance([1.0], [-1.0]) == common.ulp_distance([1.0], [-1.0])
    assert common.differing_bits([0.0, 1.0], [-0.0, 1.0]) == 1


class RejectingWorker:
    """Stands in for ciw.energy_cuda.CudaGaussianWorker: the kernel ran, then solve()'s own output check raised."""

    def __init__(self, problem, solver, iterations, replicas=4096, device_index=0):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def identity(self):
        return {"device_name": "NVIDIA GeForce RTX 2080",
                "prepared_input_sha256": hashlib.sha256(common.prepared_inputs().tobytes()).hexdigest()}

    def solve(self):
        raise ValueError("GPU covariance lost symmetry")


def test_rejected_gpu_outputs_refute_the_comparisons(tmp_path, clean_environment):
    """Outputs the CUDA worker rejects after the kernel ran are a failed check, never an expected gap."""
    clean_environment.setattr(runner, "_probe_hardware", lambda name: name == "nvidia-gpu")
    clean_environment.setattr(energy_cuda, "CudaGaussianWorker", RejectingWorker)
    assert common.run_gpu()["invalid"] == ("the worker rejected the GPU outputs: ValueError: GPU covariance lost "
                                           "symmetry")
    for task_id, claim in (("T117", energy_gpu.GPU_AGREE), ("T121", energy_gpu.KERNEL_GPU)):
        report = run(task_id, runner.Context(tmp_path / task_id))
        record = by_claim(report, claim)
        assert record["evidence_status"] == "not_established" and not record.get("expected_not_established")
        assert record["value"]["rejected_by_worker"].endswith("GPU covariance lost symmetry")
        assert record["basis"]["checks"][0]["reference"] == energy_gpu.GPU_REJECTED
        assert report["evidence_status"]["primary"] == "not_established" and report["state"] == "partial"
        assert "GPU comparison refuted" in report["numerical_result"]


def test_run_gpu_drives_the_cuda_worker_on_the_common_workload(monkeypatch):
    calls = []
    reference = common.run_numpy(np.float64, replicas=4)

    class Worker:
        def __init__(self, problem, solver, iterations, replicas=4096, device_index=0):
            calls.append((problem, solver, iterations, replicas, device_index))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def identity(self):
            return {"device_name": "fake"}

        def solve(self):
            return reference

    monkeypatch.setattr(energy_cuda, "CudaGaussianWorker", Worker)
    result = common.run_gpu(replicas=4)
    assert calls == [(common.SPEC["problem"], common.SPEC["solver"], 38, 4, 0)]
    assert np.array_equal(result["outputs"], reference) and result["identity"] == {"device_name": "fake"}

    class Broken(Worker):
        def __init__(self, *args, **kwargs):
            raise energy_cuda.CudaError("CUDA driver library is unavailable; no GPU solve was performed")

    monkeypatch.setattr(energy_cuda, "CudaGaussianWorker", Broken)
    assert common.run_gpu()["unavailable"].startswith("the PTX kernel did not run: CudaError: CUDA driver library")

    class Unprepared(Worker):
        def __init__(self, *args, **kwargs):  # energy_cuda._prepare refuses before any GPU work
            raise ValueError("Fixed GPU iterations exceed the declared solver budget")

    monkeypatch.setattr(energy_cuda, "CudaGaussianWorker", Unprepared)
    assert common.run_gpu()["unavailable"].startswith("the PTX kernel did not run: ValueError")

    class LaunchFailure(Worker):
        def solve(self):
            raise energy_cuda.CudaError("cuLaunchKernel: CUDA_ERROR_LAUNCH_FAILED (719): unspecified launch failure")

    monkeypatch.setattr(energy_cuda, "CudaGaussianWorker", LaunchFailure)
    assert common.run_gpu()["unavailable"].startswith("the PTX kernel did not complete: CudaError")
    monkeypatch.setattr(energy_cuda, "CudaGaussianWorker", RejectingWorker)
    rejected = common.run_gpu()
    assert "outputs" not in rejected and rejected["invalid"].endswith("ValueError: GPU covariance lost symmetry")


def test_common_workload_precision_study(lab):
    report = lab("T120")
    same = by_claim(report, "float32 and float64 runs of the common Gaussian VI workload first meet")
    assert same["value"] == {"float32": 38, "float64": 38, "planned_iterations": 38}
    assert same["evidence_status"] == "numerically_verified"
    floor = by_claim(report, "The float32 iteration stalls at a KL floor")
    assert floor["evidence_status"] == "numerically_verified"
    assert 3.6e-15 < floor["value"]["float32_floor_kl"] < 1e-13
    assert floor["value"]["float64_kl_at_256"] < 1e-6 * floor["value"]["float32_floor_kl"]
    counter = by_claim(report, "Lowering the common Gaussian VI workload to float32 cannot reach")
    assert counter["value"]["float64_iteration"] == 69 and "counterexample" in counter
    counts = by_claim(report, "Per replica iteration the float32 and float64 runs")
    assert counts["evidence_status"] == "numerically_verified"
    assert {p: counts["value"][p]["iteration_flops"] for p in common.PRECISIONS} == {"float64": 24, "float32": 24}


def test_kernel_reductions_under_contraction(lab, tmp_path, clean_environment):
    report = lab("T121")
    fused = by_claim(report, energy_gpu.KERNEL_FMA)
    assert fused["evidence_status"] == "numerically_verified" and "counterexample" in fused
    assert min(fused["value"][v]["max_ulp"] for v in ("fma-first", "fma-second")) >= 1
    decision = by_claim(report, energy_gpu.KERNEL_DECISION)
    assert decision["evidence_status"] == "numerically_verified"
    assert decision["value"]["kl_reference"] < decision["value"]["target_kl_nats"] == 1e-8
    assert all(value < 1e-8 for value in decision["value"]["kl_nats"].values())
    # The contraction check can fail: if contraction changed nothing, the finding would be refuted.
    clean_environment.setattr(runner, "_probe_hardware", lambda name: False)
    clean_environment.setattr(common, "run_variant", lambda variant, iterations=None, values=None:
                              common.run_numpy(np.float64, replicas=1)[0])
    refuted = by_claim(run("T121", runner.Context(tmp_path)), energy_gpu.KERNEL_FMA)
    assert refuted["evidence_status"] == "not_established" and not refuted.get("expected_not_established")


def realistic_log():
    """A resealed fixture of the common workload that declares a physical measurement and has no placeholder identity."""
    log = common_workload_log(json.loads(telemetry.fixture_bytes()["baseline"]))
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()  # noqa: E731
    uuid = "GPU-5d3f9a2c-7e41-4b8a-9c06-1f2e3d4c5b6a"
    log["sensor"].update(device_uuid=uuid, name="NVIDIA GeForce RTX 2080", driver_version="550.54.14",
                         nvml_version="12.550.54.14", library_sha256=sha("libnvidia-ml"))
    log["runtime"]["workload"].update(device_uuid=uuid, device_name="NVIDIA GeForce RTX 2080",
                                      compute_capability=[7, 5], kernel_sha256=common.kernel_sha256())
    log["runtime"]["python"]["executable_sha256"] = sha("python")
    log["runtime"]["implementation"]["code_sha256"] = sha("implementation")
    log["clock"]["implementation"] = "clock_gettime(CLOCK_MONOTONIC)"
    return telemetry.reseal(dict(log, origin="physical_measurement"))


@needs_fixtures
def test_t124_retains_an_operator_log_through_the_gate(lab, tmp_path, clean_environment):
    """T124 completes only when a real-looking operator log is retained and bound through the acquisition gate."""
    report = lab("T124")
    inventory = by_claim(report, energy_gpu.T124_OPERATOR_INVENTORY)
    assert inventory["evidence_status"] == "not_established" and inventory["expected_not_established"] is True
    assert by_claim(report, energy_gpu.T124_OPERATOR_PHYSICAL)["evidence_status"] == "not_established"
    assert energy_gpu.GPU_LAB_RUN in report["recommended_next_task"]
    log = realistic_log()
    context = simulate_gpu_host(clean_environment, tmp_path, log)
    bound = run("T124", context)
    physical = by_claim(bound, energy_gpu.T124_OPERATOR_PHYSICAL)
    assert physical["evidence_status"] == "hardware_measured" and bound["state"] == "completed"
    assert physical["value"]["device_name"] == "NVIDIA GeForce RTX 2080"
    # The claim states the identity binding the gate establishes, not that the readings came from the device: this
    # log's readings are the synthetic fixture's, which is why the origin caveat stays in the report.
    assert "names an NVML device present on this analyzing host" in physical["claim"]
    assert "come from" not in physical["claim"] and "unauthenticated" in physical["claim"]
    assert energy_gpu.T124_ORIGIN_CAVEAT in bound["unresolved_assumptions"]
    retained = by_claim(bound, energy_gpu.T124_OPERATOR_INVENTORY)
    assert retained["evidence_status"] == "numerically_verified" and retained["value"]["placeholder_fields"] == []
    entry, raw = artifact(bound, context, "operator-log.json")
    assert raw == canonical(log) and entry["sha256"] == physical["basis"]["acquisition"]["raw_sha256"]
    assert energy_gpu.NOT_RETAINED_REAL not in bound["unresolved_assumptions"]
    # A relabelled synthetic fixture passes the host-identity gate but not T124's placeholder rule.
    (tmp_path / "relabelled").mkdir()
    relabelled = run("T124", simulate_gpu_host(clean_environment, tmp_path / "relabelled", relabelled_log()))
    assert by_claim(relabelled, energy_gpu.T124_OPERATOR_PHYSICAL)["evidence_status"] == "not_established"
    assert any("placeholder values" in note
               for note in by_claim(relabelled, energy_gpu.T124_OPERATOR_PHYSICAL)["basis"]["notes"])
    assert by_claim(relabelled, energy_gpu.T124_OPERATOR_INVENTORY)["evidence_status"] == "not_established"
    assert relabelled["state"] == "partial"
    # Without the GPU probe the same log is retained but not bound.
    clean_environment.setattr(runner, "_probe_hardware", lambda name: False)
    no_gpu = run("T124", runner.Context(tmp_path / "no-gpu"))
    assert by_claim(no_gpu, energy_gpu.T124_OPERATOR_PHYSICAL)["evidence_status"] == "not_established"
    assert no_gpu["state"] == "partial"


def with_workload(log, iterations=None, replicas=None, kernel_sha256=None):
    """A copy of a common-workload log that names another K, replica count or kernel, still valid and eligible."""
    log = json.loads(json.dumps(log))
    work, plan = log["runtime"]["workload"], log["plan"]
    if iterations is not None:
        work["iterations"] = plan["iterations"] = iterations
    if replicas is not None:
        work["replicas"] = plan["replicas"] = replicas
        result = energy_records.encode_result(common.run_numpy(np.float64, replicas=replicas))
        for phase in log["phases"]:
            for batch in phase["batches"]:
                batch["result"] = dict(result)
    if kernel_sha256 is not None:
        work["kernel_sha256"] = kernel_sha256
    return telemetry.reseal(log)


@needs_fixtures
def test_operator_logs_of_another_workload_are_withheld(tmp_path, clean_environment):
    """T116, T118, T119 and T124 label a log hardware_measured only when it names the common workload."""
    other_kernel = hashlib.sha256(b"another PTX kernel").hexdigest()
    variants = {"fixture problem, K 128, 4 replicas": (relabelled_log("NVIDIA GeForce RTX 2080", workload=False),
                                                         "problem_sha256"),
                "K 39": (with_workload(relabelled_log("NVIDIA GeForce RTX 2080"), iterations=39), "iterations"),
                "2048 replicas": (with_workload(relabelled_log("NVIDIA GeForce RTX 2080"), replicas=2048), "replicas"),
                "another kernel": (with_workload(realistic_log(), kernel_sha256=other_kernel), "kernel_sha256")}
    for label, (log, field) in variants.items():
        energy_records.validate_log(log)
        assert energy_records.analyze(log)["comparison"]["eligible"], label
        assert field in telemetry.workload_reasons(log)[0], label
        directory = tmp_path / label.replace(" ", "-").replace(",", "")
        directory.mkdir()
        context = simulate_gpu_host(clean_environment, directory, log)
        for task_id, claim in (("T116", energy_gpu.GPU_ENERGY), ("T118", energy_gpu.POWER),
                               ("T119", energy_gpu.T119_PHYSICAL), ("T124", energy_gpu.T124_OPERATOR_PHYSICAL)):
            report = run(task_id, runner.Context(directory / task_id))
            record = by_claim(report, claim)
            assert record["evidence_status"] == "not_established", (label, task_id)
            assert any("different GPU workload than the common workload" in note and field in note
                       for note in record["basis"]["notes"]), (label, task_id)
            assert report["state"] == "partial", (label, task_id)
    # The same log naming the common workload passes (the gate is not vacuous).
    context = simulate_gpu_host(clean_environment, tmp_path, relabelled_log("NVIDIA GeForce RTX 2080"))
    assert telemetry.workload_reasons(relabelled_log()) == []
    assert by_claim(run("T116", context), energy_gpu.GPU_ENERGY)["evidence_status"] == "hardware_measured"


def test_workload_tasks_digest_the_workload_sources(lab):
    """Tasks whose results depend on the common workload digest the modules that define it and record it."""
    for task_id in ("T115", "T117", "T119", "T120", "T121", "T124"):
        identity = lab(task_id)["provider_runtime_identity"]
        assert set(common.SOURCES) <= set(identity["sources"]), task_id
        assert all(len(identity["sources"][name]) == 64 for name in common.SOURCES), task_id
        assert identity["common_workload"] == common.workload(), task_id
    assert common.SOURCES == ("src/ciw/energy_cuda.py", "src/ciw/energy_bench.py", "src/ciw/free_energy_math.py")


def test_next_steps_name_work_that_delivers(lab):
    """No next step points at a queue task; hardware steps name capture roles and the retention command."""
    for task_id in TASK_IDS:
        report = lab(task_id)
        assert pointers(report) == ([], []), task_id
    for task_id in ("T115", "T117", "T120"):
        assert energy_gpu.RAPL_LAB_RUN in lab(task_id)["recommended_next_task"]
        assert energy_gpu.RAPL_RETAIN in lab(task_id)["recommended_next_task"]
    for task_id in ("T116", "T117", "T118", "T119", "T121", "T124"):
        assert energy_gpu.GPU_RETAIN in lab(task_id)["recommended_next_task"], task_id
    for task_id in ("T122", "T123", "T125"):
        assert lab(task_id)["recommended_next_task"].startswith("Deferred research question:")


def test_native_build_failures_name_no_host_path(tmp_path, monkeypatch):
    """A failed rustc or kernel launch is reported by exception type: a retained report must hold no host path."""
    secret = str(tmp_path / "operator" / "secret")

    def refuse(*args, **kwargs):
        raise OSError(f"[Errno 2] No such file or directory: '{secret}'")

    monkeypatch.setattr(kernels.shutil, "which", lambda name: secret + "/rustc")
    monkeypatch.setattr(kernels.subprocess, "run", refuse)
    for build in (lambda: kernels.build_rust_kernel(tmp_path), lambda: common.build_rust_port(tmp_path),
                  lambda: kernels.run_rust_kernel(secret, [[1.0, 0.0, 1.0, 0.0]], 1.0, 1),
                  lambda: common.run_rust_port(secret, replicas=1)):
        with pytest.raises(kernels.NativeKernelUnavailable) as refused:
            build()
        assert secret not in str(refused.value) and str(refused.value).endswith("OSError")


def test_rust_energy_is_read_from_the_captures_rust_bracket(tmp_path, clean_environment):
    """The Rust bracket binds to the port T117 builds (same rustc, same binary digest); then T117 reports it."""
    monkeypatch = clean_environment
    (tmp_path / "probe-build").mkdir()
    built = rust_port(tmp_path / "probe-build")
    host = {"cpu_model": "Simulated CPU", "machine": "x86_64", "system": "Linux", "zones": ["intel-rapl:0 package-0"]}
    record, capture = simulated_rapl_capture(tmp_path, monkeypatch, host, rust=True)
    assert record["skipped"] == {} and telemetry.VI_RUST64 in record["brackets"]
    assert record["workloads"][telemetry.VI_RUST64]["binary_sha256"] == built["binary_sha256"]
    monkeypatch.setattr(telemetry, "rapl_host_identity", lambda root=None, separator=":": dict(host))
    monkeypatch.setenv(telemetry.RAPL_ENV, str(capture))
    monkeypatch.setattr(runner, "_probe_hardware", lambda name: name == "rapl")
    report = run("T117", runner.Context(tmp_path / "t117"))
    energy = by_claim(report, energy_gpu.RUST_ENERGY)
    assert energy["evidence_status"] == "hardware_measured"
    assert energy["value"]["rust"] == {"gross_j": pytest.approx(0.5), "idle_subtracted_j": pytest.approx(0.25)}
    assert energy["value"]["rust_over_numpy_idle_subtracted"] == pytest.approx(0.25 / 0.75)
    # A capture whose Rust bracket names another binary (another rustc) is withheld, not compared.
    edited = json.loads(json.dumps(record))
    edited["workloads"][telemetry.VI_RUST64]["binary_sha256"] = "0" * 64
    capture.write_text(json.dumps(edited), encoding="utf-8")
    other = by_claim(run("T117", runner.Context(tmp_path / "other-binary")), energy_gpu.RUST_ENERGY)
    assert other["evidence_status"] == "not_established"
    assert "different gaussian-vi-rust-float64 workload" in " ".join(other["basis"]["notes"])
