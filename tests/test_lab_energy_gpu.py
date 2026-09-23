"""Energy and GPU experiments T115-T125: labels, key numbers and refusals.

No test reads hardware counters or starts GPU work. Hardware probes are forced
off so every assertion is deterministic; the Rust comparison is skipped when
rustc is absent.
"""
import json
import shutil

import numpy as np
import pytest

from ciw import energy_records
from ciw.lab import energy_gpu, energy_gpu_kernels as kernels, energy_gpu_telemetry as telemetry, runner
from ciw.lab.registry import _REGISTRY, load_queue
from ciw.lab.report import validate_report
from ciw.telemetry import canonical

QUEUE = {t["id"]: t for t in load_queue()["tasks"]}
needs_fixtures = pytest.mark.skipif(telemetry.fixture_bytes() is None, reason="source checkout fixtures absent")


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    """Run tasks through the runner with one shared context and hardware probes forced off."""
    context = runner.Context(tmp_path_factory.mktemp("lab-energy-gpu"))
    patch = pytest.MonkeyPatch()
    patch.setattr(runner, "_probe_hardware", lambda name: False)
    patch.delenv(telemetry.LOG_ENV, raising=False)
    reports = {}

    def run(task_id):
        if task_id not in reports:
            reports[task_id] = validate_report(runner.run_task(QUEUE[task_id], _REGISTRY[task_id], context, {}))
        return reports[task_id]

    run.context = context
    yield run
    patch.undo()


def by_claim(report, prefix):
    matches = [f for f in report["findings"] if f["claim"].startswith(prefix)]
    assert len(matches) == 1, prefix
    return matches[0]


def physical_labels(report):
    return {f["evidence_status"] for f in report["findings"] if f["domain"] == "physical"}


def test_every_section_task_is_registered_with_tests():
    for number in range(115, 126):
        implementation = _REGISTRY[f"T{number}"]
        assert implementation.regression_tests
        assert all(node.startswith("tests/test_lab_energy_gpu.py::") for node in implementation.regression_tests)


def test_cpu_energy_task_counts_work_and_leaves_energy_unestablished(lab):
    report = lab("T115")
    assert report["state"] == "partial"
    work = by_claim(report, "Fixed-step RK4 spends exactly")
    assert work["value"] == 4 * energy_gpu.STEPS and work["evidence_status"] == "numerically_verified"
    accepted = by_claim(report, "The fixed-step trajectories are accepted")
    assert accepted["evidence_status"] == "numerically_verified" and accepted["value"] < 1e-8
    energy = by_claim(report, "CPU package energy per geodesic trajectory")
    assert energy["domain"] == "physical" and energy["evidence_status"] == "not_established" and energy["value"] is None
    assert report["physical_validation_status"]["status"] == "not_established"
    names = {a["path"].rsplit("/", 1)[-1] for a in report["generated_artifacts"]}
    assert {"work-proxies.json", "timing.json", "rapl-probe.json"} <= names
    # Elapsed time is retained as an artifact, never as a finding.
    assert not any("time" in f["claim"].lower() for f in report["findings"])


def test_rapl_helpers_read_counters_and_one_wrap(tmp_path):
    # '_' stands in for ':' so the tree can be built on Windows too.
    for zone, name, value in (("intel-rapl_0", "package-0", 1000), ("intel-rapl_0_0", "core", 5)):
        directory = tmp_path / zone
        directory.mkdir()
        (directory / "energy_uj").write_text(f"{value}\n")
        (directory / "name").write_text(name + "\n")
        (directory / "max_energy_range_uj").write_text("262143328850\n")
    domains = telemetry.rapl_domains(tmp_path, separator="_")
    assert [d["name"] for d in domains] == ["package-0"]  # subzones are inside their package
    assert telemetry.rapl_read(domains) == [1000]
    assert telemetry.rapl_delta_uj(1000, 1500, 262143328850) == (500, False)
    assert telemetry.rapl_delta_uj(262143328800, 49, 262143328850) == (100, True)
    with pytest.raises(ValueError, match="without a declared wrap range"):
        telemetry.rapl_delta_uj(10, 5, None)


def test_rapl_path_divides_counter_difference_by_trajectories(tmp_path, monkeypatch):
    """Simulated counters exercise the RAPL branch; the label follows the acquisition record."""
    readings = iter([[1_000_000], [1_000_000 + 9_000_000]])
    domain = {"zone": "intel-rapl:0", "name": "package-0", "path": "unused", "max_energy_range_uj": 262143328850}
    monkeypatch.setattr(runner, "_probe_hardware", lambda name: name == "rapl")
    monkeypatch.setattr(telemetry, "rapl_domains", lambda root=None, separator=":": [domain])
    monkeypatch.setattr(telemetry, "rapl_read", lambda domains: next(readings))
    report = validate_report(runner.run_task(QUEUE["T115"], _REGISTRY["T115"], runner.Context(tmp_path), {}))
    energy = by_claim(report, "CPU package energy per geodesic trajectory")
    assert energy["value"] == pytest.approx(9.0 / (3 * len(kernels.HEADINGS)))
    assert energy["evidence_status"] == "hardware_measured" and report["state"] == "completed"
    assert energy["basis"]["acquisition"]["calibration"].startswith("not_applied")
    raw = json.loads((tmp_path / "artifacts" / "T115" / "rapl-raw.json").read_text())
    assert raw["after_uj"][0] - raw["before_uj"][0] == 9_000_000


def test_gpu_tasks_are_blocked_with_the_recording_protocol(lab):
    for task_id in ("T116", "T118"):
        report = lab(task_id)
        assert report["state"] == "blocked" and report["findings"] == []
        assert "hardware:nvidia-gpu" in report["experiment"]
        assert "ciw energy record --problem examples/energy-accuracy/problem.json" in report["experiment"]
        assert report["physical_validation_status"]["status"] == "not_established"
    assert "nvidia-smi --query-gpu" in lab("T118")["experiment"] and "nsys" in lab("T118")["experiment"]


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
    # A relabelled log is still withheld when this host's NVML identity differs or the device is not an RTX 2080.
    relabelled = telemetry.reseal(dict(log, origin="physical_measurement"))
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
    kernel = by_claim(rtx, "RTX 2080 kernel-only duration")
    assert kernel["value"] is None and kernel["evidence_status"] == "not_established"


@needs_fixtures
def test_operator_log_gate_trust_boundary_is_the_host_identity(tmp_path):
    """The gate binds a declared physical log to this host's NVML identity; it cannot authenticate the capture.

    A synthetic fixture relabelled as a physical measurement and paired with a
    host identity equal to its sensor block passes the gate. This is the
    documented trust boundary (see the T124 origin-relabelling counterexample).
    """
    log = telemetry.reseal(dict(json.loads(telemetry.fixture_bytes()["baseline"]), origin="physical_measurement"))
    context = runner.Context(tmp_path)
    context.begin("T116")
    outcome = energy_gpu.operator_log_outcome(context, "T116", canonical(log), log, energy_records.analyze(log),
                                              dict(log["sensor"]))
    energy = by_claim(outcome, "GPU-domain gross energy per measured batch")
    assert energy["evidence_status"] == "hardware_measured" and energy["value"] == pytest.approx(0.2)
    assert energy["basis"]["acquisition"]["calibration"].startswith("not_applied")


@pytest.mark.skipif(shutil.which("rustc") is None, reason="rustc is not installed")
def test_rust_kernel_matches_python_kernel(lab, tmp_path):
    report = lab("T117")
    assert report["state"] == "partial"
    agreement = by_claim(report, "Rust and Python closed-form RK4 kernels agree")
    assert agreement["evidence_status"] == "numerically_verified" and agreement["value"] <= 1e-12
    assert by_claim(report, "Rust kernel endpoint error")["value"] < 1e-8
    identity = kernels.build_rust_kernel(tmp_path)
    assert identity["source_sha256"] == kernels.rust_source_sha256()
    with pytest.raises(kernels.NativeKernelUnavailable, match="states must be a list of 4-vectors"):
        kernels.run_rust_kernel(identity["executable"], [[1.0, 0.0, 1.0]], 1.0, 4)
    with pytest.raises(kernels.NativeKernelUnavailable, match="steps must be an integer"):
        kernels.run_rust_kernel(identity["executable"], [[1.0, 0.0, 1.0, 0.0]], 1.0, 0)


def test_cross_language_agreement_is_not_independent(lab):
    report = lab("T117")
    refusal = by_claim(report, "Declaring the Rust kernel an independent check")
    assert refusal["evidence_status"] == "numerically_verified"
    assert refusal["basis"]["checks"][0]["observed_refusal"] == \
        "independent_check producer and checker share an implementation origin"
    assert all(f["evidence_status"] != "independently_verified" for f in report["findings"])
    for prefix in ("A Julia implementation", "A GPU implementation"):
        assert by_claim(report, prefix)["evidence_status"] == "not_established"
    generic = by_claim(report, "Generic Christoffel-symbol RK4")
    assert generic["evidence_status"] == "numerically_verified" and generic["value"] <= 1e-12
    assert physical_labels(report) == {"not_established"}


@needs_fixtures
def test_energy_per_accepted_result_on_fixtures(lab):
    report = lab("T119")
    assert report["state"] == "partial"
    metric = by_claim(report, "Energy per accepted result of the synthetic baseline")
    assert metric["value"] == pytest.approx(0.05, abs=1e-15) and metric["evidence_status"] == "numerically_verified"
    withheld = by_claim(report, "The metric is withheld")
    assert sorted(withheld["value"]) == ["missing", "reset", "under-target"]
    naive = by_claim(report, "Dividing gross energy by executed solves")
    assert naive["value"]["accepted_solves"] == 0 and naive["value"]["naive_j_per_solve"] == pytest.approx(0.05)
    assert "counterexample" in naive
    assert by_claim(report, "Widening the boundary")["value"] == pytest.approx(7.0)
    assert by_claim(report, "Physical GPU energy per accepted")["evidence_status"] == "not_established"


def test_precision_study_float32_floor_and_counterexample(lab):
    report = lab("T120")
    assert report["state"] == "partial"
    order = by_claim(report, "float64 RK4 endpoint error converges at order 4")
    assert order["evidence_status"] == "numerically_verified" and abs(order["value"] - 4) < 0.3
    floor = by_claim(report, "float32 RK4 endpoint error saturates")
    assert 1e-7 <= floor["value"] <= 1e-5 and floor["evidence_status"] == "numerically_verified"
    counter = by_claim(report, "Lowering precision to float32 cannot reach")
    assert counter["counterexample"]["witness"]["float64_steps"] == 128
    reach = by_claim(report, "Steps needed to reach each accuracy target")["value"]
    assert reach["1e-07"] == {"float32": None, "float64": 128}
    assert by_claim(report, "Per RK4 step")["evidence_status"] == "analytic"
    assert physical_labels(report) == {"not_established"}
    # float32 must stay float32 through the whole integration.
    assert kernels.rk4_batch(kernels.initial_states()[:1], 1.0, 4, np.float32).dtype == np.float32


def test_reduction_orders_bound_and_sign_counterexample(lab):
    report = lab("T121")
    assert report["state"] == "partial"
    labels = {f["claim"]: f["evidence_status"] for f in report["findings"]}
    assert sum(label == "numerically_verified" for label in labels.values()) == 7
    flip = by_claim(report, "Reduction order alone flips the sign")
    assert set(flip["value"].values()) == {-1, 1}
    witness = flip["counterexample"]["witness"]
    assert witness["exact_sum"] == 0.25
    assert by_claim(report, "float64 reductions of the same cancellation data")["value"]["sequential"] == 1
    guard = by_claim(report, "Threshold decisions guarded")
    assert all(row["contradictions"] == 0 for row in guard["value"].values())
    assert by_claim(report, "Max reductions are order-invariant")["value"] == 1
    assert by_claim(report, "Emulated atomicAdd completion orders")["value"] >= 2
    assert physical_labels(report) == {"not_established"}
    # Kahan summation is within 2u sum|x| of the exact sum on positive data.
    x = np.arange(1, 4097, dtype=np.float32) / np.float32(7)
    exact = sum(float(v) for v in x.astype(np.float64))
    assert abs(float(kernels.sum_kahan(x)) - exact) <= 2 * kernels.unit_roundoff(np.float32) * exact


def test_bounded_free_energy_identity_and_counterexamples(lab):
    report = lab("T122")
    assert report["state"] == "completed"
    identity = by_claim(report, "The variational free-energy identity")
    assert identity["value"] <= 1e-10 and identity["evidence_status"] == "numerically_verified"
    descent = by_claim(report, "KL to the exact posterior decreases")["value"]
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
    with pytest.raises(kernels.QuantityRefusal, match="Cannot express energy in nat"):
        Q(1.0, "J").to("nat")
    assert (Q(0.2, "J") + Q(300, "mJ")).to("J") == pytest.approx(0.5)
    assert (Q(1.0, "J") / Q(2.0, "nat")).describe() == "energy/information"
    report = lab("T123")
    assert report["state"] == "completed"
    refusals = by_claim(report, "Typed arithmetic refuses")
    assert refusals["evidence_status"] == "numerically_verified" and None not in refusals["value"].values()
    audit = by_claim(report, "CIW energy records and free-energy records keep joules and nats")
    if telemetry.fixture_bytes() is not None:
        assert audit["evidence_status"] == "numerically_verified"
        assert "gross_energy_j" in audit["value"]["energy_record_joule_fields"]
        assert "free_energy" in audit["value"]["variational_nat_fields"]
        assert audit["value"]["panel_units"]["accuracy"] == ["nat"]
    assert "counterexample" in by_claim(report, "An untyped sum of free energy")
    assert physical_labels(report) == {"not_established"}


@needs_fixtures
def test_raw_telemetry_retention_and_tampering(lab):
    report = lab("T124")
    assert report["state"] == "completed"
    retained = by_claim(report, "Every fixture retains raw timestamped counter readings")
    assert retained["evidence_status"] == "numerically_verified" and retained["value"]["baseline"]["samples"] == 12
    tampering = by_claim(report, "The energy-log validator refuses")
    assert tampering["evidence_status"] == "numerically_verified"
    assert len(tampering["basis"]["checks"]) == len(telemetry.TAMPERING)
    assert all(check["passed"] for check in tampering["basis"]["checks"])
    resealed = by_claim(report, "A log whose counter readings were doubled")
    assert resealed["value"]["gross_energy_j_resealed"] == pytest.approx(2 * resealed["value"]["gross_energy_j_original"])
    assert "counterexample" in resealed
    relabel = by_claim(report, "Relabelling a synthetic fixture")
    assert relabel["value"]["hardware_provenance"] == "retained_operator_record_not_authenticated"
    assert by_claim(report, "The fixtures' counter readings were produced")["evidence_status"] == "not_established"


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
