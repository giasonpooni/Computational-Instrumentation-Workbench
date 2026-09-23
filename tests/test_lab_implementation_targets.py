import dataclasses
import importlib.util
import json
import math
import shutil

import numpy as np
import pytest

from ciw.core.identities import canonical_json
from ciw.lab import runner
from ciw.lab import implementation_targets as targets
from ciw.lab import implementation_targets_architecture as arch
from ciw.lab import implementation_targets_authority as authority
from ciw.lab import implementation_targets_fpga as fpga
from ciw.lab import implementation_targets_kernels as kernels
from ciw.lab import implementation_targets_serial as serial
from ciw.lab.registry import load_queue, section_implementations
from ciw.lab.report import validate_report
from ciw.telemetry import canonical as telemetry_canonical

TASKS = {t["id"]: t for t in load_queue()["tasks"]}
RUST = shutil.which("rustc") is not None
NOW = "2026-09-23T00:00:00Z"


def _run(task_id, directory):
    implementations = section_implementations("implementation-targets")
    report = runner.run_task(TASKS[task_id], implementations[task_id], runner.Context(directory), {})
    validate_report(report)
    return report, {f["claim"]: f for f in report["findings"]}


def test_every_section_task_is_registered_with_existing_tests():
    implementations = section_implementations("implementation-targets")
    assert set(implementations) == {f"T{n}" for n in range(142, 155)}
    for implementation in implementations.values():
        assert implementation.regression_tests
        for node in implementation.regression_tests:
            path, name = node.split("::")
            assert path == "tests/test_lab_implementation_targets.py" and callable(globals()[name])


def test_t142_kernel_counts_and_ranking(tmp_path):
    report, findings = _run("T142", tmp_path)
    counts = findings["Exact floating-point operation counts of the scalar kernel restatements"]
    assert counts["evidence_status"] == "numerically_verified"
    flops = {name: value["flops"] for name, value in counts["value"].items()}
    assert flops == {"geodesic_rhs": 173, "jacobi_rhs": 178, "rk4_step": 819, "jacobi_transfer": 8191,
                     "kalman_update": 453}
    assert flops["rk4_step"] == 4 * flops["jacobi_rhs"] + kernels.rk4_combination_ops(8)
    assert counts["value"]["geodesic_rhs"]["trig"] == 4
    ranking = findings["Ranked Rust port recommendation"]
    assert ranking["evidence_status"] == "analytic"
    assert ranking["value"][0].startswith("jacobi_transfer") and ranking["value"][-1].startswith("rk4_step alone")
    dispatches = findings["Interpreter calls issued by ciw code per kernel call"]
    assert dispatches["evidence_status"] == "numerically_verified" and dispatches["value"]["geodesic_rhs"] > 20
    assert findings["Rust ports of the ranked kernels are ready for industrial deployment"]["evidence_status"] == \
        "not_established"
    rust = findings["Rust fused RK4 loop reproduces ciw.lab.jacobi.transfer on the unit sphere"]
    assert rust["evidence_status"] == ("numerically_verified" if RUST else "not_established")
    assert report["state"] == ("completed" if RUST else "partial")
    timings = json.loads((tmp_path / "artifacts" / "T142" / "kernel-timings.json").read_text(encoding="utf-8"))
    assert "seconds_per_call" in timings["timings"]


@pytest.mark.skipif(not RUST, reason="rustc is not on PATH")
def test_rust_fused_sphere_loop():
    result = targets.rust_sphere_agreement(steps=200, length=2.0)
    assert result["max_abs_difference"] <= 1e-11
    assert result["rust_jacobi_error"] <= 1e-7 and result["core_jacobi_error"] <= 1e-7


def test_t143_interface_inventory(tmp_path):
    report, findings = _run("T143", tmp_path)
    assert report["state"] == "completed"
    names = findings["Industrial interfaces that need C/C++ libraries behind a pinned subprocess boundary"]
    assert names["evidence_status"] == "analytic" and len(names["value"]) == 5
    refusals = findings["Inventory validator refuses in-process bindings, write-capable directions and unpinned entries"]
    assert refusals["value"] == 4 and refusals["evidence_status"] == "numerically_verified"
    assert findings["The listed interfaces are qualified for plant integration"]["evidence_status"] == "not_established"
    with pytest.raises(arch.ArchitectureRefusal) as refused:
        arch.validate_inventory([dict(arch.INTERFACES[1], write_path="enabled")])
    assert refused.value.code == "write_path_enabled"


def test_t144_architecture_scan(tmp_path):
    report, findings = _run("T144", tmp_path)
    closure = findings["Evidence and identity closure is standard-library Python with no native loading or process spawns"]
    assert closure["value"] == ["ciw.core.identities", "ciw.lab.evidence", "ciw.lab.report"]
    assert closure["evidence_status"] == "numerically_verified"
    assert findings["Scanner flags forged modules that cross the boundary"]["value"] == 5
    assert findings["Keeping native code behind subprocess boundaries makes machine interfaces safe"][
        "evidence_status"] == "not_established"
    scanned = arch.scan_source("ciw.lab.x", "import subprocess as sp\nsp.check_output('ls', shell=True)\n")
    assert scanned["spawn_lines"] == [2] and scanned["shell_lines"] == [2] and not scanned["identity_tokens"]
    relative = arch.scan_source("ciw.lab.report", "from ..core.identities import digest\nfrom .evidence import finding\n")
    assert "ciw.core.identities" in relative["imports"] and "ciw.lab.evidence" in relative["imports"]


def test_t145_julia_partial_with_plan(tmp_path):
    report, findings = _run("T145", tmp_path)
    assert report["state"] == "partial"
    assert report["evidence_status"]["primary"] == "not_established"
    assert findings["Julia environment pinned and exercised through the CIW to SCR boundary"]["evidence_status"] == \
        "not_established"
    assert findings["Julia provider pin procedure"]["evidence_status"] == "analytic"
    plan = json.loads((tmp_path / "artifacts" / "T145" / "julia-pin-procedure.json").read_text(encoding="utf-8"))
    assert "manifest_sha256" in plan["identity_fields"] and plan["steps"]
    symbolic = findings.get("Symbolic torus Christoffel symbols and curvature agree with ciw.lab.surfaces.Torus")
    if importlib.util.find_spec("sympy") is not None:
        assert symbolic["evidence_status"] == "independently_verified" and symbolic["value"] <= 1e-12


def test_sympy_torus_geometry_matches_core():
    pytest.importorskip("sympy")
    result = targets.sympy_torus_check(samples=4)
    assert result["max_abs_difference"] <= 1e-12
    assert result["metric_off_diagonal"] == "0"


def test_t146_python_canonicalizers_and_vectors(tmp_path):
    assert serial.canonical_bytes([1e16, 1e15, 1e-5, 1e-4, -0.0, 5e-324, 1.0]) == \
        b"[1e+16,1000000000000000.0,1e-05,0.0001,-0.0,5e-324,1.0]"
    assert serial.canonical_bytes({"b": 1, "\U0001f600": 2, "Ａ": 3}).decode() == '{"b":1,"Ａ":3,"\U0001f600":2}'
    for value, code in (([math.nan], "nonfinite_number"), (2 ** 53, "unsafe_integer"), ({1: "x"}, "non_string_key"),
                        ("\ud800", "invalid_unicode")):
        with pytest.raises(serial.CanonicalRefusal) as refused:
            serial.canonical_bytes(value)
        assert refused.value.code == code
    text = "é\x7f"
    assert telemetry_canonical(text) == serial.canonical_bytes(text) != canonical_json(text).encode()
    assert canonical_json({1: "x"}) == canonical_json({"1": "x"})
    assert serial.ecmascript_number(1.0) == "1" and serial.ecmascript_number(1e-7) == "1e-7"
    report, findings = _run("T146", tmp_path)
    assert findings["Existing Python canonicalizers reproduce the specification bytes on every accepted vector"][
        "value"] == {"ciw.core.identities.canonical_json": 0, "ciw.telemetry.canonical": 0}
    differs = findings["ciw.core.identities.canonical_json and ciw.telemetry.canonical produce different bytes for "
                       "non-ASCII text"]
    assert "unicode-bmp" in differs["value"]["differing_vectors"] and differs["counterexample"]
    floats = findings["CPython shortest float digits agree with NumPy Dragon4 on vector and random binary64 values"]
    assert floats["evidence_status"] == "independently_verified" and floats["value"]["mismatches"] == 0
    assert findings["Byte-identical canonical JSON holds for Julia, C++ and GPU-host implementations"][
        "evidence_status"] == "not_established"
    vectors = json.loads((tmp_path / "artifacts" / "T146" / "test-vectors.json").read_text(encoding="utf-8"))
    by_name = {row["name"]: row for row in vectors["vectors"]}
    assert by_name["signed-zero"]["canonical"] == '[0.0,-0.0,{"z":-0.0}]'
    assert by_name["empty-object"]["sha256"] == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"


@pytest.mark.skipif(not RUST, reason="rustc is not on PATH")
def test_t146_rust_byte_identity():
    accepted, invalid = serial.vectors(), serial.invalid_vectors()
    results = serial.rust_canonical([v for _, v in accepted] + [v for _, v, _ in invalid])
    for (name, value), got in zip(accepted, results):
        assert got == (serial.canonical_bytes(value), serial.canonical_bytes(value, ascii_only=True)), name
    assert [got for got in results[len(accepted):]] == [code for _, _, code in invalid]


def test_t147_harness_detects_differences(tmp_path):
    report, findings = _run("T147", tmp_path)
    assert report["state"] == "partial"
    assert report["evidence_status"]["primary"] == "not_established"
    assert findings["Bitwise policy detects reduction-order differences between float64 CPU orders"]["value"] >= 1
    within = findings["Float64 reduction-order differences lie within the analytic error-bound policy"]
    assert within["evidence_status"] == "numerically_verified" and within["value"] < 1.0
    f32 = findings["Float32 results violate the float64 policy and satisfy the float32 policy"]["value"]
    assert f32["violations_under_f64_policy"] > 0 and f32["max_ratio_under_f32_policy"] < 1.0
    assert findings["GPU/CPU agreement establishes industrial readiness"]["evidence_status"] == "not_established"
    with pytest.raises(ValueError, match="equal shapes"):
        kernels.compare_outputs(np.zeros(2), np.zeros(3), {"mode": "bitwise"})
    with pytest.raises(ValueError, match="finite"):
        kernels.compare_outputs(np.zeros(2), np.array([0.0, np.nan]), {"mode": "bitwise"})
    assert kernels.ulp_distance(np.array([1.0]), np.array([np.nextafter(1.0, 2.0)]))[0] == 1.0


def test_t148_reduction_policies(tmp_path):
    case = [1.0, 1e100, 1.0, -1e100]
    assert kernels.sum_kahan(case) == 0.0 and kernels.sum_neumaier(case) == 2.0 and kernels.sum_exact(case) == 2.0
    rng = np.random.Generator(np.random.PCG64(7))
    xs = [float(v) for v in rng.uniform(-1, 1, 300) * 10.0 ** rng.integers(-8, 8, 300)]
    for _ in range(5):
        order = [xs[i] for i in rng.permutation(len(xs))]
        assert kernels.sum_exact(order) == math.fsum(xs)
    report, findings = _run("T148", tmp_path)
    assert report["state"] == "completed"
    exact = findings["Correctly rounded exact accumulation is permutation-invariant"]
    assert exact["value"] == 1 and exact["evidence_status"] == "independently_verified"
    pairwise = findings["Fixed-order pairwise summation is reproducible for one order but not permutation-invariant"]
    assert pairwise["value"] >= 2 and pairwise["counterexample"]
    assert all(ratio <= 1.0 for ratio in
               findings["Observed errors of every algorithm lie within their analytic bounds"]["value"].values())


def test_t149_telemetry_only_frames(tmp_path):
    assert fpga.crc32(b"123456789") == 0xCBF43926
    frame = fpga.encode_frame(2 ** 32 - 1, 5, 7, [-1, 2 ** 31 - 1])
    assert fpga.decode_frame(frame)["channels"] == [-1, 2 ** 31 - 1] and len(frame) == 28 + 8 + 4
    for forged, code in ((fpga.forge_frame(0x81, 0), "command_path_refused"),
                         (fpga.forge_frame(1, fpga.FLAG_WRITE_REQUEST), "command_path_refused"),
                         (fpga.forge_frame(0x02, 0), "unknown_frame_type"),
                         (frame[:-1] + bytes([frame[-1] ^ 1]), "crc_mismatch")):
        with pytest.raises(fpga.TelemetryRefusal) as refused:
            fpga.decode_frame(forged)
        assert refused.value.code == code
    report, findings = _run("T149", tmp_path)
    detection = findings["Every single-bit error and every 2-32 bit burst in a frame is refused by the decoder"]["value"]
    assert detection["single_bit"] == [384, 384] and detection["bursts"][0] == detection["bursts"][1]
    assert findings["CIW table-driven CRC-32 agrees with zlib and the catalogue check value"]["evidence_status"] == \
        "independently_verified"
    assert findings["Decoder, encoder and interface validator refuse every command or write path"]["value"] == 9
    assert findings["Host receiver exposes no sending or writing method"]["value"] == []
    assert findings["A telemetry-only interface guarantees the FPGA cannot actuate the machine"]["evidence_status"] == \
        "not_established"


def test_t150_bitstream_identity(tmp_path):
    report, findings = _run("T150", tmp_path)
    bound = findings["Bitstream identity record binds bitstream, toolchain, constraints and source tree"]
    assert bound["value"]["refused_mutations"] == 10 and bound["evidence_status"] == "numerically_verified"
    assert findings["The bitstream is approved for production deployment"]["evidence_status"] == "not_established"
    record = json.loads((tmp_path / "artifacts" / "T150" / "bitstream-identity.json").read_text(encoding="utf-8"))
    assert record["synthetic"] is True and record["record_sha256"] == bound["value"]["record_sha256"]
    with pytest.raises(fpga.IdentityRefusal) as refused:
        fpga.deployment_decision(record)
    assert refused.value.code == "synthetic_bitstream"


def test_t151_compatibility_and_rollback(tmp_path):
    report, findings = _run("T151", tmp_path)
    agreement = findings["Compatibility rules and the set construction agree on every combination"]["value"]
    assert agreement == {"combinations": 36, "compatible": len(fpga.compatible_set(fpga.COMPATIBILITY)),
                         "disagreements": 0}
    counter = findings["Rolling back to the previous bitstream version can be incompatible"]
    assert counter["value"]["cases"] >= 1 and counter["counterexample"]["witness"] == counter["value"]["witness"]
    assert findings["Rollback validation refuses incompatible, unregistered, no-op, forward and unexplained rollbacks"][
        "value"] == 8
    assert not fpga.compatible(fpga.COMPATIBILITY, "1.4.1", "2.0", "revB")


def test_t152_loss_latency_staleness(tmp_path):
    report, findings = _run("T152", tmp_path)
    gaps = findings["Sequence-number gap detection recovers every lost frame across the 32-bit wrap"]
    assert gaps["value"]["lost"] == gaps["value"]["detected"] > 0 and gaps["value"]["reordered"] > 0
    stale = findings["Staleness detection with the declared clock offset matches ground truth exactly"]["value"]
    assert stale["stale"] == stale["detected"] > 0
    biased = findings["An offset estimated from minimum delay misses exactly the stale frames within its bias"]
    assert biased["value"]["missed"] > 0 and biased["value"]["false_alarms"] == 0
    naive = findings["A detector without modular sequence arithmetic miscounts losses"]["value"]
    assert naive["naive_count"] != naive["true_lost"]
    assert findings["Simulated loss, latency and staleness represent the real FPGA telemetry link"][
        "evidence_status"] == "not_established"


def test_t153_writes_disabled_by_default(tmp_path):
    policy = authority.ActuatorWritePolicy()
    with pytest.raises(authority.AuthorityRefusal) as refused:
        policy.check_write("spindle_speed", 1.0, now=NOW)
    assert refused.value.code == "writes_disabled_by_default"
    with pytest.raises(authority.AuthorityRefusal) as refused:
        authority.issue_authorization()
    assert refused.value.code == "lab_cannot_issue_authority"
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.enabled = True
    report, findings = _run("T153", tmp_path)
    assert findings["Default policy refuses every write attempt"]["value"] == {"attempts": 1000, "accepted": 0}
    assert findings["Enabling writes is refused on every route, including a well-formed external record"]["value"] == 8
    assert findings["A frozen in-process policy object can be mutated"]["counterexample"]
    assert findings["The lab holds actuator write authority"]["evidence_status"] == "not_established"
    assert findings["The lab holds actuator write authority"]["domain"] == "actuator_authority"


def test_t154_control_outputs_are_proposals(tmp_path):
    report, findings = _run("T154", tmp_path)
    assert report["state"] == "completed"
    outputs = findings["Every control output carries status proposal and cannot be converted to a command here"]
    assert outputs["value"] == {"refusals": 7, "status": "proposal"}
    orders = findings["Jacobi heading proposal cancels a lateral offset to second order"]["value"]
    assert abs(orders["corrected_order"] - 2.0) < 0.2 and abs(orders["uncorrected_order"] - 1.0) < 0.2
    assert findings["Heading proposals are authorized for execution as actuator commands"]["evidence_status"] == \
        "not_established"
    record = json.loads((tmp_path / "artifacts" / "T154" / "proposal-record.json").read_text(encoding="utf-8"))
    assert record["status"] == "proposal"
    with pytest.raises(authority.AuthorityRefusal) as refused:
        authority.ControlProposal("heading_offset", 0.1, "rad", "test", "0" * 64, "test", status="command")
    assert refused.value.code == "proposal_status_fixed"
    proposal = authority.ControlProposal("heading_offset", 0.1, "rad", "test", "0" * 64, "test")
    with pytest.raises(authority.AuthorityRefusal) as refused:
        authority.to_command(proposal, now=NOW)
    assert refused.value.code == "proposal_not_authorized"
