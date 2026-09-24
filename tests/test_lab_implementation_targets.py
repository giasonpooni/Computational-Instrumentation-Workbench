import dataclasses
import importlib.util
import json
import math
import shutil

import numpy as np
import pytest

from ciw.core.identities import canonical_json
from ciw.lab import energy_gpu_workload as common
from ciw.lab import planner, runner
from ciw.lab import implementation_targets as targets
from ciw.lab import implementation_targets_architecture as arch
from ciw.lab import implementation_targets_authority as authority
from ciw.lab import implementation_targets_fpga as fpga
from ciw.lab import implementation_targets_kernels as kernels
from ciw.lab import implementation_targets_serial as serial
from ciw.lab.evidence import COMPUTATIONAL_DOMAINS, finding
from ciw.lab.registry import load_queue, section_implementations
from ciw.lab.report import validate_report
from ciw.telemetry import canonical as telemetry_canonical

TASKS = {t["id"]: t for t in load_queue()["tasks"]}
NOW = "2026-09-23T00:00:00Z"


def _rust_build():
    """None when rustc is absent or cannot build a trivial program; the build otherwise (compiled once)."""
    if shutil.which("rustc") is None:
        return None
    build = serial.rust_build()
    return build if build["usable"] else None


@pytest.fixture(scope="module")
def rust_probe():
    build = _rust_build()
    if build is None:
        pytest.skip("rustc is absent or cannot build a trivial program here")
    if not build["available"]:
        pytest.fail("rustc works but the embedded Rust probe did not build: " + build.get("stderr", "")[-500:])
    return build


def _run(task_id, directory):
    implementations = section_implementations("implementation-targets")
    report = runner.run_task(TASKS[task_id], implementations[task_id], runner.Context(directory), {})
    validate_report(report)
    return report, {f["claim"]: f for f in report["findings"]}


def _refuted(report):
    return [f["claim"] for f in report["findings"] if f["domain"] in COMPUTATIONAL_DOMAINS
            and f["evidence_status"] == "not_established" and not f.get("expected_not_established")]


def _assert_clean(report, state, primary):
    assert _refuted(report) == []
    assert report["state"] == state
    assert report["evidence_status"]["primary"] == primary
    assert all(f.get("uncertainty") is not None for f in report["findings"]
               if f["domain"] in COMPUTATIONAL_DOMAINS and f["value"] is not None)


def test_every_section_task_is_registered_with_existing_tests():
    implementations = section_implementations("implementation-targets")
    assert set(implementations) == {f"T{n}" for n in range(142, 155)}
    for implementation in implementations.values():
        assert implementation.regression_tests
        for node in implementation.regression_tests:
            path, name = node.split("::")
            assert path == "tests/test_lab_implementation_targets.py" and callable(globals()[name])


def test_refusal_that_does_not_happen_is_a_refuted_finding_not_a_blocked_task(tmp_path, monkeypatch):
    check = targets._refusal("action that does not refuse", "expected_code", lambda: None)
    assert check["observed_refusal"] == "none" and check["passed"] is False
    assert finding("claim", "computational_pipeline", 0, {"checks": [check]})["evidence_status"] == "not_established"
    crash = targets._refusal("action that crashes", "expected_code", lambda: {}["missing"])
    assert crash["observed_refusal"] == "unexpected KeyError"
    monkeypatch.setattr(fpga, "execute_rollback", lambda record: None)
    report, findings = _run("T151", tmp_path)
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "not_established"
    assert _refuted(report) == ["Rollback validation refuses incompatible, unregistered, no-op, forward and "
                                "unexplained rollbacks"]
    assert findings["Compatibility rules and the set construction agree on every combination"][
        "evidence_status"] == "numerically_verified"


def test_t142_kernel_counts_and_ranking(tmp_path):
    report, findings = _run("T142", tmp_path)
    available = _rust_build() is not None
    _assert_clean(report, "completed" if available else "partial", "numerically_verified")
    counts = findings["Exact floating-point operation counts of the scalar kernel restatements"]
    assert counts["evidence_status"] == "numerically_verified"
    flops = {name: value["flops"] for name, value in counts["value"].items()}
    assert flops == {"geodesic_rhs": 173, "jacobi_rhs": 178, "rk4_step": 819, "jacobi_transfer": 8191,
                     "kalman_update": 453}
    assert flops["rk4_step"] == 4 * flops["jacobi_rhs"] + kernels.rk4_combination_ops(8)
    assert counts["value"]["geodesic_rhs"]["trig"] == 4
    ranking = findings["Ranked Rust port recommendation"]
    assert ranking["evidence_status"] == "numerically_verified"
    assert ranking["value"][0].startswith("jacobi_transfer") and ranking["value"][-1].startswith("rk4_step alone")
    assert "not a core kernel" in ranking["value"][2]
    dispatches = findings["Interpreter calls issued by ciw code per kernel call"]
    assert dispatches["evidence_status"] == "numerically_verified" and dispatches["value"]["geodesic_rhs"] > 20
    # Containment: the fused loop's lead over geodesic_rhs is structural, and the derivation says so.
    assert "contains every geodesic_rhs call" in ranking["basis"]["derivation"]
    overhead = findings["Python dispatch overhead, not arithmetic, dominates the run time of the geodesic/Jacobi kernels"]
    assert overhead["evidence_status"] == "not_established" and overhead["expected_not_established"] is True
    assert findings["Rust ports of the ranked kernels are ready for industrial deployment"]["evidence_status"] == \
        "not_established"
    rust = findings[targets.RUST_SPHERE_CLAIM]
    assert rust["evidence_status"] == ("numerically_verified" if available else "not_established")
    rows = json.loads((tmp_path / "artifacts" / "T142" / "kernel-ranking.json").read_text(encoding="utf-8"))
    assert all(row["arithmetic_intensity"] == row["flops_per_call"] / row["bytes_per_call"] for row in rows)
    timings = json.loads((tmp_path / "artifacts" / "T142" / "kernel-timings.json").read_text(encoding="utf-8"))
    assert "seconds_per_call" in timings["timings"]


def test_rust_fused_sphere_loop(rust_probe):
    result = targets.rust_sphere_agreement(steps=200, length=2.0)
    assert result["max_abs_difference"] <= 1e-11
    assert result["rust_jacobi_error"] <= 1e-7 and result["core_jacobi_error"] <= 1e-7
    # The build directory is remapped, so the binary digest is reproducible.
    assert "binary_sha256" in rust_probe["identity"]
    assert any(flag.startswith("--remap-path-prefix=<build dir>=") for flag in rust_probe["identity"]["flags"])


def test_t143_interface_inventory(tmp_path):
    report, findings = _run("T143", tmp_path)
    _assert_clean(report, "completed", "analytic")
    names = findings["Industrial interfaces assigned to C/C++ libraries behind a pinned subprocess boundary (required, "
                     "or preferred over existing pure-Python stacks)"]
    assert names["evidence_status"] == "analytic" and len(names["value"]["required"]) == 4
    assert "EtherCAT passive monitoring (network TAP capture)" in names["value"]["required"]
    # OPC UA has pure-Python stacks (asyncua, python-opcua): C/C++ is a stated preference, not a necessity.
    assert names["value"]["preferred"] == ["OPC UA client (read-only subscriptions)"]
    assert any("asyncua" in name for name in arch.INTERFACES[0]["pure_python_alternatives"])
    refusals = findings["Inventory validator refuses in-process bindings, write-capable directions or bus roles, "
                        "unpinned entries and unexplained native preferences"]
    assert refusals["value"] == 12 and refusals["evidence_status"] == "numerically_verified"
    inventory = json.loads((tmp_path / "artifacts" / "T143" / "interface-inventory.json").read_text(encoding="utf-8"))
    assert set(inventory["python_packages_present_here"]) == set(targets.PYTHON_PACKAGES)
    assert findings["The listed interfaces are qualified for plant integration"]["evidence_status"] == "not_established"
    for entry, code in ((dict(arch.INTERFACES[1], write_path="enabled"), "write_path_enabled"),
                        (dict(arch.INTERFACES[1], fieldbus_role="master"), "write_capable_direction"),
                        (dict(arch.INTERFACES[0], identity=["library version", "*"]), "unpinned_identity")):
        with pytest.raises(arch.ArchitectureRefusal) as refused:
            arch.validate_inventory([entry])
        assert refused.value.code == code
    for pin in ("", "latest-stable", "nightly", "main", "HEAD", ">=1.0", "1.x", "2023.2_nightly", "N/A", "~1.2"):
        with pytest.raises(arch.ArchitectureRefusal) as refused:
            arch.validate_inventory([dict(arch.INTERFACES[0], identity=[pin])])
        assert refused.value.code == "unpinned_identity", pin
    # Prose that merely contains a branch-like word stays a valid description of what is pinned.
    for pin in ("main board firmware version", "current sensor firmware version", "/dev/ttyUSB0 adapter serial number",
                "x86-64 build flags", "printhead firmware version"):
        arch.validate_inventory([dict(arch.INTERFACES[0], identity=[pin])])


def test_t144_architecture_scan(tmp_path):
    report, findings = _run("T144", tmp_path)
    _assert_clean(report, "completed", "numerically_verified")
    closure = findings["Evidence and identity closure is standard-library Python with no native loading or process spawns"]
    # Ancestor packages run their __init__ on import, so they belong to the closure.
    assert closure["value"] == ["ciw", "ciw.core", "ciw.core.identities", "ciw.lab", "ciw.lab.evidence",
                                "ciw.lab.report"]
    assert closure["evidence_status"] == "numerically_verified"
    structural = findings["Package-wide structural rules hold (no shell spawns, native loading only in declared "
                          "hardware probes, no compiled extensions)"]
    assert structural["value"] == 0 and structural["evidence_status"] == "numerically_verified"
    assert findings["Scanner flags forged modules that cross the boundary"]["value"] == 11
    scan = arch.scan_package()
    forged = arch.mutated_scan(scan, "ciw.lab", "import numpy\n")
    assert ("evidence_closure_not_stdlib", "ciw.lab", "numpy") in arch.violations(forged)
    unpinned = findings["Some process spawns run a PATH-resolved executable without comparing it to a pinned identity"]
    assert unpinned["counterexample"]["statement"] == "Numerical providers are invoked only through pinned executables"
    pinned = findings["Numerical providers are invoked only through pinned executables"]
    assert pinned["evidence_status"] == "not_established" and pinned["expected_not_established"] is True
    assert len(report["provider_runtime_identity"]["scanned_package_sha256"]) == 64
    assert findings["Keeping native code behind subprocess boundaries makes machine interfaces safe"][
        "evidence_status"] == "not_established"
    scanned = arch.scan_source("ciw.lab.x", "import subprocess as sp\nsp.check_output('ls', shell=True)\n")
    assert scanned["spawn_lines"] == [2] and scanned["shell_lines"] == [2] and not scanned["identity_tokens"]
    comment_only = arch.scan_source("ciw.lab.x", '"""sha256 digest"""\nimport subprocess\n# digest\n'
                                                 'subprocess.run(["x"])\n')
    assert comment_only["spawn_lines"] == [4] and not comment_only["identity_tokens"]
    asynchronous = arch.scan_source("ciw.lab.x", "import asyncio\nasyncio.create_subprocess_shell('x')\n")
    assert asynchronous["spawn_lines"] == [2] and asynchronous["shell_lines"] == [2]
    imported = arch.scan_source("ciw.lab.x", "from os import posix_spawn as ps\nfrom shutil import which\n"
                                             "ps(which('x'), ['x'], {})\n")
    assert imported["spawn_lines"] == [3] and imported["which_lines"] == [3]
    relative = arch.scan_source("ciw.lab.report", "from ..core.identities import digest\nfrom .evidence import finding\n")
    assert "ciw.core.identities" in relative["imports"] and "ciw.lab.evidence" in relative["imports"]


def test_t145_julia_partial_with_plan(tmp_path):
    report, findings = _run("T145", tmp_path)
    _assert_clean(report, "partial", "analytic")
    assert findings["Julia environment pinned and exercised through the CIW to SCR boundary"]["evidence_status"] == \
        "not_established"
    assert findings["Julia provider pin procedure"]["evidence_status"] == "analytic"
    # No Julia execution path exists, so the next task is to build one, not merely to provision Julia.
    assert report["recommended_next_task"].startswith("Implement the Julia worker behind the SCR boundary")
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


def test_t146_reference_float_rule(monkeypatch):
    # The reference encoder never calls json: it must work with json.dumps disabled.
    monkeypatch.setattr(json, "dumps", lambda *args, **kwargs: pytest.fail("reference encoder used json.dumps"))
    for value, expected in serial.SPEC_EXAMPLES:
        assert serial.canonical_bytes(value) == expected
    assert serial.format_float(1e15 + 0.25) == "1000000000000000.2" and serial.decimal_tie(1e15 + 0.25)
    assert serial.format_float(123456789012345.625) == "123456789012345.62"
    assert not serial.decimal_tie(0.1) and serial.format_float(0.1) == "0.1"
    monkeypatch.undo()
    corpus = serial.float_corpus()
    ties = corpus["ties"]
    assert len(ties) >= 50 and all(serial.decimal_tie(x) for x in ties)
    for x in corpus["vector"] + corpus["random"][:500] + ties + [1.7976931348623157e308, 5e-324, -0.0]:
        assert serial.format_float(x) == repr(x), x


def test_t146_python_canonicalizers_and_vectors(tmp_path):
    assert serial.canonical_bytes([1e16, 1e15, 1e-5, 1e-4, -0.0, 5e-324, 1.0]) == \
        b"[1e+16,1000000000000000.0,1e-05,0.0001,-0.0,5e-324,1.0]"
    assert serial.canonical_bytes({"b": 1, "\U0001f600": 2, "\uff21": 3}).decode() == \
        '{"b":1,"\uff21":3,"\U0001f600":2}'
    for value, code in (([math.nan], "nonfinite_number"), (2 ** 53, "unsafe_integer"), ({1: "x"}, "non_string_key"),
                        ("\ud800", "invalid_unicode")):
        with pytest.raises(serial.CanonicalRefusal) as refused:
            serial.canonical_bytes(value)
        assert refused.value.code == code
    text = "\u00e9\x7f"
    assert telemetry_canonical(text) == serial.canonical_bytes(text) != canonical_json(text).encode()
    assert canonical_json({1: "x"}) == canonical_json({"1": "x"})
    assert serial.ecmascript_number(1.0) == "1" and serial.ecmascript_number(1e-7) == "1e-7"
    report, findings = _run("T146", tmp_path)
    _assert_clean(report, "completed" if _rust_build() is not None else "partial", "numerically_verified")
    telemetry = findings["ciw.telemetry.canonical reproduces the specification bytes on every accepted vector and "
                         "tested float"]
    assert telemetry["value"]["vector_mismatches"] == 0 and telemetry["value"]["float_mismatches"] == 0
    assert telemetry["value"]["floats"] > 2000
    identities = findings["ciw.core.identities.canonical_json reproduces the ASCII-escaped variant, not the "
                          "specification bytes"]["value"]
    assert identities["ascii_variant_mismatches"] == 0 and "unicode-bmp" in identities["differs_from_specification"]
    differs = findings["ciw.core.identities.canonical_json and ciw.telemetry.canonical produce different bytes for "
                       "non-ASCII text"]
    assert differs["value"]["differing_vectors"] == identities["differs_from_specification"]
    assert differs["counterexample"]["witness"]["telemetry_sha256"] != differs["counterexample"]["witness"][
        "identities_sha256"]
    floats = findings["The exact shortest-digit rule agrees with CPython repr and NumPy Dragon4 on vector, random and "
                      "decimal-tie binary64 values"]
    assert floats["evidence_status"] == "independently_verified"
    assert floats["value"]["cpython_mismatches"] == 0 and floats["value"]["numpy_mismatches"] == 0
    assert findings["Byte-identical canonical JSON holds for Julia, C++ and GPU-host implementations"][
        "evidence_status"] == "not_established"
    vectors = json.loads((tmp_path / "artifacts" / "T146" / "test-vectors.json").read_text(encoding="utf-8"))
    by_name = {row["name"]: row for row in vectors["vectors"]}
    assert by_name["signed-zero"]["canonical"] == '[0.0,-0.0,{"z":-0.0}]'
    # Each value is an exact tie between two shortest candidates; the even last digit wins (CPython repr).
    assert by_name["decimal-ties"]["canonical"] == \
        "[1000000000000000.2,123456789012345.62,252611590017749.62,-29474221479446.812]"
    assert all(serial.decimal_tie(x) for x in dict(serial.vectors())["decimal-ties"])
    assert by_name["empty-object"]["sha256"] == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    assert [ord(ch) for ch in by_name["no-normalization"]["canonical"]] == \
        [0x5B, 0x22, 0xE9, 0x22, 0x2C, 0x22, 0x65, 0x301, 0x22, 0x5D]


def test_t146_rust_byte_identity(rust_probe):
    accepted, invalid = serial.vectors(), serial.invalid_vectors()
    corpus = serial.float_corpus()
    floats = corpus["vector"] + corpus["random"] + corpus["ties"]
    results = serial.rust_canonical([v for _, v in accepted] + [v for _, v, _ in invalid] + floats)
    for (name, value), got in zip(accepted, results):
        assert got == (serial.canonical_bytes(value), serial.canonical_bytes(value, ascii_only=True)), name
    assert results[len(accepted):len(accepted) + len(invalid)] == [code for _, _, code in invalid]
    for x, got in zip(floats, results[len(accepted) + len(invalid):]):
        assert got[0] == repr(x).encode(), x
    # Rust's own {:e} rounds exact decimal ties up: the reason the probe formats at an explicit precision.
    assert serial.rust_shortest([1e15 + 0.25]) == ["1.0000000000000003e15"]


GUARANTEE_CLAIM = ("The harness detects every dropped partial product larger than twice the row tolerance under both "
                   "policies")
MISS_CLAIM = ("The float32 policy misses dropped partial products up to about its bound, as often as the product "
              "distribution predicts")
BITWISE_CLAIM = "Bitwise policy detects reduction-order differences between float64 CPU orders"


def test_t147_harness_detects_differences(tmp_path):
    report, findings = _run("T147", tmp_path)
    _assert_clean(report, "partial", "numerically_verified")
    assert findings[BITWISE_CLAIM]["value"] >= 1
    within = findings["Float64 reduction-order differences lie within the analytic error-bound policy"]
    assert within["evidence_status"] == "numerically_verified" and within["value"] < 1.0
    f32 = findings["Float32 results violate the float64 policy and satisfy the float32 policy"]["value"]
    assert f32["violations_under_f64_policy"] > 0 and f32["max_ratio_under_f32_policy"] < 1.0
    power = findings[GUARANTEE_CLAIM]
    assert power["evidence_status"] == "numerically_verified"
    power = power["value"]
    assert power["float64"] == {"faults": 131072, "fault_free_violations": 0, "above_twice_tolerance": 131072,
                                "undetected_above_twice_tolerance": 0, "undetected": 0}
    assert power["float32"]["undetected_above_twice_tolerance"] == 0 and power["float32"]["above_twice_tolerance"] > 0
    missed = findings[MISS_CLAIM]
    assert missed["evidence_status"] == "numerically_verified" and missed["value"]["undetected"] == 708
    value = missed["value"]
    assert abs(value["undetected"] - value["expected_undetected"]) <= 4 * value["sigma"]
    assert value["detected_below_band"] == 0 and value["undetected_above_band"] == 0
    witness = missed["counterexample"]["witness"]
    assert witness["largest_undetected"]["magnitude"] <= witness["largest_undetected"]["row_tolerance"]
    # A fault slightly above its row bound escapes: the candidate's own deviation points the other way.
    above = witness["above_bound"]
    assert (above["row"], above["column"]) == (14, 538) and 1.0 < above["ratio"] < 1.001
    assert value["max_undetected_ratio"] == above["ratio"] and value["undetected_above_bound"] == 1
    gpu = findings[targets.GPU_CLAIM]
    assert gpu["expected_not_established"] is True and gpu["basis"]["notes"] == [common.NO_GPU_PROBE]
    harness = findings[targets.HARNESS_CLAIM]
    assert harness["evidence_status"] == "numerically_verified"
    assert harness["value"] == {"python-scalar": False, "fma-first": True, "fma-second": True}
    assert "--capture energy-log=" in report["recommended_next_task"]
    assert "ciw lab hardware retain" in report["recommended_next_task"]
    assert common.GPU_QUESTION in report["recommended_next_task"]
    assert findings["GPU/CPU agreement establishes industrial readiness"]["evidence_status"] == "not_established"
    with pytest.raises(ValueError, match="equal shapes"):
        kernels.compare_outputs(np.zeros(2), np.zeros(3), {"mode": "bitwise"})
    with pytest.raises(ValueError, match="finite"):
        kernels.compare_outputs(np.zeros(2), np.array([0.0, np.nan]), {"mode": "bitwise"})
    assert kernels.ulp_distance(np.array([1.0]), np.array([np.nextafter(1.0, 2.0)]))[0] == 1.0


@pytest.mark.parametrize("scale", [3.0, 1.0 / 3.0])
def test_t147_fault_study_is_judged_by_the_harness(tmp_path, monkeypatch, scale):
    """A harness with a wrong tolerance or a bitwise mode that flags nothing refutes the fault findings."""
    real = kernels.compare_outputs

    def broken(reference, candidate, policy):
        if policy["mode"] == "bound":
            policy = dict(policy, tolerance=scale * np.asarray(policy["tolerance"]))
        elif policy["mode"] == "bitwise":
            policy = {"mode": "abs_rel", "abs": 1.0, "rel": 0.0}
        return real(reference, candidate, policy)

    monkeypatch.setattr(kernels, "compare_outputs", broken)
    report, _ = _run("T147", tmp_path)
    refuted = set(_refuted(report))
    assert BITWISE_CLAIM in refuted and MISS_CLAIM in refuted and targets.HARNESS_CLAIM in refuted
    # A lenient harness (3x tolerance) misses faults above twice the true tolerance.
    assert (GUARANTEE_CLAIM in refuted) == (scale > 1)
    assert report["evidence_status"]["primary"] == "not_established"


def test_t148_reduction_policies(tmp_path):
    case = [1.0, 1e100, 1.0, -1e100]
    assert kernels.sum_kahan(case) == 0.0 and kernels.sum_neumaier(case) == 2.0 and kernels.sum_exact(case) == 2.0
    rng = np.random.Generator(np.random.PCG64(7))
    xs = [float(v) for v in rng.uniform(-1, 1, 300) * 10.0 ** rng.integers(-8, 8, 300)]
    for _ in range(5):
        order = [xs[i] for i in rng.permutation(len(xs))]
        assert kernels.sum_exact(order) == math.fsum(xs)
    report, findings = _run("T148", tmp_path)
    # The policy record is a derivation, so the headline is analytic although every number is exact.
    _assert_clean(report, "completed", "analytic")
    exact = findings["Correctly rounded exact accumulation is permutation-invariant"]
    assert exact["value"] == 1 and exact["evidence_status"] == "independently_verified"
    pairwise = findings["Fixed-order pairwise summation is reproducible for one order but not permutation-invariant"]
    assert pairwise["value"]["distinct_results"] >= 2 and pairwise["counterexample"]
    # The same tree walked by NumPy level-wise additions gives the same bits on every sampled order.
    assert pairwise["value"]["tree_mismatches"] == 0 and pairwise["value"]["orders"] == 25
    assert float.fromhex(pairwise["value"]["first_order_sum_hex"]) == kernels.sum_pairwise(
        kernels.reduction_datasets()["uniform"])
    with pytest.raises(ValueError, match="power-of-two"):
        kernels.sum_pairwise_levels([1.0, 2.0, 3.0])
    rigorous = findings["Observed errors of the sequential, pairwise, Neumaier and exact sums lie within their "
                        "rigorous bounds"]
    assert set(rigorous["value"]) == {"exact", "neumaier", "pairwise", "sequential"}
    assert all(ratio <= 1.0 for ratio in rigorous["value"].values())
    old = findings["The bound 2u|S| + 4n u^2 sum|x| does not bound Neumaier summation"]["value"]
    assert old["n"] == 1002 and old["ratio_to_superseded_bound"] > 10 and old["ratio_to_rigorous_bound"] < 0.1
    distinct = findings["Distinct results under permutation for each algorithm and dataset"]
    assert distinct["evidence_status"] == "numerically_verified"
    assert all(row["exact"] == 1 for row in distinct["value"].values())


def test_t149_telemetry_only_frames(tmp_path):
    assert fpga.crc32(b"123456789") == 0xCBF43926
    frame = fpga.encode_frame(2 ** 32 - 1, 5, 7, [-1, 2 ** 31 - 1])
    assert fpga.decode_frame(frame)["channels"] == [-1, 2 ** 31 - 1] and len(frame) == 28 + 8 + 4
    assert frame[-4:] == fpga.crc32(frame[:-4]).to_bytes(4, "little")
    for forged, code in ((fpga.forge_frame(0x81, 0), "command_path_refused"),
                         (fpga.forge_frame(1, fpga.FLAG_WRITE_REQUEST), "command_path_refused"),
                         (fpga.forge_frame(0x02, 0), "unknown_frame_type"),
                         (fpga.forge_frame(1, 0, payload_bytes=12), "length_mismatch"),
                         (fpga.forge_frame(1, 0, magic=b"XXXX"), "bad_magic"),
                         (fpga.forge_frame(1, 0, version=2), "unsupported_version"),
                         (frame[:-1] + bytes([frame[-1] ^ 1]), "crc_mismatch")):
        with pytest.raises(fpga.TelemetryRefusal) as refused:
            fpga.decode_frame(forged)
        assert refused.value.code == code
    report, findings = _run("T149", tmp_path)
    _assert_clean(report, "completed", "numerically_verified")
    detection = findings["Every burst of at most 32 bits and every single-bit error in a frame is refused by the "
                         "decoder"]["value"]
    assert detection["single_bit"] == [384, 384] and detection["rank_deficient_windows"] == 0
    assert detection["burst_windows"] == 353
    big_endian = findings["A big-endian CRC trailer lets a 32-bit burst across the payload/CRC boundary escape"]["value"]
    assert big_endian["rank_deficient_windows"] == 16 and big_endian["first_window"] == 321
    msb = findings["The burst guarantee holds only in the LSB-first bit order of the reflected CRC"]["value"]
    assert msb["rank_deficient_windows_msb_first"] > 0 and min(msb["witness_bits"]) >= 8 * fpga.HEADER.size
    assert findings["CIW table-driven CRC-32 agrees with zlib and the catalogue check value"]["evidence_status"] == \
        "independently_verified"
    assert findings["Decoder, encoder and interface validator refuse every command, write or malformed path"][
        "value"] == 13
    receiver = findings["Host receiver exposes only accept() and plain data attributes, with no sending method or "
                        "transport handle"]
    assert receiver["value"] == {"public_methods": ["accept"], "bases": ["object"], "non_data_attributes": []}
    assert receiver["evidence_status"] == "numerically_verified"

    class Relay(fpga.TelemetryReceiver):  # a method name no denylist would catch
        def push(self, data):
            return data

    relay = Relay(0, 1)
    relay.port = object()
    assert fpga.receiver_interface(Relay, relay) == {"public_methods": ["accept", "push"],
                                                     "bases": ["TelemetryReceiver", "object"],
                                                     "non_data_attributes": ["port"]}
    assert findings["A telemetry-only interface guarantees the FPGA cannot actuate the machine"]["evidence_status"] == \
        "not_established"


def test_t150_bitstream_identity(tmp_path):
    report, findings = _run("T150", tmp_path)
    _assert_clean(report, "completed", "numerically_verified")
    bound = findings["Bitstream identity record binds bitstream, toolchain version and installation manifest, "
                     "constraints and source tree"]
    assert bound["value"]["refused_mutations"] == 26 and bound["value"]["accepted_positive_controls"] == 1
    codes = {check["expected_refusal"] for check in bound["basis"]["checks"]}
    assert {"toolchain_installation_mismatch", "toolchain_version_mismatch"} <= codes
    assert bound["evidence_status"] == "numerically_verified"
    assert findings["The bitstream is approved for production deployment"]["evidence_status"] == "not_established"
    record = json.loads((tmp_path / "artifacts" / "T150" / "bitstream-identity.json").read_text(encoding="utf-8"))
    assert record["synthetic"] is True and record["record_sha256"] == bound["value"]["record_sha256"]
    with pytest.raises(fpga.IdentityRefusal) as refused:
        fpga.deployment_decision(record)
    assert refused.value.code == "synthetic_bitstream"
    for version in ("2024.1.x", "1.0-latest", "2023.2_nightly", "2024.*", "2024.1\n", "\u0662\u0660\u0662\u0664.1"):
        assert not fpga._PINNED_VERSION.fullmatch(version)
    assert fpga._PINNED_VERSION.fullmatch("2024.1") and fpga._PINNED_VERSION.fullmatch("23.1.0-rc2")
    assert not fpga._HEX64.fullmatch("0" * 64 + "\n") and fpga._HEX64.fullmatch("0" * 64)
    with pytest.raises(fpga.IdentityRefusal) as refused:
        fpga.validate_identity(dict(record, source_tree="unknown"), source_files={"a.v": b""})
    assert refused.value.code == "missing_field"
    fpga.validate_identity(record, toolchain_files=targets.PLACEHOLDER_TOOLCHAIN_FILES, toolchain_version="0.0.1")
    with pytest.raises(fpga.IdentityRefusal) as refused:
        fpga.validate_identity(record, toolchain_files=dict(targets.PLACEHOLDER_TOOLCHAIN_FILES, extra=b""))
    assert refused.value.code == "toolchain_installation_mismatch"


def test_t151_compatibility_and_rollback(tmp_path):
    report, findings = _run("T151", tmp_path)
    _assert_clean(report, "completed", "numerically_verified")
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
    _assert_clean(report, "completed", "numerically_verified")
    gaps = findings["Sequence-number gap detection recovers every lost frame across the 32-bit wrap"]
    assert gaps["value"]["lost"] == gaps["value"]["detected"] > 0 and gaps["value"]["reordered"] > 0
    declared = findings["With the declared clock offset no stale frame is missed and false alarms match the timestamp "
                        "quantization and drift prediction"]
    assert declared["evidence_status"] == "numerically_verified"
    assert declared["value"]["missed"] == 0 and declared["value"]["false_alarms"] > 0
    assert abs(declared["value"]["false_alarms"] - declared["value"]["expected_false_alarms"]) < 20
    biased = findings["An offset estimated from minimum delay misses stale frames at the rate its bias predicts"]
    assert biased["evidence_status"] == "numerically_verified" and biased["value"]["false_alarms"] == 0
    assert biased["value"]["missed"] > 0
    model = findings["Loss rate and mean latency of a long run match the declared link model"]
    assert model["evidence_status"] == "numerically_verified" and model["value"]["frames"] == 200000
    naive = findings["Differencing sequence numbers in arrival order miscounts losses under reordering"]["value"]
    assert naive["arrival_order"] > naive["true"] and naive["in_order"] == naive["true"] - 1
    wrap = findings["Non-modular differencing misses the loss at the 32-bit wrap"]["value"]
    assert wrap["wrap_gap"] == [2 ** 32 - 1]
    assert findings["Simulated loss, latency and staleness represent the real FPGA telemetry link"][
        "evidence_status"] == "not_established"
    moments = fpga.gilbert_elliott_moments(fpga.LINK_MODEL)
    assert abs(moments["stationary_loss"] - (0.2 / 0.205 * 0.01 + 0.005 / 0.205 * 0.5)) < 1e-15
    assert abs(float(fpga.latency_cdf(2_000_000 + 500_000)) - (1 - 2 * math.exp(-1))) < 1e-12


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
    # The validity window is compared as instants, not strings.
    for valid_until, code in (("2026-09-23T01:00:00+02:00", "authorization_expired"),
                              ("2026-09-23T00:00:00.500Z", "no_trust_anchor"),
                              ("tomorrow", "malformed_timestamp"), ("2027-01-01T00:00:00", "malformed_timestamp"),
                              ("2026-09-23T00:00:00Z", "authorization_expired")):
        with pytest.raises(authority.AuthorityRefusal) as refused:
            authority.verify_authorization(dataclasses.replace(targets.EXTERNAL, valid_until=valid_until),
                                           channel="spindle_speed", now=NOW)
        assert refused.value.code == code, valid_until
    with pytest.raises(authority.AuthorityRefusal) as refused:
        authority.verify_authorization(targets.EXTERNAL, channel="spindle_speed", now="2026-09-23T02:00:00+02:00")
    assert refused.value.code == "no_trust_anchor"
    report, findings = _run("T153", tmp_path)
    _assert_clean(report, "completed", "numerically_verified")
    assert findings["Default policy refuses writes on every declared channel"]["value"] == len(targets.WRITE_CHANNELS)
    assert findings["Enabling writes is refused on every route, including a well-formed external record"]["value"] == 12
    paths = findings["No ciw module imports a device, serial or fieldbus library or names a device node, and network "
                     "libraries appear only in the declared workbench servers (source scan)"]
    assert paths["evidence_status"] == "numerically_verified"
    assert paths["value"]["device_paths"] == [] and paths["value"]["undeclared_network"] == []
    assert set(paths["value"]["declared_network"]) <= set(arch.DECLARED_NETWORK)
    scanned = arch.scan_source("ciw.lab.x", "import serial\nopen('/dev/ttyACM0', 'wb').write(b'M3')\n")
    assert scanned["device_libraries"] == ["serial"] and scanned["device_nodes"] == ["/dev/ttyACM0"]
    assert findings["A frozen in-process policy object can be mutated"]["counterexample"]
    assert findings["The lab holds actuator write authority"]["evidence_status"] == "not_established"
    assert findings["The lab holds actuator write authority"]["domain"] == "actuator_authority"


def test_t154_control_outputs_are_proposals(tmp_path):
    report, findings = _run("T154", tmp_path)
    _assert_clean(report, "completed", "numerically_verified")
    outputs = findings["Control outputs built by this section are constructed with status proposal and cannot be "
                       "converted to a command here"]
    assert outputs["value"] == {"refusals": 7, "status": "proposal"}
    tampered = findings["A frozen control proposal's status can be forced in memory, and the forced object is refused"]
    assert tampered["value"] == {"refusals": 3, "status_forced": True} and tampered["counterexample"]
    orders = findings["Jacobi heading proposal cancels a lateral offset to second order"]["value"]
    assert abs(orders["corrected_order"] - 2.0) < 0.2 and abs(orders["uncorrected_order"] - 1.0) < 0.2
    assert findings["Heading proposals are authorized for execution as actuator commands"]["evidence_status"] == \
        "not_established"
    inventory = findings["Control-like outputs named in the package are all inventoried (keyword scan)"]
    assert inventory["evidence_status"] == "numerically_verified"
    assert "ciw.lab.lyapunov_research" in inventory["value"]["modules_naming_control_terms"]
    assert inventory["value"]["proposals"] == ["Jacobi heading correction"]
    scope = findings["Every control-like output of the workbench is a ControlProposal"]
    assert scope["expected_not_established"] is True and "stop request" in scope["basis"]["notes"]
    assert any("T114 servo-axis abort" in item for item in report["unresolved_assumptions"])
    record = json.loads((tmp_path / "artifacts" / "T154" / "proposal-record.json").read_text(encoding="utf-8"))
    assert record["status"] == "proposal"
    with pytest.raises(authority.AuthorityRefusal) as refused:
        authority.ControlProposal("heading_offset", 0.1, "rad", "test", "0" * 64, "test", status="command")
    assert refused.value.code == "proposal_status_fixed"
    proposal = authority.ControlProposal("heading_offset", 0.1, "rad", "test", "0" * 64, "test")
    with pytest.raises(authority.AuthorityRefusal) as refused:
        authority.to_command(proposal, now=NOW)
    assert refused.value.code == "proposal_not_authorized"
    object.__setattr__(proposal, "status", "command")
    with pytest.raises(authority.AuthorityRefusal) as refused:
        proposal.record()
    assert refused.value.code == "proposal_status_tampered"


_REAL_RUN_GPU = common.run_gpu


def _gpu_identity():
    import hashlib
    return {"device_name": "NVIDIA GeForce RTX 2080", "device_uuid": "GPU-5d3f9a2c-7e41-4b8a-9c06-1f2e3d4c5b6a",
            "compute_capability": [7, 5], "cuda_driver_version": 12040, "kernel_sha256": common.kernel_sha256(),
            "prepared_input_sha256": hashlib.sha256(common.prepared_inputs().tobytes()).hexdigest()}


class _RejectingWorker:
    """Stands in for ciw.energy_cuda.CudaGaussianWorker: the kernel ran, then solve()'s own output check raised."""

    def __init__(self, problem, solver, iterations, replicas=4096, device_index=0):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def identity(self):
        return _gpu_identity()

    def solve(self):
        raise ValueError("GPU covariance lost symmetry")


def _simulated_gpu(monkeypatch, outputs=None, unavailable=None):
    """A host whose nvidia-gpu probe answers and whose PTX kernel returns ``outputs`` (or fails with ``unavailable``)."""
    monkeypatch.setattr(runner, "_probe_hardware", lambda name: name == "nvidia-gpu")
    result = {"unavailable": unavailable} if unavailable else {"outputs": outputs, "identity": _gpu_identity()}
    monkeypatch.setattr(common, "run_gpu", lambda: result)


def test_t147_common_workload_comparison_follows_the_gpu_probe(tmp_path, monkeypatch):
    """T147 judges the PTX kernel's outputs under T148's fixed-order policy where a GPU answers, and only there."""
    monkeypatch.setattr(runner, "_probe_hardware", lambda name: False)
    report, findings = _run("T147", tmp_path / "no-gpu")
    assert report["state"] == "partial" and findings[targets.GPU_CLAIM]["expected_not_established"] is True
    assert targets.FIXED_ORDER_POLICY["mode"] == "bitwise" and "fixed_layout_arrays" in targets.FIXED_ORDER_POLICY["rule"]
    _simulated_gpu(monkeypatch, outputs=common.run_numpy(np.float64))
    report, findings = _run("T147", tmp_path / "gpu")
    gpu = findings[targets.GPU_CLAIM]
    assert gpu["evidence_status"] == "numerically_verified"
    assert gpu["value"] == {"elements": 6 * common.REPLICAS, "violations": 0, "max_ulp": 0.0, "policy": "bitwise"}
    _assert_clean(report, "completed", "numerically_verified")
    assert report["provider_runtime_identity"]["gpu"]["device_name"] == "NVIDIA GeForce RTX 2080"
    # A device that contracted the kernel's multiply-adds violates the policy: refuted, not hidden.
    _simulated_gpu(monkeypatch, outputs=np.tile(common.run_variant("fma-second"), (common.REPLICAS, 1)))
    report, findings = _run("T147", tmp_path / "contracted")
    assert targets.GPU_CLAIM in _refuted(report) and findings[targets.GPU_CLAIM]["value"]["violations"] > 0
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "not_established"
    _simulated_gpu(monkeypatch, outputs=common.run_numpy(np.float64)[:5])
    report, findings = _run("T147", tmp_path / "short")
    assert targets.GPU_CLAIM in _refuted(report) and report["state"] == "partial"
    _simulated_gpu(monkeypatch, unavailable="the PTX kernel did not run: CudaError: simulated")
    report, findings = _run("T147", tmp_path / "failed")
    assert findings[targets.GPU_CLAIM]["basis"]["notes"] == ["the PTX kernel did not run: CudaError: simulated"]
    assert report["state"] == "partial"
    # A device that drops the kernel's neg.f64 flips the sign of one output column: every flipped value violates.
    flipped = common.run_numpy(np.float64)
    flipped[:, 3] = -flipped[:, 3]
    _simulated_gpu(monkeypatch, outputs=flipped)
    report, findings = _run("T147", tmp_path / "sign-flipped")
    assert targets.GPU_CLAIM in _refuted(report) and findings[targets.GPU_CLAIM]["value"]["violations"] == common.REPLICAS
    assert report["evidence_status"]["primary"] == "not_established"
    # Outputs the CUDA worker rejected after the kernel ran (solve() raises ValueError after launch, sync and copy)
    # are a refutation, never an expected gap.
    from ciw import energy_cuda
    monkeypatch.setattr(common, "run_gpu", _REAL_RUN_GPU)
    monkeypatch.setattr(energy_cuda, "CudaGaussianWorker", _RejectingWorker)
    report, findings = _run("T147", tmp_path / "rejected")
    rejected = findings[targets.GPU_CLAIM]
    assert targets.GPU_CLAIM in _refuted(report) and not rejected.get("expected_not_established")
    assert rejected["value"]["rejected_by_worker"].endswith("GPU covariance lost symmetry")
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "not_established"


def test_t147_identity_digests_the_common_workload_sources(tmp_path, monkeypatch):
    """The kernel, planner and information-system modules that define the workload are part of T147's identity."""
    monkeypatch.setattr(runner, "_probe_hardware", lambda name: False)
    report, _ = _run("T147", tmp_path)
    identity = report["provider_runtime_identity"]
    assert set(common.SOURCES) <= set(identity["sources"]) and all(len(identity["sources"][s]) == 64
                                                                   for s in common.SOURCES)
    assert identity["common_workload"] == common.workload()


def test_ciw_producers_of_independent_checks_carry_a_revision(tmp_path):
    """Every ciw-side producer of an independent check names the package version and its module digest (C7)."""
    from ciw import __version__
    # T145's only independent check is the SymPy derivation, which runs where SymPy (the lab extra) is installed.
    symbolic = ("T145",) if importlib.util.find_spec("sympy") is not None else ()
    for task_id in (*symbolic, "T146", "T148", "T149"):
        report, _ = _run(task_id, tmp_path / task_id)
        checks = [f["basis"]["independent_check"] for f in report["findings"] if "independent_check" in f["basis"]]
        assert checks, task_id
        for check in checks:
            for side in ("producer", "checker"):
                identity = check[side]
                assert isinstance(identity.get("revision"), str) and identity["revision"], (task_id, side)
            if check["producer"]["implementation"].startswith("ciw.") and task_id != "T145":
                assert check["producer"]["revision"] == f"ciw {__version__}"
                assert len(check["producer"]["source_sha256"]) == 64


def test_next_steps_point_at_work_that_delivers(tmp_path):
    """No next step points at a queue task (all of T142-T154 have run); open work is a named question."""
    implementations = section_implementations("implementation-targets")
    context = runner.Context(tmp_path)
    for task_id in sorted(implementations):
        report = runner.run_task(TASKS[task_id], implementations[task_id], context, {})
        kept, stale = planner.next_step_items(report["recommended_next_task"], task_id, set(TASKS))
        assert stale == [] and not [item for item in kept if item[0] == "pointer"], task_id
    t142 = runner.run_task(TASKS["T142"], implementations["T142"], context, {})["recommended_next_task"]
    assert t142.startswith("Done in this task") and "rapl-capture" in t142
    t148 = runner.run_task(TASKS["T148"], implementations["T148"], context, {})["recommended_next_task"]
    assert t148 == common.GPU_QUESTION
