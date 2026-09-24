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
    names = findings["Industrial interfaces that need C/C++ libraries behind a pinned subprocess boundary"]
    assert names["evidence_status"] == "analytic" and len(names["value"]) == 5
    assert "EtherCAT passive monitoring (network TAP capture)" in names["value"]
    refusals = findings["Inventory validator refuses in-process bindings, write-capable directions or bus roles and "
                        "unpinned entries"]
    assert refusals["value"] == 10 and refusals["evidence_status"] == "numerically_verified"
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
    assert closure["value"] == ["ciw.core.identities", "ciw.lab.evidence", "ciw.lab.report"]
    assert closure["evidence_status"] == "numerically_verified"
    structural = findings["Package-wide structural rules hold (no shell spawns, native loading only in declared "
                          "hardware probes, no compiled extensions)"]
    assert structural["value"] == 0 and structural["evidence_status"] == "numerically_verified"
    assert findings["Scanner flags forged modules that cross the boundary"]["value"] == 8
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


def test_t147_harness_detects_differences(tmp_path):
    report, findings = _run("T147", tmp_path)
    _assert_clean(report, "partial", "numerically_verified")
    assert findings["Bitwise policy detects reduction-order differences between float64 CPU orders"]["value"] >= 1
    within = findings["Float64 reduction-order differences lie within the analytic error-bound policy"]
    assert within["evidence_status"] == "numerically_verified" and within["value"] < 1.0
    f32 = findings["Float32 results violate the float64 policy and satisfy the float32 policy"]["value"]
    assert f32["violations_under_f64_policy"] > 0 and f32["max_ratio_under_f32_policy"] < 1.0
    power = findings["Dropped partial products larger than the policy bound plus the candidate's deviation from the "
                     "reference are detected under both policies"]
    assert power["evidence_status"] == "numerically_verified"
    power = power["value"]
    assert power["float64"] == {"faults": 131072, "undetected": 0, "guarantee_violations": 0,
                                "undetected_above_bound": 0}
    assert power["float32"]["guarantee_violations"] == 0 and power["float32"]["undetected"] > 0
    assert power["float32"]["undetected_above_bound"] == 1
    missed = findings["The float32 policy misses dropped partial products up to about its bound"]
    assert missed["evidence_status"] == "numerically_verified" and missed["value"]["undetected"] == 708
    witness = missed["counterexample"]["witness"]
    assert witness["largest_undetected"]["magnitude"] <= witness["largest_undetected"]["row_tolerance"]
    # A fault slightly above its row bound escapes: the candidate's own deviation points the other way.
    above = witness["above_bound"]
    assert (above["row"], above["column"]) == (14, 538) and 1.0 < above["ratio"] < 1.001
    assert missed["value"]["max_undetected_ratio"] == above["ratio"]
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
    # The policy record is a derivation, so the headline is analytic although every number is exact.
    _assert_clean(report, "completed", "analytic")
    exact = findings["Correctly rounded exact accumulation is permutation-invariant"]
    assert exact["value"] == 1 and exact["evidence_status"] == "independently_verified"
    pairwise = findings["Fixed-order pairwise summation is reproducible for one order but not permutation-invariant"]
    assert pairwise["value"] >= 2 and pairwise["counterexample"]
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
    assert findings["Host receiver exposes no sending or writing method"]["value"] == []
    assert findings["A telemetry-only interface guarantees the FPGA cannot actuate the machine"]["evidence_status"] == \
        "not_established"


def test_t150_bitstream_identity(tmp_path):
    report, findings = _run("T150", tmp_path)
    _assert_clean(report, "completed", "numerically_verified")
    bound = findings["Bitstream identity record binds bitstream, toolchain, constraints and source tree"]
    assert bound["value"]["refused_mutations"] == 23 and bound["value"]["accepted_positive_controls"] == 1
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
    report, findings = _run("T153", tmp_path)
    _assert_clean(report, "completed", "numerically_verified")
    assert findings["Default policy refuses every write attempt"]["value"] == {"attempts": 1000, "accepted": 0}
    assert findings["Enabling writes is refused on every route, including a well-formed external record"]["value"] == 8
    assert findings["A frozen in-process policy object can be mutated"]["counterexample"]
    assert findings["The lab holds actuator write authority"]["evidence_status"] == "not_established"
    assert findings["The lab holds actuator write authority"]["domain"] == "actuator_authority"


def test_t154_control_outputs_are_proposals(tmp_path):
    report, findings = _run("T154", tmp_path)
    _assert_clean(report, "completed", "numerically_verified")
    outputs = findings["Control outputs are constructed with status proposal and cannot be converted to a command here"]
    assert outputs["value"] == {"refusals": 7, "status": "proposal"}
    tampered = findings["A frozen control proposal's status can be forced in memory, and the forced object is refused"]
    assert tampered["value"] == {"refusals": 3, "status_forced": True} and tampered["counterexample"]
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
    object.__setattr__(proposal, "status", "command")
    with pytest.raises(authority.AuthorityRefusal) as refused:
        proposal.record()
    assert refused.value.code == "proposal_status_tampered"
