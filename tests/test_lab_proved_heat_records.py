"""Retained SP1 proved-heat gate records and T099's binding to them.

The record tests build a synthetic gate output in a temporary directory. Its
bundles are CIW's own proved-heat bundles around explicit test-double proof
bytes (never a cryptographic proof), consistent enough to pass CIW's offline
validation, so each refusal below is the one its named defect causes. T099's
tests stand in for the SCR build and engine run, as the section's own tests
do, and need git. The pinned-toolchain test needs CIW_LAB_SCR_REPO, cargo with
the CI-pinned rustup toolchain and a real record (CIW_LAB_PROVED_HEAT_RECORD,
or the repository's lab/proved-heat/), and skips otherwise. Tests of the
scripts, workflows and the repository's record skip where those files are not
beside the tests (the clean room).
"""
import base64
from copy import deepcopy
import gzip
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import zlib

import pytest

from ciw.lab import exchange_provenance_bundles as section
from ciw.lab import exchange_provenance_bundles_fixtures as fixtures
from ciw.lab import exchange_provenance_bundles_providers as providers
from ciw.lab import proved_heat_records as records
from ciw.lab import registry, runner
from ciw.lab.report import finding_row

ROOT = Path(__file__).resolve().parents[1]
QUEUE = {item["id"]: item for item in registry.load_queue()["tasks"]}
ENGINE = b"synthetic execution-cli bytes, not an engine"
PROVER = b"synthetic sp1-host bytes, not a prover"
MEMORY = {"status": "measured", "unit": "byte", "bytes": 2 * 1024 * 1024, "scope": "max_waited_child_peak_rss"}
RUN_ID = "synthetic-2026-09-24"
SHOWN = f"pinned provider run ({providers.REPOSITORIES['scr']}@a59aba283b03)"
# T099's stand-in pinned-toolchain build fails when asked to build these bytes.
FAILED_BUILD = b""


def _source() -> dict:
    from ciw.proved_heat import POLICY
    return {"schema": "ciw.proved-heat-source.v1", "experiment_id": "synthetic-record-test-double",
            "initial_values": [0, 100, 200, 100, 0], "steps": 4, "configuration": deepcopy(POLICY)}


def _runtime() -> dict:
    """A pinned SCR runtime identity naming the synthetic engine and prover and the registered guest."""
    from ciw import proved_heat
    runtime = {"schema": "ciw.subprocess-runtime.v1", "adapter_version": "ciw-pinned-subprocess-v1",
               "repository_root": "synthetic-scr-checkout", **proved_heat.PIN, "python_executable": "synthetic-python",
               "python_sha256": "a" * 64, "python_version": "3.12.3", "dependencies": {"numpy": "2.4.3", "scipy": None}}
    for role, data in (("engine", ENGINE), ("prover", PROVER)):
        runtime[role] = {"sha256": "sha256:" + sha256(data).hexdigest(), "byte_count": len(data),
                         "source_binding": "operator_asserted_not_attested"}
    runtime["guest"] = {"sha256": "sha256:" + proved_heat.GUEST_SHA256, "byte_count": 50264,
                        "source_binding": "registered_guest_sha256"}
    return runtime


def _data(value: dict, label: str) -> dict:
    """One proved execution as a structurally bound test double; its proof bytes are not an SP1 proof."""
    from ciw.declared_workload import HEAT_DESCRIPTOR, _commit
    from ciw.proved_heat import BACKEND, GUEST_SHA256
    from ciw.telemetry import byte_digest
    values = fixtures.heat_reference(value["initial_values"], value["steps"])
    inputs = struct.pack("<II", value["steps"], len(values)) + b"".join(struct.pack("<q", v)
                                                                       for v in value["initial_values"])
    output = b"".join(struct.pack("<q", v) for v in values)
    native = {"specification": {"program": HEAT_DESCRIPTOR.hex(), "configuration": "", "input_payload": inputs.hex()},
              "specification_identity": _commit("specification", [HEAT_DESCRIPTOR, b"", inputs]),
              "program_identity": _commit("program", [HEAT_DESCRIPTOR]), "input_identity": _commit("input", [inputs]),
              "engine_occurrence": 0, "status": "completed", "exit_code": 0, "output": output.hex(),
              "output_identity": _commit("output", [output]), "computation_identity": "", "detail": None,
              "values": values}
    native["computation_identity"] = _commit("computation", [bytes.fromhex(native[key]) for key in (
        "program_identity", "input_identity", "output_identity")] + [struct.pack("<I", 0)])
    raw = f"TEST DOUBLE, NOT AN SP1 PROOF ({label})".encode()
    proof = {"bytes_b64": base64.b64encode(raw).decode(), "sha256": byte_digest(raw),
             "byte_count": len(raw), "identity": _commit("proof", [b"sp1-cpu", b"v6.1.0", raw]),
             "backend_name": "sp1-cpu", "backend_version": "v6.1.0", "guest_sha256": "sha256:" + GUEST_SHA256}
    return {"native": native, "proof": proof,
            "verifier": {"command": "verify", "outcome": "verified",
                         "coverage": "program=true input=true output=true exit_code=true",
                         "proof_identity": proof["identity"], "backend": BACKEND,
                         "statement_program": native["program_identity"]},
            "timings": {"native_seconds": 0.1, "prove_and_verify_seconds": 1.5, "reverify_seconds": 0.5,
                        "memory": deepcopy(MEMORY)}}


def _bundles():
    """The source bytes and an original and replayed proved-heat bundle, made by CIW's own workflow code."""
    from ciw.proved_heat import ProvedHeatWorkflow
    from ciw.telemetry import canonical
    raw = canonical(_source())
    labels = iter(("original", "replay"))
    workflow = ProvedHeatWorkflow()
    workflow._invoke = lambda value, *args, **kwargs: _data(value, next(labels))
    workflow._adapters = lambda *args, **kwargs: (None, _runtime(), {})
    original = workflow._execute(raw, (None, _runtime(), {}))
    return raw, original, workflow.replay_session(original, {})["session"]


def _reverification(original: dict) -> dict:
    from ciw.proved_heat import _identify, _verification
    from ciw.telemetry import digest
    report = _verification(original, "verification-" + "c" * 32)
    report.pop("verification_id")
    report.update(seconds=0.5, memory=deepcopy(MEMORY), verifier_runtimes={"scr": _runtime()})
    report["verifier_runtime_digest"] = digest(report["verifier_runtimes"])
    return _identify(report)


def _local_run() -> dict:
    pins = records.pins()
    return {"schema": records.LOCAL_RUN_SCHEMA, "started_utc": "2026-09-24T12:00:00Z",
            "record_origin": "written_by_driver",
            "procedure": {"kind": "local_workflow_replay", "workflow": ".github/workflows/proved-heat.yml",
                          "driver": "scripts/run_proved_heat_locally.py"},
            "steps": [{"name": "actions/checkout@v4", "status": "not_run", "reason": "synthetic"},
                      {"name": "Exercise actual proofs through an isolated installed CIW wheel", "status": "ran",
                       "seconds": 1.0, "exit_code": 0}],
            "host_facts": {"system": "Linux"}, "toolchains": {"rustc": "rustc 1.94.0 (synthetic)"},
            "sources": {"scr": dict(pins["scr"]), "sp1": dict(pins["sp1"])},
            "compiler_archive": {"sha256": pins["compiler_archive_sha256"], "checked": "synthetic"}}


def gate_output(directory: Path, gate_changes=None, local_changes=None) -> Path:
    """A passing proved-heat gate output as the workflow and the local driver lay it out (test doubles inside)."""
    from ciw.telemetry import canonical
    pins = records.pins()
    raw, original, replay = _bundles()
    (directory / "gate").mkdir(parents=True)
    (directory / "local-logs").mkdir()

    def write(name, data):
        (directory / name).write_bytes(data if isinstance(data, bytes) else json.dumps(data, indent=2).encode())

    write("gate/source.json", raw)
    write("gate/original.json", canonical(original))
    write("gate/replay.json", canonical(replay))
    write("gate/reverification.json", canonical(_reverification(original)))
    write("gate/workspace.json", {"workspace_version": 3, "workbench": {"bundles": [
        {"kind": "proved-heat", "native": original}, {"kind": "proved-heat", "native": replay}]}})
    tests = [*records.REQUIRED_NATIVE_TESTS, "test_source_contract_bounds[steps-True]"]
    write("gate/tests.xml", ('<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">'
                             + "".join(f'<testcase classname="tests.test_proved_heat_session" name="{name}" '
                                       'time="0.1" />' for name in tests) + "</testsuite></testsuites>").encode())
    natives = {"engine": ENGINE, "prover": PROVER}
    gate = {"schema": records.GATE_SCHEMA, "status": "passed", "execution": records.GATE_EXECUTION,
            "physical_truth": "not_established", "scr": dict(pins["scr"]), "sp1": dict(pins["sp1"]),
            "native_artifacts": {**{role: {"sha256": sha256(data).hexdigest(), "byte_count": len(data)}
                                    for role, data in natives.items()},
                                 "guest": {"sha256": pins["guest_sha256"], "byte_count": 50264}},
            "host_executable_binding": records.HOST_BINDING, "tests_passed": len(tests),
            "installed_wheel": {"filename": "synthetic.whl", "sha256": "d" * 64, "byte_count": 1},
            "measurements": {label: records._measurements(bundle) for label, bundle in (("original", original),
                                                                                         ("replay", replay))}}
    write("gate/gate.json", {**gate, **(gate_changes or {})})
    write("build.json", {"recipe_identity": pins["recipe_identity"], "elf_sha256": pins["guest_sha256"],
                         "elf_path": "synthetic", "toolchain_identity": "synthetic",
                         "compiler_archive_sha256": pins["compiler_archive_sha256"],
                         "sp1_revision": pins["sp1"]["revision"], "guest_build": "verified_against_committed_registry"})
    write("source-checks.json", {"sp1_revision": pins["sp1"]["revision"], "sp1_tree": pins["sp1"]["tree"],
                                 "tracked_build_source_bytes": "unchanged",
                                 "runtime_reference": "separate_clean_checkout",
                                 "generated_build_files_excluded_from_reference": ["target/generated"]})
    write("local-logs/01-step.log", b"synthetic step log\n")
    write("local-run.json", {**_local_run(), **(local_changes or {})})
    return directory


def retain(tmp_path: Path, **changes) -> Path:
    """A synthetic gate output retained with the tool; the record directory."""
    run = gate_output(tmp_path / "gate-run", **changes)
    records.retain_record(run, tmp_path / "lab", RUN_ID, "synthetic test host")
    return tmp_path / "lab" / "proved-heat" / RUN_ID


def _reseal(directory: Path) -> None:
    """Recompute every manifest entry, as a deliberate editor would; pins and bundle checks still apply."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for path in manifest["files"]:
        data = (directory / path).read_bytes()
        entry = {"sha256": sha256(data).hexdigest(), "bytes": len(data)}
        if path.endswith(".gz"):
            content = gzip.decompress(data)
            entry["content"] = {"sha256": sha256(content).hexdigest(), "bytes": len(content)}
        manifest["files"][path] = entry
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _edit(directory: Path, path: str, change) -> None:
    """Change one retained JSON file (compressed or not) and reseal the manifest."""
    location = directory / path
    raw = location.read_bytes()
    value = json.loads(gzip.decompress(raw) if path.endswith(".gz") else raw)
    change(value)
    data = json.dumps(value).encode()
    location.write_bytes(records.compress(data) if path.endswith(".gz") else data)
    _reseal(directory)


def test_a_retained_record_keeps_the_gate_outputs_and_verifies(tmp_path):
    record = retain(tmp_path)
    run = tmp_path / "gate-run"
    manifest = json.loads((record / "manifest.json").read_text(encoding="utf-8"))
    assert sorted(manifest["files"]) == sorted(["run.json", *records.RETAINED.values()])
    for source, target in records.RETAINED.items():
        data = (run / source).read_bytes()
        stored = (record / target).read_bytes()
        assert (gzip.decompress(stored) if target.endswith(".gz") else stored) == data, source
        if target.endswith(".gz"):
            assert manifest["files"][target]["content"] == {"sha256": sha256(data).hexdigest(), "bytes": len(data)}
    assert not (record / "gate" / "workspace.json").exists() and not (record / "local-logs").exists()
    described = json.loads((record / "run.json").read_text(encoding="utf-8"))
    assert described["omitted"]["gate/workspace.json"]["sha256"] == sha256((run / "gate/workspace.json").read_bytes()).hexdigest()
    assert described["omitted"]["local-logs"]["files"] == ["01-step.log"]
    assert described["host"] == "synthetic test host" and described["limitations"] == records.LIMITATIONS
    assert runner._host_paths(described, "run.json") == []
    inspected = records.inspect_record(record)
    assert inspected["problems"] == [] and inspected["files"] == 9
    summary = inspected["summary"]
    assert summary["bundles"]["values"] == [0, 65, 92, 65, 0] and summary["bundles"]["replay_numerical_match"] is True
    assert summary["gate"]["host_executable_binding"] == "operator_asserted_not_attested"
    assert summary["native_tests"] == list(records.REQUIRED_NATIVE_TESTS)
    verified = records.verify_records(tmp_path / "lab")
    assert verified["passed"] and verified["verified"] == 1 and verified["records"][0]["date"] == "2026-09-24"


def _flip_last_byte(record):
    path = record / "gate" / "gate.json"
    data = bytearray(path.read_bytes())
    data[-2] ^= 1
    path.write_bytes(bytes(data))


@pytest.mark.parametrize("defect, expected", [
    (_flip_last_byte, "integrity: gate/gate.json differs from its manifest digest"),
    (lambda record: (record / "gate" / "tests.xml").unlink(), "integrity: gate/tests.xml is missing or is a link"),
    (lambda record: (record / "gate" / "notes.txt").write_text("unrecorded"),
     "integrity: gate/notes.txt is not recorded in manifest.json"),
    (lambda record: (record / "gate" / "replay.json.gz").write_bytes(records.compress(b"{}")),
     "integrity: gate/replay.json.gz differs from its manifest digest"),
], ids=["changed-byte", "missing-file", "unrecorded-file", "replaced-bundle"])
def test_a_record_with_a_changed_missing_or_unrecorded_file_is_refused(tmp_path, defect, expected):
    record = retain(tmp_path)
    defect(record)
    inspected = records.inspect_record(record)
    assert any(problem.startswith(expected) for problem in inspected["problems"]), inspected["problems"]
    assert inspected["summary"] is None
    assert not records.verify_records(tmp_path / "lab")["passed"]


@pytest.mark.parametrize("path, change, expected", [
    ("gate/gate.json", lambda gate: gate["scr"].update(revision="0" * 40), "pins: gate/gate.json scr"),
    ("gate/gate.json", lambda gate: gate["sp1"].update(tree="0" * 40), "pins: gate/gate.json sp1"),
    ("build.json", lambda build: build.update(recipe_identity="0" * 64), "pins: build.json recipe_identity"),
    ("build.json", lambda build: build.update(elf_sha256="0" * 64), "pins: build.json elf_sha256"),
    ("build.json", lambda build: build.update(compiler_archive_sha256="0" * 64),
     "pins: build.json compiler_archive_sha256"),
    ("source-checks.json.gz", lambda checks: checks.update(sp1_revision="0" * 40), "pins: source-checks.json sp1"),
], ids=["scr-revision", "sp1-tree", "recipe", "guest", "compiler-archive", "source-checks"])
def test_a_resealed_record_naming_other_pins_is_refused(tmp_path, path, change, expected):
    # Recomputed digests hide nothing the pins decide.
    record = retain(tmp_path)
    _edit(record, path, change)
    problems = records.inspect_record(record)["problems"]
    assert not any(problem.startswith("integrity") for problem in problems)
    assert any(problem.startswith(expected) for problem in problems), problems


def _verified_at(report: dict, revision: str) -> None:
    """Reseal the re-verification report as verified by an SCR checkout at ``revision``."""
    from ciw.proved_heat import _identify
    from ciw.telemetry import digest
    report["verifier_runtimes"]["scr"]["revision"] = revision
    report["verifier_runtime_digest"] = digest(report["verifier_runtimes"])
    report.pop("verification_id")
    _identify(report)


@pytest.mark.parametrize("path, change, expected", [
    ("gate/gate.json", lambda gate: gate.update(status="failed"), "gate: gate/gate.json status is 'failed'"),
    ("gate/gate.json", lambda gate: gate.update(schema="ciw.proved-heat-gate.v0"),
     "gate: gate/gate.json is not ciw.proved-heat-gate.v1"),
    ("gate/gate.json", lambda gate: gate.update(host_executable_binding="attested"),
     "gate: gate/gate.json host_executable_binding"),
    ("run.json", lambda run: run.update(schema="ciw.lab-proved-heat-run.v0"),
     "run_record: run.json is not ciw.lab-proved-heat-run.v1"),
    ("run.json", lambda run: run.update(host="/home/operator/rig"), "run_record: run.json.host holds a host path"),
    ("gate/gate.json", lambda gate: gate["native_artifacts"]["engine"].update(sha256="e" * 64),
     "bundles: gate/original.json ran other engine"),
    ("gate/reverification.json", lambda report: report.update(subject_ref="sha256:" + "0" * 64),
     "bundles: gate/reverification.json subject_ref"),
    ("gate/original.json.gz", lambda bundle: bundle["steps"][0]["result"]["data"]["native"].update(values=[0] * 5),
     "bundles: gate/original.json is refused by CIW's proved-heat validator"),
    ("run.json", lambda run: run.update(procedure="x"), "run_record: run.json procedure is not an object"),
    ("run.json", lambda run: run["procedure"].update(kind="ci_upload"),
     "run_record: run.json procedure kind is not one of local_workflow_replay"),
    ("run.json", lambda run: run.update(observations=["x"]), "run_record: run.json observations is not an object"),
    ("run.json", lambda run: run.update(toolchains=3), "run_record: run.json toolchains is not an object"),
    ("run.json", lambda run: run.update(host_facts="x"), "run_record: run.json host_facts is not an object"),
    ("run.json", lambda run: run.update(sources=[]), "run_record: run.json sources is not an object"),
    ("run.json", lambda run: run.update(compiler_archive="x"),
     "run_record: run.json compiler_archive is not an object"),
    ("run.json", lambda run: run.update(omitted=[]), "run_record: run.json omitted is not an object"),
    ("run.json", lambda run: run.update(notes="x"), "run_record: run.json notes is not a list of sentences"),
    ("run.json", lambda run: run.update(limitations={}),
     "run_record: run.json limitations is not a list of sentences"),
    ("run.json", lambda run: run.update(observations={records.TOOLCHAIN_EXPERIMENT: {"builds": [{"sha256": "x"}]}}),
     f"run_record: run.json observations.{records.TOOLCHAIN_EXPERIMENT} does not list its builds"),
    ("run.json", lambda run: run.update(host="  "), "run_record: run.json declares no host"),
    ("run.json", lambda run: run.update(host=3), "run_record: run.json declares no host"),
    ("run.json", lambda run: run.update(date="2026-9-24"), "run_record: run.json date is not YYYY-MM-DD"),
    ("run.json", lambda run: run.update(date=20260924), "run_record: run.json date is not YYYY-MM-DD"),
    ("run.json", lambda run: run.update(steps=[]), "run_record: run.json lists no workflow steps"),
    ("run.json", lambda run: run.update(steps=["ran"]), "run_record: run.json lists no workflow steps"),
    ("gate/gate.json", lambda gate: gate.update(tests_passed=gate["tests_passed"] + 1),
     "gate: gate/gate.json counts 5 passed tests; gate/tests.xml records 4"),
    ("gate/gate.json", lambda gate: gate["measurements"].pop("replay"),
     "gate: gate/gate.json lacks the original and replay measurements"),
    ("gate/gate.json", lambda gate: gate.pop("measurements"),
     "gate: gate/gate.json lacks the original and replay measurements"),
    ("gate/gate.json", lambda gate: gate["native_artifacts"]["engine"].update(byte_count=0),
     "gate: gate/gate.json does not identify its engine, prover and guest"),
    ("gate/gate.json", lambda gate: gate["native_artifacts"].update(prover="x"),
     "gate: gate/gate.json does not identify its engine, prover and guest"),
    ("gate/gate.json", lambda gate: gate["native_artifacts"].pop("guest"),
     "gate: gate/gate.json does not identify its engine, prover and guest"),
    ("gate/gate.json", lambda gate: gate.update(native_artifacts="x"),
     "gate: gate/gate.json does not identify its engine, prover and guest"),
    # Either source-checks condition alone refuses the record.
    ("source-checks.json.gz", lambda checks: checks.update(tracked_build_source_bytes="changed"),
     "gate: source-checks.json does not record unchanged SP1 build sources"),
    ("source-checks.json.gz", lambda checks: checks.update(runtime_reference="same_checkout"),
     "gate: source-checks.json does not record unchanged SP1 build sources"),
    ("gate/source.json", lambda source: source.update(experiment_id="another-experiment"),
     "bundles: gate/source.json differs from the source bytes the bundles retain"),
    ("gate/reverification.json", lambda report: _verified_at(report, "0" * 40),
     f"pins: gate/reverification.json verifier_runtimes.scr: revision {'0' * 40} is not a revision CIW pins"),
], ids=["failed-gate", "gate-schema", "attested-claim", "run-schema", "host-path", "engine", "reverification",
        "bundle", "procedure", "procedure-kind", "observations", "toolchains", "host-facts", "sources",
        "compiler-archive", "omitted", "notes", "limitations", "toolchain-experiment", "blank-host", "host-not-text",
        "date-format", "date-not-text", "no-steps", "step-not-object", "tests-passed", "measurement-key",
        "no-measurements", "zero-byte-engine", "prover-not-object", "no-guest", "natives-not-object",
        "changed-build-sources", "shared-reference", "source", "verifier-revision"])
def test_a_resealed_record_with_a_wrong_schema_status_or_bundle_is_refused(tmp_path, path, change, expected):
    record = retain(tmp_path)
    _edit(record, path, change)
    inspected = records.inspect_record(record)
    assert any(problem.startswith(expected) for problem in inspected["problems"]), inspected["problems"]
    assert inspected["summary"] is None


def test_a_skipped_native_test_is_refused(tmp_path):
    record = retain(tmp_path)
    path = record / "gate" / "tests.xml"
    path.write_bytes(path.read_bytes().replace(b'time="0.1" />', b'time="0.1"><skipped /></testcase>', 1))
    _reseal(record)
    problems = records.inspect_record(record)["problems"]
    assert f"gate: gate/tests.xml records {records.REQUIRED_NATIVE_TESTS[0]} as skipped" in problems


@pytest.mark.parametrize("path, value, expected", [
    ("run.json", [], ["run_record: run.json is not a JSON object",
                      "run_record: run.json is not ciw.lab-proved-heat-run.v1"]),
    ("gate/gate.json", [], ["gate: gate/gate.json is not a JSON object",
                            "gate: gate/gate.json is not ciw.proved-heat-gate.v1"]),
    ("build.json", None, ["gate: build.json is not a JSON object"]),
    ("source-checks.json.gz", [], ["gate: source-checks.json is not a JSON object"]),
    ("gate/original.json.gz", [], ["bundles: gate/original.json is not a JSON object"]),
    ("gate/replay.json.gz", None, ["bundles: gate/replay.json is not a JSON object"]),
    ("gate/reverification.json", [], ["bundles: gate/reverification.json is not a JSON object"]),
], ids=["run", "gate", "build", "source-checks", "original", "replay", "reverification"])
def test_a_resealed_record_whose_json_file_is_not_an_object_is_refused(tmp_path, path, value, expected):
    # No check could read such a file, so each is refused by name (and build.json null no longer crashes).
    record = retain(tmp_path)
    data = json.dumps(value).encode()
    (record / path).write_bytes(records.compress(data) if path.endswith(".gz") else data)
    _reseal(record)
    inspected = records.inspect_record(record)
    assert inspected["problems"] == expected and inspected["summary"] is None


def _manifest_listing_an_unknown_file(record: Path, manifest: dict) -> dict:
    (record / "gate" / "notes.txt").write_bytes(b"notes")
    manifest["files"]["gate/notes.txt"] = {"sha256": sha256(b"notes").hexdigest(), "bytes": 5}
    return manifest


@pytest.mark.parametrize("change, expected", [
    (lambda record, manifest: [manifest], "manifest.json is not ciw.lab-proved-heat-manifest.v1"),
    (lambda record, manifest: {**manifest, "schema": "ciw.lab-proved-heat-manifest.v0"},
     "manifest.json is not ciw.lab-proved-heat-manifest.v1"),
    (lambda record, manifest: {**manifest, "files": {**manifest["files"], "build.json": "listed"}},
     "manifest.json lists 'build.json', which is not a file of a proved-heat record"),
    (_manifest_listing_an_unknown_file,
     "manifest.json lists 'gate/notes.txt', which is not a file of a proved-heat record"),
], ids=["not-an-object", "schema", "entry-not-an-object", "unknown-file"])
def test_a_malformed_manifest_is_refused_by_name(tmp_path, change, expected):
    record = retain(tmp_path)
    manifest = json.loads((record / "manifest.json").read_text(encoding="utf-8"))
    (record / "manifest.json").write_text(json.dumps(change(record, manifest)), encoding="utf-8")
    inspected = records.inspect_record(record)
    assert inspected["problems"] == [f"integrity: {expected}"] and inspected["summary"] is None


def _replace(directory: Path, path: str, data: bytes) -> None:
    """Replace one retained file and reseal its manifest digest; the content digest stays the gate's bytes."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    (directory / path).write_bytes(data)
    manifest["files"][path].update(sha256=sha256(data).hexdigest(), bytes=len(data))
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("change, expected", [
    (lambda data, content: data + b"trailing bytes", "is not exactly one complete gzip stream"),
    (lambda data, content: data + records.compress(b"{}"), "is not exactly one complete gzip stream"),
    (lambda data, content: data[:-8], "is not exactly one complete gzip stream"),
    (lambda data, content: zlib.compress(content, 9), "is not valid gzip (error)"),
], ids=["trailing-data", "second-member", "no-trailer", "zlib-stream"])
def test_a_retained_gz_file_must_be_exactly_one_gzip_stream(tmp_path, change, expected):
    # Each stream still holds the gate's exact bytes, so only the stream around them refuses the record.
    record = retain(tmp_path)
    path = "gate/replay.json.gz"
    data = (record / path).read_bytes()
    _replace(record, path, change(data, gzip.decompress(data)))
    inspected = records.inspect_record(record)
    assert inspected["problems"] == [f"integrity: {path} {expected}"] and inspected["summary"] is None


@pytest.mark.parametrize("spare", [0, -1], ids=["exactly-the-limit", "one-byte-over"])
def test_retained_content_may_expand_to_exactly_the_size_limit(tmp_path, monkeypatch, spare):
    record = retain(tmp_path)
    manifest = json.loads((record / "manifest.json").read_text(encoding="utf-8"))
    sizes = {path: entry["content"]["bytes"] for path, entry in manifest["files"].items() if path.endswith(".gz")}
    limit = max(sizes.values()) + spare
    monkeypatch.setattr(records, "MAX_CONTENT_BYTES", limit)
    problems = records.inspect_record(record)["problems"]
    assert problems == [f"integrity: {path} expands beyond {limit} bytes"
                        for path, size in sorted(sizes.items()) if size > limit]
    assert problems or spare == 0


def test_a_one_byte_native_artifact_is_the_smallest_the_gate_may_name(tmp_path, monkeypatch):
    # A zero-byte one is refused (above); CIW's bundle validator admits one byte, and so does the record.
    monkeypatch.setitem(globals(), "ENGINE", b"e")
    inspected = records.inspect_record(retain(tmp_path))
    assert inspected["problems"] == []
    assert inspected["summary"]["gate"]["native_artifacts"]["engine"]["byte_count"] == 1


def test_a_retained_runtime_needs_a_source_tree_ciw_records(tmp_path, monkeypatch):
    # CIW's proved-heat pin records its tree; were no tree recorded for that revision, no runtime would be bound to one.
    from ciw import proved_heat
    from ciw.lab import bridge
    record = retain(tmp_path)
    declared = bridge.declared_pins()
    trees = {revision: found for revision, found in declared["trees"].items()
             if revision != proved_heat.PIN["revision"]}
    monkeypatch.setattr(bridge, "declared_pins", lambda: {**declared, "trees": trees})
    assert records.inspect_record(record)["problems"] == [
        f"pins: {where}: CIW records no source tree for its revision"
        for where in ("gate/original.json runtimes.scr", "gate/replay.json runtimes.scr",
                      "gate/reverification.json verifier_runtimes.scr")]


@pytest.mark.parametrize("arrange, match", [
    (lambda run: _write_json(run / "gate" / "gate.json", lambda gate: gate.update(status="failed")),
     "did not pass"),
    (lambda run: (run / "local-run.json").unlink(), "no readable local-run.json"),
    (lambda run: _write_json(run / "local-run.json", lambda local: local.pop("toolchains")), "missing"),
    (lambda run: (run / "gate" / "reverification.json").unlink(), "incomplete"),
    (lambda run: _write_json(run / "gate" / "workspace.json", lambda saved: saved["workbench"]["bundles"].pop()),
     "other bundles"),
    (lambda run: _write_json(run / "gate" / "gate.json", lambda gate: gate["sp1"].update(revision="0" * 40)),
     "fails verification"),
    (lambda run: _write_json(run / "local-run.json", lambda local: local.update(schema="ciw.proved-heat-local-run.v0")),
     "is not ciw.proved-heat-local-run.v1"),
    (lambda run: (run / "local-run.json").write_text("[]"), "is not ciw.proved-heat-local-run.v1"),
    (lambda run: (run / "gate" / "gate.json").write_text("{"), "gate/gate.json is unreadable"),
    # Refused before anything is written, not by the inspection of a written copy.
    (lambda run: _write_json(run / "local-run.json", lambda local: local["host_facts"].update(home="/home/operator")),
     "Refusing the run description: run.json.host_facts.home holds a host path"),
], ids=["failed-gate", "no-run-description", "incomplete-description", "incomplete-output", "workspace", "pins",
        "description-schema", "description-not-an-object", "unreadable-gate", "description-host-path"])
def test_retention_refuses_a_run_it_cannot_retain_and_leaves_nothing(tmp_path, arrange, match):
    run = gate_output(tmp_path / "gate-run")
    arrange(run)
    with pytest.raises(ValueError, match=match):
        records.retain_record(run, tmp_path / "lab", RUN_ID, "synthetic test host")
    assert not (tmp_path / "lab" / "proved-heat" / RUN_ID).exists()


def test_retention_refuses_a_host_path_a_bad_identity_and_a_second_copy(tmp_path):
    run = gate_output(tmp_path / "gate-run")
    with pytest.raises(ValueError, match="host path"):
        records.retain_record(run, tmp_path / "lab", RUN_ID, "rig at /home/operator/rig")
    with pytest.raises(ValueError, match="kebab-case"):
        records.retain_record(run, tmp_path / "lab", "Local Run", "synthetic test host")
    records.retain_record(run, tmp_path / "lab", RUN_ID, "synthetic test host")
    with pytest.raises(ValueError, match="already retained"):
        records.retain_record(run, tmp_path / "lab", RUN_ID, "synthetic test host")


def _write_json(path: Path, change) -> None:
    value = json.loads(path.read_bytes())
    change(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_lab_verify_checks_every_retained_proved_heat_record(tmp_path, capsys):
    from ciw.cli import main
    record = retain(tmp_path)
    result = runner.verify_retained(tmp_path / "lab", tmp_path / "fresh")
    assert result["proved_heat_records"]["verified"] == 1
    assert not any(problem.startswith("proved-heat/") for problem in result["problems"])
    assert main(["lab", "proved-heat", "verify", "--retained", str(tmp_path / "lab")]) == 0
    _flip_last_byte(record)
    result = runner.verify_retained(tmp_path / "lab", tmp_path / "fresh")
    assert f"proved-heat/{RUN_ID}: integrity: gate/gate.json differs from its manifest digest" in result["problems"]
    assert not result["passed"]
    capsys.readouterr()
    assert main(["lab", "proved-heat", "verify", "--retained", str(tmp_path / "lab")]) == 3
    assert json.loads(capsys.readouterr().out)["passed"] is False


def test_a_tool_probe_can_name_a_rustup_toolchain(tmp_path, monkeypatch):
    calls = []

    def answer(command, **kwargs):
        calls.append((command, kwargs["env"].get("RUSTUP_AUTO_INSTALL")))
        return subprocess.CompletedProcess(command, 0 if command[1] == "+1.94.0" else 1, "", "")

    monkeypatch.setattr(runner.shutil, "which", lambda name: "/usr/bin/cargo" if name == "cargo" else None)
    monkeypatch.setattr(runner.subprocess, "run", answer)
    ctx = runner.Context(tmp_path)
    assert ctx.available("tool:cargo") and ctx.available("tool:cargo+1.94.0")
    assert not ctx.available("tool:cargo+1.93.0") and not ctx.available("tool:rustc+1.94.0")
    assert not ctx.available("tool:cargo+--help")
    assert calls == [(["cargo", "+1.94.0", "--version"], "0"), (["cargo", "+1.93.0", "--version"], "0")]
    assert ctx.probes["tool:cargo+1.94.0"] is True and ctx.probes["tool:cargo+1.93.0"] is False


# ---------------------------------------------------------------- T099

def _repository(path: Path) -> Path:
    """A small committed Git repository standing in for the SCR checkout."""
    (path / "crates").mkdir(parents=True)
    (path / "crates" / "Cargo.lock").write_bytes(b"# lock\n")

    def git(*arguments):
        subprocess.run(["git", "-C", str(path), "-c", "user.name=lab", "-c", "user.email=lab@example.invalid",
                        "-c", "core.autocrlf=false", *arguments], check=True, capture_output=True)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "synthetic SCR")
    return path


def _t099(tmp_path, monkeypatch, record=None, pinned=ENGINE):
    """T099 on a synthetic SCR checkout with stand-in builds and engine run.

    ``pinned`` is the engine bytes the CI-pinned toolchain builds, :data:`FAILED_BUILD` when that build fails, or
    None when that toolchain is not installed.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    scr = _repository(tmp_path / "scr")
    monkeypatch.setattr(runner, "_probe_tool", lambda name: name == "cargo" or (
        name == f"cargo+{section.PINNED_TOOLCHAIN}" and pinned is not None))
    monkeypatch.setattr(providers, "compare_with_pins", lambda role, identity, pins: {
        "role": role, "matched": ["ciw.proved_heat.PIN"], "unmatched": [], "tree_refusals": [], "accepted": True,
        "clean": True})
    default = b"default-toolchain engine bytes"
    build = {"builds": [{"returncode": 0, "binary_sha256": sha256(default).hexdigest()}] * 2, "binary": default,
             "cargo_lock_sha256_before": "0" * 64, "cargo_lock_sha256_after": "0" * 64,
             "cargo": "cargo (stand-in)", "rustc": "rustc (stand-in)"}
    monkeypatch.setattr(section, "_locked_build", lambda ctx: build)
    monkeypatch.setattr(providers, "materialize", lambda data, directory: Path(directory) / "execution-cli")
    monkeypatch.setattr(providers, "run_heat_kernel", lambda scr, engine, cases: {
        "cases": [{"values": fixtures.heat_reference(values, steps)} for steps, values in cases]})

    def build_engine(scr, builds=2, toolchain=None):
        assert toolchain == section.PINNED_TOOLCHAIN and builds == 1
        row = ({"returncode": 101, "binary_sha256": None, "byte_count": None, "log_tail": ["error (stand-in)"]}
               if pinned == FAILED_BUILD else
               {"returncode": 0, "binary_sha256": sha256(pinned).hexdigest(), "byte_count": len(pinned),
                "log_tail": []})
        return {"builds": [row], "binary": pinned or None, "command": f"cargo +{toolchain} build (stand-in)",
                "cargo": f"cargo {toolchain} (stand-in)", "rustc": f"rustc {toolchain} (stand-in)", "toolchain": toolchain}
    monkeypatch.setattr(providers, "build_engine", build_engine)
    bound = {"scr": str(scr), **({section.PROVED_HEAT_RECORD: str(record)} if record else {})}
    report = runner.run_task(QUEUE["T099"], registry._REGISTRY["T099"], runner.Context(tmp_path / "run", bound), {})
    prose = json.dumps({name: report[name] for name in runner.PROSE_FIELDS})
    assert str(tmp_path) not in prose
    return report


def _claim(report, text):
    return next(record for record in report["findings"] if record["claim"] == text)


def _artifact(tmp_path, name):
    return json.loads((tmp_path / "run" / "artifacts" / "T099" / name).read_text(encoding="utf-8"))


@pytest.mark.lab_task("T099")
def test_t099_labels_a_valid_proved_heat_record_provider_backed(tmp_path, monkeypatch):
    # A retained record at CIW's pins is provider_backed: the gate's outcome as recorded, not T099's verification.
    from ciw import proved_heat
    report = _t099(tmp_path, monkeypatch, record=retain(tmp_path))
    assert report["state"] == "completed" and report["evidence_status"]["primary"] == "provider_backed"
    record = _claim(report, section.RECORD_CLAIM)
    assert record["evidence_status"] == "numerically_verified" and record["value"]["problems"] == 0
    for claim in (section.GUEST_CLAIM, section.PROOF_CLAIM, section.REVERIFY_CLAIM):
        found = _claim(report, claim)
        assert found["evidence_status"] == "provider_backed" and SHOWN in finding_row(found), claim
        assert found["basis"]["provider"]["source_tree"] == proved_heat.PIN["source_tree"]
        assert found["basis"]["notes"]["host_executable_binding"] == "operator_asserted_not_attested"
    assert _claim(report, section.PROOF_CLAIM)["value"]["heat_values"] == [0, 65, 92, 65, 0]
    attested = _claim(report, section.ATTESTED_CLAIM)
    assert attested["evidence_status"] == "not_established" and attested["value"] == "operator_asserted_not_attested"
    pinned = _claim(report, section.PINNED_CLAIM)
    assert pinned["evidence_status"] == "numerically_verified" and pinned["value"]["equal_to_gate_engine"] is True
    identity = report["provider_runtime_identity"]
    assert identity["proved_heat_record"]["state"] == "valid"
    assert identity["proved_heat_record"]["sp1"] == records.pins()["sp1"]
    assert identity["requirement_probes"][f"provider:{section.PROVED_HEAT_RECORD}"] is True
    assert identity["requirement_probes"][f"tool:cargo+{section.PINNED_TOOLCHAIN}"] is True
    # Measurements are reported in the result and artifacts, never as compared finding values.
    assert "host-specific, not compared" in report["numerical_result"]
    assert all("prove_and_verify_seconds" not in json.dumps(f["value"]) for f in report["findings"])
    kept = _artifact(tmp_path, "proved-heat-record.json")
    assert kept["state"] == "valid" and kept["summary"]["measurements"]["original"]["memory"] == MEMORY
    toolchains = _artifact(tmp_path, "pinned-toolchain-build.json")
    assert toolchains["default_differs_from_gate_engine"] is True and toolchains["default_differs_from_pinned"] is True


@pytest.mark.lab_task("T099")
@pytest.mark.parametrize("defect, category", [("changed-byte", "integrity"), ("other-pin", "pins"),
                                              ("malformed-run", "run_record")])
def test_t099_refuses_a_tampered_proved_heat_record_by_name(tmp_path, monkeypatch, defect, category):
    record = retain(tmp_path)
    if defect == "changed-byte":
        _flip_last_byte(record)
    elif defect == "other-pin":
        _edit(record, "gate/gate.json", lambda gate: gate["scr"].update(tree="0" * 40))
    else:
        # Resealed, so only the run record's shape refuses it; T099 must refuse it by name rather than fail on it.
        _edit(record, "run.json", lambda run: run.update(procedure="x", observations=["x"]))
    report = _t099(tmp_path, monkeypatch, record=record)
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "not_established"
    refused = _claim(report, section.RECORD_CLAIM)
    assert refused["evidence_status"] == "not_established" and "expected_not_established" not in refused
    failing = {check["reference"] for check in refused["basis"]["checks"] if not check["passed"]}
    assert failing == {f"record problems ({category})"}
    for claim in (section.GUEST_CLAIM, section.PROOF_CLAIM, section.REVERIFY_CLAIM):
        found = _claim(report, claim)
        assert found["evidence_status"] == "not_established" and found["expected_not_established"] is True
        assert f"the bound proved-heat record {RUN_ID} is refused" in found["value"]
    assert _claim(report, section.PINNED_CLAIM)["value"].startswith("built, not compared")
    assert _artifact(tmp_path, "proved-heat-record.json")["problems"]


@pytest.mark.lab_task("T099")
@pytest.mark.parametrize("pinned", [None, ENGINE], ids=["no-pinned-toolchain", "pinned-toolchain"])
def test_t099_without_a_record_or_the_pinned_toolchain(tmp_path, monkeypatch, pinned):
    report = _t099(tmp_path, monkeypatch, pinned=pinned)
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "numerically_verified"
    for claim in (section.RECORD_CLAIM, section.GUEST_CLAIM, section.PROOF_CLAIM, section.REVERIFY_CLAIM):
        found = _claim(report, claim)
        assert found["evidence_status"] == "not_established" and found["expected_not_established"] is True
        assert "no proved-heat record is bound" in found["value"]
    assert all(record["claim"] != section.ATTESTED_CLAIM for record in report["findings"])
    rebuilt = _claim(report, section.PINNED_CLAIM)
    assert rebuilt["evidence_status"] == "not_established" and rebuilt["expected_not_established"] is True
    assert rebuilt["value"].startswith("not built" if pinned is None else "built, not compared")
    probes = report["provider_runtime_identity"]["requirement_probes"]
    assert probes[f"provider:{section.PROVED_HEAT_RECORD}"] is False
    assert probes[f"tool:cargo+{section.PINNED_TOOLCHAIN}"] is (pinned is not None)
    assert "no proved-heat record bound" in report["numerical_result"]


@pytest.mark.lab_task("T099")
def test_t099_refutes_a_pinned_rebuild_that_differs_from_the_gate_engine(tmp_path, monkeypatch):
    report = _t099(tmp_path, monkeypatch, record=retain(tmp_path), pinned=b"another engine")
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "not_established"
    refuted = _claim(report, section.PINNED_CLAIM)
    assert refuted["evidence_status"] == "not_established" and "expected_not_established" not in refuted
    assert refuted["value"]["equal_to_gate_engine"] is False
    assert _claim(report, section.GUEST_CLAIM)["evidence_status"] == "provider_backed"
    assert "different from the gate's" in report["numerical_result"]
    # The prose reports the refutation; it never says the rebuild reproduced the gate's engine.
    assert _claim(report, section.ATTESTED_CLAIM)["basis"]["notes"]["engine"] == section._ENGINE_REBUILT["differs"]
    assert not any("reproducible from the pinned source where" in text for text in report["unresolved_assumptions"])
    assert any(section._ENGINE_ASSUMPTION["differs"] in text for text in report["unresolved_assumptions"])


def _experiment(*builds) -> dict:
    """run.json observations with an execution-cli toolchain observation of ``(toolchain, source, engine bytes)``."""
    return {"observations": {records.TOOLCHAIN_EXPERIMENT: {
        "kind": "operator_observation", "builds": [{"toolchain": toolchain, "source": source,
                                                    "sha256": sha256(data).hexdigest()}
                                                   for toolchain, source, data in builds]}}}


SEPARATED = _experiment(("rustc pinned", "<scr>", ENGINE), ("rustc pinned", "<scr>, built again", ENGINE),
                        ("rustc newer", "<scr>", b"newer engine"), ("rustc newer", "a second clone", b"newer engine"))
UNSEPARATED = _experiment(("rustc pinned", "<scr>", ENGINE), ("rustc pinned", "a second clone", b"path-dependent"))


@pytest.mark.lab_task("T099")
@pytest.mark.parametrize("outcome, pinned, observation", [
    ("equal", ENGINE, SEPARATED), ("differs", b"another engine", SEPARATED), ("failed", FAILED_BUILD, SEPARATED),
    ("not_installed", None, SEPARATED), ("equal", ENGINE, UNSEPARATED)],
    ids=["equal", "differs", "failed", "not-installed", "unseparated-observation"])
def test_t099_words_the_pinned_rebuild_and_the_toolchain_observation_by_what_happened(tmp_path, monkeypatch, outcome,
                                                                                     pinned, observation):
    # With a valid record, every sentence about the pinned-toolchain rebuild follows its outcome, and the record's
    # toolchain observation is counted from the record, never asserted beyond it.
    report = _t099(tmp_path, monkeypatch, record=retain(tmp_path, local_changes=observation), pinned=pinned)
    assert report["state"] == ("completed" if outcome == "equal" else "partial")
    rebuilt = _claim(report, section.PINNED_CLAIM)
    if outcome == "not_installed":
        assert rebuilt["expected_not_established"] is True and rebuilt["value"].startswith("not built")
    else:
        assert rebuilt["value"]["equal_to_gate_engine"] is (outcome == "equal")
        assert rebuilt["evidence_status"] == ("numerically_verified" if outcome == "equal" else "not_established")
    attested = _claim(report, section.ATTESTED_CLAIM)
    assert attested["evidence_status"] == "not_established"
    assert attested["basis"]["notes"]["engine"] == section._ENGINE_REBUILT[outcome]
    assumptions = report["unresolved_assumptions"]
    assert any(section._ENGINE_ASSUMPTION[outcome] in text for text in assumptions), assumptions
    toolchains = assumptions[0]
    assert section._REBUILT_HERE[outcome] in toolchains
    assert section._pinned_result(outcome) in report["numerical_result"]
    if observation is SEPARATED:
        assert toolchains.startswith("The engine digest depends on the Rust toolchain and not on the checkout path")
        assert ("2 builds with rustc pinned gave the gate's engine digest; 2 builds with rustc newer gave one other "
                "digest") in toolchains
    else:
        assert toolchains.startswith("The bound record's run.json toolchain observation does not separate")
        assert "2 builds with rustc pinned gave 2 different digests" in toolchains
    assert "checkout paths" not in toolchains
    kept = _artifact(tmp_path, "proved-heat-record.json")
    assert kept["summary"]["observations"] == observation["observations"]


def _repository_record() -> str | None:
    """The newest record under the repository's lab/proved-heat/, when the tests sit in the repository."""
    root = ROOT / "lab" / "proved-heat"
    found = sorted((json.loads((path / "run.json").read_text(encoding="utf-8"))["date"], path.name, path)
                   for path in root.iterdir() if (path / "run.json").is_file()) if root.is_dir() else []
    return str(found[-1][2]) if found else None


@pytest.mark.lab_task("T099")
def test_t099_rebuilds_the_gate_engine_with_the_pinned_toolchain(tmp_path):
    scr, record = os.environ.get("CIW_LAB_SCR_REPO"), os.environ.get("CIW_LAB_PROVED_HEAT_RECORD") or _repository_record()
    if not scr or not record:
        pytest.skip("set CIW_LAB_SCR_REPO and CIW_LAB_PROVED_HEAT_RECORD (a retained proved-heat record)")
    if not runner.Context(tmp_path).available(f"tool:cargo+{section.PINNED_TOOLCHAIN}"):
        pytest.skip(f"install the rustup toolchain {section.PINNED_TOOLCHAIN}")
    bound = {"scr": scr, section.PROVED_HEAT_RECORD: record}
    report = runner.run_task(QUEUE["T099"], registry._REGISTRY["T099"], runner.Context(tmp_path, bound), {})
    rebuilt = _claim(report, section.PINNED_CLAIM)
    assert rebuilt["evidence_status"] == "numerically_verified" and rebuilt["value"]["equal_to_gate_engine"] is True
    assert _claim(report, section.GUEST_CLAIM)["evidence_status"] == "provider_backed"
    assert report["state"] == "completed" and report["evidence_status"]["primary"] == "provider_backed"


def test_t099_registers_the_record_tests_in_this_file():
    nodes = [node for node in registry._REGISTRY["T099"].regression_tests if node.startswith(section.RECORD_TESTS)]
    assert len(nodes) == 6
    for node in nodes:
        assert node.split("::")[1] in globals(), node


# ---------------------------------------------------------------- repository files (skipped in the clean room)

def _script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    if not path.is_file():
        pytest.skip("scripts/ is not beside the tests")
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(f"lab_script_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def test_the_repository_proved_heat_records_verify():
    if not (ROOT / "lab" / "proved-heat").is_dir():
        pytest.skip("lab/ is not beside the tests")
    result = records.verify_records(ROOT / "lab")
    assert result["passed"], result["problems"]
    for directory in records.record_directories(ROOT / "lab"):
        for name in ("run.json", "manifest.json"):
            assert runner._host_paths(json.loads((directory / name).read_text(encoding="utf-8")), name) == []


def test_required_native_tests_mirror_the_gate_script():
    gate = _script("check_proved_heat")
    assert set(records.REQUIRED_NATIVE_TESTS) == gate.REQUIRED_NATIVE_TESTS
    assert set(section.PROOF_TESTS + section.REVERIFY_TESTS) == gate.REQUIRED_NATIVE_TESTS
    assert records.pins()["scr"] == {"revision": gate.SCR_REVISION, "tree": gate.SCR_TREE}
    assert records.pins()["sp1"] == {"revision": gate.SP1_REVISION, "tree": gate.SP1_TREE}


def test_the_native_toolchain_mirrors_both_workflows():
    workflows = ROOT / ".github" / "workflows"
    if not workflows.is_dir():
        pytest.skip(".github/workflows is not beside the tests")
    install = f"rustup toolchain install {providers.NATIVE_TOOLCHAIN} --profile minimal"
    proved = (workflows / "proved-heat.yml").read_text(encoding="utf-8")
    assert install in proved and install in (workflows / "lab.yml").read_text(encoding="utf-8")
    assert (f"cargo +{providers.NATIVE_TOOLCHAIN} build --release --locked --offline --manifest-path "
            '"$stack/scr/crates/Cargo.toml"') in proved


def test_lab_scripts_bind_and_preserve_the_proved_heat_record(tmp_path):
    check, reproduce, refresh = _script("check_lab"), _script("reproduce_lab"), _script("refresh_lab")
    root = tmp_path / "proved-heat"
    assert check.proved_heat_record(root) is None
    for run_id, date in (("zz-older", "2026-09-01"), ("aa-newer", "2026-10-01")):
        (root / run_id).mkdir(parents=True)
        (root / run_id / "run.json").write_text(json.dumps({"date": date}), encoding="utf-8")
    (root / "README.md").write_text("not a record", encoding="utf-8")
    assert check.proved_heat_record(root) == root / "aa-newer"
    assert reproduce.TEST_VARIABLES[section.PROVED_HEAT_RECORD] == "CIW_LAB_PROVED_HEAT_RECORD"
    assert "CIW_LAB_PROVED_HEAT_RECORD" in reproduce.INHERITED_EXCLUDED
    environment = reproduce.clean_room_environment(tmp_path, [(section.PROVED_HEAT_RECORD, str(root / "aa-newer"))])
    assert environment["CIW_LAB_PROVED_HEAT_RECORD"] == str(root / "aa-newer")
    assert "proved-heat" in refresh.PRESERVED and "proved-heat" not in refresh.RETAINED
    if (ROOT / "lab" / "proved-heat").is_dir():
        assert check.proved_heat_record() is not None


def test_check_lab_binds_the_latest_record_for_the_clean_room(tmp_path, monkeypatch):
    # The clean room has no copy of lab/, so the gate binds the record from the checkout.
    check = _script("check_lab")
    commands = []
    monkeypatch.setattr(check, "call", lambda command, **kwargs: commands.append([str(part) for part in command]))
    monkeypatch.setattr(check, "validate_checkout", lambda path, revision: Path(path))
    monkeypatch.setattr(check, "proved_heat_record", lambda: tmp_path / "lab" / "proved-heat" / RUN_ID)
    monkeypatch.setattr(sys, "argv", ["check_lab.py", "--no-compare", "--stack-root", str(tmp_path / "stack"),
                                      "--output-dir", str(tmp_path / "out")])
    assert check.main() == 0
    command, = commands
    assert f"proved-heat-record={tmp_path / 'lab' / 'proved-heat' / RUN_ID}" in command


def test_the_clean_room_verifies_retained_proved_heat_records_without_a_comparison(tmp_path, monkeypatch):
    from types import SimpleNamespace
    reproduce = _script("reproduce_lab")
    commands = []

    def run(command, **kwargs):
        command = [str(part) for part in command]
        commands.append(command)
        if command[1:4] == ["-m", "pip", "wheel"]:
            dist = Path(command[command.index("--wheel-dir") + 1])
            dist.mkdir(parents=True)
            (dist / "ciw-0-py3-none-any.whl").write_bytes(b"wheel")

    retained = tmp_path / "retained"
    (retained / "proved-heat" / RUN_ID).mkdir(parents=True)
    (retained / "proved-heat" / "README.md").write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(reproduce, "run", run)
    monkeypatch.setattr(reproduce, "venv", SimpleNamespace(EnvBuilder=lambda **kwargs: SimpleNamespace(
        create=lambda path: None)))
    monkeypatch.setattr(reproduce, "subprocess", SimpleNamespace(
        run=lambda *args, **kwargs: SimpleNamespace(stdout=str(tmp_path / "site" / "ciw" / "__init__.py"))))
    monkeypatch.setattr(sys, "argv", ["reproduce_lab.py", "--no-compare", "--output-dir", "out", "--retained",
                                      str(retained), "--temporary-root", str(tmp_path)])
    assert reproduce.main() == 0
    assert ["-m", "ciw", "lab", "proved-heat", "verify", "--retained", str(retained.resolve())] in [
        command[1:] for command in commands]
    assert not any(command[1:5] == ["-m", "ciw", "lab", "hardware"] for command in commands)
    record = json.loads((tmp_path / "out" / "gate.json").read_text(encoding="utf-8"))
    assert record["proved_heat_records_verified_for_integrity"] == [RUN_ID]


def test_refresh_keeps_proved_heat_records_and_requires_their_binding(tmp_path, monkeypatch):
    refresh = _script("refresh_lab")
    monkeypatch.setattr(refresh, "ROOT", tmp_path)
    kept = tmp_path / "lab" / "proved-heat" / RUN_ID / "run.json"
    kept.parent.mkdir(parents=True)
    kept.write_text('{"schema": "ciw.lab-proved-heat-run.v1"}')
    run = tmp_path / "run"
    (run / "reports").mkdir(parents=True)
    (run / "artifacts").mkdir()
    (run / "reports" / "T001.json").write_text(json.dumps({"task_id": "T001"}))
    # A run of check_lab.py also ran T077's telemetry session on the bound telemetry stack.
    (run / "reports" / "T077.json").write_text(json.dumps({"task_id": "T077", "provider_runtime_identity": {
        "telemetry-stack": {"gsie": {"state": "ready"}}, "executed_runtimes": {"gsie": {}}}}))
    for name in ("queue-state.json", "REPORTS.md", "index.html"):
        (run / name).write_text("")
    gate = {"schema": "ciw.lab-clean-room-gate.v1", "python": "3.12.3",
            "providers": [f"{role}=/p" for role in refresh.REQUIRED_PROVIDERS]}
    (run / "gate.json").write_text(json.dumps(gate))
    monkeypatch.setattr(sys, "argv", ["refresh_lab.py", "--from-run", str(run)])
    with pytest.raises(SystemExit, match="proved-heat-record"):
        refresh.main()
    (run / "gate.json").write_text(json.dumps({**gate, "providers": [*gate["providers"], "proved-heat-record=/r"]}))
    # T099 without the CI-pinned toolchain would be retained partial and regress in CI, whose lab gate installs it.
    pinned = f"tool:cargo+{section.PINNED_TOOLCHAIN}"
    for probes in (None, {"tool:cargo": True}, {"tool:cargo": True, pinned: False}):
        if probes is not None:
            (run / "reports" / "T099.json").write_text(json.dumps(
                {"task_id": "T099", "provider_runtime_identity": {"requirement_probes": probes}}))
        with pytest.raises(SystemExit, match="CI-pinned rustup toolchain"):
            refresh.main()
    (run / "reports" / "T099.json").write_text(json.dumps(
        {"task_id": "T099", "provider_runtime_identity": {"requirement_probes": {"tool:cargo": True, pinned: True}}}))
    assert refresh.main() == 0
    assert kept.read_text() == '{"schema": "ciw.lab-proved-heat-run.v1"}'


def test_the_local_driver_knows_every_workflow_step_and_refuses_without_prerequisites(tmp_path):
    yaml = pytest.importorskip("yaml")
    driver = _script("run_proved_heat_locally")
    workflow_path = ROOT / ".github" / "workflows" / "proved-heat.yml"
    if not workflow_path.is_file():
        pytest.skip(".github/workflows is not beside the tests")
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps, skipped = driver.plan(workflow)
    assert [driver._name(step) for step in steps] == list(driver.GATE_STEPS)
    assert len(steps) + len(skipped) == len(next(iter(workflow["jobs"].values()))["steps"])
    pins = driver.workflow_pins(workflow)
    assert pins["toolchain"] == providers.NATIVE_TOOLCHAIN
    assert pins["archive_sha256"] == records.pins()["compiler_archive_sha256"]
    changed = deepcopy(workflow)
    next(iter(changed["jobs"].values()))["steps"].append({"name": "Upload somewhere", "run": "true"})
    with pytest.raises(SystemExit, match="does not know"):
        driver.plan(changed)
    # A step setting, job default or workflow-wide env the driver would not replay is refused, not dropped.
    for change in (lambda job: next(step for step in job["steps"] if step.get("name") == driver.GATE_STEPS[2])
                   .update(env={"RUSTFLAGS": "-C target-cpu=native"}),
                   lambda job: job.update(defaults={"run": {"shell": "sh"}})):
        changed = deepcopy(workflow)
        change(next(iter(changed["jobs"].values())))
        with pytest.raises(SystemExit, match="does not replay"):
            driver.plan(changed)
    with pytest.raises(SystemExit, match="does not replay"):
        driver.plan({**workflow, "env": {"CARGO_PROFILE_RELEASE_LTO": "fat"}})
    job_env = next(iter(workflow["jobs"].values()))["env"]
    caller = {"PATH": "/usr/bin", "RUSTFLAGS": "-C target-cpu=native", "CARGO_PROFILE_RELEASE_LTO": "fat",
              "RUSTUP_TOOLCHAIN": "stable", "CARGO_BUILD_JOBS": "4", "CARGO_TARGET_DIR": "target",
              "CARGO_INCREMENTAL": "1", "CARGO_HOME": "cargo-home"}
    refused = driver.build_variable_problems(caller, job_env)
    assert [problem.split()[0] for problem in refused] == ["CARGO_PROFILE_RELEASE_LTO", "RUSTFLAGS",
                                                           "RUSTUP_TOOLCHAIN"]
    assert driver.build_variable_problems({"PATH": "/usr/bin"}, job_env) == []
    arguments = driver.argparse.Namespace(scr=tmp_path / "no-scr", sp1=tmp_path / "no-sp1",
                                          compiler_archive=tmp_path / "no-archive", python=Path(sys.executable))
    problems = driver.prerequisite_problems(arguments, pins)
    assert "--compiler-archive is not a file" in problems
    assert any(problem.startswith("the SCR checkout is not clean") for problem in problems)
    assert any(problem.startswith("the SP1 checkout is not clean") for problem in problems)
