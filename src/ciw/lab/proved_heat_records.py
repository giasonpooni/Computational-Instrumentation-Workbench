"""Retained runs of the SP1 proved-heat gate: record format, retention and integrity verification.

Scope: one passing run of the proved-heat gate (the steps of
``.github/workflows/proved-heat.yml``, replayed on an operator's host by
``scripts/run_proved_heat_locally.py``) is retained under
``<retained>/proved-heat/<run-id>/`` with ``run.json``
(``ciw.lab-proved-heat-run.v1``: how, where and with which toolchains it was
produced, as role names and versions, never host paths), the gate outputs
needed to re-check its claims (the proof-bearing bundles gzip-compressed, so a
retained proof can be re-verified later with ``ciw proof verify`` after
decompressing it, ``gunzip -k gate/original.json.gz``) and
``manifest.json`` (``ciw.lab-proved-heat-manifest.v1``: the SHA-256 and size of
every retained file and of the gate's own bytes inside each compressed one).
:func:`inspect_record` recomputes every digest; compares the gate's SCR and SP1
revisions and trees, guest recipe, guest ELF and compiler archive with the pins
CIW declares; checks the gate's test record; and validates the retained bundles,
their replay receipt and the re-verification report offline with CIW's own
proved-heat validator. T099 reads a record through the ``proved-heat-record``
provider role; ``ciw lab verify``, ``ciw lab proved-heat verify`` and
``scripts/check_lab.py`` check every retained record for integrity.

Non-claims: nothing here runs a prover or a verifier; a retained record is the
gate's outcome as recorded on its host. Digests are unkeyed: they detect an
accidental edit, not who produced the record, and a fabricated record that
copies CIW's public pins and recomputes every digest passes. The gate binds its
engine and prover as operator-asserted executables, not attested builds, and
its times, memory and proof sizes are measurements of the recording host. Gate
outputs are retained as the gate wrote them, so they name that host's temporary
and checkout locations as identity metadata: ``build.json``'s guest ELF path,
and the SCR runtime identity (checkout root and interpreter) inside the bundles
and ``gate/reverification.json``, which their bundle digests and verification
id seal, so they cannot be redacted.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
import zlib

from .runner import _declared_host, _host_paths, check_name, dumps

DIRECTORY = "proved-heat"
RUN_SCHEMA = "ciw.lab-proved-heat-run.v1"
MANIFEST_SCHEMA = "ciw.lab-proved-heat-manifest.v1"
# Written by scripts/run_proved_heat_locally.py beside the gate outputs.
LOCAL_RUN = "local-run.json"
LOCAL_RUN_SCHEMA = "ciw.proved-heat-local-run.v1"
GATE_SCHEMA = "ciw.proved-heat-gate.v1"
GATE_EXECUTION = "real_sp1_proof_fresh_replay_and_verification"
HOST_BINDING = "operator_asserted_not_attested"
RUN_FILE, MANIFEST_FILE = "run.json", "manifest.json"
# Gate outputs retained, from their path in the gate's output directory to their retained path; the large ones are
# gzip-compressed. The two bundles hold the proofs, so a retained proof can be re-verified later (decompressed).
RETAINED = {"build.json": "build.json", "source-checks.json": "source-checks.json.gz",
            "gate/gate.json": "gate/gate.json", "gate/source.json": "gate/source.json",
            "gate/reverification.json": "gate/reverification.json", "gate/tests.xml": "gate/tests.xml",
            "gate/original.json": "gate/original.json.gz", "gate/replay.json": "gate/replay.json.gz"}
# Retained when the run wrote it (the workflow's resource step).
OPTIONAL = {"resources.json": "resources.json"}
# Gate outputs that are not retained, with the reason recorded in run.json.
OMITTED = {
    "gate/workspace.json": ("its two proved-heat bundles are gate/original.json and gate/replay.json (checked equal "
                            "when the run was retained); the rest is the session's synthetic demonstration run"),
    "local-logs": ("per-step console output of the local driver (compiler and pip output naming host paths); each "
                   "step's outcome and duration are in run.json"),
}
# The native tests scripts/check_proved_heat.py requires (tests/test_lab_proved_heat_records.py keeps them in step).
REQUIRED_NATIVE_TESTS = ("test_native_proved_heat_shared_session", "test_native_tampered_proof_fails_fresh_verification",
                         "test_native_retained_proof_can_be_reverified_without_reexecution")
# A proved-heat bundle is at most 24 MiB (ciw.proved_heat.MAX_BYTES); no retained file expands beyond this.
MAX_CONTENT_BYTES = 32 * 1024 * 1024
LOCAL_RUN_KEYS = {"schema", "started_utc", "record_origin", "procedure", "steps", "host_facts", "toolchains",
                  "sources", "compiler_archive"}
LOCAL_RUN_OPTIONAL = {"observations", "notes"}
RUN_KEYS = {"schema", "run_id", "host", "date", "started_utc", "record_origin", "procedure", "steps", "host_facts",
            "toolchains", "sources", "compiler_archive", "observations", "notes", "omitted", "limitations", "note"}
RECORD_ORIGINS = ("written_by_driver", "written_after_the_run")
# How a retained run was produced (procedure.kind); T099 words its assumptions by it.
PROCEDURE_KINDS = ("local_workflow_replay",)
# run.json fields that must be objects, and those that must be lists of sentences.
RUN_OBJECTS = ("procedure", "host_facts", "toolchains", "sources", "compiler_archive", "observations", "omitted")
RUN_SENTENCES = ("notes", "limitations")
# The operator's execution-cli toolchain observation T099 reads from run.json observations, when present.
TOOLCHAIN_EXPERIMENT = "execution_cli_toolchain_experiment"
LIMITATIONS = [
    "The gate binds its engine and prover as operator_asserted_not_attested executables: the record shows which "
    "bytes ran, not that they were built from the pinned sources.",
    "Times, memory and proof sizes in gate/gate.json and the bundles are measurements of the recording host.",
    "Gate outputs are retained as the gate wrote them and name the recording host's temporary and checkout "
    "locations as identity metadata; run.json and manifest.json hold no host path.",
    "Digests are unkeyed: they detect accidental edits, not who produced the record. Nothing here re-runs the "
    "prover or the verifier.",
]
NOTE = ("Retained proved-heat gate records are verified for integrity only: every retained file matches the manifest "
        "and no unrecorded file is present, the gate passed with its required native tests, its SCR and SP1 "
        "revisions and trees, guest recipe, guest ELF and compiler archive are the pins CIW declares, and the "
        "retained bundles, replay receipt and re-verification report pass CIW's offline proved-heat validation. "
        "No proof is re-verified and nothing is re-executed; the digests are unkeyed.")
CATEGORIES = ("integrity", "run_record", "gate", "pins", "bundles")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def pins() -> dict:
    """The identities a proved-heat gate record must name, from CIW's own declarations."""
    from .. import proved_heat
    from .exchange_provenance_bundles_providers import SP1_REQUIREMENTS
    return {"scr": {"revision": proved_heat.PIN["revision"], "tree": proved_heat.PIN["source_tree"]},
            "sp1": {"revision": SP1_REQUIREMENTS["sp1_checkout"]["revision"],
                    "tree": SP1_REQUIREMENTS["sp1_checkout"]["tree"]},
            "recipe_identity": SP1_REQUIREMENTS["guest_recipe"]["identity"],
            "guest_sha256": proved_heat.GUEST_SHA256, "requirements_guest_sha256": SP1_REQUIREMENTS["guest_sha256"],
            "compiler_archive_sha256": SP1_REQUIREMENTS["succinct_compiler_archive"]["sha256"]}


def compress(data: bytes) -> bytes:
    """Deterministic gzip bytes (no name, time zero)."""
    return gzip.compress(data, compresslevel=9, mtime=0)


def _decompress(data: bytes) -> bytes:
    """gzip content, refused beyond :data:`MAX_CONTENT_BYTES` or when anything follows the stream."""
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    content = inflater.decompress(data, MAX_CONTENT_BYTES + 1)
    if len(content) > MAX_CONTENT_BYTES or inflater.unconsumed_tail:
        raise ValueError(f"expands beyond {MAX_CONTENT_BYTES} bytes")
    if not inflater.eof or inflater.unused_data:
        raise ValueError("is not exactly one complete gzip stream")
    return content


def _entry(data: bytes) -> dict:
    return {"sha256": sha256(data).hexdigest(), "bytes": len(data)}


def _retained_paths() -> dict:
    """Retained path -> the gate's path, for every file a record may hold besides run.json and manifest.json."""
    return {retained: gate for gate, retained in {**RETAINED, **OPTIONAL}.items()}


def _files(directory: Path, problem) -> tuple[dict, bytes | None]:
    """The record's files by their gate path (decompressed), checked against manifest.json; and the manifest bytes."""
    try:
        raw = (directory / MANIFEST_FILE).read_bytes()
        manifest = json.loads(raw)
    except (OSError, ValueError) as exc:
        problem("integrity", f"{MANIFEST_FILE} unreadable: {type(exc).__name__}")
        return {}, None
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        problem("integrity", f"{MANIFEST_FILE} is not {MANIFEST_SCHEMA}")
        return {}, raw
    if manifest.get("run_id") != directory.name:
        problem("integrity", f"{MANIFEST_FILE} names run {manifest.get('run_id')!r}, retained as {directory.name!r}")
    listed = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    known = {RUN_FILE: RUN_FILE, **_retained_paths()}
    contents = {}
    for path, entry in sorted(listed.items()):
        if path not in known or not isinstance(entry, dict):
            problem("integrity", f"{MANIFEST_FILE} lists {path!r}, which is not a file of a proved-heat record")
            continue
        location = directory / path
        if location.is_symlink() or not location.is_file():
            problem("integrity", f"{path} is missing or is a link, not retained bytes")
            continue
        data = location.read_bytes()
        if {"sha256": entry.get("sha256"), "bytes": entry.get("bytes")} != _entry(data):
            problem("integrity", f"{path} differs from its manifest digest")
            continue
        if path.endswith(".gz"):
            try:
                data = _decompress(data)
            except (OSError, ValueError, EOFError, zlib.error) as exc:
                problem("integrity", f"{path} {exc}" if isinstance(exc, ValueError) else
                        f"{path} is not valid gzip ({type(exc).__name__})")
                continue
            if entry.get("content") != _entry(data):
                problem("integrity", f"the content of {path} differs from its manifest digest")
                continue
        contents[known[path]] = data
    required = {RUN_FILE, *RETAINED.values()}
    for path in sorted(required - set(listed)):
        problem("integrity", f"{path} is missing from {MANIFEST_FILE}")
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            problem("integrity", f"{relative} is a link")
        elif path.is_file() and relative != MANIFEST_FILE and relative not in listed:
            problem("integrity", f"{relative} is not recorded in {MANIFEST_FILE}")
        elif path.is_dir() and relative != "gate":
            problem("integrity", f"unexpected directory {relative}")
    return contents, raw


def _parsed(contents: dict, name: str, problem, category: str):
    """A retained JSON file's object, or None when it is absent, not JSON or not an object (a problem then)."""
    if name not in contents:
        return None
    try:
        value = json.loads(contents[name])
    except ValueError as exc:
        problem(category, f"{name} is not JSON ({exc})")
        return None
    if not isinstance(value, dict):
        problem(category, f"{name} is not a JSON object")
        return None
    return value


def _tests(raw: bytes | None, problem) -> list:
    """Test names of the gate's JUnit record; a skipped, failed or erroring case is a problem."""
    if raw is None:
        return []
    try:
        cases = list(ET.fromstring(raw).iter("testcase"))
    except ET.ParseError as exc:
        problem("gate", f"gate/tests.xml is not XML ({exc})")
        return []
    for case in cases:
        outcome = next((tag for tag in ("skipped", "failure", "error") if case.find(tag) is not None), None)
        if outcome:
            problem("gate", f"gate/tests.xml records {case.get('name')} as {outcome}")
    names = [str(case.get("name")) for case in cases]
    for name in REQUIRED_NATIVE_TESTS:
        if name not in names:
            problem("gate", f"gate/tests.xml does not record the required native test {name}")
    return names


def _toolchain_builds(experiment) -> list | None:
    """The builds of an execution-cli toolchain observation, or None unless each names its toolchain, source and
    SHA-256."""
    builds = experiment.get("builds") if isinstance(experiment, dict) else None
    if not isinstance(builds, list) or not builds or not all(
            isinstance(build, dict) and isinstance(build.get("toolchain"), str) and isinstance(build.get("source"), str)
            and _HEX64.fullmatch(str(build.get("sha256"))) for build in builds):
        return None
    return builds


def _check_run(run, run_id: str, problem) -> None:
    if not isinstance(run, dict) or run.get("schema") != RUN_SCHEMA:
        problem("run_record", f"{RUN_FILE} is not {RUN_SCHEMA}")
        return
    if set(run) != RUN_KEYS:
        problem("run_record", f"{RUN_FILE} fields differ from {RUN_SCHEMA}: {sorted(set(run) ^ RUN_KEYS)}")
    if run.get("run_id") != run_id:
        problem("run_record", f"{RUN_FILE} names run {run.get('run_id')!r}, retained as {run_id!r}")
    if not isinstance(run.get("host"), str) or not run["host"].strip():
        problem("run_record", f"{RUN_FILE} declares no host")
    if not isinstance(run.get("date"), str) or not _DATE.fullmatch(run["date"]):
        problem("run_record", f"{RUN_FILE} date is not YYYY-MM-DD")
    if run.get("record_origin") not in RECORD_ORIGINS:
        problem("run_record", f"{RUN_FILE} record_origin is not one of {', '.join(RECORD_ORIGINS)}")
    steps = run.get("steps")
    if not isinstance(steps, list) or not steps or not all(isinstance(step, dict) for step in steps):
        problem("run_record", f"{RUN_FILE} lists no workflow steps")
    else:
        for step in steps:
            if step.get("status") == "ran" and step.get("exit_code") != 0:
                problem("run_record", f"{RUN_FILE} records step {step.get('name')!r} ending with {step.get('exit_code')}")
            elif step.get("status") not in ("ran", "not_run"):
                problem("run_record", f"{RUN_FILE} step {step.get('name')!r} has no status ran or not_run")
    for key in RUN_OBJECTS:
        if key in run and not isinstance(run[key], dict):
            problem("run_record", f"{RUN_FILE} {key} is not an object")
    for key in RUN_SENTENCES:
        if key in run and not (isinstance(run[key], list) and all(isinstance(text, str) for text in run[key])):
            problem("run_record", f"{RUN_FILE} {key} is not a list of sentences")
    if isinstance(run.get("procedure"), dict) and run["procedure"].get("kind") not in PROCEDURE_KINDS:
        problem("run_record", f"{RUN_FILE} procedure kind is not one of {', '.join(PROCEDURE_KINDS)}")
    experiment = run["observations"].get(TOOLCHAIN_EXPERIMENT) if isinstance(run.get("observations"), dict) else None
    if experiment is not None and not _toolchain_builds(experiment):
        problem("run_record", f"{RUN_FILE} observations.{TOOLCHAIN_EXPERIMENT} does not list its builds by "
                              "toolchain, source and SHA-256")
    problems = _host_paths(run, RUN_FILE)
    for text in problems:
        problem("run_record", text)


def _native(gate) -> dict:
    """The gate's native artifact identities as the bundles record them (``sha256:`` digests and byte counts)."""
    return {role: {"sha256": "sha256:" + gate["native_artifacts"][role]["sha256"],
                   "byte_count": gate["native_artifacts"][role]["byte_count"]} for role in ("engine", "prover", "guest")}


def _check_gate(gate, build, checks, tests: list, problem) -> None:
    if not isinstance(gate, dict) or gate.get("schema") != GATE_SCHEMA:
        problem("gate", f"gate/gate.json is not {GATE_SCHEMA}")
        return
    for key, expected in (("status", "passed"), ("execution", GATE_EXECUTION), ("physical_truth", "not_established"),
                          ("host_executable_binding", HOST_BINDING)):
        if gate.get(key) != expected:
            problem("gate", f"gate/gate.json {key} is {gate.get(key)!r}, not {expected!r}")
    if "error" in gate:
        problem("gate", "gate/gate.json records an error")
    natives = gate.get("native_artifacts")
    if not isinstance(natives, dict) or set(natives) != {"engine", "prover", "guest"} or not all(
            isinstance(entry, dict) and _HEX64.fullmatch(str(entry.get("sha256")))
            and type(entry.get("byte_count")) is int and entry["byte_count"] > 0 for entry in natives.values()):
        problem("gate", "gate/gate.json does not identify its engine, prover and guest by digest and size")
    if type(gate.get("tests_passed")) is not int or gate["tests_passed"] != len(tests):
        problem("gate", f"gate/gate.json counts {gate.get('tests_passed')} passed tests; gate/tests.xml records "
                        f"{len(tests)}")
    if not isinstance(gate.get("measurements"), dict) or set(gate["measurements"]) != {"original", "replay"}:
        problem("gate", "gate/gate.json lacks the original and replay measurements")
    if isinstance(build, dict) and build.get("guest_build") != "verified_against_committed_registry":
        problem("gate", "build.json does not record the guest as verified against the committed registry")
    if isinstance(checks, dict) and (checks.get("tracked_build_source_bytes") != "unchanged"
                                     or checks.get("runtime_reference") != "separate_clean_checkout"):
        problem("gate", "source-checks.json does not record unchanged SP1 build sources and a separate clean reference")


def _check_pins(gate, build, checks, run, expected, problem) -> None:
    if expected["guest_sha256"] != expected["requirements_guest_sha256"]:
        problem("pins", "ciw.proved_heat.GUEST_SHA256 and SP1_REQUIREMENTS disagree on the guest ELF")
    compared = []
    if isinstance(gate, dict):
        natives = gate.get("native_artifacts") if isinstance(gate.get("native_artifacts"), dict) else {}
        guest = natives.get("guest") if isinstance(natives.get("guest"), dict) else {}
        compared += [("gate/gate.json scr", gate.get("scr"), expected["scr"]),
                     ("gate/gate.json sp1", gate.get("sp1"), expected["sp1"]),
                     ("gate/gate.json guest", guest.get("sha256"), expected["guest_sha256"])]
    if isinstance(build, dict):
        compared += [("build.json recipe_identity", build.get("recipe_identity"), expected["recipe_identity"]),
                     ("build.json elf_sha256", build.get("elf_sha256"), expected["guest_sha256"]),
                     ("build.json compiler_archive_sha256", build.get("compiler_archive_sha256"),
                      expected["compiler_archive_sha256"]),
                     ("build.json sp1_revision", build.get("sp1_revision"), expected["sp1"]["revision"])]
    if isinstance(checks, dict):
        compared += [("source-checks.json sp1", {"revision": checks.get("sp1_revision"), "tree": checks.get("sp1_tree")},
                      expected["sp1"])]
    if isinstance(run, dict) and isinstance(run.get("sources"), dict):
        for role in ("scr", "sp1"):
            source = run["sources"].get(role)
            recorded = {key: source.get(key) for key in ("revision", "tree")} if isinstance(source, dict) else None
            compared.append((f"{RUN_FILE} sources.{role}", recorded, expected[role]))
        compiler = run.get("compiler_archive")
        compared.append((f"{RUN_FILE} compiler_archive", compiler.get("sha256") if isinstance(compiler, dict) else None,
                         expected["compiler_archive_sha256"]))
    for where, recorded, pinned in compared:
        if recorded != pinned:
            problem("pins", f"{where} is {recorded!r}, not CIW's pin {pinned!r}")


def _runtime_pins(runtime, where: str, problem) -> None:
    """A retained SCR runtime identity against the proved-heat pins CIW declares (ciw.lab.bridge)."""
    from .bridge import _pin_check, declared_pins
    declared = declared_pins()
    row = _pin_check(runtime, declared["kinds"]["proved-heat"]["scr"], declared["trees"], "role scr of proved-heat")
    if row["problem"] or not row.get("tree_pinned"):
        problem("pins", f"{where}: {row['problem'] or 'CIW records no source tree for its revision'}")


def _measurements(bundle) -> dict:
    """The projection scripts/check_proved_heat.py writes into gate.json, from one bundle."""
    data = bundle["steps"][0]["result"]["data"]
    timings = data["timings"]
    return {"proof_byte_count": data["proof"]["byte_count"],
            **{key: timings[key] for key in ("native_seconds", "prove_and_verify_seconds", "reverify_seconds")},
            "memory": timings["memory"]}


def _check_bundles(gate, original, replay, reverification, source: bytes | None, problem) -> dict:
    """CIW's offline validation of the retained bundles, replay receipt and re-verification report."""
    from ..proved_heat import VERIFY_SCHEMA, ProvedHeatWorkflow, _identify
    from ..telemetry import digest
    if not isinstance(original, dict) or not isinstance(replay, dict):
        return {}
    workflow = ProvedHeatWorkflow()
    valid = True
    for name, bundle in (("gate/original.json", original), ("gate/replay.json", replay)):
        try:
            workflow._validate(bundle)
        except ValueError as exc:
            problem("bundles", f"{name} is refused by CIW's proved-heat validator: {exc}")
            valid = False
    if not valid:
        return {}
    receipts = replay.get("replay_receipts") or []
    try:
        workflow.validate_replay(original, replay, receipts[0])
    except (ValueError, IndexError) as exc:
        problem("bundles", f"gate/replay.json is not a validated replay of gate/original.json: {exc}")
        return {}
    for name, bundle in (("gate/original.json", original), ("gate/replay.json", replay)):
        runtime = bundle["runtimes"]["scr"]
        _runtime_pins(runtime, f"{name} runtimes.scr", problem)
        if isinstance(gate, dict) and isinstance(gate.get("native_artifacts"), dict):
            try:
                recorded = {role: {key: runtime[role][key] for key in ("sha256", "byte_count")}
                            for role in ("engine", "prover", "guest")}
                if recorded != _native(gate):
                    problem("bundles", f"{name} ran other engine, prover or guest bytes than gate/gate.json names")
            except (KeyError, TypeError):
                problem("bundles", "gate/gate.json native artifacts are malformed")
            label = "original" if name == "gate/original.json" else "replay"
            if (gate.get("measurements") or {}).get(label) != _measurements(bundle):
                problem("bundles", f"gate/gate.json {label} measurements differ from {name}")
    evidence = original["source"]["evidence"][0]
    if source is not None and base64.b64decode(evidence["bytes_b64"]) != source:
        problem("bundles", "gate/source.json differs from the source bytes the bundles retain")
    if not isinstance(reverification, dict):
        return {}
    try:
        expected = {"schema": VERIFY_SCHEMA, "subject_ref": original["bundle_digest"], "outcome": "passed",
                    "independent": False, "result_id": original["steps"][0]["result_id"],
                    "runtime_digest": digest(original["runtimes"]),
                    "proof_identity": original["verification"]["proof_identity"]}
        for key, value in expected.items():
            if reverification.get(key) != value:
                problem("bundles", f"gate/reverification.json {key} does not bind gate/original.json")
        unsealed = {key: value for key, value in reverification.items() if key != "verification_id"}
        if _identify(deepcopy(unsealed))["verification_id"] != reverification.get("verification_id"):
            problem("bundles", "gate/reverification.json verification_id differs from its content")
        runtimes = reverification["verifier_runtimes"]
        if reverification.get("verifier_runtime_digest") != digest(runtimes):
            problem("bundles", "gate/reverification.json verifier_runtime_digest differs from its verifier runtimes")
        _runtime_pins(runtimes["scr"], "gate/reverification.json verifier_runtimes.scr", problem)
        if isinstance(gate, dict) and {role: {key: runtimes["scr"][role][key] for key in ("sha256", "byte_count")}
                                       for role in ("engine", "prover", "guest")} != _native(gate):
            problem("bundles", "gate/reverification.json verified with other engine, prover or guest bytes than "
                               "gate/gate.json names")
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        problem("bundles", f"gate/reverification.json is malformed ({type(exc).__name__})")
        return {}
    data = [bundle["steps"][0]["result"]["data"] for bundle in (original, replay)]
    request = original["steps"][0]["request"]
    return {"initial_values": request["initial_values"], "steps": request["steps"],
            "values": data[0]["native"]["values"], "replay_values": data[1]["native"]["values"],
            "verifier_outcomes": [item["verifier"]["outcome"] for item in data],
            "replay_numerical_match": receipts[0]["numerical_match"],
            "reverification_outcome": reverification["outcome"],
            "proof_sha256": [item["proof"]["sha256"] for item in data],
            "bundle_digests": [original["bundle_digest"], replay["bundle_digest"]]}


def inspect_record(directory) -> dict:
    """Integrity of one retained proved-heat record, with every problem by category.

    Returns ``{"run_id", "problems", "categories", "files", "summary"}``:
    ``categories`` maps each of :data:`CATEGORIES` to its problems, and
    ``summary`` (the run, gate, build and bundle facts T099 reports) is None
    unless the record has no problem. Nothing is executed.
    """
    directory = Path(directory)
    found = {category: [] for category in CATEGORIES}

    def problem(category, text):
        found[category].append(text)

    try:
        check_name(directory.name, "proved-heat run identity")
    except ValueError as exc:
        problem("run_record", str(exc))
    if not directory.is_dir():
        problem("integrity", "the record directory does not exist")
        contents, manifest = {}, None
    else:
        contents, manifest = _files(directory, problem)
    run = _parsed(contents, RUN_FILE, problem, "run_record")
    gate = _parsed(contents, "gate/gate.json", problem, "gate")
    build = _parsed(contents, "build.json", problem, "gate")
    checks = _parsed(contents, "source-checks.json", problem, "gate")
    original = _parsed(contents, "gate/original.json", problem, "bundles")
    replay = _parsed(contents, "gate/replay.json", problem, "bundles")
    reverification = _parsed(contents, "gate/reverification.json", problem, "bundles")
    if RUN_FILE in contents:
        _check_run(run, directory.name, problem)
    tests = _tests(contents.get("gate/tests.xml"), problem)
    if "gate/gate.json" in contents:
        _check_gate(gate, build, checks, tests, problem)
    expected = pins()
    _check_pins(gate, build, checks, run, expected, problem)
    try:
        bundles = _check_bundles(gate, original, replay, reverification, contents.get("gate/source.json"), problem)
    except (KeyError, TypeError, IndexError, AttributeError, ValueError) as exc:
        problem("bundles", f"the retained bundles are malformed ({type(exc).__name__})")
        bundles = {}
    problems = [f"{category}: {text}" for category in CATEGORIES for text in found[category]]
    summary = None
    if not problems:
        summary = {"run_id": directory.name, "date": run["date"], "host": run["host"],
                   "record_origin": run["record_origin"], "procedure": run["procedure"], "steps": run["steps"],
                   "toolchains": run["toolchains"], "observations": run["observations"],
                   "gate": {key: gate[key] for key in ("status", "execution", "tests_passed", "host_executable_binding",
                                                       "scr", "sp1", "native_artifacts", "installed_wheel")
                            if key in gate},
                   "build": {key: build[key] for key in ("recipe_identity", "elf_sha256", "compiler_archive_sha256",
                                                         "sp1_revision", "guest_build", "toolchain_identity")
                             if key in build},
                   "native_tests": [name for name in REQUIRED_NATIVE_TESTS if name in tests],
                   "measurements": gate["measurements"], "bundles": bundles,
                   "manifest_sha256": sha256(manifest).hexdigest()}
    return {"run_id": directory.name, "problems": problems, "categories": found,
            "files": len(contents), "summary": summary}


def record_directories(retained) -> list:
    """Every record directory under ``<retained>/proved-heat`` (its README is not a record)."""
    root = Path(retained) / DIRECTORY
    return [path for path in sorted(root.iterdir()) if path.is_dir()] if root.is_dir() else []


def verify_records(retained) -> dict:
    """Integrity of every record under ``<retained>/proved-heat``; nothing is recomputed or re-verified."""
    records = []
    for directory in record_directories(retained):
        inspected = inspect_record(directory)
        records.append({"run_id": inspected["run_id"], "files": inspected["files"],
                        "date": (inspected["summary"] or {}).get("date"), "problems": inspected["problems"]})
    problems = [f"proved-heat/{record['run_id']}: {problem}" for record in records for problem in record["problems"]]
    return {"verified": len(records), "records": records, "problems": problems, "passed": not problems, "note": NOTE}


def _workspace_bundles(workspace: bytes, original: bytes, replay: bytes) -> str | None:
    """Why ``gate/workspace.json`` is not the two gate bundles plus the session around them, or None."""
    try:
        saved = json.loads(workspace)
        natives = [bundle["native"] for bundle in saved["workbench"]["bundles"]]
    except (ValueError, KeyError, TypeError) as exc:
        return f"gate/workspace.json is malformed ({type(exc).__name__})"
    if natives != [json.loads(original), json.loads(replay)]:
        return "gate/workspace.json holds other bundles than gate/original.json and gate/replay.json"
    return None


def retain_record(run_dir, retained, run_id, host) -> dict:
    """Validate a passing proved-heat gate output and retain it as ``<retained>/proved-heat/<run_id>``.

    ``run_dir`` is the gate's output directory as the workflow writes it
    (``results/proved-heat``) with ``local-run.json`` from
    ``scripts/run_proved_heat_locally.py``. The run is refused unless the gate
    passed and every file :data:`RETAINED` names is present; ``gate/workspace.json``
    is left out once its bundles are checked to be the retained ones. The copy
    is inspected with :func:`inspect_record` before this returns, and removed
    when it has any problem, so a record that names other pins than CIW's, or
    whose bundles CIW refuses, is never retained.
    """
    run_dir, run_id = Path(run_dir), check_name(run_id, "proved-heat run identity")
    host = _declared_host(host)
    destination = Path(retained) / DIRECTORY / run_id
    if destination.exists():
        raise ValueError(f"Proved-heat run {run_id} is already retained at {destination}")
    try:
        local = json.loads((run_dir / LOCAL_RUN).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Not a proved-heat gate run with a run description (no readable {LOCAL_RUN}; "
                         f"scripts/run_proved_heat_locally.py writes it): {run_dir}") from exc
    if not isinstance(local, dict) or local.get("schema") != LOCAL_RUN_SCHEMA:
        raise ValueError(f"{run_dir / LOCAL_RUN} is not {LOCAL_RUN_SCHEMA}")
    missing, unknown = LOCAL_RUN_KEYS - set(local), set(local) - LOCAL_RUN_KEYS - LOCAL_RUN_OPTIONAL
    if missing or unknown:
        raise ValueError(f"{LOCAL_RUN} fields differ from {LOCAL_RUN_SCHEMA}: missing {sorted(missing)}, "
                         f"unknown {sorted(unknown)}")
    missing = [name for name in RETAINED if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"The gate output is incomplete: {missing}")
    try:
        gate = json.loads((run_dir / "gate" / "gate.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("gate/gate.json is unreadable") from exc
    if not isinstance(gate, dict) or gate.get("status") != "passed":
        raise ValueError(f"Refusing a gate run that did not pass (gate/gate.json status "
                         f"{gate.get('status') if isinstance(gate, dict) else None!r})")
    omitted = {}
    workspace = run_dir / "gate" / "workspace.json"
    if workspace.is_file():
        data = workspace.read_bytes()
        why = _workspace_bundles(data, (run_dir / "gate" / "original.json").read_bytes(),
                                 (run_dir / "gate" / "replay.json").read_bytes())
        if why:
            raise ValueError("Refusing the run: " + why)
        omitted["gate/workspace.json"] = {**_entry(data), "reason": OMITTED["gate/workspace.json"]}
    if (run_dir / "local-logs").is_dir():
        omitted["local-logs"] = {"files": sorted(path.name for path in (run_dir / "local-logs").iterdir()),
                                 "reason": OMITTED["local-logs"]}
    run = {"schema": RUN_SCHEMA, "run_id": run_id, "host": host, "date": str(local.get("started_utc", ""))[:10],
           **{key: local[key] for key in sorted(LOCAL_RUN_KEYS - {"schema"})},
           "observations": local.get("observations") or {}, "notes": local.get("notes") or [],
           "omitted": omitted, "limitations": LIMITATIONS, "note": NOTE}
    paths = _host_paths(run, RUN_FILE)
    if paths:
        raise ValueError("Refusing the run description: " + "; ".join(paths[:5]))
    destination.mkdir(parents=True)
    try:
        files = {}
        for source, target in [*RETAINED.items(), *((name, path) for name, path in OPTIONAL.items()
                                                     if (run_dir / name).is_file())]:
            data = (run_dir / source).read_bytes()
            stored = compress(data) if target.endswith(".gz") else data
            (destination / target).parent.mkdir(parents=True, exist_ok=True)
            (destination / target).write_bytes(stored)
            files[target] = {**_entry(stored), **({"content": _entry(data)} if target.endswith(".gz") else {})}
        encoded = dumps(run).encode("utf-8")
        (destination / RUN_FILE).write_bytes(encoded)
        files[RUN_FILE] = _entry(encoded)
        manifest = {"schema": MANIFEST_SCHEMA, "run_id": run_id, "files": dict(sorted(files.items())),
                    "note": "SHA-256 and size of every file of this record except this manifest; for a .gz file also "
                            "of the gate's bytes inside it (content)"}
        (destination / MANIFEST_FILE).write_text(dumps(manifest), encoding="utf-8")
        inspected = inspect_record(destination)
        if inspected["problems"]:
            raise ValueError("The retained copy fails verification: " + "; ".join(inspected["problems"][:5]))
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return {"run_id": run_id, "retained": str(destination), "date": run["date"], "host": host,
            "files": len(files), "bytes": sum(entry["bytes"] for entry in files.values()), "note": NOTE}
