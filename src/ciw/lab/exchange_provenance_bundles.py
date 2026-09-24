"""CIW exchange and provenance, part 2: bundles, providers and visibility (T091-T100).

Scope: these tasks exercise the real CIW session, workbench, energy-accuracy,
declared-workload, exchange and candidate-evidence code offline. They save and
reopen workspaces under an execution guard, refuse replay without trusted
bindings, prove refusals leave state unchanged, reopen golden retained
bundles, refuse malformed exchange inputs, run provider-free conformance checks,
and, when pinned provider checkouts are bound, record their identities, build
the SCR engine with ``cargo build --locked --offline`` and execute SCR, SET and
PPDA integrations. The last task audits that synthetic, provider-backed and
physical results stay visibly distinct in retained reports and CIW records.

Non-claims: content consistency of a retained bundle is not numerical
correctness (a fabricated heat bundle passes reopen validation), and a retained
runtime identity is not proof that the named provider computed the values; a
pinned provider result is not independent verification by another party; a
matching checkout digest does not authenticate the upstream repository, the
toolchain or a built engine; the energy fixtures are synthetic and nothing here
measures energy, heat or any physical quantity.
"""
from __future__ import annotations

import ast
import base64
from contextlib import nullcontext
from copy import deepcopy
from hashlib import sha256
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from .. import __version__
from .evidence import (ACCEPTED_ACQUISITION, AUTHORITY_DOMAINS, COMPUTATIONAL_DOMAINS, LABELS, NO_ORIGIN, ORIGINS,
                       ORIGIN_WORDS, PHYSICAL_DOMAINS, UNACCEPTED_ACQUISITION, EvidenceRefusal, finding,
                       finding_origin, holds, origin_counts, validate_finding)
from .registry import task
from .report import FINDINGS_HEADER, FINDINGS_RULE, build_report, finding_row, render_markdown, validate_report
from .svg import line_plot
from .exchange_provenance_bundles_fixtures import (FABRICATED_SOURCE_TREE, GOLDEN_MANIFEST, GOLDEN_PLATFORM,
                                                   GOLDEN_SCR_ENGINE_SHA256, SYNTHETIC_BINDING, Client,
                                                   ExecutionForbidden, add_provider_content, build_energy_session,
                                                   energy_recomputation_within, example_bytes, execution_guard,
                                                   execution_paths, fabricated_heat_catalog, fixture_root,
                                                   heat_reference, malformed_fixtures, merge_catalogs, path_label,
                                                   platform_fingerprint, repository_example,
                                                   retained_energy_analyses, source_payload,
                                                   workflow_entry_points)
from . import exchange_provenance_bundles_providers as providers

MODULE = "src/ciw/lab/exchange_provenance_bundles.py"
FIXTURES = "src/ciw/lab/exchange_provenance_bundles_fixtures.py"
PROVIDERS = "src/ciw/lab/exchange_provenance_bundles_providers.py"
TESTS = "tests/test_lab_exchange_provenance_bundles.py"
EXACT = {"abs": 0, "rel": 0}
# Per-finding uncertainty: every value here is an exact count, digest, string or
# integer; nothing is rounded.
EXACT_COUNT = {"kind": "roundoff", "value": 0,
               "basis": "exact counts, digests and string comparisons; no rounding is involved"}
EXACT_INTEGER = {"kind": "roundoff", "value": 0, "basis": "exact int64 arithmetic compared cell by cell"}
UNBOUND = "No trusted repositories bound for this workbench workflow"
UPSTREAM_REFUSAL = "invalid_payload: Select a retained upstream bundle of the declared kind"
# Registered workflow kinds whose provider is never bound by this section, with
# an embedded source each can retain before execution is refused.
UNBOUND_SOURCES = (
    ("numerical-heat", "declared-workloads/numerical-heat.json"),
    ("proved-heat", "proved-heat/source.json"),
    ("flat-torus-reference", "geodesic-reference/flat-torus.json"),
    ("curved-path-transfer", "geodesic-reference/curved-path.json"),
    ("translation-flow", "geometry-research/translation-flow.json"),
)
# Further unbound kinds with an example source under examples/, read through
# runner.repository_path (an installed wheel without examples/ records them as
# not exercised). Telemetry also needs its example configuration.
REPOSITORY_SOURCES = (
    ("calibrated-observable", "calibrated-observable/source.json"),
    ("telemetry", "telemetry/source.json"),
    ("calibrated-window", "calibrated-window/source.json"),
    ("schematic-assessment", "declared-workloads/schematic-assessment.json"),
    ("acquired-dataset", "acquired-dataset/source.json"),
    ("measurement-chain", "measurement-chain/source.json"),
    ("geometric-circle", "geometric-circle/source.json"),
    ("covariance-geometry", "geometry-research/covariance-geometry.json"),
    ("mesh-path", "geometry-research/mesh-path.json"),
    ("variational-free-energy", "variational-free-energy/baseline.json"),
)
# Kinds that consume a retained upstream bundle of another kind: Workbench.execute
# checks the upstream selection before it consults the binding.
UPSTREAM_SOURCES = (
    ("identified-design", "identified-design/source.json"),
    ("schematic-companions", "declared-workloads/schematic-companions.json"),
)
_PROCESS_PATHS = frozenset({"subprocess.Popen", "ciw.adapters.subprocess._bounded_process",
                            "ciw.adapters.subprocess.PinnedSubprocessAdapter.__init__",
                            "ciw.candidate_evidence._bounded_process"})


def _provider_paths() -> frozenset:
    """Entry points that start or reach a provider: processes, pinned adapters and every workflow entry point
    except those the builtin energy-accuracy workflow defines itself."""
    return _PROCESS_PATHS | {path_label(entry) for entry in workflow_entry_points()
                             if entry[1] != "EnergyAccuracyWorkflow"}


def _check(reference, observed, tolerance=0.0, comparison="abs_le", kind="exact_arithmetic"):
    observed, tolerance = float(observed), float(tolerance)
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": holds(observed, tolerance, comparison)}


def _refuted(findings) -> list:
    """Computational findings whose checks failed (not honestly flagged as unestablished)."""
    return [record["claim"] for record in findings if record["domain"] in COMPUTATIONAL_DOMAINS
            and record["evidence_status"] == "not_established" and not record.get("expected_not_established")]


def _settle(state: str, findings) -> str:
    """A refuted computational finding makes a completed task partial (the contract forbids completed)."""
    return "partial" if state == "completed" and _refuted(findings) else state


def _runtime_identity(changed_files, bound: dict) -> dict:
    """Provider identities plus the built-in CIW identity of the code these tasks ran."""
    from .runner import builtin_identity
    return {**bound, "ciw": builtin_identity(changed_files)}


def _sanitized(message: str, paths) -> str:
    """Error text without operator-specific checkout paths (report prose must not depend on location)."""
    for path in paths:
        for spelling in {str(path), str(Path(path).resolve())}:
            message = message.replace(spelling, "<checkout>")
    return message


def _located(value, bindings: dict):
    """A retained artifact with host paths replaced by their role, e.g. ``<scr>`` or ``<python>``.

    Checkout and interpreter locations differ between hosts and runs; the pins, trees and digests
    recorded beside them identify the providers.
    """
    if isinstance(value, dict):
        return {key: _located(item, bindings) for key, item in value.items()}
    if isinstance(value, list):
        return [_located(item, bindings) for item in value]
    if isinstance(value, str):
        for role, path in sorted(bindings.items(), key=lambda item: -len(str(item[1]))):
            for spelling in sorted({str(path), str(Path(path).resolve())}, key=len, reverse=True):
                if spelling and spelling in value:
                    value = value.replace(spelling, f"<{role}>")
    return value


def _host_bindings(ctx) -> dict:
    return {**{role: path for role, path in ctx.providers.items() if path}, "python": sys.executable}


def _refusal(reference, expected, observed):
    observed = "none" if observed is None else str(observed)
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed, "passed": observed == expected}


def _fields(hypothesis, model, inputs, observation, invariant, experiment, result, uncertainty, failures,
            assumptions, next_task):
    return {"hypothesis": hypothesis, "mathematical_model": model, "input_data": inputs,
            "observation_model": observation, "expected_invariant": invariant, "experiment": experiment,
            "numerical_result": result, "uncertainty": uncertainty, "failure_modes_checked": failures,
            "unresolved_assumptions": assumptions, "recommended_next_task": next_task}


# Each task's recommended next step names its own open question, never a queue task that has already run.
# ``ciw lab next`` lists them as research rows (tests/test_lab_exchange_provenance_bundles.py checks with
# ciw.lab.planner that none is a pointer).
NEXT_STEPS = {
    "T091": ("Deferred research question: intercept execution below CIW's Python entry points while a workspace "
             "reopens (a sys.addaudithook hook on subprocess.Popen, os.exec*, os.posix_spawn and ctypes.dlopen, run "
             "in a child process because audit hooks cannot be removed), so that a workflow reached through a name "
             "imported into another module, or through code outside the listed entry points, is caught; and count "
             "validation recomputation for every retained kind, not only the energy analysis."),
    "T092": ("Deferred research question: observe, rather than infer, replay refusal for the provider kinds other "
             "than numerical-heat by retaining a bundle of each from its bound provider (the telemetry stack pinned "
             "in src/ciw/telemetry-runtimes.json for telemetry bundles, and the calibrated-observable stack pinned in "
             "src/ciw/calibrated-observable-runtimes.json for calibrated-observable bundles, both of which reach the "
             "'No operator-bound ESM candidate adapter' refusal) and replaying it in a reopened unbound "
             "session; and add example sources for acquired-calibrated-window, bim-quantity, identified-stability "
             "and residual-monitor so that their execution refusal is observed too."),
    "T093": ("Deferred research question: generate the refused requests from CIW's own request table (every request "
             "type Session._dispatch accepts, each with an unknown identity, a wrong-typed field and an unknown field) "
             "instead of the hand-written classes, and compare the same state digests; and decide, as a CIW design "
             "question, whether a refused recording operation should stay in the saved workspace as an execution "
             "record (the retained counterexample) or be kept out of it."),
    "T094": ("Deferred research question: reopen the golden workspaces on Windows x86-64, macOS arm64 and a NumPy "
             "built on another LAPACK to learn whether the energy recomputation stays within the rounding tolerance "
             "measured across three OpenBLAS kernels on one x86-64 host; decide, as a CIW design question, whether "
             "reopen should compare the energy recomputation within a declared tolerance instead of bit for bit, so "
             "that a workspace written on one platform reopens on another; and retain golden workspaces for "
             "provider kinds other than numerical-heat, produced by their pinned providers, so that validator drift "
             "in those kinds is caught at reopen."),
    "T095": ("Deferred research question: apply the malformed matrix to the source parsers of the workbench kinds "
             "other than energy-accuracy, which parse after the same canonical-base64 and byte-budget checks; and, "
             "once CIW makes the changes docs/lab/EXCHANGE_BUNDLES.md requests (read_json refusing an overflowing "
             "number and turning RecursionError into ValueError, Session.from_workspace refusing unknown or missing "
             "top-level fields with a ValueError, source.add naming the source rather than a runtime), re-run it to "
             "check that each of the five counterexamples becomes a refusal with its own text."),
    "T096": ("Deferred research question: explore combined mutations that restore consistency (two or more fields "
             "changed together, including a recomputed identity) against exchange._identity and "
             "candidate_evidence.validate_response, which single-field mutations cannot reach; and, as CIW changes, "
             "bind observation-batch identities to their content and refuse unknown ESM response fields, then re-run "
             "to check that both counterexamples become refusals."),
    "T097": ("Deferred research question: provision the telemetry provider stack (ppda, stfe, gsie, set and cbsr at "
             "the src/ciw/telemetry-runtimes.json pins; scripts/check_lab.py provisions only ppda and set of these, at "
             "other pins) and drive CIW's telemetry workflow end to end against it, as the exchange roundtrip here "
             "does for PPDA, SCR and SET; and attest the SCR engine a bound replay uses (CIW records a bound engine as "
             "operator_asserted_not_attested), for example by requiring a locked build of the bound checkout in the "
             "same run before a replay is accepted."),
    "T098": ("Deferred research question: authenticate what the digests only identify, by verifying signed upstream "
             "commits or tags of each bound checkout against its maintainers' published keys and recording digests "
             "of the Rust and Python toolchains, since a matching HEAD, tree and lock digest is not authentication; "
             "and settle why CIW pins two SCR revisions (a59aba2 for declared-workload and proved-heat, 5f04097 for "
             "the exchange workflow) and several SET revisions, converging them or recording each workflow's "
             "reason."),
    "T099": ("Run the SP1 proved-heat gate (scripts/check_proved_heat.py) on a machine that meets its recorded "
             "requirements (network access to crates.io and GitHub releases, the SP1 checkout b38b612, protoc, at "
             "least 7 GiB of RAM and 20 GiB of disk) and retain its outcome; and rebuild execution-cli with CI's "
             "pinned rustc 1.94.0 beside this run's toolchain to find whether the engine digest depends on the "
             "toolchain."),
    "T100": ("Deferred research question (CIW change): key the workspace seals, or have the provider sign its runtime "
             "identity at execution under a defined key custody, so that a bundle copying CIW's public pins (the "
             "retained copied-pin counterexample) is refused or classified not_established; compare a retained "
             "runtime identity with CIW's pins on reopen (Session.from_workspace), as ciw lab classify does; record "
             "in CIW's pin tables the source tree of every pinned revision that has none (the kinds "
             "ciw.lab.bridge.pins_without_tree lists, such as telemetry and calibrated-observable), so that the "
             "invented-tree check reaches them; and authenticate energy-log origin at acquisition, outside the "
             "workbench, so that a resealed relabel under a fresh occurrence is detected."),
}


def _outcome(response):
    """``code: message`` of a refused protocol response; ``accepted`` otherwise."""
    if response["type"] == "error":
        return f"{response['payload']['code']}: {response['payload']['message']}"
    payload = response["payload"]
    if isinstance(payload, dict) and payload.get("status") == "refused":
        refusal = payload["execution"].get("refusal") or {}
        return f"{refusal.get('code')}: {refusal.get('message')}"
    return "accepted"


def _digest(value) -> str:
    from ..telemetry import canonical
    return sha256(canonical(value)).hexdigest()


def _state(session) -> dict:
    """Digests of every in-memory part of a session a request could change."""
    workbench = session.workbench
    parts = {"selection": session.selection, "results": session.results, "executions": session.executions,
             "catalog": workbench.serialize(), "revision": workbench._revision, "used_bytes": workbench._used_bytes,
             "reserved_bytes": workbench._reserved_bytes, "pending": workbench._pending,
             "identities": sorted(workbench._identities), "candidates": sorted(workbench._candidates),
             "bindings": {kind: sorted(map(str, value)) for kind, value in workbench._bindings.items()}}
    return {name: _digest(value) for name, value in parts.items()}


def _listing(directory: Path) -> dict:
    return {path.name: sha256(path.read_bytes()).hexdigest() for path in sorted(Path(directory).iterdir())
            if path.is_file()}


def _binding_keys(value) -> int:
    """Count keys in a saved workspace that could carry an executable binding."""
    names = {"bindings", "_bindings", "repositories", "repository_bindings", "engine_path", "executable"}
    if isinstance(value, dict):
        return sum((key in names) + _binding_keys(item) for key, item in value.items())
    if isinstance(value, list):
        return sum(_binding_keys(item) for item in value)
    return 0


def _without_saved_at(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    value.pop("saved_at", None)
    return value


def _session_class():
    from ..session import Session
    return Session


# ---------------------------------------------------------------- T091

@task("T091", changed_files=(MODULE, FIXTURES),
      regression_tests=(f"{TESTS}::test_t091_reopen_needs_no_provider_and_reaches_no_execution_path",
                        f"{TESTS}::test_execution_guard_restores_every_patched_attribute"))
def save_reopen_without_providers(ctx):
    Session = _session_class()
    declared = json.loads(example_bytes("declared-workloads/numerical-heat.json"))
    with tempfile.TemporaryDirectory(prefix="ciw-lab-t091-") as scratch:
        scratch = Path(scratch)
        session, _, ids = build_energy_session(scratch / "original")
        # Provider-kind content and a trusted binding in the saving session: what must not survive a save/reopen.
        ids["heat"] = add_provider_content(session, heat_reference(declared["initial_values"], declared["steps"]))
        original_bindings = sorted(kind for kind, value in session.workbench._bindings.items() if value)
        original_available = sorted(o["operation_id"] for o in session.workbench.describe_operations() if o["available"])
        path = session.save_workspace(scratch / "saved" / "workspace.json")
        saved_bytes = path.read_bytes()
        saved_text = saved_bytes.decode("utf-8")
        binding_paths_saved = sum(location in saved_text for location in SYNTHETIC_BINDING.values())
        reads = {}
        with execution_guard() as guard:
            reopened = Session.from_workspace(path, scratch / "reopened")
            reader = Client(reopened)
            for kind, payload in (("session.get", {}), ("source.list", {}), ("bundle.list", {}), ("result.list", {}),
                                  ("execution.list", {}), ("operation.list", {}),
                                  ("experiment.inspect", {"bundle_id": ids["original"]})):
                reads[kind] = reader.call(kind, payload)["type"]
            reads["bundle.get numerical-heat"] = reader.call("bundle.get", {"bundle_id": ids["heat"]})["type"]
            for result_id in sorted(reopened.results):
                reads["result.get " + result_id] = reader.call("result.get", {"result_id": result_id})["type"]
            reopen_attempts = list(guard["attempts"])
            recomputations = dict(guard["recomputations"])
            # Sensitivity control: an execution attempted inside the guard must be caught.
            try:
                reopened.workbench.replay({"bundle_id": ids["original"]})
                control = "not_intercepted"
            except ExecutionForbidden:
                control = "intercepted"
        mismatches = {name: int(_digest(getattr(session, name)) != _digest(getattr(reopened, name)))
                      for name in ("selection", "results", "executions")}
        mismatches["catalog"] = int(_digest(session.workbench.serialize()) != _digest(reopened.workbench.serialize()))
        bindings = sorted(kind for kind, value in reopened.workbench._bindings.items() if value)
        available = sorted(o["operation_id"] for o in reopened.workbench.describe_operations() if o["available"])
        saved = json.loads(saved_bytes)
        binding_keys = _binding_keys(saved)
        # Positive control after leaving the guard: the binding-free builtin still replays.
        replay = Client(reopened).call("bundle.replay", {"bundle_id": ids["original"]})
        replay_match = replay["type"] == "response" and replay["payload"]["replay_receipt"]["numerical_match"] is True
        # The binding the saving session held is not recovered: replaying its provider-kind bundle is refused.
        heat_replay = _outcome(Client(reopened).call("bundle.replay", {"bundle_id": ids["heat"]}))
        unchanged = path.read_bytes() == saved_bytes
    guarded = [path_label(entry) for entry in execution_paths()]
    observed = {"workspace_version": saved["workspace_version"], "bundles": len(saved["workbench"]["bundles"]),
                "provider_kind_bundles": sorted(b["kind"] for b in saved["workbench"]["bundles"]
                                                if b["kind"] != "energy-accuracy"),
                "results": len(saved["results"]), "executions": len(saved["executions"]),
                "bindings_before_save": original_bindings, "execution_path_calls": len(reopen_attempts),
                "provider_bindings": bindings, "available_operations": available, "read_requests": len(reads),
                "read_requests_answered": sum(value == "response" for value in reads.values())}
    ctx.artifact_json("reopen.json", {"observed": observed, "reads": reads, "mismatches": mismatches,
                                      "reopen_attempts": reopen_attempts, "recomputations": recomputations,
                                      "guard_control": control, "binding_keys_in_saved_workspace": binding_keys,
                                      "synthetic_binding": SYNTHETIC_BINDING,
                                      "binding_paths_in_saved_workspace": binding_paths_saved,
                                      "available_before_save": original_available, "provider_bundle_replay": heat_replay,
                                      "saved_workspace_sha256": sha256(saved_bytes).hexdigest(),
                                      "guarded_paths": guarded})
    analyze_calls = recomputations.get("ciw.energy_records.analyze", 0)
    findings = [
        finding("A saved workspace from a session holding a trusted provider binding, with oscillator results, an "
                "energy-accuracy original and replay and a provider-kind numerical-heat bundle, reopens with no "
                "provider binding and reaches no execution path", "computational_pipeline", observed,
                {"checks": [_check("trusted provider bindings held by the saving session", len(original_bindings), 1,
                                   "ge", "invariant"),
                            _check("provider-kind bundles in the saved workspace",
                                   len(observed["provider_kind_bundles"]), 1, "ge", "invariant"),
                            _check("execution entry points reached while reopening and reading", len(reopen_attempts)),
                            _check("selection/results/executions/catalog digest mismatches after reopen",
                                   sum(mismatches.values())),
                            _check("provider bindings present after reopen", len(bindings)),
                            _check("binding-like keys in the saved workspace", binding_keys),
                            _check("synthetic binding paths present in the saved workspace bytes", binding_paths_saved),
                            _check("read requests not answered", len(reads) - observed["read_requests_answered"]),
                            _check("saved workspace bytes changed by reopen", int(not unchanged)),
                            _refusal("bundle.replay of the provider-kind bundle after reopen",
                                     f"operation_unavailable: {UNBOUND}", heat_replay)]},
                uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("The execution guard intercepts a replay attempted while it is active", "computational_pipeline",
                control, {"checks": [_refusal("Workbench.replay inside execution_guard", "intercepted", control)]},
                uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("Reopening recomputes the retained energy analysis to validate content", "computational_pipeline",
                analyze_calls, {"checks": [_check("ciw.energy_records.analyze calls during reopen and reads",
                                                  analyze_calls, 1, "ge", "invariant")]},
                uncertainty=EXACT_COUNT, tolerance=EXACT, counterexample={
                    "statement": "Reopening a saved workspace performs no numerical recomputation",
                    "witness": {"ciw.energy_records.analyze calls": analyze_calls,
                                "new execution occurrences": 0}}),
        finding("After reopen the binding-free energy-accuracy workflow still replays with a numerical match",
                "computational_pipeline", replay_match,
                {"checks": [_check("replay numerical_match is false", int(not replay_match))]},
                uncertainty=EXACT_COUNT, tolerance=EXACT),
    ]
    fields = _fields(
        "A CIW workspace saved by a session that held a trusted provider binding, with retained oscillator results, "
        "an energy-accuracy original and replay and a provider-kind bundle, can be reopened with no provider binding "
        "and without reaching any execution entry point, and the binding is not recovered.",
        "Reopen = validate(saved JSON) then construct; the guard replaces "
        f"{len(guarded)} execution entry points (session analyses, recording operations, the workbench, every "
        "workbench workflow's session/step/adapter entry points, provider adapters, subprocesses) with a refusing "
        "recorder.",
        ["ciw.instruments.make_demo_run (synthetic analytic oscillator)",
         "examples/energy-accuracy/baseline.json bytes (synthetic_fixture origin), embedded",
         "a provider-kind numerical-heat bundle (fabricated_heat_catalog: values equal to the integer reference, "
         "fabricated runtime identity) and a synthetic trusted numerical-heat binding naming no checkout"],
        "Protocol v1 requests against the reopened Session; digests of selection, results, executions and the "
        "retained catalog; guard attempt and recomputation counters; the saved bytes.",
        "Zero execution-path calls, zero bindings, no binding path in the saved bytes, identical retained content, "
        "answered read requests, provider-kind replay refused.",
        "Build a session (stats, selection update, spectrum, statistics.v1 operation, energy source, execute, "
        "replay), add a provider-kind numerical-heat bundle and a synthetic trusted binding, save, reopen under the "
        "guard, issue read requests, compare digests, run a guard sensitivity control, then after the guard a "
        "binding-free replay control and a replay of the provider-kind bundle.",
        f"{observed['bundles']} bundles ({', '.join(observed['provider_kind_bundles'])} from a provider kind), "
        f"{observed['results']} results and {observed['executions']} execution reopened with {len(reopen_attempts)} "
        f"execution-path calls; bindings before save {original_bindings}, after reopen {bindings}; provider-kind "
        f"replay: {heat_replay}; the energy analysis was recomputed {analyze_calls} times for validation.",
        "Exact: counts and digest equalities; no floating-point tolerance is involved.",
        ["execution entry point reached during reopen", "retained content drift across reopen",
         "binding restored from saved data", "binding path written into the saved workspace",
         "saved workspace rewritten by reopen", "guard insensitivity (control replay must be intercepted)"],
        ["The provider-kind bundle is fabricated (no provider ran here) and the binding names no checkout: the "
         "task tests that save/reopen carry neither execution nor binding, not provider output. A bundle produced "
         "by the real SCR is reopened under the same guard in T094 (golden) and T097 (live, when SCR is bound).",
         "The guard covers the entry points listed in reopen.json: EXECUTION_PATHS plus the entry points of every "
         "workbench workflow kind, discovered from ciw.workbench._workflow. Entry points reached through a name "
         "imported into another module, or code paths outside these, are not intercepted.",
         "Validation recomputation is counted only for the energy analysis (ciw.energy_records.analyze), separately "
         "from execution because it creates no occurrence and no result; in-process validation recomputation of "
         "other kinds is not counted."],
        NEXT_STEPS["T091"])
    return {"state": _settle("completed", findings), "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T092

def _unbound_session(scratch: Path):
    """Reopened session holding energy bundles, unbound-kind sources and a fabricated heat bundle.

    Returns the reopened session, identities, the retained source descriptors
    by kind, the fabricated catalog, reopen attempts, the upstream-gated kinds
    whose sources were retained, the repository kinds whose example was not
    reachable and the reopen outcome of the fabricated bundle (``accepted`` or
    the refusal text). If reopen refuses the fabricated bundle, the workspace is
    reopened without it so the other refusal cases still run.
    """
    Session = _session_class()
    session, client, ids = build_energy_session(scratch / "original")
    sources, unreachable, upstream = {}, [], []
    for kind, name in UNBOUND_SOURCES:
        sources[kind] = client.ok("source.add", source_payload(kind, example_bytes(name), f"Unbound {kind} source"))
    for group, kinds in (("unbound", REPOSITORY_SOURCES), ("upstream", UPSTREAM_SOURCES)):
        for kind, name in kinds:
            raw = repository_example(name)
            if raw is None:
                unreachable.append(kind)
                continue
            sources[kind] = client.ok("source.add", source_payload(kind, raw, f"Unbound {kind} source"))
            if group == "upstream":
                upstream.append(kind)
    plain = session.save_workspace(scratch / "saved" / "workspace.json")
    workspace = json.loads(plain.read_text(encoding="utf-8"))
    fabricated = fabricated_heat_catalog([0, 1, 2, 3, 0])
    workspace["workbench"] = merge_catalogs(workspace["workbench"], fabricated)
    path = scratch / "saved" / "with-fabricated-heat.json"
    path.write_text(json.dumps(workspace, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with execution_guard() as guard:
        try:
            reopened = Session.from_workspace(path, scratch / "reopened")
            outcome = "accepted"
        except ValueError as exc:
            outcome = str(exc)
            reopened = Session.from_workspace(plain, scratch / "reopened-without-fabricated")
    if outcome == "accepted":
        ids["heat"] = fabricated["bundles"][0]["bundle_id"]
    return reopened, ids, sources, fabricated, list(guard["attempts"]), upstream, unreachable, outcome


def _replay_refusal_cases(ids, sources, upstream=()):
    """Refusal cases; the retained-heat replay case needs the fabricated bundle (``ids["heat"]``) to be present."""
    heat_source = sources["numerical-heat"]["source_id"]
    cases = []
    for kind in sorted(set(sources) - set(upstream)):
        parameters = {"source_id": sources[kind]["source_id"]}
        if kind == "telemetry":
            # Workbench.execute requires telemetry's explicit configuration before the binding check.
            parameters["configuration"] = json.loads(repository_example("telemetry/configuration.json") or b"{}")
        cases.append((f"execute unbound {kind}", "operation.execute",
                      {"operation_id": f"ciw.{kind}.v1", "parameters": parameters},
                      f"operation_unavailable: {UNBOUND}"))
    cases += [(f"execute {kind} with an upstream bundle of another kind", "operation.execute",
               {"operation_id": f"ciw.{kind}.v1", "parameters": {"source_id": sources[kind]["source_id"],
                                                                "upstream_bundle_id": ids["original"]}},
               UPSTREAM_REFUSAL) for kind in sorted(upstream)]
    if "heat" in ids:
        cases.append(("replay retained numerical-heat bundle without SCR binding", "bundle.replay",
                      {"bundle_id": ids["heat"]}, f"operation_unavailable: {UNBOUND}"))
    cases += [
        ("replay with client-supplied repositories", "bundle.replay",
         {"bundle_id": ids.get("heat", ids["original"]), "repositories": {"scr": "/client/scr", "engine": "/client/engine"}},
         "invalid_payload: Unexpected or missing workbench fields"),
        ("execute with client-supplied repositories", "operation.execute",
         {"operation_id": "ciw.numerical-heat.v1", "parameters": {"source_id": heat_source,
                                                                  "repositories": {"scr": "/client/scr"}}},
         "invalid_payload: Unknown payload fields: repositories"),
        ("client request to bind a workflow", "workflow.bind",
         {"kind": "numerical-heat", "repositories": {"scr": "/client/scr"}},
         "unknown_command: Unknown request type: workflow.bind"),
        ("replay an unknown bundle", "bundle.replay", {"bundle_id": "sha256:" + "0" * 64},
         "invalid_payload: Unknown retained workbench bundle"),
        ("unversioned operation identity", "operation.execute", {"operation_id": "ciw.lab-unregistered"},
         "invalid_payload: operation_id must be versioned and parameters must be an object"),
        ("ESM candidate inspection of a non-telemetry bundle", "operation.execute",
         {"operation_id": "esm.inspect-candidate.v1",
          "parameters": {"bundle_id": ids["original"], "inspected_at": "2026-09-23T00:00:00Z"}},
         "invalid_payload: ESM requires an explicitly selected calibrated or telemetry bundle"),
        ("unregistered recording operation", "operation.execute", {"operation_id": "ciw.lab-unregistered.v1"},
         "operation_unavailable: No trusted provider bound for ciw.lab-unregistered.v1"),
    ]
    return cases


def _workflow_kinds(session) -> list:
    """Workflow kinds whose operation the workbench lists as unavailable (no trusted binding)."""
    return sorted(operation["operation_id"][len("ciw."):-len(".v1")] for operation in
                  session.workbench.describe_operations()
                  if operation["operation_id"].startswith("ciw.") and not operation["available"])


@task("T092", changed_files=(MODULE, FIXTURES),
      regression_tests=(f"{TESTS}::test_t092_unbound_replay_and_execution_are_refused",
                        f"{TESTS}::test_fabricated_heat_bundle_is_content_consistent_but_wrong",
                        f"{TESTS}::test_t092_next_step_names_the_manifest_that_pins_each_kind"))
def replay_refusal_without_binding(ctx):
    with tempfile.TemporaryDirectory(prefix="ciw-lab-t092-") as scratch:
        (reopened, ids, sources, fabricated, reopen_attempts, upstream, unreachable,
         fabricated_reopen) = _unbound_session(Path(scratch))
        unavailable = _workflow_kinds(reopened)
        client = Client(reopened)
        outcomes = []
        # Count (without refusing) every execution entry point the refusals reach.
        with execution_guard(refuse=False) as guard:
            for label, kind, payload, expected in _replay_refusal_cases(ids, sources, upstream):
                before = len(guard["attempts"])
                observed = _outcome(client.call(kind, payload))
                outcomes.append({"case": label, "request": kind, "expected": expected, "observed": observed,
                                 "entry_points": guard["attempts"][before:]})
        provider_paths = _provider_paths()
        provider_reached = sorted({p for row in outcomes for p in row["entry_points"] if p in provider_paths})
        control = client.call("bundle.replay", {"bundle_id": ids["original"]})
        control_ok = control["type"] == "response" and control["payload"]["replay_receipt"]["numerical_match"] is True
        # What a workbench reader sees of the fabricated bundle (the catalog's own copy when reopen refused it).
        heat = (reopened.workbench.get_bundle(ids["heat"]) if "heat" in ids
                else deepcopy(fabricated["bundles"][0]["native"]))
    declared = json.loads(example_bytes("declared-workloads/numerical-heat.json"))
    reference = heat_reference(declared["initial_values"], declared["steps"])
    retained = heat["steps"][0]["result"]["data"]["values"]
    mismatched = sum(a != b for a, b in zip(retained, reference))
    ctx.artifact_json("refusals.json", {"cases": outcomes, "provider_paths_reached": provider_reached,
                                        "provider_paths_counted": sorted(provider_paths),
                                        "reopen_attempts": reopen_attempts, "fabricated_bundle_reopen": fabricated_reopen,
                                        "energy_replay_control": _outcome(control)})
    ctx.artifact_json("fabricated-heat-catalog.json", fabricated)
    refused = sum(row["observed"] == row["expected"] for row in outcomes)
    refused_unbound = sorted(set(sources) - set(upstream))
    not_exercised = sorted(set(unavailable) - set(sources))
    replay_observed = ["numerical-heat"] if "heat" in ids else []
    findings = [
        finding("Requests to execute or replay unbound provider workflows, or to supply or bypass a binding, are "
                "refused with a named error and reach no provider process, adapter or workflow entry point",
                "computational_pipeline",
                {"requests": len(outcomes), "refused_as_expected": refused,
                 "codes": sorted({row["observed"].split(":", 1)[0] for row in outcomes}),
                 "kinds_refused_unbound": refused_unbound, "kinds_refused_at_upstream_selection": sorted(upstream),
                 "kinds_with_replay_refusal_observed": replay_observed,
                 "unavailable_kinds_not_exercised": not_exercised},
                {"checks": [_refusal(row["case"], row["expected"], row["observed"]) for row in outcomes]
                 + [_check("provider process, adapter or workflow entry points reached", len(provider_reached)),
                    _check("execution entry points reached while reopening", len(reopen_attempts))]},
                uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("A retained numerical-heat bundle whose values no provider computed passes reopen validation",
                "computational_pipeline",
                {"reopen_outcome": fabricated_reopen, "retained_values": retained, "reference_values": reference,
                 "mismatched_cells": mismatched},
                {"checks": [_refusal("Session.from_workspace of a workspace holding the fabricated numerical-heat bundle",
                                     "accepted", fabricated_reopen),
                            _check("cells differing from the independent integer reference", mismatched, 1, "ge",
                                   "invariant")]},
                uncertainty=EXACT_INTEGER, tolerance=EXACT, counterexample={
                    "statement": "Reopen validation of a retained numerical-heat bundle establishes that its values "
                                 "are the pinned SCR heat-kernel output",
                    "witness": {"bundle_id": fabricated["bundles"][0]["bundle_id"], "retained_values": retained,
                                "reference_values": reference,
                                "runtime_repository_root": heat["runtimes"]["scr"]["repository_root"]}}),
        finding("The binding-free energy-accuracy replay succeeds in the same reopened session", "computational_pipeline",
                control_ok, {"checks": [_check("energy replay without numerical match", int(not control_ok))]},
                uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("A content-consistent reopened bundle is acceptable as a verified production result",
                "production_acceptance", "not decided by the workbench", {}),
    ]
    assumptions = ["Replay refusal is observed only for numerical-heat, the one kind with a retained bundle here "
                   "(the fabricated one); every other unavailable kind is exercised through operation.execute. That "
                   "their retained bundles would be refused on replay too is inferred from Workbench.replay calling "
                   "Workbench._reserve(kind), which refuses any kind without a trusted binding, not observed.",
                   "No ESM candidate adapter refusal ('No operator-bound ESM candidate adapter') is reached: that "
                   "path needs a retained telemetry or calibrated-observable bundle, which needs bound providers. The "
                   "ESM case here is refused for its bundle kind.",
                   "The fabricated bundle's runtime identity is syntactically valid but names no real checkout."]
    if not_exercised:
        assumptions.insert(0, "Not exercised (no example source reachable here): " + ", ".join(not_exercised)
                           + ". That they are refused too is inferred from the shared code path (Workbench.execute "
                           "and Workbench.replay call Workbench._reserve, which refuses any kind without a trusted "
                           "binding), not observed.")
    fields = _fields(
        "Without a host-side trusted binding, CIW refuses every replay or execution of a provider workflow with a "
        "named error, and no client or saved value can supply the binding.",
        "Workbench._reserve(kind) requires kind in trusted bindings; bindings are process configuration set only by "
        "Workbench.bind_workflow, never by a protocol request or a saved workspace. Kinds that consume an upstream "
        "bundle are checked for that bundle first.",
        [f"embedded example sources for {', '.join(kind for kind, _ in UNBOUND_SOURCES)}",
         "examples/ sources (via runner.repository_path) for "
         + ", ".join(kind for kind, _ in REPOSITORY_SOURCES + UPSTREAM_SOURCES),
         "fabricated numerical-heat catalog (values [0, 1, 2, 3, 0], fabricated runtime identity)",
         "energy-accuracy session from examples/energy-accuracy/baseline.json (synthetic)"],
        "Protocol v1 error envelopes (code: message) or refused execution records; counted entry points.",
        "Every case refused with its expected code and text; no provider path reached; the builtin still replays.",
        f"{len(outcomes)} refusal cases on a reopened session, each compared with its expected error text; a "
        "counting guard records which execution entry points each request reached.",
        f"{refused}/{len(outcomes)} refused exactly as expected: {len(refused_unbound)} of {len(unavailable)} "
        f"unavailable workflow kinds refused unbound on execute, {len(upstream)} refused at upstream selection, "
        f"{len(not_exercised)} not exercised; replay refusal observed for {replay_observed or 'no kind'} only (inferred "
        f"for the rest); provider paths reached: {provider_reached or 'none'}; the fabricated heat bundle (values "
        f"{retained}, reference field {reference}) on reopen: {fabricated_reopen}.",
        "Exact string and count comparisons.",
        ["client-supplied repositories in replay/execute payloads", "client binding request",
         "unknown bundle", "unversioned and unregistered operation identities", "ESM on a non-telemetry bundle",
         "upstream bundle of the wrong kind", "provider adapter constructed before refusal"],
        assumptions,
        NEXT_STEPS["T092"])
    return {"state": _settle("completed", findings), "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T093

def _unchanged_cases(ids, heat_source, revision):
    """(label, request type, payload, expected 'code: message') of each refused request class."""
    duplicate = base64.b64encode(b'{"schema": "a", "schema": "b"}').decode()
    return [
        ("execute unbound numerical-heat", "operation.execute",
         {"operation_id": "ciw.numerical-heat.v1", "parameters": {"source_id": heat_source}},
         f"operation_unavailable: {UNBOUND}"),
        ("replay an unknown bundle", "bundle.replay", {"bundle_id": "sha256:" + "0" * 64},
         "invalid_payload: Unknown retained workbench bundle"),
        ("client binding request", "workflow.bind", {"kind": "numerical-heat"},
         "unknown_command: Unknown request type: workflow.bind"),
        ("replay with client-supplied repositories", "bundle.replay",
         {"bundle_id": ids["original"], "repositories": {"scr": "/client/scr"}},
         "invalid_payload: Unexpected or missing workbench fields"),
        ("malformed source JSON", "source.add", {"kind": "energy-accuracy", "label": "bad", "bytes_b64": duplicate},
         MALFORMED_TEXT),
        ("non-canonical base64 source", "source.add", {"kind": "energy-accuracy", "label": "bad", "bytes_b64": "e31="},
         "invalid_payload: Source bytes must use bounded canonical base64"),
        ("unknown source kind", "source.add", {"kind": "lab-unknown", "label": "bad", "bytes_b64": "e30="},
         "invalid_payload: Unknown workbench source kind"),
        ("stale selection revision", "selection.update", {"expected_revision": revision + 7, "channel": "q"},
         "revision_conflict: Selection changed; refresh session.get before retrying"),
        ("unknown selection channel", "selection.update", {"expected_revision": revision, "channel": "lab"},
         "invalid_payload: Unknown channel"),
        ("unknown result", "result.get", {"result_id": "result-" + "0" * 32},
         "not_found: Result not found in this session"),
        ("unknown bundle read", "bundle.get", {"bundle_id": "sha256:" + "1" * 64},
         "invalid_payload: Unknown retained workbench bundle"),
        ("energy execute with an upstream bundle", "operation.execute",
         {"operation_id": "ciw.energy-accuracy.v1",
          "parameters": {"source_id": "source:none", "upstream_bundle_id": ids["original"]}},
         "invalid_payload: This workflow does not accept an upstream bundle"),
        ("ESM candidate inspection of a non-telemetry bundle", "operation.execute",
         {"operation_id": "esm.inspect-candidate.v1",
          "parameters": {"bundle_id": ids["original"], "inspected_at": "2026-09-23T00:00:00Z"}},
         "invalid_payload: ESM requires an explicitly selected calibrated or telemetry bundle"),
        ("analysis outside the recording", "analysis.stats", {"interval_s": [5.0, 99.0]},
         "invalid_payload: Require 0 <= start < end <= recording duration"),
        ("workspace save to a client path", "workspace.save", {"path": "/client/workspace.json"},
         "invalid_payload: Unknown payload fields: path"),
    ]


@task("T093", changed_files=(MODULE, FIXTURES),
      regression_tests=(f"{TESTS}::test_t093_refusals_leave_workspace_and_state_unchanged",))
def unchanged_after_refusal(ctx):
    Session = _session_class()
    with tempfile.TemporaryDirectory(prefix="ciw-lab-t093-") as scratch:
        scratch = Path(scratch)
        session, client, ids = build_energy_session(scratch / "session")
        heat = client.ok("source.add", source_payload("numerical-heat",
                                                      example_bytes("declared-workloads/numerical-heat.json"),
                                                      "Unbound numerical-heat source"))
        path = session.save_workspace(scratch / "saved" / "workspace.json")
        before = {"state": _state(session), "directory": _listing(session.output_dir)}
        outcomes = []
        for label, kind, payload, expected in _unchanged_cases(ids, heat["source_id"], session.selection["revision"]):
            outcomes.append({"case": label, "expected": expected, "observed": _outcome(client.call(kind, payload))})
        # A request with an unsupported protocol version never reaches dispatch.
        outcomes.append({"case": "unsupported protocol version",
                         "expected": "unsupported_version: Supported protocol_version is 1",
                         "observed": _outcome(session.handle({"protocol_version": 2, "request_id": "lab-v2",
                                                              "type": "workspace.save", "payload": {}}))})
        after = {"state": _state(session), "directory": _listing(session.output_dir)}
        changed_parts = sorted(name for name in before["state"] if before["state"][name] != after["state"][name])
        second = session.save_workspace(scratch / "saved" / "second.json")
        content_equal = _without_saved_at(path) == _without_saved_at(second)
        # A refused reopen must not create or write its output directory.
        corrupt = json.loads(path.read_text(encoding="utf-8"))
        corrupt["results"][0]["run_id"] = "run-lab-corrupted"
        corrupt_path = scratch / "saved" / "corrupt.json"
        corrupt_path.write_text(json.dumps(corrupt), encoding="utf-8")
        target = scratch / "refused-reopen"
        try:
            Session.from_workspace(corrupt_path, target)
            reopen_refusal = "accepted"
        except ValueError as exc:
            reopen_refusal = str(exc)
        target_created = target.exists()
        # Counterexample: a refused recording operation is retained as an auditable execution record.
        pre = {"state": _state(session), "directory": _listing(session.output_dir)}
        refused_operation = _outcome(client.call("operation.execute", {"operation_id": "ciw.lab-unregistered.v1"}))
        post = {"state": _state(session), "directory": _listing(session.output_dir)}
        recorded = [name for name in pre["state"] if pre["state"][name] != post["state"][name]]
        new_files = sorted(set(post["directory"]) - set(pre["directory"]))
        third = session.save_workspace(scratch / "saved" / "third.json")
        record_resaved_equal = _without_saved_at(path) == _without_saved_at(third)
    refused = sum(row["observed"] == row["expected"] for row in outcomes)
    ctx.artifact_json("unchanged.json", {"cases": outcomes, "state_parts": sorted(before["state"]),
                                         "changed_parts": changed_parts, "reopen_refusal": reopen_refusal,
                                         "refused_operation": refused_operation, "recorded_parts": recorded,
                                         "new_files": new_files})
    findings = [
        finding(f"Each of the {len(outcomes)} exercised non-recording request classes is refused and leaves the "
                "in-memory session state and the session directory unchanged, and a re-save reproduces the saved "
                "workspace content (a refused recording operation is the exception: it is retained as an execution "
                "record)", "computational_pipeline",
                {"refused_as_expected": refused, "requests": len(outcomes), "state_parts_changed": changed_parts,
                 "directory_changed": before["directory"] != after["directory"],
                 "resaved_content_equal": content_equal},
                {"checks": [_refusal(row["case"], row["expected"], row["observed"]) for row in outcomes]
                 + [_check("in-memory state parts changed", len(changed_parts)),
                    _check("session directory listing changed", int(before["directory"] != after["directory"])),
                    _check("re-saved content differs apart from saved_at", int(not content_equal))]},
                uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("A refused reopen of a corrupted workspace creates no output directory", "computational_pipeline",
                {"refusal": reopen_refusal, "output_directory_created": target_created},
                {"checks": [_refusal("Session.from_workspace on a result bound to another run",
                                     "Saved result does not refer to this evidence", reopen_refusal),
                            _check("refused reopen created its output directory", int(target_created))]},
                uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("A refused recording operation is retained as a refused execution record", "computational_pipeline",
                {"outcome": refused_operation, "state_parts_changed": recorded, "new_files": len(new_files),
                 "resaved_workspace_changed": not record_resaved_equal},
                {"checks": [_refusal("operation.execute of an unregistered recording operation",
                                     "operation_unavailable: No trusted provider bound for ciw.lab-unregistered.v1",
                                     refused_operation),
                            _check("state parts changed by the refused operation", len(recorded), 1, "ge", "invariant")]},
                uncertainty=EXACT_COUNT, tolerance=EXACT, counterexample={
                    "statement": "Every refused request leaves the session's in-memory state unchanged",
                    "witness": {"request": "operation.execute ciw.lab-unregistered.v1", "changed": recorded,
                                "new_files": len(new_files)}}),
    ]
    fields = _fields(
        "A refused request other than a recording operation changes neither the session state a save would write nor "
        "the session directory; a refused recording operation is retained by design as an execution record.",
        "State = digests of selection, results, executions, retained catalog, catalog revision, byte and "
        "reservation counters, identity claims, candidates and bindings; plus the session directory listing and "
        "the re-saved workspace content without its saved_at timestamp.",
        ["energy-accuracy session (synthetic baseline) with oscillator results and one unbound numerical-heat source"],
        "Exact 'code: message' of each refusal; SHA-256 digests before and after the refused requests.",
        "Each non-recording request refused with its expected text; all digests equal; re-saving reproduces the "
        "workspace content except its saved_at timestamp; a refused recording operation adds an execution record.",
        f"Send {len(outcomes)} refused requests (unbound execution, forged bindings, malformed sources, stale "
        "revision, unknown identities, out-of-range analysis, save redirection, unsupported version), then compare; "
        "reopen a corrupted copy; finally send one refused recording operation separately.",
        f"{refused}/{len(outcomes)} refused with the expected text; changed state parts: {changed_parts or 'none'}; "
        f"corrupted reopen refused with '{reopen_refusal}'; a refused recording operation changed {recorded}.",
        "Exact digest and string equality.",
        ["reservation or pending counters leaked by a refusal", "partial source retained after refusal",
         "selection revision advanced by a refused update", "save path redirected by a client",
         "refused reopen writing into its target directory", "refusal for an unintended reason (text compared)"],
        ["The unchanged-state result covers the exercised non-recording request classes only. Refused recording "
         "operations are retained by design as auditable execution records (the counterexample finding); this is a "
         "documented exception, not a leak.",
         "The saved workspace file is not rewritten by requests (no request can target it); the evidence is the "
         "state digests and the re-save comparison."],
        NEXT_STEPS["T093"])
    return {"state": _settle("completed", findings), "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T094

# ciw.energy_workflow refuses a retained analysis whose fresh recomputation differs in any bit.
ENERGY_PLATFORM_REFUSAL = "Retained energy analysis binding differs"
# T094's own comparison of a recomputed golden energy analysis with the retained one, per float
# |retained - fresh| <= abs + rel |retained|, every other field exact (energy_recomputation_within).
ENERGY_ROUNDING_TOLERANCE = {"abs": 1e-14, "rel": 1e-14}
ENERGY_ROUNDING_BASIS = (
    "Per-float tolerance 1e-14 absolute plus 1e-14 relative, set from the spread measured with the golden energy "
    "workspace on one x86-64 host under three OpenBLAS kernels: SkylakeX, which wrote it, recomputes it bit for bit; "
    "under Haswell and Sandybridge the reference mean and covariance (order-one results of 2x2 LAPACK solves) move "
    "by at most 2 ulps (2.8e-16 relative) and the error fields derived from them, which cancel to near zero, by at "
    "most 2.2e-16 absolute. The margin is about 45 times the absolute and 36 times the relative spread; this run's "
    "differences are in golden.json.")


def _retained_analyses(raw: bytes) -> dict:
    try:
        return retained_energy_analyses(json.loads(raw))
    except ValueError:  # not JSON: reopen refuses it and the manifest check names it
        return {}


def _reopen_golden(path: Path, target: Path, retained=None) -> dict:
    """Reopen one golden workspace under the execution guard; a refusal is returned as its text.

    With ``retained`` (retained energy analyses by log digest) the energy recomputation is compared to
    ENERGY_ROUNDING_TOLERANCE instead of bit for bit; without it CIW's reopen runs unmodified.
    """
    Session = _session_class()
    session, outcome, compared = None, None, []
    with execution_guard() as guard:
        within = nullcontext() if retained is None else energy_recomputation_within(
            retained, ENERGY_ROUNDING_TOLERANCE, compared)
        try:
            with within:
                session = Session.from_workspace(path, target)
        except (ValueError, ExecutionForbidden) as exc:
            outcome = str(exc)
    return {"outcome": outcome, "session": session, "attempts": list(guard["attempts"]), "compared": compared,
            "bundles": [] if session is None else session.workbench.serialize()["bundles"],
            "recomputations": dict(guard["recomputations"])}


@task("T094", changed_files=(MODULE, FIXTURES, "src/ciw/lab/blas_probe.py", "tests/fixtures/lab/golden"),
      regression_tests=(f"{TESTS}::test_t094_golden_workspaces_reopen_with_current_code",
                        f"{TESTS}::test_golden_manifest_matches_committed_fixtures"))
def golden_retained_bundles(ctx):
    root = fixture_root(ctx.providers)
    if root is None:
        fields = _fields(
            "Golden retained workspaces still reopen and validate with the current code.",
            "Byte identity against GOLDEN_MANIFEST, then Session.from_workspace under the execution guard.",
            [], "None: the golden fixture directory is not reachable from this installation.",
            "Every golden file matches its digest and reopens without execution.",
            "Blocked: tests/fixtures/lab/golden is not present (installed wheel without the source tree) and no "
            "'ciw-fixtures' provider binding was given.",
            "none", "not quantified", [],
            ["Bind the fixtures with --provider ciw-fixtures=<checkout>/tests/fixtures/lab."],
            "Rerun T094 with the ciw-fixtures binding (or from the source checkout).")
        return {"state": "blocked", "fields": fields, "findings": []}
    from ..declared_workload import PINS
    from ..proved_heat import PIN as PROVED_HEAT_PIN
    rows, attempts, failures, digests, compared, unmodified = {}, [], [], {}, [], {}
    reference_mismatch, heat_refusal, heat_values, heat_runtime = None, None, None, None
    with tempfile.TemporaryDirectory(prefix="ciw-lab-t094-") as scratch:
        for index, (name, expected) in enumerate(sorted(GOLDEN_MANIFEST.items())):
            path = root / name
            if not path.is_file():
                failures.append(f"{name}: missing")
                continue
            raw = path.read_bytes()
            digests[name] = sha256(raw).hexdigest()
            # Reopen recomputes the energy analysis (LAPACK solves) and compares it bit for bit, which depends on
            # the platform. The first reopen compares it to rounding tolerance instead, so every other part of
            # reopen validation decides on any platform; the second is CIW's, unmodified.
            opened = _reopen_golden(path, Path(scratch) / f"{index}-rounding", _retained_analyses(raw))
            strict = _reopen_golden(path, Path(scratch) / f"{index}-unmodified")
            attempts += opened["attempts"] + strict["attempts"]
            compared += opened["compared"]
            # Bitwise prediction: CIW's reopen ends as the first, except that a recomputation differing from the
            # retained analysis in any bit is refused by name.
            differs = any(not record["bit_identical"] for record in opened["compared"])
            unmodified[name] = {"outcome": strict["outcome"],
                                "predicted": ENERGY_PLATFORM_REFUSAL if differs else opened["outcome"]}
            if opened["outcome"] is not None:
                failures.append(f"{name}: {opened['outcome']}")
                continue
            session, bundles = opened["session"], opened["bundles"]
            row = {"sha256": digests[name], "bundles": len(bundles), "results": len(session.results),
                   "executions": len(session.executions), "kinds": sorted({b["kind"] for b in bundles})}
            replays = [b for b in bundles if b["native"].get("replay_receipts")]
            for record in replays:
                receipt = record["native"]["replay_receipts"][0]
                original = next(b for b in bundles if b["bundle_id"] == receipt["source_bundle_digest"])
                row["replay_numerical_identity_equal"] = (
                    original["native"]["steps"][0]["numerical_result_id"]
                    == record["native"]["steps"][0]["numerical_result_id"])
            if "energy-accuracy" in row["kinds"]:
                data = bundles[0]["native"]["steps"][0]["result"]["data"]
                row["energy_origin"] = data["origin"]
                row["energy_classification"] = data["comparison"]["classification"]
            if "numerical-heat" in row["kinds"]:
                native = bundles[0]["native"]
                source = json.loads(base64.b64decode(native["source"]["evidence"][0]["bytes_b64"]))
                heat_values = native["steps"][0]["result"]["data"]["values"]
                reference = heat_reference(source["initial_values"], source["steps"])
                reference_mismatch = sum(a != b for a, b in zip(heat_values, reference)) + abs(
                    len(heat_values) - len(reference))
                heat_runtime = native["runtimes"]["scr"]
                heat_refusal = _outcome(Client(session).call("bundle.replay", {"bundle_id": bundles[0]["bundle_id"]}))
            rows[name] = row
    manifest_mismatch = sum(digests.get(name) != expected for name, expected in GOLDEN_MANIFEST.items())
    fingerprint = platform_fingerprint()
    golden_platform = fingerprint == GOLDEN_PLATFORM
    differing = sum(not record["bit_identical"] for record in compared)
    outside = sum(not record["within_tolerance"] for record in compared)
    matched = sum(row["outcome"] == row["predicted"] for row in unmodified.values())
    ctx.artifact_json("golden.json", {"root": "tests/fixtures/lab" if root.name == "lab" else str(root),
                                      "manifest": GOLDEN_MANIFEST, "observed": rows, "failures": failures,
                                      "execution_attempts": attempts, "unmodified_reopen": unmodified,
                                      "platform_refusals": sorted(name for name, row in unmodified.items()
                                                                  if row["outcome"] == ENERGY_PLATFORM_REFUSAL),
                                      "energy_recomputations": compared, "tolerance": ENERGY_ROUNDING_TOLERANCE,
                                      "platform": fingerprint, "golden_platform": GOLDEN_PLATFORM,
                                      "platform_is_golden": golden_platform})
    value = {name: {key: row[key] for key in ("sha256", "bundles", "results", "executions", "kinds")}
             for name, row in rows.items()}
    # CIW's unmodified reopen must end as predicted everywhere; on the platform that wrote the goldens every
    # recomputation must also be bit-identical. Which branch applied is in the check and in golden.json.
    strict_checks = [_refusal(f"unmodified reopen of {name}, predicted from the bitwise comparison of its energy "
                              "recomputations with the retained analysis", row["predicted"] or "none", row["outcome"])
                     for name, row in sorted(unmodified.items())]
    strict_checks.append(_check("energy recomputations compared with the retained analysis", len(compared), 1, "ge"))
    if golden_platform:
        strict_checks.append(_check("energy recomputations differing in any bit on the platform that wrote the "
                                    "goldens (platform_fingerprint() equal to GOLDEN_PLATFORM)", differing))
    first = compared[0] if compared else {}
    findings = [
        finding("The golden retained workspaces reopen and validate with the current code without execution, the "
                "energy recomputation compared to rounding tolerance instead of bit for bit",
                "computational_pipeline", value,
                {"checks": [_check("golden files differing from GOLDEN_MANIFEST", manifest_mismatch),
                            _check("golden workspaces refused or missing, the energy recomputation compared to "
                                   "ENERGY_ROUNDING_TOLERANCE", len(failures)),
                            _check("execution entry points reached while reopening (both reopens)", len(attempts)),
                            _check("replays whose numerical identity differs from their original",
                                   sum(not row.get("replay_numerical_identity_equal", True) for row in rows.values()))]},
                uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("Golden fixture SHA-256 digests", "provenance", dict(sorted(digests.items())),
                {"checks": [_check("digests differing from the recorded manifest", manifest_mismatch)]},
                uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("Unmodified reopen accepts the golden energy analysis only where it recomputes bit for bit, as on "
                "the platform that wrote it, and otherwise refuses it by name as platform-dependent", "numerical",
                {"workspaces_reopened_unmodified": len(unmodified), "outcomes_as_predicted": matched,
                 "refusal_where_a_bit_differs": ENERGY_PLATFORM_REFUSAL},
                {"checks": strict_checks},
                uncertainty={"kind": "roundoff", "value": 0,
                             "basis": "bit-exact canonical JSON comparison, as CIW makes it, of LAPACK-backed "
                                      "recomputations; which of them differ depends on the NumPy/LAPACK build and the "
                                      "OpenBLAS kernel. The platform fingerprint, whether it equals GOLDEN_PLATFORM "
                                      "and each unmodified outcome are in golden.json."},
                tolerance=EXACT),
        finding("The golden energy analysis recomputed with the current code agrees with the retained analysis to "
                "1e-14 absolute plus 1e-14 relative in every float and exactly in every other field", "numerical",
                {"recomputations": len(compared), "float_fields_each": first.get("float_fields", 0),
                 "other_fields_each": first.get("other_fields", 0), "outside_tolerance": outside},
                {"checks": [_check("energy recomputations compared with the retained analysis", len(compared), 1, "ge"),
                            _check("recomputations without a retained analysis of the same log",
                                   sum(not record["retained"] for record in compared)),
                            _check("recomputed floats outside ENERGY_ROUNDING_TOLERANCE",
                                   sum(record.get("floats_outside_tolerance", 0) for record in compared)),
                            _check("recomputed non-float fields or structure differing",
                                   sum(record.get("other_fields_differing", 0) for record in compared))]},
                uncertainty={"kind": "roundoff", "value": ENERGY_ROUNDING_TOLERANCE["abs"],
                             "basis": ENERGY_ROUNDING_BASIS},
                tolerance=EXACT),
    ]
    if heat_values is not None:
        runtime_checks = [
            _check("retained SCR revision differs from ciw.declared_workload.PINS[numerical-heat]",
                   int(heat_runtime["revision"] != PINS["numerical-heat"]["revision"])),
            _check("retained SCR source tree differs from ciw.proved_heat.PIN",
                   int(heat_runtime["source_tree"] != PROVED_HEAT_PIN["source_tree"])),
            _check("retained engine digest differs from the digest recorded with the golden fixtures",
                   int(heat_runtime["engine"]["sha256"] != GOLDEN_SCR_ENGINE_SHA256))]
        findings += [
            finding("The retained golden numerical-heat bundle names the pinned SCR revision, tree and recorded "
                    "engine digest, and its values equal the integer Jacobi reference", "numerical", heat_values,
                    {"checks": runtime_checks + [_check("cells differing from the integer Jacobi reference",
                                                        reference_mismatch)]},
                    uncertainty=EXACT_INTEGER, tolerance=EXACT),
            finding("The retained golden numerical-heat values were computed by the pinned SCR engine", "provenance",
                    "not established by reopen: a fabricated content-consistent bundle also reopens (T092); only a "
                    "bound replay re-executes (T097)", {}, expected_not_established=True),
            finding("Replay of the golden SCR bundle without a binding is refused", "computational_pipeline",
                    heat_refusal, {"checks": [_refusal("bundle.replay in the reopened golden workspace",
                                                       f"operation_unavailable: {UNBOUND}", heat_refusal)]},
                    uncertainty=EXACT_COUNT, tolerance=EXACT),
        ]
    state = _settle("completed" if not failures and not manifest_mismatch else "partial", findings)
    fields = _fields(
        "Workspaces saved by CIW earlier (golden fixtures) still reopen and validate with the current code, with "
        "no provider binding and no execution, on any platform; the bit-for-bit recomputation of the energy analysis "
        "that reopen also requires holds only where the platform rounds as the one that wrote them, and elsewhere "
        "reopen refuses it by name.",
        "Golden = exact saved bytes recorded in GOLDEN_MANIFEST; validation = Session.from_workspace (structure, "
        "identities, commitments, energy-analysis recomputation) under the execution guard, run twice: with the "
        "recomputed energy analysis compared to the retained one per float within |retained - fresh| <= 1e-14 + "
        "1e-14 |retained| (every other field exactly), and unmodified, where CIW compares it bit for bit. Bitwise "
        "prediction: the unmodified reopen ends as the first, except that a recomputation differing from the "
        "retained analysis in any bit is refused as 'Retained energy analysis binding differs'.",
        [f"tests/fixtures/lab/{name}" for name in sorted(GOLDEN_MANIFEST)],
        "SHA-256 of each file; both reopen outcomes; retained replay receipts; each energy recomputation compared "
        "with the retained analysis bit for bit and per float; the platform fingerprint (system, NumPy, BLAS and "
        "LAPACK builds, the OpenBLAS kernel in use, SIMD) against GOLDEN_PLATFORM; retained SCR values and runtime "
        "identity.",
        "All digests match; every workspace reopens with the energy recomputation compared to rounding tolerance, "
        "with zero execution attempts and replay numerical identity preserved; every recomputed float within the "
        "tolerance and every other field equal; every unmodified reopen ends as predicted; on the platform that "
        "wrote the goldens every recomputation is bit-identical; retained heat values equal the integer reference.",
        "Hash each golden file, reopen it into temporary directories under the guard with the tolerant energy "
        "comparison and unmodified, inspect its bundles, compare each energy recomputation with the retained "
        "analysis, compare the retained heat field with the integer reference and its runtime identity with CIW's "
        "pins, and attempt an unbound replay.",
        f"{len(rows)} golden workspaces reopen and validate with the energy recomputation compared to rounding "
        f"tolerance ({len(failures)} failures, {manifest_mismatch} digest mismatches); {len(compared)} energy "
        f"recomputations, {differing} of them differing from the retained analysis in some bit on this platform and "
        f"{outside} outside the tolerance; {matched} of {len(unmodified)} unmodified reopens end as predicted; "
        f"retained heat values {heat_values}.",
        "Exact counts and digests. The energy recomputation is compared to 1e-14 absolute plus 1e-14 relative per "
        "float, about 45 times the absolute and 36 times the relative spread measured across three OpenBLAS "
        "kernels, and bit for bit where CIW compares it.",
        ["fixture drift (digest)", "schema or validator drift breaking reopen", "execution during reopen",
         "replay identity drift", "unbound replay of a provider-backed golden bundle",
         "retained runtime identity differing from CIW's pins", "floating-point recomputation drift beyond rounding",
         "a platform refusal other than the named one, or one where no bit differs",
         "bit-exact recomputation lost on the platform that wrote the goldens"],
        ["Bit-exact reopen of the energy golden is only available on the platform that wrote it (GOLDEN_PLATFORM: "
         "Linux x86_64, NumPy 2.4.3, scipy-openblas 0.3.31.dev running its SkylakeX kernels). CIW compares the "
         "recomputed analysis bit for bit, and a platform whose NumPy/LAPACK build or OpenBLAS kernel rounds "
         "differently refuses the workspace as 'Retained energy analysis binding differs'; forced to the Haswell or "
         "Sandybridge kernels on the same host it does. T094 then shows that the recomputation agrees to rounding "
         "tolerance, which CIW's reopen does not accept. The platform fingerprints and which branch applied are "
         "retained in golden.json.",
         "The rounding tolerance was measured on one x86-64 host under three OpenBLAS kernels; other LAPACK builds, "
         "operating systems and architectures have not been measured. An equal platform fingerprint does not "
         "guarantee equal rounding, and outside NumPy's bundled OpenBLAS the kernel is not named.",
         "The retained runtime identity is metadata written by the saving run: it is compared with CIW's pins, "
         "but reopen cannot show that the named engine produced the values.",
         "The golden numerical-heat workspace records host paths of the machine that produced it; they are "
         "identity metadata, never bindings."],
        NEXT_STEPS["T094"])
    return {"state": state, "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T095

def _attempt(function):
    """Outcome of one validator call; any exception is retained with its type.

    Only a ``ValueError`` (CIW's refusal type) counts as ``refused``; any other
    exception is a crash (``crashed``), never a clean refusal.
    """
    try:
        function()
        return {"refused": False, "crashed": False, "error_type": None, "message": None}
    except ValueError as exc:
        return {"refused": True, "crashed": False, "error_type": type(exc).__name__, "message": str(exc)[:500]}
    except Exception as exc:  # retained, never hidden: an unexpected exception type is itself a finding
        return {"refused": False, "crashed": True, "error_type": type(exc).__name__, "message": str(exc)[:200]}


def _parsed(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError):
        return None


def _malformed_matrix(directory: Path) -> dict:
    """Apply every CIW validator a malformed exchange or workspace file can reach.

    ``exchange_inspect`` runs without a SET checkout: a file that passes every
    parsing check then fails on the missing validator, recorded as not refused.
    """
    from .. import exchange
    from ..instruments import make_demo_run
    from ..session import Session, read_json
    from ..workbench import Workbench
    session = Session(make_demo_run(), directory / "session")
    client = Client(session)
    validator_repo = directory / "no-validator"
    validator_repo.mkdir()
    fields = {exchange.RESULT_SCHEMA: "result_id", exchange.VERIFICATION_SCHEMA: "verification_id"}
    rows = {}
    for index, (name, raw) in enumerate(sorted(malformed_fixtures().items())):
        path = directory / name
        path.write_bytes(raw)
        response = client.call("source.add", source_payload("energy-accuracy", raw, name))
        try:
            exchange.inspect_exchange([path], validator_repo=validator_repo)
            inspected = {"refused": False, "crashed": False, "error_type": None, "message": None}
        except OSError:
            inspected = {"refused": False, "crashed": False, "error_type": "validator_unavailable", "message": None}
        except ValueError as exc:
            inspected = {"refused": True, "crashed": False, "error_type": type(exc).__name__, "message": str(exc)[:500]}
        except Exception as exc:  # a crash is retained, not counted as a refusal
            inspected = {"refused": False, "crashed": True, "error_type": type(exc).__name__, "message": str(exc)[:200]}
        value = _parsed(raw)
        field = fields.get(value.get("schema")) if isinstance(value, dict) else None
        rows[name] = {
            "workbench_source_add": {"refused": response["type"] == "error", "crashed": False, "error_type": None,
                                     "message": _outcome(response) if response["type"] == "error" else None},
            "session_read_json": _attempt(lambda: read_json(path)),
            "exchange_inspect": inspected,
            "exchange_identity": (_attempt(lambda: exchange._identity(value, field)) if field
                                  else {"refused": None, "crashed": None, "error_type": "not_applicable", "message": None}),
            "session_from_workspace": _attempt(lambda: Session.from_workspace(path, directory / f"never-{index}")),
            "workbench_restore": (_attempt(lambda: Workbench.restore(value)) if isinstance(value, dict)
                                  else {"refused": None, "crashed": None, "error_type": "not_applicable", "message": None}),
        }
        if rows[name]["session_read_json"]["error_type"] is None:
            # What read_json made of the bytes it accepted (the overflow witness).
            parsed = read_json(path)
            component = (parsed.get("components") or [{}])[0] if isinstance(parsed, dict) else {}
            rows[name]["session_read_json"]["parsed_component_value"] = repr(component.get("value"))
    return rows


def _runtime_malformed(directory: Path) -> dict:
    """Malformations that are too large to commit or need a full recording."""
    from .. import exchange
    from ..energy_workflow import SOURCE_LIMIT
    from ..instruments import make_demo_run
    from ..session import Session
    rows = {}
    oversize = directory / "oversize-exchange.json"
    oversize.write_bytes(b" " * (exchange.MAX_ARTIFACT_BYTES + 1))
    result = _attempt(lambda: exchange.inspect_exchange([oversize], validator_repo=directory))
    if result["message"]:
        result["message"] = result["message"].replace(str(oversize), "<path>")
    rows["oversize-exchange (1 MiB + 1 byte)"] = result
    session = Session(make_demo_run(), directory / "runtime-session")
    client = Client(session)
    for label, payload in (
            ("oversize-energy-source (4 MiB + 1 byte)",
             source_payload("energy-accuracy", b" " * (SOURCE_LIMIT + 1), "oversize")),
            ("non-canonical base64", {"kind": "energy-accuracy", "label": "x", "bytes_b64": "e31="}),
            ("extra source payload field", {**source_payload("energy-accuracy", b"{}", "x"), "provider": "scr"})):
        response = client.call("source.add", payload)
        rows[label] = {"refused": response["type"] == "error", "crashed": False, "error_type": None,
                       "message": _outcome(response) if response["type"] == "error" else None}
    saved = session.save_workspace(directory / "valid-workspace.json")
    workspace = json.loads(saved.read_text(encoding="utf-8"))
    # Single-defect type mutations of an otherwise valid saved workspace.
    typed = deepcopy(workspace)
    typed["selection"]["revision"] = "0"
    rows["wrong-type selection revision (Session.from_workspace)"] = _attempt(lambda: Session.from_workspace(
        _write(directory / "typed.json", json.dumps(typed).encode()), directory / "never-typed"))
    version = dict(workspace, workspace_version=str(workspace["workspace_version"]))
    rows["wrong-type workspace_version (Session.from_workspace)"] = _attempt(lambda: Session.from_workspace(
        _write(directory / "version.json", json.dumps(version).encode()), directory / "never-version"))
    extra = dict(workspace, lab_unexpected_field={"note": "not part of the workspace schema"})
    extra_path = _write(directory / "extra.json", json.dumps(extra).encode())
    accepted = _attempt(lambda: Session.from_workspace(extra_path, directory / "extra-reopened"))
    dropped = None
    if accepted["error_type"] is None:
        resaved = Session.from_workspace(extra_path, directory / "extra-resaved").save_workspace(directory / "resaved.json")
        dropped = "lab_unexpected_field" not in json.loads(resaved.read_text(encoding="utf-8"))
    rows[EXTRA_FIELD_INPUT] = dict(accepted, dropped_on_resave=dropped)
    return rows


def _write(path: Path, raw: bytes) -> Path:
    path.write_bytes(raw)
    return path


MALFORMED_TEXT = "MALFORMED_RESPONSE: The bound runtime did not return finite, unambiguous JSON"
# The validator each committed fixture targets and its exact CIW-generated text.
MALFORMED_EXPECTED = {
    "duplicate-key.json": ("exchange_inspect", "duplicate JSON member: schema"),
    "nan.json": ("exchange_inspect", "nonfinite JSON number: NaN"),
    "infinity.json": ("exchange_inspect", "nonfinite JSON number: -Infinity"),
    "overflow.json": ("exchange_inspect", "JSON number overflows float64"),
    "truncated-energy-log.json": ("workbench_source_add", MALFORMED_TEXT),
    "invalid-utf8.json": ("exchange_inspect", "exchange input must be bounded UTF-8 JSON"),
    "deep-nesting.json": ("exchange_inspect", "exchange input must be bounded UTF-8 JSON"),
    "wrong-schema-exchange.json": ("exchange_inspect", "unsupported instrument-exchange schema"),
    "wrong-schema-energy-log.json": ("workbench_source_add",
                                     "invalid_payload: Unsupported energy log schema or occurrence identity"),
    "wrong-type-energy-log.json": ("workbench_source_add",
                                   "invalid_payload: Declare actual measurement versus synthetic fixture"),
    "extra-field-energy-log.json": ("workbench_source_add",
                                    "invalid_payload: Require exactly the declared energy-log fields"),
    "wrong-revision-workbench.json": ("workbench_restore", "Malformed retained workbench catalog"),
    "wrong-identity-result.json": ("exchange_identity", "result_id does not match the artifact content"),
}
# A committed fixture whose targeted validator crashes instead of refusing (a counterexample).
CRASH_FIXTURE = ("incomplete-workspace.json", "session_from_workspace")
# The generated input that is accepted instead of refused (a counterexample).
EXTRA_FIELD_INPUT = "extra top-level workspace field (Session.from_workspace)"
RUNTIME_EXPECTED = {
    "oversize-exchange (1 MiB + 1 byte)": "file exceeds 1048576 bytes: <path>",
    "oversize-energy-source (4 MiB + 1 byte)":
        "invalid_payload: Energy analysis source must contain 1..4194304 exact retained bytes",
    "non-canonical base64": "invalid_payload: Source bytes must use bounded canonical base64",
    "extra source payload field": "invalid_payload: Unexpected or missing workbench fields",
    "wrong-type selection revision (Session.from_workspace)": "Invalid saved selection revision",
    "wrong-type workspace_version (Session.from_workspace)": "Unsupported workspace format",
}


@task("T095", changed_files=(MODULE, FIXTURES, "tests/fixtures/lab/malformed"),
      regression_tests=(f"{TESTS}::test_t095_malformed_fixtures_are_refused_with_retained_text",
                        f"{TESTS}::test_committed_malformed_fixtures_match_their_generator"))
def malformed_exchange_fixtures(ctx):
    with tempfile.TemporaryDirectory(prefix="ciw-lab-t095-") as scratch:
        scratch = Path(scratch)
        matrix = _malformed_matrix(scratch)
        runtime = _runtime_malformed(scratch)
    root = fixture_root(ctx.providers)
    committed = None
    if root is not None and (root / "malformed").is_dir():
        generated = malformed_fixtures()
        directory = root / "malformed"
        committed = {"differing": sorted(name for name, raw in generated.items() if (directory / name).is_file()
                                         and (directory / name).read_bytes() != raw),
                     "missing": sorted(name for name in generated if not (directory / name).is_file()),
                     "extra": sorted(path.name for path in directory.iterdir() if path.name not in generated)}
    ctx.artifact_json("malformed-refusals.json", {"committed_fixture_comparison": committed,
                                                  "fixtures": matrix, "runtime": runtime})
    # A refusal counts only as the expected CIW text; a crash or another message does not.
    unrefused = [name for name, (validator, expected) in MALFORMED_EXPECTED.items()
                 if matrix[name][validator]["message"] != expected]
    unrefused += [name for name, expected in RUNTIME_EXPECTED.items() if runtime[name]["message"] != expected]
    checks = [_refusal(f"{name} via {validator}", expected, matrix[name][validator]["message"])
              for name, (validator, expected) in sorted(MALFORMED_EXPECTED.items())]
    checks += [_refusal(name, expected, runtime[name]["message"]) for name, expected in sorted(RUNTIME_EXPECTED.items())]
    overflow = matrix["overflow.json"]
    deep = matrix["deep-nesting.json"]
    crash = matrix[CRASH_FIXTURE[0]][CRASH_FIXTURE[1]]
    extra = runtime[EXTRA_FIELD_INPUT]
    source_text = matrix["duplicate-key.json"]["workbench_source_add"]["message"]
    declared = len(MALFORMED_EXPECTED) + len(RUNTIME_EXPECTED)
    inputs = len(matrix) + len(runtime)
    # Inputs with no declared CIW refusal text; each must be one of the counterexamples below.
    without = sorted(set(matrix) - set(MALFORMED_EXPECTED)) + sorted(set(runtime) - set(RUNTIME_EXPECTED))
    outcomes_without = {CRASH_FIXTURE[0]: f"{CRASH_FIXTURE[1]}: {crash['error_type'] or 'accepted'}",
                        EXTRA_FIELD_INPUT: extra["message"] if extra["error_type"] else "accepted"}
    parsed_overflow = overflow["session_read_json"].get("parsed_component_value")
    findings = [
        finding("Every malformed exchange input with a declared CIW refusal text is refused by the validator it "
                "targets with exactly that text", "computational_pipeline",
                {"inputs": inputs, "with_declared_refusal": declared, "refused_with_declared_text": declared - len(unrefused),
                 "without_declared_refusal": {name: outcomes_without.get(name, "not covered") for name in without}},
                {"checks": [_check("malformed inputs with a declared text not refused with it", len(unrefused)),
                            _check("inputs without a declared refusal that no counterexample finding covers",
                                   sum(name not in outcomes_without for name in without))]
                 + checks}, uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("session.read_json accepts an overflowing number as infinity while the exchange and workbench "
                "parsers refuse it", "computational_pipeline",
                {"session_read_json_refused": overflow["session_read_json"]["refused"],
                 "session_read_json_value": parsed_overflow,
                 "exchange_refused": overflow["exchange_inspect"]["refused"],
                 "workbench_refused": overflow["workbench_source_add"]["refused"]},
                {"checks": [_check("read_json refusals of 1e999", int(overflow["session_read_json"]["refused"])),
                            _check("read_json values of 1e999 other than inf", int(parsed_overflow != repr(float("inf")))),
                            _check("exchange refusals of 1e999", int(overflow["exchange_inspect"]["refused"]), 1,
                                   "ge", "invariant"),
                            _check("workbench refusals of 1e999", int(overflow["workbench_source_add"]["refused"]), 1,
                                   "ge", "invariant")]},
                uncertainty=EXACT_COUNT, tolerance=EXACT, counterexample={
                    "statement": "Every CIW JSON reader refuses nonfinite numbers",
                    "witness": {"fixture": "overflow.json", "reader": "ciw.session.read_json",
                                "parsed_value": parsed_overflow}}),
        finding("session.read_json does not refuse a 20000-deep array with a ValueError", "computational_pipeline",
                "not_refused_with_value_error",
                {"checks": [_refusal("read_json on deep-nesting.json",
                                     "not_refused_with_value_error",
                                     "not_refused_with_value_error"
                                     if deep["session_read_json"]["error_type"] not in ("ValueError", "JSONDecodeError")
                                     else "value_error")]},
                uncertainty=EXACT_COUNT, tolerance=EXACT, counterexample={
                    "statement": "session.read_json refuses every malformed workspace file with a ValueError",
                    "witness": {"fixture": "deep-nesting.json",
                                "observed": deep["session_read_json"]["error_type"] or "accepted"}}),
        finding("Session.from_workspace raises AttributeError, not a ValueError refusal, on a workspace holding only "
                "its version", "computational_pipeline", crash["error_type"] or "accepted",
                {"checks": [_refusal("Session.from_workspace on incomplete-workspace.json", "AttributeError",
                                     crash["error_type"] or "accepted")]},
                uncertainty=EXACT_COUNT, tolerance=EXACT, counterexample={
                    "statement": "Session.from_workspace refuses every malformed workspace with a ValueError",
                    "witness": {"fixture": CRASH_FIXTURE[0], "content": {"workspace_version": 3},
                                "error_type": crash["error_type"]}}),
        finding("Session.from_workspace accepts an unknown top-level field and drops it on re-save",
                "computational_pipeline", {"accepted": not extra["refused"], "dropped_on_resave": extra["dropped_on_resave"]},
                {"checks": [_check("extra-field workspaces refused", int(extra["refused"])),
                            _check("extra field kept after re-save", int(extra["dropped_on_resave"] is not True))]},
                uncertainty=EXACT_COUNT, tolerance=EXACT, counterexample={
                    "statement": "Saved workspaces with fields outside the schema are refused on reopen",
                    "witness": {"field": "lab_unexpected_field", "reopened": not extra["refused"],
                                "dropped_on_resave": extra["dropped_on_resave"]}}),
        (finding("The committed malformed fixtures equal their generator's bytes", "provenance",
                 {"fixtures": len(malformed_fixtures()), **committed},
                 {"checks": [_check("committed malformed fixtures differing from, missing from or extra to the generator",
                                    sum(map(len, committed.values())))]},
                 uncertainty=EXACT_COUNT, tolerance=EXACT) if committed is not None else
         finding("The committed malformed fixtures equal their generator's bytes", "provenance",
                 "not compared: tests/fixtures/lab/malformed is not reachable from this installation", {},
                 expected_not_established=True)),
        finding("Workbench source.add reports malformed source JSON with text that names a bound runtime",
                "computational_pipeline", source_text,
                {"checks": [_refusal("source.add of duplicate-key.json", MALFORMED_TEXT, source_text)]},
                uncertainty=EXACT_COUNT, tolerance=EXACT, counterexample={
                    "statement": "Workbench source errors name the source, not a bound runtime",
                    "witness": {"fixture": "duplicate-key.json", "request": "source.add (energy-accuracy)",
                                "message": source_text}}),
    ]
    fields = _fields(
        "Each malformed exchange input (duplicate keys, NaN/Infinity, overflow, wrong schema, oversize, truncated "
        "bytes, invalid UTF-8, wrong types, extra fields, forged identity) is refused by the relevant CIW validator.",
        "Validators: Workbench source.add (energy-accuracy), ciw.session.read_json, ciw.exchange.inspect_exchange "
        "(parsing stage, no SET checkout), ciw.exchange._identity, Session.from_workspace, Workbench.restore.",
        [f"tests/fixtures/lab/malformed/{name}" for name in sorted(malformed_fixtures())]
        + sorted(RUNTIME_EXPECTED) + ["extra top-level workspace field (generated)"],
        "Refused (ValueError), crashed (any other exception) or accepted, exception type and exact message per "
        "validator (retained in malformed-refusals.json).",
        "Every input with a declared CIW refusal text refused by the validator it targets with that text; wrong-type "
        "cases are single-field mutations of a valid input so the refusal is attributable; the committed fixtures "
        "equal their generator's bytes.",
        "Write each committed fixture from its generator, apply the byte-level validators to each, apply the "
        "structural validators to the runtime-generated cases, and compare exact CIW messages.",
        f"{declared - len(unrefused)}/{inputs} malformed inputs refused with their declared CIW text ({declared} have "
        f"one); the other {len(without)} are counterexamples ({'; '.join(f'{k}: {v}' for k, v in outcomes_without.items())}); "
        f"read_json accepted 1e999 as {parsed_overflow} and "
        f"{'raised ' + str(deep['session_read_json']['error_type']) if deep['session_read_json']['error_type'] else 'accepted'}"
        f" on 20000-deep nesting; the extra top-level workspace field was "
        f"{'dropped' if extra['dropped_on_resave'] else 'kept'} on re-save; committed fixtures "
        f"{'match their generator' if committed is not None and not sum(map(len, committed.values())) else 'not compared' if committed is None else 'differ from their generator'}.",
        "Exact string comparison of CIW-generated text; messages from Python's json module and interpreter "
        "exceptions are retained but not compared, because they vary between interpreter versions.",
        ["duplicate keys", "NaN and ±Infinity literals", "float overflow", "truncation", "invalid UTF-8",
         "recursion depth", "wrong schema", "wrong types (single-field mutations)", "incomplete workspace",
         "extra fields", "oversize bytes", "forged identity", "non-canonical base64"],
        ["The workbench validator is exercised through the energy-accuracy kind; other kinds apply their own "
         "source parsers after the same canonical-base64 and byte-budget checks.",
         "The exchange stage stops before the SET validator; conformance of well-formed artifacts needs SET (T097)."],
        NEXT_STEPS["T095"])
    return {"state": _settle("completed", findings), "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T096

def _random_value(generator, depth=0):
    kind = int(generator.integers(0, 6 if depth < 2 else 4))
    if kind == 0:
        return int(generator.integers(-10 ** 6, 10 ** 6))
    if kind == 1:
        return float(generator.normal()) * 10.0 ** int(generator.integers(-3, 4))
    if kind == 2:
        alphabet = "abcxyz é–ΩéÅ0123"
        return "".join(alphabet[int(i)] for i in generator.integers(0, len(alphabet), int(generator.integers(0, 8))))
    if kind == 3:
        return bool(generator.integers(0, 2))
    if kind == 4:
        return [_random_value(generator, depth + 1) for _ in range(int(generator.integers(0, 4)))]
    return {f"k{int(i)}": _random_value(generator, depth + 1) for i in generator.integers(0, 50, int(generator.integers(1, 4)))}


def _mutated(value, generator):
    """A different JSON value of the same broad type."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + (abs(value) or 1.0) * 1e-9
    if isinstance(value, str):
        return value + "·"
    if isinstance(value, list):
        return value + [0]
    return {**value, "lab_mutation": True}


IDENTITY_SEED, IDENTITY_COUNT = 9601, 24
# The deterministic, unseeded builder of T096's synthetic ESM inspect and capture responses.
CANDIDATE_GENERATOR = "ciw.lab.exchange_provenance_bundles._candidate_cases"


def _identity_study(seed=IDENTITY_SEED, count=IDENTITY_COUNT):
    """exchange._identity over records sealed by the lab-written canonical encoder.

    ``canonical_text`` is written from the producer specification (sorted
    members, no spaces, literal non-ASCII, JSON escapes, shortest float repr)
    without calling ``json.dumps``, and the study first checks that it agrees
    byte for byte with the ``json.dumps`` call ``exchange._identity`` uses. Both
    are CIW-side code, so acceptance of valid records follows from that
    agreement; it is not an independent check of ``_identity``.
    """
    import numpy as np
    from .. import exchange
    from .exchange_provenance_bundles_fixtures import canonical_text, seal_independently
    generator = np.random.Generator(np.random.PCG64(seed))
    valid = accepted = mutated = refused = reordered_accepted = batch_mutations = batch_accepted = 0
    encoder_agreement = 0
    for index in range(count):
        schema, field = ((exchange.RESULT_SCHEMA, "result_id") if index % 2 == 0
                         else (exchange.VERIFICATION_SCHEMA, "verification_id"))
        body = {"schema": schema, **{f"field_{j}": _random_value(generator) for j in range(int(generator.integers(2, 6)))}}
        artifact = seal_independently(body, field)
        valid += 1
        encoder_agreement += canonical_text(body) == json.dumps(body, sort_keys=True, separators=(",", ":"),
                                                                ensure_ascii=False, allow_nan=False)
        accepted += exchange._identity(artifact, field) == "content_recomputed_not_authenticated"
        # Member order does not change the canonical identity.
        reordered = dict(reversed(list(artifact.items())))
        reordered_accepted += exchange._identity(reordered, field) == "content_recomputed_not_authenticated"
        for key in sorted(artifact):
            if key in (field, "schema"):
                continue
            changed = dict(artifact, **{key: _mutated(artifact[key], generator)})
            mutated += 1
            try:
                exchange._identity(changed, field)
            except ValueError:
                refused += 1
        forged = dict(artifact, **{field: artifact[field][:-1] + ("0" if artifact[field][-1] != "0" else "1")})
        mutated += 1
        try:
            exchange._identity(forged, field)
        except ValueError:
            refused += 1
        batch = {"schema": exchange.OBSERVATION_SCHEMA, "batch_id": f"example:lab-{index}", "value": body.get("field_0")}
        changed_batch = dict(batch, value=_mutated(batch["value"], generator))
        batch_mutations += 1
        batch_accepted += exchange._identity(changed_batch, "batch_id") == "caller_declared_reference"
    return {"valid": valid, "accepted": accepted, "reordered_accepted": reordered_accepted,
            "encoder_agreement": encoder_agreement, "mutations": mutated, "refused": refused,
            "batch_mutations": batch_mutations, "batch_mutations_accepted": batch_accepted}


def _telemetry_bundle():
    steps = [{"runtime_ref": role, "execution_id": f"execution-{role}"} for role in ("ppda", "stfe", "gsie", "set")]
    steps.append({"runtime_ref": "cbsr", "execution_id": "execution-cbsr", "result": {"status": "consistent"}})
    return {"schema": "ciw.telemetry-session.v1", "bundle_digest": "sha256:" + "ab" * 32, "steps": steps}


def _calibrated_bundle():
    data = {"fdir": {"residual_basis": "normalized", "detection": {"status": "no_fault"},
                     "isolability": {"status": "not_isolable", "cross_covariance_policy": "declared_zero",
                                     "isolated_fault": None}}}
    steps = [{"runtime_ref": "gsie", "execution_id": "execution-gsie", "result_id": "result-gsie",
              "result": {"data": {"state_id": "state-1"}}},
             {"runtime_ref": "oit", "execution_id": "execution-oit", "result_id": "result-oit",
              "result": {"data": {"status": "observable"}}},
             {"runtime_ref": "cbsr", "execution_id": "execution-cbsr", "result_id": "result-cbsr",
              "result": {"data": {"status": "consistent"}}},
             {"runtime_ref": "fdir", "execution_id": "execution-fdir", "result_id": "result-fdir",
              "result": {"data": data["fdir"]}}]
    return {"schema": "ciw.calibrated-observable-session.v1", "bundle_digest": "sha256:" + "cd" * 32, "steps": steps}


def _candidate_cases():
    """(label, action, bundle, response, parameters, policy, should_accept) for validate_response."""
    from ..telemetry import canonical
    context = {"requestId": "lab-request-1", "authority": "lab-reviewer"}
    registration = {"sourceId": "lab-derived-source"}
    cases = []
    for bundle in (_telemetry_bundle(), _calibrated_bundle()):
        raw = canonical(bundle)
        steps = {step["runtime_ref"]: step for step in bundle["steps"]}
        calibrated = bundle["schema"].startswith("ciw.calibrated")
        candidate = {"state": "UNADMITTED", "bundleDigest": bundle["bundle_digest"],
                     "executionIds": sorted(step["execution_id"] for step in bundle["steps"]),
                     "verification": {"outcome": "passed", "independent": False},
                     "reconciliation": {"status": (steps["cbsr"]["result"]["data"] if calibrated
                                                   else steps["cbsr"]["result"])["status"]}}
        if calibrated:
            faults = steps["fdir"]["result"]["data"]
            candidate["processAssessment"] = {
                "stateResultId": "result-gsie", "stateId": "state-1", "observabilityResultId": "result-oit",
                "observabilityStatus": "observable", "reconciliationResultId": "result-cbsr",
                "faultResultId": "result-fdir", "residualBasis": faults["residual_basis"],
                "detectionStatus": faults["detection"]["status"], "isolabilityStatus": faults["isolability"]["status"],
                "crossCovariancePolicy": faults["isolability"]["cross_covariance_policy"],
                "isolatedFault": faults["isolability"]["isolated_fault"]}

        def inspection(at):
            return {"schema": "payload.instrument-candidate-inspection.v1", "inspectedAt": at,
                    "requestId": context["requestId"], "authority": context["authority"],
                    "state": "ELIGIBLE_FOR_CANDIDATE_REVIEW", "bundleBytesDigest": "sha256:" + sha256(raw).hexdigest(),
                    "canonicalAdmission": "REFUSED", "canonicalStateMutated": False, "evidenceRetained": False,
                    "releaseActivated": False, "sourceTruthClaimed": False, "independentlyVerified": False,
                    "retractionHistoryComplete": False, "candidate": deepcopy(candidate)}
        inspect_parameters = {"bundle_id": bundle["bundle_digest"], "inspected_at": "2026-09-23T00:00:00Z"}
        base = inspection(inspect_parameters["inspected_at"])
        name = "calibrated" if calibrated else "telemetry"
        cases.append((f"{name} inspect valid", "inspect", bundle, base, inspect_parameters, {"review_context": context}, True))
        refused_state = dict(base, state="REFUSED", candidate=None)
        cases.append((f"{name} inspect refused state valid", "inspect", bundle, refused_state, inspect_parameters,
                      {"review_context": context}, True))
        mutations = {
            "canonical admission granted": lambda r: r.update(canonicalAdmission="ADMITTED"),
            "canonical state mutated": lambda r: r.update(canonicalStateMutated=True),
            "release activated": lambda r: r.update(releaseActivated=True),
            "source truth claimed": lambda r: r.update(sourceTruthClaimed=True),
            "evidence retained on inspect": lambda r: r.update(evidenceRetained=True),
            "independently verified": lambda r: r.update(independentlyVerified=True),
            "retraction history complete": lambda r: r.update(retractionHistoryComplete=True),
            "schema changed": lambda r: r.update(schema="payload.instrument-candidate-inspection.v2"),
            "inspection time changed": lambda r: r.update(inspectedAt="2026-09-24T00:00:00Z"),
            "request identity changed": lambda r: r.update(requestId="other"),
            "authority changed": lambda r: r.update(authority="other"),
            "unknown state": lambda r: r.update(state="ADMITTED"),
            "bundle bytes digest changed": lambda r: r.update(bundleBytesDigest="sha256:" + "0" * 64),
            "eligible without candidate": lambda r: r.update(candidate=None),
            "candidate admitted": lambda r: r["candidate"].update(state="ADMITTED"),
            "candidate bundle changed": lambda r: r["candidate"].update(bundleDigest="sha256:" + "1" * 64),
            "candidate executions changed": lambda r: r["candidate"].update(executionIds=["execution-other"]),
            "candidate verification failed": lambda r: r["candidate"]["verification"].update(outcome="failed"),
            "candidate verification independent": lambda r: r["candidate"]["verification"].update(independent=True),
            "candidate reconciliation changed": lambda r: r["candidate"]["reconciliation"].update(status="inconsistent"),
        }
        if calibrated:
            mutations["process assessment changed"] = lambda r: r["candidate"]["processAssessment"].update(stateId="other")
        else:
            mutations["telemetry inherits process assessment"] = lambda r: r["candidate"].update(processAssessment={})
        for label, change in mutations.items():
            response = deepcopy(base)
            change(response)
            cases.append((f"{name} inspect {label}", "inspect", bundle, response, inspect_parameters,
                          {"review_context": context}, False))
        capture_parameters = {"bundle_id": bundle["bundle_digest"], "evidence_id": "evidence-lab-1",
                              "workflow_id": "workflow-lab-1", "retained_at": "2026-09-23T01:00:00Z"}
        capture_policy = {"review_context": context, "capture_registration": registration}
        capture = {"canonicalAdmission": "REFUSED", "canonicalStateMutated": False, "releaseActivated": False,
                   "sourceTruthClaimed": False, "state": "CANDIDATE_EVIDENCE_RETAINED",
                   "inspection": inspection(capture_parameters["retained_at"]),
                   "capture": {"evidence": {"evidenceId": "evidence-lab-1", "capturedAt": "2026-09-23T01:00:00Z",
                                            "sourceId": "lab-derived-source", "sourceTruthClaimed": False,
                                            "contentDigest": "sha256:" + "ef" * 32, "storageKey": "store/lab-1"},
                               "receipt": {"evidenceId": "evidence-lab-1", "receiptId": "workflow-lab-1:receipt",
                                           "storedAt": "2026-09-23T01:00:00Z", "contentDigest": "sha256:" + "ef" * 32,
                                           "storageKey": "store/lab-1"}}}
        cases.append((f"{name} capture valid", "capture", bundle, capture, capture_parameters, capture_policy, True))
        capture_mutations = {
            "evidence identity differs": lambda r: r["capture"]["evidence"].update(evidenceId="evidence-other"),
            "receipt identity differs": lambda r: r["capture"]["receipt"].update(receiptId="other:receipt"),
            "capture time differs": lambda r: r["capture"]["evidence"].update(capturedAt="2026-09-23T02:00:00Z"),
            "storage time differs": lambda r: r["capture"]["receipt"].update(storedAt="2026-09-23T02:00:00Z"),
            "derived source differs": lambda r: r["capture"]["evidence"].update(sourceId="other"),
            "content digest differs": lambda r: r["capture"]["receipt"].update(contentDigest="sha256:" + "0" * 64),
            "storage key differs": lambda r: r["capture"]["receipt"].update(storageKey="store/other"),
            "capture claims source truth": lambda r: r["capture"]["evidence"].update(sourceTruthClaimed=True),
            "inspection not eligible": lambda r: r["inspection"].update(state="REFUSED"),
            "refused capture with receipt": lambda r: r.update(state="REFUSED"),
            "unknown capture state": lambda r: r.update(state="ADMITTED"),
        }
        for label, change in capture_mutations.items():
            response = deepcopy(capture)
            change(response)
            cases.append((f"{name} capture {label}", "capture", bundle, response, capture_parameters,
                          capture_policy, False))
        extra = deepcopy(base)
        extra["lab_admission_override"] = True
        extra["candidate"]["lab_admitted"] = True
        cases.append((f"{name} inspect unknown extra fields", "inspect", bundle, extra, inspect_parameters,
                      {"review_context": context}, None))
    return cases


def _candidate_study():
    from ..candidate_evidence import validate_response
    from ..telemetry import canonical
    rows = []
    for label, action, bundle, response, parameters, policy, expected in _candidate_cases():
        try:
            validate_response(response, action, bundle, canonical(bundle), parameters, policy)
            outcome = "accepted"
        except ValueError as exc:
            outcome = "refused: " + str(exc)
        rows.append({"case": label, "expected": {True: "accepted", False: "refused", None: "extra"}[expected],
                     "observed": outcome})
    return rows


@task("T096", changed_files=(MODULE, FIXTURES),
      regression_tests=(f"{TESTS}::test_t096_provider_free_conformance",))
def provider_free_conformance(ctx):
    identity = _identity_study()
    candidates = _candidate_study()
    ctx.artifact_json("identity-study.json", identity)
    ctx.artifact_json("candidate-study.json", candidates)
    valid = [row for row in candidates if row["expected"] == "accepted"]
    mutated = [row for row in candidates if row["expected"] == "refused"]
    extras = [row for row in candidates if row["expected"] == "extra"]
    wrong_valid = sum(row["observed"] != "accepted" for row in valid)
    wrong_mutated = sum(not row["observed"].startswith("refused") for row in mutated)
    extras_accepted = sum(row["observed"] == "accepted" for row in extras)
    # Every finding here rests on synthetic inputs and declares their generator, so its Basis column names it.
    records = {"name": "ciw.lab seeded nested JSON records", "seed": IDENTITY_SEED, "count": IDENTITY_COUNT}
    responses = {"name": CANDIDATE_GENERATOR, "count": len(candidates)}
    findings = [
        finding("exchange._identity accepts every synthetic result and verification record sealed by a lab-written "
                "canonical encoder that agrees byte for byte with json.dumps, in any member order, and refuses every "
                "single-field mutation and a forged identity", "computational_pipeline",
                {key: identity[key] for key in ("valid", "accepted", "reordered_accepted", "encoder_agreement",
                                                "mutations", "refused")},
                {"generator": records,
                 "checks": [_check("records whose lab canonical text differs from json.dumps canonical text",
                                   identity["valid"] - identity["encoder_agreement"]),
                            _check("valid records not accepted", identity["valid"] - identity["accepted"]),
                            _check("member-reordered records not accepted",
                                   identity["valid"] - identity["reordered_accepted"]),
                            _check("mutations not refused", identity["mutations"] - identity["refused"])]},
                uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("candidate_evidence.validate_response accepts valid synthetic ESM responses and refuses every "
                "boundary mutation", "computational_pipeline",
                {"valid": len(valid), "valid_accepted": len(valid) - wrong_valid, "mutations": len(mutated),
                 "mutations_refused": len(mutated) - wrong_mutated},
                {"generator": responses,
                 "checks": [_check("valid synthetic responses refused", wrong_valid),
                            _check("boundary mutations accepted", wrong_mutated)]},
                uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("Observation-batch identities are caller-declared: mutated batches pass exchange._identity",
                "computational_pipeline",
                {"mutations": identity["batch_mutations"], "accepted": identity["batch_mutations_accepted"]},
                {"generator": records,
                 "checks": [_check("mutated batches accepted", identity["batch_mutations_accepted"], 1, "ge",
                                   "invariant")]},
                uncertainty=EXACT_COUNT, tolerance=EXACT, counterexample={
                    "statement": "Every exchange artifact identity is bound to its content",
                    "witness": {"schema": "notation.instrument.observation-batch.v1",
                                "identity_status": "caller_declared_reference"}}),
        finding("validate_response accepts unknown extra fields in an ESM response", "computational_pipeline",
                {"cases": len(extras), "accepted": extras_accepted},
                {"generator": responses,
                 "checks": [_check("responses with unknown fields accepted", extras_accepted, 1, "ge", "invariant")]},
                uncertainty=EXACT_COUNT, tolerance=EXACT, counterexample={
                    "statement": "The CIW candidate boundary refuses any ESM response field it does not recognise",
                    "witness": {"fields": ["lab_admission_override", "candidate.lab_admitted"],
                                "reason": "the boundary checks bindings only; ESM owns its record schema"}}),
        finding("Passing provider-free conformance admits a candidate into canonical state", "production_acceptance",
                "not decided by the workbench", {}),
    ]
    fields = _fields(
        "The provider-free parts of the exchange and ESM candidate boundaries accept valid records and refuse "
        "every mutation of a bound field, without any provider checkout.",
        "exchange._identity: id = 'sha256:' + SHA-256(schema NUL canonical_json(record without id)); "
        "validate_response: equality of boundary fields with the explicit request and the selected bundle.",
        [f"{IDENTITY_COUNT} seeded synthetic result/verification records (PCG64 seed {IDENTITY_SEED})",
         f"{len(candidates)} synthetic ESM inspect/capture responses over telemetry and calibrated bundle stubs "
         f"(deterministic, unseeded: {CANDIDATE_GENERATOR})"],
        "Accepted or refused (with message) per record or response.",
        "All valid accepted, all single-field mutations refused, identity invariant under member order.",
        "Seal records with the lab-written canonical encoder (checked byte for byte against the json.dumps call "
        "exchange._identity uses), re-validate them with members reordered, mutate each non-identity field and the "
        "identity itself; build valid ESM responses and apply every boundary mutation.",
        f"{identity['refused']}/{identity['mutations']} identity mutations refused; "
        f"{len(mutated) - wrong_mutated}/{len(mutated)} candidate mutations refused; {extras_accepted} response(s) "
        "with unknown fields accepted.",
        "Exact.",
        ["member-order dependence", "identity forgery", "admission/state/truth flags",
         "binding to request, time, digest and bundle", "capture receipt consistency"],
        ["Acceptance of valid records follows from the lab encoder's byte-for-byte agreement with the json.dumps "
         "call exchange._identity uses (both CIW-side code, same origin), so it is not independent evidence; "
         "producer-sealed artifacts are tested only in T097's PPDA/SCR/SET roundtrip, when those checkouts are bound.",
         "JSON escaping (for example \\u00e9 for é) is resolved by the JSON parser before _identity sees a "
         "record, so it is not an identity property and is not tested here.",
         "The bundle stubs contain only the fields validate_response reads; full native bundles need providers.",
         "Mutations are single-field; combined mutations that restore consistency are not explored."],
        NEXT_STEPS["T096"])
    return {"state": _settle("completed", findings), "fields": fields, "findings": findings}


# ---------------------------------------------------------------- engines

def _locked_build(ctx) -> dict:
    """The run's shared ``cargo build --locked --offline`` of SCR execution-cli (built once, reused by T097-T099)."""
    return ctx.memo(("exchange-bundles:scr-build", str(ctx.providers["scr"])),
                    lambda: providers.build_engine(ctx.providers["scr"]))


def _engine(ctx) -> dict | None:
    """SCR engine bytes: an operator binding, else the run's locked offline build."""
    bound = ctx.providers.get("scr-engine")
    if bound is not None and Path(bound).is_file():
        data = Path(bound).read_bytes()
        return {"origin": "operator_bound_scr_engine", "binary": data, "sha256": sha256(data).hexdigest(), "build": None}
    if "scr" in ctx.providers and ctx.available("tool:cargo"):
        build = _locked_build(ctx)
        if build["binary"] is not None:
            return {"origin": "cargo_build_locked_offline_in_this_run", "binary": build["binary"],
                    "sha256": sha256(build["binary"]).hexdigest(), "build": build}
    return None


def _scr_identity(ctx):
    """(identity, None) of the bound SCR checkout, or (None, error type) when it is not a readable repository root."""
    def read():
        try:
            return providers.checkout_identity(ctx.providers["scr"]), None
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            return None, type(exc).__name__
    return ctx.memo(("exchange-bundles:identity", str(ctx.providers["scr"])), read)


def _refused_checkout(role, identity, comparison) -> str:
    matched = ", ".join(comparison["matched"]) or "no CIW pin"
    return (f"the bound {role} checkout is refused: HEAD {identity['head']} matches {matched}; "
            f"clean={comparison['clean']}; pinned-tree refusals: {', '.join(comparison['tree_refusals']) or 'none'}")


def _engine_note(engine) -> dict:
    """Where the engine bytes came from; CIW itself records a bound engine as operator_asserted_not_attested."""
    return {"engine_origin": engine["origin"], "engine_sha256": engine["sha256"],
            "engine_source_binding": ("operator_asserted_not_attested" if engine["build"] is None
                                      else "built_from_the_bound_checkout_in_this_run")}


def _provider_basis(identity, engine):
    """The executed SCR checkout and engine, declared by every finding that rests on the SCR run (even where a
    check decides its label), so the Basis column names the provider."""
    return {"provider": {"repository": providers.REPOSITORIES["scr"], "revision": identity["head"],
                         "source_tree": identity["tree"], "runtime_digest": "sha256:" + engine["sha256"],
                         "executed": True},
            "notes": _engine_note(engine)}


ENGINE_MATCH = "The operator-bound SCR engine is byte-identical to a locked offline build of the pinned checkout"


def _engine_origin_findings(ctx, engine) -> list:
    """Cross-check an operator-bound engine against the run's locked build, or record that it was not checked."""
    if engine is None or engine["build"] is not None:
        return []
    if not ctx.available("tool:cargo"):
        return [finding(ENGINE_MATCH, "provenance", "not compared: cargo is unavailable, so the engine digest is "
                        "operator-asserted", {"notes": _engine_note(engine)}, expected_not_established=True)]
    build = _locked_build(ctx)
    digests = sorted({row["binary_sha256"] for row in build["builds"] if row["binary_sha256"]})
    return [finding(ENGINE_MATCH, "provenance",
                    {"locked_build_digests": len(digests), "equal_to_bound_engine": digests == [engine["sha256"]]},
                    {"checks": [_check("locked builds that produced a binary", len(digests), 1, "ge", "invariant"),
                                _check("locked-build digests differing from the operator-bound engine",
                                       sum(digest != engine["sha256"] for digest in digests))],
                     "notes": {**_engine_note(engine), "locked_build_sha256": digests}},
                    uncertainty=EXACT_COUNT, tolerance=EXACT)]


SURVEY_CASES = [[3, [0, 0, 1000, 0, 0]], [2, [0, 0, 64, 0, 0]], [1, [0, -3, 0]], [1, [7, -3, 5]]]

_T097_PLAN = _fields(
    "The pinned SCR provider, driven through CIW's shared numerical-heat workflow, returns the declared integer "
    "heat field, replays with the same numerical identity, and its reopened workspace refuses replay unbound.",
    "u_i <- u_i + trunc((u_{i-1} - 2 u_i + u_{i+1}) / 4), fixed ends (SCR heat descriptor, ciw.declared_workload).",
    ["examples/declared-workloads/numerical-heat.json", "survey source [0, 0, 1000, 0, 0], 3 steps"],
    "Workbench bundle values, replay receipts, SCR Python API output.",
    "Provider values equal the independent integer reference; replay numerical identity equal with fresh "
    "occurrences.",
    "Bind --provider scr=<Scientific-Computation-Runtime at a59aba283b0304faeeb3e5d305087e7709e171ca> (and "
    "optionally scr-engine=<execution-cli>, set=<SET at 542e672be512bf43b61253f2b2a43cd967cb3062>, "
    "ppda=<PPDA at a29845e13e55de30b24ae752b896058041d653e6>, scr-exchange=<SCR at "
    "5f0409743e0098a0691a88302a9b3dcdcbcf25fd>); equivalent pytest gate: CIW_DECLARED_STACK_ROOT=<root with "
    "sra/ and scr/> CIW_SCR_ENGINE=<execution-cli> python -m pytest -q tests/test_declared_workloads.py.",
    "none", "not quantified",
    ["provider unavailable"],
    ["The SCR checkout is not bound in this run."],
    "Provision the pinned SCR checkout (scripts/check_lab.py does) and rerun T097.")
_T097_PLAN["findings"] = [finding("The integer heat field describes physical heat diffusion in a material", "physical",
                                  "not established: dimensionless integer arithmetic", {})]
T097_FILES = (MODULE, FIXTURES, PROVIDERS)
# The CIW pin each optional exchange checkout must satisfy.
EXCHANGE_REQUIRED = {"set": "ciw/exchange-runtime.json", "ppda": ".github/workflows/exchange.yml",
                     "scr-exchange": ".github/workflows/exchange.yml"}


def _optional_checkout(ctx, role, pins) -> dict:
    """State of an optional exchange checkout: not_bound, unreadable, refused, off_pin or ready (with a reason)."""
    pin = providers.EXCHANGE_WORKFLOW_PINS[role]
    name = providers.REPOSITORIES[role].split("/")[1]
    if role not in ctx.providers:
        return {"state": "not_bound", "reason": f"no {role} checkout bound (needs {name} at {pin})"}
    try:
        identity = providers.checkout_identity(ctx.providers[role])
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        return {"state": "unreadable",
                "reason": f"the bound {role} path is not a readable Git repository root ({type(exc).__name__})"}
    comparison = providers.compare_with_pins(role, identity, pins)
    record = {"head": identity["head"], "tree": identity["tree"], "matched": comparison["matched"],
              "clean": comparison["clean"]}
    if not comparison["accepted"]:
        return dict(record, state="refused", reason=_refused_checkout(role, identity, comparison))
    if EXCHANGE_REQUIRED[role] not in comparison["matched"]:
        return dict(record, state="off_pin", reason=f"the bound {role} checkout is at {', '.join(comparison['matched'])}, "
                                                    f"not the {EXCHANGE_REQUIRED[role]} pin {pin}")
    return dict(record, state="ready", reason=None)


def _executed_checkout(role, checkout) -> dict:
    """The provider basis of a bound, ready exchange checkout that executed: repository, HEAD and tree."""
    return {"repository": providers.REPOSITORIES[role], "revision": checkout["head"], "source_tree": checkout["tree"],
            "executed": True}


def _set_findings(result, optional) -> list:
    """The SET validator finding; its basis declares the executed SET checkout, so the Basis column names it."""
    authority = result["authority"]
    return [finding(
        "The pinned SET contracts validator reports the synthetic observation conformant with effective covariance "
        "rank 2 and refuses an indefinite covariance", "computational_pipeline",
        {"status": result["status"], "effective_rank": result["effective_rank"],
         "indefinite_covariance": result["indefinite_covariance"], "may_authorize": authority.get("may_authorize")},
        {"checks": [_check("SET inspections whose status is not conformant", int(result["status"] != "conformant")),
                    _check("effective covariance rank minus 2",
                           (result["effective_rank"] if isinstance(result["effective_rank"], int) else -1) - 2),
                    _refusal("SET on the indefinite covariance [[1, 2], [2, 1]]",
                             "covariance.matrix is not positive-semidefinite", result["indefinite_covariance"]),
                    _check("inspection input bytes changed", int(not result["unchanged_input"])),
                    _check("inspection reports that may authorize", int(authority.get("may_authorize") is not False))],
         "provider": _executed_checkout("set", optional["set"]),
         "notes": {"validator": {"revision": result["validator"]["revision"], "path": result["validator"]["path"],
                                 "sha256": result["validator"]["sha256"]}}},
        uncertainty=EXACT_COUNT, tolerance=EXACT)]


def _roundtrip_findings(result, optional) -> list:
    """The producer roundtrip finding. Its one provider slot declares the SET checker whose verdict the checks read;
    the PPDA and SCR producers that wrote the artifacts are recorded, with SET, under ``notes.provider``."""
    authority = result["authority"]
    return [finding(
        "PPDA and SCR exchange artifacts pass the pinned SET checker, link to each other, keep a failed verification "
        "failed and a changed result is refused", "computational_pipeline",
        {"status": result["status"], "links": result["links"], "verification_outcome": result["verification_outcome"],
         "changed_result_refusal": result["changed_result_refusal"], "may_authorize": authority.get("may_authorize")},
        {"checks": [_check("roundtrip inspections whose status is not conformant", int(result["status"] != "conformant")),
                    _check("links that matched a supplied artifact",
                           result["links"].count("matched_supplied_reference"), 2, "ge", "invariant"),
                    _check("failed verification outcomes not preserved", int(result["verification_outcome"] != "failed")),
                    _refusal("inspection after changing a component of the SCR result artifact",
                             "result_id does not match the artifact content", result["changed_result_refusal"]),
                    _check("inspection reports that may authorize", int(authority.get("may_authorize") is not False))],
         "provider": _executed_checkout("set", optional["set"]),
         "notes": {"provider": {role: _executed_checkout(role, optional[role])
                                for role in ("ppda", "scr-exchange", "set")}}},
        uncertainty=EXACT_COUNT, tolerance=EXACT)]


@task("T097", changed_files=T097_FILES,
      regression_tests=(f"{TESTS}::test_t097_scr_numerical_heat_integration",
                        f"{TESTS}::test_t097_is_blocked_without_scr",
                        f"{TESTS}::test_t097_keeps_set_results_when_the_engine_is_missing",
                        f"{TESTS}::test_t097_t099_refuse_a_non_repository_scr_binding",
                        f"{TESTS}::test_t097_names_each_executed_provider_beside_its_label"),
      requires=("provider:scr",), plan=_T097_PLAN)
def exact_provider_integrations(ctx):
    fields = {k: v for k, v in deepcopy(_T097_PLAN).items() if k != "findings"}
    physical = finding("The integer heat field describes physical heat diffusion in a material", "physical",
                       "not established: dimensionless integer arithmetic", {})
    identity, error = _scr_identity(ctx)
    if identity is None:
        fields.update(numerical_result=f"SCR binding refused: the bound path is not a readable Git repository root "
                                       f"({error}); nothing was executed.",
                      unresolved_assumptions=["The bound SCR path is not a Git repository root."],
                      provider_runtime_identity=_runtime_identity(T097_FILES, {"scr": {"error": error}}))
        return {"state": "blocked", "fields": fields, "findings": [physical]}
    comparison = providers.compare_with_pins("scr", identity, providers.ciw_pins())
    if not comparison["accepted"]:
        fields.update(numerical_result="SCR not executed: " + _refused_checkout("scr", identity, comparison) + ".",
                      unresolved_assumptions=["The bound SCR checkout is not at a CIW pin or not clean."],
                      provider_runtime_identity=_runtime_identity(
                          T097_FILES, {"scr": {"revision": identity["head"], "source_tree": identity["tree"]}}))
        return {"state": "blocked", "fields": fields, "findings": [
            finding("The SCR integration was not executed because the bound checkout is not at a clean CIW pin",
                    "provenance", comparison, {}, expected_not_established=True), physical]}
    engine = _engine(ctx)
    findings, parts, blocked = [], {}, []
    with tempfile.TemporaryDirectory(prefix="ciw-lab-t097-") as scratch:
        scratch = Path(scratch)
        if engine is None:
            blocked.append("SCR engine unavailable: no scr-engine binding, and cargo is absent or its locked "
                           "offline build produced no binary")
        else:
            try:
                path = providers.materialize(engine["binary"], scratch)
                workbench = providers.scr_workbench_integration(ctx.providers["scr"], path, scratch / "workbench", [
                    ("Declared integer heat workload (example)", example_bytes("declared-workloads/numerical-heat.json"))])
                api = providers.run_heat_kernel(ctx.providers["scr"], path, SURVEY_CASES)
                parts["workbench"], parts["api"] = workbench, api
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                blocked.append("SCR execution failed: " + _sanitized(f"{type(exc).__name__}: {exc}",
                                                                     [ctx.providers["scr"], scratch]))
    pins = providers.ciw_pins()
    optional = {role: _optional_checkout(ctx, role, pins) for role in ("set", "ppda", "scr-exchange")}
    bound_paths = [ctx.providers[role] for role in optional if role in ctx.providers]
    set_reason = optional["set"]["reason"]
    roundtrip_reason = "; ".join(optional[role]["reason"] for role in ("ppda", "scr-exchange", "set")
                                 if optional[role]["reason"])
    with tempfile.TemporaryDirectory(prefix="ciw-lab-t097-exchange-") as scratch:
        if optional["set"]["state"] == "ready":
            try:
                parts["set"] = providers.set_exchange_inspection(ctx.providers["set"], scratch)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                set_reason = "SET inspection failed: " + _sanitized(f"{type(exc).__name__}: {exc}",
                                                                    bound_paths + [scratch])
        if not roundtrip_reason:
            try:
                parts["roundtrip"] = providers.exchange_roundtrip(ctx.providers["ppda"], ctx.providers["scr-exchange"],
                                                                  ctx.providers["set"], scratch)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                roundtrip_reason = "roundtrip failed: " + _sanitized(f"{type(exc).__name__}: {exc}",
                                                                     bound_paths + [scratch])
    ctx.artifact_json("integration.json", _located({
        "scr_identity": {k: identity[k] for k in ("head", "tree", "tracked_sha256", "tracked_files", "lockfiles")},
        "scr_pins": comparison, "engine": None if engine is None else {"origin": engine["origin"], "sha256": engine["sha256"]},
        "optional_providers": optional, "blocked": blocked, "set_reason": set_reason,
        "roundtrip_reason": roundtrip_reason or None, "parts": parts}, _host_bindings(ctx)))
    if "workbench" in parts:
        workbench, api = parts["workbench"], parts["api"]
        demo = workbench["runs"][0]
        survey = api["cases"][0]
        mismatch = sum(sum(a != b for a, b in zip(run["values"], heat_reference(run["initial_values"], run["steps"])))
                       for run in workbench["runs"])
        mismatch += sum(sum(a != b for a, b in zip(case["values"] or [], heat_reference(values, steps)))
                        + (0 if case["values"] else len(values))
                        for case, (steps, values) in zip(api["cases"], SURVEY_CASES))
        from ..declared_workload import HEAT_DESCRIPTOR
        descriptor_equal = api["descriptor_sha256"] == sha256(HEAT_DESCRIPTOR).hexdigest()
        steps, initial = SURVEY_CASES[0]
        ctx.artifact_text("heat-fields.svg", line_plot(
            [("initial field", list(range(5)), initial), ("SCR after 3 steps", list(range(5)), survey["values"]),
             ("integer reference", list(range(5)), heat_reference(initial, steps))],
            title="SCR integer heat kernel versus reference", xlabel="cell index", ylabel="integer field value"))
        findings += [
            finding("SCR executed through CIW's shared numerical-heat workflow and through its own Python API returns "
                    "the declared integer heat fields", "numerical",
                    {"workbench_example": demo["values"], "api_cases": [case["values"] for case in api["cases"]]},
                    _provider_basis(identity, engine), uncertainty=EXACT_INTEGER, tolerance=EXACT),
            finding("SCR heat outputs from the workbench and from SCR's Python API equal an independent integer "
                    "reference", "numerical", {"cases": len(workbench["runs"]) + len(api["cases"]),
                                               "mismatched_cells": mismatch},
                    {"independent_check": {**_check("cells differing from the integer Jacobi reference", mismatch),
                                           "producer": {"implementation": "Scientific-Computation-Runtime execution-cli",
                                                        "revision": identity["head"]},
                                           "checker": {"implementation": "ciw.lab.exchange_provenance_bundles_fixtures"
                                                                         ".heat_reference", "revision": __version__}},
                     **_provider_basis(identity, engine)},
                    uncertainty=EXACT_INTEGER, tolerance=EXACT),
            finding("SCR replay reproduces the numerical identity with fresh execution occurrences and a "
                    "non-independent verification", "computational_pipeline",
                    {"runs": len(workbench["runs"]), "numerical_identity_equal": all(
                        run["numerical_identity_equal"] for run in workbench["runs"]),
                     "fresh_occurrences_per_run": sorted({run["fresh_occurrences"] for run in workbench["runs"]}),
                     "verification_independent": any(run["verification_independent"] for run in workbench["runs"])},
                    {"checks": [_check("runs with different replay numerical identity",
                                       sum(not run["numerical_identity_equal"] for run in workbench["runs"])),
                                _check("runs without four distinct occurrences",
                                       sum(run["fresh_occurrences"] != 4 for run in workbench["runs"])),
                                _check("runs whose replay receipt lacks a numerical match",
                                       sum(run["replay_numerical_match"] is not True for run in workbench["runs"])),
                                _check("runs whose verification claims independence",
                                       sum(run["verification_independent"] is not False for run in workbench["runs"])),
                                _check("SCR descriptor digest differs from CIW's HEAT_DESCRIPTOR",
                                       int(not descriptor_equal))],
                     **_provider_basis(identity, engine)},
                    uncertainty=EXACT_COUNT, tolerance=EXACT),
            finding("The reopened SCR workspace carries no binding and refuses replay", "computational_pipeline",
                    {"bindings": workbench["reopened_bindings"], "available": workbench["reopened_available"],
                     "replay": f"{workbench['unbound_replay']['code']}: {workbench['unbound_replay'].get('message')}"},
                    {"checks": [_refusal("bundle.replay after reopen", f"operation_unavailable: {UNBOUND}",
                                         f"{workbench['unbound_replay']['code']}: "
                                         f"{workbench['unbound_replay'].get('message')}"),
                                _check("execution entry points reached while reopening",
                                       len(workbench["reopen_attempts"])),
                                _check("bindings after reopen", len(workbench["reopened_bindings"]))]},
                    uncertainty=EXACT_COUNT, tolerance=EXACT),
        ]
    else:
        findings.append(finding(
            "SCR numerical-heat execution was not performed", "computational_pipeline",
            {"reason": blocked[0] if blocked else "not attempted",
             "required": "--provider scr-engine=<execution-cli> or cargo on PATH"}, {}, expected_not_established=True))
    findings += _engine_origin_findings(ctx, engine)
    if "set" in parts:
        findings += _set_findings(parts["set"], optional)
    else:
        blocked.append("SET exchange validator: " + set_reason)
        findings.append(finding(
            "SET exchange conformance was not executed", "computational_pipeline",
            {"reason": set_reason,
             "required": "--provider set=<State-Estimation-Evaluation-Testbed at 542e672be512bf43b61253f2b2a43cd967cb3062>",
             "pytest": "CIW_SET_REPO=<set> python -m pytest -q tests/test_exchange.py"}, {},
            expected_not_established=True))
    if "roundtrip" in parts:
        findings += _roundtrip_findings(parts["roundtrip"], optional)
    else:
        blocked.append("PPDA/SCR/SET producer roundtrip: " + roundtrip_reason)
        findings.append(finding(
            "The PPDA/SCR/SET exchange producer roundtrip was not executed", "computational_pipeline",
            {"reason": roundtrip_reason,
             "required": {role: revision for role, revision in sorted(providers.EXCHANGE_WORKFLOW_PINS.items())},
             "pytest": "CIW_SET_REPO=<set> CIW_ACQUISITION_REPO=<ppda> CIW_RUNTIME_REPO=<scr-exchange> "
                       "python -m pytest -q tests/test_exchange_integration.py"}, {},
            expected_not_established=True))
    findings.append(physical)
    # A part that ran (even one whose result refutes its claim) makes the report partial, never blocked: a
    # blocked report may carry nothing established and would drop that evidence.
    established = any(record["evidence_status"] != "not_established" for record in findings)
    state = _settle("completed" if not blocked else ("partial" if parts or established else "blocked"), findings)
    heat = parts.get("workbench", {}).get("runs", [])
    fields.update(
        input_data=["examples/declared-workloads/numerical-heat.json (embedded)",
                    f"API cases (steps, initial field): {SURVEY_CASES}", f"SCR checkout {identity['head']}",
                    f"engine: {None if engine is None else engine['origin']}"],
        experiment=("Bind SCR and its engine in a fresh session, add the example source, execute, replay, save, "
                    "reopen unbound under the guard and request replay; run four cases (including [0, 0, 1000, 0, 0] "
                    "for 3 steps) through SCR's own Python API; when "
                    "bound, validate a synthetic observation with SET and run the PPDA/SCR/SET producer roundtrip."),
        numerical_result=("; ".join([f"workbench {run['initial_values']} x{run['steps']} -> {run['values']}"
                                     for run in heat] + [f"API {values} x{steps} -> {case['values']}" for case,
                                     (steps, values) in zip(parts.get("api", {}).get("cases", []), SURVEY_CASES)])
                          or "SCR not executed") + f"; not executed here: {blocked or 'none'}",
        uncertainty="Exact integer arithmetic and exact string comparisons; no tolerance.",
        failure_modes_checked=["pin mismatch", "engine drift", "replay identity drift", "binding recovered from "
                               "saved data", "descriptor drift between SCR and CIW", "verification marked independent",
                               "SET status, rank and refusal text", "roundtrip link and claim-boundary drift"],
        unresolved_assumptions=blocked + ["CIW records a bound engine as operator_asserted_not_attested; this run's "
                                          "engine origin is "
                                          f"{None if engine is None else engine['origin']} (checked against a locked "
                                          "build only when an operator-bound engine and cargo are both present).",
                                          "The PPDA telemetry stack (ppda, stfe, gsie, set, cbsr at "
                                          "src/ciw/telemetry-runtimes.json pins) is a separate integration."],
        recommended_next_task=NEXT_STEPS["T097"],
        provider_runtime_identity=_runtime_identity(T097_FILES, {
            "scr": {"revision": identity["head"], "source_tree": identity["tree"],
                    "engine_sha256": None if engine is None else engine["sha256"],
                    "engine_origin": None if engine is None else engine["origin"]},
            # Head and tree, as T098 records them, so a runtime inventory (T165) lists each checkout once.
            **{role: {"state": value["state"], "head": value.get("head"), "tree": value.get("tree"),
                      "matched": value.get("matched")} for role, value in optional.items()}}))
    return {"state": state, "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T098

def _pin_consistency() -> dict:
    """CIW's declared provider revisions grouped by repository (roles alias repositories, e.g. scr-exchange)."""
    from .. import declared_workload, proved_heat
    grouped: dict = {}
    for role, entries in providers.ciw_pins().items():
        repository = providers.REPOSITORIES.get(role, role)
        for pin in entries:
            grouped.setdefault(repository, {}).setdefault(pin["revision"], []).append(pin["declared_in"])
    listing = {repository: {revision: sorted(places) for revision, places in sorted(revisions.items())}
               for repository, revisions in sorted(grouped.items())}
    return {"scr_revisions": listing.get(providers.REPOSITORIES["scr"], {}),
            "set_revisions": listing.get(providers.REPOSITORIES["set"], {}),
            "scr_workflows_agree": declared_workload.PINS["numerical-heat"]["revision"] == proved_heat.PIN["revision"],
            "by_repository": listing}


REVISION_REFUSAL = "SOURCE_PIN_MISMATCH: The bound checkout is not at the declared revision"


def _control_revision(role, identity, pins):
    """(revision, declared_in) of the first CIW pin of another repository that is not this checkout's HEAD."""
    repository = providers.REPOSITORIES.get(role, role)
    candidates = sorted((pin["revision"], pin["declared_in"]) for other, entries in pins.items()
                        if providers.REPOSITORIES.get(other, other) != repository for pin in entries
                        if pin["revision"] != identity["head"])
    return candidates[0] if candidates else None


def _adapter_outcomes(role, path, identity, pins) -> list:
    """CIW's own subprocess adapter against every module pin CIW declares for the role.

    A control row runs the role's first module pin with a revision CIW pins for
    another repository, so the refusal side is exercised even when every
    declared pin of the role is at HEAD.
    """
    from ..adapters.protocol import AdapterRefusal
    from ..adapters.subprocess import PinnedSubprocessAdapter

    def attempt(pin, revision):
        try:
            PinnedSubprocessAdapter(path, revision, pin["module"], source_root=pin["source_root"])
            return "accepted"
        except AdapterRefusal as exc:
            return f"{exc.code}: {exc}"
        except ValueError as exc:
            return f"ValueError: {exc}"

    module_pins = [pin for pin in pins.get(role, []) if pin.get("module")]
    rows = [{"declared_in": pin["declared_in"], "revision": pin["revision"], "at_head": pin["revision"] == identity["head"],
             "control": False, "outcome": attempt(pin, pin["revision"])} for pin in module_pins]
    control = _control_revision(role, identity, pins) if module_pins else None
    if control is not None:
        revision, source = control
        rows.append({"declared_in": f"control: {module_pins[0]['declared_in']} module at the {source} revision",
                     "revision": revision, "at_head": False, "control": True,
                     "outcome": attempt(module_pins[0], revision)})
    return rows


def _refusal_reasons(comparison) -> list:
    reasons = []
    if not comparison["matched"]:
        reasons.append("no CIW pin matches HEAD")
    if comparison["tree_refusals"]:
        reasons.append("pinned tree differs")
    if not comparison["clean"]:
        reasons.append("checkout not clean")
    return reasons


def _refusal_finding(role, path, identity, comparison, pins):
    """One refused checkout, each reason corroborated by a second reader of the repository state."""
    from .runner import git_identity
    reasons = _refusal_reasons(comparison)
    try:
        other = git_identity(Path(path))
    except (OSError, subprocess.SubprocessError):
        other = {"revision": None, "source_tree": None, "dirty": None}
    declared = pins.get(role, [])
    checks = []
    if "no CIW pin matches HEAD" in reasons:
        checks.append(_check(f"{role} pins at the HEAD read by ciw.lab.runner.git_identity",
                             sum(pin["revision"] == other["revision"] for pin in declared)))
    if "pinned tree differs" in reasons:
        checks.append(_check(f"{role} pins at HEAD whose tree equals the tree read by ciw.lab.runner.git_identity",
                             sum(pin["revision"] == other["revision"] and pin.get("source_tree") == other["source_tree"]
                                 for pin in declared)))
    if "checkout not clean" in reasons:
        checks.append(_check(f"{role} checkouts reported clean by git status --ignored --untracked-files=all",
                             int(not providers.status_entries(path))))
    return finding(f"The bound {role} checkout is refused against CIW's pins", "provenance",
                   {"head": identity["head"], "reasons": reasons, "matched": comparison["matched"]},
                   {"checks": checks}, uncertainty=EXACT_COUNT, tolerance=EXACT)


@task("T098", changed_files=(MODULE, PROVIDERS),
      regression_tests=(f"{TESTS}::test_t098_provider_identities",
                        f"{TESTS}::test_t098_refuses_an_unpinned_checkout_and_is_location_independent",
                        f"{TESTS}::test_tree_recomputation_matches_git_on_a_synthetic_repository"))
def provider_identities(ctx):
    changed = (MODULE, PROVIDERS)
    pins = providers.ciw_pins()
    consistency = _pin_consistency()
    roles = sorted(role for role in providers.REPOSITORIES if role in ctx.providers)
    identities, comparisons, adapters, errors = {}, {}, {}, {}
    # Bound interpreters (for example plsr-python, ftr-python) that host provider code.
    interpreters, interpreter_errors = {}, {}
    for role in sorted(role for role, path in ctx.providers.items() if role.endswith("-python") and path):
        try:
            interpreters[role] = providers.interpreter_identity(ctx.providers[role])
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            interpreter_errors[role] = type(exc).__name__
    for role in roles:
        try:
            identities[role] = providers.checkout_identity(ctx.providers[role])
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            errors[role] = type(exc).__name__
            continue
        comparisons[role] = providers.compare_with_pins(role, identities[role], pins)
        adapters[role] = _adapter_outcomes(role, ctx.providers[role], identities[role], pins)
    accepted = sorted(role for role, row in comparisons.items() if row["accepted"])
    refused = sorted(role for role, row in comparisons.items() if not row["accepted"])
    engine = _engine(ctx) if "scr" in accepted else None
    liveness = None
    if engine is not None:
        with tempfile.TemporaryDirectory(prefix="ciw-lab-t098-") as scratch:
            try:
                liveness = providers.run_heat_kernel(ctx.providers["scr"],
                                                     providers.materialize(engine["binary"], scratch), SURVEY_CASES[:1])
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                errors["scr-engine"] = type(exc).__name__
    ctx.artifact_json("provider-identities.json", _located({
        "identities": identities, "comparisons": comparisons, "adapter_pin_checks": adapters, "errors": errors,
        "interpreters": interpreters, "interpreter_errors": interpreter_errors,
        "ciw_pins": pins, "pins_by_repository": consistency["by_repository"], "engine": None if engine is None else {
            "origin": engine["origin"], "sha256": engine["sha256"], "byte_count": len(engine["binary"]),
            "toolchain": None if engine["build"] is None else {k: engine["build"][k] for k in ("cargo", "rustc")}}},
        _host_bindings(ctx)))
    input_data = [f"{role}: {providers.REPOSITORIES[role]} at {identities[role]['head']}" if role in identities
                  else f"{role}: not a readable Git repository root" for role in roles]
    input_data += [f"{role}: bound interpreter (version and digest in provider-identities.json)"
                   for role in sorted(interpreters.keys() | interpreter_errors.keys())]
    if not identities:
        fields = _fields(
            "Every bound provider checkout is clean and at a CIW pin, and its bytes reproduce its Git tree.",
            "Identity = (HEAD, HEAD^{tree}, SHA-256 over tracked working bytes, lockfile SHA-256, engine SHA-256).",
            input_data, "None: no readable provider checkout is bound.", "Pins matched; recomputed tree equals Git's.",
            "Blocked: bind providers with --provider ROLE=PATH for roles " + ", ".join(sorted(providers.REPOSITORIES)),
            f"No checkout identity computed ({len(roles)} bound paths unreadable); the CIW pin table is retained in "
            "provider-identities.json.",
            "not quantified", ["no readable provider bound"], ["No readable provider checkout is bound."],
            "Bind csg, ftr, scr (scripts/check_lab.py does) and rerun T098.")
        # A blocked report carries no established finding; the CIW pin table is in the artifact.
        return {"state": "blocked", "fields": fields, "findings": []}
    scr_revisions, set_revisions = consistency["scr_revisions"], consistency["set_revisions"]
    findings = [finding(
        "The declared-workload and proved-heat workflows pin one SCR revision, while CIW declares more than one "
        "revision of SCR and of SET across its workflows", "provenance",
        {"scr_revisions": scr_revisions, "set_revisions": set_revisions},
        {"checks": [_check("declared-workload and proved-heat SCR pins that differ",
                           int(not consistency["scr_workflows_agree"])),
                    _check("distinct SCR revisions declared across CIW", len(scr_revisions), 2, "ge", "invariant"),
                    _check("distinct SET revisions declared across CIW", len(set_revisions), 2, "ge", "invariant")]},
        uncertainty=EXACT_COUNT, tolerance=EXACT)]
    findings.append(finding(
        "Every bound provider checkout is clean and at a CIW pin and, where CIW pins one, the pinned tree",
        "provenance", {"accepted": {role: {"head": identities[role]["head"], "tree": identities[role]["tree"],
                                           "pins": comparisons[role]["matched"]} for role in accepted},
                       "refused": refused, "unreadable": sorted(errors)},
        {"checks": [_check("bound checkouts refused against CIW pins", len(refused)),
                    _check("bound paths that are not readable Git repository roots", len(errors))]},
        uncertainty=EXACT_COUNT, tolerance=EXACT))
    findings += [_refusal_finding(role, ctx.providers[role], identities[role], comparisons[role], pins)
                 for role in refused]
    clean = sorted(role for role in identities if comparisons[role]["clean"])
    modified = sorted(role for role in identities if identities[role]["mismatched_files"])
    tree_mismatch = sum(identities[role]["recomputed_tree"] != identities[role]["tree"] for role in clean)
    if clean:
        findings.append(finding(
            "The working bytes of every clean bound checkout reproduce Git's HEAD tree id", "provenance",
            {role: identities[role]["recomputed_tree"] for role in clean},
            {"independent_check": {**_check("clean checkouts whose recomputed tree differs from git rev-parse "
                                            "HEAD^{tree}", tree_mismatch),
                                   "producer": {"implementation": "ciw.lab.exchange_provenance_bundles_providers"
                                                                  ".checkout_identity", "revision": __version__},
                                   "checker": {"implementation": "git rev-parse HEAD^{tree}",
                                               "revision": _git_version()}}},
            uncertainty=EXACT_COUNT, tolerance=EXACT))
    if modified:
        findings.append(finding(
            "Modified tracked bytes in a bound checkout change its recomputed tree", "provenance",
            {role: len(identities[role]["mismatched_files"]) for role in modified},
            {"checks": [_check("checkouts with modified tracked files whose recomputed tree still equals HEAD's",
                               sum(identities[role]["recomputed_tree"] == identities[role]["tree"]
                                   for role in modified))]},
            uncertainty=EXACT_COUNT, tolerance=EXACT))
    if clean:
        # A second reader of the same digests: blob bytes from Git's object database, not the working tree.
        differing, object_errors = 0, []
        for role in clean:
            try:
                second = providers.head_object_digests(ctx.providers[role])
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                object_errors.append(f"{role}: {type(exc).__name__}")
                differing += 1
                continue
            first = identities[role]
            differing += int(second["tracked_sha256"] != first["tracked_sha256"])
            differing += int(second["tracked_files"] != first["tracked_files"])
            differing += sum(second["lockfiles"].get(name) != first["lockfiles"].get(name)
                             for name in second["lockfiles"].keys() | first["lockfiles"].keys())
        findings.append(finding(
            "Tracked-source and lockfile digests of every clean bound checkout agree between its working tree and "
            "Git's HEAD objects", "provenance",
            {role: {"tracked_sha256": identities[role]["tracked_sha256"], "tracked_files": identities[role]["tracked_files"],
                    "lockfiles": identities[role]["lockfiles"]} for role in clean},
            {"independent_check": {**_check("tracked-source and lockfile digests differing between the working-tree "
                                            "reading and the Git object reading", differing),
                                   "producer": {"implementation": "ciw.lab.exchange_provenance_bundles_providers"
                                                                  ".checkout_identity (working-tree bytes)",
                                                "revision": __version__},
                                   "checker": {"implementation": "git cat-file --batch of HEAD blobs, hashed with "
                                                                 "SHA-256", "revision": _git_version()}},
             **({"notes": {"object_read_errors": object_errors}} if object_errors else {})},
            uncertainty=EXACT_COUNT, tolerance=EXACT))
    rows = [(role, row) for role in sorted(adapters) for row in adapters[role]]
    if rows:
        checks = []
        for role, row in rows:
            reference = f"{role} against {row['declared_in']}"
            if not row["at_head"]:
                checks.append(_refusal(reference, REVISION_REFUSAL, row["outcome"]))
            elif comparisons[role]["clean"]:
                checks.append(_refusal(reference, "accepted", row["outcome"]))
            else:
                checks.append(_refusal(reference, "SOURCE_PIN_MISMATCH", row["outcome"].split(":", 1)[0]))
        findings.append(finding(
            "CIW's pinned subprocess adapter accepts each clean bound checkout for the module pins at its HEAD and "
            "refuses it at any other revision, including a control revision taken from another repository's pin",
            "provenance",
            {role: {"at_head": sorted(row["declared_in"] for row in adapters[role] if row["at_head"]),
                    "refused_declared_pins": sorted(row["declared_in"] for row in adapters[role]
                                                    if not row["control"] and row["outcome"] != "accepted"),
                    "control_refused": any(row["control"] and row["outcome"] != "accepted" for row in adapters[role])}
             for role in sorted(adapters) if adapters[role]}
            | {"refusal_cases": sum(not row["at_head"] for _, row in rows),
               "acceptance_cases": sum(row["at_head"] for _, row in rows)},
            {"checks": checks}, uncertainty=EXACT_COUNT, tolerance=EXACT))
    if liveness is not None:
        findings.append(finding(
            "The SCR engine recorded for this run executes the SCR heat descriptor on the survey input",
            "numerical", {"engine_origin": engine["origin"],
                          "cargo_lock_sha256": identities["scr"]["lockfiles"].get("crates/Cargo.lock"),
                          "survey_output": liveness["cases"][0]["values"],
                          "descriptor_sha256": liveness["descriptor_sha256"]},
            _provider_basis(identities["scr"], engine), uncertainty=EXACT_INTEGER, tolerance=EXACT))
    findings += _engine_origin_findings(ctx, engine)
    if "plsr-python" in interpreters or "plsr-python" in interpreter_errors:
        findings.append(_plsr_installation_finding(interpreters.get("plsr-python"),
                                                   interpreter_errors.get("plsr-python")))
    findings.append(finding(
        "A matching HEAD, tree and lock digest authenticates the upstream repository, toolchain and built engine",
        "provenance", "not established: digests are not signatures", {}, expected_not_established=True))
    state = _settle("partial" if refused or errors else "completed", findings)
    adapter_refused = sorted(role for role, role_rows in adapters.items()
                             if any(not row["control"] and row["outcome"] != "accepted" for row in role_rows))
    controls_refused = sorted(role for role, role_rows in adapters.items()
                              if any(row["control"] and row["outcome"] != "accepted" for row in role_rows))
    fields = _fields(
        "Every bound provider checkout is clean and at a CIW pin; its working bytes reproduce its Git tree; its "
        "lockfiles, the bound provider interpreters and the engine used in this run are recorded.",
        "Git object ids: blob = H('blob' len NUL bytes), tree = H('tree' len NUL sorted(mode name NUL id)); "
        "tracked digest = SHA-256 over 'mode kind sha256(bytes) path' lines in ls-tree order, computed from the "
        "working tree and again from Git's HEAD objects; lockfiles = Cargo.lock, uv.lock, poetry.lock, Pipfile.lock, "
        "package-lock.json and the other recognised names, plus fully pinned requirements*.txt.",
        input_data,
        "git rev-parse, git ls-tree, git cat-file, working-tree bytes, CIW pin tables, PinnedSubprocessAdapter "
        "outcomes, interpreter probes.",
        "HEAD equals a CIW pin; pinned trees equal; recomputed tree equals Git's; working-tree and Git-object "
        "digests equal; the adapter refuses every revision other than HEAD.",
        "For each bound role read HEAD and tree, hash every tracked file and lockfile, recompute the tree "
        "independently of Git, reread the digests from Git's HEAD objects, compare with every CIW pin for that role "
        "(refusing mismatches with their reasons), and let CIW's own adapter accept or refuse each module pin plus a "
        "control revision from another repository; probe each bound interpreter (version, executable digest, "
        "installed PLSR sources); execute the SCR engine once on the survey input when available.",
        f"accepted {accepted}, refused {refused}, unreadable {sorted(errors)}; {tree_mismatch} tree recomputation "
        f"mismatches among clean checkouts; adapter refusals of declared pins for {adapter_refused}, of the "
        f"control revision for {controls_refused}; "
        f"interpreters recorded {sorted(interpreters)}, unreadable {sorted(interpreter_errors)}; "
        f"engine {None if engine is None else engine['origin']}.",
        "Exact digests.",
        ["dirty or untracked files", "wrong revision", "pinned tree drift", "modified tracked bytes",
         "executable-bit drift", "one checkout serving several different CIW pins", "adapter accepting a wrong pin",
         "working-tree lockfile differing from HEAD", "installed PLSR drifting from its pin"],
        ["Engine and interpreter digests depend on the toolchain and the host's Python build; they are retained as "
         "provenance in provider-identities.json (engine, interpreters) and provider_runtime_identity, not compared as "
         "regression values. A PLSR install commit is recorded only when the installer wrote one (direct_url.json).",
         "Pins declared only in .github/workflows/exchange.yml are mirrored in EXCHANGE_WORKFLOW_PINS.",
         "Checkout paths are kept in provider-identities.json only; report prose names repositories and revisions."],
        NEXT_STEPS["T098"])
    bound = {role: {"head": identities[role]["head"], "tree": identities[role]["tree"]} for role in sorted(identities)}
    if engine is not None:
        bound["scr"]["engine_sha256"] = engine["sha256"]
    for role, record in sorted(interpreters.items()):
        bound[role] = {"python_version": record["python_version"], "sha256": record["sha256"],
                       **({"plsr_version": record["plsr"].get("version"),
                           "plsr_install_commit": record["plsr"].get("install_commit")} if record.get("plsr") else {})}
    fields["provider_runtime_identity"] = _runtime_identity(changed, bound)
    return {"state": state, "fields": fields, "findings": findings}


def _plsr_installation_finding(record, error):
    """The PLSR runtime installed in the bound plsr-python interpreter against ciw/plsr-runtime.json."""
    manifest = providers.plsr_manifest()
    plsr = (record or {}).get("plsr") or {}
    files = plsr.get("files") or {}
    differing = sorted(set(files) ^ set(manifest["files"])
                       | {name for name in files.keys() & manifest["files"].keys() if files[name] != manifest["files"][name]})
    return finding(
        "The PLSR runtime installed in the bound plsr-python interpreter has CIW's pinned version and source file "
        "digests", "provenance",
        {"pin": manifest["commit"], "package_version": plsr.get("version"), "source_files": len(files),
         "differing_source_files": differing, **({"probe_error": error} if error else {})},
        {"checks": [_check("installed PLSR versions differing from ciw/plsr-runtime.json",
                           int(plsr.get("version") != manifest["package_version"])),
                    _check("installed PLSR source files missing, extra or differing from ciw/plsr-runtime.json",
                           len(differing) + (0 if files else len(manifest["files"])))],
         "notes": {"install_commit": plsr.get("install_commit"), "install_kind": plsr.get("install_kind"),
                   "hashing": "SHA-256 of CRLF-normalised .py, .json and py.typed files, as ciw.plsr_engine hashes them"}},
        uncertainty=EXACT_COUNT, tolerance=EXACT)


def _git_version() -> str:
    try:
        return subprocess.run(["git", "--version"], capture_output=True, text=True,
                              timeout=30).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# ---------------------------------------------------------------- T099

_T099_PLAN = _fields(
    "The pinned SCR execution engine builds with cargo build --release --locked --offline, leaves the checkout "
    "and Cargo.lock unchanged, is reproducible across target directories and executes the heat kernel correctly.",
    "Cargo resolution fixed by crates/Cargo.lock (no external crates); determinism = equal binary digests.",
    ["SCR checkout at a59aba283b0304faeeb3e5d305087e7709e171ca", "Rust toolchain (cargo, rustc)"],
    "cargo exit codes, Cargo.lock digests, binary digests, checkout cleanliness, engine output.",
    "Exit 0; lock unchanged; clean checkout; equal digests; output equals the integer reference.",
    "Bind --provider scr=<checkout> with cargo on PATH; command: CARGO_TARGET_DIR=<tmp> cargo build --release "
    "--locked --offline --manifest-path <scr>/crates/Cargo.toml -p execution-cli. The SP1 proved-heat build is "
    "recorded as blocked with its requirements and never attempted.",
    "none", "not quantified", ["provider or toolchain unavailable"],
    ["The SCR checkout or cargo is unavailable in this run."],
    "Install a Rust toolchain, bind the pinned SCR checkout and rerun T099.")
_T099_PLAN["findings"] = [finding("A successful locked build makes the engine acceptable for production use",
                                  "production_acceptance", "not decided by the workbench", {})]
T099_FILES = (MODULE, FIXTURES, PROVIDERS)
BUILD_CLAIM = "cargo build --release --locked --offline of SCR execution-cli succeeds at the pinned revision"


@task("T099", changed_files=T099_FILES,
      regression_tests=(f"{TESTS}::test_t099_locked_offline_scr_build",
                        f"{TESTS}::test_t097_t099_refuse_a_non_repository_scr_binding",
                        f"{TESTS}::test_t099_names_the_built_checkout_beside_its_label"),
      requires=("provider:scr", "tool:cargo"), plan=_T099_PLAN)
def locked_cargo_build(ctx):
    fields = {k: v for k, v in deepcopy(_T099_PLAN).items() if k != "findings"}
    production = finding("A successful locked build makes the engine acceptable for production use",
                         "production_acceptance", "not decided by the workbench", {})
    identity, error = _scr_identity(ctx)
    if identity is None:
        fields.update(numerical_result=f"SCR binding refused: the bound path is not a readable Git repository root "
                                       f"({error}); nothing was built.",
                      provider_runtime_identity=_runtime_identity(T099_FILES, {"scr": {"error": error}}))
        return {"state": "blocked", "fields": fields, "findings": [production]}
    comparison = providers.compare_with_pins("scr", identity, providers.ciw_pins())
    if not comparison["accepted"]:
        fields.update(numerical_result="SCR not built: " + _refused_checkout("scr", identity, comparison) + ".",
                      provider_runtime_identity=_runtime_identity(
                          T099_FILES, {"scr": {"head": identity["head"], "tree": identity["tree"]}}))
        return {"state": "blocked", "fields": fields, "findings": [production]}
    build = _locked_build(ctx)
    after = providers.checkout_identity(ctx.providers["scr"])
    ok = [row for row in build["builds"] if row["returncode"] == 0 and row["binary_sha256"]]
    output, api_error = None, None
    if build["binary"] is not None:
        with tempfile.TemporaryDirectory(prefix="ciw-lab-t099-") as scratch:
            try:
                output = providers.run_heat_kernel(ctx.providers["scr"],
                                                   providers.materialize(build["binary"], scratch), SURVEY_CASES[:1])
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                api_error = type(exc).__name__
    ctx.artifact_json("cargo-build.json", {key: value for key, value in build.items() if key != "binary"})
    probes = {"protoc": providers.which("protoc") is not None, "sp1_bound": "sp1" in ctx.providers,
              "zk_manifest": (Path(ctx.providers["scr"]) / "zk" / "Cargo.toml").is_file()}
    ctx.artifact_json("sp1-requirements.json", {"requirements": providers.SP1_REQUIREMENTS, "probes": probes,
                                                "attempted": False})
    digests = sorted({row["binary_sha256"] for row in ok})
    lock = identity["lockfiles"].get("crates/Cargo.lock")
    # The build and the engine run rest on the bound SCR checkout: each finding declares it as its provider, so the
    # Basis column names it beside the label (a passing check still decides the label).
    built = {"repository": providers.REPOSITORIES["scr"], "revision": identity["head"], "source_tree": identity["tree"],
             "executed": True}
    findings = [finding(
        BUILD_CLAIM, "computational_pipeline",
        {"exit_codes": [row["returncode"] for row in build["builds"]], "flags": ["--release", "--locked", "--offline"],
         "package": "execution-cli", "cargo_lock_sha256": lock, "builds": len(build["builds"])},
        {"checks": [_check("builds with a nonzero or missing exit code",
                           sum(row["returncode"] != 0 for row in build["builds"])),
                    _check("builds without an execution-cli binary",
                           sum(not row["binary_sha256"] for row in build["builds"]))],
         "provider": built, "notes": {"binary_sha256": digests}},
        uncertainty=EXACT_COUNT, tolerance=EXACT)]
    if ok:
        values = output["cases"][0]["values"] if output else None
        steps, initial = SURVEY_CASES[0]
        reference = heat_reference(initial, steps)
        findings += [
            finding("The locked build leaves Cargo.lock and the checkout unchanged and is bit-reproducible across "
                    "fresh target directories", "computational_pipeline",
                    {"cargo_lock_unchanged": build["cargo_lock_sha256_before"] == build["cargo_lock_sha256_after"],
                     "checkout_clean": not after["dirty"] and not after["mismatched_files"]
                     and not after["untracked_outside_caches"], "distinct_binary_digests": len(digests)},
                    {"checks": [_check("Cargo.lock changed by the build",
                                       int(build["cargo_lock_sha256_before"] != build["cargo_lock_sha256_after"])),
                                _check("checkout dirty, modified or with new untracked files after the build",
                                       int(after["dirty"] or bool(after["mismatched_files"])
                                           or bool(after["untracked_outside_caches"]))),
                                _check("builds that failed", len(build["builds"]) - len(ok)),
                                _check("distinct binary digests beyond one", len(digests) - 1)],
                     "provider": built},
                    uncertainty=EXACT_COUNT, tolerance=EXACT),
            finding("The freshly built engine's heat output equals an independent integer reference", "numerical",
                    values, {"independent_check": {
                        **_check("cells differing from the integer Jacobi reference",
                                 sum(a != b for a, b in zip(values or [], reference)) + (0 if values else len(reference))),
                        "producer": {"implementation": "Scientific-Computation-Runtime execution-cli (locked build)",
                                     "revision": identity["head"]},
                        "checker": {"implementation": "ciw.lab.exchange_provenance_bundles_fixtures.heat_reference",
                                    "revision": __version__}},
                     "provider": dict(built, runtime_digest="sha256:" + digests[0])},
                    uncertainty=EXACT_INTEGER, tolerance=EXACT),
        ]
    findings += [
        finding("The SP1 proved-heat locked build was not attempted", "computational_pipeline",
                {"attempted": False, "requirements": providers.SP1_REQUIREMENTS}, {}, expected_not_established=True),
        production,
    ]
    fields.update(
        numerical_result=(f"{len(ok)}/{len(build['builds'])} locked offline builds succeeded; distinct binary "
                          f"digests {len(digests)}; Cargo.lock {lock}; survey output "
                          f"{None if output is None else output['cases'][0]['values']}; SP1 build not attempted."),
        uncertainty="Exact digests and integers; the binary digest depends on the Rust toolchain and is recorded as "
                    "provenance.",
        failure_modes_checked=["lockfile rewrite", "network access (--offline)", "writes into the checkout",
                               "non-deterministic build output", "engine output drift", "build timeout"],
        unresolved_assumptions=["The binary digest depends on the Rust toolchain recorded in "
                                "provider_runtime_identity; CI pins rustc 1.94.0 for the proved-heat gate, and other "
                                "toolchains may produce different digests.",
                                "The SP1 proved-heat build needs crates.io and GitHub release downloads, the SP1 "
                                "checkout, the Succinct compiler archive, protoc and >= 7 GiB RAM / 20 GiB disk "
                                "(.github/workflows/proved-heat.yml); host probes are in sp1-requirements.json."]
        + ([f"The engine heat-kernel call failed ({api_error})."] if api_error else []),
        recommended_next_task=NEXT_STEPS["T099"],
        provider_runtime_identity=_runtime_identity(T099_FILES, {
            "scr": {"head": identity["head"], "tree": identity["tree"], "engine_sha256": digests[0] if digests else None,
                    "cargo": build["cargo"], "rustc": build["rustc"]}}))
    return {"state": "partial", "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T100

_COMPUTATIONAL_LABELS = frozenset({"analytic", "synthetic", "numerically_verified", "provider_backed"})
# The packaged queue's tasks before T100 (T001-T099).
EXPECTED_EARLIER_REPORTS = 99
# The rendered finding table: header and rule as report.render_markdown writes them.
FINDINGS_TABLE = FINDINGS_HEADER + "\n" + FINDINGS_RULE + "\n"
# Basis components that name where a result's inputs or numbers came from, with the identity each shows.
DECLARED_SOURCES = ("synthetic_inputs", "provider")


def _cells(row: str) -> int:
    """Cells of a Markdown table row: unescaped pipes minus one (GFM drops cells beyond the header's)."""
    return len(re.findall(r"(?<!\\)\|", row)) - 1


def _cell_texts(row: str) -> list:
    """The text of each cell of a Markdown table row, split at unescaped pipes."""
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", row)[1:-1]]


HEADER_CELLS = _cells(FINDINGS_HEADER)


def _shown(value, limit=80) -> str:
    """A declared identity as a reader should see it: whitespace collapsed, cut at ``limit`` with an ellipsis."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


# Names of the basis components in audit problems; the audit counts a finding whose generator or provider is
# missing from its row as hiding its source.
PROBLEM_NAMES = {"synthetic_inputs": "generator"}
UPGRADED_ACQUISITION = "hardware acquisition shown without an acquisition the finding establishes"


def _expected_basis(record) -> list:
    """(component, text) of each part of the Basis cell the contract prescribes, in :data:`ORIGINS` order.

    Restated here from docs/lab/AUTHORING.md rather than read from
    ``describe_basis``, so a renderer that changes the cell cannot also change
    what the audit expects: each declared component in words, a generator
    with its name and seed, an executed provider with its repository and the
    first 12 characters of its revision, and an acquisition record as
    'hardware acquisition (<device>)' only on a physical-domain finding
    labelled ``hardware_measured`` or ``independently_verified`` (one it
    establishes), 'declared acquisition record (not accepted)' on every other.
    """
    basis, declared, parts = record["basis"], set(finding_origin(record)), []
    for item in (item for item in ORIGINS if item in declared):
        if item == "acquisition":
            accepted = (record["domain"] in PHYSICAL_DOMAINS
                        and record["evidence_status"] in ("hardware_measured", "independently_verified"))
            text = (f"{ACCEPTED_ACQUISITION} ({_shown(basis['acquisition']['device'])})" if accepted
                    else UNACCEPTED_ACQUISITION)
        elif item == "synthetic_inputs":
            generator = basis["generator"]
            seed = f", seed {_shown(generator['seed'], 40)}" if generator.get("seed") is not None else ""
            text = f"{ORIGIN_WORDS[item]} ({_shown(generator['name'])}{seed})"
        elif item == "provider":
            provider = basis["provider"]
            text = f"{ORIGIN_WORDS[item]} ({_shown(provider['repository'])}@{_shown(provider['revision'])[:12]})"
        else:
            text = ORIGIN_WORDS[item]
        parts.append((item, text))
    return parts


def _basis_problems(record, cell: str) -> list:
    """How a rendered Basis cell departs from the cell the declared basis prescribes (``_expected_basis``).

    The whole cell is compared, so an omitted component cannot hide behind the
    same words inside a declared identity, and extra or reordered text is a
    problem too. Each prescribed part missing from the comma-separated parts
    of the cell is named ('generator', 'provider', 'acquisition',
    'derivation', ...; 'none' for a finding declaring nothing that does not
    read 'no declared basis'), 'hardware acquisition' shown on a finding
    whose acquisition is not accepted (or that declares none) is named as
    such, and any other difference reads 'order or extra text'.
    """
    shown = cell.replace("\\|", "|")
    parts = _expected_basis(record)
    expected = ", ".join(text for _, text in parts) or NO_ORIGIN
    if shown == expected:
        return []
    present = f", {shown}, "
    problems = [PROBLEM_NAMES.get(item, item) for item, text in parts if f", {text}, " not in present]
    if not parts and f", {NO_ORIGIN}, " not in present:
        problems.append("none")
    # 'hardware acquisition' beyond the occurrences the prescribed cell holds (inside a declared identity, say).
    if shown.count(ACCEPTED_ACQUISITION) > expected.count(ACCEPTED_ACQUISITION):
        problems.append(UPGRADED_ACQUISITION)
    return problems or ["order or extra text"]


def _audit_reports(ctx) -> dict:
    """Label, rendering and declared-basis audit of every retained report numbered below 100 in the output directory.

    Each finding's rendered row must equal ``report.finding_row`` and have the
    header's four cells, with the label in the third and, in the fourth,
    exactly the Basis cell its declared basis prescribes (``_expected_basis``,
    built without the renderer); a finding whose basis declares a generator of
    synthetic inputs or an executed provider must name it there.
    """
    directory = ctx.output_dir / "reports"
    rows, labels, rendering, basis, hidden, audited = [], [], [], [], [], []
    for path in sorted(directory.glob("T*.json")) if directory.is_dir() else []:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            labels.append(f"{path.name}: unreadable ({type(exc).__name__})")
            continue
        if not isinstance(report.get("number"), int) or report["number"] >= 100:
            continue
        try:
            validate_report(report)
        except EvidenceRefusal as exc:
            labels.append(f"{path.name}: refused by validate_report ({exc})")
            continue
        markdown = render_markdown(report)
        table = markdown.split(FINDINGS_TABLE, 1)
        lines = table[1].splitlines() if len(table) == 2 else []
        for index, record in enumerate(report["findings"]):
            label, domain = record["evidence_status"], record["domain"]
            where = f"{report['task_id']}[{index}]"
            if label not in LABELS:
                labels.append(f"{where}: unknown label {label}")
            if domain in PHYSICAL_DOMAINS and (label in _COMPUTATIONAL_LABELS or
                                               (label != "not_established" and not record["basis"].get("acquisition"))):
                labels.append(f"{where}: physical finding labelled {label}")
            if domain in AUTHORITY_DOMAINS and label != "not_established":
                labels.append(f"{where}: authority finding labelled {label}")
            if domain not in PHYSICAL_DOMAINS and label == "hardware_measured":
                labels.append(f"{where}: computational finding labelled hardware_measured")
            row = lines[index] if index < len(lines) else ""
            cells = _cell_texts(row)
            if row != finding_row(record) or len(cells) != HEADER_CELLS or cells[2] != f"`{label}`":
                rendering.append(f"{where}: rendered row does not show `{label}` in its column")
            missing = _basis_problems(record, cells[3]) if len(cells) == HEADER_CELLS else ["basis column"]
            if missing:
                basis.append(f"{where}: rendered Basis column differs from the declared basis ({', '.join(missing)})")
            if set(DECLARED_SOURCES) & set(finding_origin(record)) and set(missing) & {"generator", "provider",
                                                                                     "basis column"}:
                hidden.append(where)
        if f"`{report['physical_validation_status']['status']}`" not in markdown:
            rendering.append(f"{report['task_id']}: physical validation status not rendered")
        audited += report["findings"]
        rows.append({"task_id": report["task_id"], "findings": len(report["findings"]),
                     "pipe_claims": sum("|" in f["claim"] or "\n" in f["claim"] for f in report["findings"]),
                     "labels": sorted({f["evidence_status"] for f in report["findings"]}),
                     "declared_sources": sorted({item for f in report["findings"] for item in finding_origin(f)
                                                 if item in DECLARED_SOURCES}),
                     "physical_or_authority": sum(f["domain"] in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS
                                                  for f in report["findings"])})
    declaring = {item: sum(item in finding_origin(record) for record in audited) for item in DECLARED_SOURCES}
    return {"rows": rows, "labels": labels, "rendering": rendering, "basis": basis, "hidden_sources": hidden,
            "declaring": declaring, "label_by_basis": origin_counts(audited), "findings": len(audited)}


def _energy_data(client, bundle_id) -> dict:
    data = client.ok("bundle.get", {"bundle_id": bundle_id})["steps"][0]["result"]["data"]
    return {"origin": data["origin"], "hardware_provenance": data["hardware_provenance"],
            "classification": data["comparison"]["classification"], "eligible": data["comparison"]["eligible"]}


def _energy_origin_study():
    from .. import energy_records
    from ..instruments import make_demo_run
    from ..session import Session
    from ..telemetry import canonical
    log = json.loads(example_bytes("energy-accuracy/baseline.json"))
    synthetic = energy_records.analyze(log)
    relabelled = dict(log, origin="physical_measurement")
    try:
        energy_records.validate_log(relabelled)
        unsealed = "accepted"
    except ValueError as exc:
        unsealed = str(exc)
    resealed = energy_records.seal({k: v for k, v in relabelled.items() if k != "log_digest"})
    physical = energy_records.analyze(resealed)
    fresh_data = None
    with tempfile.TemporaryDirectory(prefix="ciw-lab-t100-") as scratch:
        client = Client(Session(make_demo_run(), Path(scratch)))
        first = client.ok("source.add", source_payload("energy-accuracy", canonical(log), "synthetic"))
        client.ok("operation.execute", {"operation_id": "ciw.energy-accuracy.v1",
                                        "parameters": {"source_id": first["source_id"]}})
        second = client.ok("source.add", source_payload("energy-accuracy", canonical(resealed), "relabelled"))
        collision = _outcome(client.call("operation.execute", {"operation_id": "ciw.energy-accuracy.v1",
                                                               "parameters": {"source_id": second["source_id"]}}))
        fresh = energy_records.seal({k: v for k, v in dict(relabelled, run_id="energy-run-" + "2" * 32).items()
                                     if k != "log_digest"})
        third = client.ok("source.add", source_payload("energy-accuracy", canonical(fresh), "fresh occurrence"))
        response = client.call("operation.execute", {"operation_id": "ciw.energy-accuracy.v1",
                                                     "parameters": {"source_id": third["source_id"]}})
        fresh_outcome = _outcome(response)
        if fresh_outcome == "accepted":
            # The retained bundle's own classification, as a workbench reader sees it.
            fresh_data = _energy_data(client, response["payload"]["bundle_id"])
    return {"synthetic": {key: synthetic[key] for key in ("origin", "hardware_provenance")}
            | {"classification": synthetic["comparison"]["classification"],
               "eligible": synthetic["comparison"]["eligible"]},
            "unsealed_relabel": unsealed,
            "resealed_relabel": {key: physical[key] for key in ("origin", "hardware_provenance")}
            | {"classification": physical["comparison"]["classification"], "eligible": physical["comparison"]["eligible"]},
            "same_occurrence_collision": collision, "fresh_occurrence": fresh_outcome, "fresh_bundle": fresh_data}


FABRICATED_VALUES = [0, 1, 2, 3, 0]
FABRICATED_EXPERIMENT = "ciw-lab-t100-fabricated-heat"


def _classified_bundle(source_tree) -> dict:
    """A fabricated, content-consistent numerical-heat bundle sealed with ``source_tree``, as CIW readers see it.

    The bundle is saved in a workspace, then read back through the workbench
    protocol and through ``ciw.lab.bridge.classify_workspace`` (``ciw lab
    classify``), which validates the workspace like ``Session.from_workspace``
    and compares the retained runtime identity with the pins CIW declares.
    """
    from ..declared_workload import PINS
    from ..instruments import make_demo_run
    from ..proved_heat import PIN as PROVED_HEAT_PIN
    from ..workbench import Workbench
    from .bridge import classify_workspace
    Session = _session_class()
    fabricated = fabricated_heat_catalog(FABRICATED_VALUES, experiment_id=FABRICATED_EXPERIMENT,
                                         source_tree=source_tree)
    bundle_id = fabricated["bundles"][0]["bundle_id"]
    with tempfile.TemporaryDirectory(prefix="ciw-lab-t100-provider-") as scratch:
        scratch = Path(scratch)
        session = Session(make_demo_run(), scratch / "session")
        session.workbench = Workbench.restore(fabricated)
        path = session.save_workspace(scratch / "workspace.json")
        runtime = Client(session).ok("bundle.get", {"bundle_id": bundle_id})["runtimes"]["scr"]
        try:
            classification = classify_workspace(path)
            outcome = "accepted"
        except ValueError as exc:
            classification, outcome = None, str(exc)
    labels, pins, notes = [], [], None
    for item in (classification or {}).get("items", []):
        if item["kind"] == "workbench_bundle" and item["identity"] == bundle_id:
            numerical = [record for record in item["findings"] if record["claim"].endswith("numerical result")]
            labels = [record["evidence_status"] for record in numerical]
            notes = numerical[0]["basis"].get("notes") if numerical else None
            pins = [{key: row.get(key) for key in ("role", "matched", "problem", "tree_pinned")}
                    for row in item.get("runtime_pins", [])]
    return {"reopen": outcome, "numerical_labels": labels, "runtime_pins": pins, "classifier_notes": notes,
            "reader_view": {"revision_is_ciw_pin": runtime["revision"] == PINS["numerical-heat"]["revision"],
                            "source_tree_is_ciw_pin": runtime["source_tree"] == PROVED_HEAT_PIN["source_tree"],
                            "engine_source_binding": runtime["engine"]["source_binding"],
                            "adapter_version": runtime["adapter_version"],
                            "repository_root": runtime["repository_root"]},
            "bundle_id": bundle_id}


def _provider_origin_study() -> dict:
    """T092's fabricated heat bundle with an invented source tree, and the same bundle sealed with CIW's pinned tree.

    ``fabricated``: tree ``'0' * 40`` (not CIW's pin); ``copied_pin``: the
    tree ``ciw.proved_heat.PIN`` records for the pinned SCR revision, which is
    a public constant any fabricator can copy.
    """
    from ..proved_heat import PIN as PROVED_HEAT_PIN
    return {"fabricated": _classified_bundle(FABRICATED_SOURCE_TREE),
            "copied_pin": _classified_bundle(PROVED_HEAT_PIN["source_tree"])}


def _kinds_without_tree() -> list:
    """Workflow kinds with a pin whose revision has no source tree in any CIW pin table (the classifier accepts any
    tree there)."""
    from .bridge import pins_without_tree
    return sorted({kind for kind, _, _ in pins_without_tree()["kinds"]})


def _free_energy_study():
    from .. import free_energy_view
    from ..free_energy_profile import POLICY, SOURCE_SCHEMA, validate_source
    keys = ("schema", "experiment_id", "configuration", "geometry", "coordinates", "sensors", "assumed_model",
            "generator", "samples", "representative_index", "solver")
    outcomes = {}
    for label, configuration in (("declared synthetic policy", dict(POLICY)),
                                 ("observations relabelled physical", dict(POLICY, observations="physical_measurement")),
                                 ("physical validation claimed", dict(POLICY, physical_validation="established"))):
        source = {key: None for key in keys}
        source.update(schema=SOURCE_SCHEMA, configuration=configuration)
        try:
            validate_source(json.dumps(source).encode())
            outcomes[label] = "accepted"
        except ValueError as exc:
            outcomes[label] = str(exc)
    # Static check: the retained-truth panel is labelled as a synthetic reference.
    tree = ast.parse(inspect.getsource(free_energy_view))
    literals = sorted({node.value.value for node in ast.walk(tree) if isinstance(node, ast.keyword)
                       and node.arg == "basis" and isinstance(node.value, ast.Constant)})
    return {"policy_outcomes": outcomes, "view_basis_literals": literals,
            "policy_observations": POLICY["observations"], "policy_physical_validation": POLICY.get("physical_validation")}


def _rendering_probe():
    """Render a synthetic claim containing a pipe; returns the finding, its rendered row and whether the label kept
    its column (the header's cell count, with the label in the third cell)."""
    from .registry import load_queue
    witness = finding("claim with a | pipe", "numerical", 1.0, {"generator": {"name": "rendering probe"}})
    task_record = next(item for item in load_queue()["tasks"] if item["id"] == "T100")
    row = render_markdown(build_report(task_record, "partial", {}, [witness])).splitlines()[-1]
    cells = _cell_texts(row)
    kept = len(cells) == HEADER_CELLS and cells[2] == f"`{witness['evidence_status']}`"
    return witness, row, kept


KEEPS_LABEL_COLUMN = "render_markdown keeps the label column for a claim containing a pipe character"


def _probe_finding(witness, row, kept):
    """Records whichever renderer behaviour was observed, so the finding holds before and after escaping lands.

    The label-column claim is checked unless the row shows the extra cell a
    raw pipe creates: then the shift is recorded as a counterexample. A row
    that lost its label column without that extra cell (cells reordered, say)
    refutes the label-column claim and witnesses no counterexample, so no
    catalogue of counterexamples (T157, T160-T162) lists it.
    """
    extra = _cells(row) - HEADER_CELLS
    if kept or extra < 1:
        return finding(KEEPS_LABEL_COLUMN, "computational_pipeline",
                       {"row_cells": _cells(row), "label_column_shifted": not kept},
                       {"checks": [_check(f"rendered cells beyond the header's {HEADER_CELLS} for a claim containing "
                                          "a pipe", extra),
                                   _check("rendered rows whose third cell is not the finding's label", int(not kept)),
                                   _check("rendered row differing from report.finding_row", int(row != finding_row(witness)))]},
                       uncertainty=EXACT_COUNT, tolerance=EXACT)
    return finding("An unescaped pipe character in a finding claim shifts the rendered label out of its Markdown column",
                   "computational_pipeline", {"row_cells": _cells(row), "label_column_shifted": True},
                   {"checks": [_check(f"rendered cells beyond the header's {HEADER_CELLS} for a claim containing a pipe",
                                      extra, 1, "ge", "invariant")]},
                   uncertainty=EXACT_COUNT, tolerance=EXACT, counterexample={
                       "statement": "render_markdown keeps every finding's label in the label column for any claim text",
                       "witness": {"claim": witness["claim"], "row": row}})


@task("T100", changed_files=(MODULE, FIXTURES),
      regression_tests=(f"{TESTS}::test_t100_labels_and_origins_stay_distinct",
                        f"{TESTS}::test_t100_flags_a_retained_report_whose_label_leaves_its_column",
                        f"{TESTS}::test_t100_flags_a_rendered_row_that_hides_its_generator_or_provider",
                        f"{TESTS}::test_t100_classifier_study_separates_an_invented_tree_from_a_copied_pin",
                        f"{TESTS}::test_t100_flags_an_unaccepted_acquisition_shown_as_hardware_acquisition",
                        f"{TESTS}::test_basis_audit_compares_the_whole_cell",
                        f"{TESTS}::test_render_markdown_pipe_probe_matches_the_renderer",
                        f"{TESTS}::test_next_steps_name_forward_work"))
def visibly_distinct_results(ctx):
    audit = _audit_reports(ctx)
    rows, violations, rendering, basis = audit["rows"], audit["labels"], audit["rendering"], audit["basis"]
    hidden, declaring = audit["hidden_sources"], audit["declaring"]
    energy = _energy_origin_study()
    free = _free_energy_study()
    provider = _provider_origin_study()
    fabricated, copied = provider["fabricated"], provider["copied_pin"]
    # Validator self-test: a physical finding relabelled with a computational label is refused.
    physical = finding("Synthetic fixture reflects real device energy", "physical", 1.0,
                       {"generator": {"name": "synthetic fixture"}})
    try:
        validate_finding(dict(physical, evidence_status="synthetic"))
        forged = "accepted"
    except EvidenceRefusal as exc:
        forged = str(exc)
    witness, row, kept = _rendering_probe()
    pipe_claims = sum(r.get("pipe_claims", 0) for r in rows)
    present = [r["task_id"] for r in rows]
    ctx.artifact_json("visibility-audit.json", {"reports": rows, "task_ids_audited": present,
                                                "reports_expected_for_the_full_queue": EXPECTED_EARLIER_REPORTS,
                                                "label_violations": violations, "rendering_violations": rendering,
                                                "basis_violations": basis, "hidden_generator_or_provider": hidden,
                                                "energy": energy, "free_energy": free, "provider_origin": provider,
                                                "forged_relabel": forged,
                                                "rendering_probe_row": row})
    # Label x declared basis component over every audited finding (a finding counts under each component it declares).
    ctx.artifact_json("label-by-basis.json", {
        "findings": audit["findings"], "task_ids_audited": present, "counts": audit["label_by_basis"],
        "reading": "counts[label][component]: findings with that label declaring that basis component "
                   "(ciw.lab.evidence.origin_counts); 'none' counts findings declaring no component. The label is "
                   "decided by the rules; the components show what it rests on."})
    findings = []
    audited = bool(rows)
    coverage = {"reports_checked": len(rows), "reports_absent_of_T001_T099": EXPECTED_EARLIER_REPORTS - len(rows)}
    if audited:
        findings.append(finding(
            "Every earlier report present in the output directory when T100 runs keeps labels among the seven and "
            "physical and authority findings unestablished without acquisition", "computational_pipeline",
            dict(coverage, label_violations=len(violations)),
            {"checks": [_check("label or domain violations", len(violations))]},
            uncertainty=EXACT_COUNT, tolerance=EXACT))
        findings.append(finding(
            "Every finding of those reports shows its label in the label column of its rendered Markdown row",
            "computational_pipeline", dict(coverage, rendering_violations=len(rendering)),
            {"checks": [_check("findings whose rendered row differs from report.finding_row or whose label leaves "
                               "its column", len(rendering))]},
            uncertainty=EXACT_COUNT, tolerance=EXACT))
        findings.append(finding(
            "Every finding of those reports shows exactly its declared basis in the Basis column of its rendered "
            "Markdown row, an acquisition record as hardware acquisition only on a physical finding it establishes",
            "computational_pipeline", dict(coverage, basis_violations=len(basis)),
            {"checks": [_check("findings whose rendered Basis column differs from the cell their declared basis "
                               "prescribes (docs/lab/AUTHORING.md)", len(basis))]},
            uncertainty=EXACT_COUNT, tolerance=EXACT))
        if any(declaring.values()):
            findings.append(finding(
                "Every finding of those reports whose basis declares a generator of synthetic inputs or an executed "
                "provider names that generator or provider beside its label", "computational_pipeline",
                dict(coverage, generator_findings=declaring["synthetic_inputs"],
                     provider_findings=declaring["provider"], not_shown=len(hidden)),
                {"checks": [_check("findings declaring a generator or an executed provider whose rendered row does "
                                   "not name it", len(hidden))]},
                uncertainty=EXACT_COUNT, tolerance=EXACT))
        else:
            # Nothing to test: the visibility claim is recorded as unestablished, not the (counted) absence.
            findings.append(finding(
                "Every finding of those reports whose basis declares a generator of synthetic inputs or an executed "
                "provider names it beside its label (no audited finding declares one)",
                "computational_pipeline", dict(coverage, generator_findings=0, provider_findings=0), {},
                expected_not_established=True))
    else:
        findings.append(finding(
            "No earlier report was present in the output directory to audit", "computational_pipeline",
            coverage, {}, expected_not_established=True))
    fresh = energy["fresh_bundle"] or {}

    def pinned(study):
        return sum(row["matched"] is not None for row in study["runtime_pins"])

    findings += [
        finding("A physical finding relabelled with a computational label is refused by the evidence validator",
                "computational_pipeline", forged,
                {"checks": [_refusal("validate_finding on a physical finding relabelled synthetic",
                                     "Evidence label refused: basis supports not_established, finding states synthetic",
                                     forged)]}, uncertainty=EXACT_COUNT, tolerance=EXACT),
        _probe_finding(witness, row, kept),
        finding("CIW keeps the synthetic energy fixture synthetic_only and refuses an unsealed relabel to physical",
                "computational_pipeline", {"synthetic": energy["synthetic"], "unsealed_relabel": energy["unsealed_relabel"],
                                           "same_occurrence_relabel": energy["same_occurrence_collision"]},
                {"checks": [_refusal("validate_log after changing origin without resealing",
                                     "Retained log digest differs", energy["unsealed_relabel"]),
                            _refusal("workbench execute of a resealed relabel of a retained occurrence",
                                     "invalid_payload: Identity collision across retained workbench artifacts",
                                     energy["same_occurrence_collision"]),
                            _check("synthetic fixture classified other than synthetic_only",
                                   int(energy["synthetic"]["classification"] != "synthetic_only")),
                            _check("synthetic fixture eligible for a physical comparison",
                                   int(energy["synthetic"]["eligible"]))]}, uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("A resealed relabel of the synthetic energy fixture under a fresh occurrence is accepted by the "
                "workbench and classified as a physical-domain measurement", "computational_pipeline",
                {"fresh_occurrence_execute": energy["fresh_occurrence"], "fresh_bundle": energy["fresh_bundle"]},
                {"checks": [_refusal("workbench execute of a fresh-occurrence reseal", "accepted",
                                     energy["fresh_occurrence"]),
                            _check("fresh-occurrence bundle classified other than physical_domain_measurement",
                                   int(fresh.get("classification") != "physical_domain_measurement")),
                            _check("fresh-occurrence bundle whose origin is not physical_measurement",
                                   int(fresh.get("origin") != "physical_measurement")),
                            _check("fresh-occurrence bundle whose hardware provenance is not "
                                   "retained_operator_record_not_authenticated",
                                   int(fresh.get("hardware_provenance") != "retained_operator_record_not_authenticated"))]},
                uncertainty=EXACT_COUNT, tolerance=EXACT, counterexample={
                    "statement": "CIW energy records can distinguish a genuinely acquired log from a relabelled "
                                 "synthetic fixture",
                    "witness": {"origin": "physical_measurement (declared)", "run_id": "energy-run-" + "2" * 32,
                                "classification": fresh.get("classification"),
                                "hardware_provenance": fresh.get("hardware_provenance")}}),
        finding("CIW refuses free-energy sources that relabel synthetic observations as physical or claim physical "
                "validation", "computational_pipeline",
                {"outcomes": free["policy_outcomes"], "truth_panel_basis": free["view_basis_literals"]},
                {"checks": [_refusal("observations relabelled physical", "Unsupported free-energy source or authority policy",
                                     free["policy_outcomes"]["observations relabelled physical"]),
                            _refusal("physical validation claimed", "Unsupported free-energy source or authority policy",
                                     free["policy_outcomes"]["physical validation claimed"]),
                            _check("declared synthetic policy refused at the policy check",
                                   int(free["policy_outcomes"]["declared synthetic policy"]
                                       == "Unsupported free-energy source or authority policy")),
                            _check("view basis literals naming the synthetic reference",
                                   int("synthetic_reference_not_hardware_measurement" in free["view_basis_literals"]),
                                   1, "ge", "invariant")]}, uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("The workspace classifier labels a fabricated, content-consistent numerical-heat bundle whose source "
                "tree is not CIW's pinned tree not_established", "computational_pipeline",
                {"reopen": fabricated["reopen"], "numerical_labels": fabricated["numerical_labels"],
                 "runtime_pins": fabricated["runtime_pins"], "reader_view": fabricated["reader_view"]},
                {"checks": [_refusal("classify_workspace (Session.from_workspace validation) of the saved fabricated "
                                     "bundle", "accepted", fabricated["reopen"]),
                            _check("fabricated bundle numerical findings labelled other than not_established",
                                   sum(label != "not_established" for label in fabricated["numerical_labels"])
                                   + int(not fabricated["numerical_labels"])),
                            _check("runtime identities of the fabricated bundle compared with CIW's pins",
                                   len(fabricated["runtime_pins"]), 1, "ge", "invariant"),
                            _check("runtime identities of the fabricated bundle matched to a CIW pin",
                                   pinned(fabricated))]},
                uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("A fabricated numerical-heat bundle sealed with CIW's pinned SCR revision and source tree reopens and "
                "is labelled provider_backed by the workspace classifier, as a provider result is",
                "computational_pipeline",
                {"reopen": copied["reopen"], "numerical_labels": copied["numerical_labels"],
                 "runtime_pins": copied["runtime_pins"], "reader_view": copied["reader_view"]},
                {"checks": [_refusal("classify_workspace (Session.from_workspace validation) of the saved bundle "
                                     "sealed with the pinned tree", "accepted", copied["reopen"]),
                            _check("copied-pin bundle numerical findings not labelled provider_backed",
                                   sum(label != "provider_backed" for label in copied["numerical_labels"])
                                   + int(not copied["numerical_labels"]))]},
                uncertainty=EXACT_COUNT, tolerance=EXACT, counterexample={
                    "statement": "CIW's retained records and their classification keep provider-backed results "
                                 "visibly distinct from fabricated ones",
                    "witness": {"bundle_id": copied["bundle_id"], "values": FABRICATED_VALUES,
                                "classified": copied["numerical_labels"],
                                "matched_pins": sorted({pin for row in copied["runtime_pins"]
                                                        for pin in row["matched"] or []}),
                                **copied["reader_view"]}}),
        finding("The relabelled energy log is a physical GPU energy measurement", "physical",
                "not established: origin is operator-declared and unauthenticated", {}),
        finding("The synthetic energy fixture characterizes real NVML counter accuracy", "sensor_performance",
                "not established: synthetic fixture", {}),
    ]
    counts = audit["label_by_basis"]
    tree_free = _kinds_without_tree()
    fields = _fields(
        "Retained lab reports keep synthetic, provider-backed and physical results visibly distinct: every rendered "
        "finding shows its label and, beside it, the basis it declares (the generator of synthetic inputs, the "
        "executed provider, an acquisition); CIW's own records refuse a synthetic-to-physical relabel wherever the "
        "record can detect it; and the workspace classifier tells a provider result from a fabricated, "
        "content-consistent one exactly as far as its comparison with CIW's pins reaches (T092's counterexample).",
        "Allowed labels per domain: physical -> {not_established, hardware_measured, independently_verified with "
        "acquisition}; authority -> {not_established}; computational -> never hardware_measured. Rendered row = "
        "report.finding_row: claim | value | label | declared basis. The Basis cell is compared whole with the cell "
        "docs/lab/AUTHORING.md prescribes, built here without describe_basis: each declared component in basis "
        "order, a generator with its name and seed, a provider as repository@revision, and an acquisition record as "
        "hardware acquisition (device) only on a physical finding labelled hardware_measured or "
        "independently_verified, as declared acquisition record (not accepted) on every other.",
        ["retained reports ctx.output_dir/reports/T001-T099 (only those present when T100 runs)",
         "examples/energy-accuracy/baseline.json (synthetic, embedded)", "ciw.free_energy_profile.POLICY",
         "ciw.free_energy_view source text",
         "fabricated numerical-heat catalog (values [0, 1, 2, 3, 0], fabricated runtime identity at the CIW revision "
         "pin), sealed once with an invented source tree and once with ciw.proved_heat.PIN's tree"],
        "validate_report, per-finding label/domain rules, rendered Markdown rows compared with report.finding_row "
        "and their Basis cells with the declared basis; CIW analysis/refusal outputs and the retained bundle of a "
        "fresh-occurrence relabel; the workbench and classify_workspace views (labels and runtime_pins rows) of "
        "two fabricated provider-kind bundles.",
        "Zero label, rendering and basis violations; every generator or executed provider a finding declares named "
        "beside its label; relabels refused where detectable; the invented-tree bundle not_established; physical "
        "claims not established.",
        "Audit every retained report below T100 present in the output directory (labels, rendered rows, Basis "
        "cells, label x basis counts); forge a relabelled physical finding; render a claim containing a pipe "
        "character; relabel the energy fixture unsealed, resealed in the same occurrence and resealed in a fresh "
        "occurrence (reading the retained bundle's classification); relabel free-energy source policies; inspect the "
        "free-energy truth panel basis; save a fabricated numerical-heat bundle with an invented source tree and "
        "again with CIW's pinned tree, and read each through bundle.get and classify_workspace.",
        f"{len(rows)} of {EXPECTED_EARLIER_REPORTS} earlier reports present and audited ({audit['findings']} "
        f"findings): {len(violations)} label violations, {len(rendering)} rendering violations, {len(basis)} basis "
        f"violations; findings declaring synthetic inputs: {declaring['synthetic_inputs']}, an executed provider: "
        f"{declaring['provider']}, either one not named beside the label: {len(hidden)}; label x basis counts in "
        f"label-by-basis.json (numerically_verified findings declaring synthetic inputs: "
        f"{counts['numerically_verified']['synthetic_inputs']}, a provider run: "
        f"{counts['numerically_verified']['provider']}); the pipe probe "
        f"{'kept' if kept else 'lost'} its label column; fresh-occurrence energy relabel "
        f"{energy['fresh_occurrence']} and classified {fresh.get('classification')}; same-occurrence relabel: "
        f"{energy['same_occurrence_collision']}; fabricated heat bundle reopen {fabricated['reopen']}, classified "
        f"{fabricated['numerical_labels']} with an invented tree and {copied['numerical_labels']} with the pinned "
        "tree.",
        "Exact.",
        ["unknown label", "physical finding with computational label", "authority finding established",
         "label missing from rendered row", "rendered row differing from report.finding_row",
         "declared basis, generator or provider missing from the rendered row",
         "a component omitted but named inside a declared identity", "an unaccepted acquisition shown as hardware "
         "acquisition",
         "origin relabel with and without resealing", "occurrence rebinding", "free-energy policy relabel",
         "fabricated provider-kind bundle classified as provider-backed",
         "classifier accepting a runtime tree that is not CIW's pin"],
        ["The report audit covers only reports present in the output directory when T100 runs: a section-only run "
         "audits its own section, a full run T001-T099; stale reports from earlier runs in the same directory are "
         "included if present (task ids in visibility-audit.json).",
         f"Retained claims containing a pipe or newline: {pipe_claims}; the rendering probe uses a synthetic claim.",
         "The Basis column shows the components and identities a finding declares; they are as recorded, not "
         "authenticated (a generator name or provider repository is text in the basis).",
         "The audit sees only the components a finding declares: a finding that rests on a generator or an executed "
         "provider without declaring it in its basis is counted under the components it does declare, and nothing "
         "here detects the omission. One provider slot shows one provider of a result several providers produced "
         "(T097's producer roundtrip names the SET checker; the PPDA and SCR producers are in its basis notes).",
         "classify_workspace compares a retained runtime identity with the pins CIW declares "
         "(ciw.lab.bridge.declared_pins), so the invented tree is caught for numerical-heat, whose pinned revision "
         "has a tree CIW records (ciw.proved_heat.PIN); for the kinds in ciw.lab.bridge.pins_without_tree "
         f"({', '.join(tree_free)}) any tree is accepted (tree_pinned: false), so an invented tree at a pinned "
         "revision is not caught there. Session.from_workspace (reopen) does not compare it. The pins are public "
         "constants and workspace seals are unkeyed, so the bundle sealed with the pinned tree is labelled "
         "provider_backed: nothing the classifier compares tells it from a provider result. Its adapter_version and "
         "repository_root name the fabrication only because this lab wrote them so (reader_view in "
         "visibility-audit.json)."],
        NEXT_STEPS["T100"])
    return {"state": _settle("completed" if audited else "partial", findings), "fields": fields, "findings": findings}
