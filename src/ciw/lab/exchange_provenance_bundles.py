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
correctness (a fabricated heat bundle passes reopen validation); a pinned
provider result is not independent verification by another party; a matching
checkout digest does not authenticate the upstream repository or toolchain; the
energy fixtures are synthetic and nothing here measures energy, heat or any
physical quantity.
"""
from __future__ import annotations

import ast
import base64
from copy import deepcopy
from hashlib import sha256
import inspect
import json
from pathlib import Path
import subprocess
import tempfile

from .. import __version__
from .evidence import AUTHORITY_DOMAINS, LABELS, PHYSICAL_DOMAINS, EvidenceRefusal, finding, validate_finding
from .registry import task
from .report import build_report, render_markdown, validate_report
from .svg import line_plot
from .exchange_provenance_bundles_fixtures import (EXECUTION_PATHS, GOLDEN_MANIFEST, Client, ExecutionForbidden,
                                                   build_energy_session, example_bytes, execution_guard,
                                                   fabricated_heat_catalog, fixture_root, heat_reference,
                                                   malformed_fixtures, merge_catalogs, source_payload)
from . import exchange_provenance_bundles_providers as providers

MODULE = "src/ciw/lab/exchange_provenance_bundles.py"
FIXTURES = "src/ciw/lab/exchange_provenance_bundles_fixtures.py"
PROVIDERS = "src/ciw/lab/exchange_provenance_bundles_providers.py"
TESTS = "tests/test_lab_exchange_provenance_bundles.py"
EXACT = {"abs": 0, "rel": 0}
UNBOUND = "No trusted repositories bound for this workbench workflow"
# Registered workflow kinds whose provider is never bound by this section, with
# an embedded source each can retain before execution is refused.
UNBOUND_SOURCES = (
    ("numerical-heat", "declared-workloads/numerical-heat.json"),
    ("proved-heat", "proved-heat/source.json"),
    ("flat-torus-reference", "geodesic-reference/flat-torus.json"),
    ("curved-path-transfer", "geodesic-reference/curved-path.json"),
    ("translation-flow", "geometry-research/translation-flow.json"),
)
PROVIDER_PATHS = frozenset(label for label in (
    "subprocess.Popen", "ciw.adapters.subprocess._bounded_process",
    "ciw.adapters.subprocess.PinnedSubprocessAdapter.__init__", "ciw.candidate_evidence._bounded_process",
    "ciw.declared_workload.DeclaredWorkflow._adapters", "ciw.declared_workload.DeclaredWorkflow._step",
    "ciw.declared_workload.DeclaredWorkflow._execute", "ciw.declared_workload.DeclaredWorkflow.create_session",
    "ciw.declared_workload.DeclaredWorkflow.replay_session"))


def _check(reference, observed, tolerance=0.0, comparison="abs_le", kind="exact_arithmetic"):
    observed, tolerance = float(observed), float(tolerance)
    holds = {"abs_le": abs(observed) <= tolerance, "le": observed <= tolerance, "ge": observed >= tolerance}[comparison]
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": holds}


def _refusal(reference, expected, observed):
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed, "passed": observed == expected}


def _fields(hypothesis, model, inputs, observation, invariant, experiment, result, uncertainty, failures,
            assumptions, next_task):
    return {"hypothesis": hypothesis, "mathematical_model": model, "input_data": inputs,
            "observation_model": observation, "expected_invariant": invariant, "experiment": experiment,
            "numerical_result": result, "uncertainty": uncertainty, "failure_modes_checked": failures,
            "unresolved_assumptions": assumptions, "recommended_next_task": next_task}


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
    with tempfile.TemporaryDirectory(prefix="ciw-lab-t091-") as scratch:
        scratch = Path(scratch)
        session, _, ids = build_energy_session(scratch / "original")
        path = session.save_workspace(scratch / "saved" / "workspace.json")
        saved_bytes = path.read_bytes()
        reads = {}
        with execution_guard() as guard:
            reopened = Session.from_workspace(path, scratch / "reopened")
            reader = Client(reopened)
            for kind, payload in (("session.get", {}), ("source.list", {}), ("bundle.list", {}), ("result.list", {}),
                                  ("execution.list", {}), ("operation.list", {}),
                                  ("experiment.inspect", {"bundle_id": ids["original"]})):
                reads[kind] = reader.call(kind, payload)["type"]
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
        unchanged = path.read_bytes() == saved_bytes
    observed = {"workspace_version": saved["workspace_version"], "bundles": len(saved["workbench"]["bundles"]),
                "results": len(saved["results"]), "executions": len(saved["executions"]),
                "execution_path_calls": len(reopen_attempts), "provider_bindings": bindings,
                "available_operations": available, "read_requests": len(reads),
                "read_requests_answered": sum(value == "response" for value in reads.values())}
    ctx.artifact_json("reopen.json", {"observed": observed, "reads": reads, "mismatches": mismatches,
                                      "reopen_attempts": reopen_attempts, "recomputations": recomputations,
                                      "guard_control": control, "binding_keys_in_saved_workspace": binding_keys,
                                      "saved_workspace_sha256": sha256(saved_bytes).hexdigest(),
                                      "guarded_paths": [".".join(filter(None, p)) for p in EXECUTION_PATHS]})
    analyze_calls = recomputations.get("ciw.energy_records.analyze", 0)
    findings = [
        finding("A saved workspace with oscillator results and an energy-accuracy original and replay reopens "
                "with no provider binding and reaches no execution path", "computational_pipeline", observed,
                {"checks": [_check("execution entry points reached while reopening and reading", len(reopen_attempts)),
                            _check("selection/results/executions/catalog digest mismatches after reopen",
                                   sum(mismatches.values())),
                            _check("provider bindings present after reopen", len(bindings)),
                            _check("binding-like keys in the saved workspace", binding_keys),
                            _check("read requests not answered", len(reads) - observed["read_requests_answered"]),
                            _check("saved workspace bytes changed by reopen", int(not unchanged))]},
                tolerance=EXACT),
        finding("The execution guard intercepts a replay attempted while it is active", "computational_pipeline",
                control, {"checks": [_refusal("Workbench.replay inside execution_guard", "intercepted", control)]},
                tolerance=EXACT),
        finding("Reopening recomputes the retained energy analysis to validate content", "computational_pipeline",
                analyze_calls, {"checks": [_check("ciw.energy_records.analyze calls during reopen and reads",
                                                  analyze_calls, 1, "ge", "invariant")]},
                tolerance=EXACT, counterexample={
                    "statement": "Reopening a saved workspace performs no numerical recomputation",
                    "witness": {"ciw.energy_records.analyze calls": analyze_calls,
                                "new execution occurrences": 0}}),
        finding("After reopen the binding-free energy-accuracy workflow still replays with a numerical match",
                "computational_pipeline", replay_match,
                {"checks": [_check("replay numerical_match is false", int(not replay_match))]}, tolerance=EXACT),
    ]
    fields = _fields(
        "A CIW workspace saved with retained oscillator results and an energy-accuracy original and replay can be "
        "reopened with no provider binding and without reaching any execution entry point.",
        "Reopen = validate(saved JSON) then construct; the guard replaces "
        f"{len(EXECUTION_PATHS)} execution entry points (workflow steps, sessions, "
        "provider adapters, subprocesses, recording operations) with a refusing recorder.",
        ["ciw.instruments.make_demo_run (synthetic analytic oscillator)",
         "examples/energy-accuracy/baseline.json bytes (synthetic_fixture origin), embedded"],
        "Protocol v1 requests against the reopened Session; digests of selection, results, executions and the "
        "retained catalog; guard attempt and recomputation counters.",
        "Zero execution-path calls, zero bindings, identical retained content, answered read requests.",
        "Build a session (stats, selection update, spectrum, statistics.v1 operation, energy source, execute, "
        "replay), save, reopen under the guard, issue read requests, compare digests, run a guard sensitivity "
        "control and a binding-free replay control after the guard.",
        f"{observed['bundles']} bundles, {observed['results']} results and {observed['executions']} execution "
        f"reopened with {len(reopen_attempts)} execution-path calls and bindings {bindings}; "
        f"the energy analysis was recomputed {analyze_calls} times for validation.",
        "Exact: counts and digest equalities; no floating-point tolerance is involved.",
        ["execution entry point reached during reopen", "retained content drift across reopen",
         "binding restored from saved data", "saved workspace rewritten by reopen",
         "guard insensitivity (control replay must be intercepted)"],
        ["The guard covers the execution entry points listed in reopen.json; a new entry point added to CIW "
         "must be added to EXECUTION_PATHS.",
         "Validation recomputation (energy analysis) is counted separately from execution because it creates no "
         "occurrence and no result; the counterexample finding records it."],
        "T092: refuse replay of provider kinds without a binding in the reopened session.")
    return {"state": "completed", "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T092

def _unbound_session(scratch: Path):
    """Reopened session holding energy bundles, unbound-kind sources and a fabricated heat bundle."""
    Session = _session_class()
    session, client, ids = build_energy_session(scratch / "original")
    sources = {}
    for kind, name in UNBOUND_SOURCES:
        sources[kind] = client.ok("source.add", source_payload(kind, example_bytes(name), f"Unbound {kind} source"))
    path = session.save_workspace(scratch / "saved" / "workspace.json")
    workspace = json.loads(path.read_text(encoding="utf-8"))
    fabricated = fabricated_heat_catalog([0, 1, 2, 3, 0])
    workspace["workbench"] = merge_catalogs(workspace["workbench"], fabricated)
    path.write_text(json.dumps(workspace, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with execution_guard() as guard:
        reopened = Session.from_workspace(path, scratch / "reopened")
    ids["heat"] = fabricated["bundles"][0]["bundle_id"]
    return reopened, ids, sources, fabricated, list(guard["attempts"])


def _replay_refusal_cases(ids, sources):
    heat_source = sources["numerical-heat"]["source_id"]
    cases = [(f"execute unbound {kind}", "operation.execute",
              {"operation_id": f"ciw.{kind}.v1", "parameters": {"source_id": sources[kind]["source_id"]}},
              f"operation_unavailable: {UNBOUND}") for kind, _ in UNBOUND_SOURCES]
    cases += [
        ("replay retained numerical-heat bundle without SCR binding", "bundle.replay", {"bundle_id": ids["heat"]},
         f"operation_unavailable: {UNBOUND}"),
        ("replay with client-supplied repositories", "bundle.replay",
         {"bundle_id": ids["heat"], "repositories": {"scr": "/client/scr", "engine": "/client/engine"}},
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
        ("candidate inspection of an energy bundle without an ESM adapter", "operation.execute",
         {"operation_id": "esm.inspect-candidate.v1",
          "parameters": {"bundle_id": ids["original"], "inspected_at": "2026-09-23T00:00:00Z"}},
         "invalid_payload: ESM requires an explicitly selected calibrated or telemetry bundle"),
        ("unregistered recording operation", "operation.execute", {"operation_id": "ciw.lab-unregistered.v1"},
         "operation_unavailable: No trusted provider bound for ciw.lab-unregistered.v1"),
    ]
    return cases


@task("T092", changed_files=(MODULE, FIXTURES),
      regression_tests=(f"{TESTS}::test_t092_unbound_replay_and_execution_are_refused",
                        f"{TESTS}::test_fabricated_heat_bundle_is_content_consistent_but_wrong"))
def replay_refusal_without_binding(ctx):
    with tempfile.TemporaryDirectory(prefix="ciw-lab-t092-") as scratch:
        reopened, ids, sources, fabricated, reopen_attempts = _unbound_session(Path(scratch))
        client = Client(reopened)
        outcomes = []
        # Count (without refusing) every execution entry point the refusals reach.
        with execution_guard(refuse=False) as guard:
            for label, kind, payload, expected in _replay_refusal_cases(ids, sources):
                before = len(guard["attempts"])
                observed = _outcome(client.call(kind, payload))
                outcomes.append({"case": label, "request": kind, "expected": expected, "observed": observed,
                                 "entry_points": guard["attempts"][before:]})
        provider_reached = sorted({p for row in outcomes for p in row["entry_points"] if p in PROVIDER_PATHS})
        control = client.call("bundle.replay", {"bundle_id": ids["original"]})
        control_ok = control["type"] == "response" and control["payload"]["replay_receipt"]["numerical_match"] is True
        heat = reopened.workbench.get_bundle(ids["heat"])
    declared = json.loads(example_bytes("declared-workloads/numerical-heat.json"))
    reference = heat_reference(declared["initial_values"], declared["steps"])
    retained = heat["steps"][0]["result"]["data"]["values"]
    mismatched = sum(a != b for a, b in zip(retained, reference))
    ctx.artifact_json("refusals.json", {"cases": outcomes, "provider_paths_reached": provider_reached,
                                        "reopen_attempts": reopen_attempts, "energy_replay_control": _outcome(control)})
    ctx.artifact_json("fabricated-heat-catalog.json", fabricated)
    refused = sum(row["observed"] == row["expected"] for row in outcomes)
    findings = [
        finding("Replay and execution requests for provider kinds without a trusted binding are refused with a "
                "named error and reach no provider process or adapter", "computational_pipeline",
                {"requests": len(outcomes), "refused_as_expected": refused,
                 "codes": sorted({row["observed"].split(":", 1)[0] for row in outcomes})},
                {"checks": [_refusal(row["case"], row["expected"], row["observed"]) for row in outcomes]
                 + [_check("provider process or adapter entry points reached", len(provider_reached)),
                    _check("execution entry points reached while reopening", len(reopen_attempts))]},
                tolerance=EXACT),
        finding("A retained numerical-heat bundle whose values no provider computed passes reopen validation",
                "computational_pipeline",
                {"accepted_on_reopen": True, "retained_values": retained, "reference_values": reference,
                 "mismatched_cells": mismatched},
                {"checks": [_check("cells differing from the independent integer reference", mismatched, 1, "ge",
                                   "invariant")]},
                tolerance=EXACT, counterexample={
                    "statement": "Reopen validation of a retained numerical-heat bundle establishes that its values "
                                 "are the pinned SCR heat-kernel output",
                    "witness": {"bundle_id": ids["heat"], "retained_values": retained, "reference_values": reference,
                                "runtime_repository_root": heat["runtimes"]["scr"]["repository_root"]}}),
        finding("The binding-free energy-accuracy replay succeeds in the same reopened session", "computational_pipeline",
                control_ok, {"checks": [_check("energy replay without numerical match", int(not control_ok))]},
                tolerance=EXACT),
        finding("A content-consistent reopened bundle is acceptable as a verified production result",
                "production_acceptance", "not decided by the workbench", {}),
    ]
    fields = _fields(
        "Without a host-side trusted binding, CIW refuses every replay or execution of a provider workflow with a "
        "named error, and no client or saved value can supply the binding.",
        "Workbench._reserve(kind) requires kind in trusted bindings; bindings are process configuration set only by "
        "Workbench.bind_workflow, never by a protocol request or a saved workspace.",
        [f"embedded example sources for {', '.join(kind for kind, _ in UNBOUND_SOURCES)}",
         "fabricated numerical-heat catalog (values [0, 1, 2, 3, 0], fabricated runtime identity)",
         "energy-accuracy session from examples/energy-accuracy/baseline.json (synthetic)"],
        "Protocol v1 error envelopes (code: message) or refused execution records; counted entry points.",
        "Every case refused with its expected code and text; no provider path reached; the builtin still replays.",
        f"{len(outcomes)} refusal cases on a reopened session, each compared with its expected error text; a "
        "counting guard records which execution entry points each request reached.",
        f"{refused}/{len(outcomes)} refused exactly as expected; provider paths reached: {provider_reached or 'none'}; "
        f"the fabricated heat bundle (values {retained}) reopened although the reference field is {reference}.",
        "Exact string and count comparisons.",
        ["client-supplied repositories in replay/execute payloads", "client binding request",
         "unknown bundle", "unversioned and unregistered operation identities", "ESM without adapter",
         "provider adapter constructed before refusal"],
        ["The five unbound kinds stand for every registered workflow kind except energy-accuracy (the only one "
         "bound by default); all of them pass through Workbench._reserve.",
         "The fabricated bundle's runtime identity is syntactically valid but names no real checkout."],
        "T093: show these refusals leave the saved workspace and in-memory state unchanged.")
    return {"state": "completed", "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T093

def _unchanged_cases(ids, heat_source, revision):
    duplicate = base64.b64encode(b'{"schema": "a", "schema": "b"}').decode()
    return [
        ("execute unbound numerical-heat", "operation.execute",
         {"operation_id": "ciw.numerical-heat.v1", "parameters": {"source_id": heat_source}}),
        ("replay an unknown bundle", "bundle.replay", {"bundle_id": "sha256:" + "0" * 64}),
        ("client binding request", "workflow.bind", {"kind": "numerical-heat"}),
        ("replay with client-supplied repositories", "bundle.replay",
         {"bundle_id": ids["original"], "repositories": {"scr": "/client/scr"}}),
        ("malformed source JSON", "source.add", {"kind": "energy-accuracy", "label": "bad", "bytes_b64": duplicate}),
        ("non-canonical base64 source", "source.add", {"kind": "energy-accuracy", "label": "bad", "bytes_b64": "e31="}),
        ("unknown source kind", "source.add", {"kind": "lab-unknown", "label": "bad", "bytes_b64": "e30="}),
        ("stale selection revision", "selection.update", {"expected_revision": revision + 7, "channel": "q"}),
        ("unknown selection channel", "selection.update", {"expected_revision": revision, "channel": "lab"}),
        ("unknown result", "result.get", {"result_id": "result-" + "0" * 32}),
        ("unknown bundle read", "bundle.get", {"bundle_id": "sha256:" + "1" * 64}),
        ("energy execute with an upstream bundle", "operation.execute",
         {"operation_id": "ciw.energy-accuracy.v1",
          "parameters": {"source_id": "source:none", "upstream_bundle_id": ids["original"]}}),
        ("candidate inspection without adapter", "operation.execute",
         {"operation_id": "esm.inspect-candidate.v1",
          "parameters": {"bundle_id": ids["original"], "inspected_at": "2026-09-23T00:00:00Z"}}),
        ("analysis outside the recording", "analysis.stats", {"interval_s": [5.0, 99.0]}),
        ("workspace save to a client path", "workspace.save", {"path": "/client/workspace.json"}),
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
        before = {"workspace": sha256(path.read_bytes()).hexdigest(), "state": _state(session),
                  "directory": _listing(session.output_dir)}
        outcomes = []
        for label, kind, payload in _unchanged_cases(ids, heat["source_id"], session.selection["revision"]):
            outcomes.append({"case": label, "observed": _outcome(client.call(kind, payload))})
        # A request with an unsupported protocol version never reaches dispatch.
        outcomes.append({"case": "unsupported protocol version", "observed": _outcome(session.handle(
            {"protocol_version": 2, "request_id": "lab-v2", "type": "workspace.save", "payload": {}}))})
        after = {"workspace": sha256(path.read_bytes()).hexdigest(), "state": _state(session),
                 "directory": _listing(session.output_dir)}
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
        workspace_after_record = sha256(path.read_bytes()).hexdigest()
    refused = sum(row["observed"] != "accepted" for row in outcomes)
    ctx.artifact_json("unchanged.json", {"cases": outcomes, "state_parts": sorted(before["state"]),
                                         "changed_parts": changed_parts, "reopen_refusal": reopen_refusal,
                                         "refused_operation": refused_operation, "recorded_parts": recorded,
                                         "new_files": new_files})
    findings = [
        finding("Refused requests leave the saved workspace bytes, the in-memory session state and the session "
                "directory unchanged", "computational_pipeline",
                {"refused_requests": refused, "requests": len(outcomes), "workspace_changed": before["workspace"]
                 != after["workspace"], "state_parts_changed": changed_parts,
                 "directory_changed": before["directory"] != after["directory"]},
                {"checks": [_check("requests not refused", len(outcomes) - refused),
                            _check("saved workspace digest changed", int(before["workspace"] != after["workspace"])),
                            _check("in-memory state parts changed", len(changed_parts)),
                            _check("session directory listing changed",
                                   int(before["directory"] != after["directory"])),
                            _check("re-saved content differs apart from saved_at", int(not content_equal))]},
                tolerance=EXACT),
        finding("A refused reopen of a corrupted workspace creates no output directory", "computational_pipeline",
                {"refusal": reopen_refusal, "output_directory_created": target_created},
                {"checks": [_refusal("Session.from_workspace on a result bound to another run",
                                     "Saved result does not refer to this evidence", reopen_refusal),
                            _check("refused reopen created its output directory", int(target_created))]},
                tolerance=EXACT),
        finding("A refused recording operation is retained as a refused execution record", "computational_pipeline",
                {"outcome": refused_operation, "state_parts_changed": recorded, "new_files": len(new_files),
                 "saved_workspace_changed": workspace_after_record != before["workspace"]},
                {"checks": [_refusal("operation.execute of an unregistered recording operation",
                                     "operation_unavailable: No trusted provider bound for ciw.lab-unregistered.v1",
                                     refused_operation),
                            _check("state parts changed by the refused operation", len(recorded), 1, "ge", "invariant")]},
                tolerance=EXACT, counterexample={
                    "statement": "Every refused request leaves the session's in-memory state unchanged",
                    "witness": {"request": "operation.execute ciw.lab-unregistered.v1", "changed": recorded,
                                "new_files": len(new_files)}}),
    ]
    fields = _fields(
        "A request that CIW refuses changes neither the saved workspace file nor the session state it would save.",
        "State = digests of selection, results, executions, retained catalog, catalog revision, byte and "
        "reservation counters, identity claims, candidates and bindings; plus the session directory listing.",
        ["energy-accuracy session (synthetic baseline) with oscillator results and one unbound numerical-heat source"],
        "SHA-256 digests before and after each class of refused request.",
        "All digests equal; re-saving reproduces the workspace content except its saved_at timestamp.",
        f"Send {len(outcomes)} refused requests (unbound execution, forged bindings, malformed sources, stale "
        "revision, unknown identities, out-of-range analysis, save redirection, unsupported version), then compare; "
        "reopen a corrupted copy; finally send one refused recording operation separately.",
        f"{refused}/{len(outcomes)} refused; changed state parts: {changed_parts or 'none'}; corrupted reopen refused "
        f"with '{reopen_refusal}'; a refused recording operation changed {recorded}.",
        "Exact digest equality.",
        ["reservation or pending counters leaked by a refusal", "partial source retained after refusal",
         "selection revision advanced by a refused update", "save path redirected by a client",
         "refused reopen writing into its target directory"],
        ["Refused recording operations are retained by design as auditable execution records; this is a documented "
         "exception, not a leak."],
        "T094: reopen golden retained bundles with the current code.")
    return {"state": "completed", "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T094

def _reopen_golden(path: Path, target: Path) -> dict:
    Session = _session_class()
    with execution_guard() as guard:
        session = Session.from_workspace(path, target)
    catalog = session.workbench.serialize()
    return {"session": session, "attempts": list(guard["attempts"]), "bundles": catalog["bundles"],
            "recomputations": dict(guard["recomputations"])}


@task("T094", changed_files=(MODULE, FIXTURES, "tests/fixtures/lab/golden"),
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
    rows, attempts, failures, digests = {}, [], [], {}
    reference_mismatch, heat_refusal, heat_values = None, None, None
    with tempfile.TemporaryDirectory(prefix="ciw-lab-t094-") as scratch:
        for index, (name, expected) in enumerate(sorted(GOLDEN_MANIFEST.items())):
            path = root / name
            if not path.is_file():
                failures.append(f"{name}: missing")
                continue
            digests[name] = sha256(path.read_bytes()).hexdigest()
            try:
                opened = _reopen_golden(path, Path(scratch) / str(index))
            except (ValueError, ExecutionForbidden) as exc:
                failures.append(f"{name}: {exc}")
                continue
            attempts += opened["attempts"]
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
                reference_mismatch = sum(a != b for a, b in zip(heat_values, reference))
                row["scr_revision_pinned"] = native["runtimes"]["scr"]["revision"] == PINS["numerical-heat"]["revision"]
                row["scr_engine_sha256"] = native["runtimes"]["scr"]["engine"]["sha256"]
                heat_refusal = _outcome(Client(session).call("bundle.replay", {"bundle_id": bundles[0]["bundle_id"]}))
            rows[name] = row
    manifest_mismatch = sum(digests.get(name) != expected for name, expected in GOLDEN_MANIFEST.items())
    ctx.artifact_json("golden.json", {"root": "tests/fixtures/lab" if root.name == "lab" else str(root),
                                      "manifest": GOLDEN_MANIFEST, "observed": rows, "failures": failures,
                                      "execution_attempts": attempts})
    value = {name: {key: row[key] for key in ("sha256", "bundles", "results", "executions", "kinds")}
             for name, row in rows.items()}
    findings = [
        finding("The golden retained workspaces reopen and validate with the current code without execution",
                "computational_pipeline", value,
                {"checks": [_check("golden files differing from GOLDEN_MANIFEST", manifest_mismatch),
                            _check("golden workspaces refused or missing", len(failures)),
                            _check("execution entry points reached while reopening", len(attempts)),
                            _check("replays whose numerical identity differs from their original",
                                   sum(not row.get("replay_numerical_identity_equal", True) for row in rows.values()))]},
                tolerance=EXACT),
        finding("Golden fixture SHA-256 digests", "provenance", dict(sorted(digests.items())),
                {"checks": [_check("digests differing from the recorded manifest", manifest_mismatch)]},
                tolerance=EXACT),
    ]
    if heat_values is not None:
        findings += [
            finding("Retained golden SCR heat values agree with an independent integer reference", "numerical",
                    heat_values, {"independent_check": {
                        **_check("cells differing from the integer Jacobi reference", reference_mismatch),
                        "producer": {"implementation": "Scientific-Computation-Runtime execution-cli (retained)",
                                     "revision": PINS["numerical-heat"]["revision"]},
                        "checker": {"implementation": "ciw.lab.exchange_provenance_bundles_fixtures.heat_reference",
                                    "revision": __version__}}}, tolerance=EXACT),
            finding("Replay of the golden SCR bundle without a binding is refused", "computational_pipeline",
                    heat_refusal, {"checks": [_refusal("bundle.replay in the reopened golden workspace",
                                                       f"operation_unavailable: {UNBOUND}", heat_refusal)]},
                    tolerance=EXACT),
        ]
    state = "completed" if not failures and not manifest_mismatch else "partial"
    fields = _fields(
        "Workspaces saved by CIW earlier (golden fixtures) still reopen and validate with the current code, with "
        "no provider binding and no execution.",
        "Golden = exact saved bytes recorded in GOLDEN_MANIFEST; validation = Session.from_workspace (structure, "
        "identities, commitments, energy-analysis recomputation) under the execution guard.",
        [f"tests/fixtures/lab/{name}" for name in sorted(GOLDEN_MANIFEST)],
        "SHA-256 of each file; reopen outcome; retained replay receipts; retained SCR values.",
        "All digests match, all reopen, zero execution attempts, replay numerical identity preserved.",
        "Hash each golden file, reopen it into a temporary directory under the guard, inspect its bundles, compare "
        "the retained SCR heat field with the independent integer reference and attempt an unbound replay.",
        f"{len(rows)} golden workspaces reopened, {len(failures)} failures, {manifest_mismatch} digest mismatches; "
        f"retained SCR heat values {heat_values}.",
        "Exact.",
        ["fixture drift (digest)", "schema or validator drift breaking reopen", "execution during reopen",
         "replay identity drift", "unbound replay of a provider-backed golden bundle"],
        ["Reopen recomputes the retained energy analysis in floating point; a platform whose NumPy/BLAS rounds "
         "differently could refuse the golden energy workspace (not observed here).",
         "The golden numerical-heat workspace records host paths of the machine that produced it; they are "
         "identity metadata, never bindings."],
        "T095: refuse malformed exchange fixtures with retained error text.")
    return {"state": state, "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T095

def _attempt(function):
    try:
        function()
        return {"refused": False, "error_type": None, "message": None}
    except RecursionError as exc:
        return {"refused": True, "error_type": "RecursionError", "message": str(exc)[:200]}
    except ValueError as exc:
        return {"refused": True, "error_type": type(exc).__name__, "message": str(exc)[:500]}


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
            inspected = {"refused": False, "error_type": None, "message": None}
        except OSError:
            inspected = {"refused": False, "error_type": "validator_unavailable", "message": None}
        except (ValueError, RecursionError) as exc:
            inspected = {"refused": True, "error_type": type(exc).__name__, "message": str(exc)[:500]}
        value = _parsed(raw)
        field = fields.get(value.get("schema")) if isinstance(value, dict) else None
        rows[name] = {
            "workbench_source_add": {"refused": response["type"] == "error", "error_type": None,
                                     "message": _outcome(response) if response["type"] == "error" else None},
            "session_read_json": _attempt(lambda: read_json(path)),
            "exchange_inspect": inspected,
            "exchange_identity": (_attempt(lambda: exchange._identity(value, field)) if field
                                  else {"refused": None, "error_type": "not_applicable", "message": None}),
            "session_from_workspace": _attempt(lambda: Session.from_workspace(path, directory / f"never-{index}")),
            "workbench_restore": (_attempt(lambda: Workbench.restore(value)) if isinstance(value, dict)
                                  else {"refused": None, "error_type": "not_applicable", "message": None}),
        }
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
        rows[label] = {"refused": response["type"] == "error", "error_type": None,
                       "message": _outcome(response) if response["type"] == "error" else None}
    saved = session.save_workspace(directory / "valid-workspace.json")
    workspace = json.loads(saved.read_text(encoding="utf-8"))
    typed = deepcopy(workspace)
    typed["selection"]["revision"] = "0"
    rows["wrong-type selection revision (Session.from_workspace)"] = _attempt(lambda: Session.from_workspace(
        _write(directory / "typed.json", json.dumps(typed).encode()), directory / "never-typed"))
    extra = dict(workspace, lab_unexpected_field={"note": "not part of the workspace schema"})
    extra_path = _write(directory / "extra.json", json.dumps(extra).encode())
    accepted = _attempt(lambda: Session.from_workspace(extra_path, directory / "extra-reopened"))
    dropped = None
    if not accepted["refused"]:
        resaved = Session.from_workspace(extra_path, directory / "extra-resaved").save_workspace(directory / "resaved.json")
        dropped = "lab_unexpected_field" not in json.loads(resaved.read_text(encoding="utf-8"))
    rows["extra top-level workspace field (Session.from_workspace)"] = dict(accepted, dropped_on_resave=dropped)
    return rows


def _write(path: Path, raw: bytes) -> Path:
    path.write_bytes(raw)
    return path


MALFORMED_TEXT = "MALFORMED_RESPONSE: The bound runtime did not return finite, unambiguous JSON"
# The validator each committed fixture targets and its exact CIW-generated text;
# None where the text comes from Python's json module and varies by version.
MALFORMED_EXPECTED = {
    "duplicate-key.json": ("exchange_inspect", "duplicate JSON member: schema"),
    "nan.json": ("exchange_inspect", "nonfinite JSON number: NaN"),
    "infinity.json": ("exchange_inspect", "nonfinite JSON number: -Infinity"),
    "overflow.json": ("exchange_inspect", "JSON number overflows float64"),
    "truncated-energy-log.json": ("workbench_source_add", MALFORMED_TEXT),
    "invalid-utf8.json": ("exchange_inspect", "exchange input must be bounded UTF-8 JSON"),
    "deep-nesting.json": ("exchange_inspect", None),
    "wrong-schema-exchange.json": ("exchange_inspect", "unsupported instrument-exchange schema"),
    "wrong-schema-energy-log.json": ("workbench_source_add",
                                     "invalid_payload: Unsupported energy log schema or occurrence identity"),
    "wrong-type-energy-log.json": ("workbench_source_add",
                                   "invalid_payload: Declare actual measurement versus synthetic fixture"),
    "extra-field-energy-log.json": ("workbench_source_add",
                                    "invalid_payload: Require exactly the declared energy-log fields"),
    "wrong-type-workspace.json": ("session_from_workspace", "Unsupported workspace format"),
    "wrong-revision-workbench.json": ("workbench_restore", "Malformed retained workbench catalog"),
    "wrong-identity-result.json": ("exchange_identity", "result_id does not match the artifact content"),
}
RUNTIME_EXPECTED = {
    "oversize-exchange (1 MiB + 1 byte)": "file exceeds 1048576 bytes: <path>",
    "oversize-energy-source (4 MiB + 1 byte)":
        "invalid_payload: Energy analysis source must contain 1..4194304 exact retained bytes",
    "non-canonical base64": "invalid_payload: Source bytes must use bounded canonical base64",
    "extra source payload field": "invalid_payload: Unexpected or missing workbench fields",
    "wrong-type selection revision (Session.from_workspace)": "Invalid saved selection revision",
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
        committed = sum((root / "malformed" / name).read_bytes() != raw if (root / "malformed" / name).is_file() else 1
                        for name, raw in malformed_fixtures().items())
    ctx.artifact_json("malformed-refusals.json", {"committed_fixture_mismatches": committed,
                                                  "fixtures": matrix, "runtime": runtime})
    unrefused = [name for name, (validator, _) in MALFORMED_EXPECTED.items() if not matrix[name][validator]["refused"]]
    unrefused += [name for name in RUNTIME_EXPECTED if not runtime[name]["refused"]]
    checks = []
    for name, (validator, expected) in sorted(MALFORMED_EXPECTED.items()):
        observed = matrix[name][validator]
        if expected is None:
            checks.append(_check(f"{name} not refused by {validator}", int(not observed["refused"])))
        else:
            checks.append(_refusal(f"{name} via {validator}", expected, observed["message"]))
    for name, expected in sorted(RUNTIME_EXPECTED.items()):
        checks.append(_refusal(name, expected, runtime[name]["message"]))
    overflow = matrix["overflow.json"]
    deep = matrix["deep-nesting.json"]
    extra = runtime["extra top-level workspace field (Session.from_workspace)"]
    total = len(MALFORMED_EXPECTED) + len(RUNTIME_EXPECTED)
    findings = [
        finding("Every malformed exchange fixture is refused by the CIW validator it targets, with the error text "
                "retained", "computational_pipeline", {"fixtures": total, "refused": total - len(unrefused)},
                {"checks": [_check("malformed fixtures accepted by their targeted validator", len(unrefused))] + checks},
                tolerance=EXACT),
        finding("session.read_json accepts an overflowing number as infinity while the exchange and workbench "
                "parsers refuse it", "computational_pipeline",
                {"session_read_json_refused": overflow["session_read_json"]["refused"],
                 "exchange_refused": overflow["exchange_inspect"]["refused"],
                 "workbench_refused": overflow["workbench_source_add"]["refused"]},
                {"checks": [_check("read_json refusals of 1e999", int(overflow["session_read_json"]["refused"])),
                            _check("exchange refusals of 1e999", int(overflow["exchange_inspect"]["refused"]), 1,
                                   "ge", "invariant")]},
                tolerance=EXACT, counterexample={
                    "statement": "Every CIW JSON reader refuses nonfinite numbers",
                    "witness": {"fixture": "overflow.json", "reader": "ciw.session.read_json",
                                "parsed_value": "inf"}}),
        finding("session.read_json does not refuse a 20000-deep array with a ValueError", "computational_pipeline",
                "not_refused_with_value_error",
                {"checks": [_refusal("read_json on deep-nesting.json",
                                     "not_refused_with_value_error",
                                     "not_refused_with_value_error"
                                     if deep["session_read_json"]["error_type"] not in ("ValueError", "JSONDecodeError")
                                     else "value_error")]},
                tolerance=EXACT, counterexample={
                    "statement": "session.read_json refuses every malformed workspace file with a ValueError",
                    "witness": {"fixture": "deep-nesting.json",
                                "observed": deep["session_read_json"]["error_type"] or "accepted"}}),
        finding("Session.from_workspace accepts an unknown top-level field and drops it on re-save",
                "computational_pipeline", {"accepted": not extra["refused"], "dropped_on_resave": extra["dropped_on_resave"]},
                {"checks": [_check("extra-field workspaces refused", int(extra["refused"])),
                            _check("extra field kept after re-save", int(extra["dropped_on_resave"] is False))]},
                tolerance=EXACT, counterexample={
                    "statement": "Saved workspaces with fields outside the schema are refused on reopen",
                    "witness": {"field": "lab_unexpected_field", "reopened": True, "dropped_on_resave": True}}),
        finding("Workbench source.add reports malformed source JSON with text that names a bound runtime",
                "computational_pipeline", matrix["duplicate-key.json"]["workbench_source_add"]["message"],
                {"checks": [_refusal("source.add of duplicate-key.json",
                                     "MALFORMED_RESPONSE: The bound runtime did not return finite, unambiguous JSON",
                                     matrix["duplicate-key.json"]["workbench_source_add"]["message"])]},
                tolerance=EXACT),
    ]
    fields = _fields(
        "Each malformed exchange input (duplicate keys, NaN/Infinity, overflow, wrong schema, oversize, truncated "
        "bytes, invalid UTF-8, wrong types, extra fields, forged identity) is refused by the relevant CIW validator.",
        "Validators: Workbench source.add (energy-accuracy), ciw.session.read_json, ciw.exchange.inspect_exchange "
        "(parsing stage, no SET checkout), ciw.exchange._identity, Session.from_workspace, Workbench.restore.",
        [f"tests/fixtures/lab/malformed/{name}" for name in sorted(malformed_fixtures())]
        + sorted(RUNTIME_EXPECTED) + ["extra top-level workspace field (generated)"],
        "Refused/accepted, exception type and exact message per validator (retained in malformed-refusals.json).",
        "Every fixture refused by at least one validator; CIW-generated messages equal their expected text.",
        "Write each committed fixture from its generator, apply the three byte-level validators to each, apply the "
        "structural validators to the runtime-generated cases, and compare exact CIW messages.",
        f"{total - len(unrefused)}/{total} malformed inputs refused; read_json accepted 1e999 and "
        f"{'raised ' + str(deep['session_read_json']['error_type']) if deep['session_read_json']['refused'] else 'accepted'}"
        " on 20000-deep nesting; an extra top-level workspace field was accepted and dropped on re-save.",
        "Exact string comparison for CIW-generated text; messages from Python's json module are retained but not "
        "compared, because they vary between interpreter versions.",
        ["duplicate keys", "NaN and ±Infinity literals", "float overflow", "truncation", "invalid UTF-8",
         "recursion depth", "wrong schema", "wrong types", "extra fields", "oversize bytes", "forged identity",
         "non-canonical base64"],
        ["The workbench validator is exercised through the energy-accuracy kind; other kinds use their own schema "
         "validators after the same canonical-base64 and JSON checks.",
         "The exchange stage stops before the SET validator; conformance of well-formed artifacts needs SET (T097)."],
        "T096: provider-free conformance of exchange identities and ESM candidate responses.")
    return {"state": "completed", "fields": fields, "findings": findings}


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


def _identity_study(seed=9601, count=24):
    import numpy as np
    from .. import exchange
    from .exchange_provenance_bundles_fixtures import _sealed
    generator = np.random.Generator(np.random.PCG64(seed))
    valid = accepted = mutated = refused = permuted = batch_mutations = batch_accepted = 0
    for index in range(count):
        schema, field = ((exchange.RESULT_SCHEMA, "result_id") if index % 2 == 0
                         else (exchange.VERIFICATION_SCHEMA, "verification_id"))
        body = {"schema": schema, **{f"field_{j}": _random_value(generator) for j in range(int(generator.integers(2, 6)))}}
        artifact = _sealed(body, field)
        valid += 1
        accepted += exchange._identity(artifact, field) == "content_recomputed_not_authenticated"
        # Key order and JSON escaping do not change the canonical identity.
        reordered = json.loads(json.dumps(dict(reversed(list(artifact.items()))), ensure_ascii=True))
        permuted += exchange._identity(reordered, field) == "content_recomputed_not_authenticated"
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
    return {"valid": valid, "accepted": accepted, "permuted_accepted": permuted, "mutations": mutated,
            "refused": refused, "batch_mutations": batch_mutations, "batch_mutations_accepted": batch_accepted}


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


@task("T096", changed_files=(MODULE,),
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
    findings = [
        finding("exchange._identity accepts every sealed synthetic result and verification record and refuses "
                "every single-field mutation", "computational_pipeline",
                {key: identity[key] for key in ("valid", "accepted", "permuted_accepted", "mutations", "refused")},
                {"generator": {"name": "ciw.lab seeded nested JSON records", "seed": 9601, "count": 24},
                 "checks": [_check("valid records not accepted", identity["valid"] - identity["accepted"]),
                            _check("key-reordered or ASCII-escaped records not accepted",
                                   identity["valid"] - identity["permuted_accepted"]),
                            _check("mutations not refused", identity["mutations"] - identity["refused"])]},
                tolerance=EXACT),
        finding("candidate_evidence.validate_response accepts valid synthetic ESM responses and refuses every "
                "boundary mutation", "computational_pipeline",
                {"valid": len(valid), "valid_accepted": len(valid) - wrong_valid, "mutations": len(mutated),
                 "mutations_refused": len(mutated) - wrong_mutated},
                {"checks": [_check("valid synthetic responses refused", wrong_valid),
                            _check("boundary mutations accepted", wrong_mutated)]},
                tolerance=EXACT),
        finding("Observation-batch identities are caller-declared: mutated batches pass exchange._identity",
                "computational_pipeline",
                {"mutations": identity["batch_mutations"], "accepted": identity["batch_mutations_accepted"]},
                {"checks": [_check("mutated batches accepted", identity["batch_mutations_accepted"], 1, "ge",
                                   "invariant")]},
                tolerance=EXACT, counterexample={
                    "statement": "Every exchange artifact identity is bound to its content",
                    "witness": {"schema": "notation.instrument.observation-batch.v1",
                                "identity_status": "caller_declared_reference"}}),
        finding("validate_response accepts unknown extra fields in an ESM response", "computational_pipeline",
                {"cases": len(extras), "accepted": extras_accepted},
                {"checks": [_check("responses with unknown fields accepted", extras_accepted, 1, "ge", "invariant")]},
                tolerance=EXACT, counterexample={
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
        ["24 seeded synthetic result/verification records (PCG64 seed 9601)",
         f"{len(candidates)} synthetic ESM inspect/capture responses over telemetry and calibrated bundle stubs"],
        "Accepted or refused (with message) per record or response.",
        "All valid accepted, all single-field mutations refused, identity invariant under key order and escaping.",
        "Seal records, re-validate them reordered and ASCII-escaped, mutate each non-identity field and the "
        "identity itself; build valid ESM responses and apply every boundary mutation.",
        f"{identity['refused']}/{identity['mutations']} identity mutations refused; "
        f"{len(mutated) - wrong_mutated}/{len(mutated)} candidate mutations refused; {extras_accepted} response(s) "
        "with unknown fields accepted.",
        "Exact.",
        ["key-order dependence", "Unicode escaping dependence", "identity forgery", "admission/state/truth flags",
         "binding to request, time, digest and bundle", "capture receipt consistency"],
        ["The bundle stubs contain only the fields validate_response reads; full native bundles need providers.",
         "Mutations are single-field; combined mutations that restore consistency are not explored."],
        "T097: exact SCR/SET/PPDA integrations when the pinned checkouts are bound.")
    return {"state": "completed", "fields": fields, "findings": findings}


# ---------------------------------------------------------------- engines

def _engine(ctx) -> dict | None:
    """SCR engine bytes: an operator binding, else a locked offline build shared by the run."""
    bound = ctx.providers.get("scr-engine")
    if bound is not None and Path(bound).is_file():
        data = Path(bound).read_bytes()
        return {"origin": "operator_bound_scr_engine", "binary": data, "sha256": sha256(data).hexdigest(), "build": None}
    if "scr" in ctx.providers and ctx.available("tool:cargo"):
        build = ctx.memo(("exchange-bundles:scr-build", str(ctx.providers["scr"])),
                         lambda: providers.build_engine(ctx.providers["scr"]))
        if build["binary"] is not None:
            return {"origin": "cargo_build_locked_offline_in_this_run", "binary": build["binary"],
                    "sha256": sha256(build["binary"]).hexdigest(), "build": build}
    return None


def _scr_identity(ctx):
    return ctx.memo(("exchange-bundles:identity", str(ctx.providers["scr"])),
                    lambda: providers.checkout_identity(ctx.providers["scr"]))


def _provider_basis(identity, engine_sha256):
    return {"provider": {"repository": providers.REPOSITORIES["scr"], "revision": identity["head"],
                         "source_tree": identity["tree"], "runtime_digest": "sha256:" + engine_sha256,
                         "executed": True}}


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


@task("T097", changed_files=(MODULE, PROVIDERS),
      regression_tests=(f"{TESTS}::test_t097_scr_numerical_heat_integration",
                        f"{TESTS}::test_t097_is_blocked_without_scr"),
      requires=("provider:scr",), plan=_T097_PLAN)
def exact_provider_integrations(ctx):
    identity = _scr_identity(ctx)
    comparison = providers.compare_with_pins("scr", identity, providers.ciw_pins())
    fields = {k: v for k, v in deepcopy(_T097_PLAN).items() if k != "findings"}
    if not comparison["accepted"]:
        fields.update(numerical_result=f"SCR checkout refused: HEAD {identity['head']} matches {comparison['matched']}",
                      unresolved_assumptions=["The bound SCR checkout is not at a CIW pin or not clean."])
        return {"state": "blocked", "fields": fields, "findings": [
            finding("The bound SCR checkout is refused before execution", "provenance", comparison,
                    {"checks": [_refusal("SCR HEAD against ciw.declared_workload.PINS", "refused", "refused")]},
                    tolerance=EXACT)]}
    engine = _engine(ctx)
    findings, parts, blocked = [], {}, []
    with tempfile.TemporaryDirectory(prefix="ciw-lab-t097-") as scratch:
        scratch = Path(scratch)
        if engine is None:
            blocked.append("SCR engine: neither scr-engine binding nor cargo is available")
        else:
            path = providers.materialize(engine["binary"], scratch)
            workbench = providers.scr_workbench_integration(ctx.providers["scr"], path, scratch / "workbench", [
                ("Declared integer heat workload (example)", example_bytes("declared-workloads/numerical-heat.json"))])
            api = providers.run_heat_kernel(ctx.providers["scr"], path, SURVEY_CASES)
            parts["workbench"], parts["api"] = workbench, api
    pins = providers.ciw_pins()
    optional, trees = {}, {}
    for role in ("set", "ppda", "scr-exchange"):
        if role in ctx.providers:
            try:
                bound = providers.checkout_identity(ctx.providers[role])
                optional[role], trees[role] = providers.compare_with_pins(role, bound, pins), bound["tree"]
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                optional[role] = {"accepted": False, "error": str(exc)}
    set_ready = optional.get("set", {}).get("accepted") and "ciw/exchange-runtime.json" in optional["set"]["matched"]
    with tempfile.TemporaryDirectory(prefix="ciw-lab-t097-exchange-") as scratch:
        if set_ready:
            parts["set"] = providers.set_exchange_inspection(ctx.providers["set"], scratch)
        roundtrip_ready = (set_ready and optional.get("ppda", {}).get("accepted")
                           and ".github/workflows/exchange.yml" in optional["ppda"].get("matched", [])
                           and optional.get("scr-exchange", {}).get("accepted"))
        if roundtrip_ready:
            parts["roundtrip"] = providers.exchange_roundtrip(ctx.providers["ppda"], ctx.providers["scr-exchange"],
                                                              ctx.providers["set"], scratch)
    ctx.artifact_json("integration.json", {
        "scr_identity": {k: identity[k] for k in ("head", "tree", "tracked_sha256", "tracked_files", "cargo_locks")},
        "scr_pins": comparison, "engine": None if engine is None else {"origin": engine["origin"], "sha256": engine["sha256"]},
        "optional_providers": optional,
        "parts": {name: value for name, value in parts.items()}})
    if "workbench" in parts:
        workbench, api = parts["workbench"], parts["api"]
        demo = workbench["runs"][0]
        survey = api["cases"][0]
        mismatch = sum(sum(a != b for a, b in zip(run["values"], heat_reference(run["initial_values"], run["steps"])))
                       for run in workbench["runs"])
        mismatch += sum(sum(a != b for a, b in zip(case["values"], heat_reference(values, steps)))
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
                    _provider_basis(identity, engine["sha256"]), tolerance=EXACT),
            finding("SCR heat outputs from the workbench and from SCR's Python API equal an independent integer "
                    "reference", "numerical", {"cases": len(workbench["runs"]) + len(api["cases"]),
                                               "mismatched_cells": mismatch},
                    {"independent_check": {**_check("cells differing from the integer Jacobi reference", mismatch),
                                           "producer": {"implementation": "Scientific-Computation-Runtime execution-cli",
                                                        "revision": identity["head"]},
                                           "checker": {"implementation": "ciw.lab.exchange_provenance_bundles_fixtures"
                                                                         ".heat_reference", "revision": __version__}}},
                    tolerance=EXACT),
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
                                _check("SCR descriptor digest differs from CIW's HEAT_DESCRIPTOR",
                                       int(not descriptor_equal))]},
                    tolerance=EXACT),
            finding("The reopened SCR workspace carries no binding and refuses replay", "computational_pipeline",
                    {"bindings": workbench["reopened_bindings"], "available": workbench["reopened_available"],
                     "replay": f"{workbench['unbound_replay']['code']}: {workbench['unbound_replay'].get('message')}"},
                    {"checks": [_refusal("bundle.replay after reopen", f"operation_unavailable: {UNBOUND}",
                                         f"{workbench['unbound_replay']['code']}: "
                                         f"{workbench['unbound_replay'].get('message')}"),
                                _check("execution entry points reached while reopening",
                                       len(workbench["reopen_attempts"])),
                                _check("bindings after reopen", len(workbench["reopened_bindings"]))]},
                    tolerance=EXACT),
        ]
    if "set" in parts:
        findings.append(finding(
            "The pinned SET contracts validator reports the synthetic observation conformant with effective "
            "covariance rank 2 and refuses an indefinite covariance", "computational_pipeline",
            {"status": parts["set"]["status"], "effective_rank": parts["set"]["effective_rank"],
             "indefinite_refused": parts["set"]["indefinite_covariance"] != "not_refused"},
            {"provider": {"repository": providers.REPOSITORIES["set"], "revision": parts["set"]["validator"]["revision"],
                          "runtime_digest": "sha256:" + parts["set"]["validator"]["sha256"], "executed": True}},
            tolerance=EXACT))
    else:
        blocked.append("SET exchange validator: set checkout at 542e672be512bf43b61253f2b2a43cd967cb3062 not bound")
        findings.append(finding(
            "SET exchange conformance was not executed", "computational_pipeline",
            {"required": "--provider set=<State-Estimation-Evaluation-Testbed at 542e672be512bf43b61253f2b2a43cd967cb3062>",
             "pytest": "CIW_SET_REPO=<set> python -m pytest -q tests/test_exchange.py"}, {},
            expected_not_established=True))
    if "roundtrip" in parts:
        findings.append(finding(
            "PPDA and SCR exchange producers validated by the pinned SET checker link and preserve claim boundaries",
            "computational_pipeline",
            {"status": parts["roundtrip"]["status"], "links": parts["roundtrip"]["links"],
             "verification_outcome": parts["roundtrip"]["verification_outcome"],
             "changed_result_refusal": parts["roundtrip"]["changed_result_refusal"]},
            {"provider": {"repository": providers.REPOSITORIES["ppda"], "revision": providers.EXCHANGE_WORKFLOW_PINS["ppda"],
                          "source_tree": trees["ppda"], "executed": True}},
            tolerance=EXACT))
    else:
        blocked.append("PPDA/SCR/SET producer roundtrip: needs ppda@a29845e, scr-exchange@5f04097 and set@542e672")
        findings.append(finding(
            "The PPDA/SCR/SET exchange producer roundtrip was not executed", "computational_pipeline",
            {"required": {role: revision for role, revision in sorted(providers.EXCHANGE_WORKFLOW_PINS.items())},
             "pytest": "CIW_SET_REPO=<set> CIW_ACQUISITION_REPO=<ppda> CIW_RUNTIME_REPO=<scr-exchange> "
                       "python -m pytest -q tests/test_exchange_integration.py"}, {},
            expected_not_established=True))
    findings.append(finding("The integer heat field describes physical heat diffusion in a material", "physical",
                            "not established: dimensionless integer arithmetic", {}))
    state = "completed" if not blocked else ("partial" if "workbench" in parts else "blocked")
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
                          or "SCR not executed") + f"; blocked parts: {blocked or 'none'}",
        uncertainty="Exact integer arithmetic; no tolerance.",
        failure_modes_checked=["pin mismatch", "engine drift", "replay identity drift", "binding recovered from "
                               "saved data", "descriptor drift between SCR and CIW"],
        unresolved_assumptions=blocked + ["CIW records the engine digest as operator_asserted_not_attested; this "
                                          "run's engine origin is "
                                          f"{None if engine is None else engine['origin']}, which CIW itself does "
                                          "not verify.",
                                          "The PPDA telemetry stack (ppda, stfe, gsie, set, cbsr at "
                                          "src/ciw/telemetry-runtimes.json pins) is a separate integration."],
        recommended_next_task="T098: record the identities of every bound provider checkout.",
        provider_runtime_identity={"scr": {"revision": identity["head"], "source_tree": identity["tree"],
                                           "engine_sha256": None if engine is None else engine["sha256"],
                                           "engine_origin": None if engine is None else engine["origin"]},
                                   **{role: {"matched": value.get("matched")} for role, value in optional.items()}})
    return {"state": state, "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T098

def _pin_consistency():
    from .. import declared_workload, proved_heat
    pins = providers.ciw_pins()
    scr = {pin["revision"] for pin in pins["scr"]}
    set_revisions = sorted({pin["revision"] for pin in pins["set"]})
    return {"scr_revisions": sorted(scr), "set_revisions": set_revisions,
            "scr_consistent": declared_workload.PINS["numerical-heat"]["revision"] == proved_heat.PIN["revision"]}


def _adapter_refusals(role, path, pins):
    """CIW's own subprocess adapter refusing a checkout for each module pin it does not match."""
    from ..adapters.protocol import AdapterRefusal
    from ..adapters.subprocess import PinnedSubprocessAdapter
    rows = []
    for pin in pins.get(role, []):
        if not pin.get("module"):
            continue
        try:
            PinnedSubprocessAdapter(path, pin["revision"], pin["module"], source_root=pin["source_root"])
            outcome = "accepted"
        except AdapterRefusal as exc:
            outcome = f"{exc.code}: {exc}"
        except ValueError as exc:
            outcome = f"ValueError: {exc}"
        rows.append({"declared_in": pin["declared_in"], "revision": pin["revision"], "outcome": outcome})
    return rows


@task("T098", changed_files=(MODULE, PROVIDERS),
      regression_tests=(f"{TESTS}::test_t098_provider_identities",
                        f"{TESTS}::test_tree_recomputation_matches_git_on_a_synthetic_repository"))
def provider_identities(ctx):
    pins = providers.ciw_pins()
    consistency = _pin_consistency()
    roles = sorted(role for role in providers.REPOSITORIES if role in ctx.providers)
    identities, comparisons, refusals, errors = {}, {}, {}, {}
    for role in roles:
        try:
            identities[role] = providers.checkout_identity(ctx.providers[role])
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            errors[role] = str(exc)
            continue
        comparisons[role] = providers.compare_with_pins(role, identities[role], pins)
        refusals[role] = _adapter_refusals(role, ctx.providers[role], pins)
    accepted = sorted(role for role, row in comparisons.items() if row["accepted"])
    rejected = sorted(set(roles) - set(accepted))
    engine = _engine(ctx) if "scr" in accepted else None
    liveness = None
    if engine is not None:
        with tempfile.TemporaryDirectory(prefix="ciw-lab-t098-") as scratch:
            liveness = providers.run_heat_kernel(ctx.providers["scr"], providers.materialize(engine["binary"], scratch),
                                                 SURVEY_CASES[:1])
    ctx.artifact_json("provider-identities.json", {
        "identities": identities, "comparisons": comparisons, "adapter_pin_checks": refusals, "errors": errors,
        "ciw_pins": pins, "engine": None if engine is None else {
            "origin": engine["origin"], "sha256": engine["sha256"], "byte_count": len(engine["binary"]),
            "toolchain": None if engine["build"] is None else {k: engine["build"][k] for k in ("cargo", "rustc")}}})
    findings = [finding(
        "CIW declares one SCR revision across its workflows and several distinct SET revisions",
        "provenance", {"scr_revisions": consistency["scr_revisions"], "set_revisions": consistency["set_revisions"]},
        {"checks": [_check("distinct SCR revisions declared by CIW beyond one", len(consistency["scr_revisions"]) - 1),
                    _check("distinct SET revisions declared by CIW", len(consistency["set_revisions"]), 2, "ge",
                           "invariant")]}, tolerance=EXACT)]
    if not roles:
        fields = _fields(
            "Every bound provider checkout is clean and at a CIW pin, and its bytes reproduce its Git tree.",
            "Identity = (HEAD, HEAD^{tree}, SHA-256 over tracked working bytes, Cargo.lock SHA-256, engine SHA-256).",
            [], "None: no provider checkout is bound.", "Pins matched; recomputed tree equals Git's tree.",
            "Blocked: bind providers with --provider ROLE=PATH for roles " + ", ".join(sorted(providers.REPOSITORIES)),
            "CIW pin table only.", "not quantified", ["no provider bound"], ["No provider checkout is bound."],
            "Bind csg, ftr, scr (scripts/check_lab.py does) and rerun T098.")
        return {"state": "blocked", "fields": fields, "findings": findings}
    value = {role: {"head": identities[role]["head"], "tree": identities[role]["tree"],
                    "pins": comparisons[role]["matched"]} for role in accepted}
    tree_mismatch = sum(identities[role]["recomputed_tree"] != identities[role]["tree"] for role in identities)
    again = {role: providers.checkout_identity(ctx.providers[role])["tracked_sha256"] for role in identities}
    findings.insert(0, finding(
        "Every accepted provider checkout is clean and at a CIW-declared revision and pinned tree",
        "provenance", value,
        {"checks": [_check("accepted checkouts with a pinned-tree mismatch",
                           sum(bool(comparisons[role]["tree_refusals"]) for role in accepted)),
                    _check("accepted checkouts that are dirty or carry untracked or modified files",
                           sum(not comparisons[role]["clean"] for role in accepted)),
                    _check("accepted checkouts", len(accepted), 1, "ge", "invariant")]}, tolerance=EXACT))
    findings.insert(1, finding(
        "The working bytes of every bound checkout reproduce Git's HEAD tree id", "provenance",
        {role: identities[role]["recomputed_tree"] for role in sorted(identities)},
        {"independent_check": {**_check("checkouts whose recomputed tree differs from git rev-parse HEAD^{tree}",
                                        tree_mismatch),
                               "producer": {"implementation": "ciw.lab.exchange_provenance_bundles_providers"
                                                              ".checkout_identity", "revision": __version__},
                               "checker": {"implementation": "git rev-parse HEAD^{tree}",
                                           "revision": _git_version()}}}, tolerance=EXACT))
    findings.insert(2, finding(
        "Tracked-source and Cargo.lock digests of every bound checkout", "provenance",
        {role: {"tracked_sha256": identities[role]["tracked_sha256"], "tracked_files": identities[role]["tracked_files"],
                "cargo_locks": identities[role]["cargo_locks"]} for role in sorted(identities)},
        {"checks": [_check("digests that change on an immediate second read",
                           sum(again[role] != identities[role]["tracked_sha256"] for role in identities))]},
        tolerance=EXACT))
    refused_rows = {role: [row for row in rows if row["outcome"] != "accepted"] for role, rows in refusals.items()}
    refused_rows = {role: rows for role, rows in refused_rows.items() if rows}
    if refused_rows:
        findings.append(finding(
            "CIW's pinned subprocess adapter refuses each bound checkout for the pins it does not match",
            "provenance", {role: sorted(row["declared_in"] for row in rows) for role, rows in sorted(refused_rows.items())},
            {"checks": [_refusal(f"{role} against {row['declared_in']}",
                                 "SOURCE_PIN_MISMATCH: The bound checkout is not at the declared revision",
                                 row["outcome"]) for role, rows in sorted(refused_rows.items()) for row in rows]},
            tolerance=EXACT))
    if rejected:
        findings.append(finding(
            "Bound checkouts that match no CIW pin are refused", "provenance",
            {role: comparisons.get(role, {"error": errors.get(role)}) for role in rejected},
            {"checks": [_refusal(f"{role} checkout", "refused", "refused") for role in rejected]}, tolerance=EXACT))
    if liveness is not None:
        findings.append(finding(
            "The SCR engine recorded for this run executes the SCR heat descriptor on the survey input",
            "numerical", {"engine_origin": engine["origin"],
                          "cargo_lock_sha256": identities["scr"]["cargo_locks"].get("crates/Cargo.lock"),
                          "survey_output": liveness["cases"][0]["values"],
                          "descriptor_sha256": liveness["descriptor_sha256"]},
            _provider_basis(identities["scr"], engine["sha256"]), tolerance=EXACT))
    fields = _fields(
        "Every bound provider checkout is clean and at a CIW pin; its working bytes reproduce its Git tree; its "
        "lockfiles and the engine used in this run are recorded.",
        "Git object ids: blob = H('blob' len NUL bytes), tree = H('tree' len NUL sorted(mode name NUL id)); "
        "tracked digest = SHA-256 over 'mode kind sha256(bytes) path' lines in ls-tree order.",
        [f"{role}: {ctx.providers[role]}" for role in roles],
        "git rev-parse, git ls-tree, working-tree bytes, CIW pin tables, PinnedSubprocessAdapter outcomes.",
        "HEAD equals a CIW pin; pinned trees equal; recomputed tree equals Git's; digests stable.",
        "For each bound role read HEAD and tree, hash every tracked file, recompute the tree independently of "
        "Git, compare with every CIW pin for that role, and let CIW's own adapter accept or refuse each module pin; "
        "execute the SCR engine once on the survey input when available.",
        f"accepted {accepted}, refused {rejected}; {tree_mismatch} tree recomputation mismatches; adapter refusals "
        f"for {sorted(refused_rows)}; engine {None if engine is None else engine['origin']}.",
        "Exact digests.",
        ["dirty or untracked files", "wrong revision", "pinned tree drift", "modified tracked bytes",
         "executable-bit drift", "one checkout serving several different CIW pins"],
        ["Engine and interpreter digests depend on the toolchain; they are retained as provenance in the artifact "
         "and the basis, not compared as regression values.",
         "Pins declared only in .github/workflows/exchange.yml are mirrored in EXCHANGE_WORKFLOW_PINS."],
        "T099: verify the locked offline SCR build and record the SP1 build requirements.")
    fields["provider_runtime_identity"] = {role: {"head": identities[role]["head"], "tree": identities[role]["tree"]}
                                           for role in sorted(identities)}
    if engine is not None:
        fields["provider_runtime_identity"]["scr"]["engine_sha256"] = engine["sha256"]
    return {"state": "completed", "fields": fields, "findings": findings}


def _git_version() -> str:
    try:
        return subprocess.run(["git", "--version"], capture_output=True, text=True,
                                            timeout=30).stdout.strip() or "unknown"
    except OSError:
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


@task("T099", changed_files=(MODULE, PROVIDERS),
      regression_tests=(f"{TESTS}::test_t099_locked_offline_scr_build",),
      requires=("provider:scr", "tool:cargo"), plan=_T099_PLAN)
def locked_cargo_build(ctx):
    identity = _scr_identity(ctx)
    comparison = providers.compare_with_pins("scr", identity, providers.ciw_pins())
    fields = {k: v for k, v in deepcopy(_T099_PLAN).items() if k != "findings"}
    if not comparison["accepted"]:
        fields["numerical_result"] = f"SCR checkout refused: {comparison}"
        return {"state": "blocked", "fields": fields, "findings": []}
    build = ctx.memo(("exchange-bundles:scr-build", str(ctx.providers["scr"])),
                     lambda: providers.build_engine(ctx.providers["scr"]))
    after = providers.checkout_identity(ctx.providers["scr"])
    ok = [row for row in build["builds"] if row["returncode"] == 0 and row["binary_sha256"]]
    output = None
    if build["binary"] is not None:
        with tempfile.TemporaryDirectory(prefix="ciw-lab-t099-") as scratch:
            output = providers.run_heat_kernel(ctx.providers["scr"], providers.materialize(build["binary"], scratch),
                                               SURVEY_CASES[:1])
    ctx.artifact_json("cargo-build.json", {key: value for key, value in build.items() if key != "binary"})
    probes = {"protoc": providers.which("protoc") is not None, "sp1_bound": "sp1" in ctx.providers,
              "zk_manifest": (Path(ctx.providers["scr"]) / "zk" / "Cargo.toml").is_file()}
    ctx.artifact_json("sp1-requirements.json", {"requirements": providers.SP1_REQUIREMENTS, "probes": probes,
                                                "attempted": False})
    digests = sorted({row["binary_sha256"] for row in ok})
    lock = identity["cargo_locks"].get("crates/Cargo.lock")
    findings = []
    if ok:
        values = output["cases"][0]["values"] if output else None
        reference = heat_reference(SURVEY_CASES[0][1], SURVEY_CASES[0][0])
        findings += [
            finding("cargo build --release --locked --offline of SCR execution-cli succeeds at the pinned revision",
                    "computational_pipeline", {"exit_code": 0, "flags": ["--release", "--locked", "--offline"],
                                               "package": "execution-cli", "cargo_lock_sha256": lock,
                                               "builds": len(build["builds"])},
                    _provider_basis(identity, digests[0]), tolerance=EXACT),
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
                                _check("distinct binary digests beyond one", len(digests) - 1)]},
                    tolerance=EXACT),
            finding("The freshly built engine's heat output equals an independent integer reference", "numerical",
                    values, {"independent_check": {
                        **_check("cells differing from the integer Jacobi reference",
                                 sum(a != b for a, b in zip(values or [], reference)) + (0 if values else 5)),
                        "producer": {"implementation": "Scientific-Computation-Runtime execution-cli (locked build)",
                                     "revision": identity["head"]},
                        "checker": {"implementation": "ciw.lab.exchange_provenance_bundles_fixtures.heat_reference",
                                    "revision": __version__}}}, tolerance=EXACT),
        ]
    else:
        findings.append(finding("cargo build --locked --offline of SCR execution-cli failed", "computational_pipeline",
                                [row["returncode"] for row in build["builds"]], {}, expected_not_established=True))
    findings += [
        finding("The SP1 proved-heat locked build was not attempted", "computational_pipeline",
                {"attempted": False, "requirements": providers.SP1_REQUIREMENTS}, {}, expected_not_established=True),
        finding("A successful locked build makes the engine acceptable for production use", "production_acceptance",
                "not decided by the workbench", {}),
    ]
    fields.update(
        numerical_result=(f"{len(ok)}/{len(build['builds'])} locked offline builds succeeded; distinct binary "
                          f"digests {len(digests)}; Cargo.lock {lock}; survey output "
                          f"{None if output is None else output['cases'][0]['values']}; SP1 build not attempted."),
        uncertainty="Exact digests; the binary digest depends on the Rust toolchain and is recorded as provenance.",
        failure_modes_checked=["lockfile rewrite", "network access (--offline)", "writes into the checkout",
                               "non-deterministic build output", "engine output drift"],
        unresolved_assumptions=[f"Toolchain: {build['cargo']}; {build['rustc']}. CI pins rustc 1.94.0 for the "
                                "proved-heat gate; other toolchains may produce different binary digests.",
                                "SP1 proved-heat build needs crates.io, the SP1 checkout, protoc and >= 7 GiB RAM / "
                                f"20 GiB disk; probes here: {probes}."],
        recommended_next_task="Run the SP1 proved-heat gate (scripts/check_proved_heat.py) on a provisioned machine; "
                              "then T100.",
        provider_runtime_identity={"scr": {"head": identity["head"], "tree": identity["tree"],
                                           "engine_sha256": digests[0] if digests else None,
                                           "cargo": build["cargo"], "rustc": build["rustc"]}})
    return {"state": "partial", "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T100

_COMPUTATIONAL_LABELS = frozenset({"analytic", "synthetic", "numerically_verified", "provider_backed"})


def _audit_reports(ctx):
    """Label and rendering audit of every retained report numbered below 100."""
    directory = ctx.output_dir / "reports"
    rows, violations = [], []
    for path in sorted(directory.glob("T*.json")) if directory.is_dir() else []:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            violations.append(f"{path.name}: unreadable ({exc})")
            continue
        if not isinstance(report.get("number"), int) or report["number"] >= 100:
            continue
        try:
            validate_report(report)
        except EvidenceRefusal as exc:
            violations.append(f"{path.name}: refused by validate_report ({exc})")
            continue
        markdown = render_markdown(report)
        table = markdown.split("| Finding | Value | Evidence status |\n| --- | --- | --- |\n", 1)
        lines = table[1].splitlines() if len(table) == 2 else []
        for index, record in enumerate(report["findings"]):
            label, domain = record["evidence_status"], record["domain"]
            if label not in LABELS:
                violations.append(f"{report['task_id']}[{index}]: unknown label {label}")
            if domain in PHYSICAL_DOMAINS and (label in _COMPUTATIONAL_LABELS or
                                               (label != "not_established" and not record["basis"].get("acquisition"))):
                violations.append(f"{report['task_id']}[{index}]: physical finding labelled {label}")
            if domain in AUTHORITY_DOMAINS and label != "not_established":
                violations.append(f"{report['task_id']}[{index}]: authority finding labelled {label}")
            if domain not in PHYSICAL_DOMAINS and label == "hardware_measured":
                violations.append(f"{report['task_id']}[{index}]: computational finding labelled hardware_measured")
            row = lines[index] if index < len(lines) else ""
            if not row.endswith(f"| `{label}` |") or row.count(" | ") != 2:
                violations.append(f"{report['task_id']}[{index}]: rendered row does not show `{label}` in its column")
        if f"`{report['physical_validation_status']['status']}`" not in markdown:
            violations.append(f"{report['task_id']}: physical validation status not rendered")
        rows.append({"task_id": report["task_id"], "findings": len(report["findings"]),
                     "pipe_claims": sum("|" in f["claim"] or "\n" in f["claim"] for f in report["findings"]),
                     "labels": sorted({f["evidence_status"] for f in report["findings"]}),
                     "physical_or_authority": sum(f["domain"] in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS
                                                  for f in report["findings"])})
    return rows, violations


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
        fresh_outcome = _outcome(client.call("operation.execute", {"operation_id": "ciw.energy-accuracy.v1",
                                                                   "parameters": {"source_id": third["source_id"]}}))
    return {"synthetic": {key: synthetic[key] for key in ("origin", "hardware_provenance")}
            | {"classification": synthetic["comparison"]["classification"],
               "eligible": synthetic["comparison"]["eligible"]},
            "unsealed_relabel": unsealed,
            "resealed_relabel": {key: physical[key] for key in ("origin", "hardware_provenance")}
            | {"classification": physical["comparison"]["classification"], "eligible": physical["comparison"]["eligible"]},
            "same_occurrence_collision": collision, "fresh_occurrence": fresh_outcome}


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


@task("T100", changed_files=(MODULE,),
      regression_tests=(f"{TESTS}::test_t100_labels_and_origins_stay_distinct",
                        f"{TESTS}::test_render_markdown_claim_pipe_counterexample"))
def visibly_distinct_results(ctx):
    rows, violations = _audit_reports(ctx)
    energy = _energy_origin_study()
    free = _free_energy_study()
    # Validator self-test: a physical finding relabelled with a computational label is refused.
    physical = finding("Synthetic fixture reflects real device energy", "physical", 1.0,
                       {"generator": {"name": "synthetic fixture"}})
    try:
        validate_finding(dict(physical, evidence_status="synthetic"))
        forged = "accepted"
    except EvidenceRefusal as exc:
        forged = str(exc)
    # Rendering counterexample: an unescaped pipe in a claim shifts the label column.
    witness = finding("claim with a | pipe", "numerical", 1.0, {"generator": {"name": "rendering probe"}})
    from .registry import load_queue
    probe = build_report(
        load_queue()["tasks"][99], "partial", {}, [witness])
    row = render_markdown(probe).splitlines()[-1]
    shifted = row.count(" | ") != 2
    pipe_claims = sum(r.get("pipe_claims", 0) for r in rows)
    ctx.artifact_json("visibility-audit.json", {"reports": rows, "violations": violations, "energy": energy,
                                                "free_energy": free, "forged_relabel": forged,
                                                "rendering_probe_row": row})
    findings = []
    audited = bool(rows)
    if audited:
        findings.append(finding(
            "Every retained report of tasks T001-T099 keeps labels among the seven, physical and authority findings "
            "unestablished without acquisition, and renders each finding's label in its Markdown row",
            "computational_pipeline", {"reports_checked": len(rows), "violations": len(violations)},
            {"checks": [_check("label, domain or rendering violations", len(violations))]}, tolerance=EXACT))
    else:
        findings.append(finding(
            "No retained report of tasks T001-T099 was present to audit", "computational_pipeline",
            {"reports_checked": 0}, {}, expected_not_established=True))
    findings += [
        finding("A physical finding relabelled with a computational label is refused by the evidence validator",
                "computational_pipeline", forged,
                {"checks": [_refusal("validate_finding on a physical finding relabelled synthetic",
                                     "Evidence label refused: basis supports not_established, finding states synthetic",
                                     forged)]}, tolerance=EXACT),
        finding("An unescaped pipe character in a finding claim shifts the rendered label out of its Markdown column",
                "computational_pipeline", {"row_cells": row.count(" | ") + 1, "label_column_shifted": shifted},
                {"checks": [_check("rendered rows keeping three cells for a claim containing a pipe", int(not shifted))]},
                tolerance=EXACT, counterexample={
                    "statement": "render_markdown keeps every finding's label in the label column for any claim text",
                    "witness": {"claim": witness["claim"], "row": row}}),
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
                                   int(energy["synthetic"]["eligible"]))]}, tolerance=EXACT),
        finding("A resealed relabel of the synthetic energy fixture under a fresh occurrence is classified as a "
                "physical-domain measurement", "computational_pipeline",
                {"resealed_relabel": energy["resealed_relabel"], "fresh_occurrence_execute": energy["fresh_occurrence"]},
                {"checks": [_refusal("classification of the resealed relabel", "physical_domain_measurement",
                                     energy["resealed_relabel"]["classification"]),
                            _refusal("hardware provenance of the resealed relabel",
                                     "retained_operator_record_not_authenticated",
                                     energy["resealed_relabel"]["hardware_provenance"])]},
                tolerance=EXACT, counterexample={
                    "statement": "CIW energy records can distinguish a genuinely acquired log from a relabelled "
                                 "synthetic fixture",
                    "witness": {"origin": "physical_measurement (declared)", "run_id": "energy-run-" + "2" * 32,
                                "hardware_provenance": energy["resealed_relabel"]["hardware_provenance"]}}),
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
                                   1, "ge", "invariant")]}, tolerance=EXACT),
        finding("The relabelled energy log is a physical GPU energy measurement", "physical",
                "not established: origin is operator-declared and unauthenticated", {}),
        finding("The synthetic energy fixture characterizes real NVML counter accuracy", "sensor_performance",
                "not established: synthetic fixture", {}),
    ]
    fields = _fields(
        "Retained lab reports and CIW's own records keep synthetic, provider-backed and physical results visibly "
        "distinct, and relabelling is refused wherever the record can detect it.",
        "Allowed labels per domain: physical -> {not_established, hardware_measured, independently_verified with "
        "acquisition}; authority -> {not_established}; computational -> never hardware_measured.",
        ["retained reports ctx.output_dir/reports/T001-T099 (as present in this run's directory)",
         "examples/energy-accuracy/baseline.json (synthetic, embedded)", "ciw.free_energy_profile.POLICY",
         "ciw.free_energy_view source text"],
        "validate_report, per-finding label/domain rules, rendered Markdown rows; CIW analysis/refusal outputs.",
        "Zero violations; relabels refused where detectable; physical claims not established.",
        "Audit every retained report below T100; forge a relabelled physical finding; render a claim containing "
        "a pipe character; relabel the energy fixture unsealed, resealed in the same occurrence and resealed in a fresh "
        "occurrence; relabel free-energy source policies; inspect the free-energy truth panel basis.",
        f"{len(rows)} reports audited with {len(violations)} violations; resealed energy relabel classified "
        f"{energy['resealed_relabel']['classification']}; same-occurrence relabel: {energy['same_occurrence_collision']}.",
        "Exact.",
        ["unknown label", "physical finding with computational label", "authority finding established",
         "label missing from rendered row", "origin relabel with and without resealing", "occurrence rebinding",
         "free-energy policy relabel"],
        ["The report audit covers only reports present in the output directory when T100 runs (a full run "
         "retains T001-T099); stale reports from earlier runs in the same directory are included if present.",
         f"Retained claims containing a pipe or newline: {pipe_claims}; the rendering counterexample uses a "
         "synthetic claim."],
        "Escape claim text in ciw.lab.report.render_markdown; authenticate energy-log origin at acquisition "
        "(outside the workbench).")
    return {"state": "completed" if audited else "partial", "fields": fields, "findings": findings}
