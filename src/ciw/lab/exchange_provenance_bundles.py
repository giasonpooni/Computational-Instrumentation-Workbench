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
from copy import deepcopy
from hashlib import sha256
import inspect
import json
from pathlib import Path
import re
import subprocess
import tempfile

from .. import __version__
from .evidence import (AUTHORITY_DOMAINS, COMPUTATIONAL_DOMAINS, LABELS, PHYSICAL_DOMAINS, EvidenceRefusal,
                       finding, holds, validate_finding)
from .registry import task
from .report import build_report, render_markdown, validate_report
from .svg import line_plot
from .exchange_provenance_bundles_fixtures import (EXECUTION_PATHS, GOLDEN_MANIFEST, GOLDEN_PLATFORM,
                                                   GOLDEN_SCR_ENGINE_SHA256, Client, ExecutionForbidden,
                                                   build_energy_session, example_bytes, execution_guard,
                                                   fabricated_heat_catalog, fixture_root, heat_reference,
                                                   malformed_fixtures, merge_catalogs, platform_fingerprint,
                                                   repository_example, source_payload)
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
PROVIDER_PATHS = frozenset(label for label in (
    "subprocess.Popen", "ciw.adapters.subprocess._bounded_process",
    "ciw.adapters.subprocess.PinnedSubprocessAdapter.__init__", "ciw.candidate_evidence._bounded_process",
    "ciw.declared_workload.DeclaredWorkflow._adapters", "ciw.declared_workload.DeclaredWorkflow._step",
    "ciw.declared_workload.DeclaredWorkflow._execute", "ciw.declared_workload.DeclaredWorkflow.create_session",
    "ciw.declared_workload.DeclaredWorkflow.replay_session"))


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
    return {"state": _settle("completed", findings), "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T092

def _unbound_session(scratch: Path):
    """Reopened session holding energy bundles, unbound-kind sources and a fabricated heat bundle.

    Returns the reopened session, identities, the retained source descriptors
    by kind, the fabricated catalog, reopen attempts, the upstream-gated kinds
    whose sources were retained and the repository kinds whose example was not
    reachable.
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
    path = session.save_workspace(scratch / "saved" / "workspace.json")
    workspace = json.loads(path.read_text(encoding="utf-8"))
    fabricated = fabricated_heat_catalog([0, 1, 2, 3, 0])
    workspace["workbench"] = merge_catalogs(workspace["workbench"], fabricated)
    path.write_text(json.dumps(workspace, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with execution_guard() as guard:
        reopened = Session.from_workspace(path, scratch / "reopened")
    ids["heat"] = fabricated["bundles"][0]["bundle_id"]
    return reopened, ids, sources, fabricated, list(guard["attempts"]), upstream, unreachable


def _replay_refusal_cases(ids, sources, upstream=()):
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
                        f"{TESTS}::test_fabricated_heat_bundle_is_content_consistent_but_wrong"))
def replay_refusal_without_binding(ctx):
    with tempfile.TemporaryDirectory(prefix="ciw-lab-t092-") as scratch:
        reopened, ids, sources, fabricated, reopen_attempts, upstream, unreachable = _unbound_session(Path(scratch))
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
    refused_unbound = sorted(set(sources) - set(upstream))
    not_exercised = sorted(set(unavailable) - set(sources))
    findings = [
        finding("Requests to execute or replay unbound provider workflows, or to supply or bypass a binding, are "
                "refused with a named error and reach no provider process or adapter", "computational_pipeline",
                {"requests": len(outcomes), "refused_as_expected": refused,
                 "codes": sorted({row["observed"].split(":", 1)[0] for row in outcomes}),
                 "kinds_refused_unbound": refused_unbound, "kinds_refused_at_upstream_selection": sorted(upstream),
                 "unavailable_kinds_not_exercised": not_exercised},
                {"checks": [_refusal(row["case"], row["expected"], row["observed"]) for row in outcomes]
                 + [_check("provider process or adapter entry points reached", len(provider_reached)),
                    _check("execution entry points reached while reopening", len(reopen_attempts))]},
                uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("A retained numerical-heat bundle whose values no provider computed passes reopen validation",
                "computational_pipeline",
                {"accepted_on_reopen": True, "retained_values": retained, "reference_values": reference,
                 "mismatched_cells": mismatched},
                {"checks": [_check("cells differing from the independent integer reference", mismatched, 1, "ge",
                                   "invariant")]},
                uncertainty=EXACT_INTEGER, tolerance=EXACT, counterexample={
                    "statement": "Reopen validation of a retained numerical-heat bundle establishes that its values "
                                 "are the pinned SCR heat-kernel output",
                    "witness": {"bundle_id": ids["heat"], "retained_values": retained, "reference_values": reference,
                                "runtime_repository_root": heat["runtimes"]["scr"]["repository_root"]}}),
        finding("The binding-free energy-accuracy replay succeeds in the same reopened session", "computational_pipeline",
                control_ok, {"checks": [_check("energy replay without numerical match", int(not control_ok))]},
                uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("A content-consistent reopened bundle is acceptable as a verified production result",
                "production_acceptance", "not decided by the workbench", {}),
    ]
    assumptions = ["No ESM candidate adapter refusal ('No operator-bound ESM candidate adapter') is reached: that "
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
        f"unavailable workflow kinds refused unbound, {len(upstream)} refused at upstream selection, "
        f"{len(not_exercised)} not exercised; provider paths reached: {provider_reached or 'none'}; "
        f"the fabricated heat bundle (values {retained}) reopened although the reference field is {reference}.",
        "Exact string and count comparisons.",
        ["client-supplied repositories in replay/execute payloads", "client binding request",
         "unknown bundle", "unversioned and unregistered operation identities", "ESM on a non-telemetry bundle",
         "upstream bundle of the wrong kind", "provider adapter constructed before refusal"],
        assumptions,
        "T093: show these refusals leave the saved workspace and in-memory state unchanged.")
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
        finding("Refused requests leave the in-memory session state and the session directory unchanged, and a "
                "re-save reproduces the saved workspace content", "computational_pipeline",
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
        "A request that CIW refuses changes neither the session state a save would write nor the session directory.",
        "State = digests of selection, results, executions, retained catalog, catalog revision, byte and "
        "reservation counters, identity claims, candidates and bindings; plus the session directory listing and "
        "the re-saved workspace content without its saved_at timestamp.",
        ["energy-accuracy session (synthetic baseline) with oscillator results and one unbound numerical-heat source"],
        "Exact 'code: message' of each refusal; SHA-256 digests before and after the refused requests.",
        "Each request refused with its expected text; all digests equal; re-saving reproduces the workspace "
        "content except its saved_at timestamp.",
        f"Send {len(outcomes)} refused requests (unbound execution, forged bindings, malformed sources, stale "
        "revision, unknown identities, out-of-range analysis, save redirection, unsupported version), then compare; "
        "reopen a corrupted copy; finally send one refused recording operation separately.",
        f"{refused}/{len(outcomes)} refused with the expected text; changed state parts: {changed_parts or 'none'}; "
        f"corrupted reopen refused with '{reopen_refusal}'; a refused recording operation changed {recorded}.",
        "Exact digest and string equality.",
        ["reservation or pending counters leaked by a refusal", "partial source retained after refusal",
         "selection revision advanced by a refused update", "save path redirected by a client",
         "refused reopen writing into its target directory", "refusal for an unintended reason (text compared)"],
        ["Refused recording operations are retained by design as auditable execution records; this is a documented "
         "exception, not a leak.",
         "The saved workspace file is not rewritten by requests (no request can target it); the evidence is the "
         "state digests and the re-save comparison."],
        "T094: reopen golden retained bundles with the current code.")
    return {"state": _settle("completed", findings), "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T094

# ciw.energy_workflow refuses a retained analysis whose fresh recomputation differs in any bit.
ENERGY_PLATFORM_REFUSAL = "Retained energy analysis binding differs"


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
    from ..proved_heat import PIN as PROVED_HEAT_PIN
    rows, attempts, failures, digests, platform_refusals = {}, [], [], {}, []
    reference_mismatch, heat_refusal, heat_values, heat_runtime = None, None, None, None
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
                if str(exc) == ENERGY_PLATFORM_REFUSAL:
                    # Reopen recomputes the energy analysis (LAPACK solves) and compares it bit for bit.
                    platform_refusals.append(name)
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
                reference_mismatch = sum(a != b for a, b in zip(heat_values, reference)) + abs(
                    len(heat_values) - len(reference))
                heat_runtime = native["runtimes"]["scr"]
                heat_refusal = _outcome(Client(session).call("bundle.replay", {"bundle_id": bundles[0]["bundle_id"]}))
            rows[name] = row
    manifest_mismatch = sum(digests.get(name) != expected for name, expected in GOLDEN_MANIFEST.items())
    fingerprint = platform_fingerprint()
    ctx.artifact_json("golden.json", {"root": "tests/fixtures/lab" if root.name == "lab" else str(root),
                                      "manifest": GOLDEN_MANIFEST, "observed": rows, "failures": failures,
                                      "platform_refusals": platform_refusals, "execution_attempts": attempts,
                                      "platform": fingerprint, "golden_platform": GOLDEN_PLATFORM})
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
                uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("Golden fixture SHA-256 digests", "provenance", dict(sorted(digests.items())),
                {"checks": [_check("digests differing from the recorded manifest", manifest_mismatch)]},
                uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("Reopening recomputes the golden energy analysis bit for bit on this platform", "numerical",
                {"refused_as_platform_dependent": platform_refusals},
                {"checks": [_check("golden workspaces refused with 'Retained energy analysis binding differs'",
                                   len(platform_refusals))]},
                uncertainty={"kind": "roundoff", "value": 0,
                             "basis": "bit-exact canonical JSON comparison of a LAPACK-backed recomputation; "
                                      "depends on the NumPy/LAPACK build and CPU kernels (fingerprint in golden.json)"},
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
        "no provider binding and no execution.",
        "Golden = exact saved bytes recorded in GOLDEN_MANIFEST; validation = Session.from_workspace (structure, "
        "identities, commitments, energy-analysis recomputation) under the execution guard.",
        [f"tests/fixtures/lab/{name}" for name in sorted(GOLDEN_MANIFEST)],
        "SHA-256 of each file; reopen outcome; retained replay receipts; retained SCR values and runtime identity.",
        "All digests match, all reopen, zero execution attempts, replay numerical identity preserved, retained heat "
        "values equal the integer reference.",
        "Hash each golden file, reopen it into a temporary directory under the guard, inspect its bundles, compare "
        "the retained heat field with the integer reference and its runtime identity with CIW's pins, and attempt "
        "an unbound replay.",
        f"{len(rows)} golden workspaces reopened, {len(failures)} failures ({len(platform_refusals)} platform-dependent "
        f"energy recomputations), {manifest_mismatch} digest mismatches; retained heat values {heat_values}.",
        "Exact.",
        ["fixture drift (digest)", "schema or validator drift breaking reopen", "execution during reopen",
         "replay identity drift", "unbound replay of a provider-backed golden bundle",
         "retained runtime identity differing from CIW's pins", "floating-point recomputation drift"],
        ["Reopen recomputes the retained energy analysis in floating point and compares it bit for bit; a "
         "platform whose NumPy/LAPACK rounds differently refuses the golden energy workspace. The platform "
         "fingerprint of this run and of the golden's origin are retained in golden.json.",
         "The retained runtime identity is metadata written by the saving run: it is compared with CIW's pins, "
         "but reopen cannot show that the named engine produced the values.",
         "The golden numerical-heat workspace records host paths of the machine that produced it; they are "
         "identity metadata, never bindings."],
        "T095: refuse malformed exchange fixtures with retained error text.")
    return {"state": state, "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T095

def _attempt(function):
    """Outcome of one validator call; any exception is retained with its type (a crash is not a clean refusal)."""
    try:
        function()
        return {"refused": False, "error_type": None, "message": None}
    except RecursionError as exc:
        return {"refused": True, "error_type": "RecursionError", "message": str(exc)[:200]}
    except ValueError as exc:
        return {"refused": True, "error_type": type(exc).__name__, "message": str(exc)[:500]}
    except Exception as exc:  # retained, never hidden: an unexpected exception type is itself a finding
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
        if not rows[name]["session_read_json"]["refused"]:
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
        rows[label] = {"refused": response["type"] == "error", "error_type": None,
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
    if not accepted["refused"]:
        resaved = Session.from_workspace(extra_path, directory / "extra-resaved").save_workspace(directory / "resaved.json")
        dropped = "lab_unexpected_field" not in json.loads(resaved.read_text(encoding="utf-8"))
    rows["extra top-level workspace field (Session.from_workspace)"] = dict(accepted, dropped_on_resave=dropped)
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
        committed = sum((root / "malformed" / name).read_bytes() != raw if (root / "malformed" / name).is_file() else 1
                        for name, raw in malformed_fixtures().items())
    ctx.artifact_json("malformed-refusals.json", {"committed_fixture_mismatches": committed,
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
    extra = runtime["extra top-level workspace field (Session.from_workspace)"]
    source_text = matrix["duplicate-key.json"]["workbench_source_add"]["message"]
    total = len(MALFORMED_EXPECTED) + len(RUNTIME_EXPECTED)
    parsed_overflow = overflow["session_read_json"].get("parsed_component_value")
    findings = [
        finding("Every malformed exchange fixture is refused by the CIW validator it targets, with the error text "
                "retained", "computational_pipeline", {"fixtures": total, "refused": total - len(unrefused)},
                {"checks": [_check("malformed fixtures not refused with their expected text", len(unrefused))]
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
        "Refused/accepted, exception type and exact message per validator (retained in malformed-refusals.json).",
        "Every fixture refused by the validator it targets with its exact CIW text; wrong-type cases are single-field "
        "mutations of a valid input so the refusal is attributable.",
        "Write each committed fixture from its generator, apply the byte-level validators to each, apply the "
        "structural validators to the runtime-generated cases, and compare exact CIW messages.",
        f"{total - len(unrefused)}/{total} malformed inputs refused with their expected text; read_json accepted "
        f"1e999 as {parsed_overflow} and "
        f"{'raised ' + str(deep['session_read_json']['error_type']) if deep['session_read_json']['refused'] else 'accepted'}"
        f" on 20000-deep nesting; Session.from_workspace raised {crash['error_type']} on a version-only workspace; an "
        "extra top-level workspace field was accepted and dropped on re-save.",
        "Exact string comparison of CIW-generated text; messages from Python's json module and interpreter "
        "exceptions are retained but not compared, because they vary between interpreter versions.",
        ["duplicate keys", "NaN and ±Infinity literals", "float overflow", "truncation", "invalid UTF-8",
         "recursion depth", "wrong schema", "wrong types (single-field mutations)", "incomplete workspace",
         "extra fields", "oversize bytes", "forged identity", "non-canonical base64"],
        ["The workbench validator is exercised through the energy-accuracy kind; other kinds apply their own "
         "source parsers after the same canonical-base64 and byte-budget checks.",
         "The exchange stage stops before the SET validator; conformance of well-formed artifacts needs SET (T097)."],
        "T096: provider-free conformance of exchange identities and ESM candidate responses.")
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


def _identity_study(seed=9601, count=24):
    """exchange._identity over records sealed by the lab's own canonical encoder (not json.dumps).

    ``canonical_text`` is written from the producer specification (sorted
    members, no spaces, literal non-ASCII, JSON escapes, shortest float repr),
    so acceptance is agreement between two encoders, not a copy checking itself.
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
    findings = [
        finding("exchange._identity accepts every synthetic result and verification record sealed by an "
                "independently written canonical encoder, in any member order, and refuses every single-field "
                "mutation and a forged identity", "computational_pipeline",
                {key: identity[key] for key in ("valid", "accepted", "reordered_accepted", "encoder_agreement",
                                                "mutations", "refused")},
                {"generator": {"name": "ciw.lab seeded nested JSON records", "seed": 9601, "count": 24},
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
                {"checks": [_check("valid synthetic responses refused", wrong_valid),
                            _check("boundary mutations accepted", wrong_mutated)]},
                uncertainty=EXACT_COUNT, tolerance=EXACT),
        finding("Observation-batch identities are caller-declared: mutated batches pass exchange._identity",
                "computational_pipeline",
                {"mutations": identity["batch_mutations"], "accepted": identity["batch_mutations_accepted"]},
                {"checks": [_check("mutated batches accepted", identity["batch_mutations_accepted"], 1, "ge",
                                   "invariant")]},
                uncertainty=EXACT_COUNT, tolerance=EXACT, counterexample={
                    "statement": "Every exchange artifact identity is bound to its content",
                    "witness": {"schema": "notation.instrument.observation-batch.v1",
                                "identity_status": "caller_declared_reference"}}),
        finding("validate_response accepts unknown extra fields in an ESM response", "computational_pipeline",
                {"cases": len(extras), "accepted": extras_accepted},
                {"checks": [_check("responses with unknown fields accepted", extras_accepted, 1, "ge", "invariant")]},
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
        ["24 seeded synthetic result/verification records (PCG64 seed 9601)",
         f"{len(candidates)} synthetic ESM inspect/capture responses over telemetry and calibrated bundle stubs"],
        "Accepted or refused (with message) per record or response.",
        "All valid accepted, all single-field mutations refused, identity invariant under member order.",
        "Seal records with the lab's own canonical encoder (checked against json.dumps), re-validate them with "
        "members reordered, mutate each non-identity field and the identity itself; build valid ESM responses and "
        "apply every boundary mutation.",
        f"{identity['refused']}/{identity['mutations']} identity mutations refused; "
        f"{len(mutated) - wrong_mutated}/{len(mutated)} candidate mutations refused; {extras_accepted} response(s) "
        "with unknown fields accepted.",
        "Exact.",
        ["member-order dependence", "identity forgery", "admission/state/truth flags",
         "binding to request, time, digest and bundle", "capture receipt consistency"],
        ["Acceptance of valid records is agreement between the lab's canonical encoder and exchange._identity "
         "(both CIW-side code written to the same producer specification); producer-sealed artifacts are "
         "accepted only in T097's PPDA/SCR/SET roundtrip, when those checkouts are bound.",
         "JSON escaping (for example \\u00e9 for é) is resolved by the JSON parser before _identity sees a "
         "record, so it is not an identity property and is not tested here.",
         "The bundle stubs contain only the fields validate_response reads; full native bundles need providers.",
         "Mutations are single-field; combined mutations that restore consistency are not explored."],
        "T097: exact SCR/SET/PPDA integrations when the pinned checkouts are bound.")
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


def _set_findings(result) -> list:
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
         "notes": {"provider": {"repository": providers.REPOSITORIES["set"], "revision": result["validator"]["revision"],
                                "validator_path": result["validator"]["path"],
                                "validator_sha256": result["validator"]["sha256"], "executed": True}}},
        uncertainty=EXACT_COUNT, tolerance=EXACT)]


def _roundtrip_findings(result, optional) -> list:
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
         "notes": {"provider": {role: {"repository": providers.REPOSITORIES[role], "revision": optional[role]["head"],
                                       "source_tree": optional[role]["tree"], "executed": True}
                                for role in ("ppda", "scr-exchange", "set")}}},
        uncertainty=EXACT_COUNT, tolerance=EXACT)]


@task("T097", changed_files=T097_FILES,
      regression_tests=(f"{TESTS}::test_t097_scr_numerical_heat_integration",
                        f"{TESTS}::test_t097_is_blocked_without_scr",
                        f"{TESTS}::test_t097_keeps_set_results_when_the_engine_is_missing",
                        f"{TESTS}::test_t097_t099_refuse_a_non_repository_scr_binding"),
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
    ctx.artifact_json("integration.json", {
        "scr_identity": {k: identity[k] for k in ("head", "tree", "tracked_sha256", "tracked_files", "cargo_locks")},
        "scr_pins": comparison, "engine": None if engine is None else {"origin": engine["origin"], "sha256": engine["sha256"]},
        "optional_providers": optional, "blocked": blocked, "set_reason": set_reason,
        "roundtrip_reason": roundtrip_reason or None, "parts": parts})
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
                     "notes": _engine_note(engine)},
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
                                       int(not descriptor_equal))]},
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
        findings += _set_findings(parts["set"])
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
        recommended_next_task="T098: record the identities of every bound provider checkout.",
        provider_runtime_identity=_runtime_identity(T097_FILES, {
            "scr": {"revision": identity["head"], "source_tree": identity["tree"],
                    "engine_sha256": None if engine is None else engine["sha256"],
                    "engine_origin": None if engine is None else engine["origin"]},
            **{role: {"state": value["state"], "head": value.get("head"), "matched": value.get("matched")}
               for role, value in optional.items()}}))
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


def _adapter_outcomes(role, path, identity, pins) -> list:
    """CIW's own subprocess adapter against every module pin CIW declares for the role."""
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
        rows.append({"declared_in": pin["declared_in"], "revision": pin["revision"],
                     "at_head": pin["revision"] == identity["head"], "outcome": outcome})
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
    ctx.artifact_json("provider-identities.json", {
        "identities": identities, "comparisons": comparisons, "adapter_pin_checks": adapters, "errors": errors,
        "ciw_pins": pins, "pins_by_repository": consistency["by_repository"], "engine": None if engine is None else {
            "origin": engine["origin"], "sha256": engine["sha256"], "byte_count": len(engine["binary"]),
            "toolchain": None if engine["build"] is None else {k: engine["build"][k] for k in ("cargo", "rustc")}}})
    input_data = [f"{role}: {providers.REPOSITORIES[role]} at {identities[role]['head']}" if role in identities
                  else f"{role}: not a readable Git repository root" for role in roles]
    if not identities:
        fields = _fields(
            "Every bound provider checkout is clean and at a CIW pin, and its bytes reproduce its Git tree.",
            "Identity = (HEAD, HEAD^{tree}, SHA-256 over tracked working bytes, Cargo.lock SHA-256, engine SHA-256).",
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
    again = {role: providers.checkout_identity(ctx.providers[role])["tracked_sha256"] for role in identities}
    findings.append(finding(
        "Tracked-source and Cargo.lock digests of every readable bound checkout", "provenance",
        {role: {"tracked_sha256": identities[role]["tracked_sha256"], "tracked_files": identities[role]["tracked_files"],
                "cargo_locks": identities[role]["cargo_locks"]} for role in sorted(identities)},
        {"checks": [_check("digests that change on an immediate second read",
                           sum(again[role] != identities[role]["tracked_sha256"] for role in identities))]},
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
            "refuses it for every other module pin", "provenance",
            {role: {"at_head": sorted(row["declared_in"] for row in adapters[role] if row["at_head"]),
                    "refused": sorted(row["declared_in"] for row in adapters[role] if row["outcome"] != "accepted")}
             for role in sorted(adapters) if adapters[role]},
            {"checks": checks}, uncertainty=EXACT_COUNT, tolerance=EXACT))
    if liveness is not None:
        findings.append(finding(
            "The SCR engine recorded for this run executes the SCR heat descriptor on the survey input",
            "numerical", {"engine_origin": engine["origin"],
                          "cargo_lock_sha256": identities["scr"]["cargo_locks"].get("crates/Cargo.lock"),
                          "survey_output": liveness["cases"][0]["values"],
                          "descriptor_sha256": liveness["descriptor_sha256"]},
            _provider_basis(identities["scr"], engine), uncertainty=EXACT_INTEGER, tolerance=EXACT))
    findings += _engine_origin_findings(ctx, engine)
    findings.append(finding(
        "A matching HEAD, tree and lock digest authenticates the upstream repository, toolchain and built engine",
        "provenance", "not established: digests are not signatures", {}, expected_not_established=True))
    state = _settle("partial" if refused or errors else "completed", findings)
    adapter_refused = sorted(role for role, role_rows in adapters.items()
                             if any(row["outcome"] != "accepted" for row in role_rows))
    fields = _fields(
        "Every bound provider checkout is clean and at a CIW pin; its working bytes reproduce its Git tree; its "
        "lockfiles and the engine used in this run are recorded.",
        "Git object ids: blob = H('blob' len NUL bytes), tree = H('tree' len NUL sorted(mode name NUL id)); "
        "tracked digest = SHA-256 over 'mode kind sha256(bytes) path' lines in ls-tree order.",
        input_data,
        "git rev-parse, git ls-tree, working-tree bytes, CIW pin tables, PinnedSubprocessAdapter outcomes.",
        "HEAD equals a CIW pin; pinned trees equal; recomputed tree equals Git's; digests stable.",
        "For each bound role read HEAD and tree, hash every tracked file, recompute the tree independently of "
        "Git, compare with every CIW pin for that role (refusing mismatches with their reasons), and let CIW's own "
        "adapter accept or refuse each module pin; execute the SCR engine once on the survey input when available.",
        f"accepted {accepted}, refused {refused}, unreadable {sorted(errors)}; {tree_mismatch} tree recomputation "
        f"mismatches among clean checkouts; adapter refusals for {adapter_refused}; "
        f"engine {None if engine is None else engine['origin']}.",
        "Exact digests.",
        ["dirty or untracked files", "wrong revision", "pinned tree drift", "modified tracked bytes",
         "executable-bit drift", "one checkout serving several different CIW pins", "adapter accepting a wrong pin"],
        ["Engine and interpreter digests depend on the toolchain; they are retained as provenance in the artifact "
         "and the basis notes, not compared as regression values.",
         "Pins declared only in .github/workflows/exchange.yml are mirrored in EXCHANGE_WORKFLOW_PINS.",
         "Checkout paths are kept in provider-identities.json only; report prose names repositories and revisions."],
        "T099: verify the locked offline SCR build and record the SP1 build requirements.")
    bound = {role: {"head": identities[role]["head"], "tree": identities[role]["tree"]} for role in sorted(identities)}
    if engine is not None:
        bound["scr"]["engine_sha256"] = engine["sha256"]
    fields["provider_runtime_identity"] = _runtime_identity(changed, bound)
    return {"state": state, "fields": fields, "findings": findings}


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
                        f"{TESTS}::test_t097_t099_refuse_a_non_repository_scr_binding"),
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
    lock = identity["cargo_locks"].get("crates/Cargo.lock")
    provider_note = {"repository": providers.REPOSITORIES["scr"], "revision": identity["head"],
                     "source_tree": identity["tree"], "binary_sha256": digests}
    findings = [finding(
        BUILD_CLAIM, "computational_pipeline",
        {"exit_codes": [row["returncode"] for row in build["builds"]], "flags": ["--release", "--locked", "--offline"],
         "package": "execution-cli", "cargo_lock_sha256": lock, "builds": len(build["builds"])},
        {"checks": [_check("builds with a nonzero or missing exit code",
                           sum(row["returncode"] != 0 for row in build["builds"])),
                    _check("builds without an execution-cli binary",
                           sum(not row["binary_sha256"] for row in build["builds"]))],
         "notes": {"provider": provider_note}},
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
                                _check("distinct binary digests beyond one", len(digests) - 1)]},
                    uncertainty=EXACT_COUNT, tolerance=EXACT),
            finding("The freshly built engine's heat output equals an independent integer reference", "numerical",
                    values, {"independent_check": {
                        **_check("cells differing from the integer Jacobi reference",
                                 sum(a != b for a, b in zip(values or [], reference)) + (0 if values else len(reference))),
                        "producer": {"implementation": "Scientific-Computation-Runtime execution-cli (locked build)",
                                     "revision": identity["head"]},
                        "checker": {"implementation": "ciw.lab.exchange_provenance_bundles_fixtures.heat_reference",
                                    "revision": __version__}}},
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
        recommended_next_task="Run the SP1 proved-heat gate (scripts/check_proved_heat.py) on a provisioned machine; "
                              "then T100.",
        provider_runtime_identity=_runtime_identity(T099_FILES, {
            "scr": {"head": identity["head"], "tree": identity["tree"], "engine_sha256": digests[0] if digests else None,
                    "cargo": build["cargo"], "rustc": build["rustc"]}}))
    return {"state": "partial", "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T100

_COMPUTATIONAL_LABELS = frozenset({"analytic", "synthetic", "numerically_verified", "provider_backed"})
# The packaged queue's tasks before T100 (T001-T099).
EXPECTED_EARLIER_REPORTS = 99


def _cells(row: str) -> int:
    """Cells of a Markdown table row: unescaped pipes minus one (GFM drops cells beyond the header's)."""
    return len(re.findall(r"(?<!\\)\|", row)) - 1


def _audit_reports(ctx):
    """Label and rendering audit of every retained report numbered below 100 present in the output directory."""
    directory = ctx.output_dir / "reports"
    rows, labels, rendering = [], [], []
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
        table = markdown.split("| Finding | Value | Evidence status |\n| --- | --- | --- |\n", 1)
        lines = table[1].splitlines() if len(table) == 2 else []
        for index, record in enumerate(report["findings"]):
            label, domain = record["evidence_status"], record["domain"]
            if label not in LABELS:
                labels.append(f"{report['task_id']}[{index}]: unknown label {label}")
            if domain in PHYSICAL_DOMAINS and (label in _COMPUTATIONAL_LABELS or
                                               (label != "not_established" and not record["basis"].get("acquisition"))):
                labels.append(f"{report['task_id']}[{index}]: physical finding labelled {label}")
            if domain in AUTHORITY_DOMAINS and label != "not_established":
                labels.append(f"{report['task_id']}[{index}]: authority finding labelled {label}")
            if domain not in PHYSICAL_DOMAINS and label == "hardware_measured":
                labels.append(f"{report['task_id']}[{index}]: computational finding labelled hardware_measured")
            row = lines[index] if index < len(lines) else ""
            if not row.endswith(f"| `{label}` |") or _cells(row) != 3:
                rendering.append(f"{report['task_id']}[{index}]: rendered row does not show `{label}` in its column")
        if f"`{report['physical_validation_status']['status']}`" not in markdown:
            rendering.append(f"{report['task_id']}: physical validation status not rendered")
        rows.append({"task_id": report["task_id"], "findings": len(report["findings"]),
                     "pipe_claims": sum("|" in f["claim"] or "\n" in f["claim"] for f in report["findings"]),
                     "labels": sorted({f["evidence_status"] for f in report["findings"]}),
                     "physical_or_authority": sum(f["domain"] in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS
                                                  for f in report["findings"])})
    return rows, labels, rendering


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
    """Render a synthetic claim containing a pipe; returns the finding row and whether its label kept its column."""
    from .registry import load_queue
    witness = finding("claim with a | pipe", "numerical", 1.0, {"generator": {"name": "rendering probe"}})
    task_record = next(item for item in load_queue()["tasks"] if item["id"] == "T100")
    row = render_markdown(build_report(task_record, "partial", {}, [witness])).splitlines()[-1]
    kept = _cells(row) == 3 and row.endswith(f"| `{witness['evidence_status']}` |")
    return witness, row, kept


def _probe_finding(witness, row, kept):
    """Records whichever renderer behaviour was observed, so the finding holds before and after escaping lands."""
    if kept:
        return finding("render_markdown keeps the label column for a claim containing a pipe character",
                       "computational_pipeline", {"row_cells": _cells(row), "label_column_shifted": False},
                       {"checks": [_check("rendered cells beyond three for a claim containing a pipe", _cells(row) - 3)]},
                       uncertainty=EXACT_COUNT, tolerance=EXACT)
    return finding("An unescaped pipe character in a finding claim shifts the rendered label out of its Markdown column",
                   "computational_pipeline", {"row_cells": _cells(row), "label_column_shifted": True},
                   {"checks": [_check("rendered rows keeping three cells for a claim containing a pipe", int(kept))]},
                   uncertainty=EXACT_COUNT, tolerance=EXACT, counterexample={
                       "statement": "render_markdown keeps every finding's label in the label column for any claim text",
                       "witness": {"claim": witness["claim"], "row": row}})


@task("T100", changed_files=(MODULE, FIXTURES),
      regression_tests=(f"{TESTS}::test_t100_labels_and_origins_stay_distinct",
                        f"{TESTS}::test_t100_flags_a_retained_report_whose_label_leaves_its_column",
                        f"{TESTS}::test_render_markdown_pipe_probe_matches_the_renderer"))
def visibly_distinct_results(ctx):
    rows, violations, rendering = _audit_reports(ctx)
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
    witness, row, kept = _rendering_probe()
    pipe_claims = sum(r.get("pipe_claims", 0) for r in rows)
    present = [r["task_id"] for r in rows]
    ctx.artifact_json("visibility-audit.json", {"reports": rows, "task_ids_audited": present,
                                                "reports_expected_for_the_full_queue": EXPECTED_EARLIER_REPORTS,
                                                "label_violations": violations, "rendering_violations": rendering,
                                                "energy": energy, "free_energy": free, "forged_relabel": forged,
                                                "rendering_probe_row": row})
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
            {"checks": [_check("findings whose rendered label leaves its column", len(rendering))]},
            uncertainty=EXACT_COUNT, tolerance=EXACT))
    else:
        findings.append(finding(
            "No earlier report was present in the output directory to audit", "computational_pipeline",
            coverage, {}, expected_not_established=True))
    fresh = energy["fresh_bundle"] or {}
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
        ["retained reports ctx.output_dir/reports/T001-T099 (only those present when T100 runs)",
         "examples/energy-accuracy/baseline.json (synthetic, embedded)", "ciw.free_energy_profile.POLICY",
         "ciw.free_energy_view source text"],
        "validate_report, per-finding label/domain rules, rendered Markdown rows; CIW analysis/refusal outputs and "
        "the retained bundle of a fresh-occurrence relabel.",
        "Zero violations; relabels refused where detectable; physical claims not established.",
        "Audit every retained report below T100 present in the output directory; forge a relabelled physical "
        "finding; render a claim containing a pipe character; relabel the energy fixture unsealed, resealed in the "
        "same occurrence and resealed in a fresh occurrence (reading the retained bundle's classification); relabel "
        "free-energy source policies; inspect the free-energy truth panel basis.",
        f"{len(rows)} of {EXPECTED_EARLIER_REPORTS} earlier reports present and audited: {len(violations)} label "
        f"violations, {len(rendering)} rendering violations; the pipe probe "
        f"{'kept' if kept else 'lost'} its label column; fresh-occurrence energy relabel "
        f"{energy['fresh_occurrence']} and classified {fresh.get('classification')}; same-occurrence relabel: "
        f"{energy['same_occurrence_collision']}.",
        "Exact.",
        ["unknown label", "physical finding with computational label", "authority finding established",
         "label missing from rendered row", "origin relabel with and without resealing", "occurrence rebinding",
         "free-energy policy relabel"],
        ["The report audit covers only reports present in the output directory when T100 runs: a section-only run "
         "audits its own section, a full run T001-T099; stale reports from earlier runs in the same directory are "
         "included if present (task ids in visibility-audit.json).",
         f"Retained claims containing a pipe or newline: {pipe_claims}; the rendering probe uses a synthetic claim."],
        ("Authenticate energy-log origin at acquisition (outside the workbench)." if kept else
         "Escape claim text in ciw.lab.report.render_markdown; authenticate energy-log origin at acquisition "
         "(outside the workbench)."))
    return {"state": _settle("completed" if audited else "partial", findings), "fields": fields, "findings": findings}
