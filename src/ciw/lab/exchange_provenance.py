"""CIW exchange and provenance, part 1: identity matrix and workspace mutations (T077-T090).

Scope: experiments on the real CIW integrity layer, run completely offline: an
oscillator recording session (``statistics.v1`` operations sealed by the
operation runner, plus one legacy ``analysis.stats`` result), the binding-free
energy-accuracy workbench workflow (retained source bytes, analysis bundles,
replays and replay receipts), saved workspaces reopened with
``Session.from_workspace``, and the pure exchange and ESM candidate
validators. T077 records how every identity is derived, what it binds, whether
it survives replay and reopen and where it is validated, and checks each
property empirically. T078-T083 test retention, separation, stability,
freshness and receipt binding. T084-T090 edit saved workspaces, recompute
whatever unkeyed digests an edit needs to be self-consistent, reopen, and
record whether the edit is refused and with which exact message.

Non-claims: a SHA-256 content identity or an unkeyed record seal establishes
consistency, not authorship or authenticity; the retained source bytes are
themselves unauthenticated, so re-analysis on reopen protects results only
relative to those bytes. An accepted forgery is recorded as a counterexample,
never repaired, and no CIW code is changed here. The energy logs are synthetic
fixtures, so nothing here bears on real GPU energy, sensor performance,
admission authority, or verification by another party.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import uuid

from .evidence import finding, holds as compare
from .exchange_provenance_common import (
    FORGED_CODE, FORGED_DIGEST, FORGED_RUNTIME, FORGED_SESSION, FORGED_TIME, KIND, OPERATION, Mutant, View,
    attempt, build_session_fixture, build_variant_fixture, build_verification, check_esm, edited_log, esm_case,
    exchange_artifact, fixture_available, measurement_edit, reforge, reforge_source, relabel, reopen, request,
    reseal_oscillator, reseal_receipt, restep, role_labels, run_mutant, telemetry_shaped_bundle, validator_row,
    verification_id)
from .registry import task

MODULE = "src/ciw/lab/exchange_provenance.py"
COMMON = "src/ciw/lab/exchange_provenance_common.py"
TESTS = "tests/test_lab_exchange_provenance.py"
DOC = "docs/lab/EXCHANGE_PROVENANCE.md"
CHANGED = (MODULE, COMMON, TESTS, DOC)
EXACT = {"abs": 0.0, "rel": 0.0}
EXACT_UNC = {"kind": "exact", "value": 0.0,
             "basis": "exact equality of SHA-256 digests, bytes, counts, booleans or refusal strings; no rounding"}
VERIFY_SCHEMA = "ciw.declared-workload-verification.v1"
SCIENTIFIC_FIELDS = ("instrument", "metadata", "time_s", "channels")
# Bundles that analyse the baseline log: original, sibling, replay, replay
# after reopen, replay of a replay, separate session (same process and code).
SAME_SOURCE = ("B0", "B0b", "B1", "B2", "B3", "B4")
ALL_BUNDLES = SAME_SOURCE[:2] + ("Bother",) + SAME_SOURCE[2:]
UUID_HEX = "4" + "0" * 11 + "4" + "0" * 3 + "8" + "0" * 15
OSCILLATOR_INPUT = ("make_demo_run() damped-oscillator recording; statistics.v1 on channel v over [1, 2] s, twice, "
                    "plus one legacy analysis.stats result")
WORKSPACE_INPUT = "Workspaces saved by Session.save_workspace and reopened by Session.from_workspace in temporary directories"
SESSION_INPUTS = [OSCILLATOR_INPUT,
                  "examples/energy-accuracy/{baseline,reset}.json (origin synthetic_fixture): two original executions "
                  "of baseline, one of reset, replays, and a separate session in the same process",
                  WORKSPACE_INPUT]
VARIANT_INPUT = ("examples/energy-accuracy/{baseline,reset,missing,under-target}.json, eight byte variants of baseline "
                 "under one shared label (original, crlf, minified, tab-indented, sorted-keys, reversed-keys, "
                 "trailing-whitespace, float-spelling 1e-09 -> 0.000000001) and two resealed content variants "
                 "(sensor name renamed; initial_covariance diagonal written as integers)")
OBSERVATION = ("No physical observation. The observed objects are CIW records (sessions, executions, results, "
               "bundles, receipts, saved workspace JSON) produced offline by the unmodified CIW code; the energy "
               "logs are synthetic fixtures and no hardware was sampled.")
COMPARISON = ("Digest re-derivations use a local canonical-JSON/SHA-256 implementation in ciw.lab, the same "
              "implementation origin as CIW, so they are cross_implementation checks (numerically_verified, never "
              "independently_verified); every other check compares CIW's own outputs or refusal messages.")
BOM_MESSAGE = "The bound runtime did not return finite, unambiguous JSON"
LOG_DIGEST = "Retained log digest differs"
ESM_SCOPE = "Native ESM inspection binding or scope mismatch"


def _node(name: str) -> str:
    return f"{TESTS}::{name}"


def _canon(value, ascii_only=False) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=ascii_only,
                      allow_nan=False).encode("utf-8")


def _sha(value, ascii_only=False) -> str:
    return "sha256:" + hashlib.sha256(_canon(value, ascii_only)).hexdigest()


def _without(record: dict, *keys) -> dict:
    return {key: value for key, value in record.items() if key not in keys}


def _uuid4(value, prefix: str) -> bool:
    if not isinstance(value, str) or not value.startswith(prefix):
        return False
    try:
        parsed = uuid.UUID(hex=value[len(prefix):])
    except ValueError:
        return False
    return parsed.version == 4 and parsed.hex == value[len(prefix):]


def _refusal_message(call) -> str:
    """The exact CIW refusal message, or ``accepted``; any other exception is a harness failure."""
    try:
        call()
    except ValueError as exc:
        return str(exc)
    return "accepted"


def _collision(count: int) -> dict:
    return {"kind": "collision_bound", "value": count * (count - 1) / 2.0 ** 123,
            "basis": f"uuid4 carries 122 random bits; P(any collision among {count} draws) <= n(n-1)/2^123"}


# ------------------------------------------------------------------ checks
def _exact(reference: str, mismatches: int, kind: str = "exact_arithmetic") -> dict:
    return {"reference_kind": kind, "reference": reference, "observed": float(mismatches),
            "tolerance": 0.0, "comparison": "abs_le", "passed": mismatches == 0}


def _rederived(reference: str, mismatches: int) -> dict:
    """A CIW digest re-derived by the lab's same-origin canonical JSON/SHA-256 code."""
    return _exact(reference, mismatches, "cross_implementation")


def _invariant(reference: str, observed, tolerance=0.0, comparison="abs_le") -> dict:
    observed, tolerance = float(observed), float(tolerance)
    holds = compare(observed, tolerance, comparison)
    return {"reference_kind": "invariant", "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": holds}


def _refusal(reference: str, expected: str, observed: str) -> dict:
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed, "passed": observed == expected}


def _count(reference: str, observed: int, expected: int) -> dict:
    """An exact count compared with its prediction (difference must be zero)."""
    return _exact(f"{reference} (observed {observed}, predicted {expected})", observed - expected)


def _state(findings) -> str:
    """Completed only when every computational finding is established or declared unestablished."""
    from .evidence import AUTHORITY_DOMAINS, PHYSICAL_DOMAINS
    for record in findings:
        if (record["evidence_status"] == "not_established" and record["domain"] not in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS
                and not record.get("expected_not_established")):
            return "partial"
    return "completed"


def _verified(claim, value, checks, uncertainty=None, tolerance=None, **options) -> dict:
    """A provenance finding supported by passing checks; exact uncertainty and tolerance unless stated."""
    return finding(claim, "provenance", value, {"checks": checks}, uncertainty=uncertainty or EXACT_UNC,
                   tolerance=tolerance or EXACT, **options)


# ------------------------------------------------------------------ shared computations
def _temporary():
    return tempfile.TemporaryDirectory(prefix="ciw-lab-exchange-", ignore_cleanup_errors=True)


def _fixture(ctx) -> dict:
    def compute():
        with _temporary() as root:
            return build_session_fixture(Path(root))
    return ctx.memo("exchange-provenance.fixture", compute)


def _variants(ctx) -> dict:
    def compute():
        with _temporary() as root:
            return build_variant_fixture(Path(root))
    return ctx.memo("exchange-provenance.variants", compute)


def _harness(fixture: dict, root: Path) -> dict:
    """Controls: the forger's recomputation reproduces CIW exactly, and an unedited workspace reopens."""
    workspace = deepcopy(fixture["workspace"])
    reforge(workspace)
    view = View(workspace, fixture)
    reseal_oscillator(view.E1, view.E2, view.R1, view.R2)
    control, _ = reopen(deepcopy(fixture["workspace"]), root)
    return {"reforge_reproduces_ciw_digests": _canon(workspace) == _canon(fixture["workspace"]),
            "unchanged_workspace_reopens": control["outcome"] == "accepted"}


def _mutations(ctx) -> dict:
    fixture = _fixture(ctx)

    def compute():
        labels = role_labels(fixture)
        with _temporary() as root:
            rows = [run_mutant(mutant, fixture, Path(root), labels) for mutant in MUTANTS]
            harness = _harness(fixture, Path(root))
        return {"rows": rows + _validator_rows(fixture), "harness": harness}
    return ctx.memo("exchange-provenance.mutations", compute)


def _moved_step(fixture: dict) -> dict:
    """Move one energy step to a new execution occurrence and let CIW's own step validator judge it."""
    from ..energy_workflow import EnergyAccuracyWorkflow
    from ..telemetry import byte_digest
    workflow = EnergyAccuracyWorkflow()
    raw = fixture["raw"]["baseline"]
    source, evidence = workflow._source(raw), byte_digest(raw)
    step = deepcopy(fixture["natives"]["B0"]["steps"][0])
    before = {key: step[key] for key in ("operation_id", "result_id", "numerical_result_id")}
    step["execution_id"] = step["result"]["execution_ref"] = "execution-" + UUID_HEX
    stale = _refusal_message(lambda: workflow._validate_step(deepcopy(step), source, evidence))
    restep(step)
    recomputed = _refusal_message(lambda: workflow._validate_step(deepcopy(step), source, evidence))
    return {"stale_result_id": stale, "recomputed_result_id": recomputed,
            "result_id_changed": step["result_id"] != before["result_id"],
            "numerical_result_id_unchanged": step["numerical_result_id"] == before["numerical_result_id"],
            "operation_id_unchanged": step["operation_id"] == before["operation_id"]}


def _blocked(fields: dict, findings=()) -> dict:
    """A blocked report still records the physical or authority claims it could never establish."""
    fields = dict(fields)
    reason = ("Blocked: the bundled energy-accuracy fixture logs (examples/energy-accuracy/*.json) are not reachable "
              "through ciw.lab.runner.repository_path (set CIW_LAB_REPOSITORY_ROOT), so the offline workbench path "
              "cannot be exercised.")
    fields["experiment"] = reason + " Planned: " + fields.get("experiment", "")
    fields["unresolved_assumptions"] = list(fields.get("unresolved_assumptions", [])) + [reason]
    return {"state": "blocked", "fields": fields, "findings": list(findings)}


def _physical_logs(origins=None) -> dict:
    """``origins`` are read from the logs; None in a blocked run, where nothing was observed."""
    value = {"observed": False} if origins is None else {"declared_origins": origins}
    notes = ("The logs were not reachable in this run." if origins is None else
             f"The bundled logs declare origin {', '.join(origins)} and no device was sampled here.")
    return finding("The retained energy logs are real GPU energy measurements", "physical", value,
                   {"notes": notes + " Byte retention says nothing about the physical truth of a log."},
                   tolerance=EXACT)


def _admission_authority(admissions=None) -> dict:
    """``admissions`` are read from the retained receipts; None in a blocked run."""
    value = {"observed": False} if admissions is None else {"receipt_admissions": admissions}
    return finding("A retained replay receipt or verification authorizes admission of the replayed result into "
                   "canonical state", "production_acceptance", value,
                   {"notes": "Admission is an authority decision outside the workbench; the retained receipts record "
                             "admission not_performed and CIW refuses any other value."}, tolerance=EXACT)


def _fields(hypothesis, model, invariant, experiment, result, uncertainty, failures, assumptions, next_task,
            inputs=None) -> dict:
    return {"hypothesis": hypothesis, "mathematical_model": model, "input_data": inputs or list(SESSION_INPUTS),
            "observation_model": OBSERVATION, "expected_invariant": invariant, "experiment": experiment,
            "numerical_result": result, "uncertainty": uncertainty, "failure_modes_checked": failures,
            "unresolved_assumptions": assumptions, "recommended_next_task": next_task}


def _accepted(row: dict) -> bool:
    return row["observed_outcome"] == "accepted"


def _survived(row: dict) -> bool:
    """Accepted although a refusal is what a reader would expect; documented acceptances are not survivors."""
    return _accepted(row) and "accepted_by_design" not in row


def _authentication(rows) -> dict:
    survivors = sorted(row["name"] for row in rows if _survived(row) and row["kind"] == "workspace")
    return finding(
        "Retained workspace records are authenticated: a holder without a secret cannot produce a forged record "
        "that reopens", "provenance", {"surviving_mutants_in_this_task": survivors},
        {"notes": "Every CIW seal and identity on these paths (record_digest, log_digest, bundle_digest, "
                  "verification_id, replay_id, result_id, source_id) is an unkeyed SHA-256 over public canonical "
                  "JSON or raw bytes, and no keyed signature or MAC is retained. Authentication is therefore not "
                  "established; the listed surviving mutants are concrete forgeries accepted on reopen."},
        tolerance=EXACT, expected_not_established=True)


def _witness_checks(row: dict) -> list:
    """Survivors whose claim says more than 'reopens' also check that the reader sees the forged content."""
    seen = row.get("post_reopen", {})
    if row["name"] == "energy-source.resealed":
        computed = seen.get("gross_energy_j_computed_from_original_bytes")
        forged = seen.get("gross_energy_j_after_reopen", {})
        return [_exact("reopened baseline bundles (B0, B0b, B1) not reporting a gross energy different from the "
                       "original bytes' value", 3 - sum(value != computed for value in forged.values()))]
    if row["name"] == "oscillator-stats.impossible-moments":
        violated = sum(not held for record in seen.get("retained_inequalities", {}).values() for held in record.values())
        return [_count("moment inequalities violated by the reopened results", violated, 2)]
    if row["name"] == "receipt-replayed.reidentified-bundle":
        return [_exact("reopened replay not dated before its source bundle", 0 if seen.get("replay_predates_source") else 1)]
    return []


def _survivor(row: dict, claim: str, statement: str) -> dict:
    witness = {key: row[key] for key in ("name", "target", "recompute", "description", "observed")}
    if "post_reopen" in row:
        witness["post_reopen"] = row["post_reopen"]
    return _verified(claim, {"mutant": row["name"], "observed": row["observed"]},
                     [_invariant(f"{row['name']}: accepted (1 = accepted)", 1.0 if _accepted(row) else 0.0, 1.0, "ge")]
                     + _witness_checks(row), counterexample={"statement": statement, "witness": witness})


def _kills(rows, claim: str, harness: dict | None = None) -> dict:
    """One finding over every predicted refusal: each must be refused with its pinned message."""
    predicted = [row for row in rows if row["predicted_outcome"] == "killed"]
    checks = [_refusal(f"{row['name']} ({row['recompute']} recompute): refusal on reopen" if row["kind"] == "workspace"
                       else f"{row['name']}: validator refusal", row["pinned_message"], row["observed"])
              for row in predicted]
    if harness is not None:
        checks.append(_rederived("harness control: the forger's recomputation of the unedited workspace reproduces "
                                 "every CIW digest", 0 if harness["reforge_reproduces_ciw_digests"] else 1))
        checks.append(_exact("harness control: the unedited saved workspace reopens",
                             0 if harness["unchanged_workspace_reopens"] else 1))
    value = [{"mutant": row["name"], "kind": row["kind"], "recompute": row["recompute"], "observed": row["observed"]}
             for row in predicted]
    return _verified(claim, value, checks)


def _table(headers, rows) -> str:
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _verdict(row) -> str:
    if row["killed"]:
        return "killed"
    return "accepted (by design)" if "accepted_by_design" in row else "SURVIVED"


def _matrix_markdown(rows) -> str:
    return _table(("Task", "Mutant", "Kind", "Target", "Recompute", "Predicted", "Pinned message", "Observed",
                   "Outcome"),
                  [(row["task"], row["name"], row["kind"], row["target"], row["recompute"], row["predicted_outcome"],
                    row["pinned_message"], row["observed"], _verdict(row)) for row in rows])


def _retain_rows(ctx, rows, stem: str) -> None:
    ctx.artifact_json(f"{stem}.json", {"schema": "ciw.lab-mutation-matrix.v2", "rows": rows})
    ctx.artifact_text(f"{stem}.md", _matrix_markdown(rows))


def _summary(rows) -> str:
    workspace = [row for row in rows if row["kind"] == "workspace"]
    validators = [row for row in rows if row["kind"] == "validator"]
    survivors = [row["name"] for row in rows if _survived(row)]
    by_design = [row["name"] for row in rows if _accepted(row) and not _survived(row)]
    pinned = [row for row in rows if row["predicted_outcome"] == "killed"]
    text = (f"{len(rows)} rows ({len(workspace)} workspace mutants, {len(validators)} pure-validator rows): "
            f"{sum(row['killed'] for row in rows)} refused; {len(survivors)} surviving "
            f"{'forgery' if len(survivors) == 1 else 'forgeries'} "
            f"({', '.join(survivors) or 'none'})")
    if by_design:
        text += f"; {len(by_design)} documented validator acceptance ({', '.join(by_design)})"
    return (text + f"; kill/survive predictions matched {sum(row['outcome_matches_prediction'] for row in rows)}"
            f"/{len(rows)}; refusal messages matched their regression pins "
            f"{sum(row['message_matches_pin'] for row in pinned)}/{len(pinned)}")


# ------------------------------------------------------------------ mutation catalogue
def _receipt_field(field, value, reseal):
    def apply(view):
        receipt = view.receipt()
        receipt[field] = value(view) if callable(value) else deepcopy(value)
        if reseal:
            reseal_receipt(receipt)
    return apply


def _receipt_verification(field, value, reseal):
    def apply(view):
        receipt = view.receipt()
        receipt["verification"][field] = value(view) if callable(value) else deepcopy(value)
        if reseal:
            reseal_receipt(receipt)
    return apply


def _bundle_verification(role, field, value):
    def apply(view):
        verification = view.native(role)["verification"]
        verification[field] = value(view) if callable(value) else deepcopy(value)
        verification["verification_id"] = verification_id(verification)
    return apply


def _rebind_receipt(role):
    """Move the receipt's source and verification subject to another retained bundle, consistently."""
    def apply(view):
        receipt, target = view.receipt(), view.native(role)
        receipt["source_bundle_digest"] = target["bundle_digest"]
        receipt["verification"] = build_verification(target, view.native("B1")["steps"][0])
        reseal_receipt(receipt)
    return apply


def _rebind_forged(view):
    receipt = view.receipt()
    receipt["source_bundle_digest"] = receipt["verification"]["subject_ref"] = FORGED_DIGEST
    reseal_receipt(receipt)


def _rebind_self(view):
    receipt = view.receipt()
    receipt["source_bundle_digest"] = receipt["verification"]["subject_ref"] = view.native("B1")["bundle_digest"]
    reseal_receipt(receipt)


def _reidentify(view):
    native = view.native("B1")
    native["created_at"], native["session_id"] = FORGED_TIME.replace("+00:00", "Z"), FORGED_SESSION
    reforge(view.workspace)


def _oscillator(roles, edit, reseal=True):
    def apply(view):
        records = [getattr(view, role) for role in roles]
        for record in records:
            edit(record, view)
        if reseal:
            reseal_oscillator(*[record for record in records if "record_digest" in record])
    return apply


def _energy_runtime(roles, field, value):
    def apply(view):
        for role in roles:
            view.native(role)["runtimes"]["energy"][field] = value
        reforge(view.workspace)
    return apply


def _energy_runtime_naive(view):
    view.native("B1")["runtimes"]["energy"]["code_sha256"] = FORGED_CODE


def _result_authority(view):
    view.native("B1")["steps"][0]["result"]["authority"]["state_admission"] = "performed"
    reforge(view.workspace)


def _reuse_execution(which):
    def apply(view):
        original = view.native("B0")
        occurrence = original["steps"][0] if which == "step" else original["verification"]["reproduction"]
        view.native("B1")["steps"][0]["execution_id"] = occurrence["execution_id"]
        reforge(view.workspace)
    return apply


def _energy_namespace(view):
    view.native("B1")["steps"][0]["execution_id"] = view.E1["execution_id"]
    reforge(view.workspace)


def _energy_data(reforged):
    """Change one analysed number; optionally make every copy and derived digest agree (source bytes unchanged)."""
    def apply(view):
        from ..telemetry import _bundle_digest
        native = view.native("B1")
        copies = [native["steps"][0], native["verification"]["reproduction"]]
        for step in copies if reforged else copies[:1]:
            step["result"]["data"]["measurement"]["batch_count"] += 1
        if reforged:
            for step in copies:
                restep(step)
            native["bundle_digest"] = _bundle_digest(native)
            native["verification"] = build_verification(native, native["verification"]["reproduction"])
            receipt = native["replay_receipts"][0]
            receipt["replayed_bundle_digest"] = native["bundle_digest"]
            receipt["verification"] = build_verification(view.native("B0"), native["steps"][0])
            reseal_receipt(receipt)
            view.records["B1"]["bundle_id"] = native["bundle_digest"]
    return apply


def _energy_source(view):
    """Forge the retained baseline log itself: one counter sample, resealed, with every derived record rebuilt."""
    reforge_source(view.workspace, view.records["B0"]["source_id"], measurement_edit)


def _delete_receipt(view):
    del view.native("B1")["replay_receipts"]


def _transplant(reseal):
    def apply(view):
        receipt = view.native("B1").pop("replay_receipts")[0]
        target = view.native("B0b")
        target["replay_receipts"] = [receipt]
        if reseal:
            receipt["replayed_bundle_digest"] = target["bundle_digest"]
            reseal_receipt(receipt)
    return apply


def _revision_gap(view):
    view.workspace["selection"]["revision"] = 1000
    view.E1["selection_revision"] = view.R1["selection_revision"] = 999
    reseal_oscillator(view.E1, view.R1)


def _midpoint(record, view):
    record["data"]["mean"] = 0.5 * (record["data"]["minimum"] + record["data"]["maximum"])


def _impossible_moments(view):
    """In-bounds statistics that no sample set can have: R1 gets |mean| > rms, R2 gets rms > max(|min|, |max|)."""
    first, second = view.R1["data"], view.R2["data"]
    first["mean"] = first["maximum"]
    first["rms"] = 0.5 * abs(first["maximum"])
    second["rms"] = 10.0 * max(abs(second["minimum"]), abs(second["maximum"]))
    reseal_oscillator(view.R1, view.R2)


def _moments(data) -> dict:
    largest = max(abs(data["minimum"]), abs(data["maximum"]))
    return {"abs_mean_le_rms": abs(data["mean"]) <= data["rms"], "rms_le_max_abs": data["rms"] <= largest}


# Post-reopen witnesses: what a reader of the reopened session sees.
def _w_receipt(session, fixture):
    native = session.workbench.get_bundle(fixture["natives"]["B1"]["bundle_digest"])
    receipt = native["replay_receipts"][0]
    return {"replay_bundle": native["bundle_digest"], "receipt_source": receipt["source_bundle_digest"],
            "verification_subject": receipt["verification"]["subject_ref"]}


def _w_reidentified(session, fixture):
    for summary in session.workbench.list_bundles():
        native = session.workbench.get_bundle(summary["bundle_id"])
        if native["session_id"] == FORGED_SESSION:
            source = session.workbench.get_bundle(native["replay_receipts"][0]["source_bundle_digest"])
            # Both timestamps are ISO 8601 UTC strings of one layout, so string order is time order.
            return {"created_at": native["created_at"], "session_id": native["session_id"],
                    "receipt_source": source["bundle_digest"],
                    "replay_predates_source": native["created_at"] < source["created_at"]}
    return {"reidentified_bundle_found": False}


def _w_result(field):
    def witness(session, fixture):
        result = request(session, "result.get", {"result_id": fixture["oscillator"]["R1"]["result_id"]})
        return {"result.get": {key: result.get(key) for key in (field, "verification_status")}}
    return witness


def _w_runtime(session, fixture):
    execution = session.executions[fixture["oscillator"]["E1"]["execution_id"]]
    result = request(session, "result.get", {"result_id": fixture["oscillator"]["R1"]["result_id"]})
    return {"execution_runtime": execution["runtime"], "result_runtime": result["runtime"]}


def _w_energy_runtime(session, fixture):
    first = session.workbench.list_bundles()[0]["bundle_id"]
    runtime = session.workbench.get_bundle(first)["runtimes"]["energy"]
    replay = attempt(session, "bundle.replay", {"bundle_id": first})
    return {"retained_runtime_forged": {"code_sha256": runtime["code_sha256"] == FORGED_CODE,
                                        "python_version": runtime["python_version"] == "3.99.0"},
            "replay_after_reopen": {"outcome": replay["outcome"], "message": replay.get("message")}}


def _w_revision(session, fixture):
    execution = session.executions[fixture["oscillator"]["E1"]["execution_id"]]
    return {"selection_revision": session.selection["revision"],
            "execution_selection_revision": execution["selection_revision"]}


def _w_mean(role):
    def witness(session, fixture):
        original = fixture["oscillator"][role]
        retained = session.results[original["result_id"]]["data"]
        return {"retained_mean_is_midpoint": retained["mean"] == 0.5 * (retained["minimum"] + retained["maximum"]),
                "retained_mean_differs_from_computed": retained["mean"] != original["data"]["mean"]}
    return witness


def _w_moments(session, fixture):
    retained = {role: session.results[fixture["oscillator"][role]["result_id"]]["data"] for role in ("R1", "R2")}
    return {"retained": {f"result:{role}": {key: data[key] for key in ("mean", "minimum", "maximum", "rms")}
                         for role, data in retained.items()},
            "retained_inequalities": {f"result:{role}": _moments(data) for role, data in retained.items()},
            "computed_inequalities": _moments(fixture["oscillator"]["R1"]["data"])}


def _w_source(session, fixture):
    natives = fixture["natives"]
    roles = {natives[role]["session_id"]: role for role in ("B0", "B0b", "B1")}
    energy, times = {}, []
    for summary in session.workbench.list_bundles():
        native = session.workbench.get_bundle(summary["bundle_id"])
        role = roles.get(native["session_id"])
        if role is not None:
            energy[role] = native["steps"][0]["result"]["data"]["measurement"]["gross_energy_j"]
            times.append(native["created_at"] == natives[role]["created_at"])
    return {"gross_energy_j_after_reopen": dict(sorted(energy.items())),
            "gross_energy_j_computed_from_original_bytes": natives["B0"]["steps"][0]["result"]["data"]["measurement"][
                "gross_energy_j"],
            "original_session_ids_and_created_at_kept": len(times) == 3 and all(times)}


def _w_created(session, fixture):
    execution = session.executions[fixture["oscillator"]["E1"]["execution_id"]]
    return {"execution_created_at": execution["created_at"]}


def _w_deleted(session, fixture):
    replayed = fixture["natives"]["B1"]["bundle_digest"]
    return {"replay_bundle_listed": any(b["bundle_id"] == replayed for b in session.workbench.list_bundles()),
            "bundles_with_receipts": sum(bool(session.workbench.get_bundle(b["bundle_id"]).get("replay_receipts"))
                                         for b in session.workbench.list_bundles())}


BINDING = "Retained energy analysis binding differs"
RECEIPT = "Invalid retained energy replay receipt"
BUNDLE = "Energy analysis bundle identity, schema or size differs"
V1 = "Protocol v1 saved results must remain not_verified with verification_id null"
SEAL = "Operation record integrity mismatch"


def _m(name, task_id, target, recompute, description, pinned, apply, witness=None):
    return Mutant(name, task_id, target, recompute, description, pinned, apply, witness)


MUTANTS = (
    # T080: aliasing between operation, execution and result identities; selection history.
    _m("alias.result-execution", "T080", "oscillator result", "local",
       "second result's execution_id set to the first execution; resealed",
       "Saved result identity mismatch or duplication",
       _oscillator(["R2"], lambda r, v: r.update(execution_id=v.E1["execution_id"]))),
    _m("alias.execution-result", "T080", "oscillator execution", "local",
       "second execution's result_id set to the first result; resealed",
       "Execution/result execution_id binding mismatch",
       _oscillator(["E2"], lambda r, v: r.update(result_id=v.R1["result_id"]))),
    _m("alias.result-prefix", "T080", "oscillator execution and result", "local",
       "result identity given the execution- prefix in result and execution; resealed",
       "Invalid saved result identity",
       _oscillator(["R1", "E1"], lambda r, v: r.update(result_id="execution-" + UUID_HEX))),
    _m("alias.operation", "T080", "oscillator execution and result", "local",
       "operation_id changed to spectrum.periodogram.v1 in both records; resealed (killed by payload-shape "
       "validation, not by an identity binding)",
       "Invalid saved spectrum data fields or sample count",
       _oscillator(["E1", "R1"], lambda r, v: r.update(operation_id="spectrum.periodogram.v1"))),
    _m("revision.gap", "T080", "oscillator selection history", "local",
       "selection revision 1000 with execution and result claiming revision 999; resealed",
       "accepted", _revision_gap, _w_revision),
    # T081: numerical content of retained results and of the retained source.
    _m("energy-data.naive", "T081", "energy replay result data", "none",
       "measurement.batch_count incremented in the replay step; nothing recomputed", BUNDLE, _energy_data(False)),
    _m("energy-data.reforged", "T081", "energy replay result data", "full",
       "batch_count incremented in the step and its reproductions; every derived digest recomputed; source bytes "
       "unchanged", BINDING, _energy_data(True)),
    _m("energy-source.resealed", "T081", "retained energy source log", "full",
       "first measurement counter sample lowered by 50 mJ in the retained baseline log; log_digest resealed; "
       "source, evidence, experiment, request, analysis, step, bundle, verification and receipt digests recomputed; "
       "occurrence ids and times kept", "accepted", _energy_source, _w_source),
    _m("oscillator-stats.naive", "T081", "oscillator result data", "none",
       "statistics mean moved to the interval midpoint; not resealed", SEAL,
       _oscillator(["R1"], _midpoint, reseal=False)),
    _m("oscillator-stats.out-of-bounds", "T081", "oscillator result data", "local",
       "statistics mean moved above the maximum; resealed", "Saved statistics mean is outside its bounds",
       _oscillator(["R1"], lambda r, v: r["data"].update(mean=r["data"]["maximum"] + 1.0))),
    _m("oscillator-stats.resealed", "T081", "oscillator result data", "local",
       "statistics mean moved to the midpoint of [minimum, maximum]; resealed", "accepted",
       _oscillator(["R1"], _midpoint), _w_mean("R1")),
    _m("oscillator-stats.impossible-moments", "T081", "oscillator result data", "local",
       "R1 mean set to its maximum and rms to half of it (|mean| > rms); R2 rms set to ten times max(|min|, |max|) "
       "(rms > max|x|); both resealed", "accepted", _impossible_moments, _w_moments),
    _m("oscillator-stats.legacy", "T081", "legacy oscillator result data", "none",
       "legacy (unsealed) statistics mean moved to the midpoint", "accepted",
       _oscillator(["RL"], _midpoint, reseal=False), _w_mean("RL")),
    # T082: occurrence freshness and collisions.
    _m("fresh.energy-replay-reuse", "T082", "energy replay bundle", "full",
       "replay step reuses the original step's execution_id; every digest recomputed",
       "Declared workload bundles must have distinct execution and reproduction occurrences", _reuse_execution("step")),
    _m("fresh.energy-reproduction-reuse", "T082", "energy replay bundle", "full",
       "replay step reuses the original verification reproduction's execution_id; every digest recomputed",
       "Declared workload bundles must have distinct execution and reproduction occurrences",
       _reuse_execution("reproduction")),
    _m("fresh.cross-namespace", "T082", "energy replay bundle", "full",
       "replay step execution_id set to an oscillator execution identity; every digest recomputed",
       "Identity collision between recording operations and retained workflows", _energy_namespace),
    _m("fresh.oscillator-duplicate", "T082", "oscillator execution and result", "local",
       "second execution and result reuse the first execution_id; resealed",
       "Saved result identity mismatch or duplication",
       _oscillator(["E2", "R2"], lambda r, v: r.update(execution_id=v.E1["execution_id"]))),
    _m("fresh.created-at-one", "T082", "oscillator execution", "local", "execution created_at backdated; resealed",
       "Execution/result created_at binding mismatch",
       _oscillator(["E1"], lambda r, v: r.update(created_at=FORGED_TIME))),
    _m("fresh.created-at-shift", "T082", "oscillator execution and result", "local",
       "created_at backdated identically in execution and result; resealed", "accepted",
       _oscillator(["E1", "R1"], lambda r, v: r.update(created_at=FORGED_TIME)), _w_created),
    # T083: receipt placement.
    _m("receipt.numerical-match-false", "T083", "energy replay receipt", "local",
       "numerical_match false; replay_id recomputed", RECEIPT, _receipt_field("numerical_match", False, True)),
    _m("receipt.transplanted", "T083", "energy bundle", "none",
       "receipt moved from the replay bundle onto a sibling execution", RECEIPT, _transplant(False)),
    _m("receipt.transplanted-resealed", "T083", "energy bundle", "local",
       "receipt moved onto a sibling execution; replayed digest and replay_id recomputed", BINDING, _transplant(True)),
    _m("receipt.deleted", "T083", "energy replay bundle", "none", "replay_receipts removed from the replay bundle",
       "accepted", _delete_receipt, _w_deleted),
    # T084: receipt source digest.
    _m("receipt-source.naive", "T084", "energy replay receipt", "none",
       "source_bundle_digest set to a fixed forged digest", RECEIPT,
       _receipt_field("source_bundle_digest", FORGED_DIGEST, False)),
    _m("receipt-source.replay-id", "T084", "energy replay receipt", "local",
       "forged source digest; replay_id recomputed", BINDING,
       _receipt_field("source_bundle_digest", FORGED_DIGEST, True)),
    _m("receipt-source.subject-rebound", "T084", "energy replay receipt", "local",
       "forged source digest and verification subject; verification_id and replay_id recomputed",
       "Replay source must already belong to this workbench", _rebind_forged),
    _m("receipt-source.self", "T084", "energy replay receipt", "local",
       "source digest and subject set to the replay bundle itself; ids recomputed", RECEIPT, _rebind_self),
    _m("receipt-source.other-source", "T084", "energy replay receipt", "local",
       "receipt re-pointed at a retained bundle of a different source log; verification rebuilt",
       "Replay source must already belong to this workbench", _rebind_receipt("Bother")),
    _m("receipt-source.sibling-execution", "T084", "energy replay receipt", "local",
       "receipt source and verification subject moved together to a sibling execution of the same source bytes; "
       "verification rebuilt", "accepted", _rebind_receipt("B0b"), _w_receipt),
    # T085: receipt replayed digest.
    _m("receipt-replayed.naive", "T085", "energy replay receipt", "none",
       "replayed_bundle_digest set to a forged digest", RECEIPT,
       _receipt_field("replayed_bundle_digest", FORGED_DIGEST, False)),
    _m("receipt-replayed.replay-id", "T085", "energy replay receipt", "local",
       "forged replayed digest; replay_id recomputed", RECEIPT,
       _receipt_field("replayed_bundle_digest", FORGED_DIGEST, True)),
    _m("receipt-replayed.source", "T085", "energy replay receipt", "local",
       "replayed digest set to the source bundle digest; replay_id recomputed", RECEIPT,
       _receipt_field("replayed_bundle_digest", lambda v: v.native("B0")["bundle_digest"], True)),
    _m("receipt-replayed.sibling", "T085", "energy replay receipt", "local",
       "replayed digest set to a sibling execution digest; replay_id recomputed", RECEIPT,
       _receipt_field("replayed_bundle_digest", lambda v: v.native("B0b")["bundle_digest"], True)),
    _m("receipt-replayed.reidentified-bundle", "T085", "energy replay bundle", "full",
       "replay bundle dated 2001, before its source, with a new session id; bundle, verification and receipt "
       "digests recomputed", "accepted", _reidentify, _w_reidentified),
    # T086: verification subject. Moving the subject together with the receipt
    # source is T084's receipt-source.sibling-execution; it is not run twice.
    _m("receipt-subject.naive", "T086", "energy replay receipt verification", "none",
       "verification subject set to the replay bundle", RECEIPT,
       _receipt_verification("subject_ref", lambda v: v.native("B1")["bundle_digest"], False)),
    _m("receipt-subject.resealed", "T086", "energy replay receipt verification", "local",
       "subject set to the replay bundle; verification_id and replay_id recomputed", BINDING,
       _receipt_verification("subject_ref", lambda v: v.native("B1")["bundle_digest"], True)),
    _m("bundle-subject.resealed", "T086", "energy bundle verification", "local",
       "original bundle's verification subject set to a sibling; verification_id recomputed", BINDING,
       _bundle_verification("B0", "subject_ref", lambda v: v.native("B0b")["bundle_digest"])),
    _m("oscillator-verification.resealed", "T086", "oscillator result", "local",
       "verification_id and verification_status forged; resealed", V1,
       _oscillator(["R1"], lambda r, v: r.update(verification_id="verification-" + UUID_HEX,
                                                 verification_status="verified"))),
    # T087: verification method.
    _m("receipt-method.naive", "T087", "energy replay receipt verification", "none",
       "method set to independent_reimplementation", RECEIPT,
       _receipt_verification("method", "independent_reimplementation", False)),
    _m("receipt-method.resealed", "T087", "energy replay receipt verification", "local",
       "forged method; verification_id and replay_id recomputed", BINDING,
       _receipt_verification("method", "independent_reimplementation", True)),
    _m("bundle-method.resealed", "T087", "energy bundle verification", "local",
       "replay bundle's own verification method forged; verification_id recomputed", BINDING,
       _bundle_verification("B1", "method", "independent_reimplementation")),
    _m("oscillator-method.injected", "T087", "oscillator result", "local",
       "verification_method field injected into a sealed result; resealed", "accepted",
       _oscillator(["R1"], lambda r, v: r.update(verification_method="independent_reimplementation")),
       _w_result("verification_method")),
    # T088: independence flag.
    _m("receipt-independent.naive", "T088", "energy replay receipt verification", "none", "independent set true",
       RECEIPT, _receipt_verification("independent", True, False)),
    _m("receipt-independent.resealed", "T088", "energy replay receipt verification", "local",
       "independent true; verification_id and replay_id recomputed", BINDING,
       _receipt_verification("independent", True, True)),
    _m("bundle-independent.resealed", "T088", "energy bundle verification", "local",
       "original bundle's verification independent true; verification_id recomputed", BINDING,
       _bundle_verification("B0", "independent", True)),
    _m("oscillator-independent.injected", "T088", "oscillator execution and result", "local",
       "independent: true injected into a sealed execution and result; resealed", "accepted",
       _oscillator(["E1", "R1"], lambda r, v: r.update(independent=True)), _w_result("independent")),
    # T089: admission status.
    _m("receipt-admission.naive", "T089", "energy replay receipt", "none", "admission set to admitted", RECEIPT,
       _receipt_field("admission", "admitted", False)),
    _m("receipt-admission.resealed", "T089", "energy replay receipt", "local",
       "admission admitted; replay_id recomputed", RECEIPT, _receipt_field("admission", "admitted", True)),
    _m("bundle-authority.resealed", "T089", "energy bundle verification", "local",
       "verification authority.state_admission set to performed; verification_id recomputed", BINDING,
       _bundle_verification("B1", "authority", {"state_admission": "performed", "sensor_fusion": "not_performed",
                                                "physical_measurement": "not_performed_by_analysis",
                                                "hardware_provenance": "not_authenticated"})),
    _m("result-authority.reforged", "T089", "energy result authority", "full",
       "result authority.state_admission set to performed; every digest recomputed", BINDING, _result_authority),
    _m("oscillator-status.resealed", "T089", "oscillator result", "local",
       "verification_status set to admitted; resealed", V1,
       _oscillator(["R1"], lambda r, v: r.update(verification_status="admitted"))),
    _m("oscillator-admission.injected", "T089", "oscillator result", "local",
       "state_admission: admitted injected into a sealed result; resealed", "accepted",
       _oscillator(["R1"], lambda r, v: r.update(state_admission="admitted")), _w_result("state_admission")),
    # T090: provider runtime identity.
    _m("oscillator-runtime.naive", "T090", "oscillator execution", "none", "execution runtime forged; not resealed",
       SEAL, _oscillator(["E1"], lambda r, v: r.update(runtime=dict(FORGED_RUNTIME)), reseal=False)),
    _m("oscillator-runtime.execution-only", "T090", "oscillator execution", "local",
       "execution runtime forged; resealed", "Execution/result runtime binding mismatch",
       _oscillator(["E1"], lambda r, v: r.update(runtime=dict(FORGED_RUNTIME)))),
    _m("oscillator-runtime.empty", "T090", "oscillator execution and result", "local",
       "runtime emptied in execution and result; both resealed", "Invalid execution runtime identity",
       _oscillator(["E1", "R1"], lambda r, v: r.update(runtime={}))),
    _m("oscillator-runtime.both", "T090", "oscillator execution and result", "local",
       "runtime forged identically in execution and result; both resealed", "accepted",
       _oscillator(["E1", "R1"], lambda r, v: r.update(runtime=dict(FORGED_RUNTIME))), _w_runtime),
    _m("energy-runtime.naive", "T090", "energy replay bundle runtime", "none",
       "code_sha256 forged in the replay bundle; nothing recomputed", BUNDLE, _energy_runtime_naive),
    _m("energy-runtime.replay-only", "T090", "energy replay bundle runtime", "full",
       "code_sha256 forged in the replay bundle only; every digest recomputed", BINDING,
       _energy_runtime(["B1"], "code_sha256", FORGED_CODE)),
    _m("energy-runtime.malformed", "T090", "energy bundle runtimes", "full",
       "code_sha256 replaced by a non-hex string in every energy bundle; every digest recomputed",
       "Invalid retained analysis implementation identity",
       _energy_runtime(["B0", "B0b", "Bother", "B1"], "code_sha256", "forged")),
    _m("energy-runtime.all-bundles", "T090", "energy bundle runtimes", "full",
       "code_sha256 forged in every energy bundle; every digest recomputed", "accepted",
       _energy_runtime(["B0", "B0b", "Bother", "B1"], "code_sha256", FORGED_CODE), _w_energy_runtime),
    _m("energy-runtime.python-version", "T090", "energy bundle runtimes", "full",
       "python_version forged as 3.99.0 in every energy bundle; every digest recomputed", "accepted",
       _energy_runtime(["B0", "B0b", "Bother", "B1"], "python_version", "3.99.0"), _w_energy_runtime),
)

# The refuted general statement for each predicted survivor.
STATEMENTS = {
    "revision.gap": "Saved execution and result selection revisions are checked against a retained selection history",
    "energy-source.resealed": "Reopen re-analysis protects retained energy results from numerical forgery",
    "oscillator-stats.resealed": "Unkeyed record seals detect every edit to a retained numerical result",
    "oscillator-stats.impossible-moments": "Saved statistics satisfy |mean| <= rms <= max(|min|, |max|)",
    "oscillator-stats.legacy": "Every retained oscillator result is sealed against edits",
    "fresh.created-at-shift": "A retained execution occurrence binds its creation time",
    "receipt.deleted": "A replay bundle cannot be retained without its replay receipt",
    "receipt-source.sibling-execution": "Unkeyed record seals detect every replay-provenance forgery",
    "receipt-replayed.reidentified-bundle": "A retained replay cannot be dated before its source bundle",
    "oscillator-method.injected": "Sealed operation records refuse verification-method claims outside their schema",
    "oscillator-independent.injected": "Sealed operation records refuse independence claims outside their schema",
    "oscillator-admission.injected": "Sealed operation records refuse admission claims outside their schema",
    "oscillator-runtime.both": "Unkeyed record seals detect a forged provider runtime identity",
    "energy-runtime.all-bundles": "Reopening a workspace detects a forged analysis runtime identity",
    "energy-runtime.python-version": "Reopening a workspace detects forged runtime dependency versions",
}

EXCHANGE_BY_DESIGN = ("exchange._identity recomputes content only and returns content_recomputed_not_authenticated; "
                      "ciw.exchange documents that a hash does not establish verification authority, and "
                      "inspect_exchange reports verification_independence not_established")


def _validator_rows(fixture: dict) -> list:
    """Pure-validator rows (no workspace): ESM candidate responses and exchange identities."""
    from ..exchange import VERIFICATION_SCHEMA, _identity
    natives = fixture["natives"]
    selected, sibling = telemetry_shaped_bundle("selected"), telemetry_shaped_bundle("sibling")
    case = esm_case(selected)
    body = {"subject_ref": natives["B0"]["bundle_digest"], "outcome": "passed", "independent": False,
            "method": "same_runtime_fresh_occurrence_reproduction"}
    tampered = exchange_artifact(VERIFICATION_SCHEMA, body, "verification_id")
    tampered["subject_ref"] = natives["B0b"]["bundle_digest"]
    independent = exchange_artifact(VERIFICATION_SCHEMA, dict(body, independent=True,
                                                              method="independent_reimplementation"),
                                    "verification_id")
    candidate_mismatch = "ESM candidate does not bind the selected native bundle"
    esm = "ESM candidate inspection of a synthetic telemetry-shaped bundle (pure validator)"
    exchange = "exchange verification identity (pure validator)"
    return [
        validator_row("esm.bundle-subject", "T086", esm, "candidate bundleDigest set to another synthetic bundle",
                      candidate_mismatch, lambda: check_esm(case, lambda c: c["response"]["candidate"].update(
                          bundleDigest=sibling["bundle_digest"]))),
        validator_row("exchange.verification-subject", "T086", exchange,
                      "subject_ref edited without recomputing verification_id",
                      "verification_id does not match the artifact content",
                      lambda: _identity(tampered, "verification_id")),
        validator_row("esm.inspection-independent", "T088", esm, "independentlyVerified set true", ESM_SCOPE,
                      lambda: check_esm(case, lambda c: c["response"].update(independentlyVerified=True))),
        validator_row("esm.candidate-independent", "T088", esm, "candidate verification independent set true",
                      candidate_mismatch, lambda: check_esm(case, lambda c: c["response"]["candidate"][
                          "verification"].update(independent=True))),
        validator_row("exchange.verification-independent", "T088", exchange,
                      "independent true and method independent_reimplementation; verification_id recomputed",
                      "accepted:content_recomputed_not_authenticated",
                      lambda: _identity(independent, "verification_id"), recompute="local",
                      by_design=EXCHANGE_BY_DESIGN),
        validator_row("esm.canonical-admission", "T089", esm, "canonicalAdmission set to ADMITTED",
                      "ESM may retain candidate evidence only",
                      lambda: check_esm(case, lambda c: c["response"].update(canonicalAdmission="ADMITTED"))),
        validator_row("esm.candidate-admitted", "T089", esm, "candidate state set to ADMITTED", candidate_mismatch,
                      lambda: check_esm(case, lambda c: c["response"]["candidate"].update(state="ADMITTED"))),
    ]


def _task_rows(ctx, task_id: str) -> list:
    return [row for row in _mutations(ctx)["rows"] if row["task"] == task_id]


def _row(rows, name):
    return next(row for row in rows if row["name"] == name)


def _survivor_findings(rows, claims: dict) -> list:
    return [_survivor(_row(rows, name), claim, STATEMENTS[name]) for name, claim in claims.items()]


# ------------------------------------------------------------------ T077 identity matrix
def _entry(identity, scope, derivation_class, derivation, binds, across_replay, across_reopen, validated_at,
           properties, exercised=True, refusals=None) -> dict:
    """One matrix row; properties named ``recomputed_*`` re-derive a CIW digest with the lab's code."""
    row = {"identity": identity, "scope": scope, "derivation_class": derivation_class, "derivation": derivation,
           "binds": binds, "across_replay": across_replay, "across_reopen": across_reopen,
           "validated_at": list(validated_at), "exercised": exercised,
           "properties": {name: bool(value) for name, value in properties.items()}}
    if refusals:
        row["observed_refusals"] = dict(refusals)
    return row


def identity_matrix(fixture: dict, variants: dict) -> list:
    """Predicted identity semantics with each property checked on the offline fixtures."""
    from ..core.identities import evidence_id, validate_evidence_identity
    from ..energy_records import validate_log
    from ..energy_workflow import analysis_identity
    from ..exchange import OBSERVATION_SCHEMA, RESULT_SCHEMA, _identity
    from ..operations.registry import valid_operation_id
    from ..telemetry import _bundle_digest

    run, oscillator, natives = fixture["run"], fixture["oscillator"], fixture["natives"]
    E1, E2, R1, R2, RL = (oscillator[key] for key in ("E1", "E2", "R1", "R2", "RL"))
    same = [natives[role] for role in SAME_SOURCE]
    steps = [native["steps"][0] for native in same] + [native["verification"]["reproduction"] for native in same]
    receipts = [natives[role]["replay_receipts"][0] for role in ("B1", "B2", "B3")]
    verifications = [native["verification"] for native in same] + [receipt["verification"] for receipt in receipts]
    S, SC, SR = (fixture["sources"][key] for key in ("S", "SC", "Srelabelled"))
    baseline = fixture["raw"]["baseline"]
    parsed = json.loads(baseline.decode("utf-8"))
    B0, B1 = natives["B0"], natives["B1"]
    reopened_sources = {source["source_id"] for source in fixture["workspace_reopened"]["workbench"]["sources"]}
    records = {record["name"]: record for record in variants["records"] if record["kind"] != "byte_variant"}
    byte_records = [record for record in variants["records"] if record["kind"] == "byte_variant"]
    metadata, logged = records["metadata-renamed"], records["baseline"]
    moved = _moved_step(fixture)

    edited = deepcopy(run)
    channel = next(iter(edited["channels"]))
    index = next(i for i, value in enumerate(edited["channels"][channel]["values"]) if value is not None)
    edited["channels"][channel]["values"][index] += 1.0
    rendered = dict(deepcopy(run), render={"lab": "changed render state"})
    descriptor = {key: S[key] for key in ("schema", "kind", "label", "source_schema", "evidence_id", "byte_count")}
    exchange_result = exchange_artifact(RESULT_SCHEMA, {"input_refs": ["lab-input"],
                                                        "components": [{"name": "q", "value": 1.0}]}, "result_id")
    exchange_tampered = dict(exchange_result, components=[{"name": "q", "value": 2.0}])
    case = esm_case(telemetry_shaped_bundle("selected"))

    def renamed(c):
        c["response"]["requestId"] = c["policy"]["review_context"]["requestId"] = "lab-request-renamed"

    def noncanonical(c):
        layout = json.dumps(c["bundle"], indent=2).encode("utf-8")
        c["response"]["bundleBytesDigest"] = "sha256:" + hashlib.sha256(layout).hexdigest()

    def seal_ok(record):
        return record["record_digest"] == _sha(_without(record, "record_digest"), ascii_only=True)

    def kernel(log, value="e" * 64):
        log["runtime"]["workload"]["kernel_sha256"] = value
        log["runtime"]["python"]["executable_sha256"] = value

    producers = {"sensor.device_uuid": parsed["sensor"]["device_uuid"],
                 "runtime.workload.kernel_sha256": parsed["runtime"]["workload"]["kernel_sha256"],
                 "runtime.python.executable_sha256": parsed["runtime"]["python"]["executable_sha256"],
                 "runtime.implementation.code_sha256": parsed["runtime"]["implementation"]["code_sha256"]}
    refusals = {
        "recording_sample_edit": _refusal_message(lambda: validate_evidence_identity(edited)),
        "producer_edit_unsealed": _refusal_message(lambda: validate_log(edited_log(baseline, kernel, reseal=False))),
        "producer_edit_resealed": _refusal_message(lambda: validate_log(edited_log(baseline, kernel, reseal=True))),
        "exchange_tamper": _refusal_message(lambda: _identity(exchange_tampered, "result_id")),
        "esm_base": _refusal_message(lambda: check_esm(case)),
        "esm_rename": _refusal_message(lambda: check_esm(case, renamed)),
        "esm_request_mismatch": _refusal_message(lambda: check_esm(case, lambda c: c["response"].update(
            requestId="other"))),
        "esm_noncanonical_bytes": _refusal_message(lambda: check_esm(case, noncanonical)),
        "esm_forged_digest": _refusal_message(lambda: check_esm(case, lambda c: c["response"].update(
            bundleBytesDigest=FORGED_DIGEST))),
    }

    def seen(*names):
        return {name: refusals[name] for name in names}

    rows = [
        _entry("recording evidence_id", "oscillator recording", "content_hash",
               "sha256 over canonical JSON of instrument, metadata, time_s and channels",
               "all scientific content including adapter and calibration metadata; excludes run_id and render",
               "not_applicable (source evidence is not replayed)", "stable",
               ["src/ciw/core/identities.py:evidence_id", "src/ciw/core/identities.py:validate_evidence_identity",
                "src/ciw/session.py:_validate_evidence"],
               {"recomputed_from_scientific_fields": _sha({k: run[k] for k in SCIENTIFIC_FIELDS}, True) == run["evidence_id"],
                "render_excluded": evidence_id(rendered) == run["evidence_id"],
                "sample_edit_refused": refusals["recording_sample_edit"]
                == "Evidence integrity mismatch: scientific content does not match evidence_id",
                "retained_across_reopen": fixture["reopened_run_evidence_id"] == run["evidence_id"]
                == fixture["workspace"]["run"]["evidence_id"]}, refusals=seen("recording_sample_edit")),
        _entry("recording file name", "oscillator recording", "content_hash",
               "'recording-' + sha256 over canonical JSON of the whole run, written by Session",
               "entire parsed run including run_id and render; the file is a CIW re-serialization",
               "not_applicable", "stable", ["src/ciw/session.py:_recording_file",
                                            "src/ciw/session.py:_validate_saved_result"],
               {"recomputed_from_run": fixture["recording"]["file"]
                == "recording-" + hashlib.sha256(_canon(run, True)).hexdigest() + ".json",
                "file_is_reserialization_not_input_bytes": fixture["recording"]["is_reserialization"]}),
        _entry("workbench source evidence_id", "workbench source", "byte_hash", "'sha256:' + sha256(exact source bytes)",
               "exact retained bytes only (not label or kind); bundle artifact_ref and sha256 are copies of it",
               "stable (bundle artifact_ref equals it)", "stable",
               ["src/ciw/workbench.py:_source", "src/ciw/workbench.py:Workbench.restore",
                "src/ciw/energy_workflow.py:EnergyAccuracyWorkflow._validate"],
               {"recomputed_sha256_of_bytes": S["evidence_id"] == "sha256:" + hashlib.sha256(baseline).hexdigest(),
                "equals_bundle_artifact_ref": all(n["source"]["evidence"][0]["artifact_ref"] == S["evidence_id"]
                                                  for n in same),
                "label_independent": SR["evidence_id"] == SC["evidence_id"] == S["evidence_id"],
                "retained_across_reopen": S["source_id"] in reopened_sources}),
        _entry("workbench source_id", "workbench source", "content_hash",
               "'source:' + sha256 over canonical descriptor {schema, kind, label, source_schema, evidence_id, byte_count}",
               "source bytes (through evidence_id) and the caller-declared label", "stable", "stable",
               ["src/ciw/workbench.py:_source", "src/ciw/workbench.py:_source_claims",
                "src/ciw/workbench.py:Workbench.restore"],
               {"recomputed_from_descriptor": S["source_id"] == "source:" + _sha(descriptor),
                "label_bound": SR["source_id"] != SC["source_id"],
                "byte_bound_under_one_label": len({r["source_id"] for r in byte_records}) == len(byte_records) > 1,
                "same_bytes_and_label_same_id_across_sessions": SC["source_id"] == S["source_id"]}),
        _entry("bundle experiment_digest", "energy bundle source", "content_hash",
               "sha256 over canonical JSON of the parsed log", "parsed log content (whitespace and key order excluded)",
               "stable", "stable", ["src/ciw/energy_workflow.py:EnergyAccuracyWorkflow._validate"],
               {"recomputed_from_parsed_content": B0["source"]["experiment_digest"] == _sha(parsed),
                "stable_across_replay_and_sessions": len({n["source"]["experiment_digest"] for n in same}) == 1,
                "byte_layout_insensitive": len({r["experiment_digest"] for r in byte_records}) == 1}),
        _entry("energy log run_id (bundle experiment_id)", "energy log", "caller_declared",
               "copied from the retained log's run_id field (format energy-run-<32 hex>)",
               "the log author's measurement-occurrence claim; bound to log_digest by workbench claims",
               "stable", "stable", ["src/ciw/energy_records.py:_validate", "src/ciw/workbench.py:_claims"],
               {"copied_from_log": all(n["source"]["experiment_id"] == parsed["run_id"] for n in same),
                "unchanged_by_resealed_content_edit": metadata["experiment_id"] == parsed["run_id"]
                and metadata["experiment_digest"] != logged["experiment_digest"]}),
        _entry("energy log producer identities", "energy log", "caller_declared",
               "sensor.device_uuid, runtime.workload.kernel_sha256, runtime.python.executable_sha256 and "
               "runtime.implementation.code_sha256 copied verbatim from the log",
               "nothing outside the log: bound only by the log author's unkeyed log_digest; format-checked (GPU UUID "
               "pattern, 64 hex)", "stable (copied)", "stable",
               ["src/ciw/energy_records.py:_sensor", "src/ciw/energy_records.py:_runtime"],
               {"device_uuid_copied_into_analysis": B0["steps"][0]["result"]["data"]["measurement_scope"]["device_uuid"]
                == producers["sensor.device_uuid"],
                "unsealed_edit_refused_by_log_digest": refusals["producer_edit_unsealed"] == LOG_DIGEST,
                "resealed_edit_accepted_format_only": refusals["producer_edit_resealed"] == "accepted",
                "fixture_hashes_are_placeholders": all(len(set(producers[key])) == 1 for key in producers
                                                       if key.endswith("sha256"))},
               refusals=seen("producer_edit_unsealed", "producer_edit_resealed")),
        _entry("oscillator operation_id", "oscillator operation", "caller_declared_name",
               "versioned registry name chosen by the requester", "which registered operation ran", "not_applicable",
               "stable", ["src/ciw/operations/registry.py:valid_operation_id",
                          "src/ciw/operations/runner.py:validate_execution"],
               {"same_for_repeat": E1["operation_id"] == E2["operation_id"] == R1["operation_id"] == "statistics.v1",
                "versioned_name": valid_operation_id(E1["operation_id"])}),
        _entry("energy operation_id", "energy bundle step", "constant_name", "fixed name ciw.energy-accuracy.v1",
               "which workflow ran", "stable", "stable",
               ["src/ciw/energy_workflow.py:EnergyAccuracyWorkflow._validate_step", "src/ciw/workbench.py:_claims"],
               {"constant_across_occurrences": {s["operation_id"] for s in steps} == {OPERATION},
                "versioned_name": valid_operation_id(OPERATION)}),
        _entry("oscillator execution_id", "oscillator execution", "fresh_event_uuid", "'execution-' + uuid4().hex",
               "one execution occurrence (nothing about its content)", "fresh", "stable",
               ["src/ciw/operations/runner.py:execute", "src/ciw/operations/runner.py:validate_execution",
                "src/ciw/session.py:_identity"],
               {"fresh_per_execution": E1["execution_id"] != E2["execution_id"],
                "uuid4_form": _uuid4(E1["execution_id"], "execution-") and _uuid4(E2["execution_id"], "execution-"),
                "result_links_execution": R1["execution_id"] == E1["execution_id"] and E1["result_id"] == R1["result_id"]}),
        _entry("energy step execution_id", "energy bundle step", "fresh_event_uuid", "'execution-' + uuid4().hex",
               "one analysis occurrence", "fresh", "stable",
               ["src/ciw/energy_workflow.py:EnergyAccuracyWorkflow._validate_step", "src/ciw/workbench.py:_validate_links"],
               {"fresh_per_occurrence": len({s["execution_id"] for s in steps}) == len(steps),
                "uuid4_form": all(_uuid4(s["execution_id"], "execution-") for s in steps),
                "result_execution_ref_matches": all(s["result"]["execution_ref"] == s["execution_id"] for s in steps)}),
        _entry("oscillator result_id", "oscillator result", "fresh_event_uuid", "'result-' + uuid4().hex",
               "one result occurrence (content bound only by the unkeyed record seal)", "fresh", "stable",
               ["src/ciw/session.py:_validate_saved_result", "src/ciw/session.py:_identity"],
               {"fresh_per_execution": R1["result_id"] != R2["result_id"],
                "uuid4_form": _uuid4(R1["result_id"], "result-") and _uuid4(RL["result_id"], "result-")}),
        _entry("energy result_id", "energy bundle step", "content_hash_of_fresh_occurrence",
               "sha256 over the result record, which includes execution_ref", "result content and its fresh occurrence",
               "fresh", "stable", ["src/ciw/energy_workflow.py:EnergyAccuracyWorkflow._validate_step"],
               {"recomputed_from_result": all(s["result_id"] == _sha(_without(s["result"], "result_id")) for s in steps),
                "fresh_per_occurrence": len({s["result_id"] for s in steps}) == len(steps),
                "ciw_refuses_stale_result_id_after_occurrence_change": moved["stale_result_id"] == BINDING,
                "ciw_accepts_recomputed_result_id": moved["recomputed_result_id"] == "accepted"
                and moved["result_id_changed"]},
               refusals={"moved_step_stale_result_id": moved["stale_result_id"],
                         "moved_step_recomputed_result_id": moved["recomputed_result_id"]}),
        _entry("energy numerical_result_id", "energy bundle step", "content_hash",
               "sha256 over {operation_id, data}",
               "operation name, analysed numbers and the log identity (data.log_digest = canonical digest of the whole "
               "retained log including unanalysed metadata, plus data.origin and measurement_scope.device_uuid)",
               "stable for canonically identical logs", "stable",
               ["src/ciw/energy_workflow.py:EnergyAccuracyWorkflow._validate_step", "src/ciw/workbench.py:_validate_links"],
               {"recomputed_from_operation_and_data": all(
                   s["numerical_result_id"] == _sha({"operation_id": s["operation_id"], "data": s["result"]["data"]})
                   for s in steps),
                "single_value_across_replay_reopen_and_sessions": len({s["numerical_result_id"] for s in steps}) == 1,
                "occurrence_change_leaves_it_unchanged": moved["numerical_result_id_unchanged"],
                "metadata_only_resealed_edit_changes_it": metadata["numerical_result_id"] != logged["numerical_result_id"],
                "metadata_only_edit_changes_only_log_digest_in_data":
                    metadata["data_keys_differing_from_baseline"] == ["log_digest"]}),
        _entry("oscillator numerical-result identity", "oscillator result", "absent",
               "none: results carry data but no content-level numerical identity",
               "nothing; equal numbers from two executions are not linked", "absent", "absent",
               ["(no validator)"],
               {"absent": all("numerical_result_id" not in record for record in (R1, R2, RL)),
                "equal_data_distinct_ids": R1["data"] == R2["data"] and R1["result_id"] != R2["result_id"]}),
        _entry("energy verification_id", "energy verification", "content_hash",
               "sha256(schema + NUL + canonical verification without its id)",
               "subject_ref, outcome, independent, method, runtime_digest, reproduction step, authority", "fresh",
               "stable", ["src/ciw/energy_workflow.py:EnergyAccuracyWorkflow._check_verification",
                          "src/ciw/energy_workflow.py:EnergyAccuracyWorkflow._validate", "src/ciw/exchange.py:_identity"],
               {"recomputed": all(v["verification_id"] == "sha256:" + hashlib.sha256(
                   VERIFY_SCHEMA.encode() + b"\0" + _canon(_without(v, "verification_id"))).hexdigest()
                   for v in verifications),
                "fresh_per_reproduction": len({v["verification_id"] for v in verifications}) == len(verifications),
                "exchange_status_not_authenticated": all(_identity(v, "verification_id")
                                                         == "content_recomputed_not_authenticated" for v in verifications),
                "receipt_subject_is_source": all(r["verification"]["subject_ref"] == r["source_bundle_digest"]
                                                 for r in receipts)}),
        _entry("oscillator verification_id", "oscillator result", "absent", "always null; status not_verified",
               "nothing (protocol v1 results are never verified)", "absent", "stable",
               ["src/ciw/session.py:_validate_saved_result"],
               {"null_and_not_verified": all(r["verification_id"] is None and r["verification_status"] == "not_verified"
                                             for r in (R1, R2, RL))}),
        _entry("bundle_digest (workbench bundle_id)", "energy bundle", "content_hash_of_fresh_occurrence",
               "sha256 over the bundle without bundle_digest, verification and replay_receipts",
               "session_id, created_at, source bytes, configuration, runtimes and steps; NOT verification or receipts",
               "fresh", "stable", ["src/ciw/telemetry.py:_bundle_digest",
                                   "src/ciw/energy_workflow.py:EnergyAccuracyWorkflow._validate",
                                   "src/ciw/workbench.py:_validate_record"],
               {"recomputed_excluding_verification_and_receipts": all(
                   n["bundle_digest"] == _sha(_without(n, "bundle_digest", "verification", "replay_receipts"))
                   for n in same),
                "fresh_per_execution_and_replay": len({n["bundle_digest"] for n in same}) == len(same),
                "receipts_excluded": _bundle_digest(_without(B1, "replay_receipts")) == B1["bundle_digest"],
                "verification_excluded": _bundle_digest(dict(B1, verification={"schema": "lab.replaced"}))
                == B1["bundle_digest"],
                "catalog_id_equals_digest": all(r["bundle_id"] == r["native"]["bundle_digest"]
                                                for r in fixture["workspace"]["workbench"]["bundles"])}),
        _entry("bundle session_id", "energy bundle", "fresh_event_uuid", "'session-' + uuid4().hex",
               "one workflow session occurrence", "fresh", "stable",
               ["src/ciw/energy_workflow.py:EnergyAccuracyWorkflow._validate", "src/ciw/workbench.py:_claims"],
               {"fresh_per_bundle": len({n["session_id"] for n in same}) == len(same),
                "uuid4_form": all(_uuid4(n["session_id"], "session-") for n in same)}),
        _entry("protocol Session.session_id", "protocol session", "fresh_event_uuid", "'session-' + uuid4().hex",
               "one live protocol session; never saved", "not_applicable", "fresh (not persisted)",
               ["src/ciw/session.py:Session.__init__"],
               {"fresh_per_session": len(set(fixture["session_ids"])) == 3,
                "not_persisted": "session_id" not in fixture["workspace"],
                "uuid4_form": all(_uuid4(value, "session-") for value in fixture["session_ids"])}),
        _entry("replay_id", "energy replay receipt", "content_hash", "sha256 over the receipt without replay_id",
               "source and replayed bundle digests, numerical_match, verification, admission", "fresh", "stable",
               ["src/ciw/workbench.py:_validate_receipts", "src/ciw/energy_workflow.py:EnergyAccuracyWorkflow._validate"],
               {"recomputed": all(r["replay_id"] == _sha(_without(r, "replay_id")) for r in receipts),
                "fresh_per_replay": len({r["replay_id"] for r in receipts}) == len(receipts)}),
        _entry("oscillator record_digest (seal)", "oscillator execution and result", "unkeyed_seal",
               "sha256 over the record without record_digest", "the whole record, but anyone can recompute it",
               "not_applicable", "stable", ["src/ciw/operations/runner.py:seal", "src/ciw/operations/runner.py:check_seal"],
               {"recomputed_without_secret": all(seal_ok(record) for record in (E1, E2, R1, R2)),
                "legacy_result_unsealed": "record_digest" not in RL and "schema" not in RL}),
        _entry("oscillator runtime identity", "oscillator execution and result", "provider_declared",
               "the operation provider's runtime_identity() dict", "a provider name and version string",
               "not_applicable", "stable (never re-checked)", ["src/ciw/operations/runner.py:validate_execution"],
               {"declared_by_provider": E1["runtime"] == R1["runtime"] == fixture["runtime_declared"],
                "no_code_digest": not any("sha256" in str(value) for value in E1["runtime"].values())}),
        _entry("energy runtime identity", "energy bundle", "code_hash_plus_declared_versions",
               "sha256 of three CIW source files plus self-reported python and numpy versions",
               "analysis code bytes and dependency version strings", "stable (compared on replay)",
               "stable (format-checked only)", ["src/ciw/energy_workflow.py:analysis_identity",
                                                "src/ciw/energy_workflow.py:_check_runtime",
                                                "src/ciw/energy_workflow.py:EnergyAccuracyWorkflow._adapters"],
               {"equals_current_analysis_identity": B0["runtimes"]["energy"] == analysis_identity(),
                "stable_across_occurrences": len({_canon(n["runtimes"]) for n in same}) == 1}),
        _entry("exchange result_id / verification_id", "external exchange artifact", "content_hash",
               "sha256(schema + NUL + canonical artifact without its id)", "artifact content; status "
               "content_recomputed_not_authenticated", "not_applicable", "not_applicable",
               ["src/ciw/exchange.py:_identity"],
               {"status_not_authenticated": _identity(exchange_result, "result_id") == "content_recomputed_not_authenticated",
                "tamper_refused": refusals["exchange_tamper"] == "result_id does not match the artifact content"},
               refusals=seen("exchange_tamper")),
        _entry("exchange batch_id", "external exchange artifact", "caller_declared",
               "any string supplied by the producer", "nothing; reported as caller_declared_reference",
               "not_applicable", "not_applicable", ["src/ciw/exchange.py:_identity"],
               {"caller_declared": _identity({"schema": OBSERVATION_SCHEMA, "batch_id": "any producer string"},
                                             "batch_id") == "caller_declared_reference"}),
        _entry("ESM requestId / inspectedAt", "external ESM inspection", "caller_declared",
               "copied from the operator's review context and request parameters",
               "equality between request and response only", "not_applicable", "not_applicable",
               ["src/ciw/candidate_evidence.py:validate_response"],
               {"base_case_accepted": refusals["esm_base"] == "accepted",
                "consistent_rename_accepted": refusals["esm_rename"] == "accepted",
                "mismatch_refused": refusals["esm_request_mismatch"] == ESM_SCOPE},
               refusals=seen("esm_base", "esm_rename", "esm_request_mismatch")),
        _entry("ESM bundleBytesDigest", "external ESM inspection", "byte_hash",
               "'sha256:' + sha256 of the bytes CIW passes (its canonical serialization of the selected bundle)",
               "the exact canonical bytes of the selected bundle, not its content in another layout",
               "not_applicable", "not_applicable", ["src/ciw/candidate_evidence.py:validate_response",
                                                    "src/ciw/workbench.py:Workbench._validate_candidate"],
               {"noncanonical_layout_of_same_bundle_refused": refusals["esm_noncanonical_bytes"] == ESM_SCOPE,
                "forged_digest_refused": refusals["esm_forged_digest"] == ESM_SCOPE},
               refusals=seen("esm_noncanonical_bytes", "esm_forged_digest")),
        _entry("ESM candidate_id / candidate execution_id", "workbench candidate receipt",
               "content_hash / fresh_event_uuid", "'candidate:' + sha256(record); 'candidate-execution:' + uuid4",
               "the retained ESM response bytes, policy and adapter identity", "not_applicable", "stable",
               ["src/ciw/workbench.py:Workbench._validate_candidate"], {}, exercised=False),
    ]
    return rows


def _matrix_markdown_table(rows) -> str:
    return _table(("Identity", "Scope", "Derivation", "Binds", "Across replay", "Across reopen", "Validated at",
                   "Empirical"),
                  [(r["identity"], r["scope"], f"{r['derivation_class']}: {r['derivation']}", r["binds"],
                    r["across_replay"], r["across_reopen"], "; ".join(r["validated_at"]),
                    ("not exercised offline" if not r["exercised"] else
                     f"{sum(r['properties'].values())}/{len(r['properties'])} held"))
                   for r in rows])


def _property_checks(identity: str, properties: dict) -> list:
    """Re-derivations (``recomputed_*``) are same-origin cross-implementation checks; the rest observe CIW."""
    rederived = {name: held for name, held in properties.items() if name.startswith("recomputed")}
    observed = {name: held for name, held in properties.items() if name not in rederived}
    checks = []
    if rederived:
        checks.append(_rederived(f"{identity}: re-derived digest properties that did not hold",
                                 len(rederived) - sum(rederived.values())))
    if observed:
        checks.append(_exact(f"{identity}: observed properties that did not hold",
                             len(observed) - sum(observed.values())))
    return checks


CLASSES = ("content_hash, byte_hash, content_hash_of_fresh_occurrence and unkeyed_seal (SHA-256 over canonical JSON "
           "or raw bytes); fresh_event_uuid (uuid4 draws); caller_declared, caller_declared_name, constant_name and "
           "provider_declared (copied, not derived); code_hash_plus_declared_versions (a code digest beside "
           "self-reported versions); absent")

T077_PLAN = _fields(
    "Each CIW identity on the offline paths is a content or byte hash, a fresh event UUID, a caller- or "
    "provider-declared value, or absent, and its binding, replay and reopen behaviour follow from how it is derived.",
    "Derivation classes used in the matrix: " + CLASSES + ". Content identities are sha256 over canonical JSON "
    "(sort_keys, compact separators); byte identities are sha256 over raw bytes; event identities are uuid4 draws "
    "(collision probability ~2^-122 per pair).",
    "Every exercised identity shows its predicted properties: content identities recompute exactly and are stable "
    "when their content is; event identities are distinct per occurrence; declared identities are copied, not "
    "derived.",
    "Build an oscillator session and energy-accuracy bundles offline, replay, save, reopen, replay again, replay a "
    "replay, and repeat in a separate session (same process and code); retain byte and resealed content variants; "
    "check every predicted property per identity, with CIW's own validators judging refusals and moved occurrences.",
    "not run", "not run", [], [], "T078",
    inputs=SESSION_INPUTS + [VARIANT_INPUT])


@task("T077", changed_files=CHANGED, regression_tests=(_node("test_identity_matrix"),), plan=T077_PLAN)
def identity_matrix_task(ctx):
    if not fixture_available():
        return _blocked(T077_PLAN)
    from ..telemetry import _bundle_digest
    fixture, variants = _fixture(ctx), _variants(ctx)
    rows = identity_matrix(fixture, variants)
    ctx.artifact_json("identity-matrix.json", {"schema": "ciw.lab-identity-matrix.v2", "rows": rows})
    ctx.artifact_text("identity-matrix.md", _matrix_markdown_table(rows))
    exercised = [row for row in rows if row["exercised"]]
    properties = sum(len(row["properties"]) for row in exercised)
    held = sum(sum(row["properties"].values()) for row in exercised)
    rederived = sum(name.startswith("recomputed") for row in exercised for name in row["properties"])
    classes = dict(sorted(Counter(row["derivation_class"] for row in rows).items()))
    checks = [check for row in exercised for check in _property_checks(row["identity"], row["properties"])]
    R1, R2 = fixture["oscillator"]["R1"], fixture["oscillator"]["R2"]
    B1 = fixture["natives"]["B1"]
    receipt_free = _bundle_digest(_without(B1, "replay_receipts")) == B1["bundle_digest"]
    verification_free = _bundle_digest(dict(B1, verification={"schema": "lab.replaced"})) == B1["bundle_digest"]
    parsed = json.loads(fixture["raw"]["baseline"].decode("utf-8"))
    findings = [
        _verified("Every exercised identity shows its predicted derivation, freshness and binding properties on the "
                  "offline paths",
                  {"rows": len(rows), "exercised_rows": len(exercised), "properties": properties,
                   "properties_held": held, "rederived_properties": rederived, "derivation_classes": classes},
                  checks),
        _verified("Oscillator operation results carry no replay-stable numerical-result identity",
                  {"numerical_result_id_present": "numerical_result_id" in R1, "data_equal": R1["data"] == R2["data"],
                   "result_ids_equal": R1["result_id"] == R2["result_id"]},
                  [_invariant("equal data, distinct result ids and no numerical identity (1 = observed)",
                              1.0 if ("numerical_result_id" not in R1 and R1["data"] == R2["data"]
                                      and R1["result_id"] != R2["result_id"]) else 0.0, 1.0, "ge")],
                  counterexample={"statement": "Every retained CIW operation result carries a replay-stable "
                                               "numerical-result identity",
                                  "witness": {"operation": "statistics.v1", "records": ["result:R1", "result:R2"],
                                              "fields": sorted(R1)}}),
        _verified("The energy replay bundle identity excludes its replay receipt and verification",
                  {"digest_unchanged_without_receipt": receipt_free,
                   "digest_unchanged_with_replaced_verification": verification_free,
                   "excluded_fields": ["bundle_digest", "replay_receipts", "verification"]},
                  [_exact("bundle digest recomputed by CIW without replay_receipts differs from the retained one",
                          0 if receipt_free else 1),
                   _exact("bundle digest recomputed by CIW with the verification replaced differs from the retained one",
                          0 if verification_free else 1)],
                  counterexample={"statement": "A retained bundle identity binds every provenance record stored in the "
                                               "bundle",
                                  "witness": {"bundle": "bundle:B1", "removed": "replay_receipts",
                                              "replaced": "verification",
                                              "function": "src/ciw/telemetry.py:_bundle_digest"}}),
        finding("The retained log's device and kernel identities identify the producing GPU and code", "physical",
                {"sensor.device_uuid": parsed["sensor"]["device_uuid"],
                 "runtime.workload.kernel_sha256": parsed["runtime"]["workload"]["kernel_sha256"],
                 "runtime.python.executable_sha256": parsed["runtime"]["python"]["executable_sha256"],
                 "runtime.implementation.code_sha256": parsed["runtime"]["implementation"]["code_sha256"]},
                {"notes": "These values are copied from a synthetic fixture log (the hashes are repeated-character "
                          "placeholders) and CIW checks only their format; a resealed log with other well-formed "
                          "values is accepted. Nothing identifies real hardware or code."}, tolerance=EXACT),
        finding("Retained workspace records are authenticated: a holder without a secret cannot produce a forged "
                "record that reopens", "provenance",
                {"unkeyed_identity_classes": sorted({"content_hash", "content_hash_of_fresh_occurrence",
                                                     "unkeyed_seal", "byte_hash"} & set(classes))},
                {"notes": "Every content identity and seal in the matrix is an unkeyed SHA-256 over public canonical "
                          "JSON or raw bytes; no keyed signature or MAC exists on these paths (see T081-T090 for "
                          "accepted forgeries)."}, tolerance=EXACT, expected_not_established=True),
    ]
    fields = dict(T077_PLAN)
    fields.update(
        experiment=T077_PLAN["experiment"] + " " + COMPARISON + " Matrix retained as identity-matrix.json/.md "
                   "(observed refusal messages included per row).",
        numerical_result=f"{len(rows)} identities ({len(exercised)} exercised offline); {held}/{properties} predicted "
                         f"properties held ({rederived} are same-origin digest re-derivations, the rest observe CIW "
                         f"outputs, refusals or validators); classes {classes}",
        uncertainty="Exact equality tests on digests, UUID draws and exact refusal messages; a uuid4 collision among "
                    "the ~40 fresh identities has probability below 1e-33. Properties are sampled on one fixture, not "
                    "proved for all inputs. Checks that could not fail by construction (identifier prefix tests, a "
                    "lab-set ESM digest, label-driven source_id counts) were removed.",
        failure_modes_checked=["digest recomputation mismatch", "identity reused across occurrences",
                               "identity not stable across replay or reopen", "label-independent source identity",
                               "source identity insensitive to bytes under one label",
                               "result identity after an occurrence change (judged by CIW's step validator)",
                               "numerical identity after a metadata-only resealed edit",
                               "producer identities edited with and without resealing",
                               "exchange identity tamper", "ESM request/response mismatch",
                               "ESM digest over a non-canonical layout of the same bundle"],
        unresolved_assumptions=[
            "ESM candidate_id and candidate execution identities were not exercised: they need an operator-bound ESM "
            "adapter and a telemetry or calibrated bundle.",
            "The ESM rows use a synthetic telemetry-shaped record (schema, digest, three step occurrences), not a "
            "telemetry session validated by the telemetry workflow, which needs provider checkouts.",
            "Provider-backed workflows (telemetry, declared workloads, proved heat) were not exercised offline; their "
            "identity rows are inferred only where they share the energy-accuracy code path.",
            "Content identities establish consistency, not authorship."],
        recommended_next_task="T078 (exact source-byte retention); CIW change: add a numerical_result_id to "
                              "oscillator operation results and bind replay receipts into a catalog-level seal")
    return {"state": _state(findings), "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T078 exact source bytes
T078_PLAN = _fields(
    "The workbench retains the exact bytes supplied for each source: live, inside every native bundle, and after "
    "save and reopen.",
    "Let b be the supplied bytes and r the retained bytes (base64-decoded). Prediction: r == b, "
    "sha256(r) == evidence_id and len(r) == byte_count for every source; non-canonical base64 is refused, "
    "never normalized.",
    "bytes in == bytes retained for every accepted source; refused submissions leave the retained source count "
    "unchanged.",
    "Retain the four fixture logs, eight byte variants and two resealed content variants of the baseline log, "
    "execute each, compare source.get bytes live and after offline reopen and the bundle evidence bytes with the "
    "input; submit three refused re-encodings and three non-canonical base64 transports and count retained sources "
    "before and after.", "not run", "not run", [], [], "T079",
    inputs=[VARIANT_INPUT, OSCILLATOR_INPUT])


@task("T078", changed_files=CHANGED, regression_tests=(_node("test_exact_source_bytes"),), plan=T078_PLAN)
def exact_source_bytes(ctx):
    if not fixture_available():
        return _blocked(T078_PLAN, [_physical_logs()])
    variants, fixture = _variants(ctx), _fixture(ctx)
    records = variants["records"]
    ctx.artifact_json("source-retention.json", {"records": records, "refusals": variants["refusals"],
                                                "sources_before_refusals": variants["sources_before_refusals"],
                                                "sources_after_refusals": variants["sources_after_refusals"]})
    mismatches = {f"{where}_byte_mismatches": sum(not record[f"{where}_bytes_equal"] for record in records)
                  for where in ("live", "reopened", "bundle")}
    id_mismatch = sum(record["evidence_id"] != record["input_sha256"] for record in records)
    count_mismatch = sum(record["declared_byte_count"] != record["byte_count"] for record in records)
    artifact_mismatch = sum(record["artifact_ref"] != record["evidence_id"] for record in records)
    same_bytes = [record for record in records if (record["kind"], record["name"]) in
                  (("fixture_log", "baseline"), ("byte_variant", "original"))]
    transports = {name: row for name, row in variants["refusals"].items() if name.startswith("base64/")}
    expected_transport = {"base64/missing-padding": "Source bytes must use canonical base64",
                          "base64/line-wrapped": "Source bytes must use canonical base64",
                          "base64/noncanonical-trailing-bits": "Source bytes must use bounded canonical base64"}
    added = variants["sources_after_refusals"] - variants["sources_before_refusals"]
    recording = fixture["recording"]
    kinds = Counter(record["kind"] for record in records)
    findings = [
        _verified("Workbench sources retain the exact supplied bytes live, inside native bundles and after offline "
                  "reopen",
                  {"sources": len(records), "fixture_logs": kinds["fixture_log"], "byte_variants": kinds["byte_variant"],
                   "resealed_variants": kinds["resealed_variant"], **mismatches,
                   "evidence_id_mismatches": id_mismatch, "byte_count_mismatches": count_mismatch,
                   "bundle_artifact_ref_mismatches": artifact_mismatch},
                  [_exact(f"sources with {key.replace('_', ' ')}", value) for key, value in mismatches.items()]
                  + [_rederived("evidence_id differs from the lab's sha256 of the input bytes", id_mismatch),
                     _exact("declared byte_count differs from the input length", count_mismatch),
                     _exact("bundle artifact_ref differs from the source evidence_id", artifact_mismatch)]),
        _verified("Non-canonical base64 transports are refused rather than normalized, and refused submissions retain "
                  "no source",
                  {"messages": {name: row.get("message") for name, row in transports.items()},
                   "refused_submissions": len(variants["refusals"]), "sources_added_by_refused_submissions": added},
                  [_refusal(f"{name}: source.add refusal", expected_transport[name],
                            transports[name].get("message") or transports[name]["outcome"])
                   for name in expected_transport]
                  + [_exact("sources retained by the six refused submissions", added)]),
        _verified("Evidence identity depends only on bytes: one byte string under two labels shares evidence_id while "
                  "source_id differs",
                  {"sources": len(same_bytes), "distinct_evidence_ids": len({r["evidence_id"] for r in same_bytes}),
                   "distinct_source_ids": len({r["source_id"] for r in same_bytes})},
                  [_count("distinct evidence ids for identical bytes", len({r["evidence_id"] for r in same_bytes}), 1),
                   _count("distinct source ids for two labels", len({r["source_id"] for r in same_bytes}), 2)]),
        _verified("The oscillator recording path retains canonical content, not caller bytes: its file is a CIW "
                  "re-serialization named by a content digest",
                  {"recording_file_is_reserialization": recording["is_reserialization"]},
                  [_exact("recording file (newlines LF-normalized) differs from the json.dumps(run, indent=2) "
                          "re-serialization", 0 if recording["is_reserialization"] else 1)]),
        _physical_logs(variants["fixture_origins"]),
    ]
    fields = dict(T078_PLAN)
    fields.update(
        experiment=T078_PLAN["experiment"] + " Retained as source-retention.json.",
        numerical_result=f"{len(records)} sources; byte mismatches live/reopened/bundle = "
                         f"{mismatches['live_byte_mismatches']}/{mismatches['reopened_byte_mismatches']}/"
                         f"{mismatches['bundle_byte_mismatches']}; evidence_id mismatches {id_mismatch}; "
                         f"{sum(transports[name].get('message') == expected for name, expected in expected_transport.items())}"
                         f"/{len(expected_transport)} non-canonical base64 transports refused with the pinned message; "
                         f"{added} sources added by {len(variants['refusals'])} refused submissions",
        uncertainty="Exact byte equality; no numerical tolerance is involved. The sample is fourteen byte strings of "
                    "one schema, not every possible source kind.",
        failure_modes_checked=["byte normalization on retention", "re-encoding on workspace save",
                               "bundle evidence differing from the source", "base64 padding, wrapping and "
                               "non-canonical trailing bits", "refused submission leaving a retained source",
                               "label leaking into evidence identity"],
        unresolved_assumptions=["Only the energy-accuracy source kind was exercised; other kinds share "
                                "workbench._source but have their own parsers.",
                                "The oscillator recording is supplied as parsed JSON, so its input bytes do not "
                                "exist to retain.",
                                "Exact retention is not authenticity: the retained bytes carry only the log author's "
                                "unkeyed log_digest (see T081 energy-source.resealed)."],
        recommended_next_task="T079 (whitespace-preserving evidence retention)")
    return {"state": _state(findings), "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T079 whitespace-preserving retention
T079_PLAN = _fields(
    "Byte variants of one log that differ only in whitespace, key order or float spelling are retained as distinct "
    "evidence with distinct byte-level identities, while every content-level identity is equal; CIW's canonical "
    "comparison is type-sensitive, so rewriting 1.0 as 1 is a content change.",
    "Byte-level identities are functions of the bytes b (sha256(b)); content-level identities are functions of "
    "canon(parse(b)) with CIW's canonical JSON, which prints 1 and 1.0 differently. For variants with "
    "canon(parse(b_i)) equal and b_i distinct, byte-level identities are pairwise distinct and content-level "
    "identities coincide.",
    "Eight variants under one label: 8 distinct evidence_id and source_id; one experiment_digest, log_digest and "
    "numerical_result_id. A consistent int-for-float rewrite is refused by the log seal unless resealed, and a "
    "resealed one is distinct evidence, never aliased.",
    "Retain and analyse eight variants of baseline.json under one shared label; compare identities; submit a partial "
    "and a consistent (unsealed) int-for-float rewrite and a byte-order-mark variant; retain a consistent resealed "
    "int-for-float rewrite.", "not run", "not run", [], [], "T080",
    inputs=[VARIANT_INPUT])


@task("T079", changed_files=CHANGED, regression_tests=(_node("test_whitespace_variants"),), plan=T079_PLAN)
def whitespace_retention(ctx):
    if not fixture_available():
        return _blocked(T079_PLAN)
    variants = _variants(ctx)
    records = [record for record in variants["records"] if record["kind"] == "byte_variant"]
    named = {record["name"]: record for record in variants["records"] if record["kind"] != "byte_variant"}
    count = len(records)

    def distinct(key):
        return len({record[key] for record in records})
    byte_level = {key: distinct(key) for key in ("input_sha256", "evidence_id", "source_id")}
    content_level = {key: distinct(key) for key in ("experiment_digest", "log_digest", "numerical_result_id")}
    aliases = sum(record["artifact_ref"] != record["evidence_id"] for record in records)
    labels = len({record["label"] for record in records})
    levels = [
        {"identity": "workbench evidence_id (bundle artifact_ref and sha256 are copies of it)", "level": "byte",
         "distinct_values": byte_level["evidence_id"]},
        {"identity": "workbench source_id", "level": "byte + label (one shared label here)",
         "distinct_values": byte_level["source_id"]},
        {"identity": "bundle experiment_digest", "level": "canonical content", "distinct_values": content_level["experiment_digest"]},
        {"identity": "log_digest (sealed inside the log)", "level": "canonical content", "distinct_values": content_level["log_digest"]},
        {"identity": "numerical_result_id", "level": "canonical content of analysed data, which includes log_digest",
         "distinct_values": content_level["numerical_result_id"]},
        {"identity": "recording evidence_id (oscillator)", "level": "canonical content of scientific fields", "distinct_values": None},
        {"identity": "bundle_digest / result_id / verification_id / replay_id", "level": "canonical content including fresh occurrences", "distinct_values": None},
    ]
    ctx.artifact_json("identity-levels.json", {"variants": [r["name"] for r in records], "levels": levels,
                                               "records": records})
    ctx.artifact_text("identity-levels.md", _table(("Identity", "Level", f"Distinct values over {count} variants"),
                                                  [(r["identity"], r["level"], "n/a" if r["distinct_values"] is None
                                                    else r["distinct_values"]) for r in levels]))
    refusals = variants["refusals"]
    partial, unsealed, bom = (refusals[name] for name in ("int-for-float-partial", "int-for-float-consistent-unsealed",
                                                          "byte-order-mark"))
    resealed, baseline = named["int-for-float-consistent"], named["baseline"]
    exact = sum(r["live_bytes_equal"] and r["reopened_bytes_equal"] and r["bundle_bytes_equal"] for r in records)
    canonical = sum(r["canonical_equal_to_baseline"] for r in records)
    moved = {key: resealed[key] != baseline[key] for key in ("experiment_digest", "log_digest", "numerical_result_id")}

    def message(row):
        return row.get("message") or row["outcome"]
    findings = [
        _verified("Whitespace, key-order and float-spelling variants under one label retain distinct exact bytes and "
                  "distinct evidence and source identities while their parsed content is canonically equal",
                  {"variants": count, "labels": labels, "exactly_retained": exact, "canonically_equal": canonical,
                   "artifact_ref_not_equal_to_evidence_id": aliases,
                   **{f"distinct_{key}": value for key, value in byte_level.items()}},
                  [_count("labels shared by the variants", labels, 1),
                   _count("variants retained byte-exactly (live, bundle, reopen)", exact, count),
                   _count("variants canonically equal to baseline (CIW canonical JSON)", canonical, count),
                   _exact("variants whose bundle artifact_ref is not their evidence_id", aliases)]
                  + [_count(f"distinct {key}", value, count) for key, value in byte_level.items()]),
        _verified("Content-level identities coincide across the byte variants: one experiment_digest, log_digest and "
                  "numerical_result_id", content_level,
                  [_count(f"distinct {key}", value, 1) for key, value in content_level.items()]),
        _verified("CIW canonical comparison is type-sensitive (1 != 1.0): a consistent int-for-float rewrite is refused "
                  "by the stale log seal and, once resealed, retained as distinct evidence rather than aliased",
                  {"python_equal": unsealed["python_equal"], "ciw_canonical_equal": unsealed["canonical_equal"],
                   "unsealed_observed": message(unsealed), "resealed_retained": True,
                   "resealed_identities_differ_from_baseline": moved},
                  [_refusal("consistent rewrite of both initial_covariance copies, log_digest not resealed: source.add "
                            "refusal", LOG_DIGEST, message(unsealed)),
                   _exact("Python equality holds while CIW canonical equality fails (0 = observed)",
                          0 if (unsealed["python_equal"] and not unsealed["canonical_equal"]) else 1),
                   _exact("content identities of the resealed rewrite equal to baseline (experiment_digest, log_digest, "
                          "numerical_result_id)", sum(not value for value in moved.values()))],
                  counterexample={"statement": "Canonical-content identity treats numerically equal JSON numbers as "
                                               "equal content",
                                  "witness": {"edit": "unit diagonal of runtime.workload.solver_settings and "
                                                      "plan.solver initial_covariance written as 1 instead of 1.0",
                                              "python_equal": unsealed["python_equal"],
                                              "ciw_canonical_equal": unsealed["canonical_equal"],
                                              "unsealed": message(unsealed),
                                              "resealed_identities_differ_from_baseline": moved}}),
        _verified("A partial int-for-float rewrite of one copy of the solver settings is refused by CIW's cross-field "
                  "solver check", {"observed": message(partial)},
                  [_refusal("first '1.0,' (runtime.workload.solver_settings) rewritten as '1,': source.add refusal",
                            "Workload solver differs from plan", message(partial))]),
        _verified("A UTF-8 byte-order-mark variant with equal content is refused",
                  {"canonical_equal_after_bom_strip": bom["canonical_equal"], "code": bom.get("code"),
                   "observed": message(bom)},
                  [_refusal("BOM-prefixed baseline: source.add refusal", BOM_MESSAGE, message(bom))]),
    ]
    fields = dict(T079_PLAN)
    fields.update(
        experiment=T079_PLAN["experiment"] + " Identity levels retained as identity-levels.json/.md.",
        numerical_result=f"{count} variants under {labels} label: byte-level distinct counts {byte_level}; "
                         f"content-level distinct counts {content_level}; consistent int-for-float rewrite refused "
                         f"unsealed ({message(unsealed)}), distinct evidence when resealed; partial rewrite and BOM "
                         "refused",
        uncertainty="Exact equality of digests; the variant set samples whitespace, order and float spelling but "
                    "not every lexical freedom JSON allows (for example escaped string characters).",
        failure_modes_checked=["whitespace normalization before hashing", "key-order sensitivity of content "
                               "identities", "float lexical spelling", "int/float type aliasing (consistent, "
                               "resealed and unsealed)", "cross-field solver consistency", "byte-order mark",
                               "label-driven rather than byte-driven source identity"],
        unresolved_assumptions=[
            "The BOM refusal message names a 'bound runtime' although the defect is in source bytes; the refusal is "
            "correct but its wording is misleading (reported, not changed).",
            "Unicode escape variants (\\u0041 vs A) were not exercised."],
        recommended_next_task="T080 (operation/execution/result identity separation)")
    return {"state": _state(findings), "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T080 separation
T080_PLAN = _fields(
    "Operation, execution and result identities have different kinds (name, fresh occurrence, result occurrence); "
    "forged aliases between execution and result identities are refused on reopen, while the selection revision "
    "a record claims is not checked against any history.",
    "operation_id is a constant per operation; execution_id and result_id are independent uuid4 draws per "
    "occurrence (energy result_id = sha256 over a record that includes execution_ref); numerical_result_id "
    "excludes the occurrence.",
    "No string is used in two identity roles; CIW's own step validator refuses a moved occurrence with a stale "
    "result_id and accepts it with a recomputed one, whose numerical and operation identities are unchanged; "
    "execution/result aliasing forgeries are refused; revision.gap reopens.",
    "Compare identity sets across oscillator and energy occurrences, move one energy step to a new execution_ref and "
    "let EnergyAccuracyWorkflow._validate_step judge it, and reopen four aliasing forgeries and one selection-revision "
    "forgery.", "not run", "not run", [], [], "T081")


@task("T080", changed_files=CHANGED, regression_tests=(_node("test_identity_separation"),), plan=T080_PLAN)
def identity_separation(ctx):
    if not fixture_available():
        return _blocked(T080_PLAN)
    fixture = _fixture(ctx)
    oscillator, natives = fixture["oscillator"], fixture["natives"]
    E1, E2, R1, R2 = (oscillator[key] for key in ("E1", "E2", "R1", "R2"))
    steps = [natives[role]["steps"][0] for role in ALL_BUNDLES] + [natives[role]["verification"]["reproduction"]
                                                                   for role in ALL_BUNDLES]
    roles = {"operation": {E1["operation_id"], E2["operation_id"]} | {s["operation_id"] for s in steps},
             "execution": {E1["execution_id"], E2["execution_id"], oscillator["RL"]["execution_id"]}
             | {s["execution_id"] for s in steps},
             "result": {R1["result_id"], R2["result_id"], oscillator["RL"]["result_id"]} | {s["result_id"] for s in steps},
             "numerical_result": {s["numerical_result_id"] for s in steps},
             "bundle": {natives[role]["bundle_digest"] for role in ALL_BUNDLES}}
    names = sorted(roles)
    overlaps = sum(len(roles[a] & roles[b]) for i, a in enumerate(names) for b in names[i + 1:])
    moved = _moved_step(fixture)
    unmoved = sum(not moved[key] for key in ("result_id_changed", "numerical_result_id_unchanged",
                                             "operation_id_unchanged"))
    rows, harness = _task_rows(ctx, "T080"), _mutations(ctx)["harness"]
    _retain_rows(ctx, rows, "aliasing-mutations")
    counts = {role: len(values) for role, values in roles.items()}
    findings = [
        _verified("Operation, execution, result, numerical-result and bundle identities never share a value, and CIW's "
                  "step validator requires a new result identity, but no new numerical or operation identity, for a "
                  "new occurrence",
                  {"distinct_per_role": counts, "cross_role_overlaps": overlaps, "moved_step": moved},
                  [_exact("values shared between identity roles", overlaps),
                   _count("distinct operation identities", counts["operation"], 2),
                   _count("distinct oscillator+energy execution identities", counts["execution"], 3 + len(steps)),
                   _count("distinct result identities", counts["result"], 3 + len(steps)),
                   _refusal("EnergyAccuracyWorkflow._validate_step on a step moved to a new execution_ref with its "
                            "stale result_id", BINDING, moved["stale_result_id"]),
                   _exact("EnergyAccuracyWorkflow._validate_step refuses the moved step with recomputed digests",
                          0 if moved["recomputed_result_id"] == "accepted" else 1),
                   _exact("identities that did not move as predicted (result_id changes; numerical_result_id and "
                          "operation_id do not)", unmoved)]),
        _kills(rows, "Execution/result aliasing forgeries are refused on reopen with the pinned message; operation "
                     "substitution is refused only by payload-shape validation", harness),
        *_survivor_findings(rows, {"revision.gap": "Surviving mutant revision.gap: a workspace whose selection "
                                                   "revision jumps to 1000, with records claiming revision 999, "
                                                   "reopens"}),
        _authentication(rows),
    ]
    fields = dict(T080_PLAN)
    fields.update(
        experiment=T080_PLAN["experiment"] + " " + COMPARISON + " Mutation rows retained as aliasing-mutations.json/.md.",
        numerical_result=f"distinct identities per role {counts}; cross-role overlaps {overlaps}; moved step: stale "
                         f"result_id -> {moved['stale_result_id']}, recomputed -> {moved['recomputed_result_id']}; "
                         f"mutants: {_summary(rows)}",
        uncertainty="Exact comparisons. The saved selection history is not retained, so revision claims cannot be "
                    "checked by any validator.",
        failure_modes_checked=["identity reuse across roles", "result pointing at another execution", "execution "
                               "pointing at another result", "wrong identity prefix",
                               "operation substitution (alias.operation is killed incidentally by payload-shape "
                               "validation: only statistics.v1 and spectrum.periodogram.v1 are registered and their "
                               "payload shapes differ, so no shape-compatible substitution exists offline)",
                               "stale result identity after an occurrence change", "selection revision gap"],
        unresolved_assumptions=["uuid4 freshness is probabilistic, not enforced by a registry across workspaces.",
                                "Selection history is not persisted, so selection_revision is only bounded, not bound.",
                                "No binding between operation_id and the other identities was isolated from payload "
                                "validation."],
        recommended_next_task="T081 (stable numerical-result identity); CIW change: persist the selection history or "
                              "drop selection_revision from sealed records")
    return {"state": _state(findings), "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T081 stable numerical identity
def _direct_statistics(run: dict, parameters: dict) -> dict:
    """statistics.v1 recomputed from the retained samples with NumPy reductions."""
    import numpy as np
    start, end = parameters["interval_s"]
    values = run["channels"][parameters["channel"]]["values"]
    samples = [value for time, value in zip(run["time_s"], values) if start <= time < end and value is not None]
    x = np.asarray(samples, dtype=np.float64)
    return {"sample_count": len(samples), "mean": float(np.mean(x)), "minimum": float(np.min(x)),
            "maximum": float(np.max(x)), "rms": float(np.sqrt(np.mean(x * x)))}


T081_PLAN = _fields(
    "The energy numerical_result_id is a content identity of the analysed data, and that data includes the retained "
    "log's identity (data.log_digest, origin and device_uuid), so it is equal across every occurrence of a "
    "canonically identical log (original, sibling, replay, replay after reopen, replay of a replay, separate session "
    "in the same process and code) and changes under a resealed metadata-only edit. Re-analysis on reopen refuses "
    "numerical edits that leave the retained source bytes unchanged, but not a forger who rewrites and reseals the "
    "source log itself. Oscillator results lack a numerical identity and their statistics are only bounds-checked.",
    "numerical_result_id = sha256(canon({operation_id, data})) with data = analyze(parse(bytes)) deterministic "
    "in-process; data.log_digest = sha256(canon(log without log_digest)). For statistics, |mean| <= rms <= "
    "max(|min|, |max|) holds for every sample set (Cauchy-Schwarz and the maximum bound).",
    "exactly one numerical_result_id over the baseline occurrences and a different one for the metadata variant; "
    "edits that keep the source bytes are refused where content is recomputed and survive where it is only sealed; "
    "energy-source.resealed and oscillator-stats.impossible-moments reopen.",
    "Collect the numerical identity from the original, a sibling, a replay, a replay after reopen, a replay of a "
    "replay and a separate session (steps and reproductions); compare a resealed metadata-only variant; recompute "
    "statistics.v1 from the samples; reopen data-edit and source-log forgeries.",
    "not run", "not run", [], [], "T082",
    inputs=SESSION_INPUTS + ["baseline.json with the sensor name renamed and log_digest resealed (variant fixture)"])


@task("T081", changed_files=CHANGED, regression_tests=(_node("test_numerical_identity"),), plan=T081_PLAN)
def numerical_identity(ctx):
    if not fixture_available():
        return _blocked(T081_PLAN)
    from .exchange_provenance_common import STATS
    fixture, variants = _fixture(ctx), _variants(ctx)
    natives = fixture["natives"]
    occurrences = [natives[role]["steps"][0] for role in SAME_SOURCE] + [
        natives[role]["verification"]["reproduction"] for role in SAME_SOURCE]
    numerical = {s["numerical_result_id"] for s in occurrences}
    recomputed = sum(s["numerical_result_id"] != _sha({"operation_id": s["operation_id"], "data": s["result"]["data"]})
                     for s in occurrences)
    result_ids = {s["result_id"] for s in occurrences}
    named = {record["name"]: record for record in variants["records"] if record["kind"] != "byte_variant"}
    metadata, baseline = named["metadata-renamed"], named["baseline"]
    computed = fixture["oscillator"]["R1"]["data"]
    direct = _direct_statistics(fixture["run"], STATS["parameters"])
    relative = max(abs(direct[key] - computed[key]) / max(abs(computed[key]), 1e-300)
                   for key in ("mean", "minimum", "maximum", "rms"))
    inequalities = _moments(computed)
    rows, harness = _task_rows(ctx, "T081"), _mutations(ctx)["harness"]
    _retain_rows(ctx, rows, "numerical-mutations")
    ctx.artifact_json("numerical-identity.json", relabel(
        {"occurrences": [{"bundle": role, "step": natives[role]["steps"][0]["execution_id"],
                          "numerical_result_id_equal_to_B0": natives[role]["steps"][0]["numerical_result_id"]
                          == natives["B0"]["steps"][0]["numerical_result_id"]} for role in SAME_SOURCE],
         "metadata_variant": {key: metadata[key] for key in ("experiment_id", "data_keys_differing_from_baseline")},
         "statistics": {"computed": computed, "direct": direct, "max_relative_difference": relative}},
        role_labels(fixture)))
    findings = [
        _verified("The energy numerical_result_id is identical across original, sibling, replay, replay after offline "
                  "reopen, replay of a replay and a separate session (same process and code) of one canonically "
                  "identical log",
                  {"occurrences": len(occurrences), "distinct_numerical_result_ids": len(numerical),
                   "distinct_result_ids": len(result_ids), "recomputation_mismatches": recomputed},
                  [_count("distinct numerical_result_id values", len(numerical), 1),
                   _count("distinct result_id values (fresh occurrences)", len(result_ids), len(occurrences)),
                   _rederived("numerical_result_id differs from the lab's sha256 over {operation_id, data}",
                              recomputed)]),
        _verified("A resealed metadata-only log edit changes the energy numerical_result_id while every analysed "
                  "number stays equal: the identity binds the whole log through data.log_digest",
                  {"edit": "sensor.name renamed; log_digest resealed",
                   "numerical_result_id_differs": metadata["numerical_result_id"] != baseline["numerical_result_id"],
                   "data_keys_differing": metadata["data_keys_differing_from_baseline"],
                   "experiment_id_unchanged": metadata["experiment_id"] == baseline["experiment_id"]},
                  [_exact("metadata variant shares the baseline numerical_result_id",
                          int(metadata["numerical_result_id"] == baseline["numerical_result_id"])),
                   _exact("analysis data keys other than log_digest that differ from baseline",
                          len(set(metadata["data_keys_differing_from_baseline"]) - {"log_digest"})),
                   _exact("log_digest equal to baseline after the metadata edit",
                          int("log_digest" not in metadata["data_keys_differing_from_baseline"]))]),
        _verified("The computed statistics.v1 result satisfies |mean| <= rms <= max(|min|, |max|) and agrees with a "
                  "direct recomputation from the retained samples",
                  {"max_relative_difference": relative, "sample_count": direct["sample_count"],
                   "inequalities": inequalities},
                  [_invariant("max relative difference of mean, minimum, maximum and rms from a NumPy recomputation",
                              relative, 1e-12, "le"),
                   _exact("sample count differs from the recomputation", direct["sample_count"] - computed["sample_count"]),
                   _exact("moment inequalities violated by the computed result",
                          sum(not value for value in inequalities.values()))],
                  uncertainty={"kind": "roundoff", "value": 1e-12,
                               "basis": "bound on the relative disagreement of two float64 reductions over 64 samples"},
                  tolerance={"abs": 1e-12, "rel": 1e-6}),
        _verified("CIW refuses to analyse a resealed variant of a log beside the original in one workbench when both "
                  "keep one run_id", variants["shared_run_id_execution"],
                  [_refusal("operation.execute on the resealed metadata variant in a workbench that already analysed "
                            "baseline", "Identity collision across retained workbench artifacts",
                            variants["shared_run_id_execution"].get("message") or "accepted")]),
        _kills(rows, "Numerical edits that leave the retained source bytes unchanged and break a seal, a digest, the "
                     "statistics bounds or the recomputed analysis are refused on reopen", harness),
        *_survivor_findings(rows, {
            "energy-source.resealed": "Surviving mutant energy-source.resealed: a resealed edit of the retained source "
                                      "log, with every derived record rebuilt, reopens with a different gross energy",
            "oscillator-stats.resealed": "Surviving mutant oscillator-stats.resealed: an in-bounds statistics edit "
                                         "with a recomputed seal reopens",
            "oscillator-stats.impossible-moments": "Surviving mutant oscillator-stats.impossible-moments: resealed "
                                                   "statistics that no sample set can have reopen",
            "oscillator-stats.legacy": "Surviving mutant oscillator-stats.legacy: an edit to an unsealed legacy "
                                       "result reopens"}),
        _authentication(rows),
    ]
    fields = dict(T081_PLAN)
    fields.update(
        experiment=T081_PLAN["experiment"] + " " + COMPARISON + " Retained numerical-identity.json and "
                   "numerical-mutations.json/.md.",
        numerical_result=f"{len(occurrences)} baseline occurrences -> {len(numerical)} numerical_result_id, "
                         f"{len(result_ids)} result_id; metadata variant data differs only in "
                         f"{metadata['data_keys_differing_from_baseline']}; statistics max relative difference "
                         f"{relative:.1e}; mutants: {_summary(rows)}",
        uncertainty="Stability was observed within one process and platform; numerical_result_id hashes floats, so "
                    "a different BLAS or NumPy build could change the last bits of analysed data and therefore the "
                    "identity across platforms (not tested here). Value integrity after reopen is relative to the "
                    "retained source bytes, which are themselves unauthenticated.",
        failure_modes_checked=["occurrence leaking into numerical identity", "replay after reopen", "replay of a "
                               "replay", "separate session", "metadata-only resealed log edit",
                               "sealed and unsealed data edits with the source bytes unchanged",
                               "resealed edit of the retained source log", "resealed variant analysed beside the "
                               "original under one run_id", "out-of-bounds statistics",
                               "statistics violating the moment inequalities"],
        unresolved_assumptions=["Cross-platform stability of float-valued numerical identities is not established.",
                                "Oscillator results have no numerical identity, so their replays cannot be linked.",
                                "The separate session runs in the same process with the same code; it is not an "
                                "independent reproduction."],
        recommended_next_task="T082 (fresh replay execution and result identities); CIW change: content-level "
                              "numerical_result_id for operation results, recomputation of statistics (or at least "
                              "the moment inequalities) on reopen, and keyed authentication of retained source bytes")
    return {"state": _state(findings), "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T082 fresh occurrences
T082_PLAN = _fields(
    "Every execution, result, verification, bundle and replay occurrence receives a fresh identity, and a "
    "retained workspace that reuses an occurrence is refused; creation times are bound only between the two records "
    "of one occurrence.",
    "Occurrence identities are uuid4 draws or digests over records containing them; for n occurrences the "
    "probability of any uuid4 collision is at most n(n-1) / 2^123.",
    "distinct count == occurrence count for every occurrence kind; reuse forgeries refused; a creation time moved in "
    "only one record is refused, moved in both it reopens (fresh.created-at-shift).",
    "Count identities over oscillator executions and every energy bundle (original, sibling, other log, replays, "
    "separate session); reopen four reuse forgeries and two creation-time forgeries.",
    "not run", "not run", [], [], "T083")


@task("T082", changed_files=CHANGED, regression_tests=(_node("test_fresh_occurrences"),), plan=T082_PLAN)
def fresh_occurrences(ctx):
    if not fixture_available():
        return _blocked(T082_PLAN)
    fixture = _fixture(ctx)
    oscillator, natives = fixture["oscillator"], fixture["natives"]
    bundles = [natives[role] for role in ALL_BUNDLES]
    steps = [n["steps"][0] for n in bundles] + [n["verification"]["reproduction"] for n in bundles]
    receipts = [natives[role]["replay_receipts"][0] for role in ("B1", "B2", "B3")]
    kinds = {
        "execution": [s["execution_id"] for s in steps] + [oscillator[k]["execution_id"] for k in ("E1", "E2", "RL")],
        "result": [s["result_id"] for s in steps] + [oscillator[k]["result_id"] for k in ("R1", "R2", "RL")],
        "verification": [n["verification"]["verification_id"] for n in bundles]
        + [r["verification"]["verification_id"] for r in receipts],
        "bundle": [n["bundle_digest"] for n in bundles],
        "bundle_session": [n["session_id"] for n in bundles],
        "replay": [r["replay_id"] for r in receipts],
    }
    table = {kind: {"occurrences": len(values), "distinct": len(set(values))} for kind, values in kinds.items()}
    total = sum(entry["occurrences"] for entry in table.values())
    rows, harness = _task_rows(ctx, "T082"), _mutations(ctx)["harness"]
    _retain_rows(ctx, rows, "freshness-mutations")
    findings = [
        _verified("Every execution, result, verification, bundle, session and replay occurrence has a fresh identity",
                  table, [_count(f"distinct {kind} identities", entry["distinct"], entry["occurrences"])
                          for kind, entry in table.items()], uncertainty=_collision(total)),
        _kills(rows, "Reused or colliding occurrence identities are refused on reopen with the pinned message",
               harness),
        *_survivor_findings(rows, {"fresh.created-at-shift": "Surviving mutant fresh.created-at-shift: a backdated "
                                                             "execution and result pair reopens"}),
        _authentication(rows),
    ]
    fields = dict(T082_PLAN)
    fields.update(
        experiment=T082_PLAN["experiment"] + " Mutation rows retained as freshness-mutations.json/.md.",
        numerical_result=f"occurrence table {table}; mutants: {_summary(rows)}",
        uncertainty=f"uuid4 collisions among these {total} identities have probability below "
                    f"{_collision(total)['value']:.1e}; freshness is checked within one workspace, not globally "
                    "across workspaces.",
        failure_modes_checked=["replay reusing the original execution", "replay reusing the original reproduction",
                               "energy step colliding with an oscillator execution", "oscillator execution duplicate",
                               "creation time moved in one record", "creation time moved in both records"],
        unresolved_assumptions=["No global registry of occurrences exists, so two workspaces can reuse an "
                                "occurrence identity without detection.",
                                "created_at is a declared timestamp, not a trusted time."],
        recommended_next_task="T083 (replay receipt binding); CIW change: keyed or externally timestamped "
                              "execution records if creation time is to carry evidential weight")
    return {"state": _state(findings), "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T083 receipt binding
T083_PLAN = _fields(
    "A replay receipt binds the source bundle digest, the replayed bundle digest, the verification subject, the "
    "fresh reproduction step, the runtime digest and a not_performed admission, but the replay bundle's own "
    "identity does not bind the receipt.",
    "replay_id = sha256(receipt \\ replay_id); receipt.verification.subject_ref = source_bundle_digest; "
    "bundle_digest = sha256(bundle \\ {bundle_digest, verification, replay_receipts}).",
    "All binding equalities hold for every retained receipt; transplanted receipts and false numerical matches are "
    "refused; a receipt can be removed without changing any digest (receipt.deleted reopens).",
    "Check the binding equalities for three receipts (replay, replay after reopen, replay of a replay); reopen "
    "receipt transplant, false numerical match and deletion forgeries.", "not run", "not run", [], [], "T084")


@task("T083", changed_files=CHANGED, regression_tests=(_node("test_replay_receipt_binding"),), plan=T083_PLAN)
def receipt_binding(ctx):
    if not fixture_available():
        return _blocked(T083_PLAN)
    from ..energy_workflow import METHOD
    fixture = _fixture(ctx)
    natives = fixture["natives"]
    by_digest = {native["bundle_digest"]: native for native in natives.values()}
    properties = []
    for role in ("B1", "B2", "B3"):
        native = natives[role]
        receipt = native["replay_receipts"][0]
        source = by_digest.get(receipt["source_bundle_digest"])
        verification = receipt["verification"]
        held = {
            "schema": receipt["schema"] == f"ciw.{KIND}-replay.v1",
            "source_is_retained_bundle": source is not None and source["bundle_digest"] != native["bundle_digest"],
            "replayed_is_containing_bundle": receipt["replayed_bundle_digest"] == native["bundle_digest"],
            "subject_is_source": verification["subject_ref"] == receipt["source_bundle_digest"],
            "reproduction_is_fresh_step": verification["reproduction"] == native["steps"][0]
            and source is not None and source["steps"][0]["execution_id"] != native["steps"][0]["execution_id"],
            "recomputed_runtime_digest_binds_both": source is not None and verification["runtime_digest"]
            == _sha(native["runtimes"]) == _sha(source["runtimes"]),
            "numerical_match_holds": receipt["numerical_match"] is True and source is not None
            and source["steps"][0]["numerical_result_id"] == native["steps"][0]["numerical_result_id"],
            "limited_authority": receipt["admission"] == "not_performed" and verification["independent"] is False
            and verification["method"] == METHOD,
            "recomputed_replay_id": receipt["replay_id"] == _sha(_without(receipt, "replay_id")),
            "recomputed_verification_id": verification["verification_id"] == "sha256:" + hashlib.sha256(
                VERIFY_SCHEMA.encode() + b"\0" + _canon(_without(verification, "verification_id"))).hexdigest(),
        }
        properties.append({"replay": role, "properties": held})
    total = sum(len(entry["properties"]) for entry in properties)
    held_count = sum(sum(entry["properties"].values()) for entry in properties)
    rows, harness = _task_rows(ctx, "T083"), _mutations(ctx)["harness"]
    _retain_rows(ctx, rows, "receipt-mutations")
    ctx.artifact_json("receipt-binding.json", properties)
    findings = [
        _verified("Every retained replay receipt binds its source and replayed bundle digests, verification subject, "
                  "fresh reproduction step, runtime digest and limited authority",
                  {"receipts": len(properties), "binding_properties": total, "held": held_count},
                  [check for entry in properties
                   for check in _property_checks(f"{entry['replay']} receipt binding", entry["properties"])]),
        _kills(rows, "Transplanted receipts and false numerical-match claims are refused on reopen", harness),
        *_survivor_findings(rows, {"receipt.deleted": "Surviving mutant receipt.deleted: a replay bundle reopens "
                                                      "without its replay receipt, indistinguishable from an original "
                                                      "execution"}),
        finding("Replay agreement establishes verification by an independent party", "provenance",
                {"receipt_independent_flags": [natives[role]["replay_receipts"][0]["verification"]["independent"]
                                               for role in ("B1", "B2", "B3")], "method": METHOD},
                {"notes": "Every receipt records independent: false and the same analysis runtime; replay is "
                          "same-code reproduction, which the evidence boundary separates from independent "
                          "verification by another party."}, tolerance=EXACT, expected_not_established=True),
        _authentication(rows),
    ]
    fields = dict(T083_PLAN)
    fields.update(
        experiment=T083_PLAN["experiment"] + " " + COMPARISON + " Retained receipt-binding.json and "
                   "receipt-mutations.json/.md.",
        numerical_result=f"{held_count}/{total} receipt binding properties held over {len(properties)} receipts; "
                         f"mutants: {_summary(rows)}",
        uncertainty="Exact equalities. Receipt deletion is accepted because bundle_digest excludes replay_receipts "
                    "(src/ciw/telemetry.py:_bundle_digest); nothing else records that a replay happened.",
        failure_modes_checked=["receipt moved to another bundle", "receipt moved and resealed", "false "
                               "numerical_match", "receipt deleted"],
        unresolved_assumptions=["Only energy-accuracy receipts were exercised; provider-backed receipts share "
                                "workbench._validate_receipts but have workflow-specific checks."],
        recommended_next_task="T084 (mutate replay receipt source digest); CIW change: bind receipts into the replay "
                              "bundle identity or a catalog-level seal so deletion is detected")
    return {"state": _state(findings), "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T084-T090 mutation tasks
MUTATION_TASKS = {
    "T084": {
        "subject": "Replay receipt source digest", "stem": "receipt-source-mutations",
        "hypothesis": "Predicted kills: a source digest edited without resealing (stale replay_id); resealed to a "
                      "forged digest (the energy workflow rebuilds the receipt verification with subject = source "
                      "and refuses the stale subject); re-pointed together with its subject at a digest that is not "
                      "retained, at the replay itself, or at a bundle of another source log (workbench._validate_links "
                      "requires a retained source bundle of the same kind, source_id and upstream). Predicted "
                      "survivor: receipt-source.sibling-execution, because a sibling execution of the same source "
                      "bytes satisfies every one of those checks and every digest is unkeyed.",
        "invariant": "Every predicted refusal is observed; receipt-source.sibling-execution reopens with its receipt "
                     "and verification subject naming the sibling; the harness controls hold.",
        "survivors": {"receipt-source.sibling-execution": "Surviving mutant receipt-source.sibling-execution: a "
                      "receipt re-pointed, with its verification subject, at a sibling execution of the same bytes "
                      "reopens"},
        "next": "T085 (mutate replay receipt replayed digest)"},
    "T085": {
        "subject": "Replay receipt replayed digest", "stem": "receipt-replayed-mutations",
        "hypothesis": "Predicted kills: the four replayed_bundle_digest edits, resealed or not, because reopen requires "
                      "it to equal the containing bundle's own digest. Predicted survivor: "
                      "receipt-replayed.reidentified-bundle, because created_at and session_id sit inside the unkeyed "
                      "bundle_digest, the forger recomputes that digest, the verification and the receipt together, "
                      "and no check compares a replay's creation time with its source's.",
        "invariant": "Every predicted refusal is observed; the re-identified replay reopens dated before its source; "
                     "the harness controls hold.",
        "survivors": {"receipt-replayed.reidentified-bundle": "Surviving mutant receipt-replayed.reidentified-"
                      "bundle: a replay re-sessioned and dated before its source, with recomputed digests, reopens"},
        "next": "T086 (mutate verification subject)"},
    "T086": {
        "subject": "Verification subject", "stem": "verification-subject-mutations", "validators": "ESM and exchange",
        "hypothesis": "Predicted kills: a receipt verification subject edited with or without recomputed ids (reopen "
                      "rebuilds the expected verification with subject = receipt source); a bundle verification "
                      "subject other than the bundle's own digest; oscillator verification fields (protocol v1 fixes "
                      "them); an ESM candidate naming another bundle; an exchange subject edit without a recomputed "
                      "verification_id. No survivor of its own is predicted: the subject can move only together with "
                      "the receipt source, which is T084's receipt-source.sibling-execution (cross-referenced here, "
                      "not re-run).",
        "invariant": "Every predicted refusal is observed; T084's sibling-execution survivor shows the subject and "
                     "source moved together; the harness controls hold.",
        "survivors": {}, "next": "T087 (mutate verification method)"},
    "T087": {
        "subject": "Verification method", "stem": "verification-method-mutations",
        "hypothesis": "Predicted kills: the three energy verification method edits, because reopen rebuilds the "
                      "expected verification with the fixed method fresh_analysis_of_same_retained_measurement. "
                      "Predicted survivor: oscillator-method.injected, because sealed oscillator records do not "
                      "refuse extra keys and their seal is unkeyed.",
        "invariant": "Every predicted refusal is observed; the injected verification_method is returned by result.get "
                     "after reopen; the harness controls hold.",
        "survivors": {"oscillator-method.injected": "Surviving mutant oscillator-method.injected: a sealed result "
                      "carrying an injected verification_method reopens"},
        "next": "T088 (mutate independence flag)"},
    "T088": {
        "subject": "Independence flag", "stem": "independence-mutations", "validators": "ESM and exchange",
        "hypothesis": "Predicted kills: independent true in an energy receipt or bundle verification (reopen fixes "
                      "independent: false) and in an ESM inspection or candidate (the validator requires false). "
                      "Predicted survivor: oscillator-independent.injected (open key set, unkeyed seal). "
                      "exchange._identity is predicted to accept a recomputed artifact claiming independence, by "
                      "design: it checks content only and reports content_recomputed_not_authenticated.",
        "invariant": "Every predicted refusal is observed; the injected flag is returned after reopen; exchange "
                     "identity status is content_recomputed_not_authenticated; the harness controls hold.",
        "survivors": {"oscillator-independent.injected": "Surviving mutant oscillator-independent.injected: "
                      "sealed records carrying an injected independent: true reopen"},
        "next": "T089 (mutate admission status)"},
    "T089": {
        "subject": "Admission status", "stem": "admission-mutations", "validators": "ESM",
        "hypothesis": "Predicted kills: receipt admission other than not_performed; energy verification or result "
                      "authority other than the fixed not_performed record; an oscillator verification_status other "
                      "than not_verified; an ESM response or candidate claiming admission. Predicted survivor: "
                      "oscillator-admission.injected (open key set, unkeyed seal).",
        "invariant": "Every predicted refusal is observed; the injected state_admission is returned after reopen; "
                     "the harness controls hold.",
        "survivors": {"oscillator-admission.injected": "Surviving mutant oscillator-admission.injected: a "
                      "sealed result carrying an injected state_admission reopens"},
        "next": "T090 (mutate provider runtime identity)"},
    "T090": {
        "subject": "Provider runtime identity", "stem": "runtime-mutations",
        "hypothesis": "Predicted kills: oscillator runtime edits that break the seal, differ between execution and "
                      "result, or empty the identity; energy runtime edits that leave bundle_digest stale, differ "
                      "between a replay and its source (the receipt verification binds one runtime digest), or break "
                      "the code_sha256 format. Predicted survivors: oscillator-runtime.both (a provider-declared "
                      "identity checked only for equality between the two records) and energy-runtime.all-bundles and "
                      "energy-runtime.python-version (reopen format-checks the retained runtime; only a new replay "
                      "compares it with the current analysis identity).",
        "invariant": "Every predicted refusal is observed; the three survivors reopen; a replay of a reopened "
                     "forged-runtime bundle is refused; the harness controls hold.",
        "survivors": {"oscillator-runtime.both": "Surviving mutant oscillator-runtime.both: a provider runtime "
                      "forged identically in execution and result reopens",
                      "energy-runtime.all-bundles": "Surviving mutant energy-runtime.all-bundles: a consistently "
                      "forged analysis code digest reopens",
                      "energy-runtime.python-version": "Surviving mutant energy-runtime.python-version: forged "
                      "dependency versions reopen"},
        "next": "T091 (save/reopen without provider access); CIW change: keyed signatures over workspace records; "
                "label a retained runtime that differs from the current analysis identity as historical and "
                "unverified on reopen"},
}


def _mutation_plan(task_id: str) -> dict:
    spec = MUTATION_TASKS[task_id]
    validators = spec.get("validators")
    return _fields(
        spec["hypothesis"],
        "Each record seal or identity d = sha256(canon(record \\ d)) is unkeyed, so any holder can recompute d "
        "after an edit. Reopen kills a mutant only if some check compares the edited field with an independently "
        "recomputed, fixed or cross-referenced value; otherwise a consistent recomputation survives.",
        spec["invariant"],
        f"Edit the saved workspace ({spec['subject'].lower()}), recompute none, local or all unkeyed digests, write it "
        "and reopen with Session.from_workspace"
        + (f"; pure {validators} validators are also run on synthetic records." if validators else "."),
        "not run", "not run", [], [], spec["next"])


def _cross_reference(matrix: dict) -> dict:
    """T086 cites T084's survivor: moving the subject together with the source is one forgery, counted once."""
    row = _row(matrix["rows"], "receipt-source.sibling-execution")
    witness = row.get("post_reopen", {})
    moved = (_accepted(row) and witness.get("verification_subject") == witness.get("receipt_source") == "bundle:B0b")
    return _verified("The verification subject moves with the receipt source: T084's surviving mutant "
                     "receipt-source.sibling-execution rebinds both to a sibling execution and reopens",
                     {"mutant": row["name"], "task": row["task"], "post_reopen": witness},
                     [_exact("T084 sibling-execution row accepted with subject and source both naming the sibling "
                             "(0 = observed)", 0 if moved else 1)])


def _exchange_by_design(rows) -> dict:
    row = _row(rows, "exchange.verification-independent")
    return _verified("exchange._identity checks content only: a recomputed verification artifact claiming "
                     "independent: true is accepted with status content_recomputed_not_authenticated",
                     {"observed": row["observed"], "documented_behaviour": row.get("accepted_by_design"),
                      "full_inspector": "not run: inspect_exchange needs the pinned State Estimation Evaluation "
                                        "Testbed validator (CIW_SET_REPO)"},
                     [_refusal("exchange._identity on a recomputed artifact claiming independence (observed status)",
                               "accepted:content_recomputed_not_authenticated", row["observed"])])


def _validator_scope(validators: str) -> str:
    text = ("Pure-validator rows run candidate_evidence.validate_response on a synthetic telemetry-shaped record "
            "(schema, digest, three step occurrences) that no telemetry workflow validated, without an ESM process")
    if "exchange" in validators:
        text += ", and exchange._identity without the pinned exchange validator"
    return text + "."


def _mutation_task(ctx, task_id: str) -> dict:
    spec = MUTATION_TASKS[task_id]
    plan = _mutation_plan(task_id)
    if not fixture_available():
        return _blocked(plan, [_admission_authority()] if task_id == "T089" else [])
    matrix = _mutations(ctx)
    rows = [row for row in matrix["rows"] if row["task"] == task_id]
    _retain_rows(ctx, rows, spec["stem"])
    findings = [_kills(rows, f"{spec['subject']} forgeries that leave a digest stale or contradict a recomputed, fixed "
                             "or cross-referenced value are refused with the pinned message", matrix["harness"])]
    findings += _survivor_findings(rows, spec["survivors"])
    if task_id == "T086":
        findings.append(_cross_reference(matrix))
    if task_id == "T088":
        findings.append(_exchange_by_design(rows))
    if task_id == "T089":
        fixture = _fixture(ctx)
        findings.append(_admission_authority([receipt["admission"] for receipt in fixture["receipts"].values()]))
    if task_id == "T090":
        witness = _row(rows, "energy-runtime.all-bundles").get("post_reopen", {})
        replay = witness.get("replay_after_reopen", {})
        findings.append(_verified(
            "A consistently forged energy runtime identity is detected only when a replay recomputes the current "
            "analysis identity", {"replay_after_reopen": replay},
            [_refusal("bundle.replay on the reopened forged workspace", BINDING,
                      replay.get("message") or str(replay.get("outcome")))]))
        ctx.artifact_json("mutation-matrix.json", {"schema": "ciw.lab-mutation-matrix.v2", "harness": matrix["harness"],
                                                   "summary": _summary(matrix["rows"]), "rows": matrix["rows"]})
        ctx.artifact_text("mutation-matrix.md", _matrix_markdown(matrix["rows"]))
    findings.append(_authentication(rows))
    fields = dict(plan)
    fields.update(
        experiment=plan["experiment"] + f" Rows retained as {spec['stem']}.json/.md"
                   + (" and the full matrix as mutation-matrix.json/.md." if task_id == "T090" else "."),
        numerical_result=_summary(rows) + (f". Full matrix: {_summary(matrix['rows'])}" if task_id == "T090" else ""),
        uncertainty="Outcomes are exact (accepted or a refusal message). Only kill/survive is a prediction; refusal "
                    "messages are regression pins recorded from observed runs. Survivors are demonstrated forgeries; "
                    "kills show only that these particular edits are detected, not that every edit of this field is.",
        failure_modes_checked=[f"{row['name']}: {row['description']}" for row in rows],
        unresolved_assumptions=[
            "The forger model recomputes only unkeyed SHA-256 digests with CIW's public canonicalization; no key "
            "exists to steal.",
            *([_validator_scope(spec["validators"])] if spec.get("validators") else []),
            "Whether a reader of a forged workspace would be misled depends on how the record is displayed; not "
            "assessed."])
    return {"state": _state(findings), "fields": fields, "findings": findings}


def _register_mutation_task(task_id: str, test: str):
    @task(task_id, changed_files=CHANGED, regression_tests=(_node(test),), plan=_mutation_plan(task_id))
    def run(ctx):
        return _mutation_task(ctx, task_id)
    run.__name__ = f"mutation_{task_id.lower()}"
    return run


for _task_id, _test in (("T084", "test_receipt_digest_mutations"), ("T085", "test_receipt_digest_mutations"),
                        ("T086", "test_verification_mutations"), ("T087", "test_verification_mutations"),
                        ("T088", "test_verification_mutations"), ("T089", "test_admission_and_runtime_mutations"),
                        ("T090", "test_admission_and_runtime_mutations")):
    _register_mutation_task(_task_id, _test)
