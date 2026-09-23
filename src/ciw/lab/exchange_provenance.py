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
consistency, not authorship or authenticity. An accepted forgery is recorded
as a counterexample, never repaired, and no CIW code is changed here. The
energy logs are synthetic fixtures, so nothing here bears on real GPU energy,
sensor performance, admission authority, or verification by another party.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import uuid

from .evidence import finding
from .exchange_provenance_common import (
    FORGED_CODE, FORGED_DIGEST, FORGED_RUNTIME, FORGED_SESSION, FORGED_TIME, KIND, OPERATION, Mutant, View,
    attempt, build_session_fixture, build_variant_fixture, build_verification, check_esm, esm_case,
    exchange_artifact, fixture_available, reforge, relabel, reopen, request, reseal_oscillator, reseal_receipt,
    restep, role_labels, run_mutant, validator_row, verification_id)
from .registry import task

MODULE = "src/ciw/lab/exchange_provenance.py"
COMMON = "src/ciw/lab/exchange_provenance_common.py"
TESTS = "tests/test_lab_exchange_provenance.py"
DOC = "docs/lab/EXCHANGE_PROVENANCE.md"
CHANGED = (MODULE, COMMON, TESTS, DOC)
EXACT = {"abs": 0.0, "rel": 0.0}
VERIFY_SCHEMA = "ciw.declared-workload-verification.v1"
SCIENTIFIC_FIELDS = ("instrument", "metadata", "time_s", "channels")
# Bundles that analyse the baseline log: original, sibling, replay, replay
# after reopen, replay of a replay, independent session.
SAME_SOURCE = ("B0", "B0b", "B1", "B2", "B3", "B4")
ALL_BUNDLES = SAME_SOURCE[:2] + ("Bother",) + SAME_SOURCE[2:]
UUID_HEX = "4" + "0" * 11 + "4" + "0" * 3 + "8" + "0" * 15
INPUTS = ["make_demo_run() damped-oscillator recording; statistics.v1 on channel v over [1, 2] s, twice, plus one "
          "legacy analysis.stats result",
          "examples/energy-accuracy/{baseline,reset,missing,under-target}.json (origin synthetic_fixture)",
          "Workspaces saved by Session.save_workspace and reopened by Session.from_workspace in temporary directories"]
OBSERVATION = ("No physical observation. The observed objects are CIW records (sessions, executions, results, "
               "bundles, receipts, saved workspace JSON) produced offline by the unmodified CIW code; the energy "
               "logs are synthetic fixtures and no hardware was sampled.")
COMPARISON = ("Each digest is re-derived with a local canonical-JSON/SHA-256 implementation in ciw.lab; that is the "
              "same implementation origin as CIW, so agreement is labelled numerically_verified, never "
              "independently_verified.")


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


def _raises(call) -> bool:
    try:
        call()
    except (ValueError, TypeError, KeyError):
        return True
    return False


# ------------------------------------------------------------------ checks
def _exact(reference: str, mismatches: int) -> dict:
    return {"reference_kind": "exact_arithmetic", "reference": reference, "observed": float(mismatches),
            "tolerance": 0.0, "comparison": "abs_le", "passed": mismatches == 0}


def _invariant(reference: str, observed, tolerance=0.0, comparison="abs_le") -> dict:
    observed, tolerance = float(observed), float(tolerance)
    holds = {"abs_le": abs(observed) <= tolerance, "le": observed <= tolerance, "ge": observed >= tolerance}[comparison]
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


def _blocked(fields: dict) -> dict:
    fields = dict(fields)
    reason = ("Blocked: the bundled energy-accuracy fixture logs (examples/energy-accuracy/*.json) are not present "
              "next to this source tree, so the offline workbench path cannot be exercised.")
    fields["experiment"] = reason + " Planned: " + fields.get("experiment", "")
    fields["unresolved_assumptions"] = list(fields.get("unresolved_assumptions", [])) + [reason]
    return {"state": "blocked", "fields": fields, "findings": []}


def _fields(hypothesis, model, invariant, experiment, result, uncertainty, failures, assumptions, next_task,
            inputs=None) -> dict:
    return {"hypothesis": hypothesis, "mathematical_model": model, "input_data": inputs or list(INPUTS),
            "observation_model": OBSERVATION, "expected_invariant": invariant, "experiment": experiment,
            "numerical_result": result, "uncertainty": uncertainty, "failure_modes_checked": failures,
            "unresolved_assumptions": assumptions, "recommended_next_task": next_task}


def _accepted(row: dict) -> bool:
    """Accepted on reopen, or accepted by a pure validator (which may report a status)."""
    return row["observed"] == "accepted" or row["observed"].startswith("accepted:")


def _authentication(rows) -> dict:
    survivors = sorted(row["name"] for row in rows if _accepted(row))
    return finding(
        "Retained workspace records are authenticated: a holder without a secret cannot produce a forged record "
        "that reopens", "provenance", {"surviving_mutants_in_this_task": survivors},
        {"notes": "Every CIW seal and identity on these paths (record_digest, bundle_digest, verification_id, "
                  "replay_id, result_id, source_id) is an unkeyed SHA-256 over public canonical JSON, and no keyed "
                  "signature or MAC is retained. Authentication is therefore not established; the listed "
                  "surviving mutants are concrete forgeries accepted on reopen."},
        tolerance=EXACT, expected_not_established=True)


def _survivor(row: dict, claim: str, statement: str) -> dict:
    witness = {key: row[key] for key in ("name", "target", "recompute", "description", "observed")}
    if "post_reopen" in row:
        witness["post_reopen"] = row["post_reopen"]
    return finding(claim, "provenance", {"mutant": row["name"], "observed": row["observed"]},
                   {"checks": [_invariant(f"{row['name']}: accepted (1 = accepted)",
                                          1.0 if _accepted(row) else 0.0, 1.0, "ge")]},
                   tolerance=EXACT, counterexample={"statement": statement, "witness": witness})


def _kills(rows, claim: str, harness: dict | None = None) -> dict:
    """One finding over every predicted refusal: each must be refused with the predicted message."""
    predicted = [row for row in rows if not row["predicted"].startswith("accepted")]
    checks = [_refusal(f"{row['name']} ({row['recompute']} recompute): refusal on reopen", row["predicted"],
                       row["observed"]) for row in predicted]
    if harness is not None:
        checks.append(_exact("harness control: forger recomputation reproduces every CIW digest of the unedited "
                             "workspace", 0 if harness["reforge_reproduces_ciw_digests"] else 1))
        checks.append(_exact("harness control: the unedited saved workspace reopens",
                             0 if harness["unchanged_workspace_reopens"] else 1))
    value = [{"mutant": row["name"], "recompute": row["recompute"], "observed": row["observed"]} for row in predicted]
    return finding(claim, "provenance", value, {"checks": checks}, tolerance=EXACT)


def _table(headers, rows) -> str:
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _matrix_markdown(rows) -> str:
    return _table(("Task", "Mutant", "Target", "Recompute", "Predicted", "Observed", "Outcome"),
                  [(row["task"], row["name"], row["target"], row["recompute"], row["predicted"], row["observed"],
                    "killed" if row["killed"] else "SURVIVED") for row in rows])


def _retain_rows(ctx, rows, stem: str) -> None:
    ctx.artifact_json(f"{stem}.json", {"schema": "ciw.lab-mutation-matrix.v1", "rows": rows})
    ctx.artifact_text(f"{stem}.md", _matrix_markdown(rows))


def _summary(rows) -> str:
    survivors = [row["name"] for row in rows if _accepted(row)]
    matched = sum(row["matches_prediction"] for row in rows)
    return (f"{len(rows)} mutants; {sum(row['killed'] for row in rows)} refused, {len(survivors)} accepted "
            f"({', '.join(survivors) or 'none'}); {matched}/{len(rows)} outcomes match the prediction exactly")


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
    """Change one analysed number; optionally make every copy and digest agree with the edit."""
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
            return {"created_at": native["created_at"], "session_id": native["session_id"],
                    "receipt_source": native["replay_receipts"][0]["source_bundle_digest"]}
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
    return {"retained_runtime": {key: runtime[key] for key in ("code_sha256", "python_version")},
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


def _m(name, task_id, target, recompute, description, predicted, apply, witness=None):
    return Mutant(name, task_id, target, recompute, description, predicted, apply, witness)


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
       "operation_id changed to spectrum.periodogram.v1 in both records; resealed",
       "Invalid saved spectrum data fields or sample count",
       _oscillator(["E1", "R1"], lambda r, v: r.update(operation_id="spectrum.periodogram.v1"))),
    _m("revision.gap", "T080", "oscillator selection history", "local",
       "selection revision 1000 with execution and result claiming revision 999; resealed",
       "accepted", _revision_gap, _w_revision),
    # T081: numerical content of retained results.
    _m("energy-data.naive", "T081", "energy replay result data", "none",
       "measurement.batch_count incremented in the replay step; nothing recomputed", BUNDLE, _energy_data(False)),
    _m("energy-data.reforged", "T081", "energy replay result data", "full",
       "batch_count incremented in the step and its reproductions; every digest recomputed", BINDING,
       _energy_data(True)),
    _m("oscillator-stats.naive", "T081", "oscillator result data", "none",
       "statistics mean moved to the interval midpoint; not resealed", SEAL,
       _oscillator(["R1"], _midpoint, reseal=False)),
    _m("oscillator-stats.out-of-bounds", "T081", "oscillator result data", "local",
       "statistics mean moved above the maximum; resealed", "Saved statistics mean is outside its bounds",
       _oscillator(["R1"], lambda r, v: r["data"].update(mean=r["data"]["maximum"] + 1.0))),
    _m("oscillator-stats.resealed", "T081", "oscillator result data", "local",
       "statistics mean moved to the midpoint of [minimum, maximum]; resealed", "accepted",
       _oscillator(["R1"], _midpoint), _w_mean("R1")),
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
       "receipt re-pointed at a sibling execution of the same source bytes; verification rebuilt", "accepted",
       _rebind_receipt("B0b"), _w_receipt),
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
       "replay bundle backdated with a new session id; bundle, verification and receipt digests recomputed",
       "accepted", _reidentify, _w_reidentified),
    # T086: verification subject.
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
    _m("receipt-subject.rebound-with-source", "T086", "energy replay receipt verification", "local",
       "subject and source digest moved together to a sibling execution; verification rebuilt", "accepted",
       _rebind_receipt("B0b"), _w_receipt),
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
    "oscillator-stats.resealed": "Unkeyed record seals detect every edit to a retained numerical result",
    "oscillator-stats.legacy": "Every retained oscillator result is sealed against edits",
    "fresh.created-at-shift": "A retained execution occurrence binds its creation time",
    "receipt.deleted": "A replay bundle cannot be retained without its replay receipt",
    "receipt-source.sibling-execution": "Unkeyed record seals detect every replay-provenance forgery",
    "receipt-replayed.reidentified-bundle": "A retained replay bundle cannot be re-identified (backdated, re-sessioned) without detection",
    "receipt-subject.rebound-with-source": "A replay verification subject cannot be rebound once retained",
    "oscillator-method.injected": "Sealed operation records refuse verification-method claims outside their schema",
    "oscillator-independent.injected": "Sealed operation records refuse independence claims outside their schema",
    "exchange.verification-independent": "A recomputed exchange content identity rejects a forged independence claim",
    "oscillator-admission.injected": "Sealed operation records refuse admission claims outside their schema",
    "oscillator-runtime.both": "Unkeyed record seals detect a forged provider runtime identity",
    "energy-runtime.all-bundles": "Reopening a workspace detects a forged analysis runtime identity",
    "energy-runtime.python-version": "Reopening a workspace detects forged runtime dependency versions",
}


def _validator_rows(fixture: dict) -> list:
    """Pure-validator mutants (no workspace): ESM candidate responses and exchange identities."""
    from ..exchange import VERIFICATION_SCHEMA, _identity
    natives = fixture["natives"]
    case = esm_case(natives["B0"])
    body = {"subject_ref": natives["B0"]["bundle_digest"], "outcome": "passed", "independent": False,
            "method": "same_runtime_fresh_occurrence_reproduction"}
    tampered = exchange_artifact(VERIFICATION_SCHEMA, body, "verification_id")
    tampered["subject_ref"] = natives["B0b"]["bundle_digest"]
    independent = exchange_artifact(VERIFICATION_SCHEMA, dict(body, independent=True,
                                                              method="independent_reimplementation"),
                                    "verification_id")
    candidate_mismatch = "ESM candidate does not bind the selected native bundle"
    esm = "ESM candidate inspection (pure validator)"
    exchange = "exchange verification identity (pure validator)"
    return [
        validator_row("esm.bundle-subject", "T086", esm, "candidate bundleDigest set to a sibling execution",
                      candidate_mismatch, lambda: check_esm(case, lambda c: c["response"]["candidate"].update(
                          bundleDigest=natives["B0b"]["bundle_digest"]))),
        validator_row("exchange.verification-subject", "T086", exchange,
                      "subject_ref edited without recomputing verification_id",
                      "verification_id does not match the artifact content",
                      lambda: _identity(tampered, "verification_id")),
        validator_row("esm.inspection-independent", "T088", esm, "independentlyVerified set true",
                      "Native ESM inspection binding or scope mismatch",
                      lambda: check_esm(case, lambda c: c["response"].update(independentlyVerified=True))),
        validator_row("esm.candidate-independent", "T088", esm, "candidate verification independent set true",
                      candidate_mismatch, lambda: check_esm(case, lambda c: c["response"]["candidate"][
                          "verification"].update(independent=True))),
        validator_row("exchange.verification-independent", "T088", exchange,
                      "independent true and method independent_reimplementation; verification_id recomputed",
                      "accepted:content_recomputed_not_authenticated",
                      lambda: _identity(independent, "verification_id")),
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
           properties, exercised=True) -> dict:
    return {"identity": identity, "scope": scope, "derivation_class": derivation_class, "derivation": derivation,
            "binds": binds, "across_replay": across_replay, "across_reopen": across_reopen,
            "validated_at": list(validated_at), "exercised": exercised,
            "properties": {name: bool(value) for name, value in properties.items()}}


def identity_matrix(fixture: dict) -> list:
    """Predicted identity semantics with each property checked on the offline fixture."""
    from ..core.identities import evidence_id, validate_evidence_identity
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

    edited = deepcopy(run)
    channel = next(iter(edited["channels"]))
    index = next(i for i, value in enumerate(edited["channels"][channel]["values"]) if value is not None)
    edited["channels"][channel]["values"][index] += 1.0
    rendered = dict(deepcopy(run), render={"lab": "changed render state"})
    descriptor = {key: S[key] for key in ("schema", "kind", "label", "source_schema", "evidence_id", "byte_count")}
    result = steps[0]["result"]
    exchange_result = exchange_artifact(RESULT_SCHEMA, {"input_refs": ["lab-input"],
                                                        "components": [{"name": "q", "value": 1.0}]}, "result_id")
    exchange_tampered = dict(exchange_result, components=[{"name": "q", "value": 2.0}])
    case = esm_case(B0)

    def renamed(c):
        c["response"]["requestId"] = c["policy"]["review_context"]["requestId"] = "lab-request-renamed"

    def seal_ok(record):
        return record["record_digest"] == _sha(_without(record, "record_digest"), ascii_only=True)

    rows = [
        _entry("recording evidence_id", "oscillator recording", "content_hash",
               "sha256 over canonical JSON of instrument, metadata, time_s and channels",
               "all scientific content including adapter and calibration metadata; excludes run_id and render",
               "not_applicable (source evidence is not replayed)", "stable",
               ["src/ciw/core/identities.py:evidence_id", "src/ciw/core/identities.py:validate_evidence_identity",
                "src/ciw/session.py:_validate_evidence"],
               {"recomputed_from_scientific_fields": _sha({k: run[k] for k in SCIENTIFIC_FIELDS}, True) == run["evidence_id"],
                "render_excluded": evidence_id(rendered) == run["evidence_id"],
                "sample_edit_refused": _raises(lambda: validate_evidence_identity(edited)),
                "retained_across_reopen": fixture["reopened_run_evidence_id"] == run["evidence_id"]
                == fixture["workspace"]["run"]["evidence_id"]}),
        _entry("recording file name", "oscillator recording", "content_hash",
               "'recording-' + sha256 over canonical JSON of the whole run, written by Session",
               "entire parsed run including run_id and render; the file is a CIW re-serialization",
               "not_applicable", "stable", ["src/ciw/session.py:_recording_file",
                                            "src/ciw/session.py:_validate_saved_result"],
               {"recomputed_from_run": fixture["recording"]["file"]
                == "recording-" + hashlib.sha256(_canon(run, True)).hexdigest() + ".json",
                "file_is_reserialization_not_input_bytes": fixture["recording"]["is_reserialization"]}),
        _entry("workbench source evidence_id", "workbench source", "byte_hash", "'sha256:' + sha256(exact source bytes)",
               "exact retained bytes only (not label or kind)", "stable (bundle artifact_ref equals it)", "stable",
               ["src/ciw/workbench.py:_source", "src/ciw/workbench.py:Workbench.restore",
                "src/ciw/energy_workflow.py:EnergyAccuracyWorkflow._validate"],
               {"equals_sha256_of_bytes": S["evidence_id"] == "sha256:" + hashlib.sha256(baseline).hexdigest(),
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
                "same_bytes_and_label_same_id_across_sessions": SC["source_id"] == S["source_id"]}),
        _entry("bundle experiment_digest", "energy bundle source", "content_hash",
               "sha256 over canonical JSON of the parsed log", "parsed log content (whitespace and key order excluded)",
               "stable", "stable", ["src/ciw/energy_workflow.py:EnergyAccuracyWorkflow._validate"],
               {"recomputed_from_parsed_content": B0["source"]["experiment_digest"] == _sha(parsed),
                "stable_across_replay_and_sessions": len({n["source"]["experiment_digest"] for n in same}) == 1}),
        _entry("energy log run_id (bundle experiment_id)", "energy log", "caller_declared",
               "copied from the retained log's run_id field (format energy-run-<32 hex>)",
               "the log author's measurement-occurrence claim; bound to log_digest by workbench claims",
               "stable", "stable", ["src/ciw/energy_records.py:_validate", "src/ciw/workbench.py:_claims"],
               {"copied_from_log": all(n["source"]["experiment_id"] == parsed["run_id"] for n in same),
                "not_content_derived": not parsed["run_id"].startswith("sha256:")}),
        _entry("oscillator operation_id", "oscillator operation", "caller_declared_name",
               "versioned registry name chosen by the requester", "which registered operation ran", "not_applicable",
               "stable", ["src/ciw/operations/registry.py:valid_operation_id",
                          "src/ciw/operations/runner.py:validate_execution"],
               {"same_for_repeat": E1["operation_id"] == E2["operation_id"] == R1["operation_id"] == "statistics.v1",
                "versioned_name": valid_operation_id(E1["operation_id"]),
                "not_content_derived": not E1["operation_id"].startswith("sha256:")}),
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
                "depends_on_execution_ref": _sha(dict(_without(result, "result_id"), execution_ref="execution-" + UUID_HEX))
                != result["result_id"]}),
        _entry("energy numerical_result_id", "energy bundle step", "content_hash",
               "sha256 over {operation_id, data}", "operation name and analysed numbers only", "stable", "stable",
               ["src/ciw/energy_workflow.py:EnergyAccuracyWorkflow._validate_step", "src/ciw/workbench.py:_validate_links"],
               {"recomputed_from_operation_and_data": all(
                   s["numerical_result_id"] == _sha({"operation_id": s["operation_id"], "data": s["result"]["data"]})
                   for s in steps),
                "single_value_across_replay_reopen_and_sessions": len({s["numerical_result_id"] for s in steps}) == 1}),
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
                "receipt_and_verification_excluded": _bundle_digest(_without(B1, "replay_receipts", "verification"))
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
               {"recomputable_without_secret": all(seal_ok(record) for record in (E1, E2, R1, R2)),
                "legacy_result_unsealed": "record_digest" not in RL and "schema" not in RL}),
        _entry("oscillator runtime identity", "oscillator execution and result", "provider_declared",
               "the operation provider's runtime_identity() dict", "a provider name and version string",
               "not_applicable", "stable (never re-checked)", ["src/ciw/operations/runner.py:validate_execution"],
               {"declared_by_provider": E1["runtime"] == R1["runtime"] == fixture["runtime_declared"],
                "no_code_digest": not any("sha256" in str(value) for value in E1["runtime"].values())}),
        _entry("energy runtime identity", "energy bundle", "code_hash_plus_declared_versions",
               "sha256 of three CIW source files plus declared python and numpy versions",
               "analysis code bytes and dependency version strings", "stable (compared on replay)",
               "stable (format-checked only)", ["src/ciw/energy_workflow.py:analysis_identity",
                                                "src/ciw/energy_workflow.py:_check_runtime",
                                                "src/ciw/energy_workflow.py:EnergyAccuracyWorkflow._adapters"],
               {"equals_current_analysis_identity": B0["runtimes"]["energy"] == analysis_identity(),
                "stable_across_occurrences": len({_sha(n["runtimes"]) for n in same}) == 1}),
        _entry("exchange result_id / verification_id", "external exchange artifact", "content_hash",
               "sha256(schema + NUL + canonical artifact without its id)", "artifact content; status "
               "content_recomputed_not_authenticated", "not_applicable", "not_applicable",
               ["src/ciw/exchange.py:_identity"],
               {"status_not_authenticated": _identity(exchange_result, "result_id") == "content_recomputed_not_authenticated",
                "tamper_refused": _raises(lambda: _identity(exchange_tampered, "result_id"))}),
        _entry("exchange batch_id", "external exchange artifact", "caller_declared",
               "any string supplied by the producer", "nothing; reported as caller_declared_reference",
               "not_applicable", "not_applicable", ["src/ciw/exchange.py:_identity"],
               {"caller_declared": _identity({"schema": OBSERVATION_SCHEMA, "batch_id": "any producer string"},
                                             "batch_id") == "caller_declared_reference"}),
        _entry("ESM requestId / inspectedAt", "external ESM inspection", "caller_declared",
               "copied from the operator's review context and request parameters",
               "equality between request and response only", "not_applicable", "not_applicable",
               ["src/ciw/candidate_evidence.py:validate_response"],
               {"base_case_accepted": not _raises(lambda: check_esm(case)),
                "consistent_rename_accepted": not _raises(lambda: check_esm(case, renamed)),
                "mismatch_refused": _raises(lambda: check_esm(case, lambda c: c["response"].update(requestId="other")))}),
        _entry("ESM bundleBytesDigest", "external ESM inspection", "byte_hash",
               "'sha256:' + sha256(canonical bundle bytes)", "the exact canonical bytes of the selected bundle",
               "not_applicable", "not_applicable", ["src/ciw/candidate_evidence.py:validate_response"],
               {"equals_canonical_bundle_hash": case["response"]["bundleBytesDigest"]
                == "sha256:" + hashlib.sha256(_canon(B0)).hexdigest(),
                "mismatch_refused": _raises(lambda: check_esm(case, lambda c: c["response"].update(
                    bundleBytesDigest=FORGED_DIGEST)))}),
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


T077_PLAN = _fields(
    "Every CIW identity on the offline paths falls into one derivation class (content hash, byte hash, fresh "
    "event UUID, caller-declared name/reference, provider-declared, or absent), and its binding, replay and "
    "reopen behaviour follow from that class.",
    "Content identities are sha256 over canonical JSON (sort_keys, compact separators); byte identities are "
    "sha256 over raw bytes; event identities are uuid4 draws (collision probability ~2^-122 per pair).",
    "Content identities recompute exactly and are stable when their content is; event identities are distinct "
    "per occurrence; caller-declared identities are copied, not derived.",
    "Build an oscillator session and energy-accuracy bundles offline, replay, save, reopen, replay again, replay a "
    "replay, and repeat in an independent session; check every predicted property per identity.",
    "not run", "not run", [], [], "T078")


@task("T077", changed_files=CHANGED, regression_tests=(_node("test_identity_matrix"),), plan=T077_PLAN)
def identity_matrix_task(ctx):
    if not fixture_available():
        return _blocked(T077_PLAN)
    fixture = _fixture(ctx)
    rows = identity_matrix(fixture)
    ctx.artifact_json("identity-matrix.json", {"schema": "ciw.lab-identity-matrix.v1", "rows": rows})
    ctx.artifact_text("identity-matrix.md", _matrix_markdown_table(rows))
    exercised = [row for row in rows if row["exercised"]]
    properties = sum(len(row["properties"]) for row in exercised)
    held = sum(sum(row["properties"].values()) for row in exercised)
    classes = dict(sorted(Counter(row["derivation_class"] for row in rows).items()))
    checks = [_exact(f"{row['identity']}: predicted properties that did not hold", len(row["properties"])
                     - sum(row["properties"].values())) for row in exercised]
    oscillator, natives = fixture["oscillator"], fixture["natives"]
    R1, R2 = oscillator["R1"], oscillator["R2"]
    B1 = natives["B1"]
    from ..telemetry import _bundle_digest
    stripped = _without(B1, "replay_receipts")
    findings = [
        finding("Every exercised identity matches its predicted derivation class, freshness and binding on the "
                "offline paths", "provenance",
                {"rows": len(rows), "exercised_rows": len(exercised), "properties": properties,
                 "properties_held": held, "derivation_classes": classes},
                {"checks": checks}, tolerance=EXACT),
        finding("Oscillator operation results carry no replay-stable numerical-result identity", "provenance",
                {"numerical_result_id_present": "numerical_result_id" in R1, "data_equal": R1["data"] == R2["data"],
                 "result_ids_equal": R1["result_id"] == R2["result_id"]},
                {"checks": [_invariant("equal data, distinct result ids and no numerical identity (1 = observed)",
                                       1.0 if ("numerical_result_id" not in R1 and R1["data"] == R2["data"]
                                               and R1["result_id"] != R2["result_id"]) else 0.0, 1.0, "ge")]},
                tolerance=EXACT,
                counterexample={"statement": "Every retained CIW operation result carries a replay-stable "
                                             "numerical-result identity",
                                "witness": {"operation": "statistics.v1", "records": ["result:R1", "result:R2"],
                                            "fields": sorted(R1)}}),
        finding("The energy replay bundle identity excludes its replay receipt and verification", "provenance",
                {"digest_unchanged_without_receipt": _bundle_digest(stripped) == B1["bundle_digest"],
                 "excluded_fields": ["bundle_digest", "replay_receipts", "verification"]},
                {"checks": [_exact("bundle digest recomputed without replay_receipts differs from the retained one",
                                   0 if _bundle_digest(stripped) == B1["bundle_digest"] else 1)]},
                tolerance=EXACT,
                counterexample={"statement": "A retained bundle identity binds every provenance record stored in the "
                                             "bundle",
                                "witness": {"bundle": "bundle:B1", "removed": "replay_receipts",
                                            "function": "src/ciw/telemetry.py:_bundle_digest"}}),
        finding("Retained workspace records are authenticated: a holder without a secret cannot produce a forged "
                "record that reopens", "provenance",
                {"unkeyed_identity_classes": sorted({"content_hash", "content_hash_of_fresh_occurrence",
                                                     "unkeyed_seal", "byte_hash"} & set(classes))},
                {"notes": "Every content identity and seal in the matrix is an unkeyed SHA-256 over public canonical "
                          "JSON or raw bytes; no keyed signature or MAC exists on these paths (see T084-T090 for "
                          "accepted forgeries)."}, tolerance=EXACT, expected_not_established=True),
    ]
    fields = dict(T077_PLAN)
    fields.update(
        experiment=T077_PLAN["experiment"] + " " + COMPARISON + " Matrix retained as identity-matrix.json/.md.",
        numerical_result=f"{len(rows)} identities ({len(exercised)} exercised offline); {held}/{properties} predicted "
                         f"properties held; classes {classes}",
        uncertainty="Exact equality tests on digests and UUID draws; a uuid4 collision among the <30 fresh "
                    "identities has probability below 1e-33. Properties are sampled on one fixture, not proved for "
                    "all inputs.",
        failure_modes_checked=["digest recomputation mismatch", "identity reused across occurrences",
                               "identity not stable across replay or reopen", "label-independent source identity",
                               "exchange identity tamper", "ESM request/response mismatch"],
        unresolved_assumptions=[
            "ESM candidate_id and candidate execution identities were not exercised: they need an operator-bound ESM "
            "adapter and a telemetry or calibrated bundle.",
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
    "bytes in == bytes retained for every accepted source; refused transports leave no retained source.",
    "Retain four fixture logs and eight byte variants of the baseline log, execute each, compare source.get bytes "
    "live and after offline reopen and the bundle evidence bytes with the input; submit three non-canonical base64 "
    "transports.", "not run", "not run", [], [], "T079")


@task("T078", changed_files=CHANGED, regression_tests=(_node("test_exact_source_bytes"),), plan=T078_PLAN)
def exact_source_bytes(ctx):
    if not fixture_available():
        return _blocked(T078_PLAN)
    variants, fixture = _variants(ctx), _fixture(ctx)
    records = variants["records"]
    ctx.artifact_json("source-retention.json", {"records": records, "refusals": variants["refusals"]})
    mismatches = {key: sum(not record[key] for record in records)
                  for key in ("live_bytes_equal", "reopened_bytes_equal", "bundle_bytes_equal")}
    id_mismatch = sum(record["evidence_id"] != record["input_sha256"] for record in records)
    count_mismatch = sum(record["declared_byte_count"] != record["byte_count"] for record in records)
    artifact_mismatch = sum(record["artifact_ref"] != record["evidence_id"] for record in records)
    same_bytes = [record for record in records if record["label"] in ("baseline", "baseline/original")]
    transports = {name: row for name, row in variants["refusals"].items() if name.startswith("base64/")}
    expected_transport = {"base64/missing-padding": "Source bytes must use canonical base64",
                          "base64/line-wrapped": "Source bytes must use canonical base64",
                          "base64/noncanonical-trailing-bits": "Source bytes must use bounded canonical base64"}
    recording = fixture["recording"]
    findings = [
        finding("Workbench sources retain the exact supplied bytes live, inside native bundles and after offline "
                "reopen", "provenance",
                {"sources": len(records), "fixture_logs": sum(r["kind"] == "fixture_log" for r in records),
                 "byte_variants": sum(r["kind"] == "byte_variant" for r in records), **mismatches,
                 "evidence_id_mismatches": id_mismatch, "byte_count_mismatches": count_mismatch,
                 "bundle_artifact_ref_mismatches": artifact_mismatch},
                {"checks": [_exact(f"sources whose {key.replace('_', ' ')} is false", value)
                            for key, value in mismatches.items()]
                 + [_exact("evidence_id differs from sha256 of the input bytes", id_mismatch),
                    _exact("declared byte_count differs from the input length", count_mismatch),
                    _exact("bundle artifact_ref differs from the source evidence_id", artifact_mismatch)]},
                tolerance=EXACT),
        finding("Non-canonical base64 transports are refused rather than normalized into retained bytes",
                "provenance", {name: row["message"] for name, row in transports.items()},
                {"checks": [_refusal(f"{name}: source.add refusal", expected_transport[name],
                                     transports[name].get("message") or transports[name]["outcome"])
                            for name in expected_transport]}, tolerance=EXACT),
        finding("Evidence identity depends only on bytes: one byte string under two labels shares evidence_id while "
                "source_id differs", "provenance",
                {"sources": len(same_bytes), "distinct_evidence_ids": len({r["evidence_id"] for r in same_bytes}),
                 "distinct_source_ids": len({r["source_id"] for r in same_bytes})},
                {"checks": [_count("distinct evidence ids for identical bytes", len({r["evidence_id"] for r in same_bytes}), 1),
                            _count("distinct source ids for two labels", len({r["source_id"] for r in same_bytes}), 2)]},
                tolerance=EXACT),
        finding("The oscillator recording path retains canonical content, not caller bytes: its file is a CIW "
                "re-serialization named by a content digest", "provenance",
                {"recording_file_is_reserialization": recording["is_reserialization"]},
                {"checks": [_exact("recording file differs from json.dumps(run, indent=2) re-serialization",
                                   0 if recording["is_reserialization"] else 1)]}, tolerance=EXACT),
        finding("The retained energy logs are real GPU energy measurements", "physical",
                {"origins": sorted({"synthetic_fixture"})},
                {"notes": "Every bundled log declares origin synthetic_fixture and no device was sampled; byte "
                          "retention says nothing about the physical truth of a log."}, tolerance=EXACT),
    ]
    fields = dict(T078_PLAN)
    fields.update(
        input_data=["examples/energy-accuracy/{baseline,reset,missing,under-target}.json and eight byte variants of "
                    "baseline (crlf, minified, tab-indented, sorted-keys, reversed-keys, trailing-whitespace, "
                    "float-spelling, original under a second label)", "make_demo_run() oscillator recording"],
        experiment=T078_PLAN["experiment"] + " Retained as source-retention.json.",
        numerical_result=f"{len(records)} sources; byte mismatches live/reopened/bundle = "
                         f"{mismatches['live_bytes_equal']}/{mismatches['reopened_bytes_equal']}/"
                         f"{mismatches['bundle_bytes_equal']}; evidence_id mismatches {id_mismatch}; "
                         f"3/3 non-canonical base64 transports refused",
        uncertainty="Exact byte equality; no numerical tolerance is involved. The sample is twelve byte strings of "
                    "one schema, not every possible source kind.",
        failure_modes_checked=["byte normalization on retention", "re-encoding on workspace save",
                               "bundle evidence differing from the source", "base64 padding, wrapping and "
                               "non-canonical trailing bits", "label leaking into evidence identity"],
        unresolved_assumptions=["Only the energy-accuracy source kind was exercised; other kinds share "
                                "workbench._source but have their own parsers.",
                                "The oscillator recording is supplied as parsed JSON, so its input bytes do not "
                                "exist to retain."],
        recommended_next_task="T079 (whitespace-preserving evidence retention)")
    return {"state": _state(findings), "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T079 whitespace-preserving retention
T079_PLAN = _fields(
    "Byte variants of one log that differ only in whitespace, key order or float spelling are retained as distinct "
    "evidence with distinct byte-level identities, while every content-level identity is equal.",
    "Byte-level identities are functions of the bytes b (sha256(b)); content-level identities are functions of "
    "canon(parse(b)). For variants with canon(parse(b_i)) equal and b_i distinct, byte-level identities are "
    "pairwise distinct and content-level identities coincide.",
    "distinct evidence_id/artifact_ref/source_id per variant; one experiment_digest, log_digest and "
    "numerical_result_id across variants.",
    "Retain and analyse eight variants of baseline.json in one workbench; compare identities; submit an "
    "int-for-float re-encoding and a byte-order-mark variant.", "not run", "not run", [], [], "T080")


@task("T079", changed_files=CHANGED, regression_tests=(_node("test_whitespace_variants"),), plan=T079_PLAN)
def whitespace_retention(ctx):
    if not fixture_available():
        return _blocked(T079_PLAN)
    variants = _variants(ctx)
    records = [record for record in variants["records"] if record["kind"] == "byte_variant"]
    count = len(records)

    def distinct(key):
        return len({record[key] for record in records})
    byte_level = {key: distinct(key) for key in ("input_sha256", "evidence_id", "artifact_ref", "source_id")}
    content_level = {key: distinct(key) for key in ("experiment_digest", "log_digest", "numerical_result_id")}
    levels = [
        {"identity": "workbench evidence_id", "level": "byte", "distinct_values": byte_level["evidence_id"]},
        {"identity": "bundle source artifact_ref / sha256", "level": "byte", "distinct_values": byte_level["artifact_ref"]},
        {"identity": "workbench source_id", "level": "byte + label", "distinct_values": byte_level["source_id"]},
        {"identity": "bundle experiment_digest", "level": "canonical content", "distinct_values": content_level["experiment_digest"]},
        {"identity": "log_digest (sealed inside the log)", "level": "canonical content", "distinct_values": content_level["log_digest"]},
        {"identity": "numerical_result_id", "level": "canonical content of analysed data", "distinct_values": content_level["numerical_result_id"]},
        {"identity": "recording evidence_id (oscillator)", "level": "canonical content of scientific fields", "distinct_values": None},
        {"identity": "bundle_digest / result_id / verification_id / replay_id", "level": "canonical content including fresh occurrences", "distinct_values": None},
    ]
    ctx.artifact_json("identity-levels.json", {"variants": [r["label"] for r in records], "levels": levels,
                                               "records": records})
    ctx.artifact_text("identity-levels.md", _table(("Identity", "Level", f"Distinct values over {count} variants"),
                                                  [(r["identity"], r["level"], "n/a" if r["distinct_values"] is None
                                                    else r["distinct_values"]) for r in levels]))
    refusals = variants["refusals"]
    int_case, bom = refusals["int-for-float"], refusals["byte-order-mark"]
    exact = sum(r["live_bytes_equal"] and r["reopened_bytes_equal"] and r["bundle_bytes_equal"] for r in records)
    findings = [
        finding("Whitespace, key-order and float-spelling variants retain distinct exact bytes and distinct evidence "
                "identities while their parsed content is canonically equal", "provenance",
                {"variants": count, "exactly_retained": exact, "canonically_equal": sum(r["canonical_equal_to_baseline"] for r in records),
                 **{f"distinct_{key}": value for key, value in byte_level.items()}},
                {"checks": [_count("variants retained byte-exactly (live, bundle, reopen)", exact, count),
                            _count("variants canonically equal to baseline", sum(r["canonical_equal_to_baseline"] for r in records), count)]
                 + [_count(f"distinct {key}", value, count) for key, value in byte_level.items()]},
                tolerance=EXACT),
        finding("Content-level identities coincide across the byte variants: one experiment_digest, log_digest and "
                "numerical_result_id", "provenance", content_level,
                {"checks": [_count(f"distinct {key}", value, 1) for key, value in content_level.items()]},
                tolerance=EXACT),
        finding("A numerically equal int-for-float re-encoding is refused, not aliased: canonical content identity "
                "is type-sensitive", "provenance",
                {"python_equal": int_case["python_equal"], "canonical_equal": int_case["canonical_equal"],
                 "observed": int_case.get("message") or int_case["outcome"]},
                {"checks": [_refusal("first '1.0,' rewritten as '1,': source.add refusal",
                                     "Workload solver differs from plan", int_case.get("message") or int_case["outcome"]),
                            _exact("Python equality holds while canonical equality fails (0 = observed)",
                                   0 if (int_case["python_equal"] and not int_case["canonical_equal"]) else 1)]},
                tolerance=EXACT,
                counterexample={"statement": "Canonical-content identity treats numerically equal JSON numbers as "
                                             "equal content",
                                "witness": {"edit": "first '1.0,' in baseline.json rewritten as '1,'",
                                            "python_equal": int_case["python_equal"],
                                            "canonical_equal": int_case["canonical_equal"],
                                            "observed": int_case.get("message")}}),
        finding("A UTF-8 byte-order-mark variant with equal content is refused", "provenance",
                {"canonical_equal_after_bom_strip": bom["canonical_equal"], "code": bom.get("code"),
                 "observed": bom.get("message") or bom["outcome"]},
                {"checks": [_refusal("BOM-prefixed baseline: source.add refusal",
                                     "The bound runtime did not return finite, unambiguous JSON",
                                     bom.get("message") or bom["outcome"])]}, tolerance=EXACT),
    ]
    fields = dict(T079_PLAN)
    fields.update(
        input_data=["examples/energy-accuracy/baseline.json and eight byte variants (original, crlf, minified, "
                    "tab-indented, sorted-keys, reversed-keys, trailing-whitespace, float-spelling 1e-09 -> "
                    "0.000000001)", "two refused re-encodings (int-for-float, byte-order mark)"],
        experiment=T079_PLAN["experiment"] + " Identity levels retained as identity-levels.json/.md.",
        numerical_result=f"{count} variants: byte-level distinct counts {byte_level}; content-level distinct counts "
                         f"{content_level}; int-for-float and BOM variants refused",
        uncertainty="Exact equality of digests; the variant set samples whitespace, order and float spelling but "
                    "not every lexical freedom JSON allows (for example escaped string characters).",
        failure_modes_checked=["whitespace normalization before hashing", "key-order sensitivity of content "
                               "identities", "float lexical spelling", "int/float type aliasing", "byte-order mark"],
        unresolved_assumptions=[
            "The BOM refusal message names a 'bound runtime' although the defect is in source bytes; the refusal is "
            "correct but its wording is misleading (reported, not changed).",
            "Unicode escape variants (\\u0041 vs A) were not exercised."],
        recommended_next_task="T080 (operation/execution/result identity separation)")
    return {"state": _state(findings), "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T080 separation
T080_PLAN = _fields(
    "Operation, execution and result identities have different kinds (name, fresh occurrence, result occurrence) "
    "and a forged alias between them is refused on reopen.",
    "operation_id is a constant per operation; execution_id and result_id are independent uuid4 draws per "
    "occurrence (energy result_id = sha256 over a record that includes execution_ref); numerical_result_id "
    "excludes the occurrence.",
    "No string is used in two identity roles; changing execution_ref changes result_id but not the operation or "
    "numerical identity; aliasing forgeries are refused.",
    "Compare identity sets across oscillator and energy occurrences, recompute the energy result under a changed "
    "execution_ref, and reopen five aliasing forgeries.", "not run", "not run", [], [], "T081")


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
    moved = deepcopy(natives["B0"]["steps"][0])
    before = {key: moved[key] for key in ("operation_id", "result_id", "numerical_result_id")}
    moved["execution_id"] = "execution-" + UUID_HEX
    restep(moved)
    energy = {"result_id_changed": moved["result_id"] != before["result_id"],
              "numerical_result_id_unchanged": moved["numerical_result_id"] == before["numerical_result_id"],
              "operation_id_unchanged": moved["operation_id"] == before["operation_id"]}
    rows, harness = _task_rows(ctx, "T080"), _mutations(ctx)["harness"]
    _retain_rows(ctx, rows, "aliasing-mutations")
    counts = {role: len(values) for role, values in roles.items()}
    findings = [
        finding("Operation, execution, result, numerical-result and bundle identities never share a value, and an "
                "occurrence change moves the result identity but not the operation or numerical identity",
                "provenance", {"distinct_per_role": counts, "cross_role_overlaps": overlaps, "energy": energy},
                {"checks": [_exact("values shared between identity roles", overlaps),
                            _count("distinct operation identities", counts["operation"], 2),
                            _count("distinct oscillator+energy execution identities", counts["execution"], 3 + len(steps)),
                            _count("distinct result identities", counts["result"], 3 + len(steps)),
                            _exact("energy identities that did not move as predicted under a new execution_ref",
                                   sum(not value for value in energy.values()))]}, tolerance=EXACT),
        _kills(rows, "Aliasing forgeries between operation, execution and result identities are refused on reopen "
                     "with the predicted message", harness),
        *_survivor_findings(rows, {"revision.gap": "Surviving mutant revision.gap: execution and result claim a "
                                                   "selection revision that never existed and reopen"}),
        _authentication(rows),
    ]
    fields = dict(T080_PLAN)
    fields.update(
        experiment=T080_PLAN["experiment"] + " " + COMPARISON + " Mutation rows retained as aliasing-mutations.json/.md.",
        numerical_result=f"distinct identities per role {counts}; cross-role overlaps {overlaps}; mutants: {_summary(rows)}",
        uncertainty="Exact comparisons. The saved selection history is not retained, so revision claims cannot be "
                    "checked by any validator.",
        failure_modes_checked=["identity reuse across roles", "result pointing at another execution", "execution "
                               "pointing at another result", "wrong identity prefix", "operation substitution",
                               "selection revision gap"],
        unresolved_assumptions=["uuid4 freshness is probabilistic, not enforced by a registry across workspaces.",
                                "Selection history is not persisted, so selection_revision is only bounded, not bound."],
        recommended_next_task="T081 (stable numerical-result identity); CIW change: persist the selection history or "
                              "drop selection_revision from sealed records")
    return {"state": _state(findings), "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T081 stable numerical identity
T081_PLAN = _fields(
    "The energy numerical_result_id is a function of the operation and the analysed numbers only, so it is equal "
    "across original, replay, reopen and independent sessions; oscillator results lack such an identity.",
    "numerical_result_id = sha256(canon({operation_id, data})), data = analyze(parse(bytes)) is deterministic "
    "in-process, so the identity is constant over occurrences of the same bytes.",
    "exactly one numerical_result_id over all baseline occurrences; data edits are refused where content is "
    "recomputed and survive where it is only sealed.",
    "Collect the numerical identity from the original, a sibling, a replay, a replay after reopen, a replay of a "
    "replay and an independent session (steps and reproductions); reopen data-edit forgeries.",
    "not run", "not run", [], [], "T082")


@task("T081", changed_files=CHANGED, regression_tests=(_node("test_numerical_identity"),), plan=T081_PLAN)
def numerical_identity(ctx):
    if not fixture_available():
        return _blocked(T081_PLAN)
    fixture = _fixture(ctx)
    natives = fixture["natives"]
    occurrences = [natives[role]["steps"][0] for role in SAME_SOURCE] + [
        natives[role]["verification"]["reproduction"] for role in SAME_SOURCE]
    numerical = {s["numerical_result_id"] for s in occurrences}
    recomputed = sum(s["numerical_result_id"] != _sha({"operation_id": s["operation_id"], "data": s["result"]["data"]})
                     for s in occurrences)
    result_ids = {s["result_id"] for s in occurrences}
    rows, harness = _task_rows(ctx, "T081"), _mutations(ctx)["harness"]
    _retain_rows(ctx, rows, "numerical-mutations")
    ctx.artifact_json("numerical-identity.json", relabel(
        {"occurrences": [{"bundle": role, "step": natives[role]["steps"][0]["execution_id"],
                          "numerical_result_id_equal_to_B0": natives[role]["steps"][0]["numerical_result_id"]
                          == natives["B0"]["steps"][0]["numerical_result_id"]} for role in SAME_SOURCE]},
        role_labels(fixture)))
    findings = [
        finding("The energy numerical_result_id is identical across original, sibling, replay, replay after offline "
                "reopen, replay of a replay and an independent session", "provenance",
                {"occurrences": len(occurrences), "distinct_numerical_result_ids": len(numerical),
                 "distinct_result_ids": len(result_ids), "recomputation_mismatches": recomputed},
                {"checks": [_count("distinct numerical_result_id values", len(numerical), 1),
                            _count("distinct result_id values (fresh occurrences)", len(result_ids), len(occurrences)),
                            _exact("numerical_result_id differs from sha256 over {operation_id, data}", recomputed)]},
                tolerance=EXACT),
        _kills(rows, "Numerical edits that break a seal, a digest or a recomputed analysis are refused on reopen",
               harness),
        *_survivor_findings(rows, {
            "oscillator-stats.resealed": "Surviving mutant oscillator-stats.resealed: an in-bounds statistics edit "
                                         "with a recomputed seal reopens",
            "oscillator-stats.legacy": "Surviving mutant oscillator-stats.legacy: an edit to an unsealed legacy "
                                       "result reopens"}),
        _authentication(rows),
    ]
    fields = dict(T081_PLAN)
    fields.update(
        experiment=T081_PLAN["experiment"] + " " + COMPARISON + " Retained numerical-identity.json and "
                   "numerical-mutations.json/.md.",
        numerical_result=f"{len(occurrences)} baseline occurrences -> {len(numerical)} numerical_result_id, "
                         f"{len(result_ids)} result_id; mutants: {_summary(rows)}",
        uncertainty="Stability was observed within one process and platform; numerical_result_id hashes floats, so "
                    "a different BLAS or NumPy build could change the last bits of analysed data and therefore the "
                    "identity across platforms (not tested here).",
        failure_modes_checked=["occurrence leaking into numerical identity", "replay after reopen", "replay of a "
                               "replay", "independent session", "sealed and unsealed data edits",
                               "out-of-bounds statistics"],
        unresolved_assumptions=["Cross-platform stability of float-valued numerical identities is not established.",
                                "Oscillator results have no numerical identity, so their replays cannot be linked."],
        recommended_next_task="T082 (fresh replay execution and result identities); CIW change: content-level "
                              "numerical_result_id for operation results, and recomputation of statistics on reopen")
    return {"state": _state(findings), "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T082 fresh occurrences
T082_PLAN = _fields(
    "Every execution, result, verification, bundle and replay occurrence receives a fresh identity, and a "
    "retained workspace that reuses an occurrence is refused.",
    "Occurrence identities are uuid4 draws or digests over records containing them; for n occurrences the "
    "probability of any uuid4 collision is about n^2 / 2^123.",
    "distinct count == occurrence count for every occurrence kind; reuse forgeries refused.",
    "Count identities over oscillator executions and every energy bundle (original, sibling, other log, replays, "
    "independent session); reopen four reuse forgeries and two creation-time forgeries.",
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
    rows, harness = _task_rows(ctx, "T082"), _mutations(ctx)["harness"]
    _retain_rows(ctx, rows, "freshness-mutations")
    findings = [
        finding("Every execution, result, verification, bundle, session and replay occurrence has a fresh identity",
                "provenance", table,
                {"checks": [_count(f"distinct {kind} identities", entry["distinct"], entry["occurrences"])
                            for kind, entry in table.items()]}, tolerance=EXACT),
        _kills(rows, "Reused or colliding occurrence identities are refused on reopen with the predicted message",
               harness),
        *_survivor_findings(rows, {"fresh.created-at-shift": "Surviving mutant fresh.created-at-shift: a backdated "
                                                             "execution and result pair reopens"}),
        _authentication(rows),
    ]
    fields = dict(T082_PLAN)
    fields.update(
        experiment=T082_PLAN["experiment"] + " Mutation rows retained as freshness-mutations.json/.md.",
        numerical_result=f"occurrence table {table}; mutants: {_summary(rows)}",
        uncertainty="uuid4 collisions among these identities have probability below 1e-33; freshness is checked "
                    "within one workspace, not globally across workspaces.",
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
    "All binding equalities hold for every retained receipt; a receipt can be removed without changing any "
    "digest.",
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
            "runtime_digest_binds_both": source is not None and verification["runtime_digest"]
            == _sha(native["runtimes"]) == _sha(source["runtimes"]),
            "numerical_match_holds": receipt["numerical_match"] is True and source is not None
            and source["steps"][0]["numerical_result_id"] == native["steps"][0]["numerical_result_id"],
            "limited_authority": receipt["admission"] == "not_performed" and verification["independent"] is False
            and verification["method"] == METHOD,
            "replay_id_recomputes": receipt["replay_id"] == _sha(_without(receipt, "replay_id")),
            "verification_id_recomputes": verification["verification_id"] == "sha256:" + hashlib.sha256(
                VERIFY_SCHEMA.encode() + b"\0" + _canon(_without(verification, "verification_id"))).hexdigest(),
        }
        properties.append({"replay": role, "properties": held})
    total = sum(len(entry["properties"]) for entry in properties)
    held_count = sum(sum(entry["properties"].values()) for entry in properties)
    rows, harness = _task_rows(ctx, "T083"), _mutations(ctx)["harness"]
    _retain_rows(ctx, rows, "receipt-mutations")
    ctx.artifact_json("receipt-binding.json", properties)
    findings = [
        finding("Every retained replay receipt binds its source and replayed bundle digests, verification subject, "
                "fresh reproduction step, runtime digest and limited authority", "provenance",
                {"receipts": len(properties), "binding_properties": total, "held": held_count},
                {"checks": [_exact(f"{entry['replay']} receipt: binding properties that did not hold",
                                   len(entry["properties"]) - sum(entry["properties"].values())) for entry in properties]},
                tolerance=EXACT),
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
    "T084": {"subject": "Replay receipt source digest", "stem": "receipt-source-mutations",
             "survivors": {"receipt-source.sibling-execution": "Surviving mutant receipt-source.sibling-execution: a "
                           "receipt re-pointed at a sibling execution of the same bytes reopens"},
             "next": "T085 (mutate replay receipt replayed digest)"},
    "T085": {"subject": "Replay receipt replayed digest", "stem": "receipt-replayed-mutations",
             "survivors": {"receipt-replayed.reidentified-bundle": "Surviving mutant receipt-replayed.reidentified-"
                           "bundle: a backdated, re-sessioned replay bundle with recomputed digests reopens"},
             "next": "T086 (mutate verification subject)"},
    "T086": {"subject": "Verification subject", "stem": "verification-subject-mutations",
             "survivors": {"receipt-subject.rebound-with-source": "Surviving mutant receipt-subject.rebound-with-"
                           "source: subject and source moved together to a sibling execution reopen"},
             "next": "T087 (mutate verification method)"},
    "T087": {"subject": "Verification method", "stem": "verification-method-mutations",
             "survivors": {"oscillator-method.injected": "Surviving mutant oscillator-method.injected: a sealed "
                           "result carrying an injected verification_method reopens"},
             "next": "T088 (mutate independence flag)"},
    "T088": {"subject": "Independence flag", "stem": "independence-mutations",
             "survivors": {"oscillator-independent.injected": "Surviving mutant oscillator-independent.injected: "
                           "sealed records carrying an injected independent: true reopen",
                           "exchange.verification-independent": "Surviving mutant exchange.verification-independent: "
                           "a recomputed exchange identity accepts a forged independence claim"},
             "next": "T089 (mutate admission status)"},
    "T089": {"subject": "Admission status", "stem": "admission-mutations",
             "survivors": {"oscillator-admission.injected": "Surviving mutant oscillator-admission.injected: a "
                           "sealed result carrying an injected state_admission reopens"},
             "next": "T090 (mutate provider runtime identity)"},
    "T090": {"subject": "Provider runtime identity", "stem": "runtime-mutations",
             "survivors": {"oscillator-runtime.both": "Surviving mutant oscillator-runtime.both: a provider runtime "
                           "forged identically in execution and result reopens",
                           "energy-runtime.all-bundles": "Surviving mutant energy-runtime.all-bundles: a consistently "
                           "forged analysis code digest reopens",
                           "energy-runtime.python-version": "Surviving mutant energy-runtime.python-version: forged "
                           "dependency versions reopen"},
             "next": "T091 (save/reopen without provider access); CIW change: keyed signatures over workspace "
                     "records and re-derivation of the energy runtime identity on reopen"},
}


def _mutation_plan(task_id: str) -> dict:
    spec = MUTATION_TASKS[task_id]
    return _fields(
        f"{spec['subject']} forgeries that leave an unkeyed digest stale, or that contradict a binding CIW recomputes "
        "on reopen, are refused; forgeries that recompute every unkeyed digest consistently can survive because no "
        "record is authenticated.",
        "Each record seal or identity d = sha256(canon(record \\ d)) is unkeyed, so any holder can recompute d "
        "after an edit. Reopen kills a mutant only if some check compares the edited field with an independently "
        "recomputed or cross-referenced value.",
        "Every predicted refusal is observed with its exact message; every predicted survivor is accepted; an "
        "unedited workspace reopens and the forger's recomputation reproduces CIW's digests exactly.",
        f"Edit the saved workspace ({spec['subject'].lower()}), recompute none, local or all unkeyed digests, write it "
        "and reopen with Session.from_workspace; pure validators are run on synthetic ESM and exchange records.",
        "not run", "not run", [], [], spec["next"])


def _mutation_task(ctx, task_id: str) -> dict:
    spec = MUTATION_TASKS[task_id]
    plan = _mutation_plan(task_id)
    if not fixture_available():
        return _blocked(plan)
    matrix = _mutations(ctx)
    rows = [row for row in matrix["rows"] if row["task"] == task_id]
    _retain_rows(ctx, rows, spec["stem"])
    findings = [_kills(rows, f"{spec['subject']} forgeries that leave a digest stale or contradict a recomputed "
                             "binding are refused on reopen with the predicted message", matrix["harness"])]
    findings += _survivor_findings(rows, spec["survivors"])
    if task_id == "T089":
        findings.append(finding(
            "A retained replay receipt or verification authorizes admission of the replayed result into canonical "
            "state", "production_acceptance", {"receipt_admission": "not_performed"},
            {"notes": "Admission is an authority decision outside the workbench; receipts record admission "
                      "not_performed and CIW refuses any other value."}, tolerance=EXACT))
    if task_id == "T090":
        witness = _row(rows, "energy-runtime.all-bundles").get("post_reopen", {})
        replay = witness.get("replay_after_reopen", {})
        findings.append(finding(
            "A consistently forged energy runtime identity is detected only when a replay recomputes the current "
            "analysis identity", "provenance", {"replay_after_reopen": replay},
            {"checks": [_refusal("bundle.replay on the reopened forged workspace", BINDING,
                                 replay.get("message") or str(replay.get("outcome")))]}, tolerance=EXACT))
        ctx.artifact_json("mutation-matrix.json", {"schema": "ciw.lab-mutation-matrix.v1", "harness": matrix["harness"],
                                                   "rows": matrix["rows"]})
        ctx.artifact_text("mutation-matrix.md", _matrix_markdown(matrix["rows"]))
    findings.append(_authentication(rows))
    fields = dict(plan)
    fields.update(
        experiment=plan["experiment"] + f" Rows retained as {spec['stem']}.json/.md"
                   + (" and the full matrix as mutation-matrix.json/.md." if task_id == "T090" else "."),
        numerical_result=_summary(rows),
        uncertainty="Outcomes are exact (accepted or a refusal message). Survivors are demonstrated forgeries; kills "
                    "show only that these particular edits are detected, not that every edit of this field is.",
        failure_modes_checked=[f"{row['name']}: {row['description']}" for row in rows],
        unresolved_assumptions=[
            "The forger model recomputes only unkeyed SHA-256 digests with CIW's public canonicalization; no key "
            "exists to steal.",
            "Pure-validator rows exercise candidate_evidence.validate_response and exchange._identity without an ESM "
            "process or the pinned exchange validator.",
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
