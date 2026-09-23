"""Claims and the status statements a reviewer can rely on.

A claim is admitted only when the ledger holds the evidence its class needs:

==============  ==============================================================
measured        at least one observation with ``acquisition == "physical"``
estimated       a fusion candidate/admitted state, or physical observations plus a model result
predicted       a numerical result
computed        a completed execution
verified        verification records, every one of them ``passed``
interpretation  an explicit status; ``consistent``/``contradicted`` need physical observations
authorized      never created here; only the authority gate records authorization
==============  ==============================================================

``status_statements`` derives five separate statements for one experiment from
the ledger alone: whether a measurement exists, whether the calculation
completed, whether the result is numerically stable, whether the physical
interpretation is resolved, and whether any decision is authorized. They are
computed, never asserted, and never collapsed into one verdict.
"""
from __future__ import annotations

from typing import Any

from ._common import Refusal, text
from .ledger import Ledger
from .vocabulary import CLAIM_CLASSES

INTERPRETATION_STATUS = frozenset({"unresolved", "consistent", "contradicted"})


def _kinds(ledger: Ledger, refs: list[str]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for ref in refs:
        entry = ledger.get(ref)
        grouped.setdefault(entry["kind"], []).append(entry)
    return grouped


def _physical(entries: list[dict]) -> list[dict]:
    return [entry for entry in entries if entry["body"].get("acquisition") == "physical"]


def check_claim(ledger: Ledger, claim_class: str, refs: list[str], status: str | None = None) -> None:
    if claim_class not in CLAIM_CLASSES:
        raise Refusal("unknown_claim_class", f"Claim class {claim_class!r} is not declared")
    if claim_class == "authorized":
        raise Refusal("claim_unsupported", "Authorization is recorded only by the authority gate")
    if not refs:
        raise Refusal("claim_unsupported", "A claim must cite ledger evidence")
    grouped = _kinds(ledger, refs)
    observations = grouped.get("observation", [])
    if claim_class == "measured" and not _physical(observations):
        raise Refusal("claim_unsupported", "A measured claim needs a physical observation; synthetic or planned "
                      "observations never support it", cited=sorted(grouped))
    if claim_class == "estimated":
        fused = [entry for entry in grouped.get("fusion_state", []) if entry["body"].get("stage") in
                 {"candidate_state", "admitted_state"}]
        if not fused and not (_physical(observations) and grouped.get("numerical_result")):
            raise Refusal("claim_unsupported", "An estimate needs a fusion state or physical observations with a model result")
    if claim_class == "predicted" and not grouped.get("numerical_result"):
        raise Refusal("claim_unsupported", "A prediction must cite a numerical result")
    if claim_class == "computed":
        executions = grouped.get("execution", [])
        if not executions or any(entry["body"]["status"] != "completed" for entry in executions):
            raise Refusal("claim_unsupported", "A computed claim must cite only completed executions")
    if claim_class == "verified":
        verifications = grouped.get("verification", [])
        if not verifications:
            raise Refusal("claim_unsupported", "A verified claim must cite verification records")
        failing = [entry["entry_id"] for entry in verifications if entry["body"]["status"] != "passed"]
        if failing:
            raise Refusal("claim_unsupported", "Every cited verification must have passed", failing=failing)
    if claim_class == "interpretation":
        if status not in INTERPRETATION_STATUS:
            raise Refusal("claim_unsupported", f"An interpretation needs a status in {sorted(INTERPRETATION_STATUS)}")
        if status != "unresolved" and not _physical(observations):
            raise Refusal("claim_unsupported", "Only physical observations can resolve a physical interpretation")
    elif status is not None:
        raise Refusal("claim_unsupported", "Only interpretations carry a resolution status")


def make_claim(ledger: Ledger, statement: str, claim_class: str, refs: list[str], *, experiment_id: str | None = None,
               scope: Any = None, status: str | None = None, detail: Any = None) -> dict:
    text(statement, "statement")
    check_claim(ledger, claim_class, refs, status)
    body = {"statement": statement, "claim_class": claim_class, "status": status or "asserted", "scope": scope,
            "experiment_id": experiment_id, "detail": detail}
    return ledger.append("ciw.science.claim.v1", body, refs=refs)


def _for(ledger: Ledger, kind: str, experiment_id: str) -> list[dict]:
    return [entry for entry in ledger.entries(kind) if entry["body"].get("experiment_id") == experiment_id]


def status_statements(ledger: Ledger, experiment_id: str) -> dict:
    """Five independent statements about one experiment, derived from retained evidence."""
    observations = _for(ledger, "observation", experiment_id)
    physical = _physical(observations)
    executions = _for(ledger, "execution", experiment_id)
    completed = [entry for entry in executions if entry["body"]["status"] == "completed"]
    verifications = _for(ledger, "verification", experiment_id)
    passed = [entry for entry in verifications if entry["body"]["status"] == "passed"]
    receipts = _for(ledger, "replay_receipt", experiment_id)
    diverged = [entry for entry in receipts if entry["body"]["status"] in {"diverged", "refused"}]
    interpretations = [entry for entry in _for(ledger, "claim", experiment_id)
                       if entry["body"]["claim_class"] == "interpretation"]
    resolved = [entry for entry in interpretations if entry["body"]["status"] in {"consistent", "contradicted"}]
    proposals = _for(ledger, "decision_proposal", experiment_id)
    decisions = [entry for entry in ledger.entries("authority_decision")
                 if entry["body"].get("experiment_id") == experiment_id]
    authorized = [entry for entry in decisions if entry["body"]["authorized"]]

    measurement = ("the measurement exists" if physical else
                   f"no physical measurement exists ({len(observations)} synthetic or planned observations retained)")
    calculation = ("the calculation completed" if executions and len(completed) == len(executions) else
                   "no calculation was executed" if not executions else
                   f"{len(executions) - len(completed)} of {len(executions)} executions were refused")
    if verifications and len(passed) == len(verifications) and not diverged:
        stability = ("the result is numerically stable within the declared tolerances"
                     + (f" and reproduced by {len(receipts)} replays" if receipts else " (not yet replayed)"))
    elif not verifications:
        stability = "numerical stability is not established: no verification was run"
    else:
        stability = (f"numerical stability is not established: {len(verifications) - len(passed)} verifications did not "
                     f"pass and {len(diverged)} replays diverged or were refused")
    if resolved:
        latest = resolved[-1]["body"]["status"]
        interpretation = f"the physical interpretation is {latest} with physical evidence"
    else:
        interpretation = "the physical interpretation is unresolved"
    if authorized:
        decision = f"{len(authorized)} bounded actions are authorized by the authority gate"
    else:
        decision = ("the decision is not authorized" +
                    (f" ({len(decisions)} gate evaluations refused {len(proposals)} proposals)" if decisions else ""))
    return {"experiment_id": experiment_id, "measurement": measurement, "calculation": calculation,
            "numerical_stability": stability, "physical_interpretation": interpretation, "decision": decision,
            "counts": {"observations": len(observations), "physical_observations": len(physical),
                       "executions": len(executions), "completed": len(completed),
                       "verifications": len(verifications), "passed": len(passed), "replays": len(receipts),
                       "diverged_replays": len(diverged), "authority_decisions": len(decisions)}}
