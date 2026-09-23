"""Read-only authority gate: evaluates proposals against a policy and records the decision.

The gate never dispatches anything. Its output is an ``authority_decision``
ledger entry listing each requirement, whether it was satisfied and why. Four
execution modes bound what can be authorized at all:

``explore``  compute and report       ``observe``  request acquisitions
``prepare``  build candidate artifacts ``operate``  act on equipment

Equipment actions (actuate, write parameters, deploy a bitstream, send a
control command) additionally require a bound control path. This build binds
none, so they are always refused with ``control_path_unbound`` even when every
other requirement holds. Operator approvals are Ed25519 signatures over the
proposal identity from approvers named in the policy; declared rollback,
compatibility, watchdog and emergency-disable facilities are recorded as
*declared*, which is never sufficient for actuation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ._common import Refusal, canonical_json, content_identity, finite, iso_time, mapping, require_keys, text, utc_now
from . import ed25519
from .claims import status_statements
from .ledger import Ledger
from .vocabulary import EXECUTION_MODES

POLICY_SCHEMA = "ciw.authority-policy.v1"
ACTION_MODES = {"report": "explore", "compute": "explore", "acquisition_request": "observe",
                "prepare_artifact": "prepare", "actuate": "operate", "write_parameter": "operate",
                "deploy_bitstream": "operate", "control_command": "operate"}
EQUIPMENT_ACTIONS = frozenset({"actuate", "write_parameter", "deploy_bitstream", "control_command"})
SAFETY_FACILITIES = ("rollback_plan", "compatibility_check", "watchdog", "emergency_disable")


def validate_policy(policy: Any) -> dict:
    policy = require_keys(mapping(policy, "policy"), "policy", {"schema", "policy_id", "mode", "requirements"},
                          {"approvers", "description"})
    if policy["schema"] != POLICY_SCHEMA:
        raise Refusal("unsupported_schema", f"Expected {POLICY_SCHEMA}")
    text(policy["policy_id"], "policy_id", 128)
    if policy["mode"] not in EXECUTION_MODES:
        raise Refusal("unknown_mode", f"Mode must be one of {list(EXECUTION_MODES)}")
    requirements = require_keys(policy["requirements"], "requirements", set(), {
        "verified_experiments", "max_evidence_age_s", "operator_approval", "admitted_state", *SAFETY_FACILITIES})
    if "max_evidence_age_s" in requirements:
        finite(requirements["max_evidence_age_s"], "max_evidence_age_s", minimum=0.0, exclusive_minimum=True)
    for approver in policy.get("approvers", []):
        require_keys(approver, "approver", {"approver_id", "public_key"})
        if len(bytes.fromhex(approver["public_key"])) != 32:
            raise Refusal("malformed_record", "Approver public keys are 32-byte Ed25519 keys in hex")
    return policy


def propose(ledger: Ledger, action: dict, proposer: str, refs: list[str], *, experiment_id: str | None = None) -> dict:
    action = require_keys(mapping(action, "action"), "action", {"kind", "target"}, {"parameters", *SAFETY_FACILITIES})
    if action["kind"] not in ACTION_MODES:
        raise Refusal("unknown_action", f"Action kind {action['kind']!r} is not declared", allowed=sorted(ACTION_MODES))
    return ledger.append("ciw.science.decision-proposal.v1",
                         {"action": action, "proposer": text(proposer, "proposer", 128), "experiment_id": experiment_id},
                         refs=refs)


def approval_message(proposal_id: str) -> bytes:
    return canonical_json({"proposal": proposal_id, "decision": "approve"}).encode("utf-8")


def approve(secret: bytes, approver_id: str, proposal_id: str) -> dict:
    return {"approver_id": approver_id, "signature": ed25519.sign(secret, approval_message(proposal_id)).hex()}


def evaluate(ledger: Ledger, proposal_id: str, policy: dict, *, now: datetime | None = None,
             approvals: list[dict] | None = None) -> dict:
    """Evaluate one proposal and retain the decision; nothing is executed."""
    policy = validate_policy(policy)
    proposal = ledger.get(proposal_id)
    if proposal["kind"] != "decision_proposal":
        raise Refusal("not_a_proposal", "Only decision proposals can be evaluated")
    now = now or utc_now()
    action = proposal["body"]["action"]
    requirements = policy["requirements"]
    checks: list[dict] = []

    def check(name: str, satisfied: bool, detail: Any) -> None:
        checks.append({"requirement": name, "satisfied": bool(satisfied), "detail": detail})

    needed = ACTION_MODES[action["kind"]]
    check("mode", EXECUTION_MODES.index(policy["mode"]) >= EXECUTION_MODES.index(needed),
          f"action {action['kind']} needs mode {needed}; policy grants {policy['mode']}")
    for experiment_id in requirements.get("verified_experiments", []):
        statement = status_statements(ledger, experiment_id)["numerical_stability"]
        check(f"verified:{experiment_id}", statement.startswith("the result is numerically stable"), statement)
    if "max_evidence_age_s" in requirements:
        ages = [(now - iso_time(ledger.get(ref)["recorded_at"], "recorded_at")).total_seconds()
                for ref in proposal["refs"]]
        oldest = max(ages) if ages else None
        check("fresh_state", oldest is not None and oldest <= requirements["max_evidence_age_s"],
              {"oldest_evidence_age_s": oldest, "limit_s": requirements["max_evidence_age_s"]})
    if requirements.get("admitted_state"):
        admitted = [ref for ref in proposal["refs"] if ledger.get(ref)["kind"] == "fusion_state"
                    and ledger.get(ref)["body"].get("stage") == "admitted_state"]
        check("admitted_state", bool(admitted), {"admitted_states_cited": len(admitted)})
    if requirements.get("operator_approval"):
        keys = {item["approver_id"]: bytes.fromhex(item["public_key"]) for item in policy.get("approvers", [])}
        valid = []
        for item in approvals or []:
            key = keys.get(item.get("approver_id"))
            try:
                signature = bytes.fromhex(item.get("signature", ""))
            except ValueError:
                signature = b""
            if key is not None and ed25519.verify(key, approval_message(proposal_id), signature):
                valid.append(item["approver_id"])
        check("operator_approval", bool(valid), {"valid_approvals": sorted(set(valid)),
                                                 "offered": len(approvals or [])})
    for facility in SAFETY_FACILITIES:
        if requirements.get(facility):
            declared = bool(action.get(facility))
            check(facility, declared and action["kind"] not in EQUIPMENT_ACTIONS,
                  "declared by the proposer; this read-only gate cannot verify it" if declared else "not declared")
    if action["kind"] in EQUIPMENT_ACTIONS:
        check("control_path", False, "control_path_unbound: this build binds no equipment control path")
    authorized = all(item["satisfied"] for item in checks)
    body = {"proposal": proposal_id, "authorized": authorized, "reasons": checks, "policy_id": policy["policy_id"],
            "policy_identity": content_identity(policy), "mode": policy["mode"], "action_kind": action["kind"],
            "experiment_id": proposal["body"].get("experiment_id"), "read_only": True, "dispatch": "none",
            "evaluated_at": now.isoformat()}
    return ledger.append("ciw.science.authority-decision.v1", body, refs=[proposal_id])
