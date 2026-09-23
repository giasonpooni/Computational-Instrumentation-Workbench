"""Scientific governance: signed, weighted council decisions over notation artefacts.

Scope: a council of members with Ed25519 keys and positive weights votes on
proposals that adopt, deprecate, pin or unpin governed artefacts (schema
versions, reference datasets, provider pins, validator releases, calibration
standards, experiment protocols, migration rules) or open and resolve
disputes. ``tally`` counts only well-formed, correctly signed, single votes of
known members; ``enact`` retains an approved, quorate proposal in the ledger;
``current_pins`` re-verifies every governance entry and folds them in ledger
order into the current pins, deprecations and dispute states.

Honest limits: governance decides which artefacts the community uses, never
whether a result is true. Verification outcomes, invariant results, numerical
results, observations and claim truth are refused as subjects, and a dispute
can only schedule evidence gathering (replay, validator rerun, new acquisition)
or cite existing verification or replay-receipt entries. Council membership and
keys are taken as given (no key rotation, revocation or membership votes), votes
carry no timestamps or deadlines, and the pure-Python Ed25519 is not constant
time. Equivocating members (two votes) are not counted at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re
from typing import Any

from . import ed25519
from ._common import Refusal, canonical_json, content_identity, finite, iso_time, mapping, require_keys, text

BODY = "ciw.science.governance.v1"
GOVERNED = frozenset({"schema_version", "reference_dataset", "provider_pin", "validator_release",
                      "calibration_standard", "experiment_protocol", "migration_rule", "dispute"})
FORBIDDEN = frozenset({"verification_outcome", "invariant_result", "numerical_result", "observation", "claim_truth"})
TRUTH_FIELDS = frozenset({"truth", "claim_truth", "verdict", "outcome", "claim_status", "status"})
ACTIONS = frozenset({"adopt", "deprecate", "pin", "unpin", "open_dispute", "resolve_dispute"})
DISPUTE_ACTIONS = frozenset({"open_dispute", "resolve_dispute"})
EVIDENCE_REQUESTS = frozenset({"replay", "rerun_validators", "new_acquisition"})
EVIDENCE_KINDS = frozenset({"verification", "replay_receipt"})
VOTES = frozenset({"approve", "reject", "abstain"})
IDENTITY = re.compile(r"sha256:[0-9a-f]{64}")
MAX_MEMBERS, MAX_VOTES, MAX_REFS = 256, 1024, 64

__all__ = ["Member", "Council", "validate_proposal", "proposal_identity", "sign_vote", "tally", "enact",
           "current_pins"]


def _hex(value: Any, name: str, size: int) -> bytes:
    if not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{2 * size}}}", value) is None:
        raise Refusal("malformed_key", f"{name} must be {size} bytes of lowercase hex")
    return bytes.fromhex(value)


def _identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or IDENTITY.fullmatch(value) is None:
        raise Refusal("malformed_identity", f"{name} must be sha256:<64 lowercase hex>")
    return value


@dataclass(frozen=True)
class Member:
    member_id: str
    public_key: bytes
    weight: float

    @classmethod
    def from_json(cls, value: Any, name: str = "member") -> "Member":
        if isinstance(value, Member):
            return value
        value = require_keys(value, name, {"member_id", "public_key", "weight"})
        return cls(text(value["member_id"], f"{name}.member_id", 128), _hex(value["public_key"], f"{name}.public_key", 32),
                   finite(value["weight"], f"{name}.weight", minimum=0.0, maximum=1e6, exclusive_minimum=True))

    def to_json(self) -> dict:
        return {"member_id": self.member_id, "public_key": self.public_key.hex(), "weight": self.weight}


class Council:
    """Weighted members plus the quorum and approval fractions that bind a decision."""

    def __init__(self, members: Any, quorum_fraction: float, approval_fraction: float):
        if not isinstance(members, (list, tuple)) or not members or len(members) > MAX_MEMBERS:
            raise Refusal("malformed_record", f"A council needs 1 to {MAX_MEMBERS} members")
        parsed = [Member.from_json(item, f"members[{index}]") for index, item in enumerate(members)]
        if len({member.member_id for member in parsed}) != len(parsed):
            raise Refusal("duplicate_member", "Member ids must be unique")
        if len({member.public_key for member in parsed}) != len(parsed):
            raise Refusal("duplicate_key", "One key cannot hold two seats")
        self.members = {member.member_id: member for member in sorted(parsed, key=lambda item: item.member_id)}
        self.quorum_fraction = finite(quorum_fraction, "quorum_fraction", minimum=0.0, maximum=1.0,
                                      exclusive_minimum=True)
        self.approval_fraction = finite(approval_fraction, "approval_fraction", minimum=0.5, maximum=1.0)

    @classmethod
    def from_json(cls, value: Any) -> "Council":
        value = require_keys(value, "council", {"members", "quorum_fraction", "approval_fraction"})
        return cls(value["members"], value["quorum_fraction"], value["approval_fraction"])

    def to_json(self) -> dict:
        return {"members": [member.to_json() for member in self.members.values()],
                "quorum_fraction": self.quorum_fraction, "approval_fraction": self.approval_fraction}

    @property
    def identity(self) -> str:
        return content_identity(self.to_json())

    @property
    def total_weight(self) -> Fraction:
        return sum((Fraction(member.weight) for member in self.members.values()), Fraction(0))


# ---------------------------------------------------------------------- proposals
def validate_proposal(proposal: Any, ledger: Any = None) -> dict:
    """Validate a proposal; with a ledger, evidence and disputed entries must exist with admissible kinds."""
    mapping(proposal, "proposal")
    kind = proposal.get("subject_kind")
    if isinstance(kind, str) and kind in FORBIDDEN:
        raise Refusal("governance_cannot_decide_evidence", f"{kind!r} is decided by evidence and validators, "
                      "not by vote", subject_kind=kind)
    truth = TRUTH_FIELDS & (set(proposal) | (set(proposal["subject"]) if isinstance(proposal.get("subject"), dict)
                                             else set()))
    if truth:
        raise Refusal("governance_cannot_decide_evidence", "A proposal cannot set a claim's truth or a verdict",
                      fields=sorted(truth))
    proposal = require_keys(proposal, "proposal", {"proposal_id", "subject_kind", "subject", "action", "rationale",
                                                   "created_at"}, {"evidence_request", "evidence_refs"})
    if not isinstance(kind, str) or kind not in GOVERNED:
        raise Refusal("unknown_subject_kind", f"Subject kind {kind!r} is not governed", allowed=sorted(GOVERNED))
    action = proposal["action"]
    if not isinstance(action, str) or action not in ACTIONS:
        raise Refusal("unknown_action", f"Action {action!r} is not declared", allowed=sorted(ACTIONS))
    if (kind == "dispute") != (action in DISPUTE_ACTIONS):
        raise Refusal("action_not_applicable", f"{action!r} does not apply to a {kind!r} subject")
    subject = require_keys(proposal["subject"], "proposal.subject", {"name"},
                           {"digest", "version", "entry", "description"})
    normalized_subject = {"name": text(subject["name"], "subject.name", 256)}
    if "digest" in subject:
        normalized_subject["digest"] = _identity(subject["digest"], "subject.digest")
    if "version" in subject:
        normalized_subject["version"] = text(subject["version"], "subject.version", 64)
    if "entry" in subject:
        normalized_subject["entry"] = _identity(subject["entry"], "subject.entry")
        if ledger is not None:
            ledger.get(normalized_subject["entry"])
    if "description" in subject:
        normalized_subject["description"] = text(subject["description"], "subject.description", 1024)
    if action in {"adopt", "pin"} and "digest" not in subject:
        raise Refusal("malformed_record", f"{action} must name the exact digest it binds")
    iso_time(proposal["created_at"], "proposal.created_at")
    result = {"proposal_id": text(proposal["proposal_id"], "proposal_id", 256), "subject_kind": kind,
              "subject": normalized_subject, "action": action, "rationale": text(proposal["rationale"], "rationale"),
              "created_at": proposal["created_at"]}
    has_request, has_refs = "evidence_request" in proposal, "evidence_refs" in proposal
    if action != "resolve_dispute":
        if has_request or has_refs:
            raise Refusal("malformed_record", "Evidence fields belong only to resolve_dispute proposals")
        return result
    if not has_request and not has_refs:
        raise Refusal("dispute_requires_evidence", "A dispute resolves only by scheduling evidence gathering or "
                      "citing evidence outcomes")
    if has_request:
        request = proposal["evidence_request"]
        if not isinstance(request, str) or request not in EVIDENCE_REQUESTS:
            raise Refusal("unknown_evidence_request", f"Evidence request {request!r} is not declared",
                          allowed=sorted(EVIDENCE_REQUESTS))
        result["evidence_request"] = request
    if has_refs:
        refs = proposal["evidence_refs"]
        if not isinstance(refs, list) or not refs or len(refs) > MAX_REFS:
            raise Refusal("malformed_record", f"evidence_refs must list 1 to {MAX_REFS} ledger entry ids")
        refs = [_identity(ref, f"evidence_refs[{index}]") for index, ref in enumerate(refs)]
        if len(set(refs)) != len(refs):
            raise Refusal("malformed_record", "evidence_refs must not repeat an entry")
        if ledger is not None:
            for ref in refs:
                entry = ledger.get(ref)
                if entry["kind"] not in EVIDENCE_KINDS:
                    raise Refusal("inadmissible_evidence", f"{ref} is a {entry['kind']!r} entry; disputes cite only "
                                  f"{sorted(EVIDENCE_KINDS)}", entry=ref, kind=entry["kind"])
        result["evidence_refs"] = refs
    return result


def proposal_identity(proposal: Any) -> str:
    return content_identity(validate_proposal(proposal))


# ---------------------------------------------------------------------- votes
def _message(identity: str, member_id: str, vote: str) -> bytes:
    return canonical_json({"proposal_identity": identity, "member_id": member_id, "vote": vote}).encode("utf-8")


def sign_vote(secret: bytes, member_id: str, proposal: Any, vote: str) -> dict:
    """Sign ``{proposal_identity, member_id, vote}`` (canonical JSON) with a 32-byte Ed25519 secret."""
    if not isinstance(secret, (bytes, bytearray)) or len(secret) != 32:
        raise Refusal("malformed_key", "An Ed25519 secret key has 32 bytes")
    if not isinstance(vote, str) or vote not in VOTES:
        raise Refusal("malformed_vote", f"vote must be one of {sorted(VOTES)}")
    identity = proposal_identity(proposal)
    member_id = text(member_id, "member_id", 128)
    signature = ed25519.sign(bytes(secret), _message(identity, member_id, vote))
    return {"proposal_identity": identity, "member_id": member_id, "vote": vote, "signature": signature.hex()}


def tally(council: Council, proposal: Any, votes: Any) -> dict:
    """Count valid votes; unknown members, bad signatures and duplicate votes are listed, never counted.

    Quorum is participating weight / total weight (abstentions participate);
    approval is approve / (approve + reject). Comparisons are exact rationals.
    """
    if not isinstance(council, Council):
        raise Refusal("malformed_record", "tally needs a Council")
    identity = proposal_identity(proposal)
    if not isinstance(votes, list) or len(votes) > MAX_VOTES:
        raise Refusal("malformed_record", f"votes must be a list of at most {MAX_VOTES} items")
    problems, valid = [], []
    for index, vote in enumerate(votes):
        if not isinstance(vote, dict) or set(vote) != {"proposal_identity", "member_id", "vote", "signature"} or \
                not isinstance(vote["member_id"], str) or not isinstance(vote["vote"], str) or vote["vote"] not in VOTES:
            problems.append({"index": index, "problem": "malformed_vote"})
            continue
        base = {"index": index, "member_id": vote["member_id"]}
        if vote["proposal_identity"] != identity:
            problems.append({**base, "problem": "wrong_proposal"})
            continue
        member = council.members.get(vote["member_id"])
        if member is None:
            problems.append({**base, "problem": "unknown_member"})
            continue
        signature = vote["signature"]
        if not isinstance(signature, str) or re.fullmatch("[0-9a-f]{128}", signature) is None or not ed25519.verify(
                member.public_key, _message(identity, member.member_id, vote["vote"]), bytes.fromhex(signature)):
            problems.append({**base, "problem": "bad_signature"})
            continue
        valid.append((index, member, vote))
    seen: dict[str, int] = {}
    for _, member, _ in valid:
        seen[member.member_id] = seen.get(member.member_id, 0) + 1
    counted = []
    for index, member, vote in valid:
        if seen[member.member_id] > 1:
            problems.append({"index": index, "member_id": member.member_id, "problem": "duplicate_vote"})
        else:
            counted.append((index, member, vote))
    weights = {choice: sum((Fraction(member.weight) for _, member, vote in counted if vote["vote"] == choice),
                           Fraction(0)) for choice in sorted(VOTES)}
    total, participating = council.total_weight, sum(weights.values(), Fraction(0))
    decisive = weights["approve"] + weights["reject"]
    quorum_met = participating >= total * Fraction(council.quorum_fraction)
    approved = decisive > 0 and weights["approve"] >= decisive * Fraction(council.approval_fraction)
    return {"proposal_identity": identity, "council_identity": council.identity,
            "counted": sorted(({"index": index, "member_id": member.member_id, "vote": vote["vote"],
                                "weight": member.weight} for index, member, vote in counted),
                              key=lambda item: item["member_id"]),
            "problems": sorted(problems, key=lambda item: item["index"]),
            "total_weight": float(total), "participating_weight": float(participating),
            "approve_weight": float(weights["approve"]), "reject_weight": float(weights["reject"]),
            "abstain_weight": float(weights["abstain"]), "quorum": float(participating / total),
            "quorum_fraction": council.quorum_fraction, "quorum_met": quorum_met,
            "approval": float(weights["approve"] / decisive) if decisive > 0 else None,
            "approval_fraction": council.approval_fraction, "approved": approved,
            "passed": quorum_met and approved}


# ---------------------------------------------------------------------- ledger
def enact(ledger: Any, council: Council, proposal: Any, votes: Any) -> dict:
    """Append a governance entry only for a quorate, approved, not previously enacted proposal."""
    normalized = validate_proposal(proposal, ledger)
    identity = content_identity(normalized)
    if any(entry["body"].get("proposal_identity") == identity for entry in ledger.entries("governance")):
        raise Refusal("proposal_already_enacted", f"Proposal {identity} is already enacted in this ledger")
    if normalized["action"] == "resolve_dispute":
        state = current_pins(ledger, council)["disputes"].get(normalized["subject"]["name"])
        if state is None or state["status"] not in {"open", "evidence_scheduled"}:
            raise Refusal("dispute_not_open", f"No open dispute {normalized['subject']['name']!r} under this council")
    result = tally(council, normalized, votes)
    if not result["passed"]:
        raise Refusal("proposal_not_approved", "The proposal lacks quorum or approval", tally=result)
    retained = [votes[item["index"]] for item in result["counted"]]
    body = {"action": normalized["action"], "subject": normalized["subject"], "subject_kind": normalized["subject_kind"],
            "proposal": normalized, "proposal_identity": identity, "votes": retained, "tally": result,
            "council": council.to_json(), "council_identity": council.identity}
    return ledger.append(BODY, body, refs=normalized.get("evidence_refs", []))


def current_pins(ledger: Any, council: Council | None = None) -> dict:
    """Fold re-verified governance entries in ledger order into pins, deprecations and disputes.

    An entry counts only if its proposal validates, its identity and body
    match, and its retained votes still pass under its recorded council (and,
    when ``council`` is given, that council is this one). Others are listed
    under ``ignored`` with a reason.
    """
    pins: dict[str, dict] = {}
    deprecations: dict[str, dict] = {}
    disputes: dict[str, dict] = {}
    applied, ignored = [], []
    for entry in ledger.entries("governance"):
        body, entry_id = entry["body"], entry["entry_id"]
        try:
            recorded = Council.from_json(body.get("council"))
            if council is not None and recorded.identity != council.identity:
                ignored.append({"entry_id": entry_id, "reason": "council_mismatch"})
                continue
            proposal = validate_proposal(body.get("proposal"), ledger)
            if content_identity(proposal) != body.get("proposal_identity") or any(
                    body.get(key) != proposal[key] for key in ("action", "subject", "subject_kind")):
                ignored.append({"entry_id": entry_id, "reason": "body_mismatch"})
                continue
            if not tally(recorded, proposal, body.get("votes"))["passed"]:
                ignored.append({"entry_id": entry_id, "reason": "not_passed_on_reverification"})
                continue
        except Refusal as exc:
            ignored.append({"entry_id": entry_id, "reason": exc.code})
            continue
        kind, subject, action = proposal["subject_kind"], proposal["subject"], proposal["action"]
        name, digest = subject["name"], subject.get("digest")
        current, deprecated = pins.setdefault(kind, {}), deprecations.setdefault(kind, {})
        if action in {"adopt", "pin"}:
            current[name] = digest
            if name in deprecated and deprecated[name] in {digest, None}:
                del deprecated[name]
        elif action == "unpin":
            if name in current and digest in {None, current[name]}:
                del current[name]
            else:
                ignored.append({"entry_id": entry_id, "reason": "unpin_without_matching_pin"})
                continue
        elif action == "deprecate":
            deprecated[name] = digest
            if name in current and digest in {None, current[name]}:
                del current[name]
        elif action == "open_dispute":
            disputes[name] = {"status": "open", "opened_by": entry_id, **({"entry": subject["entry"]}
                                                                          if "entry" in subject else {})}
        else:
            disputes[name] = {**disputes.get(name, {}), "resolved_by": entry_id,
                              "status": "evidence_scheduled" if "evidence_request" in proposal
                              else "resolved_by_cited_evidence",
                              **{key: proposal[key] for key in ("evidence_request", "evidence_refs") if key in proposal}}
        applied.append(entry_id)
    return {"pins": {kind: dict(sorted(items.items())) for kind, items in sorted(pins.items()) if items},
            "deprecations": {kind: dict(sorted(items.items())) for kind, items in sorted(deprecations.items()) if items},
            "disputes": dict(sorted(disputes.items())), "applied": applied, "ignored": ignored}
