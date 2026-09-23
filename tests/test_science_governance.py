"""Governance: signed weighted votes, quorum/approval, enactment, pin folding and evidence-bound disputes."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib

import pytest

from ciw.science import ed25519
from ciw.science._common import Refusal
from ciw.science.governance import (FORBIDDEN, Council, current_pins, enact, proposal_identity, sign_vote, tally,
                                    validate_proposal)
from ciw.science.ledger import Ledger

SECRETS = {name: hashlib.sha256(f"governance-test-{name}".encode()).digest() for name in ("alice", "bob", "carol",
                                                                                           "mallory")}
WEIGHTS = {"alice": 2.0, "bob": 1.0, "carol": 1.0}


def digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def refused(code, function, *args, **kwargs):
    with pytest.raises(Refusal) as caught:
        function(*args, **kwargs)
    assert caught.value.code == code, caught.value.to_dict()
    return caught.value


@pytest.fixture
def council():
    return Council([{"member_id": name, "public_key": ed25519.public_key(SECRETS[name]).hex(), "weight": weight}
                    for name, weight in WEIGHTS.items()], quorum_fraction=0.5, approval_fraction=2 / 3)


@pytest.fixture
def ledger(tmp_path):
    return Ledger.create(tmp_path / "ledger", now=lambda: datetime(2026, 9, 1, tzinfo=timezone.utc))


def proposal(action="pin", kind="provider_pin", name="geodesic-solver", label="v1", **extra):
    subject = {"name": name}
    if label is not None:
        subject["digest"] = digest(label)
    return {"proposal_id": f"{action}-{kind}-{name}-{label}", "subject_kind": kind, "subject": subject,
            "action": action, "rationale": "reference solver passed the oracle suite",
            "created_at": "2026-09-01T09:00:00+00:00", **extra}


def votes(record, **choices):
    return [sign_vote(SECRETS[name], name, record, choice) for name, choice in choices.items()]


# ---------------------------------------------------------------------- proposals
@pytest.mark.parametrize("kind", sorted(FORBIDDEN))
def test_evidence_subjects_cannot_be_governed(kind):
    refused("governance_cannot_decide_evidence", validate_proposal, proposal(kind=kind))


def test_truth_cannot_be_set_through_a_subject_or_proposal():
    record = proposal()
    record["subject"]["truth"] = True
    refused("governance_cannot_decide_evidence", validate_proposal, record)
    record = proposal("open_dispute", "dispute", "claim-17", None)
    record["verdict"] = "claim holds"
    refused("governance_cannot_decide_evidence", validate_proposal, record)


@pytest.mark.parametrize("record, code", [
    (proposal(kind="favourite_colour"), "unknown_subject_kind"),
    (proposal(action="ratify"), "unknown_action"),
    (proposal(action="open_dispute"), "action_not_applicable"),
    (proposal("pin", "dispute", "claim-17"), "action_not_applicable"),
    (proposal(label=None), "malformed_record"),
    (proposal(evidence_request="replay"), "malformed_record"),
    ({**proposal(), "created_at": "2026-09-01T09:00:00"}, "clock_unspecified"),
    ({**proposal(), "subject": {"name": "x", "digest": "md5:abc"}}, "malformed_identity"),
])
def test_malformed_proposals_are_refused(record, code):
    refused(code, validate_proposal, record)


def test_proposal_identity_is_content_identity_of_the_normalized_proposal():
    assert proposal_identity(proposal()) == proposal_identity(copy.deepcopy(proposal()))
    assert proposal_identity(proposal()) != proposal_identity(proposal(label="v2"))
    assert validate_proposal(validate_proposal(proposal())) == validate_proposal(proposal())


def test_council_validation():
    key = ed25519.public_key(SECRETS["alice"]).hex()
    member = {"member_id": "alice", "public_key": key, "weight": 1}
    refused("duplicate_member", Council, [member, {**member, "public_key": ed25519.public_key(SECRETS["bob"]).hex()}],
            0.5, 0.5)
    refused("duplicate_key", Council, [member, {**member, "member_id": "bob"}], 0.5, 0.5)
    refused("out_of_domain", Council, [member], 0.5, 0.4)
    refused("out_of_domain", Council, [member], 0.0, 0.5)
    refused("out_of_domain", Council, [{**member, "weight": 0}], 0.5, 0.5)
    refused("malformed_key", Council, [{**member, "public_key": key[:-2]}], 0.5, 0.5)
    council = Council([member], 1.0, 1.0)
    assert Council.from_json(council.to_json()).identity == council.identity


# ---------------------------------------------------------------------- tally
def test_approval_and_quorum_are_weighted(council):
    record = proposal()
    result = tally(council, record, votes(record, alice="approve", bob="reject", carol="approve"))
    assert result["problems"] == [] and result["quorum"] == 1.0 and result["approval"] == 0.75
    assert result["quorum_met"] and result["approved"] and result["passed"]
    result = tally(council, record, votes(record, alice="approve", bob="reject", carol="reject"))
    assert result["approval"] == 0.5 and not result["approved"] and not result["passed"]
    result = tally(council, record, votes(record, alice="approve", carol="abstain"))
    assert result["quorum"] == 0.75 and result["approval"] == 1.0 and result["passed"]  # abstain joins quorum only
    result = tally(council, record, votes(record, bob="abstain", carol="abstain"))
    assert result["quorum_met"] and result["approval"] is None and not result["passed"]


def test_quorum_failure(council, ledger):
    record = proposal()
    result = tally(council, record, votes(record, carol="approve"))
    assert result["quorum"] == 0.25 and not result["quorum_met"] and result["approved"] and not result["passed"]
    error = refused("proposal_not_approved", enact, ledger, council, record, votes(record, carol="approve"))
    assert error.detail["tally"]["quorum_met"] is False and len(ledger) == 0


def test_forged_and_tampered_signatures_are_not_counted(council):
    record = proposal()
    forged = sign_vote(SECRETS["mallory"], "alice", record, "approve")  # mallory signs in alice's name
    tampered = sign_vote(SECRETS["bob"], "bob", record, "reject")
    tampered["vote"] = "approve"
    garbage = {**sign_vote(SECRETS["carol"], "carol", record, "approve"), "signature": "zz"}
    result = tally(council, record, [forged, tampered, garbage])
    assert [(item["member_id"], item["problem"]) for item in result["problems"]] == [
        ("alice", "bad_signature"), ("bob", "bad_signature"), ("carol", "bad_signature")]
    assert result["counted"] == [] and result["participating_weight"] == 0 and not result["passed"]


def test_duplicate_votes_are_listed_and_none_counted(council):
    record = proposal()
    ballot = votes(record, bob="approve", carol="approve") + [sign_vote(SECRETS["alice"], "alice", record, "approve"),
                                                              sign_vote(SECRETS["alice"], "alice", record, "reject")]
    result = tally(council, record, ballot)
    assert [(item["index"], item["problem"]) for item in result["problems"]] == [(2, "duplicate_vote"),
                                                                              (3, "duplicate_vote")]
    assert [item["member_id"] for item in result["counted"]] == ["bob", "carol"]
    assert result["quorum"] == 0.5 and result["passed"]


def test_unknown_members_wrong_proposals_and_malformed_votes(council):
    record = proposal()
    other = proposal(label="v2")
    ballot = [sign_vote(SECRETS["mallory"], "mallory", record, "approve"),
              sign_vote(SECRETS["alice"], "alice", other, "approve"),
              {"member_id": "bob", "vote": "approve"},
              {**sign_vote(SECRETS["carol"], "carol", record, "approve"), "vote": "maybe"}]
    result = tally(council, record, ballot)
    assert [item["problem"] for item in result["problems"]] == ["unknown_member", "wrong_proposal", "malformed_vote",
                                                                "malformed_vote"]
    assert result["counted"] == []


# ---------------------------------------------------------------------- enactment
def enact_all(ledger, council, record, **choices):
    return enact(ledger, council, record, votes(record, **choices))


def test_enact_and_fold_current_pins(council, ledger):
    majority = {"alice": "approve", "bob": "approve"}
    adopted = enact_all(ledger, council, proposal("adopt", "schema_version", "ciw.physical-protocol", "schema-v1"),
                        **majority)
    assert adopted["kind"] == "governance" and adopted["body"]["tally"]["passed"]
    assert [vote["member_id"] for vote in adopted["body"]["votes"]] == ["alice", "bob"]
    enact_all(ledger, council, proposal(label="v1"), **majority)
    enact_all(ledger, council, proposal(label="v2"), **majority)
    state = current_pins(ledger, council)
    assert state["pins"] == {"provider_pin": {"geodesic-solver": digest("v2")},
                             "schema_version": {"ciw.physical-protocol": digest("schema-v1")}}
    assert state["deprecations"] == {} and len(state["applied"]) == 3 and state["ignored"] == []
    enact_all(ledger, council, proposal("deprecate", "schema_version", "ciw.physical-protocol", "schema-v1"),
              **majority)
    enact_all(ledger, council, proposal("unpin", label="v1"), **majority)  # v1 is no longer the pin: no effect
    state = current_pins(ledger)
    assert state["pins"] == {"provider_pin": {"geodesic-solver": digest("v2")}}
    assert state["deprecations"] == {"schema_version": {"ciw.physical-protocol": digest("schema-v1")}}
    assert state["ignored"][-1]["reason"] == "unpin_without_matching_pin"
    enact_all(ledger, council, proposal("unpin", label=None), **majority)
    assert current_pins(ledger)["pins"] == {}
    refused("proposal_already_enacted", enact_all, ledger, council, proposal(label="v1"), **majority)
    assert Ledger.open(ledger.root).verify().ok


def test_folding_ignores_unverifiable_entries(council, ledger):
    record = validate_proposal(proposal(label="v1"))
    identity = proposal_identity(record)
    ledger.append("ciw.science.governance.v1", {  # appended directly, without a passing vote
        "action": "pin", "subject": record["subject"], "subject_kind": "provider_pin", "proposal": record,
        "proposal_identity": identity, "votes": [], "council": council.to_json()})
    forged = sign_vote(SECRETS["mallory"], "alice", record, "approve")
    ledger.append("ciw.science.governance.v1", {
        "action": "pin", "subject": record["subject"], "subject_kind": "provider_pin", "proposal": record,
        "proposal_identity": identity, "votes": [forged], "council": council.to_json()})
    ledger.append("ciw.science.governance.v1", {"action": "pin", "subject": {"name": "x"}})
    state = current_pins(ledger)
    assert state["pins"] == {} and [item["reason"] for item in state["ignored"]] == [
        "not_passed_on_reverification", "not_passed_on_reverification", "malformed_record"]
    rogue = Council([{"member_id": "mallory", "public_key": ed25519.public_key(SECRETS["mallory"]).hex(),
                      "weight": 1}], 1.0, 1.0)
    enact(ledger, rogue, proposal(label="rogue"), votes(proposal(label="rogue"), mallory="approve"))
    assert current_pins(ledger)["pins"] == {"provider_pin": {"geodesic-solver": digest("rogue")}}
    scoped = current_pins(ledger, council)
    assert scoped["pins"] == {} and scoped["ignored"][-1]["reason"] == "council_mismatch"


# ---------------------------------------------------------------------- disputes
def test_dispute_resolution_requires_admissible_evidence(council, ledger):
    majority = {"alice": "approve", "carol": "approve"}
    claim = ledger.append("ciw.science.claim.v1", {"statement": "geodesic length 1.2 m", "claim_class": "computed",
                                                   "status": "asserted"})
    observation = ledger.append("ciw.science.observation.v1", {"observable": "camera_chord", "acquisition": "physical",
                                                               "value": 1.0, "unit": "m"})
    verification = ledger.append("ciw.science.verification.v1", {"subject": claim["entry_id"], "verdicts": [],
                                                                 "status": "passed"})
    receipt = ledger.append("ciw.science.replay-receipt.v1", {"original": claim["entry_id"], "status": "reproduced",
                                                              "comparison": {}})
    dispute = {"name": "dispute-17", "entry": claim["entry_id"]}

    def resolution(**evidence):
        return {**proposal("resolve_dispute", "dispute", "dispute-17", None), "subject": dispute, **evidence,
                "proposal_id": "resolve-" + "-".join(sorted(evidence))}

    refused("dispute_requires_evidence", validate_proposal, resolution())
    refused("unknown_evidence_request", validate_proposal, resolution(evidence_request="majority_opinion"))
    refused("inadmissible_evidence", validate_proposal, resolution(evidence_refs=[observation["entry_id"]]), ledger)
    refused("unknown_entry", validate_proposal, resolution(evidence_refs=[digest("nowhere")]), ledger)
    refused("dispute_not_open", enact_all, ledger, council, resolution(evidence_request="replay"), **majority)

    opened = {**proposal("open_dispute", "dispute", "dispute-17", None), "subject": dispute}
    enact_all(ledger, council, opened, **majority)
    assert current_pins(ledger, council)["disputes"]["dispute-17"]["status"] == "open"
    enact_all(ledger, council, resolution(evidence_request="replay"), **majority)
    assert current_pins(ledger, council)["disputes"]["dispute-17"]["status"] == "evidence_scheduled"
    refs = [verification["entry_id"], receipt["entry_id"]]
    entry = enact_all(ledger, council, resolution(evidence_refs=refs), **majority)
    assert entry["refs"] == sorted(refs)
    state = current_pins(ledger, council)["disputes"]["dispute-17"]
    assert state["status"] == "resolved_by_cited_evidence" and state["evidence_refs"] == refs
    assert state["entry"] == claim["entry_id"] and "truth" not in state
    refused("dispute_not_open", enact_all, ledger, council, resolution(evidence_request="new_acquisition"), **majority)
