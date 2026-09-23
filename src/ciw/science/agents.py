"""Agent orchestration: typed, role-scoped proposals that never change state directly.

Every agent, human or model, acts through ``submit``: a role may propose only
the kinds of artifact its role declares, and the proposal is appended to the
ledger as ``proposed``. ``dispose`` runs the deterministic validator for that
kind and records ``accepted`` or ``rejected`` with reasons. Accepting a
proposal does not apply it: an accepted experiment specification still has to
be executed, and an accepted acquisition request is still only a request.
Kinds that would bypass evidence (editing the ledger, overriding a
verification, changing a claim's status, commanding hardware) are refused for
every role.

Three deterministic agents ship with the workbench and use the same channel:
``provenance_audit`` checks that every result is bound to runtime, parameters
and verification; ``adversarial_ledger_probe`` mutates a copy of a ledger and
confirms each mutation is detected; ``adversarial_parameter_search`` searches a
sweep for oracle failures and proposes counterexamples.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

import numpy as np

from ._common import Refusal, content_identity, mapping, plain, text
from .claims import check_claim
from .experiment import compile_spec, validate_spec
from .ledger import LEDGER_FILE, Ledger
from .oracles import evaluate
from .solvers import default_registry
from .vocabulary import AGENT_ROLES, OBSERVABLES

ROLE_KINDS: dict[str, frozenset[str]] = {
    "schematic-retrieval": frozenset({"evidence_reference"}),
    "provider-resolution": frozenset({"provider_binding"}),
    "geometry-reasoning": frozenset({"experiment_spec", "geometry_model"}),
    "experiment-design": frozenset({"experiment_spec", "design_recommendation"}),
    "numerical-method": frozenset({"solver_setting"}),
    "sensor-fusion": frozenset({"fusion_policy"}),
    "provenance-auditor": frozenset({"audit_finding"}),
    "adversarial-test": frozenset({"counterexample", "audit_finding"}),
    "documentation": frozenset({"report_draft"}),
    "hardware-interface": frozenset({"acquisition_request"}),
    "safety-reviewer": frozenset({"authority_review"}),
}
FORBIDDEN_KINDS = frozenset({"ledger_edit", "verification_override", "claim_status_change", "control_command",
                             "calibration_overwrite", "solver_registration"})


def submit(ledger: Ledger, role: str, proposal_kind: str, payload: Any, *, agent: str, rationale: str,
           evidence: list[str] | None = None) -> dict:
    if role not in AGENT_ROLES:
        raise Refusal("unknown_role", f"Agent role {role!r} is not declared", allowed=sorted(AGENT_ROLES))
    if proposal_kind in FORBIDDEN_KINDS:
        raise Refusal("proposal_forbidden", f"{proposal_kind} would bypass evidence; no role may propose it")
    if proposal_kind not in ROLE_KINDS[role]:
        raise Refusal("role_not_permitted", f"Role {role} may propose {sorted(ROLE_KINDS[role])}, not {proposal_kind}")
    body = {"role": role, "agent": text(agent, "agent", 128), "proposal_kind": proposal_kind, "payload": payload,
            "rationale": text(rationale, "rationale"), "status": "proposed",
            "payload_identity": content_identity(plain(payload))}
    return ledger.append("ciw.science.agent-proposal.v1", body, refs=evidence or [])


def _validate_spec(ledger: Ledger, payload: Any) -> list[str]:
    compile_spec(validate_spec(deepcopy(payload)))
    return ["specification validated and compiled; execution still requires an explicit run"]


def _validate_acquisition(ledger: Ledger, payload: Any) -> list[str]:
    payload = mapping(payload, "acquisition request")
    if payload.get("observable") not in OBSERVABLES:
        raise Refusal("unknown_observable", "An acquisition request must name a declared observable")
    text(payload.get("instrument"), "instrument", 128)
    if payload.get("command") is not None:
        raise Refusal("proposal_forbidden", "An acquisition request cannot carry an equipment command")
    return ["request is well formed; it is retained as a request and triggers no acquisition"]


def _validate_counterexample(ledger: Ledger, payload: Any) -> list[str]:
    payload = mapping(payload, "counterexample")
    job, oracles, tolerances = payload.get("job"), payload.get("oracles"), payload.get("tolerances")
    declaration = default_registry().get(mapping(job, "job")["solver_id"])
    try:
        result = declaration.implementation(job)
    except Refusal as exc:
        return [f"solver refuses the job on re-execution ({exc.code}); the refusal itself is the documented behaviour"]
    summary = evaluate(oracles, job, result, declaration.implementation, tolerances)
    if summary["status"] != "failed":
        raise Refusal("counterexample_not_reproduced", f"Re-execution gives verification status {summary['status']}")
    failed = [item["oracle_id"] for item in summary["verdicts"] if item["passed"] is False]
    return [f"re-execution reproduces failing oracles {failed}"]


def _validate_finding(ledger: Ledger, payload: Any) -> list[str]:
    payload = mapping(payload, "finding")
    for ref in payload.get("entries", []):
        ledger.get(ref)
    text(payload.get("finding"), "finding")
    return ["finding cites existing ledger entries"]


def _validate_claim_review(ledger: Ledger, payload: Any) -> list[str]:
    payload = mapping(payload, "authority review")
    check_claim(ledger, payload["claim_class"], payload.get("refs", []), payload.get("status"))
    return ["cited evidence admits the reviewed claim class"]


VALIDATORS: dict[str, Callable[[Ledger, Any], list[str]]] = {
    "experiment_spec": _validate_spec, "acquisition_request": _validate_acquisition,
    "counterexample": _validate_counterexample, "audit_finding": _validate_finding,
    "authority_review": _validate_claim_review,
}


def dispose(ledger: Ledger, proposal_id: str) -> dict:
    """Run the deterministic validator for a proposal and record the disposition."""
    proposal = ledger.get(proposal_id)
    if proposal["kind"] != "agent_proposal":
        raise Refusal("not_a_proposal", "Only agent proposals can be disposed")
    if ledger.referrers(proposal_id, "proposal_disposition"):
        raise Refusal("already_disposed", "This proposal already has a disposition")
    kind = proposal["body"]["proposal_kind"]
    validator = VALIDATORS.get(kind)
    if validator is None:
        disposition, reasons = "held", [f"no deterministic validator is bound for {kind}; a person must review it"]
    else:
        try:
            disposition, reasons = "accepted", validator(ledger, proposal["body"]["payload"])
        except Refusal as exc:
            disposition, reasons = "rejected", [f"{exc.code}: {exc}"]
        except (KeyError, TypeError, ValueError) as exc:
            disposition, reasons = "rejected", [f"malformed payload: {exc}"]
    return ledger.append("ciw.science.proposal-disposition.v1", {
        "proposal": proposal_id, "disposition": disposition, "reasons": reasons, "applied": False,
        "proposal_kind": kind}, refs=[proposal_id])


# ---------------------------------------------------------------- deterministic agents
def provenance_audit(ledger: Ledger) -> list[dict]:
    """Findings for results that lack runtime, parameter or verification bindings."""
    findings = []
    for result in ledger.entries("numerical_result"):
        chain = ledger.closure([result["entry_id"]])
        kinds = {entry["kind"] for entry in chain}
        missing = sorted({"execution", "parameter_identity", "runtime_identity"} - kinds)
        if missing:
            findings.append({"finding": f"result {result['body']['job_id']} is not bound to {missing}",
                             "entries": [result["entry_id"]], "severity": "high"})
        member = any(entry["kind"] == "parameter_identity" and entry["body"]["parameters"].get("ensemble", {}).get("role")
                     == "member" for entry in chain)
        if not member and not ledger.referrers(result["entry_id"], "verification"):
            findings.append({"finding": f"result {result['body']['job_id']} has no verification",
                             "entries": [result["entry_id"]], "severity": "medium"})
    for claim in ledger.entries("claim"):
        try:
            check_claim(ledger, claim["body"]["claim_class"], claim["refs"],
                        claim["body"]["status"] if claim["body"]["claim_class"] == "interpretation" else None)
        except Refusal as exc:
            findings.append({"finding": f"claim no longer admitted by its evidence: {exc}", "entries": [claim["entry_id"]],
                             "severity": "high"})
    return findings


def _mutations(lines: list[bytes], blob_paths: list[Path]) -> list[tuple[str, Callable[[Path], None]]]:
    def rewrite(transform):
        def apply(root: Path) -> None:
            path = root / LEDGER_FILE
            path.write_bytes(b"".join(transform(list(path.read_bytes().splitlines(keepends=True)))))
        return apply

    def flip(entries):
        target = len(entries) // 2
        line = bytearray(entries[target])
        position = line.find(b'"body":') + 12
        line[position] = ord("0") if line[position] != ord("0") else ord("1")
        entries[target] = bytes(line)
        return entries

    def forge(entries):
        forged = json.loads(entries[-1])
        forged["body"] = {**forged["body"], "status": "passed"}
        entries.append(json.dumps(forged, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        return entries

    cases = [("flip a byte in an entry body", rewrite(flip)),
             ("delete an entry", rewrite(lambda e: e[:len(e) // 2] + e[len(e) // 2 + 1:])),
             ("swap two entries", rewrite(lambda e: e[:1] + [e[2], e[1]] + e[3:])),
             ("truncate the final entry", rewrite(lambda e: e[:-1] + [e[-1][:-10]])),
             ("append a forged entry copying the head", rewrite(forge))]
    if blob_paths:
        def corrupt(root: Path, relative=blob_paths[0]) -> None:
            path = root / relative
            path.write_bytes(path.read_bytes() + b" ")
        cases.append(("append a byte to a retained blob", corrupt))
        cases.append(("plant an unexpected file among blobs",
                      lambda root: (root / "blobs" / "zz").mkdir(parents=True, exist_ok=True) or
                      (root / "blobs" / "zz" / "notes.txt").write_bytes(b"x")))
    return cases


def adversarial_ledger_probe(root: Path) -> list[dict]:
    """Apply tampering to copies of a ledger and report whether each was detected."""
    root = Path(root)
    Ledger.open(root)
    lines = (root / LEDGER_FILE).read_bytes().splitlines(keepends=True)
    if len(lines) < 4:
        raise Refusal("ledger_too_small", "The probe needs at least four entries")
    blob_paths = sorted(path.relative_to(root) for path in (root / "blobs").rglob("*") if path.is_file())
    results = []
    for description, mutate in _mutations(lines, blob_paths):
        with tempfile.TemporaryDirectory() as scratch:
            copy = Path(scratch) / "ledger"
            shutil.copytree(root, copy)
            mutate(copy)
            try:
                Ledger.open(copy)
                detected, code = False, None
            except Refusal as exc:
                detected, code = True, exc.code
        results.append({"mutation": description, "detected": detected, "refusal": code})
    return results


def adversarial_parameter_search(spec: dict, *, samples: int = 16, seed: int = 0) -> list[dict]:
    """Randomly perturb swept parameters within their declared ranges and look for oracle failures."""
    plan = compile_spec(spec)
    declaration = default_registry().get(plan["solver_id"])
    rng = np.random.default_rng(seed)
    candidates = [job for job in plan["jobs"] if job.get("ensemble", {}).get("role") != "member"]
    found = []
    for _ in range(samples):
        job = deepcopy(candidates[int(rng.integers(len(candidates)))])
        if job.get("heading_rad") is not None:
            job["heading_rad"] = float(job["heading_rad"] + rng.normal(0.0, 0.5))
        job["arclength"] = float(job["arclength"] * rng.uniform(0.5, 2.0))
        job["steps"] = int(max(4, job["steps"] * rng.uniform(0.05, 1.0)))
        try:
            result = declaration.implementation(job)
        except Refusal as exc:
            found.append({"job": job, "outcome": "refused", "refusal": exc.to_dict()})
            continue
        summary = evaluate(plan["validation"]["oracles"], job, result, declaration.implementation,
                           plan["validation"]["tolerances"])
        if summary["status"] == "failed":
            found.append({"job": job, "outcome": "failed",
                          "failed_oracles": [item["oracle_id"] for item in summary["verdicts"] if item["passed"] is False],
                          "oracles": plan["validation"]["oracles"], "tolerances": plan["validation"]["tolerances"]})
    return plain(found)
