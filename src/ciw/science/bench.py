"""Synthetic experiment bench: the whole define → inspect → simulate → measure → compare → retain → decide loop.

``run_bench`` builds one ledger from the shipped examples:

1. retains the declared inspection-cell frame registry and resolves a coupon-to-camera chain;
2. compiles and executes every experiment specification, then replays each one;
3. ranks candidate coupons for the chord-versus-curvature question and files the
   recommended experiment as an experiment-design agent proposal;
4. evaluates the physical protocol against its (synthetic) measurements;
5. compiles the industrial use cases against the capabilities actually registered;
6. runs the synthetic two-tracker fusion scenario to an admission decision;
7. parses the synthetic FPGA telemetry capture and asks for (and is refused) control;
8. routes a report proposal and an actuation proposal through the read-only authority gate;
9. runs the provenance auditor and the adversarial ledger probe;
10. pins the solver sources through a bench governance council;
11. writes Markdown and LaTeX reports and a signed evidence bundle.

All bench keys are derived from public strings and exist only to exercise the
signature paths. They must never approve anything real. Nothing here acquires
a physical measurement; every observation is labelled synthetic.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ._common import Refusal, canonical_json, content_identity, plain
from . import agents, authority, design, ed25519
from .backends import source_digest
from .claims import status_statements
from .frames import FrameRegistry
from .ledger import Ledger
from .report import build, latex, markdown, retain as retain_report
from .runner import execute, replay
from .solvers import default_registry

ROOT = Path(__file__).resolve().parents[3] / "examples" / "science"
QUICK = ("cylinder-chord.json", "cylinder-misdeclared.json", "cylinder-winding.json", "torus-invariants.json",
         "hyperbolic-jacobi.json", "cylinder-mesh-refinement.json")


def bench_key(name: str) -> bytes:
    """Deterministic, publicly derivable key material. Bench use only."""
    return hashlib.sha256(b"ciw-bench-only-key:" + name.encode()).digest()


def _json(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    return json.loads(raw), raw


def run_bench(output: Path, *, examples: Path | None = None, quick: bool = False) -> dict:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise Refusal("output_exists", f"{output} is not empty; the bench writes a new ledger")
    examples = Path(examples) if examples is not None else ROOT
    ledger = Ledger.create(output / "ledger")
    summary: dict[str, Any] = {"output": str(output)}

    # 1. frames
    frames_record, frames_raw = _json(examples / "frames" / "coupon-cell.json")
    registry = FrameRegistry.from_json(frames_record)
    frames_blob = ledger.put_blob(frames_raw)
    frames_entry = ledger.append("ciw.science.frame-registry.v1", {
        "registry": frames_record, "registry_identity": registry.identity(), "source_blob": frames_blob},
        blobs=[frames_blob])
    chain = registry.transform_point([0.05, 0.0, 0.02], "m", "coupon", "camera", 1000.0, "cell-ptp")
    try:
        registry.transform_point([0.05, 0.0, 0.02], "m", "coupon", "camera", 8000.0, "cell-ptp")
        stale = None
    except Refusal as exc:
        stale = exc.to_dict()
    summary["frames"] = {"coupon_point_in_camera": chain["point"], "chain": [link["transform_id"] for link in chain["chain"]],
                         "stale_query": stale["code"] if stale else None}

    # 2. experiments and replays
    experiments = {}
    for path in sorted((examples / "experiments").glob("*.json")):
        if quick and path.name not in QUICK:
            continue
        spec, raw = _json(path)
        result = execute(spec, ledger, source_bytes=raw, frame_registry_entry=frames_entry["entry_id"])
        receipt = replay(ledger, spec["experiment_id"])
        experiments[spec["experiment_id"]] = {"verifications": result["verifications"], "replay": receipt["counts"],
                                              "claims": [(c["claim_class"], c["admitted"]) for c in result["claims"]],
                                              "status": status_statements(ledger, spec["experiment_id"])}
    summary["experiments"] = experiments

    # 3. active design
    problem, _ = _json(examples / "design" / "chord-or-curvature.json")
    ranking = design.rank(problem)
    design_entry = design.retain(ledger, problem, ranking)
    proposal = agents.submit(ledger, "experiment-design", "experiment_spec", design.next_experiment(problem, ranking),
                             agent="ciw.design", rationale=ranking["summary"], evidence=[design_entry["entry_id"]])
    disposition = agents.dispose(ledger, proposal["entry_id"])
    summary["design"] = {"recommendation": ranking["recommendation"], "summary": ranking["summary"],
                         "proposal": disposition["body"]["disposition"]}

    # 4. physical protocol
    from .physical import evaluate as evaluate_protocol, retain_measurements, retain_protocol
    protocol, _ = _json(examples / "physical" / "cylinder-chord-protocol.json")
    measurements, _ = _json(examples / "physical" / "cylinder-chord-measurements.json")
    predictions, _ = _json(examples / "physical" / "cylinder-chord-predictions.json")
    evaluation = evaluate_protocol(protocol, measurements, predictions)
    protocol_entry = retain_protocol(ledger, protocol)
    retain_measurements(ledger, protocol_entry["entry_id"], measurements)
    summary["physical"] = {"physical_evidence": evaluation.get("physical_evidence"),
                           "status": evaluation.get("status") or evaluation.get("summary")}

    # 5. use cases
    from .usecase import compile_use_case, retain_requirements
    solvers = default_registry()
    available = {name for declaration in solvers.describe() for name, ok in declaration["capabilities"].items() if ok}
    use_cases = {}
    for path in sorted((examples / "usecases").glob("*.json")):
        record, _ = _json(path)
        compiled = compile_use_case(record, available)
        retain_requirements(ledger, record, compiled)
        use_cases[path.stem] = {"missing_capabilities": compiled.get("missing_capabilities")}
    summary["use_cases"] = use_cases

    # 6. fusion
    from .fusion import AdmissionPolicy, replay as fusion_replay, retain as retain_fusion, scenario_engine
    scenario, _ = _json(examples / "fusion" / "two-tracker-scenario.json")
    policy, _ = _json(examples / "fusion" / "admission-policy.json")
    engine = scenario_engine(scenario)
    records = fusion_replay(engine, scenario)
    candidate = engine.candidate()
    admission = engine.admit(candidate, AdmissionPolicy.from_json(policy), evaluated_at=candidate["time"]["value"])
    candidate_entry = retain_fusion(ledger, candidate)
    admission_entry = retain_fusion(ledger, admission, refs=[candidate_entry["entry_id"]])
    stages = {}
    for record in records:
        stages[record["stage"]] = stages.get(record["stage"], 0) + 1
    summary["fusion"] = {"records": stages, "admission": admission["stage"],
                         "refusal_reasons": [item["code"] for item in admission.get("reasons", [])],
                         "position_std": candidate["position_std"], "unit": candidate["unit"]}

    # 7. hardware boundary
    from .hardware import parse_stream, request_control, retain_capture
    reference, _ = _json(examples / "hardware" / "synthetic-capture.json")
    layouts, _ = _json(examples / "hardware" / reference["layouts"])
    capture_raw = (examples / "hardware" / reference["capture"]).read_bytes()
    capture = parse_stream(capture_raw, layouts, **reference["parse"])
    capture_entry = retain_capture(ledger, capture_raw, capture)
    control = request_control({"kind": "write_register", "register": "0x10", "value": 1},
                              {name: True for name in ("explicit_authority", "rollback_plan", "compatibility_check",
                                                       "watchdog", "fresh_state", "operator_approval", "emergency_disable")})
    summary["hardware"] = {"digest_reproduced": capture.digest == reference["retained_digest"],
                           "packets": capture.summary.get("packets_decoded"), "control": control.get("status", control.get("code")),
                           "control_reasons": control.get("reasons")}

    # 8. authority gate
    reviewer = bench_key("operator")
    policy_record = {"schema": authority.POLICY_SCHEMA, "policy_id": "bench-operate", "mode": "operate",
                     "approvers": [{"approver_id": "bench-operator", "public_key": ed25519.public_key(reviewer).hex()}],
                     "requirements": {"verified_experiments": sorted(experiments), "operator_approval": True,
                                      "rollback_plan": True, "watchdog": True, "emergency_disable": True,
                                      "max_evidence_age_s": 3600}}
    decisions = {}
    for kind, target, refs in (("report", "bench report", [design_entry["entry_id"]]),
                               ("actuate", "robot joint 3", [admission_entry["entry_id"], capture_entry["entry_id"]])):
        action = {"kind": kind, "target": target, "rollback_plan": True, "watchdog": True, "emergency_disable": True}
        entry = authority.propose(ledger, action, "ciw.bench", refs)
        approval = authority.approve(reviewer, "bench-operator", entry["entry_id"])
        decision = authority.evaluate(ledger, entry["entry_id"], policy_record, approvals=[approval])
        decisions[kind] = {"authorized": decision["body"]["authorized"],
                           "unmet": [item["requirement"] for item in decision["body"]["reasons"] if not item["satisfied"]]}
    summary["authority"] = decisions

    # 9. auditors
    findings = agents.provenance_audit(ledger)
    probe = agents.adversarial_ledger_probe(output / "ledger")
    audit = agents.submit(ledger, "adversarial-test", "audit_finding",
                          {"finding": f"{sum(item['detected'] for item in probe)} of {len(probe)} ledger mutations detected",
                           "entries": [ledger.head()], "probe": probe},
                          agent="ciw.adversarial", rationale="tamper-evidence self-test")
    agents.dispose(ledger, audit["entry_id"])
    summary["audit"] = {"provenance_findings": findings, "probe_detected": [item["detected"] for item in probe]}

    # 10. governance
    from .governance import Council, current_pins, enact, sign_vote
    members = [{"member_id": name, "public_key": ed25519.public_key(bench_key(name)).hex(), "weight": 1}
               for name in ("bench-a", "bench-b", "bench-c")]
    council = Council.from_json({"members": members, "quorum_fraction": 0.5, "approval_fraction": 0.5})
    pin = {"proposal_id": "pin-native-solvers", "subject_kind": "provider_pin",
           "subject": {"name": "ciw.science.native-solvers", "digest": source_digest()},
           "action": "pin", "rationale": "pin the solver sources the bench evidence was produced with",
           "created_at": "2026-09-23T00:00:00+00:00"}
    votes = [sign_vote(bench_key(name), name, pin, "approve") for name in ("bench-a", "bench-b")]
    enact(ledger, council, pin, votes)
    summary["governance"] = current_pins(ledger, council).get("pins")

    # 11. reports and bundle
    model = build(ledger)
    text = markdown(model)
    (output / "report.md").write_text(text, encoding="utf-8", newline="\n")
    (output / "report.tex").write_text(latex(model), encoding="utf-8", newline="\n")
    retain_report(ledger, text, "markdown")
    from .bundle import export_bundle, inspect_bundle
    signer = bench_key("bundle")
    export_bundle(ledger, output / "evidence.ciwb", signing_key=signer, key_id="bench-bundle")
    inspection = inspect_bundle(output / "evidence.ciwb", trusted_keys={"bench-bundle": ed25519.public_key(signer)})
    summary["bundle"] = {"ok": inspection.get("ok"), "signature": inspection.get("signature", {}).get("status")
                         if isinstance(inspection.get("signature"), dict) else inspection.get("signature")}
    verification = ledger.verify()
    summary["ledger"] = {"entries": len(ledger), "head": ledger.head(), "verified": verification.ok}
    (output / "summary.json").write_text(json.dumps(plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8",
                                         newline="\n")
    return plain(summary)
