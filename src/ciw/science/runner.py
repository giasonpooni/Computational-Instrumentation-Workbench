"""Execute compiled experiments into the evidence ledger, and replay them.

For every run the ledger receives, in order: the exact specification bytes,
the execution plan, acquisition requests the compiler could not satisfy, the
runtime identity, and per job a parameter identity, an execution record
(completed or refused, with backend and timing telemetry), the numerical result
with its content identity, the oracle verification, and, when declared, a
synthetic observation and its comparison. Ensemble groups receive a Monte Carlo
check of the first-order (Jacobi) uncertainty prediction. Claims are then made
only where ``claims.check_claim`` admits them.

Replay re-executes each retained job from its parameter identity with the
currently registered solver, compares result identities bitwise and, failing
that, numerically within the retained tolerances, and records runtime drift.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._common import Refusal, canonical_json, content_identity, plain
from .backends import drift, measured
from .claims import make_claim
from .experiment import compile_spec
from .geometry import surface_from_json
from .ledger import Ledger
from .observation import compare, explain, observation as make_observation, predict
from .oracles import evaluate
from .solvers import SolverRegistry, default_registry

ENSEMBLE_Z_LIMIT = 4.0


def _lateral_offsets(surface, nominal: dict, members: list[dict]) -> np.ndarray:
    """Signed offsets of member endpoints normal to the nominal geodesic at its endpoint."""
    end = nominal["endpoint"]
    chart = surface.charts[end["chart"]]
    u, du = np.asarray(end["u"], dtype=float), np.asarray(end["du"], dtype=float)
    if "x" in end:
        jacobian = chart.jacobian(u)
        tangent = jacobian @ du
        tangent /= np.linalg.norm(tangent)
        normal = np.cross(jacobian[:, 0], jacobian[:, 1])
        lateral = np.cross(normal / np.linalg.norm(normal), tangent)
        return np.array([(np.asarray(m["endpoint"]["x"]) - np.asarray(end["x"])) @ lateral for m in members])
    g = chart.metric(u)
    normal = np.array([-(g[1] @ du), g[0] @ du]) / math.sqrt(np.linalg.det(g))
    normal /= math.sqrt(normal @ g @ normal)
    return np.array([(np.asarray(m["endpoint"]["u"]) - u) @ g @ normal for m in members])


def ensemble_check(surface, nominal: dict, members: list[dict], heading_std: float) -> dict:
    """Monte Carlo lateral spread against ``|j(L)| sigma_heading`` (first-order propagation)."""
    offsets = _lateral_offsets(surface, nominal, members)
    count = len(offsets)
    sample = float(np.std(offsets, ddof=1))
    predicted = abs(nominal["jacobi"]["value"]) * heading_std if "jacobi" in nominal else None
    if predicted is None or predicted == 0:
        return {"oracle_id": "ensemble-first-order", "kind": "statistical", "applicable": False,
                "detail": {"reason": "no nonzero first-order prediction (Jacobi field absent or at a conjugate point)",
                           "sample_std": sample, "members": count}}
    z = (sample / predicted - 1.0) * math.sqrt(2 * (count - 1))
    return plain({"oracle_id": "ensemble-first-order", "kind": "statistical", "applicable": True,
                  "passed": abs(z) <= ENSEMBLE_Z_LIMIT, "quantity": "lateral endpoint std",
                  "computed": sample, "reference": predicted, "abs_error": abs(sample - predicted),
                  "tolerance": predicted * ENSEMBLE_Z_LIMIT / math.sqrt(2 * (count - 1)),
                  "detail": {"z": z, "members": count, "mean_offset": float(np.mean(offsets)),
                             "basis": "relative standard error of a sample std is 1/sqrt(2(n-1)) for normal data"}})


def execute(spec: dict, ledger: Ledger, *, registry: SolverRegistry | None = None, source_bytes: bytes | None = None,
            frame_registry_entry: str | None = None) -> dict:
    """Compile and run a specification, retaining every artifact in the ledger."""
    registry = registry or default_registry()
    plan = compile_spec(spec, registry)
    declaration = registry.get(plan["solver_id"])
    experiment_id = plan["experiment_id"]
    raw = source_bytes if source_bytes is not None else canonical_json(spec).encode("utf-8")
    blob = ledger.put_blob(raw)
    spec_entry = ledger.append("ciw.science.experiment-spec.v1", {
        "experiment_id": experiment_id, "spec": spec, "spec_identity": plan["spec_identity"], "source_blob": blob,
        "source_bytes": len(raw)}, blobs=[blob], refs=[frame_registry_entry] if frame_registry_entry else [])
    plan_entry = ledger.append("ciw.science.execution-plan.v1", {
        "experiment_id": experiment_id, "plan_identity": plan["plan_identity"], "plan": plan,
        "jobs": [job["job_id"] for job in plan["jobs"]]}, refs=[spec_entry["entry_id"]])
    for request in plan["acquisition_requests"]:
        ledger.append("ciw.science.acquisition-request.v1", {**request, "experiment_id": experiment_id},
                      refs=[plan_entry["entry_id"]])
    runtime = declaration.runtime(plan["seed"])
    runtime_entry = ledger.append("ciw.science.runtime-identity.v1", {
        "runtime": runtime, "runtime_identity": content_identity(runtime), "experiment_id": experiment_id},
        refs=[plan_entry["entry_id"]])
    tolerances = plan["validation"]["tolerances"]
    outcomes: dict[str, dict] = {}
    for job in plan["jobs"]:
        parameter_entry = ledger.append("ciw.science.parameter-identity.v1", {
            "job_id": job["job_id"], "parameters": job, "parameter_identity": content_identity(job),
            "experiment_id": experiment_id}, refs=[plan_entry["entry_id"]])
        try:
            result, telemetry = measured(lambda job=job: declaration.implementation(job))
        except Refusal as exc:
            ledger.append("ciw.science.execution.v1", {
                "job_id": job["job_id"], "solver_id": job["solver_id"], "status": "refused", "refusal": exc.to_dict(),
                "backend": plan["backend"], "experiment_id": experiment_id},
                refs=[parameter_entry["entry_id"], runtime_entry["entry_id"]])
            outcomes[job["job_id"]] = {"status": "refused", "job": job}
            continue
        execution_entry = ledger.append("ciw.science.execution.v1", {
            "job_id": job["job_id"], "solver_id": job["solver_id"], "status": "completed", "backend": plan["backend"],
            "telemetry": telemetry, "experiment_id": experiment_id},
            refs=[parameter_entry["entry_id"], runtime_entry["entry_id"]])
        result_entry = ledger.append("ciw.science.numerical-result.v1", {
            "job_id": job["job_id"], "solver_id": job["solver_id"], "result": result,
            "result_identity": content_identity(result), "experiment_id": experiment_id},
            refs=[execution_entry["entry_id"]])
        outcomes[job["job_id"]] = {"status": "completed", "job": job, "result": result,
                                   "execution": execution_entry["entry_id"], "result_entry": result_entry["entry_id"]}
        if job.get("ensemble", {}).get("role") == "member":
            continue
        summary = evaluate(plan["validation"]["oracles"], job, result, declaration.implementation, tolerances)
        verification = ledger.append("ciw.science.verification.v1", {
            "subject": result_entry["entry_id"], "job_id": job["job_id"], "status": summary["status"],
            "verdicts": summary["verdicts"], "required": plan["validation"]["oracles"],
            "not_applicable": summary["not_applicable"], "tolerances": tolerances, "experiment_id": experiment_id},
            refs=[result_entry["entry_id"]])
        outcomes[job["job_id"]]["verification"] = verification["entry_id"]
        outcomes[job["job_id"]]["verification_status"] = summary["status"]
        if plan["observation"] and plan["observation"]["acquisition"] == "synthetic":
            _synthetic_observation(ledger, plan, job, result, result_entry["entry_id"], outcomes[job["job_id"]])
    for group in plan["validation"]["ensemble_checks"]:
        _ensemble(ledger, plan, outcomes, group)
    claims = _claims(ledger, plan, outcomes)
    return plain({"experiment_id": experiment_id, "plan_identity": plan["plan_identity"],
                  "spec_entry": spec_entry["entry_id"], "plan_entry": plan_entry["entry_id"],
                  "jobs": len(plan["jobs"]),
                  "completed": sum(1 for item in outcomes.values() if item["status"] == "completed"),
                  "refused": sum(1 for item in outcomes.values() if item["status"] == "refused"),
                  "verifications": {status: sum(1 for item in outcomes.values() if item.get("verification_status") == status)
                                    for status in ("passed", "failed", "incomplete")},
                  "claims": claims, "ledger_head": ledger.head()})


def _synthetic_observation(ledger: Ledger, plan: dict, job: dict, result: dict, result_entry: str, outcome: dict) -> None:
    spec_observation = plan["observation"]
    surface = surface_from_json(job["surface"], job["length_unit"])
    start, end = job["point"], result["endpoint"]["u"]
    truth = spec_observation.get("synthetic_truth", spec_observation["observable"])
    std_quantity = spec_observation["noise_std"]
    from .units import Quantity
    std = Quantity.from_json(std_quantity, "noise_std").magnitude(job["length_unit"])
    true_value = predict(truth, surface, points=(start, end))["value"]
    seed = int(content_identity({"job": job["job_id"], "seed": plan["seed"]})[7:15], 16)
    value = true_value + float(np.random.default_rng(seed).normal(0.0, std))
    record = make_observation(spec_observation["observable"], value, job["length_unit"], std, acquisition="synthetic",
                              frame=spec_observation.get("frame"), clock=spec_observation.get("clock"),
                              instrument=spec_observation.get("instrument"))
    record.update(experiment_id=plan["experiment_id"], job_id=job["job_id"], generated_from=truth,
                  generator_seed=seed)
    entry = ledger.append("ciw.science.observation.v1", record, refs=[result_entry])
    predicted = predict(spec_observation["observable"], surface, points=(start, end))
    comparison = compare(record, predicted)
    diagnosis = explain(surface, (start, end), value, job["length_unit"], std) if surface.embedded else None
    outcome["observation"] = entry["entry_id"]
    outcome["comparison"] = comparison
    make_claim(ledger, f"Synthetic {spec_observation['observable']} observation compared with the model prediction; "
               "physical interpretation remains unresolved because no physical measurement exists",
               "interpretation", [entry["entry_id"], result_entry], experiment_id=plan["experiment_id"],
               status="unresolved", detail={"comparison": comparison, "diagnosis": diagnosis})


def _ensemble(ledger: Ledger, plan: dict, outcomes: dict, group: str) -> None:
    items = [item for item in outcomes.values() if item["job"].get("ensemble", {}).get("group") == group]
    nominal = next((item for item in items if item["job"]["ensemble"]["role"] == "nominal"), None)
    members = [item for item in items if item["job"]["ensemble"]["role"] == "member" and item["status"] == "completed"]
    if nominal is None or nominal["status"] != "completed" or len(members) < 2:
        return
    surface = surface_from_json(nominal["job"]["surface"], nominal["job"]["length_unit"])
    verdict = ensemble_check(surface, nominal["result"], [item["result"] for item in members],
                             nominal["job"]["settings"]["heading_std_rad"])
    status = "incomplete" if not verdict["applicable"] else ("passed" if verdict["passed"] else "failed")
    ledger.append("ciw.science.verification.v1", {
        "subject": nominal["result_entry"], "job_id": nominal["job"]["job_id"], "status": status,
        "verdicts": [verdict], "required": ["ensemble-first-order"], "not_applicable": [] if verdict["applicable"]
        else ["ensemble-first-order"], "ensemble_group": group, "experiment_id": plan["experiment_id"]},
        refs=[nominal["result_entry"], *[item["result_entry"] for item in members]])


def _claims(ledger: Ledger, plan: dict, outcomes: dict) -> list[dict]:
    made, sought, experiment_id = [], set(plan["claims_sought"]), plan["experiment_id"]
    completed = [item for item in outcomes.values() if item["status"] == "completed"]
    attempts = []
    if "computed" in sought and completed:
        attempts.append(("computed", f"{len(completed)} of {len(outcomes)} jobs completed with {plan['solver_id']}",
                         [item["execution"] for item in completed], plan["solver_id"]))
    if "predicted" in sought and completed:
        attempts.append(("predicted", f"Model predictions for {experiment_id} under the declared surface and solver",
                         [item["result_entry"] for item in completed], "declared model only"))
    verified = [entry["entry_id"] for entry in ledger.entries("verification")
                if entry["body"].get("experiment_id") == experiment_id]
    if "verified" in sought and verified:
        attempts.append(("verified", "Declared invariants hold within tolerance: " + ", ".join(plan["validation"]["oracles"]),
                         verified, {"oracles": plan["validation"]["oracles"], "tolerances": plan["validation"]["tolerances"]}))
    if "measured" in sought:
        attempts.append(("measured", f"Physical measurement for {experiment_id}",
                         [item["observation"] for item in outcomes.values() if "observation" in item], None))
    for claim_class, statement, refs, scope in attempts:
        try:
            entry = make_claim(ledger, statement, claim_class, refs, experiment_id=experiment_id, scope=scope)
            made.append({"claim_class": claim_class, "admitted": True, "entry": entry["entry_id"]})
        except Refusal as exc:
            made.append({"claim_class": claim_class, "admitted": False, "refusal": exc.to_dict()})
    return made


def _close(first: Any, second: Any, tolerances: dict, path: str = "") -> list[str]:
    """Paths where two JSON trees differ beyond the declared tolerance."""
    if isinstance(first, dict) and isinstance(second, dict):
        if first.keys() != second.keys():
            return [path + "{keys}"]
        return [item for key in first for item in _close(first[key], second[key], tolerances, f"{path}.{key}")]
    if isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            return [path + "[len]"]
        return [item for index, (a, b) in enumerate(zip(first, second)) for item in _close(a, b, tolerances, f"{path}[{index}]")]
    if isinstance(first, (int, float)) and isinstance(second, (int, float)) and not isinstance(first, bool):
        limit = max(float(tolerances.get("abs", 0.0)), float(tolerances.get("rel", 0.0)) * abs(float(first)))
        return [] if abs(float(first) - float(second)) <= limit else [path]
    return [] if first == second else [path]


def replay(ledger: Ledger, experiment_id: str, *, registry: SolverRegistry | None = None) -> dict:
    """Re-execute every retained completed job of an experiment and record replay receipts."""
    registry = registry or default_registry()
    plans = [entry for entry in ledger.entries("execution_plan") if entry["body"]["experiment_id"] == experiment_id]
    if not plans:
        raise Refusal("unknown_experiment", f"No retained plan for {experiment_id!r}")
    plan = plans[-1]["body"]["plan"]
    runtime_entries = [entry for entry in ledger.entries("runtime_identity")
                       if entry["body"].get("experiment_id") == experiment_id]
    original_runtime = runtime_entries[-1]["body"]["runtime"] if runtime_entries else {}
    results = {entry["body"]["job_id"]: entry for entry in ledger.entries("numerical_result")
               if entry["body"].get("experiment_id") == experiment_id}
    parameters = {entry["body"]["job_id"]: entry for entry in ledger.entries("parameter_identity")
                  if entry["body"].get("experiment_id") == experiment_id}
    try:
        declaration = registry.get(plan["solver_id"])
        current_runtime = declaration.runtime(plan.get("seed"))
    except Refusal as exc:
        declaration, current_runtime = None, None
        refusal = exc.to_dict()
    runtime_drift = drift(original_runtime, current_runtime or {})
    runtime_entry = ledger.append("ciw.science.runtime-identity.v1", {
        "runtime": current_runtime or {"unavailable": True}, "runtime_identity": content_identity(current_runtime or {}),
        "experiment_id": experiment_id, "purpose": "replay"}, refs=[plans[-1]["entry_id"]])
    counts = {"reproduced": 0, "within_tolerance": 0, "diverged": 0, "refused": 0}
    for job_id, original in results.items():
        job = parameters[job_id]["body"]["parameters"]
        if content_identity(job) != parameters[job_id]["body"]["parameter_identity"]:
            status, comparison = "refused", {"reason": "retained parameters do not match their identity"}
        elif declaration is None:
            status, comparison = "refused", {"reason": "solver unavailable", "refusal": refusal}
        else:
            try:
                fresh = declaration.implementation(job)
            except Refusal as exc:
                status, comparison = "refused", {"reason": "solver refused on replay", "refusal": exc.to_dict()}
            else:
                identity = content_identity(fresh)
                if identity == original["body"]["result_identity"]:
                    status, comparison = "reproduced", {"result_identity": identity, "bitwise": True}
                else:
                    differences = _close(original["body"]["result"], plain(fresh), plan["validation"]["tolerances"])
                    status = "within_tolerance" if not differences else "diverged"
                    comparison = {"result_identity": identity, "bitwise": False, "differences": differences[:32]}
        counts[status] += 1
        ledger.append("ciw.science.replay-receipt.v1", {
            "original": original["entry_id"], "job_id": job_id, "status": status, "comparison": comparison,
            "runtime_drift": runtime_drift, "experiment_id": experiment_id},
            refs=[original["entry_id"], runtime_entry["entry_id"]])
    return {"experiment_id": experiment_id, "counts": counts, "runtime_drift": runtime_drift,
            "ledger_head": ledger.head()}
