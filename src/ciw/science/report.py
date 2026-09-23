"""Reproducible reports derived only from a ledger.

The report shows, per experiment: the hypothesis, the mathematical model as
equations, the numerical implementation and its declared failure modes, the
runtime and provider, per-job results, oracle verdicts with errors and
tolerances, uncertainty checks, observations with their acquisition kind and
diagnosis, admitted claims, replays with runtime drift, and authority
decisions with every unmet requirement. Other retained evidence (fusion,
hardware captures, designs, protocols, governance, agent proposals) is
summarized by kind.

Given the same ledger bytes the output bytes are identical: nothing reads the
clock, the filesystem outside the ledger, or the environment. The ledger head
and its verification status are printed so a reader can check which evidence
the report was generated from.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ._common import Refusal
from .claims import status_statements
from .ledger import Ledger
from .solvers import default_registry


def _number(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    if isinstance(value, list):
        return "[" + ", ".join(_number(item) for item in value[:6]) + (", …" if len(value) > 6 else "") + "]"
    return str(value)


def _quantity(value: Any) -> str:
    if isinstance(value, dict) and set(value) >= {"value", "unit"}:
        return f"{_number(value['value'])} {value['unit']}"
    return _number(value)


def build(ledger: Ledger, experiment_ids: list[str] | None = None) -> dict:
    """Collect the report model from the ledger (no rendering)."""
    integrity = ledger.verify()
    specs = [entry for entry in ledger.entries("experiment_spec")]
    ids = experiment_ids or list(dict.fromkeys(entry["body"]["experiment_id"] for entry in specs))
    known = {entry["body"]["experiment_id"] for entry in specs}
    missing = [item for item in ids if item not in known]
    if missing:
        raise Refusal("unknown_experiment", f"No retained specification for {missing}")
    registry = default_registry()
    by_experiment: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for entry in ledger.entries():
        experiment = entry["body"].get("experiment_id")
        if experiment in ids:
            by_experiment[experiment][entry["kind"]].append(entry)
    experiments = []
    for experiment_id in ids:
        grouped = by_experiment[experiment_id]
        spec = grouped["experiment_spec"][-1]["body"]["spec"]
        plan = grouped["execution_plan"][-1]["body"]["plan"]
        try:
            declaration = registry.get(plan["solver_id"]).describe()
        except Refusal:
            declaration = {"solver_id": plan["solver_id"], "equations": [], "assumptions": [], "failure_modes": [],
                           "title": "solver not registered in this process"}
        experiments.append({"experiment_id": experiment_id, "spec": spec, "plan": plan, "solver": declaration,
                            "status": status_statements(ledger, experiment_id), "entries": grouped})
    decisions = [entry for entry in ledger.entries("authority_decision")]
    other = Counter(entry["kind"] for entry in ledger.entries()
                    if entry["body"].get("experiment_id") not in ids)
    return {"ledger_head": ledger.head(), "entries": len(ledger), "integrity": integrity.to_json(),
            "experiments": experiments, "decisions": decisions, "other": dict(sorted(other.items())),
            "ledger": ledger}


def _oracle_rows(verifications: list[dict]) -> list[dict]:
    worst: dict[str, dict] = {}
    for entry in verifications:
        for verdict in entry["body"]["verdicts"]:
            current = worst.setdefault(verdict["oracle_id"], {"oracle_id": verdict["oracle_id"], "kind": verdict["kind"],
                                                               "applicable": 0, "passed": 0, "failed": 0, "max_error": None,
                                                               "tolerance": None, "detail": None})
            if not verdict["applicable"]:
                current["detail"] = (verdict.get("detail") or {}).get("reason")
                continue
            current["applicable"] += 1
            current["passed" if verdict["passed"] else "failed"] += 1
            error = verdict.get("abs_error")
            if error is not None and (current["max_error"] is None or error >= current["max_error"]):
                current["max_error"], current["tolerance"] = error, verdict.get("tolerance")
                detail = verdict.get("detail") or {}
                if "observed_order" in detail:
                    current["detail"] = f"order {_number(detail['observed_order'])} ({detail.get('order_status')})"
                elif "z" in detail:
                    current["detail"] = f"z = {_number(detail['z'])}, n = {detail['members']}"
    return list(worst.values())


def markdown(model: dict) -> str:
    lines = ["# Scientific evidence report", "",
             f"Ledger head `{model['ledger_head']}` · {model['entries']} entries · integrity "
             + ("**verified**" if model["integrity"]["ok"] else f"**FAILED** ({len(model['integrity']['problems'])} problems)"),
             "", "Generated from retained evidence only. Synthetic observations are labelled and never count as "
             "physical measurements.", ""]
    for experiment in model["experiments"]:
        spec, plan, solver, status, entries = (experiment[key] for key in ("spec", "plan", "solver", "status", "entries"))
        lines += [f"## `{experiment['experiment_id']}` — {spec['title']}", "",
                  f"**Hypothesis.** {spec['hypothesis']['statement']}", ""]
        for competing in spec["hypothesis"].get("competing", []):
            lines.append(f"- *{competing['id']}*: {competing['statement']}")
        lines += ["", "| Statement | Derived from the ledger |", "| --- | --- |"]
        for key in ("measurement", "calculation", "numerical_stability", "physical_interpretation", "decision"):
            lines.append(f"| {key.replace('_', ' ')} | {status[key]} |")
        surface = spec["model"]["surface"]
        parameters = ", ".join(f"{key} = {_quantity(value)}" for key, value in surface.items() if key != "type")
        lines += ["", "### Model and implementation", "",
                  f"{spec['model']['kind']} on a **{surface['type']}** surface" + (f" ({parameters})" if parameters else "")
                  + f"; working length unit `{spec['model']['length_unit']}`.", "",
                  f"Solver `{solver['solver_id']}` — {solver['title']}.", ""]
        lines += [f"$$ {equation} $$" for equation in solver.get("equations", [])]
        if solver.get("method"):
            lines += ["", f"Method: {solver['method']}; precision {solver['precision']}; error behaviour: "
                      f"{solver['error_behavior']}."]
        lines += ["", "Assumptions: " + "; ".join(solver.get("assumptions", [])) + ".",
                  "", "Known failure modes: " + "; ".join(solver.get("failure_modes", [])) + "."]
        runtime = entries["runtime_identity"][0]["body"]["runtime"] if entries["runtime_identity"] else {}
        lines += ["", "### Provider and runtime", "", "| Field | Value |", "| --- | --- |"]
        for key in ("provider", "backend", "solver_id", "source_digest", "python", "numpy", "machine", "system", "seed"):
            lines.append(f"| {key} | `{runtime.get(key)}` |")
        lines.append(f"| rejected backends | {', '.join(item['backend_id'] + ': ' + item['reason'] for item in plan['backend']['rejected']) or 'none'} |")
        lines += ["", f"### Jobs ({len(plan['jobs'])} planned)", "",
                  "| Job | Sweep point | Status | Length | Endpoint | Verification |", "| --- | --- | --- | --- | --- | --- |"]
        results = {entry["body"]["job_id"]: entry for entry in entries["numerical_result"]}
        executions = {entry["body"]["job_id"]: entry for entry in entries["execution"]}
        verifications = {entry["body"]["job_id"]: entry for entry in entries["verification"]
                         if "ensemble_group" not in entry["body"]}
        shown = 0
        for job in plan["jobs"]:
            if job.get("ensemble", {}).get("role") == "member":
                continue
            shown += 1
            if shown > 40:
                lines.append("| … | further jobs omitted from the table; all are in the ledger | | | | |")
                break
            result = results.get(job["job_id"], {}).get("body", {}).get("result", {})
            endpoint = result.get("endpoint", {})
            lines.append(f"| `{job['job_id'][:16]}` | {', '.join(f'{k}={_quantity(v)}' for k, v in job['sweep_point'].items()) or '—'} "
                         f"| {executions.get(job['job_id'], {}).get('body', {}).get('status', 'absent')} "
                         f"| {_number(result.get('length'))} | {_number(endpoint.get('u', result.get('v')))} "
                         f"| {verifications.get(job['job_id'], {}).get('body', {}).get('status', '—')} |")
        members = sum(1 for job in plan["jobs"] if job.get("ensemble", {}).get("role") == "member")
        if members:
            lines.append(f"\n{members} ensemble members were executed for uncertainty checks (not tabulated).")
        lines += ["", "### Oracle verdicts", "", "Worst case per oracle across jobs.", "",
                  "| Oracle | Kind | Applied | Passed | Failed | Max error | Tolerance | Note |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in _oracle_rows(entries["verification"]):
            lines.append(f"| `{row['oracle_id']}` | {row['kind']} | {row['applicable']} | {row['passed']} | {row['failed']} "
                         f"| {_number(row['max_error'])} | {_number(row['tolerance'])} | {row['detail'] or ''} |")
        jacobi = [entry["body"]["result"]["jacobi"] for entry in entries["numerical_result"]
                  if "jacobi" in entry["body"]["result"] and "lateral_std" in entry["body"]["result"]["jacobi"]]
        if jacobi:
            lines += ["", "### Uncertainty", "", f"First-order lateral endpoint standard deviation "
                      f"|j(L)|·σ_heading ranges over [{_number(min(item['lateral_std'] for item in jacobi))}, "
                      f"{_number(max(item['lateral_std'] for item in jacobi))}] {spec['model']['length_unit']}."]
        if entries["observation"]:
            lines += ["", "### Observations", "",
                      "| Job | Observable | Acquisition | Generated from | Value | Normalized residual | Diagnosis |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            interpretations = {ref: entry for entry in entries["claim"] if entry["body"]["claim_class"] == "interpretation"
                               for ref in entry["refs"]}
            for entry in entries["observation"]:
                body = entry["body"]
                detail = interpretations.get(entry["entry_id"], {}).get("body", {}).get("detail") or {}
                comparison, diagnosis = detail.get("comparison") or {}, detail.get("diagnosis") or {}
                lines.append(f"| `{body.get('job_id', '')[:16]}` | {body['observable']} | {body['acquisition']} "
                             f"| {body.get('generated_from', '—')} | {_number(body['value'])} {body['unit']} "
                             f"| {_number(comparison.get('normalized_residual'))} | {diagnosis.get('reading', '—')} |")
        for request in entries["acquisition_request"]:
            lines += ["", f"Acquisition request: {request['body']['observable']} with `{request['body']['instrument']}` — "
                      f"{request['body']['status']} ({request['body']['reason']})."]
        admitted = [entry for entry in entries["claim"] if entry["body"]["claim_class"] != "interpretation"]
        if admitted:
            lines += ["", "### Admitted claims", "", "| Class | Statement | Evidence entries |", "| --- | --- | --- |"]
            for entry in admitted:
                lines.append(f"| {entry['body']['claim_class']} | {entry['body']['statement']} | {len(entry['refs'])} |")
        if entries["replay_receipt"]:
            counts = Counter(entry["body"]["status"] for entry in entries["replay_receipt"])
            drift = entries["replay_receipt"][-1]["body"]["runtime_drift"]
            lines += ["", "### Replays", "", ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))
                      + ". Runtime drift: " + (", ".join(item["key"] for item in drift) if drift else "none") + "."]
        lines.append("")
    if model["decisions"]:
        lines += ["## Authority decisions", "", "| Action | Mode | Authorized | Unmet requirements |", "| --- | --- | --- | --- |"]
        for entry in model["decisions"]:
            unmet = [f"{item['requirement']} ({item['detail'] if isinstance(item['detail'], str) else 'see ledger'})"
                     for item in entry["body"]["reasons"] if not item["satisfied"]]
            lines.append(f"| {entry['body']['action_kind']} | {entry['body']['mode']} | {entry['body']['authorized']} "
                         f"| {'; '.join(unmet) or 'none'} |")
        lines.append("")
    if model["other"]:
        lines += ["## Other retained evidence", "", "| Kind | Entries |", "| --- | --- |"]
        lines += [f"| {kind} | {count} |" for kind, count in model["other"].items()]
        ledger = model["ledger"]
        for entry in ledger.entries("design_recommendation"):
            lines += ["", f"**Design recommendation:** {entry['body']['summary']}"]
        for entry in ledger.entries("hardware_capture"):
            summary = entry["body"]["summary"]
            lines += ["", "**Hardware capture:** " + ", ".join(f"{key}={_number(value)}" for key, value in sorted(summary.items())
                                                             if not isinstance(value, (dict, list)))]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _tex(value: Any) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                     ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}"),
                     ("·", r"$\cdot$"), ("σ", r"$\sigma$"), ("—", "---"), ("…", r"\dots{}")):
        text = text.replace(old, new)
    return text


def latex(model: dict) -> str:
    lines = [r"\documentclass{article}", r"\usepackage{amsmath}", r"\usepackage[margin=2cm]{geometry}",
             r"\begin{document}", r"\section*{Scientific evidence report}",
             r"Ledger head \texttt{" + _tex(model["ledger_head"]) + "}, " + str(model["entries"]) + " entries, integrity "
             + ("verified." if model["integrity"]["ok"] else "FAILED.")]
    for experiment in model["experiments"]:
        spec, solver, status = experiment["spec"], experiment["solver"], experiment["status"]
        lines += [r"\subsection*{" + _tex(spec["title"]) + "}", r"\textbf{Hypothesis.} " + _tex(spec["hypothesis"]["statement"]),
                  r"\begin{itemize}"]
        for key in ("measurement", "calculation", "numerical_stability", "physical_interpretation", "decision"):
            lines.append(r"\item " + _tex(status[key]))
        lines += [r"\end{itemize}", r"Solver \texttt{" + _tex(solver["solver_id"]) + "}:"]
        if solver.get("equations"):
            lines += [r"\begin{align}", r" \\ ".join(solver["equations"]), r"\end{align}"]
        lines += [r"\begin{tabular}{lrrrl}", r"oracle & applied & failed & max error & tolerance \\ \hline"]
        for row in _oracle_rows(experiment["entries"]["verification"]):
            lines.append(r"\texttt{" + _tex(row["oracle_id"]) + "} & " + str(row["applicable"]) + " & " + str(row["failed"])
                         + " & " + _tex(_number(row["max_error"])) + " & " + _tex(_number(row["tolerance"])) + r" \\")
        lines.append(r"\end{tabular}")
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def retain(ledger: Ledger, text: str, fmt: str) -> dict:
    """Retain rendered report bytes; the entry binds the ledger head the report was built from."""
    head = ledger.head()
    blob = ledger.put_blob(text.encode("utf-8"))
    return ledger.append("ciw.science.report.v1", {"blob": blob, "format": fmt, "ledger_head": head},
                         blobs=[blob], refs=[head] if head else [])
