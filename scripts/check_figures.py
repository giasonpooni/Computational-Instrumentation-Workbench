"""Re-execute every figure task of a retained lab run and compare its figures with the retained copies.

T158 re-executes the figure tasks that fit the research section's time budget,
on the platform of the run. This script re-executes every task whose retained
report lists an SVG figure, with no time budget, into a new output directory
(``ciw lab run`` in this process, from the installed ``ciw``) and compares
each figure with the retained copy: byte for byte, except a figure its task
declared as a wall-clock timing figure (``wall_clock_timing`` in the report's
generated artifacts) or as a rounding-level figure (``rounding_level``: it plots
values at binary64 rounding level, whose last bits follow the BLAS kernel and
platform), which is compared for presence and structure (series and points)
only; the two declarations are counted separately. Run on another platform
(Windows) against the same retained run, it gives the second-platform
comparison that T158 names as its next step; run under a forced OpenBLAS
kernel (``OPENBLAS_CORETYPE``, as CI's lab-blas-kernels job does), it fails
on an undeclared figure whose bytes follow the kernel.

A task is not re-executed when its retained report used a provider that is
not bound here (``--provider ROLE=PATH``, as for ``ciw lab run``) or recorded
source digests that differ from this installation's, and a re-executed task
that ends in another state or with other requirement-probe outcomes is not
comparable; neither is ever counted as a match. ``figure-check.json``
(``ciw.lab-figure-check.v1``) records every figure's outcome with the platform
(OS, Python, NumPy, its BLAS and the OpenBLAS kernel it runs, read from the
loaded library by ``ciw.lab.blas_probe.openblas_core``, with any forced
``OPENBLAS_CORETYPE``) and ``figure-check.md`` summarizes it; the exit status
is 3 when a figure mismatches or none was compared. BLAS runs single-threaded
unless the caller sets its thread variables, as in the clean-room run that
retained the figures. The record is a reproducibility check, not a lab
finding, and stays outside ``lab/``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
RECORD_SCHEMA = "ciw.lab-figure-check.v1"
# Pinned as in scripts/reproduce_lab.py: single-threaded BLAS keeps reduction order, and so figure data, stable.
BLAS_THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
# A figure the re-executed task writes although its retained report lists none is a mismatch as well.
NOT_RETAINED = "not retained"


def bindings(values) -> dict:
    """``ROLE=PATH`` provider bindings, as ``ciw lab run --provider`` takes them."""
    providers = {}
    for binding in values:
        role, separator, path = binding.partition("=")
        if not separator or not role or not path:
            raise SystemExit(f"Provider bindings use ROLE=PATH, not {binding!r}")
        providers[role] = Path(path)
    return providers


def blas_identity() -> dict:
    """NumPy's BLAS and LAPACK build, the OpenBLAS kernel it runs and the CPU features it uses, with threadpoolctl's
    runtime view when installed."""
    import numpy as np

    from ciw.lab.blas_probe import openblas_core
    # The kernel the loaded library runs (the host's, or the one OPENBLAS_CORETYPE forces); the build
    # configuration below names only the build target.
    identity = {"thread_variables": {name: os.environ.get(name) for name in (*BLAS_THREADS, "OPENBLAS_CORETYPE")},
                "openblas_core": openblas_core()}
    try:
        config = np.show_config(mode="dicts")
    except TypeError:  # NumPy before 1.25 only prints its configuration
        config = {}
    for name, entry in (config.get("Build Dependencies") or {}).items():
        identity[name] = {key: entry[key] for key in ("name", "version", "openblas configuration") if entry.get(key)}
    identity["simd_found"] = (config.get("SIMD Extensions") or {}).get("found")
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return identity
    identity["runtime"] = [{key: pool.get(key) for key in ("internal_api", "version", "architecture", "num_threads")}
                           for pool in threadpool_info() if pool.get("user_api") == "blas"]
    return identity


def platform_identity() -> dict:
    import numpy as np

    import ciw
    from ciw.lab import runner
    return {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "python_implementation": platform.python_implementation(), "numpy": np.__version__,
            "blas": blas_identity(), "ciw": {"version": ciw.__version__, "package_digest": runner.package_digest()}}


def figure_reports(retained: Path) -> dict:
    """Retained reports that list an SVG figure, by task identity."""
    from ciw.lab import runner
    return {report["task_id"]: report for report in runner.load_reports(retained)
            if any(artifact["path"].endswith(".svg") for artifact in report["generated_artifacts"])}


def _figures(report) -> list:
    return [artifact for artifact in report["generated_artifacts"] if artifact["path"].endswith(".svg")]


def _probes(report) -> dict:
    identity = report.get("provider_runtime_identity")
    probes = identity.get("requirement_probes") if isinstance(identity, dict) else None
    return probes if isinstance(probes, dict) else {}


def not_reexecuted(report, providers) -> str | None:
    """Why a retained figure task is not re-executed here: a provider it used is unbound, or its sources changed."""
    from ciw.lab import runner
    from ciw.lab.research_portfolio import providers_used
    unbound = [role for role in providers_used(report) if role not in providers]
    if unbound:
        return f"provider {', '.join(unbound)} not bound here; the retained run used it"
    identity = report.get("provider_runtime_identity")
    identity = identity if isinstance(identity, dict) else {}
    # Provider-backed tasks (T005, T008, T097) record their CIW sources under "ciw", beside the provider's identity.
    sources = {}
    for record in (identity, identity.get("ciw")):
        if isinstance(record, dict) and isinstance(record.get("sources"), dict):
            sources.update(record["sources"])
    changed = sorted(name for name, digest in sources.items() if runner.source_digest(name) != digest)
    if changed:
        return "sources differ from the retained run's: " + ", ".join(changed)
    return None


def _declared(artifact) -> dict:
    from ciw.lab.report import FIGURE_DECLARATIONS
    return {key: artifact.get(key) is True for key in FIGURE_DECLARATIONS}


def compare_task(retained_dir: Path, fresh_dir: Path, old, new) -> tuple[list, str | None]:
    """The figure outcomes of one re-executed task, or why its figures are not comparable.

    A figure's declaration is read from the retained report, so a declaration a task adds takes effect once its
    report is retained again.
    """
    from ciw.lab.research_portfolio import compare_figure
    if old["state"] != new["state"]:
        return [], f"state {old['state']} -> {new['state']}"
    before, after = _probes(old), _probes(new)
    changed = [f"{name} {before.get(name)} -> {after.get(name)}" for name in sorted(set(before) | set(after))
               if before.get(name) != after.get(name)]
    if changed:
        return [], "requirement probes differ: " + ", ".join(changed)
    written = {artifact["path"]: artifact for artifact in _figures(new)}
    outcomes = []
    for artifact in _figures(old):
        path, declared = artifact["path"], _declared(artifact)
        fresh = (fresh_dir / path).read_bytes() if path in written else None
        outcomes.append({"task_id": old["task_id"], "path": path, **declared,
                         "outcome": compare_figure((retained_dir / path).read_bytes(), fresh, any(declared.values())),
                         "retained_sha256": artifact["sha256"], "fresh_sha256": written.get(path, {}).get("sha256")})
    for path in sorted(set(written) - {artifact["path"] for artifact in _figures(old)}):
        outcomes.append({"task_id": old["task_id"], "path": path, **_declared(written[path]), "outcome": NOT_RETAINED,
                         "retained_sha256": None, "fresh_sha256": written[path]["sha256"]})
    return outcomes, None


def summarize(reports: dict, outcomes: list, skipped: dict, not_comparable: dict) -> dict:
    from ciw.lab.report import FIGURE_DECLARATIONS, ROUNDING_LEVEL, WALL_CLOCK_TIMING
    from ciw.lab.research_portfolio import MISMATCHES

    def figures(task_ids):
        return sum(len(_figures(reports[task_id])) for task_id in task_ids)

    timing = [o for o in outcomes if o[WALL_CLOCK_TIMING]]
    rounding = [o for o in outcomes if o[ROUNDING_LEVEL]]
    return {"figure_tasks": len(reports), "figures": figures(reports),
            "compared_tasks": len({o["task_id"] for o in outcomes}),
            "compared_figures": sum(o["outcome"] != NOT_RETAINED for o in outcomes),
            "identical": sum(o["outcome"] == "identical" and not any(o[key] for key in FIGURE_DECLARATIONS)
                             for o in outcomes),
            "declared_timing_same_structure": sum(o["outcome"] == "same structure" for o in timing),
            "declared_timing_identical": sum(o["outcome"] == "identical" for o in timing),
            "declared_rounding_level_same_structure": sum(o["outcome"] == "same structure" for o in rounding),
            "declared_rounding_level_identical": sum(o["outcome"] == "identical" for o in rounding),
            "mismatched": sum(o["outcome"] in (*MISMATCHES, NOT_RETAINED) for o in outcomes),
            "not_reexecuted_tasks": len(skipped), "not_reexecuted_figures": figures(skipped),
            "not_comparable_tasks": len(not_comparable), "not_comparable_figures": figures(not_comparable)}


def _declaration(outcome) -> str:
    from ciw.lab.report import ROUNDING_LEVEL, WALL_CLOCK_TIMING
    return ("wall-clock timing" if outcome[WALL_CLOCK_TIMING] else "rounding level" if outcome[ROUNDING_LEVEL]
            else "no")


def render(record: dict) -> str:
    """The Markdown summary: platform, counts, then every declared or not byte-identical figure and every task left out."""
    host, counts = record["platform"], record["summary"]
    blas = host["blas"].get("blas") or {}
    forced = host["blas"]["thread_variables"].get("OPENBLAS_CORETYPE")
    lines = [f"# Figure re-execution on {host['platform']}", "",
             f"- Python {host['python']} ({host['python_implementation']}), NumPy {host['numpy']}, BLAS "
             f"{blas.get('name', 'unknown')} {blas.get('version', '')}".rstrip()
             + (f" ({blas['openblas configuration']})" if blas.get("openblas configuration") else ""),
             f"- OpenBLAS core: {host['blas'].get('openblas_core') or 'unknown'}"
             + (f" (OPENBLAS_CORETYPE={forced})" if forced else ""),
             f"- CIW {host['ciw']['version']}, package digest {host['ciw']['package_digest']}",
             f"- Retained run: `{record['retained']}`; providers bound: {', '.join(record['providers']) or 'none'}",
             f"- Figure tasks: {counts['figure_tasks']} ({counts['figures']} figures); re-executed and compared: "
             f"{counts['compared_tasks']} tasks ({counts['compared_figures']} figures)",
             f"- Byte-identical: {counts['identical']}; declared wall-clock timing figures with the same structure: "
             f"{counts['declared_timing_same_structure']}, byte-identical: {counts['declared_timing_identical']}; "
             f"declared rounding-level figures with the same structure: "
             f"{counts['declared_rounding_level_same_structure']}, byte-identical: "
             f"{counts['declared_rounding_level_identical']}",
             f"- Mismatched: {counts['mismatched']}",
             f"- Not re-executed: {counts['not_reexecuted_tasks']} tasks ({counts['not_reexecuted_figures']} figures); "
             f"not comparable: {counts['not_comparable_tasks']} tasks ({counts['not_comparable_figures']} figures)"]
    rows = [o for o in record["figures"] if o["outcome"] != "identical" or _declaration(o) != "no"]
    if rows:
        lines += ["", "| Task | Figure | Declared | Outcome |", "| --- | --- | --- | --- |"]
        lines += [f"| {o['task_id']} | `{o['path']}` | {_declaration(o)} | {o['outcome']} |" for o in rows]
    for title, reasons in (("Not re-executed", record["not_reexecuted"]), ("Not comparable", record["not_comparable"])):
        if reasons:
            lines += ["", f"| {title} | Reason |", "| --- | --- |"]
            lines += [f"| {task_id} | {reason.replace('|', '/')} |" for task_id, reason in sorted(reasons.items())]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="*", metavar="TASK", help="Retained figure tasks to re-execute (default: all)")
    parser.add_argument("--retained", type=Path, default=ROOT / "lab")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--provider", action="append", default=[], metavar="ROLE=PATH",
                        help="Provider binding, as for ciw lab run")
    args = parser.parse_args()
    from ciw.lab import runner
    providers = bindings(args.provider)
    retained, output = args.retained.resolve(), args.output_dir.resolve()
    if output == retained or retained in output.parents:
        raise SystemExit(f"The output directory must lie outside the retained run: {output}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory must be new or empty: {output}")
    reports = figure_reports(retained)
    if not reports:
        raise SystemExit(f"No retained figure tasks in {retained / 'reports'}: nothing to compare")
    unknown = sorted(set(args.tasks) - set(reports))
    if unknown:
        raise SystemExit(f"Not figure tasks of the retained run: {', '.join(unknown)}")
    reports = {task_id: reports[task_id] for task_id in sorted(set(args.tasks) or reports)}
    # Comparing with an edited retained copy would verify nothing.
    problems = runner.artifact_problems(retained, set(reports))
    if problems:
        raise SystemExit("Retained artifacts differ from their reports: " + "; ".join(problems[:5]))
    skipped = {task_id: reason for task_id, report in reports.items()
               if (reason := not_reexecuted(report, providers))}
    run_ids = [task_id for task_id in reports if task_id not in skipped]
    fresh_dir = output / "run"
    outcomes, not_comparable = [], {}
    if run_ids:
        print(f"Re-executing {len(run_ids)} figure tasks into {fresh_dir}", flush=True)
        runner.run_queue(fresh_dir, run_ids, providers)
        fresh = {report["task_id"]: report for report in runner.load_reports(fresh_dir)}
        for task_id in run_ids:
            figures, reason = compare_task(retained, fresh_dir, reports[task_id], fresh[task_id])
            outcomes += figures
            if reason:
                not_comparable[task_id] = reason
    record = {"schema": RECORD_SCHEMA,
              "note": "Figures of a retained lab run re-executed on this platform; a reproducibility check, "
                      "not a lab finding. Declared wall-clock timing and rounding-level figures are compared for "
                      "presence and structure only; tasks not re-executed or not comparable are never counted as "
                      "matches.",
              "retained": str(args.retained), "platform": platform_identity(), "providers": sorted(providers),
              "summary": summarize(reports, outcomes, skipped, not_comparable), "figures": outcomes,
              "not_reexecuted": skipped, "not_comparable": not_comparable}
    output.mkdir(parents=True, exist_ok=True)
    (output / "figure-check.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    summary = render(record)
    (output / "figure-check.md").write_text(summary, encoding="utf-8")
    print(summary, end="")
    counts = record["summary"]
    if counts["mismatched"] or not counts["compared_figures"]:
        print(f"FAIL: {counts['mismatched']} figures mismatched, {counts['compared_figures']} retained figures "
              f"compared; see {output / 'figure-check.json'}")
        return 3
    print(f"PASS: {counts['compared_figures']} figures compared, none mismatched; {counts['not_reexecuted_tasks']} "
          f"tasks not re-executed, {counts['not_comparable_tasks']} not comparable")
    return 0


if __name__ == "__main__":
    for variable in BLAS_THREADS:  # before NumPy is imported
        os.environ.setdefault(variable, "1")
    raise SystemExit(main())
