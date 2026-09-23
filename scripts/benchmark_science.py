"""Measure native solver cost and cross-backend agreement on the shipped experiment specifications.

For every example specification the nominal jobs (ensemble members excluded) are
run on every backend that binds the specification's solver, ``--repeats`` times
each. The report gives median wall and process-CPU time per job and backend,
the result content identity, whether repeated runs were bitwise identical, and
the largest absolute difference between backends on each job's numerical
leaves. Energy is not measured here; see ``ciw energy`` and the energy channel
of ``ciw.science.backends``.

This is a measurement of this machine and these builds, not a performance
claim: timings depend on CPU, load, Python/NumPy and compiler versions, and
the first call on a compiled or worker backend includes its start-up cost,
which is reported separately and excluded from the medians.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ciw.science._common import Refusal, content_identity, plain  # noqa: E402
from ciw.science.experiment import compile_spec  # noqa: E402
from ciw.science.solvers import default_registry  # noqa: E402


def implementations(declaration) -> dict:
    """Backend bindings when the registry declares several, else the native one."""
    bound = getattr(declaration, "implementations", None)
    return dict(bound) if bound else {"python-numpy-cpu": declaration.implementation}


def leaves(value, path=""):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from leaves(value[key], f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield path, float(value)


def max_difference(first: dict, second: dict) -> dict:
    a, b = dict(leaves(first)), dict(leaves(second))
    shared = sorted(a.keys() & b.keys())
    if not shared:
        return {"compared": 0, "max_abs": None, "max_rel": None, "missing": len(a.keys() ^ b.keys())}
    abs_diff = [abs(a[key] - b[key]) for key in shared]
    rel_diff = [abs(a[key] - b[key]) / max(abs(a[key]), 1e-300) for key in shared if a[key] != 0]
    worst = shared[int(np.argmax(abs_diff))]
    return {"compared": len(shared), "max_abs": max(abs_diff), "max_rel": max(rel_diff) if rel_diff else 0.0,
            "worst_path": worst, "missing": len(a.keys() ^ b.keys())}


def time_call(function, job):
    wall, cpu = time.perf_counter_ns(), time.process_time_ns()
    try:
        result = function(job)
        error = None
    except Refusal as exc:
        result, error = None, exc.to_dict()
    return result, error, (time.perf_counter_ns() - wall) / 1e9, (time.process_time_ns() - cpu) / 1e9


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples", type=Path, default=ROOT / "examples" / "science" / "experiments")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-jobs", type=int, default=4, help="nominal jobs per specification")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.max_jobs < 1:
        parser.error("--repeats and --max-jobs must be positive")
    registry = default_registry()
    rows = []
    for path in sorted(args.examples.glob("*.json")):
        spec = json.loads(path.read_bytes())
        try:
            plan = compile_spec(spec, registry)
        except Refusal as exc:
            rows.append({"spec": path.name, "refused": exc.to_dict()})
            continue
        declaration = registry.get(plan["solver_id"])
        bindings = implementations(declaration)
        jobs = [job for job in plan["jobs"] if job.get("ensemble", {}).get("role") != "member"][: args.max_jobs]
        for job in jobs:
            row = {"spec": path.name, "experiment_id": plan["experiment_id"], "job_id": job["job_id"],
                   "solver_id": plan["solver_id"], "steps": job.get("steps"), "backends": {}}
            results = {}
            for backend_id, function in sorted(bindings.items()):
                first, error, first_wall, _ = time_call(function, job)
                if error is not None:
                    row["backends"][backend_id] = {"refused": error}
                    continue
                walls, cpus, identities = [], [], {content_identity(plain(first))}
                for _ in range(args.repeats):
                    result, error, wall, cpu = time_call(function, job)
                    if error is not None:
                        break
                    walls.append(wall)
                    cpus.append(cpu)
                    identities.add(content_identity(plain(result)))
                results[backend_id] = plain(first)
                row["backends"][backend_id] = {
                    "first_call_wall_s": first_wall, "median_wall_s": statistics.median(walls) if walls else None,
                    "median_cpu_s": statistics.median(cpus) if cpus else None,
                    "result_identity": sorted(identities)[0], "repeat_bitwise_identical": len(identities) == 1,
                    "per_step_us": (statistics.median(walls) / job["steps"] * 1e6
                                    if walls and job.get("steps") else None)}
            names = sorted(results)
            row["agreement"] = {f"{a}|{b}": dict(max_difference(results[a], results[b]),
                                                bitwise=content_identity(results[a]) == content_identity(results[b]))
                                for i, a in enumerate(names) for b in names[i + 1:]}
            rows.append(row)
    report = {"schema": "ciw.science-benchmark.v1",
              "machine": {"python": platform.python_version(), "numpy": np.__version__, "system": platform.system(),
                          "machine": platform.machine(), "processor": platform.processor() or None},
              "repeats": args.repeats, "rows": rows,
              "note": "medians exclude the first call; energy is not measured by this script"}
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=lambda v: None if (
        isinstance(v, float) and not math.isfinite(v)) else str(v))
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
