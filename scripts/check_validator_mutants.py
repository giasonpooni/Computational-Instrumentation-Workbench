"""Mutation gate for the source and result validators.

Every ``if <condition>: raise ...`` of a target module is weakened one clause
at a time (the whole condition, or one operand of an ``or``, becomes
``False``) and the module's guarding tests must fail. A surviving mutant
means a check that no test reaches. Mutants run in parallel git worktrees
that carry the live tree's copy of the target and its tests, so the working
tree is never modified. Run from the repository root:

    python scripts/check_validator_mutants.py                 # every target
    python scripts/check_validator_mutants.py energy_records  # one target
    python scripts/check_validator_mutants.py --jobs 2

Clauses listed in EQUIVALENT are redundant with a neighbouring clause for
every input the module can receive (for example a finiteness test beside a
range comparison that NaN also fails); their survival is reported, not
counted. Exit status 0 when every other mutant is killed, 1 otherwise.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "energy_records": ("src/ciw/energy_records.py",
                       ["tests/test_energy_log_checks.py", "tests/test_energy_records.py"]),
    "thermal_contract": ("src/ciw/thermal_contract.py",
                         ["tests/test_thermal_contract_checks.py", "tests/test_thermal_contract.py"]),
    "uncertainty_validation": ("src/ciw/uncertainty_validation.py",
                               ["tests/test_uncertainty_source_checks.py", "tests/test_uncertainty_validation.py"]),
    "consistency_math": ("src/ciw/consistency_math.py",
                         ["tests/test_consistency_math_checks.py", "tests/test_consistency_math.py"]),
    "machine_workflow": ("src/ciw/machine_workflow.py",
                         ["tests/test_machine_source_checks.py", "tests/test_machine_workflow.py"]),
    "project_workflow": ("src/ciw/project_workflow.py",
                         ["tests/test_project_source_checks.py", "tests/test_project_workflow.py"]),
    "free_energy_math": ("src/ciw/free_energy_math.py",
                         ["tests/test_free_energy_math_checks.py", "tests/test_free_energy_math.py"]),
    "covariance": ("src/ciw/core/covariance.py",
                   ["tests/test_covariance_artifact_checks.py", "tests/test_covariance_records.py"]),
    "session": ("src/ciw/session.py",
                ["tests/test_session_checks.py", "tests/test_audit_hardening.py", "tests/test_replay.py",
                 "tests/test_protocol.py"]),
}
EQUIVALENT = {
    "thermal_contract": {
        # A numpy scalar is the only non-(int, float) value equal to 1e-10, and the request arrives as JSON.
        'type(choice["tie_tolerance_nats"]) not in (int, float)',
        # sorted(permutation) == [1, 2] already forces two entries.
        "len(permutation) != 2",
        # NaN and infinities also fail the range comparison beside this clause.
        "not math.isfinite(value)",
        # np.array_equal against the 2x2 identity already rejects any other row count.
        "len(h) != 2",
    },
    "uncertainty_validation": {
        # The shared covariance validator rejects negative variances upstream, and a zero smallest
        # eigenvalue also fails the condition-number clause beside this one.
        "eigenvalues.min() <= 0",
    },
    "consistency_math": {
        # NaN and infinities also fail the unit-interval comparison beside this clause.
        "not math.isfinite(x)",
    },
    "free_energy_math": {
        # Every array leaf has already passed the finite scalar check.
        "not np.all(np.isfinite(result))",
        # Guards on matrices and results computed from admitted (finite, bounded, well conditioned)
        # inputs: no admitted input reaches them, so no input-level test can distinguish them.
        "not np.all(np.isfinite(matrix))",
        "np.max(np.abs(matrix - matrix.T)) > 128 * np.finfo(float).eps * max(1.0, float(np.linalg.norm(matrix)))",
        "not np.all(np.isfinite(eigenvalues))",
        "not all(math.isfinite(number) for number in [value, *terms.values()])",
        "np.any(eigenvalues <= 0)",
        "not math.isfinite(result)",
        "result < 0",
        "not np.all(np.isfinite(proposed_mean))",
    },
    "covariance": {
        # Any identity that is not the recomputed content identity already fails the equality clause.
        '_CONTENT_ID.fullmatch(artifact["covariance_id"]) is None',
        # A nonfinite correlation entry is refused before the eigenvalues are computed.
        "not np.all(np.isfinite(eigenvalues))",
    },
    "session": {
        # calibration_status repeats this check with the same message before the value is used.
        '"evaluated_at" in payload and not isinstance(payload["evaluated_at"], str)',
        # datetime.fromisoformat never yields a tzinfo whose utcoffset() is None, and a missing
        # tzinfo is exactly the case the other clause names.
        "parsed.tzinfo is None",
        "parsed.utcoffset() is None",
        # Any suffix of another length either fails uuid parsing or, in a hyphenated or braced
        # spelling, differs from its canonical hex, which the next check refuses.
        "len(value) != len(prefix) + 32",
        # validate_execution already requires a completed execution's result to name it back, so a
        # retained operation result cannot reach these clauses with a refused or foreign execution.
        'execution["status"] != "completed"',
        'execution["result_id"] != result["result_id"]',
    },
}


def mutants_for(source: str) -> list[tuple[str, str, str]]:
    """(label, clause text, mutated source) for every guarded raise."""
    tree = ast.parse(source)
    lines = source.encode("utf-8").split(b"\n")

    def replaced(node: ast.AST) -> str:
        first, last = node.lineno - 1, node.end_lineno - 1
        new = list(lines)
        head = new[first][:node.col_offset]
        tail = new[last][node.end_col_offset:]
        new[first] = head + b"False" + tail
        del new[first + 1:last + 1]
        return b"\n".join(new).decode("utf-8")

    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If) and len(node.body) == 1 and isinstance(node.body[0], ast.Raise)):
            continue
        test = node.test
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
            for index, clause in enumerate(test.values):
                text = ast.get_source_segment(source, clause) or "?"
                out.append((f"line {node.lineno} clause {index}", text, replaced(clause)))
        else:
            text = ast.get_source_segment(source, test) or "?"
            out.append((f"line {node.lineno}", text, replaced(test)))
    return out


def run_target(name: str, jobs: int, base: Path) -> tuple[int, list[tuple[str, str]], list[tuple[str, str]]]:
    relative, tests = TARGETS[name]
    source = (ROOT / relative).read_text(encoding="utf-8")
    mutants = mutants_for(source)
    trees = []
    for index in range(min(jobs, len(mutants)) or 1):
        tree = base / f"{name}-{index}"
        subprocess.run(["git", "worktree", "add", "--detach", "-q", str(tree), "HEAD"], cwd=ROOT, check=True)
        for item in [relative, *tests]:
            (tree / item).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / item, tree / item)
        trees.append(tree)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    # A failing guarding test would "kill" every mutant; the unmutated module must pass first.
    baseline = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "-x", *tests],
                              cwd=trees[0], capture_output=True, text=True, timeout=1800, env=env)
    if baseline.returncode != 0:
        for tree in trees:
            subprocess.run(["git", "worktree", "remove", "--force", str(tree)], cwd=ROOT, check=False)
        summary = baseline.stdout.strip().splitlines()[-1] if baseline.stdout.strip() else baseline.stderr.strip()[-200:]
        print(f"{name}: guarding tests fail before any mutation ({summary}); nothing measured", flush=True)
        return 0, [("baseline", "guarding tests fail unmutated")], []

    def run_bucket(bucket: list[tuple[Path, tuple[str, str, str]]]) -> list[tuple[str, str, str]]:
        results = []
        for tree, (label, clause, mutated) in bucket:
            target = tree / relative
            target.write_text(mutated, encoding="utf-8")
            try:
                completed = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
                                            "-x", *tests], cwd=tree, capture_output=True, text=True, timeout=1800, env=env)
            finally:
                target.write_text(source, encoding="utf-8")
            results.append((label, clause, "killed" if completed.returncode != 0 else "survived"))
        return results

    buckets: list[list] = [[] for _ in trees]
    for index, mutant in enumerate(mutants):
        buckets[index % len(trees)].append((trees[index % len(trees)], mutant))
    outcomes: list[tuple[str, str, str]] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(trees)) as pool:
            for chunk in pool.map(run_bucket, buckets):
                outcomes.extend(chunk)
    finally:
        for tree in trees:
            subprocess.run(["git", "worktree", "remove", "--force", str(tree)], cwd=ROOT, check=False)
    equivalent = EQUIVALENT.get(name, set())
    survivors = [(label, clause) for label, clause, outcome in outcomes if outcome == "survived" and clause not in equivalent]
    expected = [(label, clause) for label, clause, outcome in outcomes if outcome == "survived" and clause in equivalent]
    killed = sum(outcome == "killed" for _, _, outcome in outcomes)
    print(f"{name}: mutants {len(outcomes)} | killed {killed} | equivalent survivors {len(expected)} | unexpected survivors {len(survivors)}", flush=True)
    for label, clause in survivors:
        print(f"  SURVIVED {label}: {clause}", flush=True)
    return killed, survivors, expected


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="*", help="module names (default: every target): " + ", ".join(TARGETS))
    parser.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    args = parser.parse_args(argv)
    unknown = sorted(set(args.targets) - set(TARGETS))
    if unknown:
        parser.error("unknown targets: " + ", ".join(unknown))
    chosen = args.targets or list(TARGETS)
    failed = False
    with tempfile.TemporaryDirectory(prefix="ciw-validator-mutants-") as scratch:
        for name in chosen:
            _, survivors, _ = run_target(name, args.jobs, Path(scratch))
            failed = failed or bool(survivors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
