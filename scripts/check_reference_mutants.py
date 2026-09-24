"""Mutation gate for the shared reference lifecycle.

Each mutant removes one identity check from ``src/ciw/reference_workflow.py``
and the reference tests must fail; a mutant that survives means a check is no
longer guarded by any test. The file is restored whatever happens. Run from
the repository root:

    python scripts/check_reference_mutants.py

Exit status 0 when every mutant is killed, 1 when one survives or a mutant's
pattern no longer matches the source (the gate itself then needs updating).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "ciw" / "reference_workflow.py"
TESTS = ["tests/test_reference_identity_checks.py", "tests/test_retained_compatibility.py",
         "tests/test_reference_kernel.py", "tests/test_review_fixes.py", "tests/test_thermal_workflow.py",
         "tests/test_machine_workflow.py", "tests/test_project_workflow.py",
         "tests/test_uncertainty_validation.py", "tests/test_energy_workflow.py"]
MUTANTS = {
    "input_refs unchecked": ('                step["input_refs"] != [evidence_id] or\n', '                False or\n'),
    "authority unchecked": ('                result["authority"] != self.AUTHORITY or\n', '                False or\n'),
    "same occurrence allowed in verification": (
        '        if (old["execution_id"] == new["execution_id"] or old["result_id"] == new["result_id"] or\n',
        '        if (False or\n'),
    "bundle digest unchecked": ('                    bundle["bundle_digest"] != _bundle_digest(bundle) or\n', '                    False or\n'),
    "code digest shape unchecked": (
        'not isinstance(algorithm["code_sha256"], str) or not CODE_DIGEST.fullmatch(algorithm["code_sha256"]) or',
        'False or'),
    "self-referencing receipt allowed": (
        '                        receipt["source_bundle_digest"] == bundle["bundle_digest"] or\n', '                        False or\n'),
    "runtime identity never compared": ('            if current != retained:\n', '            if False:\n'),
    "request digest unchecked": ('                step["request_sha256"] != digest(request)):\n', '                False):\n'),
    "numerical identity unchecked": (
        '        if step["numerical_result"] != numerical or step["numerical_result_id"] != digest(numerical):\n',
        '        if False:\n'),
    "session id shape unchecked": (
        '                    not isinstance(bundle["session_id"], str) or not SESSION_ID.fullmatch(bundle["session_id"])):\n',
        '                    False):\n'),
    "tolerance loosened": ('REL_TOL = 1e-9\n', 'REL_TOL = 1e-3\n'),
    "kernel probe constant": ('    return sha256(raw).hexdigest()\n', '    return "0" * 64\n'),
}


def main() -> int:
    original = TARGET.read_text(encoding="utf-8")
    outcomes = {}
    try:
        for name, (old, new) in MUTANTS.items():
            if original.count(old) != 1:
                outcomes[name] = "pattern not found"
                print(f"  {name:42s} PATTERN NOT FOUND", flush=True)
                continue
            TARGET.write_text(original.replace(old, new, 1), encoding="utf-8")
            completed = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
                                        "-x", *TESTS], cwd=ROOT, capture_output=True, text=True, timeout=1800)
            outcomes[name] = "killed" if completed.returncode != 0 else "survived"
            print(f"  {name:42s} {outcomes[name]}", flush=True)
    finally:
        TARGET.write_text(original, encoding="utf-8")
    survivors = sorted(name for name, outcome in outcomes.items() if outcome != "killed")
    print("mutants:", len(MUTANTS), "| killed:", sum(o == "killed" for o in outcomes.values()), "| survivors:", survivors or "none")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
