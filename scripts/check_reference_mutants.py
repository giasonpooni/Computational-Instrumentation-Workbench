"""Mutation gate for the shared reference lifecycle and the retained workbench.

Each mutant removes one identity check from ``src/ciw/reference_workflow.py``
or one record check from ``src/ciw/workbench.py`` and the guarding tests must
fail; a mutant that survives means a check is no longer guarded by any test.
The mutated file is restored whatever happens. Run from the repository root:

    python scripts/check_reference_mutants.py            # both targets
    python scripts/check_reference_mutants.py workbench  # one target

Exit status 0 when every mutant is killed, 1 when one survives or a mutant's
pattern no longer matches the source (the gate itself then needs updating).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_TESTS = ["tests/test_reference_identity_checks.py", "tests/test_retained_compatibility.py",
         "tests/test_reference_kernel.py", "tests/test_review_fixes.py", "tests/test_thermal_workflow.py",
         "tests/test_machine_workflow.py", "tests/test_project_workflow.py",
         "tests/test_uncertainty_validation.py", "tests/test_energy_workflow.py"]
REFERENCE_MUTANTS = {
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

WORKBENCH_TESTS = ["tests/test_workbench_record_checks.py", "tests/test_retained_compatibility.py",
                   "tests/test_workspace_verify.py", "tests/test_review_fixes.py", "tests/test_audit_hardening.py",
                   "tests/test_workbench_session.py", "tests/test_energy_workflow.py", "tests/test_replay.py"]
WORKBENCH_MUTANTS = {
    "source kind unchecked": ('    if source is None or source["kind"] != record["kind"]:\n', '    if source is None:\n'),
    "bundle id and source bytes unchecked": (
        '    if (native["bundle_digest"] != record["bundle_id"] or\n'
        '            raw != base64.b64decode(source["bytes_b64"], validate=True)):\n',
        '    if False:\n'),
    "verification presence unchecked": ('    if "verification" not in native:\n', '    if False:\n'),
    "second receipt allowed": ('    if not isinstance(receipts, list) or len(receipts) > 1:\n',
                               '    if not isinstance(receipts, list):\n'),
    "receipt replayed digest unchecked": (
        '                receipt["replayed_bundle_digest"] != native["bundle_digest"] or\n', '                False or\n'),
    "receipt replay id unchecked": (
        '                receipt["replay_id"] != _digest({k: v for k, v in receipt.items() if k != "replay_id"})):\n',
        '                False):\n'),
    "restored source content unchecked": (
        '                if _canonical(source) != _canonical(retained) or source["source_id"] in restored._sources:\n',
        '                if False:\n'),
    "duplicate bundle allowed": (
        '                if record["bundle_id"] in restored._bundles:\n'
        '                    raise ValueError("Duplicate retained bundle identity")\n',
        '                pass\n'),
    "identity collision ignored": (
        '            if identity in self._identities and self._identities[identity] != claim:\n', '            if False:\n'),
    "upstream requirement dropped": (
        '    elif record["upstream_bundle_id"] is None:\n'
        '        raise ValueError("This workflow needs an explicitly selected upstream bundle")\n',
        '    elif False:\n        pass\n'),
}
TARGETS = {
    "reference": (ROOT / "src" / "ciw" / "reference_workflow.py", REFERENCE_TESTS, REFERENCE_MUTANTS),
    "workbench": (ROOT / "src" / "ciw" / "workbench.py", WORKBENCH_TESTS, WORKBENCH_MUTANTS),
}


def run(target: Path, tests: list[str], mutants: dict[str, tuple[str, str]]) -> list[str]:
    original = target.read_text(encoding="utf-8")
    outcomes = {}
    try:
        for name, (old, new) in mutants.items():
            if original.count(old) != 1:
                outcomes[name] = "pattern not found"
                print(f"  {name:42s} PATTERN NOT FOUND", flush=True)
                continue
            target.write_text(original.replace(old, new, 1), encoding="utf-8")
            completed = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
                                        "-x", *tests], cwd=ROOT, capture_output=True, text=True, timeout=1800)
            outcomes[name] = "killed" if completed.returncode != 0 else "survived"
            print(f"  {name:42s} {outcomes[name]}", flush=True)
    finally:
        target.write_text(original, encoding="utf-8")
    survivors = sorted(name for name, outcome in outcomes.items() if outcome != "killed")
    print(f"{target.name}: mutants {len(mutants)} | killed {sum(o == 'killed' for o in outcomes.values())} |",
          "survivors:", survivors or "none", flush=True)
    return survivors


def main(argv: list[str]) -> int:
    chosen = argv or list(TARGETS)
    unknown = sorted(set(chosen) - set(TARGETS))
    if unknown:
        print("unknown targets:", unknown, "| known:", sorted(TARGETS))
        return 2
    survivors = []
    for name in chosen:
        print(f"{name}:", flush=True)
        survivors += run(*TARGETS[name])
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
