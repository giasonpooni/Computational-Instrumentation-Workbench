"""Fail-closed native Julia/JuMP gate; fixtures or mocks cannot satisfy this script."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from ciw import design_adjustment as design, jump_design  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", type=Path, required=True)
    parser.add_argument("--depot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        runs = []
        for index, name in enumerate(("interior", "boundary", "zero-response", "fixed-bound", "interior")):
            problem = design.load_problem(ROOT/"examples"/"design-adjustment"/(name+".json"))
            study = jump_design.run_study(problem, args.julia, depot=args.depot, timeout=300)
            if study["check"]["status"] != "accepted":
                raise ValueError(f"{name}: exact certificate rejected")
            # Independent closed-form reference for this scalar convex box QP.
            def q(field):
                v = problem[field]
                return Fraction(v["numerator"], v["denominator"])
            x, target, regularization = q("x"), q("target"), q("regularization")
            expected = min(max(2*x*(target-x*x)/(4*x*x+regularization), q("lower")), q("upper"))
            if abs(Fraction.from_float(float(study["solver"]["delta"]))-expected) > Fraction(1, 100_000_000):
                raise ValueError(f"{name}: raw solver candidate differs from the independent reference")
            path = args.output_dir/f"{index}-{name}.json"
            jump_design.save_study(path, study)
            with patch.object(jump_design, "run_study", side_effect=AssertionError("Offline open ran Julia")), \
                 patch.object(design, "check_candidate", side_effect=AssertionError("Offline open rechecked")):
                if jump_design.load_study(path) != study:
                    raise ValueError("Offline reopen differs from retained study")
            runs.append(study)
        if (runs[0]["producer_invocation_id"] == runs[-1]["producer_invocation_id"]
                or runs[0]["problem_digest"] != runs[-1]["problem_digest"]):
            raise ValueError("Explicit repeat failed occurrence/content separation")
        summary = {"native_studies": len(runs), "offline_reopens": len(runs),
                   "independent_reference_tolerance": "1/100000000 on raw solver delta",
                   "runs": [{"invocation": r["producer_invocation_id"], "study_id": r["study_id"],
                             "solver_delta": r["solver"]["delta"], "checks": r["check"]["checks"],
                             "gap": r["check"]["candidate"]["gap"], "timing": r["timing"]} for r in runs],
                   "runtime": runs[0]["runtime"], "sp1_verification": "not_performed"}
        (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
        print("5 native studies, 5 offline reopens, independent reference and explicit repeat passed.")
        return 0
    except (OSError, ValueError, AssertionError) as exc:
        print("Native design gate failed:", exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
