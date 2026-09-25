"""Retain a second-platform figure check from its CI artifact zip as ``lab/figure-platforms/<record-id>/``.

``.github/workflows/figures.yml`` runs ``scripts/check_figures.py --retained
lab`` on ``windows-latest`` and uploads ``figure-check-windows``: the check's
``figure-check.json`` and ``figure-check.md`` with the fresh SVGs and reports.
Download that artifact's zip (``gh run download`` unpacks it; download the zip
itself, for example with ``gh api repos/OWNER/REPO/actions/artifacts/ID/zip >
figure-check-windows.zip``) and pass it with the run's provenance as GitHub
reports it (``gh api repos/OWNER/REPO/actions/runs/RUN`` and ``.../artifacts``):

    python scripts/retain_figure_check.py figure-check-windows.zip --retained lab \\
        --repository OWNER/REPO --workflow .github/workflows/figures.yml --run-id RUN --run-attempt 1 \\
        --head-sha SHA --artifact-id ID --artifact-name figure-check-windows --artifact-digest sha256:HEX

The zip is refused unless its SHA-256 is the declared artifact digest. Only
``figure-check.json``, ``figure-check.md`` and the fresh SVG of each figure the
check reports as a mismatch are retained, beside ``record.json`` (the
provenance and the check's platform) and ``manifest.json`` (the digest of every
file); the copy must pass ``ciw lab figure-platform verify``. T158 reads the
latest record, which ``scripts/check_lab.py`` binds as its
``figure-platform-record`` provider. Review ``git diff lab/figure-platforms``
before committing a record, and retain one from a run against the ``lab/`` it
will be committed with: T158 counts only the entries whose retained figure is
still its run's.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifact", type=Path, help="The downloaded artifact zip")
    parser.add_argument("--retained", type=Path, default=ROOT / "lab")
    parser.add_argument("--repository", required=True, help="OWNER/REPO of the workflow run")
    parser.add_argument("--workflow", required=True, help="Workflow path, e.g. .github/workflows/figures.yml")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--run-attempt", type=int, help="The run attempt, when known")
    parser.add_argument("--head-sha", required=True, help="The commit the run checked out")
    parser.add_argument("--artifact-id", type=int, required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--artifact-digest", required=True, help="sha256:<hex>, as GitHub reports the artifact")
    parser.add_argument("--record-id", help="kebab-case identity (default: <os>-<run id>)")
    parser.add_argument("--date", help="YYYY-MM-DD (default: the date of figure-check.json in the zip)")
    args = parser.parse_args()
    from ciw.lab.figure_platform_records import retain_record
    try:
        result = retain_record(args.artifact, args.retained, repository=args.repository, workflow=args.workflow,
                               run_id=args.run_id, run_attempt=args.run_attempt, head_sha=args.head_sha,
                               artifact_id=args.artifact_id, artifact_name=args.artifact_name,
                               artifact_digest=args.artifact_digest, record_id=args.record_id, date=args.date)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
