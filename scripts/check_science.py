"""Acceptance gate for the scientific foundation: full synthetic bench plus offline round trip.

Runs every shipped experiment (including the 245-job sphere ensemble), replays
them, and then checks that a signed evidence bundle imports into a fresh ledger
whose report is byte-identical to one rendered from the source ledger.
Exit status is nonzero on any failed expectation.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

from ciw.science import ed25519
from ciw.science.bench import bench_key, run_bench
from ciw.science.bundle import import_bundle, inspect_bundle
from ciw.science.ledger import Ledger
from ciw.science.report import build, markdown

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "science"


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAILED: {message}")
    print(f"ok  {message}")


def main() -> int:
    with tempfile.TemporaryDirectory() as scratch:
        output = Path(scratch) / "bench"
        summary = run_bench(output, examples=EXAMPLES)
        expect(summary["ledger"]["verified"], "bench ledger verifies")
        expect(len(summary["experiments"]) >= 7, "every shipped experiment ran")
        for name, experiment in sorted(summary["experiments"].items()):
            counts = experiment["verifications"]
            expect(counts["failed"] == 0 and counts["incomplete"] == 0, f"{name}: every verification passed")
            expect(experiment["replay"]["diverged"] == 0 and experiment["replay"]["refused"] == 0,
                   f"{name}: replay reproduced every result")
            expect(experiment["status"]["measurement"].startswith("no physical measurement"),
                   f"{name}: synthetic evidence is not reported as a measurement")
        expect(summary["design"]["recommendation"] == "cylinder-r50", "design recommends the cylinder coupon")
        expect(summary["authority"]["actuate"]["authorized"] is False, "actuation is refused by the read-only gate")
        expect(summary["hardware"]["control"] == "refused", "hardware control path is refused")
        expect(all(summary["audit"]["probe_detected"]), "every ledger mutation is detected")
        expect(summary["bundle"] == {"ok": True, "signature": "valid"}, "signed bundle inspects clean")

        trusted = {"bench-bundle": ed25519.public_key(bench_key("bundle"))}
        imported = import_bundle(output / "evidence.ciwb", Path(scratch) / "imported", trusted_keys=trusted)
        source = Ledger.open(output / "ledger")
        expect(imported.head() == source.head(), "imported ledger has the source head")
        expect(markdown(build(imported)) == markdown(build(source)), "report is byte-identical after the round trip")
        untrusted = inspect_bundle(output / "evidence.ciwb", trusted_keys={})
        expect(untrusted["signature"]["status"] == "untrusted_key" if isinstance(untrusted["signature"], dict)
               else untrusted["signature"] == "untrusted_key", "an unknown signer is not trusted")
        print(json.dumps({"entries": summary["ledger"]["entries"], "head": summary["ledger"]["head"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
