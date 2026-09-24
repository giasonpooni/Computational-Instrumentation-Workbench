"""Exercise exchange producers, the pinned SET validator and the read-only CLI.

SET is the instrument-exchange pipeline's provider and comes from its
descriptor; the PPDA and SCR checkouts only produce exchange artifacts for the
tests and come from ``ci/gates.json``.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from provider_checkouts import clone_at, descriptor_pin, extra_pin

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="New directory for the exact checkouts")
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    validator = descriptor_pin("instrument-exchange", "set")
    repository = validator["repository"].removeprefix("https://github.com/")
    set_repo = clone_at(repository, validator["revision"], work / "set")
    producers = {name: extra_pin(name) for name in ("exchange-ppda", "exchange-scr")}
    acquisition = clone_at(producers["exchange-ppda"]["repository"], producers["exchange-ppda"]["revision"], work / "acquisition")
    runtime = clone_at(producers["exchange-scr"]["repository"], producers["exchange-scr"]["revision"], work / "runtime")
    env = {**os.environ, "CIW_SET_REPO": str(set_repo), "CIW_SET_PINNED_REPO": str(set_repo),
           "CIW_ACQUISITION_REPO": str(acquisition), "CIW_RUNTIME_REPO": str(runtime)}
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_exchange.py", "tests/test_exchange_integration.py",
                    "tests/test_exchange_adapter.py"], cwd=ROOT, env=env, check=True, timeout=900)
    print("PASS: exchange producers, pinned SET validator and read-only CLI")


if __name__ == "__main__":
    main()
