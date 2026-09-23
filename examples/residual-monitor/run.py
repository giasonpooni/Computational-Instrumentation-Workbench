"""Monitor an explicitly ordered set of retained calibrated window bundles.

Example: python examples/residual-monitor/run.py --stack-root /trusted/providers \
    --window window-1.json window-2.json window-3.json --output /tmp/residual-monitor

Each input is an existing calibrated-window or acquired-calibrated-window native
bundle. The same reference prior is required; this example never fits a baseline
or feeds a posterior into the next window. The FDIR and OIT checkout pins are
checked before execution. Replay produces fresh monitor execution identities.
"""
import argparse
from copy import deepcopy
from pathlib import Path

from ciw import residual_monitor as monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path, required=True)
    parser.add_argument("--window", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundles = [monitor._json(path.read_bytes()) for path in args.window]
    source = {"schema": monitor.SOURCE_SCHEMA, "experiment_id": "declared-fixed-reference-drift-monitor",
              "window_bundle_ids": [bundle["bundle_digest"] for bundle in bundles],
              "configuration": deepcopy(monitor.DEFAULT_CONFIGURATION)}
    raw = monitor.canonical(source)
    repositories = {role: args.stack_root / role for role in monitor.ROLES}
    original = monitor.create_session(raw, {bundle["bundle_digest"]: bundle for bundle in bundles}, repositories)
    replay = monitor.replay_session(original, repositories)["session"]
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("source", raw), ("original", monitor.canonical(original)), ("replay", monitor.canonical(replay))):
        (args.output / (name + ".json")).write_bytes(data)
    print("Retained residual monitoring and fresh replay passed; physical drift and alarm probability remain unestablished.")


if __name__ == "__main__":
    main()
