#!/usr/bin/env python3
"""Regenerate src/ciw/julia-runtime.json from the packaged Julia environment.

Run this after changing the worker source, Project.toml or Manifest.toml, and
only after the environment has been instantiated and exercised with Julia.
The pin records digests of committed files and the manifest's package
identities; it is a commitment to reviewed bytes, not proof that they work.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / "src/ciw/julia"
PIN = ROOT / "src/ciw/julia-runtime.json"
PACKAGES = ("OrdinaryDiffEqTsit5", "OrdinaryDiffEqCore", "DiffEqBase", "SciMLBase", "SHA", "TOML")
OPERATIONS = ["ciw.julia-oscillator.integrate.v1"]


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_pin(environment: Path = ENVIRONMENT) -> dict:
    manifest = tomllib.loads((environment / "Manifest.toml").read_text(encoding="utf-8"))
    packages = {}
    for name in PACKAGES:
        entry, = manifest["deps"][name]
        packages[name] = {"uuid": entry["uuid"], "version": entry["version"], "git_tree_sha1": entry.get("git-tree-sha1")}
    return {
        "schema": "ciw.julia-runtime-pin.v1",
        "protocol_version": 1,
        "julia_version": manifest["julia_version"],
        "operations": OPERATIONS,
        "worker": {"path": "worker/oscillator_worker.jl", "sha256": digest(environment / "worker/oscillator_worker.jl")},
        "project": {"path": "Project.toml", "sha256": digest(environment / "Project.toml")},
        "manifest": {"path": "Manifest.toml", "sha256": digest(environment / "Manifest.toml"),
                     "project_hash": manifest["project_hash"]},
        "packages": packages,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the committed pin differs from the packaged files")
    args = parser.parse_args(argv)
    pin = build_pin()
    rendered = json.dumps(pin, indent=2, sort_keys=True) + "\n"
    if args.check:
        if PIN.read_text(encoding="utf-8") != rendered:
            print("julia-runtime.json differs from the packaged Julia environment")
            return 1
        print("julia-runtime.json matches the packaged Julia environment")
        return 0
    PIN.write_text(rendered, encoding="utf-8")
    print(json.dumps(pin, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
