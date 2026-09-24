"""Regenerate the retained-workspace compatibility fixture.

``workbench.json`` is a snapshot of one executed bundle per provider-free kind,
built from the committed example inputs, in the shape ``Workbench.serialize``
writes into a version 3 workspace. ``tests/test_retained_compatibility.py``
requires the current code to reopen it without executing a provider and to
reproduce every retained numerical identity.

Run this only when the retained format or a reference changes on purpose, and
say so in the commit: regenerating it silently would hide exactly the
incompatibility the fixture exists to catch. A regeneration also makes the
replay branch of the gate live again, since replay requires the retained
runtime identity, which covers the reference source files. ``manifest.json``
records where each input came from and the identities the test pins.
"""
from __future__ import annotations

import base64
import datetime
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np

from ciw.telemetry import canonical
from ciw.workbench import OPERATIONS, Workbench

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = Path(__file__).resolve().parent
SCHEMA = "ciw.retained-compatibility-fixture.v1"
INPUTS = {
    "energy-accuracy": "examples/energy-accuracy/baseline.json",
    "thermal-observer": "examples/thermal-observer/source.json",
    "machine-manifest": "examples/machine-manifest/source.json",
    "project-graph": "examples/project-graph/make_source.py",
    "uncertainty-validation": "examples/uncertainty-validation/consistent.json",
}


def _script_source(path):
    spec = importlib.util.spec_from_file_location("retained_fixture_" + path.parent.name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.source()


def input_bytes(kind):
    """The exact bytes retained for a kind: a committed file, or a deterministic authoring script."""
    path = ROOT / INPUTS[kind]
    if path.suffix == ".py":
        return canonical(_script_source(path))
    return path.read_bytes()


def build():
    workbench = Workbench()
    bundles = []
    for kind in INPUTS:
        raw = input_bytes(kind)
        source = workbench.add_source({"kind": kind, "label": "Retained compatibility fixture: " + kind,
                                       "bytes_b64": base64.b64encode(raw).decode("ascii")})
        summary = workbench.execute({"operation_id": OPERATIONS[kind], "source_id": source["source_id"]})
        native = workbench.get_bundle(summary["bundle_id"])
        runtime = native["runtimes"][native["steps"][0]["runtime_ref"]]
        bundles.append({"kind": kind, "operation_id": OPERATIONS[kind], "input": INPUTS[kind],
                        "source_id": source["source_id"], "bundle_id": summary["bundle_id"],
                        "execution_id": native["steps"][0]["execution_id"],
                        "result_id": native["steps"][0]["result_id"],
                        "numerical_result_id": native["steps"][0]["numerical_result_id"],
                        "code_sha256": runtime.get("algorithm", runtime)["code_sha256"]})
    manifest = {"schema": SCHEMA, "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "generator": "tests/fixtures/retained/generate.py", "python_version": platform.python_version(),
                "numpy_version": np.__version__, "bundles": bundles}
    return workbench.serialize(), manifest


if __name__ == "__main__":
    retained, manifest = build()
    (FIXTURE / "workbench.json").write_text(json.dumps(retained, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    (FIXTURE / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for entry in manifest["bundles"]:
        print(entry["kind"], entry["bundle_id"])
    print("wrote workbench.json and manifest.json")
