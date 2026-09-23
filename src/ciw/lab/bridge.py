"""Evidence labels for results retained in an existing CIW workspace.

The workspace is first validated exactly as ``Session.from_workspace`` does.
That binds no provider and creates no execution record, but CIW's built-in
validators may recompute deterministic offline analyses to check retained data
(the energy-accuracy validator re-runs its log analysis). Each retained run,
operation result, workbench bundle
and replay receipt then receives findings labelled by ``ciw.lab.evidence``.
The labels are a derived projection: nothing is written into sealed records.

Rules applied:

* Results computed by a runtime identity with a revision and source tree are
  ``provider_backed``; results over generated inputs are ``synthetic``;
  built-in analyses over caller-declared data are ``not_established``, which
  matches CIW's own ``verification_status: not_verified``.
* Replay receipts with a numerical match are a same-runtime reproduction
  check (``numerically_verified`` for replay determinism, never independent).
* Every retained item gets a physical-truth claim. It is ``hardware_measured``
  only for an energy log that declares ``physical_measurement`` with device,
  digest, clock and calibration fields, and even then the recorder's assertion
  is not authenticated. Everything else is ``not_established``.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile

from ..session import Session, read_json
from .evidence import LABELS, finding

SCHEMA = "ciw.lab-workspace-classification.v1"
SYNTHETIC_KEYS = frozenset({"origin", "claim_scope", "observations", "basis", "source", "coverage"})


def _providers(value, found=None):
    """Runtime identities that pin a revision and source tree, found anywhere in a record."""
    found = [] if found is None else found
    if isinstance(value, dict):
        if isinstance(value.get("revision"), str) and isinstance(value.get("source_tree"), str):
            found.append({"repository": str(value.get("module") or value.get("repository") or "pinned provider"),
                          "revision": value["revision"], "source_tree": value["source_tree"], "executed": True})
        for child in value.values():
            _providers(child, found)
    elif isinstance(value, list):
        for child in value:
            _providers(child, found)
    return found


def _synthetic(value) -> bool:
    """True when a record declares synthetic inputs under a provenance-bearing key."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in SYNTHETIC_KEYS and isinstance(child, str) and "synthetic" in child.lower():
                return True
            if _synthetic(child):
                return True
    elif isinstance(value, list):
        return any(_synthetic(child) for child in value)
    return False


def _result_basis(record, synthetic_input, name):
    providers = _providers(record)
    if providers:
        return {"provider": providers[0], "notes": f"{len(providers)} pinned runtime identities in the record"}
    if synthetic_input:
        return {"generator": {"name": name}}
    return {}


def _energy_acquisition(source):
    """Acquisition record for an energy log that declares a physical measurement."""
    if source.get("schema") != "ciw.energy-accuracy-log.v1" or source.get("origin") != "physical_measurement":
        return None
    sensor, clock = source["sensor"], source["clock"]
    starts = [phase["start_ns"] for phase in source.get("phases", []) if isinstance(phase, dict)]
    accuracy = sensor.get("accuracy_j")
    return {"device": f"{sensor.get('backend')}:{sensor.get('device_uuid')}",
            "raw_sha256": source["log_digest"].removeprefix("sha256:"),
            "acquired_at": f"{clock['epoch_id']}+{min(starts) if starts else clock['monotonic_origin_ns']}ns",
            "calibration": ("declared_accuracy_j=" + repr(accuracy)) if accuracy is not None
            else "vendor_counter_accuracy_not_declared"}


def classify_workspace(path) -> dict:
    """Validate a saved workspace without provider execution and label every retained result."""
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix="ciw-lab-classify-") as directory:
        Session.from_workspace(path, Path(directory))
    saved = read_json(path)
    run = saved["run"]
    provenance = run["metadata"].get("provenance", {})
    synthetic_run = _synthetic(provenance) or "generator" in provenance
    items = []

    def item(kind, identity, findings):
        items.append({"kind": kind, "identity": identity, "findings": findings,
                      "labels": sorted({f["evidence_status"] for f in findings})})

    run_basis = {"generator": {"name": str(provenance.get("generator", "declared synthetic source"))}} if synthetic_run else {}
    item("run", run["evidence_id"], [
        finding(f"Run {run['run_id']} content", "numerical", run["evidence_id"], run_basis,
                expected_not_established=not synthetic_run),
        finding(f"Run {run['run_id']} represents a physical acquisition", "physical", None, {}),
    ])
    results = saved.get("results", {})
    for result in (results.values() if isinstance(results, dict) else results):
        identity = result.get("result_id", "legacy-result")
        basis = _result_basis(result.get("runtime") or {}, synthetic_run, f"{result.get('operation_id')} over {run['run_id']}")
        item("operation_result", identity, [
            finding(f"{result.get('operation_id')} result {identity}", "numerical", result.get("record_digest"), basis,
                    expected_not_established=not basis),
            finding(f"{result.get('operation_id')} result {identity} is physically valid", "physical", None, {}),
        ])
    workbench = saved.get("workbench") or {}
    sources = {s["source_id"]: s for s in workbench.get("sources", [])}
    for bundle in workbench.get("bundles", []):
        native, kind = bundle["native"], bundle["kind"]
        source_record = sources.get(bundle.get("source_id"))
        try:
            source = json.loads(base64.b64decode(source_record["bytes_b64"])) if source_record else {}
        except (ValueError, KeyError):
            source = {}
        synthetic_input = _synthetic(source) or _synthetic(native.get("configuration")) or _synthetic(native.get("steps"))
        basis = _result_basis({"runtimes": native.get("runtimes"), "steps": native.get("steps")}, synthetic_input, f"{kind} source")
        findings = [finding(f"{kind} bundle {bundle['bundle_id']} numerical result", "numerical", bundle["bundle_id"], basis,
                            expected_not_established=not basis)]
        acquisition = _energy_acquisition(source) if isinstance(source, dict) else None
        findings.append(finding(f"{kind} bundle {bundle['bundle_id']} rests on a physical measurement", "physical",
                                source.get("log_digest") if acquisition else None,
                                {"acquisition": acquisition, "notes": "recorder assertion; hardware provenance not authenticated"}
                                if acquisition else {}))
        for receipt in native.get("replay_receipts", []) or []:
            matched = receipt.get("numerical_match") is True and receipt.get("verification", {}).get("outcome") == "passed"
            findings.append(finding(
                f"{kind} replay {receipt.get('replay_id')} reproduces the retained numerical result", "provenance",
                matched, {"checks": [{"reference_kind": "self_convergence", "reference": "same-runtime fresh reproduction",
                                      "observed": 0.0 if matched else 1.0, "tolerance": 0.0, "passed": matched}]}))
        item("workbench_bundle", bundle["bundle_id"], findings)
    counts = {label: 0 for label in LABELS}
    for entry in items:
        for record in entry["findings"]:
            counts[record["evidence_status"]] += 1
    return {"schema": SCHEMA, "workspace": str(path), "validated_without_provider_execution": True,
            "validation_scope": "Session.from_workspace; built-in offline analyses may be recomputed to check retained data",
            "items": items, "label_counts": counts,
            "note": "Derived projection; retained records are unchanged and replay is never independent verification."}
