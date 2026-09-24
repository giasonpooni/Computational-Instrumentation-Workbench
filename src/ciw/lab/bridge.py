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
* Generated inputs are recognised only from structured declarations: a run
  provenance ``generator`` that is a known CIW generator identity
  (:data:`KNOWN_GENERATORS`, today ``ciw.instruments.make_demo_run``) or one
  of the exact values CIW's own validators pin (``origin: synthetic_fixture``
  and the variational free-energy policy values). Any other generator name,
  such as an acquisition script, a vendor API or an instrument driver class,
  and free text such as "synthetic aperture radar" never make a recording
  synthetic; such results stay ``not_established``.
* Replay receipts with a numerical match are a same-runtime reproduction
  check (``numerically_verified`` for replay determinism, never independent).
* Every retained item gets a physical-truth claim, ``not_established`` unless
  an energy log that declares ``physical_measurement`` also declares a
  ``raw_sha256`` that the raw bytes of another workspace source hash to, has
  device, clock and calibration fields, and nothing in it declares synthetic
  or generated inputs. A log's ``log_digest`` is a self-digest of the JSON
  record, not raw device bytes, and CIW's energy logs carry no raw digest, so
  a log that merely calls itself a measurement stays ``not_established``.
  Seals are unkeyed: they detect alteration, not origin, so even a
  ``hardware_measured`` label is the recorder's assertion as recorded.
"""
from __future__ import annotations

import base64
import binascii
from hashlib import sha256
import json
from pathlib import Path
import tempfile

from ..session import Session, read_json
from .evidence import LABELS, finding

SCHEMA = "ciw.lab-workspace-classification.v1"
# Exact (key, value) declarations of generated inputs pinned by CIW's validators
# (energy_records origin, free_energy_profile POLICY, free_energy_contract scope).
SYNTHETIC_DECLARATIONS = frozenset({
    ("origin", "synthetic_fixture"),
    ("observations", "retained_synthetic_values"),
    ("coverage", "synthetic_prior_predictive_ensemble"),
    ("claim_scope", "synthetic_static_linear_gaussian_inference"),
})
# Closed allowlist of the run generators CIW itself emits (adapters/oscillator.py).
# A dotted name alone proves nothing: acquisition scripts, vendor APIs and
# instrument driver classes look the same, so any other name stays not_established.
KNOWN_GENERATORS = frozenset({"ciw.instruments.make_demo_run"})
UNAUTHENTICATED = ("Workspace seals and digests are unkeyed self-digests: they detect alteration of a record, "
                   "not who produced it or whether it came from hardware. Every label is derived from the "
                   "records as retained, and declared origins are not authenticated.")


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
    """True when a record carries an exact, CIW-pinned declaration of generated inputs."""
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, str) and (key, child) in SYNTHETIC_DECLARATIONS:
                return True
            if _synthetic(child):
                return True
    elif isinstance(value, list):
        return any(_synthetic(child) for child in value)
    return False


def _generator(provenance) -> str | None:
    """The known CIW generator identity a run provenance declares, if any."""
    name = provenance.get("generator") if isinstance(provenance, dict) else None
    return name if isinstance(name, str) and name in KNOWN_GENERATORS else None


def _mentions_generated(value) -> bool:
    """Conservative screen for the physical label: any generator key or synthetic/fixture wording.

    Used only to withhold ``hardware_measured``, never to assign ``synthetic``.
    """
    if isinstance(value, dict):
        return any(key == "generator" or _mentions_generated(child) for key, child in value.items())
    if isinstance(value, list):
        return any(_mentions_generated(child) for child in value)
    return isinstance(value, str) and any(word in value.lower() for word in ("synthetic", "fixture"))


def _result_basis(record, synthetic_input, name):
    providers = _providers(record)
    if providers:
        return {"provider": providers[0], "notes": f"{len(providers)} pinned runtime identities in the record"}
    if synthetic_input:
        return {"generator": {"name": name}}
    return {}


def _declares_physical(source) -> bool:
    return (isinstance(source, dict) and source.get("schema") == "ciw.energy-accuracy-log.v1"
            and source.get("origin") == "physical_measurement")


def _energy_acquisition(source, raw_digests):
    """Acquisition record for an energy log whose raw acquisition bytes are retained, else None.

    ``raw_digests`` holds the SHA-256 of the bytes of the workspace's other
    sources. The log must declare ``raw_sha256`` equal to one of them (its own
    ``log_digest`` hashes the JSON record, not device bytes), name its device,
    clock and calibration, and declare nothing synthetic or generated.
    """
    if not _declares_physical(source) or _synthetic(source) or _mentions_generated(source):
        return None
    sensor, clock = source.get("sensor"), source.get("clock")
    if not isinstance(sensor, dict) or not isinstance(clock, dict) or "accuracy_j" not in sensor:
        return None
    backend, device, epoch = sensor.get("backend"), sensor.get("device_uuid"), clock.get("epoch_id")
    raw = source.get("raw_sha256")
    if not all(isinstance(text, str) and text for text in (backend, device, epoch, raw)):
        return None
    raw = raw.removeprefix("sha256:")
    if raw not in raw_digests:
        return None
    starts = [phase["start_ns"] for phase in source.get("phases", []) if isinstance(phase, dict) and "start_ns" in phase]
    origin = min(starts) if starts else clock.get("monotonic_origin_ns")
    if origin is None:
        return None
    accuracy = sensor["accuracy_j"]
    return {"device": f"{backend}:{device}", "raw_sha256": raw, "acquired_at": f"{epoch}+{origin}ns",
            "calibration": ("declared_accuracy_j=" + repr(accuracy)) if accuracy is not None
            else "vendor_counter_accuracy_not_declared"}


def _source_bytes(record):
    try:
        return base64.b64decode(record["bytes_b64"], validate=True)
    except (KeyError, TypeError, ValueError, binascii.Error):
        return None


def classify_workspace(path) -> dict:
    """Validate a saved workspace without provider execution and label every retained result."""
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix="ciw-lab-classify-") as directory:
        Session.from_workspace(path, Path(directory))
    saved = read_json(path)
    run = saved["run"]
    provenance = run["metadata"].get("provenance", {})
    generator = _generator(provenance)
    synthetic_run = generator is not None or _synthetic(provenance)
    items = []

    def item(kind, identity, findings):
        items.append({"kind": kind, "identity": identity, "findings": findings,
                      "labels": sorted({f["evidence_status"] for f in findings})})

    run_basis = {"generator": {"name": generator or "declared synthetic source"}} if synthetic_run else {}
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
    raw = {source_id: _source_bytes(record) for source_id, record in sources.items()}
    digests = {source_id: sha256(data).hexdigest() for source_id, data in raw.items() if data is not None}
    for bundle in workbench.get("bundles", []):
        native, kind = bundle["native"], bundle["kind"]
        try:
            source = json.loads(raw.get(bundle.get("source_id")) or b"{}")
        except ValueError:
            source = {}
        if not isinstance(source, dict):
            source = {}
        synthetic_input = _synthetic(source) or _synthetic(native.get("configuration")) or _synthetic(native.get("steps"))
        basis = _result_basis({"runtimes": native.get("runtimes"), "steps": native.get("steps")}, synthetic_input, f"{kind} source")
        findings = [finding(f"{kind} bundle {bundle['bundle_id']} numerical result", "numerical", bundle["bundle_id"], basis,
                            expected_not_established=not basis)]
        # Raw acquisition bytes must be another retained source, rehashed here.
        acquisition = _energy_acquisition(source, {digest for source_id, digest in digests.items()
                                                   if source_id != bundle.get("source_id")})
        if acquisition:
            physical = {"acquisition": acquisition, "notes": "Retained raw bytes hash to the declared raw_sha256; "
                        "the recorder's assertion as recorded, origin not authenticated."}
        elif _declares_physical(source):
            physical = {"notes": "Operator declares physical_measurement (retained_operator_record_not_authenticated); "
                        "no retained raw acquisition bytes hash to a declared raw_sha256, or the log declares "
                        "synthetic or generated inputs. The self-sealed log_digest does not authenticate origin."}
        else:
            physical = {}
        findings.append(finding(f"{kind} bundle {bundle['bundle_id']} rests on a physical measurement", "physical",
                                acquisition["raw_sha256"] if acquisition else None, physical))
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
            "items": items, "label_counts": counts, "origin_authentication": UNAUTHENTICATED,
            "note": "Derived projection; retained records are unchanged and replay is never independent verification."}
