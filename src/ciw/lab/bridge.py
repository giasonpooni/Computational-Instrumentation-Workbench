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
  ``provider_backed`` only when that identity is a pin CIW itself declares for
  the workflow kind (:func:`declared_pins`): the revision equals a declared
  pin of the role it is recorded under (an identity nested inside a role's
  runtime, or recorded in a step, may match any pin of the kind), the module
  and source root equal the pin's where it declares them, and the source tree
  equals the tree CIW records for that revision wherever any CIW pin table
  records one. Where no table records a tree for the revision
  (:func:`pins_without_tree`), any tree is accepted and the row says
  ``tree_pinned: false``. Any other runtime identity leaves the result
  ``not_established`` with the reason. Results over
  generated inputs are ``synthetic``; built-in analyses over caller-declared
  data are ``not_established``, which matches CIW's own
  ``verification_status: not_verified``. The pin tables are public constants,
  so a fabricated record that copies a pinned revision and tree still passes
  this comparison: it is a check against CIW's declarations, not
  authentication.
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
from .evidence import LABELS, finding, supported_label

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
PIN_RULE = ("provider_backed only when every retained runtime identity with a revision and source tree is a pin CIW "
            "declares for the workflow kind (revision, and module, source root and source tree where CIW records "
            "them); otherwise not_established with the reason. Where CIW records no source tree for a pinned "
            "revision, any tree is accepted and the row shows tree_pinned false. An identity nested inside a role's "
            "runtime, or recorded in a step, may match any pin of the kind. The pins are public constants: a "
            "record that copies them still matches.")
UNAUTHENTICATED = ("Workspace seals and digests are unkeyed self-digests: they detect alteration of a record, "
                   "not who produced it or whether it came from hardware. Every label is derived from the "
                   "records as retained, and declared producers are not authenticated.")


def _providers(value, found=None):
    """Runtime identities that pin a revision and source tree, found anywhere in a record."""
    found = [] if found is None else found
    if isinstance(value, dict):
        if isinstance(value.get("revision"), str) and isinstance(value.get("source_tree"), str):
            found.append(value)
        for child in value.values():
            _providers(child, found)
    elif isinstance(value, list):
        for child in value:
            _providers(child, found)
    return found


def _manifest(name):
    from importlib import resources
    return json.loads(resources.files("ciw").joinpath(name).read_text(encoding="utf-8"))


def _pin(entry, declared_in, **extra):
    fields = {key: entry[key] for key in ("revision", "module", "source_root", "source_tree") if key in entry}
    return {**fields, **extra, "declared_in": declared_in}


def declared_pins() -> dict:
    """Runtime pins CIW declares, read from CIW's own pin tables.

    Returns ``{"kinds": {kind: {role: [pin, ...]}}, "operations": [pin, ...],
    "trees": {revision: [source_tree, ...]}}``. A pin holds ``revision`` and,
    where its table declares them, ``module``, ``source_root`` and
    ``source_tree``, plus ``declared_in``. ``operations`` are the session
    adapter pins (``adapter-runtimes.json``, historical revisions included).
    ``trees`` gathers every source tree any table records for a revision: a
    source tree is the commit's root tree, so it binds every kind that pins
    that revision (the SCR numerical-heat pin records no tree, the proved-heat
    pin of the same revision does).
    """
    from .. import (acquired_dataset, bim_quantity, declared_workload, free_energy_native, geodesic_reference,
                    geometric_circle, geometry_research, identified_stability, measurement_chain, proved_heat,
                    residual_monitor, schematic_companions)
    from ..adapters.ppda_acquisition import VENDOR_REVISION
    kinds: dict = {}

    def add(kind, role, pin):
        kinds.setdefault(kind, {}).setdefault(role, []).append(pin)

    for module, name in ((geodesic_reference, "geodesic_reference"), (declared_workload, "declared_workload"),
                         (geometry_research, "geometry_research")):
        for kind, entry in sorted(module.PINS.items()):
            add(kind, entry["role"], _pin(entry, f"ciw.{name}.PINS[{kind}]"))
    add("proved-heat", "scr", _pin(proved_heat.PIN, "ciw.proved_heat.PIN"))
    add("geometric-circle", "gte",
        _pin(geometric_circle.PIN, "ciw.geometric_circle.PIN", source_tree=geometric_circle.SOURCE_TREE))
    add("bim-quantity", bim_quantity.PIN["role"], _pin(bim_quantity.PIN, "ciw.bim_quantity.PIN"))
    add("identified-stability", "plsr",
        _pin(identified_stability.PIN, "ciw.identified_stability.PIN", source_tree=identified_stability.SOURCE_TREE))
    for kind, module, name in (("schematic-companions", schematic_companions, "schematic_companions"),
                               ("measurement-chain", measurement_chain, "measurement_chain"),
                               ("residual-monitor", residual_monitor, "residual_monitor"),
                               ("variational-free-energy", free_energy_native, "free_energy_native")):
        for role, entry in sorted(module.PINS.items()):
            add(kind, role, _pin(entry, f"ciw.{name}.PINS[{role}]"))
    manifests = {name: _manifest(name) for name in ("calibrated-observable-runtimes.json",
                                                    "calibrated-window-runtimes.json",
                                                    "identified-design-runtimes.json", "telemetry-runtimes.json")}
    for kind, names in (("calibrated-observable", ("calibrated-observable-runtimes.json",)),
                        ("calibrated-window", ("calibrated-window-runtimes.json",)),
                        ("acquired-calibrated-window", ("calibrated-window-runtimes.json",)),
                        ("identified-design",
                         ("calibrated-observable-runtimes.json", "identified-design-runtimes.json")),
                        ("telemetry", ("telemetry-runtimes.json",))):
        for name in names:
            for role, entry in sorted(manifests[name].items()):
                add(kind, role, _pin(entry, f"ciw/{name}[{role}]"))
    # The PPDA runtime and the vendor runtime nested inside it (a nested identity may match any pin of
    # its kind), with the trees acquired_dataset pins for both.
    vendor = {"revision": VENDOR_REVISION, "module": "evidence.types", "source_root": "."}
    ppda = acquired_dataset.AcquisitionWorkflow().pin
    for role, pin, declared_in in (("ppda", ppda, "ciw.acquired_dataset.AcquisitionWorkflow.pin"),
                                   ("ppda.vendor", vendor, "ciw.acquired_dataset vendor pin")):
        tree = acquired_dataset.SOURCE_TREES[pin["revision"]]
        add("acquired-dataset", role, _pin(pin, declared_in, source_tree=tree))
    operations = []
    for role, entry in sorted(_manifest("adapter-runtimes.json").items()):
        for index, pin in enumerate([entry, *entry.get("historical", [])]):
            suffix = f".historical[{index - 1}]" if index else ""
            operations.append(_pin(pin, f"ciw/adapter-runtimes.json[{role}]{suffix}"))
    trees: dict = {}
    for pin in [pin for roles in kinds.values() for group in roles.values() for pin in group] + operations:
        if pin.get("source_tree"):
            trees.setdefault(pin["revision"], set()).add(pin["source_tree"])
    return {"kinds": kinds, "operations": operations,
            "trees": {revision: sorted(values) for revision, values in sorted(trees.items())}}


def pins_without_tree(pins: dict | None = None) -> dict:
    """Declared pins whose revision has no source tree in any CIW pin table.

    For these, :func:`_pin_check` compares revision, module and source root
    only and accepts any source tree (``tree_pinned: false``), so a record that
    invents a tree still matches. Returns ``{"kinds": [(kind, role, revision),
    ...], "operations": [declared_in, ...]}``, sorted.
    """
    pins = declared_pins() if pins is None else pins
    trees = pins["trees"]
    kinds = sorted({(kind, role, pin["revision"]) for kind, roles in pins["kinds"].items()
                    for role, group in roles.items() for pin in group if not trees.get(pin["revision"])})
    operations = sorted(pin["declared_in"] for pin in pins["operations"] if not trees.get(pin["revision"]))
    return {"kinds": kinds, "operations": operations}


def _pin_check(identity, candidates, trees, scope) -> dict:
    """Compare one retained runtime identity with the pins CIW declares for its scope."""
    revision, tree = identity["revision"], identity["source_tree"]
    row = {"revision": revision, "source_tree": tree, "module": identity.get("module"),
           "repository": str(identity.get("module") or identity.get("repository") or "pinned provider"),
           "matched": None, "problem": None}
    same = [pin for pin in candidates if pin["revision"] == revision]
    fitting = [pin for pin in same if all(identity.get(key) == pin[key] for key in ("module", "source_root")
                                          if key in pin)]
    recorded = trees.get(revision, [])
    if not candidates:
        row["problem"] = f"CIW declares no runtime pin for {scope}"
    elif not same:
        row["problem"] = f"revision {revision} is not a revision CIW pins for {scope}"
    elif not fitting:
        row["problem"] = f"module or source root differs from every CIW pin of revision {revision} for {scope}"
    elif len(recorded) > 1:
        row["problem"] = f"CIW records conflicting source trees for revision {revision}: {', '.join(recorded)}"
    elif recorded and tree != recorded[0]:
        row["problem"] = f"source tree {tree} is not the tree {recorded[0]} CIW records for revision {revision}"
    else:
        row["matched"] = sorted(pin["declared_in"] for pin in fitting)
        row["tree_pinned"] = bool(recorded)
    return row


def _runtime_rows(runtimes, steps, kind, pins) -> list:
    """Pin comparison for every revision-and-tree identity in a bundle's runtimes (by role) and steps."""
    roles = pins["kinds"].get(kind, {})
    everything = [pin for group in roles.values() for pin in group]
    rows = []
    for role, value in sorted(runtimes.items()) if isinstance(runtimes, dict) else []:
        for identity in _providers(value):
            # A nested identity (for example PPDA's vendor runtime) may match any pin of the kind.
            candidates = roles.get(role, []) if identity is value else everything
            rows.append({"role": role, **_pin_check(identity, candidates, pins["trees"],
                                                    f"role {role} of workflow kind {kind}")})
    for identity in _providers(steps):
        rows.append({"role": None, **_pin_check(identity, everything, pins["trees"], f"workflow kind {kind}")})
    return rows


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


def _result_basis(rows, synthetic_input, name):
    """Provider basis when every retained runtime identity is a CIW pin; otherwise the reason, or synthetic/none."""
    if rows:
        problems = [row["problem"] for row in rows if row["problem"]]
        if problems:
            return {"notes": "Runtime identity does not match a CIW pin: " + "; ".join(problems)
                    + ". provider_backed needs the retained revision and source tree to be a pin CIW declares "
                    "for this workflow kind."}
        first = rows[0]
        trees = "every source tree equals the tree CIW records" if all(row.get("tree_pinned") for row in rows) \
            else "CIW records no source tree for some pinned revisions; those trees are as recorded"
        return {"provider": {"repository": first["repository"], "revision": first["revision"],
                             "source_tree": first["source_tree"], "executed": True},
                "notes": f"{len(rows)} pinned runtime identities in the record; every revision is a CIW pin "
                         f"({', '.join(sorted({d for row in rows for d in row['matched']}))}) and {trees}."}
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
    pins = declared_pins()
    run = saved["run"]
    provenance = run["metadata"].get("provenance", {})
    generator = _generator(provenance)
    synthetic_run = generator is not None or _synthetic(provenance)
    items = []

    def item(kind, identity, findings, rows=None):
        entry = {"kind": kind, "identity": identity, "findings": findings,
                 "labels": sorted({f["evidence_status"] for f in findings})}
        if rows is not None:
            entry["runtime_pins"] = rows
        items.append(entry)

    def numerical(claim, value, basis):
        return finding(claim, "numerical", value, basis,
                       expected_not_established=supported_label(basis, "numerical") == "not_established")

    run_basis = {"generator": {"name": generator or "declared synthetic source"}} if synthetic_run else {}
    item("run", run["evidence_id"], [
        finding(f"Run {run['run_id']} content", "numerical", run["evidence_id"], run_basis,
                expected_not_established=not synthetic_run),
        finding(f"Run {run['run_id']} represents a physical acquisition", "physical", None, {}),
    ])
    results = saved.get("results", {})
    for result in (results.values() if isinstance(results, dict) else results):
        identity = result.get("result_id", "legacy-result")
        rows = [_pin_check(runtime, pins["operations"], pins["trees"], "session operation adapters")
                for runtime in _providers(result.get("runtime") or {})]
        basis = _result_basis(rows, synthetic_run, f"{result.get('operation_id')} over {run['run_id']}")
        item("operation_result", identity, [
            numerical(f"{result.get('operation_id')} result {identity}", result.get("record_digest"), basis),
            finding(f"{result.get('operation_id')} result {identity} is physically valid", "physical", None, {}),
        ], rows)
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
        rows = _runtime_rows(native.get("runtimes"), native.get("steps"), kind, pins)
        basis = _result_basis(rows, synthetic_input, f"{kind} source")
        findings = [numerical(f"{kind} bundle {bundle['bundle_id']} numerical result", bundle["bundle_id"], basis)]
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
        item("workbench_bundle", bundle["bundle_id"], findings, rows)
    counts = {label: 0 for label in LABELS}
    for entry in items:
        for record in entry["findings"]:
            counts[record["evidence_status"]] += 1
    return {"schema": SCHEMA, "workspace": str(path), "validated_without_provider_execution": True,
            "validation_scope": "Session.from_workspace; built-in offline analyses may be recomputed to check retained data",
            "items": items, "label_counts": counts, "origin_authentication": UNAUTHENTICATED,
            "runtime_pin_rule": PIN_RULE,
            "note": "Derived projection; retained records are unchanged and replay is never independent verification."}
