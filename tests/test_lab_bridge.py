import base64
import json
from pathlib import Path

import pytest

from ciw import cli, energy_records
from ciw.core.identities import evidence_id
from ciw.instruments import make_demo_run
from ciw.lab import bridge
from ciw.lab.bridge import classify_workspace
from ciw.lab.evidence import validate_finding
from ciw.session import Session

ROOT = Path(__file__).resolve().parents[1]


def _request(session, kind, payload):
    response = session.handle({"protocol_version": 1, "request_id": "lab-bridge", "type": kind, "payload": payload})
    assert response["type"] == "response", response
    return response["payload"]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    directory = tmp_path_factory.mktemp("lab-bridge")
    session = Session(make_demo_run(), directory / "out")
    _request(session, "operation.execute", {"operation_id": "statistics.v1",
                                            "parameters": {"channel": "v", "interval_s": [1.0, 2.0]}})
    synthetic = json.loads((ROOT / "examples" / "energy-accuracy" / "baseline.json").read_text(encoding="utf-8"))
    physical = dict(synthetic, origin="physical_measurement", run_id="energy-run-" + "2" * 32)
    physical.pop("log_digest")
    for log in (synthetic, energy_records.seal(physical)):
        source = _request(session, "source.add", {"kind": "energy-accuracy", "label": log["origin"],
                                                  "bytes_b64": base64.b64encode(json.dumps(log).encode()).decode()})
        original = _request(session, "operation.execute", {"operation_id": "ciw.energy-accuracy.v1",
                                                           "parameters": {"source_id": source["source_id"]}})
        _request(session, "bundle.replay", {"bundle_id": original["bundle_id"]})
    return session.save_workspace(directory / "workspace.json")


def _labels(items, kind):
    return [[f["evidence_status"] for f in item["findings"]] for item in items if item["kind"] == kind]


def test_every_retained_result_receives_validated_labels(workspace):
    result = classify_workspace(workspace)
    assert result["validated_without_provider_execution"] is True
    for item in result["items"]:
        for record in item["findings"]:
            validate_finding(record)
    assert _labels(result["items"], "run") == [["synthetic", "not_established"]]
    assert _labels(result["items"], "operation_result") == [["synthetic", "not_established"]]
    bundles = _labels(result["items"], "workbench_bundle")
    assert ["synthetic", "not_established"] in bundles
    assert ["synthetic", "not_established", "numerically_verified"] in bundles
    assert ["not_established", "not_established", "numerically_verified"] in bundles
    assert result["label_counts"]["independently_verified"] == 0
    assert result["label_counts"]["hardware_measured"] == 0


def test_a_resealed_log_that_calls_itself_physical_is_not_hardware_measured(workspace):
    # The fixture above is the synthetic baseline with origin changed to
    # physical_measurement and resealed: the seal is an unkeyed self-digest.
    result = classify_workspace(workspace)
    physical = [f for item in result["items"] for f in item["findings"] if f["domain"] == "physical"]
    assert len(physical) == 6 and {f["evidence_status"] for f in physical} == {"not_established"}
    declared = [f for f in physical if "physical_measurement" in f["basis"].get("notes", "")]
    assert len(declared) == 2 and all("acquisition" not in f["basis"] and f["value"] is None for f in declared)
    assert "unkeyed" in result["origin_authentication"] and "not authenticated" in result["origin_authentication"]


def _physical_log(**changes):
    log = json.loads((ROOT / "examples" / "energy-accuracy" / "baseline.json").read_text())
    log.update(origin="physical_measurement", raw_sha256="ab" * 32)
    log["sensor"] = dict(log["sensor"], name="bench device", driver_version="550.54", nvml_version="12.550")
    log["clock"] = dict(log["clock"], epoch_id="boot-7", implementation="time.monotonic_ns")
    log["runtime"] = dict(log["runtime"], implementation={"profile": "ciw.fixed-gaussian-gpu-energy.v1", "code_sha256": "d" * 64})
    log["runtime"]["workload"] = dict(log["runtime"]["workload"], device_name="bench device")
    return dict(log, **changes)


def test_hardware_label_needs_retained_raw_bytes_and_no_generated_declaration():
    log = _physical_log()
    acquisition = bridge._energy_acquisition(log, {"ab" * 32})
    assert acquisition["raw_sha256"] == "ab" * 32 and acquisition["device"].startswith("nvml:GPU-")
    assert acquisition["calibration"] == "vendor_counter_accuracy_not_declared"
    assert bridge._energy_acquisition(log, set()) is None                       # raw bytes not in the workspace
    assert bridge._energy_acquisition(dict(log, raw_sha256=None), {"ab" * 32}) is None
    assert bridge._energy_acquisition(dict(log, origin="synthetic_fixture"), {"ab" * 32}) is None
    assert bridge._energy_acquisition(dict(log, provenance={"generator": "ciw.fake"}), {"ab" * 32}) is None
    fixture = dict(log, sensor=dict(log["sensor"], name="synthetic fixture device"))
    assert bridge._energy_acquisition(fixture, {"ab" * 32}) is None
    no_calibration = dict(log, sensor={k: v for k, v in log["sensor"].items() if k != "accuracy_j"})
    assert bridge._energy_acquisition(no_calibration, {"ab" * 32}) is None
    no_clock = dict(log, clock={k: v for k, v in log["clock"].items() if k != "epoch_id"})
    assert bridge._energy_acquisition(no_clock, {"ab" * 32}) is None


def _run_workspace(directory, provenance):
    run = make_demo_run()
    run["metadata"]["provenance"] = provenance
    run["evidence_id"] = evidence_id(run)
    session = Session(run, directory / "out")
    _request(session, "operation.execute", {"operation_id": "statistics.v1",
                                            "parameters": {"channel": "v", "interval_s": [1.0, 2.0]}})
    return session.save_workspace(directory / "workspace.json")


@pytest.mark.parametrize("provenance, label", [
    ({"generator": "Keysight 33500B function generator driving the shaker", "source": "accelerometer recording"},
     "not_established"),
    ({"source": "synthetic aperture radar acquisition, field campaign 2026-03"}, "not_established"),
    ({"source": "bench accelerometer recording"}, "not_established"),
    # Acquisition code named as the generator: a dotted identity is not a CIW generator identity.
    ({"generator": "shaker_capture.py", "source": "bench accelerometer recording"}, "not_established"),
    ({"generator": "pymeasure.instruments.agilent.Agilent33500", "source": "accelerometer recording"},
     "not_established"),
    ({"generator": "NI.DAQmx", "source": "bench accelerometer recording"}, "not_established"),
    ({"source": "numpy draws", "generator": "numpy.random.default_rng"}, "not_established"),
    ({"source": "demo", "generator": "ciw.instruments.make_demo_run", "generator_version": 1}, "synthetic"),
])
def test_synthetic_needs_a_structured_generator_declaration(tmp_path, provenance, label):
    result = classify_workspace(_run_workspace(tmp_path, provenance))
    assert _labels(result["items"], "run") == [[label, "not_established"]]
    assert _labels(result["items"], "operation_result") == [[label, "not_established"]]


def test_classification_binds_no_provider_and_changes_nothing(workspace, monkeypatch):
    from ciw.adapters import subprocess as adapters

    before = workspace.read_bytes()
    monkeypatch.setattr(adapters.PinnedSubprocessAdapter, "__init__",
                        lambda *_, **__: pytest.fail("classification must not bind a provider"))
    classify_workspace(workspace)
    assert workspace.read_bytes() == before


def test_reopen_recomputes_the_builtin_energy_analysis(workspace, monkeypatch):
    # Recorded behavior: the energy-accuracy validator re-runs its offline log
    # analysis on reopen, so "validated without execution" would overstate it.
    calls = []
    original = energy_records.analyze
    monkeypatch.setattr(energy_records, "analyze", lambda log: calls.append(1) or original(log))
    classify_workspace(workspace)
    assert calls


def test_tampered_workspace_is_refused_before_labelling(workspace, tmp_path):
    saved = json.loads(workspace.read_text(encoding="utf-8"))
    saved["run"]["channels"]["v"]["values"][0] += 1.0
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="integrity"):
        classify_workspace(forged)


def test_cli_classify(workspace, capsys):
    assert cli.main(["lab", "classify", str(workspace)]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "ciw.lab-workspace-classification.v1"


def _fabricated_heat_workspace(directory, tree=None):
    """A content-consistent numerical-heat bundle no provider computed (T092/T100), optionally resealed with ``tree``."""
    from ciw.declared_workload import _verification
    from ciw.lab.exchange_provenance_bundles_fixtures import fabricated_heat_catalog
    from ciw.telemetry import _bundle_digest
    from ciw.workbench import Workbench

    catalog = fabricated_heat_catalog([0, 1, 2, 3, 0], experiment_id="ciw-lab-bridge-fabricated-heat")
    bundle = catalog["bundles"][0]["native"]
    if tree is not None:
        bundle["runtimes"]["scr"]["source_tree"] = tree
        reproduction = bundle.pop("verification")["reproduction"]
        bundle["bundle_digest"] = _bundle_digest(bundle)
        bundle["verification"] = _verification(bundle, reproduction)
        catalog["bundles"][0]["bundle_id"] = bundle["bundle_digest"]
    session = Session(make_demo_run(), directory / "out")
    session.workbench = Workbench.restore(catalog)
    return session.save_workspace(directory / "workspace.json"), catalog["bundles"][0]["bundle_id"]


def _bundle_item(result, bundle_id):
    return next(item for item in result["items"] if item["kind"] == "workbench_bundle" and item["identity"] == bundle_id)


def test_a_fabricated_runtime_whose_tree_is_not_the_pin_is_not_provider_backed(tmp_path):
    # The retained revision, module and source root are CIW's numerical-heat pin; the tree is invented.
    path, bundle_id = _fabricated_heat_workspace(tmp_path)
    item = _bundle_item(classify_workspace(path), bundle_id)
    numerical = item["findings"][0]
    assert numerical["evidence_status"] == "not_established" and numerical["expected_not_established"] is True
    assert "provider" not in numerical["basis"] and numerical["origin"] == []
    assert "is not the tree 4068a711534932e8d89bb0d87d373376dafdf6cd" in numerical["basis"]["notes"]
    row, = item["runtime_pins"]
    assert row["role"] == "scr" and row["matched"] is None and row["source_tree"] == "0" * 40


def test_a_record_copying_the_pinned_tree_still_classifies_provider_backed(tmp_path):
    # The residual limit: pins are public constants and seals are unkeyed, so a copy of the pin matches.
    from ciw.proved_heat import PIN
    path, bundle_id = _fabricated_heat_workspace(tmp_path, tree=PIN["source_tree"])
    result = classify_workspace(path)
    numerical = _bundle_item(result, bundle_id)["findings"][0]
    assert numerical["evidence_status"] == "provider_backed" and numerical["origin"] == ["provider"]
    assert "ciw.declared_workload.PINS[numerical-heat]" in numerical["basis"]["notes"]
    assert "public constants" in result["runtime_pin_rule"] and "not authenticated" in result["origin_authentication"]


def test_runtime_pin_comparison_rules():
    pins = bridge.declared_pins()
    heat = pins["kinds"]["numerical-heat"]["scr"]
    good = {"revision": heat[0]["revision"], "source_tree": pins["trees"][heat[0]["revision"]][0],
            "module": "execution.engine", "source_root": "."}
    assert bridge._pin_check(good, heat, pins["trees"], "scope")["matched"] == ["ciw.declared_workload.PINS[numerical-heat]"]
    for change, reason in ((dict(revision="f" * 40), "is not a revision CIW pins"),
                           (dict(module="execution.proving"), "module or source root differs"),
                           (dict(source_tree="0" * 40), "is not the tree")):
        assert reason in bridge._pin_check(dict(good, **change), heat, pins["trees"], "scope")["problem"]
    assert "declares no runtime pin" in bridge._pin_check(good, [], pins["trees"], "kind x")["problem"]
    # Every tree CIW records is one tree per revision: a commit has exactly one root tree.
    assert all(len(trees) == 1 for trees in pins["trees"].values())
    assert set(pins["kinds"]) >= {"numerical-heat", "proved-heat", "flat-torus-reference", "curved-path-transfer"}


# Pins whose revision has no source tree in any CIW pin table, by kind. For these the classifier compares
# revision, module and source root only and accepts any tree (tree_pinned false); docs/LAB.md and rule 12 of
# docs/lab/SPECIFICATIONS.md name them. Recording a tree for one of them, or adding a treeless pin, changes this set.
TREELESS_ROLES = {
    "acquired-calibrated-window": ["gsie", "mcur", "set", "stfe", "tbrt"],
    "bim-quantity": ["cse"],
    "calibrated-observable": ["cbsr", "fdir", "fsrt", "mcur", "oit", "set", "tbrt"],
    "calibrated-window": ["gsie", "mcur", "set", "stfe", "tbrt"],
    "identified-design": ["cbsr", "edspt", "fdir", "fsrt", "mcur", "oit", "set", "sidt", "tbrt", "ywir"],
    "residual-monitor": ["fdir", "oit"],
    "schematic-assessment": ["sra"],
    "schematic-companions": ["jspt", "plsr", "sra"],
    "telemetry": ["cbsr", "gsie", "ppda", "set", "stfe"],
}
TREELESS_REVISIONS = [
    "1467ec5058b3e7ebd6ba4a45f2d9b49148a2560d", "209985a8c7482aba037021035c2f0b550c9b1ddc",
    "29e4b306492b793487d47a96b97a56548217aa12", "2f838f4e196f453efc3a59045b0b3ec4b5680296",
    "40507060ca7a9126a9d641999b994a757eef3bfd", "49405ecd623474ddf601989b0a8195be401544e0",
    "4b74abda40bba3277de69bf61e9e09283ae2d5b3", "54dd43b657b92d87c2fc263dfd99437c664be353",
    "5e7bda36f521a5c1b0082b512f35e29803bffafc", "7399ab03087b27683620b4c57f97b2ac14546c7f",
    "89d5ea52f46676eabf13e906166e3ab701451680", "9d0e7b4a1162e038150a71c63d986945d78135d4",
    "a6e79585950bb6860e5edce5ebd2cce39ea481f2", "b543969cb80a69a5df1274eee10d22549bcea773",
    "b7e81402d5f4e854b3c8bb0d0354afe4de9269c6", "d7c181fb9967883e085b3728d38096f59e54efbd",
    "daf43fc870ba926289c3bc4908db784691c7addd", "db4c564bddbe1911f96585bd18f58659a0026fb7",
    "de38873db1ba98744d395dfa7df1d3b5b7bb3059", "e1f4a8c128a9c14e5b82acad76696219a5b5b6a9",
]
TREELESS_OPERATIONS = ["ciw/adapter-runtimes.json[fsrt].historical[0]", "ciw/adapter-runtimes.json[rci].historical[0]"]


def _identity(pin, tree):
    return {"revision": pin["revision"], "source_tree": tree, **{k: pin[k] for k in ("module", "source_root") if k in pin}}


def test_the_pins_without_a_recorded_tree_are_the_reviewed_set():
    pins = bridge.declared_pins()
    gap = bridge.pins_without_tree(pins)
    roles = {}
    for kind, role, _ in gap["kinds"]:
        roles.setdefault(kind, []).append(role)
    assert roles == TREELESS_ROLES
    assert sorted({revision for _, _, revision in gap["kinds"]}) == TREELESS_REVISIONS
    assert gap["operations"] == TREELESS_OPERATIONS
    # For these an invented tree is accepted and the row says so; for every pin with a recorded tree it is refused.
    invented = "0" * 40
    for kind, groups in sorted(pins["kinds"].items()):
        for role, group in sorted(groups.items()):
            for pin in group:
                row, = bridge._runtime_rows({role: _identity(pin, invented)}, None, kind, pins)
                if (kind, role, pin["revision"]) in gap["kinds"]:
                    assert row["matched"] and row["tree_pinned"] is False, (kind, role)
                    basis = bridge._result_basis([row], False, "x")
                    assert basis["provider"]["executed"] is True and "records no source tree" in basis["notes"]
                else:
                    assert row["matched"] is None and "is not the tree" in row["problem"], (kind, role)
    for pin in pins["operations"]:
        row = bridge._pin_check(_identity(pin, invented), pins["operations"], pins["trees"], "ops")
        assert bool(row["matched"]) is (pin["declared_in"] in TREELESS_OPERATIONS)
    assert "tree_pinned false" in bridge.PIN_RULE and "any pin of the kind" in bridge.PIN_RULE
    assert "declared producers are not authenticated" in bridge.UNAUTHENTICATED


def test_nested_and_step_identities_may_match_any_pin_of_the_kind():
    pins = bridge.declared_pins()
    ppda, vendor = pins["kinds"]["acquired-dataset"]["ppda"][0], pins["kinds"]["acquired-dataset"]["ppda.vendor"][0]
    tree = pins["trees"][vendor["revision"]][0]
    nested = dict(_identity(ppda, pins["trees"][ppda["revision"]][0]), vendor=_identity(vendor, tree))
    rows = bridge._runtime_rows({"ppda": nested}, [{"runtime": _identity(ppda, pins["trees"][ppda["revision"]][0])}],
                                "acquired-dataset", pins)
    assert [row["matched"] is not None for row in rows] == [True, True, True]
    # Recorded directly under a role, the vendor pin is not the ppda role's pin.
    row, = bridge._runtime_rows({"ppda": _identity(vendor, tree)}, None, "acquired-dataset", pins)
    assert "is not a revision CIW pins for role ppda" in row["problem"]


def test_the_classify_documentation_names_every_kind_whose_tree_is_not_compared():
    root = Path(__file__).resolve().parents[1] / "docs"
    if not (root / "LAB.md").is_file():
        pytest.skip("docs/ is not available")
    lab = " ".join((root / "LAB.md").read_text(encoding="utf-8").split())
    classify = lab[lab.index("`ciw lab classify WORKSPACE` applies"):lab.index("## Assistant access over MCP")]
    spec = " ".join((root / "lab" / "SPECIFICATIONS.md").read_text(encoding="utf-8").split())
    rule = spec[spec.index("12. Workspace classification"):spec.index("T155 compares")]
    for text in (classify, rule):
        assert "tree_pinned: false" in text and "pins_without_tree" in text and "any pin of the kind" in text
        for kind in TREELESS_ROLES:
            assert f"`{kind}`" in text, kind
        assert "except `gsie`" in text and "historical `fsrt` and `rci` adapter pins" in text
