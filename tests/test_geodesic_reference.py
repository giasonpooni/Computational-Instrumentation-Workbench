"""Native geometry contracts, bounded refusal and independent analytic checks."""
import base64
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from ciw.declared_workload import _verification
from ciw.geodesic_reference import GeodesicReferenceWorkflow, PINS, _check_data
from ciw.geodesic_reference_view import project
from ciw.telemetry import canonical, digest, byte_digest, _bundle_digest

ROOT = Path(__file__).resolve().parents[1]
FILES = {"flat-torus-reference": "flat-torus.json", "curved-path-transfer": "curved-path.json"}


def source(kind):
    return json.loads((ROOT / "examples/geodesic-reference" / FILES[kind]).read_bytes())


def repositories(kind):
    role = PINS[kind]["role"]
    path = os.environ.get("CIW_" + role.upper() + "_REPO")
    if not path:
        pytest.skip("Set CIW_" + role.upper() + "_REPO to the approved native checkout")
    return {role: path}


@pytest.fixture(scope="module", params=tuple(FILES))
def native(request):
    kind = request.param
    declaration = source(kind)
    raw = b"\n " + json.dumps(declaration, indent=2).encode() + b"\n"
    workflow = GeodesicReferenceWorkflow(kind)
    paths = repositories(kind)
    original = workflow.create_session(raw, paths)
    replay = workflow.replay_session(original, paths)["session"]
    return kind, workflow, paths, raw, original, replay


@pytest.mark.parametrize("kind", tuple(FILES))
@pytest.mark.parametrize("raw", [b"null", b"[]", b"false", b"{}", b'{"schema":1,"schema":2}', b"", b" " * 32769],
                         ids=["null", "array", "boolean", "empty-object", "duplicate-key", "empty", "oversize"])
def test_malformed_sources_refuse_before_runtime(kind, raw, monkeypatch):
    workflow = GeodesicReferenceWorkflow(kind)
    monkeypatch.setattr(workflow, "_adapters", lambda *_: pytest.fail("Refusal must precede runtime binding"))
    with pytest.raises(ValueError):
        workflow.create_session(raw, {})


@pytest.mark.parametrize("field,value", [
    ("winding", [True, 1]), ("winding", [1.5, 1]), ("winding", [0, 0]), ("winding", [17, 0]),
    ("tau", [0, 0]), ("tau", [0, .1]), ("tau", [5, 1]), ("start", [9, 0]),
    ("samples", True), ("samples", 1), ("samples", 129), ("start", [0, False]),
])
def test_flat_source_bounds(field, value, monkeypatch):
    value_source = source("flat-torus-reference")
    value_source[field] = value
    workflow = GeodesicReferenceWorkflow("flat-torus-reference")
    monkeypatch.setattr(workflow, "_adapters", lambda *_: pytest.fail("Invalid geometry reached provider"))
    with pytest.raises(ValueError):
        workflow.create_session(canonical(value_source), {})


@pytest.mark.parametrize("field,value", [
    ("arclength", [0, 0]), ("arclength", [1, 2]), ("arclength", [0, 8]),
    ("arclength", [0, True]), ("gaussian_curvature", True), ("gaussian_curvature", 2),
    ("initial_perturbation", [.2, 0]), ("units", {"length": "unknown", "angle": "radian"}),
    ("relative_tolerance", 0), ("relative_tolerance", .1),
    ("starting_covariance", {"matrix": [[1, 2], [2, 1]], "basis": "assumed", "note": "indefinite"}),
    ("starting_covariance", {"matrix": [[1, .1], [0, 1]], "basis": "assumed", "note": "nonsymmetric"}),
    ("starting_covariance", {"matrix": [[1, 0], [0, 1]], "basis": "measured", "note": "undeclared calibration"}),
])
def test_curved_source_bounds(field, value, monkeypatch):
    value_source = source("curved-path-transfer")
    value_source[field] = value
    workflow = GeodesicReferenceWorkflow("curved-path-transfer")
    monkeypatch.setattr(workflow, "_adapters", lambda *_: pytest.fail("Invalid declaration reached provider"))
    with pytest.raises(ValueError):
        workflow.create_session(canonical(value_source), {})


@pytest.mark.parametrize("name,kind", [("flat-invalid-winding.json", "flat-torus-reference"),
    ("curved-invalid-covariance.json", "curved-path-transfer"), ("curved-coarse-grid.json", "curved-path-transfer")])
def test_shipped_refusal_fixtures(name, kind):
    with pytest.raises(ValueError):
        GeodesicReferenceWorkflow(kind)._source((ROOT / "examples/geodesic-reference" / name).read_bytes())


def test_native_occurrence_replay_and_exact_evidence(native):
    kind, workflow, _, raw, original, replay = native
    assert base64.b64decode(original["source"]["evidence"][0]["bytes_b64"]) == raw
    assert workflow._validate(original) == raw
    assert workflow._validate(replay) == raw
    assert original["source"] == replay["source"]
    assert original["session_id"] != replay["session_id"]
    assert original["bundle_digest"] != replay["bundle_digest"]
    old, new = original["steps"][0], replay["steps"][0]
    assert old["numerical_result_id"] == new["numerical_result_id"]
    assert canonical(old["numerical_result"]) == canonical(new["numerical_result"])
    assert old["execution_id"] != new["execution_id"] and old["result_id"] != new["result_id"]
    assert original["verification"]["independent"] is False
    assert original["runtimes"][PINS[kind]["role"]]["source_tree"] == PINS[kind]["source_tree"]
    assert "flat_torus.companion" not in sys.modules and "geodesic_testbed.jacobi" not in sys.modules


def _analytic(curvature, grid):
    grid = np.asarray(grid)
    if curvature > 0:
        root = math.sqrt(curvature)
        a, b = np.cos(root * grid), np.sin(root * grid) / root
    elif curvature < 0:
        root = math.sqrt(-curvature)
        a, b = np.cosh(root * grid), np.sinh(root * grid) / root
    else:
        a, b = np.ones_like(grid), grid
    return np.stack((np.stack((a, b), axis=-1), np.stack((-curvature * b, a), axis=-1)), axis=-2)


def _assert_curved_oracle(declaration, data):
    record = data["record"]
    phi = _analytic(declaration["gaussian_curvature"], declaration["arclength"])
    actual_phi = np.stack((np.stack((record["a"], record["b"]), axis=-1),
                          np.stack((record["a_rate"], record["b_rate"]), axis=-1)), axis=-2)
    # Fixed native RK4 agreement allowance; declaration relative_tolerance is
    # the physical first-order validity bound, not a numerical solver setting.
    np.testing.assert_allclose(actual_phi, phi, rtol=2e-6, atol=2e-7)
    initial = np.asarray(declaration["initial_perturbation"])
    expected = phi @ initial
    np.testing.assert_allclose(data["separation"], expected[:, 0], rtol=2e-6, atol=2e-9)
    np.testing.assert_allclose(data["heading_change"], expected[:, 1], rtol=2e-6, atol=2e-9)
    covariance = np.asarray(declaration["starting_covariance"]["matrix"])
    np.testing.assert_allclose(data["propagated_covariance"], phi @ covariance @ phi.transpose(0, 2, 1), rtol=2e-6, atol=2e-10)
    np.testing.assert_allclose(data["determinant"], 1, rtol=2e-6, atol=2e-7)
    assert record["calibration"]["bound"] is False
    assert record["covariance"]["basis"] == "assumed"
    assert record["resolution"]["convergence"]["established"] is False


def test_native_reference_agrees_with_independent_geometry(native):
    kind, _, _, raw, original, _ = native
    declaration = json.loads(raw)
    data = original["steps"][0]["result"]["data"]
    if kind == "curved-path-transfer":
        _assert_curved_oracle(declaration, data)
    else:
        geometry, path = data["reference"]["geometry"], data["trajectory"]
        assert geometry["area"] == 1.0
        assert geometry["length"] == pytest.approx(math.sqrt(2), rel=1e-15)
        times = np.asarray(path["times"])
        expected = np.asarray(declaration["start"]) + times[:, None] * np.array([1, 1])
        pairs = lambda name: [[p["re"], p["im"]] for p in path[name]]
        np.testing.assert_allclose(pairs("cover_points"), expected, atol=1e-14, rtol=0)
        np.testing.assert_allclose(pairs("parallelogram_points"), expected % 1, atol=1e-14, rtol=0)


@pytest.mark.parametrize("curvature", [0.0, -.25])
def test_native_flat_and_negative_curvature_oracles(curvature):
    kind = "curved-path-transfer"
    declaration = source(kind)
    declaration["gaussian_curvature"] = curvature
    declaration["experiment_id"] += ":" + str(curvature)
    workflow = GeodesicReferenceWorkflow(kind)
    paths = repositories(kind)
    original = workflow.create_session(canonical(declaration), paths)
    replay = workflow.replay_session(original, paths)["session"]
    _assert_curved_oracle(declaration, original["steps"][0]["result"]["data"])
    destination = os.environ.get("CIW_GEODESIC_FIXTURE_DIR")
    if destination:
        folder = Path(destination) / ("curved-zero" if curvature == 0 else "curved-negative")
        folder.mkdir(parents=True, exist_ok=True)
        for name, content in (("source", declaration), ("original", original), ("replay", replay)):
            (folder / (name + ".json")).write_bytes(canonical(content))


def test_native_oblique_lattice_and_negative_winding():
    kind = "flat-torus-reference"
    declaration = source(kind)
    declaration.update(tau=[.4, 1.7], start=[-.8, .3], winding=[-2, 3], samples=33)
    workflow = GeodesicReferenceWorkflow(kind)
    paths = repositories(kind)
    original = workflow.create_session(canonical(declaration), paths)
    replay = workflow.replay_session(original, paths)["session"]
    data = original["steps"][0]["result"]["data"]
    geometry, trajectory = data["reference"]["geometry"], data["trajectory"]
    x, y = declaration["tau"]
    omega1, omega2 = 1 / math.sqrt(y), complex(x, y) / math.sqrt(y)
    cover_vector = declaration["winding"][0] * omega1 + declaration["winding"][1] * omega2
    assert geometry["area"] == pytest.approx(1, abs=1e-14)
    assert geometry["length"] == pytest.approx(abs(cover_vector), rel=1e-14)
    for sample, t in zip(trajectory["cover_points"], trajectory["times"]):
        expected = complex(geometry["start"]["re"], geometry["start"]["im"]) + t * cover_vector
        assert complex(sample["re"], sample["im"]) == pytest.approx(expected, abs=1e-12)
    destination = os.environ.get("CIW_GEODESIC_FIXTURE_DIR")
    if destination:
        folder = Path(destination) / "flat-oblique"
        folder.mkdir(parents=True, exist_ok=True)
        for name, content in (("source", declaration), ("original", original), ("replay", replay)):
            (folder / (name + ".json")).write_bytes(canonical(content))


def _seal(bundle, changed_source=None):
    if changed_source is not None:
        raw = canonical(changed_source)
        evidence = byte_digest(raw)
        bundle["source"] = {"experiment_id": changed_source["experiment_id"], "experiment_digest": digest(changed_source),
            "evidence": [{"artifact_ref": evidence, "sha256": evidence, "bytes_b64": base64.b64encode(raw).decode()}]}
    for step in (bundle["steps"][0], bundle["verification"]["reproduction"]):
        if changed_source is not None:
            step["request"] = deepcopy(changed_source)
            step["request_sha256"] = digest(changed_source)
            step["input_refs"] = [evidence]
            step["result"]["input_refs"] = [evidence]
        result = step["result"]
        result["result_id"] = digest({k: v for k, v in result.items() if k != "result_id"})
        step["result_id"] = result["result_id"]
        step["result_sha256"] = digest(result)
        step["numerical_result"] = {"operation_id": step["operation_id"], "data": deepcopy(result["data"])}
        step["numerical_result_id"] = digest(step["numerical_result"])
    bundle["bundle_digest"] = _bundle_digest(bundle)
    bundle["verification"] = _verification(bundle, bundle["verification"]["reproduction"])


def test_resealed_initial_input_tamper_is_rejected(native):
    kind, workflow, _, raw, original, _ = native
    changed = deepcopy(original)
    declaration = json.loads(raw)
    if kind == "flat-torus-reference":
        declaration["start"] = [.25, .375]
    else:
        declaration["initial_perturbation"] = [.05, .01]
    _seal(changed, declaration)
    with pytest.raises(ValueError, match="start|perturbation"):
        workflow._validate(changed)


@pytest.mark.parametrize("tamper", ["tree", "request_bool", "authority", "shape", "verification_bool"])
def test_resealed_contract_tampering_refuses(native, tamper):
    kind, workflow, _, _, original, _ = native
    changed = deepcopy(original)
    if tamper == "tree":
        changed["runtimes"][PINS[kind]["role"]]["source_tree"] = "0" * 40
    for step in (changed["steps"][0], changed["verification"]["reproduction"]):
        if tamper == "request_bool":
            if kind == "flat-torus-reference":
                step["request"]["winding"][0] = True
            else:
                step["request"]["arclength"][0] = False
            step["request_sha256"] = digest(step["request"])
        elif tamper == "authority":
            step["result"]["authority"]["state_admission"] = "performed"
        elif tamper == "shape":
            data = step["result"]["data"]
            (data["trajectory"]["cover_points"] if kind == "flat-torus-reference" else data["propagated_covariance"]).pop()
    _seal(changed)
    if tamper == "verification_bool":
        changed["verification"]["independent"] = 0
        content = {k: v for k, v in changed["verification"].items() if k != "verification_id"}
        changed["verification"]["verification_id"] = byte_digest(content["schema"].encode() + b"\0" + canonical(content))
    with pytest.raises(ValueError):
        workflow._validate(changed)


def test_reproduction_detects_resealed_numerical_forgery(native):
    kind, workflow, paths, _, original, _ = native
    changed = deepcopy(original)
    for step in (changed["steps"][0], changed["verification"]["reproduction"]):
        data = step["result"]["data"]
        if kind == "flat-torus-reference":
            data["trajectory"]["cover_points"][-1]["re"] += .1
        else:
            data["separation"][-1] += .1
    _seal(changed)
    # Inspection binds content and shape; a fresh native operation establishes
    # reproducibility. A submitted pass is not independent authentication.
    workflow._validate(changed)
    with pytest.raises(ValueError, match="replay mismatch"):
        workflow.replay_session(changed, paths)


def test_view_is_copy_only_and_preserves_native_uncertainty(native):
    kind, _, _, raw, original, _ = native
    before = canonical(original)
    declaration = json.loads(raw)
    evidence = original["source"]["evidence"][0]["artifact_ref"]
    view = project({"native": original, "kind": kind, "bundle_id": original["bundle_digest"]},
        {"source_id": "source", "evidence_id": evidence, "label": "reference"}, declaration, 1)
    assert view["fusion_context"] is None and view["raw_observations"] == []
    assert view["authority"]["read_only"] is True
    panels = {panel["panel_id"]: panel for panel in view["panels"]}
    if kind == "flat-torus-reference":
        assert all(panel["covariance"] is None for panel in panels.values())
        assert panels["loop-length"]["units"] == ["normalized_length"]
    else:
        data = original["steps"][0]["result"]["data"]
        assert panels["endpoint-pose"]["covariance"] == data["propagated_covariance"][-1]
        assert panels["endpoint-pose"]["units"] == ["m", "radian"]
        assert panels["separation"]["covariance"] is None
    view["raw_declaration"]["experiment_id"] = "changed view"
    assert canonical(original) == before
    assert declaration["experiment_id"] != "changed view"


def test_flat_start_lattice_equivalence_is_allowed(native):
    kind, workflow, _, raw, original, _ = native
    if kind != "flat-torus-reference":
        return
    declaration = json.loads(raw)
    declaration["start"] = [1.125, -1.75]
    _check_data(kind, declaration, original["steps"][0]["result"]["data"])


@pytest.mark.parametrize("tamper", ["container", "count", "extra_field", "source_type", "source_hash", "self_source",
    "target", "schema", "numerical_boolean", "admission", "verification_schema", "subject", "outcome", "independent_boolean",
    "method", "runtime_digest", "authority", "reproduction", "verification_id", "replay_id"])
def test_resealed_replay_receipt_refusal(native, tamper):
    _, workflow, _, _, original, replay = native
    changed = deepcopy(replay)
    receipt = changed["replay_receipts"][0]
    verification = receipt["verification"]
    if tamper == "container":
        changed["replay_receipts"] = {"0": receipt}
    elif tamper == "count":
        changed["replay_receipts"].append(deepcopy(receipt))
    elif tamper == "extra_field":
        receipt["extra"] = True
    elif tamper == "source_type":
        receipt["source_bundle_digest"] = False
    elif tamper == "source_hash":
        receipt["source_bundle_digest"] = "sha256:short"
    elif tamper == "self_source":
        receipt["source_bundle_digest"] = changed["bundle_digest"]
        verification["subject_ref"] = changed["bundle_digest"]
    elif tamper == "target":
        receipt["replayed_bundle_digest"] = original["bundle_digest"]
    elif tamper == "schema":
        receipt["schema"] = "ciw.unrelated-replay.v1"
    elif tamper == "numerical_boolean":
        receipt["numerical_match"] = 1
    elif tamper == "admission":
        receipt["admission"] = "performed"
    elif tamper == "verification_schema":
        verification["schema"] = "ciw.unrelated-verification.v1"
    elif tamper == "subject":
        verification["subject_ref"] = changed["bundle_digest"]
    elif tamper == "outcome":
        verification["outcome"] = "failed"
    elif tamper == "independent_boolean":
        verification["independent"] = 0
    elif tamper == "method":
        verification["method"] = "authenticated"
    elif tamper == "runtime_digest":
        verification["runtime_digest"] = "not-a-digest"
    elif tamper == "authority":
        verification["authority"]["physical_truth"] = "established"
    elif tamper == "reproduction":
        verification["reproduction"] = deepcopy(changed["verification"]["reproduction"])
    content = {k: v for k, v in verification.items() if k != "verification_id"}
    verification["verification_id"] = byte_digest(content["schema"].encode() + b"\0" + canonical(content))
    if tamper == "verification_id":
        verification["verification_id"] = "sha256:" + "0" * 64
    receipt["replay_id"] = digest({k: v for k, v in receipt.items() if k != "replay_id"})
    if tamper == "replay_id":
        receipt["replay_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        workflow._validate(changed)


def test_replay_accepts_relocated_runtime_preserving_historical_digest(native):
    kind, workflow, paths, raw, original, _ = native
    relocated_history = deepcopy(original)
    runtime = relocated_history["runtimes"][PINS[kind]["role"]]
    runtime["repository_root"] = "/previous-host-location/provider"
    runtime["python_executable"] = "/previous-host-location/python"
    _seal(relocated_history)
    assert workflow._validate(relocated_history) == raw
    fresh = workflow.replay_session(relocated_history, paths)["session"]
    assert workflow._validate(fresh) == raw
    receipt = fresh["replay_receipts"][0]
    assert receipt["verification"]["runtime_digest"] == digest(relocated_history["runtimes"])
    assert receipt["verification"]["runtime_digest"] != digest(fresh["runtimes"])
    assert receipt["source_bundle_digest"] == relocated_history["bundle_digest"]
    assert receipt["replayed_bundle_digest"] == fresh["bundle_digest"]
