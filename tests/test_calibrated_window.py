"""Actual provider composition through the shared session, plus refusal boundaries."""
import base64
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from ciw import calibrated_window as window
from ciw.cli import parser
from ciw.instruments import make_demo_run
from ciw.session import Session
from ciw.core.canonical import bundle_digest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "examples/calibrated-window/source.json").read_bytes()


def call(session, kind, payload=None, *, error=False):
    response = session.handle({"protocol_version": 1, "request_id": "window-test", "type": kind, "payload": payload or {}})
    assert response["type"] == ("error" if error else "response"), response
    return response["payload"]


def add(session, raw):
    return call(session, "source.add", {"kind": "calibrated-window", "label": "Calibrated level window",
                                       "bytes_b64": base64.b64encode(raw).decode()})


@pytest.fixture(scope="module")
def repositories():
    root = os.environ.get("CIW_CALIBRATED_WINDOW_STACK_ROOT")
    if not root:
        pytest.skip("set CIW_CALIBRATED_WINDOW_STACK_ROOT to exact role-named provider checkouts")
    return {role: Path(root) / role for role in window.ROLES}


@pytest.fixture(scope="module")
def retained(repositories, tmp_path_factory):
    directory = tmp_path_factory.mktemp("calibrated-window")
    session = Session(make_demo_run(), directory)
    session.workbench.bind_workflow("calibrated-window", repositories)
    raw = b"\n" + SOURCE + b"\n "
    source = add(session, raw)
    summary = call(session, "operation.execute", {"operation_id": "ciw.calibrated-window.v1", "parameters": {"source_id": source["source_id"]}})
    original = call(session, "bundle.get", {"bundle_id": summary["bundle_id"]})
    replay = call(session, "bundle.replay", {"bundle_id": summary["bundle_id"]})
    fresh = call(session, "bundle.get", {"bundle_id": replay["bundle"]["bundle_id"]})
    path = session.save_workspace(directory / "workspace.json")
    fixture_root = os.environ.get("CIW_WINDOW_FIXTURE_DIR")
    if fixture_root:
        root = Path(fixture_root); root.mkdir(parents=True, exist_ok=True)
        for name, bundle in (("original", original), ("replay", fresh)):
            (root / (name + ".json")).write_bytes(window.canonical(bundle))
    return session, source, original, fresh, path, raw


def test_operation_joins_existing_session_and_startup(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    source = add(session, SOURCE)
    assert source["kind"] == "calibrated-window"
    operations = call(session, "operation.list")["operations"]
    entry, = [o for o in operations if o["operation_id"] == "ciw.calibrated-window.v1"]
    assert entry["available"] is False
    call(session, "operation.execute", {"operation_id": entry["operation_id"], "parameters": {"source_id": source["source_id"]}}, error=True)
    assert call(session, "bundle.list")["bundles"] == []
    args = parser().parse_args(["serve", "--calibrated-window-stack-root", "/trusted/window", "--telemetry-stack-root", "/trusted/telemetry"])
    assert args.calibrated_window_stack_root == Path("/trusted/window")


def test_experiment_projection_preserves_joint_covariance_and_replay_occurrences(retained, monkeypatch):
    session, source, original, fresh, path, raw = retained
    def forbidden(*args, **kwargs):
        raise AssertionError("View must not recompute native science")
    monkeypatch.setattr(window, "create_session", forbidden)
    monkeypatch.setattr(window, "replay_session", forbidden)
    before = session.workbench.serialize()
    view = call(session, "experiment.inspect", {"bundle_id": original["bundle_digest"]})
    panels = {p["panel_id"]: p for p in view["panels"]}
    assert panels["indications"]["values"] == [2, 4]
    assert panels["indications"]["context"]["event_times"] == [10, 12]
    assert panels["aligned-time"]["values"] == [0, 1]
    assert panels["aligned-time"]["covariance"] == [[.25, .25], [.25, .5]]
    assert panels["measurements"]["values"] == [5, 9]
    assert panels["measurements"]["covariance"] == [[5.5, 3.5], [3.5, 8.5]]
    assert panels["measurements"]["context"]["joint_time_value_covariance"][0][2] == .125
    assert panels["feature"]["values"] == [7]
    assert panels["feature"]["covariance"] == [[5.25]]
    assert panels["state"]["values"] == [1.12]
    assert panels["innovation"]["covariance"] == [[6.25]]
    assert panels["posterior-residual"]["covariance"] is None
    assert view["raw_observations"] == json.loads(raw)["samples"]
    assert [n["input_refs"] for n in view["graph"]["nodes"]] == [s["input_refs"] for s in original["steps"]]
    assert view["verification"] == original["verification"]
    assert view["fusion_context"]["observability"]["status"] == "unresolved"
    replay = call(session, "experiment.inspect", {"bundle_id": fresh["bundle_digest"]})
    assert replay["bundle_id"] != view["bundle_id"]
    assert [p["values"] for p in replay["panels"]] == [p["values"] for p in view["panels"]]
    assert [n["execution_id"] for n in replay["graph"]["nodes"]] != [n["execution_id"] for n in view["graph"]["nodes"]]
    view["panels"][0]["values"][0] = 999
    assert session.workbench.serialize() == before
    restored = Session.from_workspace(path, path.parent / "view-restored")
    assert call(restored, "experiment.inspect", {"bundle_id": fresh["bundle_digest"]}) == replay


@pytest.mark.parametrize("attack", ["unknown", "diagonal_claim", "non_psd", "order", "nonlinear", "missing_map", "coefficient_block", "different_sensor", "moving_state", "lossy_raw", "boolean_gain", "lossy_clock"])
def test_undeclared_inputs_refuse_before_provider_execution(attack, monkeypatch):
    source = json.loads(SOURCE)
    if attack == "unknown": source["joint_covariance"]["cross_covariance_policy"] = "unknown"
    elif attack == "diagonal_claim": source["joint_covariance"]["cross_covariance_policy"] = "declared_zero"
    elif attack == "non_psd": source["joint_covariance"]["matrix"][4][5] = source["joint_covariance"]["matrix"][5][4] = 2
    elif attack == "order": source["joint_covariance"]["order"].reverse()
    elif attack == "nonlinear": source["configuration"]["composition"]["transformation"] = "quadratic"
    elif attack == "missing_map": del source["clock_model"]
    elif attack == "coefficient_block": source["calibration_profile"]["coefficient_covariance"][0][0] = 1
    elif attack == "different_sensor": source["samples"][1]["sensor_id"] = "sensor:other"
    elif attack == "moving_state": source["configuration"]["gsie"]["dynamics"]["matrix"] = [[2]]
    elif attack == "lossy_raw": source["samples"][0]["raw_value"] = 2**54 + 1
    elif attack == "boolean_gain": source["calibration_profile"]["gain"] = True
    else: source["clock_model"]["device_origin"] = 2**54 + 1
    def deny(*a, **k): raise AssertionError("invalid source reached executable binding")
    monkeypatch.setattr(window, "_adapters", deny)
    with pytest.raises(ValueError): window.create_session(window.canonical(source), {})


@pytest.mark.parametrize("field", ["epoch", "valid_from", "valid_until"])
def test_timestamp_precision_refuses_before_provider_execution(field, monkeypatch):
    source = json.loads(SOURCE)
    target = source if field == "epoch" else source["calibration_profile"]
    target[field] = "2026-01-01T00:00:00.0000009Z"

    def deny(*args, **kwargs):
        pytest.fail("Timestamp precision must be validated before binding providers")

    monkeypatch.setattr(window, "_adapters", deny)
    with pytest.raises(ValueError, match="microsecond precision"):
        window.create_session(window.canonical(source), {})


def test_representable_timestamp_spellings_preserve_source_bytes():
    source = json.loads(SOURCE)
    source["epoch"] = "2026-01-01 00:00:00.123456000+00:00"
    source["calibration_profile"]["valid_from"] = "2025-12-01T01:00:00,123456000+01:00"
    source["calibration_profile"]["valid_until"] = "2026-12-01T00:00:00.000000000Z"
    assert window._source(window.canonical(source)) == source


def test_shared_parameter_covariance_and_native_lineage(retained):
    session, source, original, _, _, raw = retained
    assert window._validate(original) == raw
    assert base64.b64decode(call(session, "source.get", {"source_id": source["source_id"]})["bytes_b64"]) == raw
    clock, calibration, feature, state = original["steps"]
    assert [row["event_time"] for row in clock["result"]["data"]["samples"]] == [0, 1]
    assert clock["result"]["data"]["temporal_covariance"] == [[.25, .25], [.25, .5]]
    calibrated = calibration["result"]["data"]
    assert [row["raw_value"] for row in calibrated["samples"]] == [2, 4]
    assert [row["corrected_value"] for row in calibrated["samples"]] == [5, 9]
    assert calibrated["temporal_covariance"] == [[5.5, 3.5], [3.5, 8.5]]
    assert calibrated["joint_time_value_covariance"][0][2:] == [.125, .125]
    assert feature["numerical_result"]["mean"] == 7
    assert feature["numerical_result"]["variance"] == 5.25
    # Affine mean interchange oracle, including the single shared profile:
    assert feature["numerical_result"]["variance"] == 2**2 * .625 + 3**2 * .25 + .5
    assert state["result"]["result_artifact"]["components"][0]["value"] == pytest.approx(1.12)
    assert state["result"]["result_artifact"]["covariance"]["matrix"][0][0] == pytest.approx(.84)
    assert feature["request"]["source_batch_ref"] == calibration["result_id"]
    assert original["verification"]["outcome"] == "passed"


def test_replay_preserves_numbers_and_creates_fresh_occurrences(retained):
    _, _, original, replay, _, _ = retained
    assert original["session_id"] != replay["session_id"]
    for old, fresh in zip(original["steps"], replay["steps"]):
        assert old["numerical_result_id"] == fresh["numerical_result_id"]
        assert old["execution_id"] != fresh["execution_id"]
        assert old["result_id"] != fresh["result_id"]
    assert replay["replay_receipts"][0]["source_bundle_digest"] == original["bundle_digest"]


def test_instruments_fusion_restore_and_candidate_boundary(retained, monkeypatch, tmp_path):
    session, _, original, _, path, _ = retained
    for instrument in ("tbrt", "mcur", "stfe", "gsie"):
        view = call(session, "instrument.inspect", {"bundle_id": original["bundle_digest"], "instrument": instrument})
        assert view["step"] == next(s for s in original["steps"] if s["runtime_ref"] == instrument)
        assert view["fusion_context"]["observability"]["status"] == "unresolved"
        assert view["fusion_context"]["calibration_validity"] == "checked_at_nominal_mapped_event_times"
        assert view["fusion_context"]["state_admission"] == "not_performed"
    def deny(*a, **k): raise AssertionError("restore executed scientific provider")
    monkeypatch.setattr(window, "_adapters", deny)
    restored = Session.from_workspace(path, tmp_path / "restored")
    assert restored.workbench.fusion_contexts() == session.workbench.fusion_contexts()
    assert not next(o for o in restored.workbench.describe_operations() if o["operation_id"] == "ciw.calibrated-window.v1")["available"]
    call(restored, "operation.execute", {"operation_id": "esm.inspect-candidate.v1", "parameters": {
        "bundle_id": original["bundle_digest"], "inspected_at": "2026-09-22T00:00:00Z"}}, error=True)


@pytest.mark.parametrize("attack", ["clock_expiry", "calibration_expiry", "late_sample"])
def test_native_refusal_does_not_retain_partial_state(attack, repositories, tmp_path):
    source = json.loads(SOURCE)
    if attack == "clock_expiry": source["clock_model"]["valid_device_interval"] = [10, 11]
    elif attack == "calibration_expiry": source["calibration_profile"]["valid_until"] = source["epoch"]
    else: source["samples"][1]["received_at"] = 1.5
    session = Session(make_demo_run(), tmp_path)
    session.workbench.bind_workflow("calibrated-window", repositories)
    descriptor = add(session, window.canonical(source))
    call(session, "operation.execute", {"operation_id": "ciw.calibrated-window.v1", "parameters": {"source_id": descriptor["source_id"]}}, error=True)
    assert session.workbench.list_bundles() == []
    assert session.workbench.fusion_contexts() == []
    assert session.workbench.pending_operations == 0


def test_resealed_request_cannot_drop_shared_uncertainty(retained):
    bundle = deepcopy(retained[2])
    step = bundle["steps"][2]
    step["request"]["uncertainty"]["matrix"][0][1] = step["request"]["uncertainty"]["matrix"][1][0] = 0
    step["request_sha256"] = window.digest(step["request"])
    bundle["bundle_digest"] = bundle_digest(bundle)
    with pytest.raises(ValueError, match="lineage"):
        window._validate(bundle)


@pytest.mark.parametrize("replacement", [True, 1.0], ids=["boolean", "float"])
@pytest.mark.parametrize("target", ["configuration", "request"])
def test_equal_python_numbers_cannot_substitute_retained_json(retained, target, replacement):
    bundle = deepcopy(retained[2])
    bundle.pop("verification", None)
    configuration = (bundle["configuration"] if target == "configuration"
                     else bundle["steps"][0]["request"]["source"]["configuration"])
    matrix = configuration["gsie"]["dynamics"]["matrix"]
    assert type(matrix[0][0]) is int
    matrix[0][0] = replacement
    bundle["bundle_digest"] = bundle_digest(bundle)
    with pytest.raises(ValueError, match="configuration binding|lineage"):
        window._validate(bundle)


def test_rehashed_numerical_projection_must_preserve_json_number_type(retained):
    bundle = deepcopy(retained[2])
    bundle.pop("verification", None)
    step = bundle["steps"][0]
    sample = step["numerical_result"]["data"]["samples"][0]
    assert sample["event_time"] == 0.0
    sample["event_time"] = False
    step["numerical_result_id"] = window.digest(step["numerical_result"])
    bundle["bundle_digest"] = bundle_digest(bundle)
    with pytest.raises(ValueError, match="Numerical projection"):
        window._validate(bundle)
