"""Historical applicability and current expiry are distinct, read-only facts."""

from copy import deepcopy
import json

import pytest

from ciw.calibration_status import calibration_status
from ciw.cli import print_investigation
from ciw.core.identities import evidence_id
from ciw.instruments import make_demo_run
from ciw.session import Session


def source_record(version="v2"):
    profile = {"calibration_id": "declared-calibration",
               "valid_from": "2026-01-01T00:00:00Z", "valid_until": "2026-02-01T00:00:00Z"}
    record = {"schema": f"measurement-record.{version}", "observed_at": "2026-01-15T12:00:00Z"}
    if version == "v2":
        record["acquisition_applicability"] = {
            "observed_at": record["observed_at"], "applicable": True,
            "valid_from": profile["valid_from"], "valid_until": profile["valid_until"],
            "basis": "caller_declared_acquisition_time",
        }
    return {"name": "sensor-a", "measurement": {"calibration": profile, "records": [record]}}


def status_run(version="v2"):
    return {"metadata": {"rci_source": {"sensors": [source_record(version)]}}}


@pytest.mark.parametrize("instant,current,expired,future", [
    ("2025-12-31T23:59:59Z", False, False, True),
    ("2026-01-01T00:00:00Z", True, False, False),
    ("2026-01-31T23:59:59Z", True, False, False),
    ("2026-02-01T00:00:00Z", False, True, False),
    ("2026-02-01T01:00:00+01:00", False, True, False),
])
def test_current_interval_is_half_open_and_does_not_change_acquisition(instant, current, expired, future):
    run = status_run()
    original = json.dumps(run, sort_keys=True)
    status = calibration_status(run, instant)[0]
    assert status["acquisition"]["applicable_at_acquisition"] is True
    assert status["acquisition"]["provenance"] == "persisted.v2"
    assert status["serving"]["current_applicability"] is current
    assert status["serving"]["expired"] is expired
    assert status["serving"]["not_yet_valid"] is future
    assert json.dumps(run, sort_keys=True) == original


@pytest.mark.parametrize("instant", ["2026-01-15", "2026-01-15T12:00:00", "2026-01-15T12:00Z",
                                     "2026-01-15 12:00:00Z", "2026-02-30T00:00:00Z", True, 0])
def test_evaluated_at_requires_explicit_valid_aware_timestamp(instant):
    with pytest.raises(ValueError, match="evaluated_at"):
        calibration_status(status_run(), instant)


def test_legacy_applicability_is_a_derivation_and_legacy_timestamp_bytes_survive():
    run = status_run("v1")
    sensor = run["metadata"]["rci_source"]["sensors"][0]
    sensor["measurement"]["calibration"]["valid_from"] = "2026-01-01T00:00+00:00"
    sensor["measurement"]["records"][0]["observed_at"] = "2026-01-15 12:00:00+00:00"
    original = deepcopy(run)
    acquisition = calibration_status(run, "2026-09-20T12:00:00Z")[0]["acquisition"]
    assert acquisition["applicable_at_acquisition"] is True
    assert acquisition["provenance"] == "derived.legacy-v1"
    assert acquisition["basis"] == "legacy_declared_acquisition_time"
    assert run == original
    assert "acquisition_applicability" not in sensor["measurement"]["records"][0]


@pytest.mark.parametrize("mutation", [
    lambda a: a.update(applicable=1),
    lambda a: a.update(applicable=False),
    lambda a: a.update(valid_until="2027-02-01T00:00:00Z"),
    lambda a: a.update(observed_at="2026-01-16T12:00:00Z"),
])
def test_inconsistent_persisted_applicability_is_refused(mutation):
    run = status_run()
    mutation(run["metadata"]["rci_source"]["sensors"][0]["measurement"]["records"][0]["acquisition_applicability"])
    with pytest.raises(ValueError, match="contradicts"):
        calibration_status(run, "2026-09-20T00:00:00Z")


def request(session, kind, **payload):
    return session.handle({"protocol_version": 1, "request_id": "status-read", "type": kind, "payload": payload})


def test_session_read_surfaces_agree_and_leave_saved_result_bytes_unchanged(tmp_path):
    # Exercise the serving boundary with deterministic built-in science and
    # synthetic declared calibration metadata, without binding an RCI runtime.
    run = make_demo_run()
    run["metadata"].update(status_run()["metadata"])
    run["evidence_id"] = evidence_id(run)
    session = Session(run, tmp_path)
    result = request(session, "analysis.stats")["payload"]
    session.save_workspace(tmp_path / "workspace.json")
    originals = {path: path.read_bytes() for path in tmp_path.iterdir()}
    result_before = deepcopy(session.results)
    instant = "2026-09-20T12:00:00Z"
    snapshot = request(session, "session.get", evaluated_at=instant)["payload"]
    listed = request(session, "result.list", evaluated_at=instant)["payload"]
    fetched = request(session, "result.get", result_id=result["result_id"], evaluated_at=instant)
    assert snapshot["calibration"] == listed["calibration"] == fetched["calibration"]
    assert snapshot["calibration"][0]["serving"]["expired"] is True
    assert fetched["payload"] == result
    assert "calibration" not in fetched["payload"]
    assert session.results == result_before and session.run == run
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == originals


@pytest.mark.parametrize("kind", ["session.get", "result.list", "result.get"])
@pytest.mark.parametrize("instant", [None, "2026-01-01T00:00:00", True])
def test_read_protocol_rejects_explicit_non_aware_or_null_time(tmp_path, kind, instant):
    session = Session(make_demo_run(), tmp_path)
    result = request(session, "analysis.stats")["payload"]
    payload = {"evaluated_at": instant}
    if kind == "result.get":
        payload["result_id"] = result["result_id"]
    response = request(session, kind, **payload)
    assert response["type"] == "error"
    assert response["payload"]["code"] == "invalid_payload"


def test_terminal_displays_off_diagonal_covariance_and_both_validity_times(capsys):
    summary = {"status": "completed", "calibration": calibration_status(status_run(), "2026-09-20T12:00:00Z"),
               "covariances": [{"name": "joint state", "quantity_ids": ["mass-a", "mass-b"],
                                "units": ["kg", "kg"], "matrix": [[2.0, -0.125], [-0.125, 3.0]],
                                "frame": "reservoir-mass", "method": "declared", "covariance_id": "sha256:example"}]}
    original = deepcopy(summary)
    print_investigation(summary)
    output = capsys.readouterr().out
    for text in ("Acquisition applicable: true", "expired: true", "2026-09-20T12:00:00+00:00",
                 "mass-a, mass-b", "[2, -0.125]", "[-0.125, 3]", "unit[i] * unit[j]", "sha256:example"):
        assert text in output
    assert summary == original


def test_new_operation_version_reaches_admission_without_loading_a_provider(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    response = request(session, "operation.execute", operation_id="unbound-provider.v2", parameters={})
    assert response["type"] == "response"
    assert response["payload"]["status"] == "refused"
    assert response["payload"]["execution"]["refusal"]["code"] == "operation_unavailable"
    assert response["payload"]["result"] is None
