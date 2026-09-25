"""Probes from the system audit that became permanent tests."""

import base64
import logging

import pytest
from pathlib import Path


from ciw import uncertainty_validation as validation
from ciw.instruments import make_demo_run
from ciw.session import Session

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "uncertainty-validation" / "consistent.json"
OPERATION = "ciw.uncertainty-validation.v1"


def _call(session, kind, payload, request_id="audit"):
    return session.handle({"protocol_version": 1, "request_id": request_id, "type": kind, "payload": payload})


def test_a_defect_inside_one_request_answers_an_internal_error_envelope_and_leaves_the_session_usable(tmp_path, monkeypatch, caplog):
    session = Session(make_demo_run(), tmp_path)
    source = _call(session, "source.add", {"kind": "uncertainty-validation", "label": "u",
                                           "bytes_b64": base64.b64encode(SOURCE.read_bytes()).decode()})["payload"]
    def boom(self, raw, repositories):
        raise RuntimeError("provider blew up unexpectedly")
    monkeypatch.setattr(validation.UncertaintyValidationWorkflow, "create_session", boom)
    with caplog.at_level(logging.ERROR, logger="ciw.session"):
        response = _call(session, "operation.execute", {"operation_id": OPERATION, "parameters": {"source_id": source["source_id"]}}, "req-7")
    assert response["type"] == "error" and response["request_id"] == "req-7"
    assert response["payload"]["code"] == "internal_error"
    assert "blew up" not in response["payload"]["message"], "internal details stay in the log"
    assert any("req-7" in record.getMessage() and record.exc_info for record in caplog.records)
    assert session.workbench.pending_operations == 0 and session.workbench.list_bundles() == []
    monkeypatch.undo()
    completed = _call(session, "operation.execute", {"operation_id": OPERATION, "parameters": {"source_id": source["source_id"]}})
    assert completed["type"] == "response" and len(session.workbench.list_bundles()) == 1
    listed = _call(session, "execution.list", {})["payload"]["executions"]
    assert {entry["status"] for entry in listed} == {"completed"}


def test_the_operation_catalogue_page_indexes_every_operation():
    from ciw.workbench import OPERATIONS
    catalogue = (ROOT / "docs" / "INSTRUMENTS.md").read_text(encoding="utf-8")
    protocol = (ROOT / "docs" / "PROTOCOL.md").read_text(encoding="utf-8")
    for kind, operation in OPERATIONS.items():
        assert f"`{operation}`" in catalogue and f"`{kind}`" in catalogue, operation
    for code in ("invalid_request", "invalid_payload", "unsupported_version", "internal_error", "storage_error", "read_only_view"):
        assert f"`{code}`" in protocol, code


def test_retained_provider_bytes_are_decoded_with_the_same_nonfinite_guard_as_the_wire():
    from ciw import geodesic, investigation
    with pytest.raises(ValueError, match="Nonfinite JSON number"):
        geodesic._parse(b'{"schema": "gte.circle-request.v1", "observations": [NaN], "constraint": {}, "policy": {}}')
    import inspect
    assert "_reject_constant" in inspect.getsource(investigation._validate_batch)


def test_unknown_workspace_and_oscillator_run_keys_are_refused_on_reopen(tmp_path):
    import json
    from ciw.adapters.oscillator import validate_run
    session = Session(make_demo_run(), tmp_path / "s")
    saved = session.save_workspace(tmp_path / "w.json")
    document = json.loads(saved.read_text(encoding="utf-8"))
    for extra in ("audit_extra", "results_shadow"):
        tampered = dict(document, **{extra: 1})
        (tmp_path / "t.json").write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown workspace keys"):
            Session.from_workspace(tmp_path / "t.json", tmp_path / "r")
    tampered = json.loads(json.dumps(document))
    tampered["run"]["audit_extra"] = {"stash": True}
    (tmp_path / "t.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown run keys"):
        Session.from_workspace(tmp_path / "t.json", tmp_path / "r")
    run = make_demo_run()
    validate_run(dict(run, run_schema="run.v1"))
    with pytest.raises(ValueError, match="Unknown run keys"):
        validate_run(dict(run, sidecar=[]))
    Session.from_workspace(saved, tmp_path / "clean")


def test_ill_conditioned_covariances_are_refused_without_a_numerical_warning():
    import warnings
    from ciw import uncertainty_validation as validation
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="condition number"):
            validation._definite_covariance([[1e300, 0.0], [0.0, 1e-300]], 2, "covariance")
        with pytest.raises(ValueError, match="condition number"):
            validation._definite_covariance([[1.0, 1.0], [1.0, 1.0]], 2, "covariance")
        assert validation._definite_covariance([[2.0, 0.5], [0.5, 1.0]], 2, "covariance") == [[2.0, 0.5], [0.5, 1.0]]


@pytest.mark.parametrize("kind,example,changes", [
    ("calibrated-observable", "examples/calibrated-observable/source.json", [
        lambda v: v["channels"][0]["observation"].__setitem__("indicated_value", ""),
        lambda v: v["configuration"]["gsie"]["prior"]["covariance"][1].__setitem__(0, True),
        lambda v: v["channels"][1]["clock_joint_covariance"].__setitem__(1, [0.0, 0.0]),
        lambda v: v["configuration"]["cbsr"]["constraints"]["coefficients"][0].__setitem__(0, {}),
        lambda v: v["configuration"]["fdir"]["fault_signatures"].__setitem__("sensor:tank-1:bias", ["a", 1]),
        lambda v: v["configuration"].__setitem__("alignment", []),
        lambda v: v["configuration"]["gsie"].__setitem__("prior_measurement_crosscov_policy", 1e308)]),
    ("calibrated-window", "examples/calibrated-window/source.json", [
        lambda v: v["configuration"]["window"].__setitem__("sample_period", []),
        lambda v: v["configuration"]["gsie"]["prior"]["covariance"][0].__setitem__(0, ""),
        lambda v: v["configuration"]["window"].update(start=v["configuration"]["window"]["end"] + 1)]),
])
def test_provider_backed_sources_refuse_non_numeric_quantities_before_retention(kind, example, changes):
    import json
    from copy import deepcopy
    from ciw.workbench import Workbench
    source = json.loads((ROOT / example).read_text(encoding="utf-8"))
    def add(value):
        return Workbench().add_source({"kind": kind, "label": "probe", "bytes_b64": base64.b64encode(json.dumps(value).encode()).decode()})
    assert add(source)["kind"] == kind
    for change in changes:
        mutated = deepcopy(source)
        change(mutated)
        with pytest.raises(ValueError):
            add(mutated)


def test_identified_design_inputs_are_refused_before_retention_when_no_prior_could_complete_them():
    import json
    from copy import deepcopy
    from ciw.workbench import Workbench
    source = json.loads((ROOT / "examples" / "identified-design" / "source.json").read_text(encoding="utf-8"))
    def add(value):
        return Workbench().add_source({"kind": "identified-design", "label": "probe", "bytes_b64": base64.b64encode(json.dumps(value).encode()).decode()})
    assert add(source)["source_schema"] == source["schema"]
    changes = [
        lambda v: v["identification"]["training"]["states"][2].__setitem__(0, ""),
        lambda v: v["identification"]["training"]["inputs"][3].__setitem__(0, {}),
        lambda v: v["identification"]["training"]["sample_times"].__setitem__(1, ""),
        lambda v: v["identification"]["training"]["sample_times"].__setitem__(1, v["identification"]["training"]["sample_times"][0]),
        lambda v: v["design"]["state_scales"].__setitem__(0, True),
        lambda v: v["design"]["candidates"][2]["noise_covariance"][0].__setitem__(0, ""),
        lambda v: v["design"]["candidates"][0]["observation_matrix"][0].__setitem__(1, {}),
        lambda v: v["token_admission"]["yield_claim"].__setitem__("expected_rank_delta", {}),
        lambda v: v["token_admission"]["yield_claim"].__setitem__("expected_new_morphism", 1),
        lambda v: v["prediction"]["process_covariance"].__setitem__(0, [1.0]),
        lambda v: v.__setitem__("extra", 1),
    ]
    for change in changes:
        mutated = deepcopy(source)
        change(mutated)
        with pytest.raises(ValueError):
            add(mutated)
    with pytest.raises(ValueError, match="finite, unambiguous JSON"):
        Workbench().add_source({"kind": "identified-design", "label": "probe", "bytes_b64": base64.b64encode(b'{"schema": "ciw.identified-design-input.v1", "a": NaN}').decode()})


def test_a_file_beyond_the_reopen_budget_is_refused_by_its_size_before_it_is_read(tmp_path):
    import os
    from ciw import session as session_module
    from ciw.cli import main
    path = tmp_path / "huge.json"
    path.write_bytes(b"{}")
    os.truncate(path, session_module.READ_LIMIT + 1)  # a sparse file: the size without the bytes
    with pytest.raises(ValueError, match="reopen budget"):
        session_module.read_json(path)
    assert main(["workspace", "verify", str(path)]) == 2
    os.truncate(path, 2)
    assert session_module.read_json(path) == {}
