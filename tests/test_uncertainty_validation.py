"""Shared CIW lifecycle and statistical semantics of the uncertainty validation operation."""

import base64
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from ciw import consistency_math as consistency
from ciw import uncertainty_validation as validation
from ciw.instruments import make_demo_run
from ciw.session import Session
from ciw.telemetry import _bundle_digest, canonical, digest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "examples" / "uncertainty-validation"
OPERATION = "ciw.uncertainty-validation.v1"


def _load(name):
    return json.loads((FIXTURES / (name + ".json")).read_text(encoding="utf-8"))


def _call(session, kind, payload):
    response = session.handle({"protocol_version": 1, "request_id": "test-" + kind, "type": kind, "payload": payload})
    assert response["type"] == "response", response
    return response["payload"]


def _refused(session, kind, payload):
    response = session.handle({"protocol_version": 1, "request_id": "refused-" + kind, "type": kind, "payload": payload})
    assert response["type"] == "error", response
    return response["payload"]["message"]


def _add(session, source):
    return _call(session, "source.add", {"kind": "uncertainty-validation", "label": source["experiment_id"],
                                         "bytes_b64": base64.b64encode(canonical(source)).decode()})["source_id"]


def _data(session, source):
    completed = _call(session, "operation.execute", {"operation_id": OPERATION,
                                                      "parameters": {"source_id": _add(session, source)}})
    return completed, _call(session, "bundle.get", {"bundle_id": completed["bundle_id"]})["steps"][0]["result"]["data"]


def test_committed_fixtures_are_the_generator_output():
    spec = importlib.util.spec_from_file_location("uncertainty_fixture_generator", FIXTURES / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated = module.build_sources()
    assert set(generated) == {"consistent", "covariance-too-small", "covariance-too-large", "unknown-dependence"}
    for name, source in generated.items():
        assert canonical(source) == canonical(_load(name)), name


def test_consistent_fixture_lifecycle_save_reopen_and_replay(tmp_path):
    session = Session(make_demo_run(), tmp_path / "original")
    operation = next(item for item in _call(session, "operation.list", {})["operations"]
                     if item["operation_id"] == OPERATION)
    assert operation["available"] is True and operation["role"] == "uncertainty_consistency"
    completed, data = _data(session, _load("consistent"))
    assert data["verdict"] == {"status": "consistent", "failures": [], "authority": "statistical"}
    assert data["nees"]["status"] == consistency.CONSISTENT and data["nees"]["dof"] == 128
    assert data["nees"]["mean_band"][0] <= data["nees"]["mean"] <= data["nees"]["mean_band"][1]
    assert data["nis"]["status"] == consistency.CONSISTENT and data["nis"]["dof"] == 64
    assert [item["status"] for item in data["coverage"]["components"]] == [consistency.CONSISTENT] * 2
    assert [item["name"] for item in data["coverage"]["components"]] == ["position", "velocity"]
    assert [item["status"] for item in data["bias"]["components"]] == [consistency.UNBIASED] * 2
    assert data["statistical_scope"] == "finite_sample_consistency_under_declared_independence"
    assert data["authority"]["reference_uncertainty"] == "not_modelled_reference_treated_as_exact"
    view = _call(session, "experiment.inspect", {"bundle_id": completed["bundle_id"]})
    assert view["object_context"]["object_kind"] == "uncertainty_consistency"
    assert view["object_context"]["verdict"]["status"] == "consistent"
    assert [panel["panel_id"] for panel in view["panels"]] == ["coverage", "normalized-squares"]
    assert view["panels"][0]["labels"] == ["position", "velocity"]
    assert view["panels"][1]["labels"] == ["nees", "nis"]
    listed = _call(session, "result.list", {})["results"]
    assert any(item["operation_id"] == OPERATION for item in listed)

    saved = session.save_workspace(tmp_path / "saved.json")
    original_create = validation.UncertaintyValidationWorkflow.create_session
    validation.UncertaintyValidationWorkflow.create_session = lambda *args, **kwargs: pytest.fail("restore executed a provider")
    try:
        reopened = Session.from_workspace(saved, tmp_path / "reopened")
    finally:
        validation.UncertaintyValidationWorkflow.create_session = original_create
    original = _call(session, "bundle.get", {"bundle_id": completed["bundle_id"]})
    assert _call(reopened, "bundle.get", {"bundle_id": completed["bundle_id"]}) == original
    replay = _call(reopened, "bundle.replay", {"bundle_id": completed["bundle_id"]})
    fresh = _call(reopened, "bundle.get", {"bundle_id": replay["bundle"]["bundle_id"]})
    assert fresh["steps"][0]["execution_id"] != original["steps"][0]["execution_id"]
    assert fresh["steps"][0]["numerical_result_id"] == original["steps"][0]["numerical_result_id"]
    assert replay["replay_receipt"]["numerical_match"] is True


def test_declared_covariance_scale_is_separated_not_merged(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    _, small = _data(session, _load("covariance-too-small"))
    assert small["nees"]["status"] == consistency.TOO_SMALL and small["nis"]["status"] == consistency.TOO_SMALL
    assert [item["status"] for item in small["coverage"]["components"]] == [consistency.UNDER] * 2
    assert small["verdict"]["status"] == "inconsistent"
    assert "nees:" + consistency.TOO_SMALL in small["verdict"]["failures"]
    assert "coverage:position:" + consistency.UNDER in small["verdict"]["failures"]
    _, large = _data(session, _load("covariance-too-large"))
    assert large["nees"]["status"] == consistency.TOO_LARGE and large["nis"]["status"] == consistency.TOO_LARGE
    assert large["nees"]["mean"] < large["nees"]["mean_band"][0]
    assert consistency.UNDER not in [item["status"] for item in large["coverage"]["components"]]
    assert large["verdict"]["status"] == "inconsistent"
    assert large["verdict"]["failures"][0] == "nees:" + consistency.TOO_LARGE


def test_unknown_dependence_reports_the_same_numbers_as_diagnostics_only(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    _, consistent = _data(session, _load("consistent"))
    _, unknown = _data(session, _load("unknown-dependence"))
    assert unknown["nees"] == consistent["nees"] and unknown["coverage"] == consistent["coverage"]
    assert unknown["verdict"]["authority"] == "diagnostic_only"
    assert unknown["statistical_scope"] == "diagnostic_only_cross_sample_dependence_unknown"


@pytest.mark.parametrize("mutate,message", [
    (lambda r: r["samples"][3].__setitem__("covariance", [[0.01, 0.005], [0.005, 0.0025]]), "positive definite"),
    (lambda r: r.__setitem__("confidence", 0.3), "confidence"),
    (lambda r: r["samples"][5].__setitem__("time", r["samples"][4]["time"]), "strictly increase"),
    (lambda r: r["samples"][2].__setitem__("estimate", [1.0]), "exactly 2"),
    (lambda r: r.pop("measurement_names"), "measurement_names"),
    (lambda r: r.__setitem__("reference_origin", "measured_truth"), "reference_origin"),
    (lambda r: r.__setitem__("cross_sample_dependence", "assumed"), "cross_sample_dependence"),
    (lambda r: r["samples"][0]["covariance"][0].__setitem__(1, 0.02), "symmetric"),
])
def test_malformed_requests_are_refused_before_retention(tmp_path, mutate, message):
    session = Session(make_demo_run(), tmp_path)
    source = _load("consistent")
    mutate(source["request"])
    assert message in _refused(session, "source.add", {"kind": "uncertainty-validation", "label": "bad",
                                                        "bytes_b64": base64.b64encode(canonical(source)).decode()})
    assert _call(session, "source.list", {})["sources"] == []


def test_replay_refuses_changed_reference_identity(tmp_path, monkeypatch):
    session = Session(make_demo_run(), tmp_path)
    completed, _ = _data(session, _load("consistent"))
    old_identity = validation.runtime_identity

    def changed_identity():
        value = deepcopy(old_identity())
        value["algorithm"]["code_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(validation, "runtime_identity", changed_identity)
    assert "runtime identity" in _refused(session, "bundle.replay", {"bundle_id": completed["bundle_id"]})


def _reseal(bundle):
    workflow = validation.UncertaintyValidationWorkflow()
    for step in (bundle["steps"][0], bundle["verification"]["reproduction"]):
        result = step["result"]
        result["result_id"] = digest({key: value for key, value in result.items() if key != "result_id"})
        step["result_id"] = result["result_id"]
        step["result_sha256"] = digest(result)
        step["numerical_result"] = {"operation_id": validation.OPERATION, "data": deepcopy(result["data"])}
        step["numerical_result_id"] = digest(step["numerical_result"])
    bundle["bundle_digest"] = _bundle_digest(bundle)
    bundle["verification"] = workflow._verification(bundle, bundle["verification"]["reproduction"])
    return bundle


def test_retained_statistics_tolerate_rounding_but_not_changed_conclusions(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    completed, _ = _data(session, _load("consistent"))
    bundle = session.workbench.get_bundle(completed["bundle_id"])
    workflow = validation.UncertaintyValidationWorkflow()
    rounded = deepcopy(bundle)
    for step in (rounded["steps"][0], rounded["verification"]["reproduction"]):
        step["result"]["data"]["nees"]["mean"] *= 1 + 1e-13
    workflow._validate(_reseal(rounded))
    promoted = deepcopy(bundle)
    for step in (promoted["steps"][0], promoted["verification"]["reproduction"]):
        step["result"]["data"]["nees"]["mean"] *= 1.5
    with pytest.raises(ValueError, match="differs numerically"):
        workflow._validate(_reseal(promoted))
    relabelled = deepcopy(bundle)
    for step in (relabelled["steps"][0], relabelled["verification"]["reproduction"]):
        step["result"]["data"]["verdict"]["authority"] = "physical"
    with pytest.raises(ValueError, match="differs from the deterministic reference"):
        workflow._validate(_reseal(relabelled))
