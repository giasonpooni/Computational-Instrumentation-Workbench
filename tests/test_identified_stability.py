"""Selected state/model binding and bounded native discrete stability checks."""
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from ciw import identified_stability as stability
from ciw.adapters.subprocess import PinnedSubprocessAdapter
from ciw.declared_workload import _verification
from ciw.core.canonical import bundle_digest

ROOT = Path(__file__).resolve().parents[1]
make_source = runpy.run_path(str(ROOT / "examples/identified-stability/make_source.py"))["make_source"]


@pytest.fixture(scope="module")
def upstream():
    path = os.environ.get("CIW_IDENTIFIED_STABILITY_UPSTREAM")
    if path is None:
        pytest.skip("set CIW_IDENTIFIED_STABILITY_UPSTREAM to an actual retained identified-design bundle")
    value = json.loads(Path(path).read_bytes())
    stability.identified_design._validate(value)
    return value


@pytest.fixture(scope="module")
def repositories():
    path = os.environ.get("CIW_IDENTIFIED_STABILITY_REPO")
    if path is None:
        pytest.skip("set CIW_IDENTIFIED_STABILITY_REPO to the source-pinned PLSR checkout")
    return {"plsr": Path(path)}


@pytest.fixture(scope="module")
def retained(upstream, repositories):
    source = make_source(upstream)
    raw = b" \n" + stability.canonical(source) + b"\n"
    original = stability.create_session(raw, upstream, repositories)
    replay = stability.replay_session(original, repositories)["session"]
    outside = stability.create_session(stability.canonical(make_source(upstream, level=0.0)), upstream, repositories)
    return raw, original, replay, outside


def test_no_optional_native_import_on_module_load():
    subprocess.run([sys.executable, "-c", "import sys;import ciw.identified_stability;assert 'lyapunov' not in sys.modules"], check=True)


def test_native_selected_state_and_complete_status_semantics(retained, upstream):
    raw, original, replay, outside = retained
    assert stability._validate(original) == raw == stability._validate(replay)
    data = original["steps"][0]["result"]["data"]
    assert original["upstream_design"] == replay["upstream_design"] == upstream
    assert data["sample"]["x"] == upstream["steps"][2]["result"]["data"]["mean"]
    assert data["record"]["code"] in {"NUMERICAL_INCONCLUSIVE", "DECREASE_NOT_DEFINITE", "NOT_CERTIFIED", "MARGIN_LOW", "CERTIFIED_WITH_MARGIN"}
    assert data["record"]["may_authorize"] is False
    assert data["binding"]["parameter_covariance_status"] == "unknown"
    assert data["record"]["proof_status"] == "NOT_CHECKED"
    assert data["record"]["diagnostics"]["P_rate"] is None
    assert outside["steps"][0]["result"]["data"]["record"]["code"] == "OUTSIDE_LEVEL_SET"
    assert original["steps"][0]["numerical_result_id"] == replay["steps"][0]["numerical_result_id"]
    assert original["steps"][0]["execution_id"] != replay["steps"][0]["execution_id"]
    assert original["steps"][0]["result_id"] != replay["steps"][0]["result_id"]
    assert original["verification"]["independent"] is False
    assert original["runtimes"]["plsr"]["source_tree"] == stability.SOURCE_TREE


@pytest.mark.parametrize("fault", ["continuous", "sample-period", "state-order", "unit", "frame", "equilibrium", "A", "state-occurrence", "model-occurrence", "estimator", "parameter-covariance", "state-covariance", "asymmetric-P", "negative-P", "boolean-P", "certificate-units", "duplicate-name", "digest"])
def test_mismatched_model_or_state_refuses(upstream, repositories, fault):
    source = make_source(upstream)
    model = source["model_artifact"]
    if fault == "continuous": model["time"]["convention"] = "continuous"
    elif fault == "sample-period": model["time"]["sample_period_s"] *= 2
    elif fault == "state-order": model["state"]["coordinates"].reverse()
    elif fault == "unit": model["state"]["coordinates"][0]["unit"] = "g"
    elif fault == "frame": source["equilibrium"]["frame_id"] = "another-frame"
    elif fault == "equilibrium": source["equilibrium"]["value"][0] = 1
    elif fault == "A": model["plant"]["A"][0][0] += 0.1
    elif fault == "state-occurrence": source["selection"]["state_result_id"] = "sha256:" + "a" * 64
    elif fault == "model-occurrence": source["selection"]["model_execution_id"] = "execution-" + "a" * 32
    elif fault == "estimator": model["estimator"]["configuration_digest"] = "a" * 64
    elif fault == "parameter-covariance": source["configuration"]["parameter_covariance"] = "declared_zero"
    elif fault == "state-covariance": source["configuration"]["state_covariance"] = "robustly_propagated"
    elif fault == "asymmetric-P": model["certificate"]["P"][0][1] = 1e-15
    elif fault == "negative-P": model["certificate"]["P"][0][0] = -1
    elif fault == "boolean-P": model["certificate"]["P"][0][0] = True
    elif fault == "certificate-units": source["certificate_unit"] = "1"
    elif fault == "duplicate-name": model["state"]["coordinates"][1]["name"] = model["state"]["coordinates"][0]["name"]
    if fault != "digest": model["artifact_digest"] = stability._native_digest(model, {"artifact_digest"})
    else: model["artifact_digest"] = "a" * 64
    with pytest.raises(ValueError):
        stability.create_session(stability.canonical(source), upstream, repositories)


def test_inspection_does_not_execute_any_provider(retained, monkeypatch):
    def forbidden(*a, **k): raise AssertionError("Inspection cannot execute a provider")
    monkeypatch.setattr(PinnedSubprocessAdapter, "__init__", forbidden)
    monkeypatch.setattr(stability.identified_design, "create_session", forbidden)
    monkeypatch.setattr(stability.identified_design, "replay_session", forbidden)
    assert stability._validate(retained[1]) == retained[0]


def _reseal(bundle):
    for step in (bundle["steps"][0], bundle["verification"]["reproduction"]):
        result = step["result"]
        record = result["data"]["record"]
        record["record_digest"] = stability._native_digest(record, {"record_digest"})
        result["result_id"] = stability.digest({k:v for k,v in result.items() if k != "result_id"})
        step.update(result_id=result["result_id"], result_sha256=stability.digest(result),
                    numerical_result={"operation_id": stability.OPERATION, "data": deepcopy(result["data"])})
        step["numerical_result_id"] = stability.digest(step["numerical_result"])
    bundle["bundle_digest"] = bundle_digest(bundle)
    bundle["verification"] = _verification(bundle, bundle["verification"]["reproduction"])


@pytest.mark.parametrize("fault", ["form", "value", "status", "authorization", "covariance", "proof", "tree", "native-boolean", "schema-dependency"])
def test_resealed_fault_cannot_promote_a_native_result(retained, fault):
    bundle = deepcopy(retained[1])
    for step in (bundle["steps"][0], bundle["verification"]["reproduction"]):
        data = step["result"]["data"]
        record = data["record"]
        if fault == "form": record["diagnostics"]["decrease_matrix"][0][0] += 1
        elif fault == "value": record["diagnostics"]["scaled_value"] += 1
        elif fault == "status": record["code"] = "CERTIFIED_WITH_MARGIN"
        elif fault == "authorization": record["may_authorize"] = True
        elif fault == "covariance": data["binding"]["covariance"][0][0] = 0
        elif fault == "native-boolean": record["inequality_certified"] = 0
    if fault == "proof": bundle["verification"]["reproduction"] = deepcopy(bundle["steps"][0])
    elif fault == "tree": bundle["runtimes"]["plsr"]["source_tree"] = "a" * 40
    elif fault == "schema-dependency": bundle["runtimes"]["plsr"]["model_schema_dependency"]["name"] = "another-validator"
    _reseal(bundle)
    with pytest.raises(ValueError): stability._validate(bundle)


def test_replay_of_different_upstream_is_not_interchangeable(retained):
    original = retained[1]
    substituted = deepcopy(original["upstream_design"])
    substituted["session_id"] = "session-" + "a" * 32
    with pytest.raises(ValueError, match="exact selected"):
        stability.validate_upstream(original, substituted)


def test_live_view_is_detached_and_keeps_state_covariance_in_context(retained, monkeypatch):
    from ciw.identified_stability_view import project
    original = retained[1]
    declaration = stability._source(retained[0])
    record = {"native": original, "kind": stability.KIND, "bundle_id": original["bundle_digest"],
              "upstream_bundle_id": original["upstream_binding"]["upstream_bundle_id"]}
    source = {"source_id": "source-1", "evidence_id": stability.byte_digest(retained[0]), "label": "Retained stability"}
    def forbidden(*a, **k): raise AssertionError("Inspection cannot execute a provider")
    monkeypatch.setattr(PinnedSubprocessAdapter, "__init__", forbidden)
    view = project(record, source, declaration, 1)
    assert view["fusion_context"] is None
    assert view["panels"][0]["covariance"] == original["upstream_binding"]["covariance"]
    assert all(p["covariance"] is None for p in view["panels"][1:])
    assert view["object_context"]["physical_stability"] == "not_established"
    view["object_context"]["native_record"]["code"] = "forged"
    assert original["steps"][0]["result"]["data"]["record"]["code"] != "forged"
