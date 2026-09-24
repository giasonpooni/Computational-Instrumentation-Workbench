"""Declared pipeline descriptors bind to the executing code; investigations are the operator surface."""
from copy import deepcopy
from pathlib import Path

import pytest

from ciw import kernel, pipelines
from ciw.instruments import make_demo_run
from ciw.session import Session
from ciw.workbench import OPERATIONS

ROOT = Path(__file__).resolve().parents[1]


def test_every_frozen_kind_has_a_descriptor_bound_to_its_code():
    descriptors = pipelines.check()
    assert set(descriptors) == kernel.FROZEN_KINDS
    for kind, value in descriptors.items():
        assert value["pipeline_id"] == OPERATIONS[kind]
        assert {step["role"]: step["pin"] for step in value["steps"] if "pin" in step} == pipelines.live_pins(kind)


def test_descriptor_pin_drift_is_detected():
    descriptors = deepcopy(pipelines.load())
    step = next(step for step in descriptors["geometric-circle"]["steps"] if "pin" in step)
    step["pin"]["revision"] = "0" * 40
    with pytest.raises(ValueError, match="pins differ"):
        pipelines.check(descriptors)


@pytest.mark.parametrize("change, message", [
    (lambda d: d["steps"][0].pop("pin"), "exact 40-hex pin"),
    (lambda d: d.update(surface="featured"), "surface"),
    (lambda d: d.update(authority="operate"), "read-only"),
    (lambda d: d["verification"].update(role="zzz"), "verification role"),
    (lambda d: d.update(refusals=["ok", "ok"]), "distinct"),
    (lambda d: d["inputs"].update(upstream_cardinality="one"), "upstream"),
    (lambda d: d.update(extra=True), "fields"),
    (lambda d: d["implementation"].update(delegates=["os"]), "delegates"),
])
def test_descriptor_structure_is_validated(change, message):
    descriptor = deepcopy(pipelines.load()["geometric-circle"])
    change(descriptor)
    with pytest.raises(ValueError, match=message):
        pipelines.validate(descriptor)


def test_investigation_chains_must_connect(monkeypatch):
    catalog = pipelines.load_investigations()
    broken = deepcopy(catalog)
    broken["process-balance"]["default_pipeline"] = ["ciw.residual-monitor.v1"]
    monkeypatch.setattr(pipelines, "load_investigations", lambda: broken)
    with pytest.raises(ValueError, match="earlier in its default pipeline"):
        pipelines.check()


def test_provider_matrix_keys_gates_by_pin():
    matrix = pipelines.provider_matrix()
    keys = [(entry["role"], entry["pin"]["revision"]) for entry in matrix]
    assert len(keys) == len(set(keys))
    tbrt = [entry for entry in matrix if entry["role"] == "tbrt"]
    assert len(tbrt) == 1 and len(tbrt[0]["pipelines"]) == 4
    assert len([entry for entry in matrix if entry["role"] == "set"]) == 4


def test_generated_catalog_is_current():
    import runpy
    render = runpy.run_path(str(ROOT / "scripts" / "generate_pipeline_catalog.py"))["render"]
    assert (ROOT / "docs" / "PIPELINES.md").read_text(encoding="utf-8") == render(), \
        "Run python scripts/generate_pipeline_catalog.py"


def _list(session, payload):
    return session.handle({"protocol_version": 1, "request_id": "r", "type": "operation.list", "payload": payload})


def test_operator_sees_investigations_and_inner_pipelines_stay_listed(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    investigations = _list(session, {"view": "investigations"})["payload"]["investigations"]
    assert [item["investigation_id"] for item in investigations] == ["process-balance", "manufacturing-cycle", "geometry-bim"]
    stages = [stage["pipeline_id"] for stage in investigations[1]["default_pipeline"]]
    assert stages == ["ciw.acquired-dataset.v1", "ciw.acquired-calibrated-window.v1", "ciw.residual-monitor.v1"]
    everything = _list(session, {})["payload"]["operations"]
    thermal = next(item for item in everything if item["operation_id"] == "ciw.thermal-observer.v1")
    assert thermal["surface"] == "bench" and thermal["investigations"] == []
    assert _list(session, {"view": "flat"})["payload"]["code"] == "invalid_payload"


def test_refusal_vocabulary_is_the_code_on_the_execution_path():
    descriptors = pipelines.load()
    assert "CALIBRATED_WINDOW_TBRT_REFUSED" in descriptors["acquired-calibrated-window"]["refusals"]  # delegate
    assert "TELEMETRY_RUNTIME_REFUSED" not in descriptors["geometric-circle"]["refusals"]  # helper import only
    changed = deepcopy(descriptors)
    changed["telemetry"]["refusals"] = [code for code in changed["telemetry"]["refusals"] if code != "TELEMETRY_RUNTIME_REFUSED"]
    with pytest.raises(ValueError, match="refusals differ"):
        pipelines.check(changed)


def test_domain_rules_cite_resolvable_code_not_line_numbers():
    descriptors = pipelines.load()
    for value in descriptors.values():
        for rule in value["domain_rules"]:
            for reference in rule["code"]:
                assert pipelines.resolve_symbol(reference) is not None
    value = deepcopy(next(v for v in descriptors.values() if v["domain_rules"]))
    stale = deepcopy(value)
    stale["domain_rules"][0]["evidence"] = "geometric_circle.py:103-114"
    with pytest.raises(ValueError, match="line numbers"):
        pipelines.validate(stale)
    missing = deepcopy(value)
    missing["domain_rules"][0]["code"] = ["ciw.pipelines.runner:NoSuchSymbol"]
    pipelines.validate(missing)
    with pytest.raises(ValueError, match="does not resolve"):
        pipelines.check({**descriptors, missing["source_kind"]: missing})
    unbound = deepcopy(value)
    del unbound["domain_rules"][0]["code"]
    with pytest.raises(ValueError):
        pipelines.validate(unbound)
