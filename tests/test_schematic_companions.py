"""Real selected SRA→JSPT→PLSR runs and adversarial retained bindings."""
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import uuid

import pytest

from ciw.declared_workload import DeclaredWorkflow, _verification
from ciw.schematic_companions import (SchematicCompanionWorkflow, POLICY, SOURCE_SCHEMA,
                                     _source, native_occurrences, numerical_projection)
from ciw.telemetry import canonical, digest, _bundle_digest

ROOT = Path(__file__).resolve().parents[1]


def declaration():
    return json.loads((ROOT / "examples/declared-workloads/schematic-companions.json").read_bytes())


def assessment_source(source):
    value = json.loads((ROOT / "examples/declared-workloads/schematic-assessment.json").read_bytes())
    value["schematic"] = deepcopy(source["schematic"])
    return value


def attributes(source):
    return next(n["attrs"] for n in source["schematic"]["nodes"] if n["id"] == source["function_id"])


@pytest.mark.parametrize("fault", ["unknown-cross", "uncertain-c", "hidden-parameter-covariance", "discrete", "non-equilibrium-claim", "negative-covariance", "nonfinite", "foreign-model", "function-not-node", "nonlinear-as-linear", "bool-coordinate", "omitted-c"])
def test_refuse_undeclared_scientific_semantics(fault):
    source = declaration()
    attrs = attributes(source)
    if fault == "unknown-cross": source["configuration"]["cross_covariance"] = "unknown"
    if fault == "uncertain-c": source["configuration"]["model_parameter_uncertainty"] = "unknown"
    if fault == "hidden-parameter-covariance": attrs["parameter_covariance"] = "unknown"
    if fault == "discrete": source["configuration"]["model_semantics"] = "discrete_transition"
    if fault == "non-equilibrium-claim": source["configuration"]["equilibrium_status"] = "certified"
    if fault == "negative-covariance": attrs["sigma_x"] = [[-1.0]]
    if fault == "nonfinite": attrs["x_star"] = [float("inf")]
    if fault == "foreign-model": attrs["model_ref"] = "jspt.reference.exp2"
    if fault == "function-not-node": source["function_id"] = "observer"
    if fault == "nonlinear-as-linear": attrs["class"] = "linear"
    if fault == "bool-coordinate": attrs["x_star"] = [True]
    if fault == "omitted-c": attrs.pop("c")
    with pytest.raises(ValueError): _source(canonical(source))


@pytest.fixture(scope="module")
def bindings():
    value = os.environ.get("CIW_SCHEMATIC_STACK_ROOT")
    if not value:
        pytest.skip("Set CIW_SCHEMATIC_STACK_ROOT with pinned sra/jspt/plsr checkouts")
    return {role: Path(value) / role for role in ("sra", "jspt", "plsr")}


def execute(source, bindings):
    upstream = DeclaredWorkflow("schematic-assessment").create_session(canonical(assessment_source(source)), {"sra": bindings["sra"]})
    selected = deepcopy(source)
    selected["schematic"] = deepcopy(upstream["steps"][0]["result"]["data"]["schematic"])
    original = SchematicCompanionWorkflow().create_session(canonical(selected), upstream, bindings)
    return selected, upstream, original


@pytest.fixture(scope="module")
def retained(bindings):
    source, upstream, original = execute(declaration(), bindings)
    replay = SchematicCompanionWorkflow().replay_session(original, bindings)["session"]
    target = os.environ.get("CIW_SCHEMATIC_FIXTURE_DIR")
    if target:
        output = Path(target)
        output.mkdir(parents=True, exist_ok=True)
        for name, value in (("source", source), ("upstream", upstream), ("original", original), ("replay", replay)):
            (output / (name + ".json")).write_bytes(canonical(value))
    return source, upstream, original, replay


def test_actual_three_provider_results_and_fresh_replay(retained):
    source, upstream, original, replay = retained
    assert original["upstream_assessment"] == upstream
    assert original["upstream_binding"]["result_id"] == upstream["steps"][0]["result_id"]
    assert original["steps"][0]["input_refs"][1] == upstream["steps"][0]["result_id"]
    old, fresh = original["steps"][0], replay["steps"][0]
    assert old["numerical_result_id"] == fresh["numerical_result_id"]
    assert old["execution_id"] != fresh["execution_id"] and old["result_id"] != fresh["result_id"]
    first = old["result"]["data"]["events"]
    second = fresh["result"]["data"]["events"]
    assert all(e["result"] == "SAMPLED" for e in first)
    assert first[0]["detail"]["execution_ref"] != second[0]["detail"]["execution_ref"]
    assert first[0]["detail"]["A"] == [[-1.0]]
    assert first[1]["detail"]["rank"] == 1 and first[1]["detail"]["invisible_dim"] == 0
    assert first[2]["detail"]["sigma_y"] == [[0.04]]
    assert first[3]["detail"]["V"] == 0.5 and first[3]["detail"]["decrease"] == -1.0
    assert first[3]["detail"]["verdict"] == "certified"
    assert old["result"]["data"]["scope"]["equilibrium_status"] == "not_established"
    assert old["result"]["data"]["scope"]["observability"] == "not_evaluated"
    assert original["verification"]["independent"] is False
    assert SchematicCompanionWorkflow()._validate(replay) == canonical(source)


def test_native_occurrence_sets_exclude_upstream_and_are_fresh(retained):
    source, upstream, original, replay = retained
    first, second = native_occurrences(original), native_occurrences(replay)
    assert len(first) == len(second) == 2 and first.isdisjoint(second)
    assert upstream["steps"][0]["execution_id"] not in first
    injected = deepcopy(original)
    injected["verification"]["reproduction"]["result"]["data"]["events"][0]["detail"]["execution_ref"] = injected["steps"][0]["result"]["data"]["events"][0]["detail"]["execution_ref"]
    with pytest.raises(ValueError, match="distinct native"):
        native_occurrences(injected)


@pytest.mark.parametrize("change,expected", [("unknown", ["NOT_ELIGIBLE"] * 4),
    ("missing-covariance", ["SAMPLED", "SAMPLED", "NOT_ELIGIBLE", "SAMPLED"]),
    ("zero-derivative", ["SAMPLED", "SAMPLED", "SAMPLED", "REFUSED"])])
def test_native_held_conditions_and_singular_lyapunov(bindings, change, expected):
    source = declaration()
    attrs = attributes(source)
    if change == "unknown": attrs["class"] = "unknown"
    if change == "missing-covariance": attrs.pop("sigma_x")
    if change == "zero-derivative": attrs["x_star"] = [0.0]
    _, _, result = execute(source, bindings)
    data = result["steps"][0]["result"]["data"]
    assert [event["result"] for event in data["events"]] == expected
    assert next(n for n in data["schematic"]["nodes"] if n["id"] == "observer")["attrs"]["status"] == "UNRESOLVED"


def test_native_rank_uses_declared_absolute_threshold(bindings):
    source = declaration()
    attributes(source)["c"] = 1e-11
    _, _, result = execute(source, bindings)
    data = result["steps"][0]["result"]["data"]
    assert data["events"][1]["detail"]["rank"] == 0
    assert data["events"][1]["detail"]["invisible_dim"] == 1
    assert data["scope"]["rank_absolute_tolerance"] == 1e-10


def test_prior_stale_numeric_evidence_is_retained_in_selected_upstream(bindings):
    source = declaration()
    source["schematic"]["nodes"].append({"id": "old-jacobian", "kind": "certificate", "attrs": {"owner": "jspt", "result": "SAMPLED", "A": [[-99.0]], "fixture": True}})
    source["schematic"]["edges"].append({"src": "old-jacobian", "dst": "f", "kind": "linearizes", "attrs": {}})
    selected, upstream, result = execute(source, bindings)
    old = next(n for n in result["steps"][0]["result"]["data"]["schematic"]["nodes"] if n["id"] == "old-jacobian")
    assert old["attrs"]["A"] == [[-99.0]] and old["attrs"]["result"] == "NOT_ELIGIBLE"
    assert old["attrs"]["historical_result"] == "SAMPLED"
    raw_source = json.loads(base64.b64decode(upstream["source"]["evidence"][0]["bytes_b64"]))
    original_old = next(n for n in raw_source["schematic"]["nodes"] if n["id"] == "old-jacobian")
    assert original_old["attrs"]["result"] == "SAMPLED"


def test_selected_upstream_must_match_graph_exactly(retained, bindings):
    source, upstream, _, _ = retained
    changed = deepcopy(source)
    attributes(changed)["c"] = 0.75
    with pytest.raises(ValueError, match="exactly match"):
        SchematicCompanionWorkflow().create_session(canonical(changed), upstream, bindings)


def reseal(bundle):
    for step in (bundle["steps"][0], bundle["verification"]["reproduction"]):
        result = step["result"]
        result["result_id"] = digest({k: v for k, v in result.items() if k != "result_id"})
        step["result_id"] = result["result_id"]
        step["result_sha256"] = digest(result)
        step["numerical_result"] = numerical_projection(result["data"])
        step["numerical_result_id"] = digest(step["numerical_result"])
    bundle["bundle_digest"] = _bundle_digest(bundle)
    bundle["verification"] = _verification(bundle, bundle["verification"]["reproduction"])


@pytest.mark.parametrize("fault", ["foreign-jacobian", "covariance", "lyapunov", "global-verdict", "observer-promotion", "linearization-input", "native-reuse", "upstream-result", "upstream-input", "provider",
    "zero-visible-basis", "nonunit-visible-basis", "boolean-singular", "boolean-event-rank", "boolean-function-rank",
    "empty-decisions-before", "empty-decisions-after", "duplicate-decision", "decision-owner", "decision-reason"])
def test_fully_resealed_bad_bindings_are_refused(retained, fault):
    bundle = deepcopy(retained[2])
    for step in (bundle["steps"][0], bundle["verification"]["reproduction"]):
        data = step["result"]["data"]
        nodes = {n["id"]: n for n in data["schematic"]["nodes"]}
        if fault == "foreign-jacobian": nodes["cert:jspt.cov:f"]["attrs"]["source_result_ref"] = "sha256:" + "0" * 64
        if fault == "covariance":
            nodes["cert:jspt.cov:f"]["attrs"]["sigma_y"] = [[99.0]]
            data["events"][2]["detail"]["sigma_y"] = [[99.0]]
        if fault == "lyapunov":
            nodes["cert:lyapunov:f"]["attrs"]["decrease"] = 1.0
            data["events"][3]["detail"]["decrease"] = 1.0
        if fault == "global-verdict":
            nodes["cert:lyapunov:f"]["attrs"]["verdict"] = "nonlinear_globally_stable"
            data["events"][3]["detail"]["verdict"] = "nonlinear_globally_stable"
        if fault == "observer-promotion": nodes["observer"]["attrs"]["status"] = "OBSERVABLE"
        if fault == "linearization-input": nodes["cert:jspt:f"]["attrs"]["binding"]["call_inputs"]["parameters"]["c"] = 99.0
        if fault == "zero-visible-basis": nodes["cert:jspt.structure:f"]["attrs"]["visible"] = [[0.0]]
        if fault == "nonunit-visible-basis": nodes["cert:jspt.structure:f"]["attrs"]["visible"] = [[2.0]]
        if fault == "boolean-singular": nodes["cert:jspt.structure:f"]["attrs"]["singular_values"] = [True]
        if fault == "boolean-event-rank": data["events"][1]["detail"]["rank"] = True
        if fault == "boolean-function-rank": nodes["f"]["attrs"]["rank"] = True
        if fault == "empty-decisions-before": data["decisions_before"] = []
        if fault == "empty-decisions-after": data["decisions_after"] = []
        if fault == "duplicate-decision": data["decisions_after"][1] = deepcopy(data["decisions_after"][0])
        if fault == "decision-owner": data["decisions_after"][0]["owner"] = "plsr"
        if fault == "decision-reason": data["decisions_after"][0]["reason"] = ""
    if fault == "native-reuse":
        bundle["verification"]["reproduction"]["result"]["data"] = deepcopy(bundle["steps"][0]["result"]["data"])
    if fault == "upstream-result": bundle["upstream_binding"]["result_id"] = "sha256:" + "0" * 64
    if fault == "upstream-input":
        for step in (bundle["steps"][0], bundle["verification"]["reproduction"]):
            step["input_refs"][1] = "sha256:" + "0" * 64
            step["result"]["input_refs"][1] = "sha256:" + "0" * 64
    if fault == "provider": bundle["runtimes"]["sra"]["companions"]["jspt"]["revision"] = "0" * 40
    reseal(bundle)
    with pytest.raises(ValueError): SchematicCompanionWorkflow()._validate(bundle)


def test_inspection_does_not_bind_or_execute_runtime(retained, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("Inspection must not execute a provider")
    monkeypatch.setattr(SchematicCompanionWorkflow, "_adapters", forbidden)
    assert SchematicCompanionWorkflow()._validate(retained[2]) == canonical(retained[0])


def test_fresh_outer_wrappers_cannot_reuse_native_occurrences(retained):
    from ciw.workbench import _validate_links

    original = retained[2]
    duplicate = deepcopy(original)
    duplicate["session_id"] = "session-" + uuid.uuid4().hex
    for step in (duplicate["steps"][0], duplicate["verification"]["reproduction"]):
        step["execution_id"] = "execution-" + uuid.uuid4().hex
        step["result"]["execution_ref"] = step["execution_id"]
    reseal(duplicate)
    # It is structurally valid in isolation; global freshness belongs to the
    # retained Workbench registry, not a guessed source of native UUIDs.
    SchematicCompanionWorkflow()._validate(duplicate)
    def record(native):
        return {"kind": "schematic-companions", "source_id": "source-test", "upstream_bundle_id": retained[1]["bundle_digest"],
                "bundle_id": native["bundle_digest"], "native": native}
    with pytest.raises(ValueError, match="fresh native execution"):
        _validate_links(record(duplicate), {original["bundle_digest"]: record(original)})
