"""The shared runner owns every identity-bearing record; pipelines supply hooks."""
from copy import deepcopy
import json

import pytest

from ciw.adapters.protocol import AdapterRefusal
from ciw.core.canonical import bundle_digest, canonical, digest
from ciw.pipelines import _check_runner
from ciw.pipelines.runner import (PipelineRunner, check_receipt_envelope, check_receipts, check_step, host_projection,
                                  seal_step)

TREE = "a" * 40
PIN = {"role": "toy", "revision": "b" * 40, "source_tree": TREE, "module": "toy.provider", "source_root": "src"}


class FakeAdapter:
    def __init__(self, repository, identity):
        self.repository, self.identity, self.calls = repository, identity, 0

    def runtime_identity(self):
        return deepcopy(self.identity)


def runtime(root="/host/a", python="/usr/bin/python3"):
    return {"schema": "ciw.subprocess-runtime.v1", "adapter_version": "ciw-pinned-subprocess-v1",
            "repository_root": root, "revision": PIN["revision"], "source_tree": TREE, "module": PIN["module"],
            "source_root": PIN["source_root"], "python_executable": python, "python_sha256": "c" * 64,
            "python_version": "3.12.3", "dependencies": {"numpy": "2.4.3"}}


class ToyRunner(PipelineRunner):
    LABEL = "Toy"

    def __init__(self, identity=None, refuse=False, drift=None):
        super().__init__("toy-kind", PIN)
        self.identity, self.refuse, self.drift = identity or runtime(), refuse, drift

    def parse_source(self, raw):
        value = json.loads(raw)
        if set(value) != {"schema", "experiment_id", "configuration", "values"} or value["schema"] != self.SOURCE_SCHEMA:
            raise ValueError("Toy source is not declared")
        return value

    def make_adapter(self, repository, retained):
        return FakeAdapter(repository, self.identity)

    def invoke(self, source, bound):
        adapter = bound[0]
        adapter.calls += 1
        if self.refuse:
            raise AdapterRefusal("TOY_REFUSED", "The toy provider refused")
        if self.drift and adapter.calls == self.drift:
            adapter.identity = {**adapter.identity, "source_tree": "d" * 40}
        return {"sum": sum(source["values"])}

    def check_data(self, source, data):
        if data != {"sum": sum(source["values"])}:
            raise ValueError("Toy result differs from its source")


def source_bytes():
    return canonical({"schema": "ciw.toy-kind-source.v1", "experiment_id": "toy", "configuration": {"mode": "sum"},
                      "values": [1, 2, 3]})


def test_create_seal_verify_and_reopen_without_the_provider():
    runner = ToyRunner()
    bundle = runner.create_session(source_bytes(), {"toy": "/host/a"})
    assert bundle["schema"] == "ciw.toy-kind-session.v1" and bundle["steps"][0]["operation_id"] == "ciw.toy-kind.v1"
    assert bundle["bundle_digest"] == bundle_digest(bundle)
    step, proof = bundle["steps"][0], bundle["verification"]
    assert proof["reproduction"]["execution_id"] != step["execution_id"]
    assert proof["reproduction"]["numerical_result_id"] == step["numerical_result_id"]
    assert proof["independent"] is False and proof["subject_ref"] == bundle["bundle_digest"]
    # Reopen is validation of retained records; it never binds a provider.
    offline = ToyRunner()
    offline.make_adapter = lambda *_: pytest.fail("Reopen bound a provider")
    assert offline._validate(json.loads(canonical(bundle))) == source_bytes()


def test_replay_is_a_fresh_occurrence_with_a_bound_receipt():
    runner = ToyRunner()
    original = runner.create_session(source_bytes(), {"toy": "/host/a"})
    # Host paths are bindings, not pins: another checkout location replays.
    replayed = ToyRunner(runtime(root="/host/b", python="/opt/python3")).replay_session(original, {"toy": "/host/b"})
    fresh, receipt = replayed["session"], replayed["replay_receipt"]
    assert fresh["session_id"] != original["session_id"] and fresh["bundle_digest"] != original["bundle_digest"]
    assert fresh["steps"][0]["numerical_result_id"] == original["steps"][0]["numerical_result_id"]
    assert receipt["source_bundle_digest"] == original["bundle_digest"] and receipt["admission"] == "not_performed"
    ToyRunner()._validate(fresh)
    for mutate in (lambda r: r.update(numerical_match=False), lambda r: r.update(source_bundle_digest=fresh["bundle_digest"]),
                   lambda r: r["verification"].update(independent=True)):
        changed = deepcopy(fresh)
        mutate(changed["replay_receipts"][0])
        with pytest.raises(ValueError):
            check_receipts(changed, "toy-kind")
    doubled = deepcopy(fresh)
    doubled["replay_receipts"].append(deepcopy(receipt))
    with pytest.raises(ValueError, match="at most one"):
        check_receipts(doubled, "toy-kind")


def _reseal(receipt):
    receipt["replay_id"] = digest({key: value for key, value in receipt.items() if key != "replay_id"})


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r.update(schema="ciw.other-kind-replay.v1"), "does not bind"),
    (lambda r: r.update(numerical_match=False), "does not bind"),
    (lambda r: r.update(admission="performed"), "does not bind"),
    (lambda r: r.update(replayed_bundle_digest="sha256:" + "2" * 64), "does not bind"),
    (lambda r: r.update(source_bundle_digest="not-a-digest"), "does not bind"),
    (lambda r: r.update(extra=True), "Invalid record fields"),
    (lambda r: r["verification"].update(subject_ref="sha256:" + "3" * 64), "verification_id"),
    (lambda r: r.update(verification=[]), "must be a record"),
])
def test_a_provider_shaped_replay_receipt_binds_its_fresh_bundle(mutate, message):
    original = ToyRunner().create_session(source_bytes(), {"toy": "/host/a"})
    fresh = ToyRunner().replay_session(original, {"toy": "/host/a"})["session"]
    check_receipt_envelope(fresh, "toy-kind")
    changed = deepcopy(fresh)
    mutate(changed["replay_receipts"][0])
    _reseal(changed["replay_receipts"][0])
    with pytest.raises(ValueError, match=message):
        check_receipt_envelope(changed, "toy-kind")
    # The resealed subject is the only content change; its identity no longer holds.
    moved = deepcopy(fresh)
    proof = moved["replay_receipts"][0]["verification"]
    proof["subject_ref"] = "sha256:" + "4" * 64
    proof["verification_id"] = "sha256:" + __import__("hashlib").sha256(
        proof["schema"].encode() + b"\0" + canonical({k: v for k, v in proof.items() if k != "verification_id"})).hexdigest()
    _reseal(moved["replay_receipts"][0])
    with pytest.raises(ValueError, match="subject differs from replay source"):
        check_receipt_envelope(moved, "toy-kind")
    doubled = deepcopy(fresh)
    doubled["replay_receipts"].append(deepcopy(doubled["replay_receipts"][0]))
    with pytest.raises(ValueError, match="retains one receipt"):
        check_receipt_envelope(doubled, "toy-kind")


def test_replay_refuses_a_different_pin():
    original = ToyRunner().create_session(source_bytes(), {"toy": "/host/a"})
    moved = runtime()
    moved["dependencies"] = {"numpy": "2.5.0"}
    with pytest.raises(ValueError, match="runtime differs from the retained execution"):
        ToyRunner(moved).replay_session(original, {"toy": "/host/a"})
    other = runtime()
    other["source_tree"] = "e" * 40
    with pytest.raises(ValueError, match="source tree differs"):
        ToyRunner(other).create_session(source_bytes(), {"toy": "/host/a"})


def test_binding_refusal_and_drift():
    with pytest.raises(ValueError, match="declared provider roles"):
        ToyRunner().create_session(source_bytes(), {"toy": "/a", "other": "/b"})
    with pytest.raises(AdapterRefusal):
        ToyRunner(refuse=True).create_session(source_bytes(), {"toy": "/host/a"})
    with pytest.raises(ValueError, match="changed during execution"):
        ToyRunner(drift=1).create_session(source_bytes(), {"toy": "/host/a"})


@pytest.mark.parametrize("path, value", [
    (("steps", 0, "result", "data", "sum"), 7),
    (("steps", 0, "result", "authority", "state_admission"), "performed"),
    (("steps", 0, "input_refs", 0), "sha256:" + "0" * 64),
    (("steps", 0, "execution_id"), "execution-" + "0" * 31),
    (("steps", 0, "numerical_result_id"), "sha256:" + "1" * 64),
    (("runtimes", "toy", "revision"), "f" * 40),
    (("configuration", "mode"), "product"),
])
def test_every_identity_is_checked_on_reopen(path, value):
    bundle = ToyRunner().create_session(source_bytes(), {"toy": "/host/a"})
    target = bundle
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    bundle["bundle_digest"] = bundle_digest(bundle)
    with pytest.raises(ValueError):
        ToyRunner()._validate(bundle)


def test_sealed_step_binds_request_result_and_numbers():
    source = json.loads(source_bytes())
    step = seal_step("toy", "ciw.toy-kind.v1", source, ["sha256:" + "2" * 64], {"sum": 6})
    assert step["result_id"] == digest({k: v for k, v in step["result"].items() if k != "result_id"})
    check = dict(role="toy", operation="ciw.toy-kind.v1", source=source, input_refs=["sha256:" + "2" * 64],
                 check_data=lambda *_: None, label="Toy")
    check_step(step, **check)
    changed = deepcopy(step)
    changed["request"]["values"] = [9]
    with pytest.raises(ValueError, match="request differs"):
        check_step(changed, **check)


def test_host_fields_never_count_as_pins():
    nested = {"repository_root": "/x", "revision": "r", "vendor": {"python_executable": "/y", "revision": "v"},
              "dependencies": {"numpy": "2.4.3", "python_executable": "/kept"}}
    # Host bindings are dropped from the runtime and its nested provider runtime
    # only; every other value, dependencies included, is compared exactly.
    assert host_projection(nested) == {"revision": "r", "vendor": {"revision": "v"},
                                       "dependencies": {"numpy": "2.4.3", "python_executable": "/kept"}}


class ScriptRunner(ToyRunner):
    def invoke(self, source, bound):
        code, raw = self.run_provider(bound, "script", [], b"")
        if code:
            raise AdapterRefusal("TOY_REFUSED", "The toy provider refused")
        return json.loads(raw)


class ScriptAdapter(FakeAdapter):
    def _run(self, script, arguments, payload):
        self.identity = {**self.identity, "source_tree": "d" * 40}
        return 1, b"not json"


def test_drift_during_a_run_outranks_the_providers_own_refusal():
    runner = ScriptRunner()
    runner.make_adapter = lambda repository, retained: ScriptAdapter(repository, runtime())
    with pytest.raises(ValueError, match="changed during execution") as caught:
        runner.create_session(source_bytes(), {"toy": "/host/a"})
    assert not isinstance(caught.value, AdapterRefusal)


def test_generic_runner_classes_supply_hooks_only():
    _check_runner("toy", "generic_runner", ToyRunner())

    class Overrides(ToyRunner):
        def _step(self, source, evidence_id, bound):
            return {}

    with pytest.raises(ValueError, match="overrides runner methods"):
        _check_runner("toy", "generic_runner", Overrides())
    with pytest.raises(ValueError, match="shared runner"):
        _check_runner("toy", "generic_runner", object())
    with pytest.raises(ValueError, match="declare it generic_runner"):
        _check_runner("toy", "hand_written", ToyRunner())


def test_stage_chain_seals_cumulative_order_and_check_chain_refuses_every_rebinding():
    from ciw.pipelines.runner import StageChain, check_chain
    operations = {"first": "ciw.first.v1", "second": "ciw.second.v1", "third": "ciw.third.v1"}
    chain = StageChain("sha256:" + "e" * 64)
    for role, operation in operations.items():
        chain.seal(role, operation, {"role": role}, {"value": role})
    stages = chain.stages
    assert stages[2]["input_refs"] == ["sha256:" + "e" * 64, stages[0]["result_id"], stages[1]["result_id"]]
    check_chain(deepcopy(stages), operations, "sha256:" + "e" * 64, seen={"execution-" + "0" * 32}, label="Probe")

    def reseal(stage):
        result = stage["result"]
        result["result_id"] = digest({k: v for k, v in result.items() if k != "result_id"})
        stage["result_id"], stage["result_sha256"] = result["result_id"], digest(result)

    swapped = deepcopy(stages)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    reused = deepcopy(stages)
    lineage = deepcopy(stages)
    lineage[2]["input_refs"] = lineage[2]["result"]["input_refs"] = lineage[2]["input_refs"][:2]
    reseal(lineage[2])
    for broken, seen in ((swapped, set()), (stages[:2], set()), (lineage, set()),
                         (reused, {stages[1]["execution_id"]})):
        with pytest.raises(ValueError):
            check_chain(deepcopy(broken), operations, "sha256:" + "e" * 64, seen=set(seen), label="Probe")


def test_workbench_reads_native_occurrences_from_the_workflow_not_the_kind():
    from ciw import workbench
    for kind in ("variational-free-energy", "measurement-chain", "schematic-companions"):
        flow = workbench._workflow(kind)
        assert flow.native_occurrences is not PipelineRunner.native_occurrences and flow.FRESH_OCCURRENCE_MESSAGE
    plain = workbench._workflow("geometric-circle")
    assert plain.native_occurrences({}) is None and plain.identity_claims({}) == {} and plain.catalog_steps({}) == []
    assert workbench._native_hook({"schema": "ciw.unknown-kind-session.v1"}, "catalog_steps") is None
    source = (workbench.__file__ and open(workbench.__file__, encoding="utf-8").read())
    assert "from .free_energy_workflow import native_occurrences" not in source


def test_acquired_windows_sharing_an_occurrence_are_refused_through_the_hook():
    from ciw import workbench

    def record(bundle_id, execution, result):
        return {"kind": "acquired-calibrated-window", "bundle_id": bundle_id, "upstream_bundle_id": None,
                "native": {"steps": [{"execution_id": execution, "result_id": result}]}}

    first = record("a", "execution-1", "result-1")
    workbench._validate_links(first, {"a": first, "b": record("b", "execution-2", "result-2")})
    for other in (record("b", "execution-1", "result-2"), record("b", "execution-2", "result-1")):
        with pytest.raises(ValueError, match="fresh native execution and result"):
            workbench._validate_links(first, {"a": first, "b": other})


def test_every_upstream_kind_binds_its_upstream_through_its_workflow():
    from ciw import identified_design, workbench
    for kind in workbench.UPSTREAM_KINDS:
        assert callable(getattr(workbench._workflow(kind), "validate_upstream", None)), kind
    identified_design.validate_upstream({"upstream": {"a": 1}}, {"a": 1})
    with pytest.raises(ValueError, match="Design upstream must exactly match"):
        identified_design.validate_upstream({"upstream": {"a": 1}}, {"a": 2})
    record = {"kind": "identified-design", "bundle_id": "d", "upstream_bundle_id": "u", "native": {"upstream": {"a": 1}}}
    upstream = {"kind": "calibrated-observable", "bundle_id": "u", "native": {"a": 2}}
    with pytest.raises(ValueError, match="Design upstream must exactly match"):
        workbench._validate_links(record, {"d": record, "u": upstream})


def test_declared_and_reproduced_kinds_follow_the_descriptor_verification_method():
    from ciw import pipelines, workbench
    methods = {kind: value["verification"]["method"] for kind, value in pipelines.load().items()}
    assert "proved-heat" in workbench.DECLARED_KINDS - workbench.REPRODUCED_KINDS
    assert not {"telemetry", "calibrated-observable", "identified-design"} & workbench.DECLARED_KINDS
    assert workbench.CONTRACT_KINDS == {"instrument-exchange"}
    assert not workbench.CONTRACT_KINDS & workbench.REPRODUCED_KINDS
    for kind in workbench.REPRODUCED_KINDS:
        assert pipelines.workflow(kind).PROFILE.verify_method == methods[kind], kind


def test_a_descriptor_cannot_declare_a_verification_method_its_workflow_does_not_record():
    import copy
    from ciw import pipelines
    for kind, method in (("geometric-circle", "fresh_registered_guest_verification"),
                         ("energy-accuracy", "same_runtime_fresh_occurrence_reproduction"),
                         ("telemetry", "pinned_set_contract_validation"),
                         ("instrument-exchange", "pinned_set_replay_verification")):
        descriptors = copy.deepcopy(pipelines.load())
        descriptors[kind]["verification"]["method"] = method
        with pytest.raises(ValueError, match=kind + ": descriptor verification method"):
            pipelines.check(descriptors)


def test_a_declared_record_validates_beside_a_retained_exchange_record():
    from ciw import workbench
    # A contract-validated exchange retains no reproduction of its own.
    record = {"kind": "numerical-heat", "bundle_id": "a", "upstream_bundle_id": None,
              "native": {"steps": [{"execution_id": "e1"}], "verification": {"reproduction": {"execution_id": "e2"}}}}
    exchange = {"kind": "instrument-exchange", "bundle_id": "b", "upstream_bundle_id": None,
                "native": {"steps": [{"execution_id": "x1"}], "verification": {"verifier_ref": "set"}}}
    workbench._validate_links(record, {"a": record, "b": exchange})
    exchange["native"]["steps"][0]["execution_id"] = "e2"
    with pytest.raises(ValueError, match="distinct execution and reproduction occurrences"):
        workbench._validate_links(record, {"a": record, "b": exchange})


def test_verification_records_never_alias_the_profile_or_the_reproduced_step():
    from ciw.pipelines.runner import DECLARED, verification
    bundle = {"bundle_digest": "sha256:" + "1" * 64, "runtimes": {}, "steps": [{"numerical_result": {"x": 1}}]}
    reproduced = {"numerical_result": {"x": 1}}
    value = verification(bundle, reproduced)
    value["authority"]["state_admission"] = "performed"
    value["reproduction"]["numerical_result"]["x"] = 2
    assert DECLARED.authority["state_admission"] == "not_performed" and reproduced["numerical_result"]["x"] == 1
