"""Unit boundary tests with explicitly synthetic proof bytes.

These test doubles exercise persistence and orchestration, NEVER cryptographic
success. The real SP1 acceptance gate is in test_proved_heat_session.py.
"""
import base64
from copy import deepcopy
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

from ciw import proved_heat as module
from ciw.declared_workload import HEAT_DESCRIPTOR, _commit
from ciw.proved_heat import ProvedHeatWorkflow, POLICY, PIN, GUEST_SHA256, _verification
from ciw.core.canonical import canonical, byte_digest, digest, bundle_digest


def source():
    return {"schema": "ciw.proved-heat-source.v1", "experiment_id": "unit-test-double-not-a-real-proof",
        "initial_values": [0, 100, 200, 100, 0], "steps": 4, "configuration": deepcopy(POLICY)}


def fake_runtime():
    """Synthetic runtime metadata for offline binding tests only."""
    result = {"schema": "ciw.subprocess-runtime.v1", "adapter_version": "ciw-pinned-subprocess-v1",
        "repository_root": "/unit-test/scr", **PIN, "python_executable": "/unit-test/python",
        "python_sha256": "a" * 64, "python_version": "3.12.0", "dependencies": {"numpy": None, "scipy": None}}
    for role in module.BINARY_LIMITS:
        result[role] = {"sha256": "sha256:" + (GUEST_SHA256 if role == "guest" else "b" * 64),
            "byte_count": 42, "source_binding": "registered_guest_sha256" if role == "guest" else "operator_asserted_not_attested"}
    return result


def fake_data(value=None):
    """Return a structurally bound test double, not a valid SP1 proof."""
    value = value or source()
    values = value["initial_values"][:]
    for _ in range(value["steps"]):
        updated = values[:]
        for i in range(1, len(values) - 1):
            change = values[i - 1] - 2 * values[i] + values[i + 1]
            updated[i] += (1 if change >= 0 else -1) * (abs(change) // 4)
        values = updated
    inputs = struct.pack("<II", value["steps"], len(values)) + b"".join(struct.pack("<q", v) for v in value["initial_values"])
    output = b"".join(struct.pack("<q", v) for v in values)
    native = {"specification": {"program": HEAT_DESCRIPTOR.hex(), "configuration": "", "input_payload": inputs.hex()},
        "specification_identity": _commit("specification", [HEAT_DESCRIPTOR, b"", inputs]),
        "program_identity": _commit("program", [HEAT_DESCRIPTOR]), "input_identity": _commit("input", [inputs]),
        "engine_occurrence": 0, "status": "completed", "exit_code": 0, "output": output.hex(),
        "output_identity": _commit("output", [output]), "computation_identity": "", "detail": None, "values": values}
    native["computation_identity"] = _commit("computation", [bytes.fromhex(native[k]) for k in
        ("program_identity", "input_identity", "output_identity")] + [struct.pack("<I", 0)])
    proof_bytes = b"SYNTHETIC UNIT TEST DOUBLE - NOT A CRYPTOGRAPHIC PROOF"
    proof = {"bytes_b64": base64.b64encode(proof_bytes).decode(), "sha256": byte_digest(proof_bytes), "byte_count": len(proof_bytes),
        "identity": _commit("proof", [b"sp1-cpu", b"v6.1.0", proof_bytes]), "backend_name": "sp1-cpu",
        "backend_version": "v6.1.0", "guest_sha256": "sha256:" + GUEST_SHA256}
    return {"native": native, "proof": proof,
        "verifier": {"command": "verify", "outcome": "verified", "coverage": "program=true input=true output=true exit_code=true",
            "proof_identity": proof["identity"], "backend": module.BACKEND, "statement_program": native["program_identity"]},
        "timings": {"native_seconds": 0.1, "prove_and_verify_seconds": 0.2, "reverify_seconds": 0.3,
            "memory": deepcopy(module.UNMEASURED_MEMORY)}}


def fake_bundle(raw=None):
    """Historical structural fixture; cannot pass a real cryptographic verifier."""
    workflow = ProvedHeatWorkflow()
    workflow._invoke = lambda value, *_args, **_kwargs: fake_data(value)
    return workflow._execute(raw or canonical(source()), (None, fake_runtime(), {}))


def reseal(bundle):
    """Recompute outer hashes only, to test semantic checks beyond digest checks."""
    step = bundle["steps"][0]
    result = step["result"]
    result["result_id"] = digest({k:v for k,v in result.items() if k != "result_id"})
    step["result_id"], step["result_sha256"] = result["result_id"], digest(result)
    step["request_sha256"] = digest(step["request"])
    step["numerical_result"] = {"operation_id": module.ProvedHeatWorkflow().operation, "data": deepcopy(result["data"]["native"])}
    step["numerical_result_id"] = digest(step["numerical_result"])
    bundle["bundle_digest"] = bundle_digest(bundle)
    bundle["verification"] = _verification(bundle, bundle["verification"]["verification_operation_id"])
    return bundle


@pytest.mark.parametrize("raw", [b"", b"null", b"[]", b"false", b"{}", b'{"a":1,"a":2}', b" " * (module.SOURCE_LIMIT + 1)],
    ids=["empty", "null", "array", "bool", "object", "duplicate", "oversize"])
def test_malformed_source_refuses_before_binding(raw, monkeypatch):
    workflow = ProvedHeatWorkflow()
    monkeypatch.setattr(workflow, "_adapters", lambda *_: pytest.fail("Malformed input reached runtime binding"))
    with pytest.raises(ValueError):
        workflow.create_session(raw, {})


@pytest.mark.parametrize("field,value", [("initial_values", [0, 1]), ("initial_values", [0] * 33),
    ("initial_values", [0, True, 0]), ("initial_values", [0, 1.0, 0]), ("initial_values", [0, 2**40 + 1, 0]),
    ("steps", True), ("steps", -1), ("steps", 65), ("schema", "ciw.numerical-heat-source.v1"),
    ("configuration", {**POLICY, "proof_policy": "optional"})])
def test_source_contract_bounds(field, value):
    data = source()
    data[field] = value
    with pytest.raises(ValueError):
        ProvedHeatWorkflow()._source(canonical(data))


@pytest.mark.parametrize("steps,values", [(0, [-2**40, 0, 2**40]), (64, [0] * 32)])
def test_source_boundary_values_are_supported(steps, values):
    data = source()
    data.update(steps=steps, initial_values=values)
    assert ProvedHeatWorkflow()._source(canonical(data)) == data


def test_guest_pin_refuses_before_loading_scr(tmp_path, monkeypatch):
    paths = {"scr": tmp_path}
    for role in module.BINARY_LIMITS:
        path = tmp_path / role
        path.write_bytes(b"NOT THE REGISTERED GUEST")
        paths[role] = path
    monkeypatch.setattr(module, "PinnedSubprocessAdapter", lambda *_a, **_k: pytest.fail("Unregistered guest reached adapter"))
    with pytest.raises(ValueError, match="registered"):
        ProvedHeatWorkflow()._adapters(paths)


def test_binding_roles_are_explicit():
    with pytest.raises(ValueError, match="Bind SCR"):
        ProvedHeatWorkflow()._adapters({"scr": "untrusted"})


def test_creation_does_not_prove_twice(monkeypatch):
    workflow = ProvedHeatWorkflow()
    calls = []
    monkeypatch.setattr(workflow, "_adapters", lambda *_a: (None, fake_runtime(), {}))
    def invoke(value, bound):
        calls.append(value)
        return fake_data(value)
    monkeypatch.setattr(workflow, "_invoke", invoke)
    bundle = workflow.create_session(canonical(source()), {})
    assert len(calls) == 1
    assert "reproduction" not in bundle["verification"]
    assert bundle["verification"]["method"] == "fresh_registered_guest_verification"
    assert bundle["verification"]["independent"] is False


def test_offline_validation_checks_historical_report_without_running_code(monkeypatch):
    raw = b"\n " + canonical(source()) + b"\n"
    bundle = fake_bundle(raw)
    workflow = ProvedHeatWorkflow()
    monkeypatch.setattr(workflow, "_adapters", lambda *_a: pytest.fail("Offline validation bound an executable"))
    monkeypatch.setattr(workflow, "_invoke", lambda *_a: pytest.fail("Offline validation executed code"))
    assert workflow._validate(bundle) == raw
    assert bundle["verification"]["trust_scope"] == module.TRUST_SCOPE
    assert bundle["verification"]["authority"]["state_admission"] == "not_performed"
    assert b"NOT A CRYPTOGRAPHIC PROOF" in base64.b64decode(bundle["steps"][0]["result"]["data"]["proof"]["bytes_b64"])


@pytest.mark.parametrize("section,field,value", [
    ("proof", "identity", "0" * 64), ("proof", "sha256", "sha256:" + "0" * 64),
    ("proof", "byte_count", True), ("proof", "bytes_b64", "!!!!"),
    ("proof", "guest_sha256", "sha256:" + "0" * 64), ("proof", "backend_version", "v7.0.0"),
    ("proof", "backend_name", "mock"), ("proof", "vkey_hash", "0x" + "c" * 64),
    ("verifier", "outcome", "failed"), ("verifier", "coverage", "program=true input=false output=true exit_code=true"),
    ("verifier", "proof_identity", "0" * 64), ("verifier", "statement_program", "0" * 64),
    ("verifier", "backend", "sp1-mock v6.1.0"), ("verifier", "command", "verify-vk"),
    ("timings", "native_seconds", True), ("timings", "prove_and_verify_seconds", -1),
    ("timings", "memory", "50MB"), ("native", "output_identity", "0" * 64),
    ("native", "specification_identity", "0" * 64), ("native", "values", [0, 0, 0, 0, 0]),
])
def test_resealed_result_tampering_is_refused(section, field, value):
    bundle = fake_bundle()
    bundle["steps"][0]["result"]["data"][section][field] = value
    reseal(bundle)
    with pytest.raises(ValueError):
        ProvedHeatWorkflow()._validate(bundle)


@pytest.mark.parametrize("field,value", [("revision", "0" * 40), ("source_tree", "0" * 40),
    ("module", "execution.engine"), ("source_root", "src"), ("adapter_version", "arbitrary"),
    ("python_version", "3.9.0"), ("python_sha256", "no")])
def test_resealed_runtime_pin_tampering_is_refused(field, value):
    bundle = fake_bundle()
    bundle["runtimes"]["scr"][field] = value
    reseal(bundle)
    with pytest.raises(ValueError):
        ProvedHeatWorkflow()._validate(bundle)


@pytest.mark.parametrize("field,value", [("independent", True), ("trust_scope", "fresh_authenticated_proof"),
    ("method", "independent_mathematical_check"), ("verification_operation_id", "execution-" + "0" * 32)])
def test_resealed_verification_cannot_expand_authority(field, value):
    bundle = fake_bundle()
    report = bundle["verification"]
    report[field] = value
    report.pop("verification_id")
    module._identify(report)
    with pytest.raises(ValueError):
        ProvedHeatWorkflow()._validate(bundle)


def test_numerical_identity_excludes_proof_and_timing():
    first, second = fake_bundle(), fake_bundle()
    second["steps"][0]["result"]["data"]["timings"]["reverify_seconds"] = 2.0
    reseal(second)
    assert first["steps"][0]["numerical_result_id"] == second["steps"][0]["numerical_result_id"]
    assert first["steps"][0]["result_id"] != second["steps"][0]["result_id"]


def test_fresh_verification_dispatches_only_verify(monkeypatch):
    bundle = fake_bundle()
    workflow = ProvedHeatWorkflow()
    calls = []
    monkeypatch.setattr(workflow, "_adapters", lambda *_a: (None, fake_runtime(), {}))
    def invoke(value, bound, retained):
        calls.append(retained)
        return {"verifier": retained["verifier"], "seconds": 0.4, "memory": deepcopy(module.UNMEASURED_MEMORY)}
    monkeypatch.setattr(workflow, "_invoke", invoke)
    report = workflow.verify_session(bundle, {})
    assert calls == [bundle["steps"][0]["result"]["data"]]
    assert report["verification_operation_id"] != bundle["verification"]["verification_operation_id"]
    assert report["proof_identity"] == bundle["verification"]["proof_identity"]


def test_fresh_verification_accepts_another_compatible_host_and_records_it(monkeypatch):
    # A transport double tests binding policy only, not cryptographic acceptance.
    bundle = fake_bundle()
    original = deepcopy(bundle)
    runtime = fake_runtime()
    runtime.update(repository_root="/other-host/scr", python_executable="/other-host/python",
                   python_sha256="d" * 64, python_version="3.13.1")
    runtime["engine"]["sha256"] = "sha256:" + "e" * 64
    runtime["prover"]["sha256"] = "sha256:" + "f" * 64
    workflow = ProvedHeatWorkflow()
    workflow._check_runtime(runtime)
    calls = []
    def bind(repositories, expected=None):
        calls.append((repositories, expected))
        assert expected is None, "Fresh verification must not require producer host equivalence"
        return None, runtime, {}
    def verify(value, bound, retained):
        assert bound[1] == runtime
        return {"verifier": retained["verifier"], "seconds": 0.4, "memory": deepcopy(module.UNMEASURED_MEMORY)}
    monkeypatch.setattr(workflow, "_adapters", bind)
    monkeypatch.setattr(workflow, "_invoke", verify)
    report = workflow.verify_session(bundle, {"trusted_host": "other"})
    assert calls == [({"trusted_host": "other"}, None)]
    assert report["runtime_digest"] == digest(original["runtimes"])
    assert report["verifier_runtimes"] == {"scr": runtime}
    assert report["verifier_runtime_digest"] == digest({"scr": runtime})
    assert report["verifier_runtime_digest"] != report["runtime_digest"]
    assert report["subject_ref"] == original["bundle_digest"]
    assert bundle == original
    runtime["engine"]["sha256"] = "sha256:" + "0" * 64
    assert report["verifier_runtime_digest"] == digest(report["verifier_runtimes"])


def test_fresh_verification_still_refuses_an_unregistered_guest(tmp_path, monkeypatch):
    paths = {"scr": tmp_path}
    for role in module.BINARY_LIMITS:
        path = tmp_path / role
        path.write_bytes(b"NOT THE REGISTERED GUEST")
        paths[role] = path
    workflow = ProvedHeatWorkflow()
    monkeypatch.setattr(workflow, "_invoke", lambda *_a, **_k: pytest.fail("Unregistered guest reached verification"))
    with pytest.raises(ValueError, match="registered"):
        workflow.verify_session(fake_bundle(), paths)


def fake_replay(monkeypatch, original=None):
    original = original or fake_bundle()
    workflow = ProvedHeatWorkflow()
    monkeypatch.setattr(workflow, "_adapters", lambda *_a: (None, deepcopy(original["runtimes"]["scr"]), {}))
    monkeypatch.setattr(workflow, "_invoke", lambda value, *_a: fake_data(value))
    return workflow, original, workflow.replay_session(original, {})


def test_replay_binds_fresh_occurrence_without_duplicating_proof(monkeypatch):
    workflow, original, replay = fake_replay(monkeypatch)
    fresh, receipt = replay["session"], replay["replay_receipt"]
    workflow.validate_replay(original, fresh, receipt)
    assert original["steps"][0]["execution_id"] != fresh["steps"][0]["execution_id"]
    assert original["steps"][0]["numerical_result_id"] == fresh["steps"][0]["numerical_result_id"]
    assert "bytes_b64" not in canonical(receipt).decode()
    assert "reproduction" not in receipt["verification"]


@pytest.mark.parametrize("field,value", [("independent", True), ("proof_identity", "0" * 64),
    ("fresh_result_id", "sha256:" + "0" * 64), ("fresh_execution_id", "execution-" + "0" * 32),
    ("fresh_verification_id", "sha256:" + "0" * 64), ("trust_scope", "fresh_verified")])
def test_resealed_replay_receipt_tampering_is_refused(monkeypatch, field, value):
    workflow, original, replay = fake_replay(monkeypatch)
    receipt = replay["replay_receipt"]
    receipt["verification"][field] = value
    receipt["verification"].pop("verification_id")
    module._identify(receipt["verification"])
    receipt["replay_id"] = digest({k:v for k,v in receipt.items() if k != "replay_id"})
    with pytest.raises(ValueError):
        workflow.validate_replay(original, replay["session"], receipt)


def test_replay_old_runtime_digest_is_checked_against_actual_history(monkeypatch):
    workflow, original, replay = fake_replay(monkeypatch)
    receipt = replay["replay_receipt"]
    receipt["verification"]["runtime_digest"] = "sha256:" + "0" * 64
    receipt["verification"].pop("verification_id")
    module._identify(receipt["verification"])
    receipt["replay_id"] = digest({k:v for k,v in receipt.items() if k != "replay_id"})
    workflow._validate(replay["session"])
    with pytest.raises(ValueError, match="historical context"):
        workflow.validate_replay(original, replay["session"], receipt)


def test_same_occurrence_cannot_be_replayed():
    bundle = fake_bundle()
    with pytest.raises(ValueError, match="distinct execution"):
        ProvedHeatWorkflow._receipt_verification(bundle, bundle)


@pytest.mark.parametrize("tamper", [False, True])
def test_bridge_executes_private_binary_snapshots_and_checks_their_bytes(tmp_path, tamper):
    # The transport double records the paths and writes synthetic proof bytes;
    # it does not execute or purport to test an SP1 prover.
    runtime, data = fake_runtime(), fake_data()
    provider = {k:v for k,v in runtime.items() if k not in module.BINARY_LIMITS}
    binaries = {role: ("SYNTHETIC " + role).encode() for role in module.BINARY_LIMITS}
    calls = []
    class TransportDouble:
        source_root = tmp_path
        def runtime_identity(self):
            return deepcopy(provider)
        def _run(self, _bootstrap, arguments, payload):
            assert module._json(payload) == {"source": source()}
            assert arguments[-1] == "create"
            for role, path in zip(("engine", "prover", "guest"), arguments[1:4]):
                assert Path(path).read_bytes() == binaries[role]
                assert not Path(path).is_relative_to(tmp_path)
            proof_path = Path(arguments[4])
            proof_path.write_bytes(base64.b64decode(data["proof"]["bytes_b64"]))
            calls.append(arguments)
            if tamper:
                Path(arguments[2]).write_bytes(b"changed prover")
            return 0, canonical(data)
    workflow = ProvedHeatWorkflow()
    if tamper:
        with pytest.raises(ValueError, match="snapshot changed"):
            workflow._invoke(source(), (TransportDouble(), runtime, binaries))
    else:
        assert workflow._invoke(source(), (TransportDouble(), runtime, binaries)) == data
    assert len(calls) == 1
    assert not Path(calls[0][4]).exists(), "Private runtime directory is removed after return/refusal"


@pytest.mark.parametrize("field,value", [("outcome", "failed"),
    ("coverage", "program=true input=false output=true exit_code=true"),
    ("proof_identity", "0" * 64), ("statement_program", "0" * 64),
    ("backend", "sp1-test v6.1.0")])
def test_fresh_verification_refuses_incomplete_or_detached_verifier_reports(monkeypatch, field, value):
    bundle = fake_bundle()
    workflow = ProvedHeatWorkflow()
    monkeypatch.setattr(workflow, "_adapters", lambda *_a: (None, fake_runtime(), {}))
    def refuse(*_args, **_kwargs):
        verifier = deepcopy(bundle["steps"][0]["result"]["data"]["verifier"])
        verifier[field] = value
        return {"verifier": verifier, "seconds": 0.4, "memory": deepcopy(module.UNMEASURED_MEMORY)}
    monkeypatch.setattr(workflow, "_invoke", refuse)
    with pytest.raises(ValueError, match="Verifier"):
        workflow.verify_session(bundle, {})


@pytest.mark.parametrize("field,value", [("status", "estimated"), ("unit", "KiB"), ("unit", "MB"),
    ("scope", "prover_only"), ("scope", "concurrent_process_tree_peak"),
    ("bytes", True), ("bytes", 0), ("bytes", -1024), ("bytes", 1.5),
    ("bytes", 2**53), ("bytes", 1234), ("bytes", None)])
def test_resealed_memory_cannot_change_scope_units_or_invent_measurements(field, value):
    bundle = fake_bundle()
    memory = {"status": "measured", "unit": "byte", "bytes": 8192, "scope": module.MEMORY_SCOPE}
    memory[field] = value
    bundle["steps"][0]["result"]["data"]["timings"]["memory"] = memory
    reseal(bundle)
    with pytest.raises(ValueError):
        ProvedHeatWorkflow()._validate(bundle)


def test_unmeasured_memory_must_not_have_a_byte_count():
    with pytest.raises(ValueError, match="fabricated"):
        module._memory({**module.UNMEASURED_MEMORY, "bytes": 8192})


@pytest.mark.parametrize("count", [1024, 8192, (2**53 - 1) // 1024 * 1024])
def test_valid_measured_memory_does_not_change_numerical_identity(count):
    bundle = fake_bundle()
    numerical = bundle["steps"][0]["numerical_result_id"]
    bundle["steps"][0]["result"]["data"]["timings"]["memory"] = {
        "status": "measured", "unit": "byte", "bytes": count, "scope": module.MEMORY_SCOPE}
    reseal(bundle)
    ProvedHeatWorkflow()._validate(bundle)
    assert bundle["steps"][0]["numerical_result_id"] == numerical


def test_linux_probe_reads_waited_children_and_converts_kib_to_bytes(monkeypatch):
    # Exercise the exact standalone collector with a syscall double. This tests
    # unit/scope conversion, not a claimed measurement of the native SP1 prover.
    namespace, observed = {}, []
    exec(module._MEMORY_PROBE, namespace)
    def usage(who):
        observed.append(who)
        return SimpleNamespace(ru_maxrss=7168)
    resource_double = SimpleNamespace(RUSAGE_CHILDREN=-1, getrusage=usage)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", "linux")
        patch.setitem(sys.modules, "resource", resource_double)
        result = namespace["child_memory"]()
    assert observed == [-1]
    assert result == {"status": "measured", "unit": "byte", "bytes": 7168 * 1024, "scope": module.MEMORY_SCOPE}
    module._memory(result)


@pytest.mark.parametrize("platform,peak,error", [("win32", 100, None), ("darwin", 100, None),
    ("linux", 0, None), ("linux", -1, None), ("linux", True, None),
    ("linux", 1.5, None), ("linux", 2**53, None), ("linux", None, OSError),
    ("linux", None, AttributeError)])
def test_memory_probe_marks_unavailable_or_inapplicable_measurement(monkeypatch, platform, peak, error):
    namespace = {}
    exec(module._MEMORY_PROBE, namespace)
    def usage(_who):
        if platform != "linux":
            pytest.fail("Linux measurement must not be applied to another platform's units")
        if error:
            raise error("unavailable")
        return SimpleNamespace(ru_maxrss=peak)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", platform)
        patch.setitem(sys.modules, "resource", SimpleNamespace(RUSAGE_CHILDREN=-1, getrusage=usage))
        result = namespace["child_memory"]()
    assert result == module.UNMEASURED_MEMORY
    module._memory(result)


def test_fresh_verification_retains_its_own_memory_measurement(monkeypatch):
    bundle = fake_bundle()
    workflow = ProvedHeatWorkflow()
    measured = {"status": "measured", "unit": "byte", "bytes": 2048, "scope": module.MEMORY_SCOPE}
    monkeypatch.setattr(workflow, "_adapters", lambda *_a: (None, fake_runtime(), {}))
    def verify(_value, _bound, retained):
        return {"verifier": retained["verifier"], "seconds": 0.4, "memory": measured}
    monkeypatch.setattr(workflow, "_invoke", verify)
    report = workflow.verify_session(bundle, {})
    assert report["memory"] == measured
    assert bundle["steps"][0]["result"]["data"]["timings"]["memory"] == module.UNMEASURED_MEMORY


def _leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, path + (index,))
    else:
        yield path


def test_every_single_leaf_change_to_a_retained_replay_is_refused():
    """Runner-owned envelopes and proof-owned verification leave no unbound field."""
    workflow = ProvedHeatWorkflow()
    original, fresh = fake_bundle(), fake_bundle()
    receipt = {"schema": "ciw.proved-heat-replay.v1", "source_bundle_digest": original["bundle_digest"],
               "replayed_bundle_digest": fresh["bundle_digest"], "numerical_match": True,
               "verification": workflow._receipt_verification(original, fresh), "admission": "not_performed"}
    receipt["replay_id"] = digest(receipt)
    fresh["replay_receipts"] = [receipt]
    workflow.validate_replay(original, fresh, receipt)
    checked = 0
    for bundle in (original, fresh):
        for path in _leaves(bundle):
            if path == ("bundle_digest",):
                continue
            for reseal in (False, True):
                changed = deepcopy(bundle)
                parent = changed
                for key in path[:-1]:
                    parent = parent[key]
                value = parent[path[-1]]
                parent[path[-1]] = (value + "x" if isinstance(value, str) else not value if isinstance(value, bool)
                                    else value + 1 if isinstance(value, (int, float)) else "x")
                if reseal:
                    changed["bundle_digest"] = bundle_digest(changed)
                with pytest.raises(ValueError):
                    workflow._validate(changed)
                checked += 1
    assert checked > 500
