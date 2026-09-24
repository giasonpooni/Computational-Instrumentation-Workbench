"""Registered SCR heat execution with retained SP1 proofs.

Creating a result runs the real prover and a separate full-ELF verifier. Loading
JSON only checks retained bytes and bindings: a historical report is not a fresh
cryptographic verification. Executable bindings always come from the operator.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
import math
import os
from pathlib import Path
import re
import tempfile
import uuid

from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import PinnedSubprocessAdapter, _json
from .declared_workload import (
    AUTHORITY, HEAT_POLICY, _check_data, _commit, _read_bound, _text,
)
from .exchange import _identity
from .core.canonical import canonical, digest, byte_digest, exact_keys
from .pipelines import pin_map
from .pipelines.runner import PipelineRunner, check_step, seal_step

MAX_BYTES = 24 * 1024 * 1024
PROOF_LIMIT = 8 * 1024 * 1024
SOURCE_LIMIT = 32 * 1024
PROVE_TIMEOUT = 1800
BINARY_LIMITS = {"engine": 32 * 1024 * 1024, "prover": 256 * 1024 * 1024, "guest": 16 * 1024 * 1024}
# The pipeline descriptor is the pin definition this module executes.
PIN = pin_map("proved-heat")["scr"]
GUEST_SHA256 = "a14e3750da7e221d31842bd6cf983fcc8c0f530b2811537e2a9a9fe803dacf82"
BACKEND = "sp1-cpu v6.1.0"
POLICY = {**HEAT_POLICY, "proof_policy": "required_before_result"}
VERIFY_SCHEMA = "ciw.proved-heat-verification.v1"
REPLAY_VERIFY_SCHEMA = "ciw.proved-heat-replay-verification.v1"
TRUST_SCOPE = "retained_runtime_report_requires_fresh_verification"
MEMORY_SCOPE = "max_waited_child_peak_rss"
UNMEASURED_MEMORY = {"status": "not_measured", "unit": "byte", "bytes": None, "scope": MEMORY_SCOPE}
_DIGEST = r"sha256:[a-f0-9]{64}"


def _same(actual, expected, message):
    if canonical(actual) != canonical(expected):
        raise ValueError(message)


def _source(raw):
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= SOURCE_LIMIT:
        raise ValueError("Proved heat source requires 1..32768 exact bytes")
    value = _json(raw)
    exact_keys(value, {"schema", "experiment_id", "configuration", "initial_values", "steps"})
    if value["schema"] != "ciw.proved-heat-source.v1":
        raise ValueError("Unsupported proved heat source")
    _text(value["experiment_id"])
    _same(value["configuration"], POLICY, "Require the bounded heat and mandatory proof policy")
    cells = value["initial_values"]
    if (not isinstance(cells, list) or not 3 <= len(cells) <= 32 or
            any(type(v) is not int or abs(v) > 2**40 for v in cells) or
            type(value["steps"]) is not int or not 0 <= value["steps"] <= 64):
        raise ValueError("Proved heat requires 3..32 integer cells, |u|<=2^40 and 0..64 steps")
    return value


_MEMORY_PROBE = r'''
def child_memory():
    # Linux getrusage(2): ru_maxrss for RUSAGE_CHILDREN is the largest
    # waited-for child's peak RSS in KiB, not simultaneous process-tree RSS.
    import sys
    result = {'status':'not_measured', 'unit':'byte', 'bytes':None,
              'scope':'max_waited_child_peak_rss'}
    if sys.platform != 'linux':
        return result
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    except (ImportError, OSError, ValueError, AttributeError):
        return result
    if type(peak) is int and 0 < peak <= (2**53 - 1) // 1024:
        result.update(status='measured', bytes=peak * 1024)
    return result
'''


_BOOTSTRAP = _MEMORY_PROBE + r'''
import base64, dataclasses, hashlib, json, struct, sys, time
from pathlib import Path
root, engine, host, guest, proof_path, action = sys.argv[1:7]
sys.path.insert(0, root)
from execution.specification import ExecutionSpecification, HEAT_DIFFUSION_DESCRIPTOR, encode_heat_input
from execution.engine import run_specification, ExecutionResult
from execution.proving import prove_and_verify_result, verify_existing_proof
payload = json.loads(sys.stdin.buffer.read())
source = payload['source']
spec = ExecutionSpecification(HEAT_DIFFUSION_DESCRIPTOR, b'', encode_heat_input(source['steps'], source['initial_values']))
proof_path, host, guest = Path(proof_path), Path(host), Path(guest)
if action == 'create':
    started = time.perf_counter()
    native = run_specification(spec, cli_path=Path(engine))
    native_seconds = time.perf_counter() - started
    if native.status != 'completed' or native.exit_code != 0 or native.output is None:
        raise ValueError('Heat execution did not complete')
    started = time.perf_counter()
    proved = prove_and_verify_result(native, spec, proof_path, host, guest, prove_timeout=1800)
    prove_seconds = time.perf_counter() - started
    if not 1 <= proof_path.stat().st_size <= 8 * 1024 * 1024:
        raise ValueError('Proof exceeds the retained artifact budget')
    proof = proof_path.read_bytes()
else:
    retained = payload['native'].copy()
    retained.pop('values')
    retained['specification'] = spec
    retained['output'] = bytes.fromhex(retained['output'])
    native = ExecutionResult(**retained)
started = time.perf_counter()
verifier = verify_existing_proof(native, spec, proof_path, host, guest)
verify_seconds = time.perf_counter() - started
if verifier.get('outcome') != 'verified':
    raise ValueError('Fresh registered guest verification did not succeed')
memory = child_memory()
if action == 'create':
    value = dataclasses.asdict(native)
    value['specification'] = {k:v.hex() for k,v in value['specification'].items()}
    value['output'] = native.output.hex()
    value['values'] = list(struct.unpack('<' + 'q'*len(source['initial_values']), native.output))
    result = {'native':value, 'proof':{'bytes_b64':base64.b64encode(proof).decode(),
        'sha256':'sha256:'+hashlib.sha256(proof).hexdigest(), 'byte_count':len(proof),
        'identity':proved.proof_identity, 'backend_name':proved.backend_name,
        'backend_version':proved.backend_version,
        'guest_sha256':'sha256:'+hashlib.sha256(guest.read_bytes()).hexdigest()},
        'verifier':verifier, 'timings':{'native_seconds':native_seconds,
            'prove_and_verify_seconds':prove_seconds, 'reverify_seconds':verify_seconds,
            'memory':memory}}
else:
    result = {'verifier':verifier, 'seconds':verify_seconds, 'memory':memory}
print(json.dumps(result, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False))
'''


def _proof_bytes(proof):
    exact_keys(proof, {"bytes_b64", "sha256", "byte_count", "identity", "backend_name", "backend_version", "guest_sha256"})
    encoded = proof["bytes_b64"]
    if not isinstance(encoded, str) or len(encoded) > 4 * ((PROOF_LIMIT + 2) // 3):
        raise ValueError("Retained proof exceeds its byte budget")
    raw = base64.b64decode(encoded, validate=True)
    if (not 1 <= len(raw) <= PROOF_LIMIT or type(proof["byte_count"]) is not int or
            proof["byte_count"] != len(raw) or proof["sha256"] != byte_digest(raw) or
            encoded != base64.b64encode(raw).decode() or
            proof["backend_name"] != "sp1-cpu" or proof["backend_version"] != "v6.1.0" or
            proof["guest_sha256"] != "sha256:" + GUEST_SHA256):
        raise ValueError("Retained proof identity, backend or guest binding mismatch")
    expected = _commit("proof", [b"sp1-cpu", b"v6.1.0", raw])
    if proof["identity"] != expected:
        raise ValueError("Proof commitment differs from retained exact bytes")
    return raw


def _verifier(value, native, proof):
    _same(value, {"command": "verify", "outcome": "verified",
        "coverage": "program=true input=true output=true exit_code=true",
        "proof_identity": proof["identity"], "backend": BACKEND,
        "statement_program": native["program_identity"]},
        "Verifier did not report the exact registered program and complete statement")


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 3600:
        raise ValueError("Invalid measured monotonic stage duration")


def _memory(value):
    exact_keys(value, {"status", "unit", "bytes", "scope"})
    if value["unit"] != "byte" or value["scope"] != MEMORY_SCOPE:
        raise ValueError("Memory must retain the largest waited-child peak RSS scope and byte units")
    if value["status"] == "not_measured":
        if value["bytes"] is not None:
            raise ValueError("Unmeasured memory cannot carry a fabricated byte count")
    elif value["status"] == "measured":
        count = value["bytes"]
        if type(count) is not int or not 1 <= count <= 2**53 - 1 or count % 1024:
            raise ValueError("Measured Linux peak RSS must be a bounded positive integer number of KiB converted to bytes")
    else:
        raise ValueError("Unknown memory measurement status")


def _data(source, value):
    exact_keys(value, {"native", "proof", "verifier", "timings"})
    _check_data("numerical-heat", source, value["native"])
    _proof_bytes(value["proof"])
    _verifier(value["verifier"], value["native"], value["proof"])
    exact_keys(value["timings"], {"native_seconds", "prove_and_verify_seconds", "reverify_seconds", "memory"})
    for key in ("native_seconds", "prove_and_verify_seconds", "reverify_seconds"):
        _seconds(value["timings"][key])
    _memory(value["timings"]["memory"])


def _identify(report):
    report["verification_id"] = byte_digest(report["schema"].encode() + b"\0" + canonical(report))
    return report


def _verification(bundle, occurrence):
    step = bundle["steps"][0]
    data = step["result"]["data"]
    return _identify({"schema": VERIFY_SCHEMA, "subject_ref": bundle["bundle_digest"],
        "outcome": "passed", "independent": False, "method": "fresh_registered_guest_verification",
        "runtime_digest": digest(bundle["runtimes"]), "result_id": step["result_id"],
        "proof_identity": data["proof"]["identity"], "verifier_digest": digest(data["verifier"]),
        "verification_operation_id": occurrence, "trust_scope": TRUST_SCOPE,
        "authority": deepcopy(AUTHORITY)})


class ProvedHeatWorkflow(PipelineRunner):
    """SCR heat execution proved by SP1; verification is a registered-guest proof check, not reproduction."""

    MAX_BYTES = MAX_BYTES
    LABEL = "Proved heat"
    ROLES = frozenset({"scr", "engine", "prover", "guest"})
    EXTRA_ROLES = frozenset({"engine", "prover", "guest"})

    def __init__(self, kind="proved-heat"):
        if kind != "proved-heat":
            raise ValueError("Unsupported proved operation")
        super().__init__(kind, {**PIN, "role": "scr"})

    def parse_source(self, raw):
        return _source(raw)

    def _adapters(self, repositories, expected=None):
        if set(repositories) != self.ROLES:
            raise ValueError("Bind SCR, execution engine, SP1 prover and registered heat guest")
        binaries = {role: _read_bound(repositories[role], limit) for role, limit in BINARY_LIMITS.items()}
        if any(not value for value in binaries.values()) or sha256(binaries["guest"]).hexdigest() != GUEST_SHA256:
            raise ValueError("Require nonempty executables and the registered SP1 heat guest")
        retained = expected["scr"] if expected else {}
        adapter = PinnedSubprocessAdapter(repositories["scr"], PIN["revision"], PIN["module"], source_root=".",
            timeout_seconds=3100, max_output_bytes=MAX_BYTES,
            expected_python_sha256=retained.get("python_sha256"), expected_python_version=retained.get("python_version"),
            expected_dependencies=retained.get("dependencies"))
        runtime = adapter.runtime_identity()
        for role, raw in binaries.items():
            runtime[role] = {"sha256": byte_digest(raw), "byte_count": len(raw),
                "source_binding": "registered_guest_sha256" if role == "guest" else "operator_asserted_not_attested"}
        self._check_runtime(runtime)
        if retained:
            _same(self._runtime_projection(runtime), self._runtime_projection(retained), "Proved heat runtime differs from retained bindings")
        return adapter, runtime, binaries

    @staticmethod
    def _check_runtime(runtime):
        exact_keys(runtime, {"schema", "adapter_version", "repository_root", "revision", "source_tree", "module", "source_root",
            "python_executable", "python_sha256", "python_version", "dependencies", "engine", "prover", "guest"})
        if (runtime["schema"] != "ciw.subprocess-runtime.v1" or runtime["adapter_version"] != "ciw-pinned-subprocess-v1" or
                any(runtime[k] != v for k, v in PIN.items()) or
                not isinstance(runtime["python_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", runtime["python_sha256"])):
            raise ValueError("Unapproved SCR runtime source or interpreter pin")
        for key in ("repository_root", "python_executable", "python_version"):
            _text(runtime[key])
        if not re.fullmatch(r"\d+\.\d+\.\d+", runtime["python_version"]) or tuple(map(int, runtime["python_version"].split(".")[:2])) < (3, 10):
            raise ValueError("SCR requires Python 3.10 or newer")
        exact_keys(runtime["dependencies"], {"numpy", "scipy"})
        for value in runtime["dependencies"].values():
            if value is not None:
                _text(value)
        for role, limit in BINARY_LIMITS.items():
            binary = runtime[role]
            exact_keys(binary, {"sha256", "byte_count", "source_binding"})
            if (not isinstance(binary["sha256"], str) or not re.fullmatch(_DIGEST, binary["sha256"]) or
                    type(binary["byte_count"]) is not int or not 1 <= binary["byte_count"] <= limit or
                    binary["source_binding"] != ("registered_guest_sha256" if role == "guest" else "operator_asserted_not_attested")):
                raise ValueError("Invalid host-bound proof runtime identity")
        if runtime["guest"]["sha256"] != "sha256:" + GUEST_SHA256:
            raise ValueError("Unregistered heat guest")

    def _invoke(self, source, bound, retained=None):
        adapter, runtime, binaries = bound
        provider = {key: value for key, value in runtime.items() if key not in BINARY_LIMITS}
        _same(self._runtime_projection(adapter.runtime_identity()), self._runtime_projection(provider), "SCR changed before execution")
        with tempfile.TemporaryDirectory(prefix="ciw-proved-heat-") as directory:
            root = Path(directory)
            paths = {}
            for role, raw in binaries.items():
                path = root / (role + (".exe" if os.name == "nt" and role != "guest" else ".elf" if role == "guest" else ""))
                path.write_bytes(raw)
                path.chmod(0o700)
                paths[role] = path
            proof_path = root / "heat.proof"
            request = {"source": source}
            if retained is not None:
                proof_path.write_bytes(_proof_bytes(retained["proof"]))
                request["native"] = retained["native"]
            code, raw = adapter._run(_BOOTSTRAP,
                [str(adapter.source_root), str(paths["engine"]), str(paths["prover"]), str(paths["guest"]),
                 str(proof_path), "create" if retained is None else "verify"], canonical(request))
            for role, path in paths.items():
                if _read_bound(path, BINARY_LIMITS[role]) != binaries[role]:
                    raise ValueError("A private proof runtime snapshot changed during execution")
            if code:
                raise AdapterRefusal("PROVED_HEAT_REFUSED", "Pinned SCR proof generation or registered-guest verification failed")
            answer = _json(raw)
            proof = _read_bound(proof_path, PROOF_LIMIT)
            if proof != _proof_bytes(answer["proof"] if retained is None else retained["proof"]):
                raise ValueError("Proof file differs from retained exact artifact bytes")
        _same(self._runtime_projection(adapter.runtime_identity()), self._runtime_projection(provider), "SCR changed during execution")
        return answer

    def _numerical(self, data):
        return {"operation_id": self.operation, "data": data["native"]}

    def _step(self, source, evidence_id, bound):
        data = self._invoke(source, bound)
        _data(source, data)
        return seal_step(self.role, self.operation, source, [evidence_id], data, self._numerical(data))

    def _validate_step(self, step, source, evidence):
        check_step(step, role=self.role, operation=self.operation, source=source, input_refs=[evidence],
                   check_data=_data, label=self.LABEL, numerical=self._numerical)

    def _check_runtimes(self, runtimes):
        exact_keys(runtimes, {"scr"})
        self._check_runtime(runtimes["scr"])

    def _verify(self, bundle, source, evidence, bound):
        return _verification(bundle, "verification-" + uuid.uuid4().hex)

    def _check_verification(self, bundle, report, source, evidence):
        """The retained statement only; never execute or assert fresh proof trust."""
        occurrence = report["verification_operation_id"]
        if not isinstance(occurrence, str) or not re.fullmatch(r"verification-[a-f0-9]{32}", occurrence):
            raise ValueError("Invalid verification occurrence")
        _same(report, _verification(bundle, occurrence), "Historical verification report differs from its retained statement")
        _identity(report, "verification_id")

    def _check_receipts(self, bundle):
        receipts = bundle.get("replay_receipts", [])
        if not isinstance(receipts, list) or len(receipts) > 1:
            raise ValueError("A proved heat occurrence retains at most one replay receipt")
        for receipt in receipts:
            self._check_receipt(bundle, receipt)

    def verify_session(self, bundle, repositories):
        """Fresh cryptographic verification of retained bytes; no native rerun or proof production."""
        source = self._source(self._validate(bundle))
        # Verification needs the approved source, guest and backend, but a
        # compatible verifier need not reproduce the producer's host binaries.
        # Replay retains the stronger original-runtime equivalence requirement.
        bound = self._adapters(repositories)
        data = bundle["steps"][0]["result"]["data"]
        answer = self._invoke(source, bound, retained=data)
        exact_keys(answer, {"verifier", "seconds", "memory"})
        _verifier(answer["verifier"], data["native"], data["proof"])
        _seconds(answer["seconds"])
        _memory(answer["memory"])
        report = _verification(bundle, "verification-" + uuid.uuid4().hex)
        report.pop("verification_id")
        report["seconds"], report["memory"] = answer["seconds"], answer["memory"]
        report["verifier_runtimes"] = {"scr": deepcopy(bound[1])}
        report["verifier_runtime_digest"] = digest(report["verifier_runtimes"])
        return _identify(report)

    @staticmethod
    def _receipt_verification(original, fresh):
        old, new = original["steps"][0], fresh["steps"][0]
        _same(old["numerical_result"], new["numerical_result"], "Fresh proof execution produced a different numerical result")
        if old["execution_id"] == new["execution_id"] or old["result_id"] == new["result_id"]:
            raise ValueError("Replay requires a distinct execution occurrence")
        return _identify({"schema": REPLAY_VERIFY_SCHEMA, "subject_ref": original["bundle_digest"],
            "outcome": "passed", "independent": False, "method": "same_runtime_fresh_proved_occurrence",
            "runtime_digest": digest(original["runtimes"]), "fresh_runtime_digest": digest(fresh["runtimes"]),
            "original_execution_id": old["execution_id"], "fresh_execution_id": new["execution_id"],
            "fresh_result_id": new["result_id"], "numerical_result_id": new["numerical_result_id"],
            "proof_identity": new["result"]["data"]["proof"]["identity"],
            "fresh_verification_id": fresh["verification"]["verification_id"],
            "trust_scope": TRUST_SCOPE, "authority": deepcopy(AUTHORITY)})

    def _check_receipt(self, fresh, receipt):
        exact_keys(receipt, {"schema", "source_bundle_digest", "replayed_bundle_digest", "numerical_match", "verification", "admission", "replay_id"})
        source_id = receipt["source_bundle_digest"]
        if (receipt["schema"] != "ciw.proved-heat-replay.v1" or not isinstance(source_id, str) or not re.fullmatch(_DIGEST, source_id) or
                source_id == fresh["bundle_digest"] or receipt["replayed_bundle_digest"] != fresh["bundle_digest"] or
                receipt["numerical_match"] is not True or receipt["admission"] != "not_performed" or
                receipt["replay_id"] != digest({k:v for k,v in receipt.items() if k != "replay_id"})):
            raise ValueError("Invalid proved heat replay receipt")
        verification = receipt["verification"]
        exact_keys(verification, {"schema", "subject_ref", "outcome", "independent", "method", "runtime_digest", "fresh_runtime_digest",
            "original_execution_id", "fresh_execution_id", "fresh_result_id", "numerical_result_id", "proof_identity",
            "fresh_verification_id", "trust_scope", "authority", "verification_id"})
        old_id, runtime = verification["original_execution_id"], verification["runtime_digest"]
        if (not isinstance(old_id, str) or not re.fullmatch(r"execution-[a-f0-9]{32}", old_id) or
                old_id == fresh["steps"][0]["execution_id"] or not isinstance(runtime, str) or not re.fullmatch(_DIGEST, runtime)):
            raise ValueError("Invalid historical replay occurrence or runtime digest")
        new = fresh["steps"][0]
        expected = {"schema": REPLAY_VERIFY_SCHEMA, "subject_ref": source_id, "outcome": "passed", "independent": False,
            "method": "same_runtime_fresh_proved_occurrence", "runtime_digest": runtime, "fresh_runtime_digest": digest(fresh["runtimes"]),
            "original_execution_id": old_id, "fresh_execution_id": new["execution_id"], "fresh_result_id": new["result_id"],
            "numerical_result_id": new["numerical_result_id"], "proof_identity": new["result"]["data"]["proof"]["identity"],
            "fresh_verification_id": fresh["verification"]["verification_id"], "trust_scope": TRUST_SCOPE, "authority": deepcopy(AUTHORITY)}
        _same(verification, _identify(expected), "Replay receipt does not bind the exact fresh proof occurrence")
        _identity(verification, "verification_id")

    def validate_replay(self, original, fresh, receipt):
        self._validate(original)
        self._validate(fresh)
        self._check_receipt(fresh, receipt)
        _same(original["source"], fresh["source"], "Replay must retain exact source bytes")
        _same(self._runtime_projection(original["runtimes"]["scr"]), self._runtime_projection(fresh["runtimes"]["scr"]), "Replay runtime changed")
        if (original["session_id"] == fresh["session_id"] or
                original["verification"]["verification_operation_id"] == fresh["verification"]["verification_operation_id"]):
            raise ValueError("Replay requires fresh session and verification occurrences")
        if receipt["source_bundle_digest"] != original["bundle_digest"]:
            raise ValueError("Replay receipt names another original bundle")
        _same(receipt["verification"], self._receipt_verification(original, fresh), "Replay differs from retained historical context")

    def replay_session(self, bundle, repositories):
        raw = self._validate(bundle)
        fresh = self._execute(raw, self._adapters(repositories, bundle["runtimes"]))
        receipt = {"schema": "ciw.proved-heat-replay.v1", "source_bundle_digest": bundle["bundle_digest"],
            "replayed_bundle_digest": fresh["bundle_digest"], "numerical_match": True,
            "verification": self._receipt_verification(bundle, fresh), "admission": "not_performed"}
        receipt["replay_id"] = digest(receipt)
        fresh["replay_receipts"] = [receipt]
        self.validate_replay(bundle, fresh, receipt)
        return {"session": fresh, "replay_receipt": receipt}
