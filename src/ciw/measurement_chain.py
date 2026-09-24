"""Shared RCI/FSRT/JSPT measurement investigation using existing native seams.

The inner workspace is retained unchanged. Only a separate numerical projection
replaces explicitly enumerated occurrence-dependent references for replay. No
provider science is implemented here and no GSIE fusion context is inferred.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from pathlib import Path
import re
import tempfile
import uuid

from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import _json
from .covariance_workflow import MAP_FIELDS, execute_covariance
from .declared_workload import DeclaredWorkflow, RESULT_SCHEMA, AUTHORITY, _text
from .investigation import _runtime, _make_run, _validate_model_independence, create_investigation
from .session import Session, read_json, write_json
from .core.canonical import canonical, digest, byte_digest, bundle_digest, exact_keys
from .pipelines import pin_map

MAX_BYTES = 4 * 1024 * 1024
SOURCE_SCHEMA = "ciw.measurement-chain-source.v1"
DATA_SCHEMA = "ciw.measurement-chain-native.v1"
ROLES = {"rci", "fsrt", "jspt"}
# The pipeline descriptor is the pin definition this module executes.
PINS = pin_map("measurement-chain")
POLICY = {
    "scope": "measurement-chain-testbed", "acquisition": "retained_simultaneous_records",
    "cross_assembly_covariance": "explicit_independence_required",
    "jacobian_source": "caller_declared_not_derivative_verified",
    "output_reference": "caller_declared_not_a_new_measurement",
    "state_admission": "not_performed", "gsie_fusion": "not_performed",
    "physical_traceability": "not_established",
}
FSRT_OPERATION = "fsrt.tank-reconstruct.v2"
JSPT_OPERATION = "jspt.covariance-propagate.v1"


def _source(raw):
    if not isinstance(raw, bytes) or len(raw) > 262144:
        raise ValueError("Measurement-chain source exceeds the 256 KiB budget")
    source = _json(raw)
    canonical(source)
    exact_keys(source, {"schema", "experiment_id", "configuration", "investigation", "covariance_map"})
    if source["schema"] != SOURCE_SCHEMA or source["configuration"] != POLICY:
        raise ValueError("Require the explicit measurement-chain scope")
    _text(source["experiment_id"])
    declared = source["investigation"]
    exact_keys(declared, {"schema", "sensors", "cross_assembly_independent", "model", "source_description", "model_independence"})
    if declared["schema"] != "ciw.tank-investigation-input.v2" or declared["cross_assembly_independent"] is not True:
        raise ValueError("The v2 investigation requires explicit cross-assembly independence")
    _validate_model_independence(declared["model_independence"])
    _text(declared["source_description"])
    if not isinstance(declared["model"], dict) or not isinstance(declared["sensors"], list) or len(declared["sensors"]) != 2:
        raise ValueError("Declare exactly two sensors and their native FSRT model")
    names = []
    for sensor in declared["sensors"]:
        exact_keys(sensor, {"name", "request"})
        _text(sensor["name"])
        names.append(sensor["name"])
        request = sensor["request"]
        exact_keys(request, {"schema", "operation_id", "inputs"})
        if request["schema"] != "ciw.adapter-request.v1" or request["operation_id"] != "rci.calibrate.v2":
            raise ValueError("Require the original native RCI v2 calibration request")
        inputs = request["inputs"]
        exact_keys(inputs, {"assembly_toml", "calibration", "records", "raw_covariance"})
        if not isinstance(inputs["assembly_toml"], str) or len(inputs["assembly_toml"].encode()) > 65536:
            raise ValueError("Require bounded exact assembly text")
        if not isinstance(inputs["calibration"], dict) or inputs["calibration"].get("schema") != "rci-calibration-binding.v2":
            raise ValueError("Require native calibration covariance provenance")
        if not isinstance(inputs["records"], list) or len(inputs["records"]) != 1:
            raise ValueError("The snapshot consumes exactly one record per sensor")
        record = inputs["records"][0]
        exact_keys(record, {"raw_record_b64", "observed_at"})
        _text(record["observed_at"])
        encoded = record["raw_record_b64"]
        if not isinstance(encoded, str) or len(encoded) > 65536:
            raise ValueError("Require bounded retained raw record bytes")
        original = base64.b64decode(encoded, validate=True)
        if base64.b64encode(original).decode() != encoded or not isinstance(_json(original), dict):
            raise ValueError("Raw records require exact canonical base64 JSON objects")
    if len(set(names)) != 2:
        raise ValueError("Require distinct declared sensor names")
    mapping = source["covariance_map"]
    exact_keys(mapping, MAP_FIELDS | {"source_artifact"})
    if mapping["source_artifact"] not in {"posterior", "reconciled"}:
        raise ValueError("Select the retained posterior or retained reconciliation covariance explicitly")
    if mapping["map_kind"] not in {"linear", "local_linearization", "weighted_aggregation", "coordinate_change"}:
        raise ValueError("Unsupported native JSPT mapping kind")
    n = len(mapping["output_quantity_ids"]) if isinstance(mapping["output_quantity_ids"], list) else 0
    if not 1 <= n <= 16 or len(set(mapping["output_quantity_ids"])) != n:
        raise ValueError("Require 1..16 distinct ordered output quantities")
    for field in ("output_units", "output_reference_values", "jacobian"):
        if not isinstance(mapping[field], list) or len(mapping[field]) != n:
            raise ValueError("Covariance mapping dimensions disagree")
    for value in mapping["output_quantity_ids"] + mapping["output_units"] + [mapping["output_frame"]]:
        _text(value)
    if any(not isinstance(row, list) or len(row) != 2 for row in mapping["jacobian"]):
        raise ValueError("JSPT columns must follow the two retained reservoir states")
    if any(type(value) not in (int, float) for value in mapping["output_reference_values"] + sum(mapping["jacobian"], [])):
        raise ValueError("Require finite numeric reference values and Jacobian entries")
    return source


def _runtime_records(workspace):
    records = {"rci": workspace["run"]["metadata"]["rci_source"]["runtime"]}
    records.update({role: next(e["runtime"] for e in workspace["executions"] if e["operation_id"] == operation)
                    for role, operation in (("fsrt", FSRT_OPERATION), ("jspt", JSPT_OPERATION))})
    return records


def _check_runtime(runtime, role, companions=False):
    exact_keys(runtime, {"schema", "adapter_version", "repository_root", "revision", "source_tree", "module", "source_root",
                    "python_executable", "python_sha256", "python_version", "dependencies"} | ({"companions"} if companions else set()))
    if (runtime["schema"] != "ciw.subprocess-runtime.v1" or any(runtime[k] != value for k, value in PINS[role].items())
            or not re.fullmatch("[a-f0-9]{40}", runtime["source_tree"])
            or not re.fullmatch("[a-f0-9]{64}", runtime["python_sha256"]) or not isinstance(runtime["dependencies"], dict)):
        raise ValueError("Unapproved measurement provider pin")
    for field in ("adapter_version", "repository_root", "python_executable", "python_version"):
        _text(runtime[field])


def _check_data(source, data):
    exact_keys(data, {"schema", "scope", "native_workspace"})
    if data["schema"] != DATA_SCHEMA or data["scope"] != POLICY:
        raise ValueError("Invalid measurement-chain scope")
    workspace = data["native_workspace"]
    exact_keys(workspace, {"workspace_version", "saved_at", "run", "selection", "results", "view_settings", "executions"})
    if type(workspace["workspace_version"]) is not int or workspace["workspace_version"] != 2 or workspace["view_settings"] != {}:
        raise ValueError("Require the unchanged native investigation workspace v2")
    if not isinstance(workspace["run"]["run_id"], str) or not re.fullmatch(r"run-[a-f0-9]{32}", workspace["run"]["run_id"]):
        raise ValueError("Require the native run occurrence identity")
    if [r["operation_id"] for r in workspace["results"]] != [FSRT_OPERATION, JSPT_OPERATION] or [e["operation_id"] for e in workspace["executions"]] != [FSRT_OPERATION, JSPT_OPERATION]:
        raise ValueError("Retain exactly the native FSRT and dependent JSPT occurrences")
    # Reuse all offline native evidence, seals, result schemas and dependency
    # checks. Restoring into a temporary directory never binds a provider.
    with tempfile.TemporaryDirectory(prefix="ciw-measurement-inspect-") as directory:
        path = write_json(Path(directory) / "workspace.json", workspace)
        Session.from_workspace(path, Path(directory) / "restored")
    run = workspace["run"]
    retained = run["metadata"]["rci_source"]
    if [s["name"] for s in retained["sensors"]] != [s["name"] for s in source["investigation"]["sensors"]]:
        raise ValueError("Measurement channel order differs from its source")
    for native, declared in zip(retained["sensors"], source["investigation"]["sensors"]):
        if canonical(native["request"]) != canonical(declared["request"]):
            raise ValueError("Retained raw calibration request differs from the source")
    expected = _make_run(source["investigation"], deepcopy(retained["sensors"]), retained["runtime"])
    if canonical({k: v for k, v in expected.items() if k != "run_id"}) != canonical({k: v for k, v in run.items() if k != "run_id"}):
        raise ValueError("Native recording differs from declared measurement/model mapping")
    fsrt, jspt = workspace["results"]
    if fsrt["parameters"] != {"model": source["investigation"]["model"]}:
        raise ValueError("FSRT model differs from the declared source")
    mapping = source["covariance_map"]
    expected_parameters = {**mapping, "source_result_id": fsrt["result_id"],
                           "source_covariance": fsrt["data"]["covariance_artifacts"][mapping["source_artifact"]]}
    if canonical(jspt["parameters"]) != canonical(expected_parameters):
        raise ValueError("JSPT must consume the selected unchanged FSRT covariance")
    for role, runtime in _runtime_records(workspace).items():
        _check_runtime(runtime, role)


def _numerical(data):
    """Normalize only named native bindings; retain all numerical data.

Provider paths are absent from this comparison; native records still retain
them. Source bytes and every native source/occurrence relationship are checked
separately, before projecting. This is not a replacement native artifact.
"""
    workspace = data["native_workspace"]
    run = workspace["run"]
    fsrt, jspt = workspace["results"]
    sensors = run["metadata"]["rci_source"]["sensors"]
    aliases = {run["evidence_id"]: "@measurement-chain/run-evidence"}
    for index, sensor in enumerate(sensors):
        aliases[sensor["covariance"]["covariance_id"]] = "@measurement-chain/calibration/" + str(index)
    for stage, artifact in fsrt["data"]["covariance_artifacts"].items():
        aliases[artifact["covariance_id"]] = "@measurement-chain/fsrt/" + stage
    aliases[jspt["data"]["output_covariance"]["covariance_id"]] = "@measurement-chain/jspt/output"
    # The native coordinate-map identity includes the input covariance ID.
    # Its complete declaration is validated offline and retained unchanged;
    # normalize that dependent binding alongside the selected covariance ID.
    aliases[jspt["data"]["output_covariance"]["basis"]["id"]] = "@measurement-chain/jspt/coordinate-map"
    def normalize(value):
        if isinstance(value, str):
            return aliases.get(value, value)
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return value
    return {"operation_id": "ciw.measurement-chain.v1", "data": normalize({
        "measurements": sensors, "fsrt": fsrt["data"], "jspt": jspt["data"], "scope": POLICY})}


def native_ids(step):
    workspace = step["result"]["data"]["native_workspace"]
    return {workspace["run"]["run_id"]} | {r["result_id"] for r in workspace["results"]} | {e["execution_id"] for e in workspace["executions"]}


def _retained_steps(bundle):
    yield bundle["steps"][0]
    yield bundle["verification"]["reproduction"]
    for receipt in bundle.get("replay_receipts", []):
        yield receipt["verification"]["reproduction"]


def catalog_steps(bundle):
    """Native primary occurrences for the common execution/result catalog."""
    workspace = bundle["steps"][0]["result"]["data"]["native_workspace"]
    result_by_id = {r["result_id"]: r for r in workspace["results"]}
    records = []
    for execution in workspace["executions"]:
        result = result_by_id[execution["result_id"]]
        refs = [workspace["run"]["evidence_id"]]
        if execution["operation_id"] == JSPT_OPERATION:
            refs += [execution["parameters"]["source_result_id"], execution["parameters"]["source_covariance"]["covariance_id"]]
        records.append({"operation_id": execution["operation_id"], "runtime_ref": "fsrt" if execution["operation_id"] == FSRT_OPERATION else "jspt",
                        "execution_id": execution["execution_id"], "result_id": result["result_id"],
                        "input_refs": refs, "result": deepcopy(result), "execution": deepcopy(execution)})
    return records


def identity_claims(bundle):
    """All retained native identities, including verification occurrences."""
    claims = {}
    def claim(identity, role, body):
        if identity in claims and canonical(claims[identity]) != canonical((role, body)):
            raise ValueError("Native measurement identity collision")
        claims[identity] = (role, deepcopy(body))
    for step in _retained_steps(bundle):
        workspace = step["result"]["data"]["native_workspace"]
        run = workspace["run"]
        claim(run["run_id"], "native_run", run)
        claim(run["evidence_id"], "native_evidence", {k: run[k] for k in ("instrument", "metadata", "time_s", "channels")})
        for sensor in run["metadata"]["rci_source"]["sensors"]:
            batch = sensor["measurement"]
            for native in batch["records"]:
                claim(native["observation_id"], "native_observation", native["raw_record_b64"])
                claim("sha256:" + native["source_evidence_digest"], "evidence", native["raw_record_b64"])
                claim("sha256:" + native["derived_evidence_digest"], "native_evidence",
                      {k: v for k, v in native.items() if k != "derived_evidence_digest"})
            claim("sha256:" + batch["assembly_digest"], "evidence", base64.b64encode(batch["assembly_toml"].encode()).decode())
            claim("sha256:" + batch["calibration_digest"], "native_calibration", batch["calibration"])
            claim("sha256:" + batch["uncertainty_digest"], "native_uncertainty", batch["uncertainty"])
            claim("sha256:" + batch["covariance_basis_digest"], "native_covariance_basis", batch["calibration"]["covariance_basis"])
        for execution in workspace["executions"]:
            claim(execution["execution_id"], "execution", execution)
        for result in workspace["results"]:
            claim(result["result_id"], "result", result)
        artifacts = [s["covariance"] for s in run["metadata"]["rci_source"]["sensors"]]
        artifacts += list(workspace["results"][0]["data"]["covariance_artifacts"].values())
        artifacts += [workspace["results"][1]["data"]["input_covariance"], workspace["results"][1]["data"]["output_covariance"]]
        for artifact in artifacts:
            claim(artifact["covariance_id"], "native_covariance", artifact)
        transformed = workspace["results"][1]
        mapping = {"operation_id": JSPT_OPERATION,
                   "source_covariance_id": transformed["data"]["input_covariance"]["covariance_id"],
                   **{k: transformed["parameters"][k] for k in MAP_FIELDS}}
        claim(transformed["data"]["output_covariance"]["basis"]["id"], "native_coordinate_map", mapping)
    return claims


def native_occurrences(bundle):
    return {e["execution_id"] for step in _retained_steps(bundle)
            for e in step["result"]["data"]["native_workspace"]["executions"]}


class MeasurementChainWorkflow(DeclaredWorkflow):
    FRESH_OCCURRENCE_MESSAGE = "Measurement chains must retain fresh native execution occurrences"

    def catalog_steps(self, bundle):
        return catalog_steps(bundle)

    def identity_claims(self, bundle):
        claims = {step["operation_id"]: ("operation", step["operation_id"]) for step in catalog_steps(bundle)}
        for identity, content in identity_claims(bundle).items():
            if identity in claims and canonical(claims[identity]) != canonical(content):
                raise ValueError("Retained identity collision")
            claims[identity] = content
        return claims

    def native_occurrences(self, bundle):
        return native_occurrences(bundle)

    def __init__(self):
        self.kind, self.role = "measurement-chain", "rci"
        self.ROLES, self.SOURCE_SCHEMA = ROLES, SOURCE_SCHEMA
        self.schema, self.operation = "ciw.measurement-chain-session.v1", "ciw.measurement-chain.v1"
        self.pin = PINS["rci"]

    def _source(self, raw):
        return _source(raw)

    @staticmethod
    def _runtime_projection(runtime):
        value = DeclaredWorkflow._runtime_projection(runtime)
        value["companions"] = {r: DeclaredWorkflow._runtime_projection(v) for r, v in runtime["companions"].items()}
        return value

    def _adapters(self, repositories, expected=None):
        if set(repositories) != ROLES:
            raise ValueError("Bind exact RCI, FSRT and JSPT native repositories")
        old = expected["rci"] if expected else None
        adapters, runtimes = {}, {}
        for role in sorted(ROLES):
            retained = (old if role == "rci" else old["companions"][role]) if old else None
            adapters[role] = _runtime(role, repositories[role], expected=retained)
            runtimes[role] = adapters[role].runtime_identity()
        runtime = {**runtimes["rci"], "companions": {r: runtimes[r] for r in ("fsrt", "jspt")}}
        if old and self._runtime_projection(runtime) != self._runtime_projection(old):
            raise ValueError("Native measurement runtime differs from its retained pin")
        return adapters, runtime, {r: Path(p) for r, p in repositories.items()}

    def _step(self, source, evidence_id, bound):
        adapters, runtime, repositories = bound
        before = {r: a.runtime_identity() for r, a in adapters.items()}
        current = {**before["rci"], "companions": {r: before[r] for r in ("fsrt", "jspt")}}
        if self._runtime_projection(current) != self._runtime_projection(runtime):
            raise ValueError("Native measurement runtime changed before execution")
        with tempfile.TemporaryDirectory(prefix="ciw-measurement-native-") as directory:
            root = Path(directory)
            summary = create_investigation(deepcopy(source["investigation"]), repositories["rci"], repositories["fsrt"], root / "investigation")
            if summary["status"] != "completed":
                raise AdapterRefusal("MEASUREMENT_CHAIN_REFUSED", "Native FSRT did not produce a retained result")
            parameters = {**deepcopy(source["covariance_map"]), "source_result_id": summary["results"][0]["result_id"]}
            transformed = execute_covariance(summary["workspace_file"], repositories["jspt"], parameters, root / "covariance")
            if transformed["status"] != "completed":
                raise AdapterRefusal("MEASUREMENT_CHAIN_REFUSED", "Native JSPT refused the declared covariance mapping")
            data = {"schema": DATA_SCHEMA, "scope": POLICY, "native_workspace": read_json(Path(transformed["workspace_file"]))}
        for role, adapter in adapters.items():
            if adapter.runtime_identity() != before[role]:
                raise ValueError("Native measurement runtime changed during execution")
        _check_data(source, data)
        if _runtime_records(data["native_workspace"]) != before:
            raise ValueError("Native records differ from the preflight runtime bindings")
        occurrence = "execution-" + uuid.uuid4().hex
        result = {"schema": RESULT_SCHEMA, "operation_id": self.operation, "execution_ref": occurrence,
                  "input_refs": [evidence_id], "data": data, "authority": AUTHORITY}
        result["result_id"] = digest(result)
        numerical = _numerical(data)
        return {"runtime_ref": self.role, "operation_id": self.operation, "execution_id": occurrence,
                "input_refs": [evidence_id], "request": source, "request_sha256": digest(source),
                "result": result, "result_sha256": digest(result), "result_id": result["result_id"],
                "numerical_result": numerical, "numerical_result_id": digest(numerical)}

    def _validate_step(self, step, source, evidence_id):
        exact_keys(step, {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256", "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"})
        if (step["runtime_ref"] != self.role or step["operation_id"] != self.operation or step["input_refs"] != [evidence_id]
                or canonical(step["request"]) != canonical(source) or not isinstance(step["execution_id"], str)
                or not re.fullmatch(r"execution-[a-f0-9]{32}", step["execution_id"])):
            raise ValueError("Measurement-chain request/occurrence binding mismatch")
        result = step["result"]
        exact_keys(result, {"schema", "operation_id", "execution_ref", "input_refs", "data", "authority", "result_id"})
        _check_data(source, result["data"])
        if (result != {"schema": RESULT_SCHEMA, "operation_id": self.operation, "execution_ref": step["execution_id"],
                       "input_refs": [evidence_id], "data": result["data"], "authority": AUTHORITY,
                       "result_id": digest({k: v for k, v in result.items() if k != "result_id"})}
                or result["result_id"] != step["result_id"] or canonical(step["numerical_result"]) != canonical(_numerical(result["data"]))):
            raise ValueError("Measurement result, projection or authority binding mismatch")
        if step["execution_id"] in native_ids(step):
            raise ValueError("Outer and native occurrences must remain distinct")
        for key, value in (("request_sha256", source), ("result_sha256", result), ("numerical_result_id", step["numerical_result"])):
            if step[key] != digest(value):
                raise ValueError("Measurement-chain content mismatch")

    def _check_verification(self, bundle, verification, source, evidence):
        super()._check_verification(bundle, verification, source, evidence)
        if native_ids(bundle["steps"][0]) & native_ids(verification["reproduction"]):
            raise ValueError("Verification must execute fresh native measurement occurrences")

    def _validate(self, bundle):
        try:
            exact_keys(bundle, {"schema", "session_id", "created_at", "source", "configuration", "runtimes", "steps", "bundle_digest", "verification"}, {"replay_receipts"})
            if (len(canonical(bundle)) > MAX_BYTES or bundle["schema"] != self.schema
                    or bundle["bundle_digest"] != bundle_digest(bundle)
                    or not re.fullmatch(r"session-[a-f0-9]{32}", bundle["session_id"])):
                raise ValueError("Measurement-chain bundle identity mismatch")
            _text(bundle["created_at"])
            evidence, = bundle["source"]["evidence"]
            raw = base64.b64decode(evidence["bytes_b64"], validate=True)
            source = _source(raw)
            if (evidence != {"artifact_ref": byte_digest(raw), "sha256": byte_digest(raw), "bytes_b64": base64.b64encode(raw).decode()}
                    or bundle["source"] != {"experiment_id": source["experiment_id"], "experiment_digest": digest(source), "evidence": [evidence]}
                    or bundle["configuration"] != POLICY or set(bundle["runtimes"]) != {"rci"}):
                raise ValueError("Measurement source or runtime binding mismatch")
            primary = bundle["runtimes"]["rci"]
            _check_runtime(primary, "rci", companions=True)
            if set(primary["companions"]) != {"fsrt", "jspt"}:
                raise ValueError("Missing native measurement companion runtimes")
            for role, runtime in primary["companions"].items():
                _check_runtime(runtime, role)
            step, = bundle["steps"]
            self._validate_step(step, source, evidence["artifact_ref"])
            self._check_verification(bundle, bundle["verification"], source, evidence["artifact_ref"])
            expected = {"rci": {k: v for k, v in primary.items() if k != "companions"}, **primary["companions"]}
            for retained in (step, bundle["verification"]["reproduction"]):
                if _runtime_records(retained["result"]["data"]["native_workspace"]) != expected:
                    raise ValueError("Outer and native measurement runtime bindings differ")
            return raw
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError, StopIteration) as exc:
            raise ValueError("Malformed measurement-chain session") from exc


workflow = MeasurementChainWorkflow()
_validate = workflow._validate
create_session = workflow.create_session
replay_session = workflow.replay_session
