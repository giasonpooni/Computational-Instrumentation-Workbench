"""Pinned telemetry operation script with retained bytes and scientific replay.

This additive session does not alter run.v1 or confer physical/admission truth.
Repository paths are operator bindings, never instructions loaded from a file.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
import math
from pathlib import Path
import sys
import uuid

from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import PinnedSubprocessAdapter, _json
from .exchange import _read, _identity, RESULT_SCHEMA
from .core.canonical import bundle_digest, byte_digest, canonical, digest, exact_keys, utc_instant, utc_now
from .session import write_json
from .pipelines import pin_map

SCHEMA = "ciw.telemetry-session.v1"
MAX_BYTES = 4 * 1024 * 1024
ROLES = {"ppda", "stfe", "gsie", "set", "cbsr"}


class _PPDAProjection(PinnedSubprocessAdapter):
    """Only the standalone stdlib bridge is executable, never PPDA's vendor tree."""
    def __init__(self, *args, source_sha256, **kwargs):
        self.approved_source_sha256 = source_sha256
        super().__init__(*args, **kwargs)

    def checked_source(self):
        path = self.repository_root / "bridge" / "instrumentation.py"
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("PPDA standalone source must not be a symlink")
        source = _read(path, 131_072)
        if sha256(source).hexdigest() != self.approved_source_sha256:
            raise ValueError("PPDA standalone projection differs from its approved source bytes")
        return source

    def _verify_source(self):
        root = self._git("rev-parse", "--show-toplevel").decode().strip()
        if Path(root).resolve() != self.repository_root or self._git("rev-parse", "HEAD").decode().strip() != self.revision:
            raise ValueError("PPDA standalone projection checkout revision mismatch")
        self.checked_source()
        return self._git("rev-parse", "HEAD^{tree}").decode().strip()

    def runtime_identity(self):
        identity = super().runtime_identity()
        identity.update(execution_scope="standalone_checked_source_only",
                        source_sha256=self.approved_source_sha256)
        return identity


def _runtime(role, repositories, expected=None):
    manifest = pin_map("telemetry")
    if role not in ROLES or role not in repositories:
        raise ValueError("Missing explicitly bound telemetry runtime: " + role)
    pin = manifest[role]
    cls = _PPDAProjection if role == "ppda" else PinnedSubprocessAdapter
    options = {"source_sha256": pin["source_sha256"]} if role == "ppda" else {}
    adapter = cls(
        repositories[role], pin["revision"], pin["module"], source_root=pin["source_root"],
        expected_python_sha256=expected.get("python_sha256") if expected else None,
        expected_python_version=expected.get("python_version") if expected else None,
        expected_dependencies=expected.get("dependencies") if expected else None,
        **options,
    )
    identity = adapter.runtime_identity()
    if expected and any(identity[key] != expected[key] for key in (
        "revision", "source_tree", "module", "source_root", "python_sha256", "python_version", "dependencies"
    )):
        raise ValueError("Telemetry replay runtime identity mismatch")
    if expected and role == "ppda" and any(identity[key] != expected.get(key) for key in ("source_sha256", "execution_scope")):
        raise ValueError("Telemetry PPDA standalone source identity mismatch")
    return adapter


# Fixed, reviewed Python adapter code. Neither module names nor executable code
# are taken from retained artifacts. -I removes PYTHONPATH/cwd/user-site imports;
# the existing bounded-process helper limits runtime and combined output bytes.
_BOOTSTRAP = r'''
import base64, json, pathlib, sys
role, source_root, set_root = sys.argv[1:4]
if role != "ppda":
    sys.path.insert(0, source_root)
request = json.loads(sys.stdin.buffer.read())
if role == "ppda":
    namespace = {"__name__": "_ciw_checked_ppda_projection"}
    exec(compile(base64.b64decode(sys.argv[4], validate=True), "checked_ppda_projection", "exec"), namespace)
    result = namespace["observation_batch_v1"](**request)
elif role == "stfe":
    from stfe import window_mean
    result = window_mean(request)
elif role == "gsie":
    from geometric_state_inference import StatePrior, LinearDynamics, LinearObservation, predict, update
    from geometric_state_inference.exchange import import_observation_batch, export_result_artifact
    declaration = request["declaration"]
    source = request["observation_batch"]
    observed = import_observation_batch(source, epoch=request["epoch"],
        variables=[x["name"] for x in source["components"]],
        units=[x["unit"] for x in source["components"]],
        frame=source["covariance"]["frame"], validator_repo=set_root)
    prior = StatePrior(**declaration["prior"])
    dynamics = LinearDynamics(**declaration["dynamics"])
    model = LinearObservation(**declaration["observation_model"])
    prior = predict(prior, dynamics, declaration["target_time"])
    estimate = update(prior, observed.observation, model)
    artifact = export_result_artifact(estimate, source=observed,
        variables=declaration["variables"], frame=declaration["frame"],
        execution_ref=request["execution_id"], created_at=request["created_at"],
        applicability="declared scalar telemetry feature observation model only",
        validator_repo=set_root)
    result = {"schema": "ciw.estimation-step.v1", "operation_id": "ciw.gsie-predict-update.v1",
        "component_operations": ["geometric-state-inference.predict.v1", "geometric-state-inference.update.v1"],
        "result_artifact": artifact}
elif role == "cbsr":
    from cbsr import reconcile_affine_exact
    result = reconcile_affine_exact(request)
elif role == "set":
    from state_estimation_testbed.replay import verify_replay_bundle
    result = verify_replay_bundle(request["bundle"], replay_results=request.get("replay_results"),
                                  created_at=request["created_at"])
else:
    raise ValueError("Unknown fixed telemetry adapter")
print(json.dumps(result, allow_nan=False, ensure_ascii=False))
'''


def _invoke(role, adapters, request):
    encoded = canonical(request)
    if len(encoded) > MAX_BYTES:
        raise ValueError("Telemetry operation exceeds byte budget")
    adapter = adapters[role]
    adapter.runtime_identity()
    adapters["set"].runtime_identity()
    source_bytes = base64.b64encode(adapter.checked_source()).decode("ascii") if role == "ppda" else ""
    code, raw = adapter._run(_BOOTSTRAP, [role, str(adapter.source_root),
                                        str(adapters["set"].repository_root), source_bytes], encoded)
    if code:
        raise AdapterRefusal("TELEMETRY_RUNTIME_REFUSED", "Pinned " + role + " operation refused its declared inputs")
    adapter.runtime_identity()
    adapters["set"].runtime_identity()
    result = _json(raw)
    if not isinstance(result, dict):
        raise ValueError("Telemetry result must be an object")
    return result


def _source(raw):
    try:
        return _source_inner(raw)
    except (KeyError, TypeError, OverflowError, AttributeError, RecursionError) as exc:
        raise ValueError("Malformed retained telemetry source") from exc


def _source_inner(raw):
    if not isinstance(raw, bytes) or len(raw) > MAX_BYTES:
        raise ValueError("Telemetry source must be bounded bytes")
    source = _json(raw)
    canonical(source)
    exact_keys(source, {"schema", "samples", "channel_id", "frame", "clock_basis", "epoch",
                   "clock_mapping_ref", "frame_mapping_ref", "crosscov_policy", "covariance_status",
                   "covariance", "calibration_refs", "admission_ref", "mappings"})
    if source["schema"] != "ciw.telemetry-source.v1":
        raise ValueError("Unsupported retained telemetry source")
    if not isinstance(source["samples"], list) or not 1 <= len(source["samples"]) <= 32:
        raise ValueError("Telemetry requires 1 to 32 retained scalar samples")
    for sample in source["samples"]:
        exact_keys(sample, {"name", "value", "unit", "event_time", "received_at"})
        for field in ("value", "event_time", "received_at"):
            if type(sample[field]) not in (int, float) or not math.isfinite(sample[field]):
                raise ValueError("Raw sample values and times must be finite real numbers")
        for field in ("event_time", "received_at"):
            utc_instant(source["epoch"], sample[field])
    # This operation has no general clock/frame transformation authority. A
    # named explicit identity map is the only implemented mapping, and its
    # complete declaration is retained. Other maps require a future operation.
    expected_maps = {
        "clock": {"ref": source["clock_mapping_ref"], "kind": "identity",
                  "source": source["clock_basis"], "target": source["clock_basis"], "epoch": source["epoch"]},
        "frame": {"ref": source["frame_mapping_ref"], "kind": "identity",
                  "source": source["frame"]["id"], "target": source["frame"]["id"]},
    }
    if source["mappings"] != expected_maps:
        raise ValueError("Only fully declared identity clock/frame mappings are implemented")
    if source["crosscov_policy"] not in {"declared", "declared_zero"}:
        raise ValueError("Unknown cross-covariance cannot enter this bounded estimation path")
    if source["covariance"] is None or source["covariance_status"] not in {"reported", "estimated", "propagated"}:
        raise ValueError("This path requires declared sample covariance")
    units = {row["unit"] for row in source["samples"]}
    if len(units) != 1:
        raise ValueError("A window contains one scalar channel with one declared unit")
    return source


def _ppda_request(source, evidence_ref):
    samples = source["samples"]
    return {
        "batch_id": "batch:" + evidence_ref.split(":", 1)[-1],
        "observed_at": utc_instant(source["epoch"], max(row["event_time"] for row in samples)),
        "received_at": utc_instant(source["epoch"], max(row["received_at"] for row in samples)),
        "clock_basis": source["clock_basis"],
        "components": [{key: row[key] for key in ("name", "value", "unit")} for row in samples],
        "covariance_status": source["covariance_status"], "covariance": source["covariance"],
        "frame": source["frame"], "covariance_method": "caller-declared full temporal covariance",
        "covariance_source_refs": [evidence_ref], "source_artifact_refs": [evidence_ref],
        "admission_ref": source["admission_ref"], "calibration_refs": source["calibration_refs"],
        "telemetry": {"channel_id": source["channel_id"],
            "sample_times": [row["event_time"] for row in samples],
            "received_times": [row["received_at"] for row in samples],
            "clock_mapping_ref": source["clock_mapping_ref"], "frame_mapping_ref": source["frame_mapping_ref"],
            "crosscov_policy": source["crosscov_policy"], "epoch": source["epoch"]},
    }


def _stfe_request(batch, configuration, runtime, execution_id, created_at):
    temporal = batch["telemetry"]
    window = configuration["window"]
    selected = [i for i, time in enumerate(temporal["sample_times"]) if window["start"] <= time < window["end"]]
    covariance = batch["covariance"]["matrix"]
    return {
        "operation_id": "stfe.window-mean.v1",
        "source_batch_ref": batch["batch_id"], "source_batch_digest": digest(batch),
        "samples": [{"observation_ref": row["name"], "event_time": temporal["sample_times"][i],
                     "received_at": temporal["received_times"][i], "value": row["value"], "missing": False}
                    for i, row in enumerate(batch["components"])],
        "window": deepcopy(window), "channel_id": temporal["channel_id"],
        "value_unit": batch["components"][0]["unit"], "frame": batch["covariance"]["frame"]["id"],
        "clock_basis": batch["clock_basis"], "clock_mapping_ref": temporal["clock_mapping_ref"],
        "frame_mapping_ref": temporal["frame_mapping_ref"], "calibration_refs": batch["calibration_refs"],
        "uncertainty": {"status": "known", "crosscov_policy": temporal["crosscov_policy"],
                        "matrix": [[covariance[i][j] for j in selected] for i in selected],
                        "observation_refs": [batch["components"][i]["name"] for i in selected]},
        "execution_id": execution_id, "created_at": created_at, "implementation_revision": runtime["revision"],
    }


def _feature_batch(feature, source, configuration):
    artifact = feature["result_artifact"]
    # An explicit transport projection of a feature result, not an acquisition
    # receipt. The observation model declares how a window mean measures state.
    return {"schema": "notation.instrument.observation-batch.v1",
        "batch_id": "feature-observation:" + artifact["result_id"].split(":", 1)[-1],
        "observed_at": utc_instant(source["epoch"], configuration["gsie"]["target_time"]),
        "received_at": utc_instant(source["epoch"], configuration["window"]["received_by"]),
        "clock_basis": source["clock_basis"], "components": deepcopy(artifact["components"]),
        "covariance": deepcopy(artifact["covariance"]), "source_artifact_refs": [artifact["result_id"]],
        "admission_ref": "ciw:feature-transport-only", "admission_status": "reference_only",
        "calibration_refs": deepcopy(artifact["calibration_refs"])}


def _numerical(role, result):
    if role == "ppda":
        return deepcopy(result)
    if role == "stfe":
        return deepcopy(result["numerical_result"])
    if role == "gsie":
        artifact = result["result_artifact"]
        return {"components": artifact["components"], "covariance": artifact["covariance"]["matrix"],
                "frame": artifact["covariance"]["frame"], "diagnostics": artifact["diagnostics"],
                "time": artifact["observation_binding"]["elapsed_seconds"]}
    if role == "cbsr":
        return {key: value for key, value in result.items() if key not in {
            "schema", "execution_id", "input_state_id", "output_state_id", "constraint_id",
            "request_digest", "request", "evidence_refs", "result_id", "source_binding_verification"}}
    raise ValueError("Unknown numerical operation")


def _step(role, operation_id, request, inputs, adapters, execution_id):
    result = _invoke(role, adapters, request)
    artifact = result.get("result_artifact", result)
    numerical = _numerical(role, result)
    return {"runtime_ref": role, "operation_id": operation_id, "execution_id": execution_id,
        "input_refs": inputs, "request": request, "request_sha256": digest(request),
        "result": result, "result_sha256": digest(result),
        "result_id": artifact.get("result_id", artifact.get("batch_id")),
        "numerical_result": numerical, "numerical_result_id": digest(numerical)}


def _execute(raw, configuration, adapters, occurrence_template=None):
    source = _source(raw)
    exact_keys(configuration, {"window", "gsie"}, {"cbsr"})
    exact_keys(configuration["gsie"], {"prior", "dynamics", "observation_model", "target_time", "variables", "frame",
                                  "prior_measurement_crosscov_policy", "feature_observation_semantics"})
    if configuration["gsie"]["prior_measurement_crosscov_policy"] != "declared_zero":
        raise ValueError("This update requires explicitly declared prior-feature independence")
    if configuration["gsie"]["feature_observation_semantics"] != "window_mean_observes_declared_state_at_window_end":
        raise ValueError("An explicit window-feature-to-state observation interpretation is required")
    if configuration["gsie"]["target_time"] != configuration["window"]["end"]:
        raise ValueError("The scalar feature observation is declared at the window end")
    created_at = occurrence_template["created_at"] if occurrence_template else utc_now()
    runtimes = {role: adapter.runtime_identity() for role, adapter in adapters.items()}
    evidence_ref = byte_digest(raw)
    identities = iter(step["execution_id"] for step in occurrence_template["steps"]) if occurrence_template else None
    execute_id = lambda: next(identities) if identities is not None else "execution-" + uuid.uuid4().hex
    ppda = _step("ppda", "ppda.observation-batch.v1", _ppda_request(source, evidence_ref),
                 [evidence_ref], adapters, execute_id())
    batch = ppda["result"]
    execution_id = execute_id()
    stfe = _step("stfe", "stfe.window-mean.v1", _stfe_request(batch, configuration, runtimes["stfe"],
                  execution_id, created_at), [batch["batch_id"]], adapters, execution_id)
    execution_id = execute_id()
    request = {"declaration": deepcopy(configuration["gsie"]), "epoch": source["epoch"],
               "observation_batch": _feature_batch(stfe["result"], source, configuration),
               "execution_id": execution_id, "created_at": created_at}
    gsie = _step("gsie", "ciw.gsie-predict-update.v1", request,
                 [stfe["result_id"]], adapters, execution_id)
    steps = [ppda, stfe, gsie]
    if "cbsr" in configuration:
        candidate = gsie["result"]["result_artifact"]
        request = deepcopy(configuration["cbsr"])
        _cbsr_layout(request, candidate)
        request.update({"estimate": [row["value"] for row in candidate["components"]],
            "covariance": candidate["covariance"]["matrix"], "state_id": candidate["state_id"],
            "evidence_refs": [gsie["result_id"]], "source_result_id": gsie["result_id"],
            "source_result_digest": digest(candidate), "execution_id": execute_id()})
        steps.append(_step("cbsr", "cbsr.affine-exact.v1", request, [gsie["result_id"]],
                           adapters, request["execution_id"]))
    bundle = {"schema": SCHEMA, "session_id": "session-" + uuid.uuid4().hex, "created_at": created_at,
        "source": {"batch": batch, "batch_sha256": digest(batch),
            "batch_bytes_b64": base64.b64encode(canonical(batch)).decode("ascii"),
            "evidence": [{"artifact_ref": evidence_ref, "sha256": evidence_ref,
                "bytes_b64": base64.b64encode(raw).decode("ascii")}]},
        "configuration": deepcopy(configuration), "runtimes": runtimes, "steps": steps}
    bundle["bundle_digest"] = bundle_digest(bundle)
    return bundle


def _cbsr_layout(request, candidate):
    declared = {"state_labels": [row["name"] for row in candidate["components"]],
                "state_units": [row["unit"] for row in candidate["components"]],
                "frame_ref": candidate["covariance"]["frame"]["id"]}
    if any(request.get(key) != value for key, value in declared.items()):
        raise ValueError("Reconciliation state labels, units and frame must match the estimator artifact")


def _validate_retained(bundle):
    try:
        return _validate_retained_inner(bundle)
    except (KeyError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed retained telemetry session") from exc


# The workbench validates every retained kind through ``_validate``.
_validate = _validate_retained


def _validate_retained_inner(bundle):
    if len(canonical(bundle)) > MAX_BYTES:
        raise ValueError("Telemetry session exceeds byte budget")
    exact_keys(bundle, {"schema", "session_id", "created_at", "source", "configuration", "runtimes", "steps", "bundle_digest"},
          {"verification", "replay_receipts"})
    if bundle["schema"] != SCHEMA or bundle["bundle_digest"] != bundle_digest(bundle):
        raise ValueError("Telemetry session content binding mismatch")
    if "verification" in bundle:
        verification = bundle["verification"]
        if verification.get("subject_ref") != bundle["bundle_digest"]:
            raise ValueError("Retained verification does not bind this telemetry session")
        _identity(verification, "verification_id")
    evidence = bundle["source"]["evidence"]
    if len(evidence) != 1:
        raise ValueError("This bounded path retains exactly one raw telemetry source")
    raw = base64.b64decode(evidence[0]["bytes_b64"], validate=True)
    if byte_digest(raw) != evidence[0]["sha256"] or evidence[0]["artifact_ref"] != evidence[0]["sha256"]:
        raise ValueError("Telemetry evidence byte digest mismatch")
    source = _source(raw)
    batch = _json(base64.b64decode(bundle["source"]["batch_bytes_b64"], validate=True))
    if batch != bundle["source"]["batch"] or digest(batch) != bundle["source"]["batch_sha256"]:
        raise ValueError("Telemetry observation batch byte binding mismatch")
    steps = bundle["steps"]
    expected_roles = ["ppda", "stfe", "gsie"] + (["cbsr"] if "cbsr" in bundle["configuration"] else [])
    if [step["runtime_ref"] for step in steps] != expected_roles or set(bundle["runtimes"]) != set(expected_roles) | {"set"}:
        raise ValueError("Telemetry operation order/runtime set mismatch")
    prior_ids = {evidence[0]["artifact_ref"]}
    executions = set()
    for step in steps:
        if step["request_sha256"] != digest(step["request"]) or step["result_sha256"] != digest(step["result"]):
            raise ValueError("Telemetry step content binding mismatch")
        if step["numerical_result_id"] != digest(step["numerical_result"]) or step["numerical_result"] != _numerical(step["runtime_ref"], step["result"]):
            raise ValueError("Telemetry numerical result binding mismatch")
        if not step["input_refs"] or not set(step["input_refs"]) <= prior_ids:
            raise ValueError("Telemetry step has unresolved input bindings")
        if step["execution_id"] in executions or step["result_id"] in prior_ids or step["execution_id"] == step["result_id"]:
            raise ValueError("Telemetry occurrence and result identities must remain distinct")
        executions.add(step["execution_id"])
        prior_ids.add(step["result_id"])
        artifact = step["result"].get("result_artifact", step["result"])
        if artifact.get("schema") == RESULT_SCHEMA:
            _identity(artifact, "result_id")
        if step["result_id"] != artifact.get("result_id", artifact.get("batch_id")):
            raise ValueError("Telemetry result identity differs from its artifact")
        if "execution_ref" in artifact and artifact["execution_ref"] != step["execution_id"]:
            raise ValueError("Telemetry execution reference differs from its artifact")
    expected_operations = ["ppda.observation-batch.v1", "stfe.window-mean.v1", "ciw.gsie-predict-update.v1"]
    if "cbsr" in bundle["configuration"]:
        expected_operations.append("cbsr.affine-exact.v1")
    if [step["operation_id"] for step in steps] != expected_operations:
        raise ValueError("Telemetry operation identity mismatch")
    exact_inputs = [[evidence[0]["artifact_ref"]]] + [[previous["result_id"]] for previous in steps[:-1]]
    if [step["input_refs"] for step in steps] != exact_inputs:
        raise ValueError("Telemetry inputs must follow the exact retained operation graph")
    if steps[0]["request"] != _ppda_request(source, evidence[0]["artifact_ref"]) or steps[0]["result"] != batch:
        raise ValueError("Retained raw evidence differs from PPDA projection inputs")
    expected_stfe = _stfe_request(batch, bundle["configuration"], bundle["runtimes"]["stfe"],
                                 steps[1]["execution_id"], steps[1]["request"]["created_at"])
    if steps[1]["request"] != expected_stfe:
        raise ValueError("Retained feature request differs from PPDA evidence or window")
    gsie_request = steps[2]["request"]
    if (gsie_request["declaration"] != bundle["configuration"]["gsie"]
            or gsie_request["observation_batch"] != _feature_batch(steps[1]["result"], source, bundle["configuration"])
            or gsie_request["epoch"] != source["epoch"] or gsie_request["execution_id"] != steps[2]["execution_id"]):
        raise ValueError("Estimator input differs from retained feature/model binding")
    if len(steps) == 4:
        candidate = steps[2]["result"]["result_artifact"]
        expected_cbsr = deepcopy(bundle["configuration"]["cbsr"])
        _cbsr_layout(expected_cbsr, candidate)
        expected_cbsr.update({"estimate": [row["value"] for row in candidate["components"]],
            "covariance": candidate["covariance"]["matrix"], "state_id": candidate["state_id"],
            "evidence_refs": [steps[2]["result_id"]], "source_result_id": steps[2]["result_id"],
            "source_result_digest": digest(candidate), "execution_id": steps[3]["execution_id"]})
        if steps[3]["request"] != expected_cbsr:
            raise ValueError("Reconciliation input differs from retained estimator binding")
    return raw


def inspect_session(bundle):
    """Check retained content without executing engines or accepting receipts."""
    _validate_retained(bundle)
    return {"schema": "ciw.telemetry-inspection.v1", "session_id": bundle["session_id"],
        "bundle_digest": bundle["bundle_digest"], "status": "content_consistent",
        "operation_ids": [step["operation_id"] for step in bundle["steps"]],
        "numerical_replay": "not_performed", "admission": "not_performed",
        "verification_trust": "retained_content_checked_not_authenticated_or_recomputed",
        "verification": deepcopy(bundle.get("verification"))}


def _adapters(configuration, repositories, expected=None):
    roles = {"ppda", "stfe", "gsie", "set"} | ({"cbsr"} if "cbsr" in configuration else set())
    if set(repositories) != roles:
        raise ValueError("Bind exactly the required repositories explicitly")
    return {role: _runtime(role, repositories, expected.get(role) if expected else None)
            for role in sorted(roles)}


REQUIRED_ROLES = frozenset({"ppda", "stfe", "gsie", "set"})


def check_bindings(bindings):
    """Validate host bindings before the workbench advertises telemetry; CBSR is optional."""
    _adapters({"cbsr": {}} if "cbsr" in bindings else {}, bindings)


def select_bindings(configuration, bindings):
    """The bound providers one execution under this configuration uses."""
    roles = set(REQUIRED_ROLES) | ({"cbsr"} if "cbsr" in configuration else set())
    if not roles <= bindings.keys():
        raise AdapterRefusal("operation_unavailable", "Telemetry reconciliation needs an explicitly bound CBSR checkout")
    return {role: bindings[role] for role in roles}


def replay_bindings(bundle, bindings):
    """Replay binds exactly the provider set the retained session ran."""
    roles = set(bundle["runtimes"])
    if not roles <= bindings.keys():
        raise AdapterRefusal("operation_unavailable", "Replay requires the original telemetry provider set")
    return {role: bindings[role] for role in roles}


def create_session(source_bytes, configuration, repositories):
    """Execute all declared operations, reexecute, then attach SET verification."""
    source = _source(source_bytes)
    del source
    adapters = _adapters(configuration, repositories)
    bundle = _execute(source_bytes, configuration, adapters)
    _validate_retained(bundle)
    recomputed = _execute(source_bytes, configuration, adapters, occurrence_template=bundle)
    _compare_reproduction(bundle, recomputed)
    replay_results = {old["execution_id"]: new["numerical_result"]
                      for old, new in zip(bundle["steps"], recomputed["steps"])}
    bundle["verification"] = _invoke("set", adapters, {"bundle": bundle,
        "replay_results": replay_results, "created_at": utc_now()})
    if bundle["verification"].get("outcome") != "passed":
        raise ValueError("SET did not verify the replayed telemetry session")
    return bundle


def replay_session(bundle, repositories):
    """Recompute from retained source/configuration under exact approved pins."""
    raw = _validate_retained(bundle)
    adapters = _adapters(bundle["configuration"], repositories, bundle["runtimes"])
    reproduced = _execute(raw, bundle["configuration"], adapters, occurrence_template=bundle)
    _compare_reproduction(bundle, reproduced)
    fresh = _execute(raw, bundle["configuration"], adapters)
    replay_results = {old["execution_id"]: new["numerical_result"]
                      for old, new in zip(bundle["steps"], fresh["steps"])}
    verification = _invoke("set", adapters, {"bundle": bundle,
        "replay_results": replay_results, "created_at": utc_now()})
    matched = all(digest(replay_results[step["execution_id"]]) == step["numerical_result_id"] for step in bundle["steps"])
    receipt = {"schema": "ciw.telemetry-replay.v1", "source_bundle_digest": bundle["bundle_digest"],
        "replayed_bundle_digest": fresh["bundle_digest"], "numerical_match": matched,
        "verification": verification, "admission": "not_performed"}
    receipt["replay_id"] = digest(receipt)
    if not matched or verification.get("outcome") != "passed":
        raise ValueError("Telemetry numerical replay or SET verification mismatch")
    fresh["verification"] = _invoke("set", adapters, {"bundle": fresh,
        "replay_results": {new["execution_id"]: old["numerical_result"]
                           for old, new in zip(bundle["steps"], fresh["steps"])}, "created_at": utc_now()})
    fresh["replay_receipts"] = [receipt]
    return {"session": fresh, "replay_receipt": receipt, "replay_results": replay_results}


def _compare_reproduction(original, reproduced):
    # Exact rerun of the original caller metadata proves all retained result
    # fields, not only a selected numerical view. It is a reproducibility
    # check, not a claim that the original occurrence just happened again.
    for old, new in zip(original["steps"], reproduced["steps"]):
        if any(old[field] != new[field] for field in ("request_sha256", "result_sha256", "numerical_result_id")):
            raise ValueError("Retained operation result differs from exact pinned recomputation")


def read_session(path):
    return _json(_read(Path(path), MAX_BYTES))


def save_session(bundle, output_dir):
    _validate_retained(bundle)
    path = Path(output_dir) / "telemetry-session.json"
    if path.exists():
        raise ValueError("Refusing to overwrite an existing telemetry session")
    return write_json(path, bundle)
