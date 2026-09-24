"""JSPT covariance operations attached to retained investigation results.

CIW resolves source identities and retains the operation. JSPT alone computes
the covariance push; a saved artifact never supplies executable code.
"""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import uuid

from .adapters.protocol import AdapterRefusal
from .core.identities import digest
from .investigation import _runtime, _summary
from .operations.registry import Operation
from .session import Session


JSPT_OPERATION = "jspt.covariance-propagate.v1"
MAP_FIELDS = {"jacobian", "output_quantity_ids", "output_units", "output_frame", "output_reference_values", "map_kind"}
PARAMETER_FIELDS = MAP_FIELDS | {"source_result_id", "source_artifact", "source_covariance"}



def provider_binding():
    """The terminal JSPT operation this module runs, for ``pipelines.check_providers``."""
    return {"jspt": {"pin": {}, "operations": {"default": [JSPT_OPERATION]}}}

def result_covariance(result, name):
    if not isinstance(name, str):
        raise ValueError("A covariance source needs an artifact name")
    data = result["data"]
    if result["operation_id"] == "fsrt.tank-reconstruct.v2":
        artifact = data.get("covariance_artifacts", {}).get(name)
    elif result["operation_id"] == JSPT_OPERATION and name == "output_covariance":
        artifact = data.get(name)
    else:
        artifact = None
    if not isinstance(artifact, dict):
        raise ValueError("The selected result has no supported covariance artifact with that name")
    return copy.deepcopy(artifact)


def operation_inputs(parameters):
    if not isinstance(parameters, dict) or set(parameters) != PARAMETER_FIELDS:
        raise ValueError("Covariance parameters require one retained source and an explicit ordered map")
    return {"covariance": copy.deepcopy(parameters["source_covariance"]),
            **{key: copy.deepcopy(parameters[key]) for key in MAP_FIELDS}}


def validate_source_link(parameters, results):
    operation_inputs(parameters)
    source = results.get(parameters["source_result_id"])
    if source is None:
        raise ValueError("Covariance source result is not retained in this investigation")
    if parameters["source_covariance"] != result_covariance(source, parameters["source_artifact"]):
        raise ValueError("Covariance source artifact differs from its retained result")


def bind_jspt(session, repo, python_executable=None, expected=None):
    adapter = _runtime("jspt", repo, python_executable, expected)

    def execute(run, parameters):
        with session._lock:
            results = copy.deepcopy(session.results)
        validate_source_link(parameters, results)
        return adapter.invoke(JSPT_OPERATION, operation_inputs(parameters))

    session.operations.register(Operation(JSPT_OPERATION, "backend", execute, adapter.runtime_identity))
    return adapter


def _execute(session, parameters, output_dir):
    response = session.handle({"protocol_version": 1, "request_id": uuid.uuid4().hex,
                               "type": "operation.execute", "payload": {
                                   "operation_id": JSPT_OPERATION, "parameters": parameters}})
    if response["type"] == "error":
        raise ValueError(response["payload"]["message"])
    path = Path(output_dir) / "workspace.json"
    session.save_workspace(path)
    return _summary(session, path)


def execute_covariance(workspace, jspt_repo, parameters, output_dir, python_executable=None):
    if not isinstance(parameters, dict) or not MAP_FIELDS <= parameters.keys() or set(parameters) - MAP_FIELDS - {"source_result_id", "source_artifact"}:
        raise ValueError("Declare the ordered covariance map; source result/artifact selectors are optional")
    with tempfile.TemporaryDirectory(prefix="ciw-covariance-inspect-") as directory:
        original = Session.from_workspace(Path(workspace), Path(directory))
        source_id = parameters.get("source_result_id")
        if source_id is None:
            eligible = [r for r in original.results.values() if r["operation_id"] in {"fsrt.tank-reconstruct.v2", JSPT_OPERATION}]
            if not eligible:
                raise ValueError("No covariance-bearing result is retained")
            source_id = eligible[-1]["result_id"]
        source = original.results.get(source_id)
        if source is None:
            raise ValueError("Unknown source result")
        name = parameters.get("source_artifact", "output_covariance" if source["operation_id"] == JSPT_OPERATION else "reconciled")
        resolved = {**copy.deepcopy(parameters), "source_result_id": source_id, "source_artifact": name,
                    "source_covariance": result_covariance(source, name)}
        _runtime("jspt", jspt_repo, python_executable).runtime_identity()
    session = Session.from_workspace(Path(workspace), Path(output_dir))
    bind_jspt(session, jspt_repo, python_executable)
    return _execute(session, resolved, output_dir)


def replay_covariance(workspace, jspt_repo, output_dir, python_executable=None):
    with tempfile.TemporaryDirectory(prefix="ciw-covariance-replay-") as directory:
        original = Session.from_workspace(Path(workspace), Path(directory))
        previous = [e for e in original.executions.values() if e["operation_id"] == JSPT_OPERATION]
        if not previous:
            raise AdapterRefusal("replay_unavailable", "No JSPT covariance execution retained")
        last = previous[-1]
        expected, parameters = last["runtime"], copy.deepcopy(last["parameters"])
        if expected is None:
            raise AdapterRefusal("replay_unavailable", "The retained attempt had no bound numerical runtime; execute a new operation explicitly")
        _runtime("jspt", jspt_repo, python_executable, expected).runtime_identity()
        old = original.results.get(last["result_id"])
    session = Session.from_workspace(Path(workspace), Path(output_dir))
    bind_jspt(session, jspt_repo, python_executable, expected)
    summary = _execute(session, parameters, output_dir)
    if old and summary["status"] == "completed":
        matches = digest(old["data"]) == digest(summary["results"][-1]["data"])
        summary["replay_data_digest_matches"] = matches
        if not matches:
            raise AdapterRefusal("replay_mismatch", "JSPT replay differs; both result identities were retained")
    return summary
