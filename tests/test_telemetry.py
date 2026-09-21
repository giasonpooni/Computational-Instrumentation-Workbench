"""Native telemetry sessions are additive, bounded and authority-free."""
from copy import deepcopy
import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ciw.telemetry import (_source, _instant, _bundle_digest, _cbsr_layout, _numerical, _PPDAProjection,
                           _BOOTSTRAP, canonical, digest, create_session,
                           inspect_session, replay_session, read_session, save_session)
from ciw.adapters.protocol import AdapterRefusal
from ciw.adapters.subprocess import _json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "examples/telemetry/source.json").read_bytes()
CONFIG = json.loads((ROOT / "examples/telemetry/configuration.json").read_text())
NAMES = {"ppda": "Provenance-Preserving-Data-Acquisition", "stfe": "Streaming-Telemetry-Feature-Extraction",
         "gsie": "Geometric-State-Inference-Engine", "set": "State-Estimation-Evaluation-Testbed",
         "cbsr": "Constraint-Based-State-Reconciliation"}


@pytest.fixture(scope="module")
def repositories():
    root = os.environ.get("CIW_TELEMETRY_STACK_ROOT")
    if not root:
        pytest.skip("set CIW_TELEMETRY_STACK_ROOT to explicitly bound sibling checkouts")
    return {role: Path(root) / name for role, name in NAMES.items() if role != "cbsr"}


@pytest.fixture(scope="module")
def bundle(repositories):
    return create_session(SOURCE, CONFIG, repositories)


@pytest.mark.parametrize("field,value", [("crosscov_policy", "unknown"), ("covariance", None)])
def test_unknown_uncertainty_cannot_enter_path(field, value):
    source = json.loads(SOURCE)
    source[field] = value
    with pytest.raises(ValueError):
        _source(canonical(source))


@pytest.mark.parametrize("value", [True, "1", float("nan")])
def test_source_does_not_coerce_sample_values(value):
    source = json.loads(SOURCE)
    source["samples"][0]["value"] = value
    with pytest.raises(ValueError):
        _source(canonical(source))


def test_mapping_must_be_explicit_identity():
    source = json.loads(SOURCE)
    source["mappings"]["clock"]["target"] = "other-clock"
    with pytest.raises(ValueError, match="identity"):
        _source(canonical(source))


def test_time_rounding_is_refused():
    with pytest.raises(ValueError, match="microsecond"):
        _instant("2026-09-20T00:00:00Z", 1e-7)
    assert _instant("2026-09-20T00:00:00Z", .1).endswith("00.100000Z")


def test_json_literal_underflow_is_not_false_zero():
    with pytest.raises(AdapterRefusal):
        _source(SOURCE.replace(b'"value": 2.0', b'"value": 1e-999'))
    assert _json(b'{"zero":0e-999}') == {"zero": 0.0}


@pytest.mark.parametrize("field,value", [("state_labels", ["renamed"]), ("state_units", ["kg"]),
                                         ("frame_ref", "other-frame")])
def test_reconciliation_cannot_relabel_estimator_state(field, value):
    candidate = {"components": [{"name": "x", "value": 1, "unit": "m"}],
                 "covariance": {"frame": {"id": "declared-frame"}}}
    declaration = {"state_labels": ["x"], "state_units": ["m"], "frame_ref": "declared-frame"}
    declaration[field] = value
    with pytest.raises(ValueError, match="must match"):
        _cbsr_layout(declaration, candidate)


def test_numerical_equivalence_retains_output_coordinate_frame():
    result = {"result_artifact": {"components": [{"name": "x", "value": 1.0, "unit": "m"}],
              "covariance": {"matrix": [[1.0]], "frame": {"id": "frame-a", "semantics": "feature_space"}},
              "diagnostics": {}, "observation_binding": {"elapsed_seconds": 2.0}}}
    before = digest(_numerical("gsie", result))
    result["result_artifact"]["covariance"]["frame"]["id"] = "frame-b"
    assert digest(_numerical("gsie", result)) != before


def test_ppda_standalone_source_cannot_import_ignored_shadow(tmp_path):
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    source = b'def observation_batch_v1(**request):\n import math\n return {"finite": math.isfinite(1.0)}\n'
    (bridge / "instrumentation.py").write_bytes(source)
    (tmp_path / ".gitignore").write_text("bridge/math.py\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "bridge/instrumentation.py", ".gitignore"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=CIW test", "-c", "user.email=ciw@example.invalid",
                    "commit", "-qm", "standalone source fixture"], check=True)
    revision = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    (bridge / "math.py").write_text('raise RuntimeError("unverified shadow executed")\n')
    adapter = _PPDAProjection(tmp_path, revision, "instrumentation", source_root="bridge",
                              source_sha256=sha256(source).hexdigest())
    code, raw = adapter._run(_BOOTSTRAP,
        ["ppda", str(bridge), "", base64.b64encode(adapter.checked_source()).decode("ascii")], b"{}")
    assert code == 0
    assert _json(raw) == {"finite": True}


def test_bound_path_runs_correlated_window_with_analytic_scalar_result(bundle):
    assert bundle["verification"]["outcome"] == "passed"
    assert bundle["verification"]["independent"] is False
    feature = bundle["steps"][1]["result"]["result_artifact"]
    assert feature["components"][0]["value"] == 3.0
    assert feature["covariance"]["matrix"] == [[0.625]]
    estimate = bundle["steps"][2]["result"]["result_artifact"]
    assert estimate["components"][0]["value"] == pytest.approx(24 / 13)
    assert estimate["covariance"]["matrix"][0][0] == pytest.approx(5 / 13)
    assert inspect_session(bundle)["numerical_replay"] == "not_performed"


def test_replay_reexecutes_and_keeps_occurrences_distinct(bundle, repositories, tmp_path):
    replay = replay_session(bundle, repositories)
    assert replay["replay_receipt"]["numerical_match"] is True
    assert set(replay["replay_results"]) == {step["execution_id"] for step in bundle["steps"]}
    for before, after in zip(bundle["steps"], replay["session"]["steps"]):
        assert before["execution_id"] != after["execution_id"]
        assert before["numerical_result_id"] == after["numerical_result_id"]
    path = save_session(replay["session"], tmp_path)
    assert read_session(path) == replay["session"]
    with pytest.raises(ValueError, match="overwrite"):
        save_session(replay["session"], tmp_path)


@pytest.mark.parametrize("mutate", [
    lambda b: b["source"]["batch"]["components"][0].update(value=999),
    lambda b: b["configuration"]["gsie"]["prior"].update(mean=[999]),
    lambda b: b["steps"][2]["request"]["declaration"]["observation_model"].update(matrix=[[2.0]]),
    lambda b: b["steps"][1].update(operation_id="stfe.invented.v1"),
    lambda b: b["steps"][2].update(result_id="forged"),
    lambda b: b["steps"][1].update(input_refs=["unbound:evidence"]),
    lambda b: b["steps"][1].update(input_refs=[b["source"]["evidence"][0]["artifact_ref"]]),
])
def test_tampering_rejected_even_if_outer_bundle_rehashed(bundle, mutate):
    changed = deepcopy(bundle)
    changed.pop("verification", None)
    mutate(changed)
    changed["bundle_digest"] = _bundle_digest(changed)
    with pytest.raises(ValueError):
        inspect_session(changed)


def test_cli_inspect(bundle, tmp_path):
    path = save_session(bundle, tmp_path)
    result = subprocess.run([sys.executable, "-m", "ciw", "telemetry", "inspect", str(path)],
                            cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["admission"] == "not_performed"


def test_cli_create_and_replay(repositories, tmp_path):
    command = [sys.executable, "-m", "ciw", "telemetry"]
    bindings = [part for role, path in repositories.items() for part in ("--" + role + "-repo", str(path))]
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    create = subprocess.run(command + ["create", "--source", str(ROOT / "examples/telemetry/source.json"),
        "--configuration", str(ROOT / "examples/telemetry/configuration.json"),
        "--output-dir", str(tmp_path / "created"), *bindings],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
    assert create.returncode == 0, create.stderr
    saved = json.loads(create.stdout)["session_file"]
    replay = subprocess.run(command + ["replay", saved, "--output-dir", str(tmp_path / "replayed"), *bindings],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
    assert replay.returncode == 0, replay.stderr
    assert json.loads(replay.stdout)["replay_receipt"]["numerical_match"] is True


def test_runtime_pin_cannot_be_changed_by_bundle(bundle, repositories):
    changed = deepcopy(bundle)
    changed.pop("verification", None)
    changed["runtimes"]["gsie"]["revision"] = "0" * 40
    changed["bundle_digest"] = _bundle_digest(changed)
    with pytest.raises(ValueError, match="identity mismatch"):
        replay_session(changed, repositories)


def test_rehashed_non_numeric_result_tamper_refused_by_exact_reexecution(bundle, repositories):
    changed = deepcopy(bundle)
    changed.pop("verification", None)
    step = changed["steps"][2]
    # A hash-only or numerical-subset-only verifier could miss this changed
    # scientific provenance claim. Exact producer reproduction must reject it.
    step["result"]["result_artifact"]["replay_binding"]["snapshot"]["tampered_claim"] = "invented"
    artifact = step["result"]["result_artifact"]
    artifact["result_id"] = "sha256:" + sha256(artifact["schema"].encode() + b"\0" + canonical(
        {key: value for key, value in artifact.items() if key != "result_id"})).hexdigest()
    step["result_id"] = artifact["result_id"]
    step["result_sha256"] = digest(step["result"])
    changed["bundle_digest"] = _bundle_digest(changed)
    with pytest.raises(ValueError, match="exact pinned recomputation"):
        replay_session(changed, repositories)


def test_declared_zero_window_covariance(repositories):
    source = json.loads(SOURCE)
    source["crosscov_policy"] = "declared_zero"
    source["covariance"] = [[1.0, 0.0], [0.0, 1.0]]
    result = create_session(canonical(source), CONFIG, repositories)
    assert result["steps"][1]["result"]["result_artifact"]["covariance"]["matrix"] == [[0.5]]


def test_optional_cbsr_reconciliation(repositories):
    config = deepcopy(CONFIG)
    config["cbsr"] = {
        "state_labels": ["synthetic-state"], "state_units": ["m"],
        "frame_ref": "stfe-feature:frame:synthetic", "crosscov_policy": "declared",
        "constraints": {"constraint_id": "synthetic:fixed-state-two", "coefficients": [[1.0]],
                        "rhs": [2.0], "row_units": ["m"], "coefficient_policy": "declared_exact"},
        "max_normalized_residual": 10.0}
    bindings = {**repositories, "cbsr": next(iter(repositories.values())).parent / NAMES["cbsr"]}
    result = create_session(SOURCE, config, bindings)
    receipt = result["steps"][3]["result"]
    assert receipt["status"] == "accepted"
    assert receipt["reconciled"] == {"estimate": [2.0], "covariance": [[0.0]]}
    assert receipt["output_constraint_residual_zero"] is True
    candidate = result["steps"][2]["result"]["result_artifact"]
    assert receipt["request"]["source_result_digest"] == digest(candidate)
