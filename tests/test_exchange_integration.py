"""Actual acquisition/runtime producers -> pinned SET validator -> CIW CLI.

The completed execution is a synthetic native-byte commitment fixture, not a
Rust-engine execution or a physical measurement. Producer interpretation and
admission remain explicitly declared; no operational result is promoted.
"""

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRODUCE = r'''
import json
import os
from pathlib import Path

from bridge import instrumentation as acquisition
from execution import instrumentation as runtime
from execution.commitments import COMPUTATION_TAG, OUTPUT_TAG, canonical_u32, commit_hex
from execution.engine import ExecutionResult
from execution.specification import ExecutionSpecification

assert Path(acquisition.__file__).resolve() == Path(os.environ["CIW_ACQUISITION_REPO"]).resolve() / "bridge/instrumentation.py"
assert Path(runtime.__file__).resolve() == Path(os.environ["CIW_RUNTIME_REPO"]).resolve() / "execution/instrumentation.py"
status = os.environ["EXCHANGE_FIXTURE_COVARIANCE"]
matrix = None if status == "unknown" else [[0.04, 0.01], [0.01, 0.09]]
common = dict(
    components=[dict(name="east", value=1.2, unit="m"), dict(name="north", value=-0.4, unit="m")],
    covariance_status=status, covariance=matrix,
    frame=dict(id="frame:synthetic-enu", semantics="tangent", basis=["east", "north"],
               evaluation_point=[-79.5, 43.7, 100.0]),
    covariance_method="synthetic declaration", covariance_source_refs=["example:source"],
    calibration_refs=[],
)
observation = acquisition.observation_batch_v1(
    batch_id="example:synthetic-batch", observed_at="2026-09-20T12:00:00Z",
    received_at="2026-09-20T12:00:01Z", clock_basis="synthetic UTC fixture",
    source_artifact_refs=["example:source"], admission_ref="example:unresolved-admission", **common,
)
spec = ExecutionSpecification(program=b"synthetic-fixture", configuration=b"",
                              input_payload=json.dumps(observation, sort_keys=True).encode())
output = b"synthetic-output-not-an-engine-run"
output_id = commit_hex(OUTPUT_TAG, [output])
computation_id = commit_hex(COMPUTATION_TAG, [bytes.fromhex(spec.program_identity()),
    bytes.fromhex(spec.input_identity()), bytes.fromhex(output_id), canonical_u32(0)])
execution = ExecutionResult(specification=spec, specification_identity=spec.identity(),
    program_identity=spec.program_identity(), input_identity=spec.input_identity(),
    engine_occurrence=0, status="completed", exit_code=0, output=output,
    output_identity=output_id, computation_identity=computation_id, detail=None)
result = runtime.result_artifact_v1(execution, input_refs=[observation["batch_id"]],
    applicability="synthetic interoperability fixture only", created_at="2026-09-20T12:00:02Z",
    **common)
verification = runtime.verification_artifact_v1(subject_ref=result["result_id"],
    verifier_ref="example:fixture-checker", created_at="2026-09-20T12:00:03Z",
    checks=[dict(name="fixture assertion", outcome="failed", basis="intentional failed-check fixture")],
    limitations=["synthetic check; not independent verification"])
print(json.dumps([observation, result, verification], allow_nan=False))
'''


@pytest.mark.parametrize("covariance_status", ["reported", "unknown"])
def test_actual_producer_cli_roundtrip_preserves_claim_boundaries(tmp_path, covariance_status):
    environment = os.environ.copy()
    names = ["CIW_ACQUISITION_REPO", "CIW_RUNTIME_REPO", "CIW_SET_REPO"]
    if not all(environment.get(name) for name in names):
        pytest.skip("Set CIW_ACQUISITION_REPO, CIW_RUNTIME_REPO and CIW_SET_REPO for actual producer integration")
    for name in names:
        environment[name] = str(Path(environment[name]).resolve())
    environment["PYTHONPATH"] = os.pathsep.join(environment[name] for name in names[:2])
    environment["EXCHANGE_FIXTURE_COVARIANCE"] = covariance_status
    produced = subprocess.run([sys.executable, "-c", PRODUCE], cwd=tmp_path, env=environment,
                              text=True, capture_output=True, timeout=30, check=False)
    assert produced.returncode == 0, produced.stderr
    artifacts = json.loads(produced.stdout)
    paths = []
    for index, artifact in enumerate(artifacts):
        path = tmp_path / f"artifact-{index}.json"
        path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        paths.append(path)
    original = [path.read_bytes() for path in paths]
    environment["PYTHONPATH"] = str(ROOT / "src")
    inspected = subprocess.run([sys.executable, "-m", "ciw", "exchange", "inspect",
                                *map(str, paths), "--validator-repo", environment["CIW_SET_REPO"]],
                               cwd=tmp_path, env=environment, text=True, capture_output=True,
                               timeout=30, check=False)
    assert inspected.returncode == 0, inspected.stderr
    report = json.loads(inspected.stdout)
    assert report["status"] == "conformant"
    assert [item["artifact"] for item in report["artifacts"]] == artifacts
    assert [path.read_bytes() for path in paths] == original
    assert [item["source_bytes_sha256"] for item in report["artifacts"]] == [sha256(raw).hexdigest() for raw in original]
    assert {item["status"] for item in report["links"]} == {"matched_supplied_reference", "unresolved_external_reference"}
    assert sum(item["status"] == "matched_supplied_reference" for item in report["links"]) == 2
    assert artifacts[0]["admission_status"] == "reference_only"
    assert artifacts[1]["execution_binding"]["components_status"] == "caller_declared"
    assert artifacts[1]["execution_binding"]["behavior_status"] == "unverified"
    assert report["artifacts"][2]["artifact"]["outcome"] == "failed"
    assert report["authority"]["verification_independence"] == "not_established"
    assert report["authority"]["may_authorize"] is False
    for item in report["artifacts"][:2]:
        assert item["artifact"]["covariance"]["status"] == covariance_status
        assert item["covariance_validation"]["effective_rank"] == (None if covariance_status == "unknown" else 2)
    assert sorted(tmp_path.iterdir()) == sorted(paths)
    # A changed interpretation cannot retain the previous result content ID.
    artifacts[1]["components"][0]["value"] += 1
    paths[1].write_text(json.dumps(artifacts[1]), encoding="utf-8")
    refused = subprocess.run([sys.executable, "-m", "ciw", "exchange", "inspect",
                              *map(str, paths), "--validator-repo", environment["CIW_SET_REPO"]],
                             cwd=tmp_path, env=environment, text=True, capture_output=True,
                             timeout=30, check=False)
    assert refused.returncode == 2
    assert "result_id does not match" in refused.stderr
    assert refused.stdout == ""
