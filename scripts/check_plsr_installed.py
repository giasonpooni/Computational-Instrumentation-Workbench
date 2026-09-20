"""Exercise the optional PLSR flow from an installed wheel outside the checkout."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def call(*args: str, cwd: Path, env: dict[str, str], expect: int = 0) -> dict:
    completed = subprocess.run(
        [sys.executable, "-I", "-m", "ciw", "plsr", *args], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=300,
    )
    if completed.returncode != expect:
        raise AssertionError(
            f"ciw plsr {' '.join(args)} exited {completed.returncode}, expected {expect}: "
            f"{completed.stderr.strip()}")
    return json.loads(completed.stdout)


def check_batch_milestone(repository: Path, root: Path, env: dict[str, str]) -> None:
    """Run an offline collection, interrupt it, resume, and compare two batches."""
    plan = repository / "examples" / "plsr" / "batch-parameter-sweep.json"
    first = root / "batch-a"
    partial = call("batch", "run", str(plan), "--output-dir", str(first), "--limit", "4",
                   cwd=root, env=env, expect=7)
    assert partial["status"] == "incomplete"
    assert partial["outcomes"]["unfinished"] == partial["requested"] - 4
    journal = (first / "journal.jsonl").read_bytes()
    retained = {path.name: path.read_bytes() for path in (first / "runs").iterdir()}
    assert len(retained) == 4

    # A run killed mid-write leaves an unterminated line. Resuming must discard
    # exactly those bytes and keep every bundle that finished.
    (first / "journal.jsonl").write_bytes(journal + b'{"entry_schema": "ciw-plsr-bat')
    finished = call("batch", "run", str(plan), "--output-dir", str(first), "--resume",
                    cwd=root, env=env)
    assert finished["status"] == "complete" and finished["discarded_partial_bytes"] == 30
    assert finished["outcomes"]["unfinished"] == 0
    assert sum(finished["outcomes"].values()) == finished["requested"]
    assert finished["resumed"] is True and finished["evaluated_this_run"] > 0
    for name, content in retained.items():
        assert (first / "runs" / name).read_bytes() == content
    assert (first / "journal.jsonl").read_bytes().startswith(journal)
    assert call("batch", "status", str(first), cwd=root, env=env) == finished
    for entry in finished["entries"]:
        bundle = call("inspect", str(first / "runs" / entry["saved_file"]),
                      cwd=root, env=env)["bundle"]
        assert bundle["record"]["record_digest"] == entry["record_digest"]

    second = root / "batch-b"
    again = call("batch", "run", str(plan), "--output-dir", str(second), cwd=root, env=env)
    assert again["status"] == "complete"
    comparison = call("batch", "compare", str(first), str(second), cwd=root, env=env)
    assert comparison["status"] == "identical", comparison["compatibility"]
    assert comparison["summary"]["compared"] == finished["requested"]
    assert comparison["summary"]["changed"] == 0
    assert {entry["baseline"]["resolution"] for entry in comparison["samples"]} == {"validated"}
    print(f"PASS: batch of {finished['requested']} samples interrupted, resumed and compared; "
          f"outcomes {finished['outcomes']}")


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="ciw-plsr-installed-") as directory:
        root = Path(directory)
        location = subprocess.run(
            [sys.executable, "-I", "-c", "import ciw; print(ciw.__file__)"],
            cwd=root, env=env, check=True, capture_output=True, text=True, timeout=300,
        )
        installed = Path(location.stdout.strip()).resolve()
        if installed.is_relative_to(repository / "src"):
            raise AssertionError("Run this check with a built wheel, not an editable installation")
        for kind, model_name in (("continuous", "continuous-affine"), ("discrete", "discrete-linear")):
            source = root / f"{model_name}.json"
            sample = root / f"{kind}-sample.json"
            shutil.copyfile(repository / "examples" / "plsr" / source.name, source)
            shutil.copyfile(repository / "examples" / "plsr" / sample.name, sample)
            imported = root / "models" / source.name
            declaration = call("import", str(source), "--output", str(imported), cwd=root, env=env)
            evaluated = call("evaluate", "--model", str(imported), "--sample", str(sample),
                             "--output-dir", str(root / "results"), cwd=root, env=env)
            bundle = evaluated["bundle"]
            assert bundle["model"]["artifact_digest"] == declaration["model_artifact_digest"]
            assert bundle["record"]["code"] == "CERTIFIED_WITH_MARGIN"
            assert bundle["verification_status"] == "not_verified"
            assert bundle["record"]["may_authorize"] is False
            saved = Path(evaluated["saved_file"])
            original = saved.read_bytes()
            # The retained bundle must be sufficient without either original input.
            source.unlink()
            imported.unlink()
            sample.unlink()
            inspected = call("inspect", str(saved), cwd=root, env=env)
            assert inspected["bundle"] == bundle
            replayed = call("replay", str(saved), "--output-dir", str(root / "replayed"), cwd=root, env=env)
            replay = replayed["bundle"]
            assert replay["replay_of"]["record_digest_matches"] is True
            assert replay["record"]["record_digest"] == bundle["record"]["record_digest"]
            assert replay["evidence_id"] == bundle["evidence_id"]
            assert replay["result_id"] != bundle["result_id"]
            assert replay["execution_id"] != bundle["execution_id"]
            assert saved.read_bytes() == original
            assert Path(replayed["saved_file"]).is_file()
        corpus = repository / "examples" / "plsr" / "corpus.json"
        before = corpus.read_bytes()
        report = call("corpus", "check", str(corpus), cwd=root, env=env)
        assert report["runtime_status"] == "supported", report["runtime_differences"]
        assert report["status"] == "reproduced", [
            case for case in report["cases"] if case["differences"]]
        assert report["outcome_counts"]["reproduced"] == report["case_count"]
        assert report["unused_tolerances"] == []
        # Checking is a read: an expectation is never refreshed by reproducing it.
        assert corpus.read_bytes() == before
        digests = [case["record_digest_matches"] for case in report["cases"]
                   if case["observed_outcome"] == "evaluated"]
        print(f"PASS: reference corpus reproduced {report['case_count']} declared cases; "
              f"{sum(1 for match in digests if match)}/{len(digests)} record digests matched")
        check_batch_milestone(repository, root, env)
    print("PASS: installed PLSR adapter imports, evaluates, inspects, retains and replays both model types")


if __name__ == "__main__":
    main()
