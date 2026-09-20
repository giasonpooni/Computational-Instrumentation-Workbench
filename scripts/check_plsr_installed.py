"""Exercise the optional PLSR flow from an installed wheel outside the checkout."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def call(*args: str, cwd: Path, env: dict[str, str]) -> dict:
    completed = subprocess.run(
        [sys.executable, "-I", "-m", "ciw", "plsr", *args], cwd=cwd, env=env,
        check=True, capture_output=True, text=True, timeout=300,
    )
    return json.loads(completed.stdout)


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
    print("PASS: installed PLSR adapter imports, evaluates, inspects, retains and replays both model types")


if __name__ == "__main__":
    main()
