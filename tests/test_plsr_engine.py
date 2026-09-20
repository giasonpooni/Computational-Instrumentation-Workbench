"""The optional instrument must reject an unpinned installation before use."""

from importlib import resources
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ciw import plsr_engine as engine


def test_base_import_does_not_load_optional_runtime():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import ciw.plsr_engine; "
         "assert 'lyapunov' not in sys.modules; assert 'jsonschema' not in sys.modules"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def installed_runtime():
    pytest.importorskip("lyapunov")
    if sys.version_info < (3, 12):
        pytest.skip("PLSR needs Python 3.12+")
    engine._runtime.cache_clear()
    yield
    engine._runtime.cache_clear()


@pytest.fixture
def package_copy(installed_runtime, tmp_path, monkeypatch):
    original_files = resources.files
    source = Path(str(original_files("lyapunov")))
    destination = tmp_path / "lyapunov"
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(engine.resources, "files", lambda name:
                        destination if name == "lyapunov" else original_files(name))
    return destination


def test_pin_accepts_equivalent_source_with_crlf(package_copy):
    source = package_copy / "runtime.py"
    source.write_bytes(source.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    identity = engine.runtime_identity()
    assert identity["commit"] == engine.RUNTIME_COMMIT
    assert identity["repository"] == engine.RUNTIME_REPOSITORY
    assert identity["adapter_version"] == engine.ADAPTER_VERSION
    assert len(identity["source_digest"]) == 64


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra", "schema"])
def test_pin_rejects_changed_missing_and_extra_package_source(package_copy, mutation):
    if mutation == "changed":
        source = package_copy / "runtime.py"
        source.write_bytes(source.read_bytes() + b"\n# unexpected installation change\n")
    elif mutation == "missing":
        (package_copy / "py.typed").unlink()
    elif mutation == "extra":
        (package_copy / "unexpected.py").write_text("# unexpected source\n", encoding="utf-8")
    else:
        schema = package_copy / "schemas" / "model-artifact-v1.schema.json"
        schema.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="installed source differs"):
        engine.runtime_identity()


def test_pin_rejects_different_distribution_version(installed_runtime, monkeypatch):
    original_version = engine.metadata.version
    monkeypatch.setattr(engine.metadata, "version", lambda name:
                        "0.1.0rc2" if name == engine._PACKAGE_NAME else original_version(name))
    with pytest.raises(RuntimeError, match="version does not match"):
        engine.runtime_identity()


def test_optional_runtime_import_failure_has_install_hint(installed_runtime, monkeypatch):
    def missing_import(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(engine, "import_module", missing_import)
    with pytest.raises(RuntimeError, match=r"pip install.*\[plsr\]"):
        engine.runtime_identity()


def test_pin_manifest_covers_every_installed_source_file(installed_runtime):
    assert engine._source_files(resources.files("lyapunov")) == engine._manifest()["files"]


@pytest.mark.parametrize("mutation", ["value", "margin_ratio", "outside_level"])
def test_inspection_rejects_resealed_diagnostic_contradictions(installed_runtime, mutation, monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples" / "plsr"
    model = engine.load_model(examples / "continuous-affine.json")
    sample = json.loads((examples / "continuous-sample.json").read_text(encoding="utf-8"))
    record = engine.evaluate(model, sample)
    if mutation == "outside_level":
        record.update(code="OUTSIDE_LEVEL_SET", presentation_category="outside_declared_domain",
                      operationally_acceptable=False)
    else:
        record["diagnostics"][mutation] += 1
    runtime = engine._runtime()
    record["record_digest"] = runtime.record_digest(record)

    def forbidden_verdict(*args, **kwargs):
        raise AssertionError("Inspection must not invoke the evaluator")

    monkeypatch.setattr(runtime.ModelArtifact, "verdict", forbidden_verdict)
    with pytest.raises(ValueError, match="invalid PLSR record"):
        engine.validate_record(model, sample, record)
