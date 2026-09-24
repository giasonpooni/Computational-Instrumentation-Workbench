"""One CI shape: kernel surfaces, descriptor validation and a provider-gate matrix keyed by pin."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ci_matrix  # noqa: E402

from ciw import pipelines  # noqa: E402


def test_registry_gates_every_pinned_pipeline_and_provider():
    registry = ci_matrix.check()
    gated = {kind for gate in registry["gates"] for kind in gate["kinds"]}
    pinned = {kind for kind, value in pipelines.load().items() if any("pin" in step for step in value["steps"])}
    assert pinned <= gated
    assert {name for gate in registry["gates"] for name in gate.get("providers", [])} == set(pipelines.load_providers())


def test_a_pinned_pipeline_without_a_gate_is_refused():
    registry = ci_matrix.load()
    broken = deepcopy(registry)
    for gate in broken["gates"]:
        gate["kinds"] = [kind for kind in gate["kinds"] if kind != "geometric-circle"]
    with pytest.raises(ValueError, match="geometric-circle"):
        ci_matrix.check(broken)


def test_pin_key_follows_the_exact_pins():
    registry = ci_matrix.load()
    gate, = (entry for entry in registry["gates"] if entry["gate"] == "geometry-research")
    descriptors = ci_matrix.descriptors()
    before = ci_matrix.pin_key(ci_matrix.gate_pins(gate, registry, descriptors))
    moved = deepcopy(descriptors)
    moved["mesh-path"]["steps"][0]["pin"]["revision"] = "0" * 40
    after = ci_matrix.pin_key(ci_matrix.gate_pins(gate, registry, moved))
    assert before != after
    rows = [row for row in ci_matrix.matrix(registry)["providers"]["include"] if row["gate"] == "geometry-research"]
    assert {row["pin_key"] for row in rows} == {before} and {row["os"] for row in rows} == {"ubuntu-latest", "windows-latest"}


def test_cache_keys_are_derived_from_pins():
    rows = ci_matrix.matrix()["providers"]["include"]
    cached = [row for row in rows if row["cache_key"]]
    assert cached and all(row["cache_key"].endswith("-pins-" + row["pin_key"]) for row in cached)


def test_a_restated_revision_in_a_workflow_is_refused(tmp_path, monkeypatch):
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "extra.yml").write_text("ref: " + "a" * 40 + "\n")
    monkeypatch.setattr(ci_matrix, "WORKFLOWS", workflows)
    with pytest.raises(ValueError, match="restates revision"):
        ci_matrix.check()


def test_the_workflow_is_one_shape_driven_by_the_registry():
    workflows = sorted(path.name for path in (ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflows == ["ci.yml"]
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for job in ("descriptors:", "kernel:", "providers:"):
        assert "\n  " + job in text
    assert "scripts/ci_matrix.py matrix --github-output" in text and "scripts/ci_matrix.py run" in text
    assert "fromJSON(needs.descriptors.outputs.providers)" in text and "fromJSON(needs.descriptors.outputs.kernel)" in text


def test_extra_pins_are_exact_and_used():
    registry = ci_matrix.load()
    for name, pin in registry["extra_pins"].items():
        assert len(pin["revision"]) == 40 and set(pin["revision"]) <= set("0123456789abcdef"), name
        assert pin["purpose"].strip()
    ci_matrix.check(registry)
    unused = deepcopy(registry)
    unused["extra_pins"]["orphan"] = {"repository": "x/y", "revision": "b" * 40, "purpose": "unused"}
    with pytest.raises(ValueError, match="bound by a gate"):
        ci_matrix.check(unused)


def test_run_expands_placeholders_without_a_shell(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ci_matrix.subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs)))
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    registry = {"surfaces": [{"gate": "probe", "summary": "probe", "os": ["ubuntu-latest"], "python": ["3.12"],
                              "timeout": 1, "run": [["{python}", "tool.py", "--out", "{output}", "--work", "{work}/x"]]}],
                "gates": []}
    ci_matrix.run("probe", registry=registry)
    (argv, kwargs), = calls
    assert argv[0] == sys.executable and Path(argv[3]) == ROOT / "results" / "probe"
    assert Path(argv[5]) == tmp_path / "ciw-probe" / "x" and kwargs["check"] is True and "shell" not in kwargs


def test_provider_descriptor_binds_the_julia_worker():
    value = pipelines.provider_descriptor("julia-model-worker")
    from ciw.model import worker
    assert worker.pinned_runtime() == value["pin"]
    binding = worker.provider_binding()
    assert all(value["pin"][key] == digest for key, digest in binding["pin"].items())
    broken = {"julia-model-worker": deepcopy(value)}
    broken["julia-model-worker"]["pin"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest_sha256"):
        pipelines.check_providers(broken)
    broken["julia-model-worker"] = deepcopy(value)
    broken["julia-model-worker"]["operations"]["core"].append("ciw.model.unknown.v1")
    with pytest.raises(ValueError, match="operations"):
        pipelines.check_providers(broken)
    assert json.loads((ROOT / "ci" / "gates.json").read_text())["gates"]  # registry stays readable JSON


def test_manifest_pins_include_historical_checkouts():
    registry = ci_matrix.load()
    gate, = (entry for entry in registry["gates"] if entry["gate"] == "adapters")
    pins = {(pin["role"], pin["revision"]) for pin in ci_matrix.gate_pins(gate, registry)}
    manifest = json.loads((ROOT / "src/ciw/adapter-runtimes.json").read_text())
    for role, pin in manifest.items():
        for historical in pin.get("historical", []):
            assert (role, historical["revision"]) in pins


def test_an_extra_pin_defined_in_package_code_must_equal_it():
    registry = ci_matrix.load()
    assert registry["extra_pins"]["scout"]["defined_by"].endswith(":VENDOR_REVISION")
    moved = deepcopy(registry)
    moved["extra_pins"]["scout"]["revision"] = "c" * 40
    with pytest.raises(ValueError, match="differs from its definition"):
        ci_matrix.check(moved)


def test_proved_heat_measures_resources_before_packages_and_keeps_recommends():
    rows = {row["gate"]: row for row in ci_matrix.matrix()["providers"]["include"]}
    assert rows["proved-heat"]["preflight"] is True and rows["proved-heat"]["apt_flags"] == ""
    assert rows["model-core"]["apt_flags"] == "--no-install-recommends"
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert text.index("--phase preflight") < text.index("apt-get install") < text.index("actions/cache/restore")
    kernel = {row["gate"]: row for row in ci_matrix.matrix()["kernel"]["include"]}
    assert kernel["installed-package"]["artifact_name"] == "ciw-python-wheel"
