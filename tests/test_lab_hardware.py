"""Operator captures, retained hardware runs and their integrity checks.

A fake hardware run stands in for one made on the capture host: the hardware
probe is replaced for the run only, and the raw bytes are a test fixture.
Nothing here is a measurement.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from ciw import cli
from ciw.lab import planner, registry, runner
from ciw.lab.evidence import finding
from ciw.lab.registry import Implementation, load_queue
from ciw.lab.report import report_identity

RUN_ID = "rtx2080-2026-10-01"
RAW = b"timestamp_ns,energy_mj\n0,0\n1000000,5\n"
CHECK = {"reference_kind": "analytic", "reference": "closed form", "observed": 0.0, "tolerance": 1e-9, "passed": True}
TASKS = {t["id"]: t for t in load_queue()["tasks"]}


def _acquisition(raw):
    return {"device": "test GPU behind a replaced probe", "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "acquired_at": "2026-10-01T00:00:00Z", "calibration": "not_applied: test fixture"}


def _fields(next_step="T118: record power from the same capture session; deferred research question: bind a signed "
                      "capture to the device"):
    return {"hypothesis": "h", "experiment": "e", "recommended_next_task": next_step}


def _measured(ctx):
    assert ctx.available("hardware:nvidia-gpu")
    ctx.artifact_text("operator-log.csv", RAW.decode())
    return {"fields": _fields(), "findings": [
        finding("GPU energy per trajectory", "physical", 5.0, {"acquisition": _acquisition(RAW)}, unit="mJ"),
        finding("Energy log parses", "numerical", 0.0, {"checks": [CHECK]})]}


def _fake(monkeypatch, task_id, run, requires=("hardware:nvidia-gpu",)):
    monkeypatch.setattr(runner, "load_implementations",
                        lambda: ({task_id: Implementation(task_id, run, requires=requires)}, {}))


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """A retained directory whose main run has T116 blocked, plus the output of a fake hardware run of T116."""
    retained, run = tmp_path / "lab", tmp_path / "hardware-run"
    monkeypatch.setattr(runner, "_probe_hardware", lambda name: False)
    runner.run_queue(retained, ["T116"])                       # the packaged task: blocked without a GPU
    assert runner.load_reports(retained)[0]["state"] == "blocked"
    with monkeypatch.context() as patch:
        patch.setattr(runner, "_probe_hardware", lambda name: name == "nvidia-gpu")
        _fake(patch, "T116", _measured)
        runner.run_queue(run, ["T116"])
    return retained, run


def _retain(retained, run, capsys):
    assert cli.main(["lab", "hardware", "retain", str(run), "--retained", str(retained), "--run-id", RUN_ID,
                     "--host", "RTX 2080 workstation (operator test)"]) == 0
    return json.loads(capsys.readouterr().out)


def test_a_retained_hardware_run_is_verified_and_shown_beside_the_main_run(lab, tmp_path, capsys):
    retained, run = lab
    summary = _retain(retained, run, capsys)
    assert summary["hardware_measured"] == 1 and summary["tasks"] == ["T116"]
    kept = retained / "hardware" / RUN_ID
    assert sorted(p.name for p in kept.iterdir()) == ["artifacts", "capture.json", "reports", "run-log.json"]
    capture = json.loads((kept / "capture.json").read_text(encoding="utf-8"))
    assert capture["schema"] == "ciw.lab-hardware-run.v1" and capture["run_id"] == RUN_ID
    assert capture["host"] == "RTX 2080 workstation (operator test)" and capture["tasks"] == ["T116"]
    assert capture["ciw"]["version"] and len(capture["ciw"]["package_digest"]) == 64
    assert str(tmp_path) not in (kept / "capture.json").read_text(encoding="utf-8")   # never a host path
    # Integrity only: nothing is recomputed, and the output says why.
    assert cli.main(["lab", "hardware", "verify", "--retained", str(retained)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] and result["verified"] == 1 and "cannot be recomputed in CI" in result["note"]
    assert cli.main(["lab", "verify", "--retained", str(retained), "--fresh", str(retained)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["hardware_runs"]["verified"] == 1 and "integrity only" in result["hardware_runs"]["note"]
    # Views outside the queue (`ciw lab unmeasured`, the planner) read valid runs through this; no queue task
    # does, so the clean-room reproduction never depends on lab/hardware.
    [found] = runner.hardware_runs(retained)
    assert found["run_id"] == RUN_ID and found["hardware_measured"] == 1 and set(found["reports"]) == {"T116"}
    # The queue keeps the main state and label and names the hardware run beside them.
    assert cli.main(["lab", "queue", "--retained", str(retained), "--state", "blocked"]) == 0
    [line] = capsys.readouterr().out.splitlines()
    assert line.startswith("T116  blocked") and "not_established" in line
    assert f"[hardware run {RUN_ID} (2" in line and "completed, numerically_verified, 1 hardware-measured finding]" in line
    row = next(r for r in runner.queue_view(retained) if r["id"] == "T116")
    assert row["state"] == "blocked" and row["hardware_run"]["physical_validation_status"] == "hardware_measured"
    # The planner stops proposing T116 as blocked on hardware and ranks it from the hardware run.
    plan = planner.next_tasks(retained, limit=200)
    assert "T116" not in [r["task_id"] for r in plan["still_blocked"]]
    rows = [r for r in plan["next"] + plan["research"] if r["task_id"] == "T116"]
    assert {r["kind"] for r in rows} == {"follow_up", "research"}
    assert all(r["hardware_run"]["run_id"] == RUN_ID and r["state"] == "blocked" for r in rows)
    assert plan["hardware_runs"] == [RUN_ID] and plan["hardware_run_problems"] == []


def test_the_unmeasured_view_lists_hardware_runs_beside_the_clean_room_count(lab, capsys):
    """R01: hardware-measured counts from retained hardware runs are aggregated outside the queue, never merged."""
    retained, run = lab
    runner.run_queue(retained, ["T167"])            # the clean-room ledger reads the main run's reports only
    _retain(retained, run, capsys)
    assert cli.main(["lab", "unmeasured", "--retained", str(retained)]) == 0
    view = json.loads(capsys.readouterr().out)
    assert view["schema"] == "ciw.lab-unmeasured.v1" and "never added" in view["note"]
    main = view["main_run"]
    assert main["hardware_measured"] == 0 and main["t167"]["hardware_measured_findings"] == 0
    assert view["hardware_runs"] == [{"run_id": RUN_ID, "date": view["hardware_runs"][0]["date"],
                                      "host": "RTX 2080 workstation (operator test)", "tasks": ["T116"],
                                      "hardware_measured": 1}]
    row = next(r for r in view["tasks"] if r["task_id"] == "T116")
    assert row["main"]["state"] == "blocked" and row["main"]["hardware_measured"] == 0
    assert row["main"]["not_established"]         # the main run's open physical claims stay listed
    [measured] = row["hardware_runs"]
    assert measured["run_id"] == RUN_ID and measured["state"] == "completed" and measured["hardware_measured"] == 1
    assert measured["measured_claims"] == ["GPU energy per trajectory"]
    # The clean-room ledger is unchanged by the hardware run.
    assert json.loads((retained / "artifacts" / "T167" / "unmeasured.json").read_text())["hardware_measured_findings"] == []
    # A tampered run is left out and named.
    (retained / "hardware" / RUN_ID / "artifacts" / "T116" / "operator-log.csv").write_bytes(b"edited\n")
    assert cli.main(["lab", "unmeasured", "--retained", str(retained)]) == 0
    view = json.loads(capsys.readouterr().out)
    assert view["hardware_runs"] == [] and view["hardware_run_problems"]
    assert "hardware_runs" not in next(r for r in view["tasks"] if r["task_id"] == "T116")


def test_host_paths_in_reports_never_reach_retained_hardware_runs(tmp_path, monkeypatch):
    """A report quoting an operator's path (an OSError on a missing log) is refused by retain."""
    secret = tmp_path / "private-captures" / "rtx" / "log.json"

    def quotes_the_path(ctx):
        assert ctx.available("hardware:nvidia-gpu")
        try:
            secret.read_bytes()
        except OSError as exc:
            reason = f"the operator log is unreadable or invalid: {exc}"
        return {"state": "blocked", "fields": dict(_fields("none"), unresolved_assumptions=[reason]), "findings": []}

    monkeypatch.setattr(runner, "_probe_hardware", lambda name: name == "nvidia-gpu")
    _fake(monkeypatch, "T116", quotes_the_path)
    run = tmp_path / "run"
    runner.run_queue(run, ["T116"])
    # The OSError message shows the path raw or, on Windows, as its repr (doubled backslashes).
    quoted = json.dumps(json.loads((run / "reports" / "T116.json").read_text(encoding="utf-8")))
    assert any(json.dumps(form)[1:-1] in quoted for form in (str(secret), repr(str(secret))[1:-1]))
    with pytest.raises(ValueError, match="holds a host path"):
        runner.retain_hardware_run(run, tmp_path / "lab", RUN_ID, "RTX 2080 workstation")
    assert not (tmp_path / "lab" / "hardware").exists()
    # Quantities that only look like paths are not refused.
    def screened(text):
        return runner._report_host_paths({"T116": {"u": text}})
    assert screened("~1e-6 at 0.00151 /mm; ~2e-10/s; m/s; ratio 3/2; RTX 2080 / Linux; http://example.org/a") == []
    for text in ("C:\\Users\\op\\log.json", "see ~/captures/log.json", "'/srv/x/y'", "x=/a/b", "\\\\rig\\share"):
        assert screened(text) == ["reports/T116.json.u holds a host path"], text


def _reseal(path, edit, capture=None):
    report = json.loads(path.read_text(encoding="utf-8"))
    edit(report)
    report["report_id"] = report_identity(report)
    path.write_text(runner.dumps(report), encoding="utf-8")
    if capture is not None:  # keep capture.json consistent so only the edited property fails
        data = json.loads(capture.read_text(encoding="utf-8"))
        data["reports"][report["task_id"]] = report["report_id"]
        capture.write_text(runner.dumps(data), encoding="utf-8")


def _capture_edit(key, value):
    def edit(kept):
        data = json.loads((kept / "capture.json").read_text(encoding="utf-8"))
        data[key] = value
        (kept / "capture.json").write_text(runner.dumps(data), encoding="utf-8")
    return edit


def _physical(report):
    return next(f for f in report["findings"] if f["domain"] == "physical")


TAMPERING = {
    "artifact bytes": (lambda kept: (kept / "artifacts" / "T116" / "operator-log.csv").write_bytes(RAW + b"2,9\n"),
                       "differs from its recorded digest"),
    "raw digest resealed": (lambda kept: _reseal(kept / "reports" / "T116.json", lambda r: _physical(r)["basis"][
        "acquisition"].__setitem__("raw_sha256", "d" * 64), kept / "capture.json"), "not a retained artifact of T116"),
    "probe record resealed": (lambda kept: _reseal(kept / "reports" / "T116.json", lambda r: r[
        "provider_runtime_identity"].pop("requirement_probes"), kept / "capture.json"),
        "no hardware probe succeeded in the task itself"),
    "own probe resealed": (lambda kept: _reseal(kept / "reports" / "T116.json", lambda r: r[
        "provider_runtime_identity"].pop("hardware_probed_in_task"), kept / "capture.json"),
        "no hardware probe succeeded in the task itself"),
    "host path in a report": (lambda kept: _reseal(kept / "reports" / "T116.json", lambda r: r[
        "unresolved_assumptions"].append("log read from /home/operator/rig/log.json"), kept / "capture.json"),
        "reports/T116.json.unresolved_assumptions[1] holds a host path"),
    "report edited": (lambda kept: _reseal(kept / "reports" / "T116.json", lambda r: r.__setitem__("title", "x")),
                      "report identities differ"),
    "identity broken": (lambda kept: (kept / "reports" / "T116.json").write_text(
        (kept / "reports" / "T116.json").read_text(encoding="utf-8").replace('"h"', '"H"'), encoding="utf-8"),
        "refused"),
    "unrecorded file": (lambda kept: (kept / "artifacts" / "T116" / "extra.csv").write_bytes(b"1\n"),
                        "not recorded by any report"),
    "count edited": (_capture_edit("hardware_measured", 2), "counts 2 hardware-measured findings"),
    "host path": (_capture_edit("host", "/home/operator/rig"), "holds a host path"),
    "renamed": (lambda kept: kept.rename(kept.with_name("other-run")), "names run"),
}


@pytest.mark.parametrize("case", sorted(TAMPERING))
def test_a_tampered_hardware_run_fails_verification_and_leaves_every_view(lab, capsys, case):
    retained, run = lab
    _retain(retained, run, capsys)
    tamper, message = TAMPERING[case]
    tamper(retained / "hardware" / RUN_ID)
    assert cli.main(["lab", "hardware", "verify", "--retained", str(retained)]) == 3
    result = json.loads(capsys.readouterr().out)
    assert not result["passed"] and any(message in problem for problem in result["problems"]), result["problems"]
    assert cli.main(["lab", "verify", "--retained", str(retained), "--fresh", str(retained)]) == 3
    capsys.readouterr()
    assert runner.hardware_runs(retained) == []
    assert next(r for r in runner.queue_view(retained) if r["id"] == "T116")["hardware_run"] is None
    plan = planner.next_tasks(retained, limit=200)
    assert "T116" in [r["task_id"] for r in plan["still_blocked"]] and plan["hardware_run_problems"]


def test_retain_refuses_what_is_not_one_hardware_run(lab, tmp_path, monkeypatch, capsys):
    retained, run = lab
    host = "RTX 2080 workstation"
    with pytest.raises(ValueError, match="kebab-case"):
        runner.retain_hardware_run(run, retained, "RTX 2080", host)
    with pytest.raises(ValueError, match="host path"):
        runner.retain_hardware_run(run, retained, RUN_ID, "rig at /home/operator")
    # A run of a computational task is not a hardware run.
    plain = tmp_path / "plain"
    runner.run_queue(plain, ["T156"])
    with pytest.raises(ValueError, match="not a hardware run"):
        runner.retain_hardware_run(plain, retained, RUN_ID, host)
    # Reports from more than the recorded run, or no run record at all, are refused.
    mixed = tmp_path / "mixed"
    shutil.copytree(run, mixed)
    shutil.copy2(plain / "reports" / "T156.json", mixed / "reports" / "T156.json")
    with pytest.raises(ValueError, match="retain the output directory of one"):
        runner.retain_hardware_run(mixed, retained, RUN_ID, host)
    (mixed / "run-record.json").unlink()
    with pytest.raises(ValueError, match="run-record.json"):
        runner.retain_hardware_run(mixed, retained, RUN_ID, host)
    # A hardware-measured finding must cite bytes its task retained.
    broken = tmp_path / "broken"
    shutil.copytree(run, broken)
    (broken / "artifacts" / "T116" / "operator-log.csv").write_bytes(b"edited\n")
    with pytest.raises(ValueError, match="differs from its recorded digest"):
        runner.retain_hardware_run(broken, retained, RUN_ID, host)
    assert not (retained / "hardware").exists()
    _retain(retained, run, capsys)
    assert cli.main(["lab", "hardware", "retain", str(run), "--retained", str(retained), "--run-id", RUN_ID,
                     "--host", host]) == 2
    assert "already retained" in capsys.readouterr().err


CMM = b"pair,distance_mm\nA-B,1.0\n"
INVENTED = b"pair,distance_mm\nA-B,2.0\n"


def _captured(ctx):
    """Claims a physical label for capture bytes: refused, since no probe of the CMM on this host backs them."""
    raw = ctx.capture("cmm")
    return {"fields": _fields("Compare the retained marker distances with the model"), "findings": [
        finding("Marker-pair distance", "physical", 1.0, {"acquisition": _acquisition(raw)}, unit="mm"),
        finding("Manifest parses", "numerical", 0.0, {"checks": [CHECK]})]}


def _analysed(ctx):
    """Retains the capture, computes from it and records the physical claim as not established."""
    raw = ctx.capture("cmm")
    ctx.artifact_text("invented.csv", INVENTED.decode())
    rows = raw.decode().splitlines()[1:]
    return {"fields": _fields("Compare the retained marker distances with the model"), "findings": [
        finding("Marker-pair distance", "physical", None, {}),
        finding("Manifest parses", "numerical", 0.0, {"checks": [dict(CHECK, observed=float(len(rows) - 1))]})]}


def test_an_operator_capture_alone_never_makes_a_physical_label(tmp_path, monkeypatch):
    """R02: bytes bound with --capture are retained and unauthenticated; they never make a physical label."""
    monkeypatch.setattr(runner, "_probe_hardware", lambda name: False)
    capture = tmp_path / "cmm-export.CSV"
    capture.write_bytes(CMM)
    digest = hashlib.sha256(CMM).hexdigest()
    report = runner.run_task(TASKS["T138"], Implementation("T138", _captured),
                             runner.Context(tmp_path / "run", captures={"cmm": capture}), {})
    assert report["state"] == "blocked" and "no hardware probe succeeded" in report["experiment"]
    assert report["physical_validation_status"]["status"] == "not_established"
    assert report["generated_artifacts"][0] == {"path": "artifacts/T138/capture-cmm.csv", "sha256": digest,
                                                "bytes": len(CMM)}
    # Computing from the retained bytes, with the physical claim recorded as not established, completes.
    report = runner.run_task(TASKS["T138"], Implementation("T138", _analysed),
                             runner.Context(tmp_path / "analysed", captures={"cmm": capture}), {})
    assert report["state"] == "completed" and report["physical_validation_status"]["status"] == "not_established"
    assert [f["evidence_status"] for f in report["findings"]] == ["not_established", "numerically_verified"]
    identity = report["provider_runtime_identity"]
    assert identity["requirement_probes"] == {"capture:cmm": True} and identity["operator_captures"] == {"cmm": digest}
    assert "hardware_probed_in_task" not in identity


def test_a_probe_of_another_instrument_does_not_back_a_capture(tmp_path, monkeypatch):
    """A GPU probe on this host says nothing about bytes from a CMM, or about other bytes of the task."""
    monkeypatch.setattr(runner, "_probe_hardware", lambda name: name == "nvidia-gpu")
    capture = tmp_path / "cmm.csv"
    capture.write_bytes(CMM)

    def gpu_then_capture(ctx):
        assert ctx.available("hardware:nvidia-gpu")
        return _captured(ctx)

    report = runner.run_task(TASKS["T138"], Implementation("T138", gpu_then_capture),
                             runner.Context(tmp_path / "run", captures={"cmm": capture}), {})
    assert report["state"] == "blocked" and "operator capture cmm" in report["experiment"]
    assert "no probe of its instrument" in report["experiment"]


@pytest.mark.parametrize("through", ["--capture", "variable"])
def test_the_energy_fixture_is_never_hardware_measured_without_a_gpu_probe(tmp_path, monkeypatch, capsys, through):
    """A synthetic energy log bound as energy-log stays unmeasured unless hardware:nvidia-gpu answered here."""
    fixture = runner.repository_path("examples", "energy-accuracy", "baseline.json")
    if fixture is None or not fixture.is_file():
        pytest.skip("examples/ is not available")
    probed = []

    def reads_the_log(ctx):
        raw = ctx.capture("energy-log")
        for name in probed:
            ctx.available(f"hardware:{name}")
        return {"fields": _fields("none"), "findings": [
            finding("GPU energy per accepted solve", "physical", 1.0, {"acquisition": _acquisition(raw)}, unit="J")]}

    _fake(monkeypatch, "T119", reads_the_log, requires=())
    monkeypatch.delenv("CIW_LAB_ENERGY_LOG", raising=False)
    for hardware, refusal in (((), "no hardware probe succeeded"), (("rapl",), "hardware:nvidia-gpu")):
        probed[:] = hardware
        monkeypatch.setattr(runner, "_probe_hardware", lambda name: name in hardware)
        out = tmp_path / f"run-{len(hardware)}"
        argv = ["lab", "run", "T119", "--output-dir", str(out)]
        if through == "--capture":
            argv += ["--capture", f"energy-log={fixture}"]
        else:
            monkeypatch.setenv("CIW_LAB_ENERGY_LOG", str(fixture))
        assert cli.main(argv) == 0
        capsys.readouterr()
        report = json.loads((out / "reports" / "T119.json").read_text(encoding="utf-8"))
        assert report["state"] == "blocked" and refusal in report["experiment"], report["experiment"]
        assert report["physical_validation_status"]["status"] == "not_established"
        assert not [f for f in report["findings"] if f["evidence_status"] == "hardware_measured"]


def test_unbound_or_unreadable_captures_are_unavailable(tmp_path):
    ctx = runner.Context(tmp_path / "run", captures={"folder": tmp_path, "gone": tmp_path / "missing.csv"})
    ctx.begin("T138")
    assert not ctx.available("capture:folder") and not ctx.available("capture:gone")
    assert not ctx.available("capture:unbound")
    with pytest.raises(ValueError, match="--capture unbound=PATH"):
        ctx.capture("unbound")
    with pytest.raises(ValueError, match="kebab-case"):
        runner.Context(tmp_path, captures={"CMM Export": tmp_path})
    # A hard capture requirement blocks the task with the role named.
    report = runner.run_task(TASKS["T138"], Implementation("T138", _captured, requires=("capture:cmm",)),
                             runner.Context(tmp_path / "blocked"), {})
    assert report["state"] == "blocked" and "capture:cmm" in report["experiment"]


def test_capture_requirements_register(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    registry.task("T138", requires=("capture:cmm",))(lambda ctx: {})
    with pytest.raises(ValueError, match="capture:<name>"):
        registry.task("T139", requires=("capture:",))


CAPTURE_RUN = "cmm-2026-10-02"


def _capture_run(tmp_path, monkeypatch, name, implementations, hardware=(), gate=True):
    """One `ciw lab run` of fake tasks with the CMM capture bound; ``gate=False`` forges what verification must catch."""
    capture = tmp_path / "private" / "cmm.csv"
    capture.parent.mkdir(exist_ok=True)
    capture.write_bytes(CMM)
    out = tmp_path / name
    with monkeypatch.context() as patch:
        patch.setattr(runner, "load_implementations", lambda: (
            {task_id: Implementation(task_id, run) for task_id, run in implementations.items()}, {}))
        patch.setattr(runner, "_probe_hardware", lambda probe: probe in hardware)
        if not gate:  # a CIW without the physical gate
            patch.setattr(runner, "_gate_physical", lambda findings, ctx: None)
        runner.run_queue(out, sorted(implementations), captures={"cmm": capture})
    return out


def test_a_capture_run_is_recorded_by_digest_and_retained(tmp_path, monkeypatch, capsys):
    capture = tmp_path / "private" / "cmm.csv"
    capture.parent.mkdir()
    capture.write_bytes(CMM)
    _fake(monkeypatch, "T138", _analysed, requires=("capture:cmm",))
    out = tmp_path / "run"
    assert cli.main(["lab", "run", "T138", "--output-dir", str(out), "--capture", f"cmm={capture}"]) == 0
    assert json.loads(capsys.readouterr().out)["states"]["completed"] == 1
    text = (out / "run-record.json").read_text(encoding="utf-8")
    record = json.loads(text)
    assert record["schema"] == "ciw.lab-run-record.v1" and record["tasks"] == ["T138"]
    assert record["captures"]["cmm"]["sha256"] == hashlib.sha256(capture.read_bytes()).hexdigest()
    assert str(tmp_path) not in text
    # The raw bytes are retained; the run measured nothing, since no probe of the CMM on this host backs them.
    summary = runner.retain_hardware_run(out, tmp_path / "lab", CAPTURE_RUN, "Metrology bench (test)")
    assert summary["hardware_measured"] == 0
    capture_record = json.loads((tmp_path / "lab" / "hardware" / CAPTURE_RUN / "capture.json").read_text())
    assert capture_record["captures"]["cmm"]["artifacts"] == ["artifacts/T138/capture-cmm.csv"]
    assert runner.verify_hardware_runs(tmp_path / "lab")["passed"]
    assert cli.main(["lab", "run", "T138", "--output-dir", str(out), "--capture", "cmm"]) == 2
    assert "Capture bindings use ROLE=PATH" in capsys.readouterr().err


def _gpu_capture(ctx):
    assert ctx.available("hardware:nvidia-gpu")
    return _captured(ctx)


def _other_bytes(ctx):
    ctx.capture("cmm")
    ctx.artifact_text("invented.csv", INVENTED.decode())
    return {"fields": _fields("none"), "findings": [
        finding("Marker-pair distance", "physical", 2.0, {"acquisition": _acquisition(INVENTED)}, unit="mm")]}


def _probes_in_memo(ctx):
    ctx.memo("gpu", lambda: ctx.available("hardware:nvidia-gpu"))
    return {"fields": _fields("none"), "findings": [finding("Probe recorded", "numerical", 0.0, {"checks": [CHECK]})]}


def _replayed(ctx):
    assert ctx.memo("gpu", lambda: None)            # T137 probed; this task only replays the outcome
    ctx.artifact_text("operator-log.csv", RAW.decode())
    return {"fields": _fields("none"), "findings": [
        finding("GPU energy per trajectory", "physical", 5.0, {"acquisition": _acquisition(RAW)}, unit="mJ")]}


NO_PROBE = "no hardware probe succeeded in the task itself"
FORGED = {  # a report written without the physical gate: task, hardware on the host, verification problem
    "capture only": (_captured, (), NO_PROBE),
    "resealed to another artifact": (_other_bytes, (), NO_PROBE),
    "probe of another instrument": (_gpu_capture, ("nvidia-gpu",), "cites the operator capture cmm"),
    "probe replayed from a memo": (_replayed, ("nvidia-gpu",), NO_PROBE),
}


@pytest.mark.parametrize("case", sorted(FORGED))
def test_verification_mirrors_the_physical_gate(tmp_path, monkeypatch, case):
    """What the run-time gate refuses, retain refuses and verification reports, whatever the report claims."""
    run, hardware, problem = FORGED[case]
    tasks = {"T137": _probes_in_memo, "T138": run}
    # The gate refuses it at run time ...
    honest = _capture_run(tmp_path, monkeypatch, "honest", tasks, hardware)
    assert json.loads((honest / "reports" / "T138.json").read_text())["state"] == "blocked"
    # ... so a report that carries it was forged, and retain refuses the run.
    forged = _capture_run(tmp_path, monkeypatch, "forged", tasks, hardware, gate=False)
    report = json.loads((forged / "reports" / "T138.json").read_text(encoding="utf-8"))
    assert report["physical_validation_status"]["status"] == "hardware_measured"
    with pytest.raises(ValueError, match=problem):
        runner.retain_hardware_run(forged, tmp_path / "refused", CAPTURE_RUN, "Metrology bench (test)")
    # Swapped into a retained capture run with capture.json kept consistent, verification reports it.
    retained = tmp_path / "lab"
    runner.retain_hardware_run(_capture_run(tmp_path, monkeypatch, "retained", {"T138": _analysed}), retained,
                               CAPTURE_RUN, "Metrology bench (test)")
    kept = retained / "hardware" / CAPTURE_RUN
    shutil.rmtree(kept / "artifacts" / "T138")
    shutil.copytree(forged / "artifacts" / "T138", kept / "artifacts" / "T138")
    (kept / "reports" / "T138.json").write_text(runner.dumps(report), encoding="utf-8")
    capture = json.loads((kept / "capture.json").read_text(encoding="utf-8"))
    capture["reports"]["T138"], capture["hardware_measured"] = report["report_id"], 1
    (kept / "capture.json").write_text(runner.dumps(capture), encoding="utf-8")
    result = runner.verify_hardware_runs(retained)
    assert not result["passed"] and any(problem in p for p in result["problems"]), result["problems"]
    assert runner.hardware_runs(retained) == []


def test_operator_capture_variables_are_capture_roles(tmp_path, monkeypatch):
    log, other = tmp_path / "log.json", tmp_path / "other.json"
    log.write_text("{}")
    other.write_text("[]")
    seen = []

    def reads_variable(ctx):
        seen.append(os.environ.get("CIW_LAB_ENERGY_LOG"))
        return {"state": "blocked", "findings": []}
    _fake(monkeypatch, "T116", reads_variable, requires=())
    monkeypatch.delenv("CIW_LAB_ENERGY_LOG", raising=False)
    runner.run_queue(tmp_path / "a", ["T116"], captures={"energy-log": log})
    assert seen == [str(log)] and "CIW_LAB_ENERGY_LOG" not in os.environ   # set for the run only
    record = json.loads((tmp_path / "a" / "run-record.json").read_text())
    assert record["captures"]["energy-log"]["variable"] == "CIW_LAB_ENERGY_LOG"
    # The variable an operator sets is recorded as the role's binding; a conflicting binding is refused.
    monkeypatch.setenv("CIW_LAB_ENERGY_LOG", str(other))
    monkeypatch.setenv("CIW_LAB_NVIDIA_SMI_UTC_OFFSET", "+00:00")
    runner.run_queue(tmp_path / "b", ["T116"])
    record = json.loads((tmp_path / "b" / "run-record.json").read_text())
    assert record["captures"]["energy-log"]["sha256"] == hashlib.sha256(b"[]").hexdigest()
    assert record["settings"] == {"CIW_LAB_NVIDIA_SMI_UTC_OFFSET": "+00:00"}
    with pytest.raises(ValueError, match="bind one"):
        runner.run_queue(tmp_path / "c", ["T116"], captures={"energy-log": log})
    # The roles cover exactly the variables the energy tasks read and the clean room removes.
    from ciw.lab import energy_gpu_telemetry as telemetry
    assert set(runner.OPERATOR_CAPTURE_VARIABLES.values()) | set(runner.OPERATOR_SETTINGS) == {
        telemetry.LOG_ENV, telemetry.SMI_ENV, telemetry.SMI_OFFSET_ENV, telemetry.RAPL_ENV}


def _script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    if not path.is_file():
        pytest.skip("scripts/ is not part of the clean-room copy of the tests")
    spec = importlib.util.spec_from_file_location(f"lab_hardware_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refresh_never_touches_retained_hardware_runs(tmp_path, monkeypatch):
    refresh = _script("refresh_lab")
    monkeypatch.setattr(refresh, "ROOT", tmp_path)
    kept = tmp_path / "lab" / "hardware" / RUN_ID / "capture.json"
    kept.parent.mkdir(parents=True)
    kept.write_text('{"schema": "ciw.lab-hardware-run.v1"}')
    (tmp_path / "lab" / "reports").mkdir()
    (tmp_path / "lab" / "reports" / "T001.json").write_text("old")
    run = tmp_path / "run"
    (run / "reports").mkdir(parents=True)
    (run / "artifacts").mkdir()
    (run / "reports" / "T001.json").write_text(json.dumps({"task_id": "T001"}))
    # A run of check_lab.py also ran T077's telemetry session on the bound telemetry stack.
    (run / "reports" / "T077.json").write_text(json.dumps({"task_id": "T077", "provider_runtime_identity": {
        "telemetry-stack": {"gsie": {"state": "ready"}}, "executed_runtimes": {"gsie": {}}}}))
    for name in ("queue-state.json", "REPORTS.md", "index.html"):
        (run / name).write_text("")
    (run / "gate.json").write_text(json.dumps({"schema": "ciw.lab-clean-room-gate.v1", "python": "3.12.3",
                                               "providers": [f"{role}=/p" for role in refresh.REQUIRED_PROVIDERS]}))
    monkeypatch.setattr(sys, "argv", ["refresh_lab.py", "--from-run", str(run)])
    assert refresh.main() == 0
    assert (tmp_path / "lab" / "reports" / "T001.json").read_text() == json.dumps({"task_id": "T001"})
    assert kept.read_text() == '{"schema": "ciw.lab-hardware-run.v1"}'
    assert "hardware" in refresh.PRESERVED and "hardware" not in refresh.RETAINED


def test_the_clean_room_verifies_retained_hardware_runs_without_a_comparison(tmp_path, monkeypatch):
    from types import SimpleNamespace
    reproduce = _script("reproduce_lab")
    commands = []

    def run(command, **kwargs):
        command = [str(part) for part in command]
        commands.append(command)
        if command[1:4] == ["-m", "pip", "wheel"]:
            dist = Path(command[command.index("--wheel-dir") + 1])
            dist.mkdir(parents=True)
            (dist / "ciw-0-py3-none-any.whl").write_bytes(b"wheel")

    retained = tmp_path / "retained"
    (retained / "hardware" / RUN_ID).mkdir(parents=True)
    (retained / "hardware" / "README.md").write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(reproduce, "run", run)
    monkeypatch.setattr(reproduce, "venv", SimpleNamespace(EnvBuilder=lambda **kwargs: SimpleNamespace(
        create=lambda path: None)))
    monkeypatch.setattr(reproduce, "subprocess", SimpleNamespace(
        run=lambda *args, **kwargs: SimpleNamespace(stdout=str(tmp_path / "site" / "ciw" / "__init__.py"))))
    monkeypatch.setattr(sys, "argv", ["reproduce_lab.py", "--no-compare", "--output-dir", "out", "--retained",
                                      str(retained), "--temporary-root", str(tmp_path)])
    assert reproduce.main() == 0
    assert ["-m", "ciw", "lab", "hardware", "verify", "--retained", str(retained.resolve())] in [
        command[1:] for command in commands]
    record = json.loads((tmp_path / "out" / "gate.json").read_text(encoding="utf-8"))
    assert record["hardware_runs_verified_for_integrity"] == [RUN_ID]


def test_lab_docs_describe_hardware_runs():
    root = Path(__file__).resolve().parents[1]
    if not (root / "docs" / "LAB.md").is_file() or not (root / "lab" / "hardware" / "README.md").is_file():
        pytest.skip("docs/ or the retained lab/ directory is not available (the clean room copies no lab/)")

    def text(*parts):
        return " ".join((root.joinpath(*parts)).read_text(encoding="utf-8").split())
    lab, authoring, kept = text("docs", "LAB.md"), text("docs", "lab", "AUTHORING.md"), text("lab", "hardware", "README.md")
    for phrase in ("ciw lab hardware retain", "ciw lab hardware verify", "lab/hardware/<run-id>/", "--capture ROLE=PATH",
                   "cannot be recomputed in CI", "ciw lab unmeasured --retained lab", "never added"):
        assert phrase in lab, phrase
    assert "ctx.capture(" in authoring and "capture:<role>" in authoring
    # Captures are unauthenticated and bound to no device; only a probe of the instrument in the task measures.
    for doc in (lab, authoring, kept):
        assert "retained and not authenticated" in doc and "exactly as for operator GPU and RAPL" not in doc
        assert "or an operator capture retained in that task" not in doc
    assert "acquisition gate binds a capture to the analysing host's device" not in lab


def test_no_queue_task_reads_retained_hardware_runs():
    """R01: the clean-room reproduction must not depend on lab/hardware, so only views outside the queue read it."""
    import re
    readers = re.compile(r"\b(?:hardware_runs|latest_hardware|verify_hardware_runs|unmeasured_view|queue_view)\s*\(|"
                         r"HARDWARE_DIRECTORY")
    # Views over the retained directory, not queue tasks: they may show hardware runs beside the main run.
    outside = {"runner.py", "planner.py", "mcp_server.py", "dashboard.py"}
    found = sorted(path.name for path in (runner.PACKAGE_ROOT / "lab").glob("*.py")
                   if path.name not in outside and readers.search(path.read_text(encoding="utf-8")))
    assert found == [], f"queue modules read retained hardware runs: {found}"


def test_retain_refuses_a_malformed_run_and_removes_a_copy_that_fails_verification(lab, tmp_path, monkeypatch):
    """Every refusal of retain_hardware_run leaves lab/hardware untouched; a failed verification removes the copy."""
    retained, run = lab
    host = "RTX 2080 workstation"

    def variant(name, edit):
        copy = tmp_path / name
        shutil.copytree(run, copy)
        edit(copy)
        return copy

    def edit_record(change):
        def edit(copy):
            record = json.loads((copy / "run-record.json").read_text(encoding="utf-8"))
            change(record)
            (copy / "run-record.json").write_text(json.dumps(record), encoding="utf-8")
        return edit

    wrong_schema = variant("schema", edit_record(lambda record: record.update(schema="ciw.lab-run-record.v0")))
    with pytest.raises(ValueError, match="is not ciw.lab-run-record"):
        runner.retain_hardware_run(wrong_schema, retained, RUN_ID, host)
    invalid = variant("invalid", lambda copy: (copy / "reports" / "T116.json").write_text("{}", encoding="utf-8"))
    with pytest.raises(ValueError, match="Refusing the run: reports/T116.json refused"):
        runner.retain_hardware_run(invalid, retained, RUN_ID, host)
    # Bindings reach capture.json, so a host path in them is refused before anything is copied.
    leaked = variant("leaked", edit_record(lambda record: record.update(providers={"scr": "/home/operator/scr"})))
    with pytest.raises(ValueError, match="Refusing the run: .*host path"):
        runner.retain_hardware_run(leaked, retained, RUN_ID, host)
    assert not (retained / "hardware").exists()
    # A copy that fails its own verification is removed again.
    monkeypatch.setattr(runner, "_inspect_hardware_run", lambda path: ({"problems": ["reports/T116.json edited"]}, None))
    with pytest.raises(ValueError, match="The retained copy fails verification: reports/T116.json edited"):
        runner.retain_hardware_run(run, retained, RUN_ID, host)
    assert not (retained / "hardware" / RUN_ID).exists()


def test_capture_json_keeps_the_run_records_bindings_and_settings(lab, tmp_path):
    retained, run = lab
    bound = tmp_path / "bound"
    shutil.copytree(run, bound)
    record = json.loads((bound / "run-record.json").read_text(encoding="utf-8"))
    record["providers"] = {"scr": {"role": "scr", "head": "a" * 40}}
    record["settings"] = {"OPENBLAS_NUM_THREADS": "1"}
    (bound / "run-record.json").write_text(json.dumps(record), encoding="utf-8")
    runner.retain_hardware_run(bound, retained, RUN_ID, "RTX 2080 workstation")
    capture = json.loads((retained / "hardware" / RUN_ID / "capture.json").read_text(encoding="utf-8"))
    assert capture["providers"] == record["providers"] and capture["settings"] == record["settings"]
