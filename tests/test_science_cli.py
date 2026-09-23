"""`ciw science` terminal commands and the quick synthetic bench."""
import json
from pathlib import Path

import pytest

from ciw.cli import main as ciw_main
from ciw.science.cli import main
from ciw.science.ledger import Ledger

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "science"


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr().out
    try:
        return code, json.loads(out)
    except json.JSONDecodeError:
        return code, out


def test_spec_units_frames_and_solvers(capsys):
    code, result = run(capsys, "spec", "validate", str(EXAMPLES / "experiments" / "cylinder-chord.json"))
    assert code == 0 and result["valid"]
    code, text = run(capsys, "spec", "latex", str(EXAMPLES / "experiments" / "cylinder-chord.json"))
    assert code == 0 and r"\Gamma" in text
    code, result = run(capsys, "units", "25.4", "mm", "inch")
    assert result["value"] == pytest.approx(1.0)
    code, result = run(capsys, "units", "1", "px", "m")
    assert code == 2 and result["refused"]["code"] == "dimension_mismatch"
    code, result = run(capsys, "frames", str(EXAMPLES / "frames" / "coupon-cell.json"), "coupon", "camera",
                       "--at", "1000", "--clock", "cell-ptp", "--point", "0.05", "0", "0.02")
    assert code == 0 and result["frame"] == "camera" and len(result["chain"]) == 2
    code, result = run(capsys, "frames", str(EXAMPLES / "frames" / "coupon-cell.json"), "coupon", "camera",
                       "--at", "9000", "--clock", "cell-ptp")
    assert code == 2 and result["refused"]["code"] == "transform_stale"
    code, result = run(capsys, "solvers")
    assert any(item["solver_id"] == "geometry.mesh-heat.v1" for item in result)


def test_run_status_replay_report_and_probe(tmp_path, capsys):
    ledger = tmp_path / "ledger"
    code, result = run(capsys, "run", str(EXAMPLES / "experiments" / "cylinder-winding.json"), "--ledger", str(ledger),
                       "--frames", str(EXAMPLES / "frames" / "coupon-cell.json"))
    assert code == 0 and result[0]["verifications"]["passed"] == 5
    code, result = run(capsys, "replay", "cylinder-winding-classes", "--ledger", str(ledger))
    assert result["counts"]["reproduced"] == 5
    code, result = run(capsys, "status", "cylinder-winding-classes", "--ledger", str(ledger))
    assert "reproduced by 5 replays" in result["numerical_stability"]
    code, result = run(capsys, "ledger", "verify", str(ledger))
    assert code == 0 and result["ok"]
    code, result = run(capsys, "ledger", "audit", str(ledger))
    assert code == 0 and result == []
    code, result = run(capsys, "ledger", "probe", str(ledger))
    assert code == 0 and all(item["detected"] for item in result)
    report = tmp_path / "report.md"
    code, _ = run(capsys, "report", "--ledger", str(ledger), "--output", str(report), "--retain")
    assert code == 0 and "cylinder-winding-classes" in report.read_text(encoding="utf-8")
    assert next(Ledger.open(ledger).entries("report"))["body"]["format"] == "markdown"
    lines = (ledger / "ledger.jsonl").read_bytes().splitlines(keepends=True)
    (ledger / "ledger.jsonl").write_bytes(b"".join(lines[:-2] + lines[-1:]))
    code, result = run(capsys, "ledger", "verify", str(ledger))
    assert code == 2 and not result["ok"]
    code, result = run(capsys, "status", "cylinder-winding-classes", "--ledger", str(ledger))
    assert code == 2 and result["refused"]["code"] == "ledger_tampered"


def test_design_usecase_physical_fusion_and_hardware(capsys):
    code, result = run(capsys, "design", str(EXAMPLES / "design" / "chord-or-curvature.json"))
    assert code == 0 and result["recommendation"] == "cylinder-r50"
    code, result = run(capsys, "usecase", str(EXAMPLES / "usecases" / "injection-moulding-freeform.json"))
    assert code == 0 and result["missing_capabilities"] == []
    code, result = run(capsys, "physical", *(str(EXAMPLES / "physical" / name) for name in (
        "cylinder-chord-protocol.json", "cylinder-chord-measurements.json", "cylinder-chord-predictions.json")))
    assert code == 0 and result["physical_evidence"] is False
    code, result = run(capsys, "fusion", str(EXAMPLES / "fusion" / "two-tracker-scenario.json"), "--policy",
                       str(EXAMPLES / "fusion" / "admission-policy.json"))
    assert code == 0 and result["admission"]["stage"] in {"admitted_state", "admission_refused"}
    reference = json.loads((EXAMPLES / "hardware" / "synthetic-capture.json").read_text(encoding="utf-8"))
    code, result = run(capsys, "hardware", str(EXAMPLES / "hardware" / reference["capture"]), "--layouts",
                       str(EXAMPLES / "hardware" / reference["layouts"]), "--tick-rate-hz",
                       str(reference["parse"]["tick_rate_hz"]), "--device-clock", reference["parse"]["device_clock"])
    assert code == 0 and result["digest"] == reference["retained_digest"]


def test_quick_bench_through_the_main_entry_point(tmp_path, capsys):
    output = tmp_path / "bench"
    code = ciw_main(["science", "bench", "--output", str(output), "--quick"])
    summary = json.loads(capsys.readouterr().out)
    assert code == 0 and summary["ledger"]["verified"]
    assert summary["design"]["recommendation"] == "cylinder-r50"
    assert summary["authority"]["report"]["authorized"] is True
    assert summary["authority"]["actuate"]["authorized"] is False
    assert "control_path" in summary["authority"]["actuate"]["unmet"]
    assert summary["hardware"]["digest_reproduced"] and summary["hardware"]["control"] == "refused"
    assert all(summary["audit"]["probe_detected"]) and summary["audit"]["provenance_findings"] == []
    assert summary["bundle"] == {"ok": True, "signature": "valid"}
    assert summary["physical"]["physical_evidence"] is False
    for experiment in summary["experiments"].values():
        assert experiment["verifications"]["failed"] == 0 and experiment["verifications"]["incomplete"] == 0
        assert experiment["replay"]["diverged"] == 0 and experiment["replay"]["refused"] == 0
    for name in ("report.md", "report.tex", "evidence.ciwb", "summary.json"):
        assert (output / name).is_file()
    with pytest.raises(SystemExit):
        ciw_main(["science", "bench"])
    code = ciw_main(["science", "bench", "--output", str(output), "--quick"])
    assert code == 2 and json.loads(capsys.readouterr().out)["refused"]["code"] == "output_exists"
