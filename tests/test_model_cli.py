"""Terminal commands for the model core; Julia commands run only when CIW_JULIA is bound."""
import json
import os
from pathlib import Path

import pytest

from ciw.cli import main
from ciw.model.spec import validate_spec

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "model-core"
JULIA = os.environ.get("CIW_JULIA")


def run(capsys, *argv):
    code = main(["model", *map(str, argv)])
    captured = capsys.readouterr()
    return code, (json.loads(captured.out) if captured.out.strip() else None), captured.err


def test_validate_latex_rescale_compose_and_exploratory(tmp_path, capsys):
    spec = EXAMPLES / "damped-oscillator.json"
    code, table, _ = run(capsys, "validate", spec)
    assert code == 0 and table["state_order"] == ["q", "v"]

    code, out, _ = run(capsys, "rescale", spec, "--unit", "q=mm", "--unit", "v=mm/s", "--unit", "y=mm",
                       "--output", tmp_path / "mm.json", "--request", EXAMPLES / "oscillator-simulate.json",
                       "--request-output", tmp_path / "mm-request.json")
    assert code == 0 and out["derived_from"]["factors"]["q"] == pytest.approx(1000.0)
    mm = validate_spec(json.loads((tmp_path / "mm.json").read_text()))
    assert json.loads((tmp_path / "mm-request.json").read_text())["spec_digest"] == mm.digest

    code, out, _ = run(capsys, "compose", "--component", f"plant={spec}",
                       "--component", f"ctrl={EXAMPLES / 'pd-controller.json'}",
                       "--connect", "plant.position", "ctrl.measurement", "--connect", "ctrl.command", "plant.force",
                       "--model-id", "closed-loop", "--title", "Closed loop", "--output", tmp_path / "loop.json")
    assert code == 0 and out["state_order"] == ["plant.q", "plant.v", "ctrl.z"]

    code, out, _ = run(capsys, "exploratory", "--title", "Energy decay",
                       "--latex-file", EXAMPLES / "energy-derivation.tex", "--relates-to", spec,
                       "--output", tmp_path / "note.json")
    assert code == 0 and out["executable"] is False

    code, out, _ = run(capsys, "latex", spec, "--exploratory", tmp_path / "note.json",
                       "--output", tmp_path / "model.tex", "--view-output", tmp_path / "view.json")
    assert code == 0 and out["status"] == "unbound_model"
    assert "Exploratory derivations" in (tmp_path / "model.tex").read_text()


def test_outputs_are_never_overwritten_and_refusals_exit_nonzero(tmp_path, capsys):
    spec = EXAMPLES / "damped-oscillator.json"
    target = tmp_path / "model.tex"
    target.write_text("keep")
    code, _, error = run(capsys, "latex", spec, "--output", target)
    assert code == 2 and target.read_text() == "keep"
    code, _, error = run(capsys, "rescale", spec, "--unit", "q=s", "--output", tmp_path / "bad.json")
    assert code == 2 and "dimensions differ" in error
    code, _, error = run(capsys, "compose", "--component", f"plant={spec}",
                         "--component", f"road={EXAMPLES / 'path-heading.json'}", "--model-id", "x", "--title", "x",
                         "--output", tmp_path / "x.json")
    assert code == 2 and "path_length" in error


@pytest.mark.skipif(not JULIA, reason="CIW_JULIA names the pinned Julia 1.10.12 executable")
def test_run_inspect_replay_and_bound_latex(tmp_path, capsys):
    spec = EXAMPLES / "damped-oscillator.json"
    code, out, err = run(capsys, "run", "linearize", spec, EXAMPLES / "oscillator-linearize.json",
                         "--julia", JULIA, "--output-dir", tmp_path / "runs")
    assert code == 0, err
    assert out["status"] == "completed" and out["reference_passed"] is True
    code, inspected, _ = run(capsys, "inspect", out["run_file"])
    assert code == 0 and inspected["computation_identity"] == out["computation_identity"]
    code, replayed, err = run(capsys, "replay", out["run_file"], "--julia", JULIA, "--output-dir", tmp_path / "runs")
    assert code == 0, err
    assert replayed["replay_of"]["run_id"] == out["run_id"] and replayed["execution_id"] != out["execution_id"]
    code, view, _ = run(capsys, "latex", spec, "--run", out["run_file"], "--output", tmp_path / "bound.tex")
    assert code == 0 and view["status"] == "bound_to_run"
    assert out["result_id"] in (tmp_path / "bound.tex").read_text()
