import json

from ciw import cli
from ciw.lab import runner, svg
from ciw.lab.dashboard import render
from ciw.lab.evidence import finding
from ciw.lab.registry import Implementation, load_queue


def _figure_task(ctx):
    ctx.artifact_text("plot.svg", svg.line_plot([("a", [1, 2], [3, 4])], title="t", xlabel="x", ylabel="y"))
    check = {"reference_kind": "analytic", "reference": "r", "observed": 0.0, "tolerance": 0.0, "passed": True}
    return {"findings": [finding("Shown <claim>", "numerical", 1.5, {"generator": {"name": "g"}, "checks": [check]},
                                 counterexample={"statement": "s"}),
                         finding("Physical <claim>", "physical", None, {})]}


def test_dashboard_is_deterministic_escaped_and_labels_everything(tmp_path, capsys):
    ctx = runner.Context(tmp_path)
    task = {t["id"]: t for t in load_queue()["tasks"]}["T003"]
    report = runner.run_task(task, Implementation("T003", _figure_task), ctx, {})
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "T003.json").write_text(runner.dumps(report))
    page = render(tmp_path)
    assert page == render(tmp_path)
    assert "Shown &lt;claim&gt; (counterexample)" in page and "<claim>" not in page
    assert "numerically_verified" in page and "not_established" in page and "<svg" in page
    assert "1 of 168 tasks reported" in page and "T004 — " in page and "not run" in page
    assert page.count("<script>") == 1 and "http" not in page.replace("http://www.w3.org/2000/svg", "")
    output = tmp_path / "out" / "index.html"
    assert cli.main(["lab", "dashboard", "--retained", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["dashboard"] == str(output) and output.read_text(encoding="utf-8") == page
