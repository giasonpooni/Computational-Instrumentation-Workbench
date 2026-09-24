import base64
import hashlib
import json

from ciw import cli
from ciw.lab import runner, svg
from ciw.lab.dashboard import render
from ciw.lab.evidence import finding
from ciw.lab.registry import Implementation, load_queue
from ciw.lab.report import report_identity


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
    assert "numerically_verified" in page and "not_established" in page
    image = "data:image/svg+xml;base64," + base64.b64encode((tmp_path / "artifacts/T003/plot.svg").read_bytes()).decode()
    assert f'<img src="{image}"' in page and "<svg" not in page       # inert image, never inline markup
    assert "1 of 168 tasks reported" in page and "T004 — " in page and "not run" in page
    assert page.count("<script>") == 1 and "http" not in page.replace(image, "")
    output = tmp_path / "out" / "index.html"
    assert cli.main(["lab", "dashboard", "--retained", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["dashboard"] == str(output) and output.read_text(encoding="utf-8") == page


def test_figures_are_shown_only_when_contained_and_matching_their_digest(tmp_path):
    retained = tmp_path / "retained"
    task = {t["id"]: t for t in load_queue()["tasks"]}["T003"]
    report = runner.run_task(task, Implementation("T003", _figure_task), runner.Context(retained), {})
    (retained / "reports").mkdir()

    def retain(**artifact):
        report["generated_artifacts"][0].update(artifact)
        report["report_id"] = report_identity(report)
        (retained / "reports" / "T003.json").write_text(runner.dumps(report))
        return render(retained)

    hostile = b'<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"><script>alert(2)</script></svg>'
    (retained / report["generated_artifacts"][0]["path"]).write_bytes(hostile)   # replaced after the digest was sealed
    page = retain()
    assert "differs from its recorded sha256" in page and "alert(" not in page and "data:image/svg" not in page
    (tmp_path / "outside.svg").write_bytes(hostile)
    page = retain(path="../outside.svg", sha256=hashlib.sha256(hostile).hexdigest())
    assert "outside the retained directory" in page and "data:image/svg" not in page and "alert(" not in page
    assert "the file is missing" in retain(path="artifacts/T003/gone.svg")


def test_an_unreadable_figure_is_a_note_not_a_crash(tmp_path, capsys):
    retained = tmp_path / "retained"
    task = {t["id"]: t for t in load_queue()["tasks"]}["T003"]
    report = runner.run_task(task, Implementation("T003", _figure_task), runner.Context(retained), {})
    (retained / "reports").mkdir()
    (retained / "reports" / "T003.json").write_text(runner.dumps(report))
    figure = retained / report["generated_artifacts"][0]["path"]
    figure.unlink()
    figure.symlink_to(figure.name)                      # a self-referencing link: resolve() raises
    page = render(retained)
    assert "plot.svg not shown: the file cannot be read" in page and "data:image/svg" not in page
    output = tmp_path / "index.html"
    assert cli.main(["lab", "dashboard", "--retained", str(retained), "--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == page
    capsys.readouterr()
    report["generated_artifacts"][0]["path"] = "artifacts/T003/x\x00.svg"   # resolve() raises ValueError
    report["report_id"] = report_identity(report)
    (retained / "reports" / "T003.json").write_text(runner.dumps(report))
    assert "the file cannot be read" in render(retained)
