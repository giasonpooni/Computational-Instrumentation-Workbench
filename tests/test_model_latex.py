"""LaTeX is a generated, non-executable view; exploratory derivations stay separate."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from ciw.model.latex import (BINDING_SCHEMA, escape_text, exploratory_derivation, latex_view, render_document,
                             validate_binding, validate_exploratory)
from ciw.model.spec import SpecificationError, validate_spec

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "model-core"


def oscillator():
    return validate_spec(json.loads((EXAMPLES / "damped-oscillator.json").read_text()))


def binding(model):
    return {"schema": BINDING_SCHEMA, "spec_digest": model.digest, "result_id": "result-" + "a" * 32,
            "execution_id": "execution-" + "b" * 32,
            "estimates": {"omega_0": {"value": 5.0265, "unit": "rad/s", "standard_uncertainty": 0.01,
                                      "source": "estimated", "evidence_ids": ["sha256:" + "c" * 64]}}}


def test_unbound_view_generates_equations_assumptions_constraints_and_symbols():
    model = oscillator()
    view = latex_view(model)
    text = view["latex"]
    assert view["status"] == "unbound_model" and view["representation"] == "derived_view_not_executable"
    assert r"\frac{\mathrm{d} v}{\mathrm{d} t} &= -2 \, \gamma \, v - {\omega_0}^{2} \, q + \frac{F}{m}" in text
    assert r"Unbound model view" in text and model.digest in text
    assert r"-10\,\mathrm{m} \le q \le 10\,\mathrm{m}" in text
    assert "no process noise" in text and "cal-displacement-probe-1-2026-09" in text
    assert r"$\mathrm{m}\,\mathrm{s}^{-1}$" in text  # m/s rendered from declared factors
    assert view["latex_sha256"] == __import__("hashlib").sha256(text.encode()).hexdigest()


def test_bound_view_links_symbols_to_estimate_uncertainty_and_evidence():
    model = oscillator()
    text = latex_view(model, binding=binding(model))["latex"]
    assert "Bound to run" in text and "result-" + "a" * 32 in text
    assert "5.0265 (estimated) & 0.01" in text and r"\texttt{sha256:cccccccc\ldots}" in text


@pytest.mark.parametrize("change, message", [
    (lambda b: b.update(spec_digest="sha256:" + "0" * 64), "different model"),
    (lambda b: b["estimates"]["omega_0"].update(unit="rad/ms"), "Rescale the model instead of relabelling"),
    (lambda b: b["estimates"]["omega_0"].update(evidence_ids=["not-evidence"]), "sha256 evidence"),
    (lambda b: b["estimates"]["omega_0"].update(standard_uncertainty=-1.0), "nonnegative"),
    (lambda b: b["estimates"].update(ghost=b["estimates"]["omega_0"]), "undeclared symbol"),
    (lambda b: b.update(result_id="execution-" + "a" * 32), "retained result"),
])
def test_binding_refusals(change, message):
    model = oscillator()
    broken = deepcopy(binding(model))
    change(broken)
    with pytest.raises(SpecificationError, match=message):
        validate_binding(model, broken)


def test_exploratory_derivations_are_escaped_unbound_and_not_executable():
    model = oscillator()
    source = (EXAMPLES / "energy-derivation.tex").read_text()
    note = exploratory_derivation("Energy decay", source + "\n\\input{/etc/passwd}", relates_to=model.digest)
    assert note["executable"] is False and note["bound_to_run"] is None
    validate_exploratory(note)
    text = render_document(model, binding=binding(model), exploratory=[note])
    section = text.split("Exploratory derivations (not bound to a run; not executable)")[1]
    assert r"\textbackslash{}input\{/etc/passwd\}" in section
    assert "\\input{/etc/passwd}" not in text
    with pytest.raises(SpecificationError, match="latex_is_not_a_model"):
        validate_spec(note)
    tampered = dict(note, latex="\\dot E = 0")
    with pytest.raises(SpecificationError, match="integrity"):
        validate_exploratory(tampered)
    with pytest.raises(SpecificationError, match="unbound, non-executable"):
        validate_exploratory(dict(note, executable=True))
    other = exploratory_derivation("Other", "x", relates_to="sha256:" + "1" * 64)
    with pytest.raises(SpecificationError, match="different model"):
        render_document(model, exploratory=[other])


def test_text_escaping_covers_latex_specials():
    assert escape_text(r"a_b^c{d}$%&#~\x") == (r"a\_b\textasciicircum{}c\{d\}\$\%\&\#\textasciitilde{}\textbackslash{}x")


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
def test_generated_documents_compile(tmp_path):
    from ciw.model.compose import compose
    plant = oscillator()
    controller = validate_spec(json.loads((EXAMPLES / "pd-controller.json").read_text()))
    loop = compose("closed-loop", "Closed loop", [{"name": "plant", "model": plant}, {"name": "ctrl", "model": controller}],
                   [{"from": "plant.position", "to": "ctrl.measurement"}, {"from": "ctrl.command", "to": "plant.force"}])
    note = exploratory_derivation("Energy decay", (EXAMPLES / "energy-derivation.tex").read_text())
    for name, text in (("plant", render_document(plant, binding=binding(plant), exploratory=[note])),
                       ("loop", render_document(loop))):
        (tmp_path / f"{name}.tex").write_text(text)
        completed = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape",
                                    f"{name}.tex"], cwd=tmp_path, capture_output=True, timeout=120)
        assert completed.returncode == 0, completed.stdout.decode(errors="replace")[-3000:]
        log = (tmp_path / f"{name}.log").read_text(errors="replace")
        assert "Overfull \\hbox" not in log, [line for line in log.splitlines() if "Overfull" in line]
