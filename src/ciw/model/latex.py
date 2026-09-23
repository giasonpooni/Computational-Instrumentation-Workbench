"""LaTeX views generated from the structured model; never parsed back.

``ciw.model-latex-view.v1`` is a derived representation of one specification
digest. A ``ciw.model-binding.v1`` links symbols to a retained run: each
estimate carries its declared unit, value, standard uncertainty, source and
evidence identities. Without a binding the view states that it is unbound.

``ciw.exploratory-derivation.v1`` retains handwritten LaTeX as text. It is
never executable, never accepted as a model and is displayed as escaped
source in its own section, so it cannot be mistaken for equations bound to a
run or inject commands into a generated report.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import math
import re

from ..core.identities import content_identity
from . import expression
from .spec import Model, SpecificationError, _keys, _text
from .units import latex_unit

VIEW_SCHEMA = "ciw.model-latex-view.v1"
BINDING_SCHEMA = "ciw.model-binding.v1"
EXPLORATORY_SCHEMA = "ciw.exploratory-derivation.v1"
ESTIMATE_SOURCES = frozenset({"declared_in_specification", "requested_initial_condition",
                              "requested_operating_point", "estimated", "measured"})
_CONTENT = re.compile(r"sha256:[0-9a-f]{64}\Z")
_EVENT = re.compile(r"(result|execution)-[0-9a-f]{32}\Z")
EXPLORATORY_LIMIT = 65536


def escape_text(text: str) -> str:
    """Escape plain text for LaTeX; the result typesets the characters themselves."""
    replacements = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&",
                    "%": r"\%", "#": r"\#", "_": r"\_", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}"}
    return "".join(replacements.get(char, char) for char in text)


def _number(value: float) -> str:
    text = format(value, ".12g")
    if "e" in text:
        mantissa, exponent = text.split("e")
        return rf"{mantissa} \times 10^{{{int(exponent)}}}"
    return text


def validate_binding(model: Model, binding) -> dict:
    _keys(binding, {"schema", "spec_digest", "result_id", "execution_id", "estimates"}, name="model binding")
    if binding["schema"] != BINDING_SCHEMA:
        raise SpecificationError("Unsupported model binding schema")
    if binding["spec_digest"] != model.digest:
        raise SpecificationError("Binding names a different model specification")
    for key, prefix in (("result_id", "result"), ("execution_id", "execution")):
        value = binding[key]
        if not isinstance(value, str) or not _EVENT.fullmatch(value) or not value.startswith(prefix):
            raise SpecificationError(f"Binding {key} must be a retained {prefix} identity")
    estimates = binding["estimates"]
    if not isinstance(estimates, dict) or not estimates:
        raise SpecificationError("A binding must link at least one symbol")
    for symbol, estimate in estimates.items():
        if symbol not in model.entries:
            raise SpecificationError(f"Binding names undeclared symbol {symbol}")
        _keys(estimate, {"value", "unit", "standard_uncertainty", "source", "evidence_ids"}, name=f"estimate {symbol}")
        if estimate["unit"] != model.entries[symbol]["unit"]:
            raise SpecificationError(f"Estimate {symbol} is in {estimate['unit']}; the model declares "
                                     f"{model.entries[symbol]['unit']}. Rescale the model instead of relabelling.")
        if type(estimate["value"]) not in (int, float) or not math.isfinite(estimate["value"]):
            raise SpecificationError(f"Estimate {symbol} value must be finite")
        sigma = estimate["standard_uncertainty"]
        if sigma is not None and (type(sigma) not in (int, float) or not math.isfinite(sigma) or sigma < 0):
            raise SpecificationError(f"Estimate {symbol} standard uncertainty must be finite, nonnegative or null")
        if estimate["source"] not in ESTIMATE_SOURCES:
            raise SpecificationError(f"Estimate {symbol} source must be one of {sorted(ESTIMATE_SOURCES)}")
        evidence = estimate["evidence_ids"]
        if (not isinstance(evidence, list) or not evidence or len(set(evidence)) != len(evidence)
                or any(not isinstance(item, str) or not _CONTENT.fullmatch(item) for item in evidence)):
            raise SpecificationError(f"Estimate {symbol} needs distinct sha256 evidence identities")
    return binding


def exploratory_derivation(title: str, latex: str, *, relates_to: str | None = None,
                           author: str | None = None, created_at: str | None = None) -> dict:
    _text(title, "exploratory title")
    if not isinstance(latex, str) or not latex.strip() or len(latex) > EXPLORATORY_LIMIT:
        raise SpecificationError(f"Exploratory LaTeX must be nonempty text of at most {EXPLORATORY_LIMIT} characters")
    if relates_to is not None and not _CONTENT.fullmatch(str(relates_to)):
        raise SpecificationError("relates_to must be a model specification digest")
    record = {"schema": EXPLORATORY_SCHEMA, "title": title, "latex": latex, "relates_to_spec": relates_to,
              "author": author, "created_at": created_at or datetime.now(timezone.utc).isoformat(),
              "status": "exploratory", "executable": False, "bound_to_run": None}
    record["derivation_id"] = content_identity(record)
    return record


def validate_exploratory(record) -> dict:
    _keys(record, {"schema", "title", "latex", "relates_to_spec", "author", "created_at", "status",
                   "executable", "bound_to_run", "derivation_id"}, name="exploratory derivation")
    if (record["schema"] != EXPLORATORY_SCHEMA or record["status"] != "exploratory"
            or record["executable"] is not False or record["bound_to_run"] is not None):
        raise SpecificationError("An exploratory derivation is unbound, non-executable text")
    if record["derivation_id"] != content_identity({key: value for key, value in record.items() if key != "derivation_id"}):
        raise SpecificationError("Exploratory derivation integrity mismatch")
    return record


def _equation_rows(model: Model, names: dict) -> list[str]:
    spec = model.spec
    t = names[model.independent]
    rows = []
    for entry in spec["dynamics"]:
        rows.append(rf"\frac{{\mathrm{{d}} {names[entry['state']]}}}{{\mathrm{{d}} {t}}} &= "
                    + expression.to_latex(entry["rhs"], names))
    return rows


def _symbol_row(model: Model, symbol: str, names: dict, binding) -> str:
    entry, kind = model.entries[symbol], model.kind[symbol]
    role = entry.get("role") or {"observation": "measured", "state": "state", "parameter": "parameter",
                                 "independent": entry.get("kind")}.get(kind, kind)
    estimate, uncertainty, evidence = "--", "--", "--"
    if kind == "parameter":
        estimate = _number(entry["value"]) + " (declared)"
    if binding and symbol in binding["estimates"]:
        item = binding["estimates"][symbol]
        estimate = _number(item["value"]) + f" ({escape_text(item['source'].replace('_', ' '))})"
        uncertainty = "--" if item["standard_uncertainty"] is None else _number(item["standard_uncertainty"])
        evidence = r"\newline ".join(rf"\texttt{{{escape_text(value[:15])}\ldots}}" for value in item["evidence_ids"])
    calibration = entry.get("calibration")
    if calibration:
        role += "; " + (f"cal. {calibration['calibration_id']}" if calibration["status"] == "calibrated" else "uncalibrated")
    frame = entry.get("frame") or "--"
    return (rf"${names[symbol]}$ & \texttt{{{escape_text(symbol)}}} & {escape_text(role)} & "
            rf"${latex_unit(model.units[symbol])}$ & {escape_text(frame)} & {estimate} & {uncertainty} & {evidence} \\")


def render_document(model: Model, *, binding: dict | None = None, exploratory: list[dict] | None = None) -> str:
    """Render a standalone LaTeX document (amsmath, longtable) for the model."""
    if binding is not None:
        validate_binding(model, binding)
    names = model.names()
    spec = model.spec
    lines = [
        r"\documentclass{article}", r"\usepackage{amsmath}", r"\usepackage{longtable}",
        r"\usepackage[margin=2cm]{geometry}", r"\begin{document}",
        rf"\section*{{{escape_text(spec['title'])}}}",
        rf"\noindent Model \texttt{{{escape_text(spec['model_id'])}}}\\",
        rf"Specification \texttt{{{model.digest}}}\\",
    ]
    if binding is None:
        lines.append(r"\textbf{Unbound model view}: these equations are generated from the specification and are "
                     r"not linked to a retained run.")
    else:
        lines += [r"\textbf{Bound to run}. Equations are generated from the specification executed by that run.\\",
                  rf"Result \texttt{{{binding['result_id']}}}\\",
                  rf"Execution \texttt{{{binding['execution_id']}}}"]
    independent = spec["independent_variable"]
    lines += [
        r"\subsection*{Dynamics}",
        rf"Independent variable ${names[model.independent]}$ ({escape_text(independent['kind'].replace('_', ' '))}, "
        rf"${latex_unit(model.units[model.independent])}$, origin: {escape_text(independent['origin'])}). "
        r"State order: $(" + ", ".join(names[s] for s in model.state_order) + r")$.",
        r"\begin{align*}", r" \\ ".join(_equation_rows(model, names)), r"\end{align*}",
    ]
    if spec["derived"]:
        lines += [r"\subsection*{Derived quantities}", r"\begin{align*}",
                  r" \\ ".join(f"{names[e['symbol']]} &= {expression.to_latex(e['expression'], names)}"
                               for e in spec["derived"]), r"\end{align*}"]
    if spec["observations"]:
        lines += [r"\subsection*{Observation model}", r"\begin{align*}",
                  r" \\ ".join(f"{names[e['symbol']]} &= {expression.to_latex(e['expression'], names)}"
                               for e in spec["observations"]), r"\end{align*}"]
        lines.append(r"\begin{itemize}")
        for entry in spec["observations"]:
            calibration = entry["calibration"]
            status = (f"calibration {calibration['calibration_id']}" if calibration["status"] == "calibrated"
                      else "uncalibrated")
            lines.append(rf"\item ${names[entry['symbol']]}$: sensor {escape_text(entry['sensor'])}, {escape_text(status)}")
        lines.append(r"\end{itemize}")
    uncertainty = spec["uncertainty"]
    lines.append(r"\subsection*{Uncertainty assumptions}")
    lines.append(r"\begin{itemize}")
    for item in uncertainty["assumptions"]:
        lines.append(rf"\item {escape_text(item.replace('_', ' '))}")
    for key in ("parameters", "initial_state", "measurement_noise", "process_noise"):
        block = uncertainty[key]
        if block is None:
            lines.append(rf"\item {escape_text(key.replace('_', ' '))} covariance: not declared")
            continue
        order = ", ".join(names[s] for s in block["order"])
        rows = r" \\ ".join(" & ".join(_number(v) for v in row) for row in block["matrix"])
        lines.append(rf"\item {escape_text(key.replace('_', ' '))} covariance, order $({order})$, declared units: "
                     rf"$\begin{{pmatrix}} {rows} \end{{pmatrix}}$")
    lines.append(r"\end{itemize}")
    domain = spec["validity_domain"]
    lines += [r"\subsection*{Validity domain}", r"\begin{itemize}"]
    if domain["independent_variable"] is not None:
        lines.append(r"\item " + _bound_latex(names[model.independent], domain["independent_variable"],
                                              latex_unit(model.units[model.independent])))
    for symbol, bounds in domain["bounds"].items():
        lines.append(r"\item " + _bound_latex(names[symbol], bounds, latex_unit(model.units[symbol])))
    for note in domain["notes"]:
        lines.append(rf"\item {escape_text(note)}")
    lines.append(r"\end{itemize}")
    lines += [r"\subsection*{Symbols}", r"{\footnotesize\setlength{\tabcolsep}{3pt}",
              r"\begin{longtable}{@{}l l p{2.4cm} l p{1.7cm} p{2.2cm} l p{3.2cm}@{}}",
              r"Symbol & Name & Role & Unit & Frame & Estimate & Std.\ unc. & Evidence \\ \hline \endhead"]
    lines += [_symbol_row(model, symbol, names, binding) for symbol in model.entries]
    lines.append(r"\end{longtable}}")
    if binding is not None:
        evidence = sorted({value for item in binding["estimates"].values() for value in item["evidence_ids"]})
        lines += [r"\paragraph{Evidence identities}", r"\begin{itemize}\footnotesize"]
        lines += [rf"\item \texttt{{{escape_text(value)}}}" for value in evidence]
        lines.append(r"\end{itemize}")
    for record in exploratory or []:
        validate_exploratory(record)
        if record["relates_to_spec"] not in (None, model.digest):
            raise SpecificationError("An exploratory derivation relates to a different model")
    if exploratory:
        lines.append(r"\subsection*{Exploratory derivations (not bound to a run; not executable)}")
        for record in exploratory:
            source = r"\\".join(escape_text(line) or r"\mbox{}" for line in record["latex"].splitlines())
            lines += [rf"\paragraph{{{escape_text(record['title'])}}} \texttt{{{record['derivation_id'][:19]}\ldots}}",
                      r"\begin{flushleft}\ttfamily\small", source, r"\end{flushleft}"]
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def _bound_latex(name, bounds, unit):
    low, high = bounds
    if low is not None and high is not None:
        return rf"${_number(low)}\,{unit} \le {name} \le {_number(high)}\,{unit}$"
    if low is not None:
        return rf"${name} \ge {_number(low)}\,{unit}$"
    if high is not None:
        return rf"${name} \le {_number(high)}\,{unit}$"
    return rf"${name}$ unbounded"


def latex_view(model: Model, *, binding: dict | None = None, exploratory: list[dict] | None = None) -> dict:
    document = render_document(model, binding=binding, exploratory=exploratory)
    return {"schema": VIEW_SCHEMA, "spec_digest": model.digest,
            "status": "bound_to_run" if binding else "unbound_model",
            "binding": binding, "exploratory_ids": [item["derivation_id"] for item in exploratory or []],
            "representation": "derived_view_not_executable",
            "latex_sha256": sha256(document.encode("utf-8")).hexdigest(), "latex": document}
