"""Self-contained HTML view of retained lab reports.

The page is a deterministic function of the retained reports and their SVG
artifacts: no clock, no network resources and no computation beyond counting.
It displays labels exactly as retained and never restates a result that is not
in a finding. A figure is shown only when its file lies inside the retained
directory and hashes to the digest its report recorded, and it is embedded as
an ``<img>`` data URI, so markup or script inside an SVG never runs in the page.
"""
from __future__ import annotations

import base64
import hashlib
from html import escape
import json
from pathlib import Path

from .evidence import LABELS
from .registry import load_queue
from .report import FIELDS
from .runner import load_reports

LABEL_COLORS = {"analytic": "#6d28d9", "synthetic": "#b45309", "numerically_verified": "#15803d",
                "provider_backed": "#1d4ed8", "hardware_measured": "#0f766e",
                "independently_verified": "#166534", "not_established": "#b91c1c"}
STATE_ORDER = ("completed", "partial", "blocked", "deferred")

CSS = """
:root{--bg:#fbfbfa;--fg:#1c1c1c;--muted:#5b5b5b;--line:#e2e2de;--card:#ffffff}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141413;--fg:#ecebe6;--muted:#a3a29c;--line:#2e2e2b;--card:#1d1d1b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 8px}
p.lead{color:var(--muted);margin:0 0 20px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}.tile b{display:block;font-size:22px}
.tile span{color:var(--muted);font-size:13px}.label,.tile span.label{display:inline-block;max-width:100%;overflow:hidden;text-overflow:ellipsis;vertical-align:middle;padding:1px 8px;border-radius:999px;color:#fff;font-size:12px;white-space:nowrap}
table{width:100%;border-collapse:collapse;margin:8px 0}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:13px;color:var(--muted);font-weight:600}.scroll{overflow-x:auto}details{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:8px 12px}
summary{cursor:pointer;font-weight:600}dl{display:grid;grid-template-columns:minmax(120px,220px) 1fr;gap:4px 12px;margin:10px 0}
dt{color:var(--muted);font-size:13px}dd{margin:0;overflow-wrap:anywhere}dd ul{margin:0;padding-left:18px}figure{margin:12px 0}figure img{max-width:100%;height:auto;background:#fff;border-radius:6px}
.filters{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.filters button{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:999px;padding:3px 10px;cursor:pointer}
.filters button[aria-pressed=true]{outline:2px solid var(--fg)}p.note{color:var(--muted);font-size:13px}
"""

SCRIPT = """
document.querySelectorAll('.filters button').forEach(b=>b.addEventListener('click',()=>{
const on=b.getAttribute('aria-pressed')!=='true';document.querySelectorAll('.filters button').forEach(x=>x.setAttribute('aria-pressed','false'));
b.setAttribute('aria-pressed',on?'true':'false');const want=on?b.dataset.label:null;
document.querySelectorAll('details[data-labels]').forEach(d=>{d.style.display=!want||d.dataset.labels.split(' ').includes(want)?'':'none'})}))
"""


def _chip(label):
    return f'<span class="label" style="background:{LABEL_COLORS.get(label, "#555")}">{escape(label)}</span>'


def _text(value, limit=900):
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return escape(text if len(text) <= limit else text[:limit - 1] + "…")


def _value(value, limit=160):
    """Display form only: floats to four significant digits; retained JSON keeps exact values."""
    if isinstance(value, float):
        return escape(format(value, ".4g"))
    return _text(value, limit)


def _answer(name, value):
    """Render one report answer: string lists as bullets, identities compactly."""
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return "<ul>" + "".join(f"<li>{_text(item, 400)}</li>" for item in value) + "</ul>"
    if isinstance(value, list) and not value:
        return "none"
    if name == "provider_runtime_identity" and isinstance(value, dict):
        shown = {k: (f"{len(v)} source digests" if k == "sources" and isinstance(v, dict) else v) for k, v in value.items()}
        return "; ".join(f"{escape(str(k))}: {_text(v, 200)}" for k, v in sorted(shown.items()))
    return _text(value)


def _figure(retained: Path, artifact) -> str:
    """An SVG artifact as an inert image, or a note saying why it is not shown."""
    name = escape(artifact["path"])
    try:  # a link loop, a NUL byte or an unreadable file must not abort the page
        path = (retained / artifact["path"]).resolve()
        if not path.is_relative_to(retained.resolve()):
            return f'<p class="note">Figure {name} not shown: it lies outside the retained directory.</p>'
        if not path.is_file():
            return f'<p class="note">Figure {name} not shown: the file is missing.</p>'
        data = path.read_bytes()
    except (OSError, RuntimeError, ValueError):
        return f'<p class="note">Figure {name} not shown: the file cannot be read.</p>'
    if hashlib.sha256(data).hexdigest() != artifact.get("sha256"):
        return f'<p class="note">Figure {name} not shown: the file differs from its recorded sha256.</p>'
    source = "data:image/svg+xml;base64," + base64.b64encode(data).decode("ascii")
    return f'<figure><img src="{source}" alt="Figure {name}"><figcaption>{name}</figcaption></figure>'


def render(retained) -> str:
    """Render the dashboard for a directory of retained lab reports."""
    retained = Path(retained)
    queue = load_queue()
    reports = {r["task_id"]: r for r in load_reports(retained)}
    labels = {label: 0 for label in LABELS}
    states = {state: 0 for state in STATE_ORDER}
    for report in reports.values():
        states[report["state"]] += 1
        for label, count in report["evidence_status"]["counts"].items():
            labels[label] += count
    physical = sum(1 for r in reports.values() if r["physical_validation_status"]["status"] != "not_established")
    out = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width,initial-scale=1">',
           f"<title>Lab queue evidence</title><style>{CSS}</style></head><body><main>",
           "<h1>Computational experimentalist queue</h1>",
           f'<p class="lead">{len(reports)} of {len(queue["tasks"])} tasks reported. Every finding carries one '
           "evidence label assigned by <code>ciw.lab.evidence</code>. Tasks reporting physical validation: "
           f"{physical}.</p>", '<div class="tiles">']
    for state in STATE_ORDER:
        out.append(f'<div class="tile"><b>{states[state]}</b><span>{state} tasks</span></div>')
    out.append('</div><h2>Findings by evidence label</h2><div class="tiles">')
    for label in LABELS:
        out.append(f'<div class="tile"><b>{labels[label]}</b><span>{_chip(label)}</span></div>')
    out.append('</div><div class="filters" role="group" aria-label="Filter tasks by label">')
    for label in LABELS:
        out.append(f'<button type="button" data-label="{label}" aria-pressed="false">{escape(label)}</button>')
    out.append("</div>")
    for section in queue["sections"]:
        tasks = [t for t in queue["tasks"] if t["section"] == section["section"]]
        out.append(f'<h2>{section["section"]}. {escape(section["name"])}</h2>')
        for item in tasks:
            report = reports.get(item["id"])
            if report is None:
                out.append(f'<details data-labels=""><summary>{item["id"]} — {escape(item["title"])} · not run</summary></details>')
                continue
            used = sorted({f["evidence_status"] for f in report["findings"]})
            out.append(f'<details data-labels="{" ".join(used)}"><summary>{item["id"]} — {escape(item["title"])} · '
                       f'{escape(report["state"])} · {_chip(report["evidence_status"]["primary"])}</summary><dl>')
            for name, label in FIELDS:
                if name in ("evidence_status", "generated_artifacts", "tests_passed", "tests_skipped", "changed_files"):
                    continue
                value = report[name]
                if name == "physical_validation_status":
                    value = f'{value["status"]}: {value["statement"]}'
                out.append(f"<dt>{escape(label)}</dt><dd>{_answer(name, value)}</dd>")
            out.append(f'<dt>Tests</dt><dd>{len(report["tests_passed"])} passed, {len(report["tests_skipped"])} skipped'
                       f'{", " + str(len(report.get("tests_failed", []))) + " failed" if report.get("tests_failed") else ""}</dd></dl>')
            if report["findings"]:
                out.append('<div class="scroll"><table><tr><th>Finding</th><th>Value</th><th>Label</th></tr>')
                for record in report["findings"]:
                    unit = f' {escape(record["unit"])}' if record.get("unit") else ""
                    flag = " (counterexample)" if record.get("counterexample") else ""
                    out.append(f'<tr><td>{escape(record["claim"])}{flag}</td><td>{_value(record["value"])}{unit}</td>'
                               f'<td>{_chip(record["evidence_status"])}</td></tr>')
                out.append("</table></div>")
            for artifact in report["generated_artifacts"]:
                if artifact["path"].endswith(".svg"):
                    out.append(_figure(retained, artifact))
            out.append("</details>")
    out.append(f"</main><script>{SCRIPT}</script></body></html>")
    return "\n".join(out) + "\n"
