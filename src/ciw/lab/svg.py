"""Deterministic SVG line plots for retained lab figures.

The output depends only on the plotted numbers and labels, so a regenerated
figure has the same bytes. No plotting library is required.

A figure that plots values at binary64 rounding level, whose last bits follow
the BLAS kernel and platform, records its plotted values with each point's
rounding bound (``line_plot(..., rounding=...)``); ``recorded_values`` reads
them back, so a re-execution compares the values rather than the bytes.
"""
from __future__ import annotations

from html import escape
import json
import math
import numbers
import xml.etree.ElementTree as ET

PALETTE = ("#1f5fa8", "#c2410c", "#15803d", "#7e22ce", "#b91c1c", "#0f766e", "#a16207", "#475569")
WIDTH, HEIGHT = 640, 420
LEFT, RIGHT, TOP, BOTTOM = 78, 170, 40, 58
# Identity of the metadata element holding a figure's recorded values and rounding bounds.
VALUES_ID = "ciw-plotted-values"


def _transform(values, log):
    if log:
        return [math.log10(v) for v in values]
    return list(values)


def _ticks(low, high, log):
    if log:
        start, stop = math.floor(low), math.ceil(high)
        step = max(1, (stop - start) // 6)
        return [float(v) for v in range(start, stop + 1, step)]
    if high == low:
        return [low]
    raw = (high - low) / 5
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min((m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw), default=raw)
    first = math.ceil(low / step) * step
    return [first + i * step for i in range(int((high - first) / step + 1e-9) + 1)]


def _label(value, log):
    if log:
        return f"1e{int(round(value))}"
    return format(value, ".3g")


def _number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _per_point(bound, count, name) -> list:
    bounds = [float(bound)] * count if _number(bound) else [float(b) for b in bound if _number(b)]
    if len(bounds) != count or not all(math.isfinite(b) and b >= 0 for b in bounds):
        raise ValueError(f"Series {name!r} needs one finite, nonnegative rounding bound per point")
    return bounds


def line_plot(series, *, title, xlabel, ylabel, logx=False, logy=False, markers=True, rounding=None) -> str:
    """Plot ``series`` = [(name, xs, ys), ...]; nonpositive values are dropped on log axes.

    ``rounding``, for a figure declared as plotting values at rounding level
    (``ctx.artifact_text(..., rounding_level=True)``), records every plotted
    point (x and y exactly) with its rounding bound: the largest change rounding
    (another BLAS kernel or platform) can make to its y value between two runs,
    in y units. It is one number for every point, or a list with one entry per
    series, each a number or one bound per point of that series. Without it the
    figure is unchanged.
    """
    series = [(name, list(xs), list(ys)) for name, xs, ys in series]
    if rounding is None:
        per_series = [None] * len(series)
    elif isinstance(rounding, bool):
        raise ValueError("rounding is a bound, or one per series, not a flag")
    else:
        per_series = [rounding] * len(series) if _number(rounding) else list(rounding)
        if len(per_series) != len(series):
            raise ValueError("Give one rounding bound, or one per series")
    cleaned = []
    for (name, xs, ys), bound in zip(series, per_series):
        bounds = [None] * len(xs) if bound is None else _per_point(bound, min(len(xs), len(ys)), name)
        points = [(float(x), float(y), b) for x, y, b in zip(xs, ys, bounds)
                  if math.isfinite(x) and math.isfinite(y) and (not logx or x > 0) and (not logy or y > 0)]
        if points:
            cleaned.append((name, [p[0] for p in points], [p[1] for p in points], [p[2] for p in points]))
    if not cleaned:
        raise ValueError("No finite points to plot")
    recorded = None if rounding is None else [{"name": str(name), "x": xs, "y": ys, "bound": bounds}
                                              for name, xs, ys, bounds in cleaned]
    cleaned = [(name, xs, ys) for name, xs, ys, _ in cleaned]
    tx = [_transform(xs, logx) for _, xs, _ in cleaned]
    ty = [_transform(ys, logy) for _, _, ys in cleaned]
    xmin, xmax = min(min(v) for v in tx), max(max(v) for v in tx)
    ymin, ymax = min(min(v) for v in ty), max(max(v) for v in ty)
    if xmax == xmin:
        xmin, xmax = xmin - 1, xmax + 1
    if ymax == ymin:
        ymin, ymax = ymin - 1, ymax + 1
    pad = 0.04 * (ymax - ymin)
    ymin, ymax = ymin - pad, ymax + pad
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def px(x):
        return LEFT + (x - xmin) / (xmax - xmin) * plot_w

    def py(y):
        return TOP + (ymax - y) / (ymax - ymin) * plot_h

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
           f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="sans-serif" font-size="11">',
           f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
           f'<text x="{LEFT}" y="22" font-size="13" font-weight="bold">{escape(title)}</text>',
           f'<rect x="{LEFT}" y="{TOP}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333"/>']
    for tick in _ticks(xmin, xmax, logx):
        if xmin <= tick <= xmax:
            x = px(tick)
            out.append(f'<line x1="{x:.2f}" y1="{TOP}" x2="{x:.2f}" y2="{TOP + plot_h}" stroke="#e5e7eb"/>')
            out.append(f'<text x="{x:.2f}" y="{TOP + plot_h + 16}" text-anchor="middle">{_label(tick, logx)}</text>')
    for tick in _ticks(ymin, ymax, logy):
        if ymin <= tick <= ymax:
            y = py(tick)
            out.append(f'<line x1="{LEFT}" y1="{y:.2f}" x2="{LEFT + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
            out.append(f'<text x="{LEFT - 6}" y="{y + 4:.2f}" text-anchor="end">{_label(tick, logy)}</text>')
    out.append(f'<text x="{LEFT + plot_w / 2:.2f}" y="{HEIGHT - 14}" text-anchor="middle">{escape(xlabel)}</text>')
    out.append(f'<text x="16" y="{TOP + plot_h / 2:.2f}" text-anchor="middle" '
               f'transform="rotate(-90 16 {TOP + plot_h / 2:.2f})">{escape(ylabel)}</text>')
    for index, ((name, _, _), xs, ys) in enumerate(zip(cleaned, tx, ty)):
        color = PALETTE[index % len(PALETTE)]
        path = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.2f},{py(y):.2f}" for i, (x, y) in enumerate(zip(xs, ys)))
        out.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        if markers:
            for x, y in zip(xs, ys):
                out.append(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="2.6" fill="{color}"/>')
        ly = TOP + 12 + 18 * index
        out.append(f'<line x1="{LEFT + plot_w + 12}" y1="{ly}" x2="{LEFT + plot_w + 32}" y2="{ly}" '
                   f'stroke="{color}" stroke-width="2"/>')
        out.append(f'<text x="{LEFT + plot_w + 38}" y="{ly + 4}">{escape(name)}</text>')
    if recorded is not None:
        values = escape(json.dumps({"series": recorded}, separators=(",", ":")), quote=False)
        out.append(f'<metadata id="{VALUES_ID}">{values}</metadata>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def recorded_values(data) -> list | None:
    """The plotted values and rounding bounds a figure records (``line_plot(..., rounding=...)``): a list of
    ``{"name", "x", "y", "bound"}`` series, or None when the figure records none or is not well-formed."""
    try:
        root = ET.fromstring(data)
        element = next((e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == "metadata"
                        and e.get("id") == VALUES_ID), None)
        series = json.loads(element.text)["series"] if element is not None else None
    except (ET.ParseError, TypeError, ValueError, KeyError):
        return None
    if not isinstance(series, list) or not all(
            isinstance(s, dict) and set(s) == {"name", "x", "y", "bound"} and isinstance(s["name"], str)
            and all(isinstance(s[k], list) and len(s[k]) == len(s["x"])
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                            for v in s[k]) for k in ("x", "y", "bound")) for s in series):
        return None
    return series
