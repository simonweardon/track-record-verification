"""Dashboard: one self-contained HTML page rendered from an output directory.

Reads only the CSVs the phases already wrote, so it regenerates from real data
with the same command.  Charts are inline SVG built here (no chart library);
a small script adds hover tooltips and a crosshair on the line charts.
"""
from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .composite import series_stats
from .returns import chain, annualize

# ---------------------------------------------------------------- formatting

def pct(x, d=2, sign=True):
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return "n/a"
    return f"{x * 100:{'+' if sign else ''}.{d}f}%"


def num(x, d=2):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}f}"


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min([1, 2, 2.5, 5, 10], key=lambda s: abs(s * mag - raw)) * mag
    t0 = np.floor(lo / step) * step
    t1 = np.ceil(hi / step) * step          # always cover the data; never let a mark sit on the edge
    ticks = list(np.arange(t0, t1 + step * 0.5, step))
    return [round(t, 10) for t in ticks if t >= t0 - 1e-9 and t <= t1 + 1e-9]


class Lin:
    def __init__(self, d0, d1, r0, r1):
        self.d0, self.d1, self.r0, self.r1 = d0, d1, r0, r1

    def __call__(self, v):
        if self.d1 == self.d0:
            return (self.r0 + self.r1) / 2
        return self.r0 + (v - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


def cell_label(c) -> str:
    return str(c)


def cell_year(c) -> int:
    s = str(c)
    return int(s[:4])


# ---------------------------------------------------------------- SVG charts

def svg_open(w, h, cls="", extra=""):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" class="chart {cls}" role="img" '
            f'preserveAspectRatio="xMidYMid meet" {extra}>')


def money(v):
    return f"${v / 1000:,.1f}k" if v >= 1000 else (f"${v:,.0f}" if v >= 10 else f"${v:,.2f}")


def line_chart(cells, series, height=300, width=760, y_fmt=lambda v: f"{v:.0f}", y_log=False,
               end_labels=True, uid="lc"):
    """series: list of dict(name, values(list float|None), cls, dash(bool), emph(bool))."""
    ml, mr, mt, mb = 52, (128 if end_labels else 16), 14, 30
    n = len(cells)
    xs = Lin(0, max(n - 1, 1), ml, width - mr)
    allv = [v for s in series for v in s["values"] if v is not None and not pd.isna(v)]
    if y_log:
        tr = np.log10
        lo, hi = tr(min(allv)), tr(max(allv))
        ticks_v = [10 ** k for k in range(int(np.floor(lo)), int(np.ceil(hi)) + 1)]
        ticks_v = [t for t in ticks_v if t >= min(allv) * 0.999 and t <= max(allv) * 1.001] or [min(allv), max(allv)]
        ys = Lin(lo, hi, height - mb, mt)
        Y = lambda v: ys(tr(v))
    else:
        tr = lambda v: v
        ticks_v = nice_ticks(min(allv + [0]), max(allv), 5)
        lo, hi = min(ticks_v), max(ticks_v)
        ys = Lin(lo, hi, height - mb, mt)
        Y = ys
    out = [svg_open(width, height, "line", f'data-line="{uid}"')]
    # grid + y axis
    for t in ticks_v:
        y = Y(t)
        out.append(f'<line class="grid" x1="{ml}" x2="{width - mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="ax" x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(y_fmt(t))}</text>')
    # x axis labels: ~8 evenly spaced
    step = max(1, n // 8)
    for i in range(0, n, step):
        out.append(f'<text class="ax" x="{xs(i):.1f}" y="{height - 8}" text-anchor="middle">{esc(cell_year(cells[i]))}</text>')
    out.append(f'<line class="axis" x1="{ml}" x2="{width - mr}" y1="{height - mb}" y2="{height - mb}"/>')
    # series
    for s in series:
        pts = [(xs(i), Y(v)) for i, v in enumerate(s["values"]) if v is not None and not pd.isna(v)]
        if not pts:
            continue
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="5 4"' if s.get("dash") else ""
        out.append(f'<path class="ser {s["cls"]}" d="{d}" fill="none"{dash}/>')
        if s.get("area"):
            base = Y(ticks_v[0]) if not y_log else height - mb
            out.append(f'<path class="area {s["cls"]}" d="{d} L{pts[-1][0]:.1f},{base:.1f} L{pts[0][0]:.1f},{base:.1f} Z"/>')
        x, y = pts[-1]
        out.append(f'<circle class="dot {s["cls"]}" cx="{x:.1f}" cy="{y:.1f}" r="4"/>')
        if end_labels:
            lastv = [v for v in s["values"] if v is not None and not pd.isna(v)][-1]
            out.append(f'<text class="lab" x="{x + 8:.1f}" y="{y + 4:.1f}">{esc(s["name"])} {esc(y_fmt(lastv))}</text>')
    # crosshair + hit area
    out.append(f'<line class="xh" x1="0" x2="0" y1="{mt}" y2="{height - mb}" visibility="hidden"/>')
    out.append(f'<rect class="hit" x="{ml}" y="{mt}" width="{width - ml - mr}" height="{height - mt - mb}" fill="transparent"/>')
    out.append("</svg>")
    xs_list = [round(xs(i), 1) for i in range(n)]
    rows = []
    for i in range(n):
        parts = [f"<b>{esc(cell_label(cells[i]))}</b>"]
        for s in series:
            v = s["values"][i]
            if v is not None and not pd.isna(v):
                parts.append(f'<span class="k {s["cls"]}"></span>{esc(s["name"])} {esc(y_fmt(v))}')
        rows.append("<br>".join(parts))
    data = json.dumps({"xs": xs_list, "rows": rows})
    return "".join(out) + f'<script type="application/json" id="{uid}-data">{data}</script>'


def diverging_bars(labels, values, height=240, width=760, y_fmt=lambda v: f"{v * 100:+.0f}%", tips=None):
    ml, mr, mt, mb = 52, 16, 14, 30
    n = len(values)
    ticks_v = nice_ticks(min(values + [0]), max(values + [0]), 5)
    lo, hi = min(ticks_v), max(ticks_v)
    ys = Lin(lo, hi, height - mb, mt)
    band = (width - ml - mr) / max(n, 1)
    bw = min(24, band - 2)
    out = [svg_open(width, height, "bars")]
    for t in ticks_v:
        y = ys(t)
        out.append(f'<line class="grid" x1="{ml}" x2="{width - mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="ax" x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(y_fmt(t))}</text>')
    y0 = ys(0)
    out.append(f'<line class="axis" x1="{ml}" x2="{width - mr}" y1="{y0:.1f}" y2="{y0:.1f}"/>')
    step = max(1, n // 10)
    for i, (lab, v) in enumerate(zip(labels, values)):
        x = ml + band * i + (band - bw) / 2
        y1 = ys(v)
        top, h = (min(y0, y1), abs(y0 - y1))
        cls = "pos" if v >= 0 else "neg"
        r = 4 if h > 6 else 0
        # rounded at the data end only, square at baseline
        if v >= 0:
            path = (f"M{x:.1f},{y0:.1f} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} "
                    f"H{x + bw - r:.1f} Q{x + bw:.1f},{top:.1f} {x + bw:.1f},{top + r:.1f} V{y0:.1f} Z")
        else:
            bot = top + h
            path = (f"M{x:.1f},{y0:.1f} V{bot - r:.1f} Q{x:.1f},{bot:.1f} {x + r:.1f},{bot:.1f} "
                    f"H{x + bw - r:.1f} Q{x + bw:.1f},{bot:.1f} {x + bw:.1f},{bot - r:.1f} V{y0:.1f} Z")
        tip = esc(tips[i]) if tips else esc(f"{lab}: {y_fmt(v)}")
        out.append(f'<path class="bar {cls}" d="{path}" data-tip="{tip}"/>')
        if i % step == 0:
            out.append(f'<text class="ax" x="{x + bw / 2:.1f}" y="{height - 8}" text-anchor="middle">{esc(lab)}</text>')
    out.append("</svg>")
    return "".join(out)


def coverage_grid(mat: pd.DataFrame, stmt: pd.DataFrame, width=760):
    years = [int(c) for c in mat.columns]
    accts = list(mat.index)
    lw = 110
    cw = min(22, (width - lw - 8) / max(len(years), 1))
    ch = 22
    top = 26
    height = top + ch * len(accts) + 6
    by_id = {}
    if len(stmt):
        s = stmt.copy(); s["year"] = pd.to_datetime(s.period_end).dt.year
        for (a, y), g in s.groupby(["account_id", "year"]):
            by_id[(a, y)] = g
    out = [svg_open(width, height, "cov")]
    for j, y in enumerate(years):
        if j % 2 == 0 or cw > 26:
            out.append(f'<text class="ax" x="{lw + cw * j + cw / 2:.1f}" y="{top - 8}" text-anchor="middle">’{str(y)[2:]}</text>')
    letter = {"V": "V", "U": "U", "X": "X", "-": "–"}
    for i, a in enumerate(accts):
        yy = top + ch * i
        out.append(f'<text class="rowlab" x="{lw - 10}" y="{yy + ch / 2 + 4:.1f}" text-anchor="end">{esc(a)}</text>')
        for j, y in enumerate(years):
            v = str(mat.loc[a, str(y)] if str(y) in mat.columns else mat.loc[a, y])
            if v in ("", "nan"):
                continue
            code = v[0]; count = v[1:]
            x = lw + cw * j
            g = by_id.get((a, y))
            if g is not None:
                ids = ", ".join(g.statement_id.head(3)) + (" …" if len(g) > 3 else "")
                flags = "; ".join(f for f in g["flags"].dropna().unique() if f) or "no flags"
                tip = f"{a} · {y} · {len(g)} statement(s) · {ids}<br>{flags}"
            else:
                tip = f"{a} · {y} · no statement (hole inside observed span)"
            out.append(f'<rect class="cell {code}" x="{x + 1:.1f}" y="{yy + 1}" width="{cw - 2:.1f}" height="{ch - 2}" rx="3" data-tip="{esc(tip)}"/>')
            out.append(f'<text class="cellt" x="{x + cw / 2:.1f}" y="{yy + ch / 2 + 3.5:.1f}" text-anchor="middle">{letter.get(code, code)}{"" if cw < 20 else count}</text>')
    out.append("</svg>")
    return "".join(out)


def stacked_h(segments, width=760, height=64, fmt=lambda v: f"{v * 100:+.1f}%"):
    """segments: list of (label, value, cls). Values may be negative; drawn from 0."""
    total = sum(v for _, v, _ in segments)
    ml, mr = 8, 8
    scale = (width - ml - mr) / max(abs(total), 1e-9)
    out = [svg_open(width, height, "stack")]
    x = ml
    for lab, v, cls in segments:
        w = abs(v) * scale
        out.append(f'<rect class="seg {cls}" x="{x + 1:.1f}" y="10" width="{max(w - 2, 0):.1f}" height="24" rx="3" data-tip="{esc(lab)}: {esc(fmt(v))}"/>')
        if w > 90:
            out.append(f'<text class="segt" x="{x + w / 2:.1f}" y="26" text-anchor="middle">{esc(lab)} {esc(fmt(v))}</text>')
        x += w
    out.append(f'<text class="ax" x="{ml}" y="{height - 10}">0</text>')
    out.append(f'<text class="ax" x="{width - mr}" y="{height - 10}" text-anchor="end">{esc(fmt(total))} total</text>')
    out.append("</svg>")
    return "".join(out)


def dot_whisker(rows, width=760, x_fmt=lambda v: f"{v * 100:+.0f}%"):
    """rows: dict(label, value, lo, hi, cls, note)."""
    ml, mr, mt, mb = 150, 70, 10, 26
    rh = 26
    height = mt + rh * len(rows) + mb
    lo = min(min(r["lo"] for r in rows), 0); hi = max(max(r["hi"] for r in rows), 0)
    ticks = nice_ticks(lo, hi, 5)
    xs = Lin(min(ticks), max(ticks), ml, width - mr)
    out = [svg_open(width, height, "dw")]
    for t in ticks:
        x = xs(t)
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{mt}" y2="{height - mb}"/>')
        out.append(f'<text class="ax" x="{x:.1f}" y="{height - 8}" text-anchor="middle">{esc(x_fmt(t))}</text>')
    x0 = xs(0)
    out.append(f'<line class="axis zero" x1="{x0:.1f}" x2="{x0:.1f}" y1="{mt}" y2="{height - mb}"/>')
    for i, r in enumerate(rows):
        y = mt + rh * i + rh / 2
        out.append(f'<text class="rowlab" x="{ml - 12}" y="{y + 4:.1f}" text-anchor="end">{esc(r["label"])}</text>')
        out.append(f'<line class="whisk {r["cls"]}" x1="{xs(r["lo"]):.1f}" x2="{xs(r["hi"]):.1f}" y1="{y:.1f}" y2="{y:.1f}" data-tip="{esc(r["note"])}"/>')
        out.append(f'<circle class="dot {r["cls"]}" cx="{xs(r["value"]):.1f}" cy="{y:.1f}" r="5" data-tip="{esc(r["note"])}"/>')
        out.append(f'<text class="lab" x="{width - mr + 10}" y="{y + 4:.1f}">{esc(x_fmt(r["value"]))}</text>')
    out.append("</svg>")
    return "".join(out)


def histogram(samples: np.ndarray, actual: float, width=760, height=220, bins=48,
              x_fmt=lambda v: f"{v * 100:+.0f}%"):
    ml, mr, mt, mb = 40, 16, 22, 30
    lo, hi = float(np.percentile(samples, 0.2)), float(max(np.percentile(samples, 99.8), actual * 1.05))
    counts, edges = np.histogram(samples, bins=bins, range=(lo, hi))
    xs = Lin(lo, hi, ml, width - mr)
    ys = Lin(0, counts.max(), height - mb, mt)
    out = [svg_open(width, height, "hist")]
    for t in nice_ticks(lo, hi, 6):
        out.append(f'<text class="ax" x="{xs(t):.1f}" y="{height - 8}" text-anchor="middle">{esc(x_fmt(t))}</text>')
    out.append(f'<line class="axis" x1="{ml}" x2="{width - mr}" y1="{height - mb}" y2="{height - mb}"/>')
    n = len(samples)
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        if c == 0:
            continue
        x0, x1 = xs(e0), xs(e1)
        cls = "emph" if e0 >= actual else "dim"
        tip = f"{x_fmt(e0)} to {x_fmt(e1)}: {c:,} of {n:,} simulated managers ({c / n:.1%})"
        out.append(f'<rect class="hbar {cls}" x="{x0 + 1:.1f}" y="{ys(c):.1f}" width="{max(x1 - x0 - 2, 1):.1f}" height="{height - mb - ys(c):.1f}" data-tip="{esc(tip)}"/>')
    xa = xs(actual)
    out.append(f'<line class="marker" x1="{xa:.1f}" x2="{xa:.1f}" y1="{mt - 4}" y2="{height - mb}"/>')
    anchor = "end" if xa > width * 0.7 else "start"
    out.append(f'<text class="lab strong" x="{xa + (-8 if anchor == "end" else 8):.1f}" y="{mt + 6}" text-anchor="{anchor}">this record {esc(x_fmt(actual))}/yr</text>')
    out.append("</svg>")
    return "".join(out)


def band_line(cells, mid, lo, hi, width=760, height=230, y_fmt=lambda v: f"{v * 100:+.0f}%", uid="bl", name="alpha"):
    ml, mr, mt, mb = 52, 16, 14, 30
    n = len(cells)
    xs = Lin(0, max(n - 1, 1), ml, width - mr)
    vals = [v for v in list(mid) + list(lo) + list(hi) if v is not None and not pd.isna(v)]
    ticks = nice_ticks(min(vals + [0]), max(vals + [0]), 5)
    ys = Lin(min(ticks), max(ticks), height - mb, mt)
    out = [svg_open(width, height, "line", f'data-line="{uid}"')]
    for t in ticks:
        y = ys(t)
        out.append(f'<line class="{"axis zero" if t == 0 else "grid"}" x1="{ml}" x2="{width - mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="ax" x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(y_fmt(t))}</text>')
    step = max(1, n // 8)
    for i in range(0, n, step):
        out.append(f'<text class="ax" x="{xs(i):.1f}" y="{height - 8}" text-anchor="middle">{esc(cell_year(cells[i]))}</text>')
    idx = [i for i in range(n) if not pd.isna(mid[i])]
    if idx:
        up = " L".join(f"{xs(i):.1f},{ys(hi[i]):.1f}" for i in idx)
        dn = " L".join(f"{xs(i):.1f},{ys(lo[i]):.1f}" for i in reversed(idx))
        out.append(f'<path class="area s1" d="M{up} L{dn} Z"/>')
        d = "M" + " L".join(f"{xs(i):.1f},{ys(mid[i]):.1f}" for i in idx)
        out.append(f'<path class="ser s1" d="{d}" fill="none"/>')
        i = idx[-1]
        out.append(f'<circle class="dot s1" cx="{xs(i):.1f}" cy="{ys(mid[i]):.1f}" r="4"/>')
    out.append(f'<line class="xh" x1="0" x2="0" y1="{mt}" y2="{height - mb}" visibility="hidden"/>')
    out.append(f'<rect class="hit" x="{ml}" y="{mt}" width="{width - ml - mr}" height="{height - mt - mb}" fill="transparent"/>')
    out.append("</svg>")
    rows = [f"<b>{esc(cell_label(cells[i]))}</b><br>{esc(name)} {esc(y_fmt(mid[i]))}<br>95% CI {esc(y_fmt(lo[i]))} … {esc(y_fmt(hi[i]))}"
            if not pd.isna(mid[i]) else f"<b>{esc(cell_label(cells[i]))}</b><br>window not yet full" for i in range(n)]
    data = json.dumps({"xs": [round(xs(i), 1) for i in range(n)], "rows": rows})
    return "".join(out) + f'<script type="application/json" id="{uid}-data">{data}</script>'


def h_bars_ref(rows, ref, ref_label, width=760, x_fmt=lambda v: f"{v * 100:.1f}%"):
    """rows: dict(label, value, emph(bool), why)."""
    ml, mr, mt, mb = 300, 70, 10, 26
    rh = 28
    height = mt + rh * len(rows) + mb
    vals = [r["value"] for r in rows if not pd.isna(r["value"])] + ([ref] if ref is not None else [])
    ticks = nice_ticks(0, max(vals) * 1.05, 5)
    xs = Lin(0, max(ticks), ml, width - mr)
    out = [svg_open(width, height, "hb")]
    for t in ticks:
        x = xs(t)
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{mt}" y2="{height - mb}"/>')
        out.append(f'<text class="ax" x="{x:.1f}" y="{height - 8}" text-anchor="middle">{esc(x_fmt(t))}</text>')
    for i, r in enumerate(rows):
        y = mt + rh * i + 4
        if pd.isna(r["value"]):
            continue
        w = xs(r["value"]) - ml
        cls = "emph" if r["emph"] else "dim"
        out.append(f'<text class="rowlab{" strong" if r["emph"] else ""}" x="{ml - 12}" y="{y + 14:.1f}" text-anchor="end">{esc(r["label"])}</text>')
        out.append(f'<path class="bar {cls}" d="M{ml},{y} H{ml + w - 4:.1f} Q{ml + w:.1f},{y} {ml + w:.1f},{y + 4} V{y + 16} Q{ml + w:.1f},{y + 20} {ml + w - 4:.1f},{y + 20} H{ml} Z" data-tip="{esc(r["why"])}"/>')
        out.append(f'<text class="lab" x="{ml + w + 8:.1f}" y="{y + 14:.1f}">{esc(x_fmt(r["value"]))}</text>')
    if ref is not None:
        xr = xs(ref)
        out.append(f'<line class="marker" x1="{xr:.1f}" x2="{xr:.1f}" y1="{mt - 4}" y2="{height - mb}"/>')
        out.append(f'<text class="lab strong" x="{xr + 6:.1f}" y="{mt + 4}">{esc(ref_label)}</text>')
    out.append("</svg>")
    return "".join(out)


def scatter(points, width=760, height=320, fmt=lambda v: f"{v * 100:.0f}%"):
    """points: dict(name, kind, ret, vol, n, sharpe). Sharpe line from RF through the composite."""
    ml, mr, mt, mb = 56, 24, 18, 36
    pts = [q for q in points if not pd.isna(q["ret"]) and not pd.isna(q["vol"])]
    xt = nice_ticks(0, max(q["vol"] for q in pts) * 1.12, 5)
    yt = nice_ticks(min(min(q["ret"] for q in pts), 0), max(q["ret"] for q in pts) * 1.12, 5)
    xs = Lin(min(xt), max(xt), ml, width - mr); ys = Lin(min(yt), max(yt), height - mb, mt)
    out = [svg_open(width, height, "sc")]
    for t in xt:
        out.append(f'<line class="grid" x1="{xs(t):.1f}" x2="{xs(t):.1f}" y1="{mt}" y2="{height - mb}"/>')
        out.append(f'<text class="ax" x="{xs(t):.1f}" y="{height - 16}" text-anchor="middle">{esc(fmt(t))}</text>')
    for t in yt:
        out.append(f'<line class="{"axis" if t == 0 else "grid"}" x1="{ml}" x2="{width - mr}" y1="{ys(t):.1f}" y2="{ys(t):.1f}"/>')
        out.append(f'<text class="ax" x="{ml - 8}" y="{ys(t) + 4:.1f}" text-anchor="end">{esc(fmt(t))}</text>')
    out.append(f'<text class="ax" x="{(ml + width - mr) / 2:.1f}" y="{height - 3}" text-anchor="middle">annualized volatility →</text>')
    out.append(f'<text class="ax" transform="translate(12,{(mt + height - mb) / 2:.1f}) rotate(-90)" text-anchor="middle">annualized return →</text>')
    rf = next((q for q in pts if q["kind"] == "rf"), None); cp = next((q for q in pts if q["kind"] == "composite"), None)
    if rf and cp and cp["vol"] > 0:
        slope = (cp["ret"] - rf["ret"]) / cp["vol"]
        x_end = max(xt); y_end = rf["ret"] + slope * x_end
        if y_end > max(yt):
            y_end = max(yt); x_end = (y_end - rf["ret"]) / slope
        out.append(f'<line class="sharpe" x1="{xs(0):.1f}" y1="{ys(rf["ret"]):.1f}" x2="{xs(x_end):.1f}" y2="{ys(y_end):.1f}" '
                   f'data-tip="Sharpe line: every point on it has the composite&#39;s Sharpe ratio ({cp["sharpe"]:.2f})"/>')
    order = {"account": 0, "rf": 1, "modelnet": 2, "benchmark": 3, "composite": 4}
    for q in sorted(pts, key=lambda q: order.get(q["kind"], 0)):
        x, y = xs(q["vol"]), ys(q["ret"])
        r = {"composite": 7, "benchmark": 6, "modelnet": 6, "rf": 4}.get(q["kind"], 4.5)
        tip = f"<b>{esc(q['name'])}</b><br>return {q['ret'] * 100:+.2f}%/yr · volatility {q['vol'] * 100:.1f}%/yr" + \
              (f" · Sharpe {q['sharpe']:.2f}" if not pd.isna(q.get("sharpe", np.nan)) else "") + f" · n {int(q['n'])}"
        out.append(f'<circle class="pt {q["kind"]}" cx="{x:.1f}" cy="{y:.1f}" r="{r}" data-tip="{esc(tip)}"/>')
        if q["kind"] in ("composite", "benchmark", "modelnet", "rf"):
            out.append(f'<text class="lab{" strong" if q["kind"] == "composite" else ""}" x="{x + r + 5:.1f}" y="{y + 4:.1f}">{esc(q["name"])}</text>')
    out.append("</svg>")
    return "".join(out)


def vol_by_year(years, vol_p, vol_b, width=760, height=220, name_b="benchmark"):
    """Composite volatility as columns, benchmark volatility as a grey line — one axis, one unit."""
    ml, mr, mt, mb = 52, 16, 14, 30
    n = len(years)
    vals = [v for v in list(vol_p) + list(vol_b) if not pd.isna(v)]
    if not vals:
        return ""
    yt = nice_ticks(0, max(vals) * 1.1, 5)
    ys = Lin(0, max(yt), height - mb, mt)
    band = (width - ml - mr) / max(n, 1); bw = min(24, band - 2)
    xc = lambda i: ml + band * i + band / 2
    out = [svg_open(width, height, "bars")]
    for t in yt:
        out.append(f'<line class="grid" x1="{ml}" x2="{width - mr}" y1="{ys(t):.1f}" y2="{ys(t):.1f}"/>')
        out.append(f'<text class="ax" x="{ml - 8}" y="{ys(t) + 4:.1f}" text-anchor="end">{t * 100:.0f}%</text>')
    y0 = ys(0)
    out.append(f'<line class="axis" x1="{ml}" x2="{width - mr}" y1="{y0:.1f}" y2="{y0:.1f}"/>')
    step = max(1, n // 10)
    for i, (y, v) in enumerate(zip(years, vol_p)):
        if not pd.isna(v):
            x = xc(i) - bw / 2; top = ys(v); r = 4 if (y0 - top) > 6 else 0
            path = (f"M{x:.1f},{y0:.1f} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} H{x + bw - r:.1f} "
                    f"Q{x + bw:.1f},{top:.1f} {x + bw:.1f},{top + r:.1f} V{y0:.1f} Z")
            out.append(f'<path class="bar emph" d="{path}" data-tip="{y}: composite volatility {v * 100:.1f}%/yr"/>')
        if i % step == 0:
            out.append(f'<text class="ax" x="{xc(i):.1f}" y="{height - 8}" text-anchor="middle">{esc(y)}</text>')
    pts = [(xc(i), ys(v)) for i, v in enumerate(vol_b) if not pd.isna(v)]
    if pts:
        out.append('<path class="ser s0" fill="none" d="M' + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
        for i, v in enumerate(vol_b):
            if not pd.isna(v):
                out.append(f'<circle class="dot s0" cx="{xc(i):.1f}" cy="{ys(v):.1f}" r="3.5" data-tip="{years[i]}: {esc(name_b)} volatility {v * 100:.1f}%/yr"/>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- page

def _read(out: Path, rel: str, **kw) -> pd.DataFrame:
    p = out / rel
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


def how(body: str, label: str = "How this was calculated") -> str:
    """Click-to-expand method note (native <details>, keyboard accessible)."""
    return f'<details class="how"><summary>{esc(label)}</summary><div class="howb">{body}</div></details>'


def build_dashboard(out_dir: str | Path, claimed: float | None = None, placeholder: bool = True,
                    placeholder_note: str = "placeholder data", fee_desc: str = "assumed fee schedule",
                    data_label: str = "", claimed_note: str = "", headline_model: str = "FF3") -> Path:
    out = Path(out_dir)
    comp = _read(out, "phase2/composite_returns.csv", index_col=0)
    monthly = len(comp) and "-" in str(comp.index[0])
    if monthly:
        comp.index = pd.PeriodIndex(comp.index, freq="M")
    else:
        comp.index = comp.index.astype(int)
    accts = _read(out, "phase2/account_summary.csv")
    st = _read(out, "statements_reconciled.csv")
    flags = _read(out, "flags.csv")
    gaps = _read(out, "gaps.csv")
    mat = _read(out, "coverage_matrix.csv", index_col=0)
    bd = _read(out, "phase3/beta_decomposition.csv", index_col=0)
    alloc = _read(out, "phase3/allocation.csv", index_col=0)
    regs = _read(out, "phase4/regressions.csv")
    metrics = _read(out, "phase4/metrics.csv")
    boot = _read(out, "phase4/bootstrap.csv")
    cohort = _read(out, "phase4/cohort.csv")
    samples = _read(out, "phase4/cohort_samples.csv")
    roll = _read(out, "phase4/rolling.csv", index_col=0)
    sub = _read(out, "phase4/subperiods.csv")
    rec = _read(out, "reconciliation.csv")
    rby = _read(out, "phase4/risk_by_year.csv", index_col=0)
    rrp = _read(out, "phase4/risk_return.csv")
    skill = _read(out, "phase4/skill.csv")

    ppy = 12 if monthly else 1
    unit = "month" if monthly else "year"
    cells = list(comp.index)
    g = series_stats(comp.gross, comp.years); b = series_stats(comp.benchmark, comp.years)
    n_ = series_stats(comp.model_net, comp.years)
    bench_id = str(accts.benchmark.mode().iloc[0]) if len(accts) and "benchmark" in accts else "benchmark"
    bench_name = {"US_MKT": "US market", "DEV_MKT": "Developed markets", "DXUS_MKT": "Developed ex-US"}.get(bench_id, bench_id)
    fset = regs.factor_set.iloc[0] if len(regs) else ""
    hm = headline_model if len(regs) and (regs.model == headline_model).any() else "CAPM"
    capm = regs[(regs.factor_set == fset) & (regs.model == hm)].iloc[0] if len(regs) else None      # headline
    jens = regs[(regs.factor_set == fset) & (regs.model == "CAPM")].iloc[0] if len(regs) else None    # Jensen
    coh = cohort.iloc[0] if len(cohort) else None
    bt = boot[boot.model == hm].iloc[0] if len(boot) and (boot.model == hm).any() else None
    hm_factors = {"CAPM": "the market premium", "FF3": "market, size (SMB) and value (HML) premia",
                  "Carhart4": "market, size, value and momentum premia", "FF5": "market, size, value, profitability and investment premia"}.get(hm, hm)

    tot = int(comp.n.sum()); ver = int(comp.n_verified.sum()); unv = int(comp.n_unverified.sum())
    n_flag = int(comp.n_excluded_flagged.sum())
    ver_share = ver / tot if tot else 0

    def chip(kind, text=None):
        names = {"verified": "Verified", "unverified": "Unverified", "estimated": "Estimated",
                 "unknown": "Unknown", "public": "Public data", "inference": "Inference"}
        return f'<span class="chip {kind}">{esc(text or names[kind])}</span>'

    series_chip = chip("verified", f"{ver_share:.0%} verified") if ver_share >= 0.5 else chip("unverified", f"{ver_share:.0%} verified")

    # ---- growth of $1
    idx_p = list(np.cumprod(1 + comp.gross.values)); idx_b = list(np.cumprod(1 + comp.benchmark.values))
    idx_n = list(np.cumprod(1 + comp.model_net.values))
    growth = line_chart(cells, [
        dict(name="Composite", values=idx_p, cls="s1"),
        dict(name=bench_name, values=idx_b, cls="s0"),
        dict(name="Model-net", values=idx_n, cls="s1l", dash=True),
    ], y_fmt=money, y_log=True, uid="growth")

    # ---- year by year excess (compound months to years when monthly)
    if monthly:
        yr = comp.assign(y=[c.year for c in comp.index]).groupby("y")
        full = yr.size() == 12
        ann = pd.DataFrame({"p": yr.gross.apply(lambda s: float((1 + s).prod() - 1)),
                            "b": yr.benchmark.apply(lambda s: float((1 + s).prod() - 1))})[full]
    else:
        ann = pd.DataFrame({"p": comp.gross, "b": comp.benchmark})
    ann["x"] = ann.p - ann.b
    ybars = diverging_bars([str(y) for y in ann.index], list(ann.x),
                           tips=[f"{y}: composite {pct(r.p)} · {bench_name} {pct(r.b)} · excess {pct(r.x)}" for y, r in ann.iterrows()])
    beat = int((ann.x > 0).sum())

    # ---- coverage
    cov = coverage_grid(mat, st) if len(mat) else "<p class='muted'>No coverage matrix.</p>"
    errs = flags[flags.severity == "error"] if len(flags) else flags

    # ---- attribution
    if len(bd):
        k = ppy
        segs = [("Risk-free", bd.rf_contribution.mean() * k, "s0"),
                ("Market exposure (β × market)", bd.beta_contribution.mean() * k, "s1"),
                ("Alpha + residual", bd.alpha_plus_residual.mean() * k, "s3")]
        attrib = stacked_h(segs)
        beta_txt = f"β {bd.beta.iloc[0]:.2f}"
    else:
        attrib, beta_txt = "<p class='muted'>Not computed.</p>", ""
    av = alloc[alloc.available == True] if len(alloc) and "available" in alloc else pd.DataFrame()

    # ---- regressions
    rows = []
    for _, r in regs.iterrows():
        if pd.isna(r.get("alpha_annual")):
            continue
        loads = " · ".join(f"β{f.replace('_RF', '')} {r[f'b_{f}']:.2f}" for f in ["MKT_RF", "SMB", "HML", "MOM", "RMW", "CMA"]
                           if f"b_{f}" in r and not pd.isna(r[f"b_{f}"]))
        rows.append(dict(label=f"{r.factor_set} · {r.model}", value=r.alpha_annual, lo=r.ci_low_annual, hi=r.ci_high_annual,
                         cls="s1" if r.factor_set == fset else "s0",
                         note=f"{r.factor_set} {r.model}: alpha {pct(r.alpha_annual)}/yr, t = {r.t:.2f}, p = {r.p:.3f}, R² {r.r2:.2f}, n = {int(r.n)}<br>{loads}"))
    dw = dot_whisker(rows) if rows else ""
    hist = ""
    if len(samples) and coh is not None:
        hist = histogram(samples.iloc[:, 0].dropna().values, float(coh.actual_headline_excess))

    # ---- rolling
    dcol = [c for c in roll.columns if c.startswith("excess_")]
    acol = [c for c in roll.columns if c.startswith("alpha_") and c.count("_") == 1]
    roll_html = ""
    if acol and len(roll):
        a = acol[0]; w_a = a.split("_")[1]
        rc = list(roll.index)
        if monthly:
            rc = [pd.Period(c, freq="M") for c in rc]
        roll_html = band_line(rc, list(roll[a]), list(roll[a + "_ci_low"]), list(roll[a + "_ci_high"]), uid="rollalpha",
                              name=f"{w_a}-{unit} alpha")
        w_d = dcol[0].split("_")[1] if dcol else None
        roll_ex = line_chart(rc, [dict(name=f"{w_d}-{unit} excess", values=list(roll[dcol[0]]), cls="s1")],
                             y_fmt=lambda v: f"{v * 100:+.0f}%", end_labels=False, uid="rollex", height=200) if dcol else ""
        alp = roll[a].dropna(); sig = (roll.loc[alp.index, a + "_ci_low"] > 0).sum()
        roll_note = (f"{w_a}-{unit} rolling alpha positive in {int((alp > 0).sum())} of {len(alp)} windows; "
                     f"95% CI excluded zero in {int(sig)}.")
    else:
        roll_ex, roll_note, w_a = "", "", ""
    sp = sub[sub.model == "CAPM"] if len(sub) else sub

    # ---- reconciliation
    rec_html = ""
    if len(rec):
        rr = [dict(label=r.calculation, value=r.value, emph=bool(r.is_verified), why=r.why) for _, r in rec.iterrows()]
        rec_html = h_bars_ref(rr, claimed, f"claimed {pct(claimed, 1, False)}" if claimed is not None else "")

    # ---- risk: scatter, vol by year, VaR table
    sc = scatter(rrp.to_dict("records")) if len(rrp) else ""
    within = len(rby) and rby.vol_p.notna().any()
    if len(rby):
        vp = list(rby.vol_p if within else rby.trailing_vol_p); vb = list(rby.vol_b if within else rby.trailing_vol_b)
        vol_chart = vol_by_year([str(y) for y in rby.index], vp, vb, name_b=bench_name)
        if within:
            var_head = "<tr><th>year</th><th class='n'>return</th><th class='n'>volatility</th><th class='n'>variance</th><th class='n'>VaR 95% hist</th><th class='n'>VaR 95% param</th><th class='n'>ES 95%</th><th class='n'>worst month</th><th class='n'>bench return</th><th class='n'>bench VaR hist</th></tr>"
            var_rows = "".join(f"<tr><td>{y}</td><td class='n'>{pct(r.return_p)}</td><td class='n'>{pct(r.vol_p, 1, False)}</td><td class='n'>{pct(r.variance_p, 2, False)}</td>"
                               f"<td class='n loss'>{pct(r.var95_hist_p)}</td><td class='n loss'>{pct(r.var95_param_p)}</td><td class='n loss'>{pct(r.es95_p)}</td><td class='n loss'>{pct(r.worst_p)}</td>"
                               f"<td class='n'>{pct(r.return_b)}</td><td class='n loss'>{pct(r.var95_hist_b)}</td></tr>" for y, r in rby.iterrows())
            var_note = ("VaR is the 95% one-month loss threshold from that year's twelve monthly returns: <em>historical</em> is the 5th percentile "
                        "(between the worst and second-worst month — coarse by construction), <em>parametric</em> is mean − 1.645σ. "
                        "Expected shortfall is the average of the months at or below the historical VaR.")
        else:
            tw = int(rby.attrs.get("trailing", 10)) if rby.attrs else 10
            var_head = "<tr><th>year</th><th class='n'>return</th><th class='n'>trailing volatility</th><th class='n'>trailing VaR 95% (one year)</th><th class='n'>trailing ES 95%</th><th class='n'>bench return</th><th class='n'>bench trailing VaR</th></tr>"
            var_rows = "".join(f"<tr><td>{y}</td><td class='n'>{pct(r.return_p)}</td><td class='n'>{pct(r.trailing_vol_p, 1, False)}</td><td class='n loss'>{pct(r.trailing_var95_hist_p)}</td>"
                               f"<td class='n loss'>{pct(r.trailing_es95_p)}</td><td class='n'>{pct(r.return_b)}</td><td class='n loss'>{pct(r.trailing_var95_hist_b)}</td></tr>" for y, r in rby.iterrows())
            var_note = ("Within-year volatility and VaR are <b>not observable from annual statements</b> — there is one number per year. "
                        "Shown instead: each figure over the trailing ten years of annual returns. Monthly custodian data would make the within-year table available.")
    else:
        vol_chart = var_head = var_rows = var_note = ""

    # ---- skill vs luck table
    skill_rows = ""
    for _, k in skill.iterrows():
        v = {"pct": pct(k.value), "num": num(k.value), "pctile": f"{k.value:.0f}th"}.get(k.fmt, num(k.value))
        bv = "" if pd.isna(k.benchmark) else {"pct": pct(k.benchmark), "num": num(k.benchmark), "pctile": ""}.get(k.fmt, "")
        yes = str(k.skill).startswith("Yes") or str(k.skill).startswith("The direct")
        skill_rows += (f"<tr class='{'yes' if yes else ''}'><td><b>{esc(k.ratio)}</b><br><span class='muted'><code>{esc(k.formula)}</code></span></td>"
                       f"<td class='n'>{v}</td><td class='n'>{bv}</td><td>{esc(k.measures)}</td><td>{esc(k.skill)}</td></tr>")

    # ---- metrics table
    def mfmt(row, col):
        v = row[col]
        if row.fmt == "str": return esc(v)
        v = pd.to_numeric(v, errors="coerce")
        if pd.isna(v): return "n/a"
        if row.fmt == "pct": return pct(v)
        if row.fmt == "int": return str(int(v))
        return num(v)
    met_rows = "".join(f"<tr><td>{esc(m.metric)}</td><td class='n'>{mfmt(m, 'portfolio')}</td><td class='n'>{mfmt(m, 'benchmark')}</td></tr>"
                       for _, m in metrics.iterrows()) if len(metrics) else ""

    # ---- accounts table
    acc_rows = ""
    for _, a in accts.iterrows():
        acc_rows += ("<tr>"
                     f"<td><code>{esc(a.account_id)}</code><br><span class='muted'>{esc(a.label)}</span></td>"
                     f"<td>{esc(a.owner_type)}</td><td><span class='pill {esc(a.status)}'>{esc(a.status)}</span></td>"
                     f"<td class='n'>{esc(a.twr_span)}</td><td class='n'>{pct(a.twr_annualized)}</td>"
                     f"<td class='n'>{pct(a.bench_annualized)}</td><td class='n'>{pct(a.irr)}</td>"
                     f"<td class='n'>{int(a.verified)} / {int(a.unverified)} / {int(a.flagged)}</td></tr>")

    gap_txt = "; ".join(f"{r.account_id} {r.gap_start}→{r.gap_end}" for _, r in gaps.iterrows()) if len(gaps) else "none"
    err_items = "".join(f"<li><code>{esc(r.statement_id if isinstance(r.statement_id, str) else r.account_id)}</code> "
                        f"<b>{esc(r.code)}</b> — {esc(r.detail)}</li>" for _, r in errs.head(8).iterrows()) if len(errs) else "<li>none</li>"

    claimed_block = ""
    if claimed is not None:
        gap = g["annualized"] - claimed
        claimed_block = (f'<div class="tile"><div class="tl">Claimed vs verified</div>'
                         f'<div class="tv small">{pct(claimed, 1, False)} <span class="arrow">→</span> {pct(g["annualized"], 1, False)}</div>'
                         f'<div class="td {"bad" if gap < 0 else "good"}">{pct(gap)}/yr gap</div>'
                         + how("<b>Claimed</b> is the number the manager states — an input to this pipeline, not something it computes. "
                               "<b>Verified</b> is the composite time-weighted return computed here from the statements. The gap is the "
                               "arithmetic difference per year; the <i>Why not…?</i> section below recomputes the usual ways a self-calculated "
                               "figure ends up higher than the verified one." + (f"<br><br>{esc(claimed_note)}" if claimed_note else ""), "What do these mean?")
                         + '</div>')

    banner = ""
    if placeholder:
        banner = (f'<div class="banner" role="note"><b>Placeholder data.</b> {esc(placeholder_note)}. '
                  f'Nothing on this page describes the record under verification.</div>')

    first, last = cell_label(cells[0]), cell_label(cells[-1])
    css = CSS
    page = f"""<title>Track Record Verification</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}</style>
{banner}
<main class="wrap">
<header class="head">
  <div>
    <h1>Track Record Verification</h1>
    <p class="sub">{esc(data_label or 'composite of all discretionary accounts')} · {esc(first)} → {esc(last)} · {len(cells)} {unit}s · generated {date.today()}</p>
  </div>
  <div class="legend-tiers">
    {chip("verified")} at least one independent check passed &nbsp; {chip("unverified")} value exists, nothing confirms it &nbsp; {chip("estimated")} assumed or inferred &nbsp; {chip("unknown")} not observable
  </div>
</header>

<section class="verdict">
  <div class="hero">
    <div class="tl">{"Verified composite return, annualized" if ver_share >= 0.5 else "Composite return, annualized — unverified"}</div>
    <div class="hv">{pct(g["annualized"], 2, False)}</div>
    <div class="hs">time-weighted · gross (no fees were charged) · {esc(first)} → {esc(last)}</div>
    <div class="meter" title="share of account-{unit}s by evidence status">
      <span class="m verified" style="width:{ver_share * 100:.1f}%"></span><span class="m unverified" style="width:{(unv / tot * 100) if tot else 0:.1f}%"></span><span class="m flagged" style="width:{(n_flag / (tot + n_flag) * 100) if tot else 0:.1f}%"></span>
    </div>
    <div class="ms"><b>{ver:,}</b> verified · <b>{unv:,}</b> unverified · <b>{n_flag:,}</b> flagged &amp; excluded — of {tot + n_flag:,} account-{unit}s</div>
    {how(f"<b>Per {unit}, per account:</b> Modified Dietz return = (End − Begin − net flows) ÷ (Begin + Σ w<sub>i</sub> × flow<sub>i</sub>), where w<sub>i</sub> is the fraction of the {unit} the flow was invested. Deposits and withdrawals are therefore neither counted as gains nor losses.<br><br>"
          f"<b>Composite:</b> all eligible accounts' values and flows are summed first, as if one portfolio (GIPS aggregate method), so larger accounts weigh more. Closed accounts stay in through their last full {unit}; partial first and last {unit}s and flagged statements are excluded.<br><br>"
          f"<b>Annualized:</b> the {unit}ly returns are chained, (1+r<sub>1</sub>)(1+r<sub>2</sub>)…, and the total is converted to a per-year rate. Gross = net here because no fee was ever charged.<br><br>"
          f"<b>The meter:</b> each statement is <i>verified</i> when at least one number printed on it was independently confirmed (positions sum, chain to prior ending value, printed change in value, printed flows or printed return) and none failed; <i>flagged</i> if any check failed; <i>unverified</i> if nothing on the page could be tested.")}
  </div>
  <div class="tiles">
    <div class="tile"><div class="tl">{esc(bench_name)}</div><div class="tv">{pct(b["annualized"], 2, False)}</div><div class="td muted">{esc(bench_id)} · same period</div>
      {how(f"Ken French's {esc(bench_name)} total return (Mkt−RF plus RF) for exactly the same {unit}s as the composite, chained and annualized the same way. Fixed before any data was examined (COMPOSITE_RULES.md §6).", "How?")}</div>
    <div class="tile"><div class="tl">Excess over benchmark</div><div class="tv">{pct(g["annualized"] - b["annualized"])}</div><div class="td muted">per year, gross</div>
      {how("Annualized composite minus annualized benchmark. A plain difference — it does not adjust for the composite taking more or less market risk than the benchmark; the alphas below do.", "How?")}</div>
    <div class="tile"><div class="tl">{esc(hm)} alpha</div><div class="tv">{pct(capm.alpha_annual) if capm is not None else "n/a"}</div><div class="td muted">{f"95% CI {pct(capm.ci_low_annual, 1)} … {pct(capm.ci_high_annual, 1)} · t {capm.t:.2f} · p {capm.p:.3f}" if capm is not None else ""}</div>
      {how(f"Fama–French regression: each {unit}'s return over the risk-free rate is regressed on {esc(hm_factors)} (Ken French {esc(fset)} factors). The intercept, × {ppy} per year, is the alpha — the part of the return an index fund holding those factor exposures could not have delivered. Confidence interval = ±t<sub>0.975</sub> × standard error{'; Newey–West errors on monthly data' if ppy == 12 else ''}.", "How?")}</div>
    <div class="tile"><div class="tl">Jensen's alpha (CAPM)</div><div class="tv">{pct(jens.alpha_annual) if jens is not None else "n/a"}</div><div class="td muted">{f"95% CI {pct(jens.ci_low_annual, 1)} … {pct(jens.ci_high_annual, 1)} · β {jens.b_MKT_RF:.2f}" if jens is not None else ""}</div>
      {how(f"The same regression with only the market premium: R<sub>p</sub> − RF = α + β(Mkt − RF) + ε. Its intercept is Jensen's alpha. It is larger than the {esc(hm)} alpha whenever part of the return came from size or value tilts — exposures an index fund could buy, which {esc(hm)} removes.", "How?")}</div>
    <div class="tile"><div class="tl">Zero-skill managers doing this well</div><div class="tv">{f"{coh.share_of_zero_skill_managers_beating_actual:.1%}" if coh is not None else "n/a"}</div><div class="td muted">{f"of {int(coh.n_managers):,} simulated, same {esc(coh.model)} loadings &amp; residual risk" if coh is not None else ""}</div>
      {how(f"{int(coh.n_managers):,} simulated managers with <b>no skill</b>: alpha set to zero, the record's own {esc(coh.model)} factor loadings ({esc(coh.loadings)}), and residual noise resampled from the record's own residuals, facing the same market history. Each one's annualized excess over the benchmark is computed; this is the share that matched or beat the real record by luck alone." if coh is not None else "", "How?")}</div>
    <div class="tile"><div class="tl">Model-net at proposed fees</div><div class="tv">{pct(n_["annualized"], 2, False)}</div><div class="td muted">{esc(fee_desc)} — {chip("estimated")}</div>
      {how(f"The gross series with a modelled fee subtracted each {unit}: {esc(fee_desc)}. The management fee is pro-rated per {unit}; the performance fee is charged only on gains above the previous high-water mark. No fee was actually charged historically — this is what a paying client would have received.", "How?")}</div>
    {claimed_block}
  </div>
</section>

<section>
  <div class="sh"><h2>Growth of $1</h2>{series_chip} {chip("public", bench_name)} {chip("estimated", "model-net")}</div>
  <div class="card">{growth}
  <p class="cap">Log scale. Composite chains each {unit}'s Modified Dietz return; the benchmark is {esc(bench_name)} total return (Ken French); model-net applies the proposed fee to the gross series.</p>
  {how(f"Each line is the cumulative product of (1 + return) per {unit}, starting from $1 at {esc(first)}. The log scale makes equal percentage moves the same height anywhere on the chart, so the slope is the growth rate. Hover for the three values at any {unit}.")}</div>
</section>

<section>
  <div class="sh"><h2>Year by year against the benchmark</h2>{series_chip}</div>
  <div class="card">{ybars}
  <p class="cap">Excess return per calendar year. Above the benchmark in <b>{beat}</b> of <b>{len(ann)}</b> years. Hover a bar for both returns.</p>
  {how(f"For each calendar year: composite return {'(the twelve months compounded)' if monthly else ''} minus the benchmark's return for the same year. Blue = beat the benchmark, red = trailed it. {'Years with fewer than twelve months are omitted.' if monthly else ''}")}</div>
</section>

<section>
  <div class="sh"><h2>Risk and return</h2>{series_chip} {chip("public", bench_name)}</div>
  <div class="grid2">
    <div class="card"><h3>Volatility vs return</h3>{sc}
      <p class="cap"><span class="k s1"></span>composite &nbsp; <span class="k s0"></span>{esc(bench_name)}, accounts &nbsp; <span class="k s1l"></span>model-net. The thin line from the risk-free point through the composite is its Sharpe ratio; anything above the line earned more per unit of risk.</p>
      {how(f"x = annualized volatility = standard deviation of {unit}ly returns × √{ppy}. y = annualized return (chained). Each account is plotted over its own eligible {unit}s, so spans differ. Sharpe = (return − risk-free) ÷ volatility; the line's slope is the composite's Sharpe.")}</div>
    <div class="card"><h3>{"Volatility by year (from monthly returns)" if within else "Trailing volatility by year"}</h3>{vol_chart}
      <p class="cap">Columns: composite. Line: {esc(bench_name)}. Same axis, same unit.</p>
      {how("Standard deviation of that year's twelve monthly returns × √12." if within else "Standard deviation of the trailing ten annual returns — within-year volatility is not observable from annual statements.")}</div>
  </div>
  <div class="card"><h3>{"Value at risk by year" if within else "Value at risk, trailing window"}</h3>
    <div class="tscroll"><table class="metrics var"><thead>{var_head}</thead><tbody>{var_rows}</tbody></table></div>
    <p class="cap">{var_note}</p>
    {how("<b>Historical VaR 95%</b>: the 5th percentile of the period's returns — the loss exceeded one time in twenty. <b>Parametric VaR</b>: mean − 1.645 × standard deviation, assuming normality. <b>Expected shortfall</b>: the average return of the periods at or below the historical VaR — how bad the bad tail was, not just where it starts. <b>Variance</b> = volatility². All reported as return levels: negative = loss.")}
  </div>
  <div class="card"><h3>Risk-adjusted metrics, full period</h3>
    <div class="tscroll"><table class="metrics"><thead><tr><th>metric</th><th class="n">portfolio</th><th class="n">benchmark</th></tr></thead><tbody>{met_rows}</tbody></table></div>
    {how(f"<b>Sharpe</b> = mean excess return over RF ÷ its standard deviation, × √{ppy}. <b>Sortino</b>: same numerator over downside deviation only (returns below RF). <b>Max drawdown</b>: largest peak-to-trough fall in the chained index{' — from ' + unit + 'ly points, so intra-' + unit + ' lows are missed' if not monthly else ''}. <b>Calmar</b> = annualized return ÷ |max drawdown|. <b>Tracking error</b> = standard deviation of (composite − benchmark). <b>Information ratio</b> = mean(composite − benchmark) ÷ tracking error. <b>Up/down capture</b>: the composite's compounded return in {unit}s the benchmark rose (fell), divided by the benchmark's.")}
  </div>
</section>

<section>
  <div class="sh"><h2>Evidence</h2>{chip("verified", "V")} {chip("unverified", "U")} {chip("unknown", "X flagged")} {chip("unknown", "– hole")}</div>
  <div class="card">{cov}
  <div class="two">
    <div><h3>Reconciliation failures (excluded until re-read)</h3><ul class="flags">{err_items}</ul>{f"<p class='muted'>… and {len(errs) - 8} more in flags.csv</p>" if len(errs) > 8 else ""}</div>
    <div><h3>Holes in coverage</h3><p>{esc(gap_txt)}</p><h3>What "verified" means here</h3><p>A statement is verified when at least one number printed on it is independently confirmed — the sum of its positions, its printed beginning value against the prior statement's ending, its printed change-in-value, or its printed flows. A value with nothing to test it against is <em>unverified</em>, even when it's probably right.</p></div>
  </div></div>
</section>

<section>
  <div class="sh"><h2>Where the return came from</h2>{chip("public", "market data")} {chip("estimated")}</div>
  <div class="card">{attrib}
  <p class="cap">Average per year. {esc(beta_txt)} vs {esc(bench_name)}: the middle segment is what simply holding that much market exposure returned; the last is everything else — stock selection, timing, and noise together.</p>
  {how(f"R<sub>p</sub> = RF + β × (Market − RF) + (α + ε). β is the full-sample CAPM slope; each {unit}'s return is split into the risk-free rate, β × that {unit}'s market premium, and the remainder. Segments are the per-{unit} averages × {ppy}. The allocation effect, where positions exist, compares the portfolio's US / international / cash weights (prior year-end positions) against the benchmark's inferred mix, each sleeve indexed.")}
  {f"<p class='cap'>Allocation effect (US / international / cash mix vs policy): <b>{pct(av.allocation_effect.mean() * ppy)}</b>/yr on average; selection + timing residual <b>{pct(av.selection_timing_residual.mean() * ppy)}</b>/yr. Weights come from year-end positions and cover {av.coverage.mean():.0%} of composite assets.</p>" if len(av) else "<p class='cap muted'>Allocation effect not available: no position data for this dataset, so US / international / cash weights are unknown.</p>"}
  </div>
</section>

<section>
  <div class="sh"><h2>Is it real? Skill or luck</h2>{chip("inference")}</div>
  <div class="card"><h3>Which ratio answers it</h3>
    <p class="cap" style="max-width:none">Risk-adjusted ratios measure return per unit of risk; none of them can tell skill from luck — a leveraged index fund scores well on all of them. The question is statistical: how consistently did the alpha show up relative to its own noise? The <b>appraisal ratio</b> (alpha ÷ residual volatility) carries exactly that, because <b>t ≈ appraisal ratio × √years</b>. Highlighted rows are the ones that speak to skill.</p>
    <div class="tscroll"><table class="metrics skill"><thead><tr><th>ratio</th><th class="n">portfolio</th><th class="n">benchmark</th><th>measures</th><th>speaks to skill?</th></tr></thead><tbody>{skill_rows}</tbody></table></div>
    {how("Every ratio uses the same annualized inputs as the tiles above. Appraisal ratio = alpha ÷ standard deviation of the regression residuals (both annualized). Because the standard error of alpha is roughly residual volatility ÷ √n, alpha ÷ SE(alpha) — the t-statistic — equals the appraisal ratio × √years, up to a small correction for the factor means. The bootstrap p-value re-runs the regression thousands of times on resampled " + unit + "s with the alpha removed, and counts how often chance produces a t this large.")}
  </div>
  <div class="grid2">
    <div class="card"><h3>Alpha with 95% confidence intervals</h3>{dw}
      <p class="cap"><span class="k s1"></span>{esc(fset)} factors (primary) &nbsp; <span class="k s0"></span>robustness set. A whisker that crosses zero means the model cannot rule out zero skill. Loadings on hover.</p>
      {how(f"Each row is a separate regression of (return − RF) on that model's factors over all {unit}s. Dot = annualized intercept (alpha); whisker = 95% confidence interval. CAPM = market only (Jensen's alpha); FF3 adds size and value; Carhart adds momentum; FF5 adds profitability and investment. The primary set is the {esc(fset)} factors; the other set is a robustness check.")}</div>
    <div class="card"><h3>Could luck alone do this?</h3>{hist}
      <p class="cap">{f"{int(coh.n_managers):,} simulated managers with zero skill, the record's own {esc(coh.model)} loadings and residual volatility ({pct(coh.resid_sd_annual, 1, False)}/yr), facing the same markets. <b>{coh.share_of_zero_skill_managers_beating_actual:.1%}</b> matched or beat the record's excess return by luck; null bootstrap of t(α): p = {bt.p_null_one_sided:.3f}." if coh is not None and bt is not None else ""}</p>
      {how(f"Histogram of the simulated managers' annualized excess returns over the benchmark. Grey bars: managers who did worse than the record; blue bars: the share that did as well or better by luck alone. The vertical line is the actual record.")}</div>
  </div>
</section>

<section>
  <div class="sh"><h2>Is it stable?</h2>{chip("inference")} {chip("estimated", "noisy windows")}</div>
  <div class="grid2">
    <div class="card"><h3>Rolling {esc(w_a)}-{unit} alpha, with 95% band</h3>{roll_html}<p class="cap">{esc(roll_note)}</p>
      {how(f"A CAPM regression re-run on each trailing window of {esc(w_a)} {unit}s. The line is that window's annualized alpha; the band is its 95% confidence interval. With so few points per window the band is wide by construction — the noise floor is stated in the sub-period card.")}</div>
    <div class="card"><h3>Rolling excess return</h3>{roll_ex}<p class="cap">Annualized excess over {esc(bench_name)} for each trailing window — descriptive, no model.</p>
      {how("For each trailing window: annualized composite return minus annualized benchmark return, both chained over the window. No regression, no risk adjustment.")}</div>
  </div>
  <div class="card"><h3>Before and after {esc(sp.segment.iloc[0].split()[-1]) if len(sp) else ""}</h3>
    <div class="tiles">{"".join(f'<div class="tile"><div class="tl">{esc(r.segment)}</div><div class="tv">{pct(r.alpha_annual)}</div><div class="td muted">SE {pct(r.se_annual, 1, False)} · n {int(r.n)} · p {r.p:.2f}</div></div>' for _, r in sp.iterrows())}</div>
    <p class="cap">CAPM alpha per year in each half. With standard errors this size, halves of a record with constant true alpha will differ by more than {pct(2 * sp[sp.segment.str.startswith('difference')].se_annual.iloc[0], 1, False) if len(sp) else ""} about one time in twenty — read the difference against that, not against zero.</p>
    {how("One regression with an interaction: (return − RF) = α₁ + α₂·D + β₁(Mkt−RF) + β₂·D·(Mkt−RF) + ε, where D = 1 after the split year. α₁ is the early alpha, α₁ + α₂ the late alpha, and the t-test on α₂ asks whether the change is larger than noise would produce.")}
  </div>
</section>

<section>
  <div class="sh"><h2>{"Why not " + pct(claimed, 1, False) + "?" if claimed is not None else "Other ways to compute the number"}</h2>{chip("verified", "same data")}</div>
  <div class="card">{rec_html}
  <p class="cap">Every bar is computed from this dataset. The highlighted bar is the defensible number; the others are the calculations a self-computed figure most often rests on. Hover for why each differs.</p>
  {how("<b>Naive CAGR</b>: (ending value ÷ beginning value)^(1/years) − 1, ignoring deposits — so contributions look like gains. <b>Arithmetic mean</b>: the simple average of annual returns, which overstates compounded growth by roughly half the variance. <b>Survivors-only</b>: the composite with closed accounts dropped. <b>Best single account / last 10 years</b>: cherry-picked scope or start date. <b>Model-net</b>: after the proposed fee. The claimed figure is drawn as the vertical line.")}</div>
</section>

<section>
  <div class="sh"><h2>Accounts</h2></div>
  <div class="card tscroll"><table class="acc"><thead><tr><th>account</th><th>owner</th><th>status</th><th class="n">span</th><th class="n">TWR /yr</th><th class="n">benchmark /yr</th><th class="n">IRR</th><th class="n">V / U / X</th></tr></thead><tbody>{acc_rows}</tbody></table>
  <p class="cap">Closed accounts stay in the composite through their last full {unit}. IRR is money-weighted and reflects the client's deposit timing, not the manager.</p>
  {how(f"<b>TWR /yr</b>: each account's own {unit}ly Modified Dietz returns chained over its span and annualized — the manager's return, independent of when money arrived. <b>IRR</b>: the single rate that discounts every dated deposit, withdrawal and the ending value to zero (XIRR) — the client's actual experience, which depends on deposit timing. <b>V / U / X</b>: statements verified / unverified / flagged.")}</div>
</section>

<footer class="foot">
  <p>Methodology: COMPOSITE_RULES.md · Phase detail: coverage.md, phase2/summary.md, phase3/summary.md, phase4/summary.md · Nothing interpolated, smoothed, or corrected; flagged statements are excluded, not fixed · In-sample: the confidence interval, not the point estimate, is the claim.</p>
</footer>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}</script>
"""
    path = out / "dashboard.html"
    path.write_text(page)
    return path


CSS = r"""
:root {
  color-scheme: light;
  --page: #f3f5f7; --surface: #ffffff; --surface-2: #f7f8fa; --line: #e3e6ea; --hair: #eceef1;
  --ink: #141a22; --ink-2: #4a5462; --muted: #8a93a0;
  --accent: #2a78d6; --accent-l: #86b6ef; --accent-wash: rgba(42,120,214,.10);
  --dim: #9aa1ab; --dim-wash: rgba(154,161,171,.22);
  --s3: #1baf7a;
  --pos: #2a78d6; --neg: #e34948;
  --good: #0ca30c; --good-wash: rgba(12,163,12,.16); --warn: #d99a00; --warn-wash: rgba(250,178,25,.22);
  --crit: #d03b3b; --crit-wash: rgba(208,59,59,.16);
  --banner-bg: #fff4d6; --banner-ink: #5c4300; --banner-line: #f0c96a;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0f1317; --surface: #171c22; --surface-2: #1d232a; --line: #2a313a; --hair: #232a32;
    --ink: #eef1f4; --ink-2: #b6bec8; --muted: #7f8893;
    --accent: #3987e5; --accent-l: #5598e7; --accent-wash: rgba(57,135,229,.16);
    --dim: #6f7781; --dim-wash: rgba(111,119,129,.28);
    --s3: #199e70; --pos: #3987e5; --neg: #e66767;
    --good: #22b822; --good-wash: rgba(34,184,34,.20); --warn: #f0b429; --warn-wash: rgba(240,180,41,.22);
    --crit: #e05252; --crit-wash: rgba(224,82,82,.20);
    --banner-bg: #3a2e08; --banner-ink: #ffe6a3; --banner-line: #7a5f10;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0f1317; --surface: #171c22; --surface-2: #1d232a; --line: #2a313a; --hair: #232a32;
  --ink: #eef1f4; --ink-2: #b6bec8; --muted: #7f8893;
  --accent: #3987e5; --accent-l: #5598e7; --accent-wash: rgba(57,135,229,.16);
  --dim: #6f7781; --dim-wash: rgba(111,119,129,.28);
  --s3: #199e70; --pos: #3987e5; --neg: #e66767;
  --good: #22b822; --good-wash: rgba(34,184,34,.20); --warn: #f0b429; --warn-wash: rgba(240,180,41,.22);
  --crit: #e05252; --crit-wash: rgba(224,82,82,.20);
  --banner-bg: #3a2e08; --banner-ink: #ffe6a3; --banner-line: #7a5f10;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 14px/1.5 "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1240px; margin: 0 auto; padding: 20px 24px 48px; display: grid; gap: 28px; }
.banner { position: sticky; top: 0; z-index: 5; background: var(--banner-bg); color: var(--banner-ink); border-bottom: 1px solid var(--banner-line); padding: 10px 24px; font-size: 13px; }
.banner b { font-weight: 600; }
.head { display: flex; justify-content: space-between; gap: 24px; align-items: flex-end; flex-wrap: wrap; }
h1 { font: 600 30px/1.15 "IBM Plex Serif", Georgia, serif; margin: 0; letter-spacing: -.01em; text-wrap: balance; }
h2 { font: 500 20px/1.2 "IBM Plex Serif", Georgia, serif; margin: 0; }
h3 { font: 600 13px/1.3 "IBM Plex Sans", system-ui, sans-serif; margin: 0 0 8px; color: var(--ink-2); text-transform: uppercase; letter-spacing: .06em; }
.sub { margin: 4px 0 0; color: var(--ink-2); }
.legend-tiers { font-size: 12px; color: var(--ink-2); display: flex; flex-wrap: wrap; gap: 6px 4px; align-items: center; max-width: 560px; }
.chip { display: inline-block; font-size: 11px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; padding: 2px 8px; border-radius: 999px; line-height: 16px; vertical-align: middle; }
.chip.verified { background: var(--good-wash); color: var(--good); }
.chip.unverified { background: var(--warn-wash); color: var(--warn); }
.chip.estimated { background: var(--accent-wash); color: var(--accent); }
.chip.unknown { background: var(--dim-wash); color: var(--ink-2); }
.chip.public, .chip.inference { background: var(--surface-2); color: var(--ink-2); border: 1px solid var(--line); }
section { display: grid; gap: 12px; }
.sh { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 18px 20px; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 12px; }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; margin-top: 16px; }
.verdict { display: grid; grid-template-columns: minmax(300px, 1.1fr) 2fr; gap: 12px; }
@media (max-width: 900px) { .verdict { grid-template-columns: 1fr; } }
.hero { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 22px 24px; display: grid; gap: 6px; align-content: start; }
.hv { font-size: 56px; font-weight: 600; line-height: 1; letter-spacing: -.02em; }
.hs { color: var(--ink-2); font-size: 13px; }
.tl { font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; display: grid; gap: 4px; align-content: start; }
.tv { font-size: 26px; font-weight: 600; line-height: 1.1; letter-spacing: -.01em; }
.tv.small { font-size: 22px; white-space: nowrap; }
.td { font-size: 12px; color: var(--ink-2); }
.td.bad { color: var(--crit); font-weight: 600; } .td.good { color: var(--good); font-weight: 600; }
.arrow { color: var(--muted); font-weight: 400; }
.meter { display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: var(--dim-wash); margin-top: 10px; gap: 2px; }
.m { display: block; height: 100%; } .m.verified { background: var(--good); } .m.unverified { background: var(--warn); } .m.flagged { background: var(--crit); }
.ms { font-size: 12px; color: var(--ink-2); }
.cap { margin: 10px 0 0; font-size: 12.5px; color: var(--ink-2); max-width: 78ch; }
.muted { color: var(--muted); }
.strong { font-weight: 600; }
code { font: 12px "IBM Plex Mono", ui-monospace, Menlo, monospace; }
ul.flags { margin: 0; padding-left: 18px; font-size: 12.5px; } ul.flags li { margin: 4px 0; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { text-align: left; font-size: 11px; letter-spacing: .05em; text-transform: uppercase; color: var(--muted); font-weight: 600; padding: 6px 8px; border-bottom: 1px solid var(--line); }
td { padding: 7px 8px; border-bottom: 1px solid var(--hair); vertical-align: top; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.tscroll { overflow-x: auto; }
.pill { font-size: 11px; padding: 1px 8px; border-radius: 999px; background: var(--surface-2); border: 1px solid var(--line); }
.pill.closed { background: var(--dim-wash); }
.foot { color: var(--muted); font-size: 12px; border-top: 1px solid var(--line); padding-top: 14px; }
.foot p { max-width: 110ch; margin: 0; }
/* charts */
.chart { display: block; font-family: "IBM Plex Sans", system-ui, sans-serif; }
.chart text { fill: var(--ink-2); font-size: 11px; }
.chart .ax { fill: var(--muted); font-variant-numeric: tabular-nums; }
.chart .rowlab { fill: var(--ink-2); font-size: 12px; } .chart .rowlab.strong { fill: var(--ink); font-weight: 600; }
.chart .lab { fill: var(--ink); font-size: 11.5px; font-variant-numeric: tabular-nums; } .chart .lab.strong { font-weight: 600; }
.chart .grid { stroke: var(--hair); stroke-width: 1; }
.chart .axis { stroke: var(--line); stroke-width: 1; } .chart .axis.zero { stroke: var(--ink-2); }
.chart .ser { stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.chart .ser.s1 { stroke: var(--accent); } .chart .ser.s0 { stroke: var(--dim); } .chart .ser.s1l { stroke: var(--accent-l); }
.chart .area.s1 { fill: var(--accent); fill-opacity: .10; stroke: none; }
.chart .dot { stroke: var(--surface); stroke-width: 2; } .chart .dot.s1 { fill: var(--accent); } .chart .dot.s0 { fill: var(--dim); } .chart .dot.s1l { fill: var(--accent-l); }
.chart .whisk { stroke-width: 2; stroke-linecap: round; } .chart .whisk.s1 { stroke: var(--accent); } .chart .whisk.s0 { stroke: var(--dim); }
.chart .bar.pos { fill: var(--pos); } .chart .bar.neg { fill: var(--neg); }
.chart .bar.emph { fill: var(--accent); } .chart .bar.dim { fill: var(--dim); }
.chart .hbar.emph { fill: var(--accent); } .chart .hbar.dim { fill: var(--dim); fill-opacity: .55; }
.chart .marker { stroke: var(--ink); stroke-width: 1.5; }
.chart .seg.s0 { fill: var(--dim); } .chart .seg.s1 { fill: var(--accent); } .chart .seg.s3 { fill: var(--s3); }
.chart .segt { fill: #fff; font-size: 11.5px; font-weight: 600; }
.chart .cell.V { fill: var(--good-wash); } .chart .cell.U { fill: var(--warn-wash); } .chart .cell.X { fill: var(--crit-wash); } .chart .cell.- { fill: var(--dim-wash); }
.chart .cellt { fill: var(--ink); font-size: 10.5px; font-weight: 600; pointer-events: none; }
.chart .xh { stroke: var(--ink-2); stroke-width: 1; }
.chart .pt { stroke: var(--surface); stroke-width: 2; }
.chart .pt.composite { fill: var(--accent); } .chart .pt.benchmark { fill: var(--dim); } .chart .pt.modelnet { fill: var(--accent-l); }
.chart .pt.rf { fill: var(--ink-2); } .chart .pt.account { fill: var(--dim); fill-opacity: .7; }
.chart .sharpe { stroke: var(--accent); stroke-width: 1; stroke-opacity: .55; }
td.loss { color: var(--crit); }
table.var td, table.var th { padding: 5px 8px; font-size: 12.5px; }
.chart [data-tip] { cursor: default; }
.chart [data-tip]:hover, .chart .bar:hover { filter: brightness(.9); }
details.how { margin-top: 10px; font-size: 12.5px; }
details.how summary { cursor: pointer; color: var(--accent); font-weight: 500; list-style: none; display: inline-flex; align-items: center; gap: 6px; }
details.how summary::-webkit-details-marker { display: none; }
details.how summary::before { content: ""; width: 6px; height: 6px; border-right: 1.5px solid currentColor; border-bottom: 1.5px solid currentColor; transform: rotate(-45deg); transition: transform .15s; }
details.how[open] summary::before { transform: rotate(45deg); }
details.how summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 3px; }
.howb { margin-top: 8px; padding: 10px 12px; background: var(--surface-2); border: 1px solid var(--line); border-radius: 8px; color: var(--ink-2); line-height: 1.55; max-width: 90ch; }
.howb b { color: var(--ink); }
.tile details.how { margin-top: 6px; } .tile details.how summary { font-size: 11.5px; }
table.skill tr.yes td { background: var(--accent-wash); }
table.skill td { font-size: 12.5px; }
.k { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.k.s1 { background: var(--accent); } .k.s0 { background: var(--dim); } .k.s1l { background: var(--accent-l); }
.tip { position: fixed; z-index: 10; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; font-size: 12px; line-height: 1.45; box-shadow: 0 6px 24px rgba(0,0,0,.14); max-width: 320px; font-variant-numeric: tabular-nums; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (prefers-reduced-motion: no-preference) { .chart [data-tip] { transition: filter .12s; } }
"""

JS = r"""
(function(){
  const tip = document.getElementById('tip');
  function show(html, e){ tip.innerHTML = html; tip.hidden = false; move(e); }
  function move(e){
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = e.clientX + 14, y = e.clientY + 14;
    if (x + w > window.innerWidth - 8) x = e.clientX - w - 14;
    if (y + h > window.innerHeight - 8) y = e.clientY - h - 14;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  function hide(){ tip.hidden = true; }
  document.querySelectorAll('[data-tip]').forEach(el => {
    el.addEventListener('mouseenter', e => show(el.dataset.tip, e));
    el.addEventListener('mousemove', move);
    el.addEventListener('mouseleave', hide);
  });
  document.querySelectorAll('svg[data-line]').forEach(svg => {
    const id = svg.dataset.line;
    const node = document.getElementById(id + '-data');
    if (!node) return;
    const d = JSON.parse(node.textContent);
    const xh = svg.querySelector('.xh'), hit = svg.querySelector('.hit');
    hit.addEventListener('mousemove', e => {
      const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
      const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      let best = 0, bd = Infinity;
      for (let i = 0; i < d.xs.length; i++) { const dd = Math.abs(d.xs[i] - p.x); if (dd < bd) { bd = dd; best = i; } }
      xh.setAttribute('x1', d.xs[best]); xh.setAttribute('x2', d.xs[best]); xh.setAttribute('visibility', 'visible');
      show(d.rows[best], e);
    });
    hit.addEventListener('mouseleave', () => { xh.setAttribute('visibility', 'hidden'); hide(); });
  });
})();
"""
