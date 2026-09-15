"""Serve the generated dashboards over HTTP (for Railway or any host).

- Listens on $PORT immediately; builds the outputs in a background thread so
  the platform's healthcheck passes while the pipeline runs.
- Serves output/ — `/` redirects to the landing dashboard (LANDING, default
  brk/dashboard.html) once built; `/status` lists everything and shows the build log.
- Optional HTTP Basic Auth: set DASHBOARD_PASSWORD (username is anything).
  Leave unset only while the data is placeholder; real statements must never
  be reachable without it.

    python -m trackrecord serve            # PORT defaults to 8080
"""
from __future__ import annotations

import base64
import hmac
import json
import re
import html
import os
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .placeholder_ticker import FAMOUS, NOT_PUBLIC, valid as valid_ticker
from .fund_universe import STYLES
from . import accounts as AC
from . import uploads as UP
from . import pdfstatements as PS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FUNDS_ROOT = ROOT / "data" / "funds"
STATE = {"phase": "starting", "log": [], "started": time.time(), "done": False, "error": None}
JOBS: dict[str, dict] = {}          # ticker -> {phase, log, error, started}
JOBS_LOCK = threading.Lock()

DATASETS = [
    # (label, out subdir, build steps) — the BRK-A stock analysis was removed from the site (2026-09-13)
    ("Synthetic accounts (annual, injected +2% alpha)", "synthetic", [
        ["synth", "--out", "data/synthetic"],
        ["report", "--data", "data/synthetic", "--out", "output/synthetic", "--claimed", "0.14"]]),
]


def _log(line: str) -> None:
    STATE["log"].append(line)
    print(line, flush=True)


def _run(args: list[str]) -> None:
    _log(f"$ python -m trackrecord {' '.join(args)}")
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "trackrecord", *args], cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    for t in tail:
        _log("  " + t)
    _log(f"  ({time.time() - t0:.0f}s, exit {r.returncode})")
    if r.returncode not in (0, 1):      # reconcile returns 1 when flags exist; that's expected
        raise RuntimeError(f"{args[0]} failed ({r.returncode}): {' / '.join(tail)}")


def _ticker_datasets() -> list:
    """Committed ticker datasets (data/tickers/<T>/) get built at startup, after the two baselines."""
    import json
    out = []
    tdir = ROOT / "data" / "tickers"
    if tdir.exists():
        for d in sorted(tdir.iterdir()):
            if (d / "statements.csv").exists():
                try:
                    name = json.loads((d / "meta.json").read_text()).get("name", d.name)
                except Exception:
                    name = d.name
                out.append((f"{name} ({d.name})", f"t/{d.name}",
                            [["report", "--data", f"data/tickers/{d.name}", "--out", f"output/t/{d.name}", "--grid", "M", "--placeholder",
                              "--n-boot", "1500", "--n-cohort", "3000"]]))
    return out


def build_all() -> None:
    try:
        from .compact import unpack               # fresh container: restore the EDGAR/price caches from the committed bundle
        if unpack(log=_log):
            STATE["phase"] = "unpacked reference caches"
        for label, sub, steps in DATASETS + _ticker_datasets():
            if (OUT / sub / "dashboard.html").exists() and os.environ.get("REBUILD", "1") != "1":
                continue
            STATE["phase"] = f"building {label}"
            _log(f"== {STATE['phase']}")
            for step in steps:
                _run(step)
        STATE["phase"] = "ready"
        _log(f"== ready ({time.time() - STATE['started']:.0f}s after start)")
        warm_funds()                         # then every manager, featured first, so no click ever waits
    except Exception as e:                   # keep serving whatever exists; show the error
        STATE["error"] = str(e)
        STATE["phase"] = "failed"
        _log(f"== FAILED: {e}")
    STATE["done"] = True


FEATURED = ["berkshire-13f", "appaloosa", "atreides", "scion", "pershing-square", "baupost", "duquesne", "tiger-global"]


def warm_funds() -> None:
    """Pre-build every scorable manager's dashboard in the background (~30 s each), featured ones first.
    Uses the same job machinery as a click, so a visitor who arrives mid-build just joins the queue."""
    if os.environ.get("WARM_FUNDS", "1") != "1":
        return
    slugs = [r["slug"] for r in fund_index() if r.get("status") == "ok"]
    order = [s for s in FEATURED if s in slugs] + sorted(s for s in slugs if s not in FEATURED)
    for i, slug in enumerate(order):
        if (OUT / "funds" / slug / "dashboard.html").exists():
            continue
        job = start_fund(slug)
        while not job["done"]:
            time.sleep(1)
        _log(f"warm {i + 1}/{len(order)}: {slug} {'ok' if not job.get('error') else 'FAILED ' + str(job['error'])[:80]}")
    _log("== all managers pre-built")


def _run_job(job: dict, args: list[str]) -> None:
    job["log"].append(f"$ python -m trackrecord {' '.join(args)}")
    r = subprocess.run([sys.executable, "-m", "trackrecord", *args], cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-4:]
    job["log"].extend("  " + t for t in tail)
    if r.returncode not in (0, 1):
        raise RuntimeError(" / ".join(tail) or f"exit {r.returncode}")


def build_ticker(ticker: str) -> None:
    job = JOBS[ticker]
    raw = f"data/reference/tickers/{ticker}.csv"
    try:
        job["phase"] = "downloading prices"
        _run_job(job, ["fetch-ticker", "--ticker", ticker, "--out", raw])
        job["phase"] = "building statements"
        _run_job(job, ["ticker-placeholder", "--ticker", ticker, "--raw", raw, "--out", f"data/tickers/{ticker}"])
        job["phase"] = "running phases 1–4 (about 30 s)"
        _run_job(job, ["report", "--data", f"data/tickers/{ticker}", "--out", f"output/t/{ticker}", "--grid", "M", "--placeholder"])
        job["phase"] = "ready"
    except Exception as e:
        job["error"] = str(e); job["phase"] = "failed"
    finally:
        job["done"] = True


def start_ticker(ticker: str) -> dict:
    with JOBS_LOCK:
        job = JOBS.get(ticker)
        if job and not job["done"]:
            return job
        if (OUT / "t" / ticker / "dashboard.html").exists():
            JOBS[ticker] = job = {"phase": "ready", "log": [], "error": None, "started": time.time(), "done": True}
            return job
        JOBS[ticker] = job = {"phase": "queued", "log": [], "error": None, "started": time.time(), "done": False}
    threading.Thread(target=build_ticker, args=(ticker,), daemon=True).start()
    return job


def fund_index() -> list[dict]:
    """Every fund in data/funds: leaderboard.csv row if scored, else meta.json."""
    import csv, json
    rows = {}
    lb = FUNDS_ROOT / "leaderboard.csv"
    if lb.exists():
        with lb.open() as fh:
            for r in csv.DictReader(fh):
                rows[r["slug"]] = r
    if FUNDS_ROOT.exists():
        for d in sorted(FUNDS_ROOT.iterdir()):
            m = d / "meta.json"
            if d.is_dir() and m.exists() and d.name not in rows:
                try:
                    meta = json.loads(m.read_text())
                    rows[d.name] = dict(slug=d.name, name=meta.get("name", d.name), manager=meta.get("manager", ""), style=meta.get("style", ""),
                                        style_name=meta.get("style_name", ""), status=meta.get("status", ""), months=meta.get("months"),
                                        first=meta.get("first"), last=meta.get("last"), coverage=meta.get("avg_coverage"))
                except Exception:
                    pass
    # live scores from output/ override the committed leaderboard if a fund was built here
    for slug, r in rows.items():
        sc = OUT / "funds" / slug / "phase4" / "scores.csv"
        if sc.exists():
            import csv as _csv
            with sc.open() as fh:
                for x in _csv.DictReader(fh):
                    if x["score"] == "Alpha-maxing score": r["alpha_maxing"] = x["value"]; r["excess"] = x["input"]
                    if x["score"] == "Wealth-management score" and x["component"] == "TOTAL": r["wealth"] = x["value"]
                    if x["score"] == "Wealth-management score" and x["component"] == "skill evidence": r["ff3_t"] = x["input"]
        r["built"] = (OUT / "funds" / slug / "dashboard.html").exists()
    return list(rows.values())


def build_fund(slug: str) -> None:
    job = JOBS["fund:" + slug]
    try:
        job["phase"] = "running phases 1–4 on the 13F clone (about 30 s)"
        _run_job(job, ["report", "--data", f"data/funds/{slug}", "--out", f"output/funds/{slug}", "--grid", "M", "--placeholder",
                       "--n-boot", "1500", "--n-cohort", "3000"])
        job["phase"] = "ready"
    except Exception as e:
        job["error"] = str(e); job["phase"] = "failed"
    finally:
        job["done"] = True


def start_fund(slug: str) -> dict:
    key = "fund:" + slug
    with JOBS_LOCK:
        job = JOBS.get(key)
        if job and not job["done"]:
            return job
        if (OUT / "funds" / slug / "dashboard.html").exists():
            JOBS[key] = job = {"phase": "ready", "log": [], "error": None, "started": time.time(), "done": True}
            return job
        JOBS[key] = job = {"phase": "queued", "log": [], "error": None, "started": time.time(), "done": False}
    threading.Thread(target=build_fund, args=(slug,), daemon=True).start()
    return job


def _f(v, fmt):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    return fmt.format(v)


def managers_html() -> str:
    rows = fund_index()
    by_style = {}
    for r in rows:
        by_style.setdefault(r.get("style", ""), []).append(r)
    def sc(v):
        try: v = float(v)
        except (TypeError, ValueError): return ""
        c = "g" if v >= 70 else ("m" if v >= 45 else "b")
        return f'<span class="sc {c}">{v:.0f}</span>'
    sections = []
    order = ["conc", "act", "ls", "event", "fo", "multi", "macro"]
    for st in order + [k for k in by_style if k not in order]:
        if st not in by_style: continue
        name, note = STYLES.get(st, (st, ""))
        items = sorted(by_style[st], key=lambda r: -(float(r["wealth"]) if r.get("wealth") not in (None, "") else -1))
        trs = ""
        for r in items:
            if r.get("status") == "ok":
                link = f"<a href='/f/{r['slug']}/'>{html.escape(r['name'])}</a>"
                span = f"{(r.get('first') or '')[:7]} – {(r.get('last') or '')[:7]} · {r.get('months') or ''} mo · {_f(r.get('coverage'), '{:.0%}')} priced"
                cells = f"<td class='n'>{_f(r.get('excess'), '{:+.1%}')}</td><td class='n'>{sc(r.get('alpha_maxing'))}</td><td class='n'>{sc(r.get('wealth'))}</td>"
            else:
                link = html.escape(r["name"]); span = f"<span class='err'>not scorable: {html.escape(str(r.get('months') or 0))} months of filings (need 36)</span>"
                cells = "<td></td><td></td><td></td>"
            trs += f"<tr><td>{link}<br><span class='sub2'>{html.escape(r.get('manager') or '')} · {span}</span></td>{cells}</tr>"
        sections.append(f"<h2>{html.escape(name)}</h2><p class='sub'>{html.escape(note)}</p><table><thead><tr><th>manager</th><th class='n'>excess vs market /yr</th><th class='n'>alpha-maxing</th><th class='n'>wealth-mgmt</th></tr></thead><tbody>{trs}</tbody></table>")
    return page("Managers", f"""<h1>Managers</h1><p class="sub">{len(rows)} prominent funds and family offices, each represented by a <b>13F long-only clone</b> of its disclosed US holdings (SEC EDGAR, structured filings from 2013), rebalanced when each filing becomes public.</p>
<div class="banner"><b>A clone is not the fund.</b> 13F filings show US long positions only, 45 days late — no shorts, options, cash, leverage or non-US holdings. For concentrated long-biased managers the clone tracks the real book; for multi-strategy, quant and macro shops it is not meaningful, and those sections say so. Every number here is a reconstruction.</div>
<p><a href="/leaderboard">Leaderboard</a> · <a href="/">Berkshire (stock)</a> · <a href="/status">All outputs</a></p>
{''.join(sections)}
<p class="sub" style="margin-top:24px"><b>Scores.</b> Alpha-maxing = 50 + 10 × excess return over the US market (%/yr). Wealth-management = skill evidence 30% + risk-adjusted return 25% + downside protection 25% + consistency over rolling 5-year windows 20%. Fixed maps, so managers compare directly with each other and with the listed vehicles. Click a manager to build its full dashboard (about 30 s the first time).</p>""")


def leaderboard_rows() -> list[dict]:
    """Every analyzed portfolio with its two scores, from the scores.csv files on disk."""
    import csv
    rows = []
    cands = [("synthetic", OUT / "synthetic")] + \
            [(p.name, p) for p in sorted((OUT / "t").glob("*")) if p.is_dir()]
    for key, d in cands:
        f = d / "phase4" / "scores.csv"
        if not f.exists():
            continue
        am = wm = None
        with f.open() as fh:
            for r in csv.DictReader(fh):
                if r["score"] == "Alpha-maxing score": am = float(r["value"]); ex = float(r["input"])
                if r["score"] == "Wealth-management score" and r["component"] == "TOTAL": wm = float(r["value"])
        label = key
        meta = ROOT / "data" / ("tickers/" + key if d.parent.name == "t" else key) / "meta.json"
        if meta.exists():
            try:
                import json; label = json.loads(meta.read_text()).get("name", key)
            except Exception:
                pass
        elif key == "brk": label = "Berkshire Hathaway (BRK-A)"
        elif key == "synthetic": label = "Synthetic placeholder accounts"
        rows.append(dict(key=key, href=f"/{'t/' if d.parent.name == 't' else ''}{key}/dashboard.html", label=label,
                         alpha=am, wealth=wm, excess=ex if am is not None else None))
    for r in fund_index():
        if r.get("status") != "ok" or r.get("wealth") in (None, ""):
            continue
        rows.append(dict(key=r["slug"], href=f"/f/{r['slug']}/", label=f"{r['name']} — 13F clone",
                         alpha=float(r["alpha_maxing"]), wealth=float(r["wealth"]), excess=float(r["excess"]) if r.get("excess") not in (None, "") else None,
                         note=r.get("style_name", "")))
    rows.sort(key=lambda r: -(r["wealth"] or 0))
    return rows


NAV_CSS = """<style>
.tr-nav{position:sticky;top:0;z-index:6;display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:8px 32px;background:var(--surface,#fdfcf9);border-bottom:1px solid var(--line,#e4dfd2);font:13px "Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;color:var(--ink-2,#6b7078)}
.tr-nav a,.tr-nav button{display:inline-flex;align-items:center;gap:6px;font:600 10px "Helvetica Neue",Helvetica,Arial,sans-serif;letter-spacing:.18em;text-transform:uppercase;padding:8px 14px;border:1px solid var(--navy,#1b2a41);background:transparent;color:var(--navy,#1b2a41);text-decoration:none;cursor:pointer}
.tr-nav a.home,.tr-nav a.on{background:var(--navy,#1b2a41);color:var(--cover-ink,#e8e4da)}
@media(max-width:640px){.tr-nav{padding:8px 16px}.tr-nav a,.tr-nav button{padding:7px 10px;letter-spacing:.12em}}
.tr-sub{display:flex;gap:14px;align-items:center;padding:6px 32px;background:var(--surface,#fdfcf9);border-bottom:1px solid var(--line,#e4dfd2);font:13px "Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif}
.tr-sub a{font:600 10px "Helvetica Neue",Helvetica,Arial,sans-serif;letter-spacing:.18em;text-transform:uppercase;color:var(--navy,#1b2a41);text-decoration:none}
.tr-sub .crumb,.tr-nav .crumb{margin-left:auto;font:600 9px "Helvetica Neue",Helvetica,Arial,sans-serif;letter-spacing:.2em;text-transform:uppercase;color:var(--muted,#a09883)}
@media(prefers-color-scheme:dark){.tr-sub a{color:var(--gold-l,#c9b48a)}}
@media(prefers-color-scheme:dark){.tr-nav a,.tr-nav button{border-color:var(--gold-l,#c9b48a);color:var(--gold-l,#c9b48a)}.tr-nav a.home,.tr-nav a.on{background:var(--gold-l,#c9b48a);color:#1b2a40}}
</style>"""


TOOLS = [("Home", "/"), ("Screener", "/#all"), ("13F signals", "/research/13f-signals"), ("Construction", "/research/construction"),
         ("R check", "/research/r-verify"), ("Research data", "/research"), ("My records", "/me")]


def toolbar(current: str = "", extra: tuple[str, str] | None = None, crumb: str = "") -> str:
    """The same bar on every page: Back, then every tool (the current one filled). Page-specific
    links (a manager's memo, back to its dashboard) go on a thin line underneath, never in the bar."""
    links = "".join(f'<a class="{"home" if href == "/" else "on" if href == current else ""}" href="{href}">{label}</a>' for label, href in TOOLS)
    bar = ('<div class="tr-nav"><button type="button" onclick="history.length>1?history.back():location.assign(\'/\')">&larr; Back</button>' + links + '</div>')
    if extra or crumb:
        bar += ('<div class="tr-sub">' + (f'<a href="{html.escape(extra[1], quote=True)}">{html.escape(extra[0])} &rarr;</a>' if extra else '')
                + (f'<span class="crumb">{html.escape(crumb)}</span>' if crumb else '') + '</div>')
    return bar


def nav_html(crumb: str = "", me: bool = False, extra: tuple[str, str] | None = None, current: str = "") -> str:
    return NAV_CSS + toolbar(current, extra, crumb)


# ---------------------------------------------------------------- private records

def build_user_record(uid: str, slug: str) -> None:
    key = f"user:{uid}:{slug}"; job = JOBS[key]
    rec = AC.user_dir(uid) / slug
    try:
        meta = json.loads((rec / "meta.json").read_text())
        job["phase"] = "running phases 1–4 on your record"
        args = ["report", "--data", str(rec / "data"), "--out", str(rec / "out"), "--grid", meta.get("grid", "M"),
                "--label", meta.get("label", slug), "--n-boot", "2000", "--n-cohort", "4000"]
        if meta.get("claimed"):
            args += ["--claimed", str(meta["claimed"])]
        _run_job(job, args)
        job["phase"] = "ready"
    except Exception as e:
        job["error"] = str(e); job["phase"] = "failed"
    finally:
        job["done"] = True


def start_user_record(uid: str, slug: str) -> dict:
    key = f"user:{uid}:{slug}"
    with JOBS_LOCK:
        job = JOBS.get(key)
        if job and not job["done"]:
            return job
        if (AC.user_dir(uid) / slug / "out" / "dashboard.html").exists():
            JOBS[key] = job = {"phase": "ready", "log": [], "error": None, "started": time.time(), "done": True}
            return job
        JOBS[key] = job = {"phase": "queued", "log": [], "error": None, "started": time.time(), "done": False}
    threading.Thread(target=build_user_record, args=(uid, slug), daemon=True).start()
    return job


def parse_multipart(content_type: str, body: bytes) -> tuple[dict, dict]:
    """fields: name -> str; files: name -> list of (filename, bytes)."""
    from email.parser import BytesParser
    from email.policy import default
    msg = BytesParser(policy=default).parsebytes(b"Content-Type: " + content_type.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    fields, files = {}, {}
    if not msg.is_multipart():
        return fields, files
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition") or ""
        fn = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if fn:
            files.setdefault(name, []).append((Path(fn).name, payload))
        else:
            fields[name] = payload.decode("utf-8", errors="replace")
    return fields, files


ACCOUNT_CSS = """<style>
.form{max-width:440px;display:grid;gap:12px;margin:8px 0 24px}.form label{display:grid;gap:5px;font:600 9.5px/1 var(--sans);letter-spacing:.18em;text-transform:uppercase;color:var(--muted)}
.form input[type=text],.form input[type=email],.form input[type=password],.form input[type=number],.form select{font:15px var(--serif);padding:9px 12px;border:1px solid var(--line);background:var(--surface);color:var(--ink)}
.form input[type=file]{font:13px var(--sans);color:var(--ink2)}.form .row{display:flex;gap:10px;align-items:center}
.msg{padding:10px 14px;border-left:2px solid var(--gold);background:var(--surface);margin:0 0 16px;max-width:70ch}.msg.err{border-color:var(--crit);color:var(--crit)}
.recs td .st{font:600 9px/14px var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--gold)}
.shape{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:10px 0 18px}.shape div{background:var(--surface);border:1px solid var(--line);padding:12px 14px;font-size:13px}.shape b{display:block;font:400 16px var(--serif);color:var(--navy);margin-bottom:4px}.shape code{font:12px Menlo,monospace;color:var(--ink2)}
</style>"""


def sheet_svg(headers: list[str], rows: list[list[str]], widths: list[int] | None = None, note: str = "") -> str:
    """A spreadsheet-looking figure: column letters, row numbers, shaded header row."""
    widths = widths or [110] * len(headers)
    rn = 34; rh = 24; hh = 22
    W = rn + sum(widths) + 2; H = hh + rh * (len(rows) + 1) + 2
    grey = "var(--surface-2, #f3f0e8)"; head = "var(--accent-wash, rgba(27,42,65,.08))"; line = "var(--line, #e4dfd2)"
    muted = "var(--muted, #a09883)"; ink = "var(--ink, #23262b)"; navy = "var(--navy, #1b2a41)"; surf = "var(--surface, #fdfcf9)"
    sans = "Helvetica,Arial,sans-serif"; mono = "Menlo,Consolas,monospace"
    x = rn; out = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px;display:block;border:1px solid {line};background:{surf}" role="img" aria-label="example file">']
    out.append(f'<rect x="0" y="0" width="{W}" height="{hh}" fill="{grey}"/>')
    for i_, w in enumerate(widths):
        out.append(f'<text x="{x + w / 2:.0f}" y="15" text-anchor="middle" font-family="{sans}" font-size="10" fill="{muted}">{chr(65 + i_)}</text>')
        out.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{H}" stroke="{line}"/>'); x += w
    out.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{H}" stroke="{line}"/>')
    for r, cells in enumerate([headers] + rows):
        y = hh + rh * r
        if r == 0:
            out.append(f'<rect x="{rn}" y="{y}" width="{W - rn}" height="{rh}" fill="{head}"/>')
        out.append(f'<rect x="0" y="{y}" width="{rn}" height="{rh}" fill="{grey}"/>')
        out.append(f'<text x="{rn / 2:.0f}" y="{y + 16}" text-anchor="middle" font-family="{sans}" font-size="10" fill="{muted}">{r + 1}</text>')
        out.append(f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="{line}"/>')
        x = rn
        for c, w in zip(cells, widths):
            out.append(f'<text x="{x + 8}" y="{y + 16}" font-family="{mono}" font-size="11.5" font-weight="{"700" if r == 0 else "400"}" fill="{navy if r == 0 else ink}">{html.escape(c)}</text>'); x += w
    out.append(f'<line x1="0" y1="{hh + rh * (len(rows) + 1)}" x2="{W}" y2="{hh + rh * (len(rows) + 1)}" stroke="{line}"/>')
    out.append("</svg>")
    return "".join(out) + (f'<p class="sub" style="font-size:12px;margin:6px 0 0">{html.escape(note)}</p>' if note else "")


def pdf_example_svg() -> str:
    W, H = 520, 232
    L = ["Fidelity Investments", "Statement Period: January 1, 2024 – January 31, 2024", "Account Number: X12-345678", "",
         "ACCOUNT SUMMARY", "Beginning Account Value            $1,000,000.00", "Additions                                  $50,000.00",
         "Subtractions                              ($20,000.00)", "Change in Investment Value                $12,500.00", "Ending Account Value              $1,042,500.00"]
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px;display:block;border:1px solid var(--line,#e4dfd2);background:#fff" role="img" aria-label="statement example">']
    y = 26
    for i, t in enumerate(L):
        bold = i in (0, 4); mono = i >= 5
        fam = "Menlo,Consolas,monospace" if mono else "Helvetica,Arial,sans-serif"
        col = "#1b2a41" if bold or mono else "#23262b"
        if mono:
            out.append(f'<rect x="60" y="{y - 13}" width="{W - 120}" height="19" fill="{"rgba(140,122,86,.10)" if i in (5, 9) else "none"}"/>')
        out.append(f'<text x="70" y="{y}" font-family="{fam}" font-size="{12 if not mono else 11.5}" font-weight="{"700" if bold else "400"}" fill="{col}">{html.escape(t)}</text>')
        y += 20 if t else 10
    out.append(f'<text x="70" y="{H - 10}" font-family="Helvetica,Arial,sans-serif" font-size="10.5" fill="#6b7078">the block the parser reads — labels can vary by custodian; what it finds and misses is reported per file</text>')
    out.append("</svg>")
    return "".join(out)


EXAMPLES = {
    "pdf": pdf_example_svg(),
    "returns": sheet_svg(["date", "return"], [["2020-01-31", "1.8"], ["2020-02-29", "-3.1"], ["2020-03-31", "-9.4"], ["2020-04-30", "7.2"]], [130, 90],
                         "one row per month (or per year); return as a percent (1.8) or a decimal (0.018)"),
    "values": sheet_svg(["date", "value", "flow"], [["2020-01-31", "1000000", "1000000"], ["2020-02-29", "969000", "0"], ["2020-03-31", "905000", "25000"], ["2020-04-30", "998000", "-10000"]], [130, 110, 100],
                        "value = ending balance that period; flow = money in (+) or out (−) during it; first row's flow = opening deposit"),
    "statements": sheet_svg(["statement_id", "account_id", "custodian", "period_start", "period_end", "ending_value", "beginning_value", "stated_deposits", "stated_withdrawals", "stated_pnl"],
                            [["FIDE_1234_2005", "FIDE_1234", "Fidelity", "2005-01-01", "2005-12-31", "1234567.89", "1100000.00", "50000.00", "0.00", "84567.89"],
                             ["FIDE_1234_2006", "FIDE_1234", "Fidelity", "2006-01-01", "2006-12-31", "1402211.10", "1234567.89", "0.00", "20000.00", "187643.21"]],
                            [130, 90, 80, 100, 100, 100, 110, 110, 120, 100],
                            "statements.csv — one row per statement; enter what is PRINTED (beginning value, additions, subtractions, change in value); leave blank what isn't"),
    "flows": sheet_svg(["account_id", "date", "amount", "flow_type", "description"],
                       [["FIDE_1234", "2005-03-15", "50000.00", "deposit", "EFT from checking"], ["FIDE_1234", "2006-09-01", "-20000.00", "withdrawal", "wire out"]],
                       [90, 100, 90, 90, 160], "flows.csv — every deposit (+) and withdrawal (−) with its date; dividends and fees are NOT flows"),
}


def account_html(msg: str = "", err: bool = False, mode: str = "signin") -> str:
    return page("Sign in", ACCOUNT_CSS + f"""<header class="cover"><div class="cover-in"><div class="eyebrow">Private records</div><div class="rule"></div><h1>{'Create an account' if mode == 'signup' else 'Sign in'}</h1>
<p class="sub">Your uploads and results are private to your account and stay here between visits. Nothing you upload appears on the public pages.</p></div></header>
<main class="wrap">{f'<p class="msg{" err" if err else ""}">{html.escape(msg)}</p>' if msg else ''}
<form class="form" method="post" action="/account"><input type="hidden" name="mode" value="{mode}">
<label>Email<input type="email" name="email" required autocomplete="email"></label>
<label>Password<input type="password" name="password" required minlength="10" autocomplete="{'new-password' if mode == 'signup' else 'current-password'}"></label>
<div class="row"><button class="btn">{'Create account' if mode == 'signup' else 'Sign in'}</button>
<a href="/account?mode={'signin' if mode == 'signup' else 'signup'}" style="font-size:13px">{'Already have an account? Sign in' if mode == 'signup' else 'No account yet? Create one'}</a></div></form>
<form method="post" action="/guest" style="max-width:440px;margin:0 0 28px"><button class="btn" style="display:block;width:100%;padding:18px 20px;font-size:13px;letter-spacing:.22em;background:var(--goldl);color:#1b2a40;border:0">Continue as guest</button>
<p class="sub" style="font-size:12.5px;margin:8px 0 0">Upload and analyze straight away. A guest's records disappear when the browser is closed, or after 24 hours — create an account at any point to keep them.</p></form>
<p class="sub" style="font-size:12.5px;max-width:70ch">Passwords are stored only as salted PBKDF2 hashes. This is a private working tool, not a bank login: there is no email verification or password reset yet — keep your password somewhere safe.</p>
<p style="display:flex;gap:10px"><a class="btn" href="/">Home</a></p></main>""")


def me_html(uid: str, msg: str = "", err: bool = False) -> str:
    recs = AC.list_records(uid)
    guest = AC.is_guest(uid)
    rows = ""
    for r in recs:
        st = "ready" if r.get("built") else "not analyzed yet"
        if r.get("shape") == "pdf":
            st += f" · {r.get('parsed', 0)} PDF{'s' if r.get('parsed', 0) != 1 else ''} parsed, {r.get('skipped', 0)} skipped — <a href='/me/{r['slug']}/files'>report</a>"
        rows += (f"<tr><td><span class='mgr'>{html.escape(r['label'])}</span><br><span class='fund'>{html.escape(r.get('shape', ''))} · {html.escape(r.get('first', ''))} – {html.escape(r.get('last', ''))} · {r.get('periods', '')} periods · {'monthly' if r.get('grid') == 'M' else 'annual'}"
                 f"{' · claimed ' + format(float(r['claimed']) * 100, '.1f') + '%' if r.get('claimed') else ''}</span></td>"
                 f"<td><span class='st'>{st}</span></td>"
                 f"<td class='n' style='white-space:nowrap'><a class='btn' href='/me/{r['slug']}/'>Analyze</a> "
                 f"<form method='post' action='/me/{r['slug']}/delete' style='display:inline' onsubmit='return confirm(\"Delete this record and its results?\")'><button class='btn' style='background:transparent;color:var(--crit);border:1px solid var(--crit)'>Delete</button></form></td></tr>")
    return page("My records", current="/me", body=ACCOUNT_CSS + f"""<header class="cover"><div class="cover-in"><div class="eyebrow">Private records · {'guest session' if guest else html.escape(AC.user_email(uid))}</div><div class="rule"></div><h1>My records</h1>
<p class="sub">Upload a return series, a value-and-flow history, or the pipeline's own statement templates, and run the same verification the public managers get. Only you can see these.</p></div></header>
<main class="wrap">{f'<p class="msg{" err" if err else ""}">{html.escape(msg)}</p>' if msg else ''}
{'<p class="msg"><b>You are a guest.</b> These records disappear when you close the browser, or after 24 hours. <a href="/account?mode=signup">Create an account</a> and they come with you.</p>' if guest else ''}
<h2 data-n="Section 01">Your records</h2>
<table class="recs"><thead><tr><th>record</th><th>status</th><th class="n"></th></tr></thead><tbody>{rows or '<tr><td colspan=3>nothing uploaded yet</td></tr>'}</tbody></table>
<h2 data-n="Section 02">Upload a record</h2>
<div class="shape">
<div style="grid-column:1/-1;border-color:var(--gold)"><b>Statement PDFs — best</b>Upload the monthly or annual statements themselves, straight from the custodian's website (Fidelity, Schwab, Vanguard, Robinhood, IBKR…). The account summary each one prints — beginning value, additions, subtractions, change in value, ending value — is read from every file, chained, and checked, so periods can come out <i>verified</i>. Text PDFs only for now: scanned paper shows as "no text layer" and needs the template route. After upload you get a per-file report of what was found and what wasn't.
<div style="margin-top:10px">{EXAMPLES["pdf"]}</div></div>
<div><b>Returns</b>One row per month or year: <code>date, return</code>. Return as a decimal (0.012) or a percent (1.2). <a href="/templates/returns.csv">template</a>. Nothing to reconcile against → every period shows as <i>unverified</i>.
<div style="margin-top:10px">{EXAMPLES["returns"]}</div></div>
<div><b>Values and flows</b>One row per statement: <code>date, value, flow</code> — the account's ending value and any deposit (+) or withdrawal (−) that period. <a href="/templates/values.csv">template</a>. Returns are computed properly net of flows.
<div style="margin-top:10px">{EXAMPLES["values"]}</div></div>
<div style="grid-column:1/-1"><b>Statement templates</b>The pipeline's own <code>statements.csv</code>, <code>flows.csv</code>, <code>positions.csv</code> (and optional <code>accounts.csv</code>), entered from real statements. <a href="/templates/statements.csv">statements</a> · <a href="/templates/flows.csv">flows</a> · <a href="/templates/positions.csv">positions</a> · <a href="/templates/ENTRY_GUIDE.md">guide</a>. The only shape that can reach <i>verified</i>: the pipeline checks each printed number against the prior statement, the positions, and the flows.
<div style="margin-top:10px;overflow-x:auto">{EXAMPLES["statements"]}</div><div style="margin-top:10px;overflow-x:auto">{EXAMPLES["flows"]}</div></div>
</div>
<form class="form" method="post" action="/me/upload" enctype="multipart/form-data">
<label>Name for this record<input type="text" name="label" required maxlength="60" placeholder="e.g. Main account 1996–2025"></label>
<label>Shape<select name="shape"><option value="pdf">Statement PDFs (best)</option><option value="returns">Returns (date, return)</option><option value="values">Values and flows (date, value, flow)</option><option value="template">Statement templates (statements.csv + flows.csv + …)</option></select></label>
<label>File(s) — PDFs, CSV or Excel, up to 5 MB each; select many at once<input type="file" name="files" multiple required accept=".pdf,.csv,.xlsx,.xls,.txt"></label>
<label>Claimed annual return, % (optional)<input type="number" name="claimed" step="0.01" placeholder="e.g. 14"></label>
<div class="row"><button class="btn">Upload</button></div></form>
<p style="display:flex;gap:10px"><a class="btn" href="/">Home</a>{'<a class="btn" href="/account?mode=signup" style="background:var(--goldl);color:#1b2a40">Create an account to keep these</a>' if guest else ''}<a class="btn" href="/logout" style="background:transparent;color:var(--navy);border:1px solid var(--navy)">{'Leave' if guest else 'Sign out'}</a></p></main>""")


def topbar(is_home: bool = False, current: str = "") -> str:
    return toolbar(current or ("/" if is_home else ""))


def page(title: str, body: str, refresh: int | None = None, is_home: bool = False, current: str = "") -> str:
    body = NAV_CSS + topbar(is_home, current) + body
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>{f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ''}
<style>
:root{{--serif:"Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;--sans:"Helvetica Neue",Helvetica,Arial,sans-serif;--page:#f6f4ee;--surface:#fdfcf9;--line:#e4dfd2;--ink:#23262b;--ink2:#6b7078;--muted:#a09883;--navy:#1b2a41;--gold:#8c7a56;--goldl:#c9b48a;--cover:#1b2a40;--coverink:#e8e4da;--covermuted:#8a9ab4;--good:#4f7a5a;--crit:#8f3b34}}
@media(prefers-color-scheme:dark){{:root{{--page:#141f31;--surface:#1b2a40;--line:#34455f;--ink:#e8e4da;--ink2:#b7bcc6;--muted:#8a9ab4;--navy:#e8e4da;--gold:#c9b48a;--cover:#111a2a}}}}
body{{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 var(--serif)}}
.cover{{background:var(--cover);color:var(--coverink)}}.cover-in{{max-width:1100px;margin:0 auto;padding:40px 32px 34px}}
.eyebrow{{font:600 9.5px/1 var(--sans);letter-spacing:.24em;text-transform:uppercase;color:var(--goldl)}}.rule{{width:44px;height:2px;background:var(--goldl);margin:20px 0 16px}}
h1{{font:400 42px/1.1 var(--serif);margin:0;color:var(--coverink)}}.cover .sub{{color:var(--covermuted);margin:14px 0 0;font-size:16px;max-width:70ch}}
.wrap{{max-width:1100px;margin:0 auto;padding:28px 32px 60px}}
h2{{font:400 22px/1.2 var(--serif);color:var(--navy);margin:34px 0 4px;position:relative;padding-top:16px}}h2::before{{content:attr(data-n);position:absolute;top:0;left:0;font:600 9px/1 var(--sans);letter-spacing:.24em;text-transform:uppercase;color:var(--gold)}}
.note{{color:var(--ink2);margin:0 0 14px;max-width:90ch}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th{{text-align:left;font:700 9.5px/1.4 var(--sans);letter-spacing:.16em;text-transform:uppercase;color:var(--navy);padding:8px;border-bottom:1px solid var(--navy)}}td{{padding:11px 8px;border-bottom:1px solid var(--line);vertical-align:middle}}td.n,th.n{{text-align:right;font-family:var(--sans);font-size:13px;font-variant-numeric:tabular-nums}}th.n{{font-size:9.5px}}
.mgr{{font-size:16px;color:var(--navy)}}.fund{{color:var(--ink2);font-size:13px}}.tag{{display:inline-block;font:600 9px/14px var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--gold);margin-left:8px}}
.btn{{display:inline-block;font:600 10px/1 var(--sans);letter-spacing:.18em;text-transform:uppercase;padding:9px 16px;background:var(--navy);color:var(--coverink);text-decoration:none;white-space:nowrap}}.btn.off{{background:transparent;color:var(--muted);border:1px solid var(--line);cursor:default}}
@media(prefers-color-scheme:dark){{.btn{{background:var(--goldl);color:#1b2a40}}}}
.sc{{display:inline-block;min-width:30px;text-align:center;font:600 12px/1 var(--sans);padding:5px 7px;color:#fff}}.g{{background:var(--good)}}.m{{background:var(--gold)}}.b{{background:var(--crit)}}
.banner{{background:#e4dfd2;color:#1b2a41;border-bottom:1px solid #d3ccbb;padding:8px 32px;font-size:12.5px}}.banner b{{font:600 9.5px/1 var(--sans);letter-spacing:.2em;text-transform:uppercase;margin-right:12px}}
.status{{display:inline-block;font:600 9.5px/1 var(--sans);letter-spacing:.16em;text-transform:uppercase;padding:5px 9px;border:1px solid var(--line);color:var(--ink2)}}pre{{background:var(--surface);border:1px solid var(--line);padding:12px;font-size:12px;overflow-x:auto}}.err{{color:var(--crit)}}
a{{color:var(--navy)}}.foot{{margin-top:40px;padding-top:14px;border-top:1px solid var(--navy);color:var(--ink2);font-size:12px;max-width:120ch}}
</style></head><body>{body}</body></html>"""


def directory_html() -> str:
    """The home screen: cover with the system at a glance, featured managers, then every
    manager in one searchable, sortable table with an Analyze button each."""
    import csv, json
    def num(v):
        try: return float(v)
        except (TypeError, ValueError): return None
    def sc(v):
        v = num(v)
        if v is None: return ""
        return f'<span class="sc {"g" if v >= 70 else "m" if v >= 45 else "b"}">{v:.0f}</span>'
    def scores_of(out):
        f = out / "phase4" / "scores.csv"
        if not f.exists(): return (None, None, None, None)
        am = wm = ex = t = None
        with f.open() as fh:
            for r in csv.DictReader(fh):
                if r["score"] == "Alpha-maxing score": am = r["value"]; ex = r["input"]
                if r["score"] == "Wealth-management score" and r["component"] == "TOTAL": wm = r["value"]
                if r["score"] == "Wealth-management score" and r["component"] == "skill evidence": t = r["input"]
        return am, wm, ex, t
    # ---- listed vehicles: BRK stock + any committed ticker dataset
    listed = []
    tdir = ROOT / "data" / "tickers"
    if tdir.exists():
        for d in sorted(tdir.iterdir()):
            m = d / "meta.json"
            if not m.exists(): continue
            meta = json.loads(m.read_text())
            am, wm, ex, t = scores_of(OUT / "t" / d.name)
            listed.append(dict(key=d.name, href=f"/analyze?ticker={d.name}", mgr=meta.get("name", d.name), fund=f"{d.name} — listed, distributions reinvested",
                               tag=f"{(meta.get('first') or '')[:4]}–{(meta.get('last') or '')[:4]}", alpha=am, wealth=wm, excess=ex, t=t,
                               months=meta.get("months") or meta.get("rows"), memo=f"/t/{d.name}/memo", ok=True))
    # ---- funds
    funds = fund_index()
    NOT_MEANINGFUL = {"multi", "macro", "mm"}
    frows = []
    for r in funds:
        ok = r.get("status") == "ok"
        nm = r.get("style") in NOT_MEANINGFUL
        frows.append(dict(key=r["slug"], href=f"/f/{r['slug']}/", mgr=r.get("manager") or r["name"], fund=r["name"], style=r.get("style_name") or "", nm=nm,
                          tag=(f"13F clone · {(r.get('first') or '')[:4]}–{(r.get('last') or '')[:4]} · {_f(r.get('coverage'), '{:.0%}')} priced" if ok
                               else f"not scorable yet — {r.get('months') or 0} months of usable filings (36 needed)"),
                          alpha=r.get("alpha_maxing"), wealth=r.get("wealth"), excess=r.get("excess"), t=r.get("ff3_t"),
                          months=r.get("months"), memo=f"/f/{r['slug']}/memo", ok=ok))
    n_ok = sum(1 for r in frows if r["ok"])
    def row(r, kind):
        a, w, e, t = num(r["alpha"]), num(r["wealth"]), num(r["excess"]), num(r.get("t"))
        mo = num(r.get("months")) or 0
        nm = r.get("nm", False)
        btn = f"<a class='btn' href='{r['href']}'>Analyze</a>" if r["ok"] else "<span class='btn off'>Analyze</span>"
        # the memo is the research entry point: it only exists for a scorable, meaningful clone
        memo = (f"<a class='memo' href='{r['memo']}'>Memo</a>" if r["ok"] and not nm and r.get("memo") else "")
        search = html.escape(f"{r['mgr']} {r['fund']} {r.get('style', '')} {kind}".lower(), quote=True)
        style_key = html.escape((r.get("style") or ("Listed vehicle" if kind == "listed" else "")).lower(), quote=True)
        if nm and r["ok"]:
            cells = "<td colspan='3' class='n'><span class='sub2' style='color:var(--muted)'>clone not meaningful for this strategy — actual returns are private</span></td>"
            w = a = -2; e = -99
            t = None
        else:
            cells = (f"<td class='n'>{_f(r['excess'], '{:+.1%}')}</td><td class='n'>{_f(r.get('t'), '{:+.1f}')}</td>"
                     f"<td class='n'>{sc(r['alpha'])}</td><td class='n'>{sc(r['wealth'])}</td>")
        tv = t if t is not None else -99
        return (f"<tr data-s='{search}' data-n='{html.escape(r['mgr'].lower(), quote=True)}' data-w='{w if w is not None else -1}' "
                f"data-a='{a if a is not None else -1}' data-e='{e if e is not None else -99}' data-t='{tv}' data-m='{mo:.0f}' "
                f"data-y='{style_key}' data-mgr='{html.escape(r['mgr'], quote=True)}' data-fund='{html.escape(r['fund'], quote=True)}' "
                f"data-slug='{html.escape(r['key'], quote=True)}'>"
                f"<td><span class='mgr'>{html.escape(r['mgr'])}</span>{'<span class=tag>' + html.escape(r['style']) + '</span>' if r.get('style') else ''}<br>"
                f"<span class='fund'>{html.escape(r['fund'])} · {html.escape(r['tag'])}</span></td>"
                f"{cells}<td class='n'>{memo}{btn}</td></tr>")
    rcards, _have = _research_cards()
    styles = sorted({r["style"] for r in frows if r.get("style")} | ({"Listed vehicle"} if listed else set()))
    style_opts = "".join(f"<option value='{html.escape(x.lower(), quote=True)}'>{html.escape(x)}</option>" for x in styles)
    ordered = sorted(frows, key=lambda r: (1 if r.get("nm") else 0, -(num(r["wealth"]) if num(r["wealth"]) is not None else -1), r["mgr"]))
    table = "".join(row(r, "listed") for r in listed) + "".join(row(r, "hedge fund") for r in ordered)
    # ---- featured
    feat_keys = [("berkshire-13f", "Warren Buffett", "Berkshire Hathaway holdings — 13F clone"), ("appaloosa", "David Tepper", "Appaloosa — 13F clone"),
                 ("atreides", "Gavin Baker", "Atreides — 13F clone"), ("FCNTX", "Will Danoff", "Fidelity Contrafund — listed")]
    by_key = {r["key"]: r for r in listed + frows}
    cards = ""
    for k, who, what in feat_keys:
        r = by_key.get(k)
        if not r: continue
        cards += (f"<a class='card' href='{r['href']}'><div class='who'>{html.escape(who)}</div><div class='what'>{html.escape(what)}</div>"
                  f"<div class='nums'><div><span class='lbl'>excess /yr</span>{_f(r['excess'], '{:+.1%}') or '—'}</div><div><span class='lbl'>alpha-max</span>{sc(r['alpha']) or '—'}</div><div><span class='lbl'>wealth</span>{sc(r['wealth']) or '—'}</div></div>"
                  f"<div class='go'>Analyze &rarr;</div></a>")
    head = ("<thead><tr><th class='sort' data-k='n' data-dir='asc'>manager</th><th class='n sort' data-k='e' data-dir='desc'>excess vs market /yr</th>"
            "<th class='n sort' data-k='t' data-dir='desc' title='t-statistic on the FF3 alpha, Newey-West'>FF3 t</th>"
            "<th class='n sort' data-k='a' data-dir='desc'>alpha-maxing</th><th class='n sort on' data-k='w' data-dir='desc'>wealth-mgmt</th><th></th></tr></thead>")
    extra_css = """<style>
th.sort{cursor:pointer;user-select:none;white-space:nowrap}th.sort::after{content:"";display:inline-block;width:0;height:0;margin-left:7px;vertical-align:middle;border-left:4px solid transparent;border-right:4px solid transparent;border-top:5px solid var(--line)}
th.sort.on::after{border-top-color:var(--gold)}th.sort.on.asc::after{border-top:0;border-bottom:5px solid var(--gold)}th.sort:hover{color:var(--gold)}
.cta{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:26px 0 0;max-width:760px}
.cta .big{display:grid;gap:4px;padding:20px 22px;text-decoration:none;border:1px solid var(--goldl);color:var(--coverink);background:rgba(232,228,218,.04)}
.cta .big.gold{background:var(--goldl);color:#1b2a40;border-color:var(--goldl)}
.cta .big .t{font:400 22px/1.15 var(--serif)}.cta .big .s{font:600 9.5px/1.3 var(--sans);letter-spacing:.18em;text-transform:uppercase;opacity:.75}
.cta .big:hover{filter:brightness(1.08)}@media(max-width:640px){.cta{grid-template-columns:1fr}}
.stats{display:flex;flex-wrap:wrap;gap:22px 40px;margin-top:30px;padding-top:18px;border-top:1px solid rgba(232,228,218,.18)}.stats div{display:grid;gap:4px}.stats dt{font:600 9px/1 var(--sans);letter-spacing:.22em;text-transform:uppercase;color:var(--covermuted)}.stats dd{margin:0;font:400 30px/1 var(--serif);color:var(--coverink)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin:6px 0 10px}
.card{display:block;background:var(--surface);border:1px solid var(--line);border-top:2px solid var(--gold);padding:16px 18px;text-decoration:none;color:var(--ink)}
.card .who{font:400 20px/1.15 var(--serif);color:var(--navy)}.card .what{color:var(--ink2);font-size:13px;margin:2px 0 12px}
.card .nums{display:flex;gap:16px;font:600 13px var(--sans)}.card .nums div{display:grid;gap:4px}.card .lbl{font:600 8.5px/1 var(--sans);letter-spacing:.18em;text-transform:uppercase;color:var(--muted)}
.card .go{margin-top:12px;font:600 9.5px/1 var(--sans);letter-spacing:.18em;text-transform:uppercase;color:var(--gold)}
.tools{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:14px 0 10px}.tools input{font:15px var(--serif);padding:8px 12px;border:1px solid var(--line);background:var(--surface);color:var(--ink);flex:1;min-width:220px}
.count{font:600 9.5px/1 var(--sans);letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin-left:auto}
.rcards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(460px,100%),1fr));gap:14px;margin:6px 0 10px}
.rcard{display:flex;flex-direction:column;background:var(--surface);border:1px solid var(--line);border-top:2px solid var(--gold);padding:18px 20px;text-decoration:none;color:var(--ink)}
.rcard .eyeb{font:600 9px/1 var(--sans);letter-spacing:.22em;text-transform:uppercase;color:var(--muted)}
.rcard .rt{font:400 21px/1.2 var(--serif);color:var(--navy);margin:8px 0 6px}
.rcard .rw{font-size:13.5px;color:var(--ink2);line-height:1.5;flex:1}
.rcard .rs{display:flex;gap:18px;margin:14px 0 0;flex-wrap:wrap}.rcard .rs div{display:grid;gap:3px}
.rcard .rs .v{font:600 16px/1 var(--sans);color:var(--navy)}
.rcard .rs .l{font:600 8.5px/1.2 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.rcard .go{margin-top:14px;font:600 9.5px/1 var(--sans);letter-spacing:.18em;text-transform:uppercase;color:var(--gold)}
.rcard:hover{border-color:var(--gold)}
.filters{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:0 0 12px;padding:12px 14px;background:var(--surface);border:1px solid var(--line)}
.filters label{font:600 9px/1 var(--sans);letter-spacing:.16em;text-transform:uppercase;color:var(--muted);display:grid;gap:5px}
.filters select{font:14px var(--serif);padding:6px 8px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
.filters .chk{flex-direction:row;align-items:center;gap:7px;display:flex;cursor:pointer}.filters .chk input{margin:0}
.filters .lnk{background:none;border:0;color:var(--gold);font:600 9.5px/1 var(--sans);letter-spacing:.16em;text-transform:uppercase;cursor:pointer;padding:6px 0}
.filters .dlbtn{margin-left:auto;font:600 9.5px/1 var(--sans);letter-spacing:.16em;text-transform:uppercase;padding:9px 14px;border:1px solid var(--navy);background:var(--navy);color:var(--surface);cursor:pointer}
.filters .dlbtn:hover{filter:brightness(1.15)}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}
#tbl td:last-child{white-space:nowrap}
a.memo{font:600 9px/1 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--gold);text-decoration:none;margin-right:14px;vertical-align:middle}
a.memo:hover{text-decoration:underline}
@media(max-width:640px){.filters{gap:10px}.filters .dlbtn{margin-left:0;width:100%}}
</style>"""
    js = """<script>
(function(){const q=document.getElementById('q'),rows=[...document.querySelectorAll('#tbl tbody tr')],cnt=document.getElementById('cnt'),tb=document.querySelector('#tbl tbody');
const fT=document.getElementById('f-t'),fY=document.getElementById('f-y'),fM=document.getElementById('f-m'),fOk=document.getElementById('f-ok');
function visible(){return rows.filter(r=>!r.hidden);}
function apply(){const s=q.value.trim().toLowerCase(),ft=fT.value,fy=fY.value,fm=parseFloat(fM.value)||0,ok=fOk.checked;let n=0;
rows.forEach(r=>{const d=r.dataset,t=parseFloat(d.t),scorable=t>-90;
let pass=!s||d.s.includes(s);
if(pass&&ft!=='any'){pass=scorable&&(ft==='neg'?t<0:t>=parseFloat(ft));}
if(pass&&fy!=='any')pass=d.y===fy;
if(pass&&fm)pass=(parseFloat(d.m)||0)>=fm;
if(pass&&ok)pass=scorable;
r.hidden=!pass;if(pass)n++;});
cnt.textContent=n+' of '+rows.length;}
[q,fT,fY,fM,fOk].forEach(el=>el.addEventListener(el===q?'input':'change',apply));
document.getElementById('f-reset').addEventListener('click',()=>{q.value='';fT.value='any';fY.value='any';fM.value='0';fOk.checked=false;apply();});
document.getElementById('dl').addEventListener('click',()=>{
const esc=v=>{v=(v==null?'':String(v));return /[",\\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v;};
const head=['slug','manager','fund','style','months','excess_vs_market','ff3_t','alpha_maxing','wealth_management'];
const out=[head.join(',')];
visible().forEach(r=>{const d=r.dataset,g=x=>{const v=parseFloat(d[x]);return (v===undefined||isNaN(v)||v<=-90||v===-1||v===-2)?'':v;};
out.push([d.slug,d.mgr,d.fund,d.y,d.m,g('e'),g('t'),g('a'),g('w')].map(esc).join(','));});
const b=new Blob([out.join('\\n')],{type:'text/csv;charset=utf-8'}),u=URL.createObjectURL(b),a=document.createElement('a');
a.href=u;a.download='managers-'+visible().length+'.csv';document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(u);});
const gs=document.getElementById('go-search');if(gs)gs.addEventListener('click',e=>{e.preventDefault();document.getElementById('all').scrollIntoView({behavior:'smooth',block:'start'});setTimeout(()=>q.focus(),400);});
function sortBy(k,dir){const num=k!=='n';[...tb.querySelectorAll('tr')].sort((x,y)=>{let a=x.dataset[k],b=y.dataset[k];if(num){a=parseFloat(a);b=parseFloat(b);return dir==='asc'?a-b:b-a;}return dir==='asc'?a.localeCompare(b):b.localeCompare(a);}).forEach(r=>tb.appendChild(r));}
document.querySelectorAll('th.sort').forEach(th=>th.addEventListener('click',()=>{const k=th.dataset.k;let dir=th.dataset.dir;if(th.classList.contains('on')){dir=dir==='asc'?'desc':'asc';th.dataset.dir=dir;}
document.querySelectorAll('th.sort').forEach(x=>x.classList.remove('on','asc'));th.classList.add('on');if(dir==='asc')th.classList.add('asc');sortBy(k,dir);}));
apply();})();
</script>"""
    return page("Track Record Verification", f"""{extra_css}<div class="banner"><b>Illustrative data</b> Public records and SEC 13F reconstructions used to demonstrate the pipeline. Nothing here is the record under verification.</div>
<header class="cover"><div class="cover-in"><div class="eyebrow">Independent performance verification</div><div class="rule"></div><h1>Track Record Verification</h1>
<p class="sub">One system, applied the same way to every manager: reconcile the record, compute time-weighted returns, remove what the market and known factors explain, simulate how often luck alone does as well, test stability, and score. Press <b>Analyze</b> on any row.</p>
<div class="cta"><a class="big" href="#all" id="go-search"><span class="t">Search other people's returns</span><span class="s">{len(frows) + len(listed)} managers · 13F clones and listed funds</span></a>
<a class="big gold" href="/me"><span class="t">My records</span><span class="s">Upload my records</span></a></div>
<dl class="stats"><div><dt>Managers in the system</dt><dd>{len(frows) + len(listed)}</dd></div><div><dt>Scorable today</dt><dd>{n_ok + len(listed)}</dd></div><div><dt>Listed vehicles</dt><dd>{len(listed)}</dd></div><div><dt>13F clones</dt><dd>{len(frows)}</dd></div></dl>
</div></header>
<main class="wrap">
<h2 data-n="Featured">Start here</h2><p class="note">Three managers seen through their disclosed holdings (13F clones), and a listed fund with a 35-year real record.</p>
<div class="cards">{cards}</div>
<h2 data-n="Research" id="research">Use it as a research tool</h2>
<p class="note">Four things you can do with the system beyond looking up one manager. Every number on these pages is generated from the data underneath it, and the data comes down as CSV. Full index: <a href="/research">all research and datasets</a>.</p>
<div class="rcards">{rcards}</div>
<h2 data-n="All managers" id="all">Screen every manager in the system</h2>
<p class="note">Listed vehicles are actual returns (share price, distributions reinvested). Hedge funds and family offices are <b>13F long-only clones</b>: their disclosed US holdings at disclosed weights, rebalanced when each quarterly filing becomes public — a reconstruction, not the fund. No shorts, options, cash, leverage or non-US holdings; entered ~45 days late; months with too little of the book priced are left out and never bridged. Concentrated, activist, long-short and long-only clones track the real book. For multi-strategy, quant, macro and market-making firms — Citadel, Millennium, Renaissance, Bridgewater, Jane Street, Belvedere — a 13F is trading inventory, not a portfolio, so no score is shown; their actual returns are private.</p>
<div class="tools"><input id="q" placeholder="Search a manager, fund or strategy — e.g. Tepper, activist, quant" autocomplete="off"><span class="count" id="cnt"></span></div>
<div class="filters">
  <label>Evidence of alpha <select id="f-t">
    <option value="any">any</option><option value="2">FF3 t ≥ 2 — statistically significant</option>
    <option value="1">FF3 t ≥ 1 — suggestive</option><option value="0">FF3 t ≥ 0 — positive alpha</option>
    <option value="neg">FF3 t &lt; 0 — negative alpha</option></select></label>
  <label>Style <select id="f-y"><option value="any">any</option>{style_opts}</select></label>
  <label>Track length <select id="f-m">
    <option value="0">any</option><option value="36">3 years or more</option>
    <option value="60">5 years or more</option><option value="120">10 years or more</option></select></label>
  <label class="chk"><input type="checkbox" id="f-ok"> scorable only</label>
  <button type="button" id="f-reset" class="lnk">Reset</button>
  <button type="button" id="dl" class="dlbtn">Download CSV</button>
</div>
<div class="tw"><table id="tbl">{head}<tbody>{table}</tbody></table></div>
<div class="foot"><b>Scores.</b> Alpha-maxing = 50 + 10 × excess return over the US market (%/yr), return only. Wealth-management = skill evidence 30% + risk-adjusted return 25% + downside protection 25% + consistency over rolling 5-year windows 20%. Fixed maps, comparable across every row. Benchmark and factors: Kenneth R. French Data Library; prices: Yahoo Finance; holdings: SEC EDGAR. First analysis of a manager takes about half a minute; afterwards it opens instantly. Past performance is not indicative of future results; nothing here is investment advice.</div>
</main>{js}""", is_home=True)


def ticker_status_html(ticker: str, job: dict, label: str | None = None, ready_href: str | None = None) -> str:
    """Wait page: friendly steps, a live timer, and a script that polls and follows the redirect
    (meta refresh alone is throttled by some mobile browsers)."""
    steps = ["Reconciling the statement series", "Computing time-weighted returns and the composite",
             "Attribution: market, size, value and residual", "Factor regressions and Jensen's alpha",
             "Simulating zero-skill managers (luck test)", "Rolling windows, sub-periods and scores", "Rendering the dashboard"]
    elapsed = int(time.time() - job["started"])
    done_n = min(len(steps) - 1, elapsed // 6) if not job["done"] else len(steps)
    lis = "".join(f"<li class='{'done' if k < done_n else 'now' if k == done_n else ''}'>{st}</li>" for k, st in enumerate(steps))
    err = f'<p class="err">Could not complete: {html.escape(job["error"])}</p>' if job.get("error") else ""
    ready = job["done"] and not job.get("error")
    poll = "" if job["done"] else """<script>setTimeout(function(){location.reload();},4000);</script>"""
    return page(f"{label or ticker} — analyzing", f"""<style>
ol.steps{{list-style:none;padding:0;margin:18px 0 0;max-width:60ch}}ol.steps li{{padding:9px 0 9px 26px;border-bottom:1px solid var(--line);position:relative;color:var(--ink2)}}
ol.steps li::before{{content:"";position:absolute;left:4px;top:15px;width:9px;height:9px;border:1px solid var(--muted);border-radius:50%}}
ol.steps li.done{{color:var(--ink)}}ol.steps li.done::before{{background:var(--good);border-color:var(--good)}}ol.steps li.now{{color:var(--navy);font-weight:600}}ol.steps li.now::before{{background:var(--gold);border-color:var(--gold)}}
.timer{{font:600 10px/1 var(--sans);letter-spacing:.18em;text-transform:uppercase;color:var(--muted);margin-top:16px}}
.spin{{display:inline-block;width:34px;height:34px;border:2px solid rgba(201,180,138,.25);border-top-color:var(--goldl);border-radius:50%;animation:tr-spin .9s linear infinite;vertical-align:middle;margin-right:16px}}
ol.steps li.now::before{{animation:tr-pulse 1.2s ease-in-out infinite}}
@keyframes tr-spin{{to{{transform:rotate(360deg)}}}}@keyframes tr-pulse{{0%,100%{{box-shadow:0 0 0 0 rgba(201,180,138,.55)}}50%{{box-shadow:0 0 0 6px rgba(201,180,138,0)}}}}
@media(prefers-reduced-motion:reduce){{.spin{{animation:none;border-top-color:var(--goldl)}}ol.steps li.now::before{{animation:none}}}}
</style>
<header class="cover"><div class="cover-in"><div class="eyebrow">Independent performance verification</div><div class="rule"></div><h1>{'Ready' if ready else '<span class="spin" aria-hidden="true"></span>Please wait'}</h1>
<p class="sub">{'The analysis of <b>' + html.escape(label or ticker) + '</b> is ready.' if ready else 'Analyzing <b>' + html.escape(label or ticker) + '</b> — the first run takes about half a minute. This page opens the dashboard by itself when it is done; there is nothing to press.'}</p></div></header>
<main class="wrap">{err}<ol class="steps">{lis}</ol>
<p class="timer">{'Ready' if ready else html.escape(job['phase'])} &nbsp;·&nbsp; {elapsed}s</p>
{'<p><a class="btn" href="' + (ready_href or ('/t/' + ticker + '/dashboard.html' if not ticker.islower() else '/funds/' + ticker + '/dashboard.html')) + '">Open the dashboard</a></p>' if ready else ''}
<p style="display:flex;gap:10px"><a class="btn" href="/">Home</a><a class="btn" href="javascript:history.back()" style="background:transparent;color:var(--navy);border:1px solid var(--navy)">&larr; Back</a></p></main>{poll}""", refresh=None if job["done"] else 5)


def startup_wait_html(which: str) -> str:
    label = {"synthetic": "the synthetic placeholder"}.get(which, which)
    ready = (OUT / which / "dashboard.html").exists()
    log = "\n".join(html.escape(l) for l in STATE["log"][-6:])
    return page(f"{label} — preparing", f"""<style>
.spin{{display:inline-block;width:34px;height:34px;border:2px solid rgba(201,180,138,.25);border-top-color:var(--goldl);border-radius:50%;animation:tr-spin .9s linear infinite;vertical-align:middle;margin-right:16px}}
ol.steps li.now::before{{animation:tr-pulse 1.2s ease-in-out infinite}}
@keyframes tr-spin{{to{{transform:rotate(360deg)}}}}@keyframes tr-pulse{{0%,100%{{box-shadow:0 0 0 0 rgba(201,180,138,.55)}}50%{{box-shadow:0 0 0 6px rgba(201,180,138,0)}}}}
@media(prefers-reduced-motion:reduce){{.spin{{animation:none;border-top-color:var(--goldl)}}ol.steps li.now::before{{animation:none}}}}
</style><header class="cover"><div class="cover-in"><div class="eyebrow">Independent performance verification</div><div class="rule"></div><h1>{'Ready' if ready else '<span class="spin" aria-hidden="true"></span>Please wait'}</h1>
<p class="sub">{'<b>' + html.escape(label) + '</b> is ready.' if ready else 'Preparing <b>' + html.escape(label) + '</b> — the server was just restarted and is rebuilding its baseline records (about a minute). This page opens the dashboard by itself when it is done; there is nothing to press.'}</p></div></header>
<main class="wrap"><p class="timer" style="font:600 10px/1 var(--sans);letter-spacing:.18em;text-transform:uppercase;color:var(--muted)">{html.escape(STATE["phase"])} &nbsp;·&nbsp; {int(time.time() - STATE["started"])}s since restart</p>
<pre>{log or '(starting)'}</pre>{'<p><a class="btn" href="/' + which + '/dashboard.html">Open the dashboard</a></p>' if ready else ''}<p style="display:flex;gap:10px"><a class="btn" href="/">Home</a><a class="btn" href="javascript:history.back()" style="background:transparent;color:var(--navy);border:1px solid var(--navy)">&larr; Back</a></p></main>
{'' if ready else '<script>setTimeout(function(){location.reload();},4000);</script>'}""", refresh=None if ready else 5)


def not_found_html(path: str) -> str:
    return page("Not found", f"""<header class="cover"><div class="cover-in"><div class="eyebrow">Independent performance verification</div><div class="rule"></div><h1>Nothing here</h1>
<p class="sub">There is no page at <code style="color:var(--goldl)">{html.escape(path)}</code>. Every manager in the system is one click from the home page.</p></div></header>
<main class="wrap"><p style="display:flex;gap:10px"><a class="btn" href="/">Home</a><a class="btn" href="javascript:history.back()" style="background:transparent;color:var(--navy);border:1px solid var(--navy)">&larr; Back</a></p></main>""")


def _research_cards() -> tuple[str, dict]:
    """Cards for the research tools, with their headline numbers read from the committed CSVs.

    Every number shown is read back from data/research/, so a rebuild cannot leave stale
    claims on the home page; a tool whose data is missing is simply left out."""
    import csv as _csv
    R = ROOT / "data" / "research"

    def rows(path, key=None):
        f = R / path
        if not f.exists():
            return {}
        with f.open() as fh:
            rs = list(_csv.DictReader(fh))
        return {r[key]: r for r in rs} if key else rs

    def man(path):
        f = R / path
        try:
            return json.loads(f.read_text()) if f.exists() else {}
        except Exception:
            return {}

    def fnum(x, default=None):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    cards, have = [], {}

    sig, sigman = rows("13f-signals/summary.csv", "key"), man("13f-signals/manifest.json")
    if sig and sigman:
        best = sig.get("BEST1", {})
        t = fnum(best.get("carhart_t"))
        verdict = ("no edge survives the 45-day lag" if t is None or abs(t) < 2
                   else "a spread that survives the lag")
        cards.append(dict(href="/research/13f-signals", eyebrow="Signal research",
                          title="Do the disclosed books carry a signal?",
                          what=f"Best ideas, crowding and fresh buys from every filing, formed into quarterly portfolios and tested as Carhart spreads. Finding: {verdict}.",
                          stats=[(f"{sigman.get('managers', '')}", "managers"),
                                 (f"{sigman.get('quarters', '')}", "quarters"),
                                 (f"{fnum(sigman.get('positions'), 0):,.0f}", "positions")]))
        have["signals"] = True

    con, conman = rows("construction/summary.csv", "key"), man("construction/manifest.json")
    if con and conman:
        P = con.get("portfolio", {})
        cards.append(dict(href="/research/construction", eyebrow="Portfolio construction",
                          title="From a signal to a trade list",
                          what="A linear program turns alpha scores into target weights under name caps, sector and active bands and a turnover budget, then into the tickets a trader would receive. Solved in Python and again in R.",
                          stats=[(f"{fnum(P.get('active_return'), 0) * 100:+.1f}%", "active return /yr"),
                                 (f"{fnum(P.get('information_ratio'), 0):.2f}", "information ratio"),
                                 (f"{conman.get('rebalances', '')}", "rebalances")]))
        have["construction"] = True

    rv = man("r-verify/manifest.json")
    if rv.get("managers"):
        cards.append(dict(href="/research/r-verify", eyebrow="Verification",
                          title="The same alphas, recomputed in R",
                          what="Every headline number is recomputed from the aligned returns by an independent R implementation \u2014 the Newey-West sandwich written out by hand \u2014 and compared with what Python reported.",
                          stats=[(f"{rv.get('managers')}", "managers"),
                                 (f"{rv.get('checks', 0):,}", "numbers checked"),
                                 (f"{rv.get('max_abs_diff', 0):.0e}", "largest difference")]))
        have["rverify"] = True

    cards.append(dict(href="/f/appaloosa/memo", eyebrow="Due diligence",
                      title="A memo on any manager",
                      what="Verdict, alpha with its confidence interval, factor loadings and rolling drift, the replication test, risks, and the questions to put to the manager at the next quarterly meeting \u2014 every sentence generated from that manager's numbers.",
                      stats=[("91", "managers covered"), ("1", "click from any row"), ("0", "hand-written prose")]))
    have["memo"] = True

    out = ""
    for c in cards:
        st = "".join(f"<div><span class='v'>{html.escape(str(v))}</span><span class='l'>{html.escape(l)}</span></div>"
                     for v, l in c["stats"])
        out += (f"<a class='rcard' href=\"{c['href']}\"><div class='eyeb'>{html.escape(c['eyebrow'])}</div>"
                f"<div class='rt'>{html.escape(c['title'])}</div><div class='rw'>{c['what']}</div>"
                f"<div class='rs'>{st}</div><div class='go'>Open &rarr;</div></a>")
    return out, have


def leaderboard_html() -> str:
    rows = leaderboard_rows()
    def sc(v):
        if v is None: return ""
        c = "g" if v >= 70 else ("m" if v >= 45 else "b")
        return f'<span class="sc {c}">{v:.0f}</span>'
    def row(r):
        ex = "" if r["excess"] is None else f"{r['excess'] * 100:+.2f}%"
        return (f"<tr><td><a href='{r['href']}'>{html.escape(r['label'])}</a><br><span style='color:#8a93a0;font-size:12px'>{html.escape(r.get('note') or r['key'])}</span></td>"
                f"<td class='n'>{ex}</td><td class='n'>{sc(r['alpha'])}</td><td class='n'>{sc(r['wealth'])}</td></tr>")
    body = "".join(row(r) for r in rows)
    famous = "".join(f"<li><a href='/analyze?ticker={t}'>{t}</a> — {html.escape(n)}</li>" for t, n in FAMOUS)
    return page("Leaderboard", f"""<h1>Leaderboard</h1><p class="sub">Every portfolio analyzed so far, scored on fixed 0–100 maps so they compare directly.</p>
<div class="banner"><b>All placeholder / public data.</b> Nothing here is the record under verification.</div>
<form action="/analyze" method="get"><input name="ticker" placeholder="Enter a ticker — fund, ETF or stock (Yahoo format, e.g. FCNTX, BRK-A)" required pattern="[A-Za-z0-9.\-]{{1,12}}"><button>Analyze</button></form>
<table><thead><tr><th>portfolio</th><th class="n">excess vs market /yr</th><th class="n">alpha-maxing</th><th class="n">wealth-mgmt</th></tr></thead><tbody>{body or '<tr><td colspan=4>nothing analyzed yet</td></tr>'}</tbody></table>
<p><a href="/managers"><b>All managers by strategy →</b></a></p>
<h3>Listed vehicles with a real public record</h3><ul>{famous}</ul>
<p class="sub"><b>Hedge funds</b> appear as 13F clones (US long positions only, 45 days late — a reconstruction, not the fund's return). Multi-strategy, quant and macro managers are listed on the Managers page but their clones are not meaningful and are flagged.</p>
<p class="sub"><b>Alpha-maxing</b> = 50 + 10 × excess return over the US market (%/yr), return only. <b>Wealth-management</b> = skill evidence (30%) + risk-adjusted return (25%) + downside protection (25%) + consistency across rolling 5-year windows (20%). Full breakdown on each dashboard.</p>""")


def index_html() -> str:
    rows = []
    for label, sub, _ in DATASETS:
        d = OUT / sub
        ok = (d / "dashboard.html").exists()
        links = (f'<a href="/{sub}/dashboard.html">dashboard</a> · <a href="/{sub}/REPORT.md">report</a> · '
                 f'<a href="/{sub}/coverage.md">coverage</a> · <a href="/{sub}/">files</a>') if ok else "<em>not built yet</em>"
        rows.append(f"<tr><td>{html.escape(label)}</td><td>{links}</td></tr>")
    log = "\n".join(html.escape(l) for l in STATE["log"][-12:])
    err = f'<p class="err">{html.escape(STATE["error"])}</p>' if STATE["error"] else ""
    refresh = '<meta http-equiv="refresh" content="8">' if not STATE["done"] else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Track Record Verification</title>{refresh}
<style>body{{font:15px/1.5 system-ui,-apple-system,sans-serif;max-width:760px;margin:40px auto;padding:0 20px;color:#141a22;background:#f3f5f7}}
h1{{font-size:24px;margin:0 0 4px}}.sub{{color:#4a5462;margin:0 0 20px}}table{{border-collapse:collapse;width:100%}}td{{padding:10px 8px;border-bottom:1px solid #e3e6ea;vertical-align:top}}
.banner{{background:#fff4d6;border:1px solid #f0c96a;color:#5c4300;padding:10px 14px;border-radius:8px;margin-bottom:20px}}
pre{{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:12px;font-size:12px;overflow-x:auto}}.err{{color:#d03b3b}}
.status{{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;background:#e3e6ea}}
@media(prefers-color-scheme:dark){{body{{background:#0f1317;color:#eef1f4}}td{{border-color:#2a313a}}pre{{background:#171c22;border-color:#2a313a}}.banner{{background:#3a2e08;border-color:#7a5f10;color:#ffe6a3}}.status{{background:#2a313a}}}}</style></head>
<body><h1>Track Record Verification</h1><p class="sub">Pipeline outputs · <span class="status">{html.escape(STATE["phase"])}</span> · {int(time.time() - STATE["started"])}s since start</p>
<div class="banner"><b>Placeholder data.</b> Nothing here is the record under verification.</div>
<form action="/analyze" method="get" style="display:flex;gap:8px;margin:0 0 16px"><input name="ticker" placeholder="Analyze any ticker, e.g. FCNTX" required pattern="[A-Za-z0-9.\-]{{1,12}}" style="font:inherit;padding:8px 10px;border:1px solid #c9ced4;border-radius:8px;flex:1;text-transform:uppercase"><button style="font:inherit;font-weight:600;padding:8px 14px;border:0;border-radius:8px;background:#2a78d6;color:#fff">Analyze</button></form>
<p><a href="/leaderboard">Leaderboard of everything analyzed</a></p>
<table>{''.join(rows)}</table>{err}
<h3>Build log</h3><pre>{log or '(waiting)'}</pre>
<p class="sub"><a href="https://github.com/simonweardon/track-record-verification">source</a></p></body></html>"""


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(OUT), **kw)

    # ---- sessions -------------------------------------------------------------
    def _uid(self) -> str | None:
        raw = self.headers.get("Cookie", "")
        for part in raw.split(";"):
            k, _, v = part.strip().partition("=")
            if k == "trsession":
                return AC.verify_token(v)
        return None

    def _secure(self) -> bool:
        return self.headers.get("X-Forwarded-Proto", "").lower() == "https"

    def _same_origin(self) -> bool:
        host = self.headers.get("Host", "")
        for h in ("Origin", "Referer"):
            v = self.headers.get(h)
            if v:
                from urllib.parse import urlparse as _u
                return _u(v).netloc == host
        return True

    def _redirect_with_cookie(self, to: str, cookie: str):
        self.send_response(302); self.send_header("Location", to); self.send_header("Set-Cookie", cookie)
        self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", "0"); self.end_headers()

    def do_POST(self):
        u = urlparse(self.path)
        if not self._same_origin():
            return self._html(page("Blocked", "<main class='wrap'><h1>Blocked</h1><p>Cross-site request.</p></main>"), 403)
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > 6 * UP.MAX_BYTES:
            return self._html(page("Too large", "<main class='wrap'><h1>Too large</h1><p>Uploads are limited to 5 MB per file.</p></main>"), 413)
        body = self.rfile.read(length)
        ctype = self.headers.get("Content-Type", "")
        if ctype.startswith("multipart/form-data"):
            fields, files = parse_multipart(ctype, body)
        else:
            fields = {k: v[0] for k, v in parse_qs(body.decode("utf-8", errors="replace")).items()}; files = {}
        if u.path == "/guest":
            AC.purge_guests()
            gid = AC.create_guest()
            return self._redirect_with_cookie("/me", AC.cookie_header(AC.issue_token(gid), self._secure(), session_only=True))
        if u.path == "/account":
            mode = fields.get("mode", "signin"); email = fields.get("email", ""); pw = fields.get("password", "")
            if mode == "signup":
                uid, err = AC.create_user(email, pw)
                if err:
                    return self._html(account_html(err, True, "signup"))
                cur = self._uid()
                if cur and AC.is_guest(cur):
                    AC.adopt_guest(cur, uid)          # a guest who signs up keeps their records
            else:
                uid = AC.authenticate(email, pw)
                if not uid:
                    return self._html(account_html("Email or password not recognised.", True, "signin"))
            return self._redirect_with_cookie("/me", AC.cookie_header(AC.issue_token(uid), self._secure()))
        uid = self._uid()
        if not uid:
            return self._redirect("/account")
        if u.path == "/me/upload":
            label = (fields.get("label") or "").strip()[:60] or "My record"
            shape = fields.get("shape", "returns")
            claimed = fields.get("claimed", "").strip()
            slug = AC.slugify(label)
            base = AC.user_dir(uid); rec = base / slug
            n = 2
            while rec.exists():
                rec = base / f"{slug}-{n}"; n += 1
            slug = rec.name
            try:
                up = files.get("files") or []
                if not up:
                    raise UP.UploadError("no file received")
                if shape == "pdf":
                    pdfs = [(fn, data) for fn, data in up if fn.lower().endswith(".pdf")]
                    if not pdfs:
                        raise UP.UploadError("no PDF files received")
                    if any(len(d) > UP.MAX_BYTES for _, d in pdfs):
                        raise UP.UploadError("a PDF is larger than 5 MB")
                    info, _ = PS.from_pdfs(pdfs, rec / "data", label)
                elif shape == "template":
                    info = UP.from_template({fn: data for fn, data in up}, rec / "data", label)
                elif shape == "values":
                    info = UP.from_values(up[0][0], up[0][1], rec / "data", label)
                else:
                    info = UP.from_returns(up[0][0], up[0][1], rec / "data", label)
                meta = dict(slug=slug, label=label, created=time.time(), **info)
                if claimed:
                    try:
                        meta["claimed"] = float(claimed) / 100.0
                    except ValueError:
                        pass
                (rec / "meta.json").write_text(json.dumps(meta))
            except UP.UploadError as e:
                import shutil; shutil.rmtree(rec, ignore_errors=True)
                return self._html(me_html(uid, f"Could not use that upload: {e}", True), 400)
            except Exception as e:
                import shutil; shutil.rmtree(rec, ignore_errors=True)
                return self._html(me_html(uid, f"Could not use that upload: {e}", True), 400)
            start_user_record(uid, slug)
            return self._redirect(f"/me/{slug}/")
        m = re.match(r"^/me/([a-z0-9\-]{1,48})/delete$", u.path)
        if m:
            import shutil
            shutil.rmtree(AC.user_dir(uid) / m.group(1), ignore_errors=True)
            JOBS.pop(f"user:{uid}:{m.group(1)}", None)
            return self._redirect("/me")
        return self._html(not_found_html(u.path), 404)

    def _authorized(self) -> bool:
        pw = os.environ.get("DASHBOARD_PASSWORD")
        if not pw:
            return True
        hdr = self.headers.get("Authorization", "")
        if hdr.startswith("Basic "):
            try:
                _, _, got = base64.b64decode(hdr[6:]).decode().partition(":")
                return hmac.compare_digest(got, pw)
            except Exception:
                return False
        return False

    def _deny(self) -> None:
        self.send_response(401); self.send_header("WWW-Authenticate", 'Basic realm="track record"')
        self.send_header("Content-Length", "0"); self.end_headers()

    def do_HEAD(self):
        if self.path == "/health":
            self.send_response(200); self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "2"); self.end_headers(); return
        if not self._authorized():
            return self._deny()
        return super().do_HEAD()

    def do_GET(self):
        if self.path == "/health":
            body = b"ok"
            self.send_response(200); self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if not self._authorized():
            return self._deny()
        u = urlparse(self.path)
        if u.path in ("/", "/managers", "/leaderboard", "/index.html"):
            return self._html(directory_html())
        if u.path == "/account":
            cur = self._uid()
            if cur and not AC.is_guest(cur):
                return self._redirect("/me")
            return self._html(account_html(mode=parse_qs(u.query).get("mode", ["signin"])[0]))
        if u.path == "/logout":
            return self._redirect_with_cookie("/", AC.clear_cookie_header(self._secure()))
        if u.path.startswith("/templates/"):
            name = u.path.split("/", 2)[2]
            if name in UP.TEMPLATES:
                b = UP.TEMPLATES[name].encode(); self.send_response(200); self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f"attachment; filename={name}"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
            f = ROOT / "templates" / name
            if f.exists() and f.is_file() and re.match(r"^[A-Za-z0-9_.\-]+$", name):
                b = f.read_bytes(); self.send_response(200); self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Disposition", f"attachment; filename={name}"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
            return self._html(not_found_html(u.path), 404)
        if u.path == "/me" or u.path.startswith("/me/"):
            uid = self._uid()
            if not uid:
                return self._redirect("/account")
            if u.path == "/me":
                return self._html(me_html(uid))
            mf = re.match(r"^/me/([a-z0-9\-]{1,48})/files$", u.path)
            if mf:
                rec = AC.user_dir(uid) / mf.group(1); rp = rec / "data" / "pdf_report.csv"
                if not rp.exists():
                    return self._html(not_found_html(u.path), 404)
                import csv as _csv
                rows_ = list(_csv.DictReader(rp.open()))
                cols = ["file", "period_start", "period_end", "account_last4", "beginning_value", "stated_deposits", "stated_withdrawals", "stated_pnl", "ending_value", "missing", "error"]
                trs = "".join("<tr>" + "".join(f"<td class='{'n' if c not in ('file', 'missing', 'error') else ''}'>{html.escape(str(r.get(c, '') or ''))}</td>" for c in cols) + "</tr>" for r in rows_)
                return self._html(page("PDF report", f"""<header class="cover"><div class="cover-in"><div class="eyebrow">Private records</div><div class="rule"></div><h1>What was read from each PDF</h1>
<p class="sub">A blank cell means the label was not found on that statement; the pipeline leaves it blank rather than guessing. Skipped files say why.</p></div></header>
<main class="wrap"><div style="overflow-x:auto"><table><thead><tr>{''.join(f'<th>{c.replace("_", " ")}</th>' for c in cols)}</tr></thead><tbody>{trs}</tbody></table></div>
<p style="display:flex;gap:10px;margin-top:20px"><a class="btn" href="/me">My records</a><a class="btn" href="/me/{mf.group(1)}/" style="background:var(--goldl);color:#1b2a40">Analyze</a></p></main>"""))
            m = re.match(r"^/me/([a-z0-9\-]{1,48})/(dashboard\.html)?$", u.path)
            if not m:
                return self._html(not_found_html(u.path), 404)
            slug = m.group(1); rec = AC.user_dir(uid) / slug
            if not (rec / "meta.json").exists():
                return self._html(not_found_html(u.path), 404)
            dash = rec / "out" / "dashboard.html"
            if m.group(2):
                if not dash.exists():
                    return self._redirect(f"/me/{slug}/")
                doc = dash.read_text(encoding="utf-8")
                label = json.loads((rec / "meta.json").read_text()).get("label", slug)
                doc = doc.replace('<main class="wrap">', nav_html(label + " — private", me=True, current="/me") + '<main class="wrap">', 1)
                return self._html(doc)
            job = start_user_record(uid, slug)
            if job["done"] and not job.get("error") and dash.exists():
                return self._redirect(f"/me/{slug}/dashboard.html")
            label = json.loads((rec / "meta.json").read_text()).get("label", slug)
            return self._html(ticker_status_html(slug, job, label=label, ready_href=f"/me/{slug}/dashboard.html"))
        if u.path in ("/research", "/research/"):
            from .research_pages import research_index_html
            return self._html(research_index_html().replace('<main class="wrap">', nav_html("Research", current="/research") + '<main class="wrap">', 1))
        if u.path.startswith("/research/data/"):
            # any CSV under data/research, addressed as <note>/<file>.csv — no traversal
            rel = u.path[len("/research/data/"):]
            if re.match(r"^[a-z0-9-]+/[a-z0-9_]+\.csv$", rel):
                f = ROOT / "data" / "research" / rel
                if f.exists():
                    b = f.read_bytes(); self.send_response(200); self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.send_header("Content-Disposition", f"attachment; filename={f.name}")
                    self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
            return self._html(not_found_html(u.path), 404)
        if u.path == "/research/13f-signals":
            from .research_pages import signals13f_html
            doc = signals13f_html()
            if doc is None:
                return self._html(page("Not built", "<main class='wrap'><h1>Research note not built</h1><p>Run <code>python -m trackrecord signals13f</code>.</p></main>"), 404)
            return self._html(doc.replace('<main class="wrap">', nav_html("Research · 13F signals", current="/research/13f-signals") + '<main class="wrap">', 1))
        if u.path == "/research/construction":
            from .research_pages import construction_html
            doc = construction_html()
            if doc is None:
                return self._html(page("Not built", "<main class='wrap'><h1>Research note not built</h1><p>Run <code>python -m trackrecord construct</code>.</p></main>"), 404)
            return self._html(doc.replace('<main class="wrap">', nav_html("Research · portfolio construction", current="/research/construction") + '<main class="wrap">', 1))
        if u.path == "/research/r-verify":
            from .research_pages import rverify_html
            doc = rverify_html()
            if doc is None:
                return self._html(page("Not built", "<main class='wrap'><h1>Research note not built</h1><p>Run <code>python -m trackrecord r-verify</code> where R and data.table are installed.</p></main>"), 404)
            return self._html(doc.replace('<main class="wrap">', nav_html("Research \u00b7 R reproduction", current="/research/r-verify") + '<main class="wrap">', 1))
        if u.path.startswith("/research/r-verify/") and u.path.endswith(".csv"):
            from .rverify import OUT_DIR as RV_DIR
            f = RV_DIR / u.path.rsplit("/", 1)[1]
            if f.exists() and re.match(r"^[a-z_]+\.csv$", f.name):
                b = f.read_bytes(); self.send_response(200); self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f"attachment; filename={f.name}"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
            return self._html(not_found_html(u.path), 404)
        if u.path.startswith("/research/construction/") and u.path.endswith(".csv"):
            from .construct import OUT_DIR as CON_DIR
            f = CON_DIR / u.path.rsplit("/", 1)[1]
            if f.exists() and re.match(r"^[a-z_]+\.csv$", f.name):
                b = f.read_bytes(); self.send_response(200); self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f"attachment; filename={f.name}"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
            return self._html(not_found_html(u.path), 404)
        if u.path == "/analyze":
            t = (parse_qs(u.query).get("ticker", [""])[0] or "").strip().upper()
            if not valid_ticker(t):
                return self._html(page("Invalid ticker", f"<h1>Invalid ticker</h1><p>{html.escape(t)!r} — use Yahoo Finance format (letters, digits, '.', '-'), e.g. FCNTX or BRK-A.</p><p><a href='/leaderboard'>Back</a></p>"), 400)
            job = start_ticker(t)
            if job["done"] and not job.get("error"):
                self.send_response(302); self.send_header("Location", f"/t/{t}/dashboard.html"); self.send_header("Content-Length", "0"); self.end_headers(); return
            self.send_response(302); self.send_header("Location", f"/t/{t}/"); self.send_header("Content-Length", "0"); self.end_headers(); return
        if u.path.startswith("/t/") and u.path.count("/") == 3 and u.path.endswith("/"):
            t = u.path.split("/")[2]
            job = JOBS.get(t)
            if job and job["done"] and not job.get("error") and (OUT / "t" / t / "dashboard.html").exists():
                self.send_response(302); self.send_header("Location", f"/t/{t}/dashboard.html"); self.send_header("Content-Length", "0"); self.end_headers(); return
            if job is None:
                if (OUT / "t" / t / "dashboard.html").exists():
                    self.send_response(302); self.send_header("Location", f"/t/{t}/dashboard.html"); self.send_header("Content-Length", "0"); self.end_headers(); return
                return self._html(page("Not analyzed", f"<h1>{html.escape(t)}</h1><p>Not analyzed yet. <a href='/analyze?ticker={html.escape(t)}'>Analyze it</a>.</p>"), 404)
            return self._html(ticker_status_html(t, job))
        if u.path.startswith("/f/") and u.path.count("/") == 3 and u.path.endswith("/"):
            slug = u.path.split("/")[2]
            if not re.match(r"^[a-z0-9\-]{1,40}$", slug) or not (FUNDS_ROOT / slug / "meta.json").exists():
                return self._html(page("Unknown manager", f"<h1>Unknown manager</h1><p><a href='/managers'>Managers</a></p>"), 404)
            job = start_fund(slug)
            if job["done"] and not job.get("error") and (OUT / "funds" / slug / "dashboard.html").exists():
                self.send_response(302); self.send_header("Location", f"/funds/{slug}/dashboard.html"); self.send_header("Content-Length", "0"); self.end_headers(); return
            try:
                import json as _j; lbl = _j.loads((FUNDS_ROOT / slug / "meta.json").read_text()).get("name", slug)
            except Exception:
                lbl = slug
            return self._html(ticker_status_html(slug, job, label=lbl))
        mm = re.match(r"^/(f|t)/([A-Za-z0-9.\-]{1,40})/memo$", u.path)
        if mm:
            kind, key = mm.group(1), mm.group(2)
            out = OUT / ("funds" if kind == "f" else "t") / key
            data = (FUNDS_ROOT / key) if kind == "f" else (ROOT / "data" / "tickers" / key)
            if not (out / "phase4" / "regressions.csv").exists():
                return self._redirect(f"/f/{key}/" if kind == "f" else f"/analyze?ticker={key}")
            from .memo import build_memo
            doc = build_memo(out, data)
            if doc is None:
                return self._html(not_found_html(u.path), 404)
            back = f"/funds/{key}/dashboard.html" if kind == "f" else f"/t/{key}/dashboard.html"
            return self._html(doc.replace('<main class="wrap memo">', nav_html("Due-diligence memo", extra=("Dashboard", back)) + '<main class="wrap memo">', 1))
        if u.path.endswith("dashboard.html"):
            f = OUT / u.path.lstrip("/")
            if f.exists():
                doc = f.read_text(encoding="utf-8")
                parts = u.path.strip("/").split("/")
                crumb = {"synthetic": "Synthetic placeholder"}.get(parts[0], "")
                extra = None
                if parts[0] == "funds":
                    try:
                        import json as _j; crumb = _j.loads((FUNDS_ROOT / parts[1] / "meta.json").read_text()).get("name", parts[1]) + " — 13F clone"
                    except Exception:
                        crumb = parts[1]
                    extra = ("Due-diligence memo", f"/f/{parts[1]}/memo")
                elif parts[0] == "t":
                    crumb = parts[1] + " — listed"; extra = ("Due-diligence memo", f"/t/{parts[1]}/memo")
                doc = doc.replace('<main class="wrap">', nav_html(crumb, extra=extra) + '<main class="wrap">', 1)
                return self._html(doc)
            parts = u.path.strip("/").split("/")          # not built (fresh container?) -> build it
            if parts[0] == "funds" and len(parts) == 3 and (FUNDS_ROOT / parts[1] / "meta.json").exists():
                return self._redirect(f"/f/{parts[1]}/")
            if parts[0] == "t" and len(parts) == 3 and valid_ticker(parts[1]):
                return self._redirect(f"/analyze?ticker={parts[1]}")
            if parts[0] == "synthetic":
                return self._html(startup_wait_html(parts[0]))
            return self._html(not_found_html(u.path), 404)
        if u.path in ("/", "/status", "/index.html"):
            body = index_html().encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        return super().do_GET()

    def _redirect(self, to: str):
        self.send_response(302); self.send_header("Location", to); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", "0"); self.end_headers()

    def send_error(self, code, message=None, explain=None):
        """Styled 404 instead of the stdlib 'Error response' page."""
        if code == 404:
            body = not_found_html(self.path).encode("utf-8")
            self.send_response(404); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
            return
        return super().send_error(code, message, explain)

    def _html(self, doc: str, code: int = 200):
        body = doc.encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def guess_type(self, path):
        sp = str(path)
        if sp.endswith((".md", ".txt")):
            return "text/plain; charset=utf-8"
        if sp.endswith((".html", ".htm")):
            return "text/html; charset=utf-8"
        if sp.endswith(".csv"):
            return "text/csv; charset=utf-8"
        return super().guess_type(path)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main(port: int | None = None, build: bool = True) -> None:
    OUT.mkdir(exist_ok=True)
    port = port or int(os.environ.get("PORT", "8080"))
    if build:
        threading.Thread(target=build_all, daemon=True).start()
    else:
        STATE["phase"] = "ready"; STATE["done"] = True
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"serving {OUT} on 0.0.0.0:{port}" + (" (password protected)" if os.environ.get("DASHBOARD_PASSWORD") else " (no password — placeholder only)"), flush=True)
    srv.serve_forever()
