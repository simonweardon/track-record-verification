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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FUNDS_ROOT = ROOT / "data" / "funds"
STATE = {"phase": "starting", "log": [], "started": time.time(), "done": False, "error": None}
JOBS: dict[str, dict] = {}          # ticker -> {phase, log, error, started}
JOBS_LOCK = threading.Lock()

DATASETS = [
    # (label, out subdir, build steps)
    ("Berkshire Hathaway BRK-A (monthly, 1985–2026)", "brk", [
        ["fetch-brk"], ["brk-placeholder"],
        ["report", "--data", "data/brk", "--out", "output/brk", "--grid", "M", "--claimed", "0.198"]]),
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


def build_all() -> None:
    try:
        for label, sub, steps in DATASETS:
            if (OUT / sub / "dashboard.html").exists() and os.environ.get("REBUILD", "1") != "1":
                continue
            STATE["phase"] = f"building {label}"
            _log(f"== {STATE['phase']}")
            for step in steps:
                _run(step)
        STATE["phase"] = "ready"
        _log(f"== ready ({time.time() - STATE['started']:.0f}s after start)")
    except Exception as e:                   # keep serving whatever exists; show the error
        STATE["error"] = str(e)
        STATE["phase"] = "failed"
        _log(f"== FAILED: {e}")
    STATE["done"] = True


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
    cands = [("brk", OUT / "brk"), ("synthetic", OUT / "synthetic")] + \
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
.tr-nav{position:sticky;top:0;z-index:6;display:flex;gap:8px 18px;align-items:center;padding:8px 32px;background:var(--surface,#fdfcf9);border-bottom:1px solid var(--line,#e4dfd2);font:13px "Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;color:var(--ink-2,#6b7078)}
.tr-nav form{display:flex;gap:6px;align-items:center}.tr-nav input{font:inherit;padding:5px 8px;border:1px solid var(--line,#e4dfd2);width:130px;background:var(--surface-2,#f3f0e8);color:inherit;text-transform:uppercase}
.tr-nav button{font:600 10px "Helvetica Neue",Helvetica,Arial,sans-serif;letter-spacing:.16em;text-transform:uppercase;padding:7px 12px;border:0;background:var(--navy,#1b2a41);color:#e8e4da;cursor:pointer}
.tr-nav a{color:var(--navy,#1b2a41);text-decoration:none}.tr-nav .quick a{margin-right:10px;font:600 10px "Helvetica Neue",Helvetica,Arial,sans-serif;letter-spacing:.12em}.tr-nav .lb{margin-left:auto;font-weight:700}
</style>"""


def nav_html() -> str:
    return (NAV_CSS + '<div class="tr-nav"><a href="/">&larr; All managers</a></div>')


def page(title: str, body: str, refresh: int | None = None) -> str:
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
    """The one home page: every fund in the system, one Analyze button each."""
    def sc(v):
        try: v = float(v)
        except (TypeError, ValueError): return ""
        return f'<span class="sc {"g" if v >= 70 else "m" if v >= 45 else "b"}">{v:.0f}</span>'
    # listed vehicles with a real public record
    listed = [dict(key="brk", href="/f/brk/", mgr="Warren Buffett", fund="Berkshire Hathaway Class A — the stock (not the 13F holdings)", tag="listed", out=OUT / "brk")]
    for p in sorted((OUT / "t").glob("*")) if (OUT / "t").exists() else []:
        if (p / "dashboard.html").exists():
            import json
            meta = ROOT / "data" / "tickers" / p.name / "meta.json"
            nm = json.loads(meta.read_text()).get("name", p.name) if meta.exists() else p.name
            listed.append(dict(key=p.name, href=f"/t/{p.name}/dashboard.html", mgr=nm, fund=f"{p.name} — listed, distributions reinvested", tag="listed", out=p))
    def scores_of(out):
        f = out / "phase4" / "scores.csv"
        if not f.exists(): return (None, None, None)
        import csv
        am = wm = ex = None
        with f.open() as fh:
            for r in csv.DictReader(fh):
                if r["score"] == "Alpha-maxing score": am = r["value"]; ex = r["input"]
                if r["score"] == "Wealth-management score" and r["component"] == "TOTAL": wm = r["value"]
        return am, wm, ex
    rows = ""
    for r in listed:
        am, wm, ex = scores_of(r["out"])
        rows += (f"<tr><td><span class='mgr'>{html.escape(r['mgr'])}</span><span class='tag'>{r['tag']}</span><br><span class='fund'>{html.escape(r['fund'])}</span></td>"
                 f"<td class='n'>{_f(ex, '{:+.1%}')}</td><td class='n'>{sc(am)}</td><td class='n'>{sc(wm)}</td><td class='n'><a class='btn' href='{r['href']}'>Analyze</a></td></tr>")
    funds = fund_index()
    ok = [r for r in funds if r.get("status") == "ok"]; bad = [r for r in funds if r.get("status") != "ok"]
    ok.sort(key=lambda r: (-(float(r["wealth"]) if r.get("wealth") not in (None, "") else -1), r["name"]))
    frows = ""
    for r in ok:
        span = f"{(r.get('first') or '')[:4]}–{(r.get('last') or '')[:4]} · {_f(r.get('coverage'), '{:.0%}')} of book priced"
        frows += (f"<tr><td><span class='mgr'>{html.escape(r.get('manager') or r['name'])}</span><span class='tag'>{html.escape(r.get('style_name') or '')}</span><br>"
                  f"<span class='fund'>{html.escape(r['name'])} · {span}</span></td>"
                  f"<td class='n'>{_f(r.get('excess'), '{:+.1%}')}</td><td class='n'>{sc(r.get('alpha_maxing'))}</td><td class='n'>{sc(r.get('wealth'))}</td>"
                  f"<td class='n'><a class='btn' href='/f/{r['slug']}/'>Analyze</a></td></tr>")
    for r in bad:
        frows += (f"<tr><td><span class='mgr'>{html.escape(r.get('manager') or r['name'])}</span><span class='tag'>{html.escape(r.get('style_name') or '')}</span><br>"
                  f"<span class='fund'>{html.escape(r['name'])} · <span class='err'>not scorable yet — {html.escape(str(r.get('months') or 0))} months of filings (36 needed)</span></span></td>"
                  f"<td></td><td></td><td></td><td class='n'><span class='btn off'>Analyze</span></td></tr>")
    head = "<thead><tr><th>manager</th><th class='n'>excess vs market /yr</th><th class='n'>alpha-maxing</th><th class='n'>wealth-mgmt</th><th></th></tr></thead>"
    return page("Track Record Verification", f"""<div class="banner"><b>Illustrative data</b> Public records and SEC 13F reconstructions used to demonstrate the pipeline. Nothing here is the record under verification.</div>
<header class="cover"><div class="cover-in"><div class="eyebrow">Independent performance verification</div><div class="rule"></div><h1>Track Record Verification</h1>
<p class="sub">Every manager in the system. Press <b>Analyze</b> to run the full verification — returns, factor alphas, luck simulation, stability, scores — on that record.</p></div></header>
<main class="wrap">
<h2 data-n="Section 01">Listed vehicles — real public records</h2><p class="note">Share prices with distributions reinvested. These are the vehicles' actual returns.</p>
<table>{head}<tbody>{rows}</tbody></table>
<h2 data-n="Section 02">Hedge funds and family offices — 13F clones</h2><p class="note">Private funds publish no returns. Each is represented by a long-only clone of its disclosed US holdings (SEC 13F, quarterly, from 2013), rebalanced when each filing becomes public. A clone is a reconstruction, not the fund: no shorts, options, cash, leverage or non-US holdings, entered ~45 days late. It tracks concentrated and activist managers well; for multi-strategy, quant and macro shops it is not meaningful, and each page says so. Sorted by wealth-management score; unscored managers last. First analysis of a manager takes about half a minute.</p>
<table>{head}<tbody>{frows or '<tr><td colspan=5>no fund datasets yet</td></tr>'}</tbody></table>
<div class="foot"><b>Scores.</b> Alpha-maxing = 50 + 10 × excess return over the US market (%/yr), return only. Wealth-management = skill evidence 30% + risk-adjusted return 25% + downside protection 25% + consistency over rolling 5-year windows 20%. Fixed maps, comparable across every row. Benchmark and factors: Kenneth R. French Data Library; prices: Yahoo Finance; holdings: SEC EDGAR. Past performance is not indicative of future results; nothing here is investment advice.</div>
</main>""")


def ticker_status_html(ticker: str, job: dict) -> str:
    log = "\n".join(html.escape(l) for l in job["log"][-12:])
    err = f'<p class="err">{html.escape(job["error"])}</p>' if job.get("error") else ""
    return page(f"{ticker} — analyzing", f"""<header class="cover"><div class="cover-in"><div class="eyebrow">Independent performance verification</div><div class="rule"></div><h1>Analyzing {html.escape(ticker)}</h1>
<p class="sub">Running the full pipeline — reconciliation, returns, attribution, factor regressions, luck simulation, scores. This page refreshes itself and opens the dashboard when ready.</p></div></header>
<main class="wrap"><p><span class="status">{html.escape(job["phase"])}</span> &nbsp; {int(time.time() - job["started"])}s</p>{err}<pre>{log or '(starting)'}</pre><p><a href="/">&larr; All managers</a></p></main>""", refresh=None if job["done"] else 5)


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
        if u.path == "/f/brk/":
            self.send_response(302); self.send_header("Location", "/brk/dashboard.html"); self.send_header("Content-Length", "0"); self.end_headers(); return
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
            return self._html(ticker_status_html(slug, job))
        if u.path.endswith("dashboard.html"):
            f = OUT / u.path.lstrip("/")
            if f.exists():
                doc = f.read_text(encoding="utf-8")
                doc = doc.replace('<main class="wrap">', nav_html() + '<main class="wrap">', 1)
                return self._html(doc)
        if u.path in ("/", "/status", "/index.html"):
            body = index_html().encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        return super().do_GET()

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
