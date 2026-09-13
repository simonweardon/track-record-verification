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
        for label, sub, steps in DATASETS + _ticker_datasets():
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
.tr-nav{position:sticky;top:0;z-index:6;display:flex;gap:10px;align-items:center;padding:8px 32px;background:var(--surface,#fdfcf9);border-bottom:1px solid var(--line,#e4dfd2);font:13px "Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;color:var(--ink-2,#6b7078)}
.tr-nav a,.tr-nav button{display:inline-flex;align-items:center;gap:6px;font:600 10px "Helvetica Neue",Helvetica,Arial,sans-serif;letter-spacing:.18em;text-transform:uppercase;padding:8px 14px;border:1px solid var(--navy,#1b2a41);background:transparent;color:var(--navy,#1b2a41);text-decoration:none;cursor:pointer}
.tr-nav a.home{background:var(--navy,#1b2a41);color:var(--cover-ink,#e8e4da)}
.tr-nav .crumb{margin-left:auto;font:600 9px "Helvetica Neue",Helvetica,Arial,sans-serif;letter-spacing:.2em;text-transform:uppercase;color:var(--muted,#a09883)}
@media(prefers-color-scheme:dark){.tr-nav a,.tr-nav button{border-color:var(--gold-l,#c9b48a);color:var(--gold-l,#c9b48a)}.tr-nav a.home{background:var(--gold-l,#c9b48a);color:#1b2a40}}
</style>"""


def nav_html(crumb: str = "") -> str:
    return (NAV_CSS + '<div class="tr-nav"><button type="button" onclick="history.length>1?history.back():location.assign(\'/\')">&larr; Back</button>'
            '<a class="home" href="/">Home</a>' + (f'<span class="crumb">{html.escape(crumb)}</span>' if crumb else '') + '</div>')


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
        if not f.exists(): return (None, None, None)
        am = wm = ex = None
        with f.open() as fh:
            for r in csv.DictReader(fh):
                if r["score"] == "Alpha-maxing score": am = r["value"]; ex = r["input"]
                if r["score"] == "Wealth-management score" and r["component"] == "TOTAL": wm = r["value"]
        return am, wm, ex
    # ---- listed vehicles: BRK stock + any committed ticker dataset
    listed = []
    am, wm, ex = scores_of(OUT / "brk")
    listed.append(dict(key="brk", href="/f/brk/", mgr="Warren Buffett", fund="Berkshire Hathaway Class A — the share price, not the 13F holdings", tag="listed · 1985–2026",
                       alpha=am, wealth=wm, excess=ex, ok=True))
    tdir = ROOT / "data" / "tickers"
    if tdir.exists():
        for d in sorted(tdir.iterdir()):
            m = d / "meta.json"
            if not m.exists(): continue
            meta = json.loads(m.read_text())
            am, wm, ex = scores_of(OUT / "t" / d.name)
            listed.append(dict(key=d.name, href=f"/analyze?ticker={d.name}", mgr=meta.get("name", d.name), fund=f"{d.name} — listed, distributions reinvested",
                               tag=f"listed · {(meta.get('first') or '')[:4]}–{(meta.get('last') or '')[:4]}", alpha=am, wealth=wm, excess=ex, ok=True))
    # ---- funds
    funds = fund_index()
    frows = []
    for r in funds:
        ok = r.get("status") == "ok"
        frows.append(dict(key=r["slug"], href=f"/f/{r['slug']}/", mgr=r.get("manager") or r["name"], fund=r["name"], style=r.get("style_name") or "",
                          tag=(f"13F clone · {(r.get('first') or '')[:4]}–{(r.get('last') or '')[:4]} · {_f(r.get('coverage'), '{:.0%}')} priced" if ok
                               else f"not scorable yet — {r.get('months') or 0} months of usable filings (36 needed)"),
                          alpha=r.get("alpha_maxing"), wealth=r.get("wealth"), excess=r.get("excess"), ok=ok))
    n_ok = sum(1 for r in frows if r["ok"])
    def row(r, kind):
        a, w, e = num(r["alpha"]), num(r["wealth"]), num(r["excess"])
        btn = f"<a class='btn' href='{r['href']}'>Analyze</a>" if r["ok"] else "<span class='btn off'>Analyze</span>"
        search = html.escape(f"{r['mgr']} {r['fund']} {r.get('style', '')} {kind}".lower(), quote=True)
        return (f"<tr data-s='{search}' data-w='{w if w is not None else -1}' data-a='{a if a is not None else -1}' data-e='{e if e is not None else -99}'>"
                f"<td><span class='mgr'>{html.escape(r['mgr'])}</span>{'<span class=tag>' + html.escape(r['style']) + '</span>' if r.get('style') else ''}<br>"
                f"<span class='fund'>{html.escape(r['fund'])} · {html.escape(r['tag'])}</span></td>"
                f"<td class='n'>{_f(r['excess'], '{:+.1%}')}</td><td class='n'>{sc(r['alpha'])}</td><td class='n'>{sc(r['wealth'])}</td><td class='n'>{btn}</td></tr>")
    ordered = sorted(frows, key=lambda r: (-(num(r["wealth"]) if num(r["wealth"]) is not None else -1), r["mgr"]))
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
    head = "<thead><tr><th>manager</th><th class='n'>excess vs market /yr</th><th class='n'>alpha-maxing</th><th class='n'>wealth-mgmt</th><th></th></tr></thead>"
    extra_css = """<style>
.stats{display:flex;gap:40px;margin-top:30px;padding-top:18px;border-top:1px solid rgba(232,228,218,.18)}.stats div{display:grid;gap:4px}.stats dt{font:600 9px/1 var(--sans);letter-spacing:.22em;text-transform:uppercase;color:var(--covermuted)}.stats dd{margin:0;font:400 30px/1 var(--serif);color:var(--coverink)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin:6px 0 10px}
.card{display:block;background:var(--surface);border:1px solid var(--line);border-top:2px solid var(--gold);padding:16px 18px;text-decoration:none;color:var(--ink)}
.card .who{font:400 20px/1.15 var(--serif);color:var(--navy)}.card .what{color:var(--ink2);font-size:13px;margin:2px 0 12px}
.card .nums{display:flex;gap:16px;font:600 13px var(--sans)}.card .nums div{display:grid;gap:4px}.card .lbl{font:600 8.5px/1 var(--sans);letter-spacing:.18em;text-transform:uppercase;color:var(--muted)}
.card .go{margin-top:12px;font:600 9.5px/1 var(--sans);letter-spacing:.18em;text-transform:uppercase;color:var(--gold)}
.tools{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:14px 0 10px}.tools input{font:15px var(--serif);padding:8px 12px;border:1px solid var(--line);background:var(--surface);color:var(--ink);flex:1;min-width:220px}
.tools .sort{font:600 9.5px/1 var(--sans);letter-spacing:.16em;text-transform:uppercase;color:var(--ink2)}.tools button{font:600 9.5px/1 var(--sans);letter-spacing:.16em;text-transform:uppercase;padding:9px 12px;border:1px solid var(--line);background:var(--surface);color:var(--navy);cursor:pointer}.tools button.on{background:var(--navy);color:var(--coverink);border-color:var(--navy)}
.count{font:600 9.5px/1 var(--sans);letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin-left:auto}
</style>"""
    js = """<script>
(function(){const q=document.getElementById('q'),rows=[...document.querySelectorAll('#tbl tbody tr')],cnt=document.getElementById('cnt');
function apply(){const s=q.value.trim().toLowerCase();let n=0;rows.forEach(r=>{const ok=!s||r.dataset.s.includes(s);r.hidden=!ok;if(ok)n++;});cnt.textContent=n+' of '+rows.length;}
q.addEventListener('input',apply);
document.querySelectorAll('.tools button').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.tools button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
const k=b.dataset.k,tb=document.querySelector('#tbl tbody');[...tb.querySelectorAll('tr')].sort((x,y)=>parseFloat(y.dataset[k])-parseFloat(x.dataset[k])).forEach(r=>tb.appendChild(r));}));
apply();})();
</script>"""
    return page("Track Record Verification", f"""{extra_css}<div class="banner"><b>Illustrative data</b> Public records and SEC 13F reconstructions used to demonstrate the pipeline. Nothing here is the record under verification.</div>
<header class="cover"><div class="cover-in"><div class="eyebrow">Independent performance verification</div><div class="rule"></div><h1>Track Record Verification</h1>
<p class="sub">One system, applied the same way to every manager: reconcile the record, compute time-weighted returns, remove what the market and known factors explain, simulate how often luck alone does as well, test stability, and score. Press <b>Analyze</b> on any row.</p>
<dl class="stats"><div><dt>Managers in the system</dt><dd>{len(frows) + len(listed)}</dd></div><div><dt>Scorable today</dt><dd>{n_ok + len(listed)}</dd></div><div><dt>Listed vehicles</dt><dd>{len(listed)}</dd></div><div><dt>13F clones</dt><dd>{len(frows)}</dd></div></dl>
</div></header>
<main class="wrap">
<h2 data-n="Featured">Start here</h2><p class="note">Three managers seen through their disclosed holdings (13F clones), and a listed fund with a 35-year real record. Berkshire's <i>stock</i> is in the table below, separately.</p>
<div class="cards">{cards}</div>
<h2 data-n="All managers">Every manager in the system</h2>
<p class="note">Listed vehicles are actual returns (share price, distributions reinvested). Hedge funds and family offices are <b>13F long-only clones</b>: their disclosed US holdings at disclosed weights, rebalanced when each quarterly filing becomes public — a reconstruction, not the fund. No shorts, options, cash, leverage or non-US holdings; entered ~45 days late; months with too little of the book priced are left out and never bridged. Concentrated, activist and long-short clones track the real book; multi-strategy, quant and macro clones are flagged as not meaningful on their pages.</p>
<div class="tools"><input id="q" placeholder="Search a manager, fund or strategy — e.g. Tepper, activist, quant" autocomplete="off"><span class="sort">Sort</span><button data-k="w" class="on">Wealth-mgmt</button><button data-k="a">Alpha-maxing</button><button data-k="e">Excess return</button><span class="count" id="cnt"></span></div>
<table id="tbl">{head}<tbody>{table}</tbody></table>
<div class="foot"><b>Scores.</b> Alpha-maxing = 50 + 10 × excess return over the US market (%/yr), return only. Wealth-management = skill evidence 30% + risk-adjusted return 25% + downside protection 25% + consistency over rolling 5-year windows 20%. Fixed maps, comparable across every row. Benchmark and factors: Kenneth R. French Data Library; prices: Yahoo Finance; holdings: SEC EDGAR. First analysis of a manager takes about half a minute; afterwards it opens instantly. Past performance is not indicative of future results; nothing here is investment advice.</div>
</main>{js}""")


def ticker_status_html(ticker: str, job: dict, label: str | None = None) -> str:
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
{'<p><a class="btn" href="' + ('/t/' + ticker + '/dashboard.html' if not ticker.islower() else '/funds/' + ticker + '/dashboard.html') + '">Open the dashboard</a></p>' if ready else ''}
<p style="display:flex;gap:10px"><a class="btn" href="/">Home</a><a class="btn" href="javascript:history.back()" style="background:transparent;color:var(--navy);border:1px solid var(--navy)">&larr; Back</a></p></main>{poll}""", refresh=None if job["done"] else 5)


def startup_wait_html(which: str) -> str:
    label = {"brk": "Berkshire Hathaway (the stock)", "synthetic": "the synthetic placeholder"}.get(which, which)
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
            if (OUT / "brk" / "dashboard.html").exists():
                return self._redirect("/brk/dashboard.html")
            return self._html(startup_wait_html("brk"))
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
        if u.path.endswith("dashboard.html"):
            f = OUT / u.path.lstrip("/")
            if f.exists():
                doc = f.read_text(encoding="utf-8")
                parts = u.path.strip("/").split("/")
                crumb = {"brk": "Berkshire Hathaway — the stock", "synthetic": "Synthetic placeholder"}.get(parts[0], "")
                if parts[0] == "funds":
                    try:
                        import json as _j; crumb = _j.loads((FUNDS_ROOT / parts[1] / "meta.json").read_text()).get("name", parts[1]) + " — 13F clone"
                    except Exception:
                        crumb = parts[1]
                elif parts[0] == "t":
                    crumb = parts[1] + " — listed"
                doc = doc.replace('<main class="wrap">', nav_html(crumb) + '<main class="wrap">', 1)
                return self._html(doc)
            parts = u.path.strip("/").split("/")          # not built (fresh container?) -> build it
            if parts[0] == "funds" and len(parts) == 3 and (FUNDS_ROOT / parts[1] / "meta.json").exists():
                return self._redirect(f"/f/{parts[1]}/")
            if parts[0] == "t" and len(parts) == 3 and valid_ticker(parts[1]):
                return self._redirect(f"/analyze?ticker={parts[1]}")
            if parts[0] in ("brk", "synthetic"):
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
