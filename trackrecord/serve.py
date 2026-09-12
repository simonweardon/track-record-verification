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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
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
    rows.sort(key=lambda r: -(r["wealth"] or 0))
    return rows


NAV_CSS = """<style>
.tr-nav{position:sticky;top:0;z-index:6;display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;padding:7px 32px;background:var(--surface,#fdfcf9);border-bottom:1px solid var(--line,#e4dfd2);font:13px "Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;color:var(--ink-2,#6b7078)}
.tr-nav form{display:flex;gap:6px;align-items:center}.tr-nav input{font:inherit;padding:5px 8px;border:1px solid var(--line,#e4dfd2);width:130px;background:var(--surface-2,#f3f0e8);color:inherit;text-transform:uppercase}
.tr-nav button{font:600 10px "Helvetica Neue",Helvetica,Arial,sans-serif;letter-spacing:.16em;text-transform:uppercase;padding:7px 12px;border:0;background:var(--navy,#1b2a41);color:#e8e4da;cursor:pointer}
.tr-nav a{color:var(--navy,#1b2a41);text-decoration:none}.tr-nav .quick a{margin-right:10px;font:600 10px "Helvetica Neue",Helvetica,Arial,sans-serif;letter-spacing:.12em}.tr-nav .lb{margin-left:auto;font-weight:700}
</style>"""


def nav_html() -> str:
    quick = " ".join(f'<a href="/analyze?ticker={t}" title="{html.escape(n)}">{t}</a>' for t, n in FAMOUS[:6])
    return (NAV_CSS + '<div class="tr-nav"><form action="/analyze" method="get"><label for="tk">Analyze any listed fund or stock:</label>'
            '<input id="tk" name="ticker" placeholder="e.g. FCNTX" required pattern="[A-Za-z0-9.\-]{1,12}"><button>Go</button></form>'
            f'<span class="quick">{quick}</span><a class="lb" href="/leaderboard">Leaderboard</a> <a href="/status">All outputs</a></div>')


def page(title: str, body: str, refresh: int | None = None) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>{f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ''}
<style>body{{font:15px/1.5 system-ui,-apple-system,sans-serif;max-width:860px;margin:32px auto;padding:0 20px;color:#141a22;background:#f3f5f7}}
h1{{font-size:24px;margin:0 0 4px}}.sub{{color:#4a5462;margin:0 0 20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px 8px;border-bottom:1px solid #e3e6ea;vertical-align:top;text-align:left}}th{{font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:#8a93a0}}td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
.banner{{background:#fff4d6;border:1px solid #f0c96a;color:#5c4300;padding:10px 14px;border-radius:8px;margin-bottom:20px}}
pre{{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:12px;font-size:12px;overflow-x:auto}}.err{{color:#d03b3b}}
.status{{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;background:#e3e6ea}}a{{color:#2a78d6}}
.sc{{display:inline-block;min-width:34px;text-align:center;padding:2px 8px;border-radius:6px;font-weight:600;color:#fff}}.g{{background:#0ca30c}}.m{{background:#d99a00}}.b{{background:#d03b3b}}
form{{display:flex;gap:8px;margin:12px 0 20px}}input{{font:inherit;padding:8px 10px;border:1px solid #c9ced4;border-radius:8px;flex:1;text-transform:uppercase}}button{{font:inherit;font-weight:600;padding:8px 14px;border:0;border-radius:8px;background:#2a78d6;color:#fff}}
@media(prefers-color-scheme:dark){{body{{background:#0f1317;color:#eef1f4}}td,th{{border-color:#2a313a}}pre{{background:#171c22;border-color:#2a313a}}.banner{{background:#3a2e08;border-color:#7a5f10;color:#ffe6a3}}.status{{background:#2a313a}}input{{background:#171c22;color:#eef1f4;border-color:#2a313a}}}}</style></head>
<body>{body}</body></html>"""


def ticker_status_html(ticker: str, job: dict) -> str:
    log = "\n".join(html.escape(l) for l in job["log"][-12:])
    err = f'<p class="err">{html.escape(job["error"])}</p>' if job.get("error") else ""
    return page(f"{ticker} — building", f"""<h1>Analyzing {html.escape(ticker)}</h1>
<p class="sub"><span class="status">{html.escape(job["phase"])}</span> · {int(time.time() - job["started"])}s · this page refreshes itself</p>
<div class="banner"><b>Public price series.</b> A price feed is not a custodian statement, so every period will show as unverified; the point is the return series and the statistics.</div>
{err}<pre>{log or '(starting)'}</pre><p><a href="/leaderboard">Leaderboard</a> · <a href="/">Home</a></p>""", refresh=None if job["done"] else 5)


def leaderboard_html() -> str:
    rows = leaderboard_rows()
    def sc(v):
        if v is None: return ""
        c = "g" if v >= 70 else ("m" if v >= 45 else "b")
        return f'<span class="sc {c}">{v:.0f}</span>'
    def row(r):
        ex = "" if r["excess"] is None else f"{r['excess'] * 100:+.2f}%"
        return (f"<tr><td><a href='{r['href']}'>{html.escape(r['label'])}</a><br><span style='color:#8a93a0;font-size:12px'>{html.escape(r['key'])}</span></td>"
                f"<td class='n'>{ex}</td><td class='n'>{sc(r['alpha'])}</td><td class='n'>{sc(r['wealth'])}</td></tr>")
    body = "".join(row(r) for r in rows)
    famous = "".join(f"<li><a href='/analyze?ticker={t}'>{t}</a> — {html.escape(n)}</li>" for t, n in FAMOUS)
    return page("Leaderboard", f"""<h1>Leaderboard</h1><p class="sub">Every portfolio analyzed so far, scored on fixed 0–100 maps so they compare directly.</p>
<div class="banner"><b>All placeholder / public data.</b> Nothing here is the record under verification.</div>
<form action="/analyze" method="get"><input name="ticker" placeholder="Enter a ticker — fund, ETF or stock (Yahoo format, e.g. FCNTX, BRK-A)" required pattern="[A-Za-z0-9.\-]{{1,12}}"><button>Analyze</button></form>
<table><thead><tr><th>portfolio</th><th class="n">excess vs market /yr</th><th class="n">alpha-maxing</th><th class="n">wealth-mgmt</th></tr></thead><tbody>{body or '<tr><td colspan=4>nothing analyzed yet</td></tr>'}</tbody></table>
<h3>Famous managers with a listed vehicle</h3><ul>{famous}</ul>
<p class="sub"><b>Not possible from public data:</b> {html.escape(NOT_PUBLIC)} — private funds don't publish returns. Their 13F filings show only US long positions, 45 days late, with no shorts, options, cash or foreign holdings: a clone could be built from those, but it would be a reconstruction, not their record.</p>
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
        if self.path == "/":
            landing = os.environ.get("LANDING", "brk/dashboard.html").lstrip("/")
            if (OUT / landing).exists():          # go straight to the dashboard once it's built
                self.send_response(302); self.send_header("Location", "/" + landing)
                self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", "0")
                self.end_headers(); return
        u = urlparse(self.path)
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
        if u.path == "/leaderboard":
            return self._html(leaderboard_html())
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
