"""Serve the generated dashboards over HTTP (for Railway or any host).

- Listens on $PORT immediately; builds the outputs in a background thread so
  the platform's healthcheck passes while the pipeline runs.
- Serves output/ — index page, dashboards, markdown reports, CSVs.
- Optional HTTP Basic Auth: set DASHBOARD_PASSWORD (username is anything).
  Leave unset only while the data is placeholder; real statements must never
  be reachable without it.

    python -m trackrecord serve            # PORT defaults to 8080
"""
from __future__ import annotations

import base64
import html
import os
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
STATE = {"phase": "starting", "log": [], "started": time.time(), "done": False, "error": None}

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
                return got == pw
            except Exception:
                return False
        return False

    def do_GET(self):
        if self.path == "/health":
            body = b"ok"
            self.send_response(200); self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if not self._authorized():
            self.send_response(401); self.send_header("WWW-Authenticate", 'Basic realm="track record"')
            self.send_header("Content-Length", "0"); self.end_headers(); return
        if self.path in ("/", "/index.html"):
            body = index_html().encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        return super().do_GET()

    def guess_type(self, path):
        if str(path).endswith(".md"):
            return "text/plain; charset=utf-8"
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
