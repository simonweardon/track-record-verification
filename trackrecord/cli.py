"""python -m trackrecord <command>"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .load import load, SchemaError
from .reconcile import reconcile, Tolerance
from .coverage import write_report


def cmd_reconcile(a):
    try:
        ds = load(a.data)
    except SchemaError as e:
        print(f"schema error: {e}", file=sys.stderr)
        return 2
    res = reconcile(ds, Tolerance(abs_dollars=a.abs_tol, rel=a.rel_tol))
    path = write_report(res, a.out, title=f"Coverage report — {Path(a.data).resolve()}")
    s = res.summary()
    print(f"{s['accounts']} accounts, {s['statements']} statements: "
          f"{s['verified']} verified, {s['flagged']} flagged, {s['unverified']} unverified; "
          f"{s['errors']} errors, {s['warnings']} warnings, {s['load_problems']} load problems")
    print(f"report: {path}")
    return 1 if (s["flagged"] or s["load_problems"]) else 0


def cmd_returns(a):
    from .returns import period_returns, FeeSchedule
    from .composite import build, CompositeConfig
    from .report2 import write_phase2
    try:
        ds = load(a.data)
    except SchemaError as e:
        print(f"schema error: {e}", file=sys.stderr)
        return 2
    res = reconcile(ds)
    ref = None
    if not a.no_reference:
        from .reference import load_all
        try:
            ref = load_all()
        except Exception as e:
            print(f"warning: reference data unavailable ({e}); benchmarks blank", file=sys.stderr)
    pr = period_returns(res.statements, ds.flows)
    cfg = CompositeConfig(grid=a.grid, include_flagged=a.include_flagged,
                          fee=FeeSchedule(mgmt_pct=a.mgmt_fee, perf_pct=a.perf_fee))
    r2 = build(res.statements, ds.flows, pr, ds.accounts, ref, cfg)
    path = write_phase2(r2, a.out)
    if len(r2.composite):
        from .composite import series_stats
        g = series_stats(r2.composite.gross, r2.composite.years)
        b = series_stats(r2.composite.benchmark, r2.composite.years)
        print(f"composite {g['first']}–{g['last']}: gross {g['annualized']:+.2%}/yr, "
              f"benchmark {b['annualized']:+.2%}/yr" if b else f"composite gross {g['annualized']:+.2%}/yr")
    print(f"report: {path}")
    return 0


def _stack(a):
    """Run every phase; returns (ds, res1, res2, res3, res4, ref)."""
    from .returns import period_returns, FeeSchedule
    from .composite import build as build2, CompositeConfig
    from . import attribution, validation
    from .reference import load_all
    ds = load(a.data)
    res1 = reconcile(ds)
    ref, refA = load_all(), load_all(freq="A")
    pr = period_returns(res1.statements, ds.flows)
    cfg = CompositeConfig(grid=a.grid, include_flagged=a.include_flagged,
                          fee=FeeSchedule(mgmt_pct=a.mgmt_fee, perf_pct=a.perf_fee))
    res2 = build2(res1.statements, ds.flows, pr, ds.accounts, ref, cfg)
    res3 = attribution.build(res2.periods, ds.positions, res2.composite, ref)
    bench = str(res2.accounts.benchmark.mode().iloc[0]) if len(res2.accounts) else cfg.default_benchmark
    primary = "US" if bench.startswith("US") else "DEV"
    vcfg = validation.ValidationConfig(periods_per_year=1 if a.grid == "A" else 12,
                                       factor_set=primary, robustness_set="DEV" if primary == "US" else "US",
                                       n_boot=a.n_boot, n_cohort=a.n_cohort, split_year=a.split_year)
    res4 = validation.build(res2.composite, refA if a.grid == "A" else ref, vcfg, account_periods=res2.periods)
    return ds, res1, res2, res3, res4


def _common(p):
    p.add_argument("--data", default="data/entered")
    p.add_argument("--grid", default="A", choices=["A", "M"])
    p.add_argument("--include-flagged", action="store_true")
    p.add_argument("--mgmt-fee", type=float, default=0.005)
    p.add_argument("--perf-fee", type=float, default=0.20)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--n-cohort", type=int, default=10000)
    p.add_argument("--split-year", type=int, default=2010)


def cmd_attribution(a):
    from .attribution import write_phase3
    try:
        _, _, _, res3, _ = _stack(a)
    except SchemaError as e:
        print(f"schema error: {e}", file=sys.stderr); return 2
    print(f"report: {write_phase3(res3, a.out)}"); return 0


def cmd_validate(a):
    from .validation import write_phase4
    try:
        _, _, _, _, res4 = _stack(a)
    except SchemaError as e:
        print(f"schema error: {e}", file=sys.stderr); return 2
    R = res4.regressions.set_index(["factor_set", "model"]).loc[(res4.config.factor_set, res4.config.headline_model)]
    print(f"{res4.config.headline_model} alpha {R.alpha_annual:+.2%}/yr (95% CI {R.ci_low_annual:+.2%} … {R.ci_high_annual:+.2%}), t={R.t:.2f}, p={R.p:.3f}")
    print(f"report: {write_phase4(res4, a.out)}"); return 0


def cmd_report(a):
    from pathlib import Path
    from .coverage import write_report as write1
    from .report2 import write_phase2
    from .attribution import write_phase3
    from .validation import write_phase4
    from .report import write_report
    try:
        ds, res1, res2, res3, res4 = _stack(a)
    except SchemaError as e:
        print(f"schema error: {e}", file=sys.stderr); return 2
    out = Path(a.out)
    write1(res1, out); write_phase2(res2, out / "phase2"); write_phase3(res3, out / "phase3"); write_phase4(res4, out / "phase4")
    placeholder = a.placeholder or Path(a.data).name in ("synthetic", "brk") or Path(a.data).parent.name in ("tickers", "funds")
    note = a.placeholder_note or {
        "synthetic": "a synthetic dataset built to exercise the pipeline (real market history, invented accounts, +2%/yr injected alpha)",
        "brk": "Berkshire Hathaway Class A's public monthly price series (1985–2026), a real and famous record used to demonstrate the pipeline; a price feed is not a custodian statement, so every period is honestly marked unverified",
    }.get(Path(a.data).name, "placeholder data")
    label = a.label or {"synthetic": "synthetic placeholder accounts", "brk": "Berkshire Hathaway Class A, public price series"}.get(Path(a.data).name, "")
    if not label and (Path(a.data) / "meta.json").exists():
        import json
        m = json.loads((Path(a.data) / "meta.json").read_text())
        if "style_name" in m:                      # 13F clone
            label = f"{m['name']} — 13F long-only clone ({m.get('manager', '')})"
            if not a.placeholder_note:
                note = (f"an SEC 13F clone of {m['name']}'s disclosed US long positions ({m.get('first', '')[:4]}–{m.get('last', '')[:4]}, "
                        f"rebalanced at each filing, {m.get('avg_coverage', 0):.0%} of value priced). A reconstruction, not the fund's return: "
                        f"no shorts, options, cash, leverage or non-US holdings, entered ~45 days late. {m.get('style_note', '')}")
        else:
            label = f"{m.get('name', '')} ({m.get('ticker', '')}), public price series"
        if not a.placeholder_note and "style_name" not in m:
            note = (f"{m.get('name', m.get('ticker'))}'s public monthly price series ({m.get('first', '')[:4]}–{m.get('last', '')[:4]}, "
                    "distributions reinvested), a real listed record used to demonstrate the pipeline; a price feed is not a custodian "
                    "statement, so every period is honestly marked unverified")
    claimed_note = a.claimed_note or {
        "brk": "For this placeholder the 'claim' is Berkshire's own letter figure — 19.8%/yr compounded gain in per-share market value, 1965–2024 — while the verified series covers 1985–2026 only; most of the gap is the missing 1965–84 years, which were the strongest.",
        "synthetic": "For this placeholder the 'claim' is a hypothetical 14%; the true injected alpha is +2%/yr over the market.",
    }.get(Path(a.data).name, "")
    path = write_report(res1, res2, res3, res4, out, a.data, claimed=a.claimed, placeholder=placeholder,
                        placeholder_note=note, claimed_note=claimed_note)
    from .dashboard import build_dashboard
    from .dashboard import build_dashboard
    import os
    dash = build_dashboard(out, claimed=a.claimed, placeholder=placeholder, placeholder_note=note,
                           fee_desc=res2.config.fee.describe(), data_label=label, claimed_note=claimed_note,
                           headline_model=res4.config.headline_model,
                           firm=a.firm or os.environ.get("FIRM_NAME", ""), prepared_for=a.prepared_for or os.environ.get("PREPARED_FOR", ""),
                           data_dir=a.data)
    print(f"report: {path}\ndashboard: {dash}"); return 0


def cmd_funds_build(a):
    """Stage the 13F clones for the whole universe (needs SEC contact; heavy, cached)."""
    import json, os
    from . import hedge13f as H
    contact = a.contact or os.environ.get("SEC_CONTACT")
    if not contact:
        print("SEC requires a contact: --contact 'Name email' or SEC_CONTACT env", file=sys.stderr); return 2
    H.set_contact(contact)
    funds = json.load(open(a.universe))
    only = [x.strip() for x in a.only.split(",")] if a.only else None
    metas = H.build_all(funds, Path(a.out), only=only, figi=not a.no_figi)
    ok = [m for m in metas if m["status"] == "ok"]
    print(f"{len(ok)} funds with >=36 months; {len(metas) - len(ok)} insufficient"); return 0


def cmd_funds_score(a):
    """Run the full report for every fund dataset and write data/funds/leaderboard.csv."""
    import json, csv, time
    root = Path(a.funds)
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists()):
        meta = json.loads((d / "meta.json").read_text())
        row = dict(slug=meta["slug"], name=meta["name"], manager=meta.get("manager", ""), style=meta.get("style", ""),
                   style_name=meta.get("style_name", ""), status=meta["status"], months=meta.get("months"),
                   first=meta.get("first"), last=meta.get("last"), coverage=meta.get("avg_coverage"),
                   excess=None, alpha_maxing=None, wealth=None, ff3_alpha=None, ff3_t=None, twr=None, bench=None)
        out = Path(a.out) / meta["slug"]
        if meta["status"] == "ok" and (a.force or not (out / "phase4" / "scores.csv").exists()):
            t0 = time.time()
            rc = main(["report", "--data", str(d), "--out", str(out), "--grid", "M", "--placeholder",
                       "--n-boot", str(a.n_boot), "--n-cohort", str(a.n_cohort)])
            print(f"  scored {meta['slug']} in {time.time() - t0:.0f}s (rc {rc})")
        sc = out / "phase4" / "scores.csv"
        if sc.exists():
            import pandas as pd
            s_ = pd.read_csv(sc); reg = pd.read_csv(out / "phase4" / "regressions.csv"); comp = pd.read_csv(out / "phase2" / "composite_returns.csv", index_col=0)
            comp.index = pd.PeriodIndex(comp.index, freq="M")
            from .composite import series_stats
            am = s_[s_.score == "Alpha-maxing score"].iloc[0]; wm = s_[(s_.score == "Wealth-management score") & (s_.component == "TOTAL")].iloc[0]
            r = reg[(reg.factor_set == "US") & (reg.model == "FF3")].iloc[0]
            g = series_stats(comp.gross, comp.years); b = series_stats(comp.benchmark, comp.years)
            row.update(excess=float(am.input), alpha_maxing=float(am.value), wealth=float(wm.value), ff3_alpha=float(r.alpha_annual),
                       ff3_t=float(r.t), twr=g["annualized"], bench=b["annualized"])
        rows.append(row)
    with open(root / "leaderboard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"leaderboard: {len(rows)} funds -> {root / 'leaderboard.csv'}"); return 0


def cmd_serve(a):
    from .serve import main as serve_main
    serve_main(port=a.port, build=not a.no_build)
    return 0


def cmd_fetch_brk(a):
    """Download BRK-A monthly history via yfinance into data/reference (not committed)."""
    import yfinance as yf
    from pathlib import Path
    h = yf.Ticker("BRK-A").history(period="max", interval="1mo", auto_adjust=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(a.out)
    print(f"wrote {len(h)} monthly rows to {a.out}"); return 0


def cmd_fetch_ticker(a):
    from .placeholder_ticker import fetch, valid
    if not valid(a.ticker):
        print(f"invalid ticker {a.ticker!r}", file=sys.stderr); return 2
    m = fetch(a.ticker, a.out)
    print(f"{m['name']} ({m['ticker']}): {m['rows']} monthly rows {m['first']}..{m['last']} -> {a.out}"); return 0


def cmd_ticker(a):
    from .placeholder_ticker import build
    out = build(a.ticker, a.raw, a.out)
    print(f"wrote {a.ticker} placeholder to {out}"); return 0


def cmd_brk(a):
    from .placeholder_brk import build
    out = build(a.raw, a.out)
    print(f"wrote Berkshire (BRK-A) placeholder to {out}  — run: report --data {out} --grid M --claimed 0.198")
    return 0


def cmd_synth(a):
    from .synthetic import generate, write
    ds = generate(seed=a.seed, n_accounts=a.accounts, defects=not a.clean)
    write(ds, a.out)
    print(f"wrote synthetic dataset ({a.accounts} accounts, "
          f"{'with' if not a.clean else 'no'} injected defects) to {a.out}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="trackrecord")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reconcile", help="load normalized CSVs, reconcile, write coverage report")
    r.add_argument("--data", default="data/entered")
    r.add_argument("--out", default="output")
    r.add_argument("--abs-tol", type=float, default=1.00)
    r.add_argument("--rel-tol", type=float, default=1e-4)
    r.set_defaults(fn=cmd_reconcile)

    t = sub.add_parser("returns", help="Phase 2: return series, composite, benchmarks, IRR")
    t.add_argument("--data", default="data/entered")
    t.add_argument("--out", default="output/phase2")
    t.add_argument("--grid", default="A", choices=["A", "M"])
    t.add_argument("--include-flagged", action="store_true")
    t.add_argument("--no-reference", action="store_true", help="skip benchmark data")
    t.add_argument("--mgmt-fee", type=float, default=0.005)
    t.add_argument("--perf-fee", type=float, default=0.20)
    t.set_defaults(fn=cmd_returns)

    for name, fn, out, help_ in [("attribution", cmd_attribution, "output/phase3", "Phase 3: beta / allocation / timing attribution"),
                                 ("validate", cmd_validate, "output/phase4", "Phase 4: regressions, bootstrap, cohort, rolling, sub-periods"),
                                 ("report", cmd_report, "output", "run every phase and write REPORT.md")]:
        q = sub.add_parser(name, help=help_)
        _common(q); q.add_argument("--out", default=out)
        if name == "report":
            q.add_argument("--claimed", type=float, default=None, help="claimed annualized return, e.g. 0.14")
            q.add_argument("--placeholder", action="store_true", help="stamp the report as placeholder data")
            q.add_argument("--placeholder-note", default=None, help="what the placeholder data is")
            q.add_argument("--claimed-note", default=None, help="where the claimed figure comes from")
            q.add_argument("--label", default=None, help="dataset label shown under the dashboard title")
            q.add_argument("--firm", default=None, help="wordmark / 'prepared by' on the cover (or env FIRM_NAME)")
            q.add_argument("--prepared-for", default=None, help="'prepared for' line on the cover (or env PREPARED_FOR)")
        q.set_defaults(fn=fn)

    fbld = sub.add_parser("funds-build", help="build 13F clone datasets for the fund universe (SEC EDGAR)")
    fbld.add_argument("--universe", default="data/reference/edgar/funds.json"); fbld.add_argument("--out", default="data/funds")
    fbld.add_argument("--contact", default=None, help="SEC User-Agent 'Name email' (or SEC_CONTACT env)")
    fbld.add_argument("--only", default=None, help="comma-separated slugs"); fbld.add_argument("--no-figi", action="store_true")
    fbld.set_defaults(fn=cmd_funds_build)
    fsc = sub.add_parser("funds-score", help="run the report for every fund dataset; write leaderboard.csv")
    fsc.add_argument("--funds", default="data/funds"); fsc.add_argument("--out", default="output/funds")
    fsc.add_argument("--n-boot", type=int, default=1000); fsc.add_argument("--n-cohort", type=int, default=2000)
    fsc.add_argument("--force", action="store_true")
    fsc.set_defaults(fn=cmd_funds_score)

    sg = sub.add_parser("signals13f", help="13F research: best-ideas / crowding / conviction portfolios, spreads, ICs -> data/research/13f-signals")
    sg.set_defaults(fn=lambda a: (__import__("trackrecord.signals13f", fromlist=["build"]).build(), 0)[1])
    ct = sub.add_parser("construct", help="LP portfolio construction demo: backtest, latest trade list, R twin check -> data/research/construction")
    ct.set_defaults(fn=lambda a: (__import__("trackrecord.construct", fromlist=["backtest"]).backtest(), 0)[1])
    cp = sub.add_parser("compact-pack", help="pack the EDGAR/price/factor caches into the committed data/reference/compact bundle")
    cp.set_defaults(fn=lambda a: (__import__("trackrecord.compact", fromlist=["pack"]).pack(), 0)[1])
    cu = sub.add_parser("compact-unpack", help="rebuild data/reference caches from the committed bundle (fresh clone)")
    cu.add_argument("--force", action="store_true")
    cu.set_defaults(fn=lambda a: (__import__("trackrecord.compact", fromlist=["unpack"]).unpack(force=a.force), 0)[1])

    sv = sub.add_parser("serve", help="serve output/ over HTTP; builds placeholder outputs in the background")
    sv.add_argument("--port", type=int, default=None)
    sv.add_argument("--no-build", action="store_true", help="serve existing outputs only")
    sv.set_defaults(fn=cmd_serve)

    fb = sub.add_parser("fetch-brk", help="download BRK-A monthly prices (yfinance) to data/reference")
    fb.add_argument("--out", default="data/reference/brk-a_monthly_raw.csv")
    fb.set_defaults(fn=cmd_fetch_brk)

    ft = sub.add_parser("fetch-ticker", help="download any listed vehicle's monthly adjusted history (yfinance)")
    ft.add_argument("--ticker", required=True); ft.add_argument("--out", required=True)
    ft.set_defaults(fn=cmd_fetch_ticker)
    tp = sub.add_parser("ticker-placeholder", help="build a one-account placeholder from a fetched ticker CSV")
    tp.add_argument("--ticker", required=True); tp.add_argument("--raw", required=True); tp.add_argument("--out", required=True)
    tp.set_defaults(fn=cmd_ticker)

    b = sub.add_parser("brk-placeholder", help="build the Berkshire BRK-A monthly placeholder dataset")
    b.add_argument("--raw", default="data/reference/brk-a_monthly_raw.csv")
    b.add_argument("--out", default="data/brk")
    b.set_defaults(fn=cmd_brk)

    s = sub.add_parser("synth", help="generate a synthetic dataset for testing the pipeline")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--accounts", type=int, default=12)
    s.add_argument("--clean", action="store_true", help="no injected defects")
    s.set_defaults(fn=cmd_synth)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
