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
    R = res4.regressions.set_index(["factor_set", "model"]).loc[(res4.config.factor_set, "CAPM")]
    print(f"CAPM alpha {R.alpha_annual:+.2%}/yr (95% CI {R.ci_low_annual:+.2%} … {R.ci_high_annual:+.2%}), t={R.t:.2f}, p={R.p:.3f}")
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
    placeholder = a.placeholder or Path(a.data).name in ("synthetic", "brk")
    note = a.placeholder_note or {
        "synthetic": "a synthetic dataset built to exercise the pipeline (real market history, invented accounts, +2%/yr injected alpha)",
        "brk": "Berkshire Hathaway Class A's public monthly price series (1985–2026), a real and famous record used to demonstrate the pipeline; a price feed is not a custodian statement, so every period is honestly marked unverified",
    }.get(Path(a.data).name, "placeholder data")
    path = write_report(res1, res2, res3, res4, out, a.data, claimed=a.claimed, placeholder=placeholder,
                        placeholder_note=note)
    from .dashboard import build_dashboard
    label = {"synthetic": "synthetic placeholder accounts", "brk": "Berkshire Hathaway Class A, public price series"}.get(Path(a.data).name, "")
    dash = build_dashboard(out, claimed=a.claimed, placeholder=placeholder, placeholder_note=note,
                           fee_desc=res2.config.fee.describe(), data_label=label)
    print(f"report: {path}\ndashboard: {dash}"); return 0


def cmd_fetch_brk(a):
    """Download BRK-A monthly history via yfinance into data/reference (not committed)."""
    import yfinance as yf
    from pathlib import Path
    h = yf.Ticker("BRK-A").history(period="max", interval="1mo", auto_adjust=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(a.out)
    print(f"wrote {len(h)} monthly rows to {a.out}"); return 0


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
        q.set_defaults(fn=fn)

    fb = sub.add_parser("fetch-brk", help="download BRK-A monthly prices (yfinance) to data/reference")
    fb.add_argument("--out", default="data/reference/brk-a_monthly_raw.csv")
    fb.set_defaults(fn=cmd_fetch_brk)

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
