"""Phase 2 outputs: CSVs plus summary.md that states what is verified, what is
modeled, and what is missing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .composite import Phase2Result, series_stats, trailing


def pct(x, d=2):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:+.{d}f}%"


def span(s: dict) -> str:
    if not s:
        return ""
    t = f"{s['first']}–{s['last']}"
    return t + (" ⚠ longest contiguous run; a hole exists outside it" if s.get("truncated") else "")


def money(x):
    return "n/a" if pd.isna(x) else f"${x:,.0f}"


def _table(df: pd.DataFrame, cols: list[tuple[str, str, callable]]) -> list[str]:
    L = ["| " + " | ".join(h for _, h, _ in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for idx, r in df.iterrows():
        cells = []
        for c, _, f in cols:
            v = idx if c == "__index__" else r[c]
            cells.append(f(v))
        L.append("| " + " | ".join(cells) + " |")
    return L


def write_phase2(res: Phase2Result, out_dir: str | Path, title="Phase 2 — return series and composite") -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pr, comp, acc, irr, cfg = res.periods, res.composite, res.accounts, res.irr, res.config

    p = pr.copy()
    for c in ["period_start", "period_end", "eff_start", "eff_end"]:
        p[c] = p[c].dt.date
    p.to_csv(out / "account_period_returns.csv", index=False)
    comp.to_csv(out / "composite_returns.csv")
    acc.to_csv(out / "account_summary.csv", index=False)
    irr.to_csv(out / "irr.csv", index=False)

    L = [f"# {title}", ""]
    if comp.empty:
        L += ["No composite-eligible periods. See account_period_returns.csv exclusion_reason."]
        (out / "summary.md").write_text("\n".join(L)); return out / "summary.md"

    g = series_stats(comp.gross, comp.years)
    n = series_stats(comp.model_net, comp.years)
    b = series_stats(comp.benchmark, comp.years)
    bu = series_stats(comp.benchmark_us, comp.years)
    unit = "year" if cfg.grid == "A" else "month"

    L += ["## Headline (composite of all discretionary accounts, closed accounts included)", ""]
    L += [f"Period: **{span(g)}** ({g['periods']} {unit}s, {g['years']:.1f} years). "
          f"Accounts per {unit}: {int(comp.n.min())}–{int(comp.n.max())}.", ""]
    L += ["| series | annualized | cumulative | growth of $1 | best {u} | worst {u} | negative {u}s |".format(u=unit),
          "|---|---|---|---|---|---|---|"]
    for name, s in [("Composite gross (= net: no fees were charged)", g),
                    (f"Composite model-net ({cfg.fee.describe()}) — ASSUMED schedule", n),
                    ("Benchmark (asset-weighted account benchmarks)", b),
                    (f"US market ({cfg.secondary_benchmark})", bu)]:
        if s:
            L.append(f"| {name} | **{pct(s['annualized'])}** | {pct(s['cumulative'], 0)} | "
                     f"{s['growth_of_1']:.2f} | {pct(s['best'], 1)} | {pct(s['worst'], 1)} | {s['negative_periods']} |")
    if g and b:
        L += ["", f"Excess over benchmark: **{pct(g['annualized'] - b['annualized'])}/yr** gross, "
              f"{pct(n['annualized'] - b['annualized'])}/yr model-net. "
              f"Composite outperformed its benchmark in {int((comp.excess > 0).sum())} of "
              f"{int(comp.excess.notna().sum())} {unit}s."]
    L.append("")

    # trailing windows
    per_year = 1 if cfg.grid == "A" else 12
    L += [f"## Trailing windows (ending {g['last']})", "",
          "| window | gross | model-net | benchmark | excess (gross) |", "|---|---|---|---|---|"]
    for yrs_ in [1, 3, 5, 10, 15, 20, 25, 30]:
        k = yrs_ * per_year
        if k > len(comp):
            continue
        L.append(f"| {yrs_}y | {pct(trailing(comp, 'gross', k))} | {pct(trailing(comp, 'model_net', k))} | "
                 f"{pct(trailing(comp, 'benchmark', k))} | "
                 f"{pct(trailing(comp, 'gross', k) - trailing(comp, 'benchmark', k))} |")
    L.append(f"| since inception | {pct(g['annualized'])} | {pct(n['annualized'])} | "
             f"{pct(b.get('annualized'))} | {pct(g['annualized'] - b['annualized']) if b else 'n/a'} |")
    L.append("")

    # survivorship
    L += ["## Survivorship check", ""]
    surv = series_stats(comp.survivors_only, comp.years)
    closed = acc[acc.status != "open"]
    if surv:
        diff = surv["annualized"] - g["annualized"]
        L += [f"- All discretionary accounts (correct): **{pct(g['annualized'])}** ({span(g)})",
              f"- Only accounts still open today (what a naive analysis would show): **{pct(surv['annualized'])}** ({span(surv)})",
              f"- Survivorship effect: **{pct(diff)}/yr**"
              + (" — the naive number is inflated." if diff > 0.0005 else
                 " — closed accounts did not flatter or hurt the record materially." if abs(diff) <= 0.0005
                 else " — closed accounts actually outperformed survivors.")]
    if len(closed):
        L += ["", "Closed / terminated accounts (included through their last full period):", ""]
        for _, a in closed.iterrows():
            L.append(f"- {a.account_id} ({a.label}): {a.inception} → {a.last_period_end}, "
                     f"TWR {pct(a.twr_annualized)} vs benchmark {pct(a.bench_annualized)} over {a.twr_span}")
    else:
        L += ["", "No closed accounts in the data. If any account ever closed and its statements are "
              "missing, this composite is subject to survivorship bias of unknown size — say so."]
    L.append("")

    # principal vs clients
    pri = series_stats(comp.principal_only, comp.years)
    cli = series_stats(comp.clients_only, comp.years)
    L += ["## Principal's own account vs client accounts", ""]
    if pri:
        L.append(f"- Principal only: **{pct(pri['annualized'])}** ({span(pri)})")
    if cli:
        L.append(f"- Clients only: **{pct(cli['annualized'])}** ({span(cli)})")
    if pri and cli:
        same = pri["first"] == cli["first"] and pri["last"] == cli["last"]
        L.append(f"- Gap: {pct(pri['annualized'] - cli['annualized'])}/yr"
                 + ("." if same else " — **spans differ**, so this is not a like-for-like comparison; "
                    "compare year by year in the table below."))
    L.append("")

    # money-weighted
    pooled = irr[irr.account_id == "POOLED_DISCRETIONARY"]
    L += ["## Money-weighted (IRR) vs time-weighted", ""]
    if len(pooled):
        L.append(f"- Pooled IRR over each discretionary account's longest clean segment: **{pct(float(pooled.irr.iloc[0]))}**")
    L.append(f"- Time-weighted composite: **{pct(g['annualized'])}**")
    L += ["- TWR is what the manager controls; IRR is what the dollars earned. If IRR < TWR, money "
          "arrived after the good years (or left before them). Both are reported; TWR is the "
          "GIPS-standard measure for a manager's record.", ""]

    # year table
    L += [f"## {unit.capitalize()}-by-{unit}", "",
          f"| {unit} | n | gross | model-net | benchmark | excess | US mkt | dispersion sd | high | low | unverified | excl. flagged |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in comp.iterrows():
        L.append(f"| {c} | {int(r.n)} | {pct(r.gross)} | {pct(r.model_net)} | {pct(r.benchmark)} | {pct(r.excess)} | "
                 f"{pct(r.benchmark_us)} | {pct(r.dispersion_sd) if not pd.isna(r.dispersion_sd) else '—'} | "
                 f"{pct(r.high, 1) if not pd.isna(r.high) else '—'} | {pct(r.low, 1) if not pd.isna(r.low) else '—'} | "
                 f"{int(r.n_unverified)} | {int(r.n_excluded_flagged)} |")
    L.append("")

    # accounts
    L += ["## Accounts", "",
          "| account | label | type | discr. | status | inception | last | TWR span | TWR ann. | bench ann. | IRR | naive CAGR (ignores flows) | arith. mean | evidence V/U/X |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, a in acc.iterrows():
        L.append(f"| {a.account_id} | {a.label} | {a.owner_type} | {a.discretionary} | {a.status} | {a.inception} | "
                 f"{a.last_period_end} | {a.twr_span} | {pct(a.twr_annualized)} | {pct(a.bench_annualized)} | "
                 f"{pct(a.irr)}{(' (' + a.irr_note + ')') if a.irr_note else ''} | {pct(a.naive_cagr_ignoring_flows)} | "
                 f"{pct(a.arithmetic_mean)} | {a.verified}/{a.unverified}/{a.flagged} |")
    L += ["", "*naive CAGR* = (ending value / beginning value)^(1/years) − 1 with deposits counted as if "
          "they were gains, and *arithmetic mean* of annual returns, are shown because they are the "
          "two most common ways a self-calculated number ends up higher than the true time-weighted "
          "return. Neither is a performance measure.", ""]

    # evidence
    tot = int(comp.n.sum())
    L += ["## Evidence behind the composite", "",
          f"- Account-{unit}s in composite: **{tot}** — verified {int(comp.n_verified.sum())}, "
          f"unverified {int(comp.n_unverified.sum())}",
          f"- Excluded as flagged (reconciliation failed; fix or disclose): {int(comp.n_excluded_flagged.sum())}",
          f"- Excluded as partial first/last periods: {int(comp.n_excluded_partial.sum())}",
          f"- Excluded for other reasons (non-discretionary, no beginning value, gap): {int(comp.n_excluded_other.sum())}"]
    ex = pr[pr.exclusion_reason == "flagged"]
    if len(ex):
        L += ["", "Flagged account-periods (re-read the page; fix the entry or disclose):", ""]
        for _, x in ex.iterrows():
            L.append(f"- {x.statement_id}: {x['flags']}")
    nr = pr[pr.exclusion_reason == "no_return"]
    if len(nr):
        L += ["", f"{len(nr)} period(s) with no computable return (no printed beginning value at "
              f"inception or after a gap): " + ", ".join(nr.statement_id)]
    L.append("")

    # method
    L += ["## Method and caveats", "",
          f"- Grid: {'calendar years' if cfg.grid == 'A' else 'calendar months'}. Period returns are Modified "
          "Dietz with day-weighted external flows; composite is the aggregate method (all members pooled as "
          "one portfolio). Asset- and equal-weighted returns in composite_returns.csv as cross-checks.",
          "- Annual statements cannot support GIPS's monthly-valuation requirement; this composite is "
          "*GIPS-methodology-inspired*, not GIPS-compliant, until monthly data covers the period.",
          f"- Model-net applies {cfg.fee.describe()} to the composite gross series on a NAV index. "
          "The schedule is an assumption pending confirmation of the RIA's actual fee terms.",
          "- Benchmarks are Ken French total-market series (Mkt-RF + RF): DEV_MKT = developed markets "
          "incl. US; US_MKT = US total market; DXUS_MKT = developed ex-US. Blends are monthly rebalanced.",
          "- Partial first/last periods are measured from first flow / to last flow and shown per account; "
          "they are not composite-eligible (GIPS: include from first full period, through last full period).",
          "- Nothing is interpolated. A missing statement is a missing period."]
    for note in res.notes:
        L.append(f"- {note}")
    L.append("")
    (out / "summary.md").write_text("\n".join(L))
    return out / "summary.md"
