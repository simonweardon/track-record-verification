"""Phase 3 — portfolio-level attribution.  Three levels, by what the data supports:

Level 1 (verified returns, real factor data)  BETA DECOMPOSITION
    R_p = RF + beta*(MKT-RF) + [alpha + residual]
    "How much of the return was just being in equities?"  Beta is the
    full-sample CAPM beta (in-sample by construction; Phase 4 does inference).

Level 2 (verified year-end weights, real sleeve benchmarks)  ALLOCATION vs REST
    Sleeves: US equity / international equity / cash (/ unclassified).
    allocation effect  = sum_i w_p,i * R_b,i  -  R_b          (mix vs policy)
    selection+timing   = R_p  -  sum_i w_p,i * R_b,i          (everything else)
    Weights are beginning-of-year = prior year-end positions.  Without
    transaction data the residual cannot be split further and is not.

Level 3 (needs security-level price data)  SELECTION vs TRADING
    Buy-and-hold return of the beginning holdings gives sleeve returns;
    selection = sum_i w_p,i * (R_bh,i - R_b,i); trading/timing = R_p - buy-and-hold.
    Implemented; runs only when a prices table is supplied.

TIMING (manager)  Treynor-Mazuy and Henriksson-Merton regressions.
    Client flow timing is the IRR-vs-TWR gap in Phase 2, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

SLEEVES = {"us_equity": "US", "intl_equity": "INTL", "cash": "CASH"}
SLEEVE_BENCH = {"US": "US_MKT", "INTL": "DXUS_MKT", "CASH": "DEV_RF"}
UNCLASSIFIED = "UNCLASSIFIED"     # fund / bond / other / blank asset_class


@dataclass
class Phase3Result:
    beta_decomp: pd.DataFrame
    allocation: pd.DataFrame
    policy: pd.DataFrame
    timing: pd.DataFrame
    selection: pd.DataFrame | None
    notes: list[str] = field(default_factory=list)


# ---- helpers ----------------------------------------------------------------

def ppy_of(cells) -> int:
    """Periods per year implied by the cell labels: 12 for months, 1 for years."""
    return 12 if len(cells) and isinstance(cells[0], pd.Period) else 1


def annual(ref: pd.DataFrame, col: str, cells) -> pd.Series:
    """Factor/market return per cell.  Year cells compound the 12 months; month
    cells look the month up directly."""
    out = {}
    for c in cells:
        if isinstance(c, pd.Period):
            m = ref.loc[c.start_time:c.end_time, col]
            out[c] = float(m.iloc[0]) if len(m) == 1 and not pd.isna(m.iloc[0]) else np.nan
        else:
            m = ref.loc[f"{c}-01-01":f"{c}-12-31", col]
            out[c] = float((1 + m).prod() - 1) if len(m) == 12 and m.notna().all() else np.nan
    return pd.Series(out, name=col)


def infer_policy_weights(ref: pd.DataFrame, years, window: int = 12) -> pd.DataFrame:
    """US share of DEV_MKT, inferred each year from the prior `window` months by
    least squares on DEV = w*US + (1-w)*DXUS.  Sanity: ~0.40 in 1995 -> ~0.65 in 2025."""
    d = ref[["DEV_MKT", "US_MKT", "DXUS_MKT"]].dropna()
    x, yv = d.US_MKT - d.DXUS_MKT, d.DEV_MKT - d.DXUS_MKT
    w = ((x * yv).rolling(window).sum() / (x * x).rolling(window).sum()).clip(0, 1)
    rows = {}
    for y in years:
        cutoff = (y - 1).end_time if isinstance(y, pd.Period) else pd.Timestamp(y - 1, 12, 31)
        prior = w[w.index <= cutoff]
        wu = float(prior.iloc[-1]) if len(prior) and not pd.isna(prior.iloc[-1]) else np.nan
        rows[y] = {"US": wu, "INTL": 1 - wu if not pd.isna(wu) else np.nan, "CASH": 0.0, UNCLASSIFIED: 0.0}
    return pd.DataFrame(rows).T.rename_axis("year")


def composite_sleeve_weights(periods: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Beginning-of-year composite weights by sleeve, from member accounts' prior
    year-end positions, value-weighted.  `coverage` = share of composite
    beginning assets for which positions existed."""
    el = periods[periods.eligible]
    po = positions.dropna(subset=["statement_id", "market_value"]).copy()
    po["sleeve"] = po.asset_class.map(SLEEVES).fillna(UNCLASSIFIED)
    by_stmt = po.groupby(["statement_id", "sleeve"]).market_value.sum().unstack(fill_value=0.0)
    prev_sid = {}   # (account, cell) -> statement_id of the prior period
    allp = periods.sort_values(["account_id", "period_end"])
    for acct, g in allp.groupby("account_id"):
        ids = g.statement_id.tolist(); cells = g.cell.tolist()
        for i in range(1, len(ids)):
            if cells[i] is not None:
                prev_sid[(acct, cells[i])] = ids[i - 1]
    rows = []
    for cell, g in el.groupby("cell"):
        tot = g.begin.sum()
        covered = 0.0
        sums = {s: 0.0 for s in list(SLEEVE_BENCH) + [UNCLASSIFIED]}
        for _, x in g.iterrows():
            sid = prev_sid.get((x.account_id, cell))
            if sid in by_stmt.index and by_stmt.loc[sid].sum() > 0:
                v = by_stmt.loc[sid]
                scale = x.begin / v.sum()       # positions sum ~= prior ending = this beginning
                for s in sums:
                    sums[s] += float(v.get(s, 0.0)) * scale
                covered += x.begin
        row = {s: (sums[s] / covered if covered > 0 else np.nan) for s in sums}
        row["coverage"] = covered / tot if tot > 0 else np.nan
        row["n_with_positions"] = int(sum(1 for _, x in g.iterrows()
                                          if prev_sid.get((x.account_id, cell)) in by_stmt.index))
        rows.append(pd.Series(row, name=cell))
    return pd.DataFrame(rows).rename_axis("year")


# ---- Level 1: beta decomposition --------------------------------------------

def beta_decomposition(comp: pd.DataFrame, ref: pd.DataFrame, mkt="DEV_MKT", rf="DEV_RF") -> pd.DataFrame:
    years = list(comp.index)
    m, r = annual(ref, mkt, years), annual(ref, rf, years)
    df = pd.DataFrame({"portfolio": comp.gross, "market": m, "rf": r}).dropna()
    y, x = df.portfolio - df.rf, df.market - df.rf
    fit = sm.OLS(y.values, sm.add_constant(x.values)).fit()
    a, b = fit.params
    df["beta"] = b
    df["rf_contribution"] = df.rf
    df["beta_contribution"] = b * x
    df["alpha_plus_residual"] = y - b * x
    df["market_excess"] = x
    df.attrs["alpha"] = a; df.attrs["beta"] = b; df.attrs["r2"] = fit.rsquared
    return df


# ---- Level 2: allocation vs rest ----------------------------------------------

def allocation_attribution(comp: pd.DataFrame, weights: pd.DataFrame, policy: pd.DataFrame,
                           ref: pd.DataFrame) -> pd.DataFrame:
    years = list(comp.index)
    bench = {s: annual(ref, c, years) for s, c in SLEEVE_BENCH.items()}
    rows = []
    for y in years:
        if y not in weights.index or pd.isna(weights.loc[y, "coverage"]) or weights.loc[y, "coverage"] == 0:
            rows.append(dict(year=y, available=False)); continue
        wp, wb = weights.loc[y], policy.loc[y]
        rb = {s: bench[s][y] for s in SLEEVE_BENCH}
        # unclassified holdings get the policy benchmark return (no information -> no effect)
        r_policy = float(sum(wb[s] * rb[s] for s in SLEEVE_BENCH))
        rb[UNCLASSIFIED] = r_policy
        r_alloc = float(sum(wp[s] * rb[s] for s in list(SLEEVE_BENCH) + [UNCLASSIFIED]))
        bench_actual = float(comp.loc[y, "benchmark"]) if "benchmark" in comp else np.nan
        row = dict(year=y, available=True, portfolio=comp.loc[y, "gross"],
                   benchmark_actual=bench_actual, policy_return=r_policy,
                   policy_construction_error=r_policy - bench_actual,
                   allocation_only_return=r_alloc,
                   allocation_effect=r_alloc - r_policy,
                   selection_timing_residual=comp.loc[y, "gross"] - r_alloc,
                   total_active=comp.loc[y, "gross"] - bench_actual,
                   coverage=wp["coverage"])
        for s in list(SLEEVE_BENCH) + [UNCLASSIFIED]:
            row[f"w_{s}"] = wp[s]; row[f"wb_{s}"] = wb[s]; row[f"rb_{s}"] = rb[s]
            row[f"alloc_{s}"] = (wp[s] - wb[s]) * (rb[s] - r_policy)   # Brinson-Fachler by sleeve
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")


# ---- Level 3: selection vs trading (needs prices) -------------------------------

def selection_attribution(periods: pd.DataFrame, positions: pd.DataFrame, prices: pd.DataFrame,
                          comp: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """prices: columns identifier, year, total_return (the security's calendar-year
    total return).  Buy-and-hold of the prior year-end holdings, by sleeve."""
    el = periods[periods.eligible]
    po = positions.dropna(subset=["statement_id", "market_value"]).copy()
    po["sleeve"] = po.asset_class.map(SLEEVES).fillna(UNCLASSIFIED)
    px = prices.set_index(["identifier", "year"]).total_return
    prev_sid = {}
    for acct, g in periods.sort_values(["account_id", "period_end"]).groupby("account_id"):
        ids = g.statement_id.tolist(); cells = g.cell.tolist()
        for i in range(1, len(ids)):
            if cells[i] is not None:
                prev_sid[(acct, cells[i])] = ids[i - 1]
    years = list(comp.index)
    bench = {s: annual(ref, c, years) for s, c in SLEEVE_BENCH.items()}
    rows = []
    for y, g in el.groupby("cell"):
        hold = []
        for _, x in g.iterrows():
            sid = prev_sid.get((x.account_id, y))
            if sid is None:
                continue
            h = po[po.statement_id == sid].copy()
            if h.empty:
                continue
            h["mv"] = h.market_value / h.market_value.sum() * x.begin
            hold.append(h)
        if not hold:
            rows.append(dict(year=y, available=False)); continue
        h = pd.concat(hold)
        h["ret"] = [px.get((i, y), np.nan) if s != "CASH" else bench["CASH"][y]
                    for i, s in zip(h.identifier, h.sleeve)]
        priced = h.ret.notna()
        if priced.sum() == 0:
            rows.append(dict(year=y, available=False)); continue
        tot = h.mv.sum()
        row = dict(year=y, available=True, priced_coverage=float(h.loc[priced, "mv"].sum() / tot))
        bh_total, sel_total = 0.0, 0.0
        for s in list(SLEEVE_BENCH) + [UNCLASSIFIED]:
            hs = h[(h.sleeve == s) & priced]
            w = float(h.loc[h.sleeve == s, "mv"].sum() / tot)
            if len(hs) == 0 or w == 0:
                row[f"bh_{s}"] = np.nan; row[f"sel_{s}"] = 0.0; continue
            r_bh = float((hs.mv * hs.ret).sum() / hs.mv.sum())
            rb = bench[s][y] if s in bench else np.nan
            row[f"bh_{s}"] = r_bh
            row[f"sel_{s}"] = w * (r_bh - rb) if not pd.isna(rb) else 0.0
            bh_total += w * r_bh
            sel_total += row[f"sel_{s}"]
        row["buy_and_hold_return"] = bh_total
        row["selection_effect"] = sel_total
        row["trading_timing_residual"] = float(comp.loc[y, "gross"]) - bh_total
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")


# ---- Timing regressions --------------------------------------------------------

def timing_tests(comp: pd.DataFrame, ref: pd.DataFrame, mkt="DEV_MKT", rf="DEV_RF") -> pd.DataFrame:
    years = list(comp.index)
    m, r = annual(ref, mkt, years), annual(ref, rf, years)
    df = pd.DataFrame({"p": comp.gross, "m": m, "rf": r}).dropna()
    y, x = (df.p - df.rf).values, (df.m - df.rf).values
    out = []
    X = sm.add_constant(np.column_stack([x, x ** 2]))
    f = sm.OLS(y, X).fit()
    out.append(dict(model="Treynor-Mazuy", alpha=f.params[0], beta=f.params[1], gamma=f.params[2],
                    gamma_t=f.tvalues[2], gamma_p=f.pvalues[2], n=len(y),
                    reading="gamma>0 with small p = evidence of market timing (convex payoff)"))
    X = sm.add_constant(np.column_stack([x, np.maximum(x, 0)]))
    f = sm.OLS(y, X).fit()
    out.append(dict(model="Henriksson-Merton", alpha=f.params[0], beta=f.params[1], gamma=f.params[2],
                    gamma_t=f.tvalues[2], gamma_p=f.pvalues[2], n=len(y),
                    reading="gamma>0 = higher beta in up markets than down markets"))
    return pd.DataFrame(out)


# ---- build + report ------------------------------------------------------------

def build(periods: pd.DataFrame, positions: pd.DataFrame, comp: pd.DataFrame, ref: pd.DataFrame,
          prices: pd.DataFrame | None = None) -> Phase3Result:
    notes = []
    years = list(comp.index)
    bd = beta_decomposition(comp, ref)
    policy = infer_policy_weights(ref, years)
    weights = composite_sleeve_weights(periods, positions)
    alloc = allocation_attribution(comp, weights, policy, ref)
    timing = timing_tests(comp, ref)
    sel = None
    if prices is not None and len(prices):
        sel = selection_attribution(periods, positions, prices, comp, ref)
    else:
        notes.append("Level 3 (selection vs trading) not run: no security price table supplied.")
    if (alloc.available == False).any():
        notes.append(f"Allocation effect unavailable in {int((~alloc.available).sum())} year(s): "
                     "no prior-year positions for any composite member.")
    return Phase3Result(bd, alloc, policy, timing, sel, notes)


def _pct(x, d=2):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:+.{d}f}%"


def write_phase3(res: Phase3Result, out_dir, title="Phase 3 — attribution"):
    from pathlib import Path
    from .returns import chain, annualize
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    bd, al, tm = res.beta_decomp, res.allocation, res.timing
    bd.to_csv(out / "beta_decomposition.csv"); al.to_csv(out / "allocation.csv")
    res.policy.to_csv(out / "policy_weights.csv"); tm.to_csv(out / "timing_tests.csv", index=False)
    if res.selection is not None:
        res.selection.to_csv(out / "selection.csv")

    L = [f"# {title}", "", "Portfolio-level only. Each level uses only the evidence it names.", ""]
    k = ppy_of(list(bd.index))

    # Level 1
    yrs = len(bd)
    L += ["## Level 1 — Beta decomposition (verified returns × real market data)", "",
          f"Full-sample CAPM vs DEV_MKT over {bd.index.min()}–{bd.index.max()}: "
          f"beta **{bd.attrs['beta']:.2f}**, R² {bd.attrs['r2']:.2f}, "
          f"alpha {_pct(bd.attrs['alpha'] * k)}/yr (arithmetic; inference in Phase 4).", ""]
    tot_p = bd.portfolio.mean() * k; tot_rf = bd.rf.mean() * k; tot_b = bd.beta_contribution.mean() * k; tot_a = bd.alpha_plus_residual.mean() * k
    L += ["| component | avg per year | share of return |", "|---|---|---|",
          f"| Risk-free rate | {_pct(tot_rf)} | {tot_rf / tot_p:.0%} |",
          f"| Market exposure (beta × market excess) | {_pct(tot_b)} | {tot_b / tot_p:.0%} |",
          f"| Alpha + residual (what's left) | {_pct(tot_a)} | {tot_a / tot_p:.0%} |",
          f"| **Portfolio** | **{_pct(tot_p)}** | 100% |", "",
          "Arithmetic annual averages. In plain terms: of the return, this is how much came from "
          "simply holding equities versus everything the manager did.", ""]

    # Level 2
    av = al[al.available == True] if "available" in al else al
    L += ["## Level 2 — Allocation effect (verified year-end weights × sleeve benchmarks)", ""]
    if len(av):
        L += [f"Years with weights: {len(av)} of {len(al)}. Average coverage of composite assets "
              f"by positions: {av.coverage.mean():.0%}.", "",
              "| avg per year | allocation effect | selection + timing residual | policy construction error | = total active vs benchmark |",
              "|---|---|---|---|---|",
              f"| | {_pct(av.allocation_effect.mean())} | {_pct(av.selection_timing_residual.mean())} | "
              f"{_pct(av.policy_construction_error.mean())} | {_pct(av.total_active.mean())} |", "",
              f"The policy benchmark's US / ex-US split is *inferred* from returns (no free source publishes "
              f"it); its construction error vs the actual benchmark averages {av.policy_construction_error.abs().mean():.2%} "
              f"in absolute terms (max {av.policy_construction_error.abs().max():.2%}). With a named index "
              "whose weights are published this line vanishes.", "",
              "| avg weight | US | Intl | Cash | Unclassified |", "|---|---|---|---|---|",
              f"| portfolio | {av.w_US.mean():.0%} | {av.w_INTL.mean():.0%} | {av.w_CASH.mean():.0%} | {av.w_UNCLASSIFIED.mean():.0%} |",
              f"| policy (inferred) | {av.wb_US.mean():.0%} | {av.wb_INTL.mean():.0%} | 0% | 0% |",
              f"| allocation effect | {_pct(av.alloc_US.mean())} | {_pct(av.alloc_INTL.mean())} | "
              f"{_pct(av.alloc_CASH.mean())} | {_pct(av.alloc_UNCLASSIFIED.mean())} |", "",
              "Allocation effect = what the portfolio's US/international/cash mix would have earned if "
              "each sleeve were indexed, minus the policy benchmark. The residual is stock selection, "
              "intra-year trading, and flow timing together — annual holdings cannot separate them.", "",
              "| year | portfolio | policy | alloc-only | allocation | residual | w US | w Intl | w Cash | coverage |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for y, r in av.iterrows():
            L.append(f"| {y} | {_pct(r.portfolio)} | {_pct(r.policy_return)} | {_pct(r.allocation_only_return)} | "
                     f"{_pct(r.allocation_effect)} | {_pct(r.selection_timing_residual)} | {r.w_US:.0%} | "
                     f"{r.w_INTL:.0%} | {r.w_CASH:.0%} | {r.coverage:.0%} |")
    else:
        L.append("No year had prior-year positions for composite members; allocation effect not computable.")
    L.append("")

    # Level 3
    L += ["## Level 3 — Selection vs trading (needs security price history)", ""]
    if res.selection is not None and (res.selection.get("available", pd.Series(dtype=bool)) == True).any():
        s = res.selection[res.selection.available == True]
        L += [f"Years priced: {len(s)}; average priced coverage {s.priced_coverage.mean():.0%}.",
              f"- Selection effect (buy-and-hold of beginning holdings vs sleeve benchmarks): **{_pct(s.selection_effect.mean())}/yr**",
              f"- Trading / timing residual (actual minus buy-and-hold): **{_pct(s.trading_timing_residual.mean())}/yr**"]
    else:
        L += ["Not run. Requires a `prices` table (identifier, year, total_return) for every holding "
              "including delisted ones — a CRSP-quality source, not a free ticker feed, or the result "
              "inherits the price database's own survivorship bias."]
    L.append("")

    # Timing
    L += ["## Manager timing tests", "", "| model | alpha/yr | beta | gamma | t(gamma) | p | n | reading |", "|---|---|---|---|---|---|---|---|"]
    for _, t in tm.iterrows():
        L.append(f"| {t.model} | {_pct(t.alpha * k)} | {t.beta:.2f} | {t.gamma:+.3f} | {t.gamma_t:+.2f} | {t.gamma_p:.2f} | {int(t.n)} | {t.reading} |")
    L += ["", "With ~30 annual observations these tests have very low power; a p-value above 0.10 "
          "means 'no evidence either way', not 'no timing'.", ""]
    for n in res.notes:
        L.append(f"- {n}")
    (out / "summary.md").write_text("\n".join(L))
    return out / "summary.md"
