"""Phase 2b — composites.  Rules are documented in COMPOSITE_RULES.md; the code
here is the executable form of that document.  If they ever disagree, the
document is wrong and must be fixed to match, or vice versa — never silently.

Composite cell return (calendar year for annual data) uses the GIPS-accepted
*aggregate* method: treat all member accounts as one portfolio and compute
Modified Dietz on pooled beginning values, flows and ending values.  The
asset-weighted (by beginning value) and equal-weighted returns are reported
alongside as cross-checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .returns import (FeeSchedule, model_net, chain, annualize, contiguous_segments,
                      twr_summary, segment_cashflows, xirr)
from .reference import benchmark_series, compound_to_periods


@dataclass
class CompositeConfig:
    grid: str = "A"                       # "A" annual cells; "M" once monthly data exists
    strategy: str | None = None           # None -> one composite of every discretionary account
    include_flagged: bool = False         # flagged statements are known-wrong; excluded by default
    include_non_discretionary: bool = False
    min_accounts_for_dispersion: int = 6  # GIPS: internal dispersion when >=6 for the full period
    fee: FeeSchedule = field(default_factory=FeeSchedule)
    default_benchmark: str = "DEV_MKT"    # used when accounts.csv has no benchmark
    secondary_benchmark: str = "US_MKT"   # always reported alongside


@dataclass
class Phase2Result:
    periods: pd.DataFrame        # per statement-period, with eligibility + exclusion reason
    composite: pd.DataFrame      # per cell
    accounts: pd.DataFrame       # per account summary
    irr: pd.DataFrame            # per account + pooled
    config: CompositeConfig
    notes: list[str]


def _default_accounts(period_rows: pd.DataFrame) -> pd.DataFrame:
    ids = sorted(period_rows.account_id.unique())
    return pd.DataFrame(dict(account_id=ids, label=ids, owner_type="client", discretionary="Y",
                             strategy="default", benchmark=pd.NA, notes="no accounts.csv; defaults"))


def _cell(row, grid: str):
    if grid == "A":
        ok = (row.period_start.month == 1 and row.period_start.day == 1 and
              row.period_end.month == 12 and row.period_end.day == 31 and
              row.period_start.year == row.period_end.year)
        return row.period_end.year if ok else None
    if grid == "M":
        ok = (row.period_start.day == 1 and row.period_end == row.period_start + pd.offsets.MonthEnd(0))
        return row.period_end.to_period("M") if ok else None
    raise ValueError(grid)


def classify(pr: pd.DataFrame, accounts: pd.DataFrame, cfg: CompositeConfig) -> pd.DataFrame:
    """Add cell / eligible / exclusion_reason to the period rows."""
    pr = pr.copy()
    meta = accounts.set_index("account_id")
    cells, elig, why = [], [], []
    for _, x in pr.iterrows():
        c = _cell(x, cfg.grid)
        m = meta.loc[x.account_id] if x.account_id in meta.index else None
        reason = ""
        if m is None:
            reason = "not_in_accounts_csv"
        elif str(m.discretionary).upper() != "Y" and not cfg.include_non_discretionary:
            reason = "non_discretionary"
        elif cfg.strategy and str(m.strategy) != cfg.strategy:
            reason = "other_strategy"
        elif pd.isna(x.r):
            reason = "no_return"
        elif not x.full_period:
            reason = "partial_period"
        elif c is None:
            reason = "not_grid_aligned"
        elif x.evidence == "flagged" and not cfg.include_flagged:
            reason = "flagged"
        cells.append(c); elig.append(reason == ""); why.append(reason)
    pr["cell"] = cells
    pr["eligible"] = elig
    pr["exclusion_reason"] = why
    return pr


def _aggregate(g: pd.DataFrame) -> float:
    denom = g.begin.sum() + g.weighted_flows.sum()
    return float((g.end.sum() - g.begin.sum() - g.net_flows.sum()) / denom) if denom > 0 else np.nan


def _asset_weighted(g: pd.DataFrame) -> float:
    return float((g.begin * g.r).sum() / g.begin.sum()) if g.begin.sum() > 0 else np.nan


def _dispersion(g: pd.DataFrame, n_min: int):
    if len(g) < n_min:
        return np.nan, np.nan, np.nan
    w = g.begin / g.begin.sum()
    mu = float((w * g.r).sum())
    sd = float(np.sqrt((w * (g.r - mu) ** 2).sum()))
    return sd, float(g.r.max()), float(g.r.min())


def _survivors(statements: pd.DataFrame) -> set[str]:
    last_end = statements.period_end.max()
    out = set()
    for acct, g in statements.groupby("account_id"):
        tail = g.sort_values("period_end").iloc[-1]
        if tail.period_end == last_end and tail.ending_value > 0:
            out.add(acct)
    return out


def _cell_bounds(cell, grid):
    if grid == "A":
        return pd.Timestamp(cell, 1, 1), pd.Timestamp(cell, 12, 31)
    return cell.start_time.normalize(), cell.end_time.normalize()


def build(statements: pd.DataFrame, flows: pd.DataFrame, period_rows: pd.DataFrame,
          accounts: pd.DataFrame | None, ref: pd.DataFrame | None,
          cfg: CompositeConfig = CompositeConfig()) -> Phase2Result:
    notes: list[str] = []
    if accounts is None:
        accounts = _default_accounts(period_rows)
        notes.append("No accounts.csv: every account treated as discretionary client money "
                     f"benchmarked to {cfg.default_benchmark}.")
    accounts = accounts.copy()
    accounts["benchmark"] = accounts["benchmark"].fillna(cfg.default_benchmark)
    meta = accounts.set_index("account_id")
    pr = classify(period_rows, accounts, cfg)
    survivors = _survivors(statements)
    pr["survivor"] = pr.account_id.isin(survivors)
    pr["owner_type"] = pr.account_id.map(meta.owner_type).fillna("client")

    # ---- benchmark return per period row -------------------------------------
    pr["benchmark"] = pr.account_id.map(meta.benchmark)
    pr["bench_r"] = np.nan
    pr["bench2_r"] = np.nan
    if ref is not None:
        cache = {}
        for spec in pr.benchmark.dropna().unique():
            try:
                cache[spec] = benchmark_series(ref, spec)
            except KeyError:
                notes.append(f"benchmark '{spec}' not in reference data; left blank")
        for spec, s in cache.items():
            m = pr.benchmark == spec
            pr.loc[m, "bench_r"] = compound_to_periods(s, pr.loc[m, ["period_start", "period_end"]]).values
        try:
            s2 = benchmark_series(ref, cfg.secondary_benchmark)
            pr["bench2_r"] = compound_to_periods(s2, pr[["period_start", "period_end"]]).values
        except KeyError:
            notes.append(f"secondary benchmark '{cfg.secondary_benchmark}' not in reference data")
    else:
        notes.append("No reference data loaded: benchmark columns blank.")

    # ---- composite per cell ---------------------------------------------------
    el = pr[pr.eligible]
    rows = []
    for cell, g in el.groupby("cell", sort=True):
        sd, hi, lo = _dispersion(g, cfg.min_accounts_for_dispersion)
        surv, cli, pri = g[g.survivor], g[g.owner_type != "principal"], g[g.owner_type == "principal"]
        bw = g.begin / g.begin.sum()
        allc = pr[pr.cell == cell]
        rows.append(dict(
            cell=cell, n=len(g), begin_assets=float(g.begin.sum()), end_assets=float(g.end.sum()),
            net_flows=float(g.net_flows.sum()),
            gross=_aggregate(g), asset_weighted=_asset_weighted(g), equal_weighted=float(g.r.mean()),
            dispersion_sd=sd, high=hi, low=lo,
            n_survivors=len(surv), survivors_only=_aggregate(surv) if len(surv) else np.nan,
            n_clients=len(cli), clients_only=_aggregate(cli) if len(cli) else np.nan,
            n_principal=len(pri), principal_only=_aggregate(pri) if len(pri) else np.nan,
            benchmark=float((bw * g.bench_r).sum()) if g.bench_r.notna().all() else np.nan,
            benchmark_us=float((bw * g.bench2_r).sum()) if g.bench2_r.notna().all() else np.nan,
            n_verified=int((g.evidence == "verified").sum()),
            n_unverified=int((g.evidence == "unverified").sum()),
            n_excluded_flagged=int((allc.exclusion_reason == "flagged").sum()),
            n_excluded_partial=int((allc.exclusion_reason == "partial_period").sum()),
            n_excluded_other=int(allc.exclusion_reason.isin(
                ["non_discretionary", "other_strategy", "no_return", "not_grid_aligned"]).sum()),
        ))
    comp = pd.DataFrame(rows).set_index("cell") if rows else pd.DataFrame()
    if len(comp):
        yrs = pd.Series([((b[1] - b[0]).days + 1) / 365.25
                         for b in (_cell_bounds(c, cfg.grid) for c in comp.index)], index=comp.index)
        comp["model_net"] = model_net(comp.gross, yrs, cfg.fee)
        comp["excess"] = comp.gross - comp.benchmark
        comp["excess_us"] = comp.gross - comp.benchmark_us
        comp["years"] = yrs

    # ---- per-account summary + IRR ---------------------------------------------
    acc_rows, irr_rows = [], []
    pooled = []
    for acct, g in pr.groupby("account_id"):
        m = meta.loc[acct] if acct in meta.index else None
        segs = contiguous_segments(g, break_on_flagged=not cfg.include_flagged)
        longest = max(segs, key=len) if segs else None
        tw = twr_summary(longest) if longest is not None else {}
        bench_cum = chain(longest.bench_r) if longest is not None and longest.bench_r.notna().all() else np.nan
        cf = segment_cashflows(longest, flows)
        irr = xirr(cf.date, cf.amount) if cf is not None else np.nan
        whole = (longest is not None and len(longest) == len(g))
        irr_note = "" if whole else ("over TWR span only" if cf is not None else "not computable")
        last = g.sort_values("period_end").iloc[-1]
        first = g.sort_values("period_end").iloc[0]
        first_flow = flows.loc[flows.account_id == acct, "date"].min()
        inception = first.eff_start
        if (pd.isna(first.begin) or first.begin == 0) and not pd.isna(first_flow):
            inception = min(first_flow, first.eff_start) if first.begin == 0 else first_flow
        naive = np.nan
        if longest is not None:
            seg2 = longest[longest.begin > 0]
            if len(seg2) and seg2.iloc[-1].end > 0:
                naive = annualize(seg2.iloc[-1].end / seg2.iloc[0].begin - 1, float(seg2.days.sum()) / 365.25)
        acc_rows.append(dict(
            account_id=acct,
            label=m.label if m is not None else acct,
            owner_type=m.owner_type if m is not None else "client",
            discretionary=m.discretionary if m is not None else "Y",
            benchmark=m.benchmark if m is not None else cfg.default_benchmark,
            inception=inception.date(), last_period_end=last.eff_end.date(),
            status=("open" if acct in survivors else "closed/terminated"),
            periods=len(g), composite_periods=int(g.eligible.sum()),
            verified=int((g.evidence == "verified").sum()),
            unverified=int((g.evidence == "unverified").sum()),
            flagged=int((g.evidence == "flagged").sum()),
            segments=len(segs),
            twr_span=(f"{tw['start'].date()}..{tw['end'].date()}" if tw else ""),
            twr_years=round(tw["years"], 2) if tw else np.nan,
            twr_annualized=tw.get("annualized", np.nan),
            twr_cumulative=tw.get("cumulative", np.nan),
            bench_annualized=annualize(bench_cum, tw["years"]) if tw else np.nan,
            naive_cagr_ignoring_flows=naive,
            arithmetic_mean=float(longest.r.mean()) if longest is not None else np.nan,
            irr=irr,
            irr_note=irr_note,
        ))
        if cf is not None and m is not None and str(m.discretionary).upper() == "Y":
            pooled.append(cf)
        irr_rows.append(dict(account_id=acct, irr=irr, gap=cf is None))
    acc = pd.DataFrame(acc_rows)
    if pooled:
        allcf = pd.concat(pooled)
        irr_rows.append(dict(account_id="POOLED_DISCRETIONARY", irr=xirr(allcf.date, allcf.amount),
                             gap=False))
    irr_df = pd.DataFrame(irr_rows)
    return Phase2Result(pr, comp, acc, irr_df, cfg, notes)


# ---- statistics on a cell series -------------------------------------------

def _consecutive(a, b) -> bool:
    """Are two cell labels adjacent?  Years are ints; months are pd.Period."""
    return (b - a == 1) if isinstance(a, (int, np.integer)) else (b - a).n == 1


def longest_run(r: pd.Series) -> pd.Series:
    """Longest run of adjacent cells with a value.  Statistics are never chained
    across a hole — a missing cell ends the run."""
    best, cur = [], []
    prev = None
    for k, v in r.items():
        if pd.isna(v) or (prev is not None and not _consecutive(prev, k)):
            if len(cur) > len(best):
                best = cur
            cur = []
        if not pd.isna(v):
            cur.append(k)
        prev = k
    if len(cur) > len(best):
        best = cur
    return r.loc[best] if best else r.iloc[0:0]


def series_stats(r: pd.Series, years: pd.Series) -> dict:
    """Summary over the longest contiguous run of `r`.  `truncated` is True when
    that run is shorter than the available cells (a hole was found)."""
    run = longest_run(r)
    if run.empty:
        return {}
    y = years.reindex(run.index)
    cum = chain(run)
    return dict(first=run.index.min(), last=run.index.max(), periods=len(run), years=float(y.sum()),
                truncated=bool(len(run) < r.notna().sum() or len(run) < len(r)),
                cumulative=cum, annualized=annualize(cum, float(y.sum())),
                growth_of_1=1 + cum, best=float(run.max()), worst=float(run.min()),
                negative_periods=int((run < 0).sum()), arithmetic_mean=float(run.mean()),
                stdev=float(run.std(ddof=1)) if len(run) > 1 else np.nan)


def trailing(comp: pd.DataFrame, col: str, n: int) -> float:
    """Annualized return over the last n cells (NaN if fewer available or any missing)."""
    if len(comp) < n:
        return np.nan
    tail = comp.iloc[-n:]
    if tail[col].isna().any():
        return np.nan
    return annualize(chain(tail[col]), float(tail.years.sum()))
