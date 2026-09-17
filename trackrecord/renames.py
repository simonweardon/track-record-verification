"""Recover the companies that only changed their name.

A position that cannot be matched to a priced security is dropped from every portfolio on the
site, and in the early years that is a third of the disclosed money.  Most of it is companies
that were bought and no longer have a price history anywhere, which nothing can recover.  Some
of it, though, is companies that are still trading under a different name and a different
identifier, and those can be recovered — but only if the recovery is *derived* rather than
asserted, because a wrong match silently prices one company's positions with another's returns.

The evidence is in the filings themselves.  When a company is re-identified, the managers
holding the old line all appear in the next quarter holding one particular new line, in the
same number of shares.  When a manager has simply sold one company and bought another, the
holder base does not move together and the share counts are unrelated.  So a match is accepted
only when all of these hold:

    the old line stops being filed, and never returns
    at least 80% of the managers holding it at the end hold the same new line the next quarter,
      and at least five of them do
    the new line's ticker is priced over the whole span of the old one
    the price path the filings imply for the old line — its reported value over its reported
      share count, taken across managers — moves quarter for quarter with the successor's own
      price, over at least eight quarters

The last condition is what makes the recovery safe, and it is the one that matters: what the
pipeline needs from a ticker is its return series, so the test is whether the successor's
returns *are* the old line's returns.  The levels never match, because the filings quote the
price on the day and the price history is adjusted for everything paid since; but if the two are
the same listing, the two paths move together and the gap between them is a constant.  A wrong
ticker fails this immediately, however plausible the two names look.

    python -m trackrecord renames        # rewrite data/reference/compact/renames.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "reference" / "compact" / "renames.csv"

MIN_HOLDERS = 5          # fewer than this and a coincidence is too easy
MIN_MOVED = 0.80         # share of the old line's holders that must appear on the new one
PATH_TOL = 0.02          # how far the two price paths may drift apart in a quarter, at the 90th percentile
MIN_QUARTERS = 8         # quarters of agreement needed before a match is believed
MIN_SIZE = 0.0005        # ignore lines too small to change anything


def implied_price(h: pd.DataFrame, cusip: str) -> pd.Series:
    """What the filings say a share of this line was worth: reported value over reported shares.

    Taken as the median across the managers filing it, so that one manager reporting a share
    count in the wrong unit — which happens, and is documented elsewhere on the site — moves
    nothing.  Reported values changed unit at the end of 2022 (thousands of dollars to dollars),
    so the series is put back on one basis before it is compared with anything."""
    g = h[h.cusip == cusip]
    p = (g.value / g.shares).replace([np.inf, -np.inf], np.nan).groupby(g.period).median().dropna()
    if p.empty:
        return p
    p.index = pd.to_datetime(p.index) + pd.offsets.MonthEnd(0)
    return p * np.where(p.index < pd.Timestamp("2022-12-31"), 1000.0, 1.0)


def path_agreement(implied: pd.Series, px: pd.Series) -> tuple[float, int]:
    """How far two price paths drift apart in a quarter, and over how many quarters.

    Levels are not comparable — the filings quote the price on the day, the history is adjusted
    for every dividend and split since — so the comparison is between the changes.  For the same
    listing the difference is the dividend paid in the quarter, a fraction of a percent."""
    # differences have to be taken on the quarter grid, not after dropping the quarters a line
    # was not filed: otherwise a gap compares two quarters of one path with one of the other
    grid = pd.date_range(implied.index.min(), implied.index.max(), freq="QE")
    a = np.log(implied.reindex(grid).where(lambda x: x > 0)).diff()
    b = np.log(px.reindex(grid).where(lambda x: x > 0)).diff()
    d = (a - b).dropna()
    if len(d) < 3:
        return float("inf"), int(len(d))
    return float(np.abs(d).quantile(0.9)), int(len(d))


def load() -> dict[str, str]:
    """The accepted renames as old cusip -> ticker, or {} if they have not been derived."""
    if not OUT.exists():
        return {}
    d = pd.read_csv(OUT, dtype=str)
    return dict(zip(d.cusip, d.ticker))


def fingerprint(log=print) -> pd.DataFrame:
    """Search every priced company for the one whose returns are the old line's returns.

    The holder migration above can only find a company whose owners all moved in one quarter.
    This finds the rest: it takes the price path each unmatched line implies and compares it with
    every ticker in the price history, quarter by quarter.  A listing that was merely renamed
    matches one ticker almost exactly and nothing else; a company that was bought matches
    nothing, which is the correct answer and the reason most of the gap cannot be closed."""
    from .compact import EDGAR, holdings_frame
    from .signals13f import load_prices

    h = holdings_frame()
    h = h[(h.putcall == "") & (h.value > 0) & (h.shares > 0)]
    cmap = json.loads((EDGAR / "cusip_map.json").read_text())   # the cache alone: what is still unmatched
    h["ticker"] = h.cusip.map(cmap)
    prices = load_prices()
    h["priced"] = h.ticker.notna() & h.ticker.isin(prices.columns)
    names = h.groupby("cusip")["name"].first()
    h["yr"] = pd.to_datetime(h.period).dt.year
    size = (h.groupby(["cusip", "yr"]).value.sum() / h.groupby("yr").value.sum()).groupby("cusip").sum()

    quarters = pd.DatetimeIndex(sorted(pd.to_datetime(h.period.unique()) + pd.offsets.MonthEnd(0)))
    Q = np.log(prices.reindex(quarters).where(lambda x: x > 0))
    dQ = Q.diff()                                              # every ticker's quarterly log return
    ok_cols = dQ.notna().sum() >= MIN_QUARTERS
    dQ = dQ.loc[:, ok_cols]
    tickers = np.array(dQ.columns)
    M = dQ.values                                              # quarters x tickers

    cands = [c for c in h.loc[~h.priced, "cusip"].unique() if size.get(c, 0) >= MIN_SIZE]
    log(f"  fingerprinting {len(cands)} unmatched lines against {len(tickers)} priced companies")
    rows = []
    for c in cands:
        ip = implied_price(h, c).reindex(quarters)
        d = np.log(ip.where(ip > 0)).diff().values
        have = ~np.isnan(d)
        if have.sum() < MIN_QUARTERS:
            continue
        D = np.abs(M[have] - d[have, None])                    # quarters x tickers
        n = (~np.isnan(D)).sum(axis=0)
        D = np.where(np.isnan(D), np.inf, D)
        gaps = np.full(len(tickers), np.inf)
        good = n >= MIN_QUARTERS
        if not good.any():
            continue
        gaps[good] = np.nanquantile(np.where(np.isinf(D[:, good]), np.nan, D[:, good]), 0.9, axis=0)
        order = np.argsort(gaps)
        best, second = order[0], order[1] if len(order) > 1 else order[0]
        rows.append(dict(cusip=c, ticker=str(tickers[best]), was=str(names[c]).strip(), now="",
                         last_filed=str(h.loc[h.cusip == c, "period"].max()), picked_up="", holders=0, moved=0,
                         share_ratio=np.nan, path_gap=round(float(gaps[best]), 5), quarters=int(n[best]),
                         runner_up=str(tickers[second]), runner_up_gap=round(float(gaps[second]), 5),
                         value_share=round(float(size.get(c, 0)), 5), how="price path",
                         accepted=bool(gaps[best] <= PATH_TOL and n[best] >= MIN_QUARTERS
                                       and gaps[second] > max(3 * gaps[best], 0.05))))
    return pd.DataFrame(rows)


def detect(log=print) -> pd.DataFrame:
    from .compact import EDGAR, holdings_frame
    from .signals13f import load_prices

    h = holdings_frame()
    h = h[(h.putcall == "") & (h.value > 0) & (h.shares > 0)]
    cmap = json.loads((EDGAR / "cusip_map.json").read_text())   # the cache alone: what is still unmatched
    h["ticker"] = h.cusip.map(cmap)
    prices = load_prices()
    h["priced"] = h.ticker.notna() & h.ticker.isin(prices.columns)
    periods = sorted(h.period.unique())
    nxt = {p: periods[i + 1] for i, p in enumerate(periods[:-1])}
    first, last = h.groupby("cusip").period.min(), h.groupby("cusip").period.max()
    names = h.groupby("cusip")["name"].first()
    h["yr"] = pd.to_datetime(h.period).dt.year
    size = (h.groupby(["cusip", "yr"]).value.sum() / h.groupby("yr").value.sum()).groupby("cusip").sum()
    shares = h.groupby(["cusip", "period", "slug"]).shares.sum()
    hold = {k: set(v) for k, v in h.groupby(["cusip", "period"]).slug.apply(set).items()}
    starts: dict[str, list[str]] = {}
    for c, p in first.items():
        if h.loc[h.cusip == c, "priced"].any():
            starts.setdefault(p, []).append(c)

    rows = []
    for c in [x for x in h.loc[~h.priced, "cusip"].unique() if x in last.index]:
        T = last[c]
        if T == periods[-1] or size.get(c, 0) < MIN_SIZE:
            continue                                    # still held, or too small to matter
        A = hold.get((c, T), set())
        if len(A) < MIN_HOLDERS:
            continue
        for T2 in (T, nxt.get(T)):
            if T2 is None:
                continue
            for c2 in starts.get(T2, []):
                if c2 == c:
                    continue
                common = sorted(A & hold.get((c2, T2), set()))
                if len(common) < MIN_HOLDERS or len(common) / len(A) < MIN_MOVED:
                    continue
                t = cmap[c2]
                px = prices[t].dropna()
                if px.empty or pd.isna(px.index.min()) or pd.isna(px.index.max()):
                    continue
                if not (px.index.min().strftime("%Y-%m") <= first[c][:7] and px.index.max().strftime("%Y-%m") >= T[:7]):
                    continue
                r = np.array([shares[(c2, T2, s)] / shares[(c, T, s)] for s in common])
                dev, nq = path_agreement(implied_price(h, c), prices[t])
                rows.append(dict(cusip=c, ticker=t, was=str(names[c]).strip(), now=str(names[c2]).strip(),
                                 last_filed=T, picked_up=T2, holders=len(A), moved=len(common),
                                 share_ratio=round(float(np.median(r)), 4),
                                 path_gap=round(dev, 5) if np.isfinite(dev) else None, quarters=nq,
                                 value_share=round(float(size.get(c, 0)), 5),
                                 accepted=bool(dev <= PATH_TOL and nq >= MIN_QUARTERS)))
    D = pd.DataFrame(rows)
    if D.empty:
        return D
    # one successor per old line: the best share-ratio match
    D = D.sort_values(["cusip", "path_gap"]).drop_duplicates("cusip")
    return D.sort_values("value_share", ascending=False).reset_index(drop=True)


def build(out: Path = OUT, log=print) -> dict:
    """Both searches, the price fingerprint first; the holder migration corroborates it."""
    F = fingerprint(log)
    M = detect(log)
    if not M.empty:
        mig = M.set_index("cusip")[["now", "picked_up", "holders", "moved", "share_ratio"]]
        F = F.drop(columns=["now", "picked_up", "holders", "moved", "share_ratio"]).join(mig, on="cusip")
        # a line the migration found and the fingerprint did not is kept only if the fingerprint agrees
        extra = M[M.accepted & ~M.cusip.isin(F.cusip)]
        if len(extra):
            F = pd.concat([F, extra.assign(how="holder migration", runner_up="", runner_up_gap=np.nan)], ignore_index=True)
        F["corroborated"] = F.moved.notna() & (F.moved > 0)
    else:
        F["corroborated"] = False
    D = F.sort_values("value_share", ascending=False).reset_index(drop=True)
    acc = D[D.accepted] if not D.empty else D
    out.parent.mkdir(parents=True, exist_ok=True)
    acc.to_csv(out, index=False)
    (out.parent / "renames_rejected.csv").write_text(D[~D.accepted].to_csv(index=False) if not D.empty else "")
    for _, r in acc.head(12).iterrows():
        corr = (f", and {int(r.moved)} of {int(r.holders)} managers moved across in one quarter" if r.corroborated else "")
        log(f"  {r.was} → {r.ticker}: paths agree to {r.path_gap:.2%} a quarter over {int(r.quarters)} quarters "
            f"(next best {r.runner_up} at {r.runner_up_gap:.1%}){corr}")
    if len(acc) > 12:
        log(f"  … and {len(acc) - 12} more, all in {out.name}")
    info = dict(accepted=int(len(acc)), rejected=int((~D.accepted).sum()) if not D.empty else 0,
                corroborated=int(acc.corroborated.sum()) if len(acc) else 0,
                value_share=float(acc.value_share.sum()) if len(acc) else 0.0,
                min_holders=MIN_HOLDERS, min_moved=MIN_MOVED, path_tol=PATH_TOL, min_quarters=MIN_QUARTERS)
    log(f"{info['accepted']} renames recovered, {info['rejected']} rejected as mergers or coincidences")
    return info
