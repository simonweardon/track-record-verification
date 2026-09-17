"""Alpha model lab: which stock-level signals predict next month's return, and does a
gradient-boosted model beat a linear composite of them?

Universe: the construction universe — US names held by ≥ 5 of the managers in the system,
priced at $1 or more — refreshed at each 13F formation date and used for the following
three month-ends.  Every month-end t, for every stock, the signals below are built from
what was public at t (prices through t; a filing only after its `filed` date), z-scored
across the universe (winsorised at ±3), and signed so that higher = "better" in the
published anomaly:

    momentum      12-1 month price return                          (Jegadeesh–Titman)
    reversal      minus the last month's return                     (short-term reversal)
    low_vol       minus the 12-month volatility of monthly returns  (low-volatility anomaly)
    size          minus log market cap                              (small-firm effect)
    value         book equity / market cap                          (B/M, Fama–French)
    profitability trailing-year net income / total assets           (ROA, quality)
    cash_flow     trailing-year operating cash flow / total assets  (quality)
    earnings_yield trailing-year net income / market cap

Tests, all out of sample by construction:
    IC       Spearman correlation between the signal at t and the return over t → t+1
    deciles  equal-weight decile portfolios each month; D10 − D1 spread, t-stat
    linear   composite = mean of the available z-scores (equal weights, no fitting)
    xgboost  gradient boosting on the same z-scores, trained walk-forward on an expanding
             window (first 36 months held out entirely; retrained every 12 months, never
             with data from the month being predicted)

The point is the machinery and the honesty of the comparison, not a claim that these
signals work: the sample is 2014–2026, one universe, survivor-biased prices.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .compact import COMPACT
from .construct import MIN_HOLDERS, load_sectors
from .fundamentals import load as load_fundamentals
from .signals13f import clean_returns, load_books, load_prices

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "research" / "alpha-lab"
SIGNALS = {
    "momentum": "Momentum (12-1)", "reversal": "Short-term reversal", "low_vol": "Low volatility", "size": "Size (small)",
    "value": "Value (book / market)", "profitability": "Profitability (ROA)", "cash_flow": "Cash flow / assets", "earnings_yield": "Earnings yield",
}
MIN_TRAIN = 36           # months held out before the first xgboost prediction
RETRAIN = 12             # months between refits
MIN_NAMES = 50           # a month needs this many names with a signal to count


# ---------------------------------------------------------------- point-in-time fundamentals

def pit_table(fund: pd.DataFrame, fact: str, kind: str) -> pd.DataFrame:
    """Long table (ticker, filed, end, value) of one fact: 'instant' facts (balance sheet, shares)
    or 'annual' duration facts (flows), so a value can be looked up as of any date."""
    f = fund[fund.fact == fact].copy()
    if kind == "annual":
        f = f[f.start.notna()]
        dur = (pd.to_datetime(f.end) - pd.to_datetime(f.start)).dt.days
        f = f[(dur >= 330) & (dur <= 400)]
    else:
        f = f[f.start.isna()]
    f = f.dropna(subset=["value", "filed", "end"]).sort_values(["ticker", "filed", "end"])
    return f[["ticker", "filed", "end", "value", "form"]]


SPLIT_RATIOS = [1.5, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 50]
DOMESTIC = ("10-K", "10-Q", "10-K/A", "10-Q/A")     # foreign filers (20-F/40-F) report ordinary shares; the price is per ADR — not comparable


def clean_shares(tbl: pd.DataFrame, tol: float = 0.04) -> pd.DataFrame:
    """Shares outstanding as filed, made usable with split-adjusted prices.

    Two problems in raw XBRL 'shares outstanding': (1) unit errors — a filing 100× or 1000× its
    neighbours that reverts next filing; (2) splits — the series jumps by 2, 4, 10… and stays
    there, while Yahoo prices are already split-adjusted to today's basis.  Fix: drop one-off
    outliers (≥ 5× against both neighbours), then walk backwards and multiply everything before
    a persistent jump matching a known split ratio (±4%) so the whole series is on today's basis."""
    out = []
    tbl = tbl[(tbl.value > 0) & (tbl.form.isin(DOMESTIC) if "form" in tbl else True)]
    for t, g in tbl.sort_values(["ticker", "end", "filed"]).groupby("ticker"):
        g = g.drop_duplicates("end", keep="last").reset_index(drop=True)
        v = g.value.values.astype(float)
        if len(v) >= 3:
            # unit errors: ≥ 5× off the median of the 8 surrounding filings, in a run of at most 3 filings
            lv = np.log(np.where(v > 0, v, np.nan))
            bad = np.zeros(len(v), bool)
            for i in range(len(v)):
                nb = np.concatenate([lv[max(0, i - 4):i], lv[i + 1:i + 5]]); nb = nb[~np.isnan(nb)]
                if len(nb) >= 2 and not np.isnan(lv[i]) and abs(lv[i] - np.median(nb)) >= np.log(5):
                    bad[i] = True
            i = 0
            while i < len(v):
                if bad[i]:
                    j = i
                    while j < len(v) and bad[j]: j += 1
                    if j - i > 3: bad[i:j] = False           # a long run is a level shift (split or issuance), not an error
                    i = j
                else:
                    i += 1
            g = g[~bad].reset_index(drop=True); v = g.value.values.astype(float)
        factor = 1.0; adj = v.copy()
        for i in range(len(v) - 1, 0, -1):
            r = v[i] / v[i - 1] if v[i - 1] > 0 else 1.0
            for k in SPLIT_RATIOS:
                if abs(r / k - 1) <= tol:
                    factor *= k; break
                if abs(r * k - 1) <= tol:
                    factor /= k; break
            adj[i - 1] = v[i - 1] * factor
        g = g.copy(); g["value"] = adj
        med = np.median(adj)
        g = g[(adj <= 20 * med) & (adj >= med / 20)]          # anything still 20× off the series median is not shares outstanding
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else tbl


def as_of(tbl: pd.DataFrame, tickers: pd.Index, t: pd.Timestamp, max_age_days: int = 456) -> pd.Series:
    """Latest value filed on or before t, with a period end no older than max_age_days."""
    f = tbl[(tbl.filed <= t) & (tbl.end >= t - pd.Timedelta(days=max_age_days))]
    if f.empty:
        return pd.Series(np.nan, index=tickers)
    last = f.groupby("ticker").tail(1).set_index("ticker").value
    return last.reindex(tickers)


def load_close(prices: pd.DataFrame) -> pd.DataFrame:
    """Split-adjusted but NOT dividend-adjusted closes (Yahoo 'Close'), for market caps with
    back-adjusted shares.  Falls back to the total-return panel if the file is missing."""
    from .compact import EDGAR
    f = EDGAR / "prices_close_monthly.csv"
    if not f.exists():
        return prices
    c = pd.read_csv(f, index_col=0, parse_dates=True)
    c.index = pd.DatetimeIndex(c.index).to_period("M").to_timestamp("M")
    c = c.reindex(index=prices.index)
    return c.where(c > 0).combine_first(prices[[t for t in prices.columns if t in c.columns]]).reindex(columns=prices.columns)


# ---------------------------------------------------------------- panel

def _z(x: pd.Series) -> pd.Series:
    x = x.replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() < 10 or x.std() == 0 or pd.isna(x.std()):
        return x * np.nan
    z = (x - x.mean()) / x.std()
    return z.clip(-3, 3)


def build_panel(log=print, min_holders: int = MIN_HOLDERS) -> tuple[pd.DataFrame, dict]:
    books = load_books(log=log); prices = load_prices(); rets = clean_returns(prices); close = load_close(prices)
    fund = load_fundamentals()
    have_fund = not fund.empty
    if have_fund:
        fund["filed"] = pd.to_datetime(fund.filed); fund["end"] = pd.to_datetime(fund["end"])
        T = {"equity": pit_table(fund, "equity", "instant"), "assets": pit_table(fund, "assets", "instant"), "shares": clean_shares(pit_table(fund, "shares", "instant")),
             "net_income": pit_table(fund, "net_income", "annual"), "op_cf": pit_table(fund, "op_cf", "annual")}
    b = books[books.ticker.notna() & books.ticker.isin(prices.columns)]
    forms = sorted(f for f in b.formation.unique() if f in prices.index)
    uni_at = {}
    for F in forms:
        q = b[b.formation == F].groupby("ticker").slug.nunique()
        uni_at[F] = list(q[q >= min_holders].index)
    months = [m for m in prices.index if m >= forms[0] and prices.index.get_loc(m) >= 13]
    rows = []
    for m in months:
        F = max(f for f in forms if f <= m)
        uni = [t for t in uni_at[F] if prices.at[m, t] >= 1.0] if m in prices.index else []
        if len(uni) < MIN_NAMES:
            continue
        i = prices.index.get_loc(m)
        P = prices[uni]
        p0, p1, p12 = P.iloc[i], P.iloc[i - 1], P.iloc[i - 12]
        R12 = rets[uni].iloc[i - 11:i + 1]
        d = pd.DataFrame(index=uni)
        d["momentum"] = p1 / p12 - 1
        d["reversal"] = -(p0 / p1 - 1)
        d["low_vol"] = -R12.std()
        if have_fund:
            sh = as_of(T["shares"], d.index, m); eq = as_of(T["equity"], d.index, m); at = as_of(T["assets"], d.index, m)
            ni = as_of(T["net_income"], d.index, m); cf = as_of(T["op_cf"], d.index, m)
            mcap = sh * close.loc[m].reindex(d.index)
            d["size"] = -np.log(mcap.where(mcap > 0))
            d["value"] = (eq / mcap).where((eq > 0) & (mcap > 0))
            d["profitability"] = (ni / at).where(at > 0)
            d["cash_flow"] = (cf / at).where(at > 0)
            d["earnings_yield"] = (ni / mcap).where(mcap > 0)
        d["fwd"] = rets[uni].iloc[i + 1] if i + 1 < len(prices) else np.nan
        d["month"] = m
        rows.append(d.reset_index().rename(columns={"index": "ticker"}))
    panel = pd.concat(rows, ignore_index=True)
    for s in SIGNALS:
        if s in panel:
            panel[s + "_raw"] = panel[s]
            panel[s] = panel.groupby("month")[s].transform(_z)
    info = dict(months=int(panel.month.nunique()), universe_avg=int(round(panel.groupby("month").size().mean())), min_holders=int(min_holders),
                first=str(panel.month.min().date()), last=str(panel.month.max().date()), fundamentals=have_fund,
                coverage={s: float(panel[s].notna().mean()) for s in SIGNALS if s in panel})
    log(f"panel: {info['months']} months × ~{info['universe_avg']} names; signal coverage " + ", ".join(f"{k} {v:.0%}" for k, v in info["coverage"].items()))
    return panel, info


# ---------------------------------------------------------------- tests

def ic_series(panel: pd.DataFrame, col: str) -> pd.Series:
    out = {}
    for m, g in panel.groupby("month"):
        g = g.dropna(subset=[col, "fwd"])
        if len(g) >= MIN_NAMES:
            out[m] = float(stats.spearmanr(g[col], g.fwd).statistic)
    return pd.Series(out)


def decile_spread(panel: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    """Monthly D10 − D1 equal-weight return, and average return by decile."""
    sp, dec = {}, []
    for m, g in panel.groupby("month"):
        g = g.dropna(subset=[col, "fwd"])
        if len(g) < MIN_NAMES:
            continue
        q = pd.qcut(g[col].rank(method="first"), 10, labels=False) + 1
        by = g.fwd.groupby(q).mean()
        sp[m] = float(by.get(10, np.nan) - by.get(1, np.nan)); dec.append(by)
    return pd.Series(sp), pd.concat(dec, axis=1).mean(axis=1)


def _stats(ic: pd.Series, spread: pd.Series) -> dict:
    return dict(months=int(len(ic)), ic_mean=float(ic.mean()), ic_sd=float(ic.std()), ic_t=float(ic.mean() / ic.std() * np.sqrt(len(ic))) if ic.std() > 0 else np.nan,
                ic_pct_positive=float((ic > 0).mean()), spread_ann=float(spread.mean() * 12), spread_t=float(spread.mean() / spread.std() * np.sqrt(len(spread))) if spread.std() > 0 else np.nan,
                spread_sharpe=float(spread.mean() / spread.std() * np.sqrt(12)) if spread.std() > 0 else np.nan)


def walk_forward_xgb(panel: pd.DataFrame, feats: list[str], log=print) -> tuple[pd.Series, pd.DataFrame]:
    """Expanding-window gradient boosting: predict each month's cross-section of forward returns
    (demeaned within the month) from that month's z-scores, using only earlier months to fit."""
    import xgboost as xgb
    months = sorted(panel.month.unique())
    d = panel.dropna(subset=["fwd"]).copy()
    d["y"] = d.fwd - d.groupby("month").fwd.transform("mean")
    pred = pd.Series(np.nan, index=d.index); model = None; last_fit = None; imps = []
    params = dict(objective="reg:squarederror", max_depth=3, eta=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=50, reg_lambda=5.0, nthread=4, seed=0)
    ROUNDS = 300
    for k, m in enumerate(months):
        if k < MIN_TRAIN:
            continue
        if model is None or (k - last_fit) >= RETRAIN:
            train = d[d.month < m]
            dtrain = xgb.DMatrix(train[feats].fillna(0.0).values, label=train.y.values, feature_names=feats)
            model = xgb.train(params, dtrain, num_boost_round=ROUNDS); last_fit = k
            gain = model.get_score(importance_type="total_gain"); tot = sum(gain.values()) or 1.0
            imps.append(pd.Series({f: gain.get(f, 0.0) / tot for f in feats}, name=str(m.date())))
            log(f"  xgboost refit at {m.date()} on {len(train):,} rows")
        test = d[d.month == m]
        if len(test):
            pred.loc[test.index] = model.predict(xgb.DMatrix(test[feats].fillna(0.0).values, feature_names=feats))
    d["xgboost"] = pred
    return d["xgboost"], pd.concat(imps, axis=1)


def build(out_dir: Path = OUT_DIR, log=print) -> dict:
    panel, info = build_panel(log)
    feats = [s for s in SIGNALS if s in panel and panel[s].notna().mean() > 0.3]
    panel["linear"] = panel[feats].mean(axis=1, skipna=True).where(panel[feats].notna().sum(axis=1) >= max(2, len(feats) // 2))
    panel["linear"] = panel.groupby("month")["linear"].transform(_z)
    xg, imps = walk_forward_xgb(panel, feats, log)
    panel["xgboost"] = xg
    # same evaluation window for the two models: only months where xgboost has predictions
    xg_months = sorted(panel.loc[panel.xgboost.notna(), "month"].unique())
    summary, ics, decs = [], {}, {}
    for col in feats + ["linear", "xgboost"]:
        sub = panel if col not in ("linear", "xgboost") else panel[panel.month.isin(xg_months)]
        ic = ic_series(sub, col); sp, dec = decile_spread(sub, col)
        if len(ic) < 12:
            continue
        row = dict(signal=col, label=SIGNALS.get(col, {"linear": "Linear composite (equal-weight z)", "xgboost": "xgboost (walk-forward)"}[col] if col in ("linear", "xgboost") else col),
                   coverage=float(panel[col].notna().mean()), first=str(ic.index.min().date()), last=str(ic.index.max().date()), **_stats(ic, sp))
        summary.append(row); ics[col] = ic; decs[col] = dec
    # linear vs xgboost, head to head, same months
    lin_ic, xg_ic = ics.get("linear"), ics.get("xgboost")
    diff = (xg_ic - lin_ic).dropna() if lin_ic is not None and xg_ic is not None else pd.Series(dtype=float)
    head = dict(months=int(len(diff)), ic_diff_mean=float(diff.mean()) if len(diff) else np.nan,
                ic_diff_t=float(diff.mean() / diff.std() * np.sqrt(len(diff))) if len(diff) > 1 and diff.std() > 0 else np.nan,
                xgb_wins_share=float((diff > 0).mean()) if len(diff) else np.nan)
    out_dir.mkdir(parents=True, exist_ok=True)
    S = pd.DataFrame(summary); S.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(ics).to_csv(out_dir / "ic_monthly.csv", index_label="month")
    pd.DataFrame(decs).to_csv(out_dir / "deciles.csv", index_label="decile")
    imps.to_csv(out_dir / "xgboost_importance.csv", index_label="feature")
    # correlation of signals (average monthly rank correlation)
    corr = panel.groupby("month")[feats].corr(method="spearman").groupby(level=1).mean().reindex(feats)[feats]
    corr.to_csv(out_dir / "signal_correlation.csv", index_label="signal")
    latest = panel[panel.month == panel.month.max()].sort_values("linear", ascending=False)
    latest[["ticker"] + feats + ["linear", "xgboost"]].head(40).to_csv(out_dir / "latest_ranks.csv", index=False)
    man = dict(built=str(date.today()), **info, features=feats, min_train=MIN_TRAIN, retrain=RETRAIN, head_to_head=head,
               xgb_first=str(xg_months[0].date()) if xg_months else None, xgb_months=len(xg_months))
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    log(f"wrote {out_dir}")
    return man
