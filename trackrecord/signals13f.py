"""Cross-sectional research on the 13F universe: do managers' disclosed books carry a signal?

Three questions, each a long-only portfolio formed once a quarter from every
manager's public filing, held with drift for three months:

  best ideas   — each manager's largest position (rank 1) and top 3, weighted by how
                 many managers name the stock (Cohen, Polk & Silli 2010 "Best Ideas")
  crowding     — the universe of held stocks split into quintiles by how many managers
                 hold each name ("hedge-fund hotels" vs names held by one manager)
  conviction   — what managers just did: new positions, positions added to (shares
                 +25%), trimmed (−25%), and sold entirely

Portfolios form at the end of Feb / May / Aug / Nov — the month in which the
quarter's 13Fs become public (45-day deadline) — using only filings that were
public by then.  Everything is measured against the US market and Carhart four
factors with HAC standard errors, and the honest tests are the *spreads*:
best-ideas minus the whole universe, crowded minus lonely, new minus sold.

Known limits, disclosed on the page: prices are survivor-biased (delisted names
drop out at the last price), only long US positions are visible, a manager's
whole book is treated as one filer (entities merged), and the sample starts
2013 (structured filings).  Outputs are CSVs under data/research/13f-signals/.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .compact import COMPACT, EDGAR, holdings_frame, unpack
from .validation import MODELS, _fit

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "research" / "13f-signals"
MIN_BOOK = 5             # a filing needs this many long positions to define a "best idea"
ADD_CUT = 0.25           # shares ±25% counts as an add / a trim
MIN_NAMES = 5            # a portfolio month needs this many priced names, else NaN (never bridged)
MIN_PRICE = 1.0          # stocks under $1 at formation are left out (standard microcap / data-quality filter)
SPIKE = (3.0, 5.0)       # a month of > +300% from a start price under $5 is a data error (OTC / bankrupt tickers), not a return
PORTFOLIOS = {           # key: (label, what it holds)
    "ALL":      ("Every held stock", "every US stock in any manager's book, equal-weighted"),
    "BEST1":    ("Best ideas (#1)", "each manager's largest position, weighted by how many managers name it"),
    "BEST3":    ("Top-3 positions", "each manager's three largest positions, pair-weighted"),
    "CROWD_HI": ("Most crowded", "top quintile of held stocks by number of managers holding"),
    "CROWD_LO": ("Least crowded", "bottom quintile — held by a single manager, smallest dollar stake"),
    "NEW":      ("New positions", "stocks a manager bought this quarter that it did not hold last quarter"),
    "ADD":      ("Added to", "positions where shares rose ≥ 25% vs last quarter"),
    "CUT":      ("Trimmed", "positions where shares fell ≥ 25% but are still held"),
    "SOLD":     ("Sold out", "stocks a manager held last quarter and no longer holds"),
}
SPREADS = [("BEST1", "ALL"), ("BEST3", "ALL"), ("CROWD_HI", "CROWD_LO"), ("NEW", "ALL"), ("NEW", "SOLD"), ("ADD", "CUT")]


# ---------------------------------------------------------------- inputs

def _month_end(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("M").to_timestamp("M")


def formation_date(period: str) -> pd.Timestamp:
    """Quarter end → end of the month in which its 13Fs are public (45-day deadline → +2 months)."""
    return (pd.Timestamp(period).to_period("M") + 2).to_timestamp("M")


def load_books(log=print) -> pd.DataFrame:
    """One row per (manager, quarter, ticker): merged entities, puts/calls out, full-book weight,
    rank by value, shares, plus the formation date the filing is eligible for."""
    unpack(log=log)
    h = holdings_frame()
    h = h[(h.putcall == "") & (h.value > 0)]
    cmap = json.loads((EDGAR / "cusip_map.json").read_text())
    h["ticker"] = h.cusip.map(cmap)
    filed = h.groupby(["slug", "period"]).filed.max()
    g = (h.groupby(["slug", "period", "cusip"]).agg(value=("value", "sum"), shares=("shares", "sum"),
                                                     name=("name", "first"), ticker=("ticker", "first")).reset_index())
    g["filed"] = [filed[(s, p)] for s, p in zip(g.slug, g.period)]
    g["w_all"] = g.value / g.groupby(["slug", "period"]).value.transform("sum")
    g = g.sort_values(["slug", "period", "value"], ascending=[True, True, False])
    g["rank"] = g.groupby(["slug", "period"]).cumcount() + 1
    g["n_book"] = g.groupby(["slug", "period"]).cusip.transform("size")
    g["formation"] = g.period.map(formation_date)
    g = g[pd.to_datetime(g.filed) <= g.formation]          # public in time for the formation date
    g = g[g.n_book >= MIN_BOOK]
    # conviction changes vs the manager's previous quarter (only if that filing exists)
    prev_period = (pd.to_datetime(g.period).dt.to_period("Q") - 1).dt.end_time.dt.strftime("%Y-%m-%d")
    g["prev_period"] = prev_period.values
    key_prev = g.set_index(["slug", "period", "cusip"]).shares
    have_prev = set(zip(g.slug, g.period))
    g["has_prev"] = [(s, p) in have_prev for s, p in zip(g.slug, g.prev_period)]
    idx = pd.MultiIndex.from_arrays([g.slug, g.prev_period, g.cusip])
    g["shares_prev"] = key_prev.reindex(idx).values
    g["new"] = g.has_prev & g.shares_prev.isna()
    chg = g.shares / g.shares_prev - 1
    g["add"] = g.has_prev & g.shares_prev.notna() & (g.shares > 0) & (chg >= ADD_CUT)
    g["cut"] = g.has_prev & g.shares_prev.notna() & (g.shares > 0) & (chg <= -ADD_CUT)
    log(f"books: {g.slug.nunique()} managers, {g.period.nunique()} quarters, {len(g)} positions, "
        f"{g.ticker.notna().mean():.0%} mapped to a ticker")
    return g.reset_index(drop=True)


def sold_positions(books: pd.DataFrame) -> pd.DataFrame:
    """(slug, period, ticker) held in the previous quarter and absent now — attributed to the current formation."""
    cur = set(zip(books.slug, books.period, books.cusip))
    nxt = books[["slug", "period", "cusip", "ticker"]].copy()
    nxt["next_period"] = (pd.to_datetime(nxt.period).dt.to_period("Q") + 1).dt.end_time.dt.strftime("%Y-%m-%d")
    filed_next = set(zip(books.slug, books.period))
    nxt = nxt[[(s, p) in filed_next for s, p in zip(nxt.slug, nxt.next_period)]]       # manager did file next quarter
    nxt = nxt[[(s, p, c) not in cur for s, p, c in zip(nxt.slug, nxt.next_period, nxt.cusip)]]
    nxt["formation"] = nxt.next_period.map(formation_date)
    return nxt[["slug", "next_period", "ticker", "formation"]].rename(columns={"next_period": "period"})


def load_prices() -> pd.DataFrame:
    p = pd.read_csv(EDGAR / "prices_monthly.csv", index_col=0, parse_dates=True)
    p.index = pd.DatetimeIndex(p.index).to_period("M").to_timestamp("M")
    last_full = (pd.Timestamp.today().to_period("M") - 1).to_timestamp("M")
    return p[p.index <= last_full].where(lambda x: x > 0)


def clean_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Monthly returns with the data-error guard applied (counted in the manifest)."""
    r = prices.pct_change(fill_method=None)
    bad = (r > SPIKE[0]) & (prices.shift(1) < SPIKE[1])
    clean_returns.n_spikes = int(bad.sum().sum())
    return r.mask(bad)


def load_factors() -> pd.DataFrame:
    from .reference import load_all
    f = load_all()
    f.index = pd.DatetimeIndex(f.index).to_period("M").to_timestamp("M")
    cols = {"US_MKT_RF": "MKT_RF", "US_SMB": "SMB", "US_HML": "HML", "US_MOM": "MOM", "US_RF": "RF", "US_MKT": "MKT"}
    return f[list(cols)].rename(columns=cols)


# ---------------------------------------------------------------- portfolios

def formation_weights(books: pd.DataFrame, sold: pd.DataFrame, F: pd.Timestamp) -> dict[str, pd.Series]:
    """Target weights (ticker → weight) for every portfolio at formation date F."""
    b = books[(books.formation == F) & books.ticker.notna()]
    out = {}
    if b.empty:
        return out
    def pairs(mask) -> pd.Series:
        c = b[mask].groupby("ticker").size().astype(float)
        return c / c.sum() if len(c) else c
    uni = b.ticker.unique()
    out["ALL"] = pd.Series(1.0 / len(uni), index=uni)                          # equal weight per stock
    out["BEST1"] = pairs(b["rank"] == 1)
    out["BEST3"] = pairs(b["rank"] <= 3)
    x = b.groupby("ticker").agg(n_holders=("slug", "nunique"), tot_value=("value", "sum"))
    x = x.sort_values(["n_holders", "tot_value"])
    q = np.floor(np.arange(len(x)) * 5 / len(x)).astype(int)
    out["CROWD_LO"] = pd.Series(1.0 / (q == 0).sum(), index=x.index[q == 0])
    out["CROWD_HI"] = pd.Series(1.0 / (q == 4).sum(), index=x.index[q == 4])
    out["NEW"] = pairs(b.new); out["ADD"] = pairs(b["add"]); out["CUT"] = pairs(b.cut)
    s = sold[(sold.formation == F) & sold.ticker.notna()]
    c = s.groupby("ticker").size().astype(float)
    out["SOLD"] = c / c.sum() if len(c) else c
    return {k: v for k, v in out.items() if len(v)}


def run_portfolios(books: pd.DataFrame, sold: pd.DataFrame, prices: pd.DataFrame, log=print):
    """Monthly returns (month × portfolio), per-formation diagnostics, and per-formation weights."""
    rets = clean_returns(prices)
    forms = sorted(f for f in books.formation.unique() if f in prices.index)
    months = prices.index[(prices.index > forms[0])]
    R = pd.DataFrame(np.nan, index=months, columns=list(PORTFOLIOS))
    diag = []; weights = {}
    prev = {}
    for k, F in enumerate(forms):
        end = forms[k + 1] if k + 1 < len(forms) else months[-1]
        W = formation_weights(books, sold, F)
        row = dict(formation=F.date(), managers=int(books[books.formation == F].slug.nunique()))
        for key, w in W.items():
            w = w[w.index.isin(prices.columns)]
            w = w[w.index.map(prices.loc[F]) >= MIN_PRICE]              # priced at formation, not a penny stock
            row[f"n_{key}"] = int(len(w)); row[f"priced_{key}"] = float(w.sum())      # share of target weight with a price
            if w.empty:
                continue
            w = w / w.sum()
            if key in prev:
                row[f"turnover_{key}"] = float((w.subtract(prev[key], fill_value=0)).abs().sum() / 2)
            cur = w.copy()
            for m in months[(months > F) & (months <= end)]:
                r = rets.loc[m, cur.index]
                ok = r.notna()
                if int(ok.sum()) < MIN_NAMES:
                    continue
                wk = cur[ok] / cur[ok].sum()
                R.loc[m, key] = float((wk * r[ok]).sum())
                cur = cur[ok] * (1 + r[ok]); cur = cur / cur.sum()
            prev[key] = cur
            weights.setdefault(key, {})[F] = w
        diag.append(row)
    log(f"portfolios: {len(forms)} formation dates {forms[0].date()} → {forms[-1].date()}, {R.notna().any(axis=1).sum()} months")
    return R, pd.DataFrame(diag), weights


# ---------------------------------------------------------------- evaluation

def _ann(r: pd.Series) -> float:
    r = r.dropna()
    return float((1 + r).prod() ** (12 / len(r)) - 1) if len(r) else np.nan


def _max_dd(r: pd.Series) -> float:
    c = (1 + r.fillna(0)).cumprod(); return float((c / c.cummax() - 1).min())


def _alpha(y: pd.Series, fac: pd.DataFrame, model: str) -> dict:
    cols = MODELS[model]
    df = pd.concat([y.rename("y"), fac[cols]], axis=1).dropna()
    if len(df) < len(cols) + 6:
        return dict(n=len(df))
    fit, lag = _fit(df.y, df[cols], 12)
    out = dict(n=len(df), alpha=float(fit.params[0] * 12), t=float(fit.tvalues[0]), p=float(fit.pvalues[0]), hac_lag=lag,
               beta=float(fit.params[1]), r2=float(fit.rsquared))
    for i, c in enumerate(cols[1:], start=2):
        out[f"b_{c}"] = float(fit.params[i])
    return out


def summarize(R: pd.DataFrame, fac: pd.DataFrame, diag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fac = fac.reindex(R.index)
    rows = []
    for key, (label, what) in PORTFOLIOS.items():
        r = R[key]; ex = r - fac.RF
        ok = r.notna()
        row = dict(key=key, label=label, holds=what, months=int(ok.sum()), first=str(r[ok].index.min().date()) if ok.any() else "",
                   last=str(r[ok].index.max().date()) if ok.any() else "",
                   ann_return=_ann(r), ann_vol=float(r.std() * np.sqrt(12)), sharpe=float(ex.mean() / ex.std() * np.sqrt(12)) if ex.std() > 0 else np.nan,
                   max_dd=_max_dd(r), excess_vs_market=_ann(r) - _ann(fac.MKT[ok]),
                   avg_names=float(diag[f"n_{key}"].mean()) if f"n_{key}" in diag else np.nan,
                   avg_turnover=float(diag[f"turnover_{key}"].mean()) if f"turnover_{key}" in diag else np.nan)
        a1 = _alpha(ex, fac, "CAPM"); a4 = _alpha(ex, fac, "Carhart4")
        row.update(capm_alpha=a1.get("alpha"), capm_t=a1.get("t"), beta=a1.get("beta"),
                   carhart_alpha=a4.get("alpha"), carhart_t=a4.get("t"), carhart_p=a4.get("p"),
                   b_SMB=a4.get("b_SMB"), b_HML=a4.get("b_HML"), b_MOM=a4.get("b_MOM"))
        rows.append(row)
    mk = fac.MKT[R.ALL.notna()]
    rows.append(dict(key="MKT", label="US market", holds="Ken French total US market (CRSP value-weighted)", months=int(mk.notna().sum()),
                     first=str(mk.index.min().date()), last=str(mk.index.max().date()), ann_return=_ann(mk), ann_vol=float(mk.std() * np.sqrt(12)),
                     sharpe=float((mk - fac.RF).mean() / (mk - fac.RF).std() * np.sqrt(12)), max_dd=_max_dd(mk), excess_vs_market=0.0, beta=1.0))
    summ = pd.DataFrame(rows)
    srows = []
    for a, b in SPREADS:
        d = (R[a] - R[b]).dropna()
        if len(d) < 12:
            continue
        t_raw = float(d.mean() / d.std() * np.sqrt(len(d)))
        a4 = _alpha(d, fac, "Carhart4"); a1 = _alpha(d, fac, "CAPM")
        srows.append(dict(long=a, short=b, label=f"{PORTFOLIOS[a][0]} − {PORTFOLIOS[b][0]}", months=len(d),
                          ann_spread=float(d.mean() * 12), t_spread=t_raw, share_positive_months=float((d > 0).mean()),
                          capm_alpha=a1.get("alpha"), capm_t=a1.get("t"), carhart_alpha=a4.get("alpha"), carhart_t=a4.get("t"), carhart_p=a4.get("p"),
                          b_MKT=a4.get("beta"), b_SMB=a4.get("b_SMB"), b_HML=a4.get("b_HML"), b_MOM=a4.get("b_MOM"),
                          worst_year=float(d.groupby(d.index.year).sum().min()), best_year=float(d.groupby(d.index.year).sum().max())))
    return summ, pd.DataFrame(srows)


def information_coefficients(books: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman rank correlation, at each formation date, between a stock's signal and its return
    over the following quarter (formation → next formation)."""
    forms = sorted(f for f in books.formation.unique() if f in prices.index)
    rows = []
    for k, F in enumerate(forms[:-1]):
        F2 = forms[k + 1]
        b = books[(books.formation == F) & books.ticker.notna()]
        x = b.groupby("ticker").agg(n_holders=("slug", "nunique"), best_ideas=("rank", lambda s: int((s <= 3).sum())),
                                    avg_weight=("w_all", "mean"), n_new=("new", "sum"), n_add=("add", "sum"), n_cut=("cut", "sum"))
        x["net_activity"] = x.n_new + x.n_add - x.n_cut
        x = x[x.index.isin(prices.columns)]
        x = x[prices.loc[F, x.index] >= MIN_PRICE]
        q = prices.loc[F2, x.index] / prices.loc[F, x.index] - 1
        x = x[q.notna()]; q = q[q.notna()]
        if len(x) < 30:
            continue
        row = dict(formation=F.date(), n_stocks=len(x))
        for sig in ["n_holders", "best_ideas", "avg_weight", "net_activity"]:
            row[sig] = float(stats.spearmanr(x[sig], q).statistic)
        rows.append(row)
    ic = pd.DataFrame(rows)
    summ = []
    for sig, label in [("n_holders", "Crowding (managers holding)"), ("best_ideas", "Best-idea count (top-3 mentions)"),
                       ("avg_weight", "Average position weight"), ("net_activity", "Net activity (new + adds − trims)")]:
        s = ic[sig].dropna()
        summ.append(dict(signal=sig, label=label, quarters=len(s), mean_ic=float(s.mean()), sd_ic=float(s.std()),
                         t=float(s.mean() / s.std() * np.sqrt(len(s))), share_positive=float((s > 0).mean())))
    return ic, pd.DataFrame(summ)


def latest_crowding(books: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    F = books.formation.max()
    b = books[(books.formation == F) & books.ticker.notna()]
    x = (b.groupby("ticker").agg(name=("name", "first"), n_holders=("slug", "nunique"), best_idea_of=("rank", lambda s: int((s == 1).sum())),
                                 top3_of=("rank", lambda s: int((s <= 3).sum())), avg_weight=("w_all", "mean"),
                                 new_buyers=("new", "sum"), adders=("add", "sum"), trimmers=("cut", "sum"))
         .sort_values(["n_holders", "top3_of"], ascending=False).head(n).reset_index())
    x["formation"] = F.date(); x["name"] = x["name"].str.title()
    return x


# ---------------------------------------------------------------- driver

def build(out_dir: Path = OUT_DIR, log=print) -> dict:
    books = load_books(log)
    sold = sold_positions(books)
    prices = load_prices(); fac = load_factors()
    R, diag, _ = run_portfolios(books, sold, prices, log)
    summ, spreads = summarize(R, fac, diag)
    ic, ic_summ = information_coefficients(books, prices)
    crowd = latest_crowding(books)
    out_dir.mkdir(parents=True, exist_ok=True)
    R.join(fac[["MKT", "RF"]]).to_csv(out_dir / "portfolios_monthly.csv", index_label="month")
    summ.to_csv(out_dir / "summary.csv", index=False); spreads.to_csv(out_dir / "spreads.csv", index=False)
    diag.to_csv(out_dir / "formation.csv", index=False); ic.to_csv(out_dir / "ic.csv", index=False)
    ic_summ.to_csv(out_dir / "ic_summary.csv", index=False); crowd.to_csv(out_dir / "latest_crowding.csv", index=False)
    info = dict(built=str(date.today()), managers=int(books.slug.nunique()), quarters=int(books.period.nunique()),
                first_formation=str(diag.formation.min()), last_formation=str(diag.formation.max()),
                months=int(R.ALL.notna().sum()), positions=int(len(books)), min_book=MIN_BOOK, add_cut=ADD_CUT, min_names=MIN_NAMES,
                min_price=MIN_PRICE, spike_rule=f"> +{SPIKE[0]:.0%} from under ${SPIKE[1]:.0f}", spikes_removed=getattr(clean_returns, "n_spikes", 0))
    (out_dir / "manifest.json").write_text(json.dumps(info, indent=1))
    log(f"wrote {out_dir}")
    return info
