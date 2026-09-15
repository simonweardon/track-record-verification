"""Passive index management: replicate an index with a sampled portfolio, run it, report it.

The index.  A vendor index is a rulebook plus constituents; here the rulebook is: every
US-domestic filer (10-K/10-Q) held by ≥ 5 of the managers in the system, with point-in-time
shares outstanding from its SEC filings (unit errors removed, split-adjusted to today's
basis) times the split-adjusted close, weighted by market capitalisation and reconstituted
at each 13F formation date (end of Feb / May / Aug / Nov).  Call it the Universe Cap-Weighted Index.
Its constituents and weights are known in advance, exactly as an index provider's file
would be — which is what makes holdings-based replication possible.

Replication.  Three ways to hold it:
    full          every constituent at index weight — zero tracking error, hundreds of trades
    sampled N     the N names that minimise *predicted* tracking error under the risk model:
                  the largest names in each industry are candidates, then a quadratic program
                  chooses their weights so that (w − b)'(XFX' + Δ)(w − b) is smallest with
                  w ≥ 0, Σw = 1, and no name above a cap
    each rebalanced at reconstitution and held with drift in between (a passive book does
    not trade between events).

Reporting.  The Daily Portfolio Status Worksheet the desk fills in every morning: NAV,
cash, holdings, largest active weights, sector actives, predicted tracking error, month-
and quarter-to-date performance vs the index, and the event list — index additions and
deletions at the coming reconstitution and names that have stopped pricing — with the
trades each implies.  A cash-flow slice (invest an inflow pro rata) is the other daily
task and is produced the same way.

Outputs: data/research/index-tracker/.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .alphalab import as_of, clean_shares, load_close, pit_table
from .fundamentals import load as load_fundamentals
from .riskmodel import build_model, decompose, exposures, factor_cov, specific_var
from .signals13f import clean_returns, load_prices

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "research" / "index-tracker"
INDEX_NAME = "Universe Cap-Weighted Index"
SAMPLES = [40, 80, 150]
MAX_WEIGHT = 0.10
NAV = 250_000_000.0
CASH_TARGET = 0.005
COST_BPS = 5.0


# ---------------------------------------------------------------- the index

def index_weights(panel: pd.DataFrame, shares_tbl: pd.DataFrame, close: pd.DataFrame, F: pd.Timestamp) -> pd.Series:
    """Cap weights at F: back-adjusted shares × split-adjusted close."""
    names = list(panel[panel.month == F].ticker.unique())
    sh = as_of(shares_tbl, pd.Index(names), F)
    px = close.loc[F].reindex(names)
    cap = (sh * px).dropna()
    cap = cap[cap > 0]
    return (cap / cap.sum()).sort_values(ascending=False)


# ---------------------------------------------------------------- sampled replication

def candidates(b: pd.Series, sectors: pd.Series, n: int) -> list[str]:
    """Largest names per industry, seats allocated in proportion to the industry's index weight (at least one each)."""
    sec = sectors.reindex(b.index).fillna("Unknown")
    wsec = b.groupby(sec).sum().sort_values(ascending=False)
    seats = {s: max(1, int(round(n * w))) for s, w in wsec.items()}
    while sum(seats.values()) > n:
        s = max(seats, key=lambda k: seats[k]); seats[s] -= 1
    out = []
    for s, k in seats.items():
        out += list(b[sec == s].sort_values(ascending=False).head(k).index)
    return out


def sampled_weights(b: pd.Series, cand: list[str], X: pd.DataFrame, F: pd.DataFrame, D: pd.Series, max_weight: float = MAX_WEIGHT) -> tuple[pd.Series, float]:
    """Weights on the candidate names minimising predicted tracking error to b under the risk model."""
    names = list(b.index)
    cand = [c for c in cand if c in names and c in X.index]
    Xn = X.reindex(names).fillna(0.0)[F.index].values
    Fv = F.values
    Dv = D.reindex(names).fillna(float(D.median())).values
    bv = b.values
    idx = np.array([names.index(c) for c in cand])
    def full(wc):
        w = np.zeros(len(names)); w[idx] = wc; return w
    def te2(wc):
        a = full(wc) - bv; x = Xn.T @ a
        return float(x @ Fv @ x + (a ** 2 * Dv).sum())
    def grad(wc):
        a = full(wc) - bv; x = Xn.T @ a
        g = 2 * (Xn @ (Fv @ x)) + 2 * a * Dv
        return g[idx]
    w0 = b.reindex(cand).values; w0 = w0 / w0.sum()
    caps = [max(max_weight, float(b[c])) for c in cand]                  # a mega-cap above the cap may be held at index weight
    res = minimize(te2, w0, jac=grad, method="SLSQP", bounds=[(0.0, hi) for hi in caps],
                   constraints=[dict(type="eq", fun=lambda w: w.sum() - 1.0, jac=lambda w: np.ones(len(w)))], options=dict(maxiter=300, ftol=1e-12))
    w = pd.Series(np.clip(res.x, 0, None), index=cand); w = w / w.sum()
    return w, float(np.sqrt(max(te2(w.values), 0.0) * 12))


# ---------------------------------------------------------------- backtest

def run(log=print) -> dict:
    model = build_model(log)
    panel, fr, U, styles, secs = model["panel"], model["factor_returns"], model["residuals"], model["styles"], model["sectors"]
    sectors = panel.drop_duplicates("ticker").set_index("ticker").sector
    prices = load_prices(); rets = clean_returns(prices); close = load_close(prices)
    fund = load_fundamentals(); fund["filed"] = pd.to_datetime(fund.filed); fund["end"] = pd.to_datetime(fund["end"])
    shares_tbl = clean_shares(pit_table(fund, "shares", "instant"))
    forms = sorted(m for m in panel.month.unique() if m.month in (2, 5, 8, 11) and (fr.index < m).sum() >= 24)
    months = prices.index[prices.index > forms[0]]
    keys = ["index", "full"] + [f"sampled_{n}" for n in SAMPLES]
    R = pd.DataFrame(np.nan, index=months, columns=keys)
    cur = {k: None for k in keys}; log_rows = []; last = None
    for k, Fm in enumerate(forms):
        end = forms[k + 1] if k + 1 < len(forms) else months[-1]
        b = index_weights(panel, shares_tbl, close, Fm)
        if len(b) < 50:
            continue
        X = exposures(panel, styles, secs, Fm); Fc = factor_cov(fr, Fm); D = specific_var(U, Fm)
        targets = {"index": b, "full": b.copy()}
        row = dict(formation=Fm.date(), constituents=len(b), top10_weight=float(b.head(10).sum()))
        for n in SAMPLES:
            cand = candidates(b, sectors, n)
            w, te = sampled_weights(b, cand, X, Fc, D)
            targets[f"sampled_{n}"] = w
            row[f"pred_te_{n}"] = te; row[f"names_{n}"] = int((w > 0).sum())
            d = decompose(w.reindex(b.index.union(w.index)).fillna(0) - b.reindex(b.index.union(w.index)).fillna(0), X, Fc, D, styles)
            row[f"pred_te_model_{n}"] = d["total"]
        prev_books = {k: (v.copy() if v is not None else None) for k, v in cur.items()}     # the book as it stands before this rebalance
        # turnover vs the drifted previous book, then hold with drift; costs charged in the first month
        for key, wt in targets.items():
            prev = cur[key]
            to = float((wt.subtract(prev, fill_value=0)).abs().sum() / 2) if prev is not None else 1.0
            row[f"turnover_{key}"] = to
            c = wt[wt > 0].copy(); first = True
            for m in months[(months > Fm) & (months <= end)]:
                r = rets.loc[m, c.index]; ok = r.notna()
                if ok.sum() == 0: break
                wk = c[ok] / c[ok].sum()
                cost = (to * 2 * COST_BPS / 1e4) if (first and key != "index") else 0.0
                R.loc[m, key] = float((wk * r[ok]).sum()) - cost
                c = c[ok] * (1 + r[ok]); c = c / c.sum(); first = False
            cur[key] = c
        log_rows.append(row); last = dict(F=Fm, b=b, targets=targets, X=X, Fc=Fc, D=D, cur={k: v.copy() for k, v in cur.items()}, prev_books=prev_books)
        log(f"  {Fm.date()}: {len(b)} constituents; predicted TE " + ", ".join(f"{n}: {row[f'pred_te_{n}']:.2%}" for n in SAMPLES))
    return dict(R=R, log=pd.DataFrame(log_rows), last=last, model=model, prices=prices, sectors=sectors, forms=forms)


def summarize(R: pd.DataFrame, lg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = R["index"].notna()
    for key in R.columns:
        r = R[key][ok]; act = r - R["index"][ok]
        ann = float((1 + r).prod() ** (12 / len(r)) - 1)
        te = float(act.std() * np.sqrt(12))
        n = key.split("_")[1] if key.startswith("sampled") else None
        rows.append(dict(key=key, label={"index": INDEX_NAME, "full": "Full replication"}.get(key, f"Optimised sampling, {n} names"),
                         months=int(len(r)), ann_return=ann, ann_vol=float(r.std() * np.sqrt(12)),
                         tracking_error=te if key != "index" else 0.0, tracking_difference=float(act.mean() * 12) if key != "index" else 0.0,
                         worst_month_active=float(act.min()) if key != "index" else 0.0,
                         avg_names=float(lg[f"names_{n}"].mean()) if n else (float(lg.constituents.mean()) if key != "index" else float(lg.constituents.mean())),
                         turnover_yr=float(lg[f"turnover_{key}"].iloc[1:].mean() * 4) if key != "index" else float(lg["turnover_index"].iloc[1:].mean() * 4),
                         pred_te_avg=float(lg[f"pred_te_{n}"].mean()) if n else (0.0 if key == "full" else np.nan)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the daily sheet

def status_worksheet(res: dict, n: int = 80, nav: float = NAV) -> dict:
    """Daily Portfolio Status Worksheet for the sampled-N book at the latest date."""
    last, R, prices, sectors, model = res["last"], res["R"], res["prices"], res["sectors"], res["model"]
    key = f"sampled_{n}"; F = last["F"]
    styles = model["styles"]
    w_target = last["targets"][key]; w_now = last["cur"][key]           # drifted holdings today
    b_target = last["b"]; b_now = last["cur"]["index"]
    today = prices.index.max()
    port = R[key].dropna(); idx = R["index"].dropna()
    def since(start):
        p = port[port.index > start]; i = idx[idx.index > start]
        return float((1 + p).prod() - 1), float((1 + i).prod() - 1)
    mtd = (float(port.iloc[-1]), float(idx.iloc[-1]))
    prev_forms = [f for f in res["forms"] if f < today]
    qtd = since(prev_forms[-1]) if prev_forms else (0.0, 0.0)
    ytd = since(pd.Timestamp(year=today.year, month=1, day=1) - pd.Timedelta(days=1))
    active = w_now.reindex(w_now.index.union(b_now.index)).fillna(0) - b_now.reindex(w_now.index.union(b_now.index)).fillna(0)
    sec_act = active.groupby(sectors.reindex(active.index).fillna("Unknown")).sum().sort_values()
    d = decompose(active, last["X"], last["Fc"], last["D"], styles)
    # events: names that will enter / leave the index at the next reconstitution are unknown until the filings arrive;
    # what is known today is the drift from target and names that have stopped pricing
    dead = [t for t in w_target.index if pd.isna(prices.loc[today, t]) or prices.loc[today, t] <= 0]
    drift = (w_now - w_target.reindex(w_now.index).fillna(0)).abs()
    cash = nav * CASH_TARGET
    holdings = pd.DataFrame({"weight": w_now, "target": w_target.reindex(w_now.index).fillna(0), "index_weight": b_now.reindex(w_now.index).fillna(0)})
    holdings["active"] = holdings.weight - holdings.index_weight
    holdings["market_value"] = holdings.weight * (nav - cash)
    holdings["price"] = prices.loc[today].reindex(holdings.index)
    holdings["shares"] = np.floor(holdings.market_value / holdings.price.where(holdings.price > 0))
    holdings["sector"] = sectors.reindex(holdings.index).fillna("Unknown")
    holdings = holdings.sort_values("weight", ascending=False)
    # cash-flow slice: invest an inflow of 2% of NAV pro rata to target weights
    inflow = 0.02 * nav
    slice_ = pd.DataFrame({"weight": w_target, "usd": w_target * inflow}); slice_["price"] = prices.loc[today].reindex(slice_.index)
    slice_["shares"] = np.floor(slice_.usd / slice_.price.where(slice_.price > 0)); slice_["sector"] = sectors.reindex(slice_.index).fillna("Unknown")
    return dict(date=str(today.date()), formation=str(F.date()), nav=nav, cash=cash, cash_pct=CASH_TARGET, n_holdings=int((w_now > 0).sum()), n_index=int(len(b_now)),
                mtd=mtd, qtd=qtd, ytd=ytd, pred_te=d["total"], pred_te_factor=d["factor"], pred_te_specific=d["specific"],
                max_active=float(active.abs().max()), max_active_name=str(active.abs().idxmax()), sector_active=sec_act.to_dict(),
                drift_max=float(drift.max()), drift_max_name=str(drift.idxmax()), dead=dead, holdings=holdings, slice=slice_, inflow=inflow,
                largest_active=active.reindex(active.abs().sort_values(ascending=False).index).head(10).to_dict(),
                top_risk=d["top"][:6])


def build(out_dir: Path = OUT_DIR, log=print) -> dict:
    res = run(log)
    R, lg, last = res["R"], res["log"], res["last"]
    out_dir.mkdir(parents=True, exist_ok=True)
    R.to_csv(out_dir / "backtest_monthly.csv", index_label="month")
    lg.to_csv(out_dir / "rebalances.csv", index=False)
    S = summarize(R, lg); S.to_csv(out_dir / "summary.csv", index=False)
    ws = status_worksheet(res, 80)
    ws["holdings"].to_csv(out_dir / "dpsw_holdings.csv", index_label="ticker")
    ws["slice"].to_csv(out_dir / "dpsw_cashflow_slice.csv", index_label="ticker")
    meta = {k: v for k, v in ws.items() if k not in ("holdings", "slice")}
    (out_dir / "dpsw.json").write_text(json.dumps(meta, indent=1, default=float))
    # the latest reconstitution's trade list for the 80-name book: the drifted previous book -> the new target
    key = "sampled_80"; F = last["F"]
    prev_book = last["prev_books"].get(key); target = last["targets"][key]
    n_trades, traded = 0, 0.0
    if prev_book is not None:
        tick = sorted(set(target.index) | set(prev_book.index))
        tl = pd.DataFrame(index=tick)
        tl["sector"] = res["sectors"].reindex(tick).fillna("Unknown")
        tl["current_wt"] = prev_book.reindex(tick).fillna(0); tl["target_wt"] = target.reindex(tick).fillna(0)
        tl["index_wt"] = last["b"].reindex(tick).fillna(0)
        tl["price"] = res["prices"].loc[F].reindex(tick)
        tl["trade_usd"] = (tl.target_wt - tl.current_wt) * NAV * (1 - CASH_TARGET)
        tl["shares"] = np.where(tl.price > 0, np.round(tl.trade_usd / tl.price), 0).astype(int)
        tl["side"] = np.select([tl.shares > 0, tl.shares < 0], ["BUY", "SELL"], "")
        tl["reason"] = np.select([(tl.current_wt == 0) & (tl.target_wt > 0), (tl.current_wt > 0) & (tl.target_wt == 0)], ["enters sample", "leaves sample"], "reweight")
        tl = tl[tl.shares != 0].sort_values("trade_usd", key=np.abs, ascending=False)
        tl.to_csv(out_dir / "rebalance_trades.csv", index_label="ticker")
        n_trades, traded = int(len(tl)), float(tl.trade_usd.abs().sum())
    s80 = S.set_index("key").loc["sampled_80"]
    man = dict(built=str(date.today()), benchmark=INDEX_NAME, first=str(lg.formation.min()), last=str(lg.formation.max()), rebalances=int(len(lg)),
               names_index=int(round(lg.constituents.mean())), names_held=80, samples=SAMPLES, te_realized=float(s80.tracking_error),
               te_predicted=float(s80.pred_te_avg), turnover=float(s80.turnover_yr), tracking_difference=float(s80.tracking_difference),
               nav=NAV, cash_target=CASH_TARGET, cost_bps=COST_BPS, max_weight=MAX_WEIGHT, latest_trades=n_trades, latest_traded_usd=traded,
               dpsw_date=ws["date"])
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    log(f"wrote {out_dir}")
    return man
