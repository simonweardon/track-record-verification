"""The live passive book: refresh prices, recompute the daily sheet, persist it.

The index tracker's backtest ends at the last complete month.  This module carries the
80-name book (and the full index) forward to today with daily prices from Yahoo, and
recomputes everything on the Daily Portfolio Status Worksheet: NAV, cash versus target,
MTD / QTD / YTD versus the index, predicted tracking error (factor versus specific) under
the risk model estimated at the last reconstitution, sector and name actives, the holdings
with shares, the cash-flow slice at today's prices, and the trades that would bring the
drifted book back to target.  It runs on a schedule inside the server (daily by default)
and on demand; results are written to output/live/passive/ with a timestamp.

What it still lacks versus a real desk, stated on the page: corporate-action notices
(dividends, splits, spin-offs, tenders) and the index vendor's add/delete calendar.  A name
that stops pricing shows up as an event to resolve; nothing is interpolated.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data" / "research" / "index-tracker" / "live_inputs"
OUT_DIR = ROOT / "output" / "live" / "passive"
BT = ROOT / "data" / "research" / "index-tracker" / "backtest_monthly.csv"
STATE = {"last_run": 0.0, "running": False, "error": None}


def _daily_prices(tickers: list[str], start: pd.Timestamp, log=print) -> pd.DataFrame:
    import yfinance as yf
    frames = []
    for i in range(0, len(tickers), 200):
        chunk = tickers[i:i + 200]
        try:
            df = yf.download(chunk, start=str(start.date()), interval="1d", auto_adjust=True, progress=False, threads=True, group_by="column")
        except Exception as e:
            log(f"  live: chunk failed: {e}"); continue
        close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": chunk[0]})
        close.index = pd.DatetimeIndex(close.index).tz_localize(None)
        frames.append(close)
    if not frames:
        return pd.DataFrame()
    P = pd.concat(frames, axis=1)
    return P.loc[:, ~P.columns.duplicated()].sort_index()


def refresh(log=print) -> dict | None:
    if not (IN_DIR / "manifest.json").exists():
        return None
    from .riskmodel import decompose
    man = json.loads((IN_DIR / "manifest.json").read_text())
    last_month = pd.Timestamp(man["last_month"]); F = pd.Timestamp(man["formation"]); nav0 = float(man["nav"]); cash_t = float(man["cash_target"])
    b80 = pd.read_csv(IN_DIR / "book_80.csv").set_index("ticker"); bidx = pd.read_csv(IN_DIR / "book_index.csv").set_index("ticker")
    X = pd.read_csv(IN_DIR / "exposures.csv", index_col=0); Fc = pd.read_csv(IN_DIR / "factor_cov.csv", index_col=0)
    D = pd.read_csv(IN_DIR / "specific_var.csv").set_index("ticker").specific_var; sectors = pd.read_csv(IN_DIR / "sectors.csv").set_index("ticker").sector
    tickers = sorted(set(b80.index) | set(bidx.index))
    P = _daily_prices(tickers, last_month - pd.Timedelta(days=10), log)
    if P.empty:
        raise RuntimeError("no prices returned")
    base_rows = P[P.index <= last_month]
    if base_rows.empty:
        raise RuntimeError("no price on or before the last month-end")
    base = base_rows.iloc[-1]; today = P.iloc[-1]; price_date = P.index[-1]
    r = (today / base - 1).reindex(tickers)
    dead = [t for t in b80.index if pd.isna(today.get(t)) or today.get(t, 0) <= 0]
    r = r.fillna(0.0)
    # drift both books from the month-end
    def drift(w):
        g = w * (1 + r.reindex(w.index).fillna(0.0)); ret = float(g.sum() - w.sum()) / float(w.sum()); return g / g.sum(), ret
    w80, ret80 = drift(b80.weight); widx, retidx = drift(bidx.weight)
    # completed months since formation / year start / month, from the backtest series
    R = pd.read_csv(BT, index_col=0, parse_dates=True)[["index", "sampled_80"]].dropna()
    def growth(since):
        s = R[R.index > since]
        return float((1 + s.sampled_80).prod()), float((1 + s["index"]).prod())
    gq = growth(F); gy = growth(pd.Timestamp(year=price_date.year, month=1, day=1) - pd.Timedelta(days=1))
    mtd = (ret80, retidx); qtd = (gq[0] * (1 + ret80) - 1, gq[1] * (1 + retidx) - 1); ytd = (gy[0] * (1 + ret80) - 1, gy[1] * (1 + retidx) - 1)
    nav = nav0 * gq[0] * (1 + ret80)
    cash = nav * cash_t
    names = w80.index.union(widx.index)
    active = w80.reindex(names).fillna(0) - widx.reindex(names).fillna(0)
    styles = man.get("styles", [])
    d = decompose(active, X, Fc, D, styles)
    sec_act = active.groupby(sectors.reindex(active.index).fillna("Unknown")).sum().sort_values()
    target = b80.target.reindex(w80.index).fillna(0)
    hold = pd.DataFrame({"weight": w80, "target": target, "index_weight": widx.reindex(w80.index).fillna(0)})
    hold["active"] = hold.weight - hold.index_weight; hold["market_value"] = hold.weight * (nav - cash)
    hold["price"] = today.reindex(hold.index); hold["shares"] = np.floor(hold.market_value / hold.price.where(hold.price > 0))
    hold["sector"] = sectors.reindex(hold.index).fillna("Unknown"); hold["day_return_since_month_end"] = r.reindex(hold.index)
    hold = hold.sort_values("weight", ascending=False)
    inflow = 0.02 * nav
    slc = pd.DataFrame({"weight": target[target > 0], "usd": target[target > 0] * inflow}); slc["price"] = today.reindex(slc.index)
    slc["shares"] = np.floor(slc.usd / slc.price.where(slc.price > 0)); slc["sector"] = sectors.reindex(slc.index).fillna("Unknown")
    # trades to return the drifted book to target, at today's prices
    tl = pd.DataFrame({"current_wt": w80, "target_wt": target}); tl["trade_usd"] = (tl.target_wt - tl.current_wt) * (nav - cash)
    tl["price"] = today.reindex(tl.index); tl["shares"] = np.where(tl.price > 0, np.round(tl.trade_usd / tl.price), 0).astype(int)
    tl["side"] = np.select([tl.shares > 0, tl.shares < 0], ["BUY", "SELL"], ""); tl["sector"] = sectors.reindex(tl.index).fillna("Unknown")
    tl = tl[tl.shares != 0].sort_values("trade_usd", key=np.abs, ascending=False)
    drift_to = float((w80 - target).abs().sum() / 2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hold.to_csv(OUT_DIR / "dpsw_holdings_live.csv", index_label="ticker"); slc.to_csv(OUT_DIR / "dpsw_cashflow_slice_live.csv", index_label="ticker")
    tl.to_csv(OUT_DIR / "drift_trades_live.csv", index_label="ticker")
    out = dict(updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), price_date=str(price_date.date()), last_month=str(last_month.date()), formation=str(F.date()),
               nav=nav, cash=cash, cash_pct=cash_t, n_holdings=int((w80 > 0).sum()), n_index=int((widx > 0).sum()), mtd=mtd, qtd=qtd, ytd=ytd,
               pred_te=d["total"], pred_te_factor=d["factor"], pred_te_specific=d["specific"], max_active=float(active.abs().max()), max_active_name=str(active.abs().idxmax()),
               sector_active=sec_act.to_dict(), largest_active=active.reindex(active.abs().sort_values(ascending=False).index).head(10).to_dict(),
               top_risk=d["top"][:6], dead=dead, drift_max=float((w80 - target).abs().max()), drift_max_name=str((w80 - target).abs().idxmax()), drift_total=drift_to,
               drift_trades=int(len(tl)), drift_traded_usd=float(tl.trade_usd.abs().sum()), inflow=inflow, priced=int(r.reindex(b80.index).notna().sum()))
    (OUT_DIR / "dpsw_live.json").write_text(json.dumps(out, indent=1, default=float))
    STATE["last_run"] = time.time(); STATE["error"] = None
    log(f"live book refreshed: prices {price_date.date()}, NAV ${nav / 1e6:,.1f}m, MTD {mtd[0]:+.2%} vs {mtd[1]:+.2%}, pred TE {d['total']:.2%}, {len(dead)} unpriced")
    return out


def load() -> dict | None:
    f = OUT_DIR / "dpsw_live.json"
    return json.loads(f.read_text()) if f.exists() else None


def loop(hours: float, log=print) -> None:
    """Run in a daemon thread: refresh now, then every `hours`."""
    while True:
        try:
            STATE["running"] = True
            refresh(log)
        except Exception as e:
            STATE["error"] = str(e); log(f"live book refresh failed: {e}")
        finally:
            STATE["running"] = False
        time.sleep(max(600, hours * 3600))
