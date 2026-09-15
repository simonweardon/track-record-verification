"""Portfolio construction as a linear program, with a trade list — the daily work of an
internal quant portfolio.

Given an alpha score per stock, a benchmark, sector labels and the portfolio as it
stands, choose target weights that maximise expected alpha net of a linear trading
cost, subject to the constraints a mandate actually carries:

    long-only, fully invested          0 ≤ w_i,  Σ w_i = 1
    name cap                           w_i ≤ max(max_weight, b_i)   (never forced below benchmark by the cap)
    active-weight band                 |w_i − b_i| ≤ active_band
    sector bands vs benchmark          |Σ_{i∈s} w_i − Σ_{i∈s} b_i| ≤ sector_band
    active share cap                   Σ |w_i − b_i| / 2 ≤ active_share   (the linear stand-in for a TE limit)
    turnover budget                    Σ |w_i − w0_i| / 2 ≤ turnover
    names leaving the universe         forced to zero (they still cost turnover)

Everything is linear (w = w0 + buy − sell and w = b + over − under, all ≥ 0), so the same problem
solves in scipy's HiGHS here and in R's Rglpk (r/construct.R) — the two are checked
against each other.  Tracking error is quadratic and so is reported ex ante from a
shrunk covariance, not constrained; the active and sector bands are its linear proxy,
which is how index-relative mandates are usually run in practice.

Demo mandate (data/research/construction/): universe = US names held by ≥ 5 of the
managers in the system each quarter; benchmark = the aggregate disclosed book,
dollar-weighted ("what the managers own, in proportion"); alpha = 12-1 month price
momentum z-score — a standard, transparent signal, chosen because it is available for
every name from the price panel, not because it is good.  Rebalanced at each 13F
formation date, held with drift.  The optimizer does not care where the alphas come
from; swap the signal and the rest is unchanged.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .signals13f import clean_returns, load_books, load_factors, load_prices, OUT_DIR as SIG_DIR
from .compact import COMPACT

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "research" / "construction"
R_SCRIPT = ROOT / "r" / "construct.R"
MIN_HOLDERS = 5
NAV = 100_000_000.0


@dataclass
class Constraints:
    max_weight: float = 0.04      # absolute cap per name
    active_band: float = 0.02     # |w − b| per name
    sector_band: float = 0.03     # |sector weight − benchmark sector weight|
    active_share: float = 0.30    # Σ|w − b| / 2 — without it an LP corners into ~50 names at +band each
    turnover: float = 0.20        # one-way, per rebalance
    cost_bps: float = 10.0        # linear cost per dollar traded (in the objective and in the realized returns)


# ---------------------------------------------------------------- the LP

def solve_lp(alpha: pd.Series, bench: pd.Series, sectors: pd.Series, w0: pd.Series, c: Constraints) -> tuple[pd.Series, dict]:
    """Target weights and solver diagnostics.  Names in w0 but not in alpha are forced out."""
    names = list(dict.fromkeys(list(alpha.index) + [t for t in w0.index if w0[t] > 0]))
    n = len(names)
    a = alpha.reindex(names).fillna(0.0).values
    b = bench.reindex(names).fillna(0.0).values
    w0v = w0.reindex(names).fillna(0.0).values
    inuni = np.array([t in alpha.index for t in names])
    cost = c.cost_bps / 1e4
    # variables x = [w, buy, sell, over, under] (n each); minimise −α'w + cost·Σ(buy + sell)
    Z = np.zeros(n); O = np.ones(n); I = np.eye(n); ZZ = np.zeros((n, n))
    obj = np.concatenate([-a, np.full(n, cost), np.full(n, cost), Z, Z])
    A_eq = np.vstack([np.concatenate([O, Z, Z, Z, Z]),              # Σw = 1
                      np.hstack([I, -I, I, ZZ, ZZ]),                # w − buy + sell = w0
                      np.hstack([I, ZZ, ZZ, -I, I])])               # w − over + under = b
    b_eq = np.concatenate([[1.0], w0v, b])
    rows, rhs = [], []
    rows.append(np.concatenate([Z, O, O, Z, Z])); rhs.append(2 * c.turnover)          # Σ(buy+sell) ≤ 2·turnover
    rows.append(np.concatenate([Z, Z, Z, O, O])); rhs.append(2 * c.active_share)      # Σ(over+under) ≤ 2·active share
    sec = sectors.reindex(names).fillna("Unknown")
    for s in sorted(sec.unique()):
        m = (sec == s).values.astype(float); bs = float(b[m > 0].sum())
        rows.append(np.concatenate([m, Z, Z, Z, Z])); rhs.append(bs + c.sector_band)
        rows.append(np.concatenate([-m, Z, Z, Z, Z])); rhs.append(-(bs - c.sector_band))
    A_ub = np.vstack(rows); b_ub = np.array(rhs)
    lo = np.where(inuni, np.maximum(0.0, b - c.active_band), 0.0)
    hi = np.where(inuni, np.minimum(np.maximum(c.max_weight, b), b + c.active_band), 0.0)   # a mega-cap above the cap may be held at benchmark weight
    bounds = [(float(l), float(h)) for l, h in zip(lo, hi)] + [(0, None)] * (4 * n)
    tries, T = 0, c.turnover
    while True:
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.status == 0 or tries >= 3:
            break
        tries += 1; T *= 2; b_ub[0] = 2 * T                     # infeasible (e.g. sector drift > budget): relax turnover, say so
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    w = pd.Series(res.x[:n], index=names).clip(lower=0)
    w[w < 1e-9] = 0.0; w = w / w.sum()
    turnover = float((w - pd.Series(w0v, index=names)).abs().sum() / 2)
    active_share = float((w - pd.Series(b, index=names)).abs().sum() / 2)
    info = dict(status=res.message, n_vars=5 * n, active_share=active_share, n_constraints=int(A_ub.shape[0] + A_eq.shape[0]), names=int((w > 0).sum()),
                turnover=turnover, turnover_budget=T, relaxed=tries, expected_alpha=float((w.values * a).sum()),
                bench_alpha=float((b * a).sum()), objective=float(-res.fun))
    return w, info


def shrunk_cov(rets: pd.DataFrame, delta: float = 0.5) -> pd.DataFrame:
    """Sample covariance shrunk halfway toward constant correlation (a Ledoit–Wolf-style target)."""
    r = rets.dropna(axis=1, thresh=int(len(rets) * 0.8)).fillna(0.0)
    S = r.cov().values; sd = np.sqrt(np.diag(S)); n = len(sd)
    corr = S / np.outer(sd, sd); rho = (corr.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    F = rho * np.outer(sd, sd); np.fill_diagonal(F, sd ** 2)
    return pd.DataFrame(delta * F + (1 - delta) * S, index=r.columns, columns=r.columns)


def ex_ante_te(w: pd.Series, b: pd.Series, cov: pd.DataFrame) -> float:
    names = [t for t in set(w.index) | set(b.index) if t in cov.index]
    a = (w.reindex(names).fillna(0) - b.reindex(names).fillna(0)).values
    return float(np.sqrt(max(a @ cov.loc[names, names].values @ a, 0.0) * 12))


def trade_list(w: pd.Series, w0: pd.Series, prices: pd.Series, names: dict, sectors: pd.Series, alpha: pd.Series, bench: pd.Series,
               nav: float = NAV, cost_bps: float = 10.0) -> pd.DataFrame:
    tick = sorted(set(w.index) | set(w0.index))
    df = pd.DataFrame(index=tick)
    df["name"] = [names.get(t, t) for t in tick]; df["sector"] = sectors.reindex(tick).fillna("Unknown").values
    df["current_wt"] = w0.reindex(tick).fillna(0).values; df["target_wt"] = w.reindex(tick).fillna(0).values
    df["benchmark_wt"] = bench.reindex(tick).fillna(0).values; df["active_wt"] = df.target_wt - df.benchmark_wt
    df["alpha_z"] = alpha.reindex(tick).values
    df["price"] = prices.reindex(tick).values
    df["trade_usd"] = (df.target_wt - df.current_wt) * nav
    df["shares"] = np.where(df.price > 0, np.round(df.trade_usd / df.price), 0).astype(int)
    df["side"] = np.select([df.shares > 0, df.shares < 0], ["BUY", "SELL"], "")
    df["est_cost_usd"] = df.trade_usd.abs() * cost_bps / 1e4
    df = df[(df.shares != 0) | (df.target_wt > 0)].sort_values("trade_usd", key=np.abs, ascending=False)
    return df.reset_index().rename(columns={"index": "ticker"})


# ---------------------------------------------------------------- the demo mandate

def momentum_alpha(prices: pd.DataFrame, F: pd.Timestamp, universe: list[str]) -> pd.Series:
    """12-1 momentum: return from 12 months before F to 1 month before F, z-scored across the universe."""
    idx = prices.index
    i = idx.get_loc(F)
    if i < 12:
        return pd.Series(dtype=float)
    p1, p12 = prices.iloc[i - 1], prices.iloc[i - 12]
    m = (p1 / p12 - 1).reindex(universe).dropna()
    m = m[np.isfinite(m)]
    z = (m - m.mean()) / m.std() if m.std() > 0 else m * 0
    return z.clip(-3, 3)


def load_sectors() -> tuple[pd.Series, pd.Series]:
    f = COMPACT / "sectors.csv"
    if not f.exists():
        return pd.Series(dtype=str), pd.Series(dtype=str)
    s = pd.read_csv(f, dtype=str).fillna("")
    s = s[s.ticker != ""]
    return s.set_index("ticker").sector.replace("", "Unknown"), s.set_index("ticker").industry


def backtest(c: Constraints = Constraints(), out_dir: Path = OUT_DIR, log=print) -> dict:
    books = load_books(log=log); prices = load_prices(); fac = load_factors()
    rets = clean_returns(prices)
    sectors, industries = load_sectors()
    names = {}
    for t, nm in zip(books.ticker, books["name"]):
        if isinstance(t, str) and t not in names: names[t] = str(nm).title()
    b = books[books.ticker.notna() & books.ticker.isin(prices.columns)]
    forms = sorted(f for f in b.formation.unique() if f in prices.index)
    forms = [F for F in forms if prices.index.get_loc(F) >= 12]
    months = prices.index[prices.index > forms[0]]
    R = pd.DataFrame(np.nan, index=months, columns=["portfolio", "benchmark", "unconstrained"])
    reb = []; cur_p = None; cur_b = None; cur_u = None; latest = None
    for k, F in enumerate(forms):
        end = forms[k + 1] if k + 1 < len(forms) else months[-1]
        q = b[b.formation == F]
        held = q.groupby("ticker").agg(n=("slug", "nunique"), value=("value", "sum"))
        held = held[(held.n >= MIN_HOLDERS) & (prices.loc[F].reindex(held.index) >= 1.0)]
        uni = list(held.index)
        alpha = momentum_alpha(prices, F, uni)
        uni = list(alpha.index)
        if len(uni) < 30:
            continue
        bench = held.loc[uni, "value"] / held.loc[uni, "value"].sum()
        sec = sectors.reindex(uni).fillna("Unknown")
        w0 = cur_p if cur_p is not None else bench.copy()
        w, info = solve_lp(alpha, bench, sec, w0, c)
        # unconstrained comparison: equal-weight top decile of the same alpha
        top = alpha.sort_values(ascending=False).head(max(10, len(alpha) // 10))
        wu = pd.Series(1.0 / len(top), index=top.index)
        tu = float((wu.subtract(cur_u, fill_value=0)).abs().sum() / 2) if cur_u is not None else 1.0
        cov = shrunk_cov(rets.loc[:F].tail(36)[[t for t in uni if t in rets.columns]])
        te = ex_ante_te(w, bench, cov)
        act = (w.reindex(uni).fillna(0) - bench)
        sec_act = act.groupby(sec).sum()
        row = dict(formation=F.date(), universe=len(uni), names=info["names"], turnover=info["turnover"], turnover_budget=info["turnover_budget"],
                   active_share=info["active_share"],
                   relaxed=info["relaxed"], cost_drag=info["turnover"] * 2 * c.cost_bps / 1e4, ex_ante_te=te,
                   expected_alpha=info["expected_alpha"], bench_alpha=info["bench_alpha"], max_active=float(act.abs().max()),
                   max_sector_active=float(sec_act.abs().max()), sum_abs_active=float(act.abs().sum()), n_constraints=info["n_constraints"],
                   unconstrained_names=len(wu), unconstrained_turnover=tu)
        reb.append(row)
        # hold with drift; cost charged in the first month after each rebalance
        def run(wt, key, cost_first):
            cur = wt[wt > 0].copy(); first = True
            for m in months[(months > F) & (months <= end)]:
                r = rets.loc[m, cur.index]; ok = r.notna()
                if ok.sum() == 0: break
                wk = cur[ok] / cur[ok].sum()
                R.loc[m, key] = float((wk * r[ok]).sum()) - (cost_first if first else 0.0)
                cur = cur[ok] * (1 + r[ok]); cur = cur / cur.sum(); first = False
            return cur
        cur_p = run(w, "portfolio", info["turnover"] * 2 * c.cost_bps / 1e4)
        cur_b = run(bench, "benchmark", 0.0)
        cur_u = run(wu, "unconstrained", tu * 2 * c.cost_bps / 1e4)
        latest = dict(F=F, w=w, w0=w0, bench=bench, alpha=alpha, sec=sec, info=info, te=te)
        log(f"  {F.date()}: universe {len(uni)}, held {info['names']}, turnover {info['turnover']:.0%}, ex-ante TE {te:.1%}"
            + (f" (turnover relaxed ×{2 ** info['relaxed']})" if info["relaxed"] else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    R = R.join(fac[["MKT", "RF"]])
    R.to_csv(out_dir / "backtest_monthly.csv", index_label="month")
    rb = pd.DataFrame(reb); rb.to_csv(out_dir / "rebalances.csv", index=False)
    summ = _summary(R, rb, c); summ.to_csv(out_dir / "summary.csv", index=False)
    # latest rebalance: target, trade list, and the inputs for the R twin
    F = latest["F"]
    tl = trade_list(latest["w"], latest["w0"], prices.loc[F], names, sectors, latest["alpha"], latest["bench"], NAV, c.cost_bps)
    tl.to_csv(out_dir / "trade_list.csv", index=False)
    sec_tbl = pd.DataFrame({"portfolio": latest["w"].groupby(latest["sec"].reindex(latest["w"].index).fillna("Unknown")).sum(),
                            "benchmark": latest["bench"].groupby(latest["sec"]).sum()}).fillna(0)
    sec_tbl["active"] = sec_tbl.portfolio - sec_tbl.benchmark
    sec_tbl.sort_values("benchmark", ascending=False).to_csv(out_dir / "sectors_latest.csv", index_label="sector")
    inp = out_dir / "inputs_latest"; inp.mkdir(exist_ok=True)
    pd.DataFrame({"ticker": latest["alpha"].index, "alpha": latest["alpha"].values, "bench": latest["bench"].reindex(latest["alpha"].index).values,
                  "sector": latest["sec"].reindex(latest["alpha"].index).values, "w0": latest["w0"].reindex(latest["alpha"].index).fillna(0).values}).to_csv(inp / "universe.csv", index=False)
    leaving = latest["w0"][(latest["w0"] > 0) & ~latest["w0"].index.isin(latest["alpha"].index)]
    pd.DataFrame({"ticker": leaving.index, "w0": leaving.values}).to_csv(inp / "leaving.csv", index=False)
    (inp / "constraints.json").write_text(json.dumps(asdict(c), indent=1))
    pd.DataFrame({"key": list(asdict(c)), "value": list(asdict(c).values())}).to_csv(inp / "constraints.csv", index=False)
    pd.DataFrame({"ticker": latest["w"].index, "weight": latest["w"].values}).to_csv(inp / "weights_python.csv", index=False)
    r_check = compare_with_r(inp, log)
    man = dict(built=str(date.today()), first=str(rb.formation.min()), last=str(rb.formation.max()), rebalances=len(rb), nav=NAV,
               constraints=asdict(c), min_holders=MIN_HOLDERS, latest=dict(formation=str(F.date()), **{k: v for k, v in latest["info"].items() if k != "status"},
               ex_ante_te=latest["te"], trades=int((tl.shares != 0).sum()), buys=int((tl.side == "BUY").sum()), sells=int((tl.side == "SELL").sum()),
               traded_usd=float(tl.trade_usd.abs().sum()), est_cost_usd=float(tl.est_cost_usd.sum())), r_twin=r_check,
               sectors_known=int((latest["sec"] != "Unknown").sum()), sectors_unknown=int((latest["sec"] == "Unknown").sum()))
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    log(f"wrote {out_dir}")
    return man


def _summary(R: pd.DataFrame, rb: pd.DataFrame, c: Constraints) -> pd.DataFrame:
    rows = []
    ok = R.portfolio.notna()
    for key, label in [("portfolio", "Constrained LP portfolio"), ("unconstrained", "Unconstrained top decile"), ("benchmark", "Benchmark (aggregate book)"), ("MKT", "US market")]:
        r = R[key][ok]
        ann = float((1 + r).prod() ** (12 / len(r)) - 1)
        ex = r - R.RF[ok]
        active = r - R.benchmark[ok]
        te = float(active.std() * np.sqrt(12)); ir = float(active.mean() * 12 / te) if te > 0 else np.nan
        cum = (1 + r).cumprod(); dd = float((cum / cum.cummax() - 1).min())
        rows.append(dict(key=key, label=label, months=int(len(r)), ann_return=ann, ann_vol=float(r.std() * np.sqrt(12)),
                         sharpe=float(ex.mean() / ex.std() * np.sqrt(12)), max_dd=dd,
                         active_return=float(active.mean() * 12) if key != "benchmark" else 0.0, tracking_error=te if key != "benchmark" else 0.0,
                         information_ratio=ir if key != "benchmark" else np.nan, hit_rate=float((active > 0).mean()) if key != "benchmark" else np.nan,
                         avg_names=float(rb.names.mean()) if key == "portfolio" else (float(rb.unconstrained_names.mean()) if key == "unconstrained" else np.nan),
                         avg_turnover=float(rb.turnover.mean()) if key == "portfolio" else (float(rb.unconstrained_turnover.mean()) if key == "unconstrained" else np.nan),
                         avg_ex_ante_te=float(rb.ex_ante_te.mean()) if key == "portfolio" else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- R twin

def compare_with_r(inp: Path, log=print) -> dict:
    """Run r/construct.R on the same inputs if Rscript + Rglpk are available; report the agreement."""
    rs = shutil.which("Rscript")
    if not rs or not R_SCRIPT.exists():
        return dict(available=False, reason="Rscript not found" if not rs else "r/construct.R missing")
    try:
        p = subprocess.run([rs, str(R_SCRIPT), str(inp)], capture_output=True, text=True, timeout=300)
    except Exception as e:
        return dict(available=False, reason=str(e)[:200])
    if p.returncode != 0:
        return dict(available=False, reason=(p.stderr or p.stdout)[-400:])
    rw = pd.read_csv(inp / "weights_r.csv").set_index("ticker").weight
    pw = pd.read_csv(inp / "weights_python.csv").set_index("ticker").weight
    both = rw.index.union(pw.index)
    d = (rw.reindex(both).fillna(0) - pw.reindex(both).fillna(0)).abs()
    u = pd.read_csv(inp / "universe.csv").set_index("ticker")
    obj_r = float((rw.reindex(u.index).fillna(0) * u.alpha).sum()); obj_p = float((pw.reindex(u.index).fillna(0) * u.alpha).sum())
    out = dict(available=True, max_abs_diff=float(d.max()), sum_abs_diff=float(d.sum()), names_r=int((rw > 0).sum()), names_python=int((pw > 0).sum()),
               expected_alpha_r=obj_r, expected_alpha_python=obj_p, r_stdout=p.stdout.strip()[-300:])
    log(f"  R twin (Rglpk): max |Δw| {out['max_abs_diff']:.2e}, Σ|Δw| {out['sum_abs_diff']:.2e}, alpha R {obj_r:.4f} vs Python {obj_p:.4f}")
    return out
