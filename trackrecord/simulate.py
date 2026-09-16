"""Simulation: pick managers, compare them side by side, blend them into a portfolio, and run
that portfolio through the same pipeline as any single manager — dashboard, scores and memo.

A simulation is a spec: managers with weights, a fee schedule applied to each manager on its
own gains above its own high-water mark, and a rebalancing rule (monthly back to target, or
buy-and-hold drift).  The blended monthly series is written as a one-account dataset in the
statement format and `report --grid M` is run on it, so the simulated portfolio gets exactly
the Manager Verification dashboard, the two fixed scores and the Due Diligence Memo — nothing
is re-implemented.  The fee breakdown (gross, net of manager fees, fee drag by manager) is
computed here and shown on the simulation page; the dashboard's own model-net tile applies
the same schedule at the portfolio level, which is stated next to it.

Specs are hashed so an identical pick reuses its output.  Datasets: data/sims/<id>/; outputs:
output/sims/<id>/.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .returns import FeeSchedule, model_net

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
SIMS = ROOT / "data" / "sims"
MIN_MONTHS = 36
NOT_MEANINGFUL = {"multi", "macro", "mm"}


@dataclass
class Spec:
    managers: list[tuple[str, float]]            # (key, weight) — key is a fund slug or a ticker
    mgmt: float = 0.0                            # annual management fee applied to each manager
    perf: float = 0.0                            # performance fee on each manager's gains above its HWM
    rebalance: str = "monthly"                   # "monthly" | "drift"
    name: str = ""

    def normalized(self) -> "Spec":
        w = np.array([max(0.0, x) for _, x in self.managers], float)
        if w.sum() <= 0:
            w = np.ones(len(w))
        w = w / w.sum()
        return Spec([(k, float(x)) for (k, _), x in zip(self.managers, w)], self.mgmt, self.perf, self.rebalance, self.name)

    def sim_id(self) -> str:
        s = self.normalized()
        payload = json.dumps(dict(m=sorted((k, round(x, 6)) for k, x in s.managers), mgmt=round(s.mgmt, 6), perf=round(s.perf, 6), reb=s.rebalance), sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:10]


# ---------------------------------------------------------------- candidates

def candidates() -> list[dict]:
    """Every scorable, meaningful manager and listed fund with a built output (monthly grid)."""
    out = []
    lb = ROOT / "data" / "funds" / "leaderboard.csv"
    if lb.exists():
        for r in pd.read_csv(lb).to_dict("records"):
            if r.get("status") == "ok" and r.get("style") not in NOT_MEANINGFUL:
                out.append(dict(key=r["slug"], kind="fund", name=r["name"], manager=r.get("manager", ""), style=r.get("style_name", ""), months=int(r.get("months") or 0),
                                ff3_t=float(r["ff3_t"]) if pd.notna(r.get("ff3_t")) else None, built=(OUT / "funds" / r["slug"] / "phase2" / "composite_returns.csv").exists()))
    tdir = ROOT / "data" / "tickers"
    if tdir.exists():
        for d in sorted(tdir.iterdir()):
            m = d / "meta.json"
            if m.exists():
                meta = json.loads(m.read_text())
                out.append(dict(key=d.name, kind="ticker", name=meta.get("name", d.name), manager="", style="Listed fund", months=int(meta.get("rows") or 0),
                                ff3_t=None, built=(OUT / "t" / d.name / "phase2" / "composite_returns.csv").exists()))
    return out


def out_dir_for(key: str) -> Path:
    return OUT / ("t" if (ROOT / "data" / "tickers" / key).exists() else "funds") / key


def gross_series(key: str) -> pd.Series | None:
    f = out_dir_for(key) / "phase2" / "composite_returns.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if not str(d.cell.iloc[0]).count("-"):
        return None                                                  # annual grid: not blendable monthly
    s = pd.Series(d.gross.astype(float).values, index=pd.PeriodIndex(d.cell, freq="M").to_timestamp("M"))
    return s.dropna()


def stats_of(key: str) -> dict:
    """Headline numbers for the side-by-side table, read from the manager's own outputs."""
    o = out_dir_for(key); out = dict(key=key)
    m = o / "phase4" / "metrics.csv"
    if m.exists():
        mm = pd.read_csv(m).set_index("metric").portfolio
        for k, col in [("annualized return", "ret"), ("annualized volatility", "vol"), ("Sharpe (excess over RF)", "sharpe"),
                       ("max drawdown (cell-level; understated on annual cells)", "max_dd"), ("down capture", "down_cap"), ("tracking error (ann.)", "te"), ("information ratio", "ir")]:
            try: out[col] = float(mm[k])
            except Exception: out[col] = np.nan
        try: out["bench_ret"] = float(pd.read_csv(m).set_index("metric").benchmark["annualized return"])
        except Exception: out["bench_ret"] = np.nan
    r = o / "phase4" / "regressions.csv"
    if r.exists():
        rr = pd.read_csv(r); ff3 = rr[(rr.factor_set == "US") & (rr.model == "FF3")]
        if len(ff3):
            x = ff3.iloc[0]; out.update(alpha=float(x.alpha_annual), t=float(x.t), beta=float(x.b_MKT_RF), smb=float(x.b_SMB), hml=float(x.b_HML), n=int(x.n))
    s = o / "phase4" / "scores.csv"
    if s.exists():
        ss = pd.read_csv(s)
        for _, row in ss.iterrows():
            if row.score == "Alpha-maxing score": out["alpha_max"] = float(row.value)
            if row.score == "Wealth-management score" and row.component == "TOTAL": out["wealth"] = float(row.value)
    return out


def residual_correlation(keys: list[str]) -> pd.DataFrame:
    """Correlation of FF3 residuals between the picked managers, on overlapping months."""
    import statsmodels.api as sm
    res = {}
    for k in keys:
        f = out_dir_for(k) / "phase4" / "aligned_data.csv"
        if not f.exists(): continue
        d = pd.read_csv(f).dropna(subset=["y", "MKT_RF", "SMB", "HML"])
        if len(d) < 24 or not str(d.cell.iloc[0]).count("-"): continue
        fit = sm.OLS(d.y.values, sm.add_constant(d[["MKT_RF", "SMB", "HML"]].values)).fit()
        res[k] = pd.Series(fit.resid, index=pd.PeriodIndex(d.cell, freq="M").to_timestamp("M"))
    return pd.DataFrame(res).corr(min_periods=24)


# ---------------------------------------------------------------- blending

def blend(spec: Spec) -> dict:
    spec = spec.normalized()
    series = {k: gross_series(k) for k, _ in spec.managers}
    missing = [k for k, s in series.items() if s is None]
    if missing:
        raise ValueError(f"not built: {', '.join(missing)}")
    idx = None
    for s in series.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    idx = idx.sort_values()
    if len(idx) < MIN_MONTHS:
        raise ValueError(f"only {len(idx)} months in common (need {MIN_MONTHS})")
    G = pd.DataFrame({k: s.reindex(idx) for k, s in series.items()})
    fee = FeeSchedule(mgmt_pct=spec.mgmt, perf_pct=spec.perf)
    N = pd.DataFrame({k: model_net(G[k], pd.Series(1 / 12, index=idx), fee) for k in G.columns}) if (spec.mgmt or spec.perf) else G.copy()
    w = pd.Series(dict(spec.managers))
    def combine(R: pd.DataFrame) -> pd.Series:
        if spec.rebalance == "monthly":
            return (R * w).sum(axis=1)
        nav = (1 + R).cumprod() * w                                  # buy and hold: each sleeve grows from its starting weight
        tot = nav.sum(axis=1)
        return tot.pct_change().fillna(tot.iloc[0] - 1.0)
    gross = combine(G); net = combine(N)
    ann = lambda s: float((1 + s).prod() ** (12 / len(s)) - 1)
    per = []
    for k in G.columns:
        per.append(dict(key=k, weight=float(w[k]), gross=ann(G[k]), net=ann(N[k]), fee_drag=ann(G[k]) - ann(N[k]), vol=float(G[k].std() * np.sqrt(12)),
                        contribution_gross=float((w[k] * G[k]).mean() * 12) if spec.rebalance == "monthly" else np.nan))
    return dict(spec=spec, months=len(idx), first=str(idx.min().date()), last=str(idx.max().date()), gross=gross, net=net,
                ann_gross=ann(gross), ann_net=ann(net), fee_drag=ann(gross) - ann(net), per_manager=per, fee=fee.describe() if (spec.mgmt or spec.perf) else "no fees",
                corr=G.corr())


# ---------------------------------------------------------------- dataset

def write_dataset(b: dict, cands: dict[str, dict]) -> Path:
    """The blended gross series as a one-account statement dataset (the pipeline applies the fee at portfolio level)."""
    spec = b["spec"]; sid = spec.sim_id(); d = SIMS / sid; d.mkdir(parents=True, exist_ok=True)
    acct = "SIM_" + sid.upper()
    val = 1_000_000.0; rows = []; g = b["gross"]
    first = (g.index[0].to_period("M") - 1).to_timestamp("M")          # deposit at the month-end before the first return month
    rows.append(dict(statement_id=f"{acct}_{first.to_period('M')}", account_id=acct, custodian="Simulation (blend of built manager series)",
                     period_start=first.to_period("M").start_time.date(), period_end=first.date(), ending_value=round(val, 2), source_file="simulate", source_pages="",
                     notes="notional $1m at the month-end before the first common month"))
    flows = [dict(account_id=acct, date=first.date(), amount=round(val, 2), flow_type="deposit", description="notional $1m", source_statement_id=f"{acct}_{first.to_period('M')}")]
    for m in g.index:
        val *= (1 + float(g[m]))
        rows.append(dict(statement_id=f"{acct}_{m.to_period('M')}", account_id=acct, custodian="Simulation (blend of built manager series)",
                         period_start=m.to_period("M").start_time.date(), period_end=m.date(), ending_value=round(val, 2), source_file="simulate", source_pages="", notes=""))
    cols = ["statement_id", "account_id", "custodian", "period_start", "period_end", "ending_value", "beginning_value", "stated_deposits", "stated_withdrawals",
            "stated_income", "stated_fees", "stated_pnl", "stated_return_pct", "source_file", "source_pages", "notes"]
    pd.DataFrame(rows).reindex(columns=cols).to_csv(d / "statements.csv", index=False)
    pd.DataFrame(flows).to_csv(d / "flows.csv", index=False)
    pd.DataFrame(columns=["statement_id", "account_id", "as_of_date", "identifier", "description", "asset_class", "quantity", "price", "market_value", "weight_pct"]).to_csv(d / "positions.csv", index=False)
    label = spec.name or "Simulated portfolio " + sid
    pd.DataFrame([dict(account_id=acct, label=label, owner_type="principal", discretionary="Y", strategy="default", benchmark="US_MKT",
                       notes=f"{len(spec.managers)} managers, {spec.rebalance} rebalancing, {b['fee']}")]).to_csv(d / "accounts.csv", index=False)
    names = ", ".join(f"{cands.get(k, {}).get('name', k)} {x:.0%}" for k, x in spec.managers)
    meta = dict(slug=sid, name=label, manager=names, style="sim", style_name="Simulated fund of managers", style_note=f"{spec.rebalance} rebalancing; {b['fee']} applied per manager",
                months=b["months"], first=b["first"], last=b["last"], status="ok", spec=dict(managers=spec.managers, mgmt=spec.mgmt, perf=spec.perf, rebalance=spec.rebalance, name=spec.name),
                ann_gross=b["ann_gross"], ann_net=b["ann_net"], fee_drag=b["fee_drag"], per_manager=b["per_manager"], fee=b["fee"])
    (d / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    pd.DataFrame({"month": g.index.strftime("%Y-%m"), "gross": g.values, "net": b["net"].values}).to_csv(d / "blend_monthly.csv", index=False)
    b["corr"].to_csv(d / "correlation.csv", index_label="key")
    return d


def report_args(sid: str, spec: Spec) -> list[str]:
    return ["report", "--data", f"data/sims/{sid}", "--out", f"output/sims/{sid}", "--grid", "M", "--placeholder",
            "--placeholder-note", "a simulated blend of built manager series — a what-if, not a record",
            "--mgmt-fee", str(spec.mgmt), "--perf-fee", str(spec.perf), "--n-boot", "1500", "--n-cohort", "3000",
            "--label", spec.name or f"Simulated portfolio {sid}"]


# ---------------------------------------------------------------- suggestion: from targets to a portfolio

@dataclass
class Targets:
    target_return: float = 0.10      # annualized, gross
    max_vol: float = 0.15
    max_drawdown: float = 0.30       # positive number: 0.30 = no worse than −30%
    min_sharpe: float = 0.5
    max_te: float = 0.06             # vs the US market
    n_managers: int = 6
    max_weight: float = 0.30
    min_months: int = 60
    styles: list[str] = field(default_factory=list)      # allowed style names; empty = all
    prefer: str = "wealth"           # which score ranks the candidate pool: "wealth" | "alpha_max" | "t"


def suggest(t: Targets, cands: list[dict], pool: int = 40) -> dict:
    """Pick up to n managers and weights to meet the targets on the common history.

    Optimises historical mean-variance under the caps (SLSQP), then keeps the n largest
    weights and re-solves on those.  Max drawdown, Sharpe and tracking error are checked
    after the fact and reported hit / missed — they are not convex, and an honest tool
    says which targets the history could not deliver rather than pretending."""
    from scipy.optimize import minimize
    from .reference import load_all
    rows = [c for c in cands if c["built"] and c["months"] >= t.min_months and (not t.styles or c["style"] in t.styles)]
    key = {"wealth": "wealth", "alpha_max": "alpha_max", "t": "t"}[t.prefer]
    scored = []
    for c in rows:
        s = stats_of(c["key"]); s["name"] = c["name"]; s["style"] = c["style"]
        if not pd.isna(s.get(key, np.nan)):
            scored.append(s)
    scored.sort(key=lambda s: -s.get(key, -1e9))
    pool_keys = [s["key"] for s in scored[:pool]]
    series = {k: gross_series(k) for k in pool_keys}
    series = {k: s for k, s in series.items() if s is not None}
    if len(series) < 2:
        return dict(ok=False, reason="fewer than two candidates with built monthly series")
    # common window: drop the shortest histories until the overlap is at least min_months
    keys = list(series)
    while True:
        idx = None
        for k in keys:
            idx = series[k].index if idx is None else idx.intersection(series[k].index)
        if len(idx) >= t.min_months or len(keys) <= 2:
            break
        keys.remove(min(keys, key=lambda k: len(series[k])))
    idx = idx.sort_values()
    R = pd.DataFrame({k: series[k].reindex(idx) for k in keys})
    mu = R.mean().values * 12; cov = R.cov().values * 12; n = len(keys)
    def solve(sel_idx):
        m = len(sel_idx); mu_s = mu[sel_idx]; C = cov[np.ix_(sel_idx, sel_idx)]
        cap = min(1.0, max(t.max_weight, 1.0 / m + 1e-9))
        cons = [dict(type="eq", fun=lambda w: w.sum() - 1.0)]
        # primary: minimise variance subject to reaching the return target; fall back to max return under the vol cap
        r1 = minimize(lambda w: w @ C @ w, np.full(m, 1 / m), method="SLSQP", bounds=[(0, cap)] * m,
                      constraints=cons + [dict(type="ineq", fun=lambda w: w @ mu_s - t.target_return)], options=dict(maxiter=500))
        if r1.success and (r1.x @ mu_s) >= t.target_return - 1e-6:
            return r1.x, "min variance at the return target"
        r2 = minimize(lambda w: -(w @ mu_s), np.full(m, 1 / m), method="SLSQP", bounds=[(0, cap)] * m,
                      constraints=cons + [dict(type="ineq", fun=lambda w: t.max_vol ** 2 - w @ C @ w)], options=dict(maxiter=500))
        if r2.success:
            return r2.x, "max return under the volatility cap (return target not reachable)"
        r3 = minimize(lambda w: -(w @ mu_s) / np.sqrt(max(w @ C @ w, 1e-12)), np.full(m, 1 / m), method="SLSQP", bounds=[(0, cap)] * m, constraints=cons, options=dict(maxiter=500))
        return r3.x, "max Sharpe (neither target reachable under the caps)"
    w, how = solve(list(range(n)))
    top = list(np.argsort(-w)[:t.n_managers])
    w2, how = solve(top)
    weights = pd.Series(w2, index=[keys[i] for i in top]); weights = weights[weights > 0.005]; weights = weights / weights.sum()
    port = (R[weights.index] * weights).sum(axis=1)
    mkt = load_all()["US_MKT"]; mkt.index = pd.DatetimeIndex(mkt.index).to_period("M").to_timestamp("M"); mkt = mkt.reindex(port.index)
    rf = load_all()["US_RF"]; rf.index = pd.DatetimeIndex(rf.index).to_period("M").to_timestamp("M"); rf = rf.reindex(port.index).fillna(0)
    ann = float((1 + port).prod() ** (12 / len(port)) - 1); vol = float(port.std() * np.sqrt(12))
    cum = (1 + port).cumprod(); dd = float((cum / cum.cummax() - 1).min())
    ex = port - rf; sharpe = float(ex.mean() / ex.std() * np.sqrt(12))
    act = port - mkt; te = float(act.std() * np.sqrt(12)); ir = float(act.mean() * 12 / te) if te > 0 else np.nan
    mkt_ann = float((1 + mkt.dropna()).prod() ** (12 / mkt.notna().sum()) - 1)
    checks = [("Return ≥ target", ann, t.target_return, ann >= t.target_return), ("Volatility ≤ cap", vol, t.max_vol, vol <= t.max_vol),
              ("Max drawdown ≥ −cap", dd, -t.max_drawdown, dd >= -t.max_drawdown), ("Sharpe ≥ min", sharpe, t.min_sharpe, sharpe >= t.min_sharpe),
              ("Tracking error ≤ cap", te, t.max_te, te <= t.max_te)]
    return dict(ok=True, weights=weights.to_dict(), how=how, months=len(port), first=str(port.index.min().date()), last=str(port.index.max().date()),
                pool=len(keys), ann=ann, vol=vol, max_dd=dd, sharpe=sharpe, te=te, ir=ir, mkt_ann=mkt_ann, checks=checks,
                names={k: next((c["name"] for c in cands if c["key"] == k), k) for k in weights.index})
