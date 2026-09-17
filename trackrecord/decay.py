"""Manager decay model: does what a manager's public book looks like today say anything
about whether the record will lag the market over the next year?

This is the question a manager-research team actually has to answer between annual
reviews — not "was this manager good", which the verification pages settle, but "is
there anything in the current filings that should put them on watch".  The inputs are
things a due-diligence analyst reads off a 13F and a return stream:

    n_positions   log of how many names the book holds
    hhi           concentration, sum of squared weights
    top1_w        weight of the largest position
    top10_w       weight of the ten largest
    turnover      1 − Σ min(w_t, w_{t−1}) between consecutive filings
    new_frac      share of the book (by names) not held last quarter
    sold_frac     share of last quarter's names no longer held
    book_growth   log change in the disclosed long book's dollar value (returns + flows), ×1000 unit steps removed, capped ±2
    crowding      value-weighted number of managers in the system holding each name
    excess_12/36  trailing 12- and 36-month return of the clone minus the market
    vol_12        trailing 12-month volatility, annualized
    mdd_12        trailing 12-month maximum drawdown
    alpha36_t     Newey-West t of the trailing 36-month FF3 alpha

One row per manager per formation date (the month-end the quarter's 13F becomes
public).  Label: 1 if the clone lags the US market over the following twelve months.

Three predictors, compared on exactly the same rows:
    persist   no model — last year's laggards lag again (rank by −excess_12)
    logistic  a logistic regression on the standardized features
    xgboost   gradient-boosted trees on the raw features

All out of sample, walk-forward: at each formation date the models are fit only on
rows whose twelve-month label window had *closed* by then (a twelve-month embargo), so
nothing about the test year is in the training set.  The regime problem — a bad year
hits every manager at once — is handled by scoring *within* each formation date
(cross-sectional AUC and a riskiest-fifth minus safest-fifth spread), and then asking
whether the average across dates is distinguishable from zero with Newey-West errors.

The sample is small in the way that matters: ~90 managers, ~12 non-overlapping years.
That is stated on the page; a result here is a scouting finding, not a validated model.
The R twin (r/decay.R, data.table + glm + xgboost) rebuilds the panel from the raw
filings and statements and re-runs the walk-forward; Python compares both.
"""
from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .signals13f import ROOT, load_books, load_factors
from .validation import _fit

FUNDS = ROOT / "data" / "funds"
OUT_DIR = ROOT / "data" / "research" / "decay"
R_SCRIPT = ROOT / "r" / "decay.R"

HORIZON = 12              # months in the label window
MIN_TRAIN_DATES = 12      # formation dates with closed labels before the first prediction (three years)
TOP_SHARE = 0.2           # riskiest / safest fifth for the spread
HAC_LAG = 3               # quarterly dates, twelve-month overlapping labels
FILING_FEATURES = ["n_positions", "hhi", "top1_w", "top10_w", "turnover", "new_frac", "sold_frac", "book_growth", "crowding"]
RETURN_FEATURES = ["excess_12", "excess_36", "vol_12", "mdd_12", "alpha36_t"]
FEATURES = FILING_FEATURES + RETURN_FEATURES
LABELS = {
    "n_positions": "Positions held (log)", "hhi": "Concentration (HHI)", "top1_w": "Largest position", "top10_w": "Top-10 weight",
    "turnover": "Quarterly turnover", "new_frac": "New names, share of book", "sold_frac": "Names sold, share of last book",
    "book_growth": "Book value growth (log, q/q)", "crowding": "Crowding (managers per name)",
    "excess_12": "Trailing 12m excess over market", "excess_36": "Trailing 36m excess over market", "vol_12": "Trailing 12m volatility",
    "mdd_12": "Trailing 12m max drawdown", "alpha36_t": "Trailing 36m FF3 alpha t",
}
MODELS = {"persist": "Persistence (no model)", "logistic": "Logistic regression", "xgboost": "xgboost (walk-forward)"}
XGB_PARAMS = dict(objective="binary:logistic", max_depth=3, eta=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0, nthread=4, seed=0)
XGB_ROUNDS = 200


# ---------------------------------------------------------------- features from the filings

def filing_features(books: pd.DataFrame) -> pd.DataFrame:
    """One row per (slug, period): what the book looks like and how it changed since last quarter."""
    g = books.copy()
    holders = g.groupby(["period", "cusip"]).slug.nunique().rename("holders")
    g = g.join(holders, on=["period", "cusip"])
    g["w2"] = g.w_all ** 2
    g["wh"] = g.w_all * g.holders
    g["top10"] = g.w_all.where(g["rank"] <= 10, 0.0)
    f = g.groupby(["slug", "period"]).agg(n_positions=("cusip", "size"), hhi=("w2", "sum"), top1_w=("w_all", "max"), top10_w=("top10", "sum"),
                                          crowding=("wh", "sum"), value=("value", "sum"), new_frac=("new", "mean"), has_prev=("has_prev", "first"),
                                          prev_period=("prev_period", "first"), formation=("formation", "first"), filed=("filed", "first")).reset_index()
    # overlap with the previous filing (value weights): turnover = 1 − Σ min(w_t, w_{t−1})
    prev = g[["slug", "period", "cusip", "w_all"]].rename(columns={"period": "prev_period", "w_all": "w_prev"})
    m = g[["slug", "period", "prev_period", "cusip", "w_all"]].merge(prev, on=["slug", "prev_period", "cusip"], how="left")
    m["overlap"] = np.minimum(m.w_all, m.w_prev.fillna(0.0))
    f = f.merge(m.groupby(["slug", "period"]).overlap.sum().rename("overlap").reset_index(), on=["slug", "period"], how="left")
    f["turnover"] = (1 - f.overlap).where(f.has_prev)
    # names sold: last quarter's names no longer held, as a share of last quarter's book
    sets = g.groupby(["slug", "period"]).cusip.apply(set)
    def sold(r):
        if not r.has_prev:
            return np.nan
        p = sets.get((r.slug, r.prev_period), set())
        return len(p - sets[(r.slug, r.period)]) / len(p) if p else np.nan
    f["sold_frac"] = [sold(r) for r in f.itertuples()]
    pv = f.set_index(["slug", "period"]).value
    growth = np.log(f.value / pv.reindex(pd.MultiIndex.from_arrays([f.slug, f.prev_period])).values)
    # 13F values switched from thousands to dollars for periods ending 2022-12-31 (some filers later, one or two
    # earlier by mistake): take out any ×1000 step, then cap at ±2 (a 7× quarter is already extreme)
    growth = growth - np.round(growth / np.log(1000)) * np.log(1000)
    f["book_growth"] = growth.clip(-2, 2).where(f.has_prev)
    f["n_positions"] = np.log(f.n_positions)
    f["new_frac"] = f.new_frac.where(f.has_prev)
    return f[["slug", "period", "formation", "filed"] + FILING_FEATURES]


# ---------------------------------------------------------------- features from the clone's returns

def load_clone_returns(funds_dir: Path = FUNDS) -> pd.DataFrame:
    """Monthly returns of every 13F clone from its statement-format dataset (one notional deposit, then drift)."""
    out = {}
    for d in sorted(funds_dir.iterdir()):
        f = d / "statements.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f, parse_dates=["period_end"]).sort_values("period_end")
        r = s.ending_value.pct_change()
        r.index = s.period_end.dt.to_period("M").dt.to_timestamp("M")
        r = r.dropna()
        if len(r):
            out[d.name] = r
    return pd.DataFrame(out).sort_index()


def _cum(logr: pd.DataFrame, n: int) -> pd.DataFrame:
    """Cumulative return over the n months ending at each row (NaN unless all n are present)."""
    return np.expm1(logr.rolling(n, min_periods=n).sum())


def return_features(R: pd.DataFrame, fac: pd.DataFrame) -> pd.DataFrame:
    """Long frame (slug, formation) of trailing features and the forward label, for every month-end."""
    idx = R.index.union(fac.index)
    R = R.reindex(idx); mkt = fac.MKT.reindex(idx)
    lp, lm = np.log1p(R), np.log1p(mkt)
    ex12, ex36 = _cum(lp, 12).sub(_cum(lm, 12), axis=0), _cum(lp, 36).sub(_cum(lm, 36), axis=0)
    vol12 = R.rolling(12, min_periods=12).std() * np.sqrt(12)
    def mdd(x):
        c = np.cumprod(1 + x); return (c / np.maximum.accumulate(c) - 1).min()
    mdd12 = R.rolling(12, min_periods=12).apply(mdd, raw=True)
    fwd = _cum(lp, HORIZON).sub(_cum(lm, HORIZON), axis=0).shift(-HORIZON)          # window F+1 … F+12
    fwd_p = _cum(lp, HORIZON).shift(-HORIZON)
    long = pd.concat({"excess_12": ex12, "excess_36": ex36, "vol_12": vol12, "mdd_12": mdd12, "fwd_excess": fwd, "fwd_return": fwd_p}, axis=1)
    long = long.stack(future_stack=True).reset_index()
    long.columns = ["formation", "slug"] + list(long.columns[2:])
    return long


def alpha36_t(R: pd.DataFrame, fac: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """Newey-West t of the FF3 intercept over the 36 months ending at each (slug, formation) row."""
    y = R.sub(fac.RF.reindex(R.index), axis=0)
    X = fac[["MKT_RF", "SMB", "HML"]]
    out = pd.Series(np.nan, index=rows.index)
    for i, r in rows.iterrows():
        if r.slug not in y:
            continue
        w = pd.concat([y[r.slug].rename("y"), X], axis=1).loc[:r.formation].tail(36).dropna()
        if len(w) < 36:
            continue
        fit, _ = _fit(w.y, w[["MKT_RF", "SMB", "HML"]], 12)
        out[i] = float(fit.tvalues[0])
    return out


def build_panel(log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    books = load_books(log)
    F = filing_features(books)
    fac = load_factors()
    R = load_clone_returns()
    rf = return_features(R, fac)
    P = F.merge(rf, on=["slug", "formation"], how="left")
    P = P[P.excess_12.notna()].copy()                       # a manager needs a year of clone history to be scored
    P["alpha36_t"] = alpha36_t(R, fac, P).values
    P["y"] = (P.fwd_excess < 0).astype(float).where(P.fwd_excess.notna())
    P = P.sort_values(["formation", "slug"]).reset_index(drop=True)
    log(f"panel: {P.slug.nunique()} managers, {P.formation.nunique()} formation dates, {len(P)} rows, {int(P.y.notna().sum())} labelled, base rate {P.y.mean():.0%}")
    return P, fac


# ---------------------------------------------------------------- models

def auc(y, s) -> float:
    """Area under the ROC curve by the rank (Mann-Whitney) formula; NaN if one class is missing."""
    y = np.asarray(y, dtype=bool); s = np.asarray(s, dtype=float)
    ok = ~np.isnan(s); y, s = y[ok], s[ok]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logistic(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, pd.Series]:
    """Plain (unpenalised) logistic regression on standardized features; missing → the training mean."""
    mu, sd = train[feats].mean(), train[feats].std().replace(0, 1.0)
    Xtr = ((train[feats] - mu) / sd).fillna(0.0).values
    Xte = ((test[feats] - mu) / sd).fillna(0.0).values
    try:
        fit = sm.GLM(train.y.values, sm.add_constant(Xtr), family=sm.families.Binomial()).fit()
        coef = pd.Series(fit.params[1:], index=feats)
        p = fit.predict(sm.add_constant(Xte, has_constant="add"))
    except Exception:
        return np.full(len(test), np.nan), pd.Series(np.nan, index=feats)
    return np.asarray(p), coef


def fit_xgb(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, pd.Series]:
    import xgboost as xgb
    dtr = xgb.DMatrix(train[feats].values, label=train.y.values, feature_names=feats, missing=np.nan)
    model = xgb.train(XGB_PARAMS, dtr, num_boost_round=XGB_ROUNDS)
    gain = model.get_score(importance_type="total_gain"); tot = sum(gain.values()) or 1.0
    imp = pd.Series({f: gain.get(f, 0.0) / tot for f in feats})
    return model.predict(xgb.DMatrix(test[feats].values, feature_names=feats, missing=np.nan)), imp


def walk_forward(P: pd.DataFrame, feats: list[str] = FEATURES, log=print) -> tuple[pd.DataFrame, dict]:
    """Fit at every formation date on rows whose label window has closed; predict that date's managers."""
    P = P.copy()
    for c in ("p_persist", "p_logistic", "p_xgboost", "p_base"):
        P[c] = np.nan
    dates = sorted(P.formation.unique())
    last = {}
    n_fits = 0
    for F in dates:
        closed = P.formation + pd.DateOffset(months=HORIZON) <= F
        train = P[closed & P.y.notna()]
        if train.formation.nunique() < MIN_TRAIN_DATES:
            continue
        test_idx = P.index[P.formation == F]
        test = P.loc[test_idx]
        P.loc[test_idx, "p_persist"] = stats.rankdata(-test.excess_12.fillna(0).values) / len(test)
        P.loc[test_idx, "p_base"] = train.y.mean()
        pl, coef = fit_logistic(train, test, feats)
        px, imp = fit_xgb(train, test, feats)
        P.loc[test_idx, "p_logistic"] = pl; P.loc[test_idx, "p_xgboost"] = px
        last = dict(date=str(pd.Timestamp(F).date()), n_train=int(len(train)), train_dates=int(train.formation.nunique()), coef=coef, imp=imp)
        n_fits += 1
    log(f"walk-forward: {n_fits} fits, last on {last.get('n_train', 0):,} rows through {last.get('date')}")
    return P, last


def _hac_t(x: pd.Series) -> tuple[float, float]:
    x = x.dropna()
    if len(x) < 4:
        return float(x.mean()) if len(x) else np.nan, np.nan
    fit = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAG})
    return float(fit.params[0]), float(fit.tvalues[0])


def evaluate(P: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per model: pooled AUC, cross-sectional AUC by date (mean, HAC t), riskiest-fifth minus safest-fifth forward excess."""
    T = P[P.y.notna() & P.p_xgboost.notna()]
    rows, by_date = [], []
    for m in MODELS:
        col = f"p_{m}"
        per = []
        for F, g in T.groupby("formation"):
            k = max(1, int(round(len(g) * TOP_SHARE)))
            s = g.sort_values(col)
            per.append(dict(formation=F, model=m, n=len(g), auc=auc(g.y, g[col]), base_rate=g.y.mean(),
                            risky=s.fwd_excess.tail(k).mean(), safe=s.fwd_excess.head(k).mean()))
        D = pd.DataFrame(per); D["spread"] = D.risky - D.safe
        by_date.append(D)
        auc_mean, auc_t = _hac_t(D.auc - 0.5)
        sp_mean, sp_t = _hac_t(D.spread)
        brier = float(((T[col] - T.y) ** 2).mean()) if m != "persist" else np.nan
        brier_base = float(((T.p_base - T.y) ** 2).mean())
        rows.append(dict(model=m, label=MODELS[m], n_rows=len(T), n_dates=int(D.formation.nunique()), auc_pooled=auc(T.y, T[col]),
                         auc_cs_mean=auc_mean + 0.5, auc_cs_t=auc_t, auc_cs_pct_above=float((D.auc > 0.5).mean()),
                         spread_mean=sp_mean, spread_t=sp_t, brier=brier, brier_base=brier_base,
                         brier_skill=(1 - brier / brier_base) if brier == brier else np.nan))
    return pd.DataFrame(rows), pd.concat(by_date, ignore_index=True)


def grouped_cv(P: pd.DataFrame, feats: list[str] = FEATURES, folds: int = 5) -> dict:
    """Leave-managers-out: does the model carry over to managers it has never seen?  Optimistic on time
    (train and test share calendar years), so reported only as a robustness check."""
    T = P[P.y.notna()].copy()
    slugs = sorted(T.slug.unique())
    fold = {s: i % folds for i, s in enumerate(slugs)}
    T["fold"] = T.slug.map(fold)
    pl_all, px_all = pd.Series(np.nan, index=T.index), pd.Series(np.nan, index=T.index)
    for k in range(folds):
        tr, te = T[T.fold != k], T[T.fold == k]
        pl_all[te.index], _ = fit_logistic(tr, te, feats)
        px_all[te.index], _ = fit_xgb(tr, te, feats)
    return dict(folds=folds, auc_logistic=auc(T.y, pl_all), auc_xgboost=auc(T.y, px_all), auc_persist=auc(T.y, -T.excess_12.fillna(0)))


def univariate(P: pd.DataFrame, feats: list[str] = FEATURES) -> pd.DataFrame:
    """Each feature on its own: rank correlation with the next year's excess return, across managers, date by date."""
    T = P[P.y.notna()]
    rows = []
    for f in feats:
        ics = []
        for F, g in T.groupby("formation"):
            g = g[[f, "fwd_excess"]].dropna()
            if len(g) >= 10:
                ics.append(stats.spearmanr(g[f], g.fwd_excess)[0])
        ics = pd.Series(ics)
        mean, t = _hac_t(ics)
        rows.append(dict(feature=f, label=LABELS[f], coverage=float(P[f].notna().mean()), dates=len(ics), ic_mean=mean, ic_t=t,
                         ic_pct_positive=float((ics > 0).mean()) if len(ics) else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- build

def _names() -> pd.DataFrame:
    lb = FUNDS / "leaderboard.csv"
    if not lb.exists():
        return pd.DataFrame(columns=["slug", "name", "manager", "style_name"])
    return pd.read_csv(lb)[["slug", "name", "manager", "style_name"]]


def build(out_dir: Path = OUT_DIR, log=print, run_r: bool = True) -> dict:
    P, fac = build_panel(log)
    P, last = walk_forward(P, log=log)
    S, D = evaluate(P)
    U = univariate(P)
    G = grouped_cv(P)
    imp, coef = last.get("imp", pd.Series(dtype=float)), last.get("coef", pd.Series(dtype=float))
    U["xgb_gain_share"] = U.feature.map(imp); U["logistic_coef"] = U.feature.map(coef)
    # head to head, same dates
    Dx = D.pivot(index="formation", columns="model", values="auc")
    diff = (Dx["xgboost"] - Dx["logistic"]).dropna()
    hh_mean, hh_t = _hac_t(diff)
    head = dict(dates=int(len(diff)), auc_diff_mean=hh_mean, auc_diff_t=hh_t, xgb_wins_share=float((diff > 0).mean()) if len(diff) else np.nan)
    # the latest formation date: what the models say now (no label yet)
    latest_F = P.formation.max()
    W = P[P.formation == latest_F].merge(_names(), on="slug", how="left")
    W = W.sort_values("p_xgboost", ascending=False)
    W["rank_xgboost"] = np.arange(1, len(W) + 1)
    W["rank_logistic"] = W.p_logistic.rank(ascending=False).astype("Int64")
    W["rank_persist"] = W.p_persist.rank(ascending=False).astype("Int64")

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["slug", "period", "formation", "filed"] + FEATURES + ["fwd_excess", "fwd_return", "y", "p_persist", "p_base", "p_logistic", "p_xgboost"]
    P[cols].to_csv(out_dir / "panel.csv", index=False)
    S.to_csv(out_dir / "summary.csv", index=False)
    D.to_csv(out_dir / "by_date.csv", index=False)
    U.to_csv(out_dir / "features.csv", index=False)
    W[["slug", "name", "manager", "style_name", "period"] + FEATURES + ["p_persist", "p_logistic", "p_xgboost", "rank_xgboost", "rank_logistic", "rank_persist"]].to_csv(out_dir / "watchlist.csv", index=False)
    fac[["MKT", "MKT_RF", "SMB", "HML", "RF"]].dropna().to_csv(out_dir / "factors.csv", index_label="month")
    T = P[P.y.notna()]
    man = dict(built=str(date.today()), managers=int(P.slug.nunique()), managers_labelled=int(T.slug.nunique()), formation_dates=int(P.formation.nunique()),
               first=str(P.formation.min().date()), last=str(latest_F.date()), rows=int(len(P)), rows_labelled=int(len(T)), base_rate=float(T.y.mean()),
               tested_rows=int(S.n_rows.iloc[0]) if len(S) else 0, tested_dates=int(S.n_dates.iloc[0]) if len(S) else 0,
               first_test=str(P[P.p_xgboost.notna()].formation.min().date()) if P.p_xgboost.notna().any() else None,
               horizon=HORIZON, min_train_dates=MIN_TRAIN_DATES, top_share=TOP_SHARE, hac_lag=HAC_LAG, features=FEATURES,
               xgb_params={k: v for k, v in XGB_PARAMS.items() if k != "nthread"}, xgb_rounds=XGB_ROUNDS,
               last_fit=dict(date=last.get("date"), n_train=last.get("n_train"), train_dates=last.get("train_dates")),
               head_to_head=head, grouped_cv=G, r=None)
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    if run_r:
        man["r"] = compare_with_r(out_dir, log=log)
        (out_dir / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    log(f"wrote {out_dir}")
    return man


# ---------------------------------------------------------------- R twin

def compare_with_r(out_dir: Path = OUT_DIR, log=print, timeout: int = 900) -> dict:
    """Run r/decay.R (data.table panel from the raw filings and statements; glm and xgboost walk-forward)
    and compare its panel and its walk-forward AUCs with Python's.  Reports unavailability rather than failing."""
    import shutil
    if shutil.which("Rscript") is None:
        return dict(available=False, reason="Rscript not installed")
    try:
        r = subprocess.run(["Rscript", str(R_SCRIPT), str(out_dir)], capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return dict(available=False, reason="R timed out")
    if r.returncode != 0:
        return dict(available=False, reason=(r.stderr or r.stdout).strip()[-800:])
    log(r.stdout.strip())
    py = pd.read_csv(out_dir / "panel.csv", parse_dates=["formation"])
    rr = pd.read_csv(out_dir / "panel_r.csv", parse_dates=["formation"])
    key = ["slug", "formation"]
    both = py.merge(rr, on=key, suffixes=("_py", "_r"))
    cmp_cols = FEATURES + ["fwd_excess", "y", "p_logistic"]
    diffs = {}
    for c in cmp_cols:
        a, b = both[f"{c}_py"], both[f"{c}_r"]
        same_nan = (a.isna() == b.isna()).mean()
        diffs[c] = dict(max_abs_diff=float((a - b).abs().max()) if (a.notna() & b.notna()).any() else 0.0, same_missing=float(same_nan))
    worst = max(diffs.values(), key=lambda d: d["max_abs_diff"])["max_abs_diff"]
    out = dict(available=True, rows_python=int(len(py)), rows_r=int(len(rr)), rows_matched=int(len(both)), max_abs_diff=worst, by_column=diffs)
    sr = out_dir / "summary_r.csv"
    if sr.exists():
        S = pd.read_csv(sr).set_index("model")
        out["auc_cs_mean_r"] = {m: float(S.loc[m, "auc_cs_mean"]) for m in S.index}
        out["auc_pooled_r"] = {m: float(S.loc[m, "auc_pooled"]) for m in S.index}
    return out
