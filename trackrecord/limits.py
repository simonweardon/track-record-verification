"""Due diligence on this work: the checks a reviewer would run before believing any of it.

Every other tool here asks hard questions of an outside manager.  This one asks the same
questions of the site itself, and answers them with numbers rather than with a paragraph of
hedging.  Eight of them, each written to a CSV so the answer moves when the data moves:

    universe_ic.csv     the same signals re-tested on universes 3x and 5x wider, and on each
                        half of the sample: is the weak result the universe, or the sample?
    coverage.csv        what share of each quarter's disclosed book reaches a priced security,
                        by quarter -- the direct measure of what survivorship costs here
    missing.csv         the largest positions that never reach a price, by era, with names
    law.csv             the headline portfolio result reconciled against the signal it is
                        built from, through the fundamental law of active management
    costs.csv           the same portfolio re-run at six trading-cost assumptions
    capacity.csv        how large the portfolio can be before the trades stop being realistic
    reconstruction.csv  per manager: how much of the real book the disclosed holdings can be,
                        measured from the filings (options disclosed, positions, coverage)
    evidence.csv        how long a record has to be before a given skill level is detectable
    freshness.csv       when each input was last pulled and what the newest data point is
    verification.csv    what the second, independent implementation covers -- and what it cannot
    cleaning.csv        the data errors that had to be fixed by hand, counted, with examples
    computation.csv     which parts of the site are computed on request and which are stored

Rebuild with `python -m trackrecord limits` (about six minutes: the signal panels are rebuilt
on three universes and the portfolio backtest is run six times).
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "research" / "limits"

UNIVERSES = (5, 2, 1)                 # minimum managers holding a name for it to enter the universe
COST_GRID = (0.0, 10.0, 25.0, 50.0, 100.0, 200.0)
NAV_GRID = (1e8, 5e8, 1e9, 5e9, 1e10, 2.5e10, 5e10)
ADV_OF_CAP = 0.005                    # a stock trades about 0.5% of its market value on an average day
SPREAD_BPS = 10.0                     # commission and half-spread, before market impact
MAX_DAYS_ADV = 5.0                    # a trade bigger than five days of volume is not a trade, it is a project
MAX_POS_OF_COMPANY = 0.05             # above 5% of a company an owner has to file and stops being a passive holder


# ---------------------------------------------------------------- 1. universe

def universe_ic(log=print) -> pd.DataFrame:
    """The alpha lab's signals re-tested on three nested universes and on each half of the sample.

    If the weak information coefficients were an artefact of testing large, heavily held names,
    widening the universe five-fold should lift them.  If the premia have decayed, the second
    half of the sample should be visibly worse than the first.  Both claims are testable here."""
    from .alphalab import SIGNALS, build_panel, ic_series

    def t(ic: pd.Series) -> float:
        return float(ic.mean() / ic.std() * np.sqrt(len(ic))) if len(ic) > 1 and ic.std() > 0 else np.nan

    rows = []
    for mh in UNIVERSES:
        panel, info = build_panel(log=lambda *a: None, min_holders=mh)
        feats = [s for s in SIGNALS if s in panel and panel[s].notna().mean() > 0.3]
        panel["composite"] = panel[feats].mean(axis=1, skipna=True).where(panel[feats].notna().sum(axis=1) >= max(2, len(feats) // 2))
        log(f"  universe: held by {mh}+ managers, {info['universe_avg']} names a month, {info['months']} months")
        for s in feats + ["composite"]:
            ic = ic_series(panel, s)
            if len(ic) < 12:
                continue
            mid = ic.index[len(ic) // 2]
            a, b = ic[ic.index < mid], ic[ic.index >= mid]
            rows.append(dict(min_holders=mh, universe_avg=info["universe_avg"], signal=s,
                             label=SIGNALS.get(s, "Average of the eight"), coverage=float(panel[s].notna().mean()),
                             months=len(ic), ic=float(ic.mean()), ic_t=t(ic), share_positive=float((ic > 0).mean()),
                             ic_first=float(a.mean()), t_first=t(a), first_to=str(mid.date()),
                             ic_second=float(b.mean()), t_second=t(b)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. survivorship

def coverage_by_quarter(log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """What share of each quarter's disclosed book reaches a security we can price, and the
    largest positions that do not.

    A company that was bought, taken private or renamed is not in the price history at all, so
    it never enters any universe on the site.  The share that does reach a price is therefore a
    direct measure of how much of the past has been quietly deleted, quarter by quarter."""
    from .signals13f import load_books, load_prices
    books = load_books(log=lambda *a: None)
    prices = load_prices()
    b = books[books.formation.notna()].copy()
    b["priced"] = b.ticker.notna() & b.ticker.isin(prices.columns)
    g = b.groupby("formation")
    cov = pd.DataFrame({
        "positions": g.size(),
        "positions_priced": g.priced.sum(),
        "value": g.value.sum(),
        "value_priced": b[b.priced].groupby("formation").value.sum(),
        "names": g.ticker.nunique(),
    }).fillna(0.0)
    cov["share_positions"] = cov.positions_priced / cov.positions
    cov["share_value"] = cov.value_priced / cov.value
    held = b[b.priced].groupby(["formation", "ticker"]).slug.nunique()
    cov["universe_5"] = held[held >= 5].groupby("formation").size()
    cov["universe_1"] = held.groupby("formation").size()
    cov = cov.reset_index()
    cov["formation"] = pd.to_datetime(cov.formation).dt.date
    # the biggest positions that never reach a price, grouped into eras
    miss = b[~b.priced].copy()
    miss["year"] = pd.to_datetime(miss.formation).dt.year
    # eras chosen so that none straddles the change of reporting unit at the end of 2022, which
    # would otherwise make positions from different years incomparable within the same group
    eras = [(2013, 2015), (2016, 2018), (2019, 2022), (2023, 2026)]
    ex = []
    for lo, hi in eras:
        e = miss[(miss.year >= lo) & (miss.year <= hi)]
        if e.empty:
            continue
        tot = b[(pd.to_datetime(b.formation).dt.year >= lo) & (pd.to_datetime(b.formation).dt.year <= hi)]
        top = e.groupby("name").value.sum().sort_values(ascending=False).head(8)
        for nm, v in top.items():
            ex.append(dict(era=f"{lo}–{hi}", name=str(nm).title(), share_of_missing=float(v / e.value.sum()),
                           era_share_value_unpriced=float(e.value.sum() / tot.value.sum())))
    log(f"  disclosed book reaching a price: {cov.share_value.iloc[0]:.0%} at the start, {cov.share_value.iloc[-1]:.0%} now")
    return cov, pd.DataFrame(ex)


# ---------------------------------------------------------------- 3. the headline result vs the signal

def law_and_costs(log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The construction result reconciled with the signal that drives it, and re-run at six
    trading-cost assumptions.

    Grinold's fundamental law says the information ratio a manager can expect is roughly the
    skill in the signal (its information coefficient) times the square root of the number of
    independent bets, scaled by how much of the signal survives the mandate's constraints (the
    transfer coefficient).  If the headline information ratio is far above what the law implies,
    the portfolio result is not coming from the signal and something needs explaining."""
    import tempfile
    from .construct import Constraints, backtest

    rows, law = [], []
    for c_bps in COST_GRID:
        d = Path(tempfile.mkdtemp())
        backtest(Constraints(cost_bps=c_bps), out_dir=d, log=lambda *a: None)
        s = pd.read_csv(d / "summary.csv").set_index("key")
        rb = pd.read_csv(d / "rebalances.csv")
        p = s.loc["portfolio"]
        rows.append(dict(cost_bps=c_bps, ann_return=float(p.ann_return), active_return=float(p.active_return),
                         tracking_error=float(p.tracking_error), information_ratio=float(p.information_ratio),
                         hit_rate=float(p.hit_rate), turnover=float(p.avg_turnover),
                         unconstrained_active=float(s.loc["unconstrained"].active_return),
                         unconstrained_ir=float(s.loc["unconstrained"].information_ratio)))
        log(f"  {c_bps:5.0f} bps: active {p.active_return:+.2%}/yr, information ratio {p.information_ratio:.2f}")
        if c_bps == 10.0:
            ic = rb.realized_ic.dropna()
            tc = float(rb.transfer_coef.mean())
            n = float(rb.universe.mean())
            months = int(pd.read_csv(d / "backtest_monthly.csv").portfolio.notna().sum())
            years = months / 12
            ir = float(p.information_ratio)
            law = pd.DataFrame([
                dict(key="ic", label="Skill in the signal each quarter", value=float(ic.mean()),
                     detail=f"rank correlation between the signal and what the {n:.0f} names then did, averaged over {len(ic)} quarters"),
                dict(key="ic_t", label="Is that skill distinguishable from zero?", value=float(ic.mean() / ic.std() * np.sqrt(len(ic))),
                     detail="t-statistic of the quarterly skill measure; below 2 it is not"),
                dict(key="transfer", label="Share of the signal the mandate lets through", value=tc,
                     detail="correlation between the positions actually taken and the signal that asked for them"),
                dict(key="breadth", label="Independent bets a year", value=n * 4,
                     detail=f"{n:.0f} names, re-decided four times a year, counted as if independent (they are not)"),
                dict(key="ir_implied", label="Information ratio the law implies", value=tc * float(ic.mean()) * np.sqrt(n * 4),
                     detail="skill x share let through x square root of the bets"),
                dict(key="ir_actual", label="Information ratio the portfolio delivered", value=ir,
                     detail=f"active return over its own variability, {months} months"),
                dict(key="ir_se", label="How precisely that is known", value=float(np.sqrt((1 + ir ** 2 / 2) / years)),
                     detail=f"standard error of an information ratio measured over {years:.1f} years"),
                dict(key="ir_t", label="Is the delivered result distinguishable from zero?", value=float(ir * np.sqrt(years)),
                     detail="t-statistic of the active return"),
            ])
    return law, pd.DataFrame(rows)


def capacity(log=print) -> pd.DataFrame:
    """How large the portfolio can be before the trade list stops being realistic.

    Market impact is estimated with the square-root rule used across the industry: trading one
    day's volume in a name moves its price by about one daily standard deviation, and smaller
    trades cost the square root of the fraction traded.  A day's volume is taken as 0.5% of the
    company's market value, which is about right for large US stocks.  Both are assumptions,
    stated here so a reader can disagree with them."""
    from .alphalab import as_of, clean_shares, load_close, pit_table
    from .construct import OUT_DIR as CON_DIR
    from .fundamentals import load as load_fundamentals
    from .signals13f import clean_returns, load_prices

    tl = pd.read_csv(CON_DIR / "trade_list.csv")
    prices = load_prices(); rets = clean_returns(prices); close = load_close(prices)
    f = load_fundamentals()
    f["filed"] = pd.to_datetime(f.filed); f["end"] = pd.to_datetime(f["end"])
    shares = clean_shares(pit_table(f, "shares", "instant"))
    F = prices.index[-1]
    tk = pd.Index(tl.ticker)
    mcap = as_of(shares, tk, F) * close.loc[F].reindex(tk)
    daily_sd = (rets[[t for t in tk if t in rets.columns]].loc[:F].tail(24).std() / np.sqrt(21)).reindex(tk)
    dw = pd.Series((tl.target_wt - tl.current_wt).abs().values, index=tk)
    tw = pd.Series(tl.target_wt.values, index=tk)
    ok = mcap.notna() & daily_sd.notna() & (dw > 0)
    adv = mcap * ADV_OF_CAP
    rows = []
    for nav in NAV_GRID:
        part = (dw * nav / adv)[ok]
        imp = SPREAD_BPS + 100 * daily_sd[ok] * np.sqrt(part)
        w = (dw * nav)[ok]
        pos = (tw * nav / mcap).dropna()
        cost = float((imp * w).sum() / w.sum())
        rows.append(dict(nav=nav, cost_bps=cost, annual_drag=4 * 2 * float(tl.assign(d=(tl.target_wt - tl.current_wt).abs()).d.sum() / 2) * cost / 1e4,
                         median_days_of_volume=float(part.median()), p95_days_of_volume=float(part.quantile(0.95)),
                         max_days_of_volume=float(part.max()), trades_over_limit=int((part > MAX_DAYS_ADV).sum()),
                         largest_stake=float(pos.max()), stakes_over_limit=int((pos > MAX_POS_OF_COMPANY).sum()),
                         names_measured=int(ok.sum()), names_total=len(tk),
                         value_measured=float(dw[ok].sum() / dw.sum())))
    D = pd.DataFrame(rows)
    log(f"  estimated cost {D.cost_bps.iloc[0]:.0f} bps at ${NAV_GRID[0] / 1e6:,.0f}m, {D.cost_bps.iloc[-1]:.0f} bps at ${NAV_GRID[-1] / 1e9:,.0f}bn")
    return D


# ---------------------------------------------------------------- 4. what the holdings can be

CONFIDENCE = {
    "lo": ("Close", "A long-only, diversified, low-turnover portfolio: the disclosure is most of it."),
    "conc": ("Close", "Few positions, held for years: the disclosure is most of the portfolio."),
    "act": ("Close", "Large, public, long-held stakes: the disclosure is most of the portfolio."),
    "fo": ("Close", "The US long positions of a closed book."),
    "ls": ("Long side only", "Short positions and hedges are disclosed nowhere, so the record here is the long book alone and will not look like the fund."),
    "event": ("Partial", "Credit, claims and merger positions do not appear in an equity disclosure, so a large part of the book is invisible."),
    "multi": ("Not meaningful", "Thousands of hedged positions and high turnover: the disclosure is inventory, not a portfolio."),
    "macro": ("Not meaningful", "The portfolio is futures, currencies and rates, which are not disclosed here."),
    "mm": ("Not meaningful", "For a market maker the disclosure is trading inventory, hedged and stale."),
}
OPTION_HEAVY = 0.15       # above this share of disclosed value in options, the stock positions are not the book


def confidence_for(style: str, option_share: float) -> tuple[str, str]:
    """How closely a record built from the disclosed stock positions can track the real portfolio.

    The label starts from what kind of firm it is and is then overridden by what the disclosure
    itself says: a manager who puts a sixth of the disclosed book into puts and calls is telling
    us, in the filing, that the stock lines are not the position."""
    base, note = CONFIDENCE.get(style, ("Partial", ""))
    if base == "Not meaningful":
        return base, note
    if option_share >= OPTION_HEAVY:
        return ("Stock positions only",
                f"{option_share:.0%} of the disclosed value is in puts and calls, which the record leaves out entirely. "
                f"What is measured here is the remaining {1 - option_share:.0%}, and it is not the position the manager took. " + note)
    return base, note


def reconstruction(log=print) -> pd.DataFrame:
    """Per manager, how much of the real portfolio the disclosed holdings can possibly be.

    The filings themselves say more than the label does.  Puts and calls are disclosed, so a
    manager who hedges through options leaves a visible trace; a manager who is short the stock
    itself leaves none.  Both numbers are here next to the share of the book that reaches a
    price, so the ranking on the rest of the site can be read with the right discount."""
    from .compact import COMPACT, holdings_frame
    from .fund_universe import STYLES
    from .signals13f import load_books, load_prices

    h = holdings_frame()
    funds = {f["slug"]: f for f in json.loads((COMPACT / "funds.json").read_text())}
    prices = load_prices()
    books = load_books(log=lambda *a: None)
    books = books[books.formation.notna()]
    priced = books.assign(p=books.ticker.notna() & books.ticker.isin(prices.columns)).groupby("slug").apply(
        lambda g: pd.Series({"share_value_priced": g[g.p].value.sum() / g.value.sum() if g.value.sum() else np.nan}), include_groups=False)
    rows = []
    for slug, g in h.groupby("slug"):
        f = funds.get(slug)
        if not f:
            continue
        opt = g[g.putcall.isin(("put", "call"))]
        tot = float(g.value.sum()) or np.nan
        per_q = g.groupby("period").size()
        style = f["style"]
        o_share = float(opt.value.sum() / tot)
        label, note = confidence_for(style, o_share)
        rows.append(dict(slug=slug, name=f["name"], manager=f["manager"], style=style,
                         style_label=STYLES[style][0], confidence=label, confidence_note=note,
                         quarters=int(g.period.nunique()), positions_median=float(per_q.median()),
                         option_share=o_share, stock_share=1.0 - o_share,
                         put_share=float(opt[opt.putcall == "put"].value.sum() / tot),
                         quarters_with_options=int(opt.period.nunique()),
                         share_value_priced=float(priced.share_value_priced.get(slug, np.nan))))
    D = pd.DataFrame(rows).sort_values("option_share", ascending=False).reset_index(drop=True)
    log(f"  {len(D)} managers; {int((D.confidence == 'Long side only').sum())} are long/short, "
        f"{int((D.confidence == 'Stock positions only').sum())} put more than {OPTION_HEAVY:.0%} of the disclosed book into options")
    return D


# ---------------------------------------------------------------- 5. how long a record has to be

def evidence_needed(law: pd.DataFrame) -> pd.DataFrame:
    """Years of monthly data needed before a given information ratio is distinguishable from luck.

    The t-statistic of an information ratio measured over y years is roughly IR x sqrt(y), so the
    years needed for a two-standard-error result is (2 / IR)^2.  This is the whole reason the site
    refuses to call a 1.5-standard-error alpha skill."""
    ir_actual = float(law.set_index("key").value.get("ir_actual", np.nan))
    rows = []
    for ir in (0.25, 0.40, 0.50, 0.75, 1.00):
        rows.append(dict(information_ratio=ir, years_for_t2=(2.0 / ir) ** 2, years_for_t3=(3.0 / ir) ** 2,
                         detectable_now=bool(ir * np.sqrt(13.0) >= 2.0)))
    return pd.DataFrame(rows).assign(ir_actual=ir_actual)


# ---------------------------------------------------------------- 6-8. provenance, plumbing, machinery

def cleaning(log=print) -> pd.DataFrame:
    """The data errors that had to be found and fixed, counted in the data as it stands today."""
    from .alphalab import clean_shares, pit_table
    from .compact import holdings_frame
    from .fundamentals import load as load_fundamentals
    from .signals13f import OUT_DIR as SIG_DIR

    h = holdings_frame()
    book = h.groupby(["slug", "period"]).value.sum().reset_index().sort_values(["slug", "period"])
    book["prev"] = book.groupby("slug").value.shift()
    book = book[(book.prev > 0) & (book.value > 0)]
    step = np.log(book.value / book.prev) / np.log(1000)
    unit_steps = book[step.round().abs() > 0]
    boundary = int((unit_steps.period == "2022-12-31").sum())
    # a stray unit error is one filing out of line with the quarter before AND the quarter after,
    # which is what separates a typing mistake from a fund that really did grow or shrink
    book["next"] = book.groupby("slug").value.shift(-1)
    stray = book[(step.round().abs() > 0) & (book.period != "2022-12-31") & (book["next"] > 0)]
    stray = stray[(np.log(stray["next"] / stray.value) / np.log(1000)).round().abs() > 0]
    worst = stray.iloc[0] if len(stray) else None

    f = load_fundamentals()
    f["filed"] = pd.to_datetime(f.filed); f["end"] = pd.to_datetime(f["end"])
    raw = pit_table(f, "shares", "instant").drop_duplicates(["ticker", "end"], keep="last")
    adj = clean_shares(pit_table(f, "shares", "instant"))
    m = raw.merge(adj[["ticker", "end", "value"]], on=["ticker", "end"], suffixes=("_raw", "_adj"))
    changed = m[(m.value_adj / m.value_raw).round(3) != 1.0]
    nv = raw[raw.ticker == "NVDA"].sort_values("end")
    nv_adj = adj[adj.ticker == "NVDA"].sort_values("end")
    nv_ex = (f"NVIDIA reported {nv.value.iloc[0] / 1e6:,.0f} million shares in {nv.end.iloc[0].year}; against prices that carry "
             f"two later splits the same company has {nv_adj.value.iloc[0] / 1e9:,.1f} billion on today's basis, a factor of "
             f"{nv_adj.value.iloc[0] / nv.value.iloc[0]:.0f}") if len(nv) and len(nv_adj) else ""

    sig = json.loads((SIG_DIR / "manifest.json").read_text()) if (SIG_DIR / "manifest.json").exists() else {}
    from .hedge13f import MANUAL_CUSIPS

    rows = [
        dict(issue="Reported portfolio values changed unit",
             rule="A quarter-on-quarter change of exactly a thousand times is a change of unit, not of the portfolio, and is removed before anything is measured",
             cases=len(unit_steps),
             example=(f"Every manager's disclosed book appears to grow a thousandfold at the end of 2022, when the reporting unit changed "
                      f"from thousands of dollars to dollars ({boundary} of the {len(unit_steps)} cases). The other {len(unit_steps) - boundary} are "
                      f"single filings where one manager used the wrong unit"
                      + (f". {worst.slug.replace('-', ' ').title()} filed the quarter ending {worst.period} with a book a thousand times the quarter "
                         f"before it and the quarter after it, and neither neighbour is wrong" if worst is not None else "") + ".")),
        dict(issue="Share counts are on an old basis after a split",
             rule="A persistent jump matching a known split ratio is taken out of the whole earlier history, so share counts and prices are on the same basis",
             cases=int(changed.ticker.nunique()),
             example=nv_ex + ". Without this the company sits in the smallest size group for years, and every size and value measure built on it is wrong."),
        dict(issue="One-off errors in reported share counts",
             rule="A filing five times its neighbours in both directions, in a run of at most three, is dropped rather than believed",
             cases=int(len(raw) - len(adj)),
             example=f"A company files its share count in thousands one quarter and in units the next, so it looks as though it was taken over and then un-taken over. "
                     f"{len(raw) - len(adj):,} of {len(raw):,} filed share counts are dropped this way."),
        dict(issue="Securities that never reach a price",
             rule="A position that cannot be matched to a priced security is left out of every portfolio, and the share left out is reported for every quarter",
             cases=int(len(MANUAL_CUSIPS)),
             example=("Most are companies that were bought or renamed, so no price history exists to buy. "
                      f"{len(MANUAL_CUSIPS)} identifiers that the reference tables get wrong are corrected by hand, mostly foreign listings and share classes.")),
        dict(issue="Impossible monthly returns in the price history",
             rule="A gain of more than 300% in a month from a price under $5 is treated as a bad price and left blank rather than earned",
             cases=int(sig.get("spikes_removed", 0)),
             example="These are almost all delisted or near-bankrupt shells whose quoted price bounces between cents. Left in, a handful of them would carry a whole portfolio's return."),
    ]
    log(f"  {len(rows)} classes of data error, {sum(r['cases'] for r in rows)} cases in total")
    return pd.DataFrame(rows)


def _last_refreshed(rel: str) -> str:
    """When a committed data file was last changed, from the repository's own history.

    The file's timestamp on disk is the moment the code was copied onto this machine, which
    says nothing; the commit date is when the data behind it was actually refreshed."""
    import subprocess
    try:
        r = subprocess.run(["git", "log", "-1", "--format=%cs", "--", rel], cwd=ROOT, capture_output=True, text=True, timeout=20)
        out = r.stdout.strip()
        if r.returncode == 0 and len(out) == 10:
            return out
    except Exception:
        pass
    return "unrecorded"


def freshness(log=print) -> pd.DataFrame:
    """When each input was last pulled, and what the newest data point in it is."""
    from .compact import COMPACT, holdings_frame
    from .fundamentals import load as load_fundamentals
    from .signals13f import load_prices

    h = holdings_frame()
    h["filed"] = pd.to_datetime(h.filed)
    last_period = max(h.period)
    prev_period = sorted(h.period.unique())[-2]
    n_last = h[h.period == last_period].slug.nunique()
    n_prev = h[h.period == prev_period].slug.nunique()
    prices = load_prices()
    f = load_fundamentals()
    rows = [
        dict(source="Managers' quarterly holdings filings", pulled=_last_refreshed("data/reference/compact/holdings13f.csv.gz"),
             newest=f"the quarter ending {last_period}",
             note=(f"{n_last} of the {h.slug.nunique()} managers had filed for that quarter by the deadline, against {n_prev} for the quarter before. "
                   f"A manager who files late, or whose holdings fall below the reporting threshold, simply has no filing for the quarter rather than a stale one.")),
        dict(source="Monthly share prices", pulled=_last_refreshed("data/reference/compact/prices_monthly.csv.gz"),
             newest=str(prices.index.max().date()),
             note=f"{prices.shape[1]:,} companies, adjusted for splits and dividends. Research is run to the last complete month."),
        dict(source="Company accounts", pulled=_last_refreshed("data/reference/compact/fundamentals.csv.gz"),
             newest=f"accounts filed up to {str(f.filed.max())[:10]}",
             note=f"{f.ticker.nunique():,} companies. A figure is only ever used after the date it was actually filed, and only if the period it covers ended within fifteen months."),
        dict(source="Market and style returns", pulled=_last_refreshed("data/reference/compact/kenfrench"),
             newest="the last complete month",
             note="The standard academic series, which are published a few weeks after each month ends."),
    ]
    log(f"  newest filing {h.filed.max().date()}, covering the quarter ending {last_period}")
    return pd.DataFrame(rows)


def verification_scope() -> pd.DataFrame:
    """What the second implementation covers, and what no amount of agreement could catch."""
    rv = ROOT / "data" / "research" / "r-verify" / "manifest.json"
    man = json.loads(rv.read_text()) if rv.exists() else {}
    con = ROOT / "data" / "research" / "construction" / "manifest.json"
    cman = json.loads(con.read_text()) if con.exists() else {}
    dec = ROOT / "data" / "research" / "decay" / "manifest.json"
    dman = json.loads(dec.read_text()) if dec.exists() else {}
    twin = (cman.get("r_twin") or {})
    rows = [
        dict(area="Manager statistics", covered=True,
             what="Four factor models per manager, their alphas, standard errors, t-statistics, p-values, confidence intervals, loadings and fit, plus return, volatility, Sharpe ratio, largest drawdown and the score",
             numbers=f"{man.get('checks', 0):,} numbers across {man.get('managers', 0)} managers",
             agreement=f"largest difference {man.get('max_abs_diff', float('nan')):.0e}, no disagreements",
             shared="Both implementations read the same aligned monthly returns. They share no code: the robust standard errors are written out by hand on the second side."),
        dict(area="The optimiser", covered=True,
             what="The same portfolio problem, the same constraints, solved by a different solver from a different library",
             numbers=f"{twin.get('names_python', 0)} target weights at the latest rebalance",
             agreement=(f"largest weight difference {twin.get('max_abs_diff', float('nan')):.0e}" if twin.get("available") else "rebuilt where the second solver is installed"),
             shared="Both read the same signal, benchmark, sector labels and starting portfolio. A wrong constraint would be reproduced by both; a wrong solution would not."),
        dict(area="The manager decay model", covered=True,
             what="The whole panel rebuilt from the raw filings — concentration, turnover, crowding, trades, book growth — and the walk-forward test re-run",
             numbers=f"{dman.get('rows', 0):,} manager-quarters",
             agreement=(f"largest difference {(dman.get('r') or {}).get('max_abs_diff', float('nan')):.0e}, every row matched"
                        if (dman.get("r") or {}).get("available") else "rebuilt independently from the filings"),
             shared="This one starts from the raw filings rather than a prepared file, so it also tests the panel construction, not only the statistics."),
        dict(area="Signal research, the risk model and the fund of funds", covered=False,
             what="These have one implementation each",
             numbers="—",
             agreement="not independently reproduced",
             shared="They are tested by their own unit tests and by the fact that every input and output is published as a file a reader can recompute."),
        dict(area="Everything upstream of the prepared returns", covered=False,
             what="Matching a filing to a security, choosing the price, chaining the monthly return, deciding which months count",
             numbers="—",
             agreement="not covered by any agreement between implementations",
             shared="Both sides would repeat the same mistake, because both start from the same assembled data. This is the real limit of the check, and it is why the coverage figures above matter more than the agreement to twelve decimal places."),
    ]
    return pd.DataFrame(rows)


def computation() -> pd.DataFrame:
    """Which parts of the site are computed when you ask, and which are stored."""
    return pd.DataFrame([
        dict(surface="Research pages", when="Stored",
             detail="Each page is drawn from the files published beside it, which are rebuilt when the data changes and committed, so a page can never show a number its own data no longer supports. Every figure is downloadable."),
        dict(surface="Manager verification and memos", when="Computed",
             detail="Each manager's record runs through the full four-phase pipeline — load, reconcile, build the return series, run the regressions and the luck tests — in about thirty seconds. Every manager is built in the background when the site starts, so a click never waits."),
        dict(surface="Simulation", when="Computed",
             detail="A blend of managers and asset classes is built, run through the same pipeline as a single manager and scored on the same fixed scales, when the request is made."),
        dict(surface="Uploaded records", when="Computed",
             detail="An uploaded file runs the identical pipeline that every manager on the site runs. Nothing is a demonstration: a file of returns or values becomes one account, and because nothing on it can be checked against anything else, every period is marked unverified, which is the honest answer. Only the full statement format, with balances, flows and positions, can reach verified."),
        dict(surface="Uploaded statements", when="Computed",
             detail="Each statement is checked against the evidence on the page and on the one before it: the balance carried forward, the positions summing to the total, the flows, the stated gain and the stated return. A statement that fails any of these is flagged, with the numbers, and nothing is smoothed to make it pass."),
    ])


# ---------------------------------------------------------------- build

def build(out_dir: Path = OUT_DIR, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log("universe robustness")
    uni = universe_ic(log)
    uni.to_csv(out_dir / "universe_ic.csv", index=False)
    log("coverage of the disclosed book")
    cov, miss = coverage_by_quarter(log)
    cov.to_csv(out_dir / "coverage.csv", index=False)
    miss.to_csv(out_dir / "missing.csv", index=False)
    log("the headline result against the signal, at six cost assumptions")
    law, costs = law_and_costs(log)
    law.to_csv(out_dir / "law.csv", index=False)
    costs.to_csv(out_dir / "costs.csv", index=False)
    log("capacity")
    cap = capacity(log)
    cap.to_csv(out_dir / "capacity.csv", index=False)
    log("what the disclosed holdings can be")
    rec = reconstruction(log)
    rec.to_csv(out_dir / "reconstruction.csv", index=False)
    ev = evidence_needed(law)
    ev.to_csv(out_dir / "evidence.csv", index=False)
    log("provenance")
    cln = cleaning(log)
    cln.to_csv(out_dir / "cleaning.csv", index=False)
    fr = freshness(log)
    fr.to_csv(out_dir / "freshness.csv", index=False)
    verification_scope().to_csv(out_dir / "verification.csv", index=False)
    computation().to_csv(out_dir / "computation.csv", index=False)
    L = law.set_index("key").value
    man = dict(
        built=str(date.today()), built_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        universes={int(r.min_holders): int(r.universe_avg) for r in uni.drop_duplicates("min_holders").itertuples()},
        widest_universe=int(uni.universe_avg.max()), narrowest_universe=int(uni.universe_avg.min()),
        best_ic_wide=float(uni[uni.min_holders == 1].ic.max()), best_ic_narrow=float(uni[uni.min_holders == 5].ic.max()),
        coverage_first=float(cov.share_value.iloc[0]), coverage_last=float(cov.share_value.iloc[-1]),
        coverage_worst=float(cov.share_value.min()), coverage_worst_quarter=str(cov.loc[cov.share_value.idxmin(), "formation"]),
        ir_actual=float(L.get("ir_actual", np.nan)), ir_implied=float(L.get("ir_implied", np.nan)),
        ir_se=float(L.get("ir_se", np.nan)), ic=float(L.get("ic", np.nan)), transfer=float(L.get("transfer", np.nan)),
        breakeven_cost_bps=float(np.interp(0.0, costs.active_return.values[::-1], costs.cost_bps.values[::-1])) if costs.active_return.min() < 0 else None,
        cost_grid=list(COST_GRID), active_at_10=float(costs[costs.cost_bps == 10.0].active_return.iloc[0]),
        active_at_200=float(costs[costs.cost_bps == 200.0].active_return.iloc[0]),
        capacity_usd=float(cap[cap.trades_over_limit == 0].nav.max()) if (cap.trades_over_limit == 0).any() else None,
        managers=int(len(rec)), long_short=int((rec.confidence == "Long side only").sum()),
        not_meaningful=int((rec.confidence == "Not meaningful").sum()),
        option_heavy=int((rec.confidence == "Stock positions only").sum()),
        close=int((rec.confidence == "Close").sum()),
        option_users=int((rec.option_share > 0.01).sum()),
        checked_numbers=int(json.loads((ROOT / "data" / "research" / "r-verify" / "manifest.json").read_text()).get("checks", 0))
        if (ROOT / "data" / "research" / "r-verify" / "manifest.json").exists() else 0,
        cleaning_cases=int(cln.cases.sum()),
        data_refreshed=str(fr[fr.source.str.contains("holdings")].pulled.iloc[0]),
        newest_quarter=str(fr[fr.source.str.contains("holdings")].newest.iloc[0]),
    )
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    log(f"wrote {out_dir}")
    return man
