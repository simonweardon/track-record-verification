"""The Active tab as one back-tested story rather than four separate tools.

The numbers are the same committed CSVs the four research notes already publish.
This page is the reading order: what was tested, how risk was measured, which
stocks were chosen and when, and the independent check. It is a historical
backtest — each step uses only what was known at that date — not a live book
and not a forecast.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dashboard import CSS, JS, diverging_bars, esc, line_chart, num, pct
from .explain import CSS as EXPLAIN_CSS, figure_note, plain
from .research_pages import _signal_evidence

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data" / "research"

STORY_CSS = EXPLAIN_CSS + """<style>
.story-toc{display:flex;flex-wrap:wrap;gap:8px 18px;margin:22px 0 0}
.story-toc a{font:600 9.5px/1 var(--sans);letter-spacing:.16em;text-transform:uppercase;color:var(--goldl);text-decoration:none;border-bottom:1px solid transparent;padding-bottom:3px}
.story-toc a:hover{border-bottom-color:var(--goldl)}
.chap{margin:48px 0 8px;padding-top:8px}
.chap .step{font:600 9.5px/1 var(--sans);letter-spacing:.22em;text-transform:uppercase;color:var(--gold);margin:0 0 8px}
.chap h2{margin-top:0;padding-top:0}
.chap h2::before{content:none}
.lede{font-size:17px;line-height:1.55;max-width:68ch;color:var(--ink);margin:0 0 16px}
.reveal{opacity:0;transform:translateY(18px);transition:opacity .7s ease, transform .7s ease}
.reveal.in{opacity:1;transform:none}
@media (prefers-reduced-motion: reduce){.reveal,.reveal.in{opacity:1;transform:none;transition:none}}
.story-rail{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0 6px}
.story-rail a{font:600 10px/1 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--navy);border:1px solid var(--line);padding:8px 12px;text-decoration:none}
.story-rail a:hover{border-color:var(--gold)}
.picks{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(280px,100%),1fr));gap:14px;margin:12px 0}
.pick{background:var(--surface);border:1px solid var(--line);padding:14px 16px;min-width:0}
.pick .tk{font:600 18px/1.2 var(--sans);color:var(--navy)}
.pick .nm{font-size:13px;color:var(--ink2);margin:4px 0 10px}
.pick .row{display:flex;justify-content:space-between;gap:12px;font-size:13px;padding:3px 0;border-top:1px solid var(--line)}
.pick .row span:last-child{font-variant-numeric:tabular-nums;font-family:var(--sans)}
.when{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(200px,100%),1fr));gap:12px;margin:14px 0 4px}
.when div{border-left:2px solid var(--gold);padding:4px 0 4px 12px}
.when .k{font:600 9px/1 var(--sans);letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
.when .v{font-size:15px;color:var(--navy)}
.legend{display:flex;flex-wrap:wrap;gap:14px;font:12px var(--sans);color:var(--ink-2,#6b7078);margin:8px 0 4px}
.k{display:inline-block;width:10px;height:10px;margin-right:6px;vertical-align:middle;background:var(--navy)}
.k.s0{background:#8a9ab4}.k.s1{background:var(--navy)}.k.s4{background:var(--crit,#8f3b34)}
@media(max-width:720px){.lede{font-size:16px}.chap{margin-top:36px}}
</style>"""

STORY_JS = """
(function(){
  var nodes = document.querySelectorAll(".reveal");
  if (!nodes.length) return;
  if (!("IntersectionObserver" in window)) {
    nodes.forEach(function(n){ n.classList.add("in"); });
    return;
  }
  var io = new IntersectionObserver(function(entries){
    entries.forEach(function(e){ if (e.isIntersecting) e.target.classList.add("in"); });
  }, { threshold: 0.12, rootMargin: "0px 0px -8% 0px" });
  nodes.forEach(function(n){ io.observe(n); });
})();
"""


def _man(rel: str) -> dict:
    f = R / rel
    try:
        return json.loads(f.read_text()) if f.exists() else {}
    except Exception:
        return {}


def _csv(rel: str, **kw) -> pd.DataFrame:
    f = R / rel
    return pd.read_csv(f, **kw) if f.exists() else pd.DataFrame()


def active_html(page) -> str:
    labman = _man("alpha-lab/manifest.json")
    riskman = _man("risk-model/manifest.json")
    conman = _man("construction/manifest.json")
    rvman = _man("r-verify/manifest.json")
    lab = _csv("alpha-lab/summary.csv")
    con = _csv("construction/summary.csv")
    rb = _csv("construction/rebalances.csv")
    tl = _csv("construction/trade_list.csv")
    monthly = _csv("construction/backtest_monthly.csv", index_col=0, parse_dates=True)
    if "signal" in lab.columns:
        lab = lab.set_index("signal")
    if "key" in con.columns:
        con = con.set_index("key")

    P = con.loc["portfolio"] if "portfolio" in con.index else None
    U = con.loc["unconstrained"] if "unconstrained" in con.index else None
    B = con.loc["benchmark"] if "benchmark" in con.index else None
    mom = lab.loc["momentum"] if "momentum" in lab.index else None
    feats = [s for s in lab.index if s not in ("linear", "xgboost")] if len(lab) else []
    n_clear = int((lab.loc[feats].spread_t >= 2).sum()) if feats and "spread_t" in lab.columns else 0
    ev_summary, _ = _signal_evidence()
    c = conman.get("constraints", {})
    L = conman.get("latest", {})
    bias = riskman.get("bias", {})
    first = (conman.get("first") or labman.get("first") or "")[:7]
    last = (conman.get("last") or labman.get("last") or "")[:7]
    form = str(L.get("formation") or last)

    growth = legend = act_bars = ""
    if len(monthly) and {"benchmark", "portfolio"}.issubset(monthly.columns):
        cols = [k for k in ("benchmark", "portfolio", "unconstrained") if k in monthly.columns]
        Rg = monthly[cols].dropna(how="all")
        cells = [x.strftime("%Y-%m") for x in Rg.index]
        series = []
        names = [("benchmark", "Benchmark — everything the managers own", "s0"),
                 ("portfolio", "Constrained portfolio (the backtest)", "s1"),
                 ("unconstrained", "Top tenth, no limits", "s4")]
        for k, name, cls in names:
            if k not in Rg:
                continue
            g = (1 + Rg[k].fillna(0)).cumprod()
            series.append(dict(name=name, values=[float(v) for v in g], cls=cls, emph=(k == "portfolio")))
        growth = line_chart(cells, series, height=300, width=860, y_fmt=lambda v: f"{v:.1f}×", y_log=True, end_labels=False, uid="storyg")
        legend = "".join(f'<span><span class="k {s["cls"]}"></span>{esc(s["name"])}</span>' for s in series)
        act = (monthly.portfolio - monthly.benchmark).dropna()
        ay = act.groupby(act.index.year).sum()
        act_bars = diverging_bars([str(i) for i in ay.index], [float(v) for v in ay.values], height=180, width=860,
                                  tips=[f"{i}: portfolio minus benchmark {v * 100:+.1f} pp" for i, v in ay.items()])

    picks = ""
    if len(tl):
        buys = tl[tl.side == "BUY"].head(6) if "side" in tl.columns else tl.head(6)
        cards = []
        for _, r in buys.iterrows():
            z = r.get("alpha_z", np.nan)
            ztxt = "—" if pd.isna(z) else f"{float(z):+.2f}"
            cards.append(
                f"<div class='pick'><div class='tk'>{esc(str(r.get('ticker', '')))}</div>"
                f"<div class='nm'>{esc(str(r.get('name', ''))[:42])}</div>"
                f"<div class='row'><span>Sector</span><span>{esc(str(r.get('sector', '')))}</span></div>"
                f"<div class='row'><span>Momentum score</span><span>{ztxt}</span></div>"
                f"<div class='row'><span>Target weight</span><span>{float(r.get('target_wt', 0)):.2%}</span></div>"
                f"<div class='row'><span>Trade</span><span>${float(r.get('trade_usd', 0)) / 1e6:+.2f}m</span></div></div>"
            )
        picks = "".join(cards)

    n_univ = ""
    if len(rb) and "universe" in rb.columns:
        n_univ = f"{int(rb.universe.iloc[-1]):,}" if len(rb) else ""
    n_held = L.get("names") or (int(P.avg_names) if P is not None else "")
    cost_bps = c.get("cost_bps", 10)

    tiles1 = ""
    if mom is not None:
        tiles1 = f"""<div class="tiles">
    <div class="tile"><div class="tl">Signals with any evidence</div><div class="tv">{n_clear} of {len(feats)}</div><div class="td muted">counted on the spread between the top and bottom tenth, at a t-statistic of 2</div></div>
    <div class="tile"><div class="tl">Best of the eight</div><div class="tv">{pct(mom.spread_ann, 1)}</div><div class="td muted">twelve-month momentum, top tenth minus bottom tenth a year; t = {mom.spread_t:+.1f}</div></div>
  </div>"""
    tiles2 = ""
    if bias:
        tiles2 = f"""<div class="tiles">
    <div class="tile"><div class="tl">Bias statistic, random portfolios</div><div class="tv">{float(bias.get('random', float('nan'))):.2f}</div><div class="td muted">1.00 would mean predicted risk matched what then happened</div></div>
    <div class="tile"><div class="tl">Explanatory power</div><div class="tv">{float(riskman.get('r2_avg', 0)):.0%}</div><div class="td muted">the average share of a month's stock returns the factors explain</div></div>
  </div>"""
    tiles3 = ""
    if P is not None:
        te = pct(P.tracking_error, 1, False)
        tiles3 = f"""<div class="tiles">
    <div class="tile"><div class="tl">Active return, constrained</div><div class="tv">{pct(P.active_return, 1)}</div><div class="td muted">per year against the benchmark, after {cost_bps:.0f} bps of trading costs</div></div>
    <div class="tile"><div class="tl">Information ratio</div><div class="tv">{num(P.information_ratio)}</div><div class="td muted">active return divided by tracking error; beat the benchmark in {P.hit_rate:.0%} of months</div></div>
    <div class="tile"><div class="tl">Tracking error</div><div class="tv">{te}</div><div class="td muted">realized; the model predicted {pct(P.avg_ex_ante_te, 1, False)} on average</div></div>
  </div>"""
    tiles4 = ""
    if rvman.get("managers"):
        tiles4 = f"""<div class="tiles">
    <div class="tile"><div class="tl">Managers agreeing</div><div class="tv">{rvman.get('managers')}</div><div class="td muted">every headline number recomputed by a second implementation</div></div>
    <div class="tile"><div class="tl">Numbers compared</div><div class="tv">{rvman.get('checks', 0):,}</div><div class="td muted">alphas, t-statistics, risk figures and the score, manager by manager</div></div>
    <div class="tile"><div class="tl">Largest difference</div><div class="tv">{rvman.get('max_abs_diff', 0):.0e}</div><div class="td muted">rounding noise, not a discrepancy of method</div></div>
  </div>"""

    u_ret = pct(U.active_return, 1) if U is not None else "n/a"
    b_ret = pct(B.ann_return, 1, False) if B is not None else "n/a"
    n_reb = conman.get("rebalances", "")
    n_tr = L.get("trades", "")
    n_buy = L.get("buys", "")
    n_sell = L.get("sells", "")

    body = f"<style>{CSS}</style>" + STORY_CSS + f"""
<div class="banner" role="note"><span class="bl">A backtest</span> This page walks through one demonstration mandate on public data, date by date, using only what was known then. It is not a live book, not a forecast, and not investment advice.</div>
<header class="cover"><div class="cover-in">
  <div class="eyebrow">Active portfolios · one story</div>
  <div class="rule"></div>
  <h1>How a signal became a list of trades</h1>
  <p class="sub">Four steps, in the order a desk would actually run them. First, which stock characteristics predicted the next month. Second, how much risk a book of those stocks was taking. Third, which names were chosen, on which date, inside a real set of limits. Fourth, whether the arithmetic survives a second implementation. The portfolio you see compounded is what that process would have held from {esc(first)} to {esc(last)} — a historical backtest, not a simulation of the coming weeks.</p>
  <nav class="story-toc" aria-label="Chapters">
    <a href="#research">1 · The research</a>
    <a href="#risk">2 · The risk</a>
    <a href="#build">3 · The portfolio</a>
    <a href="#check">4 · The check</a>
  </nav>
</div></header>
<main class="wrap">

<section class="reveal">
  <p class="lede">A live simulation of the next two weeks would be a different object: it would need a book that can still be traded tomorrow, and a clock that has not already seen the outcome. What this site can show honestly is the historical version of the same process. At every quarter-end the signal is rebuilt from prices that were already public, the optimiser is run, and the following quarter's returns are applied. That is a backtest. The findings below, including the weak ones, are what that backtest produced.</p>
</section>

<article class="chap reveal" id="research">
  <p class="step">Step one</p>
  <h2 data-n="01">What was researched</h2>
  <p class="lede">The question is not “which stocks look interesting?” It is “which well-known characteristics of a stock predicted the next month’s return, using only information that was public at the time?” Eight candidates were built each month for about {labman.get('universe_avg', '')} names: momentum, short-term reversal, low volatility, size, value, profitability, cash flow and earnings yield. The universe is the stocks that five or more of the managers in the system already held and that still had a price of a dollar or more — a large, liquid set, which is where such characteristics are usually weakest.</p>
  {tiles1}
  <p class="note">{ev_summary.strip() or "Twelve-month momentum is the characteristic the later steps trade."} A second test, the month-by-month rank correlation, does not clear the same bar even for momentum. Learning from all eight at once, walk-forward, does not beat a simple average of them. So the later steps do not average the eight, and they do not use the learned model. They trade momentum, because that is the only column with evidence behind it.</p>
  <p class="story-rail"><a href="/research/alpha-lab">Open the full research note</a></p>
</article>

<article class="chap reveal" id="risk">
  <p class="step">Step two</p>
  <h2 data-n="02">How the risk was measured</h2>
  <p class="lede">Before any optimiser runs, each stock’s next month is explained by its exposure to the market, to those same eight characteristics, and to its industry. The leftovers are that stock’s specific risk. Together they become a forecast of how much a portfolio will move, and a calibration test that asks whether the forecast can be trusted.</p>
  {tiles2}
  <p class="note">A bias statistic near 1.00 means that when the model said a random fifty-stock book would move by a certain amount, it then did. This model, on this universe, is a little too confident ({float(bias.get('random', float('nan'))):.2f} against 1.00) and explains about {float(riskman.get('r2_avg', 0)):.0%} of a typical month’s spread of returns — inside the range commercial models publish, on a much smaller set of names. Construction uses that forecast as a reported tracking-error number, not as a hard cap: the hard caps are the position and sector bands a mandate can actually write down.</p>
  <p class="story-rail"><a href="/research/risk-model">Open the full risk note</a></p>
</article>

<article class="chap reveal" id="build">
  <p class="step">Step three</p>
  <h2 data-n="03">Which stocks were chosen, and when</h2>
  <p class="lede">The mandate is a demonstration: ${conman.get('nav', 0) / 1e6:,.0f} million, long-only, fully invested. Each name may be at most {c.get('max_weight', 0):.0%} of the book and may sit no more than {c.get('active_band', 0):.0%} away from the benchmark weight. Sectors may drift {c.get('sector_band', 0):.0%} from the benchmark. Active share is capped at {c.get('active_share', 0):.0%}, and one-way turnover at {c.get('turnover', 0):.0%} a quarter. Trading costs {cost_bps:.0f} basis points on every dollar traded. The benchmark is not a published index: it is everything the managers own, added together in dollars that quarter.</p>
  <div class="when">
    <div><div class="k">When</div><div class="v">Every quarter-end, {n_reb} times from {esc(first)} to {esc(last)}. The latest list is dated {esc(form)}.</div></div>
    <div><div class="k">Who can be bought</div><div class="v">{esc(n_univ) or "Hundreds of"} US stocks held by at least {conman.get('min_holders', 5)} managers, priced at $1 or more.</div></div>
    <div><div class="k">How they are ranked</div><div class="v">Twelve-month return, skipping the most recent month, standardized across that universe on that date.</div></div>
    <div><div class="k">What then happens</div><div class="v">The optimiser turns the ranking into weights inside the bands. Names that left the universe are sold. The book is held without trading until the next quarter.</div></div>
  </div>
  <p class="note" style="margin-top:16px">The latest rebalance produced {n_tr} tickets: {n_buy} buys and {n_sell} sells, holding {n_held} names. The six largest buys on {esc(form)} are below. A high momentum score is why a name is being added; the 2% active band and the 4% name cap are why it is not being added at an unconstrained size. This is the actual trade list from the backtest, not an illustration.</p>
  <div class="picks">{picks}</div>
  {figure_note(
      how=f"On {esc(form)} each eligible stock is scored by its return from twelve months earlier to one month earlier. An optimiser then maximises the weighted score subject to the name cap, the band around the benchmark, the sector bands, the active-share cap and the turnover budget. The six cards are the largest buy tickets that solution produced, with the momentum score (alpha z) that earned them a place. Share counts use the closing price on that date and a demonstration portfolio of ${conman.get('nav', 0) / 1e6:,.0f} million.",
      where="The universe and the benchmark come from the managers' public quarterly holdings. Prices include dividends. Sectors come from the public price source. Construction rebuilds momentum itself; it does not read Signal Research's ranking file.",
      why="A backtest that never names the stocks is a return series, not a process. These cards are the process on the latest date: which names, what score, what size, constrained by the mandate rather than by hindsight.")}

  <div class="card" style="margin-top:22px">
    <div class="legend">{legend}</div>
    {growth}
    <p class="cap">One dollar, grown by the monthly returns of the backtest, of the same signal with no limits, and of the managers' aggregate book. Log scale. Costs of {cost_bps:.0f} bps are charged after each rebalance on both active lines.</p>
    {figure_note(
        how="Each line is one dollar compounded by that book's monthly return. At every quarter-end the constrained book is the optimiser's solution from that date; between rebalances weights drift with prices and nothing is traded. The unconstrained line is the top tenth by momentum, equally weighted. The following quarter's returns — which the optimiser could not see — are then applied. That is what makes this a backtest rather than a fitted path.",
        where=f"Monthly prices from {esc(first)} to {esc(last)}, the same holdings universe as the research step, and the constraints listed above. Delisted companies drop out at their last price.",
        why="The question after 'which stocks' is 'what would that rule have done'. The constrained line is the honest answer for a mandate that has to live inside bands. The unconstrained line is what the bands cost.")}
  </div>
  <div class="card" style="margin-top:14px">
    <h3>Active return by year — constrained book minus the benchmark</h3>
    {act_bars}
    <p class="cap">Each bar is that calendar year's monthly gaps, added up. Blue means the backtest beat the managers' aggregate book that year.</p>
    {figure_note(
        how="For each calendar year the monthly difference between the constrained portfolio and the benchmark is summed. No factors are removed. A red year is a year the rule lagged the managers' own aggregate holdings.",
        where="The same monthly backtest series as the growth chart.",
        why="A yearly active return of a few percent can be one wonderful year. The bars show whether the rule was persistent, which is the only way a backtest is useful as a process description.")}
  </div>
  <p class="note">Read the constrained and unconstrained figures together. With no limits the top tenth delivered {u_ret} a year of active return; inside the mandate the same signal delivered {pct(P.active_return, 1) if P is not None else 'n/a'}, against a benchmark that itself returned {b_ret} a year. The gap is the price of being able to hold the book. The information ratio of {num(P.information_ratio) if P is not None else 'n/a'} has a standard error of about 0.3 over {n_reb} quarters, which is why this page does not call it evidence of a live edge — only a description of what the process did.</p>
  {tiles3}
  <p class="story-rail"><a href="/research/construction">Open the full construction note, including every ticket</a></p>
</article>

<article class="chap reveal" id="check">
  <p class="step">Step four</p>
  <h2 data-n="04">Whether the numbers survive a second look</h2>
  <p class="lede">A story about a backtest is only as good as the arithmetic underneath it. Every headline statistic on every manager — the factor alphas, their t-statistics, the yearly return, volatility, Sharpe ratio, largest fall and the score — was recalculated by a second implementation written from scratch against the same aligned returns. The two share no code. They agree to the limit of computer precision. That check does not make the backtest a good idea to trade tomorrow. It rules out the quiet kind of error: a wrong lag, a dropped month, an annualization off by one.</p>
  {tiles4}
  <p class="note">What the check cannot reach is the data itself: names that were bought or taken private never enter the universe, disclosed holdings are the long US book rather than the fund, and the 45-day filing delay is already baked into every date on this page. Those limits are measured, not asserted, on <a href="/research/limits">Due Diligence on This Work</a>.</p>
  <p class="story-rail"><a href="/research/r-verify">Open the full verification note</a></p>
</article>

<section class="reveal" style="margin-top:48px">
  <h2 data-n="End">What this is, and what it is not</h2>
  <p class="lede">It is a dated process: a universe from public holdings, a characteristic chosen because it was the only one with evidence, a risk model that reports how much the book might move, an optimiser that names the stocks and the day they were chosen, and a second implementation that repeats the arithmetic. It is a backtest of that process from {esc(first)} to {esc(last)}. It is not a recommendation, not a live portfolio, and not a claim that the next quarter will look like the last fifty-three.</p>
</section>

<div class="foot">
  <div class="running"><span>Track record verification · active portfolios</span><span>{esc(first)} – {esc(last)}</span></div>
  <h4>Important information</h4>
  <p>This page is built from public SEC EDGAR holdings filings and Yahoo Finance prices and sector labels, with factor data from the Kenneth R. French Data Library. It is a demonstration of method on public data. Past performance is not indicative of future results, and nothing here is investment advice.</p>
</div>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}{STORY_JS}</script>
"""
    return page("Active portfolios", body, current="/active")
