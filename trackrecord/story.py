"""The Active tab as one back-tested story rather than four separate tools.

Told in three parts: the objective, the process that pursued it, and the product
that process produced. Numbers come from the same committed CSVs as the four
research notes. It is a historical backtest — each step uses only what was known
at that date — not a live book and not a forecast.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .dashboard import CSS, JS, diverging_bars, esc, line_chart, num, pct
from .explain import CSS as EXPLAIN_CSS, figure_note
from .research_pages import _signal_evidence

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data" / "research"

STORY_CSS = EXPLAIN_CSS + """<style>
/* dashboard .wrap is a wide grid with large gaps; the story is a single column */
main.wrap.story{display:block;max-width:920px;padding-top:20px;padding-bottom:48px}
main.wrap.story > * + *{margin-top:28px}
.story .banner{margin:0}
.story-cover .cover-in{padding:28px 32px 24px;max-width:920px}
.story-cover h1{font-size:34px;line-height:1.15;max-width:22ch}
.story-cover .sub{margin:10px 0 0;max-width:62ch;font-size:15.5px}
.story-toc{display:flex;flex-wrap:wrap;gap:6px 16px;margin:16px 0 0;padding-top:14px;border-top:1px solid rgba(232,228,218,.18)}
.story-toc a{font:600 9.5px/1 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--goldl);text-decoration:none}
.story-toc a:hover{text-decoration:underline}
.part{margin:0;padding:0}
.part + .part{padding-top:8px;border-top:1px solid var(--line)}
.part .eyebrow{font:600 9.5px/1 var(--sans);letter-spacing:.2em;text-transform:uppercase;color:var(--gold);margin:0 0 8px}
.part h2{font:400 24px/1.2 var(--serif);color:var(--navy);margin:0 0 10px;padding:0;position:static}
.part h2::before{content:none !important}
.part h3{font:700 10px/1.3 var(--sans);margin:18px 0 8px;color:var(--navy);text-transform:uppercase;letter-spacing:.14em}
.part .lede{font-size:16px;line-height:1.5;max-width:68ch;color:var(--ink);margin:0 0 12px}
.part .note{margin:10px 0 0;max-width:68ch}
.define{margin:12px 0 0;padding:12px 14px;border-left:3px solid var(--gold);background:var(--surface);max-width:68ch}
.define .dt{font:600 9.5px/1 var(--sans);letter-spacing:.16em;text-transform:uppercase;color:var(--gold);margin:0 0 6px}
.define p{margin:0;font-size:15px;line-height:1.5;color:var(--ink)}
.story .tiles{margin:12px 0 0;gap:10px}
.story .tile{padding:12px 14px}
.picks{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(240px,100%),1fr));gap:10px;margin:10px 0 0}
.pick{background:var(--surface);border:1px solid var(--line);padding:12px 14px;min-width:0}
.pick .tk{font:600 16px/1.2 var(--sans);color:var(--navy)}
.pick .nm{font-size:12.5px;color:var(--ink2);margin:3px 0 8px}
.pick .row{display:flex;justify-content:space-between;gap:10px;font-size:12.5px;padding:2px 0;border-top:1px solid var(--line)}
.pick .row span:last-child{font-variant-numeric:tabular-nums;font-family:var(--sans)}
/* do not reuse dashboard .k (legend swatch) for these labels */
.when{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(180px,100%),1fr));gap:10px;margin:12px 0 0}
.when > div{border-left:2px solid var(--gold);padding:0 0 0 10px;min-width:0}
.when .when-lab{display:block;font:600 9px/1.2 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 0 6px}
.when .when-val{display:block;font-size:14px;line-height:1.4;color:var(--navy);margin:0}
.legend{display:flex;flex-wrap:wrap;gap:12px;font:12px var(--sans);color:var(--ink-2,#6b7078);margin:6px 0 4px}
.legend .swatch{display:inline-block;width:10px;height:10px;margin-right:6px;vertical-align:middle;background:var(--navy)}
.legend .swatch.s0{background:#8a9ab4}.legend .swatch.s1{background:var(--navy)}.legend .swatch.s4{background:var(--crit,#8f3b34)}
.story .card{padding:16px 18px;margin-top:12px}
.story .cap{margin:8px 0 0}
.story .foot{margin-top:28px;padding-top:12px}
.reveal{opacity:0;transform:translateY(10px);transition:opacity .45s ease,transform .45s ease}
.reveal.in{opacity:1;transform:none}
@media (prefers-reduced-motion: reduce){.reveal,.reveal.in{opacity:1;transform:none;transition:none}}
@media(max-width:720px){
.story-cover .cover-in{padding:22px 16px 20px}
.story-cover h1{font-size:28px}
main.wrap.story{padding-top:16px}
.part .lede{font-size:15.5px}
}
</style>"""

STORY_JS = """
(function(){
  var nodes = document.querySelectorAll(".reveal");
  if (!nodes.length) return;
  function show(n){ n.classList.add("in"); }
  if (!("IntersectionObserver" in window)) {
    nodes.forEach(show);
    return;
  }
  // above the fold should not wait for a scroll event
  nodes.forEach(function(n){
    var r = n.getBoundingClientRect();
    if (r.top < (window.innerHeight || 800) * 0.92) show(n);
  });
  var io = new IntersectionObserver(function(entries){
    entries.forEach(function(e){ if (e.isIntersecting) show(e.target); });
  }, { threshold: 0.08, rootMargin: "0px 0px -4% 0px" });
  nodes.forEach(function(n){ if (!n.classList.contains("in")) io.observe(n); });
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
    # story is self-contained: do not link out to the old Active research modules
    if ev_summary:
        ev_summary = re.sub(r'<a href="/research/[^"]+">([^<]+)</a>', r"\1", ev_summary)
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
        growth = line_chart(cells, series, height=280, width=860, y_fmt=lambda v: f"{v:.1f}×", y_log=True, end_labels=False, uid="storyg")
        legend = "".join(f'<span><span class="swatch {s["cls"]}"></span>{esc(s["name"])}</span>' for s in series)
        act = (monthly.portfolio - monthly.benchmark).dropna()
        ay = act.groupby(act.index.year).sum()
        act_bars = diverging_bars([str(i) for i in ay.index], [float(v) for v in ay.values], height=170, width=860,
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
    <div class="tile"><div class="tl">Signals with any evidence</div><div class="tv">{n_clear} of {len(feats)}</div><div class="td muted">top tenth minus bottom tenth, t-statistic of 2</div></div>
    <div class="tile"><div class="tl">Best of the eight</div><div class="tv">{pct(mom.spread_ann, 1)}</div><div class="td muted">twelve-month momentum a year; t = {mom.spread_t:+.1f}</div></div>
  </div>"""
    tiles2 = ""
    if bias:
        tiles2 = f"""<div class="tiles">
    <div class="tile"><div class="tl">Bias statistic, random portfolios</div><div class="tv">{float(bias.get('random', float('nan'))):.2f}</div><div class="td muted">1.00 means predicted risk matched what then happened</div></div>
    <div class="tile"><div class="tl">Explanatory power</div><div class="tv">{float(riskman.get('r2_avg', 0)):.0%}</div><div class="td muted">share of a month's stock returns the factors explain</div></div>
  </div>"""
    tiles3 = ""
    if P is not None:
        te = pct(P.tracking_error, 1, False)
        tiles3 = f"""<div class="tiles">
    <div class="tile"><div class="tl">Active return, constrained</div><div class="tv">{pct(P.active_return, 1)}</div><div class="td muted">per year vs the benchmark, after {cost_bps:.0f} bps costs</div></div>
    <div class="tile"><div class="tl">Information ratio</div><div class="tv">{num(P.information_ratio)}</div><div class="td muted">beat the benchmark in {P.hit_rate:.0%} of months</div></div>
    <div class="tile"><div class="tl">Tracking error</div><div class="tv">{te}</div><div class="td muted">realized; model predicted {pct(P.avg_ex_ante_te, 1, False)}</div></div>
  </div>"""
    tiles4 = ""
    if rvman.get("managers"):
        tiles4 = f"""<div class="tiles">
    <div class="tile"><div class="tl">Managers agreeing</div><div class="tv">{rvman.get('managers')}</div><div class="td muted">every headline number recomputed a second time</div></div>
    <div class="tile"><div class="tl">Numbers compared</div><div class="tv">{rvman.get('checks', 0):,}</div><div class="td muted">alphas, t-statistics, risk figures and the score</div></div>
    <div class="tile"><div class="tl">Largest difference</div><div class="tv">{rvman.get('max_abs_diff', 0):.0e}</div><div class="td muted">rounding noise, not a method discrepancy</div></div>
  </div>"""

    u_ret = pct(U.active_return, 1) if U is not None else "n/a"
    b_ret = pct(B.ann_return, 1, False) if B is not None else "n/a"
    n_reb = conman.get("rebalances", "")
    n_tr = L.get("trades", "")
    n_buy = L.get("buys", "")
    n_sell = L.get("sells", "")
    nav_m = conman.get("nav", 0) / 1e6

    body = f"<style>{CSS}</style>" + STORY_CSS + f"""
<div class="banner" role="note"><span class="bl">A backtest</span> One demonstration mandate on public data, date by date. Not a live book, not a forecast, not investment advice.</div>
<header class="cover story-cover"><div class="cover-in">
  <div class="eyebrow">Active · trade list from ranking scores</div>
  <div class="rule"></div>
  <h1>How ranking scores became this trade list</h1>
  <p class="sub">A signal is a score for each stock, built only from public facts known at the time, used to decide which names to overweight. This page is the historical backtest of that idea from {esc(first)} to {esc(last)}: pick the score that worked, respect a mandate’s limits, name the stocks, and show what the book would have done.</p>
  <nav class="story-toc" aria-label="Sections">
    <a href="#objective">Objective</a>
    <a href="#process">Process</a>
    <a href="#product">Product</a>
  </nav>
</div></header>
<main class="wrap story">

<article class="part reveal in" id="objective">
  <p class="eyebrow">Objective</p>
  <h2>Turn a public stock score into a tradable book — and say what it would have done</h2>
  <div class="define">
    <p class="dt">What “signal” means here</p>
    <p>A <b>signal</b> is one number per stock, computed the same way every month from information that was already public — prices, company accounts, industry — and then used to <b>rank</b> the universe. Higher score means “prefer this name”; lower means “prefer less of it.” It is not a tip, not a forecast from a person, and not a buy list on its own. It is a ranking rule. Example: <b>twelve-month momentum</b> scores each stock by how much its price rose over the past year (skipping the most recent month). Value would score cheapness of book value versus price; profitability would score return on assets. Eight such rules were tested; only momentum cleared the evidence bar, so that is the signal this page trades.</p>
  </div>
  <p class="lede">The job is to pick a ranking rule that actually predicted next month’s return, measure the risk of a book built on it, turn those ranks into weights inside a real mandate, and check the arithmetic. The site cannot honestly simulate the next two weeks without a live book and an unread future. What it can show is the same process run through history: rebuild the ranking each quarter from public prices, form the trades, then apply the returns that followed.</p>
  <p class="note">That is a backtest. The findings below, including the weak ones, are what it produced.</p>
</article>

<article class="part reveal" id="process">
  <p class="eyebrow">Process</p>
  <h2>How that objective was pursued</h2>

  <h3>1 · Research — which ranking rule survived</h3>
  <p class="lede">Eight candidate signals were built each month for about {labman.get('universe_avg', '')} stocks held by five or more of the managers and priced at a dollar or more: momentum, short-term reversal, low volatility, size, value, profitability, cash flow and earnings yield. Each signal is just that stock’s score on that rule, using only information public at the time. The test asks whether a higher score went with a higher return next month.</p>
  {tiles1}
  <p class="note">{ev_summary.strip() or "Twelve-month momentum is the characteristic the later steps trade."} A rank-correlation test does not clear the same bar even for momentum, and a learned model of all eight does not beat a simple average. So the book trades momentum alone.</p>

  <h3>2 · Risk — how much the book might move</h3>
  <p class="lede">Each stock’s next month is explained by the market, those same eight characteristics, and its industry. What is left is stock-specific risk. Together they forecast portfolio volatility and are checked against what then happened.</p>
  {tiles2}
  <p class="note">This model is a little too confident ({float(bias.get('random', float('nan'))):.2f} vs 1.00) and explains about {float(riskman.get('r2_avg', 0)):.0%} of a typical month. Construction reports that forecast as tracking error; the hard caps are the position and sector bands below.</p>

  <h3>3 · Construction — rules that turn the ranking into tickets</h3>
  <p class="lede">Mandate: ${nav_m:,.0f} million, long-only, fully invested. Name cap {c.get('max_weight', 0):.0%}; active band {c.get('active_band', 0):.0%} vs the benchmark; sector band {c.get('sector_band', 0):.0%}; active share ≤ {c.get('active_share', 0):.0%}; one-way turnover ≤ {c.get('turnover', 0):.0%} a quarter; {cost_bps:.0f} bps per dollar traded. The benchmark is everything the managers own that quarter, in dollars — not a published index.</p>
  <div class="when">
    <div><span class="when-lab">When</span><span class="when-val">Every quarter-end, {n_reb} times from {esc(first)} to {esc(last)}. Latest list: {esc(form)}.</span></div>
    <div><span class="when-lab">Who can be bought</span><span class="when-val">{esc(n_univ) or "Hundreds of"} US stocks held by at least {conman.get('min_holders', 5)} managers, priced at $1 or more.</span></div>
    <div><span class="when-lab">How they are ranked</span><span class="when-val">By the momentum signal: twelve-month return, skipping the most recent month, then standardized across the universe on that date so the typical stock is near zero and a strong recent winner is positive.</span></div>
    <div><span class="when-lab">What then happens</span><span class="when-val">The optimiser sets weights inside the bands. Names that left are sold. The book is held until the next quarter.</span></div>
  </div>
</article>

<article class="part reveal" id="product">
  <p class="eyebrow">Product</p>
  <h2>What that process produced</h2>

  <h3>The latest trade list — {esc(form)}</h3>
  <p class="lede">{n_tr} tickets ({n_buy} buys, {n_sell} sells), holding {n_held} names. The six largest buys are below. A high momentum score earns the place; the bands keep the size in check. This is the actual list from the backtest.</p>
  <div class="picks">{picks}</div>
  {figure_note(
      how=f"On {esc(form)} each eligible stock is scored by its return from twelve months earlier to one month earlier. An optimiser maximises the weighted score inside the name cap, benchmark band, sector bands, active-share cap and turnover budget. These six cards are the largest buy tickets, with the momentum score that earned them a place. Share counts use that day’s close and a ${nav_m:,.0f} million demonstration book.",
      where="Universe and benchmark from public quarterly holdings. Prices include dividends. Construction rebuilds momentum itself; it does not read Signal Research’s ranking file.",
      why="A backtest that never names the stocks is only a return series. These cards are the process on the latest date.")}

  <h3>Growth of one dollar</h3>
  <div class="card">
    <div class="legend">{legend}</div>
    {growth}
    <p class="cap">Constrained backtest, unconstrained top tenth, and the managers’ aggregate book. Log scale. {cost_bps:.0f} bps charged after each rebalance on both active lines.</p>
    {figure_note(
        how="Each line is one dollar compounded by that book’s monthly return. At every quarter-end the constrained book is the optimiser’s solution from that date; between rebalances nothing is traded. The following quarter’s returns — unseen by the optimiser — are then applied.",
        where=f"Monthly prices from {esc(first)} to {esc(last)}, the same holdings universe, and the constraints above.",
        why="The constrained line is what a mandate could hold. The unconstrained line is what the bands cost.")}
  </div>
  <div class="card">
    <h3 style="margin-top:0">Active return by year</h3>
    {act_bars}
    <p class="cap">Constrained book minus the benchmark, calendar year sums. Blue means the backtest won that year.</p>
    {figure_note(
        how="Each year sums the monthly gaps between the constrained portfolio and the benchmark. No factors are removed.",
        where="The same monthly backtest series as the growth chart.",
        why="A few percent a year can be one lucky year. The bars show whether the rule was persistent.")}
  </div>
  <p class="note">With no limits the top tenth delivered {u_ret}/yr of active return; inside the mandate, {pct(P.active_return, 1) if P is not None else 'n/a'} against a benchmark that returned {b_ret}/yr. Information ratio {num(P.information_ratio) if P is not None else 'n/a'} over {n_reb} quarters (standard error about 0.3) — a description of the process, not a claim of a live edge.</p>
  {tiles3}

  <h3>Independent check</h3>
  <p class="lede">Every headline statistic on every manager was recalculated by a second implementation against the same aligned returns. The two share no code and agree to computer precision. That rules out quiet arithmetic errors; it does not make the next quarter look like the last {n_reb}.</p>
  {tiles4}
  <p class="note">What the check cannot reach: names bought or taken private never enter the universe; disclosed holdings are the long US book; the 45-day filing delay is baked into every date.</p>
</article>

<div class="foot">
  <div class="running"><span>Track record verification · active portfolios</span><span>{esc(first)} – {esc(last)}</span></div>
  <h4>Important information</h4>
  <p>Built from public SEC EDGAR holdings filings and Yahoo Finance prices and sectors, with factors from the Kenneth R. French Data Library. Demonstration of method on public data. Past performance is not indicative of future results. Not investment advice.</p>
</div>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}{STORY_JS}</script>
"""
    return page("How ranking scores became this trade list", body, current="/active")
