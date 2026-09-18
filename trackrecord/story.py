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
from .explain import CSS as EXPLAIN_CSS
from .research_pages import _signal_evidence

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data" / "research"

STORY_CSS = EXPLAIN_CSS + """<style>
/* Same 1100px column as Manager Analysis. The story used to cap itself at 920px
   with a 34px title, 22-character wrap, and tighter tiles — which made a page of
   numbers and charts feel scrunched next to the area page beside it. */
main.wrap.story{display:block;max-width:1100px;padding:36px 32px 64px}
main.wrap.story > * + *{margin-top:0}
.story .banner{margin:0}
.story-cover .cover-in{padding:44px 32px 40px;max-width:1100px}
.story-cover h1{font-size:42px;line-height:1.1;max-width:none;letter-spacing:-.005em}
.story-cover .sub{margin:16px 0 0;max-width:68ch;font-size:16px}
.story-cover .stats{display:flex;flex-wrap:wrap;gap:22px 40px;margin-top:28px;padding-top:18px;border-top:1px solid rgba(232,228,218,.18)}
.story-cover .stats div{display:grid;gap:4px;min-width:0}
.story-cover .stats dt{font:600 9px/1 var(--sans);letter-spacing:.22em;text-transform:uppercase;color:var(--cover-muted,var(--covermuted,#8a9ab4))}
.story-cover .stats dd{margin:0;font:400 30px/1 var(--serif);color:var(--cover-ink,var(--coverink,#e8e4da))}
.story-toc{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:28px 0 0}
.story-toc a{display:grid;gap:6px;padding:18px 20px;text-decoration:none;border:1px solid var(--gold-l,var(--goldl,#c9b48a));color:var(--cover-ink,var(--coverink,#e8e4da));background:rgba(232,228,218,.04);min-width:0}
.story-toc a:hover{filter:brightness(1.12);text-decoration:none}
.story-toc .toc-n{font:600 9px/1 var(--sans);letter-spacing:.2em;text-transform:uppercase;color:var(--gold-l,var(--goldl,#c9b48a))}
.story-toc .toc-t{font:400 22px/1.15 var(--serif);color:var(--cover-ink,var(--coverink,#e8e4da))}
.story-toc .toc-s{font-size:13.5px;line-height:1.45;color:var(--cover-muted,var(--covermuted,#8a9ab4))}
.part{margin:0;padding:44px 0;border-bottom:1px solid var(--line)}
.part:first-child{padding-top:8px}
.part:last-of-type{border-bottom:0;padding-bottom:16px}
.part .eyebrow{font:600 9.5px/1 var(--sans);letter-spacing:.2em;text-transform:uppercase;color:var(--gold);margin:0 0 10px}
.part h2{font:400 28px/1.2 var(--serif);color:var(--navy);margin:0 0 16px;padding:0;position:static}
.part h2::before{content:none !important}
.part h3{font:700 10px/1.3 var(--sans);margin:0 0 10px;color:var(--navy);text-transform:uppercase;letter-spacing:.14em}
.part .lede{font-size:16.5px;line-height:1.55;max-width:70ch;color:var(--ink);margin:0 0 14px}
.part .note{margin:12px 0 0;max-width:78ch}
.define{margin:4px 0 18px;padding:18px 22px;border-left:3px solid var(--gold);background:var(--surface);max-width:none}
.define .dt{font:600 9.5px/1 var(--sans);letter-spacing:.16em;text-transform:uppercase;color:var(--gold);margin:0 0 8px}
.define p{margin:0;font-size:15.5px;line-height:1.55;color:var(--ink)}
.step{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:22px 32px;align-items:start;margin:32px 0 8px}
.step:not(:has(.tiles)){grid-template-columns:1fr}
.step .step-copy,.step .tiles,.step > *{min-width:0}
.story .tiles{margin:16px 0 0;gap:14px;grid-template-columns:repeat(auto-fit,minmax(min(220px,100%),1fr))}
.story .step .tiles{margin:0;grid-template-columns:1fr;gap:14px}
.story .tile{padding:18px 20px;border-top:2px solid var(--gold)}
.picks{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(280px,100%),1fr));gap:14px;margin:16px 0 0}
.pick{background:var(--surface);border:1px solid var(--line);border-top:2px solid var(--gold);padding:16px 18px;min-width:0}
.pick .tk{font:400 20px/1.15 var(--serif);color:var(--navy)}
.pick .nm{font-size:13px;color:var(--ink2);margin:4px 0 12px}
.pick .row{display:flex;justify-content:space-between;gap:10px;font-size:13px;padding:5px 0;border-top:1px solid var(--line)}
.pick .row span:last-child{font-variant-numeric:tabular-nums;font-family:var(--sans)}
/* do not reuse dashboard .k (legend swatch) for these labels */
.when{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px 28px;margin:18px 0 0;padding:20px 22px;background:var(--surface);border:1px solid var(--line)}
.when > div{border-left:2px solid var(--gold);padding:2px 0 2px 14px;min-width:0}
.when .when-lab{display:block;font:600 9px/1.2 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
.when .when-val{display:block;font-size:14.5px;line-height:1.45;color:var(--navy);margin:0}
.mgrs{margin:18px 0 0;max-width:none}
.mgrs > summary{cursor:pointer;font-size:14px;line-height:1.4;color:var(--navy);list-style:none}
.mgrs > summary::-webkit-details-marker{display:none}
.mgrs > summary::before{content:"▸ ";color:var(--muted);font-size:12px}
.mgrs[open] > summary::before{content:"▾ "}
.mgrs .mgr-list{list-style:none;margin:12px 0 0;padding:14px 16px;display:grid;grid-template-columns:repeat(auto-fill,minmax(min(220px,100%),1fr));gap:8px 18px;max-height:min(320px,50vh);overflow:auto;border:1px solid var(--line);background:var(--surface);min-width:0}
.mgrs .mgr-list li{min-width:0}
.mgrs .mgr-list a{font-size:13.5px;color:var(--navy);text-decoration:none;line-height:1.3}
.mgrs .mgr-list a:hover{text-decoration:underline}
.mgrs .mgr-list .fund{display:block;font-size:11.5px;color:var(--ink2);margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mgrs .mgr-note{margin:10px 0 0;font-size:13px;color:var(--ink2);line-height:1.45;max-width:78ch}
.legend{display:flex;flex-wrap:wrap;gap:14px;font:12px var(--sans);color:var(--ink-2,#6b7078);margin:0 0 10px}
.legend .swatch{display:inline-block;width:10px;height:10px;margin-right:6px;vertical-align:middle;background:var(--navy)}
.legend .swatch.s0{background:#8a9ab4}.legend .swatch.s1{background:var(--navy)}.legend .swatch.s4{background:var(--crit,#8f3b34)}
.story .card{padding:22px 24px;margin-top:16px}
.story .charts{display:grid;grid-template-columns:1fr;gap:16px}
.story .chart .ser.s4{stroke:var(--crit,#8f3b34)}.story .chart .dot.s4{fill:var(--crit,#8f3b34)}
.story .cap{margin:10px 0 0}
.story .foot{margin-top:12px;padding-top:14px}
.reveal{opacity:0;transform:translateY(10px);transition:opacity .45s ease,transform .45s ease}
.reveal.in{opacity:1;transform:none}
@media (prefers-reduced-motion: reduce){.reveal,.reveal.in{opacity:1;transform:none;transition:none}}
@media(max-width:900px){
.step{grid-template-columns:1fr}
.step .tiles{grid-template-columns:repeat(auto-fit,minmax(min(220px,100%),1fr))}
}
@media(max-width:720px){
.story-cover .cover-in{padding:30px 16px 26px}
.story-cover h1{font-size:32px}
.story-cover .stats{gap:16px 28px;margin-top:22px}
.story-cover .stats dd{font-size:24px}
.story-toc{gap:8px}.story-toc a{padding:12px 10px;gap:4px}
.story-toc .toc-t{font-size:15px}.story-toc .toc-s{display:none}
main.wrap.story{padding:22px 16px 48px}
.part{padding:28px 0}
.part h2{font-size:24px}
.part .lede{font-size:15.5px}
.when{grid-template-columns:1fr;padding:16px 16px;gap:16px}
.picks{grid-template-columns:1fr}
.story .tile,.pick{padding:16px}
.define{padding:14px 16px}
.story .card{padding:18px 16px}
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


def _universe_managers() -> list[dict]:
    """Managers whose packed holdings define the Active universe and benchmark."""
    hold = ROOT / "data" / "reference" / "compact" / "holdings13f.csv.gz"
    lb_path = ROOT / "data" / "funds" / "leaderboard.csv"
    if not hold.exists():
        return []
    slugs = set(pd.read_csv(hold, usecols=["slug"])["slug"].astype(str).unique())
    if not lb_path.exists():
        return [dict(slug=s, who=s, fund="") for s in sorted(slugs)]
    lb = pd.read_csv(lb_path)
    rows = []
    for _, r in lb[lb.slug.isin(slugs)].iterrows():
        who = str(r.get("manager") or r.get("name") or r.slug).strip()
        fund = str(r.get("name") or "").strip()
        if fund.lower() == who.lower():
            fund = ""
        rows.append(dict(slug=str(r.slug), who=who, fund=fund))
    rows.sort(key=lambda d: d["who"].casefold())
    # any holdings slug missing from the leaderboard still appears
    seen = {d["slug"] for d in rows}
    for s in sorted(slugs - seen):
        rows.append(dict(slug=s, who=s, fund=""))
    return rows


def _managers_html(managers: list[dict]) -> str:
    if not managers:
        return ""
    items = []
    for m in managers:
        label = esc(m["who"])
        fund = f'<span class="fund">{esc(m["fund"])}</span>' if m.get("fund") else ""
        items.append(f'<li><a href="/f/{esc(m["slug"])}/">{label}</a>{fund}</li>')
    n = len(managers)
    return (
        f'<details class="mgrs" id="managers">'
        f'<summary>View the {n} managers whose holdings define this universe</summary>'
        f'<ul class="mgr-list">{"".join(items)}</ul>'
        f'<p class="mgr-note">A stock enters the buyable set when at least five of these managers '
        f'own it that quarter and it is priced at a dollar or more. Multi-strategy and market-maker '
        f'books are not in this list — their filings are trading inventory, not a portfolio.</p>'
        f'</details>'
    )


def _max_dd(r: pd.Series) -> float:
    w = (1 + r.fillna(0)).cumprod()
    return float((w / w.cummax() - 1).min()) if len(w) else float("nan")


def _capture(port: pd.Series, bench: pd.Series, *, down: bool) -> tuple[float, int]:
    """Mean portfolio return / mean benchmark return in months the benchmark is down (or up)."""
    mask = (bench < 0) if down else (bench > 0)
    n = int(mask.sum())
    if n < 5 or abs(float(bench[mask].mean())) < 1e-12:
        return float("nan"), n
    return float(port[mask].mean() / bench[mask].mean()), n


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
    # story is self-contained: no outbound research links and no named-module CTAs
    if ev_summary:
        ev_summary = re.sub(r'<a href="/research/[^"]+">([^<]+)</a>', r"\1", ev_summary)
        ev_summary = re.sub(r"\s+tested on Signal Research", " tested", ev_summary)
        ev_summary = re.sub(r"\s+on Signal Research", "", ev_summary)
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
    down_cap = up_cap = float("nan")
    n_down = n_up = 0
    port_dd = bench_dd = float("nan")
    if len(monthly) and {"portfolio", "benchmark"}.issubset(monthly.columns):
        pr, br = monthly["portfolio"], monthly["benchmark"]
        down_cap, n_down = _capture(pr, br, down=True)
        up_cap, n_up = _capture(pr, br, down=False)
        port_dd, bench_dd = _max_dd(pr), _max_dd(br)

    tiles3 = ""
    if P is not None:
        te = pct(P.tracking_error, 1, False)
        down_tv = f"{down_cap:.0%}" if down_cap == down_cap else "n/a"
        up_bit = f"; upside {up_cap:.0%}" if up_cap == up_cap else ""
        tiles3 = f"""<div class="tiles">
    <div class="tile"><div class="tl">Active return, constrained</div><div class="tv">{pct(P.active_return, 1)}</div><div class="td muted">per year vs the benchmark, after {cost_bps:.0f} bps costs</div></div>
    <div class="tile"><div class="tl">Information ratio</div><div class="tv">{num(P.information_ratio)}</div><div class="td muted">beat the benchmark in {P.hit_rate:.0%} of months</div></div>
    <div class="tile"><div class="tl">Downside capture</div><div class="tv">{down_tv}</div><div class="td muted">of the benchmark’s fall in down months{up_bit}</div></div>
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
    managers = _universe_managers()
    managers_block = _managers_html(managers)
    n_mgr = len(managers) or ""
    cover_bits = []
    if feats:
        cover_bits.append(f"<div><dt>Signals with evidence</dt><dd>{n_clear} of {len(feats)}</dd></div>")
    if P is not None:
        cover_bits.append(f"<div><dt>Active return</dt><dd>{pct(P.active_return, 1)}</dd></div>")
    if n_reb:
        cover_bits.append(f"<div><dt>Rebalances</dt><dd>{esc(n_reb)}</dd></div>")
    if n_mgr:
        cover_bits.append(f"<div><dt>Managers in the universe</dt><dd>{esc(n_mgr)}</dd></div>")
    cover_stats = f'<dl class="stats">{"".join(cover_bits)}</dl>' if cover_bits else ""

    body = f"<style>{CSS}</style>" + STORY_CSS + f"""
<div class="banner" role="note"><span class="bl">A backtest</span> One demonstration mandate on public data, date by date. Not a live book, not a forecast, not investment advice.</div>
<header class="cover story-cover"><div class="cover-in">
  <div class="eyebrow">Active · trade list from ranking scores</div>
  <div class="rule"></div>
  <h1>How ranking scores became this trade list</h1>
  <p class="sub">A signal is a score for each stock, built only from public facts known at the time, used to decide which names to overweight. This page is the historical backtest of that idea from {esc(first)} to {esc(last)}: pick the score that worked, respect a mandate’s limits, name the stocks, and show what the book would have done.</p>
  {cover_stats}
  <nav class="story-toc" aria-label="Sections">
    <a href="#objective"><span class="toc-n">01</span><span class="toc-t">Objective</span><span class="toc-s">What this backtest is for</span></a>
    <a href="#process"><span class="toc-n">02</span><span class="toc-t">Process</span><span class="toc-s">How the ranking became trades</span></a>
    <a href="#product"><span class="toc-n">03</span><span class="toc-t">Product</span><span class="toc-s">The list, and what it would have done</span></a>
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

  <div class="step">
    <div class="step-copy">
      <h3>1 · Research — which ranking rule survived</h3>
      <p class="lede">Eight candidate signals were built each month for about {labman.get('universe_avg', '')} stocks held by five or more of the managers and priced at a dollar or more: momentum, short-term reversal, low volatility, size, value, profitability, cash flow and earnings yield. Each signal is just that stock’s score on that rule, using only information public at the time. The test asks whether a higher score went with a higher return next month.</p>
      <p class="note">{ev_summary.strip() or "Twelve-month momentum is the characteristic the later steps trade."} A rank-correlation test does not clear the same bar even for momentum, and a learned model of all eight does not beat a simple average. So the book trades momentum alone.</p>
    </div>
    {tiles1}
  </div>

  <div class="step">
    <div class="step-copy">
      <h3>2 · Risk — how much the book might move</h3>
      <p class="lede">Each stock’s next month is explained by the market, those same eight characteristics, and its industry. What is left is stock-specific risk. Together they forecast portfolio volatility and are checked against what then happened.</p>
      <p class="note">This model is a little too confident ({float(bias.get('random', float('nan'))):.2f} vs 1.00) and explains about {float(riskman.get('r2_avg', 0)):.0%} of a typical month. Construction reports that forecast as tracking error; the hard caps are the position and sector bands below.</p>
    </div>
    {tiles2}
  </div>

  <div class="step">
    <div class="step-copy">
      <h3>3 · Construction — rules that turn the ranking into tickets</h3>
      <p class="lede">Mandate: ${nav_m:,.0f} million, long-only, fully invested. Name cap {c.get('max_weight', 0):.0%}; active band {c.get('active_band', 0):.0%} vs the benchmark; sector band {c.get('sector_band', 0):.0%}; active share ≤ {c.get('active_share', 0):.0%}; one-way turnover ≤ {c.get('turnover', 0):.0%} a quarter; {cost_bps:.0f} bps per dollar traded. The benchmark is everything the managers own that quarter, in dollars — not a published index.</p>
      <p class="note">The buyable stocks are taken from those managers’ disclosed holdings on purpose. The ranking, the universe and the benchmark then come from one public source, so the backtest asks a single question: among names these managers already own, does ranking by momentum and applying the mandate beat <i>their</i> dollar-weighted book? A published index such as the S&amp;P 500 would answer a different question — beat the index — and would mix two datasets. A real mandate often would use an index; this page does not, so nothing here is measured against a book the rest of the site never saw.</p>
    </div>
  </div>
  <div class="when">
    <div><span class="when-lab">When</span><span class="when-val">Every quarter-end, {n_reb} times from {esc(first)} to {esc(last)}. Latest list: {esc(form)}.</span></div>
    <div><span class="when-lab">Who can be bought</span><span class="when-val">{esc(n_univ) or "Hundreds of"} US stocks held by at least {conman.get('min_holders', 5)} of {n_mgr or 'these'} managers, priced at $1 or more.</span></div>
    <div><span class="when-lab">How they are ranked</span><span class="when-val">By the momentum signal: twelve-month return, skipping the most recent month, then standardized across the universe on that date so the typical stock is near zero and a strong recent winner is positive.</span></div>
    <div><span class="when-lab">What then happens</span><span class="when-val">The optimiser sets weights inside the bands. Names that left are sold. The book is held until the next quarter.</span></div>
  </div>
  {managers_block}
</article>

<article class="part reveal" id="product">
  <p class="eyebrow">Product</p>
  <h2>What that process produced</h2>

  <h3>The latest trade list — {esc(form)}</h3>
  <p class="lede">{n_tr} tickets ({n_buy} buys, {n_sell} sells), holding {n_held} names. The six largest buys are below. A high momentum score earns the place; the bands keep the size in check. This is the actual list from the backtest.</p>
  <div class="picks">{picks}</div>
  <p class="note">On {esc(form)} each eligible stock is scored by twelve-month momentum (skipping the most recent month). An optimiser sets weights inside the name cap, benchmark band, sector bands, active-share cap and turnover budget. These six cards are the largest buy tickets from that day’s close on a ${nav_m:,.0f} million demonstration book.</p>

  <h3>Growth of one dollar</h3>
  <div class="charts">
    <div class="card">
      <div class="legend">{legend}</div>
      {growth}
      <p class="cap">Constrained backtest, unconstrained top tenth, and the managers’ aggregate book. Log scale. {cost_bps:.0f} bps charged after each rebalance on both active lines. The constrained line is what a mandate could hold; the unconstrained line is what the bands cost.</p>
    </div>
    <div class="card">
      <h3 style="margin-top:0">Active return by year</h3>
      {act_bars}
      <p class="cap">Constrained book minus the benchmark, calendar year sums. Blue means the backtest won that year. A few percent a year can be one lucky year; the bars show whether the rule was persistent.</p>
    </div>
  </div>
  <p class="note">With no limits the top tenth delivered {u_ret}/yr of active return; inside the mandate, {pct(P.active_return, 1) if P is not None else 'n/a'} against a benchmark that returned {b_ret}/yr. Information ratio {num(P.information_ratio) if P is not None else 'n/a'} over {n_reb} quarters (standard error about 0.3) — a description of the process, not a claim of a live edge.</p>
  <p class="note">{(
      f"On the downside: in the {n_down} months the benchmark fell, the constrained book took about {down_cap:.0%} of that fall"
      f"{f' (and about {up_cap:.0%} of the rise in the {n_up} up months)' if up_cap == up_cap else ''}."
      f" Peak to trough it fell {abs(port_dd):.0%} against the benchmark’s {abs(bench_dd):.0%}."
      if down_cap == down_cap else
      "Downside capture is measured against the same managers’ aggregate book used for active return."
  )}</p>
  {tiles3}

  <div class="step">
    <div class="step-copy">
      <h3>Independent check</h3>
      <p class="lede">Every headline statistic on every manager was recalculated by a second implementation against the same aligned returns. The two share no code and agree to computer precision. That rules out quiet arithmetic errors; it does not make the next quarter look like the last {n_reb}.</p>
      <p class="note">What the check cannot reach: names bought or taken private never enter the universe; disclosed holdings are the long US book; the 45-day filing delay is baked into every date.</p>
    </div>
    {tiles4}
  </div>
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
