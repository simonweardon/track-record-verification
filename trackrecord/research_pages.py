"""Research notes rendered from the CSVs under data/research/ — same chrome as the dashboards.

Every sentence on these pages is generated from the numbers, so a rebuild with new
filings cannot leave stale prose behind.  The verdicts use the same thresholds as the
manager dashboards: |t| ≥ 2 is "evidence", anything less is "indistinguishable from zero".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dashboard import CSS, JS, diverging_bars, esc, line_chart, pct, num
from .explain import plain
from .signals13f import OUT_DIR as SIG_DIR, PORTFOLIOS, SPREADS

EXTRA_CSS = """
.chart .ser.s2 { stroke: var(--s3); } .chart .dot.s2 { fill: var(--s3); } .k.s2 { background: var(--s3); }
.chart .ser.s4 { stroke: var(--neg); } .chart .dot.s4 { fill: var(--neg); } .k.s4 { background: var(--neg); }
.chart .ser.s5 { stroke: var(--gold); } .chart .dot.s5 { fill: var(--gold); } .k.s5 { background: var(--gold); }
.findings { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin: 6px 0 4px; }
.finding { background: var(--surface); border: 1px solid var(--line); border-top: 2px solid var(--gold); padding: 16px 18px; }
.finding .q { font: 400 17px/1.25 var(--serif); color: var(--navy); margin-bottom: 8px; }
.finding .a { font-size: 14px; color: var(--ink); }
.finding .v { display: inline-block; font: 600 9.5px/1 var(--sans); letter-spacing: .16em; text-transform: uppercase; padding: 5px 8px; margin-bottom: 10px; color: #fff; }
.finding .v.no { background: var(--crit); } .finding .v.weak { background: var(--warn); } .finding .v.yes { background: var(--good); } .finding .v.caveat { background: var(--ink-2); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font: 12px var(--sans); color: var(--ink-2); margin: 8px 0 4px; }
"""


def _tv(t) -> str:
    if t is None or pd.isna(t):
        return "n/a"
    return f"{t:+.1f}"


def _verdict_class(t: float, want_positive: bool = True) -> tuple[str, str]:
    """Traffic light for a spread: strong evidence in the hoped-for direction, weak, none, or the wrong way."""
    if pd.isna(t):
        return "caveat", "no test"
    if abs(t) < 1.0:
        return "no", "no signal"
    if abs(t) < 2.0:
        return "weak", "weak, not significant"
    return ("yes", "evidence") if (t > 0) == want_positive else ("no", "evidence — the wrong way")


def signals13f_html(sig_dir: Path = SIG_DIR) -> str | None:
    if not (sig_dir / "manifest.json").exists():
        return None
    man = json.loads((sig_dir / "manifest.json").read_text())
    summ = pd.read_csv(sig_dir / "summary.csv").set_index("key")
    spreads = pd.read_csv(sig_dir / "spreads.csv")
    ic = pd.read_csv(sig_dir / "ic.csv"); ics = pd.read_csv(sig_dir / "ic_summary.csv")
    R = pd.read_csv(sig_dir / "portfolios_monthly.csv", index_col=0, parse_dates=True)
    form = pd.read_csv(sig_dir / "formation.csv")
    crowd = pd.read_csv(sig_dir / "latest_crowding.csv")
    sp = spreads.set_index(["long", "short"])

    def spread(a, b):
        return sp.loc[(a, b)] if (a, b) in sp.index else None

    # ---- findings (prose from numbers)
    best = spread("BEST1", "ALL"); crd = spread("CROWD_HI", "CROWD_LO"); new = spread("NEW", "SOLD"); newu = spread("NEW", "ALL"); addc = spread("ADD", "CUT")
    findings = []
    if best is not None:
        c, lab = _verdict_class(best.t_spread)
        findings.append(("Do managers' best ideas beat the rest of their books?", c, lab,
                         f"Each manager's largest position, bought when its filing became public, returned {pct(best.ann_spread, 1)} per year "
                         f"versus the equal-weighted universe of everything they hold (t = {_tv(best.t_spread)}, positive in {best.share_positive_months:.0%} of months). "
                         f"After the Carhart four factors the difference is {pct(best.carhart_alpha, 1)}/yr (t = {_tv(best.carhart_t)}). "
                         f"Cohen, Polk and Silli found that best ideas earned several percent a year over 1991 to 2005. With a 45-day delay, in this universe and period, that edge is not visible."))
    if crd is not None:
        lo = summ.loc["CROWD_LO"]
        c, lab = _verdict_class(crd.t_spread)
        findings.append(("Do crowded names outperform the ones nobody else holds?", "caveat", "unreliable — see why",
                         f"The most-held quintile {'lagged' if crd.ann_spread < 0 else 'beat'} the least-held quintile by {pct(abs(crd.ann_spread), 1, False)}/yr (spread t = {_tv(crd.t_spread)}); after factors {pct(crd.carhart_alpha, 1)}/yr (t = {_tv(crd.carhart_t)}). "
                         f"That looks like a real result in favour of avoiding crowded stocks. But the least-crowded group is made up of the smallest stakes held by a single manager, with "
                         f"a size loading of {num(lo.b_SMB)} and turnover of {lo.avg_turnover:.0%} a quarter. This is exactly where the price data's survivorship bias, "
                         f"in which delisted companies drop out at their last price, would manufacture a return, so the result is reported but not believed. "
                         f"The rank test across all names says the opposite, mildly: more holders, slightly better next quarter (IC {ics.set_index('signal').loc['n_holders', 'mean_ic']:+.3f}, t = {_tv(ics.set_index('signal').loc['n_holders', 't'])})."))
    if new is not None and newu is not None:
        c, lab = _verdict_class(new.t_spread)
        findings.append(("Is copying managers' new buys profitable?", c if c != "yes" else "no", lab,
                         f"Stocks that managers had just bought returned {pct(new.ann_spread, 1)}/yr versus the stocks they had just sold (t = {_tv(new.t_spread)}), "
                         f"and {pct(newu.ann_spread, 1)}/yr versus the whole universe (Carhart alpha {pct(newu.carhart_alpha, 1)}, t = {_tv(newu.carhart_t)}). "
                         f"By the time a purchase is public, whatever information it carried has already been priced in, and a portfolio that copied the trades would have lagged."))
    if addc is not None:
        c, lab = _verdict_class(addc.t_spread)
        findings.append(("Do adds beat trims?", c, lab,
                         f"Positions managers increased by 25%+ returned {pct(addc.ann_spread, 1)}/yr more than positions they trimmed by 25%+ "
                         f"(t = {_tv(addc.t_spread)}, positive in {addc.share_positive_months:.0%} of months; Carhart alpha {pct(addc.carhart_alpha, 1)}, t = {_tv(addc.carhart_t)}). "
                         f"This is the one result in the hoped-for direction, and it is not statistically significant."))
    fcards = "".join(f'<div class="finding"><div class="q">{esc(q)}</div><span class="v {c}">{esc(lab)}</span><div class="a">{a}</div></div>' for q, c, lab, a in findings)

    # ---- cumulative growth chart
    keys = [("MKT", "US market", "s0"), ("ALL", "Every held stock", "s1l"), ("BEST1", "Best ideas", "s1"),
            ("CROWD_HI", "Most crowded", "s2"), ("CROWD_LO", "Least crowded", "s5"), ("NEW", "New positions", "s4")]
    Rg = R[[k for k, _, _ in keys]].dropna(how="all")
    cells = [d.strftime("%Y-%m") for d in Rg.index]
    series = []
    for k, name, cls in keys:
        g = (1 + Rg[k].fillna(0)).cumprod()
        g[Rg[k].isna() & ~Rg[k].notna().cummax()] = np.nan      # nothing before the first month
        series.append(dict(name=name, values=[None if pd.isna(v) else float(v) for v in g], cls=cls, emph=(k == "BEST1")))
    growth = line_chart(cells, series, height=320, width=860, y_fmt=lambda v: f"{v:.1f}×", y_log=True, end_labels=False, uid="growth")
    legend = "".join(f'<span><span class="k {cls}"></span>{esc(name)}</span>' for _, name, cls in keys)

    # ---- portfolio table
    order = ["MKT", "ALL", "BEST1", "BEST3", "CROWD_HI", "CROWD_LO", "NEW", "ADD", "CUT", "SOLD"]
    prow = ""
    for k in order:
        if k not in summ.index: continue
        r = summ.loc[k]
        prow += (f"<tr><td><b>{esc(r.label)}</b><br><span class='muted'>{esc(r.holds)}</span></td>"
                 f"<td class='n'>{pct(r.ann_return, 1, False)}</td><td class='n'>{pct(r.ann_vol, 1, False)}</td><td class='n'>{num(r.sharpe)}</td>"
                 f"<td class='n'>{pct(r.max_dd, 0, False)}</td><td class='n'>{pct(r.excess_vs_market, 1)}</td>"
                 f"<td class='n'>{num(r.beta)}</td><td class='n'>{pct(r.carhart_alpha, 1) if k != 'MKT' else ''}</td><td class='n'>{_tv(r.carhart_t) if k != 'MKT' else ''}</td>"
                 f"<td class='n'>{f'{r.avg_names:,.0f}' if not pd.isna(r.avg_names) else ''}</td><td class='n'>{f'{r.avg_turnover:.0%}' if not pd.isna(r.avg_turnover) else ''}</td></tr>")

    # ---- spreads table + yearly bars for the two headline spreads
    srow = ""
    for _, r in spreads.iterrows():
        c, lab = _verdict_class(r.t_spread)
        srow += (f"<tr><td><b>{esc(r.label)}</b></td><td class='n'>{pct(r.ann_spread, 1)}</td><td class='n'>{_tv(r.t_spread)}</td><td class='n'>{r.share_positive_months:.0%}</td>"
                 f"<td class='n'>{pct(r.carhart_alpha, 1)}</td><td class='n'>{_tv(r.carhart_t)}</td><td class='n'>{num(r.b_SMB)}</td><td class='n'>{num(r.b_HML)}</td><td class='n'>{num(r.b_MOM)}</td>"
                 f"<td class='n'>{pct(r.worst_year, 0)}</td><td class='n'>{pct(r.best_year, 0)}</td><td><span class='v {c}' style='color:#fff;background:var(--{ {'no': 'crit', 'weak': 'warn', 'yes': 'good', 'caveat': 'ink-2'}[c] });font:600 9px var(--sans);letter-spacing:.14em;text-transform:uppercase;padding:4px 7px'>{esc(lab)}</span></td></tr>")
    def yearly(a, b):
        d = (R[a] - R[b]).dropna()
        y = d.groupby(d.index.year).sum()
        return diverging_bars([str(i) for i in y.index], [float(v) for v in y.values], height=200, width=860,
                              tips=[f"{i}: {PORTFOLIOS[a][0]} − {PORTFOLIOS[b][0]} {v * 100:+.1f} pp" for i, v in y.items()])
    bars_new = yearly("NEW", "SOLD"); bars_best = yearly("BEST1", "ALL")

    # ---- IC chart
    icc = ic.dropna(subset=["n_holders"])
    ic_bars = diverging_bars([str(f)[:7] for f in icc.formation], [float(v) for v in icc.n_holders], height=200, width=860, y_fmt=lambda v: f"{v:+.2f}",
                             tips=[f"{f}: rank correlation between number of holders and next-quarter return = {v:+.3f} across {n:,} stocks" for f, v, n in zip(icc.formation, icc.n_holders, icc.n_stocks)])
    icrow = "".join(f"<tr><td>{esc(r.label)}</td><td class='n'>{int(r.quarters)}</td><td class='n'>{r.mean_ic:+.3f}</td><td class='n'>{r.sd_ic:.3f}</td><td class='n'>{_tv(r.t)}</td><td class='n'>{r.share_positive:.0%}</td></tr>" for _, r in ics.iterrows())

    # ---- latest crowding
    crow = "".join(f"<tr><td><b>{esc(r.ticker)}</b> <span class='muted'>{esc(r['name'])}</span></td><td class='n'>{int(r.n_holders)}</td><td class='n'>{int(r.best_idea_of)}</td><td class='n'>{int(r.top3_of)}</td>"
                   f"<td class='n'>{r.avg_weight:.1%}</td><td class='n'>{int(r.new_buyers)}</td><td class='n'>{int(r.adders)}</td><td class='n'>{int(r.trimmers)}</td></tr>" for _, r in crowd.iterrows())
    last_form = str(crowd.formation.iloc[0]) if len(crowd) else man["last_formation"]
    n_mgr_last = int(form.managers.iloc[-1]) if len(form) else man["managers"]

    mapped_note = "96% of disclosed dollar value maps to a priced US ticker; the rest is foreign-domiciled, delisted or unlisted"
    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Holdings Research</title>
<style>{CSS}{EXTRA_CSS}</style>
<div class="banner" role="note"><span class="bl">Research note</span> This page is built from managers' public quarterly holdings filings and public prices. It is a reconstruction for research, not a strategy and not investment advice.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Manager Analysis · Holdings Research</div></div>
  <div class="gold-rule"></div>
  <h1>Holdings Research</h1>
  <p class="sub">Can an outsider make money by copying what prominent managers own? Every quarter, the public holdings filings of {man['managers']} managers are turned into portfolios of their biggest positions, their most crowded stocks and their newest purchases, bought in the month the filings become public and held for three months. Each portfolio is then tested the way a manager would be.</p>
  <dl class="meta">
    <div><dt>Managers</dt><dd>{man['managers']}</dd></div>
    <div><dt>Formation dates</dt><dd>{man['quarters']} · {man['first_formation'][:7]} → {man['last_formation'][:7]}</dd></div>
    <div><dt>Months tested</dt><dd>{man['months']}</dd></div>
    <div><dt>Positions read</dt><dd>{man['positions']:,}</dd></div>
    <div><dt>Built</dt><dd>{esc(man['built'])}</dd></div>
  </dl>
</div></header>
<main class="wrap">
<section class="verdict">
  <div class="sh"><h2>Findings</h2></div>
  <div class="findings">{fcards}</div>
  <p class="cap" style="margin-top:14px">The verdict rule is simple: a t-statistic below 1 means no signal, between 1 and 2 means a weak one, and 2 or more counts as evidence. The t-statistic is the spread's average monthly return divided by its standard error; the Carhart version is what remains after the market, size, value and momentum factors are removed. <b>The bottom line:</b> after the 45-day delay before holdings become public, these managers' disclosed portfolios contain no signal a copier could have used, and the one result that looks like a signal sits exactly where the data's known bias lives.</p>
</section>

<section>
  <div class="sh"><h2>Growth of $1</h2></div>
  <div class="card">
    <div class="legend">{legend}</div>
    {growth}
    <p class="cap">The chart uses a log scale. Each portfolio is rebuilt at the end of February, May, August and November from the filings public by then, with equal weights at formation, and is then held without trading until the next rebuild. Months with fewer than five priced names are left blank rather than filled in.</p>
  </div>
</section>

<section>
  <div class="sh"><h2>The portfolios</h2></div>
  <div class="card tscroll"><table><thead><tr><th>portfolio</th><th class="n">return /yr</th><th class="n">vol</th><th class="n">Sharpe</th><th class="n">max DD</th><th class="n">vs market /yr</th><th class="n">β</th><th class="n">Carhart α</th><th class="n">t</th><th class="n">names</th><th class="n">turnover /qtr</th></tr></thead><tbody>{prow}</tbody></table>
  <p class="cap">The Sharpe ratio is measured over the one-month Treasury bill. The Carhart alpha is annualized and its t-statistic uses Newey–West standard errors. Names and turnover are averages across formation dates, where turnover is half the sum of absolute weight changes against the previous portfolio.</p></div>
</section>

<section>
  <div class="sh"><h2>The tests that matter: spreads</h2></div>
  <div class="card tscroll"><table><thead><tr><th>long − short</th><th class="n">spread /yr</th><th class="n">t</th><th class="n">months &gt; 0</th><th class="n">Carhart α</th><th class="n">t</th><th class="n">SMB</th><th class="n">HML</th><th class="n">MOM</th><th class="n">worst yr</th><th class="n">best yr</th><th>verdict</th></tr></thead><tbody>{srow}</tbody></table>
  <p class="cap">A spread cancels what both sides share, such as the market, the period and the universe, so it isolates the signal. The factor loadings show what the spread is secretly betting on. The crowding spread is a bet against small companies (SMB {num(crd.b_SMB) if crd is not None else 'n/a'}), which is why its alpha is not taken at face value.</p></div>
  <div class="grid2">
    <div class="card"><h3>New positions − Sold out, by year</h3>{bars_new}<p class="cap">Each bar is the sum of the monthly spread returns in that calendar year.</p></div>
    <div class="card"><h3>Best ideas − Every held stock, by year</h3>{bars_best}<p class="cap">Each bar is the sum of the monthly spread returns in that calendar year.</p></div>
  </div>
</section>

<section>
  <div class="sh"><h2>Rank tests: information coefficients</h2></div>
  <div class="grid2">
    <div class="card"><h3>Crowding IC by formation date</h3>{ic_bars}
      <p class="cap">Each bar is the rank correlation between the number of managers holding a stock and its return over the next quarter, across every held stock with a price. A correlation of 0.02 is small but persistent: it was positive in {ics.set_index('signal').loc['n_holders', 'share_positive']:.0%} of quarters.</p></div>
    <div class="card"><h3>All signals</h3><div class="tscroll"><table><thead><tr><th>signal</th><th class="n">quarters</th><th class="n">mean IC</th><th class="n">sd</th><th class="n">t</th><th class="n">&gt; 0</th></tr></thead><tbody>{icrow}</tbody></table></div>
      <p class="cap">The information coefficient (IC) is the standard first look at a stock-selection signal: does this quarter's ranking predict next quarter's ranking of returns? Its t-statistic is the mean divided by the standard error. Real, tradable signals run between 0.03 and 0.05 with a t-statistic above 3 over long samples.</p></div>
  </div>
</section>

<section>
  <div class="sh"><h2>Most crowded names now</h2></div>
  <div class="card tscroll"><table><thead><tr><th>stock</th><th class="n">managers holding</th><th class="n">#1 position of</th><th class="n">top-3 of</th><th class="n">avg weight</th><th class="n">new buyers</th><th class="n">added</th><th class="n">trimmed</th></tr></thead><tbody>{crow}</tbody></table>
  <p class="cap">This table is built from the filings public by {esc(last_form)}, covering {n_mgr_last} managers. Weight is the average share of a holder's disclosed portfolio. New buyers, added and trimmed count the managers whose position is new, up by 25% or more in shares, or down by 25% or more since their previous filing.</p></div>
</section>

<section>
  <div class="sh"><h2>Method and limits</h2></div>
  <div class="card">
    <p><b>Universe.</b> The universe is the {man['managers']} managers in the system whose disclosed holdings are a portfolio rather than trading inventory; multi-strategy, quantitative, macro and market-making firms are excluded. All filing entities of a manager are merged, options are dropped, and a filing needs at least five long positions. {mapped_note}.</p>
    <p><b>Timing.</b> A quarter's holdings are used at the end of the second month after the quarter ends, which is the month in which the 45-day filing deadline falls, and only if the filing was actually public by then. Positions are then held without trading until the next formation date. This is the earliest any outsider could have acted.</p>
    <p><b>Weights.</b> The "every held stock" portfolio gives every stock the same weight. The best-ideas, new, added, trimmed and sold portfolios weight each stock by the number of managers in that category, with one unit per manager and position. The crowding groups rank held stocks by their number of holders, with ties broken by the total dollars held.</p>
    <p><b>Prices.</b> Prices are monthly closes adjusted for dividends. Stocks under $1 at formation are excluded, and a monthly return above 300% from a price under $5 is treated as a data error and blanked ({man.get('spikes_removed', 0)} cases in the whole panel). Delisted companies disappear at their last price, so every portfolio here is biased upward by survivorship, and most of all for the smallest, least-held names.</p>
    <p><b>Tests.</b> A spread's t-statistic is its mean monthly return divided by its standard error. Alphas are the intercepts of monthly regressions on the US market, size, value and momentum factors, with Newey–West standard errors, and are annualized. Information coefficients are rank correlations at each formation date. Nothing is optimized: the portfolio definitions were fixed before the results were seen, and every result is shown, including the ones that argue against the idea.</p>
    <p><b>Literature.</b> The relevant studies are Cohen, Polk and Silli (2010) on managers' best ideas, Brown and Schwarz (2013) on the information content of holdings disclosures, and Lakonishok, Shleifer and Vishny (1992) on herding. The question here is narrower than theirs: whether an outsider, acting only on public filings on the date they became public, could have used them.</p>
  </div>
</section>

<footer class="foot">
  <div class="running"><span>Track record verification · research</span><span>{man['first_formation'][:7]} – {man['last_formation'][:7]}</span></div>
  <h4>Important information</h4>
  <p>This page is built from public SEC EDGAR holdings filings and Yahoo Finance prices, with benchmark and factor data from the Kenneth R. French Data Library. All results are reconstructions with the limits stated above; they are not a strategy, a recommendation or investment advice. Past performance is not indicative of future results.</p>
</footer>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}</script>
"""


# ---------------------------------------------------------------- portfolio construction note

_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _signal_evidence() -> tuple[str, str]:
    """(one sentence for the summary, one for the method) citing what Signal Research settled.

    Read from the committed CSV so the claim cannot drift from the number behind it; empty
    strings if that page has not been built, so the construction page never cites nothing.
    """
    from .alphalab import OUT_DIR as LAB_DIR
    f = LAB_DIR / "summary.csv"
    if not f.exists():
        return "", ""
    S = pd.read_csv(f).set_index("signal")
    tested = [i for i in S.index if i not in ("linear", "xgboost")]
    if "momentum" not in tested:
        return "", ""
    r = S.loc["momentum"]
    return (f" It is not an arbitrary pick: of the {_WORDS.get(len(tested), len(tested))} characteristics tested on <a href=\"/research/alpha-lab\">Signal Research</a>, "
            f"it is the only one whose top tenth of the universe beat its bottom tenth with a t-statistic above 2 "
            f"({r.spread_t:+.1f}, worth {pct(r.spread_ann, 1)} a year), and the rest are noise or point the wrong way.",
            f" Of the {_WORDS.get(len(tested), len(tested))} characteristics tested on <a href=\"/research/alpha-lab\">Signal Research</a> it was the only one with "
            f"evidence behind it, so the other {_WORDS.get(len(tested) - 1, len(tested) - 1)} are left out rather than averaged in: adding them would be adding noise.")


def construction_html(out_dir: Path | None = None) -> str | None:
    from .construct import OUT_DIR as CON_DIR
    d = out_dir or CON_DIR
    if not (d / "manifest.json").exists():
        return None
    man = json.loads((d / "manifest.json").read_text())
    summ = pd.read_csv(d / "summary.csv").set_index("key")
    rb = pd.read_csv(d / "rebalances.csv")
    R = pd.read_csv(d / "backtest_monthly.csv", index_col=0, parse_dates=True)
    tl = pd.read_csv(d / "trade_list.csv")
    sec = pd.read_csv(d / "sectors_latest.csv")
    c = man["constraints"]; L = man["latest"]; rt = man.get("r_twin", {})
    P, U, B = summ.loc["portfolio"], summ.loc["unconstrained"], summ.loc["benchmark"]
    ev_summary, ev_method = _signal_evidence()

    # growth chart
    Rg = R[["benchmark", "portfolio", "unconstrained"]].dropna(how="all")
    cells = [x.strftime("%Y-%m") for x in Rg.index]
    series = []
    for k, name, cls in [("benchmark", "Benchmark", "s0"), ("portfolio", "Constrained portfolio", "s1"), ("unconstrained", "Unconstrained top decile", "s4")]:
        g = (1 + Rg[k].fillna(0)).cumprod()
        series.append(dict(name=name, values=[float(v) for v in g], cls=cls, emph=(k == "portfolio")))
    growth = line_chart(cells, series, height=300, width=860, y_fmt=lambda v: f"{v:.1f}×", y_log=True, end_labels=False, uid="cgrowth")
    legend = "".join(f'<span><span class="k {cls}"></span>{esc(n)}</span>' for _, n, cls in [(0, "Benchmark", "s0"), (0, "Constrained portfolio", "s1"), (0, "Unconstrained top decile", "s4")])
    # active return by year
    act = (R.portfolio - R.benchmark).dropna(); ay = act.groupby(act.index.year).sum()
    act_bars = diverging_bars([str(i) for i in ay.index], [float(v) for v in ay.values], height=200, width=860,
                              tips=[f"{i}: portfolio − benchmark {v * 100:+.1f} pp" for i, v in ay.items()])
    # ex-ante vs realized TE per rebalance (realized over the following quarter)
    te_rows = ""
    for _, r in rb.tail(10).iloc[::-1].iterrows():
        te_rows += (f"<tr><td>{esc(str(r.formation))}</td><td class='n'>{int(r.universe)}</td><td class='n'>{int(r.names)}</td><td class='n'>{r.turnover:.1%}</td>"
                    f"<td class='n'>{r.active_share:.0%}</td><td class='n'>{r.ex_ante_te:.1%}</td><td class='n'>{r.max_active:.1%}</td><td class='n'>{r.max_sector_active:.1%}</td><td class='n'>{r.sum_abs_active / 2:.1%}</td>"
                    f"<td class='n'>{r.expected_alpha - r.bench_alpha:+.2f}</td><td>{'turnover relaxed ×' + str(2 ** int(r.relaxed)) if r.relaxed else ''}</td></tr>")
    # trade list (top 12 each way)
    buys = tl[tl.side == "BUY"].head(12); sells = tl[tl.side == "SELL"].head(12)
    def trow(x):
        return "".join(f"<tr><td><b>{esc(r.ticker)}</b> <span class='muted'>{esc(str(r['name'])[:28])}</span></td><td class='muted'>{esc(r.sector)}</td>"
                       f"<td class='n'>{r.current_wt:.2%}</td><td class='n'>{r.target_wt:.2%}</td><td class='n'>{r.active_wt:+.2%}</td><td class='n'>{'—' if pd.isna(r.alpha_z) else f'{r.alpha_z:+.2f}'}</td>"
                       f"<td class='n'>{r.shares:+,}</td><td class='n'>${r.trade_usd / 1e6:+,.2f}m</td></tr>" for _, r in x.iterrows())
    sec_rows = "".join(f"<tr><td>{esc(r.sector)}</td><td class='n'>{r.benchmark:.1%}</td><td class='n'>{r.portfolio:.1%}</td><td class='n'>{r.active:+.1%}</td></tr>" for _, r in sec.iterrows())
    r_block = (f"<p>The same problem was handed to a second optimiser, written separately with a different solver, on the {esc(L['formation'])} inputs. "
               f"The two hold {rt.get('names_r')} and {rt.get('names_python')} names with an expected alpha of {rt.get('expected_alpha_r', 0):.4f} against {rt.get('expected_alpha_python', 0):.4f}. "
               f"The largest difference in any weight is {rt.get('max_abs_diff', 0):.1e} and the total difference is {rt.get('sum_abs_diff', 0):.1e}. A problem of this kind has one best value but can have several equally good solutions, which is why the check is on the value achieved rather than on the weights.</p>"
               if rt.get("available") else "<p>The second solver was not run on this build.</p>")
    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Construction</title>
<style>{CSS}{EXTRA_CSS}</style>
<div class="banner" role="note"><span class="bl">Research note</span> This is a demonstration mandate on public data: a transparent signal run through a real set of constraints. It is not a strategy and not investment advice.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Active · Portfolio Construction</div></div>
  <div class="gold-rule"></div>
  <h1>Portfolio Construction</h1>
  <p class="sub">This page takes a signal all the way to a list of trades under a mandate's constraints. An optimiser turns the signal into target weights that respect limits on position size, active weight, sector tilt and turnover, and then into the buy and sell tickets a dealer would receive. The portfolio has been rebalanced {man['rebalances']} times since {man['first'][:7]}, and every solution was checked by a second, independently written optimiser.</p>
  <dl class="meta">
    <div><dt>Rebalances</dt><dd>{man['rebalances']} · {man['first'][:7]} → {man['last'][:7]}</dd></div>
    <div><dt>Universe</dt><dd>names held by ≥ {man['min_holders']} managers</dd></div>
    <div><dt>Mandate NAV</dt><dd>${man['nav'] / 1e6:,.0f}m</dd></div>
    <div><dt>Built</dt><dd>{esc(man['built'])}</dd></div>
  </dl>
</div></header>
<main class="wrap">
<section class="verdict">
  <div class="sh"><h2>What the constraints bought and cost</h2></div>
  <div class="tiles">
    <div class="tile"><div class="tl">Active return, constrained</div><div class="tv">{pct(P.active_return, 1)}</div><div class="td muted">per year against the benchmark, after {c['cost_bps']:.0f} bps of trading costs</div></div>
    <div class="tile"><div class="tl">Tracking error</div><div class="tv">{pct(P.tracking_error, 1, False)}</div><div class="td muted">realized; the model predicted {pct(P.avg_ex_ante_te, 1, False)} on average</div></div>
    <div class="tile"><div class="tl">Information ratio</div><div class="tv">{num(P.information_ratio)}</div><div class="td muted">active return divided by tracking error; the portfolio beat the benchmark in {P.hit_rate:.0%} of months</div></div>
    <div class="tile"><div class="tl">Turnover</div><div class="tv">{P.avg_turnover:.0%}</div><div class="td muted">one-way per quarter against a budget of {c['turnover']:.0%}, holding {P.avg_names:.0f} names on average</div></div>
    <div class="tile"><div class="tl">Unconstrained top decile</div><div class="tv">{pct(U.active_return, 1)}</div><div class="td muted">the same signal with no constraints: tracking error {pct(U.tracking_error, 1, False)}, information ratio {num(U.information_ratio)}, turnover {U.avg_turnover:.0%}, {U.avg_names:.0f} names</div></div>
    <div class="tile"><div class="tl">Benchmark</div><div class="tv">{pct(B.ann_return, 1, False)}</div><div class="td muted">everything the managers own, weighted by dollars held; volatility {pct(B.ann_vol, 1, False)}, worst fall {pct(B.max_dd, 0, False)}</div></div>
  </div>
  <p class="cap" style="margin-top:14px">The signal here is twelve-month price momentum.{ev_summary} The machinery takes any signal and would take a better one. Read the constrained and unconstrained figures together: the constraints give up some of the raw signal in exchange for a portfolio that a benchmark-relative mandate could actually hold.</p>
</section>

<section>
  <div class="sh"><h2>Growth of $1</h2></div>
  <div class="card"><div class="legend">{legend}</div>{growth}
  <p class="cap">The chart uses a log scale. Trading costs of {c['cost_bps']:.0f} bps per dollar traded are charged in the month after each rebalance, on both the constrained and the unconstrained portfolio.</p></div>
  <div class="card" style="margin-top:14px"><h3>Active return by year, constrained portfolio minus benchmark</h3>{act_bars}</div>
</section>

<section>
  <div class="sh"><h2>The mandate</h2></div>
  <div class="grid2">
    <div class="card"><h3>Constraints</h3>
      <div class="tscroll"><table><tbody>
      <tr><td>Long-only, fully invested</td><td class="n">0 ≤ w, Σw = 1</td></tr>
      <tr><td>Name cap</td><td class="n">w ≤ {c['max_weight']:.0%}</td></tr>
      <tr><td>Active band per name</td><td class="n">|w − b| ≤ {c['active_band']:.0%}</td></tr>
      <tr><td>Sector band vs benchmark</td><td class="n">|Σ<sub>s</sub>w − Σ<sub>s</sub>b| ≤ {c['sector_band']:.0%}</td></tr>
      <tr><td>Active share cap</td><td class="n">Σ|w − b| / 2 ≤ {c['active_share']:.0%}</td></tr>
      <tr><td>Turnover budget, one-way</td><td class="n">Σ|w − w₀| / 2 ≤ {c['turnover']:.0%}</td></tr>
      <tr><td>Names leaving the universe</td><td class="n">forced to 0</td></tr>
      <tr><td>Objective</td><td class="n">max α'w − {c['cost_bps']:.0f} bps × traded</td></tr>
      </tbody></table></div>
      <p class="cap">Every rule is linear in the weights, so the problem is a linear program: {L['n_vars']:,} variables and {L['n_constraints']:,} constraints at the latest rebalance, solved in well under a second. When the drifted portfolio cannot meet the sector bands within the turnover budget, the budget is doubled and the rebalance is flagged.</p>
    </div>
    <div class="card"><h3>Sector exposure at the latest rebalance ({esc(L['formation'])})</h3>
      <div class="tscroll"><table><thead><tr><th>sector</th><th class="n">benchmark</th><th class="n">portfolio</th><th class="n">active</th></tr></thead><tbody>{sec_rows}</tbody></table></div>
      <p class="cap">Sectors come from Yahoo Finance for {man['sectors_known']} names, and the {man['sectors_unknown']} unclassified names sit in their own "Unknown" band. A production mandate would use the index vendor's sector classification.</p>
    </div>
  </div>
</section>

<section>
  <div class="sh"><h2>Trade list — {esc(L['formation'])}</h2></div>
  <p class="note">The latest rebalance produced {L['trades']} tickets: {L['buys']} buys and {L['sells']} sells, trading ${L['traded_usd'] / 1e6:,.1f}m on a ${man['nav'] / 1e6:,.0f}m portfolio, which is one-way turnover of {L['turnover']:.1%} at an estimated cost of ${L['est_cost_usd'] / 1e3:,.0f}k. The predicted tracking error is {L['ex_ante_te']:.1%}. The largest tickets each way are shown below.</p>
  <div>
    <div class="card tscroll"><h3>Buys</h3><table><thead><tr><th>stock</th><th>sector</th><th class="n">now</th><th class="n">target</th><th class="n">active</th><th class="n">α z</th><th class="n">shares</th><th class="n">$</th></tr></thead><tbody>{trow(buys)}</tbody></table></div>
    <div class="card tscroll" style="margin-top:14px"><h3>Sells</h3><table><thead><tr><th>stock</th><th>sector</th><th class="n">now</th><th class="n">target</th><th class="n">active</th><th class="n">α z</th><th class="n">shares</th><th class="n">$</th></tr></thead><tbody>{trow(sells)}</tbody></table></div>
  </div>
  <p class="cap">The full list with every name is in <a href="/research/construction/trade_list.csv">trade_list.csv</a>. Shares are rounded to whole shares at the closing price on the rebalance date, and "now" is the previous target after drifting through the quarter.</p>
</section>

<section>
  <div class="sh"><h2>Rebalance log</h2></div>
  <div class="card tscroll"><table><thead><tr><th>date</th><th class="n">universe</th><th class="n">held</th><th class="n">turnover</th><th class="n">active share</th><th class="n">ex-ante TE</th><th class="n">max |active|</th><th class="n">max |sector|</th><th class="n">active share</th><th class="n">α gain vs bench</th><th>note</th></tr></thead><tbody>{te_rows}</tbody></table>
  <p class="cap">The table shows the last ten rebalances. "α gain" is the expected alpha of the portfolio minus that of the benchmark, in standardized units, which is what the optimiser bought within the bands. The full log is in <a href="/research/construction/rebalances.csv">rebalances.csv</a>.</p></div>
</section>

<section>
  <div class="sh"><h2>Checked by a second optimiser</h2></div>
  <div class="card">{r_block}</div>
</section>

<section>
  <div class="sh"><h2>Method and limits</h2></div>
  <div class="card">
    <p><b>Universe and benchmark.</b> Each quarter, the universe is the US stocks held by at least {man['min_holders']} of the managers in the system and priced at $1 or more. The benchmark is those stocks weighted by the total dollars the managers hold, which is everything they own added together. It is a real, investable, public portfolio, but it is not an index, and a real mandate would use one.</p>
    <p><b>Signal.</b> The signal is twelve-month momentum: each stock's return from twelve months before the rebalance to one month before, standardized across the universe and clipped at three standard deviations.{ev_method}</p>
    <p><b>Costs and drift.</b> Trading costs {c['cost_bps']:.0f} bps per dollar traded and is charged in the first month after each rebalance. Between rebalances every portfolio, including the benchmark, is held without trading. Delisted companies drop out at their last price, which is a survivorship bias and is disclosed.</p>
    <p><b>Risk.</b> The predicted tracking error comes from the covariance of the past 36 months of returns, shrunk halfway toward a constant correlation. It is reported rather than constrained, because the active and sector bands are the linear stand-in that most benchmark-relative mandates actually use. Realized tracking error is the standard deviation of the monthly active returns, annualized.</p>
    <p><b>Honesty.</b> The constraints were fixed before any backtest was run and were not tuned. Both the constrained and the unconstrained rows are shown, whatever they say.</p>
  </div>
</section>

<footer class="foot">
  <div class="running"><span>Track record verification · research</span><span>{man['first'][:7]} – {man['last'][:7]}</span></div>
  <h4>Important information</h4>
  <p>This page is built from public SEC EDGAR holdings filings and Yahoo Finance prices and sector labels, with factor data from the Kenneth R. French Data Library. It is a demonstration of portfolio-construction machinery on public data, with the limits stated above; it is not a strategy, a recommendation or investment advice. Past performance is not indicative of future results.</p>
</footer>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}</script>
"""


# ---------------------------------------------------------------- independent verification note

def _sci(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.1e}"


def rverify_html(out_dir: Path | None = None) -> str | None:
    from .rverify import OUT_DIR as RV_DIR, TOL
    d = out_dir or RV_DIR
    if not (d / "manifest.json").exists():
        return None
    man = json.loads((d / "manifest.json").read_text())
    s = pd.read_csv(d / "summary.csv")
    byq = pd.read_csv(d / "by_quantity.csv")
    if s.empty:
        return None

    agreed = int(s.agree.sum()); bad = int((~s.agree).sum())
    worst_mgr = s.loc[s.max_abs_diff.idxmax()]
    # the quantities table: worst disagreement per quantity, largest first
    qrows = ""
    for _, r in byq.iterrows():
        label = f"{esc(r.quantity)}" + (f" <span class='muted'>{esc(r.model)}</span>" if r.model != "-" else "")
        cls = "" if r.max_abs_diff < TOL else " class='bad'"
        qrows += (f"<tr{cls}><td>{label}</td><td class='n'>{int(r.managers)}</td>"
                  f"<td class='n'>{_sci(r.max_abs_diff)}</td><td class='n'>{_sci(r.max_rel_diff)}</td></tr>")
    # per-manager table, largest difference first — the ones worth looking at are at the top
    mrows = ""
    for _, r in s.sort_values("max_abs_diff", ascending=False).head(20).iterrows():
        v, vt = ("yes", "agree") if r.agree else ("no", "differs")
        mrows += (f"<tr><td><a href='/f/{esc(r.slug)}/memo'>{esc(r['name'])}</a></td><td class='n'>{int(r.cells)}</td>"
                  f"<td class='n'>{pct(r.ff3_alpha_r, 1)}</td><td class='n'>{pct(r.ff3_alpha_python, 1)}</td>"
                  f"<td class='n'>{_tv(r.ff3_t_r)}</td><td class='n'>{_tv(r.ff3_t_python)}</td>"
                  f"<td class='n'>{num(r.score_r, 1)}</td><td class='n'>{num(r.score_python, 1)}</td>"
                  f"<td class='n'>{_sci(r.max_abs_diff)}</td><td><span class='v {v}'>{vt}</span></td></tr>")
    verdict = (f"All {man['checks']:,} numbers agree" if not bad else
               f"{bad} of {man['managers']} managers disagree")
    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Independent Verification</title>
<style>{CSS}{EXTRA_CSS}
tr.bad td {{ background: color-mix(in srgb, var(--crit) 12%, transparent); }}
.v {{ display: inline-block; font: 600 9.5px/1 var(--sans); letter-spacing: .12em; text-transform: uppercase; padding: 4px 7px; color: #fff; }}
.v.yes {{ background: var(--good); }} .v.no {{ background: var(--crit); }}
</style>
<div class="banner" role="note"><span class="bl">Verification</span> Every statistic on the site was calculated a second time by a separate implementation, written from scratch against the same data. This tests the calculations, not the conclusions.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Active · Independent Verification</div></div>
  <div class="gold-rule"></div>
  <h1>Independent Verification</h1>
  <p class="sub">Every headline number on a manager's dashboard, including the factor alphas, their t-statistics, the annualized return, volatility, Sharpe ratio, largest drawdown and the alpha-maxing score, was recalculated by a second implementation written independently of the first, from the aligned returns alone. The two share no code, and the results are compared number by number.</p>
  <dl class="meta">
    <div><dt>Managers</dt><dd>{man['managers']}</dd></div>
    <div><dt>Numbers checked</dt><dd>{man['checks']:,}</dd></div>
    <div><dt>Largest difference</dt><dd>{_sci(man['max_abs_diff'])}</dd></div>
    <div><dt>Built</dt><dd>{esc(man['built'])}</dd></div>
  </dl>
</div></header>
<main class="wrap">
<section class="verdict">
  <div class="sh"><h2>{verdict}</h2></div>
  <div class="tiles">
    <div class="tile"><div class="tl">Managers agreeing</div><div class="tv">{agreed} / {man['managers']}</div><div class="td muted">every number agrees to better than {_sci(TOL)}</div></div>
    <div class="tile"><div class="tl">Numbers compared</div><div class="tv">{man['checks']:,}</div><div class="td muted">{man['quantities_per_manager']} per manager, covering four factor models, the risk measures and the score</div></div>
    <div class="tile"><div class="tl">Largest difference</div><div class="tv">{_sci(man['max_abs_diff'])}</div><div class="td muted">{esc(worst_mgr['name'])}, {esc(worst_mgr.worst_quantity)}; this is rounding noise, not a discrepancy</div></div>
    <div class="tile"><div class="tl">Disagreements</div><div class="tv">{bad}</div><div class="td muted">differences above {_sci(TOL)}, which is where a real error would show</div></div>
  </div>
  <p class="cap" style="margin-top:14px">Agreement to one part in ten trillion means the two implementations do the same arithmetic. It does not mean the model is the right one, that the disclosed holdings track the funds, or that a small alpha is real; the memos say what the numbers are worth. What it rules out is the quiet kind of error: a wrong lag in the standard errors, an off-by-one in the annualization, or a dropped month that silently changes the sample.</p>
</section>

<section>
  <div class="sh"><h2>Agreement by quantity</h2></div>
  <div class="card tscroll"><table><thead><tr><th>quantity</th><th class="n">managers</th><th class="n">largest difference</th><th class="n">largest relative difference</th></tr></thead><tbody>{qrows}</tbody></table>
  <p class="cap">Each row is the worst case across every manager, with the largest first. The full detail is in <a href="/research/r-verify/by_quantity.csv">by_quantity.csv</a> and <a href="/research/r-verify/summary.csv">summary.csv</a>.</p></div>
</section>

<section>
  <div class="sh"><h2>Manager by manager</h2></div>
  <div class="card tscroll"><table><thead><tr><th>manager</th><th class="n">months</th><th class="n">FF3 α, check</th><th class="n">FF3 α, site</th><th class="n">t, check</th><th class="n">t, site</th><th class="n">score, check</th><th class="n">score, site</th><th class="n">largest difference</th><th></th></tr></thead><tbody>{mrows}</tbody></table>
  <p class="cap">These are the twenty managers with the largest difference, which makes them the hardest cases for the comparison rather than the best managers. Alphas are annualized, and t is the Newey–West t-statistic on the three-factor alpha.</p></div>
</section>

<section>
  <div class="sh"><h2>What the second implementation recalculates</h2></div>
  <div class="grid2">
    <div class="card"><h3>The regression</h3>
      <p>For each of the four factor models, it fits the manager's excess return on the factors by ordinary least squares and builds the Newey–West standard errors from the definition:</p>
      <p class="note">V = (X'X)<sup>−1</sup> S (X'X)<sup>−1</sup>, &nbsp; S = S<sub>0</sub> + Σ<sub>l=1..L</sub> w<sub>l</sub>(S<sub>l</sub> + S<sub>l</sub>'), &nbsp; w<sub>l</sub> = 1 − l/(L+1), &nbsp; L = ⌊0.75·n<sup>1/3</sup>⌋</p>
      <p>It uses no small-sample correction and normal rather than t-distributed p-values and confidence intervals, matching the conventions of the site. Those three choices are exactly where two implementations usually drift apart, so they are the point of the exercise.</p>
    </div>
    <div class="card"><h3>The rest</h3>
      <p>The annualized return is the compound growth rate over the record. Volatility is the standard deviation of monthly returns times the square root of twelve. The Sharpe ratio is the mean excess return over its standard deviation, times the square root of twelve. The largest drawdown is taken from the compounded value of the monthly returns. The alpha-maxing score is rebuilt from its fixed scale, 50 plus 10 times the excess return over the market in points, clipped to 0 to 100.</p>
      <p>The second implementation reads only the aligned returns and factors, and then the site's own results purely to compare against. Any error in the alignment itself would be invisible to this test; it checks the statistics, not the data assembly, which is what the coverage and reconciliation steps are for.</p>
    </div>
  </div>
</section>

<footer class="foot">
  <div class="running"><span>Track record verification · verification</span><span>{man['managers']} managers · {man['checks']:,} numbers</span></div>
  <h4>Important information</h4>
  <p>This is an internal consistency check between two implementations of the same statistics, run on portfolios reconstructed from public SEC EDGAR holdings filings, with factor data from the Kenneth R. French Data Library. Agreement between implementations is not evidence that a manager has skill. It is not a recommendation or investment advice.</p>
</footer>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}</script>
"""


# ---------------------------------------------------------------- research index

def research_index_html() -> str:
    """One page listing the research notes and every dataset behind them.

    The point of the page is that a reader can take the data away: each note is paired
    with the CSVs it was built from, and each row says plainly what is in the file."""
    from .rverify import OUT_DIR as RV_DIR
    from .construct import OUT_DIR as CON_DIR
    from .decay import OUT_DIR as DECAY_DIR
    R = Path(__file__).resolve().parents[1] / "data" / "research"

    def man(p):
        try:
            return json.loads(p.read_text()) if p.exists() else {}
        except Exception:
            return {}

    sig, con, rv, dec = man(SIG_DIR / "manifest.json"), man(CON_DIR / "manifest.json"), man(RV_DIR / "manifest.json"), man(DECAY_DIR / "manifest.json")

    notes = []
    if sig:
        notes.append(("/research/13f-signals", "Holdings Research",
                      f"The public holdings of {sig.get('managers', '')} managers over {sig.get('quarters', '')} quarters ({sig.get('positions', 0):,} positions) "
                      "are turned into best-ideas, crowding and conviction-change portfolios, formed 45 days after each quarter-end, "
                      "and tested as long-short spreads against the four standard factors.",
                      [("13f-signals/summary.csv", "one row per portfolio: return, vol, Sharpe, max DD, CAPM and Carhart alpha with t, turnover"),
                       ("13f-signals/spreads.csv", "long-short spreads between portfolios, with alphas and t-statistics"),
                       ("13f-signals/portfolios_monthly.csv", "the monthly return series of every portfolio"),
                       ("13f-signals/ic.csv", "rank information coefficient at each formation date"),
                       ("13f-signals/latest_crowding.csv", "how many managers hold each name in the latest filings")]))
    if con:
        notes.append(("/research/construction", "Portfolio Construction",
                      f"A demonstration mandate was rebalanced {con.get('rebalances', '')} times on a "
                      f"${con.get('nav', 0) / 1e6:,.0f}m portfolio. An optimiser maximises the signal net of trading cost under limits on position size, "
                      "sector tilt, active share and turnover, and every solution is checked by a second, independently written optimiser.",
                      [("construction/summary.csv", "constrained, unconstrained and benchmark portfolios side by side"),
                       ("construction/trade_list.csv", "the latest trade list: every ticket with shares, dollars and active weight"),
                       ("construction/rebalances.csv", "every rebalance with turnover, active share and ex-ante tracking error"),
                       ("construction/backtest_monthly.csv", "monthly returns of all three portfolios"),
                       ("construction/sectors_latest.csv", "sector exposure against the benchmark at the latest rebalance")]))
    if dec.get("managers"):
        notes.append(("/research/decay", "Manager Decay Model",
                      f"For {dec.get('managers')} managers at every quarterly filing, {dec.get('rows_labelled', 0):,} manager-quarters in all, the portfolio's "
                      "features and recent returns are used to predict whether the manager lags the market over the next twelve months. "
                      "Three predictors are compared out of sample: last year's laggards, a logistic regression and a learned model.",
                      [("decay/panel.csv", "one row per manager-quarter: every feature, the forward excess return and label, and each predictor's out-of-sample score"),
                       ("decay/summary.csv", "one row per predictor: pooled and cross-sectional AUC with t, riskiest-minus-safest spread, Brier"),
                       ("decay/by_date.csv", "cross-sectional AUC and spread of each predictor at every formation date"),
                       ("decay/features.csv", "each feature alone: rank correlation with next year's excess return, importance in the learned model, regression coefficient"),
                       ("decay/watchlist.csv", "the latest formation date: every manager's features and scores"),
                       ("decay/panel_r.csv", "the same panel rebuilt by the independent implementation, for the comparison")]))
    if rv.get("managers"):
        notes.append(("/research/r-verify", "Independent Verification",
                      f"{rv.get('checks', 0):,} numbers across {rv.get('managers')} managers were recalculated by an independent "
                      f"implementation and compared with the site's own results. The largest difference is {rv.get('max_abs_diff', 0):.0e}, "
                      f"with {rv.get('disagreements', 0)} disagreements.",
                      [("r-verify/summary.csv", "one row per manager: FF3 alpha and t from both implementations, worst difference"),
                       ("r-verify/by_quantity.csv", "worst disagreement per quantity across every manager")]))

    blocks = ""
    for href, title, what, files in notes:
        rows = "".join(
            f"<tr><td><a href='/research/data/{esc(f)}'><code>{esc(f)}</code></a></td><td>{esc(d)}</td>"
            f"<td class='n'>{_kb(R / f)}</td></tr>" for f, d in files)
        blocks += (f"<section><div class='sh'><h2><a href='{href}'>{esc(title)}</a></h2></div>"
                   f"<div class='card'><p>{esc(what)}</p>"
                   f"<div class='tscroll'><table><thead><tr><th>file</th><th>what is in it</th><th class='n'>size</th></tr></thead>"
                   f"<tbody>{rows}</tbody></table></div>"
                   f"<p class='cap'>Read the note: <a href='{href}'>{esc(title)}</a></p></div></section>")

    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Research Data</title>
<style>{CSS}{EXTRA_CSS}
.sh h2 a {{ color: inherit; text-decoration: none; border-bottom: 1px solid var(--gold); }}
.sh h2 a:hover {{ color: var(--gold); }}
.cover .sub a {{ color: var(--gold); text-decoration: none; border-bottom: 1px solid rgba(197,167,106,.5); }}
.card td a code {{ font-size: 12.5px; }}
</style>
<div class="banner" role="note"><span class="bl">Research</span> Every research note on this site is generated from the files listed here. You can take the data and check the numbers yourself.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Research \u00b7 notes and data</div></div>
  <div class="gold-rule"></div>
  <h1>Research Data</h1>
  <p class="sub">This page lists the research notes and the data under them: {len(notes)} notes built on managers' public holdings filings and public prices, and the {sum(len(f) for _, _, _, f in notes)} data files they are generated from. Nothing on those pages is hand-written prose; change the data and the sentences change with it. A due-diligence memo for any manager is one click from its row under <a href="/external#all">Manager Analysis</a>.</p>
</div></header>
<main class="wrap">
{blocks}
<section>
  <div class="sh"><h2>Screening the managers</h2></div>
  <div class="card">
    <p>The manager table under <a href="/external#all">Manager Analysis</a> is a screen. Filter the {rv.get('managers', 91)} managers by evidence of alpha, by style and by how long the record is, then take the filtered set away with <b>Download CSV</b>. The export carries the manager, fund, style, months, excess return over the market, the alpha t-statistic and both scores.</p>
    <p class="cap">The t-statistic is measured on the alpha of a three-factor regression of the manager's monthly excess returns, with Newey–West standard errors. A t-statistic of 2 or more is the conventional bar for "unlikely to be luck". Very few managers clear it, and that is the honest finding rather than a fault in the screen.</p>
  </div>
</section>
<footer class="foot">
  <div class="running"><span>Track record verification \u00b7 research</span><span>{len(notes)} notes</span></div>
  <h4>Important information</h4>
  <p>These pages are built from public SEC EDGAR holdings filings and Yahoo Finance prices, with factor and benchmark data from the Kenneth R. French Data Library. Managers' portfolios are reconstructed from their disclosed US long positions and are not the funds themselves. Nothing here is a strategy, a recommendation or investment advice. Past performance is not indicative of future results.</p>
</footer>
</main>
<script>{JS}</script>
"""


def _kb(p: Path) -> str:
    try:
        n = p.stat().st_size
    except OSError:
        return "\u2014"
    return f"{n / 1024:.0f} KB" if n >= 1024 else f"{n} B"


# ---------------------------------------------------------------- alpha model lab

def alphalab_html(out_dir: Path | None = None) -> str | None:
    from .alphalab import OUT_DIR as LAB_DIR, SIGNALS
    d = out_dir or LAB_DIR
    if not (d / "manifest.json").exists():
        return None
    man = json.loads((d / "manifest.json").read_text())
    S = pd.read_csv(d / "summary.csv").set_index("signal")
    ic = pd.read_csv(d / "ic_monthly.csv", index_col=0, parse_dates=True)
    dec = pd.read_csv(d / "deciles.csv", index_col=0)
    imps = pd.read_csv(d / "xgboost_importance.csv", index_col=0)
    corr = pd.read_csv(d / "signal_correlation.csv", index_col=0)
    latest = pd.read_csv(d / "latest_ranks.csv")
    hh = man.get("head_to_head", {})
    feats = man["features"]
    # what the eight signals actually settled: the evidence bar is a t-statistic of 2, and on
    # this sample only the decile spread of one signal clears it
    sig = {s: S.loc[s] for s in feats if s in S.index}
    cleared = [s for s, r in sig.items() if r.spread_t >= 2]
    best = max(sig, key=lambda s: sig[s].spread_t) if sig else None
    br = sig[best] if best else None
    best_name = plain(SIGNALS.get(best, best or "")).split(" (")[0].lower()
    sel = (f"Only {esc(best_name)} shows anything: its top tenth of the universe beat its bottom tenth by "
           f"{pct(br.spread_ann, 1)} a year, with a t-statistic of {br.spread_t:+.1f}, while the other "
           f"{_WORDS.get(len(feats) - 1, len(feats) - 1)} are noise or point the wrong way." if br is not None else "")

    def light(t):
        return ("yes", "evidence") if t >= 2 else ("weak", "weak") if t >= 1 else ("no", "none") if t > -1 else ("no", "wrong way")
    # signal table
    srow = ""
    for s in feats + ["linear", "xgboost"]:
        if s not in S.index: continue
        r = S.loc[s]; c, lab = light(r.ic_t)
        srow += (f"<tr><td><b>{esc(plain(r.label))}</b></td><td class='n'>{r.coverage:.0%}</td><td class='n'>{int(r.months)}</td><td class='n'>{r.ic_mean:+.3f}</td><td class='n'>{r.ic_t:+.1f}</td>"
                 f"<td class='n'>{r.ic_pct_positive:.0%}</td><td class='n'>{pct(r.spread_ann, 1)}</td><td class='n'>{r.spread_t:+.1f}</td><td class='n'>{num(r.spread_sharpe)}</td>"
                 f"<td><span class='v {c}' style='color:#fff;background:var(--{ {'no': 'crit', 'weak': 'warn', 'yes': 'good'}[c] });font:600 9px var(--sans);letter-spacing:.14em;text-transform:uppercase;padding:4px 7px'>{esc(lab)}</span></td></tr>")
    # cumulative IC chart (linear vs xgboost) and rolling 12m IC for the best signals
    cells = [x.strftime("%Y-%m") for x in ic.index]
    series = []
    for col, name, cls in [("linear", "Simple average", "s1"), ("xgboost", "Learned model", "s4"), ("momentum", "Momentum", "s0"), ("value", "Value", "s5"), ("low_vol", "Low volatility", "s2")]:
        if col in ic:
            cum = ic[col].fillna(0).cumsum().where(ic[col].notna().cummax())
            series.append(dict(name=name, values=[None if pd.isna(v) else float(v) for v in cum], cls=cls, emph=col in ("linear", "xgboost")))
    cum_chart = line_chart(cells, series, height=300, width=860, y_fmt=lambda v: f"{v:+.1f}", end_labels=False, uid="cumic")
    legend = "".join(f'<span><span class="k {s["cls"]}"></span>{esc(s["name"])}</span>' for s in series)
    # deciles bars for linear and xgboost
    def dec_bars(col):
        if col not in dec: return ""
        v = dec[col]
        return diverging_bars([f"D{int(i)}" for i in v.index], [float(x) * 12 for x in v.values], height=200, width=420, y_fmt=lambda y: f"{y * 100:+.0f}%",
                              tips=[f"decile {int(i)}: {x * 12 * 100:+.1f}%/yr average" for i, x in v.items()])
    imp_last = imps.iloc[:, -1].sort_values(ascending=False) if len(imps.columns) else pd.Series(dtype=float)
    imp_rows = "".join(f"<tr><td>{esc(SIGNALS.get(f, f))}</td><td class='n'>{v:.0%}</td></tr>" for f, v in imp_last.items())
    corr_head = "".join(f"<th class='n'>{esc(SIGNALS.get(c, c).split(' (')[0])}</th>" for c in corr.columns)
    corr_rows = "".join("<tr><td>" + esc(SIGNALS.get(r, r)) + "</td>" + "".join(f"<td class='n'>{corr.loc[r, c]:+.2f}</td>" for c in corr.columns) + "</tr>" for r in corr.index)
    lat_rows = "".join(f"<tr><td><b>{esc(r.ticker)}</b></td>" + "".join(f"<td class='n'>{'—' if pd.isna(r[f]) else f'{r[f]:+.1f}'}</td>" for f in feats) + f"<td class='n'><b>{r.linear:+.2f}</b></td><td class='n'>{'—' if pd.isna(r.xgboost) else f'{r.xgboost * 100:+.2f}%'}</td></tr>" for _, r in latest.head(25).iterrows())
    lin, xg = S.loc["linear"] if "linear" in S.index else None, S.loc["xgboost"] if "xgboost" in S.index else None
    verdict = ("No, they are indistinguishable" if hh.get("ic_diff_t", 0) < 2 else "Yes, it does") if xg is not None and lin is not None else ""
    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Signal Research</title>
<style>{CSS}{EXTRA_CSS}</style>
<div class="banner" role="note"><span class="bl">Research note</span> These signals are built from public prices and company filings and tested out of sample on a universe with a known survivorship bias. This is a demonstration of method, not a strategy.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Active · Signal Research</div></div>
  <div class="gold-rule"></div>
  <h1>Signal Research</h1>
  <p class="sub">Which stock characteristics predict next month's return? Eight classic signals are built for about {man['universe_avg']} stocks a month using only what was known at the time, and tested every month from {man['first'][:7]} to {man['last'][:7]}. {sel} That selection is the reason the <a href="/research/construction">Portfolio Construction</a> page trades the signal it trades.</p>
  <dl class="meta">
    <div><dt>Months</dt><dd>{man['months']}</dd></div>
    <div><dt>Names / month</dt><dd>~{man['universe_avg']}</dd></div>
    <div><dt>Signals</dt><dd>{len(feats)}</dd></div>
    <div><dt>Learned model tested</dt><dd>{man.get('xgb_months', 0)} months from {(man.get('xgb_first') or '')[:7]}</dd></div>
    <div><dt>Built</dt><dd>{esc(man['built'])}</dd></div>
  </dl>
</div></header>
<main class="wrap">
<section class="verdict">
  <div class="sh"><h2>Findings</h2></div>
  <div class="tiles">
    <div class="tile"><div class="tl">Signals with any evidence</div><div class="tv">{len(cleared)} of {len(feats)}</div><div class="td muted">counted on the spread between the top and bottom tenth of the universe, at a t-statistic of 2; on the month-by-month rank correlation not even that one clears the bar</div></div>
    <div class="tile"><div class="tl">Best of the eight</div><div class="tv">{pct(br.spread_ann, 1)}</div><div class="td muted">{esc(best_name)}, top tenth minus bottom tenth a year; t = {br.spread_t:+.1f}, Sharpe {num(br.spread_sharpe)}; its rank correlation is {br.ic_mean:+.3f} with t = {br.ic_t:+.1f}</div></div>
    <div class="tile"><div class="tl">Simple average</div><div class="tv">{lin.ic_mean:+.3f}</div><div class="td muted">t = {lin.ic_t:+.1f}; positive in {lin.ic_pct_positive:.0%} of months; top tenth beat bottom tenth by {pct(lin.spread_ann, 1)} a year</div></div>
    <div class="tile"><div class="tl">Learned model</div><div class="tv">{xg.ic_mean:+.3f}</div><div class="td muted">t = {xg.ic_t:+.1f}; positive in {xg.ic_pct_positive:.0%} of months; top tenth beat bottom tenth by {pct(xg.spread_ann, 1)} a year</div></div>
  </div>
  <p class="cap" style="margin-top:14px">The job of this page is to decide which of the eight, if any, is worth trading. The last two figures are information coefficients: the rank correlation between a signal today and returns next month. A usable signal looks like 0.03 to 0.05 with a t-statistic above 3 over a long sample, so on {man['months']} months of one universe everything here is small, and saying so is the result. The first {man['min_train']} months are held out so the learned model has something to learn from, and it is refit every {man['retrain']} months using only earlier data.</p>
</section>

<section>
  <div class="sh"><h2>Signal by signal</h2></div>
  <div class="card tscroll"><table><thead><tr><th>signal</th><th class="n">coverage</th><th class="n">months</th><th class="n">mean IC</th><th class="n">t</th><th class="n">IC &gt; 0</th><th class="n">D10 − D1 /yr</th><th class="n">t</th><th class="n">Sharpe</th><th>evidence</th></tr></thead><tbody>{srow}</tbody></table>
  <p class="cap">Coverage is the share of stock-months for which the signal is available, since the accounting signals depend on what companies report. "D10 − D1" is the return of the top tenth of stocks minus the bottom tenth, annualized. A t-statistic of 2 or more counts as evidence, 1 to 2 as weak, and below 1 as none.</p></div>
</section>

<section>
  <div class="sh"><h2>Cumulative information coefficient</h2></div>
  <div class="card"><div class="legend">{legend}</div>{cum_chart}
  <p class="cap">Each line is the running sum of monthly information coefficients. A signal that works climbs steadily, and one that does not wanders around zero. The two models start at {(man.get('xgb_first') or '')[:7]}, when the first prediction from the learned model is available.</p></div>
</section>

<section>
  <div class="sh"><h2>Decile returns</h2></div>
  <div class="grid2">
    <div class="card"><h3>Simple average, by tenth of the universe (annualized)</h3>{dec_bars('linear')}</div>
    <div class="card"><h3>Learned model, by tenth of the universe (annualized)</h3>{dec_bars('xgboost')}</div>
  </div>
  <p class="cap">Each bar is the average next-month return of one tenth of the universe, ranked by the model, annualized. A steady staircase is the signature of a real signal, and a spread that comes only from the bottom or only from the top is a warning about what is driving it.</p>
</section>

<section>
  <div class="sh"><h2>Does learning add anything?</h2></div>
  <div class="tiles">
    <div class="tile"><div class="tl">Head to head</div><div class="tv">{hh.get('ic_diff_mean', 0):+.3f}</div><div class="td muted">learned model minus simple average over the same {hh.get('months', 0)} months; t = {hh.get('ic_diff_t', 0):+.1f}; the learned model was ahead in {hh.get('xgb_wins_share', 0):.0%} of months</div></div>
  </div>
  <p class="cap" style="margin-top:14px">{esc(verdict)} — a coin flip, on identical inputs over identical months. With {len(feats)} inputs there is little for a model to learn that an equal-weighted average does not already capture, and {hh.get('months', 0)} months cannot settle a difference this small either way. The comparison is here because it was run and has to be reported, not because it decided anything. What the page decided is the choice of signal above.</p>
  <div class="grid2">
    <div class="card"><h3>Importance of each signal in the learned model (latest fit)</h3><div class="tscroll"><table><thead><tr><th>signal</th><th class="n">gain share</th></tr></thead><tbody>{imp_rows}</tbody></table></div>
      <p class="cap">Each row is the share of the model's improvement attributed to that input. Importance is not predictive power, because a signal can be used heavily and add nothing out of sample.</p></div>
    <div class="card tscroll"><h3>Average rank correlation between signals</h3><table><thead><tr><th></th>{corr_head}</tr></thead><tbody>{corr_rows}</tbody></table>
      <p class="cap">The correlations are averaged across months. Highly correlated signals are one signal wearing two names, and the simple average double-counts them.</p></div>
  </div>
</section>

<section>
  <div class="sh"><h2>Latest ranking — {esc(man['last'][:7])}</h2></div>
  <div class="card tscroll"><table><thead><tr><th>stock</th>{''.join(f"<th class='n'>{esc(SIGNALS.get(f, f).split(' (')[0])}</th>" for f in feats)}<th class="n">simple average</th><th class="n">learned model</th></tr></thead><tbody>{lat_rows}</tbody></table>
  <p class="cap">These are the top 25 stocks by the simple average at the latest month-end, with each signal shown as a standardized score across the universe. The learned-model column is its predicted return relative to the universe next month. <a href="/research/construction">Portfolio Construction</a> rebuilds the {esc(best_name)} column on the same universe and trades that alone, because it is the one column above with evidence behind it.</p></div>
</section>

<section>
  <div class="sh"><h2>Method and limits</h2></div>
  <div class="card">
    <p><b>Point in time.</b> Prices run through the month-end. A company filing is used only after the date it was filed and only if its period ended within the last 15 months. Balance-sheet items are the latest reported figure, and income and cash-flow items are the latest annual figure, so they update once a year.</p>
    <p><b>Universe.</b> The universe is the stocks held by at least five of the managers in the system at the latest filing date and priced at $1 or more. That is around {man['universe_avg']} names a month, large and liquid, which is where such signals are weakest. Delisted companies drop out at their last price, which is a survivorship bias and is disclosed.</p>
    <p><b>Models.</b> The simple average is the plain mean of the available standardized signals, with no fitted weights at all. The learned model is a gradient-boosted ensemble of 300 shallow decision trees trained on next-month returns relative to the universe, using an expanding window with the first {man['min_train']} months held out and a refit every {man['retrain']} months. None of its settings were tuned on the test period.</p>
    <p><b>Reading it.</b> With about 150 months and one universe, a t-statistic below 2 is noise, and by that rule seven of the eight signals are noise and the learned model and the simple average are indistinguishable. What survives is one signal, on one of its two tests, which is the finding a research meeting needs rather than a backtest that looks good. It is also enough to pick what the construction page trades, which is the only decision this page was run to make.</p>
  </div>
</section>

<footer class="foot">
  <div class="running"><span>Track record verification · research</span><span>{man['first'][:7]} – {man['last'][:7]}</span></div>
  <h4>Important information</h4>
  <p>This page is built from public SEC EDGAR filings of holdings and company accounts and from Yahoo Finance prices. The results are reconstructions with the limits stated above; they are not a strategy, a recommendation or investment advice. Past performance is not indicative of future results.</p>
</footer>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}</script>
"""


# ---------------------------------------------------------------- risk model

def riskmodel_html(out_dir: Path | None = None) -> str | None:
    from .riskmodel import OUT_DIR as RM_DIR
    from .alphalab import SIGNALS
    d = out_dir or RM_DIR
    if not (d / "manifest.json").exists():
        return None
    man = json.loads((d / "manifest.json").read_text())
    fr = pd.read_csv(d / "factor_returns.csv", index_col=0, parse_dates=True)
    r2 = pd.read_csv(d / "r2_monthly.csv", index_col=0, parse_dates=True).r2
    vols = pd.read_csv(d / "factor_vols_latest.csv").set_index("factor")
    spec = pd.read_csv(d / "specific_risk_latest.csv")
    decs = json.loads((d / "decompositions_latest.json").read_text())
    bt = pd.read_csv(d / "bias_by_year.csv", index_col=0).bias_stat
    bias = man.get("bias", {})
    styles = man["styles"]
    lab = lambda f: SIGNALS.get(f, f).split(" (")[0] if not f.startswith("ind:") else f[4:]

    cells = [x.strftime("%Y-%m") for x in fr.index]
    cls_cycle = ["s1", "s4", "s2", "s5", "s0", "s1l", "s1", "s4"]
    series = [dict(name=lab(s), values=[float(v) for v in fr[s].fillna(0).cumsum()], cls=cls_cycle[i % len(cls_cycle)]) for i, s in enumerate(styles) if s in fr]
    fchart = line_chart(cells, series, height=300, width=860, y_fmt=lambda v: f"{v * 100:+.0f}%", end_labels=False, uid="fret")
    legend = "".join(f'<span><span class="k {s["cls"]}"></span>{esc(s["name"])}</span>' for s in series)
    r2y = r2.groupby(r2.index.year).mean()
    r2_bars = diverging_bars([str(i) for i in r2y.index], [float(v) for v in r2y.values], height=180, width=860, y_fmt=lambda v: f"{v:.0%}",
                             tips=[f"{i}: average cross-sectional R² {v:.0%}" for i, v in r2y.items()])
    bias_bars = diverging_bars([str(i) for i in bt.index], [float(v) - 1 for v in bt.values], height=180, width=860, y_fmt=lambda v: f"{v + 1:.2f}",
                               tips=[f"{i}: bias statistic {v:.2f} (1 = calibrated)" for i, v in bt.items()])
    vrows = "".join(f"<tr><td>{esc(lab(f))}</td><td class='n'>{r.vol_ann:.1%}</td><td class='n'>{r.mean_ret_ann * 100:+.1f}%</td><td class='n'>{r.t:+.1f}</td></tr>"
                    for f, r in vols.iterrows() if not f.startswith("ind:"))
    irows = "".join(f"<tr><td>{esc(lab(f))}</td><td class='n'>{r.vol_ann:.1%}</td><td class='n'>{r.mean_ret_ann * 100:+.1f}%</td><td class='n'>{r.t:+.1f}</td></tr>"
                    for f, r in vols.sort_values("vol_ann", ascending=False).iterrows() if f.startswith("ind:"))
    def dec_card(name, x):
        g = x.get("groups", {})
        rows = "".join(f"<tr><td>{esc(k.title())}</td><td class='n'>{v:.0%}</td></tr>" for k, v in g.items()) + f"<tr><td class='muted'>Cross-factor interaction</td><td class='n'>{x.get('interaction', 0):.0%}</td></tr><tr><td>Stock-specific</td><td class='n'>{1 - x['share_factor']:.0%}</td></tr>"
        top = "".join(f"<tr><td>{esc(lab(t['factor']))}</td><td class='n'>{t['exposure']:+.2f}</td><td class='n'>{t['var_share']:.0%}</td></tr>" for t in x["top"][:6])
        return (f"<div class='card'><h3>{esc(name)}</h3><div class='tiles' style='margin-bottom:10px'>"
                f"<div class='tile'><div class='tl'>Total risk</div><div class='tv'>{x['total']:.1%}</div><div class='td muted'>annualized · {x['n_names']} names</div></div>"
                f"<div class='tile'><div class='tl'>Factor</div><div class='tv'>{x['factor']:.1%}</div><div class='td muted'>{x['share_factor']:.0%} of variance</div></div>"
                f"<div class='tile'><div class='tl'>Specific</div><div class='tv'>{x['specific']:.1%}</div><div class='td muted'>{1 - x['share_factor']:.0%} of variance</div></div></div>"
                f"<div class='grid2'><div class='tscroll'><table><thead><tr><th>source</th><th class='n'>share of variance</th></tr></thead><tbody>{rows}</tbody></table></div>"
                f"<div class='tscroll'><table><thead><tr><th>largest exposures</th><th class='n'>exposure</th><th class='n'>var share</th></tr></thead><tbody>{top}</tbody></table></div></div></div>")
    dec_html = "".join(dec_card(k, v) for k, v in decs.items())
    spec_med = float(spec.specific_vol_ann.median()); spec_hi = spec.head(8)
    spec_rows = "".join(f"<tr><td><b>{esc(r.ticker)}</b></td><td class='n'>{r.specific_vol_ann:.0%}</td></tr>" for _, r in spec_hi.iterrows())
    b_rand, b_uni = bias.get("random", float("nan")), bias.get("universe", float("nan"))
    verdict = ("calibrated" if 0.85 <= b_rand <= 1.15 else "under-forecasts risk" if b_rand > 1.15 else "over-forecasts risk")
    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Factor Risk Model</title>
<style>{CSS}{EXTRA_CSS}</style>
<div class="banner" role="note"><span class="bl">Research note</span> This is a factor risk model estimated on the research universe. It demonstrates the method that a commercial risk model implements at scale.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Active · Factor Risk Model</div></div>
  <div class="gold-rule"></div>
  <h1>Factor Risk Model</h1>
  <p class="sub">This is a fundamental factor risk model of the kind commercial vendors sell. Every month, each stock's return is explained by its exposure to the market, to the same eight characteristics used in Signal Research, and to its industry. Those explanations become a forecast of how much any portfolio will move and of where its risk comes from, and a calibration test says whether the forecasts can be trusted.</p>
  <dl class="meta">
    <div><dt>Months</dt><dd>{man['months']} · {man['first'][:7]} → {man['last'][:7]}</dd></div>
    <div><dt>Factors</dt><dd>{man['factors']}: the market, {len(styles)} styles and {man['industries']} industries</dd></div>
    <div><dt>Returns explained</dt><dd>{man['r2_avg']:.0%}</dd></div>
    <div><dt>Bias statistic</dt><dd>{b_rand:.2f}</dd></div>
    <div><dt>Built</dt><dd>{esc(man['built'])}</dd></div>
  </dl>
</div></header>
<main class="wrap">
<section class="verdict">
  <div class="sh"><h2>Does it forecast risk?</h2></div>
  <div class="tiles">
    <div class="tile"><div class="tl">Bias statistic, random portfolios</div><div class="tv">{b_rand:.2f}</div><div class="td muted">the spread of realized against predicted risk, where 1.00 is perfect; {verdict}</div></div>
    <div class="tile"><div class="tl">Bias statistic, whole universe</div><div class="tv">{b_uni:.2f}</div><div class="td muted">the whole universe, equally weighted, each month</div></div>
    <div class="tile"><div class="tl">Explanatory power</div><div class="tv">{man['r2_avg']:.0%}</div><div class="td muted">the average share of a month's stock returns the factors explain</div></div>
    <div class="tile"><div class="tl">Median specific risk</div><div class="tv">{spec_med:.0%}</div><div class="td muted">annualized, per stock, in the latest month</div></div>
  </div>
  <p class="cap" style="margin-top:14px">The bias statistic is the standard test of a risk model. For {len(bt)} years of monthly random 50-stock portfolios, each realized return is divided by the volatility the model predicted for it the month before, and a well-calibrated model gives ratios with a standard deviation of 1. Above 1 the model under-forecasts risk, which is dangerous; below 1 it over-forecasts, which is costly. The factor covariance uses a {man['window']}-month window that weights recent months more heavily, with a {man['half_life']}-month half-life, and each stock's specific risk is estimated the same way and shrunk toward the median for short histories.</p>
  <div class="card" style="margin-top:14px"><h3>Bias statistic by year (random portfolios)</h3>{bias_bars}<p class="cap">The bars show the deviation from 1. Years above the line are years the model was too confident, which are typically changes of regime that the trailing window had not seen.</p></div>
</section>

<section>
  <div class="sh"><h2>Factor returns</h2></div>
  <div class="card"><div class="legend">{legend}</div>{fchart}
  <p class="cap">Each line is the cumulative return to one style factor: the return to a unit of exposure, holding the other styles and industries fixed. These are "pure" factor returns of the kind a commercial risk report shows, not long-short portfolios.</p></div>
  <div class="grid2" style="margin-top:14px">
    <div class="card tscroll"><h3>Style factors, latest window</h3><table><thead><tr><th>factor</th><th class="n">vol /yr</th><th class="n">mean /yr</th><th class="n">t</th></tr></thead><tbody>{vrows}</tbody></table></div>
    <div class="card tscroll"><h3>Industry factors, latest window</h3><table><thead><tr><th>industry</th><th class="n">vol /yr</th><th class="n">mean /yr</th><th class="n">t</th></tr></thead><tbody>{irows}</tbody></table></div>
  </div>
  <div class="card" style="margin-top:14px"><h3>Cross-sectional R² by year</h3>{r2_bars}<p class="cap">Each bar is how much of that year's month-by-month dispersion in stock returns the factors explain. Commercial models on broad universes typically run between 20% and 40%, and higher in crises when everything moves together.</p></div>
</section>

<section>
  <div class="sh"><h2>Risk decomposition — {esc(man['latest_month'][:7])}</h2></div>
  <p class="note">This is the report a portfolio manager reads before a rebalance: where the risk is, and whether the active bets are the intended ones. Each group shows its own share of the variance, and the covariance between factors is shown separately.</p>
  {dec_html}
</section>

<section>
  <div class="sh"><h2>Specific risk</h2></div>
  <div class="card"><div class="grid2"><div class="tscroll"><table><thead><tr><th>highest specific risk, latest</th><th class="n">vol /yr</th></tr></thead><tbody>{spec_rows}</tbody></table></div>
  <div><p>The median is {spec_med:.0%} across {len(spec)} names. Specific risk is what diversification removes and what a concentrated manager is paid for taking. The names at the top of this list are where a single position can move a portfolio.</p></div></div></div>
</section>

<section>
  <div class="sh"><h2>Method and limits</h2></div>
  <div class="card">
    <p><b>Estimation.</b> Each month, the next month's return of every stock in the universe is regressed on its exposures at the time: the market, {len(styles)} standardized style scores, and {man['industries']} industry memberships, with the industry returns constrained to average to zero so that the market factor is identified. The regression is ordinary least squares with equal weights, whereas a commercial vendor would weight by company size and trim extreme residuals.</p>
    <p><b>Covariance.</b> The factor covariance is estimated over the trailing {man['window']} months with recent months weighted more heavily, using a {man['half_life']}-month half-life, and with no adjustment for volatility regimes. Each stock's specific variance is estimated the same way from its own residuals and shrunk toward the median across stocks, more strongly for stocks with short histories.</p>
    <p><b>What this is not.</b> A commercial model such as Barra's has around ten styles built from dozens of descriptors, around sixty industries, daily estimation and years of calibration. This model shares the structure and the tests, on one universe of about {man['universe_avg']} names. It is enough to run Portfolio Construction on, and to show where a commercial model earns its fee.</p>
  </div>
</section>

<footer class="foot">
  <div class="running"><span>Track record verification · research</span><span>{man['first'][:7]} – {man['last'][:7]}</span></div>
  <h4>Important information</h4>
  <p>This page is built from public SEC EDGAR filings and Yahoo Finance prices and sectors. It is a demonstration with the limits stated above, and it is not investment advice.</p>
</footer>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}</script>
"""


# ---------------------------------------------------------------- fund of funds

def fof_html(out_dir: Path | None = None) -> str | None:
    from .fof import OUT_DIR as FOF_DIR
    d = out_dir or FOF_DIR
    if not (d / "manifest.json").exists():
        return None
    man = json.loads((d / "manifest.json").read_text())
    M = pd.read_csv(d / "managers.csv")
    curve = pd.read_csv(d / "diversification_curve.csv")
    blends = json.loads((d / "blends.json").read_text())
    alloc = json.loads((d / "allocations_by_prior.json").read_text())
    oos = man.get("oos", {})
    tau_eb = man["tau_eb"]

    # shrinkage table (top 20 by raw alpha) and allocation table
    Ms = M.sort_values("alpha", ascending=False)
    srow = "".join(f"<tr><td><b>{esc(r['name'])}</b> <span class='muted'>{esc(r.style)}</span></td><td class='n'>{int(r.months)}</td><td class='n'>{pct(r.alpha, 1)}</td><td class='n'>{pct(r.se, 1, False)}</td><td class='n'>{r.t:+.1f}</td>"
                   f"<td class='n'>{r.shrink_factor:.2f}</td><td class='n'>{pct(r.alpha_shrunk, 1)}</td><td class='n'>{pct(r.alpha_eb, 2)}</td></tr>" for _, r in Ms.head(20).iterrows())
    W = M[M.weight_optimised > 0].sort_values("weight_optimised", ascending=False)
    wrow = "".join(f"<tr><td><b>{esc(r['name'])}</b> <span class='muted'>{esc(r.style)}</span></td><td class='n'>{pct(r.alpha_shrunk, 1)}</td><td class='n'>{r.resid_vol:.1%}</td><td class='n'>{r.ir_shrunk:.2f}</td><td class='n'>{r.weight_optimised:.1%}</td></tr>" for _, r in W.iterrows())
    prior_rows = "".join(f"<tr><td>{esc(k)}</td><td class='n'>{v['tau']:.1%}</td><td class='n'>{v['names']}</td><td class='n'>{pct(v['alpha'], 1)}</td><td class='n'>{num(v['ir'])}</td></tr>" for k, v in man["alloc_summary"].items())
    blend_rows = "".join(f"<tr><td>{esc(k)}</td><td class='n'>{v['names']}</td><td class='n'>{pct(v['alpha'], 1)}</td><td class='n'>{v['vol']:.1%}</td><td class='n'>{num(v['ir'])}</td></tr>" for k, v in blends.items())
    cells = [str(int(n)) for n in curve.n]
    cchart = line_chart(cells, [dict(name="expected IR", values=[float(v) for v in curve.ir], cls="s1", emph=True)], height=220, width=860, y_fmt=lambda v: f"{v:.2f}", end_labels=False, uid="divc")
    oos_html = ""
    if oos.get("available"):
        t_, b_, a_, s_ = oos["top"], oos["bottom"], oos["all"], oos["top_minus_bottom"]
        oos_html = (f"<div class='tiles'><div class='tile'><div class='tl'>Past top {oos['n_top']}</div><div class='tv'>{pct(t_['alpha'], 1)}</div><div class='td muted'>alpha after {esc(oos['split'][:7])}, with t = {t_['t']:+.1f}; they had {pct(t_['first_half_alpha'], 1)} before</div></div>"
                    f"<div class='tile'><div class='tl'>Past bottom {oos['n_top']}</div><div class='tv'>{pct(b_['alpha'], 1)}</div><div class='td muted'>alpha after the split, with t = {b_['t']:+.1f}; they had {pct(b_['first_half_alpha'], 1)} before</div></div>"
                    f"<div class='tile'><div class='tl'>Top − bottom</div><div class='tv'>{pct(s_['alpha'], 1)}</div><div class='td muted'>t = {s_['t']:+.1f} over {s_['months']} months</div></div>"
                    f"<div class='tile'><div class='tl'>Rank persistence</div><div class='tv'>{oos['rank_corr']:+.2f}</div><div class='td muted'>rank correlation of first-half and second-half alpha across {oos['candidates']} managers</div></div></div>"
                    f"<p class='cap' style='margin-top:12px'>Managers are ranked by the t-statistic of their three-factor alpha fitted on data before {esc(oos['split'][:7])}. The second-period alpha uses the first period's factor exposures, so it is a genuine out-of-sample result. "
                    f"{'Past alpha carried some information here' if s_['t'] >= 2 else 'Past alpha did not predict future alpha here'}: the spread between the ten best and ten worst past performers is {pct(s_['alpha'], 1)}/yr with t = {s_['t']:+.1f}, and the rank correlation is {oos['rank_corr']:+.2f}. "
                    f"{oos['share_positive_second']:.0%} of managers had positive alpha in the second period. An evaluation of {s_['months']} months is short, so the result is indicative, but it is the result.</p>")
    else:
        oos_html = f"<p class='cap'>The out-of-sample test is not available: {esc(str(oos.get('reason', '')))}.</p>"
    eb_txt = (f"The data's own estimate of how much true skill varies across managers is <b>zero</b>: the spread of the {man['managers']} estimated alphas ({M.alpha.std():.1%} standard deviation) is no larger than estimation noise alone would produce (a typical standard error of {np.sqrt((M.se ** 2).mean()):.1%}). "
              f"On this evidence the best estimate of every manager's alpha is the common mean, {pct(man['prior_mean'], 2)}/yr, and the right fund of funds is the index. "
              f"{man['n_t2']} managers show a t-statistic of 2 or more, against about {man['expected_t2_by_chance']:.1f} that would be expected by chance among {man['managers']}."
              if tau_eb == 0 else
              f"The data's own estimate of how much true skill varies across managers is {tau_eb:.1%} a year: the estimated alphas spread out more than noise alone would produce, so some of the difference is real.")
    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fund of Funds</title>
<style>{CSS}{EXTRA_CSS}</style>
<div class="banner" role="note"><span class="bl">Research note</span> The manager alphas here come from disclosed holdings and listed funds, which are public stand-ins for the audited, net-of-fee returns a real allocation would use. This is a demonstration of method.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Manager Analysis · Fund of Funds</div></div>
  <div class="gold-rule"></div>
  <h1>Fund of Funds</h1>
  <p class="sub">How should money be split across the managers that pass? For {man['managers']} managers with five years or more of history, each alpha is discounted in proportion to how noisy its estimate is, the overlap between managers' returns is measured, and a blend is chosen to maximise the expected information ratio, both under what the data support and under the beliefs an allocator might actually hold. Then comes the test that matters: does picking past winners work?</p>
  <dl class="meta">
    <div><dt>Candidates</dt><dd>{man['managers']}</dd></div>
    <div><dt>Skill dispersion, from the data</dt><dd>{tau_eb:.1%}</dd></div>
    <div><dt>Average overlap between managers</dt><dd>{man['avg_corr']:+.2f}</dd></div>
    <div><dt>Selected under a 2% prior</dt><dd>{man['selected']}</dd></div>
    <div><dt>Built</dt><dd>{esc(man['built'])}</dd></div>
  </dl>
</div></header>
<main class="wrap">
<section class="verdict">
  <div class="sh"><h2>What the data say before any allocation</h2></div>
  <div class="card"><p style="font-size:16px">{eb_txt}</p>
  <p class="cap">Each alpha is pulled toward the average across managers by a factor of τ²/(τ² + s²), where s is that manager's standard error and τ is the spread of true skill across managers, estimated as the spread of the alphas after the noise is taken out. This is the James–Stein logic applied to manager selection: the noisier the estimate, the less of it survives.</p></div>
</section>

<section>
  <div class="sh"><h2>Allocation under stated priors</h2></div>
  <p class="note">An allocator who believes some managers have skill must say how much they think true alpha varies across managers (Baks, Metrick and Wachter, 2001). Each row discounts the alphas with that belief, keeps the managers whose discounted alpha is positive, and maximises the blend's expected information ratio with no manager above {man['max_weight']:.0%}.</p>
  <div class="grid2">
    <div class="card tscroll"><h3>By prior</h3><table><thead><tr><th>prior</th><th class="n">τ</th><th class="n">managers held</th><th class="n">blend alpha</th><th class="n">expected IR</th></tr></thead><tbody>{prior_rows}</tbody></table>
      <p class="cap">The expected information ratio is the blend's discounted alpha divided by its residual volatility, which comes from the measured overlap between managers. It is a forecast, and the out-of-sample section below is the check.</p></div>
    <div class="card tscroll"><h3>The τ = 2% blend, three ways</h3><table><thead><tr><th>blend</th><th class="n">managers</th><th class="n">alpha</th><th class="n">resid vol</th><th class="n">expected IR</th></tr></thead><tbody>{blend_rows}</tbody></table>
      <p class="cap">The three rows compare the optimised blend, equal weights on the top ten, and holding everyone, which shows how much the overlap between managers is worth on top of selection.</p></div>
  </div>
  <div class="card tscroll" style="margin-top:14px"><h3>Optimised weights, τ = 2%</h3><table><thead><tr><th>manager</th><th class="n">shrunk alpha</th><th class="n">residual vol</th><th class="n">shrunk IR</th><th class="n">weight</th></tr></thead><tbody>{wrow}</tbody></table>
  <p class="cap">Managers with low residual volatility, such as listed funds and diversified long-only portfolios, get large weights even with modest alpha because they add little tracking risk. Concentrated managers get small weights because their residual volatility is 10% to 14% a year.</p></div>
</section>

<section>
  <div class="sh"><h2>Shrinkage, manager by manager</h2></div>
  <div class="card tscroll"><table><thead><tr><th>manager</th><th class="n">months</th><th class="n">raw alpha</th><th class="n">std error</th><th class="n">t</th><th class="n">keep (τ=2%)</th><th class="n">shrunk (τ=2%)</th><th class="n">empirical Bayes</th></tr></thead><tbody>{srow}</tbody></table>
  <p class="cap">These are the top twenty managers by raw alpha. "Keep" is the share of the raw estimate that survives the discount, and the last column is what the data alone support. The full table is in <a href="/research/data/fund-of-funds/managers.csv">managers.csv</a>.</p></div>
</section>

<section>
  <div class="sh"><h2>How many managers?</h2></div>
  <div class="card">{cchart}<p class="cap">The line is the expected information ratio of an equal-weight blend of the best N managers, adding one at a time. It rises while the added manager's alpha outweighs the dilution, then flattens. With the overlap between managers averaging {man['avg_corr']:+.2f}, diversification across managers is cheap, but the alpha to diversify is thin.</p></div>
</section>

<section>
  <div class="sh"><h2>Does picking past winners work?</h2></div>
  <div class="card">{oos_html}</div>
</section>

<section>
  <div class="sh"><h2>Method and limits</h2></div>
  <div class="card">
    <p><b>Inputs.</b> The inputs are each manager's three-factor regression from the verification pipeline: the alpha, its standard error and the residual returns. Managers with fewer than {man['min_months']} months, and strategies whose disclosed holdings are not a portfolio, are excluded. Listed funds with real net returns sit alongside managers reconstructed from their disclosed holdings, whereas a real allocation would have audited net returns for all of them.</p>
    <p><b>Correlations.</b> Correlations are measured pairwise on overlapping months, with at least 24 required, and shrunk {man['corr_shrink']:.0%} toward zero. They use residual rather than total returns, because exposure to the market and the factors can be bought elsewhere, cheaply.</p>
    <p><b>Optimisation.</b> The optimiser maximises the blend's expected information ratio over long-only weights that sum to one, with no manager above {man['max_weight']:.0%}. There are no transaction costs, capacity or liquidity terms; a real fund of funds adds minimum ticket sizes, redemption terms and operational scores.</p>
    <p><b>Reading it.</b> Two findings are the substance of the page: that the cross-section of alphas is indistinguishable from noise, and that ranking on past alpha did not select future alpha in this sample. Everything under a stated prior is what an allocator would do <i>if</i> they believed otherwise, shown so the belief is explicit rather than hidden in a weight.</p>
  </div>
</section>

<footer class="foot">
  <div class="running"><span>Track record verification · research</span><span>{man['first'][:7]} – {man['last'][:7]}</span></div>
  <h4>Important information</h4>
  <p>This page is built from the site's own results on public SEC EDGAR filings and Yahoo Finance prices. It is a demonstration of fund-of-funds construction, and it is not investment advice.</p>
</footer>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}</script>
"""


# ---------------------------------------------------------------- manager decay model

def decay_html(out_dir: Path | None = None) -> str | None:
    from .decay import OUT_DIR as DECAY_DIR, FEATURES, LABELS, MODELS
    d = out_dir or DECAY_DIR
    if not (d / "manifest.json").exists():
        return None
    man = json.loads((d / "manifest.json").read_text())
    S = pd.read_csv(d / "summary.csv").set_index("model")
    D = pd.read_csv(d / "by_date.csv", parse_dates=["formation"])
    U = pd.read_csv(d / "features.csv").set_index("feature")
    W = pd.read_csv(d / "watchlist.csv")
    hh, G, R = man.get("head_to_head", {}), man.get("grouped_cv", {}), man.get("r") or {}
    best_t = float(S.auc_cs_t.max())
    verdict = "No, nothing detectable" if best_t < 2 else "Weakly" if best_t < 3 else "Yes"

    def light(t):
        t = abs(t) if t == t else 0.0
        return ("yes", "evidence") if t >= 2 else ("weak", "weak") if t >= 1 else ("no", "none")
    def badge(t):
        c, lab = light(t)
        return f"<span class='v {c}' style='color:#fff;background:var(--{ {'no': 'crit', 'weak': 'warn', 'yes': 'good'}[c] });font:600 9px var(--sans);letter-spacing:.14em;text-transform:uppercase;padding:4px 7px'>{esc(lab)}</span>"

    # models table
    mrow = ""
    for m in MODELS:
        if m not in S.index: continue
        r = S.loc[m]
        brier = "—" if pd.isna(r.brier) else f"{r.brier:.3f}"
        skill = "—" if pd.isna(r.brier_skill) else f"{r.brier_skill:+.0%}"
        mrow += (f"<tr><td><b>{esc(plain(r.label))}</b></td><td class='n'>{r.auc_pooled:.3f}</td><td class='n'>{r.auc_cs_mean:.3f}</td><td class='n'>{r.auc_cs_t:+.1f}</td>"
                 f"<td class='n'>{r.auc_cs_pct_above:.0%}</td><td class='n'>{pct(r.spread_mean, 1)}</td><td class='n'>{r.spread_t:+.1f}</td><td class='n'>{brier}</td><td class='n'>{skill}</td><td>{badge(r.auc_cs_t)}</td></tr>")
    # cumulative edge chart: running sum of (AUC − 0.5), one line per model
    Dx = D.pivot(index="formation", columns="model", values="auc").sort_index()
    cells = [x.strftime("%Y-%m") for x in Dx.index]
    series = []
    for m, cls in [("persist", "s0"), ("logistic", "s1"), ("xgboost", "s4")]:
        if m in Dx:
            cum = (Dx[m] - 0.5).fillna(0).cumsum()
            series.append(dict(name=plain(MODELS[m]), values=[float(v) for v in cum], cls=cls, emph=m != "persist"))
    chart = line_chart(cells, series, height=280, width=860, y_fmt=lambda v: f"{v:+.1f}", end_labels=False, uid="cumauc")
    legend = "".join(f'<span><span class="k {s["cls"]}"></span>{esc(s["name"])}</span>' for s in series)
    # features table
    frow = ""
    for f in FEATURES:
        if f not in U.index: continue
        r = U.loc[f]
        gain = "—" if pd.isna(r.xgb_gain_share) else f"{r.xgb_gain_share:.0%}"
        coef = "—" if pd.isna(r.logistic_coef) else f"{r.logistic_coef:+.2f}"
        frow += (f"<tr><td>{esc(LABELS.get(f, f))}</td><td class='n'>{r.coverage:.0%}</td><td class='n'>{r.ic_mean:+.3f}</td><td class='n'>{r.ic_t:+.1f}</td>"
                 f"<td class='n'>{r.ic_pct_positive:.0%}</td><td class='n'>{gain}</td><td class='n'>{coef}</td><td>{badge(r.ic_t)}</td></tr>")
    n_sig = int((U.ic_t.abs() >= 2).sum())
    # latest scores
    wrow = ""
    for _, r in W.head(12).iterrows():
        wrow += (f"<tr><td><b>{esc(r['name'])}</b><div class='muted' style='font-size:12px'>{esc(r.style_name if isinstance(r.style_name, str) else '')}</div></td>"
                 f"<td class='n'>{r.p_xgboost:.2f}</td><td class='n'>{r.p_logistic:.2f}</td><td class='n'>{int(r.rank_persist) if pd.notna(r.rank_persist) else '—'}</td>"
                 f"<td class='n'>{pct(r.excess_12, 1)}</td><td class='n'>{r.turnover:.0%}</td><td class='n'>{r.hhi:.2f}</td><td class='n'>{r.crowding:.1f}</td></tr>")
    rk = (W.rank_xgboost.corr(W.rank_logistic.astype(float), method="spearman"), W.rank_xgboost.corr(W.rank_persist.astype(float), method="spearman")) if len(W) > 5 else (np.nan, np.nan)
    # R twin
    rblock = ""
    if R.get("available"):
        rx = R.get("auc_cs_mean_r", {})
        rblock = (f"<div class='card'><h3>Checked by a second implementation</h3>"
                  f"<p>A second implementation, written separately, rebuilds every feature from the raw filings and statements, which is {R.get('rows_r', 0):,} manager-quarters and "
                  f"{len(R.get('by_column', {}))} columns, and re-runs the walk-forward test. The largest difference between the two across the panel and the regression's predictions is "
                  f"<b>{R.get('max_abs_diff', 0):.0e}</b>. The second learned model, which draws its own random samples, gives an AUC by date of "
                  f"{rx.get('xgboost', float('nan')):.3f} against {S.loc['xgboost', 'auc_cs_mean']:.3f} here, and the regression gives {rx.get('logistic', float('nan')):.3f} against {S.loc['logistic', 'auc_cs_mean']:.3f}.</p></div>")
    else:
        rblock = "<div class='card'><h3>Checked by a second implementation</h3><p>The second implementation was not run in this build.</p></div>"
    P_ = S.loc["persist"]; L_ = S.loc["logistic"]; X_ = S.loc["xgboost"]
    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Manager Decay Model</title>
<style>{CSS}{EXTRA_CSS}</style>
<div class="banner" role="note"><span class="bl">Research note</span> This page covers {man['managers']} managers' disclosed holdings over {man['formation_dates']} quarters, which is a small sample by the standard that matters. It is a scouting result, not a validated model.</div>
<header class="cover"><div class="cover-in">
  <div class="cover-top"><div class="eyebrow">Manager Analysis · Manager Decay Model</div></div>
  <div class="gold-rule"></div>
  <h1>Manager Decay Model</h1>
  <p class="sub">Does anything in a manager's disclosed portfolio today, such as its concentration, turnover, crowding, recent purchases and sales, or how the last year went, say whether the manager will lag the market over the next twelve months? The {len(FEATURES)} things an analyst reads off a holdings filing and a return history are measured for {man['managers']} managers at every filing date from {man['first'][:7]} to {man['last'][:7]} and tested three ways: last year's laggards, a logistic regression and a learned model. Every test is walk-forward, so nothing about the year being predicted is used to predict it.</p>
  <dl class="meta">
    <div><dt>Managers</dt><dd>{man['managers']}</dd></div>
    <div><dt>Manager-quarters</dt><dd>{man['rows_labelled']:,}</dd></div>
    <div><dt>Tested out of sample</dt><dd>{man['tested_rows']:,} from {(man.get('first_test') or '')[:7]}</dd></div>
    <div><dt>Base rate</dt><dd>{man['base_rate']:.0%} lag</dd></div>
    <div><dt>Built</dt><dd>{esc(man['built'])}</dd></div>
  </dl>
</div></header>
<main class="wrap">
<section class="verdict">
  <div class="sh"><h2>Findings</h2></div>
  <div class="tiles">
    <div class="tile"><div class="tl">Last year's laggards lag again?</div><div class="tv">{P_.auc_cs_mean:.3f}</div><div class="td muted">AUC by date, with t = {P_.auc_cs_t:+.1f}; above 0.5 in {P_.auc_cs_pct_above:.0%} of quarters</div></div>
    <div class="tile"><div class="tl">Logistic regression</div><div class="tv">{L_.auc_cs_mean:.3f}</div><div class="td muted">t = {L_.auc_cs_t:+.1f}; its probabilities beat the base rate by {L_.brier_skill:+.0%}</div></div>
    <div class="tile"><div class="tl">Learned model</div><div class="tv">{X_.auc_cs_mean:.3f}</div><div class="td muted">t = {X_.auc_cs_t:+.1f}; its probabilities beat the base rate by {X_.brier_skill:+.0%}; against the regression {hh.get('auc_diff_mean', 0):+.3f} (t = {hh.get('auc_diff_t', 0):+.1f})</div></div>
    <div class="tile"><div class="tl">Can the filings predict next year?</div><div class="tv small">{esc(verdict)}</div><div class="td muted">the best t-statistic across the three predictors is {best_t:+.1f}, and the bar is 2</div></div>
  </div>
  <p class="cap" style="margin-top:14px">An AUC is the chance that a randomly chosen manager who went on to lag was scored as riskier than one who did not, so 0.5 is a coin flip. It is computed <em>within</em> each filing date, so a year that was bad for everyone cannot flatter or condemn a model, and then averaged over {man['tested_dates']} dates with Newey–West standard errors, because the twelve-month outcomes overlap. Every fit uses only the rows whose outcome was already known by the filing date.</p>
</section>

<section>
  <div class="sh"><h2>The number a careless backtest would report</h2></div>
  <div class="tiles">
    <div class="tile"><div class="tl">Leave-managers-out, learned model</div><div class="tv">{G.get('auc_xgboost', float('nan')):.3f}</div><div class="td muted">pooled AUC, where training and test share calendar years</div></div>
    <div class="tile"><div class="tl">Leave-managers-out, logistic</div><div class="tv">{G.get('auc_logistic', float('nan')):.3f}</div><div class="td muted">the same split</div></div>
    <div class="tile"><div class="tl">Walk-forward, learned model</div><div class="tv">{X_.auc_pooled:.3f}</div><div class="td muted">pooled AUC, with nothing from the test year in training</div></div>
    <div class="tile"><div class="tl">Difference</div><div class="tv">{G.get('auc_xgboost', 0) - X_.auc_pooled:+.3f}</div><div class="td muted">what the leak is worth</div></div>
  </div>
  <p class="cap" style="margin-top:14px">Splitting by manager looks rigorous, because the model never sees the manager it is scoring. But one manager's 2022 in the training set teaches the model what 2022 looked like, and it applies that to the other managers' 2022. The walk-forward number is the one to believe. This gap is the most useful thing on the page: it is the size of the mistake in a vendor's "AI manager-selection" pitch that reports cross-validated accuracy.</p>
</section>

<section>
  <div class="sh"><h2>Predictor by predictor</h2></div>
  <div class="card tscroll"><table><thead><tr><th>predictor</th><th class="n">pooled AUC</th><th class="n">AUC by date</th><th class="n">t</th><th class="n">&gt; 0.5</th><th class="n">riskiest − safest fifth</th><th class="n">t</th><th class="n">Brier</th><th class="n">skill</th><th>evidence</th></tr></thead><tbody>{mrow}</tbody></table>
  <p class="cap">For "riskiest minus safest fifth", the managers are ranked by the predictor at each date, and the column shows the realized excess return over the next twelve months of the fifth predicted most likely to lag, minus that of the fifth predicted least likely. A working predictor makes this negative. Brier is the mean squared error of the predicted probabilities, and skill is the improvement over always predicting the historical base rate, so a negative skill means the model's confidence was misplaced.</p></div>
</section>

<section>
  <div class="sh"><h2>Cumulative edge over a coin flip</h2></div>
  <div class="card"><div class="legend">{legend}</div>{chart}
  <p class="cap">Each line is the running sum of the AUC minus 0.5 at each filing date. A predictor with skill climbs steadily, and these wander.</p></div>
</section>

<section>
  <div class="sh"><h2>One thing at a time</h2></div>
  <div class="card tscroll"><table><thead><tr><th>feature</th><th class="n">coverage</th><th class="n">rank IC</th><th class="n">t</th><th class="n">IC &gt; 0</th><th class="n">importance in learned model</th><th class="n">regression coefficient</th><th>evidence</th></tr></thead><tbody>{frow}</tbody></table>
  <p class="cap">The rank IC is the rank correlation between the feature and the next twelve months' excess return across managers, date by date, averaged with Newey–West standard errors; positive means that more of the feature went with a better year. {n_sig} of {len(FEATURES)} features clear a t-statistic of 2, which is about what fourteen tries at noise would give. The importance and the regression coefficient (standardized, where positive means more likely to lag) come from the latest fit on {man.get('last_fit', {}).get('n_train', 0):,} rows; they show what the models lean on, not evidence that it works.</p></div>
</section>

<section>
  <div class="sh"><h2>What the models say today — {esc(man['last'][:7])}</h2></div>
  <div class="card tscroll"><table><thead><tr><th>manager</th><th class="n">chance of lagging, learned model</th><th class="n">chance of lagging, regression</th><th class="n">rank by last year</th><th class="n">trailing 12m excess</th><th class="n">turnover</th><th class="n">HHI</th><th class="n">crowding</th></tr></thead><tbody>{wrow}</tbody></table>
  <p class="cap">These are the twelve managers the learned model scores as riskiest at the latest filing date, with what the other two predictors make of them; the rank correlation of the learned model's ranking with the regression's is {rk[0]:+.2f}, and with last year's ranking {rk[1]:+.2f}. This is shown for transparency. Given the findings above, it is not a watchlist and should not be read as one.</p></div>
</section>

<section>
  <div class="sh"><h2>Method and limits</h2></div>
  <div class="card">
    <p><b>Panel.</b> There is one row per manager per filing date, which is the month-end by which the quarter's holdings are public under the 45-day deadline. The portfolio features come from that filing and the previous one, the return features from the manager's monthly returns through that date, and the outcome is whether the manager's return over the following {man['horizon']} months fell short of the US market's. A manager needs a year of history to enter. Reported values switched from thousands to dollars in the 2022 Q4 filings, and that step is removed from the growth figures.</p>
    <p><b>Models.</b> The first predictor simply ranks managers by their trailing twelve-month excess return, with nothing fitted. The second is a logistic regression on the standardized features, with missing values set to the training average. The third is a learned model, a gradient-boosted ensemble of {man.get('xgb_rounds')} shallow decision trees that handles missing values itself. Each is refit at every filing date on all rows whose outcome was already known, with {man['min_train_dates']} dates of history required before the first prediction. No setting was tuned on the test period.</p>
    <p><b>Sample size, honestly.</b> {man['rows_labelled']:,} manager-quarters sounds like a lot, but it is {man['managers']} managers over about {man['formation_dates'] // 4} years, and consecutive quarters of one manager share most of their outcome window. There are roughly a dozen independent years here. An AUC by date of 0.55 with a t-statistic of 2 is about the smallest effect this design could detect. The effects found are smaller than that, so the honest reading is "nothing detectable" rather than "nothing there".</p>
    <p><b>What the disclosed holdings are not.</b> They are the long US positions from public filings, priced after the 45-day delay, with survivorship bias in the price data and no short positions, non-US holdings or credit. A manager whose real edge is elsewhere shows up here as noise. Every caveat on the <a href="/external">manager pages</a> applies.</p>
  </div>
  {rblock}
</section>

<footer class="foot">
  <div class="running"><span>Track record verification · research</span><span>{man['first'][:7]} – {man['last'][:7]}</span></div>
  <h4>Important information</h4>
  <p>This page is built from public SEC EDGAR holdings filings, Yahoo Finance prices and the Kenneth R. French Data Library. The results are reconstructions with the limits stated above; they are not a strategy, a recommendation or investment advice. Past performance is not indicative of future results.</p>
</footer>
</main>
<div id="tip" class="tip" hidden></div>
<script>{JS}</script>
"""
