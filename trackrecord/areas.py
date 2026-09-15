"""The site organised the way the job is: three areas, one tool per responsibility.

    Passive            index tracking, daily portfolio status, rebalance trade lists
    Active             alpha model lab, risk model, portfolio construction, verification in R
    External managers  manager verification and screening, due-diligence memos, 13F signal
                       research, fund-of-funds construction

Every card's numbers are read from the committed CSVs under data/research/ (or the
leaderboard), so nothing on these pages can go stale, and a tool whose data is missing
is simply not shown rather than shown with a broken claim.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data" / "research"

AREAS = {
    "passive": ("Passive portfolios", "/research/index-tracker",
                "Run an index-tracking portfolio the way the desk does: replicate the benchmark with a subset of names, "
                "keep tracking error and cash where the mandate says, and produce the day's status sheet and trade list."),
    "active": ("Active portfolios", "/active",
               "Build and run a quantitative equity portfolio: research the signals, model the risk, construct the "
               "portfolio under the mandate's constraints, and prove every number twice."),
    "external": ("Manager Analysis", "/external",
                 "Evaluate outside managers the way a manager-research team does: verify the record, separate skill from "
                 "exposure, test it against luck, write the memo — and combine the ones that pass into a fund of funds."),
}


def _rows(path, key=None):
    f = R / path
    if not f.exists():
        return {}
    with f.open() as fh:
        rs = list(csv.DictReader(fh))
    return {r[key]: r for r in rs} if key else rs


def _man(path):
    f = R / path
    try:
        return json.loads(f.read_text()) if f.exists() else {}
    except Exception:
        return {}


def _num(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def tool_cards() -> dict[str, list[dict]]:
    """Live tools by area.  Each: href, eyebrow, title, what, stats [(value, label)]."""
    cards = {"passive": [], "active": [], "external": []}

    # ---- active
    lab, labman = _rows("alpha-lab/summary.csv", "signal"), _man("alpha-lab/manifest.json")
    if lab and labman:
        best = max(lab.values(), key=lambda r: _num(r.get("ic_t"), -99))
        xg = lab.get("xgboost", {}); lin = lab.get("linear", {})
        cards["active"].append(dict(href="/research/alpha-lab", eyebrow="Alpha Model Lab",
            title="Which signals predict returns, and does a tree model beat a line?",
            what=f"Value, momentum, quality, low-volatility and size signals from prices and SEC filings, ranked every quarter across {labman.get('universe_avg', '')} names; "
                 f"information coefficients, decile spreads, and a walk-forward gradient-boosted model (xgboost) against a linear composite.",
            stats=[(f"{_num(best.get('ic_mean'), 0):+.3f}", f"best IC · {best.get('label', best.get('signal', ''))}"),
                   (f"{_num(xg.get('ic_mean'), 0):+.3f}", "xgboost IC, walk-forward"), (f"{_num(lin.get('ic_mean'), 0):+.3f}", "linear composite IC")]))
    risk, riskman = _man("risk-model/manifest.json"), None
    if risk.get("factors"):
        cards["active"].append(dict(href="/research/risk-model", eyebrow="Factor Risk Model",
            title="A fundamental factor risk model, Barra-style",
            what="Monthly cross-sectional regressions of stock returns on style exposures and industries give a factor covariance and specific risk; "
                 "any portfolio decomposes into factor and stock-specific risk, with a predicted tracking error to check against what happened.",
            stats=[(f"{risk.get('factors')}", "factors"), (f"{risk.get('industries')}", "industries"),
                   (f"{_num(risk.get('r2_avg'), 0):.0%}", "avg cross-sectional R²")]))
    con, conman = _rows("construction/summary.csv", "key"), _man("construction/manifest.json")
    if con and conman:
        P = con.get("portfolio", {})
        cards["active"].append(dict(href="/research/construction", eyebrow="Portfolio Construction",
            title="From a signal to a trade list",
            what="A linear program turns alpha scores into target weights under name caps, sector and active bands, an active-share cap and a turnover budget, "
                 "then into the tickets a trader would receive. Solved in Python (HiGHS) and again in R (Rglpk).",
            stats=[(f"{_num(P.get('active_return'), 0) * 100:+.1f}%", "active return /yr"), (f"{_num(P.get('information_ratio'), 0):.2f}", "information ratio"),
                   (f"{conman.get('rebalances', '')}", "rebalances")]))
    rv = _man("r-verify/manifest.json")
    if rv.get("managers"):
        cards["active"].append(dict(href="/research/r-verify", eyebrow="R Verification",
            title="The same statistics, recomputed in R",
            what="Every headline number — alphas, Newey–West t-statistics, loadings, Sharpe, drawdown, scores — is recomputed from the aligned returns by an "
                 "independent R implementation with data.table and compared with what Python reported.",
            stats=[(f"{rv.get('managers')}", "managers"), (f"{rv.get('checks', 0):,}", "numbers checked"), (f"{rv.get('max_abs_diff', 0):.0e}", "largest difference")]))

    # ---- passive
    idx = _man("index-tracker/manifest.json")
    if idx.get("benchmark"):
        cards["passive"].append(dict(href="/research/index-tracker", eyebrow="Index Tracking — live book",
            title=f"Replicate {idx['benchmark']} with fewer names, and run it daily",
            what="Optimised sampling: hold a subset of the index that minimises predicted tracking error under the risk model, rebalance on the benchmark's "
                 "reconstitution, and keep cash within the mandate. Daily Portfolio Status Worksheet and rebalance trade list included.",
            stats=[(f"{idx.get('names_held')}", f"of {idx.get('names_index')} names"), (f"{_num(idx.get('te_realized'), 0):.2%}", "realized tracking error"),
                   (f"{_num(idx.get('turnover'), 0):.0%}", "turnover /yr")]))

    # ---- external
    sig, sigman = _rows("13f-signals/summary.csv", "key"), _man("13f-signals/manifest.json")
    if sig and sigman:
        t = _num(sig.get("BEST1", {}).get("carhart_t"))
        cards["external"].append(dict(href="/research/13f-signals", eyebrow="13F Signal Research",
            title="Trading on 13F disclosures",
            what=f"Best ideas, crowding and fresh buys from every filing, formed into quarterly portfolios and tested as Carhart spreads. "
                 f"Finding: {'no edge survives the 45-day lag' if t is None or abs(t) < 2 else 'a spread that survives the lag'}.",
            stats=[(f"{sigman.get('managers', '')}", "managers"), (f"{sigman.get('quarters', '')}", "quarters"), (f"{_num(sigman.get('positions'), 0):,.0f}", "positions")]))
    fof = _man("fund-of-funds/manifest.json")
    if fof.get("managers"):
        cards["external"].append(dict(href="/research/fund-of-funds", eyebrow="Fund of Funds",
            title="Allocating across managers",
            what="Shrink each manager's alpha toward zero in proportion to its noise, estimate the correlation of their active returns, and allocate: "
                 "how many managers diversification actually rewards, and what the blend's expected information ratio is.",
            stats=[(f"{fof.get('managers')}", "candidates"), (f"{fof.get('selected')}", "selected"), (f"{_num(fof.get('ir_blend'), 0):.2f}", "blend IR, shrunk")]))
    return cards


CARD_CSS = """<style>
.rcards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(440px,100%),1fr));gap:14px;margin:6px 0 10px}
.rcard{display:flex;flex-direction:column;background:var(--surface);border:1px solid var(--line);border-top:2px solid var(--gold);padding:18px 20px;text-decoration:none;color:var(--ink)}
.rcard .eyeb{font:600 9px/1 var(--sans);letter-spacing:.22em;text-transform:uppercase;color:var(--muted)}
.rcard .rt{font:400 21px/1.2 var(--serif);color:var(--navy);margin:8px 0 6px}
.rcard .rw{font-size:13.5px;color:var(--ink2);line-height:1.5;flex:1}
.rcard .rs{display:flex;gap:18px;margin:14px 0 0;flex-wrap:wrap}.rcard .rs div{display:grid;gap:3px}
.rcard .rs .v{font:600 16px/1 var(--sans);color:var(--navy)}
.rcard .rs .l{font:600 8.5px/1.2 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.rcard .go{margin-top:14px;font:600 9.5px/1 var(--sans);letter-spacing:.18em;text-transform:uppercase;color:var(--gold)}
.rcard:hover{border-color:var(--gold)}
.areas{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(300px,100%),1fr));gap:14px;margin:26px 0 0}
.area{display:grid;gap:8px;padding:22px 24px;text-decoration:none;border:1px solid var(--goldl);color:var(--coverink);background:rgba(232,228,218,.04)}
.area .t{font:400 24px/1.15 var(--serif)}.area .s{font-size:13.5px;line-height:1.5;color:var(--covermuted)}.area .n{font:600 9.5px/1.3 var(--sans);letter-spacing:.18em;text-transform:uppercase;color:var(--goldl);margin-top:6px}
.area:hover{filter:brightness(1.12)}
.stats{display:flex;flex-wrap:wrap;gap:22px 40px;margin-top:30px;padding-top:18px;border-top:1px solid rgba(232,228,218,.18)}.stats div{display:grid;gap:4px}.stats dt{font:600 9px/1 var(--sans);letter-spacing:.22em;text-transform:uppercase;color:var(--covermuted)}.stats dd{margin:0;font:400 30px/1 var(--serif);color:var(--coverink)}
</style>"""


def cards_html(cards: list[dict]) -> str:
    out = ""
    for c in cards:
        st = "".join(f"<div><span class='v'>{html.escape(str(v))}</span><span class='l'>{html.escape(l)}</span></div>" for v, l in c["stats"])
        out += (f"<a class='rcard' href=\"{c['href']}\"><div class='eyeb'>{html.escape(c['eyebrow'])}</div><div class='rt'>{html.escape(c['title'])}</div>"
                f"<div class='rw'>{c['what']}</div><div class='rs'>{st}</div></a>")
    return out


def home_html(page, n_managers: int, n_scorable: int) -> str:
    cards = tool_cards()
    n_tools = sum(len(v) for v in cards.values())
    areas = "".join(f"<a class='area' href='{href}'><span class='t'>{html.escape(name)}</span><span class='s'>{html.escape(blurb)}</span>"
                    f"<span class='n'>{'open the live book' if k == 'passive' else str(len(cards[k])) + ' tool' + ('s' if len(cards[k]) != 1 else '')} &rarr;</span></a>" for k, (name, href, blurb) in AREAS.items() if cards[k])
    sections = "".join(f"<h2 data-n='{html.escape(name)}' id='{k}'>{html.escape(name)}</h2><p class='note'>{html.escape(blurb)}</p><div class='rcards'>{cards_html(cards[k])}</div>"
                       for k, (name, href, blurb) in AREAS.items() if cards[k])
    return page("Global Equity — quantitative portfolio management", CARD_CSS + f"""<div class="banner"><b>Public data</b> SEC 13F reconstructions, listed funds and Ken French factors, used to exercise every tool. Nothing here is investment advice.</div>
<header class="cover"><div class="cover-in"><div class="eyebrow">Quantitative portfolio management · global equity</div><div class="rule"></div><h1>The work of a portfolio manager,<br>as working software</h1>
<p class="sub">Passive index portfolios, active quantitative portfolios, and the evaluation of external managers — one tool for each responsibility, built on public data and honest about what it finds.</p>
<div class="areas">{areas}</div>
<dl class="stats"><div><dt>Tools</dt><dd>{n_tools}</dd></div><div><dt>Managers in the system</dt><dd>{n_managers}</dd></div><div><dt>Scorable today</dt><dd>{n_scorable}</dd></div><div><dt>Implementations</dt><dd>Python · R</dd></div></dl>
</div></header>
<main class="wrap">{sections}
<div class="foot">Built by Simon Weardon with Claude Code as pair programmer; methodology, data-quality rules and verification are the author's. Benchmark and factors: Kenneth R. French Data Library; prices and sectors: Yahoo Finance; holdings and fundamentals: SEC EDGAR. Past performance is not indicative of future results; nothing here is investment advice.</div>
</main>""", is_home=True)


def area_html(page, key: str, extra_body: str = "", extra_head: str = "") -> str:
    name, href, blurb = AREAS[key]
    cards = tool_cards()[key]
    body = (f"<div class='rcards'>{cards_html(cards)}</div>" if cards else "<p class='note'>Tools for this area are being built.</p>")
    return page(name, CARD_CSS + extra_head + f"""<header class="cover"><div class="cover-in"><div class="eyebrow">Global equity · {html.escape(name.lower())}</div><div class="rule"></div><h1>{html.escape(name)}</h1>
<p class="sub">{html.escape(blurb)}</p></div></header>
<main class="wrap"><h2 data-n="Tools">What you can do here</h2>{body}{extra_body}</main>""", current=href)
