# Track-record reconstruction

Reconstruct and independently verify a ~30-year investment record from
custodian statements, then decompose and statistically test it.

**Live site:** https://dashboard-production-1086.up.railway.app

## What this is
An external-manager due-diligence engine, applied the same way to every record it is given: reconcile
the statements, compute time-weighted returns, remove what the market and known factors explain, simulate
how often luck alone does as well, test stability, score, and write the memo. It runs today on public
data — 108 prominent managers reconstructed from their SEC 13F filings, listed funds, and a synthetic
dataset with a known injected alpha that the pipeline must recover — so that every method is exercised
before a real record is loaded. Alongside it sit three research notes that use the same universe:
whether the disclosed books carry a tradable signal, an LP portfolio constructor that turns a signal into
a trade list under a mandate's constraints (solved in Python and in R with Rglpk), and an R reproduction of
every headline statistic. The findings are reported whichever way they come out; most of them are negative,
and the page says so.

**How it was built.** Designed and directed by Simon Weardon, with Claude Code (Anthropic) as pair
programmer: the methodology, the evidence tiers, the data-quality rules (what counts as verified, when a
month is dropped rather than bridged, why a 13F clone is not the fund) and the checks in the test suite
are the author's decisions; much of the code was written with the model and reviewed line by line. The
R reproduction exists so that no headline number rests on a single implementation.

## Status
| phase | state |
|---|---|
| 1 · Ingestion & reconciliation | schema, reconciler, coverage report **built and tested**; statement parser/OCR **waits for sample scans** |
| 2 · Return series & composites | **built and tested**; rules in [COMPOSITE_RULES.md](COMPOSITE_RULES.md) |
| 3 · Attribution | **built and tested**: beta decomposition, allocation effect, timing tests; selection-vs-trading needs security prices |
| 4 · Statistical validation | **built and tested**: CAPM/FF3/Carhart/FF5, risk metrics, null bootstrap, zero-skill cohort, rolling, sub-periods |
| Final report | `python -m trackrecord report` runs every phase and writes `REPORT.md` (verified / estimated / unknown + reconciliation to the claimed number) and `dashboard.html` |

All of it runs today on two placeholder datasets. Swapping in real statements changes nothing but the inputs.

| placeholder | what it is | what it exercises |
|---|---|---|
| `data/synthetic` | 12 invented accounts, real market history, +2%/yr injected alpha, six injected defects | Phase 1 reconciliation, composites with closed accounts, annual-grid inference, ground-truth recovery |
| `data/brk` | Berkshire Hathaway Class A monthly closes 1985–2026 (yfinance), one account, no statements — kept as a local test dataset; **no longer shown on the site** (removed 2026-09-13; Berkshire appears as its 13F clone instead) | Monthly grid: 36/60-month rolling alpha, real drawdowns, HAC errors; results check against "Buffett's Alpha" (β≈0.7, SMB<0, HML>0, alpha shrinking under FF5) |

The BRK series is the stock, not Buffett's public-equity book, and a price feed is not a custodian statement — Phase 1 marks every period unverified, which is the correct label.

## Evidence tiers (every number in the final report carries one)
- **Verified** — annual figures reconciled from binder statements (and monthly figures from Fidelity's online PDFs, once downloaded), where at least one independent check on the page passed.
- **Estimated** — e.g. monthly returns reconstructed from year-end holdings + public prices, assuming no intra-year trading; validated against the verified annual figure and labeled as an estimate everywhere.
- **Unknown** — periods with no statement, or statements with nothing independent to check against. Reported as holes. Never interpolated.

## Layout
```
trackrecord/   schema.py     the normalized tables (statements, flows, positions, accounts)
               load.py       CSV → typed frames, row-level validation
               reconcile.py  Phase 1: per-statement checks, flags, verified/flagged/unverified
               coverage.py   Phase 1: account × year matrix, gaps, flag lists → coverage.md
               returns.py    Phase 2: Modified Dietz on effective periods, TWR, XIRR, model fees
               composite.py  Phase 2: eligibility rules, aggregate composite, survivors/clients/principal, dispersion
               report2.py    Phase 2: summary.md + CSVs
               attribution.py Phase 3: beta decomposition, allocation vs residual, selection (needs prices), timing tests
               validation.py Phase 4: factor regressions, risk metrics, bootstrap, cohort, rolling, sub-periods
               report.py     REPORT.md assembler (+ reconciliation.csv)
               dashboard.py  dashboard.html: inline-SVG charts rendered from the output CSVs, light/dark, hover
               placeholder_brk.py  Berkshire BRK-A monthly placeholder in the statement format
               placeholder_ticker.py  any listed vehicle as a placeholder (yfinance, distributions reinvested)
               hedge13f.py   13F clone engine: EDGAR filings → holdings → CUSIP map → prices → monthly clone → statement format
               signals13f.py 13F research: best-ideas / crowding / conviction portfolios, spreads, ICs → data/research/13f-signals
               construct.py  LP portfolio construction (HiGHS) with trade list, active-share/sector/turnover constraints, ex-ante TE; backtest of a demo mandate
               memo.py       due-diligence memo per manager, assembled from phases 2–4 (/f/<slug>/memo)
               rverify.py    runs the R reproduction over every manager and writes the agreement → data/research/r-verify
               research_pages.py  research notes rendered from those CSVs (/research/13f-signals, /research/construction, /research/r-verify)
r/             construct.R   the same LP in R (data.table + Rglpk); checked against the Python solve in tests and on every build
               verify.R      the headline statistics recomputed in R (data.table, HAC by hand); checked against Python in tests and over all 91 managers
               compact.py    the committed ~15 MB bundle of every cache a fresh clone needs (compact-pack / compact-unpack)
               fund_universe.py  the ~108 managers, search names and style tags
               serve.py      HTTP server: dashboards, /managers, /f/<slug>/, /analyze?ticker=, /leaderboard, /status
               reference.py  Ken French factor/market data, monthly + annual (real, cached in data/reference/)
               synthetic.py  placeholder dataset: real-market-driven returns, +2%/yr injected alpha, injected defects
               cli.py        python -m trackrecord {reconcile,returns,attribution,validate,report,synth,brk-placeholder}
templates/     empty CSVs + ENTRY_GUIDE.md for hand entry
tests/         71 tests, incl. ground-truth recovery of the injected alpha and an end-to-end CLI run
data/entered/  the real normalized CSVs go here (gitignored)
data/raw/      scans and downloaded PDFs (gitignored)
data/reference/ cached benchmark/factor data (gitignored; re-fetched on demand)
output/        reports (gitignored)
```

## Deploy (Railway)
`railway.json` starts `python -m trackrecord serve`: it listens on `$PORT` at once, builds both placeholder
datasets in the background (needs network for Ken French factors and yfinance), and serves `output/`.
`/` redirects to the Berkshire dashboard (override with `LANDING=synthetic/dashboard.html`); `/status` lists all outputs.
Set `DASHBOARD_PASSWORD` on the service to require HTTP Basic Auth — **mandatory before any real statements
are deployed**. Set `REBUILD=0` to skip rebuilding on restart.

## Research: do the disclosed books carry a signal?
`python -m trackrecord signals13f` turns every filing in the universe into quarterly long-only portfolios — each
manager's **best idea** (largest position), **crowding** quintiles (how many managers hold a name), and **conviction
changes** (new / added / trimmed / sold, from shares) — formed at the end of the month the filings become public,
held with drift, and tested against the Carhart four factors with HAC errors, as spreads and as rank ICs. Served at
`/research/13f-signals`. Result on 90 managers, 2013–2026: no exploitable signal after the 45-day lag; fresh buys
lag the names just sold; the one apparent anomaly (least-crowded names) sits exactly where the price panel's
survivorship bias lives and is reported as unreliable.

## Portfolio construction: from signal to trade list
`python -m trackrecord construct` runs a demonstration mandate: universe = names held by ≥ 5 managers each quarter,
benchmark = the aggregate disclosed book, alpha = 12-1 momentum z-score, and a **linear program** that maximises
expected alpha net of a 10 bps cost under long-only, name-cap, active-band, sector-band, active-share and turnover
constraints, with names leaving the universe forced out. Output: monthly backtest vs the benchmark and an
unconstrained top-decile portfolio, a rebalance log with ex-ante tracking error (shrunk covariance), and the latest
**trade list** (shares, $ and cost on a $100m book). The same LP is solved in R with Rglpk (`r/construct.R`) and the
two solutions are compared on every build. Served at `/research/construction`.

## Due-diligence memo
`/f/<slug>/memo` (and `/t/<ticker>/memo`) assembles a one-page investment-committee memo from the pipeline's own
outputs: what is claimed vs verified, skill vs exposure (all factor models, plain-English reading of the loadings,
replication test), whether it is believable (parametric, bootstrap, zero-skill cohort, years needed at t = 2),
stability (rolling loadings, drift, alpha by half, timing test), risk and concentration, an auto-generated list of
**questions for the manager**, and a fixed-rule recommendation. Every sentence is generated from the numbers.

## Using the site as a research tool
The home page is not only a lookup. Below the featured managers a **Research** section carries one card per
tool — 13F signals, portfolio construction, the R reproduction, and the per-manager memo — each showing the
headline numbers read back from its own CSVs, so a rebuild cannot leave a stale claim on the page. The
manager table underneath is a **screener**: filter the universe by evidence of alpha (the Newey-West
t-statistic on the FF3 intercept: t ≥ 2, ≥ 1, ≥ 0, or negative), by style, and by track length, combine that
with the search box, and take the filtered set away with **Download CSV** (slug, manager, fund, style, months,
excess return, FF3 t and both scores). Every row also links straight to that manager's due-diligence memo.
`/research` indexes all three notes and every CSV they are generated from, each row saying what is in the
file; the files are served read-only from `data/research/` under `/research/data/<note>/<file>.csv`.

## R reproduction of the headline numbers
`Rscript r/verify.R output/funds/<slug>/phase4` recomputes, in data.table and base R and from the aligned returns
alone, what Python reported for that manager: CAPM/FF3/Carhart4/FF5 alpha with Newey-West standard errors and
t-statistics, annualized return, volatility, Sharpe, max drawdown and the alpha-maxing score. The HAC sandwich is
written out by hand on the R side — Bartlett weights, lag ⌊0.75·n^(1/3)⌋, no small-sample correction, normal
p-values, matching `statsmodels`' `cov_type="HAC"` — so agreement is a genuine second opinion, not a second call to
the same library. `python -m trackrecord r-verify` runs it over every manager and writes
`data/research/r-verify/`; **91 managers, 4,641 numbers, largest difference 8.9e-13, no disagreements**. Served at
`/research/r-verify`. R is not installed on Railway, so the CSVs are committed and the page renders from them; the
pytest skips where Rscript is absent.

## Hedge funds: 13F clones
Private funds publish no returns. `python -m trackrecord funds-build --contact "Name email"` (SEC requires a
contact in the User-Agent) pulls every 13F-HR for the ~108 managers in `trackrecord/fund_universe.py` from
EDGAR (structured filings from 2013 Q2), merges filer entities (e.g. Appaloosa Management LP → Appaloosa LP),
maps CUSIPs to tickers (SEC company table, then OpenFIGI), pulls monthly adjusted prices, and builds a
**long-only clone**: buy the disclosed top-60 holdings at disclosed weights at the end of the filing month, hold
with drift until the next filing. `funds-score` runs the full report on each and writes `data/funds/leaderboard.csv`.

A clone is a reconstruction, not the fund: no shorts, options, cash, leverage or non-US holdings, entered ~45 days
late, and delisted names drop out (disclosed as priced coverage). Styles tag how much it can mean — concentrated /
activist / long-short track the long book; multi-strategy, quant and macro clones are flagged as not meaningful.
Managers with < 36 months of filings (e.g. Situational Awareness, from 2025) are listed as not scorable.
The served site has `/managers` (by strategy) and `/f/<slug>/` (lazy build); datasets are committed under `data/funds/`.

For each clone, `contributions.csv` holds every holding's contribution over the scored window — gross (Σ w·r, sums to
the clone's return) and **active** (Σ w·(r − r_market), sums to the excess over the US market) — with months held,
first/last month, average weight, share of outperformance and cumulative share in rank order; `timeline.csv` holds
month-by-month weights for the names that matter. The dashboard section "The stocks behind the outperformance"
lists the ten names that added most over the market and the three that cost most, each with its share and a
holding-timeline strip.

## Private records (accounts)
`/account` creates an account (email + password, PBKDF2 hashes, HMAC-signed HttpOnly session cookie); `/me` lists that
user's uploads and takes new ones — a returns file (`date, return`), a values-and-flows file (`date, value, flow`), or
the pipeline's own statement templates — converts them into a dataset, runs the full report, and serves the dashboard
only to the owner at `/me/<slug>/dashboard.html`. Everything lives under `USERDATA_DIR` (default `userdata/`, gitignored;
on Railway a mounted volume so it survives deploys). Set `SESSION_SECRET` in production. There is no email verification or
password reset yet.

**Statement PDFs** (`pdfstatements.py`): text PDFs from a custodian's website are read by label — period, account,
beginning value, additions, subtractions, change in value, ending value — into statements.csv (+ mid-period flows);
a per-file report says what was found and what wasn't. Scans (no text layer) are refused with a reason; OCR is not
automated. This is the only upload route that can reach *verified*.

**Brokerage APIs**: Fidelity and Robinhood have no public customer API; the legitimate path is an aggregator (Plaid /
Akoya), which needs a developer account and reaches back only ~24 months — useful for ongoing verification, not for
reconstructing decades. Not wired up.

## Analyze any listed portfolio
The served site has a ticker box on every page (`/analyze?ticker=FCNTX`): it downloads the monthly adjusted
history via yfinance, builds a one-account placeholder, runs phases 1–4 on the monthly grid (~40 s), and adds
the result to `/leaderboard`. Works for anything listed — mutual funds, ETFs, holding companies. Private funds
(Tepper, Baker, …) publish no returns; their 13Fs are a lossy clone at best and are not supported.

Two 0–100 scores on fixed maps (comparable across portfolios): **alpha-maxing** = 50 + 10 × excess %/yr over the
US market (return only); **wealth-management** = skill evidence 30% + risk-adjusted return 25% + downside
protection 25% + consistency over rolling 5-year windows 20%. Breakdown on every dashboard and in `phase4/scores.csv`.

## Setup
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Run
```
.venv/bin/python -m pytest tests -q
.venv/bin/python -m trackrecord synth --out data/synthetic                              # placeholder data
.venv/bin/python -m trackrecord reconcile --data data/synthetic --out output/synthetic   # Phase 1
.venv/bin/python -m trackrecord returns   --data data/synthetic --out output/synthetic/phase2   # Phase 2
.venv/bin/python -m trackrecord report    --data data/synthetic --out output/synthetic --claimed 0.14   # everything + REPORT.md + dashboard
.venv/bin/python -m trackrecord fetch-brk && .venv/bin/python -m trackrecord brk-placeholder            # Berkshire placeholder (network)
.venv/bin/python -m trackrecord report    --data data/brk --out output/brk --grid M --claimed 0.198      # monthly grid
```
Swap `data/synthetic` for `data/entered` once real statements are in. Nothing else changes.
`report` takes ~10 s (5,000 bootstrap resamples, 10,000 simulated managers); `--n-boot` / `--n-cohort` to adjust.

## Placeholder data (data/synthetic)
12 accounts, 1996–2025, annual statements. Returns are generated from the **real**
developed-market return each year plus an injected **+2.0%/yr alpha** (beta ≈ 1, ~5%/yr
manager noise, demeaned so the realized alpha equals the injected one). Includes: the
principal's own account, 2 closed accounts (one capitulating in Mar-2009), 1
non-discretionary account, an account with lost position pages, and 6 injected data
defects. `manifest.json` records ground truth so later phases can be tested against it.

## Reconciliation checks (reconcile.py)
| check | passes when | needs |
|---|---|---|
| CHAIN | printed beginning == prior statement's ending | printed beginning value, contiguous prior |
| POSITIONS | Σ position market values == ending value | position rows |
| FLOWS | Σ flow rows == printed additions / subtractions | printed period totals |
| PNL | ending − beginning − net flows == printed change in value | printed change in value |
| RETURN | Modified Dietz ≈ printed return (warn only) | printed % return |

Statement status: **verified** (≥1 check possible, none failed), **flagged**
(a check failed), **unverified** (no check possible). Warnings do not block
verification; errors do. Nothing is corrected — flags carry the numbers so a
human re-reads the page.

## Decisions log
- 2026-09-10 · Source data is annual Fidelity statements in a binder (30 yrs, ~10 accounts, all currently open) → the verified core is an **annual** series. Monthly data is a second tier (Fidelity online, ~10 yrs) and a third tier (holdings-based estimate).
- 2026-09-10 · Fees: none charged historically (per client). Historical record is gross = net; a model-net series at the proposed RIA fee schedule will be produced in Phase 2 for marketing use.
- 2026-09-10 · Modified Dietz in Phase 1 is a diagnostic only. End-of-day flow convention: w = (period_end − flow_date) / calendar days. Phase 2 owns the return series and will chain sub-periods properly where monthly data exists.
- 2026-09-10 · `INFERRED_BEGINNING` is info-level: expected when a custodian doesn't print a beginning value; the `unverified` status is what conveys "nothing confirmed this".
- 2026-09-11 · Benchmarks fixed before real data: `DEV_MKT` (Ken French developed markets) primary, `US_MKT` secondary. Model fee assumed 0.50% + 20% over HWM pending confirmation.
- 2026-09-11 · A flagged statement breaks an account's return chain (it is excluded from the composite *and* from that account's TWR/IRR span). Multi-period statistics are only ever computed over contiguous runs; a hole ends the run and the report says so.
- 2026-09-11 · Real reference data (1996–2025): US market 10.4%/yr, developed 8.5%, developed ex-US 6.5%, T-bills 2.3%.
- 2026-09-11 · Phase 3 allocation effect uses a policy benchmark whose US/ex-US split is inferred from returns (12-month LS on DEV = w·US + (1−w)·DXUS); construction error (~±0.35%/yr) is reported as its own line item so allocation + residual + error = portfolio − benchmark exactly.
- 2026-09-11 · Phase 4 uses French's published annual factors (not compounded monthly long-short factors) for annual cells; Developed set primary, US set as robustness. Null bootstrap and zero-skill cohort both bootstrap the record's own residuals rather than assuming normality.
- 2026-09-11 · Factor set follows the composite benchmark (US_MKT → US factors primary, DEV as robustness; DEV_MKT → the reverse).
- 2026-09-11 · Dashboard is generated, not hand-authored: one accent hue for the composite, grey for everything else, status colors reserved for evidence state and always paired with a letter. The hero label reads "unverified" when fewer than half the account-periods are verified.
- 2026-09-12 · Visual system taken from the KKW partnership proposal PDF: Palatino (system font on Mac/iOS/Windows; TeX Gyre Pagella in the PDF) for headings, body and figures; tracked Helvetica small caps for labels; navy cover band (#1b2a40) with off-white (#e8e4da) type and a short gold rule (#c9b48a); paper interior (#fdfcf9) with sand hairlines (#e4dfd2); gold (#8c7a56) section eyebrows ("Section 01"); chips are dot + tracked caps; dark mode is the cover palette. No web fonts. Optional `--firm` / `FIRM_NAME` and `--prepared-for` / `PREPARED_FOR` put a wordmark and "prepared for/by" on the cover — off by default while the data is illustrative.
- 2026-09-12 · Two scores with fixed 0–100 maps (see above) so firms compare directly; both are placed beside the verdict tiles with full breakdowns. Every disclosure has a plain-language "What it means" beside "How it was calculated". Page declares UTF-8 and the server sends charset headers (mojibake fix).
- 2026-09-12 · Headline alpha is **Fama–French 3-factor** (configurable `headline_model`); Jensen's alpha (the CAPM intercept) is shown beside it. The zero-skill cohort is matched on the headline model's loadings, not just market beta. A skill table lists Sharpe, Treynor, M², Jensen, FF3 alpha, IR, appraisal ratio, t and cohort percentile, each tagged with what it can say about skill: only the appraisal ratio (α/σε) speaks to skill vs luck, since t ≈ AR × √years. All ratios on one arithmetic-annualized basis.
- 2026-09-12 · Every dashboard tile and section carries a click-to-expand "how this was calculated" note (native `<details>`), including what "claimed vs verified" means.
- 2026-09-11 · Tail risk: 95% VaR (historical = 5th percentile, parametric = μ − 1.645σ) and expected shortfall are reported per year from monthly returns where available; on annual cells the within-year figures are marked not observable and a trailing-10-year version is given instead. Risk/return scatter includes every account over its eligible cells.
- 2026-09-11 · Every rolling / sub-period result is reported next to its noise floor (SE from residual volatility). On the placeholder, true alpha is constant yet 10-year rolling alpha ranges +4.2% to −2.7% and the 2010 split "changes" at p = 0.03 — that is what noise looks like.

## Open questions (blocking the parser, not the schema)
1. Confirm the binder pages carry Fidelity's name — not a self-generated report.
2. Confirm "no fees ever charged" to any account.
3. Were there ever accounts that closed? (survivorship)
4. Two sample scans (oldest + newest) to build the parser against.
