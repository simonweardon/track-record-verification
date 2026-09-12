# Track-record reconstruction

Reconstruct and independently verify a ~30-year investment record from
custodian statements, then decompose and statistically test it.

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
| `data/brk` | Berkshire Hathaway Class A monthly closes 1985–2026 (yfinance), one account, no statements | Monthly grid: 36/60-month rolling alpha, real drawdowns, HAC errors; results check against "Buffett's Alpha" (β≈0.7, SMB<0, HML>0, alpha shrinking under FF5) |

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
               serve.py      HTTP server: dashboards, /analyze?ticker=, /leaderboard, /status; optional Basic Auth
               reference.py  Ken French factor/market data, monthly + annual (real, cached in data/reference/)
               synthetic.py  placeholder dataset: real-market-driven returns, +2%/yr injected alpha, injected defects
               cli.py        python -m trackrecord {reconcile,returns,attribution,validate,report,synth,brk-placeholder}
templates/     empty CSVs + ENTRY_GUIDE.md for hand entry
tests/         67 tests, incl. ground-truth recovery of the injected alpha and an end-to-end CLI run
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
