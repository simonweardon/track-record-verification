# Track Record Verification — working notes for Claude

Read this first in any session (laptop or cloud). README.md describes the pipeline; this file holds
the context that is not in the code.

## Who / why
- Owner: Simon Weardon (simonweardon@gmail.com). Not a finance professional — define terms plainly,
  keep questions to a minimum, make routine calls yourself.
- **Current goal (Sept 2026): showcase for a job application.** Florida State Board of Administration,
  *Portfolio Manager II – Global Equity*, Job ID 1675, **closes Oct 2, 2026**; target submit Sept 28–30.
  Simon was referred by Ted Nation and will preview the site with a quant PM before submitting.
- The JD weights: external quant-manager due diligence (45%), running internal active/passive quant
  portfolios — rebalance, trade lists, risk models (50%), fund-of-funds construction, **R strongly
  preferred (data.table, Rglpk, xgboost)**, SQL, Barra/FactSet. Every feature should map to a JD line.
- Frame the project as an **external-manager due-diligence engine**. Keep negative results and caveats
  visible (only 5 of 89 clones have FF3 alpha t>2; clones ≠ funds; 45-day lag; delisted names drop out).
- **Presentation (Simon's decision, Sept 17, 2026):** the site and README present the work as Simon's alone — no
  mention of Claude or how it was built, no language or library names (Python, R, data.table, Rglpk, xgboost, HiGHS),
  no "13F clone" jargon (say "disclosed holdings"), complete sentences everywhere, card descriptions of one or two
  sentences. `tests/test_plain_language.py` enforces this on every rendered page and the README. Plain vocabulary:
  xgboost → "learned model", linear composite → "simple average", R verification → "Independent Verification",
  13F signals → "Holdings Research", alpha lab → "Signal Research".

## Status (Sept 17, 2026) — pick up here
**Due Diligence on This Work** (`trackrecord/limits.py`, `/research/limits`, card on the home page under "Before you trust
any of it"): the eight objections a reviewer raises about the site, each answered with a computed number, all rebuilt by
`python -m trackrecord limits` (~6 min) into `data/research/limits/`. Findings and the surfaces they changed are in the
README section of the same name. Two things it changed elsewhere: (1) `construct.backtest` now records `transfer_coef`
and `realized_ic` per rebalance (the fundamental-law reconciliation: IC 0.026 x TC 0.60 x sqrt(1,378) ⇒ implied IR 0.58
vs realised 0.78, inside the 0.32 SE); (2) the `/external` screener carries a **how closely the record tracks the fund**
chip and filter per manager, driven by `limits/reconstruction.csv` — 17 of 90 managers hold > 15% of the disclosed book
in options, which the pipeline drops, so Elliott and Scion are labelled "stock positions only" next to their rank.
Also fixed a site-wide mobile layout bug while checking the new page: `.grid2`/`.tiles` used a fixed `minmax(420px…)`
and grid items kept `min-width:auto`, so a wide table held the whole page open. Now `minmax(min(420px,100%),1fr)` plus
`min-width:0` on grid items — **every page fits at 390/768/1280 px**, which several did not before. 143 tests (4 of them need R, which is installed locally).
Rebuild order after data changes: alpha-lab → risk-model; construct, fund-of-funds and limits are independent, but
limits reads construction's trade list and the r-verify / decay manifests, so run it last.

## Earlier status (Sept 16, 2026, night)
Passive area removed entirely (Simon's call after the pandering audit: the 5% JD line was not worth a tab that read as
built-for-the-JD). Gone: toolbar entry, home card, `/research/index-tracker`, `/live/passive/*`, `trackrecord/tracker.py`,
`trackrecord/livebook.py`, `data/research/index-tracker/`, the `index-tracker` CLI command and the daily live-book scheduler.
Toolbar is now Home · Active · Manager Analysis · Simulation · My records. Rebuild order after data changes: alpha-lab →
risk-model; construct and fund-of-funds are independent.
**"How & why" under every number** (`trackrecord/explain.py`): one registry of notes keyed by tile label (+ page regex where
a label is reused); `serve._html` annotates every response by path, `dashboard.build_dashboard` annotates at write time,
home/area card stats call `note_html` directly. `tests/test_explain.py` fails if any tile or card stat lacks a note — add
an entry to `NOTES` whenever a new tile is added. 121 tests.

## Earlier status (Sept 16, 2026, evening)
Manager decay model built (`trackrecord/decay.py`, `r/decay.R`, `/research/decay`, card under Manager Analysis): the
one place xgboost and data.table do work that needed them — a classification problem on the full 576k-row 13F panel.
Honest null (AUC ≈ 0.5 walk-forward) plus the leaky-CV-vs-walk-forward gap (0.57 vs 0.49) as the teaching point.
110 tests. Simon asked for a site-wide audit of "pandering" (features that exist for the JD rather than on merit);
findings were delivered in chat on Sept 16 — decisions on what to change are Simon's, pending.

## Earlier status (Sept 16, 2026)
Site reorganised by job area (Simon's brief: tools that do the job's work; redundant tabs removed). Toolbar:
Home · Passive · Active · External managers · My records (identical on every page; areas with no built tool are hidden).
New tools, all built, tested (102 tests) and pushed: alpha lab (xgboost native API), risk model, index tracker + DPSW (since removed),
fund-of-funds. Data: SEC XBRL fundamentals (1,207 names) and split-only closes in the bundle; shares cleaned
(unit errors, split back-adjustment, 10-K/10-Q only). Rebuild order after data changes: alpha-lab → risk-model →
index-tracker; construct and fund-of-funds are independent. Remaining: interview-prep quiz; consider a factor /
index-provider / currency note under External managers if time allows.

## Earlier status (Sept 15, 2026, late evening)
DONE and deployed: (1) 13F signals research `/research/13f-signals`; (2) LP constructor + trade list
`/research/construction` with the Rglpk twin `r/construct.R`; (3) due-diligence memo `/f/<slug>/memo`
(+ `/t/<ticker>/memo`), linked from every manager dashboard; server pre-builds all managers after start.
Verified this session on the deployed commit (f62f60e): `/research/construction`, `/f/appaloosa/memo`,
`/research/13f-signals` and `/leaderboard` all 200. The Railway egress is blocked from cloud sessions, so
the check was run against the same commit served locally; the Railway deploy itself is SUCCESS.
DONE (4): **R reproduction**. `r/verify.R` (data.table + base R, HAC sandwich written out by hand) recomputes
CAPM/FF3/Carhart4/FF5 alpha, SE, t, p, CI, R², loadings, annualized return/vol/Sharpe/max-DD and the
alpha-maxing score from `output/<fund>/phase4/aligned_data.csv` and compares with regressions/metrics/scores.csv.
`trackrecord/rverify.py` drives it over every manager (`python -m trackrecord r-verify`) → `data/research/r-verify/`
(committed; Railway has no R), page `/research/r-verify` linked from the landing research line.
**91 managers, 4,641 numbers, max |R − Python| 8.9e-13, zero disagreements.** 90 tests pass (3 new; they skip
without Rscript). Note for cloud sessions: R *can* be installed here — `apt-get install -y --no-install-recommends
r-base-core r-cran-data.table r-cran-rglpk` (~2 min) — which also un-skips the Rglpk construct twin test.
DONE (5a): **home page as a research tool**. Research section with a card per tool (numbers read from the
committed CSVs, so nothing goes stale); the manager table is now a screener — filter by FF3 t (≥ 2 / ≥ 1 / ≥ 0 /
negative), style, track length and scorable-only, combined with the search box, plus **Download CSV** of the
filtered rows and a Memo link on every row; new `/research` index listing all three notes and the 12 CSVs behind
them, served read-only at `/research/data/<note>/<file>.csv`. 95 tests pass (5 new in test_research_surfaces.py).
Checked in Chromium at 390/768/1280 px: no horizontal overflow (the table now scrolls inside `.tw`, not the page).
Live filter counts sanity-check the honest finding: 5 of 112 rows clear FF3 t ≥ 2.
DONE (5b, laptop): README "What this is" + live link + "How it was built" disclosure (dddb1e6); cover-note drafts given to
Simon outside the repo. NEXT: interview-prep quiz (interactive with Simon), then submit by Sept 28–30.

## Feature plan (agreed Sept 15, 2026), in priority order
1. **13F signals backtest** — `trackrecord/signals13f.py` (DONE): best-ideas (each manager's largest
   position, Cohen–Polk–Silli), crowding (# managers holding), conviction changes (new / added / trimmed /
   sold, from shares). Equal-weight quarterly portfolios formed at end of Feb/May/Aug/Nov (45-day 13F lag),
   long-short spreads, Carhart alpha with HAC t-stats (reuse `validation._fit` / `MODELS`), IC per formation
   date, turnover. Outputs to `data/research/13f-signals/`; page `/research/13f-signals` rendered from the
   CSVs with `dashboard.line_chart` / `diverging_bars`, `dashboard.CSS` + `JS`.
2. **LP portfolio constructor + trade list** — signal scores → weights under constraints (max name weight,
   sector bands vs benchmark, turnover budget) → buy/sell list. Python (scipy/cvxpy) plus an `r/` twin using
   **Rglpk** on the same inputs; optional index-tracking mode.
3. **Due-diligence memo per manager** — `/f/<slug>/memo`: verdict, alpha with CI, factor loadings and
   rolling drift, replication test, risks, auto-generated quarterly-meeting questions. All inputs already
   exist in `output/funds/<slug>/phase4/*.csv` (regressions, rolling, metrics, skill, scores, cohort, bootstrap).
   (R is installed locally via Homebrew with Rglpk + data.table; `r/construct.R` is tested against Python.)
4. **R reproduction** of headline numbers (FF3 alpha, t, scores) with data.table; test asserting Python ≈ R.
5. Research-notes page, pre-build everything a reviewer might click (lazy 30 s builds are bad first
   impressions), cover note.
Then an interview-prep quiz on the project.

## Data you have in a fresh clone
- `data/reference/compact/` (committed, ~15 MB): all cached 13F holdings for the 90 meaningful-style
  managers (2013 Q2 →, `holdings13f.csv.gz`, columns slug/period/filed/accession/cik/cusip/name/value/shares/
  putcall/cls), EDGAR submissions index, `funds.json` (resolved CIKs), CUSIP→ticker map, monthly adjusted
  prices for 3,231 tickers (`prices_monthly.csv.gz`), Ken French factor zips.
  `python -m trackrecord compact-unpack` rebuilds the raw `data/reference/` layout so every existing loader
  works; the server does this on start. `compact.holdings_frame()` reads the long frame directly.
  Multi-strategy / macro / market-maker books (Citadel, Millennium, RenTech, …) are NOT packed — they are
  never scored; they fetch from EDGAR on demand (`SEC_CONTACT="Name email"` env, never hardcode it).
- `data/funds/<slug>/` (committed): each clone's monthly statement-format dataset + `leaderboard.csv`.
- `output/` is gitignored; `python -m trackrecord report --data data/funds/<slug> --out output/funds/<slug>
  --grid M --placeholder` rebuilds one manager in ~30 s. `funds-score` does all (~4 min).
- No real client statements are in the repo, ever (`data/entered`, `data/raw`, `userdata` are ignored).

## Conventions
- Python 3.10 in `.venv` locally (so no backslashes inside f-string expressions); `pip install -r requirements.txt` in a fresh clone. Tests: `python -m pytest -q`
  (71+ tests; keep them green, add one per new module).
- Design: Palatino serif, navy/gold/sand palette, hairline rules, light/dark via `prefers-color-scheme`.
  Page chrome comes from `serve.page()`; dashboards from `dashboard.CSS`/`JS`. No new top navigation —
  the home page is one page with one action (Analyze).
- Scores are fixed maps (see README) — never change them; it would invalidate comparisons.
- Deploy: push to `main` → Railway (`python -m trackrecord serve`). Site is currently open (no password);
  `DASHBOARD_PASSWORD` must be set before any real statements are deployed.
- Commit messages: short imperative subject. **No Co-Authored-By or other AI attribution trailer** (Simon's decision, Sept 17, 2026).
