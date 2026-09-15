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
- Disclose AI assistance plainly (README/cover note): built with Claude Code as pair programmer;
  methodology, data-quality rules and verification are Simon's.

## Feature plan (agreed Sept 15, 2026), in priority order
1. **13F signals backtest** — `trackrecord/signals13f.py` (to write): best-ideas (each manager's largest
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
- Python 3.12, `.venv` locally; `pip install -r requirements.txt` in a fresh clone. Tests: `python -m pytest -q`
  (71+ tests; keep them green, add one per new module).
- Design: Palatino serif, navy/gold/sand palette, hairline rules, light/dark via `prefers-color-scheme`.
  Page chrome comes from `serve.page()`; dashboards from `dashboard.CSS`/`JS`. No new top navigation —
  the home page is one page with one action (Analyze).
- Scores are fixed maps (see README) — never change them; it would invalidate comparisons.
- Deploy: push to `main` → Railway (`python -m trackrecord serve`). Site is currently open (no password);
  `DASHBOARD_PASSWORD` must be set before any real statements are deployed.
- Commit messages: short imperative subject; end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
