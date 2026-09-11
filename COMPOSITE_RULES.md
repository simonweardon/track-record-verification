# Composite construction rules

This document is the human-readable form of `trackrecord/composite.py`. The
code is the executable form of this document. They must agree.

Written with GIPS 2020 in mind so a third-party verifier can later check the
same rules. Where annual source data makes a GIPS requirement impossible, the
deviation is stated here rather than papered over.

## 1. Firm and composite definition
- One composite: **all discretionary accounts managed under the global-equity
  strategy**, regardless of size, fee status, or owner.
- Non-discretionary accounts (client-directed restrictions material enough that
  the manager did not control the strategy) are excluded and listed in
  `accounts.csv` with `discretionary = N` and a reason in `notes`.
- The principal's own account **is included** (managed identically, no fee) and
  is disclosed; a clients-only series and a principal-only series are reported
  alongside so the reader can see whether client money got the same result.
- No minimum account size. If one is ever adopted it applies from adoption
  forward and is recorded here with the date.

## 2. Inclusion and exclusion timing
- An account enters the composite at the **first full measurement period**
  after its first external flow (GIPS 2020 ¶ 3.A.5 spirit). With annual data
  that is the first full calendar year; with monthly data it will be the first
  full month.
- A closed or terminated account stays in the composite **through its last
  full measurement period** and is never removed retroactively (GIPS ¶ 3.A.7).
  This is the survivorship rule. `summary.md` reports the composite both with
  and without closed accounts so the size of the effect is visible.
- Partial first and last periods are measured (from first flow / to last flow)
  and shown per account, but are not composite-eligible.
- An account with a gap in statements is out of the composite for the gap
  cells only; it re-enters at the next full period **only if that period's
  beginning value is printed on the statement** (a chain across a gap is not
  evidence).

## 3. Data quality gates
- A period whose statement **failed reconciliation** (Phase 1 status
  `flagged`) is excluded until the page is re-read and the flag cleared. The
  count of excluded flagged periods is reported per cell. `--include-flagged`
  exists for sensitivity analysis only, never for a published number.
- `unverified` periods (a value exists, nothing on the page confirms it) are
  **included** but counted and disclosed per cell. The final report must state
  the share of the composite resting on unverified periods.
- Nothing is interpolated. Ever.

## 4. Return calculation
- Period (statement-level) return: Modified Dietz, external flows day-weighted,
  end-of-day convention: w = (period_end − flow_date) / calendar days.
- Composite cell return: **aggregate method** — all member accounts pooled as one
  portfolio (Σ ending − Σ beginning − Σ flows) / (Σ beginning + Σ weighted flows).
  Asset-weighted-by-beginning-value and equal-weighted returns are computed as
  cross-checks and written to `composite_returns.csv`.
- Multi-period returns are geometrically linked. Annualized = (1+cum)^(1/years) − 1
  with years = actual days / 365.25.
- Internal dispersion: asset-weighted standard deviation of member returns,
  plus high/low, reported only for cells with ≥ 6 members (GIPS ¶ 4.A.1.i).

## 5. Gross vs net
- Historically **no fees were charged** on any account (per client; to be
  confirmed). Gross = net for the historical record and the composite is
  labeled accordingly.
- A **model-net** series applies the proposed RIA schedule to the composite
  gross series on a NAV index (management fee pro-rated per period; incentive
  fee on gains above a high-water mark). The schedule is an *assumption* until
  the RIA's fee terms are final and must be labeled as such wherever shown.
  GIPS permits model fees with disclosure (¶ 2.A.31 spirit).
- Transaction costs and fund-level expenses are embedded in statement values
  and therefore already netted.

## 6. Benchmarks
- Each account has a benchmark in `accounts.csv`; the composite benchmark is
  the beginning-value-weighted average of member benchmarks per cell (so it
  is well defined even if mandates differ slightly).
- Available series (Ken French library, monthly, total return = Mkt-RF + RF):
  `DEV_MKT` developed markets incl. US · `US_MKT` US total market ·
  `DXUS_MKT` developed ex-US · blends as `60:US_MKT,40:DXUS_MKT`, monthly
  rebalanced.
- Default primary: `DEV_MKT` (closest free proxy to MSCI World for a US +
  international long-only equity mandate). `US_MKT` is always shown as a
  secondary comparison because every reader will ask.
- Benchmark choice should be fixed **before** results are examined and
  recorded here with the date. Changing it afterward is the definition of
  cherry-picking.

## 7. Money-weighted returns
- XIRR per account on dated external flows, initial value and terminal value,
  only for accounts with a gap-free history (an IRR across a gap silently
  omits unknown flows). A pooled IRR across all gap-free discretionary
  accounts is reported. TWR remains the manager-performance measure.

## 8. Known deviations from GIPS
- Valuation frequency: annual statements → annual valuation. GIPS requires at
  least monthly valuation and valuation at large external flows. Until the
  Fidelity monthly PDFs are ingested, this composite is *GIPS-methodology-
  inspired*, not GIPS-compliant, and must not be described as compliant.
- Firm definition, policies and procedures document, and third-party
  verification are out of scope for this pipeline.

## 9. Change log
- 2026-09-10 · Initial rules. Benchmark `DEV_MKT` primary / `US_MKT` secondary
  chosen before any real data was examined. Model fee assumed 0.50% + 20%
  over HWM, no hurdle — pending confirmation.
