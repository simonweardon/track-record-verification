# Hand-entry guide

Three CSV files. The parser will fill them from scans; until then, or for pages
OCR can't read, a person fills them the same way. The pipeline can't tell the
difference and doesn't need to.

**The rule that matters most:** enter what is printed. Never compute a number
to fill a blank. Blanks are fine — the `stated_*` columns exist so the pipeline
has something *independent* to check against; a number you derived is not
independent.

## Minimum viable path
`statements.csv` (ending value per account-year) + `flows.csv` (every deposit
and withdrawal with its date) gives a verifiable annual return series.
`positions.csv` can be entered later; it adds a second check on every ending
value and is needed for attribution.

## statements.csv — one row per statement
| column | what to enter |
|---|---|
| statement_id | `{CUST}_{last4}_{YYYY}` e.g. `FIDE_1234_2005`. Must be unique. |
| account_id | `{CUST}_{last4}` e.g. `FIDE_1234` |
| custodian | `Fidelity` |
| period_start / period_end | from the statement header, ISO `YYYY-MM-DD` |
| ending_value | Fidelity "Ending Value" (total account). **Required.** |
| beginning_value | Fidelity "Beginning Value" |
| stated_deposits | Fidelity "Additions" (positive number) |
| stated_withdrawals | Fidelity "Subtractions" (positive number) |
| stated_income | dividends + interest total if printed |
| stated_fees | fees deducted inside the account, if printed (positive) |
| stated_pnl | Fidelity "Change in Investment Value" |
| stated_return_pct | only if the statement prints a % return for the period |
| source_file / source_pages | scan filename and page range the numbers came from |
| notes | anything odd: restated, water-damaged, account number changed, … |

## flows.csv — one row per external flow
Signed from the account's point of view: **+ money in, − money out**.
`flow_type` ∈ deposit · withdrawal · transfer_in · transfer_out.
In-kind securities transfers use the market value on the transfer date.
Dividends, interest, and fees are **not** flows — they are part of the
account's own P&L and stay inside the statement totals.

## positions.csv — one row per holding at period_end
`asset_class` ∈ us_equity · intl_equity · fund · cash · bond · other.
Include the cash / core position line so the rows sum to ending_value.
`weight_pct` only if printed.

## Then
```
python -m trackrecord reconcile --data data/entered --out output
```
and read `output/coverage.md`. Every error must be either fixed by re-reading
the page or left in and disclosed. Never adjusted to make it balance.
