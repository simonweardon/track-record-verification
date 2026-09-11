"""Normalized schema: three CSV tables that every downstream step consumes.

The tables are produced either by a statement parser (once we have scans)
or by hand entry into the templates in templates/.  Downstream code does
not know or care which.

Conventions
-----------
- Money is in account base currency (USD assumed until told otherwise).
- Dates are ISO (YYYY-MM-DD).
- Flow amounts are signed from the account's point of view: + in, - out.
- "stated_*" columns are numbers PRINTED ON THE STATEMENT.  Leave blank if
  the statement does not print them.  Never derive them.  They exist so the
  pipeline has something independent to reconcile against.
"""
from __future__ import annotations

# ---- statements.csv : one row per (account, statement period) --------------
STATEMENT_COLUMNS: dict[str, str] = {
    "statement_id":       "str",    # unique. convention: {custodian}_{last4}_{YYYY}
    "account_id":         "str",    # convention: {custodian}_{last4}
    "custodian":          "str",    # institution that produced the statement
    "period_start":       "date",
    "period_end":         "date",   # valuation date
    "ending_value":       "float",  # total account value as printed. REQUIRED
    "beginning_value":    "float",  # as printed, if printed
    "stated_deposits":    "float",  # period total as printed, if printed (positive)
    "stated_withdrawals": "float",  # period total as printed, if printed (positive)
    "stated_income":      "float",  # dividends + interest as printed
    "stated_fees":        "float",  # fees deducted inside the account (positive)
    "stated_pnl":         "float",  # "change in investment value" ex-flows, if printed
    "stated_return_pct":  "float",  # if the statement prints a period return
    "source_file":        "str",    # scan filename
    "source_pages":       "str",    # e.g. "1-4"
    "notes":              "str",
}
STATEMENT_REQUIRED = ["statement_id", "account_id", "custodian",
                      "period_start", "period_end", "ending_value"]

# ---- flows.csv : one row per external cash / securities flow ---------------
FLOW_COLUMNS: dict[str, str] = {
    "account_id":          "str",
    "date":                "date",
    "amount":              "float",  # signed: + into account, - out of account
    "flow_type":           "str",    # see FLOW_TYPES
    "description":         "str",
    "source_statement_id": "str",    # which statement this was read from
}
FLOW_REQUIRED = ["account_id", "date", "amount", "flow_type"]
FLOW_TYPES = {
    "deposit",        # cash in from outside
    "withdrawal",     # cash out to outside
    "transfer_in",    # securities in-kind, valued at market on the date
    "transfer_out",   # securities in-kind, valued at market on the date
}

# ---- positions.csv : one row per holding at a valuation date ---------------
POSITION_COLUMNS: dict[str, str] = {
    "statement_id": "str",
    "account_id":   "str",
    "as_of_date":   "date",
    "identifier":   "str",    # ticker / CUSIP / whatever the statement prints
    "description":  "str",
    "asset_class":  "str",    # see ASSET_CLASSES
    "quantity":     "float",
    "price":        "float",
    "market_value": "float",  # REQUIRED
    "weight_pct":   "float",  # as printed, optional
}
POSITION_REQUIRED = ["statement_id", "account_id", "as_of_date",
                     "identifier", "market_value"]
ASSET_CLASSES = {"us_equity", "intl_equity", "fund", "cash", "bond", "other"}

# ---- accounts.csv : one row per account (composite membership judgments) ---
# Not a statement fact; a documented decision.  Phase 2 reads it.
ACCOUNT_COLUMNS: dict[str, str] = {
    "account_id":    "str",
    "label":         "str",    # anonymised display name
    "owner_type":    "str",    # principal | client
    "discretionary": "str",    # Y | N   (N => excluded from composites, listed in rules doc)
    "strategy":      "str",    # composite key, e.g. global_equity
    "benchmark":     "str",    # reference series id or blend, e.g. DEV_MKT or 70:US_MKT,30:DXUS_MKT
    "notes":         "str",
}

TABLES = {
    "statements": (STATEMENT_COLUMNS, STATEMENT_REQUIRED),
    "flows":      (FLOW_COLUMNS, FLOW_REQUIRED),
    "positions":  (POSITION_COLUMNS, POSITION_REQUIRED),
}
