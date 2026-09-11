"""Build a Dataset from plain dicts for micro-tests."""
import pandas as pd
from trackrecord.load import Dataset
from trackrecord.schema import STATEMENT_COLUMNS, FLOW_COLUMNS, POSITION_COLUMNS


def _frame(rows, columns):
    df = pd.DataFrame(rows, columns=list(columns))
    for c, kind in columns.items():
        if kind == "date":
            df[c] = pd.to_datetime(df[c])
        elif kind == "float":
            df[c] = pd.to_numeric(df[c])
        else:
            df[c] = df[c].astype("string")
    return df


def ds(statements, flows=(), positions=()):
    return Dataset(_frame(statements, STATEMENT_COLUMNS),
                   _frame(list(flows), FLOW_COLUMNS),
                   _frame(list(positions), POSITION_COLUMNS), [])


def stmt(sid, year, end, begin=None, acct="F_1", cust="Fidelity", **kw):
    return dict(statement_id=sid, account_id=acct, custodian=cust,
                period_start=f"{year}-01-01", period_end=f"{year}-12-31",
                ending_value=end, beginning_value=begin, **kw)


def codes(res, sid=None):
    f = res.flags
    if sid is not None:
        f = f[f.statement_id == sid]
    return set(f.code)
