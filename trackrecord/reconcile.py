"""Per-statement reconciliation.

For each statement, confirm the printed ending value against whatever
INDEPENDENT evidence exists on the page or on the prior statement:

  CHAIN      printed beginning_value == prior statement's ending_value
  POSITIONS  sum(positions.market_value) == ending_value
  FLOWS      sum of flow rows in period == stated_deposits / stated_withdrawals
  PNL        ending - beginning - net_flows == stated_pnl
  RETURN     our Modified Dietz return ~= stated_return_pct   (weak; methods may differ)

Each statement ends up as one of
  verified    at least one check was possible and none failed
  flagged     at least one error-severity check failed
  unverified  nothing independent to test against (we have a number, nothing confirms it)

Nothing is smoothed or corrected.  Failures are reported with the numbers.

Modified Dietz convention (diagnostic only; Phase 2 owns the return series):
  CD  = calendar days in period = (period_end - period_start).days + 1
  w_i = (period_end - flow_date_i).days / CD        (end-of-day flows)
  r   = (End - Begin - sum F_i) / (Begin + sum w_i F_i)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .load import Dataset

ERROR, WARN, INFO = "error", "warn", "info"


@dataclass(frozen=True)
class Tolerance:
    abs_dollars: float = 1.00       # a mismatch must exceed BOTH abs and rel to count
    rel: float = 1e-4               # 1 bp of the value being checked
    return_pp: float = 1.0          # percentage points, stated vs Modified Dietz
    min_return: float = -0.70       # below this an annual return is treated as a data error
    max_return: float = 1.50        # above this, likewise

    def matches(self, a: float, b: float) -> bool:
        if pd.isna(a) or pd.isna(b):
            return False
        return abs(a - b) <= max(self.abs_dollars, self.rel * max(abs(a), abs(b)))


@dataclass
class Flag:
    statement_id: str | None
    account_id: str
    code: str
    severity: str
    detail: str


@dataclass
class ReconcileResult:
    statements: pd.DataFrame   # input statements + computed columns + status
    flags: pd.DataFrame        # one row per flag
    problems: list[str]        # structural / row-level load problems passed through

    def summary(self) -> dict:
        s = self.statements
        return {
            "accounts": int(s["account_id"].nunique()),
            "statements": int(len(s)),
            "verified": int((s["status"] == "verified").sum()),
            "flagged": int((s["status"] == "flagged").sum()),
            "unverified": int((s["status"] == "unverified").sum()),
            "errors": int((self.flags["severity"] == ERROR).sum()) if len(self.flags) else 0,
            "warnings": int((self.flags["severity"] == WARN).sum()) if len(self.flags) else 0,
            "load_problems": len(self.problems),
        }


def _dietz(begin: float, end: float, flows: pd.DataFrame, p_start, p_end) -> float:
    if pd.isna(begin):
        return np.nan
    cd = (p_end - p_start).days + 1
    if len(flows):
        w = (p_end - flows["date"]).dt.days / cd
        weighted = float((w * flows["amount"]).sum())
        net = float(flows["amount"].sum())
    else:
        weighted = net = 0.0
    denom = begin + weighted
    if denom <= 0:
        return np.nan
    return (end - begin - net) / denom


def reconcile(ds: Dataset, tol: Tolerance = Tolerance()) -> ReconcileResult:
    st = ds.statements.copy()
    fl = ds.flows.dropna(subset=["account_id", "date", "amount"]).copy()
    po = ds.positions.dropna(subset=["statement_id", "market_value"]).copy()
    flags: list[Flag] = []

    # computed columns
    for c in ["beginning_used", "net_flows", "deposits", "withdrawals",
              "positions_sum", "computed_pnl", "dietz_return"]:
        st[c] = np.nan
    st["beginning_source"] = ""
    st["n_flows"] = 0
    st["n_positions"] = 0
    st["checks_possible"] = 0
    st["checks_passed"] = 0
    st["checks_failed"] = 0

    pos_by_stmt = {k: g for k, g in po.groupby("statement_id")}
    flows_used = pd.Series(False, index=fl.index)

    for acct, grp in st.groupby("account_id", sort=True):
        grp = grp.sort_values("period_end")
        prev = None
        for i, s in grp.iterrows():
            sid = s.statement_id
            possible = passed = failed = 0

            def flag(code, sev, detail, _sid=sid, _acct=acct):
                flags.append(Flag(_sid, _acct, code, sev, detail))

            # required fields missing -> cannot do anything with this row
            if pd.isna(s.period_start) or pd.isna(s.period_end) or pd.isna(s.ending_value):
                flag("UNUSABLE_ROW", ERROR, "missing period dates or ending_value")
                st.at[i, "checks_failed"] = 1
                continue

            if s.ending_value < 0:
                flag("NEGATIVE_VALUE", ERROR, f"ending_value {s.ending_value:,.2f} < 0")
                failed += 1
            elif s.ending_value == 0:
                flag("ZERO_VALUE", INFO, "ending_value is 0 (account closed or emptied?)")

            # ---- period continuity ------------------------------------------
            contiguous = False
            if prev is not None:
                if s.period_end == prev.period_end:
                    flag("DUPLICATE_PERIOD", ERROR,
                         f"same period_end as {prev.statement_id}")
                    failed += 1
                elif s.period_start <= prev.period_end:
                    flag("PERIOD_OVERLAP", ERROR,
                         f"starts {s.period_start.date()} but {prev.statement_id} "
                         f"ends {prev.period_end.date()}")
                    failed += 1
                else:
                    gap_days = (s.period_start - prev.period_end).days - 1
                    if gap_days == 0:
                        contiguous = True
                    else:
                        flag("PERIOD_GAP", WARN,
                             f"{gap_days} day(s) missing between {prev.period_end.date()} "
                             f"and {s.period_start.date()} (prior: {prev.statement_id})")

            # ---- beginning value ---------------------------------------------
            begin = np.nan
            if not pd.isna(s.beginning_value):
                begin = float(s.beginning_value)
                st.at[i, "beginning_source"] = "printed"
                if contiguous:
                    possible += 1
                    if tol.matches(begin, prev.ending_value):
                        passed += 1
                    else:
                        failed += 1
                        flag("CHAIN_BREAK", ERROR,
                             f"printed beginning {begin:,.2f} != prior ending "
                             f"{prev.ending_value:,.2f} ({prev.statement_id}); "
                             f"diff {begin - prev.ending_value:,.2f}")
                elif prev is not None:
                    flag("CHAIN_SKIPPED", INFO, "gap before this period; chain check not possible")
            else:
                if contiguous:
                    begin = float(prev.ending_value)
                    st.at[i, "beginning_source"] = "prior_ending"
                    flag("INFERRED_BEGINNING", INFO,
                         f"no printed beginning value; using prior ending {begin:,.2f}")
                elif prev is None:
                    st.at[i, "beginning_source"] = "none"
                    flag("INCEPTION", INFO,
                         "first statement for account, no beginning value; "
                         "return series starts at this period_end")
                else:
                    st.at[i, "beginning_source"] = "none"
                    flag("NO_BEGINNING_VALUE", WARN,
                         "gap before this period and no printed beginning; "
                         "return for this period is not computable")
            st.at[i, "beginning_used"] = begin

            # ---- flows in period ----------------------------------------------
            m = (fl["account_id"] == acct) & (fl["date"] >= s.period_start) & (fl["date"] <= s.period_end)
            pf = fl[m]
            flows_used[m] = True
            dep = float(pf.loc[pf["amount"] > 0, "amount"].sum())
            wdr = float(-pf.loc[pf["amount"] < 0, "amount"].sum())
            net = dep - wdr
            st.at[i, "n_flows"] = int(len(pf))
            st.at[i, "deposits"] = dep
            st.at[i, "withdrawals"] = wdr
            st.at[i, "net_flows"] = net

            if not pd.isna(s.stated_deposits):
                possible += 1
                if tol.matches(dep, s.stated_deposits):
                    passed += 1
                else:
                    failed += 1
                    flag("FLOWS_MISMATCH", ERROR,
                         f"flow rows sum to deposits {dep:,.2f}; statement prints "
                         f"{s.stated_deposits:,.2f}")
            if not pd.isna(s.stated_withdrawals):
                possible += 1
                if tol.matches(wdr, s.stated_withdrawals):
                    passed += 1
                else:
                    failed += 1
                    flag("FLOWS_MISMATCH", ERROR,
                         f"flow rows sum to withdrawals {wdr:,.2f}; statement prints "
                         f"{s.stated_withdrawals:,.2f}")

            # ---- positions --------------------------------------------------
            ps = pos_by_stmt.get(sid)
            if ps is not None and len(ps):
                psum = float(ps["market_value"].sum())
                st.at[i, "positions_sum"] = psum
                st.at[i, "n_positions"] = int(len(ps))
                possible += 1
                if tol.matches(psum, s.ending_value):
                    passed += 1
                else:
                    failed += 1
                    flag("POSITIONS_SUM_MISMATCH", ERROR,
                         f"{len(ps)} positions sum to {psum:,.2f}; ending_value "
                         f"{s.ending_value:,.2f}; diff {psum - s.ending_value:,.2f}")
                bad_dates = ps["as_of_date"].dropna().unique()
                bad_dates = [d for d in bad_dates if d != s.period_end]
                if bad_dates:
                    flag("POSITIONS_DATE_MISMATCH", WARN,
                         f"positions dated {[str(pd.Timestamp(d).date()) for d in bad_dates]} "
                         f"but period_end is {s.period_end.date()}")
            else:
                flag("NO_POSITIONS", INFO, "no position rows; positions-sum check not possible")

            # ---- P&L and return ------------------------------------------------
            if not pd.isna(begin):
                pnl = float(s.ending_value) - begin - net
                st.at[i, "computed_pnl"] = pnl
                if not pd.isna(s.stated_pnl):
                    possible += 1
                    if tol.matches(pnl, s.stated_pnl):
                        passed += 1
                    else:
                        failed += 1
                        flag("PNL_MISMATCH", ERROR,
                             f"computed P&L {pnl:,.2f} (end {s.ending_value:,.2f} - begin "
                             f"{begin:,.2f} - net flows {net:,.2f}); statement prints "
                             f"{s.stated_pnl:,.2f}")
                r = _dietz(begin, float(s.ending_value), pf, s.period_start, s.period_end)
                st.at[i, "dietz_return"] = r
                if not pd.isna(r):
                    if r < tol.min_return or r > tol.max_return:
                        failed += 1
                        flag("IMPLAUSIBLE_RETURN", ERROR,
                             f"Modified Dietz return {r:+.1%} outside "
                             f"[{tol.min_return:+.0%}, {tol.max_return:+.0%}]; "
                             f"likely a mis-entered value or missing flow")
                    if not pd.isna(s.stated_return_pct):
                        possible += 1
                        if abs(r * 100 - s.stated_return_pct) <= tol.return_pp:
                            passed += 1
                        else:
                            # warn, not error: statement may use a different method
                            flag("RETURN_MISMATCH", WARN,
                                 f"Modified Dietz {r*100:+.2f}% vs stated "
                                 f"{s.stated_return_pct:+.2f}% "
                                 f"(diff {r*100 - s.stated_return_pct:+.2f} pp)")
            elif not pd.isna(s.stated_pnl) or not pd.isna(s.stated_return_pct):
                flag("CHECK_SKIPPED", INFO, "stated P&L/return present but no beginning value to test against")

            st.at[i, "checks_possible"] = possible
            st.at[i, "checks_passed"] = passed
            st.at[i, "checks_failed"] = failed
            prev = s

    # ---- flows that fall in no statement period -----------------------------
    for j, f in fl[~flows_used].iterrows():
        flags.append(Flag(None, f.account_id, "ORPHAN_FLOW", ERROR,
                          f"{f.flow_type} {f.amount:,.2f} on {f.date.date()} falls in no "
                          f"statement period for this account (gap, or wrong account_id)"))

    # ---- status ------------------------------------------------------------
    fdf = pd.DataFrame([asdict(f) for f in flags],
                       columns=["statement_id", "account_id", "code", "severity", "detail"])
    err_ids = set(fdf.loc[fdf["severity"] == ERROR, "statement_id"].dropna())

    def status(row):
        if row.statement_id in err_ids or row.checks_failed > 0:
            return "flagged"
        if row.checks_passed > 0:
            return "verified"
        return "unverified"

    st["status"] = st.apply(status, axis=1)
    codes = fdf.groupby("statement_id")["code"].apply(lambda c: ";".join(sorted(set(c))))
    st["flags"] = st["statement_id"].map(codes).fillna("")
    return ReconcileResult(st, fdf, list(ds.problems))
