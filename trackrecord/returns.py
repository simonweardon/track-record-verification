"""Phase 2a — per-account return series.

Inputs are the reconciled statements (Phase 1 output) and the flows.  Every
return carries the evidence status of the statement it came from.

Period returns are Modified Dietz on the *effective* period:
  - an inception statement (printed beginning value 0) starts at its first flow
  - a closing statement (ending value 0) ends at its last flow
so partial first/last periods are measured over the days money was actually
invested, not linearly scaled to a full period.  Those partial periods are
reported per account but are NOT composite-eligible (see COMPOSITE_RULES.md).

Time-weighted return over a span = geometric chain of period returns.
Money-weighted return = XIRR on dated external flows + initial and terminal values.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RETURN_COLUMNS = ["statement_id", "account_id", "period_start", "period_end",
                  "eff_start", "eff_end", "days", "begin", "end", "net_flows",
                  "weighted_flows", "n_flows", "r", "full_period", "evidence", "flags"]


def _weights(dates: pd.Series, eff_start, eff_end) -> pd.Series:
    cd = (eff_end - eff_start).days + 1
    return (eff_end - dates).dt.days / cd


def period_returns(statements: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame:
    """One row per statement.  `r` is NaN when not computable (no beginning value)."""
    fl = flows.dropna(subset=["account_id", "date", "amount"])
    rows = []
    for _, s in statements.iterrows():
        if pd.isna(s.period_start) or pd.isna(s.period_end) or pd.isna(s.ending_value):
            continue
        pf = fl[(fl.account_id == s.account_id) & (fl.date >= s.period_start) & (fl.date <= s.period_end)]
        begin = s.beginning_used
        end = float(s.ending_value)
        eff_start, eff_end = s.period_start, s.period_end
        r = np.nan
        full = False
        if not pd.isna(begin):
            if begin == 0 and len(pf):
                eff_start = pf.date.min()                       # inception: money arrives here
            if end == 0 and len(pf) and (pf.amount < 0).any():
                eff_end = pf.loc[pf.amount < 0, "date"].max()   # closing: money leaves here
            pf = pf[(pf.date >= eff_start) & (pf.date <= eff_end)]
            w = _weights(pf.date, eff_start, eff_end) if len(pf) else pd.Series(dtype=float)
            net = float(pf.amount.sum()) if len(pf) else 0.0
            weighted = float((w * pf.amount).sum()) if len(pf) else 0.0
            denom = begin + weighted
            if denom > 0:
                r = (end - begin - net) / denom
            full = (eff_start == s.period_start and eff_end == s.period_end and begin > 0)
        else:
            net = float(pf.amount.sum()) if len(pf) else 0.0
            weighted = np.nan
        rows.append(dict(statement_id=s.statement_id, account_id=s.account_id,
                         period_start=s.period_start, period_end=s.period_end,
                         eff_start=eff_start, eff_end=eff_end,
                         days=(eff_end - eff_start).days + 1,
                         begin=begin, end=end, net_flows=net, weighted_flows=weighted,
                         n_flows=int(len(pf)), r=r, full_period=bool(full),
                         evidence=s["status"], flags=s["flags"]))
    return pd.DataFrame(rows, columns=RETURN_COLUMNS)


# ---- linking --------------------------------------------------------------

def chain(r: pd.Series) -> float:
    r = r.dropna()
    return float((1 + r).prod() - 1) if len(r) else np.nan


def annualize(cum: float, years: float) -> float:
    if pd.isna(cum) or years <= 0:
        return np.nan
    return (1 + cum) ** (1 / years) - 1


def contiguous_segments(pr: pd.DataFrame, break_on_flagged: bool = True) -> list[pd.DataFrame]:
    """Split one account's period rows into runs that can be chained: no gap, a
    return in every period, and (by default) no flagged period — a value that
    failed reconciliation cannot be linked through."""
    pr = pr.sort_values("period_end")
    segs, cur, prev_end = [], [], None
    for _, x in pr.iterrows():
        usable = not pd.isna(x.r) and not (break_on_flagged and x.evidence == "flagged")
        broken = not usable or (prev_end is not None and (x.eff_start - prev_end).days != 1)
        if broken and cur:
            segs.append(pd.DataFrame(cur))
            cur = []
        if usable:
            cur.append(x)
        prev_end = x.eff_end
    if cur:
        segs.append(pd.DataFrame(cur))
    return segs


def twr_summary(seg: pd.DataFrame) -> dict:
    cum = chain(seg.r)
    years = float(seg.days.sum()) / 365.25
    return dict(start=seg.eff_start.min(), end=seg.eff_end.max(), periods=len(seg),
                years=years, cumulative=cum, annualized=annualize(cum, years))


# ---- money-weighted ----------------------------------------------------------

def xnpv(rate: float, amounts: np.ndarray, t_years: np.ndarray) -> float:
    return float(np.sum(amounts / (1 + rate) ** t_years))


def xirr(dates: pd.Series, amounts: pd.Series, lo=-0.99, hi=10.0) -> float:
    """Annualized IRR of dated cash flows (investor's view: pay-ins negative,
    take-outs and terminal value positive).  Bisection on a bracket, so it
    converges wherever a root exists in (lo, hi); NaN otherwise."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    a = np.asarray(amounts, dtype=float)
    if len(a) < 2 or not ((a > 0).any() and (a < 0).any()):
        return np.nan
    t = ((d - d.min()).dt.days / 365.25).to_numpy()
    f_lo, f_hi = xnpv(lo, a, t), xnpv(hi, a, t)
    if np.sign(f_lo) == np.sign(f_hi):
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = xnpv(mid, a, t)
        if abs(f_mid) < 1e-9:
            break
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return float(mid)


def segment_cashflows(seg: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame | None:
    """Investor-view cash flows over one contiguous segment: initial value paid
    in, every external flow negated, terminal value received.  An IRR is only
    meaningful over a span with no gap and no flagged period, i.e. a segment."""
    if seg is None or len(seg) == 0:
        return None
    start, end = seg.eff_start.min(), seg.eff_end.max()
    acct = seg.account_id.iloc[0]
    fl = flows[(flows.account_id == acct) & (flows.date >= start) & (flows.date <= end)]
    cf = [(start, -float(seg.iloc[0].begin))] if seg.iloc[0].begin > 0 else []
    cf += [(d, -float(a)) for d, a in zip(fl.date, fl.amount)]
    cf.append((end, float(seg.iloc[-1].end)))
    return pd.DataFrame(cf, columns=["date", "amount"])


# ---- model fees --------------------------------------------------------------

@dataclass(frozen=True)
class FeeSchedule:
    """ASSUMED proposed RIA schedule; confirm before any marketing use."""
    mgmt_pct: float = 0.005        # annual, pro-rated by period length
    perf_pct: float = 0.20         # share of gains above high-water mark
    hurdle_pct: float = 0.0        # annual hurdle above HWM before perf fee accrues
    high_water_mark: bool = True

    def describe(self) -> str:
        s = f"{self.mgmt_pct:.2%} management"
        if self.perf_pct:
            s += f" + {self.perf_pct:.0%} of gains"
            s += " above high-water mark" if self.high_water_mark else " (no high-water mark)"
            if self.hurdle_pct:
                s += f" and a {self.hurdle_pct:.2%} hurdle"
        return s


def model_net(gross: pd.Series, years: pd.Series, fee: FeeSchedule = FeeSchedule()) -> pd.Series:
    """Apply a model fee to a gross return series, period by period, on a NAV
    index.  `years` = length of each period in years (1.0 for annual)."""
    nav, hwm = 1.0, 1.0
    out = []
    for r, yr in zip(gross, years):
        if pd.isna(r):
            out.append(np.nan)
            continue
        prev = nav
        nav = nav * (1 + r) * (1 - fee.mgmt_pct * yr)
        if fee.perf_pct:
            threshold = (hwm if fee.high_water_mark else prev) * (1 + fee.hurdle_pct) ** yr
            if nav > threshold:
                nav -= fee.perf_pct * (nav - threshold)
            if fee.high_water_mark:
                hwm = max(hwm, nav)
        out.append(nav / prev - 1)
    return pd.Series(out, index=gross.index)
