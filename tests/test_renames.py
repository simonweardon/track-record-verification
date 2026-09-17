"""Recovering a renamed company must never quietly price one company with another's returns.

The mechanism accepts a match only on evidence in the data, so these tests check the evidence
is really there for every accepted match, and that matches which merely look plausible — the
ones a person would be tempted to type in by hand — are rejected."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trackrecord import renames as R

ROOT = Path(__file__).resolve().parents[1]
derived = pytest.mark.skipif(not R.OUT.exists(), reason="renames not derived")


@derived
def test_every_accepted_match_meets_the_stated_evidence():
    d = pd.read_csv(R.OUT)
    assert len(d) > 0
    assert (d.path_gap <= R.PATH_TOL).all()
    assert (d.quarters >= R.MIN_QUARTERS).all()
    # each match must beat its runner-up by a wide margin, so it is the only company that fits
    fp = d[d.how == "price path"]
    assert (fp.runner_up_gap > 3 * fp.path_gap).all()
    assert (fp.runner_up_gap > 0.05).all()
    assert d.cusip.is_unique and d.ticker.notna().all()


@derived
def test_accepted_tickers_are_priced_over_the_line_they_replace():
    from trackrecord.compact import holdings_frame
    from trackrecord.signals13f import load_prices
    d = pd.read_csv(R.OUT, dtype=str)
    prices = load_prices()
    h = holdings_frame()
    h = h[(h.putcall == "") & (h.value > 0)]
    span = h.groupby("cusip").period.agg(["min", "max"])
    for _, r in d.iterrows():
        assert r.ticker in prices.columns, r.ticker
        px = prices[r.ticker].dropna()
        s = span.loc[r.cusip]
        assert px.index.min().strftime("%Y-%m") <= s["min"][:7], (r.cusip, r.ticker)
        assert px.index.max().strftime("%Y-%m") >= s["max"][:7], (r.cusip, r.ticker)


@derived
def test_a_recovered_line_really_does_move_with_its_replacement():
    """Re-derive the agreement for each accepted match rather than trusting the stored number."""
    from trackrecord.compact import holdings_frame
    from trackrecord.signals13f import load_prices
    d = pd.read_csv(R.OUT, dtype=str)
    prices = load_prices()
    h = holdings_frame()
    h = h[(h.putcall == "") & (h.value > 0) & (h.shares > 0)]
    for _, r in d.iterrows():
        gap, n = R.path_agreement(R.implied_price(h, r.cusip), prices[r.ticker])
        assert n >= R.MIN_QUARTERS and gap <= R.PATH_TOL, (r.cusip, r.ticker, gap, n)


@derived
def test_a_plausible_looking_wrong_match_is_rejected():
    """Washington Post became Graham Holdings (GHC); the reference tables offer Graham Corp (GHM),
    an unrelated company whose name normalises identically.  Nothing may accept it."""
    d = pd.read_csv(R.OUT, dtype=str)
    assert "GHM" not in set(d.ticker)
    rej = R.OUT.parent / "renames_rejected.csv"
    if rej.exists() and rej.stat().st_size > 0:
        rd = pd.read_csv(rej, dtype=str)
        bad = rd[rd.ticker == "GHM"]
        assert len(bad) == 0 or (bad.accepted == "False").all()


@derived
def test_the_recovered_lines_reach_the_cusip_map():
    from trackrecord.compact import EDGAR
    from trackrecord.hedge13f import map_cusips
    d = pd.read_csv(R.OUT, dtype=str)
    holdings = pd.DataFrame({"cusip": list(d.cusip), "name": list(d.was)})
    m = map_cusips(holdings, log=lambda *a: None, figi=False)
    for _, r in d.iterrows():
        assert m.get(r.cusip) == r.ticker, r.cusip


def test_the_price_path_test_separates_a_match_from_a_mismatch():
    idx = pd.date_range("2015-03-31", periods=16, freq="QE")
    rng = np.random.default_rng(0)
    path = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.01, 0.08, len(idx)))), index=idx)
    same = path * 0.4                      # the same listing, quoted before years of dividends
    other = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.01, 0.08, len(idx)))), index=idx)
    assert R.path_agreement(path, same)[0] < 1e-9
    assert R.path_agreement(path, other)[0] > 0.05
    assert R.path_agreement(path.head(2), same)[0] == float("inf")
