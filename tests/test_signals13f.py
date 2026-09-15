"""13F signal portfolios on a hand-made universe: formation timing (45-day lag), best-idea
and pair weighting, conviction flags from shares, sold names, drift, and the data guards."""
import numpy as np
import pandas as pd

from trackrecord import signals13f as S


def _books():
    """Two managers, two quarters.  M1: A is #1 both quarters, adds to B, sells C, buys D.  M2: B is #1.
    Filler names F0–F3 keep every book above MIN_BOOK."""
    rows = []
    def pos(slug, period, filed, cusip, value, shares):
        rows.append(dict(slug=slug, period=period, filed=filed, accession=f"{slug}-{period}", cik="1", cusip=cusip, name=cusip,
                         value=value, shares=shares, putcall="", cls=""))
    for i in range(4):
        pos("m1", "2020-03-31", "2020-05-10", f"F{i}", 1.0, 1.0); pos("m1", "2020-06-30", "2020-08-10", f"F{i}", 1.0, 1.0)
        pos("m2", "2020-03-31", "2020-05-10", f"F{i}", 1.0, 1.0); pos("m2", "2020-06-30", "2020-08-10", f"F{i}", 1.0, 1.0)
    pos("m1", "2020-03-31", "2020-05-10", "A", 50.0, 100.0); pos("m1", "2020-03-31", "2020-05-10", "B", 20.0, 100.0); pos("m1", "2020-03-31", "2020-05-10", "C", 10.0, 100.0)
    pos("m1", "2020-06-30", "2020-08-10", "A", 50.0, 100.0); pos("m1", "2020-06-30", "2020-08-10", "B", 30.0, 150.0); pos("m1", "2020-06-30", "2020-08-10", "D", 10.0, 100.0)
    pos("m2", "2020-03-31", "2020-05-10", "B", 60.0, 100.0); pos("m2", "2020-03-31", "2020-05-10", "A", 10.0, 100.0)
    pos("m2", "2020-06-30", "2020-09-01", "B", 60.0, 100.0); pos("m2", "2020-06-30", "2020-09-01", "A", 10.0, 100.0)   # filed late: after Aug 31
    return pd.DataFrame(rows)


def _prices():
    idx = pd.date_range("2020-01-31", "2020-12-31", freq="ME")
    p = {t: pd.Series(100.0, index=idx) for t in ["A", "B", "C", "D"] + [f"F{i}" for i in range(4)]}
    p["A"] = pd.Series(100 * (1.10 ** np.arange(len(idx))), index=idx)      # +10%/month
    p["B"] = pd.Series(100 * (1.00 ** np.arange(len(idx))), index=idx)      # flat
    return pd.DataFrame(p)


def _patched(monkeypatch):
    b = _books()
    monkeypatch.setattr(S, "unpack", lambda log=print: False)
    monkeypatch.setattr(S, "holdings_frame", lambda: b)
    monkeypatch.setattr(S, "EDGAR", type("E", (), {"__truediv__": lambda self, x: _Map()})())
    return b


class _Map:
    def read_text(self):
        import json
        return json.dumps({t: t for t in ["A", "B", "C", "D"] + [f"F{i}" for i in range(4)]})


def test_formation_date_is_two_months_after_quarter_end():
    assert S.formation_date("2020-03-31") == pd.Timestamp("2020-05-31")
    assert S.formation_date("2020-12-31") == pd.Timestamp("2021-02-28")


def test_books_flags_and_late_filings(monkeypatch):
    _patched(monkeypatch)
    g = S.load_books(log=lambda *a: None)
    q1 = g[(g.slug == "m1") & (g.period == "2020-03-31")].set_index("cusip")
    q2 = g[(g.slug == "m1") & (g.period == "2020-06-30")].set_index("cusip")
    assert q1.loc["A", "rank"] == 1 and q2.loc["A", "rank"] == 1
    assert abs(q1.loc["A", "w_all"] - 50 / 84) < 1e-9                       # full-book weight
    assert bool(q2.loc["D", "new"]) and bool(q2.loc["B", "add"]) and not bool(q2.loc["A", "add"])
    assert not q2.loc["A", "new"]
    # m2's Q2 filing came after the Aug-31 formation date -> excluded from that formation
    assert set(g[g.formation == pd.Timestamp("2020-08-31")].slug) == {"m1"}
    sold = S.sold_positions(g)
    assert list(sold.ticker) == ["C"] and sold.formation.iloc[0] == pd.Timestamp("2020-08-31")


def test_weights_and_drift(monkeypatch):
    _patched(monkeypatch)
    g = S.load_books(log=lambda *a: None); sold = S.sold_positions(g)
    W = S.formation_weights(g, sold, pd.Timestamp("2020-05-31"))
    assert abs(W["BEST1"]["A"] - 0.5) < 1e-9 and abs(W["BEST1"]["B"] - 0.5) < 1e-9     # one best idea each
    assert abs(W["ALL"].sum() - 1) < 1e-9 and len(W["ALL"]) == 7                       # A B C + 4 fillers
    monkeypatch.setattr(S, "MIN_NAMES", 1)                 # two-name best-ideas book would otherwise be blanked
    R, diag, _ = S.run_portfolios(g, sold, _prices(), log=lambda *a: None)
    # BEST1 at May-31: 50/50 A (+10%/m) and B (flat) -> June +5%, then A's weight drifts up -> July > 5%
    assert abs(R.loc["2020-06-30", "BEST1"] - 0.05) < 1e-9
    assert R.loc["2020-07-31", "BEST1"] > 0.05
    assert abs(R.loc["2020-06-30", "SOLD"] - 0.0) < 1e-12 or pd.isna(R.loc["2020-06-30", "SOLD"])   # no sold names before Aug
    assert abs(R.loc["2020-09-30", "SOLD"] - 0.0) < 1e-9      # C sold, C is flat


def test_price_guards():
    idx = pd.date_range("2020-01-31", "2020-04-30", freq="ME")
    p = pd.DataFrame({"X": [2.0, 10.0, 10.0, 10.0], "Y": [50.0, 300.0, 300.0, 300.0]}, index=idx)
    r = S.clean_returns(p)
    assert pd.isna(r.loc["2020-02-29", "X"])            # +400% from under $5: blanked
    assert abs(r.loc["2020-02-29", "Y"] - 5.0) < 1e-9   # +500% from $50: kept (a real move, e.g. a meme squeeze)
    assert S.clean_returns.n_spikes == 1
