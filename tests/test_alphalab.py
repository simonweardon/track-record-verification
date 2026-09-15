"""Alpha lab mechanics: point-in-time lookups never see a filing before its filed date, annual-duration
facts are picked out of the XBRL mix, z-scores are winsorised, and the IC / decile tests recover a planted signal."""
import numpy as np
import pandas as pd

from trackrecord import alphalab as A


def test_point_in_time_lookup_respects_filed_date():
    fund = pd.DataFrame([
        dict(ticker="X", fact="equity", start=np.nan, end="2020-12-31", filed="2021-02-20", value=100.0),
        dict(ticker="X", fact="equity", start=np.nan, end="2021-03-31", filed="2021-05-10", value=120.0),
        dict(ticker="X", fact="net_income", start="2020-01-01", end="2020-12-31", filed="2021-02-20", value=10.0),   # annual
        dict(ticker="X", fact="net_income", start="2020-10-01", end="2020-12-31", filed="2021-02-20", value=3.0),    # quarterly: ignored
        dict(ticker="X", fact="net_income", start="2021-01-01", end="2021-12-31", filed="2022-02-25", value=14.0),
    ])
    fund["filed"] = pd.to_datetime(fund.filed); fund["end"] = pd.to_datetime(fund["end"])
    eq = A.pit_table(fund, "equity", "instant"); ni = A.pit_table(fund, "net_income", "annual")
    idx = pd.Index(["X", "Y"])
    assert A.as_of(eq, idx, pd.Timestamp("2021-03-31"))["X"] == 100.0        # May filing not yet public
    assert A.as_of(eq, idx, pd.Timestamp("2021-05-31"))["X"] == 120.0
    assert pd.isna(A.as_of(eq, idx, pd.Timestamp("2021-01-31"))["X"])         # nothing filed yet
    assert pd.isna(A.as_of(eq, idx, pd.Timestamp("2021-05-31"))["Y"])
    assert len(ni) == 2 and A.as_of(ni, idx, pd.Timestamp("2021-06-30"))["X"] == 10.0
    assert A.as_of(ni, idx, pd.Timestamp("2022-03-31"))["X"] == 14.0
    assert pd.isna(A.as_of(ni, idx, pd.Timestamp("2023-12-31"))["X"])         # stale: period end older than 15 months


def test_zscore_winsorises_and_handles_degenerate():
    x = pd.Series([0.0] * 20 + [100.0])
    z = A._z(x)
    assert z.max() <= 3.0 and abs(z.iloc[:20].mean()) < 1
    assert A._z(pd.Series([1.0] * 15)).isna().all()


def test_ic_and_deciles_recover_a_planted_signal():
    rng = np.random.default_rng(1)
    rows = []
    for m in pd.date_range("2020-01-31", periods=24, freq="ME"):
        sig = rng.normal(size=200)
        fwd = 0.01 * sig + rng.normal(scale=0.05, size=200)
        rows.append(pd.DataFrame({"month": m, "ticker": [f"T{i}" for i in range(200)], "good": sig, "noise": rng.normal(size=200), "fwd": fwd}))
    panel = pd.concat(rows, ignore_index=True)
    ic_good, ic_noise = A.ic_series(panel, "good"), A.ic_series(panel, "noise")
    assert len(ic_good) == 24 and ic_good.mean() > 0.1 and abs(ic_noise.mean()) < 0.1
    sp, dec = A.decile_spread(panel, "good")
    assert sp.mean() > 0 and dec.loc[10] > dec.loc[1] and list(dec.index) == list(range(1, 11))
