"""Simulation mechanics: weights normalise, identical picks hash to the same id, and a blend
dataset round-trips through the statement format with the fee applied per manager."""
import numpy as np
import pandas as pd

from trackrecord import simulate as S
from trackrecord.returns import FeeSchedule, model_net


def test_spec_normalises_and_hashes_stably():
    a = S.Spec([("x", 30), ("y", 70)], mgmt=0.01, perf=0.10)
    b = S.Spec([("y", 7), ("x", 3)], mgmt=0.01, perf=0.10)
    assert abs(sum(w for _, w in a.normalized().managers) - 1) < 1e-12
    assert a.sim_id() == b.sim_id()                                    # order and scale do not matter
    assert a.sim_id() != S.Spec([("x", 30), ("y", 70)], mgmt=0.02, perf=0.10).sim_id()


def test_blend_applies_fee_per_manager(monkeypatch, tmp_path):
    idx = pd.date_range("2018-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(0)
    series = {"A": pd.Series(rng.normal(0.01, 0.04, 48), index=idx), "B": pd.Series(rng.normal(0.008, 0.03, 48), index=idx)}
    monkeypatch.setattr(S, "gross_series", lambda k: series.get(k))
    b = S.blend(S.Spec([("A", 0.5), ("B", 0.5)], mgmt=0.01, perf=0.20, rebalance="monthly"))
    assert b["months"] == 48 and b["ann_gross"] > b["ann_net"] and b["fee_drag"] > 0
    expect = 0.5 * model_net(series["A"], pd.Series(1 / 12, index=idx), FeeSchedule(0.01, 0.20)) + 0.5 * model_net(series["B"], pd.Series(1 / 12, index=idx), FeeSchedule(0.01, 0.20))
    assert np.allclose(b["net"].values, expect.values)
    monkeypatch.setattr(S, "SIMS", tmp_path)
    d = S.write_dataset(b, {"A": {"name": "Alpha Fund"}, "B": {"name": "Beta Fund"}})
    st = pd.read_csv(d / "statements.csv"); meta = __import__("json").loads((d / "meta.json").read_text())
    assert len(st) == 49 and abs(st.ending_value.iloc[-1] / 1e6 - (1 + b["gross"]).prod()) < 1e-6
    assert meta["style"] == "sim" and "Alpha Fund 50%" in meta["manager"]


def test_allocation_normalises_and_multi_asset_blend(monkeypatch):
    idx = pd.date_range("2018-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(1)
    mgr = {"A": pd.Series(rng.normal(0.01, 0.04, 48), index=idx)}
    sl = pd.DataFrame({"us_stocks": rng.normal(0.008, 0.04, 48), "bonds": rng.normal(0.002, 0.01, 48), "cash": np.full(48, 0.001)}, index=idx)
    monkeypatch.setattr(S, "gross_series", lambda k: mgr.get(k)); monkeypatch.setattr(S, "sleeve_series", lambda: sl)
    sp = S.Spec([("A", 1.0)], alloc={"managers": 50, "us_stocks": 30, "bonds": 20}).normalized()
    assert abs(sum(sp.alloc.values()) - 1) < 1e-12 and abs(sp.alloc["managers"] - 0.5) < 1e-12
    b = S.blend(sp)
    expect = 0.5 * mgr["A"] + 0.3 * sl.us_stocks + 0.2 * sl.bonds
    assert np.allclose(b["gross"].values, expect.values) and len(b["sleeves"]) == 3
    b2 = S.blend(S.Spec([], alloc={"us_stocks": 0.6, "bonds": 0.4}))          # no managers at all is fine
    assert b2["mgr_ann_gross"] is None and abs(b2["sleeves"][0]["weight"] - 0.6) < 1e-12
