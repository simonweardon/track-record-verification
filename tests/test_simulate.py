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
    assert len(st) == 48 and abs(st.ending_value.iloc[-1] / 1e6 - (1 + b["gross"]).prod()) < 1e-6
    assert meta["style"] == "sim" and "Alpha Fund 50%" in meta["manager"]
