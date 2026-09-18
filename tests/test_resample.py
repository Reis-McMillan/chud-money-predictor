from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from conftest import synthetic_ticks

from chud_predictor.resample import build, load_bars, settlement_from_ticks, to_bars


def test_bars_ohlc_mean_and_gaps():
    # one hour of ticks with a 90-second hole starting at 00:10:30
    drop = set(range(10 * 60 + 30, 10 * 60 + 30 + 90))
    ticks = synthetic_ticks(datetime(2025, 9, 18), 3_600, seed=3, drop=drop)
    bars = to_bars(ticks.lazy(), "1m")
    assert bars.height == 60
    assert bars["ts"][0] == datetime(2025, 9, 18) and bars["ts"][-1] == datetime(2025, 9, 18, 0, 59)
    full = bars.filter(pl.col("is_full"))
    assert full.height == 58  # minute 10 is half empty, minute 11 is a gap, minute 12 is full again
    b10 = bars.row(10, named=True)
    assert b10["n_ticks"] == 30 and not b10["is_full"] and not b10["is_gap"]
    b11 = bars.row(11, named=True)
    assert b11["n_ticks"] == 0 and b11["is_gap"] and b11["mean"] is None
    # hand-check bar 0
    first = ticks.filter(pl.col("ts") < datetime(2025, 9, 18, 0, 1))
    b0 = bars.row(0, named=True)
    assert b0["open"] == first["value"][0] and b0["close"] == first["value"][-1]
    assert b0["high"] == first["value"].max() and b0["low"] == first["value"].min()
    assert b0["mean"] == pytest.approx(first["value"].mean())
    assert b0["n_ticks"] == 60


def test_settlement_equals_bar_mean():
    t0 = datetime(2025, 9, 18, 1, 0)
    ticks = synthetic_ticks(datetime(2025, 9, 18), 2 * 3_600, seed=4)
    # replace the settlement minute with values 1..60 so the answer is 30.5 by hand
    lo, hi = t0 + timedelta(minutes=14), t0 + timedelta(minutes=15)
    window = ticks.filter((pl.col("ts") >= lo) & (pl.col("ts") < hi)).with_columns(pl.Series("value", np.arange(1, 61, dtype=np.float64)))
    ticks = pl.concat([ticks.filter((pl.col("ts") < lo) | (pl.col("ts") >= hi)), window]).sort("ts")
    s, n = settlement_from_ticks(ticks.lazy(), t0)
    assert (s, n) == (30.5, 60)
    bars = to_bars(ticks.lazy(), "1m")
    idx = int((lo - datetime(2025, 9, 18)).total_seconds() // 60)
    assert bars["mean"][idx] == 30.5 and bars["ts"][idx] == lo


def test_build_and_load_cache(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    synthetic_ticks(datetime(2025, 9, 18), 86_400, seed=5).write_parquet(raw / "date=2025-09-18.parquet")
    out, meta = build(raw, tmp_path / "processed", "1m")
    assert meta["n_bars"] == 1440 and meta["n_gaps"] == 0 and meta["n_ticks_histogram"] == {"60": 1440}
    bars = load_bars(tmp_path / "processed", "1m")
    assert bars.height == 1440
    out2, meta2 = build(raw, tmp_path / "processed", "1m")
    assert meta2["built_at"] == meta["built_at"]  # cache hit
