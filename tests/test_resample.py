import json
from datetime import datetime, timedelta

import polars as pl
import pytest
from conftest import synthetic_ticks, write_raw

from chud_predictor.resample import build, load_bars, meta_path, to_bars, to_bars_by_day


def test_bars_ohlc_mean_and_gaps():
    # one hour of ticks with a 90-second hole starting at 00:10:30
    drop = set(range(10 * 60 + 30, 10 * 60 + 30 + 90))
    ticks = synthetic_ticks(datetime(2025, 9, 18), 3_600, seed=3, drop=drop)
    bars = to_bars(ticks.lazy(), "1m")
    assert bars.height == 60
    assert bars["ts"][0] == datetime(2025, 9, 18) and bars["ts"][-1] == datetime(2025, 9, 18, 0, 59)
    assert bars.filter(pl.col("is_full")).height == 58  # minute 10 is half empty, minute 11 is a gap, minute 12 is full again
    b10 = bars.row(10, named=True)
    assert b10["n_ticks"] == 30 and not b10["is_full"] and not b10["is_gap"]
    b11 = bars.row(11, named=True)
    assert b11["n_ticks"] == 0 and b11["n_raw_ticks"] == 0 and b11["is_gap"] and b11["mean"] is None
    first = ticks.filter(pl.col("ts") < datetime(2025, 9, 18, 0, 1))
    b0 = bars.row(0, named=True)
    assert b0["open"] == first["value"][0] and b0["close"] == first["value"][-1]
    assert b0["high"] == first["value"].max() and b0["low"] == first["value"].min()
    assert b0["mean"] == pytest.approx(first["value"].mean())
    assert b0["n_ticks"] == 60 and b0["n_raw_ticks"] == 60


def test_once_per_second_and_five_hz_give_identical_bars():
    """The index table switched from 1 Hz to 5 Hz; bars must not change with the tick density."""
    one = to_bars(synthetic_ticks(datetime(2026, 5, 1), 3_600, seed=7, hz=1).lazy(), "1m")
    five = to_bars(synthetic_ticks(datetime(2026, 5, 1), 3_600, seed=7, hz=5).lazy(), "1m")
    stats = ["ts", "open", "high", "low", "close", "mean", "std", "n_ticks", "is_gap", "is_full"]
    assert one.select(stats).equals(five.select(stats))
    assert (one["n_raw_ticks"] == 60).all() and (five["n_raw_ticks"] == 300).all() and five["is_full"].all()


def test_offset_sub_second_grid_is_rejected():
    ticks = synthetic_ticks(datetime(2026, 5, 1), 600).with_columns(pl.col("ts") + pl.duration(milliseconds=100))
    with pytest.raises(ValueError, match="none on a whole second"):
        to_bars(ticks.lazy(), "1m")


def test_per_day_build_equals_whole_history_build(tmp_path):
    ticks = pl.concat([synthetic_ticks(datetime(2026, 4, 30), 86_400, seed=1, hz=1),
                       synthetic_ticks(datetime(2026, 5, 1), 86_400, seed=2, hz=5, start_price=101_000.0)])
    write_raw(ticks, tmp_path / "raw")
    by_day = to_bars_by_day(sorted((tmp_path / "raw").glob("date=*.parquet")), "1m")
    assert by_day.equals(to_bars(ticks.lazy(), "1m")) and by_day.height == 2880


def test_build_cache_and_schema_guard(tmp_path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    write_raw(synthetic_ticks(datetime(2025, 9, 18), 86_400, seed=5), raw)
    write_raw(synthetic_ticks(datetime(2025, 9, 19), 3_600, seed=6, hz=5), raw)
    out, meta = build(raw, processed, "1m")
    assert meta["n_bars"] == 1440 + 60 and meta["n_gaps"] == 0 and meta["n_ticks_histogram"] == {"60": 1500}
    assert meta["days_once_per_second"] == 1 and meta["days_sub_second"] == 1 and meta["schema_version"] == 3
    assert load_bars(processed, "1m").height == 1500
    _, meta2 = build(raw, processed, "1m")
    assert meta2["built_at"] == meta["built_at"]  # cache hit
    # a cache written by the previous bar schema is refused
    mp = meta_path(processed, "1m")
    mp.write_text(json.dumps(meta | {"schema_version": 2}))
    with pytest.raises(RuntimeError, match="re-run"):
        load_bars(processed, "1m")
    _, meta3 = build(raw, processed, "1m")
    assert meta3["schema_version"] == 3 and meta3["built_at"] != meta["built_at"]


def test_last_second_of_bar_is_its_close():
    ticks = synthetic_ticks(datetime(2025, 9, 18), 120, seed=9, hz=5)
    bars = to_bars(ticks.lazy(), "1m")
    assert bars["close"][0] == ticks.filter(pl.col("ts") == datetime(2025, 9, 18, 0, 0, 59))["value"][0]
    assert bars["ts"][1] == datetime(2025, 9, 18) + timedelta(minutes=1)
