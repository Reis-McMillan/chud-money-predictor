from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from chud_predictor.resample import to_bars
from chud_predictor.windows import KalshiSpec, context_matrix, make_kalshi_windows, parse_minutes, strike_ticks

START = datetime(2025, 9, 18)


def _setup(ticks: pl.DataFrame, **spec_kw):
    spec = KalshiSpec(context=spec_kw.pop("context", 64), vol_lookback=30, **spec_kw)
    bars = to_bars(ticks.lazy(), "1m")
    return spec, bars, make_kalshi_windows(bars, ticks.lazy(), spec)


def test_parse_minutes():
    assert parse_minutes("0-14") == tuple(range(15))
    assert parse_minutes("0,5,10,14") == (0, 5, 10, 14)
    assert parse_minutes("3-5,14") == (3, 4, 5, 14)
    with pytest.raises(ValueError):
        parse_minutes("15")


def test_no_lookahead_and_step_mapping(two_days):
    spec, bars, kw = _setup(two_days)
    o = kw.ok_origins
    assert o.height == kw.ok_windows.height * 15
    one_min = timedelta(minutes=1)
    for row in o.head(200).iter_rows(named=True):
        t0, m = row["t0"], row["m"]
        assert row["origin_ts"] == t0 + m * one_min
        # last context bar is labelled T0+(m-1) and fully observable at the origin
        assert row["context_end_ts"] == t0 + (m - 1) * one_min
        assert row["context_end_ts"] + one_min <= row["origin_ts"]
        assert row["h_bars"] == 15 - m and row["step"] == 14 - m
        assert bars["ts"][row["target_idx"]] == t0 + 14 * one_min
        assert row["target_idx"] == row["ctx_end_idx"] + row["h_bars"]
    # the context matrix ends exactly at the last context bar and never touches the target
    X = context_matrix(kw.features, o["ctx_end_idx"].to_numpy()[:20], spec.context)
    y = kw.features["y"].to_numpy()
    for i, row in enumerate(o.head(20).iter_rows(named=True)):
        assert np.allclose(X[i, -1], y[row["ctx_end_idx"]])
        assert np.allclose(X[i, 0], y[row["ctx_end_idx"] - spec.context + 1])
    assert X.shape == (20, spec.context)


def test_windows_are_on_the_quarter_hour_and_have_enough_history(two_days):
    spec, bars, kw = _setup(two_days, context=128)
    w = kw.ok_windows
    assert (w["t0"].dt.minute() % 15 == 0).all() and (w["t0"].dt.second() == 0).all()
    assert w["t0"][0] >= START + timedelta(minutes=128)
    assert (w["t0_idx"] - 1 - 128 + 1 >= 0).all()
    assert w["t0"][-1] + timedelta(minutes=14) <= bars["ts"][-1]


def test_settlement_strike_and_labels(two_days):
    ticks = two_days
    t0 = datetime(2025, 9, 18, 3, 0)
    lo, hi = t0 + timedelta(minutes=14), t0 + timedelta(minutes=15)
    settle = ticks.filter((pl.col("ts") >= lo) & (pl.col("ts") < hi)).with_columns(pl.Series("value", np.arange(1.0, 61.0)))
    ticks = pl.concat([ticks.filter((pl.col("ts") < lo) | (pl.col("ts") >= hi)), settle]).sort("ts")
    spec, bars, kw = _setup(ticks)
    w = kw.ok_windows.filter(pl.col("t0") == t0).row(0, named=True)
    assert w["settlement"] == 30.5 and w["settlement_n_ticks"] == 60
    # open_tick strike is the tick at exactly T0
    tick_at_t0 = ticks.filter(pl.col("ts") == t0)["value"][0]
    assert w["strike"] == tick_at_t0 and w["strike_ts"] == t0 and w["strike_stale_s"] == 0.0
    assert w["label_up"] == (30.5 > tick_at_t0)
    # open_avg60 of window k equals the settlement of window k-1
    prev = kw.ok_windows.filter(pl.col("t0") == t0 - timedelta(minutes=15)).row(0, named=True)
    assert w["open_avg60"] == pytest.approx(prev["settlement"])
    spec2, bars2, kw2 = _setup(ticks, strike_mode="open_avg60")
    w2 = kw2.ok_windows.filter(pl.col("t0") == t0).row(0, named=True)
    assert w2["strike"] == pytest.approx(prev["settlement"])


def test_strike_tick_fallback_and_drop(two_days):
    t0 = datetime(2025, 9, 18, 4, 0)
    # remove the tick at T0: fall back to T0-1s (stale 1s)
    ticks = two_days.filter(pl.col("ts") != t0)
    st = strike_ticks(ticks.lazy(), pl.Series([t0]).cast(pl.Datetime("us")), max_stale_s=5)
    assert st["strike_ts"][0] == t0 - timedelta(seconds=1) and st["strike_stale_s"][0] == 1.0
    # remove 10 s before T0 too: beyond tolerance -> null strike -> window dropped
    ticks2 = two_days.filter((pl.col("ts") < t0 - timedelta(seconds=10)) | (pl.col("ts") > t0))
    spec, bars, kw = _setup(ticks2)
    w = kw.windows.filter(pl.col("t0") == t0).row(0, named=True)
    assert not w["ok"] and w["drop_reason"] == "no_strike_tick"


def test_thin_settlement_bar_dropped(two_days):
    t0 = datetime(2025, 9, 18, 5, 0)
    lo = t0 + timedelta(minutes=14)
    ticks = two_days.filter((pl.col("ts") < lo) | (pl.col("ts") >= lo + timedelta(seconds=10)))  # 50 ticks left
    spec, bars, kw = _setup(ticks)
    w = kw.windows.filter(pl.col("t0") == t0).row(0, named=True)
    assert w["drop_reason"] == "thin_settlement_bar"


def test_context_gap_fraction_drops_origin_not_window(two_days):
    t0 = datetime(2025, 9, 18, 6, 0)
    # blank out 3 minutes ending 2 minutes before T0: with context 64 and max gap 1% (0.64 bars) every
    # origin whose context includes them is dropped; later origins (m large) are unaffected only once
    # the hole leaves the context, which at C=64 never happens within the window -> all dropped.
    hole_lo = t0 - timedelta(minutes=5)
    ticks = two_days.filter((pl.col("ts") < hole_lo) | (pl.col("ts") >= hole_lo + timedelta(minutes=3)))
    spec, bars, kw = _setup(ticks, max_ctx_gap_frac=0.02)
    w = kw.windows.filter(pl.col("t0") == t0).row(0, named=True)
    assert w["ok"]
    o = kw.origins.filter(pl.col("t0") == t0)
    assert (o["n_ctx_gaps"] == 3).all() and (~o["ok"]).all() and (o["drop_reason"] == "ctx_gaps").all()
    # a looser threshold keeps them
    spec3, bars3, kw3 = _setup(ticks, max_ctx_gap_frac=0.1)
    assert kw3.origins.filter(pl.col("t0") == t0)["ok"].all()


def test_start_end_stride_and_max_windows(two_days):
    spec = KalshiSpec(context=64, vol_lookback=30)
    bars = to_bars(two_days.lazy(), "1m")
    kw = make_kalshi_windows(bars, two_days.lazy(), spec, start=date(2025, 9, 19), end=date(2025, 9, 19), stride_windows=4, max_windows=10)
    w = kw.ok_windows
    assert w.height == 10 and (w["t0"].dt.date() == date(2025, 9, 19)).all()
    assert (w["t0"].diff().drop_nulls() == timedelta(hours=1)).all()


def test_context_clamped_to_timesfm_limit():
    spec = KalshiSpec(context=20_000)
    assert spec.context == 15_360
