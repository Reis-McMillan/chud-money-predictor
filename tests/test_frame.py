"""The joined minute frame: END-labelled candles on START-labelled bars, strike identity, flags."""

from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from conftest import CONTRACT_START, T_DEGENERATE, T_DROPPED, T_NO_SETTLE, T_NULL_STRIKE, write_raw

from chud_predictor.features import build_frame, build_frame_from, describe_frame, load_frame
from chud_predictor.resample import build

MIN = timedelta(minutes=1)
T0 = CONTRACT_START.replace(hour=3)


def _row(frame: pl.DataFrame, ts) -> dict:
    return frame.filter(pl.col("ts") == ts).row(0, named=True)


def test_end_labelled_candles_land_on_start_labelled_bars(contract_world):
    frame, candles = contract_world["frame"], contract_world["candles"]
    assert frame.height == 2880 and (frame["ts"].diff().drop_nulls() == MIN).all()
    assert (frame["k"] == frame["ts"].dt.minute() % 15).all()
    for k in range(15):
        candle = candles.filter(pl.col("ts") == T0 + (k + 1) * MIN).row(0, named=True)   # ends at T0+k+1
        row = _row(frame, T0 + k * MIN)                                                   # bar that starts at T0+k
        assert row["k"] == k and row["t0"] == T0 and row["ticker"] == candle["ticker"] == row["win_ticker"]
        assert row["mid_close"] == pytest.approx((candle["yes_bid_close"] + candle["yes_ask_close"]) / 2)
        assert row["spread"] == pytest.approx(candle["yes_ask_close"] - candle["yes_bid_close"])
        assert row["trade_close"] == candle["price_close"] and row["is_settlement"] == (k == 14)
    settle = _row(frame, T0 + 14 * MIN)
    assert settle["mid_close"] in (pytest.approx(0.999), pytest.approx(0.001))
    assert frame.filter(pl.col("has_candle")).group_by("t0").agg(pl.col("ticker").n_unique())["ticker"].max() == 1


def test_strike_is_the_mean_of_the_minute_before_the_window(contract_world):
    frame, ticks = contract_world["frame"], contract_world["ticks"]
    by_hand = ticks.filter((pl.col("ts") >= T0 - MIN) & (pl.col("ts") < T0))["value"].mean()
    rows = frame.filter(pl.col("t0") == T0)
    assert (rows["strike"] == rows["strike"][0]).all() and rows["strike"][0] == pytest.approx(by_hand, rel=1e-12)
    assert rows["floor_strike"][0] == pytest.approx(_row(frame, T0 - MIN)["brti_mean"], rel=1e-12)
    assert not rows["strike_imputed"].any()
    # a window whose Kalshi strike is null gets exactly that value, flagged
    imp = frame.filter(pl.col("t0") == T_NULL_STRIKE)
    assert imp["floor_strike"].null_count() == 15 and imp["strike_imputed"].all()
    assert imp["strike"][0] == pytest.approx(_row(frame, T_NULL_STRIKE - MIN)["brti_mean"], rel=1e-12)
    assert frame.filter(pl.col("t0") == CONTRACT_START)["strike"].null_count() == 15   # no bar before the very first window


def test_missing_candles_and_flags(contract_world):
    frame = contract_world["frame"]
    dropped = frame.filter(pl.col("t0") == T_DROPPED).sort("k")
    assert dropped["has_candle"].to_list() == [k not in (3, 4) for k in range(15)]
    assert dropped.filter(~pl.col("has_candle"))["mid_close"].null_count() == 2 and not dropped["target_valid"][3]
    assert dropped["strike"].null_count() == 0 and dropped["win_ticker"].null_count() == 0   # window-level values survive
    assert frame.filter(pl.col("k") == 0)["mid_ret_1m"].null_count() == frame.filter(pl.col("k") == 0).height
    r = _row(frame, T0 + 5 * MIN)
    assert r["mid_ret_1m"] == pytest.approx(r["mid_close"] - _row(frame, T0 + 4 * MIN)["mid_close"])
    deg = frame.filter((pl.col("t0") == T_DEGENERATE) & (pl.col("k") < 14))
    assert deg["target_valid"].all() and not deg["quote_ok"].any() and deg["spread"][0] == pytest.approx(0.6)
    assert frame.filter(pl.col("t0") == T0)["quote_ok"].all()
    zero_vol = frame.filter(pl.col("has_candle") & (pl.col("volume") == 0))
    assert zero_vol.height > 0 and zero_vol["trade_close"].null_count() == zero_vol.height
    assert (zero_vol["trade_close_imp"] == zero_vol["mid_close"]).all()


def test_fair_value_columns(contract_world):
    frame = contract_world["frame"]
    r = _row(frame, T0 + 9 * MIN)
    assert r["moneyness"] == pytest.approx(np.log(r["brti_close"] / r["strike"]))
    # seen from the close of row 9 (= minute 10 of the window) the settlement minute starts 4 bars ahead
    assert r["rw_z"] == pytest.approx(np.clip(r["moneyness"] / (r["sigma_1m"] * np.sqrt(13 - 9 + 1 / 3)), -8, 8))
    ok = frame.filter(pl.col("rw_p").is_not_null() & ~pl.col("is_settlement"))
    assert ((ok["rw_p"] >= 0) & (ok["rw_p"] <= 1)).all() and ((ok["rw_p"] > 0.5) == (ok["moneyness"] > 0)).all()
    # a settled row's fair value is the outcome itself
    done = frame.filter(pl.col("is_settlement") & pl.col("rw_p").is_not_null())
    assert ((done["rw_p"] > 0.5) == done["outcome_brti"]).all() and ((done["rw_p"] < 1e-6) | (done["rw_p"] > 1 - 1e-6)).mean() > 0.95


def test_outcomes_agree_and_misalignment_is_caught(contract_world):
    frame, bars, candles = contract_world["frame"], contract_world["bars"], contract_world["candles"]
    meta = describe_frame(frame)
    assert meta["outcome_disagreement_rate"] == 0.0 and meta["strike_identity_abs_err_p99"] < 1e-6
    assert meta["n_windows_strike_imputed"] == 1 and meta["n_complete_windows"] == meta["n_windows"] - 2
    assert frame.filter(pl.col("t0") == T_NO_SETTLE)["outcome_price"].null_count() == 15
    assert frame.filter(pl.col("t0") == T_NO_SETTLE)["outcome_brti"].null_count() == 0
    # treat the END labels as START labels (the easy mistake): the audit must light up
    wrong = build_frame_from(bars, candles.with_columns(pl.col("ts") + pl.duration(minutes=1)), vol_lookback=60)
    assert describe_frame(wrong)["outcome_disagreement_rate"] > 0.05


def test_build_cache_and_load(contract_world, tmp_path):
    raw, craw, processed = tmp_path / "raw" / "brti", tmp_path / "raw" / "contracts", tmp_path / "processed"
    write_raw(contract_world["ticks"], raw)
    write_raw(contract_world["candles"], craw)
    build(raw, processed, "1m")
    out, meta = build_frame(processed, craw, vol_lookback=60)
    frame = load_frame(processed)
    assert frame.equals(contract_world["frame"]) and meta["n_rows"] == 2880
    assert build_frame(processed, craw, vol_lookback=60)[1]["built_at"] == meta["built_at"]          # cache hit
    assert build_frame(processed, craw, vol_lookback=90)[1]["built_at"] != meta["built_at"]          # parameter change rebuilds
    # without contract files the frame still builds, BRTI only
    _, meta_b = build_frame(processed, tmp_path / "nothing", vol_lookback=60, force=True)
    assert meta_b["n_rows_with_candle"] == 0 and load_frame(processed)["mid_close"].null_count() == 2880
