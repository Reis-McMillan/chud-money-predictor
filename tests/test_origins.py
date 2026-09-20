"""Origin index arithmetic and drop rules."""

from datetime import date, timedelta

import polars as pl
import pytest
from conftest import CONTRACT_START, T_DEGENERATE, T_DROPPED, T_NO_SETTLE

from chud_predictor.origins import ContractSpec, make_origins, parse_minutes

MIN = timedelta(minutes=1)
SPEC = ContractSpec(context=64)


def test_parse_minutes_and_spec_validation():
    assert parse_minutes("0-14") == tuple(range(15)) and parse_minutes("0,5,10,14") == (0, 5, 10, 14)
    assert parse_minutes("3-5,14") == (3, 4, 5, 14)
    with pytest.raises(ValueError):
        parse_minutes("15")
    with pytest.raises(ValueError, match="multiple of 32"):
        ContractSpec(context=100)
    with pytest.raises(ValueError):
        ContractSpec(context=32 * 1000)


def test_index_arithmetic_for_every_minute(contract_world):
    frame = contract_world["frame"]
    os = make_origins(frame, SPEC)
    ok = os.ok_origins
    assert set(ok["m"].unique()) == set(range(15))
    ts, k = frame["ts"], frame["k"]
    t0 = CONTRACT_START.replace(hour=5)
    idx_t0 = frame.filter(pl.col("ts") == t0)["idx"][0]
    for m in range(15):
        o = ok.filter((pl.col("t0") == t0) & (pl.col("m") == m)).row(0, named=True)
        i = o["i"]
        assert i == idx_t0 + m - 1 and o["n_steps"] == 15 - m
        assert o["origin_ts"] == t0 + m * MIN == ts[i] + MIN               # the last context row closes at the origin
        for h in range(1, o["n_steps"] + 1):
            assert ts[i + h] == t0 + (m + h - 1) * MIN and k[i + h] == m + h - 1   # candle END ts = t0 + m + h
        assert k[i + o["n_steps"]] == 14                                    # exactly one settlement step, the last
        assert o["last_mid"] == frame["mid_close"][i] and o["brti_close"] == frame["brti_close"][i]


def test_window_open_is_an_origin_with_the_known_strike(contract_world):
    frame = contract_world["frame"]
    ok = make_origins(frame, SPEC).ok_origins
    t0 = CONTRACT_START.replace(hour=5)
    o = ok.filter((pl.col("t0") == t0) & (pl.col("m") == 0)).row(0, named=True)
    i = o["i"]
    assert frame["k"][i] == 14 and o["n_steps"] == 15
    assert o["last_mid"] in (pytest.approx(0.999), pytest.approx(0.001))      # the contract that just settled
    # strike read from rows <= i only, and equal to what Kalshi publishes for the new window
    assert o["strike"] == frame["brti_mean"][i] == pytest.approx(frame["strike"][i + 1], rel=1e-12)
    later = ok.filter((pl.col("t0") == t0) & (pl.col("m") == 7)).row(0, named=True)
    assert later["strike"] == pytest.approx(o["strike"], rel=1e-12)


def test_drop_reasons(contract_world):
    frame = contract_world["frame"]
    o = make_origins(frame, ContractSpec(context=64, max_ctx_gap_frac=0.02)).origins

    def reason(t0, m):
        return o.filter((pl.col("t0") == t0) & (pl.col("m") == m))["drop_reason"][0]

    assert o.filter(pl.col("i") < 63)["drop_reason"].unique().to_list() == ["insufficient_history"]
    tail = o.filter(pl.col("i") >= frame.height - 15)                           # contexts ending inside the last window
    assert tail["ok"].sum() == 14 and tail["drop_reason"][-1] == "horizon_beyond_data"   # the very last bar opens a window with no data
    assert reason(T_DROPPED, 4) == "no_contract" and reason(T_DROPPED, 5) == "no_contract"   # context would end on a missing candle
    assert reason(T_DROPPED, 3) is None                                       # its context ends before the hole
    # 2 missing candles are 3% of a 64-row context: above a 2% budget every context that still holds them goes
    assert reason(T_DROPPED, 6) == "ctx_gaps" and reason(T_DROPPED + 15 * MIN, 3) == "ctx_gaps"
    assert reason(T_NO_SETTLE, 14) == "no_valid_step" and reason(T_NO_SETTLE, 13) is None
    assert reason(T_NO_SETTLE + 15 * MIN, 0) == "no_contract"
    # a context that *starts* on a missing candle is dropped even inside the gap budget
    loose = make_origins(frame, ContractSpec(context=64, max_ctx_gap_frac=0.1)).origins
    gap_idx = frame.filter((pl.col("t0") == T_DROPPED) & (pl.col("k") == 3))["idx"][0]
    assert loose.filter(pl.col("i") == gap_idx + 63)["drop_reason"][0] == "ctx_start_gap"
    assert loose.filter(pl.col("i") == gap_idx + 65)["ok"][0]
    assert loose.filter((pl.col("t0") == T_DROPPED) & (pl.col("m") == 6))["ok"][0]
    strict = make_origins(frame, ContractSpec(context=64, require_quote_ok=True)).origins
    assert strict.filter((pl.col("t0") == T_DEGENERATE) & (pl.col("m") == 5))["drop_reason"][0] == "degenerate_quote"


def test_selection(contract_world):
    frame = contract_world["frame"]
    day2 = date(2025, 12, 21)
    os = make_origins(frame, ContractSpec(context=64, minutes=(0, 5, 10, 14)), start=day2, end=day2, stride_windows=4, max_windows=6)
    ok = os.ok_origins
    assert set(ok["m"].unique()) == {0, 5, 10, 14} and (ok["date"] == day2).all()
    t0s = ok["t0"].unique().sort()
    assert t0s.len() == 6 and (t0s.diff().drop_nulls() == timedelta(hours=1)).all()
    assert ok.height == 24 and os.drop_report().height >= 0
