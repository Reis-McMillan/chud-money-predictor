from datetime import date, datetime, timedelta

import polars as pl
from conftest import CONTRACT_TABLE, basic_auth_header, synthetic_candles, synthetic_ticks

from chud_predictor.qdb import BRTI, CONTRACT_NUMERIC, CONTRACTS, QuestDB, ts_literal


def test_bounds_day_counts_and_fetch(fake_qdb, settings):
    d0 = date(2025, 9, 18)
    fake_qdb.add_day(d0, synthetic_ticks(datetime(2025, 9, 18), 86_400))
    fake_qdb.add_day(d0 + timedelta(days=1), synthetic_ticks(datetime(2025, 9, 19), 3_600))
    q = QuestDB(settings)
    assert q.ping() == "fake-1"
    b = q.bounds(BRTI)
    assert b.rows == 90_000 and b.min_ts == datetime(2025, 9, 18) and b.max_ts == datetime(2025, 9, 19, 0, 59, 59)
    counts = q.day_counts(BRTI, b.max_ts + timedelta(microseconds=1))
    assert counts == {d0: 86_400, d0 + timedelta(days=1): 3_600}
    df = q.fetch_day(BRTI, d0, b.max_ts + timedelta(microseconds=1))
    assert df.height == 86_400 and df.columns == ["ts", "value"]
    assert df["ts"][0] == datetime(2025, 9, 18) and df["ts"][-1] == datetime(2025, 9, 18, 23, 59, 59)
    assert fake_qdb.auth_seen[0] == basic_auth_header("admin", "pw")
    assert "index_id = 'BRTI'" in fake_qdb.requests[-1]


def test_contract_candles_fetch(fake_qdb, settings):
    ticks = synthetic_ticks(datetime(2025, 12, 20), 4 * 3_600, seed=2)
    t_null = datetime(2025, 12, 20, 1, 0)
    candles = synthetic_candles(ticks, seed=3, null_strike_t0s={t_null}, zero_volume_frac=0.3)
    fake_qdb.add_rows(candles, CONTRACT_TABLE)
    q = QuestDB(settings)
    b = q.bounds(CONTRACTS)
    assert b.rows == candles.height and b.max_ts == candles["ts"].max()
    df = q.fetch_day(CONTRACTS, date(2025, 12, 20), b.max_ts + timedelta(microseconds=1))
    assert df.columns == ["ts", "ticker", *CONTRACT_NUMERIC] and df.height == candles.height
    assert df.schema["ticker"] == pl.Utf8 and all(df.schema[c] == pl.Float64 for c in CONTRACT_NUMERIC)
    assert "series_ticker = 'KXBTC15M'" in fake_qdb.requests[-1]
    # nulls survive the CSV round trip: a nulled strike and the trade prices of zero-volume minutes
    assert df.filter(pl.col("ts") == t_null + timedelta(minutes=1))["floor_strike"][0] is None
    zero_vol = df.filter(pl.col("volume") == 0)
    assert zero_vol.height > 0 and zero_vol["price_close"].null_count() == zero_vol.height
    assert df.filter(pl.col("volume") > 0)["price_close"].null_count() == 0
    assert df["yes_bid_close"].null_count() == 0
    assert df.equals(candles.select(df.columns))


def test_ts_literal():
    assert ts_literal(datetime(2025, 9, 18, 1, 2, 3, 4)) == "'2025-09-18T01:02:03.000004Z'"


def test_retry_on_server_error(fake_qdb, settings):
    fake_qdb.add_day(date(2025, 9, 18), synthetic_ticks(datetime(2025, 9, 18), 600))
    fake_qdb.fail_next = 1
    q = QuestDB(settings)
    df = q.fetch_day(BRTI, date(2025, 9, 18), datetime(2025, 9, 19))
    assert df.height == 600
