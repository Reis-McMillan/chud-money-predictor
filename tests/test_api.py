from dataclasses import replace
from datetime import date, datetime, timedelta

import polars as pl
import pytest
from conftest import CONTRACT_TABLE, synthetic_candles, synthetic_ticks

from chud_predictor.api import (
    BRTI,
    CONTRACT_NUMERIC,
    CONTRACTS,
    ApiError,
    ChudApi,
    SseEvent,
    iter_sse,
    parse_api_ts,
    rows_to_frame,
    ts_param,
)
from chud_predictor.settings import Settings

D0 = date(2025, 9, 18)
T0 = datetime(2025, 9, 18)
UPTO = datetime(2025, 9, 19)


def _params(fake_api, path_suffix: str = "/data/") -> list[dict]:
    return [p for path, p in fake_api.requests if path_suffix in path]


# -- pure units ------------------------------------------------------------------------------------

def test_iter_sse_parses_frames_comments_and_crlf():
    lines = [
        ":",                                        # keep-alive comment
        "event: meta", "data: {\"a\": 1}", "retry: 10000", "",
        "event: row", "data: {\"b\": 2}", "id: 2026-09-24T00:00:00.000001Z", "",
        "data: one", "data: two", "",               # no `event:` -> "message", multi-line data
        "event: done", "data: {}",                  # trailing event, never dispatched
    ]
    events = list(iter_sse(f"{line}\r\n" for line in lines))
    assert events[0] == SseEvent("meta", '{"a": 1}', None, 10_000)
    assert events[1] == SseEvent("row", '{"b": 2}', "2026-09-24T00:00:00.000001Z", None)
    assert events[2] == SseEvent("message", "one\ntwo", None, None)
    assert len(events) == 3


def test_ts_helpers_round_trip():
    assert ts_param(datetime(2025, 9, 18, 1, 2, 3, 4)) == "2025-09-18T01:02:03.000004Z"
    assert parse_api_ts("2025-09-18T01:02:03.000004Z") == datetime(2025, 9, 18, 1, 2, 3, 4)
    assert parse_api_ts("2026-09-17T02:00:30.123456789Z") == datetime(2026, 9, 17, 2, 0, 30, 123456)   # chrono nanos
    assert parse_api_ts("2026-09-17T02:00:00Z") == datetime(2026, 9, 17, 2, 0)
    assert parse_api_ts(None) is None


def test_rows_to_frame_projects_and_keeps_nulls():
    rows = [
        '{"index_id":"BRTI","source":"fake","value":2.0,"received_at":null,"ts":"2025-09-18T00:00:01.500000Z"}',
        '{"index_id":"BRTI","source":"fake","value":null,"received_at":null,"ts":"2025-09-18T00:00:00.000000Z"}',
    ]
    df = rows_to_frame(rows, BRTI)
    assert df.columns == ["ts", "value"] and df.schema == BRTI.schema
    assert df["ts"].to_list() == [datetime(2025, 9, 18), datetime(2025, 9, 18, 0, 0, 1, 500_000)]
    assert df["value"].to_list() == [None, 2.0]
    assert rows_to_frame([], BRTI).schema == BRTI.schema and rows_to_frame([], BRTI).is_empty()


# -- metadata --------------------------------------------------------------------------------------

def test_health_and_market_are_unauthenticated(api, fake_api, tokens):
    assert api.health() == "ok"
    assert api.market()["market"]["tag"] == "btc-15m"
    assert fake_api.auth_seen == [None, None] and tokens.calls == 0


def test_bounds_from_the_summary(api, fake_api):
    fake_api.add_day(D0, synthetic_ticks(T0, 86_400))
    fake_api.add_day(date(2025, 9, 19), synthetic_ticks(datetime(2025, 9, 19), 3_600))
    ticks = synthetic_ticks(datetime(2025, 12, 20), 4 * 3_600, seed=2)
    fake_api.add_rows(synthetic_candles(ticks, seed=3), CONTRACT_TABLE)

    b = api.bounds(BRTI)
    assert b.rows == 90_000 and b.min_ts == T0 and b.max_ts == datetime(2025, 9, 19, 0, 59, 59)
    c = api.bounds(CONTRACTS)
    assert c.rows == fake_api.all_rows(CONTRACT_TABLE).height
    assert c.min_ts.tzinfo is None and c.max_ts == fake_api.all_rows(CONTRACT_TABLE)["ts"].max()
    assert api.summaries()["index_values_hist"].table == "index_values_hist"


def test_bounds_before_the_first_summary_refresh(api, fake_api):
    fake_api.tables_null = True
    with pytest.raises(ApiError, match="60 s"):
        api.bounds(BRTI)


def test_unknown_market_is_fatal(settings, tokens):
    with ChudApi(replace(settings, market_tag="nope"), tokens, sleep=lambda _s: None) as api:
        with pytest.raises(ApiError, match="404"):
            api.market()


# -- fetch_day -------------------------------------------------------------------------------------

def test_fetch_day_brti(api, fake_api, tokens):
    fake_api.add_day(D0, synthetic_ticks(T0, 86_400))
    fake_api.add_day(date(2025, 9, 19), synthetic_ticks(datetime(2025, 9, 19), 3_600))
    snapshot = datetime(2025, 9, 19, 0, 59, 59, 1)
    fetch = api.fetch_day(BRTI, D0, snapshot)
    df = fetch.df
    assert fetch.ok and fetch.reported_rows == 86_400 and fetch.attempts == 1
    assert df.columns == ["ts", "value"] and df.height == 86_400 and df.schema == BRTI.schema
    assert df["ts"].is_sorted() and df["ts"][0] == T0 and df["ts"][-1] == datetime(2025, 9, 18, 23, 59, 59)
    assert df.equals(fake_api.days[D0].select("ts", "value"))
    assert _params(fake_api)[-1] == {"start": "2025-09-18T00:00:00.000000Z", "end": "2025-09-19T00:00:00.000000Z"}
    assert fake_api.requests[-1][0] == "/btc-15m/data/index-hist"

    # the snapshot clips the tail day's `end`
    api.fetch_day(BRTI, date(2025, 9, 19), snapshot)
    assert _params(fake_api)[-1]["end"] == "2025-09-19T00:59:59.000001Z"
    assert tokens.calls == 2 and all("token" not in p for p in _params(fake_api))


def test_fetch_day_contracts_keeps_nulls_and_nan(api, fake_api):
    ticks = synthetic_ticks(datetime(2025, 12, 20), 4 * 3_600, seed=2)
    t_null = datetime(2025, 12, 20, 1, 0)
    candles = synthetic_candles(ticks, seed=3, null_strike_t0s={t_null}, zero_volume_frac=0.3)
    fake_api.add_rows(candles, CONTRACT_TABLE)
    fetch = api.fetch_day(CONTRACTS, date(2025, 12, 20), candles["ts"].max() + timedelta(microseconds=1))
    df = fetch.df
    assert fetch.ok and df.columns == ["ts", "ticker", *CONTRACT_NUMERIC] and df.height == candles.height
    assert df.schema["ticker"] == pl.Utf8 and all(df.schema[c] == pl.Float64 for c in CONTRACT_NUMERIC)
    assert "series_ticker" not in df.columns and "source" not in df.columns
    assert df.filter(pl.col("ts") == t_null + timedelta(minutes=1))["floor_strike"][0] is None
    zero_vol = df.filter(pl.col("volume") == 0)
    assert zero_vol.height > 0 and zero_vol["price_close"].null_count() == zero_vol.height
    assert df.filter(pl.col("volume") > 0)["price_close"].null_count() == 0
    assert df["yes_bid_close"].null_count() == 0
    assert df.equals(candles.select(df.columns))

    # a non-finite double reaches us as null (questdb::row_json), not as a NaN float
    fake_api.nan_cell = ("open_interest", 3)
    df2 = api.fetch_day(CONTRACTS, date(2025, 12, 20), candles["ts"].max() + timedelta(microseconds=1)).df
    assert df2["open_interest"][3] is None and df2["open_interest"].null_count() == 1


def test_empty_range_makes_no_request(api, fake_api):
    fake_api.add_day(D0, synthetic_ticks(T0, 600))
    before = len(fake_api.requests)
    fetch = api.fetch_day(BRTI, date(2025, 9, 19), datetime(2025, 9, 19))    # start == end: the API 400s on it
    assert fetch.ok and fetch.df.is_empty() and fetch.df.schema == BRTI.schema and fetch.attempts == 0
    assert len(fake_api.requests) == before


def test_day_with_no_rows_is_an_honest_zero(api, fake_api):
    fake_api.add_day(D0, synthetic_ticks(T0, 600))
    fetch = api.fetch_day(BRTI, date(2025, 9, 17), UPTO)
    assert fetch.ok and fetch.reported_rows == 0 and fetch.df.is_empty()


def test_keepalive_comments_are_ignored(api, fake_api):
    fake_api.keepalive = True
    fake_api.add_day(D0, synthetic_ticks(T0, 2_000))
    assert api.fetch_day(BRTI, D0, UPTO).df.height == 2_000


# -- failure handling ------------------------------------------------------------------------------

def test_retry_on_server_error(api, fake_api):
    fake_api.add_day(D0, synthetic_ticks(T0, 600))
    fake_api.fail_next = 2
    fetch = api.fetch_day(BRTI, D0, UPTO)
    assert fetch.df.height == 600 and fetch.attempts == 3 and fetch.ok


def test_server_error_exhausted_raises(api, fake_api):
    fake_api.add_day(D0, synthetic_ticks(T0, 600))
    fake_api.fail_next = 3
    with pytest.raises(ApiError, match="after 3 attempts"):
        api.fetch_day(BRTI, D0, UPTO)


def test_retry_on_mid_stream_error(api, fake_api):
    fake_api.add_day(D0, synthetic_ticks(T0, 1_000))
    fake_api.error_next = 1
    fetch = api.fetch_day(BRTI, D0, UPTO)
    assert fetch.ok and fetch.df.height == 1_000 and fetch.attempts == 2


def test_mid_stream_error_exhausted_keeps_the_prefix(api, fake_api):
    fake_api.add_day(D0, synthetic_ticks(T0, 1_000))
    fake_api.error_next = 99
    fetch = api.fetch_day(BRTI, D0, UPTO)
    assert not fetch.ok and fetch.reported_rows is None and fetch.df.height == 500
    assert fetch.attempts == 3 and "mid-stream" in fetch.note


def test_truncated_stream_is_not_ok(api, fake_api):
    fake_api.add_day(D0, synthetic_ticks(T0, 1_000))
    fake_api.truncate_reported = D0
    fetch = api.fetch_day(BRTI, D0, UPTO)
    assert not fetch.ok and fetch.df.height == 900 and fetch.reported_rows == 1_000 and fetch.attempts == 3


def test_throttle_then_success(settings, tokens, fake_api):
    fake_api.add_day(D0, synthetic_ticks(T0, 600))
    fake_api.throttle_next = 3
    slept: list[float] = []
    with ChudApi(settings, tokens, sleep=slept.append) as api:
        fetch = api.fetch_day(BRTI, D0, UPTO)
    assert fetch.ok and fetch.attempts == 4
    assert [round(s / b, 3) for s, b in zip(slept, (5, 10, 20), strict=True)] == pytest.approx([1.0] * 3, abs=0.2)


def test_throttle_honours_retry_after(settings, tokens, fake_api):
    fake_api.add_day(D0, synthetic_ticks(T0, 600))
    fake_api.throttle_next, fake_api.retry_after = 1, "2"
    slept: list[float] = []
    with ChudApi(settings, tokens, sleep=slept.append) as api:
        assert api.fetch_day(BRTI, D0, UPTO).ok
    assert 2.0 <= slept[0] <= 2.4


def test_throttle_exhausted_names_the_stream_limit(settings, tokens, fake_api):
    fake_api.add_day(D0, synthetic_ticks(T0, 600))
    fake_api.throttle_next = 99
    with ChudApi(settings, tokens, sleep=lambda _s: None, throttle_retries=2) as api:
        with pytest.raises(ApiError, match="DATA_STREAMS = 2"):
            api.fetch_day(BRTI, D0, UPTO)
    assert len([p for path, p in fake_api.requests if "/data/" in path]) == 2


def test_rejected_token_is_invalidated_and_retried_once(api, fake_api, tokens):
    fake_api.add_day(D0, synthetic_ticks(T0, 600))
    fake_api.valid_token = "tok-2"
    fetch = api.fetch_day(BRTI, D0, UPTO)
    assert fetch.ok and fetch.attempts == 2 and tokens.invalidated == 1
    assert fake_api.auth_seen == ["Bearer tok-1", "Bearer tok-2"]

    fake_api.valid_token = "never"
    with pytest.raises(ApiError, match="401"):
        api.fetch_day(BRTI, D0, UPTO)
    assert tokens.invalidated == 2       # one invalidate, one retry, then give up


def test_token_is_fetched_per_connection(api, fake_api, tokens):
    """The client caches no token: whatever the provider returns at open time is what is sent."""
    fake_api.add_day(D0, synthetic_ticks(T0, 60))
    fake_api.add_day(date(2025, 9, 19), synthetic_ticks(datetime(2025, 9, 19), 60))
    api.fetch_day(BRTI, D0, UPTO)
    tokens.invalidate()
    api.fetch_day(BRTI, date(2025, 9, 19), datetime(2025, 9, 20))
    assert fake_api.auth_seen == ["Bearer tok-1", "Bearer tok-2"] and tokens.calls == 2


def test_unknown_alias_is_fatal(api, fake_api):
    fake_api.add_day(D0, synthetic_ticks(T0, 600))
    with pytest.raises(ApiError, match="404"):
        api.fetch_day(replace(BRTI, alias="nope"), D0, UPTO)
    assert len([p for path, p in fake_api.requests if "/data/" in path]) == 1      # no retry


def test_inverted_range_is_fatal(api, fake_api):
    """`fetch_day` short-circuits an empty range, so only a hand-built request reaches the 400."""
    fake_api.add_day(D0, synthetic_ticks(T0, 60))
    with pytest.raises(ApiError, match="400"):
        api._stream_day("/btc-15m/data/index-hist", {"start": ts_param(UPTO), "end": ts_param(T0)}, BRTI)
    assert len([p for path, p in fake_api.requests if "/data/" in path]) == 1


def test_meta_mismatch_is_fatal(api, fake_api):
    fake_api.add_day(D0, synthetic_ticks(T0, 600))
    with pytest.raises(ApiError, match="expects"):
        api.fetch_day(replace(BRTI, key="ETHUSD_RR"), D0, UPTO)
    with pytest.raises(ApiError, match="no longer exposes price"):
        api.fetch_day(replace(BRTI, columns=("price",)), D0, UPTO)
    assert len([p for path, p in fake_api.requests if "/data/" in path]) == 2      # neither was retried


def test_cannot_reach_the_api(tokens, tmp_path):
    settings = Settings(api_base="http://127.0.0.1:1", data_dir=tmp_path)
    with ChudApi(settings, tokens, sleep=lambda _s: None) as api:
        with pytest.raises(ApiError, match="cannot reach http://127.0.0.1:1"):
            api.fetch_day(BRTI, D0, UPTO)
        with pytest.raises(ApiError, match="cannot reach http://127.0.0.1:1"):
            api.health()
