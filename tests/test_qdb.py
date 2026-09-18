from datetime import date, datetime, timedelta

from conftest import basic_auth_header, synthetic_ticks

from chud_predictor.qdb import QuestDB, ts_literal


def test_bounds_day_counts_and_fetch(fake_qdb, settings):
    d0 = date(2025, 9, 18)
    fake_qdb.add_day(d0, synthetic_ticks(datetime(2025, 9, 18), 86_400))
    fake_qdb.add_day(d0 + timedelta(days=1), synthetic_ticks(datetime(2025, 9, 19), 3_600))
    q = QuestDB(settings)
    assert q.ping() == "fake-1"
    b = q.bounds("index_values_hist", "BRTI")
    assert b.rows == 90_000 and b.min_ts == datetime(2025, 9, 18) and b.max_ts == datetime(2025, 9, 19, 0, 59, 59)
    counts = q.day_counts("index_values_hist", "BRTI", b.max_ts + timedelta(microseconds=1))
    assert counts == {d0: 86_400, d0 + timedelta(days=1): 3_600}
    df = q.fetch_day("index_values_hist", "BRTI", d0, b.max_ts + timedelta(microseconds=1))
    assert df.height == 86_400 and df.columns == ["ts", "value"]
    assert df["ts"][0] == datetime(2025, 9, 18) and df["ts"][-1] == datetime(2025, 9, 18, 23, 59, 59)
    assert fake_qdb.auth_seen[0] == basic_auth_header("admin", "pw")


def test_ts_literal():
    assert ts_literal(datetime(2025, 9, 18, 1, 2, 3, 4)) == "'2025-09-18T01:02:03.000004Z'"


def test_retry_on_server_error(fake_qdb, settings):
    fake_qdb.add_day(date(2025, 9, 18), synthetic_ticks(datetime(2025, 9, 18), 600))
    fake_qdb.fail_next = 1
    q = QuestDB(settings)
    df = q.fetch_day("index_values_hist", "BRTI", date(2025, 9, 18), datetime(2025, 9, 19))
    assert df.height == 600
