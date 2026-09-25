import json
import time
from datetime import date, datetime, timedelta

import polars as pl
from conftest import CONTRACT_TABLE, synthetic_candles, synthetic_ticks

from chud_predictor.api import BRTI, CONTRACTS, ChudApi
from chud_predictor.download import MANIFEST_NAME, Manifest, candidate_days, day_path, download


def _seed(fake_api, days: int, seconds: int = 86_400, tail_seconds: int | None = None) -> None:
    for i in range(days):
        n = seconds if i < days - 1 else (seconds if tail_seconds is None else tail_seconds)
        fake_api.add_day(date(2025, 9, 18 + i), synthetic_ticks(datetime(2025, 9, 18 + i), n, seed=i))


def test_candidate_days_spans_the_whole_range():
    snapshot = datetime(2025, 9, 20, 1, 0)
    assert candidate_days(datetime(2025, 9, 18, 23), snapshot) == [date(2025, 9, 18), date(2025, 9, 19), date(2025, 9, 20)]
    assert candidate_days(datetime(2025, 9, 18), snapshot, start=date(2025, 9, 19)) == [date(2025, 9, 19), date(2025, 9, 20)]
    assert candidate_days(datetime(2025, 9, 18), snapshot, end=date(2025, 9, 18)) == [date(2025, 9, 18)]
    assert candidate_days(datetime(2025, 9, 18), datetime(2025, 9, 19)) == [date(2025, 9, 18)]   # snapshot = exclusive
    assert candidate_days(None, snapshot) == []


def test_fresh_then_idempotent_then_tail_grows(api, fake_api, settings):
    _seed(fake_api, 3, tail_seconds=3_600)  # last day partial
    out = settings.raw_dir
    rep = download(api, out, jobs=2)
    assert sorted(rep.fetched) == [date(2025, 9, 18), date(2025, 9, 19), date(2025, 9, 20)]
    assert rep.incomplete == [date(2025, 9, 20)] and rep.empty == [] and rep.failed == {}
    m = Manifest.load(out / MANIFEST_NAME)
    assert m.days["2025-09-18"].complete and m.days["2025-09-18"].rows == 86_400 == m.days["2025-09-18"].api_rows
    assert not m.days["2025-09-20"].complete
    assert pl.read_parquet(day_path(out, date(2025, 9, 19))).height == 86_400

    # idempotent: only the partial tail day is re-fetched
    rep2 = download(api, out, jobs=2)
    assert rep2.fetched == [date(2025, 9, 20)] and rep2.up_to_date == 2

    # tail grows to a full day and a new day appears -> the sealed day is now complete
    fake_api.add_day(date(2025, 9, 20), synthetic_ticks(datetime(2025, 9, 20), 86_400, seed=9))
    fake_api.add_day(date(2025, 9, 21), synthetic_ticks(datetime(2025, 9, 21), 120, seed=10))
    rep3 = download(api, out, jobs=2)
    assert rep3.fetched == [date(2025, 9, 20), date(2025, 9, 21)]
    m = Manifest.load(out / MANIFEST_NAME)
    assert m.days["2025-09-20"].complete and m.days["2025-09-20"].rows == 86_400
    assert m.complete_days() == [date(2025, 9, 18), date(2025, 9, 19), date(2025, 9, 20)]
    assert not list(out.glob("*.tmp"))


def test_date_range_filter(api, fake_api, settings):
    _seed(fake_api, 3, seconds=600)
    rep = download(api, settings.raw_dir, start=date(2025, 9, 19), end=date(2025, 9, 19))
    assert rep.fetched == [date(2025, 9, 19)] and rep.planned == [date(2025, 9, 19)]


def test_sub_second_days_verify_like_any_other(api, fake_api, settings):
    fake_api.add_day(date(2026, 6, 1), synthetic_ticks(datetime(2026, 6, 1), 3_600, hz=5))
    fake_api.add_day(date(2026, 6, 2), synthetic_ticks(datetime(2026, 6, 2), 60, hz=5))
    rep = download(api, settings.raw_dir)
    m = Manifest.load(settings.raw_dir / MANIFEST_NAME)
    assert rep.failed == {} and m.days["2026-06-01"].rows == 18_000 and m.days["2026-06-01"].complete


def test_two_sources_have_separate_dirs_and_manifests(api, fake_api, settings):
    ticks = synthetic_ticks(datetime(2025, 12, 20), 86_400 + 7_200, seed=4)
    fake_api.add_rows(ticks, "index_values_hist")
    candles = synthetic_candles(ticks, seed=5)
    fake_api.add_rows(candles, CONTRACT_TABLE)
    rb = download(api, settings.raw_dir_for("brti"), BRTI)
    rc = download(api, settings.raw_dir_for("contracts"), CONTRACTS)
    assert rb.source == "brti" and rc.source == "contracts" and settings.raw_dir != settings.contracts_raw_dir
    got = pl.concat([pl.read_parquet(p) for p in sorted(settings.contracts_raw_dir.glob("date=*.parquet"))])
    assert got.equals(candles.select(got.columns))
    mc = Manifest.load(settings.contracts_raw_dir / MANIFEST_NAME)
    assert mc.version == 2 and mc.table == CONTRACT_TABLE and mc.alias == "candles" and mc.tag == "btc-15m"
    assert download(api, settings.contracts_raw_dir, CONTRACTS).fetched == [date(2025, 12, 21)]   # only the tail day


def test_short_response_marked_incomplete(api, fake_api, settings):
    _seed(fake_api, 2, seconds=3_600)
    fake_api.truncate_reported = date(2025, 9, 18)      # 100 rows lost, `done.rows` says otherwise
    out = settings.raw_dir
    rep = download(api, out)
    assert date(2025, 9, 18) in rep.incomplete
    m = Manifest.load(out / MANIFEST_NAME)
    rec = m.days["2025-09-18"]
    assert rec.rows == 3_500 and rec.api_rows == 3_600 and not rec.complete
    assert pl.read_parquet(day_path(out, date(2025, 9, 18))).height == 3_500
    assert not list(out.glob("*.tmp"))
    # planned again next time
    assert date(2025, 9, 18) in download(api, out, verify_only=True).planned


def test_sealed_day_rewrite_is_not_detected(api, fake_api, settings):
    """No per-day counts: a server-side rewrite of a sealed day is invisible until --force."""
    _seed(fake_api, 2, seconds=3_600)
    out = settings.raw_dir
    download(api, out)
    fake_api.add_day(date(2025, 9, 18), fake_api.days[date(2025, 9, 18)].head(3_590))
    rep = download(api, out)
    assert date(2025, 9, 18) not in rep.fetched and rep.fetched == [date(2025, 9, 19)]
    assert Manifest.load(out / MANIFEST_NAME).days["2025-09-18"].rows == 3_600

    rep2 = download(api, out, force=True)
    assert rep2.fetched == [date(2025, 9, 18), date(2025, 9, 19)]
    assert Manifest.load(out / MANIFEST_NAME).days["2025-09-18"].rows == 3_590
    assert pl.read_parquet(day_path(out, date(2025, 9, 18))).height == 3_590


def test_empty_day_is_recorded_without_a_file(api, fake_api, settings):
    fake_api.add_day(date(2025, 9, 18), synthetic_ticks(datetime(2025, 9, 18), 600))
    fake_api.add_day(date(2025, 9, 20), synthetic_ticks(datetime(2025, 9, 20), 600, seed=2))
    out = settings.raw_dir
    rep = download(api, out)
    assert rep.empty == [date(2025, 9, 19)] and rep.incomplete == [date(2025, 9, 20)]   # 09-20 holds the snapshot
    rec = Manifest.load(out / MANIFEST_NAME).days["2025-09-19"]
    assert rec.rows == 0 and rec.api_rows == 0 and rec.complete and rec.bytes == 0
    assert not day_path(out, date(2025, 9, 19)).exists()
    # a recorded empty day is not requested again
    assert download(api, out, verify_only=True).planned == [date(2025, 9, 20)]


def test_lost_file_is_refetched(api, fake_api, settings):
    _seed(fake_api, 3, seconds=600)
    out = settings.raw_dir
    download(api, out)
    day_path(out, date(2025, 9, 18)).unlink()
    rep = download(api, out, verify_only=True)
    assert rep.planned == [date(2025, 9, 18), date(2025, 9, 20)]      # the lost day and the tail
    assert download(api, out).fetched == [date(2025, 9, 18), date(2025, 9, 20)]
    assert day_path(out, date(2025, 9, 18)).exists()


def test_existing_file_survives_a_zero_row_answer(api, fake_api, settings):
    """A file is never deleted because the API now reports nothing for that day, only warned about."""
    _seed(fake_api, 3, seconds=600)
    out = settings.raw_dir
    download(api, out)
    path = day_path(out, date(2025, 9, 19))
    fake_api.tables["index_values_hist"].pop(date(2025, 9, 19))
    rep = download(api, out, force=True)
    assert rep.empty == [date(2025, 9, 19)] and path.exists() and pl.read_parquet(path).height == 600
    rec = Manifest.load(out / MANIFEST_NAME).days["2025-09-19"]
    assert rec.rows == 0 and rec.complete and rec.bytes == 0


def test_mid_stream_error_writes_the_prefix_and_stays_incomplete(api, fake_api, settings):
    _seed(fake_api, 2, seconds=600)
    fake_api.error_next = 99
    out = settings.raw_dir
    rep = download(api, out)
    assert rep.incomplete == [date(2025, 9, 18), date(2025, 9, 19)] and rep.failed == {}
    rec = Manifest.load(out / MANIFEST_NAME).days["2025-09-18"]
    assert rec.rows == 300 and rec.api_rows is None and not rec.complete
    assert pl.read_parquet(day_path(out, date(2025, 9, 18))).height == 300
    assert not list(out.glob("*.tmp"))
    assert date(2025, 9, 18) in download(api, out, verify_only=True).planned


def test_hard_failure_is_reported_without_a_file(api, fake_api, settings):
    _seed(fake_api, 2, seconds=600)
    fake_api.fail_next = 99
    out = settings.raw_dir
    rep = download(api, out)
    assert sorted(rep.failed) == ["2025-09-18", "2025-09-19"] and rep.fetched == []
    assert "500" in rep.failed["2025-09-18"] and not list(out.glob("date=*.parquet"))
    assert Manifest.load(out / MANIFEST_NAME).days == {}


def test_two_jobs_survive_the_stream_limit(settings, tokens, fake_api):
    _seed(fake_api, 4, seconds=600)
    fake_api.max_concurrent = 1            # the API allows 2 process-wide; assume the SPA holds one
    with ChudApi(settings, tokens, sleep=lambda _s: time.sleep(0.02), throttle_retries=20) as api:
        rep = download(api, settings.raw_dir, jobs=2)
    assert rep.failed == {} and len(rep.fetched) == 4 and fake_api.peak == 1


def test_v1_manifest_migrates_without_refetching(api, fake_api, settings):
    _seed(fake_api, 2, seconds=600)
    out = settings.raw_dir
    download(api, out)
    path = out / MANIFEST_NAME
    v2 = json.loads(path.read_text())
    path.write_text(json.dumps({
        "table": v2["table"],
        "filter_sql": "index_id = 'BRTI'",
        "days": {k: {**{kk: vv for kk, vv in v.items() if kk != "api_rows"}, "qdb_count": v["api_rows"]}
                 for k, v in v2["days"].items()},
    }))
    m = Manifest.load(path)
    assert m.version == 1 and m.tag == "" and m.days["2025-09-18"].api_rows == 600 and m.days["2025-09-18"].complete

    rep = download(api, out)
    assert rep.fetched == [date(2025, 9, 19)] and rep.up_to_date == 1      # the sealed day is not refetched
    m = Manifest.load(path)
    assert m.version == 2 and m.tag == "btc-15m" and m.alias == "index-hist"


def test_old_manifest_format_still_loads(tmp_path):
    p = tmp_path / MANIFEST_NAME
    p.write_text(json.dumps({
        "table": "index_values_hist", "index_id": "BRTI",
        "days": {"2025-09-18": {"day": "2025-09-18", "rows": 5, "qdb_count": 5, "min_ts": None, "max_ts": None,
                                "complete": True, "snapshot_ts": "", "downloaded_at": "", "bytes": 1, "legacy": 7}},
    }))
    m = Manifest.load(p)
    assert m.version == 1 and m.table == "index_values_hist" and m.alias == "" and m.tag == ""
    assert m.days["2025-09-18"].api_rows == 5 and m.days["2025-09-18"].complete


def test_no_rows_at_all(api, fake_api, settings):
    rep = download(api, settings.raw_dir, jobs=1)
    assert rep.planned == [] and rep.snapshot_max_ts is None and "0 days fetched" in rep.summary()


def test_report_summary_mentions_empty_days(api, fake_api, settings):
    fake_api.add_day(date(2025, 9, 18), synthetic_ticks(datetime(2025, 9, 18), 60))
    fake_api.add_day(date(2025, 9, 20), synthetic_ticks(datetime(2025, 9, 20), 60, seed=1))
    rep = download(api, settings.raw_dir)
    assert "1 empty" in rep.summary() and str(fake_api.all_rows("index_values_hist")["ts"].max()) in rep.summary()


def test_snapshot_bounds_the_tail_day(api, fake_api, settings):
    """Rows written while a run is in flight are not picked up: every request is < the snapshot."""
    fake_api.add_day(date(2025, 9, 18), synthetic_ticks(datetime(2025, 9, 18), 600))
    bounds = api.bounds(BRTI)
    snapshot = bounds.max_ts + timedelta(microseconds=1)
    assert candidate_days(bounds.min_ts, snapshot) == [date(2025, 9, 18)]
    fetch = api.fetch_day(BRTI, date(2025, 9, 18), snapshot)
    assert fetch.df["ts"].max() == bounds.max_ts
