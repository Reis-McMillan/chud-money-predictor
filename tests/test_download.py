from datetime import date, datetime

import polars as pl
from conftest import synthetic_ticks

from chud_predictor.download import MANIFEST_NAME, Manifest, day_path, download
from chud_predictor.qdb import QuestDB


def _seed(fake_qdb, days: int, tail_seconds: int = 86_400):
    for i in range(days):
        n = 86_400 if i < days - 1 else tail_seconds
        fake_qdb.add_day(date(2025, 9, 18 + i), synthetic_ticks(datetime(2025, 9, 18 + i), n, seed=i))


def test_fresh_then_idempotent_then_tail_grows(fake_qdb, settings):
    _seed(fake_qdb, 3, tail_seconds=3_600)  # last day partial
    q = QuestDB(settings)
    out = settings.raw_dir
    rep = download(q, out, jobs=2)
    assert sorted(rep.fetched) == [date(2025, 9, 18), date(2025, 9, 19), date(2025, 9, 20)]
    assert rep.incomplete == [date(2025, 9, 20)]
    m = Manifest.load(out / MANIFEST_NAME)
    assert m.days["2025-09-18"].complete and m.days["2025-09-18"].rows == 86_400
    assert not m.days["2025-09-20"].complete
    assert pl.read_parquet(day_path(out, date(2025, 9, 19))).height == 86_400

    # idempotent: only the partial tail day is re-fetched
    rep2 = download(q, out, jobs=2)
    assert rep2.fetched == [date(2025, 9, 20)] and rep2.up_to_date == 2

    # tail grows to a full day and a new day appears -> the sealed day is now complete
    fake_qdb.add_day(date(2025, 9, 20), synthetic_ticks(datetime(2025, 9, 20), 86_400, seed=9))
    fake_qdb.add_day(date(2025, 9, 21), synthetic_ticks(datetime(2025, 9, 21), 120, seed=10))
    rep3 = download(q, out, jobs=2)
    assert rep3.fetched == [date(2025, 9, 20), date(2025, 9, 21)]
    m = Manifest.load(out / MANIFEST_NAME)
    assert m.days["2025-09-20"].complete and m.days["2025-09-20"].rows == 86_400
    assert m.complete_days() == [date(2025, 9, 18), date(2025, 9, 19), date(2025, 9, 20)]


def test_count_drift_triggers_refetch(fake_qdb, settings):
    _seed(fake_qdb, 2)
    q = QuestDB(settings)
    out = settings.raw_dir
    download(q, out)
    # an old day's row count changes (a dedup/backfill job rewrote it)
    fake_qdb.add_day(date(2025, 9, 18), fake_qdb.days[date(2025, 9, 18)].head(86_390))
    rep = download(q, out)
    assert date(2025, 9, 18) in rep.fetched
    assert Manifest.load(out / MANIFEST_NAME).days["2025-09-18"].rows == 86_390


def test_short_response_marked_incomplete(fake_qdb, settings):
    _seed(fake_qdb, 2)
    fake_qdb.truncate_day = date(2025, 9, 18)
    q = QuestDB(settings)
    out = settings.raw_dir
    rep = download(q, out)
    assert date(2025, 9, 18) in rep.incomplete
    m = Manifest.load(out / MANIFEST_NAME)
    assert m.days["2025-09-18"].rows == 86_300 and not m.days["2025-09-18"].complete
    assert not list(out.glob("*.tmp"))
    # planned again next time
    rep2 = download(q, out, verify_only=True)
    assert date(2025, 9, 18) in rep2.planned


def test_date_range_filter(fake_qdb, settings):
    _seed(fake_qdb, 3)
    q = QuestDB(settings)
    rep = download(q, settings.raw_dir, start=date(2025, 9, 19), end=date(2025, 9, 19))
    assert rep.fetched == [date(2025, 9, 19)]
