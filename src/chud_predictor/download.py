"""Incremental, idempotent download of raw 1-second ticks into one Parquet file per UTC day.

Correctness rules:
1. Snapshot `max_ts` first; every query is bounded by `ts < snapshot_max_ts`, so the backfill
   that is running concurrently cannot change what a single run sees.
2. Re-fetch: days missing from the manifest, the (partial) day containing the snapshot, and any
   day whose QuestDB row count no longer matches the manifest.
3. Verify each day's row count against `SAMPLE BY 1d`; a mismatch is retried once, then recorded
   as incomplete and re-planned on the next run.
4. Parquet files are written atomically (tmp + os.replace); the manifest is saved after every day.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from .qdb import QuestDB

log = logging.getLogger(__name__)

MANIFEST_NAME = "_manifest.json"
DEFAULT_TABLE = "index_values_hist"
DEFAULT_INDEX = "BRTI"


def day_path(out_dir: Path, day: date) -> Path:
    return out_dir / f"date={day.isoformat()}.parquet"


@dataclass
class DayRecord:
    day: str
    rows: int
    qdb_count: int
    min_ts: str | None
    max_ts: str | None
    complete: bool
    snapshot_ts: str
    downloaded_at: str
    bytes: int


@dataclass
class Manifest:
    table: str = DEFAULT_TABLE
    index_id: str = DEFAULT_INDEX
    days: dict[str, DayRecord] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        days = {k: DayRecord(**v) for k, v in raw.get("days", {}).items()}
        return cls(table=raw.get("table", DEFAULT_TABLE), index_id=raw.get("index_id", DEFAULT_INDEX), days=days)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "table": self.table,
            "index_id": self.index_id,
            "days": {k: asdict(v) for k, v in sorted(self.days.items())},
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, path)

    def complete_days(self) -> list[date]:
        return sorted(date.fromisoformat(k) for k, v in self.days.items() if v.complete)


@dataclass
class DownloadReport:
    snapshot_max_ts: datetime | None
    planned: list[date]
    fetched: list[date]
    up_to_date: int
    incomplete: list[date]
    failed: dict[str, str]

    def summary(self) -> str:
        return (
            f"{len(self.fetched)} days fetched, {self.up_to_date} up to date, "
            f"{len(self.incomplete)} incomplete, {len(self.failed)} failed "
            f"(snapshot max ts {self.snapshot_max_ts})"
        )


def plan_days(
    manifest: Manifest,
    day_counts: dict[date, int],
    snapshot_max_ts: datetime,
    start: date | None = None,
    end: date | None = None,
    force: bool = False,
    out_dir: Path | None = None,
) -> list[date]:
    """Days that need (re)fetching. `day_counts` covers ts < snapshot_max_ts."""
    snapshot_day = snapshot_max_ts.date()
    planned: list[date] = []
    for day, n in sorted(day_counts.items()):
        if n <= 0:
            continue
        if start and day < start:
            continue
        if end and day > end:
            continue
        rec = manifest.days.get(day.isoformat())
        file_missing = out_dir is not None and not day_path(out_dir, day).exists()
        if (
            force
            or rec is None
            or file_missing
            or not rec.complete
            or rec.qdb_count != n
            or day == snapshot_day
        ):
            planned.append(day)
    return planned


def download_day(
    qdb: QuestDB,
    day: date,
    snapshot_max_ts: datetime,
    expected: int,
    out_dir: Path,
    table: str = DEFAULT_TABLE,
    index_id: str = DEFAULT_INDEX,
) -> DayRecord:
    day_start = datetime(day.year, day.month, day.day)
    day_end = day_start + timedelta(days=1)
    df: pl.DataFrame | None = None
    for attempt in (1, 2):
        df = qdb.fetch_day(table, index_id, day, snapshot_max_ts)
        if len(df) == expected:
            break
        log.warning(
            "%s: got %d rows, QuestDB reports %d (attempt %d)", day, len(df), expected, attempt
        )
    assert df is not None
    path = day_path(out_dir, day)
    tmp = path.with_suffix(".parquet.tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(tmp, compression="zstd", compression_level=3, statistics=True)
    os.replace(tmp, path)
    sealed = day_end <= snapshot_max_ts
    rows = len(df)
    return DayRecord(
        day=day.isoformat(),
        rows=rows,
        qdb_count=expected,
        min_ts=df["ts"].min().isoformat() if rows else None,
        max_ts=df["ts"].max().isoformat() if rows else None,
        complete=bool(sealed and rows == expected),
        snapshot_ts=snapshot_max_ts.isoformat(),
        downloaded_at=datetime.now(UTC).replace(tzinfo=None).isoformat(),
        bytes=path.stat().st_size,
    )


def download(
    qdb: QuestDB,
    out_dir: Path,
    *,
    table: str = DEFAULT_TABLE,
    index_id: str = DEFAULT_INDEX,
    start: date | None = None,
    end: date | None = None,
    jobs: int = 4,
    force: bool = False,
    verify_only: bool = False,
) -> DownloadReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_NAME
    manifest = Manifest.load(manifest_path)
    manifest.table, manifest.index_id = table, index_id

    bounds = qdb.bounds(table, index_id)
    if bounds.max_ts is None or bounds.rows == 0:
        return DownloadReport(None, [], [], 0, [], {})
    # Bound every query strictly below the snapshot so the concurrent backfill cannot move it.
    snapshot = bounds.max_ts + timedelta(microseconds=1)
    counts = qdb.day_counts(table, index_id, snapshot)
    planned = plan_days(manifest, counts, snapshot, start, end, force, out_dir)
    in_range = [d for d in counts if (not start or d >= start) and (not end or d <= end) and counts[d] > 0]
    up_to_date = len(in_range) - len(planned)
    log.info(
        "snapshot max ts %s; %d days in QuestDB, %d planned, %d up to date",
        bounds.max_ts, len(in_range), len(planned), up_to_date,
    )
    if verify_only:
        return DownloadReport(bounds.max_ts, planned, [], up_to_date, [], {})

    fetched: list[date] = []
    incomplete: list[date] = []
    failed: dict[str, str] = {}
    lock = threading.Lock()

    def work(day: date) -> DayRecord:
        return download_day(qdb, day, snapshot, counts[day], out_dir, table, index_id)

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(work, d): d for d in planned}
        for fut in as_completed(futures):
            day = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001 - record and continue with other days
                log.error("%s: failed: %s", day, e)
                failed[day.isoformat()] = str(e)
                continue
            with lock:
                manifest.days[rec.day] = rec
                manifest.save(manifest_path)
            fetched.append(day)
            if not rec.complete:
                incomplete.append(day)
            log.info(
                "%s: %d rows (%s) %.1f KB", day, rec.rows, "complete" if rec.complete else "partial", rec.bytes / 1024
            )
    return DownloadReport(bounds.max_ts, planned, sorted(fetched), up_to_date, sorted(incomplete), failed)
