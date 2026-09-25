"""Incremental, idempotent download of one chud-money API table into a Parquet file per UTC day.

Works for any `api.TableSpec` (BRTI ticks, Kalshi contract candles). Correctness rules, given that
the API exposes no per-day row counts:
1. Snapshot the table's `max_ts` first (from `GET /{tag}`, up to 60 s stale); every request is
   bounded by `ts < snapshot_max_ts`, so a concurrent backfill cannot change what one run sees.
2. Candidate days are the closed range `first_ts.date() .. (snapshot - 1 µs).date()`, clipped by
   `--start/--end`. A day with no rows is requested once and recorded with `rows=0` and no file.
3. A day is verified by its own stream: a terminal `done` event whose `rows` equals the rows parsed
   (`DayFetch.ok`). That is *transport integrity only* - it proves nothing about what QuestDB holds,
   so a server-side rewrite of a day already marked complete is **not** detected. `--force` is the
   escape hatch. The check is count-agnostic, so the 1 Hz -> 5 Hz density change needs no handling.
4. Re-fetch: days missing from the manifest, any day not `complete`, the tail day (the one holding
   the snapshot), a lost Parquet file for a day known to be non-empty, and everything under `--force`.
5. Parquet files are written atomically (tmp + os.replace) and never deleted; the manifest is saved
   after every day. A day the API now reports as empty keeps its existing file, with a warning.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from .api import BRTI, ChudApi, TableSpec

log = logging.getLogger(__name__)

MANIFEST_NAME = "_manifest.json"
MANIFEST_VERSION = 2


def day_path(out_dir: Path, day: date) -> Path:
    return out_dir / f"date={day.isoformat()}.parquet"


@dataclass
class DayRecord:
    day: str
    rows: int
    api_rows: int | None = None      # the server's own `done.rows`; None when the stream never finished
    min_ts: str | None = None
    max_ts: str | None = None
    complete: bool = False
    snapshot_ts: str = ""
    downloaded_at: str = ""
    bytes: int = 0


@dataclass
class Manifest:
    version: int = MANIFEST_VERSION
    table: str = BRTI.table
    alias: str = BRTI.alias
    tag: str = ""
    days: dict[str, DayRecord] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Reads v1 (QuestDB era) too: `qdb_count` becomes `api_rows`, `filter_sql`/`index_id` are
        dropped, unknown keys are ignored and `complete` is kept, so existing files are not refetched."""
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        known = {f.name for f in fields(DayRecord)}
        days: dict[str, DayRecord] = {}
        for key, value in (raw.get("days") or {}).items():
            rec = dict(value)
            if "api_rows" not in rec and "qdb_count" in rec:
                rec["api_rows"] = rec["qdb_count"]
            days[key] = DayRecord(**{k: v for k, v in rec.items() if k in known})
        return cls(
            version=int(raw.get("version", 1)),
            table=raw.get("table", BRTI.table),
            alias=raw.get("alias", ""),
            tag=raw.get("tag", ""),
            days=days,
        )

    def save(self, path: Path) -> None:
        self.version = MANIFEST_VERSION
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "table": self.table,
            "alias": self.alias,
            "tag": self.tag,
            "days": {k: asdict(v) for k, v in sorted(self.days.items())},
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, path)

    def complete_days(self) -> list[date]:
        return sorted(date.fromisoformat(k) for k, v in self.days.items() if v.complete)


@dataclass
class DownloadReport:
    source: str
    snapshot_max_ts: datetime | None
    planned: list[date]
    fetched: list[date]
    up_to_date: int
    incomplete: list[date]
    empty: list[date]
    failed: dict[str, str]

    def summary(self) -> str:
        return (
            f"[{self.source}] {len(self.fetched)} days fetched, {self.up_to_date} up to date, "
            f"{len(self.empty)} empty, {len(self.incomplete)} incomplete, {len(self.failed)} failed "
            f"(snapshot max ts {self.snapshot_max_ts})"
        )


def candidate_days(
    first_ts: datetime | None,
    snapshot_max_ts: datetime,
    start: date | None = None,
    end: date | None = None,
) -> list[date]:
    """Every UTC day the table can hold rows for: `first_ts` .. the snapshot day, clipped."""
    if first_ts is None:
        return []
    lo = max(first_ts.date(), start) if start else first_ts.date()
    hi = (snapshot_max_ts - timedelta(microseconds=1)).date()
    if end and end < hi:
        hi = end
    out: list[date] = []
    day = lo
    while day <= hi:
        out.append(day)
        day += timedelta(days=1)
    return out


def plan_days(
    manifest: Manifest,
    days: list[date],
    snapshot_max_ts: datetime,
    force: bool = False,
    out_dir: Path | None = None,
) -> list[date]:
    """Which of `days` need (re)fetching; see rule 4 in the module docstring."""
    tail = (snapshot_max_ts - timedelta(microseconds=1)).date()
    planned: list[date] = []
    for day in days:
        rec = manifest.days.get(day.isoformat())
        file_lost = out_dir is not None and rec is not None and rec.rows > 0 and not day_path(out_dir, day).exists()
        if force or rec is None or not rec.complete or file_lost or day >= tail:
            planned.append(day)
    return planned


def download_day(api: ChudApi, spec: TableSpec, day: date, snapshot_max_ts: datetime, out_dir: Path) -> DayRecord:
    """Fetch and write one day. A 0-row day writes no file (and never deletes one)."""
    day_end = datetime(day.year, day.month, day.day) + timedelta(days=1)
    fetch = api.fetch_day(spec, day, snapshot_max_ts)
    df, rows = fetch.df, fetch.df.height
    path = day_path(out_dir, day)
    if rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.write_parquet(tmp, compression="zstd", compression_level=3, statistics=True)
        os.replace(tmp, path)
    elif path.exists():
        log.warning("[%s] %s: the API returned no rows but %s exists; keeping it (use --force to investigate)",
                    spec.name, day, path.name)
    if not fetch.ok:
        log.warning("[%s] %s: %s (%d attempts)", spec.name, day, fetch.note, fetch.attempts)
    sealed = day_end <= snapshot_max_ts
    return DayRecord(
        day=day.isoformat(),
        rows=rows,
        api_rows=fetch.reported_rows,
        min_ts=df["ts"].min().isoformat() if rows else None,
        max_ts=df["ts"].max().isoformat() if rows else None,
        complete=bool(sealed and fetch.ok),
        snapshot_ts=snapshot_max_ts.isoformat(),
        downloaded_at=datetime.now(UTC).replace(tzinfo=None).isoformat(),
        bytes=path.stat().st_size if rows else 0,
    )


def download(
    api: ChudApi,
    out_dir: Path,
    spec: TableSpec = BRTI,
    *,
    start: date | None = None,
    end: date | None = None,
    jobs: int = 1,
    force: bool = False,
    verify_only: bool = False,
) -> DownloadReport:
    """One source, one directory. `jobs` defaults to 1 because the API permits 2 exports in total."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_NAME
    manifest = Manifest.load(manifest_path)
    if manifest.days:
        if manifest.table and manifest.table != spec.table:
            log.warning("[%s] %s was written from table %s, not %s", spec.name, manifest_path, manifest.table, spec.table)
        if manifest.tag and manifest.tag != api.tag:
            log.warning("[%s] %s was written from market %s, not %s", spec.name, manifest_path, manifest.tag, api.tag)
    manifest.table, manifest.alias, manifest.tag = spec.table, spec.alias, api.tag

    bounds = api.bounds(spec)
    if bounds.max_ts is None or bounds.rows == 0:
        return DownloadReport(spec.name, None, [], [], 0, [], [], {})
    # Bound every request strictly below the snapshot so a concurrent backfill cannot move it.
    snapshot = bounds.max_ts + timedelta(microseconds=1)
    days = candidate_days(bounds.min_ts, snapshot, start, end)
    planned = plan_days(manifest, days, snapshot, force, out_dir)
    up_to_date = len(days) - len(planned)
    log.info(
        "[%s] snapshot max ts %s; %d candidate days, %d planned, %d up to date",
        spec.name, bounds.max_ts, len(days), len(planned), up_to_date,
    )
    if verify_only:
        return DownloadReport(spec.name, bounds.max_ts, planned, [], up_to_date, [], [], {})

    fetched: list[date] = []
    incomplete: list[date] = []
    empty: list[date] = []
    failed: dict[str, str] = {}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(download_day, api, spec, d, snapshot, out_dir): d for d in planned}
        for fut in as_completed(futures):
            day = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001 - record and continue with other days
                log.error("[%s] %s: failed: %s", spec.name, day, e)
                failed[day.isoformat()] = str(e)
                continue
            with lock:
                manifest.days[rec.day] = rec
                manifest.save(manifest_path)
            fetched.append(day)
            if not rec.complete:
                incomplete.append(day)
            if rec.rows == 0:
                empty.append(day)
            log.info("[%s] %s: %d rows (%s) %.1f KB", spec.name, day, rec.rows,
                     "complete" if rec.complete else "partial", rec.bytes / 1024)
    return DownloadReport(spec.name, bounds.max_ts, planned, sorted(fetched), up_to_date,
                          sorted(incomplete), sorted(empty), failed)
