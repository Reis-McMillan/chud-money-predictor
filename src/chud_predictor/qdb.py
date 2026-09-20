"""Thin QuestDB HTTP client (`/exec` JSON for metadata, `/exp` CSV for bulk rows).

Two source tables, described by a `TableSpec`:
  * BRTI      index_values_hist       raw index ticks (1 Hz until ~May 2026, 5 Hz after)
  * CONTRACTS contract_candles_hist   Kalshi KXBTC15M 1-minute candles (ts = END of the minute)

Timestamps are exchanged as epoch microseconds (`cast(ts as long)`) and converted to tz-naive UTC
`Datetime("us")` on the way in.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import httpx
import polars as pl

from .settings import Settings, require_password

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TableSpec:
    name: str                        # "brti" | "contracts": raw sub-directory and manifest label
    table: str
    where: str | None                # row filter, e.g. "index_id = 'BRTI'"
    columns: tuple[str, ...]         # select list, excluding ts
    dtypes: tuple[pl.DataType, ...]  # dtype of each column

    @property
    def schema(self) -> dict[str, pl.DataType]:
        return {"ts_us": pl.Int64, **dict(zip(self.columns, self.dtypes, strict=True))}


BRTI = TableSpec("brti", "index_values_hist", "index_id = 'BRTI'", ("value",), (pl.Float64,))

CONTRACT_NUMERIC = (
    "floor_strike",
    "yes_bid_open", "yes_bid_high", "yes_bid_low", "yes_bid_close",
    "yes_ask_open", "yes_ask_high", "yes_ask_low", "yes_ask_close",
    "price_open", "price_high", "price_low", "price_close", "price_mean",
    "volume", "open_interest",
)
CONTRACTS = TableSpec(
    "contracts", "contract_candles_hist", "series_ticker = 'KXBTC15M'",
    ("ticker", *CONTRACT_NUMERIC), (pl.Utf8, *([pl.Float64] * len(CONTRACT_NUMERIC))),
)
SOURCES: dict[str, TableSpec] = {"brti": BRTI, "contracts": CONTRACTS}


class QdbError(RuntimeError):
    pass


@dataclass(frozen=True)
class Bounds:
    rows: int
    min_ts: datetime | None
    max_ts: datetime | None


def ts_literal(t: datetime) -> str:
    """QuestDB timestamp literal, microsecond precision, UTC."""
    return "'" + t.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z'"


def _parse_ts(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ")


def _where(spec: TableSpec, *extra: str) -> str:
    parts = [p for p in (spec.where, *extra) if p]
    return (" WHERE " + " AND ".join(parts)) if parts else ""


class QuestDB:
    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        retries: int = 3,
        require_auth: bool = True,
    ):
        self.settings = settings
        self.retries = retries
        password = require_password(settings) if require_auth else settings.qdb_password
        self._client = client or httpx.Client(
            base_url=settings.qdb_url,
            auth=httpx.BasicAuth(settings.qdb_user, password),
            timeout=httpx.Timeout(settings.qdb_timeout_s, connect=10.0),
        )

    # -- transport -----------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any]) -> httpx.Response:
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self._client.get(path, params=params)
            except httpx.ConnectError as e:
                raise QdbError(
                    f"cannot connect to QuestDB at {self.settings.qdb_url}: {e}. "
                    "Is the port-forward up? Run `scripts/port-forward.sh start`."
                ) from e
            except httpx.TransportError as e:
                last_exc = e
                log.warning("qdb transport error (attempt %d/%d): %s", attempt, self.retries, e)
            else:
                if resp.status_code >= 500:
                    last_exc = QdbError(f"QuestDB {resp.status_code}: {resp.text[:300]}")
                    log.warning("qdb server error (attempt %d/%d): %s", attempt, self.retries, last_exc)
                elif resp.status_code == 401 or resp.status_code == 403:
                    raise QdbError(f"QuestDB rejected the credentials ({resp.status_code})")
                else:
                    return resp
            time.sleep(delay)
            delay *= 2
        raise QdbError(f"QuestDB request failed after {self.retries} attempts: {last_exc}")

    def exec_json(self, sql: str, *, limit: int | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"query": sql}
        if limit is not None:
            params["limit"] = str(limit)
        resp = self._get("/exec", params)
        data = resp.json()
        if "error" in data:
            raise QdbError(f"{data['error']} (query: {sql[:200]})")
        return data

    def exp_csv(self, sql: str, schema: dict[str, pl.DataType]) -> pl.DataFrame:
        """Stream `/exp` CSV into a DataFrame with an explicit schema."""
        buf = io.BytesIO()
        delay = 1.0
        for attempt in range(1, self.retries + 1):
            buf.seek(0)
            buf.truncate()
            try:
                with self._client.stream("GET", "/exp", params={"query": sql}) as resp:
                    if resp.status_code >= 400:
                        body = resp.read().decode(errors="replace")[:300]
                        if resp.status_code >= 500:
                            raise httpx.TransportError(f"QuestDB {resp.status_code}: {body}")
                        raise QdbError(f"QuestDB {resp.status_code}: {body} (query: {sql[:200]})")
                    for chunk in resp.iter_bytes():
                        buf.write(chunk)
                break
            except httpx.ConnectError as e:
                raise QdbError(
                    f"cannot connect to QuestDB at {self.settings.qdb_url}: {e}. "
                    "Is the port-forward up? Run `scripts/port-forward.sh start`."
                ) from e
            except httpx.TransportError as e:
                if attempt == self.retries:
                    raise QdbError(f"QuestDB export failed after {attempt} attempts: {e}") from e
                log.warning("qdb export error (attempt %d/%d): %s", attempt, self.retries, e)
                time.sleep(delay)
                delay *= 2
        buf.seek(0)
        if buf.getbuffer().nbytes == 0:
            return pl.DataFrame(schema=schema)
        return pl.read_csv(buf, schema_overrides=schema)

    def scalar(self, sql: str) -> Any:
        data = self.exec_json(sql)
        rows = data.get("dataset") or []
        return rows[0][0] if rows and rows[0] else None

    # -- domain helpers ------------------------------------------------------------

    def ping(self) -> str:
        return str(self.scalar("SELECT build"))

    def tables(self) -> list[str]:
        data = self.exec_json("SELECT table_name FROM tables() ORDER BY table_name")
        return [r[0] for r in data.get("dataset", [])]

    def bounds(self, spec: TableSpec) -> Bounds:
        data = self.exec_json(f"SELECT count(), min(ts), max(ts) FROM {spec.table}{_where(spec)}")
        rows = data.get("dataset") or [[0, None, None]]
        n, lo, hi = rows[0]
        return Bounds(rows=int(n or 0), min_ts=_parse_ts(lo), max_ts=_parse_ts(hi))

    def day_counts(self, spec: TableSpec, upto: datetime) -> dict[date, int]:
        """Rows per UTC day with ts < upto. One query for the whole history (~365 rows)."""
        data = self.exec_json(
            f"SELECT ts, count() AS n FROM {spec.table}{_where(spec, f'ts < {ts_literal(upto)}')} "
            "SAMPLE BY 1d ALIGN TO CALENDAR"
        )
        out: dict[date, int] = {}
        for ts_s, n in data.get("dataset", []):
            out[_parse_ts(ts_s).date()] = int(n)
        return out

    def fetch_day(self, spec: TableSpec, day: date, upto: datetime) -> pl.DataFrame:
        """All rows of one UTC day (bounded by `upto`): ts Datetime[us] + spec.columns, sorted by ts."""
        day_start = datetime(day.year, day.month, day.day)
        day_end = min(day_start + timedelta(days=1), upto)
        where = _where(spec, f"ts >= {ts_literal(day_start)}", f"ts < {ts_literal(day_end)}")
        sql = f"SELECT cast(ts as long) AS ts_us, {', '.join(spec.columns)} FROM {spec.table}{where}"
        df = self.exp_csv(sql, spec.schema)
        return (
            df.with_columns(pl.from_epoch(pl.col("ts_us"), time_unit="us").alias("ts"))
            .select("ts", *spec.columns)
            .sort("ts")
        )

    def close(self) -> None:
        self._client.close()
