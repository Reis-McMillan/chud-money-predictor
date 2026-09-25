"""Client for the chud-money HTTP API (`GET /{tag}/data/{alias}`, Server-Sent Events).

Two source tables, described by a `TableSpec`:
  * BRTI      index_values_hist      alias `index-hist`, key `index_id = 'BRTI'`
  * CONTRACTS contract_candles_hist  alias `candles`, key `series_ticker = 'KXBTC15M'`

The key filter is applied server-side from the market document; `meta.key` is asserted against the
spec so a mis-tagged market can never contaminate a local file. Rows arrive as one JSON object per
`row` event with `ts` as RFC3339 text and doubles as JSON numbers or `null` (SQL NULL *and* NaN),
and are parsed into the same tz-naive `Datetime("us")` frames the QuestDB-era client produced.

Bounds come from `GET /{tag}` (`questdb.tables`, key-filtered, refreshed server-side every 60 s);
there are no per-day counts, so a day's integrity is `done.rows == rows parsed` (`DayFetch.ok`).
The API allows `DATA_STREAMS = 2` concurrent exports process-wide and 429s beyond that.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, NoReturn

import httpx
import polars as pl

from .settings import Settings

if TYPE_CHECKING:                       # `auth` pulls in the whole Verys flow; the client only needs the protocol
    from .auth import TokenProvider

log = logging.getLogger(__name__)

#: Server-side cap on concurrent exports (`DATA_STREAMS` in chud-money).
DATA_STREAMS = 2
#: 429 backoff, seconds, before ±20 % jitter; the last value repeats.
THROTTLE_BACKOFF = (5.0, 10.0, 20.0, 40.0, 60.0, 60.0)


@dataclass(frozen=True)
class TableSpec:
    name: str                        # "brti" | "contracts": raw sub-directory and manifest label
    table: str                       # QuestDB table name (manifest + summary lookup)
    alias: str                       # URL segment
    key_column: str                  # symbol column the API filters on
    key: str                         # its value, asserted against `meta.key`
    columns: tuple[str, ...]         # kept columns, excluding ts
    dtypes: tuple[pl.DataType, ...]  # dtype of each column

    @property
    def read_schema(self) -> dict[str, pl.DataType]:
        """NDJSON projection: `ts` as text, every other API column dropped."""
        return {"ts": pl.Utf8, **dict(zip(self.columns, self.dtypes, strict=True))}

    @property
    def schema(self) -> dict[str, pl.DataType]:
        """Parquet schema (unchanged since the QuestDB client)."""
        return {"ts": pl.Datetime("us"), **dict(zip(self.columns, self.dtypes, strict=True))}


BRTI = TableSpec("brti", "index_values_hist", "index-hist", "index_id", "BRTI", ("value",), (pl.Float64,))

CONTRACT_NUMERIC = (
    "floor_strike",
    "yes_bid_open", "yes_bid_high", "yes_bid_low", "yes_bid_close",
    "yes_ask_open", "yes_ask_high", "yes_ask_low", "yes_ask_close",
    "price_open", "price_high", "price_low", "price_close", "price_mean",
    "volume", "open_interest",
)
CONTRACTS = TableSpec(
    "contracts", "contract_candles_hist", "candles", "series_ticker", "KXBTC15M",
    ("ticker", *CONTRACT_NUMERIC), (pl.Utf8, *([pl.Float64] * len(CONTRACT_NUMERIC))),
)
SOURCES: dict[str, TableSpec] = {"brti": BRTI, "contracts": CONTRACTS}


class ApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class Bounds:
    rows: int
    min_ts: datetime | None
    max_ts: datetime | None


@dataclass(frozen=True)
class TableSummary:
    """One entry of `GET /{tag}` → `questdb.tables`, already key-filtered by the server."""
    table: str
    rows: int
    first_ts: datetime | None
    last_ts: datetime | None


def parse_api_ts(s: str | None) -> datetime | None:
    """RFC3339 (`…Z`, any fractional precision) -> tz-naive UTC datetime."""
    if s is None:
        return None
    t = s.strip()
    if t.endswith(("Z", "z")):
        head, _, frac = t[:-1].partition(".")
        return datetime.fromisoformat(head + (f".{frac[:6]}" if frac else ""))
    dt = datetime.fromisoformat(t)
    return dt if dt.tzinfo is None else dt.astimezone(UTC).replace(tzinfo=None)


def ts_param(t: datetime) -> str:
    """A tz-naive UTC datetime as the `start`/`end` query parameter."""
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


# -- SSE -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SseEvent:
    event: str
    data: str
    id: str | None = None
    retry: int | None = None


def iter_sse(lines: Iterable[str]) -> Iterator[SseEvent]:
    """Parse `resp.iter_lines()` into events: a blank line dispatches, `:` comments (the 15 s
    keep-alives) are ignored, repeated `data:` fields join with `\\n`, a missing `event:` is
    `message`, and a trailing event with no blank line after it is dropped."""
    event: str | None = None
    data: list[str] = []
    ident: str | None = None
    retry: int | None = None
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if event is not None or data:
                yield SseEvent(event or "message", "\n".join(data), ident, retry)
            event, data, ident, retry = None, [], None, None
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
        elif field == "id":
            ident = value
        elif field == "retry":
            retry = int(value) if value.isdigit() else None


# -- one day ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DayFetch:
    """One day's rows plus the server's own count. `ok` is transport integrity, not a promise that
    QuestDB holds nothing more for that day (there are no per-day counts to compare against)."""
    df: pl.DataFrame
    reported_rows: int | None
    attempts: int
    note: str | None = None

    @property
    def ok(self) -> bool:
        return self.reported_rows is not None and self.reported_rows == self.df.height


def rows_to_frame(rows: list[str], spec: TableSpec) -> pl.DataFrame:
    """`row` event bodies -> `ts` + `spec.columns`, sorted by ts. The schema projects, so the API's
    `index_id`/`source`/`received_at`/`series_ticker` never reach the parquet file, and `null`
    (SQL NULL or NaN on the wire) stays a polars null."""
    if not rows:
        return pl.DataFrame(schema=spec.schema)
    df = pl.read_ndjson("\n".join(rows).encode(), schema=spec.read_schema)
    return (
        df.with_columns(pl.col("ts").str.to_datetime("%Y-%m-%dT%H:%M:%S%.fZ", time_unit="us"))
        .select("ts", *spec.columns)
        .sort("ts")
    )


class _Throttled(Exception):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class _Rejected(Exception):
    """401/403: the bearer token was refused."""


class _Transient(Exception):
    """5xx or a transport error: worth another attempt."""


def _throttle_delay(n: int, retry_after: float | None) -> float:
    """`n`-th 429 (1-based): `Retry-After` if the API ever sends one, else the fixed ladder, ±20 %."""
    if retry_after is not None:
        return retry_after * random.uniform(1.0, 1.2)
    return THROTTLE_BACKOFF[min(n, len(THROTTLE_BACKOFF)) - 1] * random.uniform(0.8, 1.2)


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("Retry-After")
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


def _error_body(resp: httpx.Response) -> str:
    """The `{"error": …}` message the API returns for every 4xx/5xx, else raw text."""
    try:
        body = resp.read().decode(errors="replace")
    except Exception:  # noqa: BLE001 - a broken body must not mask the status code
        return ""
    try:
        parsed = json.loads(body)
    except ValueError:
        return body.strip()[:300]
    return str(parsed.get("error", parsed))[:300] if isinstance(parsed, dict) else body.strip()[:300]


class ChudApi:
    """The single data seam: public metadata plus one authenticated SSE export per UTC day.

    `tokens.token()` is called once per connection open and never cached here, so a token that
    expires mid-backfill is renewed by the provider; a 401/403 invalidates it and retries once.
    """

    def __init__(
        self,
        settings: Settings,
        tokens: TokenProvider,
        client: httpx.Client | None = None,
        retries: int = 3,
        throttle_retries: int = 6,
        sleep=time.sleep,  # noqa: ANN001 - injected for tests
    ):
        self.settings = settings
        self.tokens = tokens
        self.retries = retries
        self.throttle_retries = throttle_retries
        self._sleep = sleep
        self._client = client or httpx.Client(
            base_url=settings.api_base,
            timeout=httpx.Timeout(settings.api_timeout_s, connect=10.0),
        )

    @property
    def tag(self) -> str:
        return self.settings.market_tag

    # -- public metadata ---------------------------------------------------------------

    def health(self) -> str:
        """`GET /healthz` (public): the literal `ok` of a live deployment."""
        return self._get("/healthz").text.strip()

    def market(self) -> dict[str, Any]:
        """`GET /{tag}` (public): market document, feed status and the cached QuestDB summary."""
        return self._get(f"/{self.tag}").json()

    def summaries(self) -> dict[str, TableSummary]:
        """`questdb.tables` keyed by QuestDB table name. The API keys it by role (`hist`,
        `contracts`, …); a list is tolerated in case that shape ever changes."""
        snap = self.market().get("questdb") or {}
        tables = snap.get("tables")
        error = snap.get("error")
        if not tables:
            raise ApiError(
                f"{self.tag}: the API has no QuestDB summary yet"
                + (f" ({error})" if error else "")
                + "; it is refreshed every 60 s after the feed starts, so retry shortly"
            )
        if error:
            log.warning("[%s] the API's QuestDB summary is stale (refreshed_at %s): %s", self.tag, snap.get("refreshed_at"), error)
        entries = tables.values() if isinstance(tables, dict) else tables
        out: dict[str, TableSummary] = {}
        for entry in entries:
            if not entry:
                continue
            out[entry["table"]] = TableSummary(
                table=entry["table"],
                rows=int(entry.get("rows") or 0),
                first_ts=parse_api_ts(entry.get("first_ts")),
                last_ts=parse_api_ts(entry.get("last_ts")),
            )
        return out

    def bounds(self, spec: TableSpec) -> Bounds:
        """Row count and ts range of `spec.table` for this market's key (≤60 s stale)."""
        summary = self.summaries().get(spec.table)
        if summary is None:
            raise ApiError(f"{self.tag}: the API reports no summary for {spec.table}")
        return Bounds(rows=summary.rows, min_ts=summary.first_ts, max_ts=summary.last_ts)

    # -- one day of rows ---------------------------------------------------------------

    def fetch_day(self, spec: TableSpec, day: date, upto: datetime) -> DayFetch:
        """All rows of one UTC day with `ts < upto`. Retried from scratch (the export is idempotent
        and one day is a cheap re-transfer); no `Last-Event-ID` resume, so nothing is deduped."""
        day_start = datetime(day.year, day.month, day.day)
        day_end = min(day_start + timedelta(days=1), upto)
        if day_start >= day_end:
            return DayFetch(pl.DataFrame(schema=spec.schema), 0, 0, "start >= end: nothing requested")

        path = f"/{self.tag}/data/{spec.alias}"
        params = {"start": ts_param(day_start), "end": ts_param(day_end)}
        label = f"{spec.alias} {day}"
        attempts = transient = throttled = 0
        invalidated = False
        while True:
            attempts += 1
            try:
                rows, reported, note = self._stream_day(path, params, spec)
            except _Throttled as e:
                throttled += 1
                if throttled >= self.throttle_retries:
                    raise ApiError(
                        f"{label}: still throttled after {throttled} attempts ({e}). The API runs at most "
                        f"DATA_STREAMS = {DATA_STREAMS} exports process-wide and each holds its permit for the whole "
                        "stream, so another download (or the chud-money web app) is using them; retry with --jobs 1."
                    ) from e
                log.warning("[%s] %s: throttled (%s), attempt %d/%d", spec.name, day, e, throttled, self.throttle_retries)
                self._sleep(_throttle_delay(throttled, e.retry_after))
                continue
            except _Rejected as e:
                if invalidated:
                    raise ApiError(f"{label}: {e}") from e
                invalidated = True
                log.warning("[%s] %s: %s; renewing the token and retrying once", spec.name, day, e)
                self.tokens.invalidate()
                continue
            except _Transient as e:
                transient += 1
                if transient >= self.retries:
                    raise ApiError(f"{label}: failed after {transient} attempts: {e}") from e
                log.warning("[%s] %s: %s, attempt %d/%d", spec.name, day, e, transient, self.retries)
                self._sleep(2.0 ** (transient - 1))
                continue

            df = rows_to_frame(rows, spec)
            self._check_range(df, spec, day_start, day_end)
            fetch = DayFetch(df, reported, attempts, note)
            if fetch.ok:
                return fetch
            note = note or f"{df.height} rows parsed, server reported {reported}"
            transient += 1
            if transient >= self.retries:
                log.error("[%s] %s: %s after %d attempts; keeping the prefix", spec.name, day, note, attempts)
                return DayFetch(df, reported, attempts, note)
            log.warning("[%s] %s: %s, attempt %d/%d", spec.name, day, note, transient, self.retries)
            self._sleep(2.0 ** (transient - 1))

    def _stream_day(self, path: str, params: dict[str, str], spec: TableSpec) -> tuple[list[str], int | None, str | None]:
        """One SSE connection: the `row` bodies, `done.rows` (None when the stream ended without a
        terminal `done`) and a note describing why, if it did not."""
        headers = {"Authorization": f"Bearer {self.tokens.token()}", "Accept": "text/event-stream"}
        rows: list[str] = []
        reported: int | None = None
        note: str | None = None
        seen_meta = False
        try:
            with self._client.stream("GET", path, params=params, headers=headers) as resp:
                if resp.status_code != 200:
                    self._raise_status(resp, path)
                for ev in iter_sse(resp.iter_lines()):
                    if ev.event == "row":
                        rows.append(ev.data)
                    elif ev.event == "meta":
                        seen_meta = True
                        self._check_meta(json.loads(ev.data), spec)
                    elif ev.event == "done":
                        reported = int(json.loads(ev.data).get("rows", -1))
                        break
                    elif ev.event == "error":
                        body = json.loads(ev.data)
                        note = f"server failed mid-stream after {body.get('rows')} rows: {body.get('error')}"
                        break
        except httpx.ConnectError as e:
            raise ApiError(f"cannot reach {self.settings.api_base}: {e}") from e
        except httpx.TransportError as e:
            raise _Transient(f"transport error: {e}") from e
        if reported is None and note is None:
            note = "stream ended without a `done` event" if seen_meta else "stream ended without a `meta` event"
        return rows, reported, note

    def _raise_status(self, resp: httpx.Response, path: str) -> NoReturn:
        code, message = resp.status_code, _error_body(resp)
        if code == 429:
            raise _Throttled(message or "too many data streams", _retry_after(resp))
        if code in (401, 403):
            raise _Rejected(f"the API rejected the bearer token ({code}): {message}")
        if code >= 500:
            raise _Transient(f"{code}: {message}")
        raise ApiError(f"GET {path} -> {code}: {message}")

    def _check_meta(self, meta: dict[str, Any], spec: TableSpec) -> None:
        """A mismatch means the URL or the market document points somewhere else: never retry it."""
        got = (meta.get("table"), meta.get("key_column"), meta.get("key"))
        want = (spec.table, spec.key_column, spec.key)
        if got != want:
            raise ApiError(f"{spec.alias}: the API streams {got} but this build expects {want}")
        columns = set(meta.get("columns") or ())
        missing = [c for c in ("ts", *spec.columns) if c not in columns]
        if missing:
            raise ApiError(f"{spec.table}: the API no longer exposes {', '.join(missing)} (columns: {sorted(columns)})")

    def _check_range(self, df: pl.DataFrame, spec: TableSpec, day_start: datetime, day_end: datetime) -> None:
        if df.is_empty():
            return
        lo, hi = df["ts"].min(), df["ts"].max()
        if lo < day_start or hi >= day_end:
            raise ApiError(f"{spec.alias}: got ts {lo} .. {hi} outside the requested [{day_start}, {day_end})")

    # -- transport for the public routes -----------------------------------------------

    def _get(self, path: str) -> httpx.Response:
        """Unauthenticated GET with the same 5xx/transport backoff as the stream."""
        delay = 1.0
        for attempt in range(1, self.retries + 1):
            try:
                resp = self._client.get(path)
            except httpx.ConnectError as e:
                raise ApiError(f"cannot reach {self.settings.api_base}: {e}") from e
            except httpx.TransportError as e:
                last: str = f"transport error: {e}"
            else:
                if resp.status_code < 400:
                    return resp
                if resp.status_code < 500:
                    raise ApiError(f"GET {path} -> {resp.status_code}: {_error_body(resp)}")
                last = f"{resp.status_code}: {_error_body(resp)}"
            if attempt == self.retries:
                raise ApiError(f"GET {path} failed after {attempt} attempts: {last}")
            log.warning("GET %s: %s, attempt %d/%d", path, last, attempt, self.retries)
            self._sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ChudApi:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
