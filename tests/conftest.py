"""Shared fixtures: synthetic BRTI ticks (1 Hz or sub-second), synthetic Kalshi contract candles
(END-labelled, like production) and a fake chud-money API serving both tables as SSE."""

from __future__ import annotations

import json
import math
import threading
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import polars as pl
import pytest

from chud_predictor.api import CONTRACT_NUMERIC
from chud_predictor.settings import Settings

DAY = timedelta(days=1)
MIN = timedelta(minutes=1)
BRTI_TABLE = "index_values_hist"
CONTRACT_TABLE = "contract_candles_hist"


def synthetic_ticks(start: datetime, n_seconds: int, seed: int = 0, start_price: float = 100_000.0, sigma: float = 2.0,
                    drop: set[int] | None = None, hz: int = 1) -> pl.DataFrame:
    """Random-walk ticks. The on-the-second path depends only on (seed, start_price, sigma), so the
    same call with hz=5 adds four sub-second ticks per second around an identical 1 Hz path."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, sigma, n_seconds)
    steps[0] = 0
    values = start_price + np.cumsum(steps)
    ts = [start + timedelta(seconds=i) for i in range(n_seconds)]
    df = pl.DataFrame({"ts": ts, "value": values}).with_columns(pl.col("ts").cast(pl.Datetime("us")))
    if drop:
        keep = np.array([i not in drop for i in range(n_seconds)])
        df = df.filter(pl.Series(keep))
    if hz > 1:
        sub_rng = np.random.default_rng(seed + 10_000)
        parts = [df]
        for j in range(1, hz):
            parts.append(df.with_columns(
                pl.col("ts") + pl.duration(microseconds=j * 1_000_000 // hz),
                pl.col("value") + pl.Series(sub_rng.normal(0, 5 * sigma, df.height)),   # large: must not leak into the bars
            ))
        df = pl.concat(parts).sort("ts")
    return df


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def synthetic_candles(ticks: pl.DataFrame, *, seed: int = 0, drop_candles: set[tuple[datetime, int]] | None = None,
                      null_strike_t0s: set[datetime] | None = None, degenerate_t0s: set[datetime] | None = None,
                      zero_volume_frac: float = 0.05) -> pl.DataFrame:
    """One KXBTC15M-like contract per 15-minute window covered by `ticks`.

    Candle `ts` is the END of its minute (T0+1m .. T0+15m), as in production. floor_strike is the mean
    of the on-the-second ticks in [T0-60s, T0). The mid follows a noisy random-walk fair value and
    the last candle collapses to 0.999 / 0.001 according to the real settlement (mean of the last
    minute vs strike). `drop_candles` holds (t0, k) pairs to omit, k = 0..14."""
    from chud_predictor.resample import to_bars

    drop_candles, null_strike_t0s, degenerate_t0s = drop_candles or set(), null_strike_t0s or set(), degenerate_t0s or set()
    rng = np.random.default_rng(seed)
    bars = to_bars(ticks.lazy(), "1m")
    ts0: datetime = bars["ts"][0]
    mean, close = bars["mean"].to_numpy(), bars["close"].to_numpy()
    sigma = float(np.nanstd(np.diff(np.log(close)))) or 1e-5
    n = bars.height
    rows = []
    first_t0 = ts0 + MIN
    first_t0 += timedelta(minutes=(15 - first_t0.minute % 15) % 15)
    t0 = first_t0.replace(second=0, microsecond=0)
    while True:
        i0 = int((t0 - ts0).total_seconds() // 60)
        if i0 + 14 >= n:
            break
        strike = float(mean[i0 - 1])
        up = bool(mean[i0 + 14] > strike)
        ticker = f"KXBTC15M-{t0 + 15 * MIN:%y%m%d%H%M}"
        prev_mid = None
        for k in range(15):
            if k == 14:
                bid, ask = (0.998, 1.0) if up else (0.0, 0.002)
            else:
                z = math.log(close[i0 + k] / strike) / (sigma * math.sqrt(14 - k + 1 / 3))
                mid = round(min(max(_phi(z) + rng.normal(0, 0.02), 0.02), 0.98), 3)
                bid, ask = round(mid - 0.005, 3), round(mid + 0.005, 3)
            if t0 in degenerate_t0s and k < 14:
                bid, ask = 0.4, 1.0
            mid = (bid + ask) / 2
            o = prev_mid if prev_mid is not None else mid
            volume = 0.0 if rng.random() < zero_volume_frac else float(rng.integers(1_000, 500_000))
            trade = None if volume == 0 else mid
            prev_mid = mid
            if (t0, k) in drop_candles:
                continue
            rows.append({
                "ts": t0 + (k + 1) * MIN, "ticker": ticker,
                "floor_strike": None if t0 in null_strike_t0s else strike,
                "yes_bid_open": max(o - 0.005, 0.0), "yes_bid_high": max(bid, o - 0.005), "yes_bid_low": max(min(bid, o - 0.005), 0.0), "yes_bid_close": bid,
                "yes_ask_open": min(o + 0.005, 1.0), "yes_ask_high": min(max(ask, o + 0.005), 1.0), "yes_ask_low": min(ask, o + 0.005), "yes_ask_close": ask,
                "price_open": None if trade is None else o, "price_high": None if trade is None else max(o, mid),
                "price_low": None if trade is None else min(o, mid), "price_close": trade, "price_mean": None if trade is None else (o + mid) / 2,
                "volume": volume, "open_interest": float(rng.integers(10_000, 900_000)),
            })
        t0 += 15 * MIN
    schema = {"ts": pl.Datetime("us"), "ticker": pl.Utf8} | {c: pl.Float64 for c in rows[0] if c not in ("ts", "ticker")}
    return pl.DataFrame(rows, schema=schema).sort("ts")


@pytest.fixture
def two_days() -> pl.DataFrame:
    return synthetic_ticks(datetime(2025, 9, 18), 2 * 86_400, seed=1)


CONTRACT_START = datetime(2025, 12, 20)
T_NULL_STRIKE = datetime(2025, 12, 20, 6, 0)
T_DEGENERATE = datetime(2025, 12, 20, 7, 15)
T_DROPPED = datetime(2025, 12, 20, 8, 30)     # candles k = 3, 4 missing
T_NO_SETTLE = datetime(2025, 12, 20, 9, 45)   # settlement candle missing


@pytest.fixture(scope="session")
def contract_world() -> dict:
    """Two synthetic days of ticks + candles with one of each data defect, and the joined frame."""
    from chud_predictor.features import build_frame_from
    from chud_predictor.resample import to_bars

    ticks = synthetic_ticks(CONTRACT_START, 2 * 86_400, seed=11, sigma=1.0)
    candles = synthetic_candles(
        ticks, seed=12, null_strike_t0s={T_NULL_STRIKE}, degenerate_t0s={T_DEGENERATE},
        drop_candles={(T_DROPPED, 3), (T_DROPPED, 4), (T_NO_SETTLE, 14)},
    )
    bars = to_bars(ticks.lazy(), "1m")
    return {"ticks": ticks, "candles": candles, "bars": bars, "frame": build_frame_from(bars, candles, vol_lookback=60)}


def write_raw(ticks: pl.DataFrame, raw_dir: Path) -> None:
    """Split a frame with a `ts` column into date=YYYY-MM-DD.parquet files (what `chudp download` writes)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for (d,), g in ticks.group_by(pl.col("ts").dt.date(), maintain_order=True):
        g.write_parquet(raw_dir / f"date={d.isoformat()}.parquet")


# ---------------------------------------------------------------------------------------------
# fake chud-money API (Server-Sent Events)

TS_FMT = "%Y-%m-%dT%H:%M:%S%.6fZ"           # what QuestDB prints and the API passes through
#: `questdb.tables` role -> table name, as `feeds::summary::QuestdbSummary` serialises it.
ROLES = {
    "live": "index_values_live",
    "hist": BRTI_TABLE,
    "contracts": CONTRACT_TABLE,
    "contract_ticker": "contract_ticker_live",
    "contract_book": "contract_book_live",
}
#: URL segment -> table (the API also accepts the bare table name).
ALIASES = {"index-hist": BRTI_TABLE, "candles": CONTRACT_TABLE, BRTI_TABLE: BRTI_TABLE, CONTRACT_TABLE: CONTRACT_TABLE}
KEYS = {BRTI_TABLE: ("index_id", "BRTI"), CONTRACT_TABLE: ("series_ticker", "KXBTC15M")}
#: Every column the API sends, in DDL order (`ts` last) - more than the downloader keeps.
WIRE_COLUMNS = {
    BRTI_TABLE: ("index_id", "source", "value", "received_at", "ts"),
    CONTRACT_TABLE: ("ticker", "series_ticker", "source", *CONTRACT_NUMERIC, "ts"),
}


class StaticToken:
    """Stand-in for `auth.TokenProvider` (the real one is not imported by these tests)."""

    def __init__(self, token: str = "tok-1") -> None:
        self._token = token
        self.calls = 0
        self.invalidated = 0

    def token(self) -> str:
        self.calls += 1
        return self._token

    def invalidate(self) -> None:
        self.invalidated += 1
        self._token = f"tok-{self.invalidated + 1}"


class FakeChudApi:
    """The two exported tables over the real wire shape: `GET /healthz`, `GET /{tag}` and
    `GET /{tag}/data/{alias}` as SSE (`meta`, `row`*, `done` | `error`)."""

    def __init__(self) -> None:
        self.tag = "btc-15m"
        self.tables: dict[str, dict[date, pl.DataFrame]] = {BRTI_TABLE: {}, CONTRACT_TABLE: {}}
        self.valid_token: str | None = None      # None: any bearer is accepted
        self.fail_next: int = 0                  # data requests to answer with 500
        self.error_next: int = 0                 # ... with meta + half the rows + `error`, no `done`
        self.throttle_next: int = 0              # ... with 429
        self.retry_after: str | None = None      # `Retry-After` on those 429s (the real API sends none)
        self.max_concurrent: int | None = None   # mimics DATA_STREAMS: 429 beyond this many at once
        self.truncate_day: date | None = None    # serve this day 100 rows short, `done.rows` honest
        self.truncate_reported: date | None = None   # ... short, but `done.rows` claims the full day
        self.keepalive: bool = False             # emit `:` keep-alive comments between frames
        self.tables_null: bool = False           # `questdb.tables: null`, as before the first refresh
        self.summary_error: str | None = None
        self.nan_cell: tuple[str, int] | None = None  # (column, row index within the day) -> NaN -> null
        self.requests: list[tuple[str, dict]] = []
        self.auth_seen: list[str | None] = []
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    # -- rows ------------------------------------------------------------------------

    @property
    def days(self) -> dict[date, pl.DataFrame]:
        return self.tables[BRTI_TABLE]

    def add_day(self, day: date, df: pl.DataFrame, table: str = BRTI_TABLE) -> None:
        self.tables[table][day] = df.sort("ts")

    def add_rows(self, df: pl.DataFrame, table: str) -> None:
        for (d,), g in df.group_by(pl.col("ts").dt.date(), maintain_order=True):
            self.add_day(d, g, table)

    def all_rows(self, table: str) -> pl.DataFrame | None:
        days = self.tables[table]
        return pl.concat([days[d] for d in sorted(days)]).sort("ts") if days else None

    # -- what the handler serves -----------------------------------------------------

    def summary(self, table: str) -> dict:
        rows = self.all_rows(table) if table in self.tables else None
        if rows is None:
            return {"table": table, "rows": 0, "first_ts": None, "last_ts": None}
        return {"table": table, "rows": rows.height, "first_ts": _fmt(rows["ts"].min()), "last_ts": _fmt(rows["ts"].max())}

    def market_detail(self) -> dict:
        tables = None if self.tables_null else {role: self.summary(t) for role, t in ROLES.items()} | {
            "coinbase_ticker": None, "coinbase_book": None,
        }
        return {
            "market": {"tag": self.tag, "series_ticker": "KXBTC15M", "index_id": "BRTI", "title": "fake",
                       "created_at": "2025-09-17T00:00:00Z"},
            "feed": {"tag": self.tag, "running": True},
            "questdb": {"tables": tables, "refreshed_at": "2026-09-17T02:00:30.123456789Z", "error": self.summary_error},
        }

    def wire(self, table: str, lo: datetime, hi: datetime) -> list[str]:
        """The `row` bodies for `[lo, hi)`: every API column, timestamps as RFC3339 text and
        non-finite doubles as `null` (what `questdb::row_json` does)."""
        rows = self.all_rows(table)
        df = rows.filter((pl.col("ts") >= lo) & (pl.col("ts") < hi)) if rows is not None else None
        if df is None or df.is_empty():
            return []
        if table == BRTI_TABLE:
            df = df.select(
                pl.lit("BRTI").alias("index_id"), pl.lit("fake").alias("source"), "value",
                (pl.col("ts") + pl.duration(milliseconds=5)).dt.strftime(TS_FMT).alias("received_at"),
                pl.col("ts").dt.strftime(TS_FMT).alias("ts"),
            )
        else:
            df = df.select(
                "ticker", pl.lit("KXBTC15M").alias("series_ticker"), pl.lit("fake").alias("source"),
                *CONTRACT_NUMERIC, pl.col("ts").dt.strftime(TS_FMT).alias("ts"),
            )
        if self.nan_cell is not None:
            col, i = self.nan_cell
            df = df.with_columns(
                pl.when(pl.int_range(pl.len()) == i).then(pl.lit(float("nan"))).otherwise(pl.col(col)).alias(col)
            )
        return df.write_ndjson().splitlines()


def _fmt(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _parse(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


def make_handler(db: FakeChudApi):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"        # close-delimited bodies: no Content-Length on the stream

        def log_message(self, *a) -> None:  # noqa: ANN002
            pass

        def _send(self, code: int, body: bytes, ctype: str, headers: dict[str, str] | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code: int, message: str, headers: dict[str, str] | None = None) -> None:
            self._send(code, json.dumps({"error": message}).encode(), "application/json", headers)

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(url.query).items()}
            db.requests.append((url.path, params))
            db.auth_seen.append(self.headers.get("Authorization"))
            parts = [p for p in url.path.split("/") if p]
            if parts == ["healthz"]:
                return self._send(200, b"ok", "text/plain")
            if not parts or parts[0] != db.tag:
                return self._error(404, f"market '{parts[0] if parts else ''}'")
            if len(parts) == 1:
                return self._send(200, json.dumps(db.market_detail()).encode(), "application/json")
            if len(parts) != 3 or parts[1] != "data":
                return self._error(404, f"no route for {url.path}")
            self._data(parts[2], params)

        # -- GET /{tag}/data/{alias} -------------------------------------------------

        def _data(self, alias: str, params: dict[str, str]) -> None:
            auth = self.headers.get("Authorization") or ""
            token = auth.removeprefix("Bearer ") if auth.startswith("Bearer ") else None
            if not token or (db.valid_token is not None and token != db.valid_token):
                return self._error(401, "missing or invalid access token", {"WWW-Authenticate": "Bearer"})
            table = ALIASES.get(alias)
            if table is None:
                return self._error(404, f"table '{alias}'; known tables: {', '.join(sorted(ALIASES))}")
            start, end = params.get("start"), params.get("end")
            if start and end and _parse(start) >= _parse(end):
                return self._error(400, "start must be before end")
            with db.lock:
                if db.throttle_next > 0:
                    db.throttle_next -= 1
                    return self._error(429, "at most 2 data streams may run at once",
                                       {"Retry-After": db.retry_after} if db.retry_after else None)
                if db.fail_next > 0:
                    db.fail_next -= 1
                    return self._error(500, "questdb query failed: boom")
                if db.max_concurrent is not None and db.active >= db.max_concurrent:
                    return self._error(429, f"at most {db.max_concurrent} data streams may run at once")
                broken = db.error_next > 0
                if broken:
                    db.error_next -= 1
                db.active += 1
                db.peak = max(db.peak, db.active)
            try:
                self._stream(table, start, end, broken)
            finally:
                with db.lock:
                    db.active -= 1

        def _stream(self, table: str, start: str | None, end: str | None, broken: bool) -> None:
            rows = db.all_rows(table)
            lo = _parse(start) if start else (rows["ts"].min() if rows is not None else datetime(1970, 1, 1))
            hi = _parse(end) if end else ((rows["ts"].max() + timedelta(microseconds=1)) if rows is not None else datetime(1970, 1, 1))
            lines = db.wire(table, lo, hi)
            full = len(lines)
            if db.truncate_day == lo.date() or db.truncate_reported == lo.date():
                lines = lines[: max(0, full - 100)]
            reported = full if db.truncate_reported == lo.date() else len(lines)
            key_column, key = KEYS[table]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                self._frame("meta", {"table": table, "key_column": key_column, "key": key, "start": _fmt(lo),
                                     "end": _fmt(hi), "columns": list(WIRE_COLUMNS[table])}, retry=10_000)
                cut = len(lines) // 2 if broken else len(lines)
                for i, line in enumerate(lines[:cut]):
                    if db.keepalive and i % 500 == 0:
                        self.wfile.write(b":\r\n\r\n")
                    self._frame_raw("row", line, ident=json.loads(line)["ts"])
                    if i % 256 == 0:
                        self.wfile.flush()
                if broken:
                    last = json.loads(lines[cut - 1])["ts"] if cut else None
                    self._frame("error", {"error": "questdb query failed: connection reset", "rows": cut, "resume_from": last})
                else:
                    self._frame("done", {"rows": reported, "start": _fmt(lo), "end": _fmt(hi), "elapsed_ms": 1})
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):    # the client stops reading after `done`
                pass

        def _frame(self, name: str, data: dict, *, ident: str | None = None, retry: int | None = None) -> None:
            self._frame_raw(name, json.dumps(data), ident=ident, retry=retry)

        def _frame_raw(self, name: str, data: str, *, ident: str | None = None, retry: int | None = None) -> None:
            # Field order is axum's: the builder appends as it is called (`event`, `data`, then `id`/`retry`).
            out = f"event: {name}\ndata: {data}\n"
            if ident is not None:
                out += f"id: {ident}\n"
            if retry is not None:
                out += f"retry: {retry}\n"
            self.wfile.write((out + "\n").encode())

    return Handler


@pytest.fixture
def fake_api():
    db = FakeChudApi()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(db))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    db.port = server.server_address[1]  # type: ignore[attr-defined]
    yield db
    server.shutdown()
    server.server_close()


@pytest.fixture
def tokens() -> StaticToken:
    return StaticToken()


@pytest.fixture
def settings(fake_api, tmp_path: Path) -> Settings:
    return Settings(api_base=f"http://127.0.0.1:{fake_api.port}", market_tag="btc-15m", data_dir=tmp_path / "data")


@pytest.fixture
def api(settings, tokens):
    """`ChudApi` against the fake server, with the retry backoff neutralised."""
    from chud_predictor.api import ChudApi

    with ChudApi(settings, tokens, sleep=lambda _s: None) as client:
        yield client
