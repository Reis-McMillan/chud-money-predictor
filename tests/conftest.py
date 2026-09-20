"""Shared fixtures: synthetic BRTI ticks (1 Hz or sub-second), synthetic Kalshi contract candles
(END-labelled, like production) and a table-keyed fake QuestDB HTTP server."""

from __future__ import annotations

import base64
import json
import math
import re
import threading
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import polars as pl
import pytest

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
# fake QuestDB

class FakeQuestDB:
    def __init__(self) -> None:
        self.tables: dict[str, dict[date, pl.DataFrame]] = {BRTI_TABLE: {}, CONTRACT_TABLE: {}}
        self.fail_next: int = 0                  # number of /exp requests to fail with 500
        self.truncate_day: date | None = None    # serve this day short by 100 rows
        self.requests: list[str] = []
        self.auth_seen: list[str | None] = []

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


TS_RE = re.compile(r"'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6})Z'")
FROM_RE = re.compile(r"\bFROM\s+(\w+)")
SELECT_RE = re.compile(r"^SELECT\s+(.*?)\s+FROM\s", re.S)


def _parse(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")


def _fmt(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def make_handler(db: FakeQuestDB):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:  # noqa: ANN002
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            q = parse_qs(url.query).get("query", [""])[0]
            db.requests.append(q)
            db.auth_seen.append(self.headers.get("Authorization"))
            if url.path == "/exec":
                self._exec(q)
            elif url.path == "/exp":
                self._exp(q)
            else:
                self._send(404, b"nope", "text/plain")

        def _json(self, columns: list[tuple[str, str]], rows: list[list]) -> None:
            body = json.dumps({"query": "", "columns": [{"name": n, "type": t} for n, t in columns], "dataset": rows, "count": len(rows)})
            self._send(200, body.encode(), "application/json")

        def _exec(self, q: str) -> None:
            if q.startswith("SELECT build"):
                return self._json([("build", "STRING")], [["fake-1"]])
            if "tables()" in q:
                return self._json([("table_name", "STRING")], [[t] for t in sorted(db.tables)])
            m = FROM_RE.search(q)
            table = m.group(1) if m else ""
            if table not in db.tables:
                return self._send(200, json.dumps({"query": q, "error": f"table does not exist [table={table}]"}).encode(), "application/json")
            rows_all = db.all_rows(table)
            if q.startswith("SELECT count(), min(ts), max(ts)"):
                cols = [("count()", "LONG"), ("min(ts)", "TIMESTAMP"), ("max(ts)", "TIMESTAMP")]
                if rows_all is None:
                    return self._json(cols, [[0, None, None]])
                return self._json(cols, [[rows_all.height, _fmt(rows_all["ts"].min()), _fmt(rows_all["ts"].max())]])
            if "SAMPLE BY 1d" in q:
                upto = _parse(TS_RE.findall(q)[0])
                out = []
                for d in sorted(db.tables[table]):
                    n = db.tables[table][d].filter(pl.col("ts") < upto).height
                    if n:
                        out.append([_fmt(datetime(d.year, d.month, d.day)), n])
                return self._json([("ts", "TIMESTAMP"), ("n", "LONG")], out)
            return self._send(200, json.dumps({"query": q, "error": "unsupported in fake"}).encode(), "application/json")

        def _exp(self, q: str) -> None:
            if db.fail_next > 0:
                db.fail_next -= 1
                return self._send(500, b"boom", "text/plain")
            table = FROM_RE.search(q).group(1)
            cols = [c.strip() for c in SELECT_RE.search(q).group(1).split(",")][1:]   # first item is cast(ts as long) AS ts_us
            lo, hi = (_parse(s) for s in TS_RE.findall(q)[:2])
            rows_all = db.all_rows(table)
            df = rows_all.filter((pl.col("ts") >= lo) & (pl.col("ts") < hi)) if rows_all is not None else pl.DataFrame({"ts": []})
            if db.truncate_day is not None and lo.date() == db.truncate_day:
                df = df.head(max(0, df.height - 100))
            if df.is_empty():
                return self._send(200, (",".join(["ts_us", *cols]) + "\n").encode(), "text/csv")
            out = df.select(pl.col("ts").dt.epoch("us").alias("ts_us"), *cols)
            self._send(200, out.write_csv().encode(), "text/csv")   # nulls -> empty fields, like QuestDB

    return Handler


@pytest.fixture
def fake_qdb():
    db = FakeQuestDB()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(db))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    db.port = server.server_address[1]  # type: ignore[attr-defined]
    yield db
    server.shutdown()
    server.server_close()


@pytest.fixture
def settings(fake_qdb, tmp_path: Path) -> Settings:
    return Settings(qdb_host="127.0.0.1", qdb_port=fake_qdb.port, qdb_user="admin", qdb_password="pw", data_dir=tmp_path / "data")


def basic_auth_header(user: str, pw: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()
