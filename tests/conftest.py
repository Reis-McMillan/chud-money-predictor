"""Shared fixtures: synthetic 1-second BRTI ticks and a fake QuestDB HTTP server."""

from __future__ import annotations

import base64
import json
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


def synthetic_ticks(start: datetime, n_seconds: int, seed: int = 0, start_price: float = 100_000.0, sigma: float = 2.0,
                    drop: set[int] | None = None) -> pl.DataFrame:
    """Random-walk ticks, one per second, optionally dropping some second offsets."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, sigma, n_seconds)
    steps[0] = 0
    values = start_price + np.cumsum(steps)
    ts = [start + timedelta(seconds=i) for i in range(n_seconds)]
    df = pl.DataFrame({"ts": ts, "value": values}).with_columns(pl.col("ts").cast(pl.Datetime("us")))
    if drop:
        keep = np.array([i not in drop for i in range(n_seconds)])
        df = df.filter(pl.Series(keep))
    return df


@pytest.fixture
def two_days() -> pl.DataFrame:
    return synthetic_ticks(datetime(2025, 9, 18), 2 * 86_400, seed=1)


# ---------------------------------------------------------------------------------------------
# fake QuestDB

class FakeQuestDB:
    def __init__(self) -> None:
        self.days: dict[date, pl.DataFrame] = {}
        self.max_ts: datetime | None = None
        self.fail_next: int = 0           # number of /exp requests to fail with 500
        self.truncate_day: date | None = None  # serve this day short by 100 rows
        self.requests: list[str] = []
        self.auth_seen: list[str | None] = []

    def add_day(self, day: date, df: pl.DataFrame) -> None:
        self.days[day] = df.sort("ts")
        self.max_ts = max(d["ts"].max() for d in self.days.values())

    def all_ticks(self) -> pl.DataFrame:
        return pl.concat([self.days[d] for d in sorted(self.days)]).sort("ts")


TS_RE = re.compile(r"'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6})Z'")


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
            all_ticks = db.all_ticks() if db.days else None
            if q.startswith("SELECT build"):
                self._json([("build", "STRING")], [["fake-1"]])
            elif "tables()" in q:
                self._json([("table_name", "STRING")], [["index_values_hist"], ["index_values_live"]])
            elif q.startswith("SELECT count(), min(ts), max(ts)"):
                if all_ticks is None:
                    self._json([("count()", "LONG"), ("min(ts)", "TIMESTAMP"), ("max(ts)", "TIMESTAMP")], [[0, None, None]])
                else:
                    self._json([("count()", "LONG"), ("min(ts)", "TIMESTAMP"), ("max(ts)", "TIMESTAMP")],
                               [[all_ticks.height, _fmt(all_ticks["ts"].min()), _fmt(all_ticks["ts"].max())]])
            elif "SAMPLE BY 1d" in q:
                upto = _parse(TS_RE.findall(q)[0])
                rows = []
                for d in sorted(db.days):
                    n = db.days[d].filter(pl.col("ts") < upto).height
                    if n:
                        rows.append([_fmt(datetime(d.year, d.month, d.day)), n])
                self._json([("ts", "TIMESTAMP"), ("n", "LONG")], rows)
            else:
                self._send(200, json.dumps({"query": q, "error": "unsupported in fake"}).encode(), "application/json")

        def _exp(self, q: str) -> None:
            if db.fail_next > 0:
                db.fail_next -= 1
                self._send(500, b"boom", "text/plain")
                return
            lo, hi = (_parse(s) for s in TS_RE.findall(q)[:2])
            df = db.all_ticks().filter((pl.col("ts") >= lo) & (pl.col("ts") < hi))
            if db.truncate_day is not None and lo.date() == db.truncate_day:
                df = df.head(max(0, df.height - 100))
            epoch = datetime(1970, 1, 1)
            csv = "\"ts_us\",\"value\"\n" + "\n".join(
                f"{(t - epoch) // timedelta(microseconds=1)},{v}" for t, v in zip(df["ts"].to_list(), df["value"].to_list(), strict=True)
            ) + ("\n" if df.height else "")
            self._send(200, csv.encode(), "text/csv")

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
