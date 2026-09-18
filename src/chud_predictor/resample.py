"""Regular bars from raw 1-second ticks.

Bars are closed-left and labelled by their start: the bar labelled L covers [L, L+freq) and is
fully observable at wall-clock L+freq. The output sits on a complete, gap-free time grid; bars
with no ticks are present with null values and `is_gap = True`.

The `mean` column is the settlement quantity of the Kalshi 15-minute market: the settlement of a
window opening at T0 is the mean of the ticks in [T0+14m, T0+15m), i.e. `mean` of the 1m bar
labelled T0+14m.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2
FREQ_SECONDS: dict[str, int] = {"1s": 1, "5s": 5, "15s": 15, "30s": 30, "1m": 60, "5m": 300, "15m": 900, "1h": 3600}
BAR_COLUMNS = ["ts", "open", "high", "low", "close", "mean", "std", "n_ticks", "is_gap", "is_full"]


def freq_seconds(freq: str) -> int:
    if freq not in FREQ_SECONDS:
        raise ValueError(f"unsupported freq {freq!r}; one of {sorted(FREQ_SECONDS)}")
    return FREQ_SECONDS[freq]


def load_raw(raw_dir: Path, start: date | None = None, end: date | None = None) -> pl.LazyFrame:
    files = sorted(raw_dir.glob("date=*.parquet"))
    if not files:
        raise FileNotFoundError(f"no raw tick files under {raw_dir}; run `chudp download` first")
    if start or end:
        def keep(p: Path) -> bool:
            d = date.fromisoformat(p.stem.split("=", 1)[1])
            return (not start or d >= start) and (not end or d <= end)
        files = [p for p in files if keep(p)]
        if not files:
            raise FileNotFoundError(f"no raw tick files in [{start}, {end}] under {raw_dir}")
    return pl.scan_parquet([str(p) for p in files]).select("ts", "value")


def to_bars(lf: pl.LazyFrame, freq: str) -> pl.DataFrame:
    expected = freq_seconds(freq)
    ticks = lf.sort("ts").collect()
    if ticks.is_empty():
        raise ValueError("no ticks to resample")
    bars = ticks.group_by_dynamic("ts", every=freq, closed="left", label="left").agg(
        pl.col("value").first().alias("open"),
        pl.col("value").max().alias("high"),
        pl.col("value").min().alias("low"),
        pl.col("value").last().alias("close"),
        pl.col("value").mean().alias("mean"),
        pl.col("value").std(ddof=0).alias("std"),
        pl.len().alias("n_ticks"),
    )
    first, last = bars["ts"].min(), bars["ts"].max()
    grid = pl.DataFrame({"ts": pl.datetime_range(first, last, interval=freq, eager=True, time_unit="us")})
    out = grid.join(bars, on="ts", how="left").with_columns(pl.col("n_ticks").fill_null(0).cast(pl.Int64))
    out = out.with_columns(
        (pl.col("n_ticks") == 0).alias("is_gap"),
        (pl.col("n_ticks") == expected).alias("is_full"),
    )
    return out.select(BAR_COLUMNS).sort("ts")


def bars_path(processed_dir: Path, freq: str) -> Path:
    return processed_dir / f"brti_{freq}.parquet"


def meta_path(processed_dir: Path, freq: str) -> Path:
    return processed_dir / f"brti_{freq}.meta.json"


def describe_bars(bars: pl.DataFrame, freq: str) -> dict:
    gaps = bars.filter(pl.col("is_gap"))
    # largest run of consecutive gap bars
    largest = 0
    if not gaps.is_empty():
        flags = bars["is_gap"].to_numpy()
        run = 0
        for f in flags:
            run = run + 1 if f else 0
            largest = max(largest, run)
    hist = (
        bars.group_by("n_ticks").len().sort("n_ticks")
        .select(pl.col("n_ticks").cast(pl.Utf8), "len")
        .rows()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "freq": freq,
        "n_bars": bars.height,
        "n_gaps": gaps.height,
        "n_not_full": int((~bars["is_full"]).sum()),
        "largest_gap_bars": largest,
        "first_ts": bars["ts"].min().isoformat() if bars.height else None,
        "last_ts": bars["ts"].max().isoformat() if bars.height else None,
        "n_ticks_histogram": {k: v for k, v in hist},
        "built_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
    }


def build(raw_dir: Path, processed_dir: Path, freq: str = "1m", force: bool = False) -> tuple[Path, dict]:
    out = bars_path(processed_dir, freq)
    mp = meta_path(processed_dir, freq)
    if out.exists() and mp.exists() and not force:
        meta = json.loads(mp.read_text())
        raw_files = sorted(raw_dir.glob("date=*.parquet"))
        newest_raw = max((p.stat().st_mtime for p in raw_files), default=0.0)
        if meta.get("schema_version") == SCHEMA_VERSION and meta.get("raw_mtime", 0.0) >= newest_raw:
            log.info("bars cache up to date: %s", out)
            return out, meta
    bars = to_bars(load_raw(raw_dir), freq)
    processed_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    bars.write_parquet(tmp, compression="zstd", compression_level=3, statistics=True)
    tmp.replace(out)
    meta = describe_bars(bars, freq)
    meta["raw_mtime"] = max((p.stat().st_mtime for p in raw_dir.glob("date=*.parquet")), default=0.0)
    mp.write_text(json.dumps(meta, indent=1))
    return out, meta


def load_bars(processed_dir: Path, freq: str = "1m") -> pl.DataFrame:
    out, mp = bars_path(processed_dir, freq), meta_path(processed_dir, freq)
    if not out.exists():
        raise FileNotFoundError(f"{out} missing; run `chudp resample --freq {freq}`")
    if mp.exists():
        meta = json.loads(mp.read_text())
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(
                f"{out} has bar schema v{meta.get('schema_version')}, need v{SCHEMA_VERSION}; "
                f"re-run `chudp resample --freq {freq} --force`"
            )
    bars = pl.read_parquet(out)
    step = freq_seconds(freq)
    if bars.height > 1:
        deltas = bars["ts"].diff().drop_nulls().dt.total_seconds().unique().to_list()
        if deltas != [step]:
            raise RuntimeError(f"{out} is not on a regular {freq} grid (deltas {deltas[:5]})")
    return bars


def settlement_from_ticks(raw: pl.LazyFrame, t0: datetime) -> tuple[float | None, int]:
    """Kalshi settlement for the window opening at t0: mean of ticks in [t0+14m, t0+15m)."""
    lo, hi = t0 + timedelta(minutes=14), t0 + timedelta(minutes=15)
    df = raw.filter((pl.col("ts") >= lo) & (pl.col("ts") < hi)).select(
        pl.col("value").mean().alias("mean"), pl.len().alias("n")
    ).collect()
    n = int(df["n"][0])
    return (float(df["mean"][0]) if n else None), n
