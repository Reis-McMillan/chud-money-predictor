"""Regular bars from raw BRTI ticks.

Bars are closed-left and labelled by their start: the bar labelled L covers [L, L+freq) and is
fully observable at wall-clock L+freq. The output sits on a complete, gap-free time grid; bars
with no ticks are present with null values and `is_gap = True`.

Tick density. The index table holds one tick per second until about May 2026 and five per second
(200 ms) after. Kalshi samples the index once per second, and the older history *is* the
once-per-second index, so every bar statistic (open / high / low / close / mean / std / n_ticks)
is computed on the on-the-second ticks only. That keeps the series homogeneous across the density
change. `n_raw_ticks` counts all ticks and is a diagnostic, never a model input.

Inside-the-bar statistics, from the same on-the-second ticks: `rv_1s` is the realized variance of the
bar, the sum of squared one-second log returns scaled to a full bar (null with fewer than half the
bar's returns, so a sparse bar is never a zero-variance bar); `ret_l10` / `ret_l30` are the log
returns over the last 10 / 30 seconds of the bar. A return belongs to the bar of its later tick; one
that spans more than a minute or a UTC midnight is dropped.

`mean` is the Kalshi settlement quantity: the settlement of a window opening at T0 is the mean of
the once-per-second index over [T0+14m, T0+15m), i.e. `mean` of the 1m bar labelled T0+14m. The
strike of that window is `mean` of the bar labelled T0-1m.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

SCHEMA_VERSION = 4
FREQ_SECONDS: dict[str, int] = {"1s": 1, "5s": 5, "15s": 15, "30s": 30, "1m": 60, "5m": 300, "15m": 900, "1h": 3600}
BAR_COLUMNS = ["ts", "open", "high", "low", "close", "mean", "std", "rv_1s", "ret_l10", "ret_l30", "n_ticks", "n_raw_ticks", "is_gap", "is_full"]
MAX_RETURN_GAP_S = 60   # a return across a longer hole is not one bar's variance
TAIL_TOLERANCE_S = 5    # how far from its mark a tick of the last-N-seconds return may sit
ON_SECOND = pl.col("ts").dt.truncate("1s") == pl.col("ts")


def freq_seconds(freq: str) -> int:
    if freq not in FREQ_SECONDS:
        raise ValueError(f"unsupported freq {freq!r}; one of {sorted(FREQ_SECONDS)}")
    return FREQ_SECONDS[freq]


def raw_files(raw_dir: Path, start: date | None = None, end: date | None = None) -> list[Path]:
    files = sorted(raw_dir.glob("date=*.parquet"))
    if not files:
        raise FileNotFoundError(f"no raw files under {raw_dir}; run `chudp download` first")

    def keep(p: Path) -> bool:
        d = date.fromisoformat(p.stem.split("=", 1)[1])
        return (not start or d >= start) and (not end or d <= end)

    files = [p for p in files if keep(p)]
    if not files:
        raise FileNotFoundError(f"no raw files in [{start}, {end}] under {raw_dir}")
    return files


def load_raw(raw_dir: Path, start: date | None = None, end: date | None = None) -> pl.LazyFrame:
    return pl.scan_parquet([str(p) for p in raw_files(raw_dir, start, end)]).select("ts", "value")


def aggregate(ticks: pl.DataFrame, freq: str) -> pl.DataFrame:
    """Bars for the intervals that contain ticks (no grid completion). `ticks` must be sorted."""
    S = freq_seconds(freq)
    raw = ticks.group_by_dynamic("ts", every=freq, closed="left", label="left").agg(pl.len().alias("n_raw_ticks"))
    dt = pl.col("ts").diff().dt.total_seconds()
    same_day = pl.col("ts").dt.date() == pl.col("ts").dt.date().shift(1)
    off = (pl.col("ts") - pl.col("ts").dt.truncate(freq)).dt.total_seconds()
    lv, r1 = pl.col("_lv"), pl.col("_r1")

    def tail(seconds: int) -> pl.Expr:
        mark = S - 1 - seconds
        ref = lv.filter((pl.col("_off") <= mark) & (pl.col("_off") > mark - TAIL_TOLERANCE_S)).last()
        return pl.when(pl.col("_off").last() >= S - TAIL_TOLERANCE_S).then(lv.last() - ref)

    sec = ticks.filter(ON_SECOND).with_columns(pl.col("value").log().alias("_lv"), off.alias("_off")).with_columns(
        pl.when((dt <= MAX_RETURN_GAP_S) & same_day).then(pl.col("_lv").diff()).alias("_r1"),
    ).group_by_dynamic("ts", every=freq, closed="left", label="left").agg(
        pl.col("value").first().alias("open"),
        pl.col("value").max().alias("high"),
        pl.col("value").min().alias("low"),
        pl.col("value").last().alias("close"),
        pl.col("value").mean().alias("mean"),
        pl.col("value").std(ddof=0).alias("std"),
        pl.when(r1.count() >= max(S // 2, 1)).then((r1 ** 2).sum() * S / r1.count()).alias("rv_1s"),
        tail(10).alias("ret_l10"),
        tail(30).alias("ret_l30"),
        pl.len().alias("n_ticks"),
    )
    if raw.height and sec.is_empty():
        raise ValueError(
            f"{raw['n_raw_ticks'].sum()} ticks starting {ticks['ts'][0]} but none on a whole second: "
            "the sub-second grid is offset and the once-per-second bars cannot be built"
        )
    return raw.join(sec, on="ts", how="left")


def complete_grid(bars: pl.DataFrame, freq: str) -> pl.DataFrame:
    expected = freq_seconds(freq)
    first, last = bars["ts"].min(), bars["ts"].max()
    grid = pl.DataFrame({"ts": pl.datetime_range(first, last, interval=freq, eager=True, time_unit="us")})
    out = grid.join(bars, on="ts", how="left").with_columns(
        pl.col("n_ticks").fill_null(0).cast(pl.Int64), pl.col("n_raw_ticks").fill_null(0).cast(pl.Int64)
    )
    out = out.with_columns((pl.col("n_ticks") == 0).alias("is_gap"), (pl.col("n_ticks") == expected).alias("is_full"))
    return out.select(BAR_COLUMNS).sort("ts")


def to_bars(lf: pl.LazyFrame, freq: str) -> pl.DataFrame:
    ticks = lf.sort("ts").collect()
    if ticks.is_empty():
        raise ValueError("no ticks to resample")
    return complete_grid(aggregate(ticks, freq), freq)


def to_bars_by_day(files: list[Path], freq: str) -> pl.DataFrame:
    """Same result as `to_bars` over all files, but one day in memory at a time (bars up to 1h never
    straddle a UTC midnight, and a year of 5 Hz ticks should not be sorted at once)."""
    if freq_seconds(freq) > 3600:
        raise ValueError("per-day resampling needs a bar size that divides a day")
    parts = [aggregate(pl.read_parquet(p, columns=["ts", "value"]).sort("ts"), freq) for p in files]
    parts = [p for p in parts if p.height]
    if not parts:
        raise ValueError("no ticks to resample")
    return complete_grid(pl.concat(parts).sort("ts"), freq)


def bars_path(processed_dir: Path, freq: str) -> Path:
    return processed_dir / f"brti_{freq}.parquet"


def meta_path(processed_dir: Path, freq: str) -> Path:
    return processed_dir / f"brti_{freq}.meta.json"


def describe_bars(bars: pl.DataFrame, freq: str) -> dict:
    flags = bars["is_gap"].to_numpy()
    largest = run = 0
    for f in flags:
        run = run + 1 if f else 0
        largest = max(largest, run)
    hist = bars.group_by("n_ticks").len().sort("n_ticks").select(pl.col("n_ticks").cast(pl.Utf8), "len").rows()
    daily = bars.group_by(pl.col("ts").dt.date().alias("d")).agg(pl.col("n_raw_ticks").sum().alias("raw"), pl.col("n_ticks").sum().alias("sec"))
    ratio = (daily["raw"] / daily["sec"].replace(0, None)).drop_nulls()
    return {
        "schema_version": SCHEMA_VERSION,
        "freq": freq,
        "n_bars": bars.height,
        "n_gaps": int(flags.sum()),
        "n_not_full": int((~bars["is_full"]).sum()),
        "n_no_rv": int(bars["rv_1s"].is_null().sum()),
        "largest_gap_bars": largest,
        "first_ts": bars["ts"].min().isoformat() if bars.height else None,
        "last_ts": bars["ts"].max().isoformat() if bars.height else None,
        "n_ticks_histogram": {k: v for k, v in hist},
        "days_once_per_second": int((ratio < 1.5).sum()),
        "days_sub_second": int((ratio >= 1.5).sum()),
        "built_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
    }


def build(raw_dir: Path, processed_dir: Path, freq: str = "1m", force: bool = False) -> tuple[Path, dict]:
    out = bars_path(processed_dir, freq)
    mp = meta_path(processed_dir, freq)
    files = raw_files(raw_dir)
    newest_raw = max(p.stat().st_mtime for p in files)
    if out.exists() and mp.exists() and not force:
        meta = json.loads(mp.read_text())
        if meta.get("schema_version") == SCHEMA_VERSION and meta.get("raw_mtime", 0.0) >= newest_raw:
            log.info("bars cache up to date: %s", out)
            return out, meta
    bars = to_bars_by_day(files, freq)
    processed_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    bars.write_parquet(tmp, compression="zstd", compression_level=3, statistics=True)
    tmp.replace(out)
    meta = describe_bars(bars, freq)
    meta["raw_mtime"] = newest_raw
    mp.write_text(json.dumps(meta, indent=1))
    return out, meta


def load_bars(processed_dir: Path, freq: str = "1m") -> pl.DataFrame:
    out, mp = bars_path(processed_dir, freq), meta_path(processed_dir, freq)
    if not out.exists():
        raise FileNotFoundError(f"{out} missing; run `chudp resample --freq {freq}`")
    meta = json.loads(mp.read_text()) if mp.exists() else {}
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
