"""Kalshi 15-minute windows and per-minute forecast origins.

Index arithmetic (bars are closed-left, labelled by start; idx(t) = grid row of bar labelled t):

    j(m)     = idx(T0 + (m-1) min)          last context bar; m=0 -> bar labelled T0-1m
    context  = bars[j(m)-C+1 : j(m)+1]      C rows; the last tick inside it is at T0+m-1s
    target   = idx(T0 + 14 min) = j(m) + (15-m)
    h_bars   = 15 - m  (1..15);  step = 14 - m  (index into the 64-step forecast)

Windows are returned as polars frames (one row per window / per origin) rather than object lists:
a year of history is 35k windows and 525k origins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from .resample import freq_seconds

log = logging.getLogger(__name__)

WINDOW_MINUTES = 15
SETTLE_OFFSET_MIN = 14
MAX_TIMESFM_CONTEXT = 15_360
STRIKE_MODES = ("open_tick", "open_avg60")


def parse_minutes(spec: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """'0-14' | '0,5,10,14' | '3-5,14' -> sorted unique tuple within 0..14."""
    if not isinstance(spec, str):
        vals = set(int(v) for v in spec)
    else:
        vals: set[int] = set()
        for part in spec.replace(" ", "").split(","):
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                vals.update(range(int(a), int(b) + 1))
            else:
                vals.add(int(part))
    out = tuple(sorted(vals))
    if not out or out[0] < 0 or out[-1] >= WINDOW_MINUTES:
        raise ValueError(f"minutes must be within 0..{WINDOW_MINUTES - 1}, got {spec!r}")
    return out


@dataclass(frozen=True)
class KalshiSpec:
    freq: str = "1m"
    context: int = 4096
    horizon: int = 64
    target_col: str = "mean"
    transform: str = "log"
    strike_mode: str = "open_tick"
    minutes: tuple[int, ...] = field(default_factory=lambda: tuple(range(WINDOW_MINUTES)))
    min_settle_ticks: int = 55
    max_ctx_gap_frac: float = 0.01
    max_strike_stale_s: float = 5.0
    vol_lookback: int = 240

    def __post_init__(self) -> None:
        if self.freq != "1m":
            raise ValueError("Kalshi windows require 1-minute bars (freq='1m')")
        if self.strike_mode not in STRIKE_MODES:
            raise ValueError(f"strike_mode must be one of {STRIKE_MODES}")
        if self.context > MAX_TIMESFM_CONTEXT:
            log.warning(
                "context %d exceeds TimesFM's %d-point limit; it would be truncated silently, "
                "clamping to %d", self.context, MAX_TIMESFM_CONTEXT, MAX_TIMESFM_CONTEXT
            )
            object.__setattr__(self, "context", MAX_TIMESFM_CONTEXT)
        if self.horizon < WINDOW_MINUTES:
            raise ValueError("horizon must cover at least 15 bars")
        object.__setattr__(self, "minutes", parse_minutes(self.minutes))

    def with_strike_mode(self, mode: str) -> KalshiSpec:
        return replace(self, strike_mode=mode)


@dataclass
class KalshiWindows:
    spec: KalshiSpec
    windows: pl.DataFrame   # one row per window (ok or dropped)
    origins: pl.DataFrame   # one row per (t0, m) of ok windows (ok or dropped)
    features: pl.DataFrame  # bars + derived per-bar features (same row index as bars)

    @property
    def ok_windows(self) -> pl.DataFrame:
        return self.windows.filter(pl.col("ok"))

    @property
    def ok_origins(self) -> pl.DataFrame:
        return self.origins.filter(pl.col("ok"))

    def drop_report(self) -> pl.DataFrame:
        w = self.windows.filter(~pl.col("ok")).group_by("drop_reason").len().with_columns(pl.lit("window").alias("level"))
        o = self.origins.filter(~pl.col("ok")).group_by("drop_reason").len().with_columns(pl.lit("origin").alias("level"))
        return pl.concat([w, o]).select("level", "drop_reason", "len").sort("level", "drop_reason")


# ---------------------------------------------------------------------------------------------
# per-bar features

def bar_features(bars: pl.DataFrame, spec: KalshiSpec) -> pl.DataFrame:
    """Add log target, trailing 1m realized vol and the 1-step MASE scale to the bar frame."""
    if spec.target_col not in bars.columns:
        raise ValueError(f"target column {spec.target_col!r} not in bars")
    close_ff = pl.col("close").forward_fill()
    target = pl.col(spec.target_col)
    log_target = target.log() if spec.transform == "log" else target
    out = bars.with_columns(
        pl.int_range(0, pl.len()).alias("idx"),
        log_target.alias("y"),                                   # model-space target (may be null)
        close_ff.log().diff().alias("ret_1m"),
    )
    out = out.with_columns(
        pl.col("ret_1m").rolling_std(window_size=spec.vol_lookback, min_samples=max(10, spec.vol_lookback // 4)).alias("sigma_1m"),
        # price-space 1-bar absolute change, so mase_1s is comparable to mae (both in USD)
        target.diff().abs().rolling_mean(window_size=spec.context, min_samples=max(10, spec.context // 4)).alias("mase_scale_1step"),
        pl.col("y").is_null().cast(pl.Int64).cum_sum().alias("_null_cum"),  # inclusive cumsum
    )
    return out


# ---------------------------------------------------------------------------------------------
# strikes from raw ticks

def strike_ticks(ticks: pl.LazyFrame, t0s: pl.Series, max_stale_s: float) -> pl.DataFrame:
    """For each t0: the tick at exactly t0, else the last tick within `max_stale_s` before it."""
    stale = int(min(max(max_stale_s, 0), 59))
    ts = pl.col("ts")
    cand = (
        ticks.filter(
            ((ts.dt.minute() % 15 == 0) & (ts.dt.second() == 0))
            | ((ts.dt.minute() % 15 == 14) & (ts.dt.second() >= 60 - stale))
        )
        .select("ts", "value")
        .collect()
        .sort("ts")
    )
    left = pl.DataFrame({"t0": t0s}).sort("t0")
    joined = left.join_asof(
        cand.rename({"ts": "strike_ts", "value": "strike_tick"}),
        left_on="t0", right_on="strike_ts", strategy="backward", tolerance=f"{stale}s",
    )
    return joined.with_columns(
        ((pl.col("t0") - pl.col("strike_ts")).dt.total_microseconds() / 1e6).alias("strike_stale_s")
    )


# ---------------------------------------------------------------------------------------------
# windows

def _align_up(t: datetime, minutes: int = WINDOW_MINUTES) -> datetime:
    t = t.replace(second=0, microsecond=0)
    rem = t.minute % minutes
    return t if rem == 0 else t + timedelta(minutes=minutes - rem)


def make_kalshi_windows(
    bars: pl.DataFrame,
    ticks: pl.LazyFrame | None,
    spec: KalshiSpec,
    start: date | None = None,
    end: date | None = None,
    stride_windows: int = 1,
    max_windows: int | None = None,
) -> KalshiWindows:
    step_s = freq_seconds(spec.freq)
    feats = bar_features(bars, spec)
    n = feats.height
    first_ts: datetime = feats["ts"][0]
    C = spec.context

    # -- enumerate T0 on the 15-minute grid ---------------------------------------------------
    earliest = first_ts + timedelta(seconds=step_s * C)  # need C bars strictly before T0
    t0 = _align_up(earliest)
    if start:
        t0 = max(t0, _align_up(datetime(start.year, start.month, start.day)))
    last_ts: datetime = feats["ts"][n - 1]
    t0_max = last_ts - timedelta(minutes=SETTLE_OFFSET_MIN)  # target bar must exist
    if end:
        t0_max = min(t0_max, datetime(end.year, end.month, end.day) + timedelta(days=1) - timedelta(minutes=WINDOW_MINUTES))
    if t0 > t0_max:
        raise ValueError(f"no complete windows in range (need {C} bars of history before the first window)")
    t0s = pl.datetime_range(t0, t0_max, interval=f"{WINDOW_MINUTES}m", eager=True, time_unit="us")
    if stride_windows > 1:
        t0s = t0s.gather_every(stride_windows)
    if max_windows is not None:
        t0s = t0s.head(max_windows)

    def idx_of(col: pl.Expr) -> pl.Expr:
        return ((col - pl.lit(first_ts)).dt.total_seconds() // step_s).cast(pl.Int64)

    w = pl.DataFrame({"t0": t0s}).with_columns(
        idx_of(pl.col("t0")).alias("t0_idx"),
    ).with_columns(
        (pl.col("t0_idx") + SETTLE_OFFSET_MIN).alias("target_idx"),
        (pl.col("t0_idx") - 1).alias("prev_idx"),
    )

    # -- settlement / open_avg60 / vol from the bar frame -------------------------------------
    take = feats.select("idx", "mean", "n_ticks", "is_gap", "sigma_1m")
    w = w.join(
        take.rename({"idx": "target_idx", "mean": "settlement", "n_ticks": "settlement_n_ticks", "is_gap": "settle_gap"}).drop("sigma_1m"),
        on="target_idx", how="left",
    ).join(
        take.select("idx", "mean", "sigma_1m").rename({"idx": "prev_idx", "mean": "open_avg60", "sigma_1m": "vol_1m"}),
        on="prev_idx", how="left",
    )

    # -- strike ------------------------------------------------------------------------------
    if spec.strike_mode == "open_tick":
        if ticks is None:
            raise ValueError("strike_mode='open_tick' needs the raw ticks")
        st = strike_ticks(ticks, w["t0"], spec.max_strike_stale_s)
        w = w.join(st, on="t0", how="left").with_columns(pl.col("strike_tick").alias("strike"))
    else:
        w = w.with_columns(
            pl.col("open_avg60").alias("strike"),
            (pl.col("t0") - pl.duration(minutes=1)).alias("strike_ts"),
            pl.lit(60.0).alias("strike_stale_s"),
        )
    w = w.with_columns(pl.lit(spec.strike_mode).alias("strike_mode"))

    # -- labels and drops --------------------------------------------------------------------
    w = w.with_columns(
        (pl.col("settlement") > pl.col("strike")).alias("label_up"),
        (pl.col("settlement") == pl.col("strike")).alias("tie"),
    ).with_columns(
        pl.when(pl.col("t0_idx") - 1 - C + 1 < 0).then(pl.lit("insufficient_history"))
        .when(pl.col("target_idx") >= n).then(pl.lit("no_target_bar"))
        .when(pl.col("strike").is_null()).then(pl.lit("no_strike_tick"))
        .when(pl.col("settle_gap").fill_null(True)).then(pl.lit("settlement_gap"))
        .when(pl.col("settlement_n_ticks") < spec.min_settle_ticks).then(pl.lit("thin_settlement_bar"))
        .otherwise(pl.lit(None, dtype=pl.Utf8)).alias("drop_reason")
    ).with_columns(pl.col("drop_reason").is_null().alias("ok"))
    w = w.with_columns(pl.col("t0").dt.date().alias("date"), pl.col("t0").dt.hour().cast(pl.Int8).alias("hour"))

    # -- origins -----------------------------------------------------------------------------
    ok_w = w.filter(pl.col("ok")).select("t0", "t0_idx", "target_idx")
    o = ok_w.join(pl.DataFrame({"m": list(spec.minutes)}, schema={"m": pl.Int8}), how="cross")
    o = o.with_columns(
        (pl.col("t0_idx") + pl.col("m").cast(pl.Int64) - 1).alias("ctx_end_idx"),
        (pl.col("t0") + pl.duration(minutes=pl.col("m"))).alias("origin_ts"),
        (WINDOW_MINUTES - pl.col("m")).cast(pl.Int8).alias("h_bars"),
        (SETTLE_OFFSET_MIN - pl.col("m")).cast(pl.Int8).alias("step"),
    )
    nulls = feats.select("idx", "_null_cum", "close", "mean", "sigma_1m", "mase_scale_1step", "ts")
    o = o.join(
        nulls.rename({"idx": "ctx_end_idx", "_null_cum": "_cum_end", "close": "last_close", "mean": "last_mean",
                      "sigma_1m": "sigma_1m", "ts": "context_end_ts"}),
        on="ctx_end_idx", how="left",
    )
    # nulls in [ctx_end-C+1, ctx_end] = cum[ctx_end] - cum[ctx_end-C] (cum is inclusive)
    o = o.with_columns((pl.col("ctx_end_idx") - C).alias("_before_idx"))
    o = o.join(
        feats.select(pl.col("idx").alias("_before_idx"), pl.col("_null_cum").alias("_cum_before")),
        on="_before_idx", how="left",
    ).with_columns(
        (pl.col("_cum_end") - pl.col("_cum_before").fill_null(0)).alias("n_ctx_gaps")
    ).drop("_cum_end", "_cum_before", "_before_idx")
    o = o.with_columns(
        pl.when(pl.col("n_ctx_gaps") > spec.max_ctx_gap_frac * C).then(pl.lit("ctx_gaps"))
        .when(pl.col("last_close").is_null()).then(pl.lit("ctx_end_gap"))
        .otherwise(pl.lit(None, dtype=pl.Utf8)).alias("drop_reason")
    ).with_columns(pl.col("drop_reason").is_null().alias("ok"))

    # always-on invariants
    ok_o = o.filter(pl.col("ok"))
    if ok_o.height:
        assert (ok_o["target_idx"] == ok_o["ctx_end_idx"] + ok_o["h_bars"].cast(pl.Int64)).all()
        assert (ok_o["step"] + 1 == ok_o["h_bars"]).all() and int(ok_o["step"].max()) < spec.horizon
        assert ok_o.select(((pl.col("context_end_ts") + pl.duration(minutes=1)) <= pl.col("origin_ts")).all()).item()

    return KalshiWindows(spec=spec, windows=w, origins=o.sort("t0", "m"), features=feats)


def context_matrix(features: pl.DataFrame, ctx_end_idx: np.ndarray, context: int) -> np.ndarray:
    """(n, context) float32 matrix of model-space targets ending at each ctx_end_idx (inclusive).
    Nulls become NaN (TimesFM interpolates interior NaNs)."""
    y = features["y"].to_numpy().astype(np.float64)
    y = np.where(np.isnan(y), np.nan, y)
    offsets = np.arange(-context + 1, 1, dtype=np.int64)
    idx = ctx_end_idx.astype(np.int64)[:, None] + offsets[None, :]
    if idx.min() < 0:
        raise IndexError("context window starts before the first bar")
    return y[idx].astype(np.float32)
