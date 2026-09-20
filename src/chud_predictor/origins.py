"""Forecast origins on the joined minute frame.

Every bar is the context end of exactly one origin. With i the row index, k(i) = minute mod 15:

    m           = (k(i) + 1) mod 15        minutes since the window opened, at the origin
    origin_ts   = ts_i + 1m                wall clock of the forecast = T0 + m
    t0          = truncate15(origin_ts)
    context     = rows i-C+1 .. i          everything observable at origin_ts
    n_steps     = 15 - m                   candles left in this contract
    step h      = 1 .. n_steps  ->  row i + h, whose k is m + h - 1
    settlement  = the step with k == 14, i.e. h = n_steps

Steps beyond n_steps belong to the next contract; the model still emits them (one 64-step patch)
and they are dropped everywhere. m = 0 is a genuine origin: at T0 the strike (the mean of the
minute that just ended, `brti_mean[i]`) and all of BRTI are known; the last observed price belongs
to the contract that just settled.

The same origin population feeds the backtest and fine-tuning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from .features import SETTLE_K, WINDOW_MINUTES

log = logging.getLogger(__name__)

MAX_TIMESFM_CONTEXT = 15_360
INPUT_PATCH = 32
OUTPUT_PATCH = 64


def parse_minutes(spec: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """'0-14' | '0,5,10,14' | '3-5,14' -> sorted unique tuple within 0..14."""
    if not isinstance(spec, str):
        vals = {int(v) for v in spec}
    else:
        vals = set()
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
class ContractSpec:
    freq: str = "1m"
    context: int = 1024
    horizon: int = OUTPUT_PATCH          # what the model emits; only the first 15 - m steps are kept
    covariates: str = "full"             # preset in covariates.PRESETS
    max_ctx_gap_frac: float = 0.01
    minutes: tuple[int, ...] = field(default_factory=lambda: tuple(range(WINDOW_MINUTES)))
    require_quote_ok: bool = False

    def __post_init__(self) -> None:
        if self.freq != "1m":
            raise ValueError("contract windows require 1-minute rows (freq='1m')")
        if self.context % INPUT_PATCH or not 0 < self.context <= MAX_TIMESFM_CONTEXT:
            raise ValueError(f"context must be a positive multiple of {INPUT_PATCH} and at most {MAX_TIMESFM_CONTEXT}")
        if self.horizon % OUTPUT_PATCH or self.horizon < WINDOW_MINUTES:
            raise ValueError(f"horizon must be a multiple of {OUTPUT_PATCH}")
        object.__setattr__(self, "minutes", parse_minutes(self.minutes))


@dataclass
class OriginSet:
    spec: ContractSpec
    frame: pl.DataFrame
    origins: pl.DataFrame      # one row per candidate origin, with ok / drop_reason

    @property
    def ok_origins(self) -> pl.DataFrame:
        return self.origins.filter(pl.col("ok"))

    def drop_report(self) -> pl.DataFrame:
        return self.origins.filter(~pl.col("ok")).group_by("drop_reason").len().sort("drop_reason")


def _cum(flag: np.ndarray) -> np.ndarray:
    """cum[j] = number of True in flag[:j]."""
    return np.concatenate([[0], np.cumsum(flag.astype(np.int64))])


def make_origins(
    frame: pl.DataFrame,
    spec: ContractSpec,
    start: date | None = None,
    end: date | None = None,
    stride_windows: int = 1,
    max_windows: int | None = None,
) -> OriginSet:
    n, C = frame.height, spec.context
    ts = frame["ts"]
    k = frame["k"].to_numpy().astype(np.int64)
    valid = frame["target_valid"].to_numpy().astype(bool)
    quote_ok = frame["quote_ok"].fill_null(False).to_numpy().astype(bool)
    sigma = frame["sigma_1m"].to_numpy()
    strike_row = frame["strike"].to_numpy()
    brti_mean = frame["brti_mean"].to_numpy()

    i = np.arange(n, dtype=np.int64)
    m = (k + 1) % WINDOW_MINUTES
    n_steps = WINDOW_MINUTES - m
    # the strike the forecaster knows at origin_ts, read from rows <= i only: at m = 0 the new window's
    # strike is the mean of the minute that just ended
    strike = np.where(m == 0, brti_mean, strike_row)

    cum_invalid, cum_valid = _cum(~valid), _cum(valid)
    ctx_lo = i - C + 1
    has_hist = ctx_lo >= 0
    lo = np.clip(ctx_lo, 0, n)
    n_ctx_gaps = cum_invalid[i + 1] - cum_invalid[lo]
    tgt_hi = np.minimum(i + n_steps, n - 1)
    in_data = i + n_steps <= n - 1
    n_valid_steps = cum_valid[tgt_hi + 1] - cum_valid[np.minimum(i + 1, n)]

    reason = np.full(n, None, dtype=object)

    def mark(cond: np.ndarray, name: str) -> None:
        reason[(reason == None) & cond] = name  # noqa: E711 - elementwise comparison on an object array

    mark(~has_hist, "insufficient_history")
    mark(~in_data, "horizon_beyond_data")
    mark(~valid, "no_contract")
    mark(has_hist & ~valid[lo], "ctx_start_gap")   # a masked first patch has zero running variance (NaN gradients)
    mark(n_ctx_gaps > spec.max_ctx_gap_frac * C, "ctx_gaps")
    mark(~(np.isfinite(sigma) & (np.nan_to_num(sigma) > 0)), "no_sigma")
    mark(~np.isfinite(strike), "no_strike")
    mark(n_valid_steps == 0, "no_valid_step")
    if spec.require_quote_ok:
        mark(~quote_ok, "degenerate_quote")

    o = pl.DataFrame({"i": i, "m": m.astype(np.int8), "n_steps": n_steps.astype(np.int8), "k_ctx": k.astype(np.int8),
                      "n_ctx_gaps": n_ctx_gaps, "strike": strike}).with_columns(
        (ts + pl.duration(minutes=1)).alias("origin_ts"),
        pl.Series("drop_reason", reason.tolist(), dtype=pl.Utf8),
        frame["mid_close"].alias("last_mid"), frame["yes_bid_close"].alias("last_bid"), frame["yes_ask_close"].alias("last_ask"),
        frame["spread"].alias("last_spread"), frame["brti_close"], frame["sigma_1m"], pl.Series("quote_ok", quote_ok),
    ).with_columns(
        pl.col("origin_ts").dt.truncate("15m").alias("t0"),
        pl.col("drop_reason").is_null().alias("ok"),
    ).with_columns(pl.col("t0").dt.date().alias("date"), pl.col("t0").dt.hour().cast(pl.Int8).alias("hour"))

    # selection (not a quality drop): minutes, date range, window stride / cap
    o = o.filter(pl.col("m").is_in(list(spec.minutes)))
    if start:
        o = o.filter(pl.col("t0") >= datetime(start.year, start.month, start.day))
    if end:
        o = o.filter(pl.col("t0") < datetime(end.year, end.month, end.day) + timedelta(days=1))
    if stride_windows > 1:
        o = o.filter((pl.col("t0").dt.epoch("s") // (60 * WINDOW_MINUTES)) % stride_windows == 0)
    if max_windows is not None:
        keep = o.filter(pl.col("ok"))["t0"].unique().sort().head(max_windows)
        o = o.filter(pl.col("t0").is_in(keep.implode()))

    ok = o.filter(pl.col("ok"))
    if ok.height:  # always-on invariants
        assert (ok["n_steps"] == WINDOW_MINUTES - ok["m"]).all()
        assert ok.select((pl.col("origin_ts") == pl.col("t0") + pl.duration(minutes=pl.col("m"))).all()).item()
        settle_k = frame["k"].gather(ok["i"] + ok["n_steps"].cast(pl.Int64))
        assert (settle_k == SETTLE_K).all()
    return OriginSet(spec=spec, frame=frame, origins=o.sort("i"))
