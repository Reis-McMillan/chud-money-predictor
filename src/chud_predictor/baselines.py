"""Baselines scored on exactly the rows the model is scored on.

1. Persistence. The contract price is close to a martingale, so "the price stays where it is" is
   the efficient-market forecast of the MEAN and the benchmark that matters. Its reference level is
   the last observed mid, except at the window open (m = 0), where the last observed mid belongs to
   the contract that just settled at 0 or 1: a new contract opens at the money (its strike is the
   index mean of the minute before), so the reference there is 0.50.

   A price that ends at 0 or 1 is bimodal, so the martingale mean is a poor MEDIAN and says nothing
   about spread. The probabilistic persistence baseline is therefore an empirical fan: deciles of
   the realised change (future mid - reference), by minute of the window, steps ahead and the
   price level (ten 10-cent buckets), fitted on data strictly before the evaluation period. Cells
   with too little history fall back to the pooled (minute, step) cell, then to a point mass.

2. RW fair value: P(settle above strike) from the index alone, Phi(log(brti/strike) / (sigma *
   sqrt((14 - m) + 1/3))). As an expected payoff it is the same number for every step ahead.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from .features import RW_Z_CLIP, SETTLE_K, WINDOW_MINUTES, norm_cdf

QUANTILE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
FAN_COLS = [f"dq{int(q * 100)}" for q in QUANTILE_LEVELS]
MIN_FAN_N = 30
N_BUCKETS = 10
POOLED = -1
OPEN_PRICE = 0.5


def persist_ref(last_mid: np.ndarray, k_ctx: np.ndarray) -> np.ndarray:
    """Reference level of the persistence forecast: the last mid, or 0.50 at the window open."""
    return np.where(np.asarray(k_ctx) == SETTLE_K, OPEN_PRICE, np.asarray(last_mid, dtype=np.float64))


def price_bucket(ref: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(np.nan_to_num(np.asarray(ref, dtype=np.float64), nan=0.5) * N_BUCKETS), 0, N_BUCKETS - 1).astype(np.int64)


def persistence_fan(frame: pl.DataFrame, lo: datetime | None = None, hi: datetime | None = None) -> pl.DataFrame:
    """Deciles of mid[i+h] - reference[i] for context-end rows with lo <= ts < hi, steps inside the
    contract only. Columns: k_ctx, h, bucket (-1 = pooled over price levels), n, dq10..dq90."""
    mid = frame["mid_close"].to_numpy().astype(np.float64)
    k = frame["k"].to_numpy().astype(np.int64)
    ts = frame["ts"]
    sel = np.ones(frame.height, dtype=bool)
    if lo is not None:
        sel &= (ts >= lo).to_numpy()
    if hi is not None:
        sel &= (ts < hi).to_numpy()
    ref = persist_ref(mid, k)
    bucket = price_bucket(ref)
    n_steps = WINDOW_MINUTES - (k + 1) % WINDOW_MINUTES
    n = frame.height
    rows = []
    for h in range(1, WINDOW_MINUTES + 1):
        i = np.flatnonzero(sel & (n_steps >= h) & (np.arange(n) + h < n))
        d = mid[i + h] - ref[i]
        ok = np.isfinite(d)
        i, d = i[ok], d[ok]
        for kc in range(WINDOW_MINUTES):
            in_k = k[i] == kc
            if not in_k.any():
                continue
            rows.append((kc, h, POOLED, int(in_k.sum()), *np.quantile(d[in_k], QUANTILE_LEVELS).tolist()))
            for b in np.unique(bucket[i[in_k]]):
                db = d[in_k & (bucket[i] == b)]
                rows.append((kc, h, int(b), int(db.size), *np.quantile(db, QUANTILE_LEVELS).tolist()))
    schema = {"k_ctx": pl.Int8, "h": pl.Int8, "bucket": pl.Int8, "n": pl.Int64} | {c: pl.Float64 for c in FAN_COLS}
    return pl.DataFrame(rows, schema=schema, orient="row")


def fan_cells(fan: pl.DataFrame | None) -> int:
    """Number of pooled (k_ctx, h) cells with enough observations to be used (120 = the full triangle)."""
    if fan is None or fan.is_empty():
        return 0
    return int(fan.filter((pl.col("bucket") == POOLED) & (pl.col("n") >= MIN_FAN_N)).height)


def apply_fan(last_mid: np.ndarray, k_ctx: np.ndarray, h: np.ndarray, fan: pl.DataFrame | None) -> np.ndarray:
    """(n, 9) persistence deciles, clipped to [0, 1]: the level-conditional cell where it has enough
    history, else the pooled cell, else a point mass at the reference (pinball = |error| / 2)."""
    k_ctx, h = np.asarray(k_ctx, dtype=np.int64), np.asarray(h, dtype=np.int64)
    ref = persist_ref(last_mid, k_ctx)
    table = np.zeros((WINDOW_MINUTES, WINDOW_MINUTES + 1, N_BUCKETS, len(FAN_COLS)))
    if fan is not None and fan.height:
        use = fan.filter(pl.col("n") >= MIN_FAN_N)
        pooled = use.filter(pl.col("bucket") == POOLED)
        table[pooled["k_ctx"].to_numpy(), pooled["h"].to_numpy()] = pooled.select(FAN_COLS).to_numpy()[:, None, :]
        lvl = use.filter(pl.col("bucket") != POOLED)
        table[lvl["k_ctx"].to_numpy(), lvl["h"].to_numpy(), lvl["bucket"].to_numpy()] = lvl.select(FAN_COLS).to_numpy()
    q = ref[:, None] + table[k_ctx, h, price_bucket(ref)]
    return np.sort(np.clip(q, 0.0, 1.0), axis=-1)


def rw_fair_now(brti_close: np.ndarray, strike: np.ndarray, sigma_1m: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Index-only fair value at an origin m minutes into the window."""
    sigma = np.asarray(sigma_1m, dtype=np.float64)
    ok = np.isfinite(sigma) & (sigma > 0)
    h = (WINDOW_MINUTES - 1 - np.asarray(m, dtype=np.float64)) + 1.0 / 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.log(np.asarray(brti_close, dtype=np.float64) / np.asarray(strike, dtype=np.float64)) / (np.where(ok, sigma, 1.0) * np.sqrt(h))
    z = np.where(ok & np.isfinite(z), z, 0.0)
    return norm_cdf(np.clip(z, -RW_Z_CLIP, RW_Z_CLIP))
