"""Model inputs: the target context, past-only covariates and known-future covariates.

Target (1 variate): `mid_close`, dollars in [0, 1].
Past-only covariates (K, C): every BRTI and contract bar column, directly or as a stationary
transform. Known-future covariates (W, C + 64): the window clock, time of day, and the fair value
path "if BRTI stayed where it is now" (computed from origin-time information only).

One builder feeds inference and fine-tuning, so the two paths see identical arrays. Missing values
are linearly interpolated here (what `TimesFM3Forecaster.predict_batch` would do), and every variate
is guaranteed to vary inside every 32-point input patch: TimesFM's per-variate running statistics
divide by a running sigma that is exactly 0 while a series has been flat since the start of the
context (a quoted spread sits at one cent for long stretches), which produces non-finite values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .features import RW_Z_CLIP, SETTLE_K, WINDOW_MINUTES, h_eff, norm_cdf

INPUT_PATCH = 32
MAX_STEPS = WINDOW_MINUTES
JITTER = 1e-5          # 1% of a 0.001 price tick; only ever added to a patch that is exactly flat

# name -> (expression on the frame, fill for nulls before interpolation)
PO_EXPRS: dict[str, tuple[pl.Expr, str]] = {
    "brti_log_close": (pl.col("brti_close").log(), "interp"),
    "brti_ret_1m": (pl.col("ret_1m"), "interp"),
    "brti_range": ((pl.col("brti_high") / pl.col("brti_low")).log(), "interp"),
    "brti_sigma_1m": (pl.col("sigma_1m"), "interp"),
    "brti_mean_dev": ((pl.col("brti_mean") / pl.col("brti_close")).log(), "interp"),
    "moneyness": (pl.col("moneyness"), "interp"),
    "rw_z": (pl.col("rw_z"), "interp"),
    "rw_p": (pl.col("rw_p"), "interp"),
    # the one-second family: realized vol, the fair value built on it, its distance from the price, the bar's last seconds
    "brti_sigma_rv": (pl.col("sigma_rv"), "interp"),
    "brti_vol_ratio": (pl.col("vol_ratio"), "interp"),
    "rw_z_rv": (pl.col("rw_z_rv"), "interp"),
    "rw_p_rv": (pl.col("rw_p_rv"), "interp"),
    "rw_gap_rv": (pl.col("rw_gap_rv"), "interp"),
    "brti_ret_l10": (pl.col("ret_l10_std"), "zero"),
    "brti_ret_l30": (pl.col("ret_l30_std"), "zero"),
    "yes_bid_close": (pl.col("yes_bid_close"), "interp"),
    "yes_ask_close": (pl.col("yes_ask_close"), "interp"),
    "spread": (pl.col("spread"), "interp"),
    "mid_high": (pl.col("mid_high"), "interp"),
    "mid_low": (pl.col("mid_low"), "interp"),
    "mid_ret_1m": (pl.col("mid_ret_1m"), "zero"),
    "trade_close_imp": (pl.col("trade_close_imp"), "interp"),
    "log_volume": (pl.col("log_volume"), "interp"),
    "log_oi": (pl.col("log_oi"), "interp"),
}
PO_BRTI = ("brti_log_close", "brti_ret_1m", "brti_range", "brti_sigma_1m", "brti_mean_dev", "moneyness", "rw_z", "rw_p")
PO_QUOTES = ("yes_bid_close", "yes_ask_close", "spread", "mid_high", "mid_low", "mid_ret_1m", "trade_close_imp", "log_volume", "log_oi")
PF_CALENDAR = ("k_frac", "k_sin", "k_cos", "sqrt_ttl", "tod_sin", "tod_cos")
PF_FAIR = ("rw_fair_path",)
PO_BRTI_RV = ("brti_log_close", "brti_ret_1m", "brti_range", "brti_sigma_rv", "brti_vol_ratio", "brti_mean_dev", "moneyness",
              "rw_z_rv", "rw_p_rv", "rw_gap_rv", "brti_ret_l10", "brti_ret_l30")
PF_FAIR_RV = ("rw_fair_path_rv",)   # the same path on sigma_rv / rw_p_rv

PRESETS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "full": (PO_BRTI + PO_QUOTES, PF_CALENDAR + PF_FAIR),
    "full_rv": (PO_BRTI_RV + PO_QUOTES, PF_CALENDAR + PF_FAIR_RV),
    "no_fair_path": (PO_BRTI + PO_QUOTES, PF_CALENDAR),
    "brti_only": (PO_BRTI, PF_CALENDAR + PF_FAIR),
    "quotes_only": (PO_QUOTES, PF_CALENDAR),
    "calendar_only": ((), PF_CALENDAR),
    "none": ((), ()),
}
MAX_VARIATES = 32


def _calendar(k: np.ndarray, minute_of_day: np.ndarray) -> dict[str, np.ndarray]:
    k = k.astype(np.float64)
    tod = 2 * np.pi * minute_of_day.astype(np.float64) / 1440.0
    return {
        "k_frac": k / SETTLE_K,
        "k_sin": np.sin(2 * np.pi * k / WINDOW_MINUTES),
        "k_cos": np.cos(2 * np.pi * k / WINDOW_MINUTES),
        "sqrt_ttl": np.sqrt(h_eff(k) / h_eff(0.0)),
        "tod_sin": np.sin(tod),
        "tod_cos": np.cos(tod),
    }


@dataclass
class FrameArrays:
    """Column arrays of the frame, evaluated once, from which batches are gathered by index."""

    preset: str
    po_names: tuple[str, ...]
    pf_names: tuple[str, ...]
    target: np.ndarray          # (n,) mid_close, NaN where no candle
    valid: np.ndarray           # (n,) bool
    po: np.ndarray              # (K, n)
    k: np.ndarray               # (n,) int
    minute_of_day: np.ndarray   # (n,) int
    rw_p: np.ndarray            # (n,)
    brti_close: np.ndarray
    brti_mean: np.ndarray
    strike: np.ndarray
    sigma: np.ndarray

    @property
    def n_variates(self) -> int:
        return 1 + len(self.po_names) + len(self.pf_names)


def frame_arrays(frame: pl.DataFrame, preset: str = "full") -> FrameArrays:
    if preset not in PRESETS:
        raise ValueError(f"unknown covariate preset {preset!r}; one of {sorted(PRESETS)}")
    po_names, pf_names = PRESETS[preset]
    if 1 + len(po_names) + len(pf_names) > MAX_VARIATES:
        raise ValueError("more than 32 variates: beyond the regime TimesFM 3.0 was trained in")

    def col(e: pl.Expr) -> np.ndarray:
        return frame.select(e.cast(pl.Float64).fill_nan(None).alias("x"))["x"].to_numpy().astype(np.float64)

    po = np.empty((len(po_names), frame.height))
    for j, name in enumerate(po_names):
        expr, fill = PO_EXPRS[name]
        x = col(expr)
        x[~np.isfinite(x)] = np.nan
        po[j] = np.nan_to_num(x, nan=0.0) if fill == "zero" else x
    rw_p, sigma = ("rw_p_rv", "sigma_rv") if PF_FAIR_RV[0] in pf_names else ("rw_p", "sigma_1m")
    return FrameArrays(
        preset=preset, po_names=po_names, pf_names=pf_names,
        target=col(pl.col("mid_close")), valid=frame["target_valid"].to_numpy().astype(bool), po=po,
        k=frame["k"].to_numpy().astype(np.int64), minute_of_day=frame["minute_of_day"].to_numpy().astype(np.int64),
        rw_p=col(pl.col(rw_p)), brti_close=col(pl.col("brti_close")), brti_mean=col(pl.col("brti_mean")),
        strike=col(pl.col("strike")), sigma=col(pl.col(sigma)),
    )


def interpolate_rows(x: np.ndarray, keep_leading_nan: bool = False) -> np.ndarray:
    """Linear interpolation of NaNs along the last axis, in place on a copy. Edges are held; an
    all-NaN row becomes 0. With keep_leading_nan the leading run stays NaN (the target: TimesFM
    masks a context's leading gap rather than inventing values for it)."""
    x = np.array(x, dtype=np.float64, order="C", copy=True)   # C order: reshape below must be a view, not a copy
    flat = x.reshape(-1, x.shape[-1])
    pos = np.arange(flat.shape[1])
    for r in np.flatnonzero(np.isnan(flat).any(axis=1)):
        row = flat[r]
        ok = ~np.isnan(row)
        if not ok.any():
            row[:] = np.nan if keep_leading_nan else 0.0
            continue
        first = int(np.argmax(ok))
        filled = np.interp(pos, pos[ok], row[ok])
        if keep_leading_nan:
            filled[:first] = np.nan
        flat[r] = filled
    return x


_PATTERN_CACHE: dict[int, np.ndarray] = {}


def _jitter_pattern(length: int) -> np.ndarray:
    if length not in _PATTERN_CACHE:
        _PATTERN_CACHE[length] = np.random.default_rng(1234).standard_normal(length) * JITTER
    return _PATTERN_CACHE[length]


def sanitize_variates(x: np.ndarray, patch: int = INPUT_PATCH) -> tuple[np.ndarray, int]:
    """Make every (.., L) row finite and non-constant inside every `patch`-point block. A flat block
    gets a fixed, seeded jitter far below the price tick; returns (float32 array, blocks touched).
    Flatness is judged in float32, the precision the model receives: a fair-value probability that
    saturates near 1 varies in float64 and is constant in float32. NaNs (only a target's leading
    gap) are left alone and do not count as flat."""
    x = np.array(x, dtype=np.float32, order="C", copy=True)
    L = x.shape[-1]
    if L % patch:
        raise ValueError(f"length {L} is not a multiple of the input patch {patch}")
    blocks = x.reshape(*x.shape[:-1], L // patch, patch)
    with np.errstate(invalid="ignore"):
        flat = (blocks.max(axis=-1) - blocks.min(axis=-1)) == 0.0   # NaN blocks compare False
    if flat.any():
        pattern = _jitter_pattern(L).reshape(L // patch, patch).astype(np.float32)
        blocks += np.where(flat[..., None], np.broadcast_to(pattern, blocks.shape), np.float32(0.0))
    return x, int(flat.sum())


def build_arrays(fa: FrameArrays, i: np.ndarray, context: int, horizon: int = 64) -> dict[str, np.ndarray]:
    """Batch for origins with context-end rows `i`.

    targets (n, C) f32 with a leading NaN run left in place, mask (n, C) bool marking it,
    po (n, K, C) f32 or None, pf (n, W, C + horizon) f32 or None,
    tgt (n, 15) f64 realised mid per step, valid (n, 15) bool (step <= 15 - m and a real quote),
    step_k (n, 15) the k of each step, n_flat: blocks that needed jitter."""
    i = np.asarray(i, dtype=np.int64)
    n, C, H = i.size, context, horizon
    n_rows = fa.target.size
    if i.min() - C + 1 < 0:
        raise IndexError("context window starts before the first row")
    ctx = i[:, None] + np.arange(-C + 1, 1)[None, :]                    # (n, C)
    n_flat = 0

    targets = interpolate_rows(fa.target[ctx], keep_leading_nan=True)
    mask = np.isnan(targets)
    targets, f = sanitize_variates(targets)
    n_flat += f

    po = None
    if fa.po_names:
        po = interpolate_rows(np.transpose(fa.po[:, ctx], (1, 0, 2)))   # (n, K, C)
        po, f = sanitize_variates(po)
        n_flat += f

    k_i, mod_i = fa.k[i], fa.minute_of_day[i]
    m = (k_i + 1) % WINDOW_MINUTES
    n_steps = WINDOW_MINUTES - m
    pf = None
    if fa.pf_names:
        steps = np.arange(1, H + 1)[None, :]
        k_fut = (k_i[:, None] + steps) % WINDOW_MINUTES                  # (n, H)
        mod_fut = (mod_i[:, None] + steps) % 1440
        cal_ctx, cal_fut = _calendar(fa.k[ctx], fa.minute_of_day[ctx]), _calendar(k_fut, mod_fut)
        parts = []
        for name in fa.pf_names:
            if name in PF_FAIR + PF_FAIR_RV:
                # fair value if BRTI stays at its last close; strike as known at the origin (rows <= i only)
                strike = np.where(m == 0, fa.brti_mean[i], fa.strike[i])
                with np.errstate(divide="ignore", invalid="ignore"):
                    z = np.log(fa.brti_close[i] / strike)[:, None] / (fa.sigma[i][:, None] * np.sqrt(h_eff(k_fut.astype(np.float64))))
                fut = norm_cdf(np.clip(np.nan_to_num(z, nan=0.0), -RW_Z_CLIP, RW_Z_CLIP))
                beyond = steps > n_steps[:, None]                          # next contract: hold the expiry value
                last_in = fut[np.arange(n), n_steps - 1][:, None]
                fut = np.where(beyond, last_in, fut)
                parts.append(np.concatenate([interpolate_rows(fa.rw_p[ctx]), fut], axis=1))
            else:
                parts.append(np.concatenate([cal_ctx[name], cal_fut[name]], axis=1))
        pf, f = sanitize_variates(np.stack(parts, axis=1))               # (n, W, C + H)
        n_flat += f

    rows = i[:, None] + np.arange(1, MAX_STEPS + 1)[None, :]            # (n, 15)
    inside = rows < n_rows
    safe = np.minimum(rows, n_rows - 1)
    within = np.arange(1, MAX_STEPS + 1)[None, :] <= n_steps[:, None]
    valid = inside & within & fa.valid[safe]
    return {
        "targets": targets.astype(np.float32), "mask": mask,
        "po": None if po is None else po.astype(np.float32),
        "pf": None if pf is None else pf.astype(np.float32),
        "tgt": np.where(valid, fa.target[safe], np.nan), "valid": valid,
        "step_k": (m[:, None] + np.arange(MAX_STEPS)[None, :]),
        "n_flat": np.int64(n_flat),
    }
