"""Metrics for contract-price forecasts, in cents, always next to the persistence baseline.

The mid is close to a martingale, so absolute error says little on its own. The numbers to read are
    skill_mse     = 1 - MSE(mean of the model's deciles) / MSE(persistence reference)
    skill_pinball = 1 - pinball(model) / pinball(empirical, level-conditional persistence fan)
on identical rows (skill_mae compares the two medians; see score_group), with day-block bootstrap intervals (origins inside a day overlap heavily).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

QUANTILE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
QUANTILE_COLS = [f"q{int(q * 100)}" for q in QUANTILE_LEVELS]
FAN_Q_COLS = [f"fan_q{int(q * 100)}" for q in QUANTILE_LEVELS]
CAL_EDGES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0 + 1e-9)
EPS = 1e-4


def pinball(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean pinball loss over the nine deciles. q (n, 9), y (n,) -> (n,), in the units of y.
    A point mass (all quantiles equal) scores 0.5 * |error|."""
    err = np.asarray(y, dtype=np.float64)[:, None] - np.asarray(q, dtype=np.float64)
    return np.maximum(QUANTILE_LEVELS * err, (QUANTILE_LEVELS - 1.0) * err).mean(axis=-1)


def pwl_exp_cdf(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """CDF at x from nine sorted deciles: linear between the knots (q_k, k/10), exponential tails
    whose slope matches the outer segment. q (n, 9), x (n,) -> (n,)."""
    q = np.sort(np.asarray(q, dtype=np.float64), axis=1)
    x = np.asarray(x, dtype=np.float64)
    spread = np.maximum(q[:, 8] - q[:, 0], 1e-12)
    lam_lo = np.where(q[:, 1] - q[:, 0] > 0, q[:, 1] - q[:, 0], spread / 8.0)
    lam_hi = np.where(q[:, 8] - q[:, 7] > 0, q[:, 8] - q[:, 7], spread / 8.0)
    below, above = x < q[:, 0], x > q[:, 8]
    F = np.full(q.shape[0], np.nan)
    F = np.where(below, 0.1 * np.exp(np.minimum((x - q[:, 0]) / lam_lo, 0.0)), F)   # min(): np.where evaluates both branches
    F = np.where(above, 1.0 - 0.1 * np.exp(np.minimum(-(x - q[:, 8]) / lam_hi, 0.0)), F)
    mid = ~(below | above)
    for k in range(8):
        lo, hi = q[:, k], q[:, k + 1]
        seg = mid & (x >= lo) & (x <= hi) & np.isnan(F)
        width = hi - lo
        frac = np.where(width > 0, (x - lo) / np.where(width > 0, width, 1.0), 0.0)
        F = np.where(seg, (k + 1) / 10.0 + 0.1 * frac, F)
    return np.clip(F, 0.0, 1.0)


def pit(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Probability integral transform; uniform on [0, 1] for a calibrated forecast."""
    return pwl_exp_cdf(q, y)


def ks_uniform(u: np.ndarray) -> float:
    u = np.sort(np.asarray(u, dtype=np.float64))
    n = u.size
    if n == 0:
        return float("nan")
    grid = np.arange(1, n + 1) / n
    return float(max(np.max(grid - u), np.max(u - (grid - 1.0 / n))))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = EPS) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def skill(score: float, ref: float) -> float:
    return float(1.0 - score / ref) if ref and ref > 0 and math.isfinite(ref) else float("nan")


def calibration_table(p: np.ndarray, y: np.ndarray, edges: tuple[float, ...] = CAL_EDGES) -> pl.DataFrame:
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n:
            pm, f = float(p[mask].mean()), float(y[mask].mean())
            rows.append((lo, min(hi, 1.0), n, pm, f, f - pm, math.sqrt(max(f * (1 - f), 1e-12) / n)))
    return pl.DataFrame(rows, schema=["bin_lo", "bin_hi", "n", "p_mean", "freq_up", "gap", "se"], orient="row")


def ece(p: np.ndarray, y: np.ndarray) -> float:
    tab = calibration_table(p, y)
    if tab.is_empty():
        return float("nan")
    w = tab["n"].to_numpy() / tab["n"].sum()
    return float(np.sum(w * np.abs(tab["gap"].to_numpy())))


def direction_vs_last(pred: np.ndarray, actual: np.ndarray, last: np.ndarray, tol: float = 1e-9) -> dict[str, float]:
    """Does the forecast move the right way from the last observed price? Rows where either side does
    not move are left out."""
    dp, da = np.asarray(pred) - np.asarray(last), np.asarray(actual) - np.asarray(last)
    use = (np.abs(dp) > tol) & (np.abs(da) > tol)
    n = int(use.sum())
    return {"dir_acc": float(np.mean(np.sign(dp[use]) == np.sign(da[use]))) if n else float("nan"), "dir_n": n}


# ---------------------------------------------------------------------------------------------
# frame-level aggregation

def decile_mean(q: np.ndarray) -> np.ndarray:
    """Point forecast of the MEAN from the nine deciles. The model's median is the wrong summary for a
    price that ends at 0 or 1: squared error and Brier score are about the expected price."""
    return np.asarray(q, dtype=np.float64).mean(axis=-1)


def score_group(g: pl.DataFrame) -> dict[str, float]:
    """All metrics for one group of long-format forecast rows (model and baselines, same rows).

    Point accuracy is judged twice, each against the persistence forecast that is optimal for it:
      skill_mse  mean of the model's deciles vs the persistence reference (the martingale mean)
      skill_mae  the model's median vs the median of the persistence fan
    """
    y = g["actual"].to_numpy()
    med, ref, rw = g["median"].to_numpy(), g["persist"].to_numpy(), g["rw_p"].to_numpy()
    q, fan = g.select(QUANTILE_COLS).to_numpy(), g.select(FAN_Q_COLS).to_numpy()
    mean_fc = decile_mean(q)
    mae, mae_p = float(np.mean(np.abs(med - y))), float(np.mean(np.abs(fan[:, 4] - y)))
    mse, mse_p, mse_rw = float(np.mean((mean_fc - y) ** 2)), float(np.mean((ref - y) ** 2)), float(np.mean((rw - y) ** 2))
    pb, pb_fan = float(pinball(q, y).mean()), float(pinball(fan, y).mean())
    out: dict[str, float] = {
        "n": g.height, "n_windows": g["t0"].n_unique(), "n_days": g["date"].n_unique(),
        "rmse_c": 100 * math.sqrt(mse), "rmse_c_persist": 100 * math.sqrt(mse_p), "rmse_c_rw": 100 * math.sqrt(mse_rw),
        "skill_mse": skill(mse, mse_p), "skill_mse_rw": skill(mse_rw, mse_p), "bias_c": 100 * float(np.mean(mean_fc - y)),
        "mae_c": 100 * mae, "mae_c_persist": 100 * mae_p, "skill_mae": skill(mae, mae_p),
        "pinball_c": 100 * pb, "pinball_c_fan": 100 * pb_fan, "skill_pinball": skill(pb, pb_fan),
        "cover80": float(np.mean((y >= q[:, 0]) & (y <= q[:, 8]))), "cover60": float(np.mean((y >= q[:, 1]) & (y <= q[:, 7]))),
        "cover80_fan": float(np.mean((y >= fan[:, 0]) & (y <= fan[:, 8]))),
        "width80_c": 100 * float(np.mean(q[:, 8] - q[:, 0])), "width80_c_fan": 100 * float(np.mean(fan[:, 8] - fan[:, 0])),
        "pit_ks": ks_uniform(pit(q, y)),
    }
    out.update(direction_vs_last(mean_fc, y, ref))
    s = g.filter(pl.col("is_settlement") & pl.col("outcome_price").is_not_null())
    if s.height:
        o = s["outcome_price"].cast(pl.Float64).to_numpy()
        p = np.clip(decile_mean(s.select(QUANTILE_COLS).to_numpy()), EPS, 1 - EPS)
        p_last, p_rw = (np.clip(s[c].to_numpy(), EPS, 1 - EPS) for c in ("persist", "rw_p"))
        b, b_last, b_rw = brier(p, o), brier(p_last, o), brier(p_rw, o)
        out.update({"settle_n": s.height, "settle_brier": b, "settle_brier_persist": b_last, "settle_brier_rw": b_rw,
                    "settle_bss_persist": skill(b, b_last), "settle_bss_rw": skill(b, b_rw),
                    "settle_logloss": log_loss(p, o), "settle_logloss_persist": log_loss(p_last, o), "settle_ece": ece(p, o)})
    return out


def metrics_by(forecasts: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    if not by:
        return pl.DataFrame([score_group(forecasts)])
    rows = []
    for keys, g in forecasts.group_by(by, maintain_order=True):
        rows.append(dict(zip(by, keys, strict=True)) | score_group(g))
    return pl.DataFrame(rows, infer_schema_length=None).sort(by)


# ---------------------------------------------------------------------------------------------
# day-block bootstrap

def bootstrap_ratio(num: np.ndarray, den: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """CI of sum(num)/sum(den) when the rows (days) are resampled with replacement.
    num, den: (D, K) per-day sufficient statistics -> (lo (K,), hi (K,))."""
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    D = num.shape[0]
    if D == 0 or n_boot <= 0:
        return np.full(num.shape[1], np.nan), np.full(num.shape[1], np.nan)
    W = np.random.default_rng(seed).multinomial(D, np.full(D, 1.0 / D), size=n_boot).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = (W @ num) / (W @ den)
    return np.nanquantile(stat, alpha / 2, axis=0), np.nanquantile(stat, 1 - alpha / 2, axis=0)


def bootstrap_by(forecasts: pl.DataFrame, key: str, n_boot: int = 1000, seed: int = 0) -> pl.DataFrame:
    """95% day-block intervals for rmse/mae/pinball and the three skills per value of `key`."""
    y = forecasts["actual"].to_numpy()
    q, fan = forecasts.select(QUANTILE_COLS).to_numpy(), forecasts.select(FAN_Q_COLS).to_numpy()
    df = forecasts.select("date", key).with_columns(
        pl.Series("_se", (decile_mean(q) - y) ** 2), pl.Series("_se_p", (forecasts["persist"].to_numpy() - y) ** 2),
        pl.Series("_ae", np.abs(forecasts["median"].to_numpy() - y)), pl.Series("_ae_p", np.abs(fan[:, 4] - y)),
        pl.Series("_pb", pinball(q, y)), pl.Series("_pb_f", pinball(fan, y)), pl.lit(1.0).alias("_one"),
    )
    agg = df.group_by("date", key).agg(pl.col("_se", "_se_p", "_ae", "_ae_p", "_pb", "_pb_f", "_one").sum())
    keys = sorted(agg[key].unique().to_list())

    def mat(col: str) -> np.ndarray:
        p = agg.pivot(on=key, index="date", values=col, aggregate_function="first").sort("date")
        return p.select([str(v) for v in keys]).fill_null(0.0).to_numpy()

    one, se, se_p, ae, ae_p, pb, pb_f = (mat(c) for c in ("_one", "_se", "_se_p", "_ae", "_ae_p", "_pb", "_pb_f"))
    rows = []
    for name, num, den, scale, flip in (("mae_c", ae, one, 100.0, False), ("pinball_c", pb, one, 100.0, False), ("skill_mse", se, se_p, 1.0, True),
                                        ("skill_mae", ae, ae_p, 1.0, True), ("skill_pinball", pb, pb_f, 1.0, True)):
        lo, hi = bootstrap_ratio(num, den, n_boot, seed)
        lo, hi = ((1 - hi, 1 - lo) if flip else (scale * lo, scale * hi))
        rows += [{key: v, "stat": name, "lo": float(lo[j]), "hi": float(hi[j]), "n_days": one.shape[0]} for j, v in enumerate(keys)]
    return pl.DataFrame(rows)
