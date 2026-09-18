"""Pure-numpy/polars metrics for the Kalshi 15-minute evaluation.

P(up) from the nine deciles: piecewise-linear CDF between the knots (q_k, k/10) with exponential
tails whose slope matches the outer segment, so the density is continuous at q10 and q90 and
P(up) == 0.5 exactly when the strike equals the median.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

QUANTILE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
QUANTILE_COLS = [f"q{int(q * 100)}" for q in QUANTILE_LEVELS]
Z_80 = 1.2815515655446004  # Phi^-1(0.9)
CAL_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-9)

_erf = np.vectorize(math.erf, otypes=[np.float64])


def norm_cdf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------------------------
# P(up)

def pwl_exp_cdf(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """F(x) for each row given its 9 sorted deciles. q: (n, 9), x: (n,) -> (n,)."""
    q = np.sort(np.asarray(q, dtype=np.float64), axis=1)
    x = np.asarray(x, dtype=np.float64)
    n = q.shape[0]
    spread = np.maximum(q[:, 8] - q[:, 0], 1e-12)
    lam_lo = q[:, 1] - q[:, 0]
    lam_hi = q[:, 8] - q[:, 7]
    lam_lo = np.where(lam_lo > 0, lam_lo, spread / 8.0)
    lam_hi = np.where(lam_hi > 0, lam_hi, spread / 8.0)

    F = np.full(n, np.nan)
    below = x < q[:, 0]
    above = x > q[:, 8]
    F = np.where(below, 0.1 * np.exp((x - q[:, 0]) / lam_lo), F)
    F = np.where(above, 1.0 - 0.1 * np.exp(-(x - q[:, 8]) / lam_hi), F)
    mid = ~(below | above)
    for k in range(8):
        lo, hi = q[:, k], q[:, k + 1]
        seg = mid & (x >= lo) & (x <= hi) & np.isnan(F)
        width = hi - lo
        frac = np.where(width > 0, (x - lo) / np.where(width > 0, width, 1.0), 0.0)
        F = np.where(seg, (k + 1) / 10.0 + 0.1 * frac, F)
    return np.clip(F, 0.0, 1.0)


def normal_fit_p_up(q: np.ndarray, strike: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    mu = q[:, 4]
    sigma = np.maximum((q[:, 8] - q[:, 0]) / (2 * Z_80), 1e-12)
    return norm_cdf((mu - np.asarray(strike, dtype=np.float64)) / sigma)


def p_up_from_quantiles(q: np.ndarray, strike: np.ndarray, method: str = "pwl_exp", eps: float = 1e-6) -> np.ndarray:
    if method == "pwl_exp":
        p = 1.0 - pwl_exp_cdf(q, strike)
    elif method == "normal_fit":
        p = normal_fit_p_up(q, strike)
    else:
        raise ValueError(f"unknown p_up method {method!r}")
    return np.clip(p, eps, 1.0 - eps)


def h_eff(m: np.ndarray) -> np.ndarray:
    """Effective horizon (in bars) for the variance of a 60-s average of a random walk."""
    return (14 - np.asarray(m, dtype=np.float64)) + 1.0 / 3.0


def rwvol_p_up(last_close: np.ndarray, strike: np.ndarray, sigma_1m: np.ndarray, m: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Gaussian random-walk baseline: P(settlement > strike) from the trailing 1-minute vol."""
    sigma = np.asarray(sigma_1m, dtype=np.float64)
    ok = np.isfinite(sigma) & (sigma > 0)
    z = np.zeros_like(sigma)
    denom = np.where(ok, sigma, 1.0) * np.sqrt(h_eff(m))
    z = np.where(ok, (np.log(last_close) - np.log(strike)) / denom, 0.0)
    return np.clip(norm_cdf(z), eps, 1 - eps)


# ---------------------------------------------------------------------------------------------
# scores

def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y, dtype=np.float64)) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_skill(p: np.ndarray, y: np.ndarray, p_ref: np.ndarray) -> float:
    ref = brier(p_ref, y)
    return float(1.0 - brier(p, y) / ref) if ref > 0 else float("nan")


def sharpness(p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(p) - 0.5)))


def calibration_table(p: np.ndarray, y: np.ndarray, edges: tuple[float, ...] = CAL_EDGES) -> pl.DataFrame:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append((lo, min(hi, 1.0), 0, float("nan"), float("nan"), float("nan"), float("nan")))
            continue
        pm, f = float(p[mask].mean()), float(y[mask].mean())
        rows.append((lo, min(hi, 1.0), n, pm, f, f - pm, math.sqrt(max(f * (1 - f), 1e-12) / n)))
    return pl.DataFrame(rows, schema=["bin_lo", "bin_hi", "n", "p_mean", "freq_up", "gap", "se"], orient="row")


def ece(p: np.ndarray, y: np.ndarray, edges: tuple[float, ...] = CAL_EDGES) -> float:
    tab = calibration_table(p, y, edges).filter(pl.col("n") > 0)
    if tab.is_empty():
        return float("nan")
    w = tab["n"].to_numpy() / tab["n"].sum()
    return float(np.sum(w * np.abs(tab["gap"].to_numpy())))


def point_metrics(pred: np.ndarray, actual: np.ndarray, naive: np.ndarray, mase_scale_1step: np.ndarray) -> dict[str, float]:
    pred, actual, naive = (np.asarray(a, dtype=np.float64) for a in (pred, actual, naive))
    err = np.abs(pred - actual)
    naive_err = np.abs(naive - actual)
    mae = float(err.mean())
    naive_mae = float(naive_err.mean())
    scale = np.asarray(mase_scale_1step, dtype=np.float64)
    scale_mean = float(np.nanmean(scale)) if np.isfinite(scale).any() else float("nan")
    return {
        "mae": mae,
        "mae_bp": float(np.mean(np.abs(np.log(pred / actual))) * 1e4),
        "rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "mape": float(np.mean(err / np.abs(actual))),
        "naive_mae": naive_mae,
        "mase_h": mae / naive_mae if naive_mae > 0 else float("nan"),
        "mase_1s": mae / scale_mean if scale_mean and scale_mean > 0 else float("nan"),
    }


def directional_metrics(median: np.ndarray, strike: np.ndarray, settlement: np.ndarray, eps_bp: float = 0.0) -> dict[str, float]:
    median, strike, settlement = (np.asarray(a, dtype=np.float64) for a in (median, strike, settlement))
    eps_abs = strike * eps_bp / 1e4
    pred_up = median - strike > eps_abs
    pred_down = strike - median > eps_abs
    abstain = ~(pred_up | pred_down)
    label_up = settlement > strike
    tie = settlement == strike
    n = len(median)
    act = ~abstain
    tp = int((pred_up & label_up).sum())
    fp = int((pred_up & ~label_up).sum())
    fn = int((pred_down & label_up).sum())
    tn = int((pred_down & ~label_up).sum())
    n_act = int(act.sum())
    correct = tp + tn

    def safe(a: float, b: float) -> float:
        return a / b if b > 0 else float("nan")

    return {
        "n": n,
        "n_abstain": int(abstain.sum()),
        "abstain_rate": safe(int(abstain.sum()), n),
        "accuracy": safe(correct, n_act),
        "accuracy_incl_abstain": safe(correct + 0.5 * int(abstain.sum()), n),
        "precision_up": safe(tp, tp + fp),
        "recall_up": safe(tp, int(label_up[act].sum())),
        "f1_up": safe(2 * tp, 2 * tp + fp + fn),
        "precision_down": safe(tn, tn + fn),
        "recall_down": safe(tn, int((~label_up[act]).sum())),
        "base_rate_up": safe(int(label_up.sum()), n),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tie_count": int(tie.sum()),
    }


# ---------------------------------------------------------------------------------------------
# frame-level aggregation

def score_group(g: pl.DataFrame, eps_bp: float = 0.0) -> dict[str, float]:
    """All metrics for one group of forecast rows (model + baselines)."""
    y = g["label_up"].cast(pl.Float64).to_numpy()
    out: dict[str, float] = {"n": g.height, "n_windows": g["t0"].n_unique()}
    out.update(point_metrics(g["median"].to_numpy(), g["settlement"].to_numpy(), g["last_close"].to_numpy(), g["mase_scale_1step"].to_numpy()))
    d = directional_metrics(g["median"].to_numpy(), g["strike"].to_numpy(), g["settlement"].to_numpy(), eps_bp)
    out.update({k: v for k, v in d.items() if k != "n"})
    p, p_rw, p_c = g["p_up"].to_numpy(), g["p_up_rwvol"].to_numpy(), np.full(g.height, 0.5)
    out.update({
        "brier": brier(p, y), "brier_rwvol": brier(p_rw, y), "brier_const": brier(p_c, y),
        "bss_rw": brier_skill(p, y, p_rw), "bss_const": brier_skill(p, y, p_c),
        "logloss": log_loss(p, y), "logloss_rwvol": log_loss(p_rw, y),
        "ece": ece(p, y), "ece_rwvol": ece(p_rw, y),
        "sharpness": sharpness(p), "sharpness_rwvol": sharpness(p_rw),
    })
    # naive direction baseline: sign(last_close - strike)
    dn = directional_metrics(g["last_close"].to_numpy(), g["strike"].to_numpy(), g["settlement"].to_numpy(), eps_bp)
    out["accuracy_naive"] = dn["accuracy"]
    out["abstain_rate_naive"] = dn["abstain_rate"]
    return out


def metrics_by(forecasts: pl.DataFrame, by: list[str], eps_bp: float = 0.0) -> pl.DataFrame:
    rows = []
    if by:
        for keys, g in forecasts.group_by(by, maintain_order=True):
            row = dict(zip(by, keys, strict=True))
            row.update(score_group(g, eps_bp))
            rows.append(row)
        return pl.DataFrame(rows).sort(by)
    return pl.DataFrame([score_group(forecasts, eps_bp)])


# ---------------------------------------------------------------------------------------------
# day-block bootstrap

def bootstrap_ratio(num: np.ndarray, den: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """CI of sum(num)/sum(den) under resampling of rows (days) with replacement.
    num, den: (D, K) per-day sufficient statistics -> (lo (K,), hi (K,))."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    D = num.shape[0]
    if D == 0 or n_boot <= 0:
        k = num.shape[1] if num.ndim == 2 else 1
        return np.full(k, np.nan), np.full(k, np.nan)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(D, np.full(D, 1.0 / D), size=n_boot).astype(np.float64)  # (B, D)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = (W @ num) / (W @ den)
    return np.nanquantile(stat, alpha / 2, axis=0), np.nanquantile(stat, 1 - alpha / 2, axis=0)


def bootstrap_by_m(forecasts: pl.DataFrame, n_boot: int = 1000, seed: int = 0) -> pl.DataFrame:
    """Day-block bootstrap CIs for accuracy, brier, mae and mase_h per m."""
    df = forecasts.with_columns(
        (((pl.col("median") > pl.col("strike")) == pl.col("label_up")).cast(pl.Float64)).alias("_correct"),
        ((pl.col("p_up") - pl.col("label_up").cast(pl.Float64)) ** 2).alias("_sq"),
        (pl.col("median") - pl.col("settlement")).abs().alias("_ae"),
        (pl.col("last_close") - pl.col("settlement")).abs().alias("_nae"),
        pl.lit(1.0).alias("_one"),
    )
    agg = df.group_by("date", "m").agg(
        pl.col("_correct").sum(), pl.col("_sq").sum(), pl.col("_ae").sum(), pl.col("_nae").sum(), pl.col("_one").sum()
    ).sort("date", "m")
    ms = sorted(agg["m"].unique().to_list())
    days = sorted(agg["date"].unique().to_list())
    def mat(col: str) -> np.ndarray:
        p = agg.pivot(on="m", index="date", values=col, aggregate_function="first").sort("date")
        return p.select([str(m) for m in ms]).fill_null(0.0).to_numpy()
    ones, cor, sq, ae, nae = mat("_one"), mat("_correct"), mat("_sq"), mat("_ae"), mat("_nae")
    rows = []
    for name, num, den in (("accuracy", cor, ones), ("brier", sq, ones), ("mae", ae, ones), ("mase_h", ae, nae)):
        lo, hi = bootstrap_ratio(num, den, n_boot, seed)
        for i, m in enumerate(ms):
            rows.append({"m": m, "stat": name, "lo": float(lo[i]), "hi": float(hi[i]), "n_days": len(days)})
    return pl.DataFrame(rows)
