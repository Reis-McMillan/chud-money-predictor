"""Signal check: do features computed from the 1-second BRTI ticks add anything on the 1-minute grid?

Diagnostic only (small models, minutes to run). Same chronological split as `chudp finetune`.

  A. volatility: old sigma (rolling std of 240 1-minute returns) vs realized vol from 1-second returns
  B. mispricing: does (fair value - mid) predict the mid's change over the next h minutes, and does
     a trade on that prediction survive the spread and the fee?
  C. settlement: Brier of mid / rw_p / blends (for reference; not the trading goal)

    uv run --with scikit-learn --with lightgbm python scripts/analysis/signal_check.py
"""
import math
from datetime import datetime

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from chud_predictor.features import h_eff, norm_cdf

TRAIN_END, VAL_START, VAL_END, TEST_START = datetime(2026, 6, 27), datetime(2026, 6, 28), datetime(2026, 8, 6), datetime(2026, 8, 7)
START = datetime(2025, 12, 19)
FEE = 0.07          # Kalshi taker fee per contract = FEE * p * (1 - p) (general schedule; check the series' own)
HS = (1, 2, 3, 5, 8)
rng = np.random.default_rng(0)


def split_of(ts: pl.Expr) -> pl.Expr:
    return (pl.when(ts < TRAIN_END).then(pl.lit("train")).when((ts >= VAL_START) & (ts < VAL_END)).then(pl.lit("val"))
            .when(ts >= TEST_START).then(pl.lit("test")).otherwise(None))


def boot_ci(num: np.ndarray, den: np.ndarray, days: np.ndarray, n: int = 1000) -> tuple[float, float]:
    """95% day-block CI of 1 - sum(num)/sum(den)."""
    u, inv = np.unique(days, return_inverse=True)
    a, b = np.bincount(inv, num), np.bincount(inv, den)
    idx = rng.integers(0, u.size, (n, u.size))
    s = 1 - a[idx].sum(1) / b[idx].sum(1)
    return float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def load() -> pl.DataFrame:
    # the one-second columns come from the bars (`resample.aggregate`); the frame's own vol_ratio is recomputed below
    f = pl.read_parquet("data/processed/frame_1m.parquet").sort("ts").drop("vol_ratio").rename(
        {"brti_rv_1s": "rv_1s", "brti_ret_l10": "ret_l10", "brti_ret_l30": "ret_l30"})
    f = f.with_columns(
        *[pl.col("rv_1s").rolling_mean(L, min_samples=L // 2).sqrt().alias(f"sig_rv{L}") for L in (15, 30, 60, 240)],
        (pl.col("ret_1m") ** 2).alias("r2_1m"),
    )
    hk = h_eff(pl.col("k").cast(pl.Float64)).sqrt()
    for name, sig in [("old", "sigma_1m"), ("rv15", "sig_rv15"), ("rv30", "sig_rv30"), ("rv60", "sig_rv60"), ("rv240", "sig_rv240")]:
        f = f.with_columns((pl.col("moneyness") / (pl.col(sig) * hk)).clip(-8, 8).alias(f"z_{name}"))
        z = f[f"z_{name}"].to_numpy()
        f = f.with_columns(pl.Series(f"p_{name}", np.where(np.isfinite(z), norm_cdf(np.nan_to_num(z)), np.nan)).fill_nan(None))
    # future realized variance over the next 10 minutes (per-minute units), two measurements
    f = f.with_columns(
        pl.col("rv_1s").rolling_mean(10).shift(-10).alias("fut_rv"),
        pl.col("r2_1m").rolling_mean(10).shift(-10).alias("fut_r2"),
    )
    for h in HS:
        same = pl.col("t0").shift(-h) == pl.col("t0")
        ok = same & pl.col("quote_ok").shift(-h) & (pl.col("k") + h <= 13)
        f = f.with_columns(
            pl.when(ok).then(pl.col("mid_close").shift(-h) - pl.col("mid_close")).alias(f"d{h}"),
            pl.when(ok).then(pl.col("yes_bid_close").shift(-h)).alias(f"bid{h}"),
            pl.when(ok).then(pl.col("yes_ask_close").shift(-h)).alias(f"ask{h}"),
        )
    phi = np.exp(-0.5 * f["z_rv60"].to_numpy() ** 2) / math.sqrt(2 * math.pi)
    f = f.with_columns(
        pl.Series("_phi", phi),
        split_of(pl.col("ts")).alias("split"),
    ).with_columns(
        (pl.col("p_old") - pl.col("mid_close")).alias("gap_old"),
        *[(pl.col(f"p_{n}") - pl.col("mid_close")).alias(f"gap_{n}") for n in ("rv15", "rv30", "rv60", "rv240")],
        (pl.col("p_old") - pl.col("p_old").shift(1)).alias("dp_old"),
        (pl.col("p_rv60") - pl.col("p_rv60").shift(1)).alias("dp_rv60"),
        (pl.col("ret_1m") / pl.col("sigma_1m")).alias("ret_std"),
        (pl.col("ret_l10") / pl.col("sig_rv60")).alias("l10_std"),
        (pl.col("ret_l30") / pl.col("sig_rv60")).alias("l30_std"),
        # last-10s index move expressed as a fair-value move (delta * return)
        (pl.col("_phi") * pl.col("ret_l10") / (pl.col("sig_rv60") * hk)).alias("l10_fair"),
        (pl.col("_phi") * pl.col("ret_l30") / (pl.col("sig_rv60") * hk)).alias("l30_fair"),
        (pl.col("sig_rv15") / pl.col("sig_rv240")).log().alias("vol_ratio"),
        (pl.col("sig_rv60") / pl.col("sigma_1m")).log().alias("vol_new_old"),
        pl.col("mid_ret_1m").fill_null(0.0).alias("mid_ret"),
        pl.col("k").cast(pl.Float64).alias("kf"),
    )
    return f.filter((pl.col("ts") >= START) & pl.col("split").is_not_null())


def part_a(f: pl.DataFrame) -> None:
    print("\n=== A. volatility forecast: which sigma predicts the next 10 minutes' realized variance? ===")
    print(f"mean 1m ret^2 {f['r2_1m'].mean():.3e} | mean rv_1s {f['rv_1s'].mean():.3e}   (ratio rv_1s / ret^2 = {f['rv_1s'].mean() / f['r2_1m'].mean():.3f})")
    print(f"{'sigma':<12}{'split':<6}{'corr(log) vs fut rv1s':>24}{'QLIKE vs fut rv1s':>20}{'QLIKE vs fut 1m r^2':>22}")
    for sig in ("sigma_1m", "sig_rv240", "sig_rv60", "sig_rv30", "sig_rv15"):
        for sp in ("val", "test"):
            d = f.filter((pl.col("split") == sp) & pl.col(sig).is_not_null() & (pl.col("fut_rv") > 0) & (pl.col("fut_r2") > 0) & (pl.col(sig) > 0))
            s2, a, b = d[sig].to_numpy() ** 2, d["fut_rv"].to_numpy(), d["fut_r2"].to_numpy()
            # bias-correct each forecast on its own level so QLIKE compares shape, not scale
            ql = lambda y, s: float(np.mean(y / (s * y.mean() / s.mean()) - np.log(y / (s * y.mean() / s.mean())) - 1))  # noqa: E731
            print(f"{sig:<12}{sp:<6}{np.corrcoef(np.log(s2), np.log(a))[0, 1]:>24.4f}{ql(a, s2):>20.4f}{ql(b, s2):>22.4f}")


BASE = ["mid_close", "kf", "spread", "gap_old", "z_old", "dp_old", "mid_ret", "ret_std", "log_volume", "log_oi", "hour"]
TICK = ["gap_rv15", "gap_rv30", "gap_rv60", "gap_rv240", "z_rv60", "dp_rv60", "l10_std", "l30_std", "l10_fair", "l30_fair", "vol_ratio", "vol_new_old"]


def fit_gbm(tr: pl.DataFrame, va: pl.DataFrame, cols: list[str], y: str) -> lgb.Booster:
    p = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 300, "feature_fraction": 0.8,
         "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "seed": 0, "num_threads": 16}
    dtr, dva = lgb.Dataset(tr.select(cols).to_numpy(), tr[y].to_numpy()), lgb.Dataset(va.select(cols).to_numpy(), va[y].to_numpy())
    return lgb.train(p, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])


def part_b(f: pl.DataFrame) -> None:
    print("\n=== B. mispricing -> next mid change (exit before settlement; origins k<=12, quote_ok) ===")
    o = f.filter(pl.col("quote_ok") & (pl.col("k") <= 12) & pl.col("gap_old").is_not_null() & pl.col("gap_rv60").is_not_null())
    print("B1. OLS  d_h = a + b * gap  (b = share of the gap the mid closes in h minutes); skill = 1 - MSE/MSE(no change), out of sample")
    print(f"{'h':<3}{'gap':<10}{'b (train)':>10}{'skill val':>11}{'skill test':>12}{'test 95% CI':>22}")
    for h in HS:
        d = o.filter(pl.col(f"d{h}").is_not_null())
        tr = d.filter(pl.col("split") == "train")
        for g in ("gap_old", "gap_rv60", "gap_rv15"):
            X = np.c_[np.ones(tr.height), tr[g].to_numpy()]
            beta = np.linalg.lstsq(X, tr[f"d{h}"].to_numpy(), rcond=None)[0]
            res = {}
            for sp in ("val", "test"):
                e = d.filter(pl.col("split") == sp)
                y, pred = e[f"d{h}"].to_numpy(), beta[0] + beta[1] * e[g].to_numpy()
                res[sp] = (1 - ((y - pred) ** 2).sum() / (y ** 2).sum(), boot_ci((y - pred) ** 2, y ** 2, e["date"].to_numpy()))
            print(f"{h:<3}{g:<10}{beta[1]:>10.4f}{res['val'][0]:>11.4f}{res['test'][0]:>12.4f}{'[{:+.4f}, {:+.4f}]'.format(*res['test'][1]):>22}")

    print("\nB2. LightGBM (early-stopped on val, so TEST is the clean number): 1-minute features vs + 1-second tick features")
    print(f"{'h':<3}{'features':<12}{'trees':>6}{'skill val':>11}{'skill test':>12}{'test 95% CI':>22}")
    preds = {}
    for h in HS:
        d = o.filter(pl.col(f"d{h}").is_not_null())
        tr, va, te = (d.filter(pl.col("split") == s) for s in ("train", "val", "test"))
        for name, cols in (("1m", BASE), ("1m+ticks", BASE + TICK)):
            m = fit_gbm(tr, va, cols, f"d{h}")
            out = {}
            for sp, e in (("val", va), ("test", te)):
                y, pred = e[f"d{h}"].to_numpy(), m.predict(e.select(cols).to_numpy())
                out[sp] = (1 - ((y - pred) ** 2).sum() / (y ** 2).sum(), boot_ci((y - pred) ** 2, y ** 2, e["date"].to_numpy()), pred)
            preds[(h, name, "val")], preds[(h, name, "test")] = (va, out["val"][2]), (te, out["test"][2])
            print(f"{h:<3}{name:<12}{m.best_iteration:>6}{out['val'][0]:>11.4f}{out['test'][0]:>12.4f}{'[{:+.4f}, {:+.4f}]'.format(*out['test'][1]):>22}")
            if name == "1m+ticks" and h == 3:
                imp = sorted(zip(m.feature_importance("gain"), cols, strict=True), reverse=True)[:8]
                top = ", ".join(f"{c} {g / sum(m.feature_importance('gain')):.0%}" for g, c in imp)
        if h == 3:
            print(f"     top gain (h=3, 1m+ticks): {top}")

    print("\nB3. trade the predictions: enter at the quote now, exit at the quote h minutes later (cents per contract)")
    print("    long = buy YES at ask, sell at bid; short = sell YES at bid (buy NO), buy back at ask. fee = 0.07 p (1 - p) per side")
    print("    (val picked the number of trees, so it is mildly optimistic; test is clean)")
    print(f"{'h':<3}{'features':<10}{'split':<6}{'|pred| >':>9}{'trades':>8}{'mid-to-mid':>12}{'after spread':>14}{'after spr+fee':>15}{'95% CI (day blocks)':>24}{'hit':>8}")
    for h in HS:
        for name in ("1m", "1m+ticks"):
            for sp in ("val", "test"):
                te, pred = preds[(h, name, sp)]
                bid, ask = te["yes_bid_close"].to_numpy(), te["yes_ask_close"].to_numpy()
                bid_h, ask_h, d, days = te[f"bid{h}"].to_numpy(), te[f"ask{h}"].to_numpy(), te[f"d{h}"].to_numpy(), te["date"].to_numpy()
                for thr in (0.01, 0.02):
                    s = np.sign(pred) * (np.abs(pred) > thr)
                    on = s != 0
                    if on.sum() < 30:
                        print(f"{h:<3}{name:<10}{sp:<6}{thr * 100:>8.1f}c{int(on.sum()):>8}   (too few)")
                        continue
                    gross = (s * d)[on]
                    net = np.where(s > 0, bid_h - ask, bid - ask_h)[on]
                    p_in, p_out = np.where(s > 0, ask, bid)[on], np.where(s > 0, bid_h, ask_h)[on]
                    pnl = net - FEE * (p_in * (1 - p_in) + p_out * (1 - p_out))
                    u, inv = np.unique(days[on], return_inverse=True)
                    a, b = np.bincount(inv, pnl), np.bincount(inv)
                    idx = rng.integers(0, u.size, (2000, u.size))
                    bs = a[idx].sum(1) / b[idx].sum(1)
                    ci = f"[{np.quantile(bs, 0.025) * 100:+.2f}, {np.quantile(bs, 0.975) * 100:+.2f}]"
                    print(f"{h:<3}{name:<10}{sp:<6}{thr * 100:>8.1f}c{int(on.sum()):>8}{gross.mean() * 100:>12.3f}{net.mean() * 100:>14.3f}{pnl.mean() * 100:>15.3f}{ci:>24}{(gross > 0).mean():>8.1%}")


def part_c(f: pl.DataFrame) -> None:
    print("\n=== C. settlement Brier, origins k<=13 (reference: the strategy does not hold to settlement) ===")
    o = f.filter(pl.col("quote_ok") & (pl.col("k") <= 13) & pl.col("outcome_brti").is_not_null() & pl.all_horizontal([pl.col(c).is_not_null() for c in ("z_old", "z_rv15", "z_rv60", "z_rv240")]))
    y = {s: o.filter(pl.col("split") == s)["outcome_brti"].cast(pl.Float64).to_numpy() for s in ("train", "val", "test")}
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))  # noqa: E731
    feats = {
        "logit(mid) recal": lambda d: np.c_[lg(d["mid_close"].to_numpy())],
        "+ z_old": lambda d: np.c_[lg(d["mid_close"].to_numpy()), d["z_old"].to_numpy()],
        "+ z_rv60": lambda d: np.c_[lg(d["mid_close"].to_numpy()), d["z_rv60"].to_numpy()],
        "+ z_rv15,60,240": lambda d: np.c_[lg(d["mid_close"].to_numpy()), d["z_rv15"].to_numpy(), d["z_rv60"].to_numpy(), d["z_rv240"].to_numpy()],
    }
    print(f"{'forecast':<20}{'BSS vs mid  val':>16}{'test':>10}{'test 95% CI':>22}")
    for name, col in (("rw_p old sigma", "p_old"), ("rw_p rv60", "p_rv60"), ("rw_p rv15", "p_rv15")):
        r = []
        for sp in ("val", "test"):
            e = o.filter(pl.col("split") == sp)
            a, b = (e[col].to_numpy() - y[sp]) ** 2, (e["mid_close"].to_numpy() - y[sp]) ** 2
            r.append((1 - a.sum() / b.sum(), boot_ci(a, b, e["date"].to_numpy())))
        print(f"{name:<20}{r[0][0]:>16.4f}{r[1][0]:>10.4f}{'[{:+.4f}, {:+.4f}]'.format(*r[1][1]):>22}")
    tr = o.filter(pl.col("split") == "train")
    for name, fx in feats.items():
        m = LogisticRegression(C=10.0, max_iter=500).fit(fx(tr), y["train"])
        r = []
        for sp in ("val", "test"):
            e = o.filter(pl.col("split") == sp)
            a, b = (m.predict_proba(fx(e))[:, 1] - y[sp]) ** 2, (e["mid_close"].to_numpy() - y[sp]) ** 2
            r.append((1 - a.sum() / b.sum(), boot_ci(a, b, e["date"].to_numpy())))
        print(f"{name:<20}{r[0][0]:>16.4f}{r[1][0]:>10.4f}{'[{:+.4f}, {:+.4f}]'.format(*r[1][1]):>22}")


if __name__ == "__main__":
    frame = load()
    print(frame.group_by("split").len().sort("split"))
    part_a(frame)
    part_b(frame)
    part_c(frame)
