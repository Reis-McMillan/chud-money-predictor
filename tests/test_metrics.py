import math
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from chud_predictor import baselines as B
from chud_predictor import metrics as M

Q = np.array([[0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56]])


def test_pinball():
    point = np.full((1, 9), 0.5)
    assert M.pinball(point, np.array([0.6]))[0] == pytest.approx(0.05)      # a point mass scores |e| / 2
    assert M.pinball(point, np.array([0.3]))[0] == pytest.approx(0.10)
    # by hand for y = 0.47: quantiles below y pay q*(y-q_k), above pay (1-q)*(q_k-y)
    levels, y = M.QUANTILE_LEVELS, 0.47
    by_hand = np.mean([lv * (y - qk) if y >= qk else (1 - lv) * (qk - y) for lv, qk in zip(levels, Q[0], strict=True)])
    assert M.pinball(Q, np.array([y]))[0] == pytest.approx(by_hand)
    assert M.pinball(Q, np.array([0.48]))[0] < M.pinball(Q + 0.2, np.array([0.48]))[0]


def test_pit_and_coverage_of_a_calibrated_forecast():
    rng = np.random.default_rng(0)
    n = 20_000
    mu = rng.uniform(0.3, 0.7, n)
    z = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])
    q = mu[:, None] + 0.05 * z[None, :]
    y = mu + 0.05 * rng.standard_normal(n)
    u = M.pit(q, y)
    assert M.ks_uniform(u) < 0.02 and np.mean((y >= q[:, 0]) & (y <= q[:, 8])) == pytest.approx(0.8, abs=0.01)
    assert M.ks_uniform(M.pit(q, mu + 0.15 * rng.standard_normal(n))) > 0.1      # too narrow -> far from uniform
    assert M.pwl_exp_cdf(Q, np.array([0.48]))[0] == pytest.approx(0.5) and M.pwl_exp_cdf(Q, np.array([0.45]))[0] == pytest.approx(0.35)


def test_probability_scores():
    assert M.brier(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0 and M.brier(np.array([0.8]), np.array([0])) == pytest.approx(0.64)
    assert M.log_loss(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(math.log(2))
    assert M.skill(0.2, 0.25) == pytest.approx(0.2) and math.isnan(M.skill(0.2, 0.0))
    p = np.array([0.65] * 10 + [0.25] * 4)
    y = np.array([1] * 7 + [0] * 3 + [0, 0, 0, 1])
    row = M.calibration_table(p, y).filter(pl.col("bin_lo") == 0.6).row(0, named=True)
    assert row["n"] == 10 and row["freq_up"] == pytest.approx(0.7) and M.ece(p, y) == pytest.approx(10 * 0.05 / 14)
    d = M.direction_vs_last(np.array([0.6, 0.4, 0.5, 0.7]), np.array([0.7, 0.45, 0.6, 0.4]), np.array([0.5, 0.5, 0.5, 0.5]))
    assert d == {"dir_acc": pytest.approx(2 / 3), "dir_n": 3}                     # the unchanged forecast is left out


def _mini_frame(n_windows: int = 40) -> pl.DataFrame:
    """Deterministic frame: inside every window the mid rises by exactly 0.01 per minute from 0.30."""
    ts = [datetime(2026, 1, 1) + timedelta(minutes=i) for i in range(15 * n_windows)]
    k = np.arange(15 * n_windows) % 15
    return pl.DataFrame({"ts": ts, "k": k.astype(np.int8), "mid_close": 0.30 + 0.01 * k}).with_columns(pl.col("ts").cast(pl.Datetime("us")))


def test_persistence_fan_and_application():
    frame = _mini_frame()
    fan = B.persistence_fan(frame)
    pooled = fan.filter(pl.col("bucket") == B.POOLED)
    assert pooled.height == 120 and B.fan_cells(fan) == 120                       # the full (k_ctx, h) triangle
    cell = pooled.filter((pl.col("k_ctx") == 3) & (pl.col("h") == 5)).row(0, named=True)
    assert cell["dq10"] == cell["dq90"] == pytest.approx(0.05)                   # k 3 -> 8 inside one contract
    first = pooled.filter((pl.col("k_ctx") == 14) & (pl.col("h") == 1)).row(0, named=True)
    # at the window open the last mid belongs to the settled contract: the reference is 0.50, not 0.44
    assert first["dq50"] == pytest.approx(0.30 - 0.50) and pooled.filter(pl.col("k_ctx") == 14)["h"].max() == 15
    assert np.allclose(B.persist_ref(np.array([0.999, 0.62]), np.array([14, 3])), [0.5, 0.62])
    assert pooled.filter(pl.col("k_ctx") == 13)["h"].to_list() == [1]            # nothing beyond expiry
    q = B.apply_fan(np.array([0.5, 0.99]), np.array([3, 3]), np.array([5, 5]), fan)
    assert np.allclose(q[0], 0.55) and np.allclose(q[1], 1.0)                     # clipped to the price bounds
    assert np.allclose(B.apply_fan(np.array([0.5]), np.array([3]), np.array([5]), None), 0.5)   # no history: point mass
    half = B.persistence_fan(frame, hi=datetime(2026, 1, 1, 2))                  # 8 windows < 30 observations per cell
    assert B.fan_cells(half) == 0 and np.allclose(B.apply_fan(np.array([0.5]), np.array([3]), np.array([5]), half), 0.5)


def test_fan_is_conditional_on_the_price_level():
    """Prices drift to the extremes: a fan that ignores the level cannot know which way."""
    n_w = 80
    ts = [datetime(2026, 1, 1) + timedelta(minutes=i) for i in range(15 * n_w)]
    k = np.arange(15 * n_w) % 15
    high = (np.arange(15 * n_w) // 15) % 2 == 0                 # alternate windows trade at 0.80 rising / 0.20 falling
    mid = np.where(high, 0.80 + 0.01 * k, 0.20 - 0.01 * k)
    fan = B.persistence_fan(pl.DataFrame({"ts": ts, "k": k.astype(np.int8), "mid_close": mid}).with_columns(pl.col("ts").cast(pl.Datetime("us"))))
    q = B.apply_fan(np.array([0.83, 0.17]), np.array([3, 3]), np.array([4, 4]), fan)
    assert np.allclose(q[0], 0.87) and np.allclose(q[1], 0.13)                   # each level gets its own drift
    pooled = fan.filter((pl.col("bucket") == B.POOLED) & (pl.col("k_ctx") == 3) & (pl.col("h") == 4)).row(0, named=True)
    assert pooled["dq10"] == pytest.approx(-0.04) and pooled["dq90"] == pytest.approx(0.04)


def test_rw_fair_value():
    sig = 1e-3
    up = 100.0 * math.exp(sig * math.sqrt(1 / 3))
    p = B.rw_fair_now(np.array([100.0, up, up, 101.0]), np.full(4, 100.0), np.array([sig, sig, sig, np.nan]), np.array([7, 14, 0, 3]))
    assert p[0] == pytest.approx(0.5) and p[1] == pytest.approx(0.8413447, abs=1e-6)       # one sigma with 1/3 of a bar left
    assert 0.5 < p[2] < 0.6 and p[3] == pytest.approx(0.5)                                 # 14 1/3 bars left; no vol -> no view


def _forecasts(median: np.ndarray, actual: np.ndarray, last: np.ndarray, days: int = 4) -> pl.DataFrame:
    n = len(actual)
    q = median[:, None] + (np.arange(9) - 4)[None, :] * 0.01
    cols = {c: q[:, j] for j, c in enumerate(M.QUANTILE_COLS)} | {c: last for c in M.FAN_Q_COLS}
    return pl.DataFrame({
        "t0": [datetime(2026, 1, 1) + timedelta(minutes=15 * i) for i in range(n)], "date": [date(2026, 1, 1 + i % days) for i in range(n)],
        "m": np.zeros(n, dtype=np.int8), "h": (np.arange(n) % 3 + 1).astype(np.int8), "median": median, "actual": actual, "last_mid": last, "persist": last,
        "rw_p": np.full(n, 0.5), "is_settlement": np.arange(n) % 3 == 2, "outcome_price": actual > 0.5, **cols,
    })


def test_median_is_the_wrong_point_forecast_for_a_price_that_ends_at_0_or_1():
    """An overconfident model wins on absolute error at settlement and loses on squared error: the
    reason skill_mse, not skill_mae, is the headline."""
    rng = np.random.default_rng(5)
    last = np.full(2000, 0.7)
    actual = (rng.random(2000) < 0.7).astype(float)              # the market's 0.70 is exactly right
    f = _forecasts(np.ones(2000), actual, last)                  # the model shouts 1.0
    raw_mae_model, raw_mae_market = np.mean(np.abs(1.0 - actual)), np.mean(np.abs(0.7 - actual))
    assert raw_mae_model < raw_mae_market                        # absolute error prefers the overconfident forecast
    assert M.score_group(f)["skill_mse"] < -0.3 and M.score_group(f)["settle_bss_persist"] < -0.3


def test_score_group_skill_is_relative_to_persistence():
    rng = np.random.default_rng(1)
    last = rng.uniform(0.2, 0.8, 300)
    actual = np.clip(last + rng.normal(0, 0.05, 300), 0, 1)
    same = M.score_group(_forecasts(last.copy(), actual, last))
    assert same["skill_mae"] == pytest.approx(0.0) and same["mae_c"] == pytest.approx(same["mae_c_persist"]) and same["skill_mse"] == pytest.approx(0.0)
    better = M.score_group(_forecasts((last + actual) / 2, actual, last))
    assert better["skill_mae"] == pytest.approx(0.5) and better["mae_c"] == pytest.approx(100 * np.mean(np.abs(actual - last)) / 2)
    assert better["skill_mse"] == pytest.approx(0.75)               # half the error -> a quarter of the squared error
    assert better["settle_n"] == 100 and better["settle_bss_persist"] > 0 and better["skill_pinball"] > 0
    by = M.metrics_by(_forecasts((last + actual) / 2, actual, last), ["h"])
    assert by["h"].to_list() == [1, 2, 3] and by["n"].sum() == 300


def test_bootstrap_is_deterministic_and_brackets_the_point_estimate():
    lo, hi = M.bootstrap_ratio(np.array([[1.0, 2.0]] * 3), np.ones((3, 2)), n_boot=50)
    assert np.allclose(lo, [1.0, 2.0]) and np.allclose(hi, [1.0, 2.0])
    rng = np.random.default_rng(2)
    last = rng.uniform(0.2, 0.8, 600)
    actual = np.clip(last + rng.normal(0, 0.05, 600), 0, 1)
    f = _forecasts(last + 0.6 * (actual - last), actual, last, days=20)
    a, b = M.bootstrap_by(f, "h", 200, seed=3), M.bootstrap_by(f, "h", 200, seed=3)
    assert a.equals(b) and set(a["stat"].unique()) == {"mae_c", "pinball_c", "skill_mse", "skill_mae", "skill_pinball"}
    s = a.filter((pl.col("stat") == "skill_mae") & (pl.col("h") == 1)).row(0, named=True)
    assert s["lo"] < 0.6 < s["hi"] and s["hi"] - s["lo"] < 0.1 and s["n_days"] == 20
