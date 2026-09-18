import math

import numpy as np
import polars as pl
import pytest

from chud_predictor import metrics as M

Q = np.array([[100, 101, 102, 103, 104, 105, 106, 107, 108]], dtype=np.float64)


def test_p_up_extremes_and_median():
    assert M.p_up_from_quantiles(Q, np.array([50.0]))[0] > 0.99
    assert M.p_up_from_quantiles(Q, np.array([200.0]))[0] < 0.01
    assert M.p_up_from_quantiles(Q, np.array([104.0]))[0] == pytest.approx(0.5, abs=1e-12)


def test_p_up_linear_between_knots():
    assert M.p_up_from_quantiles(Q, np.array([102.0]))[0] == pytest.approx(0.7)
    assert M.p_up_from_quantiles(Q, np.array([102.5]))[0] == pytest.approx(0.65)


def test_p_up_exponential_tail():
    # one lambda (= q20 - q10 = 1) below q10: F = 0.1/e
    p = M.p_up_from_quantiles(Q, np.array([99.0]))[0]
    assert p == pytest.approx(1 - 0.1 / math.e, abs=1e-9)


def test_p_up_monotone_and_normal_fit_agrees_on_gaussian():
    grid = np.linspace(95, 113, 200)
    p = M.p_up_from_quantiles(np.repeat(Q, 200, axis=0), grid)
    assert np.all(np.diff(p) <= 1e-12)
    z = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])
    qg = (100 + 3 * z)[None, :]
    strikes = np.array([97.0, 100.0, 102.0])
    a = M.p_up_from_quantiles(np.repeat(qg, 3, axis=0), strikes, "pwl_exp")
    b = M.p_up_from_quantiles(np.repeat(qg, 3, axis=0), strikes, "normal_fit")
    assert np.all(np.abs(a - b) < 0.02)


def test_scores():
    assert M.brier(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0
    assert M.brier(np.array([0.5, 0.5]), np.array([1, 0])) == 0.25
    assert M.brier(np.array([0.8]), np.array([0])) == pytest.approx(0.64)
    assert M.log_loss(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(math.log(2))
    assert M.log_loss(np.array([1 - 1e-9]), np.array([0])) == pytest.approx(-math.log(1e-6))
    assert M.brier_skill(np.array([0.9, 0.1]), np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(1 - 0.01 / 0.25)
    assert M.sharpness(np.array([0.5, 1.0, 0.0])) == pytest.approx(1 / 3)


def test_calibration_table_and_ece():
    p = np.array([0.65] * 10 + [0.25] * 4)
    y = np.array([1] * 7 + [0] * 3 + [0, 0, 0, 1])
    tab = M.calibration_table(p, y)
    row = tab.filter((pl.col("bin_lo") == 0.6) & (pl.col("n") > 0)).row(0, named=True)
    assert row["n"] == 10 and row["freq_up"] == pytest.approx(0.7) and row["gap"] == pytest.approx(0.05)
    assert M.ece(p, y) == pytest.approx((10 * 0.05 + 4 * 0.0) / 14)


def test_rwvol_p_up():
    m = np.array([0, 7, 14])
    assert np.allclose(M.h_eff(m), [14 + 1 / 3, 7 + 1 / 3, 1 / 3])
    same = M.rwvol_p_up(np.array([100.0] * 3), np.array([100.0] * 3), np.array([0.001] * 3), m)
    assert np.allclose(same, 0.5)
    # last one sigma*sqrt(h_eff) above strike -> Phi(1)
    sig = 0.001
    last = 100.0 * math.exp(sig * math.sqrt(1 / 3))
    p = M.rwvol_p_up(np.array([last]), np.array([100.0]), np.array([sig]), np.array([14]))
    assert p[0] == pytest.approx(0.8413447, abs=1e-6)
    # missing vol -> 0.5
    assert M.rwvol_p_up(np.array([101.0]), np.array([100.0]), np.array([np.nan]), np.array([3]))[0] == 0.5


def test_point_and_directional_metrics():
    actual = np.array([100.0, 102.0, 98.0, 101.0])
    naive = np.array([99.0, 103.0, 99.0, 100.0])
    pm = M.point_metrics(naive, actual, naive, np.array([1.0] * 4))
    assert pm["mase_h"] == pytest.approx(1.0) and pm["mae"] == pytest.approx(1.0) and pm["mase_1s"] == pytest.approx(1.0)
    strike = np.array([100.0] * 4)
    median = np.array([101.0, 99.0, 100.0, 102.0])  # up, down, abstain (tie at 0 bp), up
    d = M.directional_metrics(median, strike, actual, eps_bp=0.0)
    # labels: 100>100 False, 102 True, 98 False, 101 True; median 100 vs strike 100 -> abstain
    assert d["n_abstain"] == 1 and d["tp"] == 1 and d["fp"] == 1 and d["fn"] == 1 and d["tn"] == 0
    assert d["accuracy"] == pytest.approx(1 / 3) and d["tie_count"] == 1
    d2 = M.directional_metrics(median, strike, actual, eps_bp=500.0)  # 5% band -> abstain everything
    assert d2["n_abstain"] == 4 and math.isnan(d2["accuracy"])


def test_bootstrap_ratio_degenerate():
    num = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    den = np.ones((3, 2))
    lo, hi = M.bootstrap_ratio(num, den, n_boot=50)
    assert np.allclose(lo, [1.0, 2.0]) and np.allclose(hi, [1.0, 2.0])
