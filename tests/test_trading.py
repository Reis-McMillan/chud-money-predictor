from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from chud_predictor import trading as T


def _forecasts(p_up, label_up):
    n = len(p_up)
    t0 = [datetime(2025, 9, 18) + timedelta(minutes=15 * i) for i in range(n)]
    return pl.DataFrame({
        "t0": t0, "date": [date(2025, 9, 18)] * n, "hour": [0] * n, "m": [5] * n, "vol_bucket": [1] * n,
        "p_up": p_up, "label_up": label_up,
    })


def test_complement_pricing_and_fees():
    f = _forecasts([0.9, 0.1, 0.5], [True, False, True])
    t = T.simulate_trades(f, T.TradeConfig(c=0.5, tau=0.55))
    assert t["side"].to_list() == ["YES", "NO", "NONE"]
    assert t["pnl_gross"].to_list() == pytest.approx([0.5, 0.5, 0.0])
    assert t["fee"].to_list() == pytest.approx([0.0175, 0.0175, 0.0])
    assert t["pnl_net"].to_list() == pytest.approx([0.4825, 0.4825, 0.0])
    assert t["win"].to_list() == [True, True, None]
    t6 = T.simulate_trades(_forecasts([0.1], [False]), T.TradeConfig(c=0.6, tau=0.55))
    assert t6["price_paid"][0] == pytest.approx(0.4) and t6["pnl_gross"][0] == pytest.approx(0.6)
    assert t6["fee"][0] == pytest.approx(0.07 * 0.4 * 0.6)


def test_same_pricing_rounding_and_no_fees():
    same = T.simulate_trades(_forecasts([0.1], [False]), T.TradeConfig(c=0.6, tau=0.55, no_price_mode="same"))
    assert same["pnl_gross"][0] == pytest.approx(0.4)
    rounded = T.simulate_trades(_forecasts([0.9], [True]), T.TradeConfig(c=0.5, tau=0.55, fee_round_cents=True))
    assert rounded["fee"][0] == pytest.approx(0.02)
    nofee = T.simulate_trades(_forecasts([0.9], [False]), T.TradeConfig(c=0.5, tau=0.55, fees=False))
    assert nofee["pnl_net"][0] == pytest.approx(-0.5)


def test_threshold_vs_ev_policy():
    assert T.decide(np.array([0.56, 0.50, 0.44]), T.TradeConfig(tau=0.55)).tolist() == ["YES", "NONE", "NO"]
    ev = T.TradeConfig(c=0.6, policy="ev", ev_margin=0.02)
    # p=0.59 would be a YES under threshold 0.55 but is negative EV at a 60c price, and NO at 40c
    # only clears the margin once P(no) >= 0.42
    assert T.decide(np.array([0.59, 0.63, 0.30, 0.58]), ev).tolist() == ["NONE", "YES", "NO", "NO"]


def test_trade_summary_by_hand():
    p = [0.9, 0.9, 0.9, 0.1, 0.5]
    y = [True, False, True, False, True]
    f = _forecasts(p, y)
    trades = T.simulate_trades(f, T.TradeConfig(c=0.5, tau=0.55)).filter(pl.col("traded"))
    s = T.trade_summary(trades, ["m"], f).row(0, named=True)
    assert s["n_trades"] == 4 and s["n_yes"] == 3 and s["n_no"] == 1
    assert s["win_rate"] == pytest.approx(0.75)
    assert s["total_pnl_net"] == pytest.approx(3 * 0.4825 - 0.5175)
    assert s["realized_edge"] == pytest.approx(0.25)
    assert s["trade_rate"] == pytest.approx(0.8) and s["n_windows"] == 5
    assert s["pnl_per_window"] == pytest.approx(s["total_pnl_net"] / 5)
    assert s["max_drawdown"] == pytest.approx(0.5175)


def test_sweep_has_both_sources():
    f = _forecasts([0.9, 0.1], [True, False]).with_columns(pl.Series("p_up_rwvol", [0.6, 0.4]))
    sw = T.sweep(f, (0.5,), (0.55, 0.65), T.TradeConfig())
    assert set(sw["source"].unique()) == {"model", "rwvol"}
    assert sw.filter((pl.col("source") == "rwvol") & (pl.col("tau") == 0.65)).height == 0
