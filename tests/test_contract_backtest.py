"""End-to-end contract backtest with a stub multivariate forecaster: the (window, m, h) -> row mapping
survives chunking, actuals join to the right candle, and every artifact is produced."""

import json
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from conftest import T_DEGENERATE, T_DROPPED, write_raw
from test_model_api import StubMultiForecaster

from chud_predictor import metrics as M
from chud_predictor.backtest import ContractBacktestConfig, rescore, run_contract_backtest
from chud_predictor.features import build_frame
from chud_predictor.model import ForecasterHandle
from chud_predictor.origins import ContractSpec
from chud_predictor.resample import build
from chud_predictor.settings import Settings

DAY2 = date(2025, 12, 21)


@pytest.fixture(scope="module")
def settings(contract_world, tmp_path_factory) -> Settings:
    s = Settings(data_dir=tmp_path_factory.mktemp("bt") / "data")
    write_raw(contract_world["ticks"], s.raw_dir)
    write_raw(contract_world["candles"], s.contracts_raw_dir)
    build(s.raw_dir, s.processed_dir, "1m")
    build_frame(s.processed_dir, s.contracts_raw_dir, vol_lookback=60)
    return s


def _run(settings, stub, **kw):
    handle = ForecasterHandle(backend="stub", device="cpu", model_id="stub", batch_size=8, obj=stub)
    cfg = ContractBacktestConfig(spec=ContractSpec(context=64), start=DAY2, end=DAY2, origin_chunk=50, fan_lookback_days=1, bootstrap_days=20, **kw)
    return run_contract_backtest(settings, cfg, handle=handle)


def test_end_to_end_with_stub(settings, contract_world):
    stub = StubMultiForecaster(step=1e-3)
    run_dir = _run(settings, stub, plot=True, trajectories=2, run_id="stub")
    assert len(stub.calls) > 5 and stub.calls[0]["po"][0].shape == (17, 64) and stub.calls[0]["pf"][0].shape == (7, 128)
    fc = pl.read_parquet(run_dir / "forecasts.parquet")
    frame = contract_world["frame"]

    # 95 complete windows on day 2 (the last one has no successor row for m = 0 ... it still has all 15 candles)
    per_window = fc.group_by("t0").len()
    assert per_window["len"].max() == 120 and (fc["date"] == DAY2).all()
    assert fc.group_by("t0", "m").agg(pl.col("h").max().alias("hmax"), pl.col("h").n_unique().alias("nh")).select(
        ((pl.col("hmax") == 15 - pl.col("m")) & (pl.col("nh") == pl.col("hmax"))).all()).item()
    assert (fc["k"] == fc["m"] + fc["h"] - 1).all() and (fc["is_settlement"] == (fc["k"] == 14)).all()
    assert fc.group_by("t0", "m").agg(pl.col("is_settlement").sum())["is_settlement"].max() == 1
    assert fc.select(((pl.col("bar_ts") == pl.col("t0") + pl.duration(minutes=pl.col("k")))
                      & (pl.col("origin_ts") == pl.col("t0") + pl.duration(minutes=pl.col("m")))).all()).item()
    assert (fc["candle_ts"] == fc["bar_ts"] + timedelta(minutes=1)).all()

    # the row mapping survives chunking: the stub's median is last_mid + h * 1e-3 (clipped to the price bounds)
    expected = np.clip(fc["last_mid"].to_numpy() + 1e-3 * fc["h"].to_numpy(), 0, 1)
    assert np.allclose(fc["median"].to_numpy(), expected, atol=1e-4)
    # actuals come from the candle that closes at candle_ts
    truth = fc.select("bar_ts").join(frame.select(pl.col("ts").alias("bar_ts"), "mid_close"), on="bar_ts", how="left")["mid_close"]
    assert np.allclose(fc["actual"].to_numpy(), truth.to_numpy())
    assert fc["actual"].null_count() == 0 and ((fc["q10"] <= fc["median"]) & (fc["median"] <= fc["q90"])).all()
    assert fc.filter(pl.col("is_settlement"))["actual"].is_in([0.999, 0.001]).all()
    assert ((fc["fan_q10"] <= fc["fan_q90"]) & (fc["fan_q10"] >= 0) & (fc["fan_q90"] <= 1)).all() and fc["rw_p"].is_between(0, 1).all()

    for name in ("summary.txt", "metrics.json", "meta.json", "baselines.parquet", "drops.parquet", "metrics_by_h.parquet", "metrics_by_m.parquet",
                 "metrics_by_mh.parquet", "bootstrap.parquet", "settlement_calibration.parquet", "plots/error_vs_h.png",
                 "plots/skill_heatmap.png", "plots/calibration.png", "plots/settlement_reliability.png"):
        assert (run_dir / name).exists(), name
    assert len(list((run_dir / "plots").glob("trajectory_*.png"))) == 2
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["fan_mode"] == "pre_period" and meta["n_variates"] == 25 and meta["n_origins"] == fc.select("t0", "m").n_unique()
    by_mh = pl.read_parquet(run_dir / "metrics_by_mh.parquet")
    assert by_mh.height == 120 and by_mh["skill_mse"].is_finite().all()
    text = (run_dir / "summary.txt").read_text()
    assert "skill_mse" in text and "Settlement candle" in text and "pre_period" in text


def test_persistence_stub_has_zero_skill_and_rescore_filters_quotes(settings):
    run_dir = _run(settings, StubMultiForecaster(step=0.0, width=0.0), run_id="persist")
    fc = pl.read_parquet(run_dir / "forecasts.parquet")
    after_open = M.score_group(fc.filter(pl.col("m") > 0))
    assert abs(after_open["skill_mse"]) < 2e-3 and after_open["settle_bss_persist"] == pytest.approx(0.0, abs=2e-3)
    assert after_open["skill_pinball"] < 0         # a point mass loses to the calibrated persistence fan
    # at the window open the last mid is the settled contract's 0 or 1; the reference there is 0.50, and it is far better
    at_open = fc.filter(pl.col("m") == 0)
    assert (at_open["persist"] == 0.5).all() and at_open["last_mid"].is_in([0.999, 0.001]).all()
    assert M.score_group(at_open)["skill_mse"] < -1.0

    # day 1 holds the degenerate-quote window: re-score it without those origins, no model involved
    handle = ForecasterHandle(backend="stub", device="cpu", model_id="stub", batch_size=8, obj=StubMultiForecaster())
    day1 = ContractBacktestConfig(spec=ContractSpec(context=64, max_ctx_gap_frac=0.1), start=date(2025, 12, 20), end=date(2025, 12, 20),
                                  origin_chunk=200, fan_lookback_days=1, bootstrap_days=10, run_id="day1")
    d1 = run_contract_backtest(settings, day1, handle=handle)
    assert json.loads((d1 / "meta.json").read_text())["fan_mode"] == "in_sample_head"      # nothing before day 1: said out loud
    full = pl.read_parquet(d1 / "forecasts.parquet")
    assert full.filter((pl.col("t0") == T_DEGENERATE) & ~pl.col("quote_ok")).height > 0
    assert full.filter((pl.col("t0") == T_DROPPED) & (pl.col("m") == 1))["h"].to_list() == [1, 2] + list(range(5, 15))   # missing candles never scored
    out = rescore(settings, "day1", require_quote_ok=True, label="clean")
    clean = pl.read_parquet(out / "forecasts.parquet")
    assert clean.height < full.height and clean["quote_ok"].all() and "usable quote only" in (out / "summary.txt").read_text()
