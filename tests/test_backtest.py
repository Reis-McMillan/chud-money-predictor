"""End-to-end backtest with a stub forecaster: proves the (window, m) -> output-row mapping survives
chunking, and that every artifact is produced. No model weights needed."""

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import polars as pl
import pytest
from conftest import synthetic_ticks

from chud_predictor.backtest import KalshiBacktestConfig, rescore, run_kalshi_backtest
from chud_predictor.model import ForecasterHandle
from chud_predictor.resample import build
from chud_predictor.settings import Settings
from chud_predictor.trading import TradeConfig
from chud_predictor.windows import KalshiSpec


@dataclass
class _Out:
    ts_id: object
    forecast: np.ndarray
    quantiles: np.ndarray


class StubForecaster:
    """Median = last context value (naive) + step*1e-3 in log space, so both the row identity and the
    step picked by the backtest are verifiable well above float32 context precision (~1e-6).
    Quantiles = median + k*1e-4 (k=-4..4). Yields in input order."""

    calls: int = 0

    def predict_batch(self, contexts, horizon, **kw):
        self.calls += 1
        for c in contexts:
            last = float(c[-1])
            steps = np.arange(horizon) * 1e-3
            med = last + steps
            q = med[:, None] + (np.arange(9) - 4)[None, :] * 1e-4
            yield _Out(None, med, q)


@pytest.fixture
def data_dir(tmp_path):
    raw = tmp_path / "data" / "raw" / "brti"
    raw.mkdir(parents=True)
    for i in range(2):
        synthetic_ticks(datetime(2025, 9, 18 + i), 86_400, seed=20 + i).write_parquet(raw / f"date=2025-09-{18 + i}.parquet")
    build(raw, tmp_path / "data" / "processed", "1m")
    return tmp_path / "data"


def test_backtest_end_to_end_with_stub(data_dir, tmp_path):
    settings = Settings(data_dir=data_dir, qdb_password="x")
    stub = StubForecaster()
    handle = ForecasterHandle(backend="stub", device="cpu", model_id="stub", batch_size=8, obj=stub)
    cfg = KalshiBacktestConfig(
        spec=KalshiSpec(context=64, minutes=(0, 5, 14), vol_lookback=30),
        start=date(2025, 9, 19), end=date(2025, 9, 19), window_chunk=7,  # 7 windows x 3 minutes per call
        prices=(0.5,), taus=(0.52, 0.55), bootstrap_days=20, plot=True, trajectories=2,
        verify_settlement=True, trade=TradeConfig(tau=0.55), run_id="stubrun",
    )
    run_dir = run_kalshi_backtest(settings, cfg, handle=handle)
    assert stub.calls >= 2
    fc = pl.read_parquet(run_dir / "forecasts.parquet")
    assert fc.height == 96 * 3 and set(fc["m"].unique()) == {0, 5, 14}
    # ordering through chunking: the stub's median is the last context value = last_mean, times exp(step*1e-3);
    # a wrong row or a wrong step would be off by >= 1e-3 relative, float32 context noise is ~1e-6
    expected = fc["last_mean"].to_numpy() * np.exp(fc["step"].to_numpy() * 1e-3)
    assert np.allclose(fc["median"].to_numpy(), expected, rtol=5e-6)
    assert not np.allclose(fc["median"].to_numpy(), fc["last_mean"].to_numpy(), rtol=5e-6)
    assert np.all(fc["q10"].to_numpy() < fc["median"].to_numpy()) and np.all(fc["q90"].to_numpy() > fc["median"].to_numpy())
    assert ((fc["p_up"] > 0) & (fc["p_up"] < 1)).all()
    assert (fc["strike_mode"] == "open_tick").all()
    for name in ("summary.txt", "metrics.json", "meta.json", "metrics_by_m.parquet", "trades.parquet", "trades_summary.parquet",
                 "calibration.parquet", "bootstrap.parquet", "windows.parquet", "strike_sensitivity.parquet"):
        assert (run_dir / name).exists(), name
    assert (run_dir / "plots" / "vs_m.png").exists() and list((run_dir / "plots").glob("trajectory_*.png"))
    by_m = pl.read_parquet(run_dir / "metrics_by_m.parquet")
    assert by_m["m"].to_list() == [0, 5, 14]
    assert np.isfinite(by_m["mase_h"].to_numpy()).all() and np.isfinite(by_m["brier"].to_numpy()).all()
    summary = (run_dir / "summary.txt").read_text()
    assert "mase_h" in summary and "RW+vol" in summary

    # re-score under the other strike definition without touching the model
    out = rescore(settings, "stubrun", {"spec": {"strike_mode": "open_avg60"}, "taus": (0.52,), "prices": (0.5,), "plot": False}, label="avg60")
    fc2 = pl.read_parquet(out / "forecasts.parquet")
    assert (fc2["strike_mode"] == "open_avg60").all()
    assert np.allclose(fc2["strike"].to_numpy(), fc2["open_avg60"].to_numpy())
    assert (out / "summary.txt").exists()
