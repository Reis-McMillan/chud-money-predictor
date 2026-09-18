"""Single latest-window forecast (generic, not Kalshi-specific): the last `context` bars up to a
given time -> 64 steps of median + deciles in price space."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from . import metrics as M
from .model import ForecasterHandle, load_forecaster, predict
from .resample import freq_seconds, load_bars
from .settings import Settings
from .transforms import forward, inverse

log = logging.getLogger(__name__)


def run_forecast(
    settings: Settings,
    *,
    context: int = 4096,
    horizon: int = 64,
    at: datetime | None = None,
    freq: str = "1m",
    target_col: str = "mean",
    transform: str = "log",
    model_id: str = "google/timesfm-3.0-pytorch",
    device: str | None = None,
    batch_size: int = 8,
    handle: ForecasterHandle | None = None,
    plot: bool = False,
) -> Path:
    bars = load_bars(settings.processed_dir, freq)
    if at is not None:
        bars = bars.filter(pl.col("ts") + pl.duration(seconds=freq_seconds(freq)) <= at)
    bars = bars.filter(pl.col(target_col).is_not_null())
    if bars.height < context:
        raise ValueError(f"only {bars.height} bars available, need {context}")
    ctx = bars.tail(context)
    y = forward(ctx[target_col].to_numpy().astype(np.float64), transform).astype(np.float32)
    handle = handle or load_forecaster(model_id, device=device, batch_size=batch_size)
    med, q = predict(handle, [y], horizon)
    med, q = inverse(med[0], transform), inverse(q[0], transform)
    origin_ts: datetime = ctx["ts"][-1]
    step_td = timedelta(seconds=freq_seconds(freq))
    run_id = f"{datetime.now(UTC).replace(tzinfo=None):%Y%m%d-%H%M%S}-forecast"
    df = pl.DataFrame({
        "run_id": [run_id] * horizon,
        "origin_ts": [origin_ts] * horizon,
        "step": list(range(horizon)),
        "ts": [origin_ts + step_td * (k + 1) for k in range(horizon)],
        "median": med,
        **{col: q[:, i] for i, col in enumerate(M.QUANTILE_COLS)},
        "last_value": [float(ctx[target_col][-1])] * horizon,
    })
    settings.forecasts_dir.mkdir(parents=True, exist_ok=True)
    out = settings.forecasts_dir / f"{run_id}.parquet"
    df.write_parquet(out)
    (settings.forecasts_dir / f"{run_id}.json").write_text(json.dumps({
        "run_id": run_id, "model_id": handle.model_id, "device": handle.device, "context": context, "horizon": horizon,
        "freq": freq, "target_col": target_col, "transform": transform, "origin_ts": origin_ts.isoformat(),
    }, indent=1))
    if plot:
        try:
            from .plots import plot_forecast

            plot_forecast(ctx, df, settings.forecasts_dir / f"{run_id}.png", target_col)
        except ImportError as e:
            log.warning("plot skipped (install the `viz` extra): %s", e)
    log.info("forecast from %s written to %s", origin_ts, out)
    return out
