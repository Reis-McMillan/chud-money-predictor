"""Kalshi 15-minute backtest: build windows -> batched TimesFM inference -> score -> report.

Only `context`, `target`, `minutes` and the date range require the model. Strike mode, P(up)
method, no-trade band and the whole trading grid are post-hoc re-scorings of `forecasts.parquet`
(`rescore`).
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from . import metrics as M
from . import trading as T
from .model import ForecasterHandle, load_forecaster, predict
from .resample import load_bars, load_raw, settlement_from_ticks
from .settings import Settings
from .transforms import inverse
from .windows import KalshiSpec, KalshiWindows, context_matrix, make_kalshi_windows

log = logging.getLogger(__name__)

M_GROUPS = {"0-4": (0, 4), "5-9": (5, 9), "10-14": (10, 14)}


@dataclass(frozen=True)
class KalshiBacktestConfig:
    spec: KalshiSpec = field(default_factory=KalshiSpec)
    start: date | None = None
    end: date | None = None
    stride_windows: int = 1
    max_windows: int | None = None
    window_chunk: int = 256
    batch_size: int = 64
    device: str | None = None
    model_id: str = "google/timesfm-3.0-pytorch"
    symmetric: bool = False
    pup_method: str = "pwl_exp"
    eps_bp: float = 0.0
    prices: tuple[float, ...] = (0.40, 0.50, 0.60)
    taus: tuple[float, ...] = (0.52, 0.55, 0.60, 0.65)
    policies: tuple[str, ...] = ("threshold",)
    trade: T.TradeConfig = field(default_factory=T.TradeConfig)
    bootstrap_days: int = 1000
    seed: int = 0
    plot: bool = False
    trajectories: int = 20
    verify_settlement: bool = False
    quotes_path: Path | None = None
    allow_target_mismatch: bool = False
    run_id: str | None = None

    def headline(self) -> tuple[float, float]:
        c = 0.5 if 0.5 in self.prices else self.prices[0]
        tau = self.trade.tau if self.trade.tau in self.taus else self.taus[0]
        return c, tau


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def _jsonable(o):  # noqa: ANN001
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(f"not jsonable: {type(o)}")


# ---------------------------------------------------------------------------------------------
# inference

def forecast_origins(
    kw: KalshiWindows,
    handle: ForecasterHandle,
    cfg: KalshiBacktestConfig,
    predict_fn=predict,
) -> pl.DataFrame:
    """Run the model over all ok origins; returns origins + median/q10..q90 (price space)."""
    spec = kw.spec
    origins = kw.ok_origins.sort("t0", "m")
    n = origins.height
    ctx_end = origins["ctx_end_idx"].to_numpy()
    steps = origins["step"].to_numpy().astype(np.int64)
    per_window = len(spec.minutes)
    chunk_rows = max(per_window, cfg.window_chunk * per_window)
    med_out = np.empty(n)
    q_out = np.empty((n, len(M.QUANTILE_COLS)))
    t_start = time.time()
    for lo in range(0, n, chunk_rows):
        hi = min(n, lo + chunk_rows)
        X = context_matrix(kw.features, ctx_end[lo:hi], spec.context)
        med, q = predict_fn(handle, X, spec.horizon, quantiles=True, symmetric=cfg.symmetric)
        rows = np.arange(hi - lo)
        med_out[lo:hi] = med[rows, steps[lo:hi]]
        q_out[lo:hi] = q[rows, steps[lo:hi], :]
        done = hi
        rate = done / max(time.time() - t_start, 1e-9)
        log.info("forecast %d/%d origins (%.1f origins/s, eta %.0fs)", done, n, rate, (n - done) / max(rate, 1e-9))
    med_out = inverse(med_out, spec.transform)
    q_out = inverse(q_out, spec.transform)
    out = origins.with_columns(pl.Series("median", med_out))
    for i, col in enumerate(M.QUANTILE_COLS):
        out = out.with_columns(pl.Series(col, q_out[:, i]))
    return out


# ---------------------------------------------------------------------------------------------
# assembly and scoring

def assemble_forecasts(preds: pl.DataFrame, kw: KalshiWindows, cfg: KalshiBacktestConfig, run_id: str) -> pl.DataFrame:
    w = kw.ok_windows.select(
        "t0", "date", "hour", "strike", "strike_mode", "strike_ts", "strike_stale_s",
        pl.col("strike_tick") if "strike_tick" in kw.windows.columns else pl.lit(None, dtype=pl.Float64).alias("strike_tick"),
        "open_avg60", "settlement", "settlement_n_ticks", "vol_1m", "label_up", "tie",
    )
    df = preds.join(w, on="t0", how="inner")
    df = df.with_columns(
        pl.lit(run_id).alias("run_id"),
        pl.lit(kw.spec.context).alias("context_len"),
        pl.col("vol_1m").qcut(3, labels=["0", "1", "2"], allow_duplicates=True).cast(pl.Utf8).cast(pl.Int8).alias("vol_bucket"),
    )
    return score_columns(df, cfg)


def score_columns(df: pl.DataFrame, cfg: KalshiBacktestConfig) -> pl.DataFrame:
    """(Re)compute the strike-dependent columns: label, p_up, direction, errors."""
    spec = cfg.spec
    if spec.target_col != "mean" and not cfg.allow_target_mismatch:
        raise ValueError(
            "quantiles of the bar close are not quantiles of the 60-s settlement mean; "
            "pass --allow-target-mismatch to compute P(up) anyway"
        )
    if spec.strike_mode == "open_avg60":
        df = df.with_columns(pl.col("open_avg60").alias("strike"), pl.lit("open_avg60").alias("strike_mode"))
    elif "strike_tick" in df.columns and df["strike_tick"].null_count() < df.height:
        df = df.with_columns(pl.col("strike_tick").alias("strike"), pl.lit("open_tick").alias("strike_mode"))
    df = df.filter(pl.col("strike").is_not_null())
    q = df.select(M.QUANTILE_COLS).to_numpy()
    strike = df["strike"].to_numpy()
    p_up = M.p_up_from_quantiles(q, strike, cfg.pup_method)
    p_rw = M.rwvol_p_up(df["last_close"].to_numpy(), strike, df["sigma_1m"].to_numpy(), df["m"].to_numpy())
    return df.with_columns(
        (pl.col("settlement") > pl.col("strike")).alias("label_up"),
        (pl.col("settlement") == pl.col("strike")).alias("tie"),
        pl.Series("p_up", p_up),
        pl.Series("p_up_rwvol", p_rw),
        pl.lit(0.5).alias("p_up_const"),
        (pl.col("median") > pl.col("strike")).alias("pred_up"),
        (pl.col("median") - pl.col("settlement")).abs().alias("abs_err"),
        (pl.col("last_close") - pl.col("settlement")).abs().alias("naive_abs_err"),
    )


def _m_group(m: pl.Expr) -> pl.Expr:
    return (
        pl.when(m <= 4).then(pl.lit("0-4")).when(m <= 9).then(pl.lit("5-9")).otherwise(pl.lit("10-14"))
    )


def score_run(run_dir: Path, forecasts: pl.DataFrame, cfg: KalshiBacktestConfig, quotes: pl.DataFrame | None = None) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    by_m = M.metrics_by(forecasts, ["m"], cfg.eps_bp)
    overall = M.metrics_by(forecasts, [], cfg.eps_bp)
    by_hour = M.metrics_by(forecasts, ["hour"], cfg.eps_bp)
    by_vol = M.metrics_by(forecasts, ["vol_bucket"], cfg.eps_bp)
    by_group = M.metrics_by(forecasts.with_columns(_m_group(pl.col("m")).alias("m_group")), ["m_group"], cfg.eps_bp)
    cal_rows = []
    for name, (lo, hi) in M_GROUPS.items():
        g = forecasts.filter((pl.col("m") >= lo) & (pl.col("m") <= hi))
        if g.is_empty():
            continue
        y = g["label_up"].cast(pl.Float64).to_numpy()
        for src, col in (("model", "p_up"), ("rwvol", "p_up_rwvol")):
            cal_rows.append(M.calibration_table(g[col].to_numpy(), y).with_columns(pl.lit(name).alias("m_group"), pl.lit(src).alias("source")))
    calibration = pl.concat(cal_rows) if cal_rows else pl.DataFrame()

    trades = T.sweep(forecasts, cfg.prices, cfg.taus, cfg.trade, cfg.policies, quotes=quotes)
    ts_m = T.trade_summary(trades, ["source", "c", "tau", "policy", "m"], forecasts)
    ts_all = T.trade_summary(trades, ["source", "c", "tau", "policy"], forecasts)
    boot = M.bootstrap_by_m(forecasts, cfg.bootstrap_days, cfg.seed)

    # strike sensitivity: re-score under the other strike mode if both strikes are available
    sens = None
    other = "open_avg60" if cfg.spec.strike_mode == "open_tick" else "open_tick"
    if other == "open_avg60" or ("strike_tick" in forecasts.columns and forecasts["strike_tick"].null_count() < forecasts.height):
        alt = score_columns(forecasts, replace(cfg, spec=cfg.spec.with_strike_mode(other)))
        sens = M.metrics_by(alt, ["m"], cfg.eps_bp).select("m", "accuracy", "brier", "bss_rw").rename(
            {"accuracy": f"accuracy_{other}", "brier": f"brier_{other}", "bss_rw": f"bss_rw_{other}"}
        )

    for name, df in (
        ("forecasts", forecasts), ("metrics_by_m", by_m), ("metrics_by_hour", by_hour), ("metrics_by_vol", by_vol),
        ("metrics_by_mgroup", by_group), ("calibration", calibration), ("trades", trades),
        ("trades_summary_by_m", ts_m), ("trades_summary", ts_all), ("bootstrap", boot),
    ):
        if df is not None and not df.is_empty():
            df.write_parquet(run_dir / f"{name}.parquet")
    if sens is not None:
        sens.write_parquet(run_dir / "strike_sensitivity.parquet")

    c, tau = cfg.headline()
    summary = render_summary(cfg, forecasts, by_m, overall, ts_m, ts_all, boot, calibration, by_hour, by_vol, sens, c, tau)
    (run_dir / "summary.txt").write_text(summary)
    metrics_json = {
        "overall": overall.to_dicts()[0] if not overall.is_empty() else {},
        "by_m": by_m.to_dicts(),
        "trades_summary": ts_all.to_dicts() if not ts_all.is_empty() else [],
        "headline": {"c": c, "tau": tau},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=1, default=_jsonable))

    if cfg.plot:
        try:
            from . import plots

            plots.plot_all(run_dir, forecasts, by_m, boot, trades, ts_m, calibration, c, tau, cfg.trajectories)
        except ImportError as e:
            log.warning("plots skipped (install the `viz` extra): %s", e)
    return metrics_json


# ---------------------------------------------------------------------------------------------
# summary text

def _fmt(v, nd=3, width=8) -> str:  # noqa: ANN001
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-".rjust(width)
    if isinstance(v, (int, np.integer)):
        return str(int(v)).rjust(width)
    return f"{v:.{nd}f}".rjust(width)


def _table(df: pl.DataFrame, cols: list[tuple[str, str, int]], sort: list[str] | None = None) -> str:
    """cols: (column, header, decimals)."""
    if df.is_empty():
        return "(empty)\n"
    if sort:
        df = df.sort(sort)
    widths = [max(len(h), 8) for _, h, _ in cols]
    lines = [" ".join(h.rjust(w) for (_, h, _), w in zip(cols, widths, strict=True))]
    for row in df.iter_rows(named=True):
        cells = []
        for (c, _, nd), w in zip(cols, widths, strict=True):
            v = row.get(c)
            cells.append(_fmt(v, nd, w) if not isinstance(v, str) else v.rjust(w))
        lines.append(" ".join(cells))
    return "\n".join(lines) + "\n"


def render_summary(cfg, forecasts, by_m, overall, ts_m, ts_all, boot, calibration, by_hour, by_vol, sens, c, tau) -> str:  # noqa: ANN001
    s = cfg.spec
    out = []
    out.append(f"Kalshi 15m backtest — model {cfg.model_id}, context {s.context} x {s.freq} bars of {s.target_col} ({s.transform}), "
               f"strike {s.strike_mode}, p_up {cfg.pup_method}, eps {cfg.eps_bp} bp\n")
    out.append(f"windows {forecasts['t0'].n_unique()}, origins {forecasts.height}, days {forecasts['date'].n_unique()}, "
               f"span {forecasts['t0'].min()} .. {forecasts['t0'].max()}, base rate up {overall['base_rate_up'][0]:.3f}, ties {int(overall['tie_count'][0])}\n")
    out.append("\n== Model, by minutes since window open (m); h_bars = bars to settlement ==\n")
    main_cols = [("m", "m", 0), ("n", "n", 0), ("mae_bp", "mae_bp", 2), ("mase_h", "mase_h", 3), ("mase_1s", "mase_1s", 3),
                 ("accuracy", "acc", 3), ("accuracy_naive", "acc_naive", 3), ("precision_up", "prec_up", 3), ("recall_up", "rec_up", 3),
                 ("abstain_rate", "abstain", 3), ("brier", "brier", 4), ("brier_rwvol", "brier_rw", 4), ("bss_rw", "bss_rw", 4),
                 ("logloss", "logloss", 4), ("ece", "ece", 3), ("sharpness", "sharp", 3)]
    all_row = overall.with_columns(pl.lit("ALL").alias("m"))
    out.append(_table(pl.concat([by_m.with_columns(pl.col("m").cast(pl.Utf8)), all_row.select(by_m.with_columns(pl.col("m").cast(pl.Utf8)).columns)], how="vertical_relaxed"), main_cols))
    if not boot.is_empty():
        out.append("\n== 95% day-block bootstrap CIs (model) ==\n")
        piv = boot.select("m").unique().sort("m")
        cols = [("m", "m", 0)]
        for st in ("accuracy", "brier", "mae", "mase_h"):
            s = boot.filter(pl.col("stat") == st).select("m", pl.col("lo").alias(f"{st}_lo"), pl.col("hi").alias(f"{st}_hi"))
            if not s.is_empty():
                piv = piv.join(s, on="m", how="left")
                cols += [(f"{st}_lo", f"{st}_lo", 4), (f"{st}_hi", f"{st}_hi", 4)]
        out.append(_table(piv, cols, sort=["m"]))
    if not ts_m.is_empty():
        out.append(f"\n== Trading at c={c:.2f}, tau={tau:.2f}, policy {cfg.trade.policy}, fees {'on' if cfg.trade.fees else 'off'}: model vs RW+vol on the same policy ==\n")
        h = ts_m.filter((pl.col("c") == c) & (pl.col("tau") == tau) & (pl.col("policy") == cfg.trade.policy))
        tcols = [("source", "source", 0), ("m", "m", 0), ("n_trades", "trades", 0), ("trade_rate", "trade_rate", 3), ("win_rate", "win%", 3),
                 ("mean_price_paid", "price", 3), ("realized_edge", "edge", 4), ("mean_pnl_net", "pnl/trade", 4),
                 ("pnl_per_window", "pnl/window", 4), ("total_pnl_net", "total", 2), ("t_stat", "t", 2), ("max_drawdown", "maxdd", 2)]
        out.append(_table(h, tcols, sort=["source", "m"]))
        out.append("\n== Trading grid (all m pooled) ==\n")
        gcols = [("source", "source", 0), ("c", "c", 2), ("tau", "tau", 2), ("policy", "policy", 0), ("n_trades", "trades", 0),
                 ("win_rate", "win%", 3), ("mean_pnl_net", "pnl/trade", 4), ("pnl_per_window", "pnl/window", 4), ("total_pnl_net", "total", 2), ("t_stat", "t", 2)]
        out.append(_table(ts_all, gcols, sort=["source", "c", "tau"]))
    if not calibration.is_empty():
        out.append("\n== Calibration (model) ==\n")
        ccols = [("m_group", "m_group", 0), ("bin_lo", "bin_lo", 2), ("bin_hi", "bin_hi", 2), ("n", "n", 0), ("p_mean", "p_mean", 3), ("freq_up", "freq_up", 3), ("gap", "gap", 3)]
        out.append(_table(calibration.filter(pl.col("source") == "model"), ccols))
    out.append("\n== By hour of day (UTC) ==\n")
    out.append(_table(by_hour, [("hour", "hour", 0), ("n", "n", 0), ("accuracy", "acc", 3), ("brier", "brier", 4), ("bss_rw", "bss_rw", 4), ("mase_h", "mase_h", 3)], sort=["hour"]))
    out.append("\n== By trailing-vol tercile (0 = calm) ==\n")
    out.append(_table(by_vol, [("vol_bucket", "vol", 0), ("n", "n", 0), ("accuracy", "acc", 3), ("brier", "brier", 4), ("bss_rw", "bss_rw", 4), ("mase_h", "mase_h", 3)], sort=["vol_bucket"]))
    if sens is not None:
        out.append("\n== Strike sensitivity (same forecasts, other strike definition) ==\n")
        out.append(_table(sens, [(c_, c_, 4) for c_ in sens.columns], sort=["m"]))
    out.append("\nRead the model minus RW+vol difference, not absolute P&L: fixed-price fills ignore that real quotes already price the drift from strike.\n")
    return "".join(out)


# ---------------------------------------------------------------------------------------------
# entry points

def load_quotes(path: Path | None) -> pl.DataFrame | None:
    if path is None:
        return None
    q = pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path, try_parse_dates=True)
    need = {"t0", "m", "yes_bid", "yes_ask"}
    if not need <= set(q.columns):
        raise ValueError(f"quotes file needs columns {sorted(need)}")
    return q.select("t0", "m", "yes_bid", "yes_ask")


def run_kalshi_backtest(settings: Settings, cfg: KalshiBacktestConfig, handle: ForecasterHandle | None = None, predict_fn=predict) -> Path:
    t0 = time.time()
    bars = load_bars(settings.processed_dir, cfg.spec.freq)
    ticks = load_raw(settings.raw_dir) if (cfg.spec.strike_mode == "open_tick" or cfg.verify_settlement) else None
    kw = make_kalshi_windows(bars, ticks, cfg.spec, cfg.start, cfg.end, cfg.stride_windows, cfg.max_windows)
    n_w, n_o = kw.ok_windows.height, kw.ok_origins.height
    log.info("%d windows, %d origins ok; drops:\n%s", n_w, n_o, kw.drop_report())
    if n_o == 0:
        raise RuntimeError("no usable origins in range")

    if cfg.verify_settlement and ticks is not None:
        sample = kw.ok_windows.head(50)
        for t0_, s_bar in zip(sample["t0"], sample["settlement"], strict=True):
            s_tick, n = settlement_from_ticks(ticks, t0_)
            if s_tick is None or abs(s_tick - s_bar) > 1e-6:
                raise AssertionError(f"settlement mismatch at {t0_}: bar mean {s_bar} vs ticks {s_tick} ({n} ticks)")
        log.info("settlement check passed on %d windows", sample.height)

    handle = handle or load_forecaster(cfg.model_id, device=cfg.device, batch_size=cfg.batch_size)
    run_id = cfg.run_id or f"{datetime.now(UTC).replace(tzinfo=None):%Y%m%d-%H%M%S}-kalshi-c{cfg.spec.context}-{cfg.spec.target_col}"
    run_dir = settings.backtests_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    preds = forecast_origins(kw, handle, cfg, predict_fn)
    forecasts = assemble_forecasts(preds, kw, cfg, run_id)
    kw.windows.write_parquet(run_dir / "windows.parquet")
    kw.drop_report().write_parquet(run_dir / "drops.parquet")

    meta = {
        "run_id": run_id, "created_at": datetime.now(UTC).replace(tzinfo=None).isoformat(), "git_sha": _git_sha(),
        "model_id": handle.model_id, "backend": handle.backend, "device": handle.device, "batch_size": handle.batch_size,
        "config": asdict(cfg), "n_windows": n_w, "n_origins": n_o, "drops": kw.drop_report().to_dicts(),
        "data_first_ts": bars["ts"].min().isoformat(), "data_max_ts": bars["ts"].max().isoformat(),
        "inference_seconds": round(time.time() - t0, 1),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=_jsonable))
    score_run(run_dir, forecasts, cfg, load_quotes(cfg.quotes_path))
    log.info("run %s done in %.0fs -> %s", run_id, time.time() - t0, run_dir)
    return run_dir


def rescore(settings: Settings, run_id: str, cfg_overrides: dict, label: str | None = None) -> Path:
    src = settings.backtests_dir / run_id
    meta = json.loads((src / "meta.json").read_text())
    raw_cfg = meta["config"]
    spec_kw = dict(raw_cfg.pop("spec"))
    trade_kw = dict(raw_cfg.pop("trade"))
    for k in ("start", "end"):
        raw_cfg[k] = date.fromisoformat(raw_cfg[k]) if raw_cfg.get(k) else None
    raw_cfg["quotes_path"] = Path(raw_cfg["quotes_path"]) if raw_cfg.get("quotes_path") else None
    for k in ("prices", "taus", "policies"):
        raw_cfg[k] = tuple(raw_cfg[k])
    spec_kw["minutes"] = tuple(spec_kw["minutes"])
    spec_kw.update({k: v for k, v in cfg_overrides.pop("spec", {}).items() if v is not None})
    trade_kw.update({k: v for k, v in cfg_overrides.pop("trade", {}).items() if v is not None})
    raw_cfg.update({k: v for k, v in cfg_overrides.items() if v is not None})
    cfg = KalshiBacktestConfig(spec=KalshiSpec(**spec_kw), trade=T.TradeConfig(**trade_kw), **raw_cfg)
    forecasts = score_columns(pl.read_parquet(src / "forecasts.parquet"), cfg)
    out = src / "rescore" / (label or f"{datetime.now(UTC).replace(tzinfo=None):%Y%m%d-%H%M%S}")
    score_run(out, forecasts, cfg, load_quotes(cfg.quotes_path))
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1, default=_jsonable))
    return out
