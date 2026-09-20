"""Contract-price backtest: origins -> batched multivariate TimesFM inference -> long-format
forecasts (one row per window, origin minute m and step h) -> metrics against persistence.

Only the model inputs (context, covariate preset, minutes, date range) require the model. The
scoring of `forecasts.parquet` can be repeated without it (`rescore`).
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from . import baselines as B
from . import metrics as M
from .covariates import MAX_STEPS, build_arrays, frame_arrays
from .features import SETTLE_K, load_frame
from .model import DEFAULT_MODEL, ForecasterHandle, load_forecaster, predict
from .origins import ContractSpec, OriginSet, make_origins
from .settings import Settings

log = logging.getLogger(__name__)

M_GROUPS = {"0-4": (0, 4), "5-9": (5, 9), "10-14": (10, 14)}
MIN_FAN_CELLS = 60


@dataclass(frozen=True)
class ContractBacktestConfig:
    spec: ContractSpec = field(default_factory=ContractSpec)
    start: date | None = None
    end: date | None = None
    stride_windows: int = 1
    max_windows: int | None = None
    origin_chunk: int = 256          # origins per predict call (bounds array memory)
    batch_size: int = 32             # TimesFM per_core_batch_size
    device: str | None = None
    model_id: str = DEFAULT_MODEL
    symmetric: bool = False
    fan_lookback_days: int = 30      # persistence fan is fitted on this many days before the evaluation
    fan_from: Path | None = None     # or taken from a file (e.g. a fine-tune run's baselines.parquet)
    bootstrap_days: int = 1000
    seed: int = 0
    plot: bool = False
    trajectories: int = 12
    run_id: str | None = None


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
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    raise TypeError(f"not jsonable: {type(o)}")


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _m_group(m: pl.Expr) -> pl.Expr:
    return pl.when(m <= 4).then(pl.lit("0-4")).when(m <= 9).then(pl.lit("5-9")).otherwise(pl.lit("10-14"))


# ---------------------------------------------------------------------------------------------
# inference

def forecast_origins(os_: OriginSet, handle: ForecasterHandle, cfg: ContractBacktestConfig, predict_fn=predict) -> tuple[pl.DataFrame, int]:
    """Long-format forecasts for every valid (origin, step): columns i, h, k, median, q10..q90, actual.
    Also returns the number of flat input patches that needed jitter."""
    spec = os_.spec
    fa = frame_arrays(os_.frame, spec.covariates)
    i_all = os_.ok_origins.sort("i")["i"].to_numpy()
    cols: dict[str, list[np.ndarray]] = {c: [] for c in ("i", "h", "k", "median", "actual", *M.QUANTILE_COLS)}
    n_flat, t_start = 0, time.time()
    for lo in range(0, i_all.size, cfg.origin_chunk):
        i = i_all[lo:lo + cfg.origin_chunk]
        b = build_arrays(fa, i, spec.context, spec.horizon)
        med, q = predict_fn(handle, b["targets"], spec.horizon, past_only=b["po"], past_future=b["pf"], symmetric=cfg.symmetric, clip=(0.0, 1.0))
        n_flat += int(b["n_flat"])
        r, s = np.nonzero(b["valid"])                          # steps past expiry or without a quote never get a row
        cols["i"].append(i[r])
        cols["h"].append(s + 1)
        cols["k"].append(b["step_k"][r, s])
        cols["median"].append(med[r, s])
        cols["actual"].append(b["tgt"][r, s])
        for j, c in enumerate(M.QUANTILE_COLS):
            cols[c].append(q[r, s, j])
        done = min(lo + cfg.origin_chunk, i_all.size)
        rate = done / max(time.time() - t_start, 1e-9)
        log.info("forecast %d/%d origins (%.1f origins/s, eta %.0fs)", done, i_all.size, rate, (i_all.size - done) / max(rate, 1e-9))
    out = pl.DataFrame({c: np.concatenate(v) for c, v in cols.items()}).with_columns(pl.col("h").cast(pl.Int8), pl.col("k").cast(pl.Int8))
    return out, n_flat


def fit_fan(frame: pl.DataFrame, eval_start: datetime, cfg: ContractBacktestConfig) -> tuple[pl.DataFrame | None, dict]:
    """The persistence fan and where it came from. Never silently in-sample."""
    if cfg.fan_from is not None:
        return pl.read_parquet(cfg.fan_from), {"fan_mode": "file", "fan_source": str(cfg.fan_from)}
    lo = eval_start - timedelta(days=cfg.fan_lookback_days)
    fan = B.persistence_fan(frame, lo, eval_start)
    if B.fan_cells(fan) >= MIN_FAN_CELLS:
        return fan, {"fan_mode": "pre_period", "fan_range": [lo.isoformat(), eval_start.isoformat()]}
    hi = eval_start + timedelta(days=cfg.fan_lookback_days)
    fan = B.persistence_fan(frame, eval_start, hi)
    if B.fan_cells(fan) >= MIN_FAN_CELLS:
        log.warning("no contract history before %s: the persistence fan is fitted on the first %d evaluated days (in-sample for those days)", eval_start, cfg.fan_lookback_days)
        return fan, {"fan_mode": "in_sample_head", "fan_range": [eval_start.isoformat(), hi.isoformat()]}
    log.warning("not enough data for a persistence fan: using a point mass at the last mid (pinball = MAE / 2)")
    return None, {"fan_mode": "point"}


def assemble(long: pl.DataFrame, os_: OriginSet, fan: pl.DataFrame | None, run_id: str) -> pl.DataFrame:
    o = os_.ok_origins.select("i", "t0", "date", "hour", "m", "k_ctx", "origin_ts", "strike", "last_mid", "last_bid", "last_ask",
                              "last_spread", "brti_close", "sigma_1m", "quote_ok", "n_ctx_gaps")
    wins = os_.frame.group_by("t0").agg(pl.col("win_ticker").first().alias("ticker"), pl.col("outcome_price").first(),
                                       pl.col("outcome_brti").first(), pl.col("strike_imputed").first())
    df = long.join(o, on="i", how="inner").join(wins, on="t0", how="left")
    df = df.with_columns(
        pl.lit(run_id).alias("run_id"),
        (pl.col("k") == SETTLE_K).alias("is_settlement"),
        (pl.col("t0") + pl.duration(minutes=pl.col("k"))).alias("bar_ts"),
        _m_group(pl.col("m")).alias("m_group"),
        pl.col("sigma_1m").qcut(3, labels=["0", "1", "2"], allow_duplicates=True).cast(pl.Utf8).cast(pl.Int8).alias("vol_bucket"),
    ).with_columns((pl.col("bar_ts") + pl.duration(minutes=1)).alias("candle_ts"))
    return add_baselines(df, fan).sort("t0", "m", "h")


def add_baselines(df: pl.DataFrame, fan: pl.DataFrame | None) -> pl.DataFrame:
    rw = B.rw_fair_now(df["brti_close"].to_numpy(), df["strike"].to_numpy(), df["sigma_1m"].to_numpy(), df["m"].to_numpy())
    fq = B.apply_fan(df["last_mid"].to_numpy(), df["k_ctx"].to_numpy(), df["h"].to_numpy(), fan)
    return df.with_columns(
        pl.Series("rw_p", rw), pl.Series("persist", B.persist_ref(df["last_mid"].to_numpy(), df["k_ctx"].to_numpy())),
        pl.Series("mean_fc", M.decile_mean(df.select(M.QUANTILE_COLS).to_numpy())), *[pl.Series(c, fq[:, j]) for j, c in enumerate(M.FAN_Q_COLS)],
    )


# ---------------------------------------------------------------------------------------------
# scoring

def score_run(run_dir: Path, forecasts: pl.DataFrame, cfg: ContractBacktestConfig, info: dict) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "metrics_overall": M.metrics_by(forecasts, []),
        "metrics_by_h": M.metrics_by(forecasts, ["h"]),
        "metrics_by_m": M.metrics_by(forecasts, ["m"]),
        "metrics_by_mh": M.metrics_by(forecasts, ["m", "h"]),
        "metrics_by_mgroup": M.metrics_by(forecasts, ["m_group"]),
        "metrics_by_hour": M.metrics_by(forecasts, ["hour"]),
        "metrics_by_vol": M.metrics_by(forecasts, ["vol_bucket"]),
        "metrics_not_settlement": M.metrics_by(forecasts.filter(~pl.col("is_settlement")), []),
    }
    boot = pl.concat([
        M.bootstrap_by(forecasts, "h", cfg.bootstrap_days, cfg.seed).with_columns(pl.lit("h").alias("by"), pl.col("h").cast(pl.Utf8).alias("key")).drop("h"),
        M.bootstrap_by(forecasts, "m_group", cfg.bootstrap_days, cfg.seed).with_columns(pl.lit("m_group").alias("by"), pl.col("m_group").alias("key")).drop("m_group"),
    ])
    tables["bootstrap"] = boot
    settle = forecasts.filter(pl.col("is_settlement") & pl.col("outcome_price").is_not_null())
    cal = []
    if settle.height:
        y = settle["outcome_price"].cast(pl.Float64).to_numpy()
        for src, col in (("model", "mean_fc"), ("persistence", "persist"), ("rw", "rw_p")):
            cal.append(M.calibration_table(settle[col].to_numpy(), y).with_columns(pl.lit(src).alias("source")))
    tables["settlement_calibration"] = pl.concat(cal) if cal else pl.DataFrame()
    for name, df in tables.items():
        if not df.is_empty():
            df.write_parquet(run_dir / f"{name}.parquet")
    forecasts.write_parquet(run_dir / "forecasts.parquet", compression="zstd")

    summary = render_summary(cfg, forecasts, tables, info)
    (run_dir / "summary.txt").write_text(summary)
    out = {"overall": tables["metrics_overall"].to_dicts()[0], "by_h": tables["metrics_by_h"].to_dicts(), "by_m": tables["metrics_by_m"].to_dicts(), "info": info}
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=1, default=_jsonable))
    if cfg.plot:
        try:
            from . import plots

            plots.plot_all(run_dir, forecasts, tables, cfg.trajectories)
        except ImportError as e:
            log.warning("plots skipped (install the `viz` extra): %s", e)
    return out


def _fmt(v, nd: int, width: int) -> str:  # noqa: ANN001
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-".rjust(width)
    if isinstance(v, str):
        return v.rjust(width)
    if isinstance(v, (int, np.integer)) or nd == 0:
        return str(int(v)).rjust(width)
    return f"{v:.{nd}f}".rjust(width)


def _table(df: pl.DataFrame, cols: list[tuple[str, str, int]]) -> str:
    """cols: (column, header, decimals); missing columns are skipped."""
    cols = [c for c in cols if c[0] in df.columns]
    if df.is_empty() or not cols:
        return "(empty)\n"
    widths = [max(len(h), 8) for _, h, _ in cols]
    lines = [" ".join(h.rjust(w) for (_, h, _), w in zip(cols, widths, strict=True))]
    for row in df.iter_rows(named=True):
        lines.append(" ".join(_fmt(row.get(c), nd, w) for (c, _, nd), w in zip(cols, widths, strict=True)))
    return "\n".join(lines) + "\n"


MAIN_COLS = [("n", "n", 0), ("rmse_c", "rmse_c", 3), ("rmse_c_persist", "persist", 3), ("rmse_c_rw", "rw_fair", 3), ("skill_mse", "skill_mse", 4),
             ("mae_c", "mae_c", 3), ("mae_c_persist", "fan_med", 3), ("skill_mae", "skill_mae", 4), ("pinball_c", "pinball", 3),
             ("pinball_c_fan", "pb_fan", 3), ("skill_pinball", "skill_pb", 4), ("cover80", "cover80", 3), ("cover80_fan", "cov80fan", 3),
             ("width80_c", "width80", 2), ("width80_c_fan", "w80_fan", 2), ("bias_c", "bias_c", 3), ("dir_acc", "dir_acc", 3)]
SETTLE_COLS = [("settle_n", "n", 0), ("settle_brier", "brier", 4), ("settle_brier_persist", "persist", 4), ("settle_brier_rw", "rw_fair", 4),
               ("settle_bss_persist", "bss_pers", 4), ("settle_bss_rw", "bss_rw", 4), ("settle_logloss", "logloss", 4),
               ("settle_logloss_persist", "ll_pers", 4), ("settle_ece", "ece", 3)]


def render_summary(cfg: ContractBacktestConfig, forecasts: pl.DataFrame, t: dict[str, pl.DataFrame], info: dict) -> str:
    s, ov = cfg.spec, t["metrics_overall"]
    out = [f"Contract-price backtest: {info.get('model_id', cfg.model_id)}, context {s.context} x 1m, covariates '{s.covariates}' "
           f"({info.get('n_variates', '?')} variates), target mid_close (dollars), errors in cents\n",
           f"windows {forecasts['t0'].n_unique()}, origins {forecasts.select('t0', 'm').n_unique()}, scored steps {forecasts.height}, "
           f"days {forecasts['date'].n_unique()}, span {forecasts['t0'].min()} .. {forecasts['t0'].max()}\n",
           f"persistence fan: {info.get('fan_mode')} {info.get('fan_range', info.get('fan_source', ''))}; flat input patches jittered: {info.get('n_flat', 0)}; "
           f"origins with a degenerate quote: {info.get('n_origins_quote_bad', '?')}\n"]
    if info.get("require_quote_ok"):
        out.append("scored on origins with a usable quote only (spread <= 10c)\n")
    out.append("\n== Overall. rmse: mean of the model's deciles vs persist (last mid; 0.50 at the window open) and rw_fair (index-only fair value). "
               "mae: model median vs the persistence fan's median. skill > 0 beats persistence ==\n")
    out.append(_table(ov, MAIN_COLS))
    out.append("\n== Without the settlement candle ==\n")
    out.append(_table(t["metrics_not_settlement"], MAIN_COLS))
    out.append("\n== By steps ahead h (minutes) ==\n")
    out.append(_table(t["metrics_by_h"], [("h", "h", 0), *MAIN_COLS]))
    out.append("\n== By origin minute m (minutes since the window opened) ==\n")
    out.append(_table(t["metrics_by_m"], [("m", "m", 0), *MAIN_COLS]))
    grid = t["metrics_by_mh"].pivot(on="h", index="m", values="skill_mse").sort("m")
    out.append("\n== skill_mse over the (m, h) triangle (rows m, columns h) ==\n")
    out.append(_table(grid, [("m", "m", 0)] + [(str(h), f"h={h}", 3) for h in range(1, MAX_STEPS + 1)]))
    out.append("\n== Settlement candle, scored as a probability of settling YES (persist = the market's own last price) ==\n")
    out.append(_table(ov, SETTLE_COLS))
    out.append(_table(t["metrics_by_m"], [("m", "m", 0), *SETTLE_COLS]))
    boot = t["bootstrap"]
    if not boot.is_empty():
        out.append(f"\n== 95% day-block bootstrap intervals ({int(boot['n_days'][0])} days) ==\n")
        wide = boot.with_columns((pl.col("by") + "=" + pl.col("key")).alias("group")).pivot(on="stat", index="group", values=["lo", "hi"])
        cols = [("group", "group", 0)]
        for st in ("skill_mse", "skill_mae", "skill_pinball", "mae_c", "pinball_c"):
            cols += [(f"lo_{st}", f"{st}_lo", 4), (f"hi_{st}", f"{st}_hi", 4)]
        out.append(_table(wide, cols))
    out.append("\n== By hour of day (UTC) ==\n")
    out.append(_table(t["metrics_by_hour"], [("hour", "hour", 0), ("n", "n", 0), ("rmse_c", "rmse_c", 3), ("skill_mse", "skill_mse", 4), ("skill_pinball", "skill_pb", 4), ("cover80", "cover80", 3)]))
    out.append("\n== By index volatility tercile (0 = calm) ==\n")
    out.append(_table(t["metrics_by_vol"], [("vol_bucket", "vol", 0), ("n", "n", 0), ("rmse_c", "rmse_c", 3), ("skill_mse", "skill_mse", 4), ("skill_pinball", "skill_pb", 4), ("cover80", "cover80", 3)]))
    out.append("\nThe contract mid is close to a martingale: read skill_mse / skill_pinball and their intervals, not absolute error. "
               "An interval that straddles 0 means no demonstrated edge over the last observed price.\n")
    return "".join(out)


# ---------------------------------------------------------------------------------------------
# entry points

def _warn_if_checkpoint_mismatch(cfg: ContractBacktestConfig) -> None:
    p = Path(cfg.model_id)
    conf = p.parent / "config.json"
    if p.exists() and conf.exists():
        trained = json.loads(conf.read_text())
        for key, mine in (("context", cfg.spec.context), ("covariates", cfg.spec.covariates)):
            if key in trained and trained[key] != mine:
                log.warning("checkpoint %s was fine-tuned with %s=%r but this backtest uses %r: the inputs differ from training", p, key, trained[key], mine)


def run_contract_backtest(settings: Settings, cfg: ContractBacktestConfig, handle: ForecasterHandle | None = None, predict_fn=predict) -> Path:
    t_start = time.time()
    frame = load_frame(settings.processed_dir, cfg.spec.freq)
    os_ = make_origins(frame, cfg.spec, cfg.start, cfg.end, cfg.stride_windows, cfg.max_windows)
    ok = os_.ok_origins
    log.info("%d origins in %d windows; drops:\n%s", ok.height, ok["t0"].n_unique() if ok.height else 0, os_.drop_report())
    if ok.is_empty():
        raise RuntimeError("no usable origins in range (is the contract data downloaded and `chudp frame` built?)")
    _warn_if_checkpoint_mismatch(cfg)
    handle = handle or load_forecaster(cfg.model_id, device=cfg.device, batch_size=cfg.batch_size)
    run_id = cfg.run_id or f"{_utcnow():%Y%m%d-%H%M%S}-contract-c{cfg.spec.context}-{cfg.spec.covariates}"
    run_dir = settings.backtests_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    long, n_flat = forecast_origins(os_, handle, cfg, predict_fn)
    fan, fan_info = fit_fan(frame, ok["t0"].min(), cfg)
    forecasts = assemble(long, os_, fan, run_id)
    if fan is not None:
        fan.write_parquet(run_dir / "baselines.parquet")
    os_.drop_report().write_parquet(run_dir / "drops.parquet")
    info = fan_info | {
        "run_id": run_id, "created_at": _utcnow().isoformat(), "git_sha": _git_sha(), "model_id": handle.model_id, "backend": handle.backend,
        "device": handle.device, "n_variates": frame_arrays(frame.head(1), cfg.spec.covariates).n_variates, "n_origins": ok.height,
        "n_windows": ok["t0"].n_unique(), "n_flat": n_flat, "n_origins_quote_bad": int((~ok["quote_ok"]).sum()),
        "drops": os_.drop_report().to_dicts(), "data_max_ts": frame["ts"].max().isoformat(), "inference_seconds": round(time.time() - t_start, 1),
    }
    (run_dir / "meta.json").write_text(json.dumps({"config": asdict(cfg), **info}, indent=1, default=_jsonable))
    score_run(run_dir, forecasts, cfg, info)
    log.info("run %s done in %.0fs -> %s", run_id, time.time() - t_start, run_dir)
    return run_dir


def rescore(settings: Settings, run_id: str, *, require_quote_ok: bool = False, fan_from: Path | None = None,
            plot: bool = False, bootstrap_days: int | None = None, label: str | None = None) -> Path:
    """Re-score an existing run's forecasts without the model: drop origins with a degenerate quote,
    or swap the persistence fan."""
    src = settings.backtests_dir / run_id
    meta = json.loads((src / "meta.json").read_text())
    raw = dict(meta["config"])
    spec = ContractSpec(**{**raw.pop("spec"), "minutes": tuple(meta["config"]["spec"]["minutes"])})
    for key in ("start", "end"):
        raw[key] = date.fromisoformat(raw[key]) if raw.get(key) else None
    raw["fan_from"] = Path(raw["fan_from"]) if raw.get("fan_from") else None
    cfg = ContractBacktestConfig(spec=spec, **(raw | {"plot": plot, "bootstrap_days": bootstrap_days or raw["bootstrap_days"]}))
    forecasts = pl.read_parquet(src / "forecasts.parquet")
    info = {k: v for k, v in meta.items() if k != "config"}
    if require_quote_ok:
        forecasts = forecasts.filter(pl.col("quote_ok"))
        info["require_quote_ok"] = True
    fan_path = fan_from or (src / "baselines.parquet")
    fan = pl.read_parquet(fan_path) if fan_path.exists() else None
    if fan is not None and "bucket" not in fan.columns:
        raise RuntimeError(f"{fan_path} was written by an older persistence fan; pass --fan-from or re-run the backtest")
    forecasts = add_baselines(forecasts, fan)     # baselines are always recomputed: they are cheap and define the skill
    if fan_from is not None:
        info |= {"fan_mode": "file", "fan_source": str(fan_from)}
    out = src / "rescore" / (label or f"{_utcnow():%Y%m%d-%H%M%S}")
    score_run(out, forecasts, cfg, info)
    return out
