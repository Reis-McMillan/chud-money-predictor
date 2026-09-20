"""`chudp` command line."""

from __future__ import annotations

import logging
import tomllib
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import typer

from .settings import Settings, load_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="TimesFM 3.0 forecasts of the Kalshi KXBTC15M contract price.")
qdb_app = typer.Typer(help="QuestDB inspection.")
app.add_typer(qdb_app, name="qdb")

_state: dict[str, Any] = {"settings": None, "config": {}}
log = logging.getLogger("chudp")


def _settings() -> Settings:
    return _state["settings"]


def _load_config(path: Path | None) -> None:
    if path:
        _state["config"] = tomllib.loads(path.read_text())
        log.info("config %s: %s", path, _state["config"])


def _pick(flag: Any, section: str, key: str, default: Any) -> Any:
    """Explicit CLI flag (not None) > config file > default."""
    return flag if flag is not None else _state["config"].get(section, {}).get(key, default)


def _date(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


@app.callback()
def main(
    config: Annotated[Path | None, typer.Option("--config", help="TOML config; also accepted after the sub-command")] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    env_file: Annotated[Path, typer.Option("--env-file")] = Path(".env"),
    verbose: Annotated[int, typer.Option("-v", count=True)] = 0,
) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose > 1 else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _state["settings"] = load_settings(env_file, data_dir)
    _load_config(config)


# ---------------------------------------------------------------------------------------------

@qdb_app.command("info")
def qdb_info() -> None:
    """Row counts, time bounds and per-day coverage of both source tables."""
    from .qdb import SOURCES, QuestDB

    q = QuestDB(_settings())
    typer.echo(f"questdb {q.ping()} at {_settings().qdb_url}; tables: {', '.join(q.tables())}")
    for spec in SOURCES.values():
        b = q.bounds(spec)
        typer.echo(f"[{spec.name}] {spec.table} where {spec.where}: {b.rows:,} rows, {b.min_ts} .. {b.max_ts}")
        if b.max_ts:
            counts = q.day_counts(spec, b.max_ts)
            vals = sorted(counts.values())
            typer.echo(f"    {len(counts)} days; rows/day min {vals[0]:,}, median {vals[len(vals) // 2]:,}, max {vals[-1]:,}")


@app.command()
def download(
    source: Annotated[str, typer.Option(help="both | brti | contracts")] = "both",
    start: Annotated[str | None, typer.Option(help="first UTC day, YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option(help="last UTC day, YYYY-MM-DD")] = None,
    jobs: int = 4,
    force: bool = False,
    verify_only: bool = False,
) -> None:
    """Pull raw rows into data/raw/<source>/date=YYYY-MM-DD.parquet (incremental, idempotent)."""
    from .download import download as _download
    from .qdb import SOURCES, QuestDB

    names = list(SOURCES) if source == "both" else [source]
    if any(n not in SOURCES for n in names):
        raise typer.BadParameter(f"source must be one of both, {', '.join(SOURCES)}")
    q = QuestDB(_settings())
    failed = False
    for name in names:
        rep = _download(q, _settings().raw_dir_for(name), SOURCES[name], start=_date(start), end=_date(end), jobs=jobs, force=force, verify_only=verify_only)
        if verify_only:
            typer.echo(f"[{name}] would fetch {len(rep.planned)} days: {', '.join(d.isoformat() for d in rep.planned[:10])}{' ...' if len(rep.planned) > 10 else ''}")
        typer.echo(rep.summary())
        failed |= bool(rep.failed)
    if failed:
        raise typer.Exit(code=1)


@app.command()
def resample(freq: str = "1m", force: bool = False) -> None:
    """Build regular BRTI bars from the on-the-second ticks (homogeneous across the 1 Hz -> 5 Hz change)."""
    from .resample import build

    out, meta = build(_settings().raw_dir, _settings().processed_dir, freq, force)
    typer.echo(f"{out}: {meta['n_bars']:,} bars, {meta['n_gaps']} gaps, {meta['n_not_full']} not full, largest gap {meta['largest_gap_bars']} bars, "
               f"{meta['first_ts']} .. {meta['last_ts']}; days at 1 Hz {meta['days_once_per_second']}, sub-second {meta['days_sub_second']}")


@app.command()
def frame(force: bool = False, vol_lookback: int = 240) -> None:
    """Join the BRTI bars with the contract candles into the 1-minute model frame and print its audit."""
    from .features import build_frame

    out, meta = build_frame(_settings().processed_dir, _settings().contracts_raw_dir, vol_lookback=vol_lookback, force=force)
    typer.echo(f"{out}: {meta['n_rows']:,} rows {meta['first_ts']} .. {meta['last_ts']}")
    typer.echo(f"  contract candles on {meta['n_rows_with_candle']:,} rows ({meta['contract_first_ts']} .. {meta['contract_last_ts']}); "
               f"{meta['n_windows']:,} windows, {meta['n_complete_windows']:,} with all 15 candles")
    typer.echo(f"  strikes imputed from BRTI in {meta['n_windows_strike_imputed']} windows; |kalshi strike - BRTI mean of the minute before| "
               f"median {meta['strike_identity_abs_err_p50']}, p99 {meta['strike_identity_abs_err_p99']} USD")
    typer.echo(f"  rows with a degenerate quote (spread > 10c): {meta['n_rows_quote_degenerate']:,}; "
               f"price vs index settlement disagreement: {meta['outcome_disagreement_rate']}")


@app.command("backtest-contract")
def backtest_contract(
    config: Annotated[Path | None, typer.Option("--config", help="TOML with [spec] and [run]; flags override")] = None,
    context: Annotated[int | None, typer.Option(help="context rows (multiple of 32)")] = None,
    covariates: Annotated[str | None, typer.Option(help="full | no_fair_path | brti_only | quotes_only | calendar_only | none")] = None,
    minutes: Annotated[str | None, typer.Option(help="origin minutes, '0-14' | '0,5,10,14'")] = None,
    max_ctx_gap_frac: Annotated[float | None, typer.Option()] = None,
    require_quote_ok: Annotated[bool | None, typer.Option("--require-quote-ok/--no-require-quote-ok")] = None,
    start: Annotated[str | None, typer.Option()] = None,
    end: Annotated[str | None, typer.Option()] = None,
    stride_windows: Annotated[int | None, typer.Option()] = None,
    max_windows: Annotated[int | None, typer.Option()] = None,
    origin_chunk: Annotated[int | None, typer.Option()] = None,
    batch_size: Annotated[int | None, typer.Option()] = None,
    device: Annotated[str | None, typer.Option(help="auto | cpu | mps | cuda")] = None,
    model_id: Annotated[str | None, typer.Option(help="HF repo id or a fine-tuned checkpoint directory")] = None,
    symmetric: Annotated[bool | None, typer.Option("--symmetric/--no-symmetric")] = None,
    fan_lookback_days: Annotated[int | None, typer.Option()] = None,
    fan_from: Annotated[Path | None, typer.Option(help="persistence fan file, e.g. data/finetune/<run>/baselines.parquet")] = None,
    bootstrap_days: Annotated[int | None, typer.Option()] = None,
    plot: Annotated[bool | None, typer.Option("--plot/--no-plot")] = None,
    trajectories: Annotated[int | None, typer.Option()] = None,
    run_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Forecast the contract mid at every remaining minute, from every minute of every window, and score it against persistence."""
    from .backtest import ContractBacktestConfig, run_contract_backtest
    from .origins import ContractSpec

    _load_config(config)
    spec = ContractSpec(
        context=_pick(context, "spec", "context", 1024), covariates=_pick(covariates, "spec", "covariates", "full"),
        minutes=_pick(minutes, "spec", "minutes", "0-14"), max_ctx_gap_frac=_pick(max_ctx_gap_frac, "spec", "max_ctx_gap_frac", 0.01),
        require_quote_ok=_pick(require_quote_ok, "spec", "require_quote_ok", False),
    )
    dev = _pick(device, "run", "device", None)
    cfg = ContractBacktestConfig(
        spec=spec, start=_date(_pick(start, "run", "start", None)), end=_date(_pick(end, "run", "end", None)),
        stride_windows=_pick(stride_windows, "run", "stride_windows", 1), max_windows=_pick(max_windows, "run", "max_windows", None),
        origin_chunk=_pick(origin_chunk, "run", "origin_chunk", 256), batch_size=_pick(batch_size, "run", "batch_size", 32),
        device=None if dev == "auto" else dev, model_id=_pick(model_id, "run", "model_id", "google/timesfm-3.0-pytorch"),
        symmetric=_pick(symmetric, "run", "symmetric", False), fan_lookback_days=_pick(fan_lookback_days, "run", "fan_lookback_days", 30),
        fan_from=fan_from, bootstrap_days=_pick(bootstrap_days, "run", "bootstrap_days", 1000), plot=_pick(plot, "run", "plot", False),
        trajectories=_pick(trajectories, "run", "trajectories", 12), run_id=run_id,
    )
    log.info("resolved config: %s", cfg)
    out = run_contract_backtest(_settings(), cfg)
    typer.echo((out / "summary.txt").read_text())
    typer.echo(f"run dir: {out}")


@app.command()
def rescore(
    run_id: str,
    require_quote_ok: Annotated[bool, typer.Option("--require-quote-ok/--no-require-quote-ok", help="drop origins whose quote is degenerate")] = False,
    fan_from: Annotated[Path | None, typer.Option(help="swap the persistence fan")] = None,
    bootstrap_days: Annotated[int | None, typer.Option()] = None,
    plot: bool = False,
    label: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Re-score an existing run's forecasts (no model needed)."""
    from .backtest import rescore as _rescore

    out = _rescore(_settings(), run_id, require_quote_ok=require_quote_ok, fan_from=fan_from, plot=plot, bootstrap_days=bootstrap_days, label=label)
    typer.echo((out / "summary.txt").read_text())
    typer.echo(f"rescore dir: {out}")


@app.command()
def finetune(
    config: Annotated[Path | None, typer.Option("--config", help="TOML with a [finetune] section; flags override")] = None,
    context: Annotated[int | None, typer.Option()] = None,
    covariates: Annotated[str | None, typer.Option()] = None,
    start: Annotated[str | None, typer.Option(help="clip the data before splitting, YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option()] = None,
    val_frac: Annotated[float | None, typer.Option()] = None,
    test_frac: Annotated[float | None, typer.Option()] = None,
    val_start: Annotated[str | None, typer.Option(help="explicit validation start (overrides the fractions)")] = None,
    test_start: Annotated[str | None, typer.Option(help="explicit test start (overrides the fractions)")] = None,
    embargo_bars: Annotated[int | None, typer.Option()] = None,
    model_id: Annotated[str | None, typer.Option(help="HF repo id or a local checkpoint dir to continue from")] = None,
    device: Annotated[str | None, typer.Option(help="auto | cpu | mps | cuda")] = None,
    precision: Annotated[str | None, typer.Option(help="fp32 | bf16")] = None,
    trainable: Annotated[str | None, typer.Option(help="all | head | last:N")] = None,
    loss_scale: Annotated[str | None, typer.Option(help="none | fan")] = None,
    lr: Annotated[float | None, typer.Option()] = None,
    max_steps: Annotated[int | None, typer.Option()] = None,
    batch_size: Annotated[int | None, typer.Option(help="origins per optimiser step")] = None,
    micro_batch: Annotated[int | None, typer.Option(help="origins per forward/backward pass")] = None,
    eval_every: Annotated[int | None, typer.Option()] = None,
    eval_samples: Annotated[int | None, typer.Option(help="origins scored per split; 0 = all")] = None,
    patience: Annotated[int | None, typer.Option()] = None,
    seed: Annotated[int | None, typer.Option()] = None,
    run_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Fine-tune TimesFM on the contract price with a chronological train / validation / test split."""
    from .finetune import FinetuneConfig, run_finetune

    _load_config(config)
    flags = {k: v for k, v in locals().items() if k != "config" and v is not None and k in FinetuneConfig.__dataclass_fields__}
    merged = {**_state["config"].get("finetune", {}), **flags}
    for k in ("start", "end", "val_start", "test_start"):
        if isinstance(merged.get(k), str):
            merged[k] = date.fromisoformat(merged[k])
    if merged.get("device") == "auto":
        merged["device"] = None
    try:
        cfg = FinetuneConfig(**merged)
    except TypeError as e:
        raise typer.BadParameter(f"unknown [finetune] option: {e}") from e
    log.info("resolved config: %s", cfg)
    out = run_finetune(_settings(), cfg)
    typer.echo((out / "summary.txt").read_text())
    typer.echo(f"run dir: {out}")


if __name__ == "__main__":
    app()
