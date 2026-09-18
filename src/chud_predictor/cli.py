"""`chudp` command line."""

from __future__ import annotations

import logging
import tomllib
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from .settings import Settings, load_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="TimesFM 3.0 x BRTI x Kalshi 15m backtests.")
qdb_app = typer.Typer(help="QuestDB inspection.")
app.add_typer(qdb_app, name="qdb")

_state: dict[str, Any] = {"settings": None, "config": {}}


def _settings() -> Settings:
    return _state["settings"]


def _cfg(section: str, key: str, default: Any) -> Any:
    """Config-file value for section.key, else default."""
    return _state["config"].get(section, {}).get(key, default)


def _pick(flag: Any, section: str, key: str, default: Any) -> Any:
    """Explicit CLI flag (not None) > config file > default."""
    return flag if flag is not None else _cfg(section, key, default)


@app.callback()
def main(
    config: Annotated[Path | None, typer.Option("--config", help="TOML config with [spec]/[run]/[trading] sections")] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    env_file: Annotated[Path, typer.Option("--env-file")] = Path(".env"),
    verbose: Annotated[int, typer.Option("-v", count=True)] = 0,
) -> None:
    level = logging.DEBUG if verbose > 1 else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _state["settings"] = load_settings(env_file, data_dir)
    _state["config"] = tomllib.loads(config.read_text()) if config else {}
    if config:
        logging.getLogger("chudp").info("config %s: %s", config, _state["config"])


# ---------------------------------------------------------------------------------------------

@qdb_app.command("info")
def qdb_info(table: str = "index_values_hist", index_id: str = "BRTI") -> None:
    """Row counts, time bounds and per-day coverage of the source table."""
    from .qdb import QuestDB

    q = QuestDB(_settings())
    typer.echo(f"questdb {q.ping()} at {_settings().qdb_url}; tables: {', '.join(q.tables())}")
    b = q.bounds(table, index_id)
    typer.echo(f"{table} [{index_id}]: {b.rows:,} rows, {b.min_ts} .. {b.max_ts}")
    if b.max_ts:
        counts = q.day_counts(table, index_id, b.max_ts)
        short = {d: n for d, n in counts.items() if n < 86_400}
        typer.echo(f"{len(counts)} days; {len(short)} days with < 86,400 rows: "
                   + ", ".join(f"{d}={n}" for d, n in list(short.items())[:12]) + (" ..." if len(short) > 12 else ""))


@app.command()
def download(
    start: Annotated[str | None, typer.Option(help="first UTC day, YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option(help="last UTC day, YYYY-MM-DD")] = None,
    jobs: int = 4,
    force: bool = False,
    verify_only: bool = False,
    table: str = "index_values_hist",
    index_id: str = "BRTI",
) -> None:
    """Pull raw 1-second ticks into data/raw/brti/date=YYYY-MM-DD.parquet (incremental, idempotent)."""
    from .download import download as _download
    from .qdb import QuestDB

    q = QuestDB(_settings())
    rep = _download(q, _settings().raw_dir, table=table, index_id=index_id,
                    start=date.fromisoformat(start) if start else None, end=date.fromisoformat(end) if end else None,
                    jobs=jobs, force=force, verify_only=verify_only)
    if verify_only:
        typer.echo(f"would fetch {len(rep.planned)} days: {', '.join(d.isoformat() for d in rep.planned[:10])}{' ...' if len(rep.planned) > 10 else ''}")
    typer.echo(rep.summary())
    if rep.failed:
        raise typer.Exit(code=1)


@app.command()
def resample(freq: str = "1m", force: bool = False) -> None:
    """Build regular bars (open/high/low/close/mean/std/n_ticks) from the raw ticks."""
    from .resample import build

    out, meta = build(_settings().raw_dir, _settings().processed_dir, freq, force)
    typer.echo(f"{out}: {meta['n_bars']:,} bars, {meta['n_gaps']} gaps, {meta['n_not_full']} not full, "
               f"largest gap {meta['largest_gap_bars']} bars, {meta['first_ts']} .. {meta['last_ts']}")


@app.command()
def forecast(
    context: int = 4096,
    horizon: int = 64,
    at: Annotated[str | None, typer.Option(help="forecast origin, ISO UTC; default = last bar")] = None,
    target: str = "mean",
    transform: str = "log",
    device: Annotated[str | None, typer.Option()] = None,
    model_id: str = "google/timesfm-3.0-pytorch",
    batch_size: int = 8,
    plot: bool = False,
) -> None:
    """One generic forecast from the latest (or given) origin: 64 steps of median + deciles."""
    from .forecast import run_forecast

    out = run_forecast(_settings(), context=context, horizon=horizon, at=datetime.fromisoformat(at) if at else None,
                       target_col=target, transform=transform, model_id=model_id, device=device, batch_size=batch_size, plot=plot)
    typer.echo(str(out))


def _floats(v: list[float] | None, section: str, key: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if v:
        return tuple(v)
    return tuple(_cfg(section, key, list(default)))


@app.command("backtest-kalshi")
def backtest_kalshi(
    config: Annotated[Path | None, typer.Option("--config", help="TOML with [spec]/[run]/[trading]; flags override")] = None,
    context: Annotated[int | None, typer.Option()] = None,
    target: Annotated[str | None, typer.Option(help="mean | close")] = None,
    strike_mode: Annotated[str | None, typer.Option(help="open_tick | open_avg60")] = None,
    minutes: Annotated[str | None, typer.Option(help="'0-14' | '0,5,10,14'")] = None,
    start: Annotated[str | None, typer.Option()] = None,
    end: Annotated[str | None, typer.Option()] = None,
    stride_windows: Annotated[int | None, typer.Option()] = None,
    max_windows: Annotated[int | None, typer.Option()] = None,
    window_chunk: Annotated[int | None, typer.Option()] = None,
    batch_size: Annotated[int | None, typer.Option()] = None,
    device: Annotated[str | None, typer.Option(help="auto | cpu | mps | cuda")] = None,
    model_id: Annotated[str | None, typer.Option()] = None,
    symmetric: Annotated[bool | None, typer.Option("--symmetric/--no-symmetric")] = None,
    pup_method: Annotated[str | None, typer.Option(help="pwl_exp | normal_fit")] = None,
    eps_bp: Annotated[float | None, typer.Option(help="no-trade band for the directional metric, bp")] = None,
    price: Annotated[list[float] | None, typer.Option(help="contract price c; repeat to sweep")] = None,
    tau: Annotated[list[float] | None, typer.Option(help="P(up) threshold; repeat to sweep")] = None,
    policy: Annotated[str | None, typer.Option(help="threshold | ev")] = None,
    ev_margin: Annotated[float | None, typer.Option()] = None,
    no_price_mode: Annotated[str | None, typer.Option(help="complement | same")] = None,
    fees: Annotated[bool | None, typer.Option("--fees/--no-fees")] = None,
    fee_round: Annotated[bool | None, typer.Option("--fee-round/--no-fee-round")] = None,
    vol_lookback: Annotated[int | None, typer.Option()] = None,
    min_settle_ticks: Annotated[int | None, typer.Option()] = None,
    max_ctx_gap_frac: Annotated[float | None, typer.Option()] = None,
    bootstrap_days: Annotated[int | None, typer.Option()] = None,
    plot: Annotated[bool | None, typer.Option("--plot/--no-plot")] = None,
    trajectories: Annotated[int | None, typer.Option()] = None,
    verify_settlement: Annotated[bool | None, typer.Option("--verify-settlement/--no-verify-settlement")] = None,
    quotes: Annotated[Path | None, typer.Option(help="optional Kalshi quotes (t0, m, yes_bid, yes_ask)")] = None,
    allow_target_mismatch: bool = False,
    run_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Forecast every minute of every 15-minute window and score it as a Kalshi trader."""
    from .backtest import KalshiBacktestConfig, run_kalshi_backtest
    from .trading import TradeConfig
    from .windows import KalshiSpec

    if config:
        _state["config"] = tomllib.loads(config.read_text())
        logging.getLogger("chudp").info("config %s: %s", config, _state["config"])
    spec = KalshiSpec(
        context=_pick(context, "spec", "context", 4096),
        target_col=_pick(target, "spec", "target", "mean"),
        strike_mode=_pick(strike_mode, "spec", "strike_mode", "open_tick"),
        minutes=_pick(minutes, "spec", "minutes", "0-14"),
        min_settle_ticks=_pick(min_settle_ticks, "spec", "min_settle_ticks", 55),
        max_ctx_gap_frac=_pick(max_ctx_gap_frac, "spec", "max_ctx_gap_frac", 0.01),
        vol_lookback=_pick(vol_lookback, "spec", "vol_lookback", 240),
    )
    trade = TradeConfig(
        tau=(tau[0] if tau else _cfg("trading", "tau", [0.55])[0]),
        policy=_pick(policy, "trading", "policy", "threshold"),
        ev_margin=_pick(ev_margin, "trading", "ev_margin", 0.02),
        no_price_mode=_pick(no_price_mode, "trading", "no_price_mode", "complement"),
        fees=_pick(fees, "trading", "fees", True),
        fee_round_cents=_pick(fee_round, "trading", "fee_round", False),
    )
    s, e = _pick(start, "run", "start", None), _pick(end, "run", "end", None)
    cfg = KalshiBacktestConfig(
        spec=spec, trade=trade,
        start=date.fromisoformat(s) if s else None, end=date.fromisoformat(e) if e else None,
        stride_windows=_pick(stride_windows, "run", "stride_windows", 1),
        max_windows=_pick(max_windows, "run", "max_windows", None),
        window_chunk=_pick(window_chunk, "run", "window_chunk", 256),
        batch_size=_pick(batch_size, "run", "batch_size", 64),
        device=_pick(device, "run", "device", None),
        model_id=_pick(model_id, "run", "model_id", "google/timesfm-3.0-pytorch"),
        symmetric=_pick(symmetric, "run", "symmetric", False),
        pup_method=_pick(pup_method, "run", "pup_method", "pwl_exp"),
        eps_bp=_pick(eps_bp, "run", "eps_bp", 0.0),
        prices=_floats(price, "trading", "price", (0.40, 0.50, 0.60)),
        taus=_floats(tau, "trading", "tau", (0.52, 0.55, 0.60, 0.65)),
        policies=(trade.policy,),
        bootstrap_days=_pick(bootstrap_days, "run", "bootstrap_days", 1000),
        plot=_pick(plot, "run", "plot", False),
        trajectories=_pick(trajectories, "run", "trajectories", 20),
        verify_settlement=_pick(verify_settlement, "run", "verify_settlement", False),
        quotes_path=quotes,
        allow_target_mismatch=allow_target_mismatch,
        run_id=run_id,
    )
    logging.getLogger("chudp").info("resolved config: %s", cfg)
    out = run_kalshi_backtest(_settings(), cfg)
    typer.echo((out / "summary.txt").read_text())
    typer.echo(f"run dir: {out}")


@app.command("rescore-kalshi")
def rescore_kalshi(
    run_id: str,
    strike_mode: Annotated[str | None, typer.Option()] = None,
    pup_method: Annotated[str | None, typer.Option()] = None,
    eps_bp: Annotated[float | None, typer.Option()] = None,
    price: Annotated[list[float] | None, typer.Option()] = None,
    tau: Annotated[list[float] | None, typer.Option()] = None,
    policy: Annotated[str | None, typer.Option()] = None,
    fees: Annotated[bool | None, typer.Option("--fees/--no-fees")] = None,
    no_price_mode: Annotated[str | None, typer.Option()] = None,
    quotes: Annotated[Path | None, typer.Option()] = None,
    plot: Annotated[bool | None, typer.Option("--plot/--no-plot")] = None,
    label: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Re-score an existing run's forecasts under a different strike / P(up) / trading setup (no model)."""
    from .backtest import rescore

    overrides = {
        "spec": {"strike_mode": strike_mode},
        "trade": {"policy": policy, "fees": fees, "no_price_mode": no_price_mode, "tau": tau[0] if tau else None},
        "pup_method": pup_method, "eps_bp": eps_bp,
        "prices": tuple(price) if price else None, "taus": tuple(tau) if tau else None,
        "policies": (policy,) if policy else None, "quotes_path": quotes, "plot": plot,
    }
    out = rescore(_settings(), run_id, overrides, label)
    typer.echo((out / "summary.txt").read_text())
    typer.echo(f"rescore dir: {out}")


if __name__ == "__main__":
    app()
