"""`chudp` command line."""

from __future__ import annotations

import logging
import tomllib
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import typer

from .settings import Settings, load_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="TimesFM 3.0 forecasts of the Kalshi KXBTC15M contract price.")
api_app = typer.Typer(help="chud-money API inspection.")
auth_app = typer.Typer(help="Verys login for the chud-money API.")
app.add_typer(api_app, name="api")
app.add_typer(auth_app, name="auth")

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


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


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
# auth

def _fail(msg: str, code: int = 2) -> None:
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=code)


def _provider():
    from .auth import token_provider

    return token_provider(_settings())


def _report_session(session) -> None:  # noqa: ANN001
    typer.echo(f"saved {_settings().auth_file} (mode 600): {session.email}, roles {', '.join(session.roles) or 'none'}")


@auth_app.command("login")
def auth_login(
    email: Annotated[str | None, typer.Option(help="Verys account email")] = None,
    code: Annotated[str | None, typer.Option(help="the emailed 6-digit code (skips the prompt)")] = None,
    force: Annotated[bool, typer.Option("--force", help="ignore the saved session cookies and ask for a new code")] = False,
) -> None:
    """Sign in to Verys once (email + emailed code) and save a session that `chudp` renews unattended."""
    from .auth import AuthError, Session, VerysClient, mint_session

    s = _settings()
    kw = dict(audience=s.chud_money_client_id, redirect_uri=s.verys_redirect_uri, verys_url=s.verys_url, client_id=s.verys_client_id)
    try:
        if not force and s.auth_file.exists():
            # A saved 60-day cookie can mint a fresh refresh token with no email round trip.
            old = Session.load(s.auth_file)
            try:
                with VerysClient(s.verys_url, s.verys_client_id, cookies=old.cookies) as c:
                    session = mint_session(c, email=email or old.email, cookies=old.cookies, logged_in_at=old.logged_in_at, **kw)
                session.save(s.auth_file)
                typer.echo(f"reused the saved Verys session for {session.email}; no code needed")
                return _report_session(session)
            except AuthError as e:
                log.info("saved session unusable (%s); asking for a code", e)
                email = email or old.email
        email = email or typer.prompt("Verys email")
        with VerysClient(s.verys_url, s.verys_client_id) as c:
            c.send_code(email)
            typer.echo(f"a 6-digit code was emailed to {email} (valid 5 minutes)")
            code = code or typer.prompt("code")
            c.verify_code(email, code.strip())
            session = mint_session(c, email=email, cookies=c.cookies, logged_in_at=_utcnow(), **kw)
        session.save(s.auth_file)
        _report_session(session)
    except AuthError as e:
        _fail(str(e))


@auth_app.command("status")
def auth_status() -> None:
    """The saved session, and whether it still yields a token the API accepts."""
    from .auth import AuthError, Session, claims

    s = _settings()
    try:
        session = Session.load(s.auth_file)
    except AuthError as e:
        _fail(str(e))
    mode = oct(s.auth_file.stat().st_mode & 0o777)
    typer.echo(f"session   {s.auth_file} ({mode})")
    typer.echo(f"identity  {session.email}  sub {session.sub}")
    typer.echo(f"verys     {session.verys_url}  client {session.client_id}  audience {session.audience}")
    typer.echo(f"refresh   token age {session.refresh_age} (rotated {session.obtained_at}, login {session.logged_in_at})")
    try:
        tok = _provider().token()
    except AuthError as e:
        _fail(f"exchange  FAILED: {e}", code=1)
    c = claims(tok)
    exp = datetime.fromtimestamp(c["exp"], UTC).replace(tzinfo=None)
    typer.echo(f"exchange  ok: roles {c.get('roles')}, aud {c.get('aud')}, exp {exp} (in {exp - _utcnow()})")


@auth_app.command("token")
def auth_token(decode: Annotated[bool, typer.Option("--decode", help="print the (unverified) claims instead")] = False) -> None:
    """Print a valid chud-money access token on stdout (for `curl -H "Authorization: Bearer $(chudp auth token)"`)."""
    import json

    from .auth import AuthError, claims

    try:
        tok = _provider().token()
    except AuthError as e:
        _fail(str(e))
    typer.echo(json.dumps(claims(tok), indent=1) if decode else tok)


@auth_app.command("refresh")
def auth_refresh() -> None:
    """Force one renewal cycle: refresh the Verys token (or re-authorize from the saved cookie) and exchange."""
    from .auth import AuthError, Session

    s = _settings()
    try:
        before = Session.load(s.auth_file).refresh_token
        p = _provider()
        p.invalidate()
        p.token()
        after = Session.load(s.auth_file).refresh_token
    except AuthError as e:
        _fail(str(e))
    typer.echo("renewed: exchanged a new token; refresh token " + ("rotated" if before != after else "unchanged"))


@auth_app.command("logout")
def auth_logout() -> None:
    """Revoke our refresh token and delete the session file. The browser SPA's own session is untouched."""
    from .auth import AuthError, Session, VerysClient

    s = _settings()
    try:
        session = Session.load(s.auth_file)
    except AuthError as e:
        _fail(str(e))
    with VerysClient(session.verys_url, session.client_id) as c:
        c.revoke(session.refresh_token)
    s.auth_file.unlink(missing_ok=True)
    s.auth_file.with_suffix(".lock").unlink(missing_ok=True)
    typer.echo(f"revoked the refresh token and deleted {s.auth_file}")


# ---------------------------------------------------------------------------------------------
# data

def _api():
    from .api import ChudApi

    return ChudApi(_settings(), tokens=_provider())


@api_app.command("info")
def api_info() -> None:
    """Row counts and time bounds of both source tables from the API, and the local raw coverage."""
    from .api import SOURCES
    from .download import MANIFEST_NAME, Manifest, candidate_days

    api = _api()          # /{tag} and /healthz are public: no token is fetched here
    detail = api.market()
    m, qdb = detail["market"], detail.get("questdb") or {}
    typer.echo(f"{_settings().api_base} {api.health()}; market {m['tag']} (index {m['index_id']}, series {m['series_ticker']}); "
               f"summary refreshed {qdb.get('refreshed_at')}" + (f" [error: {qdb['error']}]" if qdb.get("error") else ""))
    for spec in SOURCES.values():
        b = api.bounds(spec)
        typer.echo(f"[{spec.name}] {spec.table} ({spec.key_column}={spec.key}): {b.rows:,} rows, {b.min_ts} .. {b.max_ts}")
        man = Manifest.load(_settings().raw_dir_for(spec.name) / MANIFEST_NAME)
        done = man.complete_days()
        local = sum(r.rows for r in man.days.values())
        line = f"    local: {len(man.days)} days ({len(done)} complete), {local:,} rows"
        if done:
            line += f", {done[0]} .. {done[-1]}"
        if b.min_ts and b.max_ts:
            n_all = len(candidate_days(b.min_ts, b.max_ts + timedelta(microseconds=1)))
            line += f"; {max(0, n_all - len(man.days))} of {n_all} days not downloaded"
        typer.echo(line)


@app.command()
def download(
    source: Annotated[str, typer.Option(help="both | brti | contracts")] = "both",
    start: Annotated[str | None, typer.Option(help="first UTC day, YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option(help="last UTC day, YYYY-MM-DD")] = None,
    jobs: Annotated[int, typer.Option(help="concurrent day streams; the API allows 2 in total")] = 1,
    force: bool = False,
    verify_only: bool = False,
) -> None:
    """Pull raw rows into data/raw/<source>/date=YYYY-MM-DD.parquet (incremental, idempotent)."""
    from .api import SOURCES
    from .download import download as _download

    names = list(SOURCES) if source == "both" else [source]
    if any(n not in SOURCES for n in names):
        raise typer.BadParameter(f"source must be one of both, {', '.join(SOURCES)}")
    api = _api()
    failed = False
    for name in names:
        rep = _download(api, _settings().raw_dir_for(name), SOURCES[name], start=_date(start), end=_date(end), jobs=jobs, force=force, verify_only=verify_only)
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
def frame(force: bool = False, vol_lookback: int = 240, rv_lookback: int = 30) -> None:
    """Join the BRTI bars with the contract candles into the 1-minute model frame and print its audit."""
    from .features import build_frame

    out, meta = build_frame(_settings().processed_dir, _settings().contracts_raw_dir, vol_lookback=vol_lookback, rv_lookback=rv_lookback, force=force)
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
    covariates: Annotated[str | None, typer.Option(help="full | full_rv | no_fair_path | brti_only | quotes_only | calendar_only | none")] = None,
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
