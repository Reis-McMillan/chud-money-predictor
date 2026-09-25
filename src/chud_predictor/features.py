"""The joined 1-minute frame: BRTI bars + the active Kalshi KXBTC15M contract's candle.

One row per BRTI bar on the complete 1-minute grid, labelled by the bar START `ts`.

Alignment. Contract candles are labelled by the END of their minute, BRTI bars by the START. Every
candle is re-labelled to `ts - 1m`, after which both sources obey one rule: *the row labelled L is
fully observable at wall clock L + 1m*. A window opening at T0 owns rows T0 .. T0+14m (k = 0..14);
its last row (k = 14) holds the settlement candle and the BRTI bar whose `mean` settles the market.

Strike. Kalshi's `floor_strike` equals the mean of the once-per-second index over [T0-60s, T0),
i.e. `brti_mean` of the row just before the window. Null strikes are imputed from that row and
flagged; the build meta reports how closely the identity holds (an alignment audit).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from .resample import bars_path, load_bars, raw_files

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2
WINDOW_MINUTES = 15
SETTLE_K = WINDOW_MINUTES - 1
MAX_QUOTE_SPREAD = 0.10
RW_Z_CLIP = 8.0
SETTLED_H = 1e-4          # "no time left": the fair value of a settled contract is 0 or 1
RV_FAST, RV_SLOW = 15, 240  # windows of the realized-vol regime ratio

_erf = np.vectorize(math.erf, otypes=[np.float64])


def norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=np.float64) / math.sqrt(2.0)))


def h_eff(k):  # noqa: ANN001, ANN201
    """Effective horizon (bars) of the settlement mean, seen from the CLOSE of row k (wall clock
    T0 + k + 1): the settlement minute starts 13 - k bars later and averaging over it adds 1/3, so the
    variance of the settlement is sigma^2 * (13 - k + 1/3). At k = 14 the contract has settled.
    Seen from an origin at minute m (= k + 1) this is the familiar (14 - m) + 1/3."""
    if isinstance(k, pl.Expr):
        return pl.when(k >= SETTLE_K).then(SETTLED_H).otherwise((SETTLE_K - 1 - k) + 1.0 / 3.0)
    k = np.asarray(k, dtype=np.float64)
    return np.where(k >= SETTLE_K, SETTLED_H, (SETTLE_K - 1 - k) + 1.0 / 3.0)


def frame_path(processed_dir: Path, freq: str = "1m") -> Path:
    return processed_dir / f"frame_{freq}.parquet"


def frame_meta_path(processed_dir: Path, freq: str = "1m") -> Path:
    return processed_dir / f"frame_{freq}.meta.json"


def load_contract_candles(contracts_raw_dir: Path, start: date | None = None, end: date | None = None) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for p in raw_files(contracts_raw_dir, start, end)]).sort("ts")


def _first_over_window(col: str) -> pl.Expr:
    return pl.col(col).drop_nulls().first().over("t0")


def _rv_sigma(window: int) -> pl.Expr:
    """Per-minute sigma from the realized variance of the last `window` bars (nulls, i.e. gaps, are skipped)."""
    return pl.col("brti_rv_1s").rolling_mean(window_size=window, min_samples=max(window // 2, 1)).sqrt()


def _rw_z(sigma: str) -> pl.Expr:
    # the settled row is judged on what settles the market: the bar mean, not its close
    num = pl.when(pl.col("is_settlement")).then((pl.col("brti_mean") / pl.col("strike")).log()).otherwise(pl.col("moneyness"))
    return (num / (pl.col(sigma) * h_eff(pl.col("k").cast(pl.Float64)).sqrt())).clip(-RW_Z_CLIP, RW_Z_CLIP)


def _rw_p(frame: pl.DataFrame, z: str, name: str) -> pl.DataFrame:
    rw_z = frame[z].to_numpy()
    return frame.with_columns(pl.Series(name, np.where(np.isfinite(rw_z), norm_cdf(np.nan_to_num(rw_z)), np.nan)).fill_nan(None))


def build_frame_from(bars: pl.DataFrame, candles: pl.DataFrame | None, vol_lookback: int = 240, rv_lookback: int = 30) -> pl.DataFrame:
    """Pure join + feature derivation (no I/O)."""
    frame = bars.rename({c: f"brti_{c}" for c in ("open", "high", "low", "close", "mean", "std", "rv_1s", "ret_l10", "ret_l30", "is_gap", "is_full")})
    if candles is not None and candles.height:
        c = candles.with_columns((pl.col("ts") - pl.duration(minutes=1)).alias("bar_ts")).drop("ts")
        per_bar = c.group_by("bar_ts").agg(pl.col("ticker").n_unique().alias("n"))
        if per_bar["n"].max() > 1:
            raise ValueError("more than one contract per minute: the one-market-per-window assumption is broken")
        c = c.unique(subset=["bar_ts"], keep="last").rename({f"price_{s}": f"trade_{s}" for s in ("open", "high", "low", "close", "mean")})
        frame = frame.join(c, left_on="ts", right_on="bar_ts", how="left")
    else:
        from .qdb import CONTRACT_NUMERIC

        empty = {c.replace("price_", "trade_"): pl.lit(None, dtype=pl.Float64) for c in CONTRACT_NUMERIC}
        frame = frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias("ticker"), **empty)

    frame = frame.sort("ts").with_columns(
        pl.int_range(0, pl.len()).alias("idx"),
        pl.col("ts").dt.truncate("15m").alias("t0"),
        (pl.col("ts").dt.minute() % WINDOW_MINUTES).cast(pl.Int8).alias("k"),
        pl.col("ts").dt.date().alias("date"),
        pl.col("ts").dt.hour().cast(pl.Int8).alias("hour"),
        (pl.col("ts").dt.hour().cast(pl.Int16) * 60 + pl.col("ts").dt.minute().cast(pl.Int16)).alias("minute_of_day"),
        pl.col("ticker").is_not_null().alias("has_candle"),
    )
    # strike: Kalshi's where present, else the previous bar's mean (the verified identity); one value per window
    frame = frame.with_columns(
        pl.when(pl.col("k") == 0).then(pl.col("brti_mean").shift(1)).otherwise(None).alias("_prev_settle"),
    ).with_columns(
        _first_over_window("floor_strike").alias("_kalshi_strike"),
        _first_over_window("_prev_settle").alias("strike_from_brti"),
        _first_over_window("ticker").alias("win_ticker"),
    ).with_columns(
        pl.coalesce("_kalshi_strike", "strike_from_brti").alias("strike"),
        (pl.col("_kalshi_strike").is_null() & pl.col("strike_from_brti").is_not_null()).alias("strike_imputed"),
    )
    frame = frame.with_columns(
        ((pl.col("yes_bid_close") + pl.col("yes_ask_close")) / 2).alias("mid_close"),
        ((pl.col("yes_bid_open") + pl.col("yes_ask_open")) / 2).alias("mid_open"),
        ((pl.col("yes_bid_high") + pl.col("yes_ask_high")) / 2).alias("mid_high"),
        ((pl.col("yes_bid_low") + pl.col("yes_ask_low")) / 2).alias("mid_low"),
        (pl.col("yes_ask_close") - pl.col("yes_bid_close")).alias("spread"),
        pl.col("brti_close").forward_fill().log().diff().alias("ret_1m"),
        (pl.col("brti_close") / pl.col("strike")).log().alias("moneyness"),
        pl.col("volume").log1p().alias("log_volume"),
        pl.col("open_interest").log1p().alias("log_oi"),
    ).with_columns(
        # within-contract price change; crossing a window boundary compares two different contracts
        pl.when(pl.col("k") > 0).then(pl.col("mid_close") - pl.col("mid_close").shift(1)).otherwise(None).alias("mid_ret_1m"),
        pl.col("ret_1m").rolling_std(window_size=vol_lookback, min_samples=max(10, vol_lookback // 4)).alias("sigma_1m"),
        # the same quantity from one-second returns: precise enough for a window short enough to follow the vol regime
        _rv_sigma(rv_lookback).alias("sigma_rv"),
        (_rv_sigma(RV_FAST) / _rv_sigma(RV_SLOW)).log().alias("vol_ratio"),
        pl.coalesce("trade_close", "mid_close").alias("trade_close_imp"),
        pl.coalesce("trade_mean", "mid_close").alias("trade_mean_imp"),
        (pl.col("has_candle") & pl.col("yes_bid_close").is_not_null() & pl.col("yes_ask_close").is_not_null()).alias("target_valid"),
        (pl.col("k") == SETTLE_K).alias("is_settlement"),
    ).with_columns(
        (pl.col("target_valid") & (pl.col("spread") <= MAX_QUOTE_SPREAD)).alias("quote_ok"),
        _rw_z("sigma_1m").alias("rw_z"),
        _rw_z("sigma_rv").alias("rw_z_rv"),
        (pl.col("brti_ret_l10") / pl.col("sigma_rv")).alias("ret_l10_std"),
        (pl.col("brti_ret_l30") / pl.col("sigma_rv")).alias("ret_l30_std"),
    )
    frame = _rw_p(_rw_p(frame, "rw_z", "rw_p"), "rw_z_rv", "rw_p_rv")
    frame = frame.with_columns((pl.col("rw_p_rv") - pl.col("mid_close")).alias("rw_gap_rv"))
    # outcomes, one per window: the price's own verdict and the index's
    frame = frame.with_columns(
        pl.when(pl.col("is_settlement") & pl.col("target_valid")).then((pl.col("mid_close") > 0.5).cast(pl.Int8)).otherwise(None).alias("_op"),
        pl.when(pl.col("is_settlement")).then((pl.col("brti_mean") > pl.col("strike")).cast(pl.Int8)).otherwise(None).alias("_ob"),
    ).with_columns(
        _first_over_window("_op").cast(pl.Boolean).alias("outcome_price"),
        _first_over_window("_ob").cast(pl.Boolean).alias("outcome_brti"),
    )
    return frame.drop("_prev_settle", "_kalshi_strike", "_op", "_ob")


def describe_frame(frame: pl.DataFrame) -> dict:
    with_candle = frame.filter(pl.col("has_candle"))
    wins = with_candle.group_by("t0").agg(
        pl.len().alias("n"), pl.col("strike_imputed").first(), pl.col("outcome_price").first(), pl.col("outcome_brti").first(),
        pl.col("floor_strike").drop_nulls().first().alias("kalshi"), pl.col("strike_from_brti").first().alias("brti"),
    )
    both = wins.filter(pl.col("outcome_price").is_not_null() & pl.col("outcome_brti").is_not_null())
    ident = wins.filter(pl.col("kalshi").is_not_null() & pl.col("brti").is_not_null()).select((pl.col("kalshi") - pl.col("brti")).abs().alias("e"))
    disagree = float((both["outcome_price"] != both["outcome_brti"]).mean()) if both.height else None
    return {
        "schema_version": SCHEMA_VERSION,
        "n_rows": frame.height,
        "first_ts": frame["ts"].min().isoformat(),
        "last_ts": frame["ts"].max().isoformat(),
        "n_rows_no_sigma_rv": int(frame["sigma_rv"].is_null().sum()),
        "n_rows_with_candle": with_candle.height,
        "contract_first_ts": with_candle["ts"].min().isoformat() if with_candle.height else None,
        "contract_last_ts": with_candle["ts"].max().isoformat() if with_candle.height else None,
        "n_windows": wins.height,
        "n_complete_windows": int((wins["n"] == WINDOW_MINUTES).sum()) if wins.height else 0,
        "n_windows_strike_imputed": int(wins["strike_imputed"].sum()) if wins.height else 0,
        "n_rows_quote_degenerate": int((with_candle["target_valid"] & ~with_candle["quote_ok"]).sum()) if with_candle.height else 0,
        "strike_identity_abs_err_p50": float(ident["e"].median()) if ident.height else None,
        "strike_identity_abs_err_p99": float(ident["e"].quantile(0.99)) if ident.height else None,
        "outcome_disagreement_rate": disagree,
        "built_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
    }


def _source_mtime(processed_dir: Path, contracts_raw_dir: Path, freq: str) -> float:
    m = bars_path(processed_dir, freq).stat().st_mtime
    if contracts_raw_dir.exists():
        m = max([m, *(p.stat().st_mtime for p in contracts_raw_dir.glob("date=*.parquet"))])
    return m


def build_frame(processed_dir: Path, contracts_raw_dir: Path, *, freq: str = "1m", vol_lookback: int = 240, rv_lookback: int = 30,
                force: bool = False) -> tuple[Path, dict]:
    out, mp = frame_path(processed_dir, freq), frame_meta_path(processed_dir, freq)
    bars = load_bars(processed_dir, freq)
    src_mtime = _source_mtime(processed_dir, contracts_raw_dir, freq)
    if out.exists() and mp.exists() and not force:
        meta = json.loads(mp.read_text())
        if meta.get("schema_version") == SCHEMA_VERSION and meta.get("source_mtime", 0.0) >= src_mtime and meta.get("vol_lookback") == vol_lookback and meta.get("rv_lookback") == rv_lookback:
            log.info("frame cache up to date: %s", out)
            return out, meta
    has_contracts = contracts_raw_dir.exists() and any(contracts_raw_dir.glob("date=*.parquet"))
    if not has_contracts:
        log.warning("no contract candles under %s; the frame will carry BRTI columns only", contracts_raw_dir)
    frame = build_frame_from(bars, load_contract_candles(contracts_raw_dir) if has_contracts else None, vol_lookback, rv_lookback)
    tmp = out.with_suffix(".parquet.tmp")
    frame.write_parquet(tmp, compression="zstd", compression_level=3, statistics=True)
    tmp.replace(out)
    meta = describe_frame(frame) | {"source_mtime": src_mtime, "vol_lookback": vol_lookback, "rv_lookback": rv_lookback, "freq": freq}
    mp.write_text(json.dumps(meta, indent=1))
    rate = meta["outcome_disagreement_rate"]
    if rate is not None and rate > 0.01:
        log.warning("price and index settle differently in %.1f%% of windows: check the candle/bar alignment", 100 * rate)
    return out, meta


def load_frame(processed_dir: Path, freq: str = "1m") -> pl.DataFrame:
    out, mp = frame_path(processed_dir, freq), frame_meta_path(processed_dir, freq)
    if not out.exists():
        raise FileNotFoundError(f"{out} missing; run `chudp frame`")
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"{out} has frame schema v{meta.get('schema_version')}, need v{SCHEMA_VERSION}; re-run `chudp frame --force`")
    return pl.read_parquet(out)
