"""Fixed-price Kalshi trading simulation on top of a forecast frame.

No historical Kalshi quotes exist in the data, so a contract is assumed to be available at a fixed
price. This is optimistic by construction (real quotes already embed the drift from the strike),
which is why every summary is also computed for the RW+vol baseline's probabilities on the same
policy: the model-minus-baseline number is the one to read.

Pricing (`complement`, arbitrage-free): YES costs c, NO costs 1-c. Payoff is $1 on a win, so
pnl_gross = +(1-price) on a win and -price on a loss. Kalshi fee = fee_rate * price * (1-price).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import polars as pl

SIDES = ("YES", "NO", "NONE")


@dataclass(frozen=True)
class TradeConfig:
    c: float = 0.50
    tau: float = 0.55
    policy: str = "threshold"          # "threshold" | "ev"
    ev_margin: float = 0.02
    no_price_mode: str = "complement"  # "complement" | "same"
    fees: bool = True
    fee_rate: float = 0.07
    fee_round_cents: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.c < 1:
            raise ValueError("price c must be in (0, 1)")
        if self.policy not in ("threshold", "ev"):
            raise ValueError("policy must be 'threshold' or 'ev'")
        if self.no_price_mode not in ("complement", "same"):
            raise ValueError("no_price_mode must be 'complement' or 'same'")

    @property
    def price_yes(self) -> float:
        return self.c

    @property
    def price_no(self) -> float:
        return 1.0 - self.c if self.no_price_mode == "complement" else self.c


def fee_for(price: np.ndarray, cfg: TradeConfig) -> np.ndarray:
    if not cfg.fees:
        return np.zeros_like(price, dtype=np.float64)
    f = cfg.fee_rate * price * (1.0 - price)
    if cfg.fee_round_cents:
        f = np.ceil(f * 100.0 - 1e-12) / 100.0
    return f


def decide(p_up: np.ndarray, cfg: TradeConfig) -> np.ndarray:
    """Side per origin: 'YES' | 'NO' | 'NONE'."""
    p = np.asarray(p_up, dtype=np.float64)
    if cfg.policy == "threshold":
        yes = p >= cfg.tau
        no = p <= 1.0 - cfg.tau
    else:
        edge_yes = p - cfg.price_yes
        edge_no = (1.0 - p) - cfg.price_no
        yes = (edge_yes >= cfg.ev_margin) & (edge_yes >= edge_no)
        no = (edge_no >= cfg.ev_margin) & ~yes
    side = np.where(yes, "YES", np.where(no, "NO", "NONE"))
    return side


def simulate_trades(
    forecasts: pl.DataFrame,
    cfg: TradeConfig,
    p_col: str = "p_up",
    source: str = "model",
    quotes: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per origin (including abstentions, side == 'NONE')."""
    p = forecasts[p_col].to_numpy().astype(np.float64)
    label = forecasts["label_up"].to_numpy().astype(bool)
    side = decide(p, cfg)
    is_yes, is_no = side == "YES", side == "NO"
    price = np.where(is_yes, cfg.price_yes, np.where(is_no, cfg.price_no, np.nan))
    if quotes is not None:
        q = forecasts.select("t0", "m").join(quotes, on=["t0", "m"], how="left")
        ask, bid = q["yes_ask"].to_numpy(), q["yes_bid"].to_numpy()
        price = np.where(is_yes, ask, np.where(is_no, 1.0 - bid, np.nan))
    p_side = np.where(is_yes, p, np.where(is_no, 1.0 - p, np.nan))
    win = (is_yes & label) | (is_no & ~label)
    traded = is_yes | is_no
    pnl_gross = np.where(traded, np.where(win, 1.0 - price, -price), 0.0)
    fee = np.where(traded, fee_for(np.nan_to_num(price), cfg), 0.0)
    out = forecasts.select("t0", "date", "hour", "m", "vol_bucket").with_columns(
        pl.lit(source).alias("source"),
        pl.lit(cfg.c).alias("c"),
        pl.lit(cfg.tau).alias("tau"),
        pl.lit(cfg.policy).alias("policy"),
        pl.lit(cfg.no_price_mode).alias("no_price_mode"),
        pl.lit(cfg.fees).alias("fees"),
        pl.Series("side", side.astype(str)),
        pl.Series("p_up", p),
        pl.Series("p_side", p_side),
        pl.Series("price_paid", price),
        pl.Series("traded", traded),
        pl.Series("_win", win),
        pl.Series("pnl_gross", pnl_gross),
        pl.Series("fee", fee),
        pl.Series("pnl_net", pnl_gross - fee),
        pl.Series("edge_ex_ante", p_side - price),
    )
    return out.with_columns(pl.when(pl.col("traded")).then(pl.col("_win")).otherwise(None).alias("win")).drop("_win")


def sweep(
    forecasts: pl.DataFrame,
    prices: tuple[float, ...],
    taus: tuple[float, ...],
    base: TradeConfig,
    policies: tuple[str, ...] = ("threshold",),
    sources: dict[str, str] | None = None,
    quotes: pl.DataFrame | None = None,
    keep_abstain: bool = False,
) -> pl.DataFrame:
    sources = sources or {"model": "p_up", "rwvol": "p_up_rwvol"}
    frames = []
    for src, col in sources.items():
        for c in prices:
            for tau in taus:
                for pol in policies:
                    cfg = replace(base, c=c, tau=tau, policy=pol)
                    t = simulate_trades(forecasts, cfg, p_col=col, source=src, quotes=quotes)
                    frames.append(t if keep_abstain else t.filter(pl.col("traded")))
    return pl.concat(frames) if frames else pl.DataFrame()


def max_drawdown(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float(np.max(peak - cum))


ORIGIN_KEYS = ("m", "hour", "vol_bucket", "date")


def trade_summary(trades: pl.DataFrame, by: list[str], forecasts: pl.DataFrame | None = None) -> pl.DataFrame:
    """Aggregate taken trades. `forecasts` (optional) supplies n_origins / n_windows per group so that
    trade_rate and pnl_per_window are relative to all opportunities, not only the trades taken."""
    taken = trades.filter(pl.col("traded")) if "traded" in trades.columns else trades
    rows = []
    keys = by
    groups = taken.group_by(keys, maintain_order=True) if keys else [((), taken)]
    for k, g in groups:
        g = g.sort("t0")
        n = g.height
        pnl = g["pnl_net"].to_numpy()
        row = dict(zip(keys, k, strict=True))
        row.update({
            "n_trades": n,
            "n_yes": int((g["side"] == "YES").sum()),
            "n_no": int((g["side"] == "NO").sum()),
            "win_rate": float(g["win"].cast(pl.Float64).mean()) if n else float("nan"),
            "mean_price_paid": float(g["price_paid"].mean()) if n else float("nan"),
            "mean_edge_ex_ante": float(g["edge_ex_ante"].mean()) if n else float("nan"),
            "mean_pnl_gross": float(g["pnl_gross"].mean()) if n else float("nan"),
            "mean_pnl_net": float(pnl.mean()) if n else float("nan"),
            "total_pnl_net": float(pnl.sum()),
            "pnl_sd": float(pnl.std(ddof=1)) if n > 1 else float("nan"),
            "max_drawdown": max_drawdown(pnl),
            "n_windows_traded": g["t0"].n_unique(),
        })
        row["realized_edge"] = row["win_rate"] - row["mean_price_paid"] if n else float("nan")
        row["t_stat"] = (row["mean_pnl_net"] / (row["pnl_sd"] / math.sqrt(n))) if n > 1 and row["pnl_sd"] > 0 else float("nan")
        rows.append(row)
    out = pl.DataFrame(rows)
    if out.is_empty():
        return out
    if forecasts is not None:
        join_keys = [k for k in keys if k in ORIGIN_KEYS and k in forecasts.columns]
        if join_keys:
            counts = forecasts.group_by(join_keys).agg(pl.len().alias("n_origins"), pl.col("t0").n_unique().alias("n_windows"))
            out = out.join(counts, on=join_keys, how="left")
        else:
            out = out.with_columns(pl.lit(forecasts.height).alias("n_origins"), pl.lit(forecasts["t0"].n_unique()).alias("n_windows"))
        out = out.with_columns(
            (pl.col("n_trades") / pl.col("n_origins")).alias("trade_rate"),
            (pl.col("total_pnl_net") / pl.col("n_windows")).alias("pnl_per_window"),
        )
    return out.sort(keys) if keys else out
