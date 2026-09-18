"""Static PNG figures for a backtest run (needs the `viz` extra).

Palette and chrome follow the reference data-viz palette: fixed categorical slot order, one hue
per entity, thin marks, recessive grid, one y-axis per panel (never a dual axis).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

log = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {"model": "#2a78d6", "rwvol": "#eb6834", "naive": "#1baf7a", "const": "#eda100", "strike": "#52514e", "settlement": "#e34948"}
MGROUP = {"0-4": "#2a78d6", "5-9": "#eb6834", "10-14": "#1baf7a"}
BAND_OUTER, BAND_INNER = 0.14, 0.28


def _style(ax) -> None:  # noqa: ANN001
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _fig(nrows=1, ncols=1, figsize=(9, 4.5), **kw):  # noqa: ANN001
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE, **kw)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def _legend(ax) -> None:  # noqa: ANN001
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    if leg:
        for t in leg.get_texts():
            t.set_color(INK2)


def _m_group(m: int) -> str:
    return "0-4" if m <= 4 else ("5-9" if m <= 9 else "10-14")


# ---------------------------------------------------------------------------------------------

def plot_window_trajectory(rows: pl.DataFrame, out: Path, tau: float) -> None:
    """One window: forecast of the settlement as strike time approaches (top) and P(up) (bottom)."""
    rows = rows.sort("m")
    m = rows["m"].to_numpy()
    fig, (ax1, ax2) = _fig(2, 1, figsize=(8, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax1.fill_between(m, rows["q10"], rows["q90"], color=SERIES["model"], alpha=BAND_OUTER, linewidth=0, label="q10–q90")
    ax1.fill_between(m, rows["q20"], rows["q80"], color=SERIES["model"], alpha=BAND_INNER, linewidth=0, label="q20–q80")
    ax1.plot(m, rows["median"], color=SERIES["model"], linewidth=2, marker="o", markersize=4, label="median forecast")
    ax1.plot(m, rows["last_close"], color=SERIES["naive"], linewidth=1.2, linestyle=":", label="last close (naive)")
    ax1.axhline(float(rows["strike"][0]), color=SERIES["strike"], linewidth=1.2, linestyle="--", label="strike")
    ax1.axhline(float(rows["settlement"][0]), color=SERIES["settlement"], linewidth=1.6, label="settlement")
    t0 = rows["t0"][0]
    ax1.set_title(f"Window {t0:%Y-%m-%d %H:%M} UTC — settlement {'UP' if rows['label_up'][0] else 'DOWN'} vs strike", fontsize=10, loc="left")
    ax1.set_ylabel("BRTI (USD)")
    _legend(ax1)
    ax2.plot(m, rows["p_up"], color=SERIES["model"], linewidth=2, marker="o", markersize=4, label="P(up) model")
    ax2.plot(m, rows["p_up_rwvol"], color=SERIES["rwvol"], linewidth=1.6, marker="o", markersize=3, label="P(up) RW+vol")
    for lvl in (tau, 1 - tau):
        ax2.axhline(lvl, color=MUTED, linewidth=0.8, linestyle="--")
    ax2.axhline(0.5, color=AXIS, linewidth=0.8)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("P(settlement > strike)")
    ax2.set_xlabel("minutes since window open (m); strike at m = 15")
    ax2.set_xticks(range(0, 15))
    _legend(ax2)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_vs_m(by_m: pl.DataFrame, boot: pl.DataFrame, ts_m: pl.DataFrame, out: Path, c: float, tau: float) -> None:
    by_m = by_m.sort("m")
    m = by_m["m"].to_numpy()
    fig, axes = _fig(2, 2, figsize=(11, 7))
    (a, b), (cx, d) = axes

    def ci(stat):  # noqa: ANN001
        if boot.is_empty():
            return None
        s = boot.filter(pl.col("stat") == stat).sort("m")
        return s["lo"].to_numpy(), s["hi"].to_numpy()

    a.plot(m, by_m["accuracy"], color=SERIES["model"], linewidth=2, marker="o", markersize=4, label="model")
    if (r := ci("accuracy")) is not None:
        a.fill_between(m, r[0], r[1], color=SERIES["model"], alpha=BAND_OUTER, linewidth=0)
    a.plot(m, by_m["accuracy_naive"], color=SERIES["naive"], linewidth=1.6, marker="o", markersize=3, label="naive (last close vs strike)")
    a.axhline(0.5, color=AXIS, linewidth=0.8)
    a.set_title("Directional accuracy vs strike", fontsize=10, loc="left")
    _legend(a)

    b.plot(m, by_m["brier"], color=SERIES["model"], linewidth=2, marker="o", markersize=4, label="model")
    if (r := ci("brier")) is not None:
        b.fill_between(m, r[0], r[1], color=SERIES["model"], alpha=BAND_OUTER, linewidth=0)
    b.plot(m, by_m["brier_rwvol"], color=SERIES["rwvol"], linewidth=1.6, marker="o", markersize=3, label="RW+vol")
    b.axhline(0.25, color=SERIES["const"], linewidth=1.2, linestyle="--", label="constant 0.5")
    b.set_title("Brier score (lower is better)", fontsize=10, loc="left")
    _legend(b)

    h = ts_m.filter((pl.col("c") == c) & (pl.col("tau") == tau)) if not ts_m.is_empty() else ts_m
    for src in ("model", "rwvol"):
        s = h.filter(pl.col("source") == src).sort("m") if not h.is_empty() else h
        if not s.is_empty():
            cx.plot(s["m"], s["mean_pnl_net"], color=SERIES[src], linewidth=2 if src == "model" else 1.6, marker="o", markersize=4, label=src)
            d.plot(s["m"], s["n_trades"], color=SERIES[src], linewidth=2 if src == "model" else 1.6, marker="o", markersize=4, label=src)
    cx.axhline(0, color=AXIS, linewidth=0.8)
    cx.set_title(f"Net P&L per contract traded (c={c:.2f}, τ={tau:.2f})", fontsize=10, loc="left")
    d.set_title("Trades taken", fontsize=10, loc="left")
    _legend(cx)
    _legend(d)
    for ax in axes.ravel():
        ax.set_xlabel("minutes since window open (m)")
        ax.set_xticks(range(0, 15))
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_calibration(calibration: pl.DataFrame, forecasts: pl.DataFrame, out: Path) -> None:
    fig, (a, b) = _fig(1, 2, figsize=(11, 4.5))
    a.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8, linestyle="--")
    for grp, color in MGROUP.items():
        s = calibration.filter((pl.col("source") == "model") & (pl.col("m_group") == grp) & (pl.col("n") > 0)).sort("bin_lo")
        if s.is_empty():
            continue
        sizes = 10 + 40 * np.sqrt(s["n"].to_numpy() / max(s["n"].max(), 1))
        a.plot(s["p_mean"], s["freq_up"], color=color, linewidth=1.6, label=f"m {grp}")
        a.scatter(s["p_mean"], s["freq_up"], s=sizes, color=color, edgecolor=SURFACE, linewidth=1, zorder=3)
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_xlabel("predicted P(up)")
    a.set_ylabel("observed frequency of up")
    a.set_title("Reliability (model)", fontsize=10, loc="left")
    _legend(a)
    for grp, color in MGROUP.items():
        lo, hi = {"0-4": (0, 4), "5-9": (5, 9), "10-14": (10, 14)}[grp]
        p = forecasts.filter((pl.col("m") >= lo) & (pl.col("m") <= hi))["p_up"].to_numpy()
        if p.size:
            b.hist(p, bins=np.linspace(0, 1, 41), histtype="step", linewidth=1.6, color=color, label=f"m {grp}")
    b.set_xlabel("predicted P(up)")
    b.set_ylabel("origins")
    b.set_title("Sharpness: distribution of P(up)", fontsize=10, loc="left")
    _legend(b)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_pnl_curve(trades: pl.DataFrame, out: Path, c: float, tau: float) -> None:
    fig, ax = _fig(figsize=(10, 4.5))
    h = trades.filter((pl.col("c") == c) & (pl.col("tau") == tau) & pl.col("traded")).sort("t0")
    for src in ("model", "rwvol"):
        s = h.filter(pl.col("source") == src)
        if s.is_empty():
            continue
        ax.plot(s["t0"], np.cumsum(s["pnl_net"].to_numpy()), color=SERIES[src], linewidth=2 if src == "model" else 1.6, label=f"{src} (all m)")
    sm = h.filter(pl.col("source") == "model")
    for grp, color in MGROUP.items():
        lo, hi = {"0-4": (0, 4), "5-9": (5, 9), "10-14": (10, 14)}[grp]
        s = sm.filter((pl.col("m") >= lo) & (pl.col("m") <= hi))
        if not s.is_empty():
            ax.plot(s["t0"], np.cumsum(s["pnl_net"].to_numpy()), color=color, linewidth=1, linestyle=":", label=f"model m {grp}")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_ylabel("cumulative net P&L ($ per 1-contract trade)")
    ax.set_title(f"Cumulative P&L at c={c:.2f}, τ={tau:.2f}", fontsize=10, loc="left")
    _legend(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_pup_vs_outcome(forecasts: pl.DataFrame, out: Path) -> None:
    fig, axes = _fig(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (grp, (lo, hi)) in zip(axes, {"0-4": (0, 4), "5-9": (5, 9), "10-14": (10, 14)}.items(), strict=True):
        s = forecasts.filter((pl.col("m") >= lo) & (pl.col("m") <= hi))
        if s.is_empty():
            continue
        x = s["p_up"].to_numpy()
        y = 1e4 * np.log(s["settlement"].to_numpy() / s["strike"].to_numpy())
        lim = np.nanpercentile(np.abs(y), 99) if y.size else 1
        ax.hexbin(x, y, gridsize=30, cmap="Blues", mincnt=1, extent=(0, 1, -lim, lim), linewidths=0.2)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.axvline(0.5, color=AXIS, linewidth=0.8)
        ax.set_title(f"m {grp}", fontsize=10, loc="left")
        ax.set_xlabel("P(up) model")
    axes[0].set_ylabel("settlement vs strike (bp)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_forecast(ctx: pl.DataFrame, fc: pl.DataFrame, out: Path, target_col: str) -> None:
    fig, ax = _fig(figsize=(10, 4.5))
    tail = ctx.tail(min(ctx.height, 240))
    ax.plot(tail["ts"], tail[target_col], color=INK2, linewidth=1.4, label=f"{target_col} (context tail)")
    ax.fill_between(fc["ts"], fc["q10"], fc["q90"], color=SERIES["model"], alpha=BAND_OUTER, linewidth=0, label="q10–q90")
    ax.fill_between(fc["ts"], fc["q20"], fc["q80"], color=SERIES["model"], alpha=BAND_INNER, linewidth=0, label="q20–q80")
    ax.plot(fc["ts"], fc["median"], color=SERIES["model"], linewidth=2, label="median forecast")
    ax.set_ylabel("BRTI (USD)")
    ax.set_title(f"TimesFM forecast from {fc['origin_ts'][0]:%Y-%m-%d %H:%M} UTC", fontsize=10, loc="left")
    _legend(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_all(run_dir: Path, forecasts, by_m, boot, trades, ts_m, calibration, c, tau, n_traj) -> None:  # noqa: ANN001
    pdir = run_dir / "plots"
    pdir.mkdir(exist_ok=True)
    plot_vs_m(by_m, boot, ts_m, pdir / "vs_m.png", c, tau)
    if not calibration.is_empty():
        plot_calibration(calibration, forecasts, pdir / "calibration.png")
    if not trades.is_empty():
        plot_pnl_curve(trades, pdir / "pnl_curve.png", c, tau)
    plot_pup_vs_outcome(forecasts, pdir / "pup_vs_outcome.png")
    # trajectories: evenly spaced windows plus the best and worst final-minute Brier
    t0s = forecasts["t0"].unique().sort()
    if n_traj > 0 and len(t0s):
        pick = set(t0s.gather_every(max(1, len(t0s) // max(1, n_traj - 2))).head(max(1, n_traj - 2)).to_list())
        last = forecasts.filter(pl.col("m") == forecasts["m"].max()).with_columns(
            ((pl.col("p_up") - pl.col("label_up").cast(pl.Float64)) ** 2).alias("_b")).sort("_b")
        if last.height:
            pick.add(last["t0"][0])
            pick.add(last["t0"][-1])
        for t0 in sorted(pick):
            rows = forecasts.filter(pl.col("t0") == t0)
            plot_window_trajectory(rows, pdir / f"trajectory_{t0:%Y%m%d_%H%M}.png", tau)
    log.info("plots written to %s", pdir)
