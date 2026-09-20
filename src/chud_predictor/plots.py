"""Static PNG figures for a contract-price backtest (needs the `viz` extra).

Palette and chrome follow the reference data-viz palette: fixed categorical slot order, one hue per
entity, thin marks, recessive grid, one y-axis per panel (never a dual axis).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from . import metrics as M  # noqa: E402

log = logging.getLogger(__name__)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"model": "#2a78d6", "persistence": "#eb6834", "rw": "#1baf7a", "actual": "#0b0b0b", "strike": "#52514e"}
MGROUP = {"0-4": "#2a78d6", "5-9": "#eb6834", "10-14": "#1baf7a"}
FAN_ORIGINS = {0: "#2a78d6", 5: "#eb6834", 10: "#1baf7a"}
DIVERGING = LinearSegmentedColormap.from_list("skill", ["#e34948", "#f0efec", "#2a78d6"])   # worse | neutral gray | better
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


def _legend(ax, **kw) -> None:  # noqa: ANN001
    leg = ax.legend(frameon=False, fontsize=8, **kw)
    if leg:
        for t in leg.get_texts():
            t.set_color(INK2)


def _save(fig, out: Path) -> None:  # noqa: ANN001
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_error_vs_h(by_h: pl.DataFrame, boot: pl.DataFrame, out: Path) -> None:
    """RMSE by steps ahead, model vs the two baselines (left) and skill vs persistence (right)."""
    by_h = by_h.sort("h")
    h = by_h["h"].to_numpy()
    fig, (a, b) = _fig(1, 2, figsize=(11, 4.5))
    for key, col, label in (("model", "rmse_c", "model (mean of deciles)"), ("persistence", "rmse_c_persist", "persistence (last mid)"), ("rw", "rmse_c_rw", "index fair value")):
        a.plot(h, by_h[col], color=SERIES[key], linewidth=2 if key == "model" else 1.6, marker="o", markersize=4 if key == "model" else 3, label=label)
    a.set_ylabel("root mean squared error (cents)")
    a.set_title("Error by minutes ahead", fontsize=10, loc="left")
    _legend(a)
    b.axhline(0, color=AXIS, linewidth=1)
    b.plot(h, by_h["skill_mse"], color=SERIES["model"], linewidth=2, marker="o", markersize=4, label="skill_mse")
    b.plot(h, by_h["skill_pinball"], color=SERIES["rw"], linewidth=1.6, marker="o", markersize=3, label="skill_pinball")
    cs = boot.filter((pl.col("by") == "h") & (pl.col("stat") == "skill_mse")).with_columns(pl.col("key").cast(pl.Int64)).sort("key")
    if cs.height == h.size:
        b.fill_between(h, cs["lo"], cs["hi"], color=SERIES["model"], alpha=BAND_OUTER, linewidth=0)
    b.set_ylabel("skill vs persistence (> 0 is better)")
    b.set_title("Skill by minutes ahead (band: 95% day-block interval)", fontsize=10, loc="left")
    _legend(b)
    for ax in (a, b):
        ax.set_xlabel("steps ahead h (minutes)")
        ax.set_xticks(range(1, 16))
    _save(fig, out)


def plot_skill_heatmap(by_mh: pl.DataFrame, out: Path) -> None:
    grid = np.full((15, 15), np.nan)
    for m, h, s in by_mh.select("m", "h", "skill_mse").iter_rows():
        grid[int(m), int(h) - 1] = s
    lim = max(0.05, float(np.nanmax(np.abs(grid)))) if np.isfinite(grid).any() else 0.05
    fig, ax = _fig(figsize=(9.5, 6))
    ax.grid(False)
    im = ax.imshow(grid, cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), origin="upper", aspect="auto")
    for m in range(15):
        for h in range(15 - m):
            v = grid[m, h]
            if np.isfinite(v):
                ax.text(h, m, f"{v:+.2f}", ha="center", va="center", fontsize=6.5, color=INK if abs(v) < 0.6 * lim else SURFACE)
    ax.set_xticks(range(15), [str(h) for h in range(1, 16)])
    ax.set_yticks(range(15))
    ax.set_xlabel("steps ahead h (minutes)")
    ax.set_ylabel("origin minute m")
    ax.set_title("skill_mse vs persistence (blue: model better, red: worse)", fontsize=10, loc="left")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    _save(fig, out)


def plot_trajectory(rows: pl.DataFrame, out: Path) -> None:
    """One contract: the realised mid by minute, and forecast fans issued at m = 0, 5, 10."""
    rows = rows.sort("m", "h")
    fig, ax = _fig(figsize=(9, 4.8))
    actual = rows.group_by("k").agg(pl.col("actual").first()).sort("k")
    for m, color in FAN_ORIGINS.items():
        f = rows.filter(pl.col("m") == m)
        if f.is_empty():
            continue
        ax.fill_between(f["k"], f["q10"], f["q90"], color=color, alpha=BAND_OUTER, linewidth=0)
        ax.fill_between(f["k"], f["q20"], f["q80"], color=color, alpha=BAND_INNER, linewidth=0)
        ax.plot(f["k"], f["median"], color=color, linewidth=1.8, label=f"forecast at m={m} (median, q10–q90)")
        ax.axvline(m - 0.5, color=color, linewidth=0.8, linestyle=":")
    ax.plot(actual["k"], actual["actual"], color=SERIES["actual"], linewidth=2.2, marker="o", markersize=4, label="realised mid")
    ax.axhline(0.5, color=AXIS, linewidth=0.8)
    r = rows.row(0, named=True)
    outcome = {True: "YES", False: "NO", None: "?"}[r["outcome_price"]]
    ax.set_title(f"{r['ticker']}  window {r['t0']:%Y-%m-%d %H:%M} UTC, strike {r['strike']:,.2f}, settled {outcome}", fontsize=10, loc="left")
    ax.set_xlabel("minute of the window k (candle k closes at T0 + k + 1)")
    ax.set_ylabel("YES price (dollars)")
    ax.set_xticks(range(15))
    ax.set_ylim(-0.02, 1.02)
    _legend(ax, loc="best")
    _save(fig, out)


def plot_calibration(forecasts: pl.DataFrame, out: Path) -> None:
    fig, (a, b) = _fig(1, 2, figsize=(11, 4.5))
    levels = np.array([0.2, 0.4, 0.6, 0.8])
    pairs = [("q40", "q60"), ("q30", "q70"), ("q20", "q80"), ("q10", "q90")]
    a.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8, linestyle="--")
    for grp, color in MGROUP.items():
        g = forecasts.filter((pl.col("m_group") == grp) & ~pl.col("is_settlement"))
        if g.is_empty():
            continue
        y = g["actual"].to_numpy()
        cov = [float(np.mean((y >= g[lo].to_numpy()) & (y <= g[hi].to_numpy()))) for lo, hi in pairs]
        a.plot(levels, cov, color=color, linewidth=1.8, marker="o", markersize=4, label=f"origin m {grp}")
        u = M.pit(g.select(M.QUANTILE_COLS).to_numpy(), y)
        b.hist(u, bins=np.linspace(0, 1, 21), histtype="step", linewidth=1.6, color=color, density=True, label=f"origin m {grp}")
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_xlabel("nominal central interval")
    a.set_ylabel("share of realised prices inside")
    a.set_title("Interval coverage (settlement candle excluded)", fontsize=10, loc="left")
    _legend(a)
    b.axhline(1.0, color=AXIS, linewidth=0.8, linestyle="--")
    b.set_xlabel("PIT of the realised price")
    b.set_ylabel("density (flat = calibrated)")
    b.set_title("Probability integral transform", fontsize=10, loc="left")
    _legend(b)
    _save(fig, out)


def plot_settlement_reliability(cal: pl.DataFrame, out: Path) -> None:
    fig, ax = _fig(figsize=(6, 5.2))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8, linestyle="--")
    for src, label in (("model", "model (mean of deciles)"), ("persistence", "last observed mid"), ("rw", "index fair value")):
        s = cal.filter(pl.col("source") == src).sort("bin_lo")
        if s.is_empty():
            continue
        sizes = 12 + 50 * np.sqrt(s["n"].to_numpy() / max(s["n"].max(), 1))
        ax.plot(s["p_mean"], s["freq_up"], color=SERIES[src], linewidth=1.6, label=label)
        ax.scatter(s["p_mean"], s["freq_up"], s=sizes, color=SERIES[src], edgecolor=SURFACE, linewidth=1, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted settlement price")
    ax.set_ylabel("share of contracts that settled YES")
    ax.set_title("Settlement candle as a probability", fontsize=10, loc="left")
    _legend(ax)
    _save(fig, out)


def plot_all(run_dir: Path, forecasts: pl.DataFrame, tables: dict[str, pl.DataFrame], n_traj: int) -> None:
    pdir = run_dir / "plots"
    pdir.mkdir(exist_ok=True)
    plot_error_vs_h(tables["metrics_by_h"], tables["bootstrap"], pdir / "error_vs_h.png")
    plot_skill_heatmap(tables["metrics_by_mh"], pdir / "skill_heatmap.png")
    plot_calibration(forecasts, pdir / "calibration.png")
    if not tables["settlement_calibration"].is_empty():
        plot_settlement_reliability(tables["settlement_calibration"], pdir / "settlement_reliability.png")
    t0s = forecasts.filter(pl.col("m") == 0)["t0"].unique().sort()
    if n_traj > 0 and t0s.len():
        for t0 in t0s.gather_every(max(1, t0s.len() // n_traj)).head(n_traj).to_list():
            plot_trajectory(forecasts.filter(pl.col("t0") == t0), pdir / f"trajectory_{t0:%Y%m%d_%H%M}.png")
    log.info("plots written to %s", pdir)
