"""Post-training (fine-tuning) of TimesFM 3.0 on the BRTI 1-minute bars.

Data split. Chronological, on UTC-midnight boundaries, never shuffled across time:

    train | embargo | validation | embargo | test

A sample is an origin bar `o`: context = bars[o-C+1 .. o], targets = bars[o+1 .. o+H]. An origin
belongs to a split only if *all* H target bars lie inside that split's target range, and the range
of train / validation ends `embargo_bars` before the next split starts. So no target bar is shared
between splits and the last train target is at least one embargo away from the first validation
target (same for validation -> test). Contexts may reach back into the previous split: that is past
data the model would also see live, not leakage. The test split runs to the end of the data, so
`chudp backtest-kalshi --start <test_start> --end <test_end>` backtests exactly the held-out dates.

Roles: train = gradient steps; validation = early stopping / checkpoint selection; test = scored
once at the end with the selected checkpoint (and once with the untouched zero-shot model).

Training path. `TimesFM3Torch.decode` is the function inference uses, but it is wrapped in
`torch.no_grad()`. We call the undecorated function, so training optimises exactly the computation
`predict_batch` runs (RevIN, detrending, CPM refinement, stitching). The module stays in eval()
mode: it has no dropout, and this keeps the two paths bit-identical.

Loss. Pinball loss over the nine deciles on the first `loss_horizon` steps (15 = one Kalshi
window), each step divided by the sample's random-walk scale sigma_1m * sqrt(k + 1/3), which makes
the loss scale-free across volatility regimes and across steps. The same normalised loss is
reported for the Gaussian RW+vol baseline, so "does the model beat a random walk" is one comparison.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import polars as pl

from .backtest import _git_sha, _jsonable
from .model import DEFAULT_MODEL, load_forecaster
from .resample import freq_seconds, load_bars
from .settings import Settings
from .transforms import forward
from .windows import MAX_TIMESFM_CONTEXT, WINDOW_MINUTES, KalshiSpec, bar_features

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MEDIAN_IDX = 4


@dataclass(frozen=True)
class FinetuneConfig:
    # data
    context: int = 4096
    horizon: int = 64
    loss_horizon: int = WINDOW_MINUTES       # steps that enter the loss and the metrics (<= horizon)
    target_col: str = "mean"
    transform: str = "log"
    vol_lookback: int = 240
    max_ctx_gap_frac: float = 0.01
    start: date | None = None                # clip the data before splitting
    end: date | None = None
    # split
    val_frac: float = 0.15
    test_frac: float = 0.15
    val_start: date | None = None            # explicit boundaries override the fractions
    test_start: date | None = None
    embargo_bars: int = 1440                 # gap between the last target of a split and the next split
    # model
    model_id: str = DEFAULT_MODEL
    device: str | None = None
    precision: str = "fp32"                  # fp32 | bf16 (autocast). Evaluation is always fp32, like the backtest.
    trainable: str = "all"                   # all | head | last:N (last N transformer layers + output head)
    # optimisation
    lr: float = 1e-5
    min_lr_frac: float = 0.1
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 2000
    batch_size: int = 64                     # origins per optimiser step
    micro_batch: int = 32                    # origins per forward/backward (gradient accumulation)
    grad_clip: float = 1.0
    # evaluation / selection
    eval_every: int = 200
    eval_samples: int = 8192                 # evenly spaced origins per split; 0 = all
    eval_batch: int = 64
    patience: int = 5                        # evaluations without improvement before stopping; 0 = off
    log_every: int = 20
    seed: int = 0
    run_id: str | None = None

    def __post_init__(self) -> None:
        if not 0 < self.loss_horizon <= self.horizon:
            raise ValueError("loss_horizon must be within 1..horizon")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if self.val_frac <= 0 or self.test_frac <= 0 or self.val_frac + self.test_frac >= 1:
            raise ValueError("val_frac and test_frac must be positive and sum to < 1")
        if min(self.micro_batch, self.batch_size, self.max_steps, self.eval_every) <= 0 or self.embargo_bars < 0:
            raise ValueError("batch_size, micro_batch, max_steps, eval_every must be > 0 and embargo_bars >= 0")
        if self.context > MAX_TIMESFM_CONTEXT:
            raise ValueError(f"context {self.context} exceeds TimesFM's {MAX_TIMESFM_CONTEXT}-point limit")
        parse_trainable(self.trainable)


def parse_trainable(spec: str) -> int | None:
    """'all' -> None, 'head' -> 0, 'last:N' -> N (transformer layers trained besides the head)."""
    if spec == "all":
        return None
    if spec == "head":
        return 0
    if spec.startswith("last:") and spec[5:].isdigit() and int(spec[5:]) > 0:
        return int(spec[5:])
    raise ValueError(f"trainable must be 'all', 'head' or 'last:N', got {spec!r}")


# ---------------------------------------------------------------------------------------------
# chronological split

@dataclass(frozen=True)
class SplitBounds:
    """[start, end) of each split's *target* range, as datetimes on the bar grid."""

    train: tuple[datetime, datetime]
    val: tuple[datetime, datetime]
    test: tuple[datetime, datetime]
    embargo_bars: int

    def as_dict(self) -> dict[str, Any]:
        return {s: [getattr(self, s)[0].isoformat(), getattr(self, s)[1].isoformat()] for s in SPLITS} | {"embargo_bars": self.embargo_bars}


def _midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day)


def split_bounds(first_usable: datetime, end_excl: datetime, cfg: FinetuneConfig, step_s: int = 60) -> SplitBounds:
    """Boundaries on UTC midnights. Fractions are taken over [first_usable, end_excl), the span in
    which origins can exist (the first `context` bars only ever serve as history)."""
    span = end_excl - first_usable

    def at_frac(f: float) -> datetime:  # nearest UTC midnight
        return _midnight((first_usable + span * f + timedelta(hours=12)).date())

    test_start = _midnight(cfg.test_start) if cfg.test_start else at_frac(1.0 - cfg.test_frac)
    val_start = _midnight(cfg.val_start) if cfg.val_start else at_frac(1.0 - cfg.test_frac - cfg.val_frac)
    emb = timedelta(seconds=step_s * cfg.embargo_bars)
    b = SplitBounds(
        train=(first_usable, val_start - emb),
        val=(val_start, test_start - emb),
        test=(test_start, end_excl),
        embargo_bars=cfg.embargo_bars,
    )
    for name in SPLITS:
        lo, hi = getattr(b, name)
        if hi <= lo:
            raise ValueError(f"empty {name} split {lo} .. {hi}: not enough data for these boundaries / embargo")
    return b


def valid_origins(feats: pl.DataFrame, cfg: FinetuneConfig) -> np.ndarray:
    """Bar indices usable as forecast origins: full context with few gaps, a real first and last
    bar, a vol estimate, the full horizon inside the data and at least one real target in the loss
    horizon.

    The first context bar must be real because a context that starts inside a gap gets a (partly)
    masked first patch, whose running variance is exactly 0; timesfm3 takes sqrt() of it, which is
    fine forward but yields NaN gradients. Interior gaps are interpolated and train fine."""
    n, C, H, L = feats.height, cfg.context, cfg.horizon, cfg.loss_horizon
    cum = np.concatenate([[0], feats["_null_cum"].to_numpy().astype(np.int64)])  # cum[i+1] = nulls in y[0..i]
    o = np.arange(C - 1, n - H, dtype=np.int64)
    if o.size == 0:
        return o
    ctx_gaps = cum[o + 1] - cum[o + 1 - C]
    tgt_nulls = cum[o + L + 1] - cum[o + 1]
    y = feats["y"].to_numpy()
    close = feats["close"].to_numpy()
    sigma = feats["sigma_1m"].to_numpy()
    ok = (
        (ctx_gaps <= cfg.max_ctx_gap_frac * C)
        & np.isfinite(y[o]) & np.isfinite(y[o - C + 1]) & np.isfinite(close[o])
        & np.isfinite(sigma[o]) & (sigma[o] > 0)
        & (tgt_nulls < L)
    )
    return o[ok]


def assign_splits(origins: np.ndarray, first_ts: datetime, bounds: SplitBounds, horizon: int, step_s: int = 60) -> dict[str, np.ndarray]:
    """Origin o goes to split S iff its targets o+1 .. o+horizon all lie in S's target range."""
    def idx(t: datetime) -> int:
        return int((t - first_ts).total_seconds() // step_s)

    out = {}
    for name in SPLITS:
        lo, hi = getattr(bounds, name)
        out[name] = origins[(origins + 1 >= idx(lo)) & (origins + horizon < idx(hi))]
    return out


# ---------------------------------------------------------------------------------------------
# batches

class Samples:
    """Array view of the bar features that turns origin indices into model-ready batches."""

    def __init__(self, feats: pl.DataFrame, cfg: FinetuneConfig) -> None:
        self.cfg = cfg
        self.y = feats["y"].to_numpy().astype(np.float64)                       # model space, NaN = gap
        close = feats["close"].to_numpy().astype(np.float64)
        self.close = close
        self.y_close = forward(np.where(np.isfinite(close), close, np.nan), cfg.transform)
        self.sigma = feats["sigma_1m"].to_numpy().astype(np.float64)            # log-return vol per bar
        self.ctx_off = np.arange(-cfg.context + 1, 1, dtype=np.int64)
        self.tgt_off = np.arange(1, cfg.horizon + 1, dtype=np.int64)
        self.sqrt_h = np.sqrt(np.arange(cfg.horizon, dtype=np.float64) + 1.0 / 3.0)  # bar-mean target, see metrics.h_eff

    def batch(self, origins: np.ndarray) -> dict[str, np.ndarray]:
        from timesfm3.torch.timesfm3_forecaster import linear_interpolation

        o = np.asarray(origins, dtype=np.int64)
        ctx = self.y[o[:, None] + self.ctx_off[None, :]]
        mask = np.zeros(ctx.shape, dtype=bool)
        # Same treatment as TimesFM3Forecaster.predict_batch: leading NaNs are masked out, the
        # rest is linearly interpolated.
        for r in np.flatnonzero(np.isnan(ctx).any(axis=1)):
            first = int(np.argmax(~np.isnan(ctx[r])))
            mask[r, :first] = True
            ctx[r, :first] = 0.0
            ctx[r, first:] = linear_interpolation(ctx[r, first:])
        tgt = self.y[o[:, None] + self.tgt_off[None, :]]
        valid = np.isfinite(tgt)
        scale = self.sigma[o][:, None] * self.sqrt_h[None, :]
        if self.cfg.transform == "none":                                        # sigma is a log-return vol
            scale = scale * self.close[o][:, None]
        return {
            "context": ctx.astype(np.float32), "mask": mask,
            "target": np.where(valid, tgt, 0.0).astype(np.float32), "valid": valid,
            "scale": scale.astype(np.float32), "naive": self.y_close[o].astype(np.float32),
        }


def subsample(idx: np.ndarray, k: int) -> np.ndarray:
    """k evenly spaced elements (deterministic, covers the whole split); k<=0 or k>=len -> all."""
    if k <= 0 or k >= idx.size:
        return idx
    return idx[np.linspace(0, idx.size - 1, k).round().astype(np.int64)]


# ---------------------------------------------------------------------------------------------
# model plumbing

def forward_train(model, context, mask, horizon: int):  # noqa: ANN001
    """Gradient-enabled twin of the inference call: (b, C) -> (b, horizon', 9) raw quantiles."""
    from timesfm3 import TimesFM3Torch

    fn = getattr(TimesFM3Torch.decode, "__wrapped__", None)
    if fn is None:
        raise RuntimeError("TimesFM3Torch.decode is no longer a no_grad-wrapped function; re-check the training path")
    out_len = math.ceil(horizon / model.output_patch_len) * model.output_patch_len  # as predict_batch does
    return fn(model, target=context[:, None, :], horizon=out_len, mask=mask)[:, 0, :horizon, :]


def set_trainable(model, spec: str) -> int:  # noqa: ANN001
    """Freeze everything the spec does not name; returns the number of trainable parameters."""
    n_layers = parse_trainable(spec)
    if n_layers is not None:
        for p in model.parameters():
            p.requires_grad_(False)
        layers = list(model.transformer_stack.layers)
        for mod in [model.output_head, *layers[len(layers) - min(n_layers, len(layers)):]]:
            for p in mod.parameters():
                p.requires_grad_(True)
    else:
        for p in model.parameters():
            p.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pinball_loss(pred, target, scale, valid, quantiles=QUANTILES):  # noqa: ANN001
    """Mean over valid (sample, step) pairs and quantiles of the scale-normalised pinball loss.
    pred (b, L, Q); target, scale, valid (b, L)."""
    import torch

    q = torch.as_tensor(quantiles, dtype=pred.dtype, device=pred.device)
    err = (target[..., None] - pred) / scale[..., None]
    loss = torch.maximum(q * err, (q - 1.0) * err).mean(dim=-1)
    w = valid.to(loss.dtype)
    return (loss * w).sum() / w.sum().clamp_min(1.0)


def lr_at(step: int, cfg: FinetuneConfig) -> float:
    """Linear warmup, then cosine decay to min_lr_frac * lr at max_steps. `step` is 1-based."""
    if step <= cfg.warmup_steps:
        return cfg.lr * step / max(cfg.warmup_steps, 1)
    t = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def _np_pinball(pred: np.ndarray, target: np.ndarray, scale: np.ndarray) -> np.ndarray:
    q = np.asarray(QUANTILES)
    err = (target[..., None] - pred) / scale[..., None]
    return np.maximum(q * err, (q - 1.0) * err).mean(axis=-1)


def evaluate(model, samples: Samples, origins: np.ndarray, cfg: FinetuneConfig, device: str) -> dict[str, float]:  # noqa: ANN001
    """fp32, no-grad, quantiles sorted: what `predict` would return. Scores the first loss_horizon
    steps for the model and for the Gaussian RW+vol baseline on the same samples."""
    import torch

    L = cfg.loss_horizon
    z = np.array([NormalDist().inv_cdf(p) for p in QUANTILES])
    acc = {k: 0.0 for k in ("pinball", "pinball_rw", "ae", "ae_naive", "cover80", "n")}
    with torch.no_grad():
        for lo in range(0, origins.size, cfg.eval_batch):
            b = samples.batch(origins[lo:lo + cfg.eval_batch])
            ctx = torch.from_numpy(b["context"]).to(device)
            msk = torch.from_numpy(b["mask"]).to(device)
            pred = np.sort(forward_train(model, ctx, msk, cfg.horizon)[:, :L].float().cpu().numpy().astype(np.float64), axis=-1)
            tgt, valid, scale = b["target"][:, :L].astype(np.float64), b["valid"][:, :L], b["scale"][:, :L].astype(np.float64)
            naive = b["naive"].astype(np.float64)[:, None]
            rw = naive[..., None] + scale[..., None] * z[None, None, :]
            med = pred[..., MEDIAN_IDX]
            acc["pinball"] += _np_pinball(pred, tgt, scale)[valid].sum()
            acc["pinball_rw"] += _np_pinball(rw, tgt, scale)[valid].sum()
            acc["ae"] += np.abs(med - tgt)[valid].sum()
            acc["ae_naive"] += np.abs(naive - tgt)[valid].sum()
            acc["cover80"] += ((tgt >= pred[..., 0]) & (tgt <= pred[..., -1]))[valid].sum()
            acc["n"] += valid.sum()
    n = max(acc["n"], 1.0)
    out = {
        "pinball": acc["pinball"] / n,
        "pinball_rw": acc["pinball_rw"] / n,
        "mase_h": acc["ae"] / acc["ae_naive"] if acc["ae_naive"] > 0 else float("nan"),
        "cover80": acc["cover80"] / n,
        "n_origins": int(origins.size),
    }
    out["skill_vs_rw"] = 1.0 - out["pinball"] / out["pinball_rw"] if out["pinball_rw"] > 0 else float("nan")
    if cfg.transform == "log":
        out["mae_bp"] = acc["ae"] / n * 1e4
    return out


# ---------------------------------------------------------------------------------------------
# run

@dataclass
class _Epochs:
    """Endless stream of shuffled train origins, one permutation per epoch."""

    idx: np.ndarray
    rng: np.random.Generator
    pos: int = 0
    epoch: int = 0
    perm: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def take(self, k: int) -> np.ndarray:
        out = []
        while k > 0:
            if self.pos >= self.perm.size:
                self.perm, self.pos, self.epoch = self.rng.permutation(self.idx), 0, self.epoch + 1
            got = self.perm[self.pos:self.pos + k]
            out.append(got)
            self.pos += got.size
            k -= got.size
        return np.concatenate(out)


def prepare_data(settings: Settings, cfg: FinetuneConfig) -> tuple[Samples, dict[str, np.ndarray], SplitBounds, pl.DataFrame]:
    spec = KalshiSpec(context=cfg.context, horizon=max(cfg.horizon, WINDOW_MINUTES), target_col=cfg.target_col,
                      transform=cfg.transform, vol_lookback=cfg.vol_lookback, max_ctx_gap_frac=cfg.max_ctx_gap_frac)
    step_s = freq_seconds(spec.freq)
    bars = load_bars(settings.processed_dir, spec.freq)
    if cfg.start:
        bars = bars.filter(pl.col("ts") >= _midnight(cfg.start))
    if cfg.end:
        bars = bars.filter(pl.col("ts") < _midnight(cfg.end) + timedelta(days=1))
    if bars.height <= cfg.context + cfg.horizon:
        raise ValueError(f"only {bars.height} bars in range, need more than context + horizon = {cfg.context + cfg.horizon}")
    feats = bar_features(bars, spec)
    first_ts: datetime = feats["ts"][0]
    last_ts: datetime = feats["ts"][-1]
    if feats.height != int((last_ts - first_ts).total_seconds() // step_s) + 1:
        raise ValueError("bars are not on a regular grid; rebuild them with `chudp resample`")
    bounds = split_bounds(first_ts + timedelta(seconds=step_s * cfg.context), last_ts + timedelta(seconds=step_s), cfg, step_s)
    splits = assign_splits(valid_origins(feats, cfg), first_ts, bounds, cfg.horizon, step_s)
    for name in SPLITS:
        if splits[name].size == 0:
            raise ValueError(f"{name} split has no usable origins ({getattr(bounds, name)})")
    return Samples(feats, cfg), splits, bounds, feats


def _fmt(m: dict[str, float]) -> str:
    keys = ("pinball", "pinball_rw", "skill_vs_rw", "mase_h", "mae_bp", "cover80")
    return "  ".join(f"{k}={m[k]:.4f}" for k in keys if k in m)


def run_finetune(settings: Settings, cfg: FinetuneConfig) -> Path:
    import torch

    torch.manual_seed(cfg.seed)
    run_id = cfg.run_id or f"{datetime.now(UTC).replace(tzinfo=None):%Y%m%d-%H%M%S}-finetune-c{cfg.context}"
    out = settings.finetune_dir / run_id
    out.mkdir(parents=True, exist_ok=True)

    samples, splits, bounds, feats = prepare_data(settings, cfg)
    ts = feats["ts"]
    split_info = bounds.as_dict() | {
        "origins": {s: {"n": int(splits[s].size), "first": ts[int(splits[s][0])].isoformat(), "last": ts[int(splits[s][-1])].isoformat()} for s in SPLITS}
    }
    (out / "splits.json").write_text(json.dumps(split_info, indent=1))
    (out / "config.json").write_text(json.dumps(asdict(cfg) | {"run_id": run_id, "git_sha": _git_sha()}, indent=1, default=_jsonable))
    for s in SPLITS:
        lo, hi = getattr(bounds, s)
        log.info("%-5s targets in [%s, %s): %d origins", s, lo, hi, splits[s].size)

    handle = load_forecaster(cfg.model_id, device=cfg.device, batch_size=cfg.eval_batch)
    model, device = handle.obj.model, handle.device
    model.eval()  # no dropout in this network; eval() keeps training identical to the inference path
    n_train = set_trainable(model, cfg.trainable)
    log.info("trainable parameters: %.1fM of %.1fM (%s)", n_train / 1e6, sum(p.numel() for p in model.parameters()) / 1e6, cfg.trainable)

    val_idx, test_idx = subsample(splits["val"], cfg.eval_samples), subsample(splits["test"], cfg.eval_samples)
    base = {"val": evaluate(model, samples, val_idx, cfg, device), "test": evaluate(model, samples, test_idx, cfg, device)}
    log.info("zero-shot val : %s", _fmt(base["val"]))
    log.info("zero-shot test: %s", _fmt(base["test"]))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": [p for p in params if p.ndim >= 2], "weight_decay": cfg.weight_decay},
         {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}],
        lr=cfg.lr,
    )
    stream = _Epochs(splits["train"], np.random.default_rng(cfg.seed))
    autocast = cfg.precision == "bf16" and device != "cpu"
    L = cfg.loss_horizon
    best = {"step": 0, "pinball": base["val"]["pinball"]}
    best_dir, stale, run_loss, run_n, skipped, skipped_in_a_row = out / "best", 0, 0.0, 0, 0, 0
    history = [{"step": 0, "val": base["val"]}]
    t0 = time.time()
    with (out / "train_log.jsonl").open("w") as logf:
        logf.write(json.dumps(history[0]) + "\n")
        for step in range(1, cfg.max_steps + 1):
            lr = lr_at(step, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            origins = stream.take(cfg.batch_size)
            step_loss = 0.0
            for lo in range(0, origins.size, cfg.micro_batch):
                b = samples.batch(origins[lo:lo + cfg.micro_batch])
                t = {k: torch.from_numpy(v).to(device) for k, v in b.items() if k != "naive"}
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
                    pred = forward_train(model, t["context"], t["mask"], cfg.horizon)
                loss = pinball_loss(pred[:, :L].float(), t["target"][:, :L], t["scale"][:, :L], t["valid"][:, :L])
                w = b["context"].shape[0] / origins.size
                (loss * w).backward()
                step_loss += float(loss.detach()) * w
            if not math.isfinite(step_loss):
                raise RuntimeError(f"non-finite training loss at step {step}; lower the learning rate")
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip if cfg.grad_clip > 0 else float("inf")))
            if math.isfinite(gnorm):
                opt.step()
                run_loss, run_n, skipped_in_a_row = run_loss + step_loss, run_n + 1, 0
            else:  # a finite loss can still have a non-finite gradient; never let it reach the weights
                skipped, skipped_in_a_row = skipped + 1, skipped_in_a_row + 1
                log.warning("step %d: non-finite gradient, update skipped (%d so far)", step, skipped)
                if skipped_in_a_row >= 10:
                    raise RuntimeError("10 consecutive non-finite gradients; lower the learning rate or use precision=fp32")
            opt.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0 or step == cfg.max_steps:
                rate = step * cfg.batch_size / max(time.time() - t0, 1e-9)
                log.info("step %d/%d  train_pinball=%.4f  grad_norm=%.3f  lr=%.2e  epoch=%d  %.0f origins/s",
                         step, cfg.max_steps, run_loss / max(run_n, 1), gnorm, lr, stream.epoch, rate)
                logf.write(json.dumps({"step": step, "train_pinball": run_loss / max(run_n, 1), "grad_norm": gnorm, "lr": lr}) + "\n")
                run_loss, run_n = 0.0, 0
            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                val = evaluate(model, samples, val_idx, cfg, device)
                improved = val["pinball"] < best["pinball"]
                log.info("step %d val: %s%s", step, _fmt(val), "  *best*" if improved else "")
                history.append({"step": step, "val": val})
                logf.write(json.dumps(history[-1]) + "\n")
                logf.flush()
                if improved:
                    best, stale = {"step": step, "pinball": val["pinball"]}, 0
                    model.save_pretrained(best_dir)
                else:
                    stale += 1
                    if cfg.patience and stale >= cfg.patience:
                        log.info("early stop: no validation improvement in %d evaluations", stale)
                        break
    model.save_pretrained(out / "last")

    tuned = None
    if best["step"] > 0:
        del opt
        tuned_model = load_forecaster(str(best_dir), device=device, batch_size=cfg.eval_batch).obj.model  # proves the checkpoint loads
        tuned = {"val": evaluate(tuned_model, samples, val_idx, cfg, device), "test": evaluate(tuned_model, samples, test_idx, cfg, device)}

    summary = {"run_id": run_id, "best_step": best["step"], "steps_run": step, "skipped_steps": skipped, "splits": split_info, "zero_shot": base,
               "finetuned": tuned, "best_checkpoint": str(best_dir) if tuned else None, "history": history}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
    (out / "summary.txt").write_text(_summary_text(summary, cfg, bounds))
    log.info("fine-tune run written to %s", out)
    return out


def _summary_text(s: dict[str, Any], cfg: FinetuneConfig, bounds: SplitBounds) -> str:
    cols = ("pinball", "pinball_rw", "skill_vs_rw", "mase_h", "mae_bp", "cover80")
    lines = [f"== Fine-tune {s['run_id']} ==", f"steps run {s['steps_run']} ({s['skipped_steps']} skipped for non-finite gradients), best validation step {s['best_step']}", "", "== Splits (target ranges, UTC) =="]
    for name in SPLITS:
        lo, hi = getattr(bounds, name)
        o = s["splits"]["origins"][name]
        lines.append(f"{name:>6}  [{lo:%Y-%m-%d %H:%M}, {hi:%Y-%m-%d %H:%M})  {o['n']:>8} origins")
    lines += [f"embargo {bounds.embargo_bars} bars between splits", "",
              f"== Scores on the first {cfg.loss_horizon} steps (pinball is RW-scale normalised; skill_vs_rw > 0 beats RW+vol) ==",
              f"{'model':>10} {'split':>6} " + " ".join(f"{c:>12}" for c in cols)]
    for label, res in (("zero-shot", s["zero_shot"]), ("finetuned", s["finetuned"])):
        for split in ("val", "test"):
            if res:
                lines.append(f"{label:>10} {split:>6} " + " ".join(f"{res[split].get(c, float('nan')):>12.4f}" for c in cols))
    test_lo, test_hi = bounds.test
    dates = f"--start {test_lo:%Y-%m-%d} --end {(test_hi - timedelta(seconds=1)):%Y-%m-%d}"
    lines.append("")
    if s["finetuned"]:
        lines += ["Backtest the held-out test dates (validation and train dates are contaminated for this checkpoint):",
                  f"  chudp backtest-kalshi --config configs/kalshi_full.toml --model-id {s['best_checkpoint']} {dates}",
                  f"  chudp backtest-kalshi --config configs/kalshi_full.toml {dates}    # zero-shot on the same dates"]
    else:
        lines.append("Validation never improved on the zero-shot model, so no checkpoint was promoted (see last/ for the final weights).")
    return "\n".join(lines) + "\n"
