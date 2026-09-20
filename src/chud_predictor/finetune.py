"""Post-training (fine-tuning) of TimesFM 3.0 on the Kalshi contract price.

Samples. Every minute of every window is a sample: the origins of `origins.make_origins`, the same
population and the same input arrays (`covariates.build_arrays`) the backtest uses. The target is
the contract mid at each remaining minute to expiry; steps past expiry never enter the loss.

Data split. Chronological, on UTC-midnight boundaries, never shuffled across time:

    train | embargo | validation | embargo | test

An origin belongs to a split only if all of its (up to 15) target rows lie inside that split's
target range, and train / validation end `embargo_bars` before the next split starts. Contexts may
reach back into the previous split: that is past data the model would also see live, not leakage.
The test split runs to the end of the data, so `chudp backtest-contract --start <test_start>`
backtests exactly the held-out dates.

Roles: train = gradient steps; validation = early stopping / checkpoint selection; test = scored
once at the end with the selected checkpoint (and once with the untouched zero-shot model).

Training path. `TimesFM3Torch.decode` is the function inference uses, but it is wrapped in
`torch.no_grad()`. We call the undecorated function with the same covariates, so training optimises
exactly the computation `predict` runs. The module stays in eval() mode: it has no dropout.

Loss. Pinball loss over the nine deciles, in price space (dollars), mean over valid (sample, step)
pairs. Reported next to the empirical persistence fan fitted on the TRAIN split only (saved as
`baselines.parquet`, reusable by the backtest with `--fan-from`): skill_pinball > 0 beats it.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from . import baselines as B
from . import metrics as M
from .backtest import _git_sha, _jsonable
from .covariates import MAX_STEPS, FrameArrays, build_arrays, frame_arrays
from .features import load_frame
from .model import DEFAULT_MODEL, load_forecaster
from .origins import MAX_TIMESFM_CONTEXT, ContractSpec, make_origins
from .settings import Settings

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
QUANTILES = tuple(M.QUANTILE_LEVELS.tolist())
MEDIAN_IDX = 4
MIN_SCALE = 0.01   # floor of the fan-width loss scale: one cent


@dataclass(frozen=True)
class FinetuneConfig:
    # data
    context: int = 1024
    horizon: int = 64
    covariates: str = "full"
    max_ctx_gap_frac: float = 0.01
    require_quote_ok: bool = False
    start: date | None = date(2025, 12, 15)  # first origin date: the contract era with a published strike
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
    loss_scale: str = "none"                 # none: dollars | fan: each step divided by the persistence fan's 10-90 width
    # optimisation
    lr: float = 1e-5
    min_lr_frac: float = 0.1
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 2000
    batch_size: int = 32                     # origins per optimiser step
    micro_batch: int = 8                     # origins per forward/backward (gradient accumulation)
    grad_clip: float = 1.0
    # evaluation / selection
    eval_every: int = 200
    eval_samples: int = 8192                 # evenly spaced origins per split; 0 = all
    eval_batch: int = 32
    patience: int = 5                        # evaluations without improvement before stopping; 0 = off
    log_every: int = 20
    seed: int = 0
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if self.loss_scale not in ("none", "fan"):
            raise ValueError("loss_scale must be 'none' or 'fan'")
        if self.val_frac <= 0 or self.test_frac <= 0 or self.val_frac + self.test_frac >= 1:
            raise ValueError("val_frac and test_frac must be positive and sum to < 1")
        if min(self.micro_batch, self.batch_size, self.max_steps, self.eval_every) <= 0 or self.embargo_bars < 0:
            raise ValueError("batch_size, micro_batch, max_steps, eval_every must be > 0 and embargo_bars >= 0")
        if self.context > MAX_TIMESFM_CONTEXT:
            raise ValueError(f"context {self.context} exceeds TimesFM's {MAX_TIMESFM_CONTEXT}-point limit")
        parse_trainable(self.trainable)

    @property
    def spec(self) -> ContractSpec:
        return ContractSpec(context=self.context, horizon=self.horizon, covariates=self.covariates,
                            max_ctx_gap_frac=self.max_ctx_gap_frac, require_quote_ok=self.require_quote_ok)


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
    """[start, end) of each split's *target* range, as datetimes on the minute grid."""

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
    which targets exist."""
    span = end_excl - first_usable

    def at_frac(f: float) -> datetime:  # nearest UTC midnight
        return _midnight((first_usable + span * f + timedelta(hours=12)).date())

    test_start = _midnight(cfg.test_start) if cfg.test_start else at_frac(1.0 - cfg.test_frac)
    val_start = _midnight(cfg.val_start) if cfg.val_start else at_frac(1.0 - cfg.test_frac - cfg.val_frac)
    emb = timedelta(seconds=step_s * cfg.embargo_bars)
    b = SplitBounds(train=(first_usable, val_start - emb), val=(val_start, test_start - emb), test=(test_start, end_excl), embargo_bars=cfg.embargo_bars)
    for name in SPLITS:
        lo, hi = getattr(b, name)
        if hi <= lo:
            raise ValueError(f"empty {name} split {lo} .. {hi}: not enough data for these boundaries / embargo")
    return b


def assign_splits(origins: np.ndarray, n_steps: np.ndarray, first_ts: datetime, bounds: SplitBounds, step_s: int = 60) -> dict[str, np.ndarray]:
    """Origin i goes to split S iff its targets i+1 .. i+n_steps all lie in S's target range."""
    def idx(t: datetime) -> int:
        return int((t - first_ts).total_seconds() // step_s)

    out = {}
    for name in SPLITS:
        lo, hi = getattr(bounds, name)
        out[name] = origins[(origins + 1 >= idx(lo)) & (origins + n_steps < idx(hi))]
    return out


# ---------------------------------------------------------------------------------------------
# batches

class Samples:
    """Turns origin row indices into model-ready batches, through the builder the backtest uses."""

    def __init__(self, fa: FrameArrays, cfg: FinetuneConfig, fan: pl.DataFrame | None = None) -> None:
        self.fa, self.cfg, self.fan = fa, cfg, fan
        self.steps = np.arange(1, MAX_STEPS + 1)

    def batch(self, origins: np.ndarray) -> dict[str, np.ndarray]:
        i = np.asarray(origins, dtype=np.int64)
        b = build_arrays(self.fa, i, self.cfg.context, self.cfg.horizon)
        n = i.size
        last_mid = self.fa.target[i]
        k_ctx = self.fa.k[i]
        fan = B.apply_fan(np.repeat(last_mid, MAX_STEPS), np.repeat(k_ctx, MAX_STEPS), np.tile(self.steps, n), self.fan).reshape(n, MAX_STEPS, -1)
        scale = np.ones((n, MAX_STEPS)) if self.cfg.loss_scale == "none" else np.maximum(fan[..., -1] - fan[..., 0], MIN_SCALE)
        out = {
            # as predict_batch does: a leading gap is masked and zero-filled (origins never start in one, see origins.py)
            "context": np.where(b["mask"], 0.0, b["targets"]).astype(np.float32), "mask": b["mask"],
            "target": np.nan_to_num(b["tgt"], nan=0.0).astype(np.float32), "valid": b["valid"], "scale": scale.astype(np.float32),
            "last_mid": last_mid, "persist": B.persist_ref(last_mid, k_ctx), "fan": fan, "step_k": b["step_k"], "n_flat": b["n_flat"],
        }
        if b["po"] is not None:
            out["po"] = b["po"]
        if b["pf"] is not None:
            out["pf"] = b["pf"]
        return out


def subsample(idx: np.ndarray, k: int) -> np.ndarray:
    """k evenly spaced elements (deterministic, covers the whole split); k<=0 or k>=len -> all."""
    if k <= 0 or k >= idx.size:
        return idx
    return idx[np.linspace(0, idx.size - 1, k).round().astype(np.int64)]


# ---------------------------------------------------------------------------------------------
# model plumbing

def forward_train(model, context, mask, horizon: int, po=None, pf=None):  # noqa: ANN001
    """Gradient-enabled twin of the inference call: (b, C) [+ covariates] -> (b, horizon, 9) raw
    quantiles of the target variate (output index 0)."""
    from timesfm3 import TimesFM3Torch

    fn = getattr(TimesFM3Torch.decode, "__wrapped__", None)
    if fn is None:
        raise RuntimeError("TimesFM3Torch.decode is no longer a no_grad-wrapped function; re-check the training path")
    out_len = math.ceil(horizon / model.output_patch_len) * model.output_patch_len  # as predict_batch does
    if pf is not None and pf.shape[-1] != context.shape[-1] + out_len:
        raise ValueError(f"known-future covariates must be {context.shape[-1]} + {out_len} wide, got {pf.shape[-1]}")
    out = fn(model, target=context[:, None, :], horizon=out_len, past_only_covariates=po, past_future_covariates=pf, mask=mask)
    return out[:, 0, :horizon, :]


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


def pinball_loss(pred, target, valid, scale=None, quantiles=QUANTILES):  # noqa: ANN001
    """Mean over valid (sample, step) pairs and quantiles of the pinball loss.
    pred (b, L, Q); target, valid, scale (b, L)."""
    import torch

    q = torch.as_tensor(quantiles, dtype=pred.dtype, device=pred.device)
    err = target[..., None] - pred
    if scale is not None:
        err = err / scale[..., None]
    loss = torch.maximum(q * err, (q - 1.0) * err).mean(dim=-1)
    w = valid.to(loss.dtype)
    return (loss * w).sum() / w.sum().clamp_min(1.0)


def lr_at(step: int, cfg: FinetuneConfig) -> float:
    """Linear warmup, then cosine decay to min_lr_frac * lr at max_steps. `step` is 1-based."""
    if step <= cfg.warmup_steps:
        return cfg.lr * step / max(cfg.warmup_steps, 1)
    t = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def _to_torch(b: dict[str, np.ndarray], device: str) -> dict:
    import torch

    return {k: torch.from_numpy(np.ascontiguousarray(b[k])).to(device) for k in ("context", "mask", "target", "valid", "scale", "po", "pf") if k in b}


def evaluate(model, samples: Samples, origins: np.ndarray, cfg: FinetuneConfig, device: str) -> dict[str, float]:  # noqa: ANN001
    """fp32, no-grad, quantiles clipped to [0, 1] and sorted: what `predict` returns. Scores the
    model against persistence on the same (origin, step) pairs, as `metrics.score_group` does: squared
    error of the decile mean vs the reference level, absolute error of the median vs the fan's median,
    pinball vs the fan."""
    import torch

    acc = {k: 0.0 for k in ("pb", "pb_fan", "ae", "ae_p", "se", "se_p", "cover", "n", "sb", "sb_p", "sn")}
    with torch.no_grad():
        for lo in range(0, origins.size, cfg.eval_batch):
            b = samples.batch(origins[lo:lo + cfg.eval_batch])
            t = _to_torch(b, device)
            raw = forward_train(model, t["context"], t["mask"], cfg.horizon, t.get("po"), t.get("pf"))[:, :MAX_STEPS].float().cpu().numpy()
            pred = np.sort(np.clip(raw.astype(np.float64), 0.0, 1.0), axis=-1)
            v = b["valid"]
            y, last = b["target"].astype(np.float64)[v], np.broadcast_to(b["persist"][:, None], v.shape)[v]
            p, f = pred[v], b["fan"][v]
            mean_fc = M.decile_mean(p)
            acc["pb"] += M.pinball(p, y).sum()
            acc["pb_fan"] += M.pinball(f, y).sum()
            acc["ae"] += np.abs(p[:, MEDIAN_IDX] - y).sum()
            acc["ae_p"] += np.abs(f[:, MEDIAN_IDX] - y).sum()
            acc["se"] += ((mean_fc - y) ** 2).sum()
            acc["se_p"] += ((last - y) ** 2).sum()
            acc["cover"] += ((y >= p[:, 0]) & (y <= p[:, -1])).sum()
            acc["n"] += y.size
            s = b["step_k"][v] == MAX_STEPS - 1                      # the settlement candle, as a probability
            o = (y[s] > 0.5).astype(np.float64)
            acc["sb"] += ((np.clip(mean_fc[s], M.EPS, 1 - M.EPS) - o) ** 2).sum()
            acc["sb_p"] += ((np.clip(last[s], M.EPS, 1 - M.EPS) - o) ** 2).sum()
            acc["sn"] += s.sum()
    n = max(acc["n"], 1.0)
    return {
        "pinball_c": 100 * acc["pb"] / n, "pinball_c_fan": 100 * acc["pb_fan"] / n, "skill_pinball": M.skill(acc["pb"], acc["pb_fan"]),
        "rmse_c": 100 * math.sqrt(acc["se"] / n), "rmse_c_persist": 100 * math.sqrt(acc["se_p"] / n), "skill_mse": M.skill(acc["se"], acc["se_p"]),
        "mae_c": 100 * acc["ae"] / n, "mae_c_persist": 100 * acc["ae_p"] / n, "skill_mae": M.skill(acc["ae"], acc["ae_p"]),
        "cover80": acc["cover"] / n, "settle_bss_persist": M.skill(acc["sb"], acc["sb_p"]), "n_origins": int(origins.size), "n_steps": int(acc["n"]),
    }


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
    frame = load_frame(settings.processed_dir, cfg.spec.freq)
    ok = make_origins(frame, cfg.spec, cfg.start, cfg.end).ok_origins
    if ok.height < 3:
        raise ValueError("not enough usable origins: download the contract candles and run `chudp frame`")
    i, n_steps = ok["i"].to_numpy(), ok["n_steps"].to_numpy().astype(np.int64)
    ts = frame["ts"]
    first_ts: datetime = ts[0]
    first_target, last_target = ts[int(i.min()) + 1], ts[int((i + n_steps).max())]
    bounds = split_bounds(first_target, last_target + timedelta(minutes=1), cfg)
    splits = assign_splits(i, n_steps, first_ts, bounds)
    for name in SPLITS:
        if splits[name].size == 0:
            raise ValueError(f"{name} split has no usable origins ({getattr(bounds, name)})")
    # the persistence fan sees the train split only; context-end rows up to the last train origin
    fan = B.persistence_fan(frame, lo=ts[int(splits["train"].min())], hi=ts[int(splits["train"].max())] + timedelta(minutes=1))
    if B.fan_cells(fan) < 60:
        log.warning("train split too short for a persistence fan (%d usable cells): skill_pinball is against a point mass", B.fan_cells(fan))
    return Samples(frame_arrays(frame, cfg.covariates), cfg, fan), splits, bounds, frame


def _fmt(m: dict[str, float]) -> str:
    keys = ("pinball_c", "pinball_c_fan", "skill_pinball", "rmse_c", "skill_mse", "skill_mae", "cover80", "settle_bss_persist")
    return "  ".join(f"{k}={m[k]:.4f}" for k in keys if k in m)


def run_finetune(settings: Settings, cfg: FinetuneConfig) -> Path:
    import torch

    torch.manual_seed(cfg.seed)
    run_id = cfg.run_id or f"{datetime.now(UTC).replace(tzinfo=None):%Y%m%d-%H%M%S}-finetune-c{cfg.context}-{cfg.covariates}"
    out = settings.finetune_dir / run_id
    out.mkdir(parents=True, exist_ok=True)

    samples, splits, bounds, frame = prepare_data(settings, cfg)
    ts = frame["ts"]
    split_info = bounds.as_dict() | {
        "origins": {s: {"n": int(splits[s].size), "first": ts[int(splits[s][0])].isoformat(), "last": ts[int(splits[s][-1])].isoformat()} for s in SPLITS}
    }
    (out / "splits.json").write_text(json.dumps(split_info, indent=1))
    (out / "config.json").write_text(json.dumps(asdict(cfg) | {"run_id": run_id, "git_sha": _git_sha(), "n_variates": samples.fa.n_variates}, indent=1, default=_jsonable))
    if samples.fan is not None:
        samples.fan.write_parquet(out / "baselines.parquet")
    for s in SPLITS:
        lo, hi = getattr(bounds, s)
        log.info("%-5s targets in [%s, %s): %d origins", s, lo, hi, splits[s].size)

    handle = load_forecaster(cfg.model_id, device=cfg.device, batch_size=cfg.eval_batch)
    model, device = handle.obj.model, handle.device
    model.eval()  # no dropout in this network; eval() keeps training identical to the inference path
    n_train = set_trainable(model, cfg.trainable)
    log.info("trainable parameters: %.1fM of %.1fM (%s); %d variates", n_train / 1e6, sum(p.numel() for p in model.parameters()) / 1e6, cfg.trainable, samples.fa.n_variates)

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
    best = {"step": 0, "pinball_c": base["val"]["pinball_c"]}
    best_dir, stale, run_loss, run_n, skipped, skipped_in_a_row, n_flat = out / "best", 0, 0.0, 0, 0, 0, 0
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
                n_flat += int(b["n_flat"])
                t = _to_torch(b, device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
                    pred = forward_train(model, t["context"], t["mask"], cfg.horizon, t.get("po"), t.get("pf"))
                loss = pinball_loss(pred[:, :MAX_STEPS].float(), t["target"], t["valid"], None if cfg.loss_scale == "none" else t["scale"])
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
                log.info("step %d/%d  train_loss=%.5f  grad_norm=%.3f  lr=%.2e  epoch=%d  %.1f origins/s",
                         step, cfg.max_steps, run_loss / max(run_n, 1), gnorm, lr, stream.epoch, rate)
                logf.write(json.dumps({"step": step, "train_loss": run_loss / max(run_n, 1), "grad_norm": gnorm, "lr": lr}) + "\n")
                run_loss, run_n = 0.0, 0
            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                val = evaluate(model, samples, val_idx, cfg, device)
                improved = val["pinball_c"] < best["pinball_c"]
                log.info("step %d val: %s%s", step, _fmt(val), "  *best*" if improved else "")
                history.append({"step": step, "val": val})
                logf.write(json.dumps(history[-1]) + "\n")
                logf.flush()
                if improved:
                    best, stale = {"step": step, "pinball_c": val["pinball_c"]}, 0
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

    summary = {"run_id": run_id, "best_step": best["step"], "steps_run": step, "skipped_steps": skipped, "flat_patches_jittered": n_flat,
               "splits": split_info, "zero_shot": base, "finetuned": tuned, "best_checkpoint": str(best_dir) if tuned else None, "history": history}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
    (out / "summary.txt").write_text(_summary_text(summary, cfg, bounds, out))
    log.info("fine-tune run written to %s", out)
    return out


def _summary_text(s: dict[str, Any], cfg: FinetuneConfig, bounds: SplitBounds, out: Path) -> str:
    cols = ("pinball_c", "pinball_c_fan", "skill_pinball", "rmse_c", "rmse_c_persist", "skill_mse", "skill_mae", "cover80", "settle_bss_persist")
    lines = [f"== Fine-tune {s['run_id']} ==",
             f"context {cfg.context}, covariates '{cfg.covariates}', trainable {cfg.trainable}, loss scale {cfg.loss_scale}",
             f"steps run {s['steps_run']} ({s['skipped_steps']} skipped for non-finite gradients), best validation step {s['best_step']}", "",
             "== Splits (target ranges, UTC) =="]
    for name in SPLITS:
        lo, hi = getattr(bounds, name)
        o = s["splits"]["origins"][name]
        lines.append(f"{name:>6}  [{lo:%Y-%m-%d %H:%M}, {hi:%Y-%m-%d %H:%M})  {o['n']:>8} origins")
    lines += [f"embargo {bounds.embargo_bars} bars between splits", "",
              "== Scores over all steps to expiry, cents (skill > 0 beats persistence: the last mid, 0.50 at the open, and its train-split fan) ==",
              f"{'model':>10} {'split':>6} " + " ".join(f"{c:>18}" for c in cols)]
    for label, res in (("zero-shot", s["zero_shot"]), ("finetuned", s["finetuned"])):
        for split in ("val", "test"):
            if res:
                lines.append(f"{label:>10} {split:>6} " + " ".join(f"{res[split].get(c, float('nan')):>18.4f}" for c in cols))
    test_lo, test_hi = bounds.test
    common = f"--config configs/contract_full.toml --context {cfg.context} --covariates {cfg.covariates} --fan-from {out / 'baselines.parquet'} " \
             f"--start {test_lo:%Y-%m-%d} --end {(test_hi - timedelta(seconds=1)):%Y-%m-%d}"
    lines.append("")
    if s["finetuned"]:
        lines += ["Backtest the held-out test dates (validation and train dates are contaminated for this checkpoint):",
                  f"  chudp backtest-contract {common} --model-id {s['best_checkpoint']}",
                  f"  chudp backtest-contract {common}    # zero-shot on the same dates"]
    else:
        lines.append("Validation never improved on the zero-shot model, so no checkpoint was promoted (see last/ for the final weights).")
    return "\n".join(lines) + "\n"
