"""TimesFM 3.0 loading and batched prediction. torch is imported lazily so the data layer and
the metrics never need it."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "google/timesfm-3.0-pytorch"
N_QUANTILES = 9
OUTPUT_PATCH = 64


def pick_device(explicit: str | None = None) -> str:
    """CHUDP_DEVICE / --device > cuda (ROCm also reports as cuda) > mps > cpu."""
    choice = explicit or os.environ.get("CHUDP_DEVICE")
    if choice and choice != "auto":
        if choice == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return choice
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_summary(device: str) -> str:
    import torch

    if device == "cuda" and torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)}, torch {torch.__version__})"
    return f"{device} (torch {torch.__version__})"


@dataclass
class ForecasterHandle:
    backend: str
    device: str
    model_id: str
    batch_size: int
    obj: Any  # anything with TimesFM3Forecaster.predict_batch's signature


def load_forecaster(
    model_id: str = DEFAULT_MODEL,
    *,
    device: str | None = None,
    batch_size: int = 32,
    cache_dir: str | None = None,
    token: str | None = None,
    evaluator: bool = False,
) -> ForecasterHandle:
    dev = pick_device(device)
    from timesfm3 import ModelConfig, TimesFM3Evaluator, TimesFM3Forecaster

    cfg = ModelConfig(
        checkpoint_path=model_id,
        per_core_batch_size=batch_size,
        device=dev,
        cache_dir=cache_dir,
        token=token or os.environ.get("HF_TOKEN") or None,
    )
    log.info("loading %s on %s (batch %d)", model_id, device_summary(dev), batch_size)
    obj = TimesFM3Evaluator(cfg) if evaluator else TimesFM3Forecaster(cfg)
    return ForecasterHandle(backend="timesfm3", device=dev, model_id=model_id, batch_size=batch_size, obj=obj)


def predict(
    handle: ForecasterHandle,
    targets: np.ndarray | list[np.ndarray],
    horizon: int = OUTPUT_PATCH,
    *,
    past_only: np.ndarray | None = None,
    past_future: np.ndarray | None = None,
    symmetric: bool = False,
    clip: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast of the target variate: (median (n, horizon), quantiles (n, horizon, 9)), input order
    preserved.

    targets      (n, C) or a list of (C,) arrays; a leading NaN run is masked by TimesFM
    past_only    (n, K, C) covariates known up to the origin, or None
    past_future  (n, W, C + H) covariates known through the horizon, or None, with H the horizon
                 rounded up to the 64-step output patch. TimesFM re-derives the horizon from this
                 width, so a wrong width silently shifts the forecast: it is validated here.
    clip         bounds of the target (e.g. (0, 1) for a price); clipping is monotone, so clipped
                 quantiles are still quantiles, and they are re-sorted afterwards
    """
    tgt_list = [np.asarray(t, dtype=np.float32) for t in targets]
    n = len(tgt_list)
    if n == 0:
        return np.zeros((0, horizon)), np.zeros((0, horizon, N_QUANTILES))
    width = tgt_list[0].shape[-1]
    h_model = -(-horizon // OUTPUT_PATCH) * OUTPUT_PATCH
    kwargs: dict[str, Any] = {}
    if past_only is not None:
        if past_only.shape[0] != n or past_only.shape[-1] != width:
            raise ValueError(f"past_only must be (n, K, {width}), got {past_only.shape}")
        kwargs["past_only_covariates"] = [np.asarray(p, dtype=np.float32) for p in past_only]
    if past_future is not None:
        if past_future.shape[0] != n or past_future.shape[-1] != width + h_model:
            raise ValueError(f"past_future must be (n, W, {width} + {h_model}), got {past_future.shape}")
        kwargs["past_future_covariates"] = [np.asarray(p, dtype=np.float32) for p in past_future]
    outs = list(handle.obj.predict_batch(
        tgt_list, horizon=h_model if past_future is not None else horizon,
        return_quantiles=True, use_symmetric_averaging=symmetric, make_positive=False, **kwargs,
    ))
    if len(outs) != n:
        raise RuntimeError(f"predict_batch returned {len(outs)} outputs for {n} contexts")
    med = np.stack([np.asarray(o.forecast, dtype=np.float64) for o in outs])[:, :horizon]
    q = np.stack([np.asarray(o.quantiles, dtype=np.float64) for o in outs])[:, :horizon]
    if q.shape[-1] == N_QUANTILES + 1:  # (mean + 9 quantiles) layout of older checkpoints
        q = q[..., 1:]
    if q.shape != (n, horizon, N_QUANTILES) or med.shape != (n, horizon):
        raise RuntimeError(f"unexpected output shapes median {med.shape}, quantiles {q.shape}")
    if clip is not None:
        med = np.clip(med, *clip)
        q = np.sort(np.clip(q, *clip), axis=-1)
    return med, q
