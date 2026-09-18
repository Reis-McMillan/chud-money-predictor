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
    obj: Any  # anything with predict_batch(contexts, horizon, return_quantiles=..., ...)


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
    contexts: np.ndarray | list[np.ndarray],
    horizon: int = 64,
    *,
    quantiles: bool = True,
    symmetric: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """-> (median (n, horizon), quantiles (n, horizon, 9)) in model space, input order preserved."""
    ctx_list = [np.asarray(c, dtype=np.float32) for c in contexts]
    n = len(ctx_list)
    if n == 0:
        return np.zeros((0, horizon)), np.zeros((0, horizon, N_QUANTILES))
    outs = list(
        handle.obj.predict_batch(
            ctx_list,
            horizon=horizon,
            return_quantiles=True,
            use_symmetric_averaging=symmetric,
            make_positive=False,
        )
    )
    if len(outs) != n:
        raise RuntimeError(f"predict_batch returned {len(outs)} outputs for {n} contexts")
    med = np.stack([np.asarray(o.forecast, dtype=np.float64) for o in outs])
    q = np.stack([np.asarray(o.quantiles, dtype=np.float64) for o in outs])
    if q.shape[-1] == N_QUANTILES + 1:  # (mean + 9 quantiles) layout of older checkpoints
        q = q[..., 1:]
    if q.shape != (n, horizon, N_QUANTILES) or med.shape != (n, horizon):
        raise RuntimeError(f"unexpected output shapes median {med.shape}, quantiles {q.shape}")
    return med, q
