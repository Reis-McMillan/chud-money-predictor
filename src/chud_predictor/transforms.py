"""Target transforms. `exp` is monotone, so quantiles invert pointwise and stay valid quantiles."""

from __future__ import annotations

import numpy as np

TRANSFORMS = ("log", "none")


def forward(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "log":
        return np.log(x)
    if kind == "none":
        return np.asarray(x, dtype=np.float64)
    raise ValueError(f"unknown transform {kind!r}")


def inverse(y: np.ndarray, kind: str) -> np.ndarray:
    if kind == "log":
        return np.exp(y)
    if kind == "none":
        return np.asarray(y, dtype=np.float64)
    raise ValueError(f"unknown transform {kind!r}")
