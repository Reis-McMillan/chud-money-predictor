"""Loads the real TimesFM 3.0 weights (~1.3 GB download on first run). Opt in with CHUDP_SLOW=1."""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not os.environ.get("CHUDP_SLOW"), reason="set CHUDP_SLOW=1 to run the real model")
def test_predict_shapes_and_quantile_order():
    from chud_predictor.model import load_forecaster, predict

    handle = load_forecaster(device="cpu", batch_size=2)
    t = np.linspace(0, 12, 512)
    ctxs = [np.log(100_000 + 500 * np.sin(t) + 20 * t).astype(np.float32), np.log(90_000 + 300 * np.cos(t)).astype(np.float32)]
    med, q = predict(handle, ctxs, 64)
    assert med.shape == (2, 64) and q.shape == (2, 64, 9)
    assert np.isfinite(med).all() and np.isfinite(q).all()
    assert np.all(np.diff(q, axis=-1) >= 0)
    assert np.allclose(q[:, :, 4], med)
