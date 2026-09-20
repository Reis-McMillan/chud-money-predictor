"""model.predict: covariate plumbing, order, validation, clipping. Real weights only in the slow test."""

import os
from dataclasses import dataclass

import numpy as np
import pytest

from chud_predictor.model import ForecasterHandle, predict


@dataclass
class _Out:
    ts_id: object
    forecast: np.ndarray
    quantiles: np.ndarray


class StubMultiForecaster:
    """median[h] = last target value + step * h (h = 1..H); deciles spread by `width` around it.
    Records what it was handed so the tests can inspect the covariate plumbing."""

    def __init__(self, step: float = 1e-3, width: float = 0.01) -> None:
        self.step, self.width, self.calls = step, width, []

    def predict_batch(self, contexts, horizon, past_only_covariates=None, past_future_covariates=None, **kw):
        self.calls.append({"n": len(contexts), "horizon": horizon, "po": past_only_covariates, "pf": past_future_covariates, "kw": kw})
        for r, c in enumerate(contexts):
            assert c.ndim == 1 and c.dtype == np.float32
            if past_only_covariates is not None:
                assert past_only_covariates[r].shape[-1] == c.shape[-1]
            if past_future_covariates is not None:
                assert past_future_covariates[r].shape[-1] == c.shape[-1] + horizon
            med = float(c[-1]) + self.step * np.arange(1, horizon + 1)
            yield _Out(None, med, med[:, None] + (np.arange(9) - 4)[None, :] * self.width)


def _handle(stub) -> ForecasterHandle:
    return ForecasterHandle(backend="stub", device="cpu", model_id="stub", batch_size=4, obj=stub)


def test_covariates_reach_the_model_in_order():
    stub = StubMultiForecaster()
    tg = np.linspace(0.1, 0.9, 5)[:, None] + np.zeros((5, 64), dtype=np.float32)
    po, pf = np.random.default_rng(0).normal(size=(5, 3, 64)), np.random.default_rng(1).normal(size=(5, 2, 128))
    med, q = predict(_handle(stub), tg, 64, past_only=po, past_future=pf)
    assert med.shape == (5, 64) and q.shape == (5, 64, 9)
    assert np.allclose(med[:, 0], tg[:, -1] + 1e-3, atol=1e-6) and np.allclose(med[:, 14], tg[:, -1] + 15e-3, atol=1e-6)
    call = stub.calls[0]
    assert call["n"] == 5 and call["horizon"] == 64 and len(call["po"]) == 5 and call["po"][2].shape == (3, 64) and call["pf"][4].shape == (2, 128)
    assert np.allclose(call["po"][2], po[2].astype(np.float32)) and call["kw"]["make_positive"] is False


def test_known_future_width_is_validated():
    h = _handle(StubMultiForecaster())
    tg = np.full((2, 64), 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match=r"past_future must be"):
        predict(h, tg, 64, past_future=np.zeros((2, 1, 64 + 15)))      # a 15-step future would silently become the horizon
    with pytest.raises(ValueError, match=r"past_only must be"):
        predict(h, tg, 64, past_only=np.zeros((2, 1, 63)))
    with pytest.raises(ValueError, match=r"past_only must be"):
        predict(h, tg, 64, past_only=np.zeros((3, 1, 64)))


def test_clipping_keeps_sorted_quantiles_and_univariate_still_works():
    stub = StubMultiForecaster(step=0.02, width=0.05)
    tg = np.full((1, 32), 0.97, dtype=np.float32)
    med, q = predict(_handle(stub), tg, 64, clip=(0.0, 1.0))
    assert med.max() <= 1.0 and q.max() <= 1.0 and q.min() >= 0.0 and np.all(np.diff(q, axis=-1) >= 0)
    assert stub.calls[0]["po"] is None and stub.calls[0]["pf"] is None
    raw, _ = predict(_handle(stub), [tg[0]], 15)
    assert raw.shape == (1, 15) and raw[0, -1] == pytest.approx(0.97 + 15 * 0.02, abs=1e-6)
    empty_med, empty_q = predict(_handle(stub), np.zeros((0, 32)), 64)
    assert empty_med.shape == (0, 64) and empty_q.shape == (0, 64, 9)


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("CHUDP_SLOW"), reason="set CHUDP_SLOW=1 to run the real model")
def test_real_model_with_25_variates_is_covariate_order_invariant():
    from chud_predictor.model import load_forecaster

    handle = load_forecaster(device="cpu", batch_size=2)
    rng = np.random.default_rng(0)
    n, C = 2, 256
    tg = np.clip(0.5 + np.cumsum(rng.normal(0, 0.02, (n, C)), axis=1), 0.01, 0.99).astype(np.float32)
    po = rng.normal(size=(n, 17, C)).cumsum(axis=-1).astype(np.float32)
    pf = np.stack([np.sin(np.arange(C + 64) * 2 * np.pi / p) for p in (15, 15.5, 30, 60, 720, 1440, 90)])[None].repeat(n, 0).astype(np.float32)
    med, q = predict(handle, tg, 64, past_only=po, past_future=pf, clip=(0.0, 1.0))
    assert med.shape == (n, 64) and q.shape == (n, 64, 9) and np.isfinite(q).all() and np.all(np.diff(q, axis=-1) >= 0)
    assert np.allclose(q[:, :, 4], med)
    med2, _ = predict(handle, tg, 64, past_only=po[:, ::-1].copy(), past_future=pf[:, ::-1].copy(), clip=(0.0, 1.0))
    assert np.allclose(med, med2, atol=1e-4)            # variate attention has no variate position encoding
    med3, _ = predict(handle, tg, 64, clip=(0.0, 1.0))
    assert not np.allclose(med, med3, atol=1e-4)        # and the covariates are actually used
