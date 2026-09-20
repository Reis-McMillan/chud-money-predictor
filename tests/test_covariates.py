"""Model input arrays: shapes, calendar continuation, no lookahead, patch-level non-constancy."""

from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from conftest import CONTRACT_START, T_DROPPED

from chud_predictor import covariates as CV
from chud_predictor.features import h_eff, norm_cdf
from chud_predictor.origins import ContractSpec, make_origins

C = 64
MIN = timedelta(minutes=1)


@pytest.fixture(scope="module")
def world(contract_world):
    frame = contract_world["frame"]
    ok = make_origins(frame, ContractSpec(context=C, max_ctx_gap_frac=0.1)).ok_origins
    return frame, CV.frame_arrays(frame, "full"), ok


def _pick(ok, t0, m):
    return ok.filter((pl.col("t0") == t0) & (pl.col("m") == m))["i"].to_numpy()


def test_shapes_and_presets(world):
    frame, fa, ok = world
    i = ok["i"].to_numpy()[:40]
    b = CV.build_arrays(fa, i, C)
    assert b["targets"].shape == (40, C) and b["targets"].dtype == np.float32
    assert b["po"].shape == (40, 17, C) and b["pf"].shape == (40, 7, C + 64)
    assert b["tgt"].shape == b["valid"].shape == b["step_k"].shape == (40, 15)
    assert np.isfinite(b["po"]).all() and np.isfinite(b["pf"]).all() and not b["mask"].any()
    assert fa.n_variates == 25
    for name, (po, pf) in CV.PRESETS.items():
        assert 1 + len(po) + len(pf) <= CV.MAX_VARIATES, name
    none = CV.build_arrays(CV.frame_arrays(frame, "none"), i, C)
    assert none["po"] is None and none["pf"] is None and np.array_equal(none["targets"], b["targets"])
    assert CV.build_arrays(CV.frame_arrays(frame, "brti_only"), i, C)["po"].shape == (40, 8, C)
    with pytest.raises(ValueError, match="unknown covariate preset"):
        CV.frame_arrays(frame, "everything")


def test_targets_and_valid_mask_per_step(world):
    frame, fa, ok = world
    t0 = CONTRACT_START.replace(hour=5)
    mid = frame["mid_close"].to_numpy()
    for m in (0, 1, 7, 14):
        i = _pick(ok, t0, m)
        b = CV.build_arrays(fa, i, C)
        n_steps = 15 - m
        assert b["valid"][0, :n_steps].all() and not b["valid"][0, n_steps:].any()      # never past expiry
        assert np.allclose(b["tgt"][0, :n_steps], mid[i[0] + 1: i[0] + 1 + n_steps]) and np.isnan(b["tgt"][0, n_steps:]).all()
        assert b["step_k"][0, :n_steps].tolist() == list(range(m, 15))
        assert b["targets"][0, -1] == pytest.approx(mid[i[0]], abs=1e-4)
    # a missing candle inside the horizon is not scored, its neighbours are
    b = CV.build_arrays(fa, _pick(ok, T_DROPPED, 1), C)
    assert b["valid"][0].tolist() == [True, True, False, False] + [True] * 10 + [False]


def test_known_future_covariates_continue_the_clock(world):
    frame, fa, ok = world
    t0 = CONTRACT_START.replace(hour=5)
    i = _pick(ok, t0, 4)                                  # context ends at k = 3
    b = CV.build_arrays(fa, i, C)
    pf = dict(zip(fa.pf_names, b["pf"][0], strict=True))
    steps = np.arange(1, 65)
    k_fut = (3 + steps) % 15
    assert np.allclose(pf["k_frac"][C:], k_fut / 14, atol=2e-5)
    assert np.allclose(pf["k_sin"][C:], np.sin(2 * np.pi * k_fut / 15), atol=2e-5)
    assert np.allclose(pf["sqrt_ttl"][C:], np.sqrt(h_eff(k_fut) / (13 + 1 / 3)), atol=2e-5)
    mod = (frame["minute_of_day"][int(i[0])] + steps) % 1440
    assert np.allclose(pf["tod_cos"][C:], np.cos(2 * np.pi * mod / 1440), atol=2e-5)
    assert np.allclose(pf["k_frac"][:C], frame["k"].to_numpy()[i[0] - C + 1: i[0] + 1] / 14, atol=2e-5)   # realised part
    # fair path: realised rw_p in the context, frozen-BRTI fair value to expiry, then held
    assert np.allclose(pf["rw_fair_path"][:C], frame["rw_p"].to_numpy()[i[0] - C + 1: i[0] + 1], atol=2e-5)
    r = frame.row(int(i[0]), named=True)
    z = np.log(r["brti_close"] / r["strike"]) / (r["sigma_1m"] * np.sqrt(h_eff(k_fut[:11])))
    assert np.allclose(pf["rw_fair_path"][C:C + 11], norm_cdf(np.clip(z, -8, 8)), atol=2e-5)
    assert np.allclose(pf["rw_fair_path"][C + 11:], pf["rw_fair_path"][C + 10], atol=2e-5)


def test_no_lookahead_by_poisoning_the_future(world):
    """Every input array must be identical when all data after the context end is destroyed."""
    frame, fa, ok = world
    data_cols = [c for c in frame.columns if c not in ("ts", "idx", "t0", "k", "date", "hour", "minute_of_day")]
    for m in (0, 1, 8, 14):
        i = _pick(ok, CONTRACT_START.replace(hour=11), m)
        clean = CV.build_arrays(fa, i, C)
        poisoned = frame.with_columns([pl.when(pl.col("idx") > int(i[0])).then(None).otherwise(pl.col(c)).alias(c) for c in data_cols])
        got = CV.build_arrays(CV.frame_arrays(poisoned, "full"), i, C)
        for key in ("targets", "po", "pf", "mask"):
            assert np.array_equal(clean[key], got[key]), (m, key)
        assert not got["valid"].any()                     # while the targets really were in the poisoned part


def test_interpolation_mirrors_timesfm():
    x = np.array([[np.nan, np.nan, 1.0, np.nan, 3.0, np.nan], [np.nan] * 6, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    cov = CV.interpolate_rows(x)
    assert cov[0].tolist() == [1.0, 1.0, 1.0, 2.0, 3.0, 3.0] and cov[1].tolist() == [0.0] * 6 and cov[2].tolist() == x[2].tolist()
    tgt = CV.interpolate_rows(x, keep_leading_nan=True)
    assert np.isnan(tgt[0, :2]).all() and tgt[0, 2:].tolist() == [1.0, 2.0, 3.0, 3.0] and np.isnan(tgt[1]).all()


def test_flat_patches_are_jittered_and_nothing_else(world):
    frame, fa, ok = world
    x = np.zeros((2, 3, 96))
    x[0, 0] = 0.01                                  # flat everywhere: 3 blocks
    x[0, 1] = np.linspace(0, 1, 96)                 # never flat
    x[1, 2, 32:64] = 5.0                            # varies across blocks but each block is flat: 3 blocks
    x[1, 0, :32] = np.nan                           # a target's leading gap is left alone
    out, n = CV.sanitize_variates(x)
    assert n == 3 + 3 + 3 + 2 + 3                    # x[0,0] + x[0,2] + x[1,1] + the two non-NaN blocks of x[1,0] + x[1,2]
    assert np.array_equal(out[0, 1], x[0, 1].astype(np.float32)) and out.dtype == np.float32 and np.isnan(out[1, 0, :32]).all()
    blocks = out[0, 0].reshape(3, 32)
    assert (blocks.max(axis=1) > blocks.min(axis=1)).all() and np.abs(out[0, 0] - 0.01).max() < 1e-4   # far below a 0.001 tick
    assert np.array_equal(CV.sanitize_variates(x)[0][0], out[0])                                        # deterministic
    with pytest.raises(ValueError, match="multiple of the input patch"):
        CV.sanitize_variates(np.zeros((1, 40)))
    # on real batches no variate is ever flat inside an input patch, target included
    b = CV.build_arrays(fa, ok["i"].to_numpy()[::7], C)
    for key in ("targets", "po", "pf"):
        arr = b[key].astype(np.float64)
        assert np.isfinite(arr).all(), key            # the sample spans missing candles: they must be interpolated
        blk = arr.reshape(*arr.shape[:-1], arr.shape[-1] // 32, 32)
        assert (blk.max(axis=-1) > blk.min(axis=-1)).all(), key
