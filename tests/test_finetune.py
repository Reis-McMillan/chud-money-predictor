"""Fine-tuning: split hygiene, batch construction, the loss, and an end-to-end run on a tiny
randomly initialised TimesFM (no weights download)."""

import json
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from conftest import synthetic_ticks

from chud_predictor import finetune as F
from chud_predictor.resample import build
from chud_predictor.settings import Settings

torch = pytest.importorskip("torch")
pytest.importorskip("timesfm3")

# three synthetic days: train = 09-18, validation = 09-19, test = 09-20 (fractions cannot resolve midnights on so little data)
CFG = dict(context=256, horizon=64, loss_horizon=15, vol_lookback=40, val_start=date(2025, 9, 19), test_start=date(2025, 9, 20), embargo_bars=60,
           device="cpu", max_steps=4, batch_size=8, micro_batch=4, eval_every=2, eval_samples=24, eval_batch=8,
           warmup_steps=1, lr=1e-3, log_every=1, patience=0)


@pytest.fixture(scope="module")
def settings(tmp_path_factory) -> Settings:
    root = tmp_path_factory.mktemp("ft")
    raw = root / "data" / "raw" / "brti"
    raw.mkdir(parents=True)
    for i in range(3):
        synthetic_ticks(datetime(2025, 9, 18 + i), 86_400, seed=40 + i).write_parquet(raw / f"date=2025-09-{18 + i}.parquet")
    build(raw, root / "data" / "processed", "1m")
    return Settings(data_dir=root / "data")


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory) -> str:
    from timesfm3 import TimesFM3Torch

    torch.manual_seed(0)
    tf = dict(model_dims=16, hidden_dims=16, num_heads=2, attention_norm="rms", feedforward_norm="rms", qk_norm="rms",
              use_bias=False, use_rope_seq=True, use_rope_var=False, ff_activation="relu", deterministic=True)
    model = TimesFM3Torch(
        residual_block_config=dict(hidden_dims=16, output_dims=16, use_bias=False, activation="relu"),
        transformer_config=dict(num_layers=2, transformer=tf),
    )
    out = tmp_path_factory.mktemp("tiny") / "model"
    model.save_pretrained(out)
    return str(out)


def test_splits_are_chronological_disjoint_and_embargoed(settings):
    cfg = F.FinetuneConfig(**CFG)
    _, splits, bounds, feats = F.prepare_data(settings, cfg)
    ts = feats["ts"]
    first_target = {s: ts[int(splits[s].min()) + 1] for s in F.SPLITS}
    last_target = {s: ts[int(splits[s].max()) + cfg.horizon] for s in F.SPLITS}
    emb = timedelta(minutes=cfg.embargo_bars)
    assert last_target["train"] + emb < first_target["val"]
    assert last_target["val"] + emb < first_target["test"]
    for s in F.SPLITS:  # every target bar of every origin is inside its own split's range
        lo, hi = getattr(bounds, s)
        assert lo <= first_target[s] and last_target[s] < hi
    assert bounds.val[0].time() == bounds.test[0].time() == datetime.min.time()  # UTC midnights -> backtest --start/--end
    assert last_target["test"] == ts[-1]  # test runs to the end of the data
    assert not set(splits["train"]) & set(splits["val"]) and not set(splits["val"]) & set(splits["test"])


def test_fractions_land_on_midnights():
    cfg = F.FinetuneConfig(context=256, val_frac=0.15, test_frac=0.15)
    b = F.split_bounds(datetime(2025, 9, 20, 4, 16), datetime(2026, 8, 11, 11), cfg)
    assert b.val[0] == datetime(2026, 5, 6) and b.test[0] == datetime(2026, 6, 24)   # 70 / 15 / 15 of 325 days
    assert b.train == (datetime(2025, 9, 20, 4, 16), datetime(2026, 5, 5)) and b.val[1] == datetime(2026, 6, 23)


def test_explicit_split_dates_override_fractions():
    cfg = F.FinetuneConfig(**CFG)
    b = F.split_bounds(datetime(2025, 9, 18, 5), datetime(2025, 9, 21), cfg)
    assert b.val == (datetime(2025, 9, 19), datetime(2025, 9, 19, 23)) and b.test[0] == datetime(2025, 9, 20)
    with pytest.raises(ValueError, match="empty train"):  # a one-day embargo swallows the 19h train day
        F.split_bounds(datetime(2025, 9, 18, 5), datetime(2025, 9, 21), F.FinetuneConfig(**CFG | {"embargo_bars": 1440}))


def test_valid_origins_respect_gaps_and_horizon(settings):
    cfg = F.FinetuneConfig(**CFG)
    _, _, _, feats = F.prepare_data(settings, cfg)
    gap = feats.with_columns(pl.when(pl.col("idx").is_between(1000, 1099)).then(None).otherwise(pl.col("y")).alias("y"))
    gap = gap.with_columns(pl.col("y").is_null().cast(pl.Int64).cum_sum().alias("_null_cum"))
    o = F.valid_origins(gap, cfg)
    assert o.min() >= cfg.context - 1 and o.max() + cfg.horizon <= feats.height - 1
    assert not np.isin(np.arange(1000, 1100), o).any()            # a gap bar is never the last context bar
    assert 1200 not in o                       # interior gap, but 100 bars > 1% of the context
    assert 1354 not in o and 1355 in o         # context may not start inside the gap (NaN gradients), see valid_origins
    assert 999 - cfg.loss_horizon in o and 999 not in o           # 999: every scored target is a gap


def test_batch_matches_inference_preprocessing(settings):
    cfg = F.FinetuneConfig(**CFG)
    samples, splits, _, _ = F.prepare_data(settings, cfg)
    o = splits["train"][[0, 300, 600]]  # far enough apart that the planted NaNs touch one row each
    # valid_origins never yields a context with leading NaNs, but batch() must still mirror predict_batch for them
    samples.y[o[0] - cfg.context + 1: o[0] - cfg.context + 4] = np.nan   # leading NaNs -> masked
    samples.y[o[1] - 10] = np.nan                                        # interior NaN -> interpolated
    samples.y[o[2] + 2] = np.nan                                         # missing target -> not scored
    b = samples.batch(o)
    assert b["mask"][0, :3].all() and not b["mask"][0, 3:].any() and not b["mask"][1:].any()
    assert np.isfinite(b["context"]).all() and b["context"].dtype == np.float32
    assert b["context"][1, -11] == pytest.approx((samples.y[o[1] - 11] + samples.y[o[1] - 9]) / 2, rel=1e-6)
    assert not b["valid"][2, 1] and b["valid"][2, 0] and b["valid"][:2].all()
    assert b["target"][0, 0] == pytest.approx(samples.y[o[0] + 1], rel=1e-6)
    assert np.allclose(b["scale"][0], samples.sigma[o[0]] * np.sqrt(np.arange(64) + 1 / 3), rtol=1e-5)


def test_pinball_loss_matches_definition():
    pred = torch.tensor([[[0.0] * 9, [1.0] * 9]])                # (1, 2, 9)
    target, scale = torch.tensor([[2.0, 0.0]]), torch.tensor([[2.0, 1.0]])
    both = F.pinball_loss(pred, target, scale, torch.tensor([[True, True]]))
    # step 0: under-prediction by 1 scale unit -> mean(q); step 1: over-prediction by 1 -> mean(1-q); both 0.5
    assert float(both) == pytest.approx(0.5)
    only_first = F.pinball_loss(pred, target * 3, scale, torch.tensor([[True, False]]))
    assert float(only_first) == pytest.approx(1.5)


def test_lr_schedule_and_trainable_spec():
    cfg = F.FinetuneConfig(**CFG | {"warmup_steps": 10, "max_steps": 110, "lr": 1.0})
    assert F.lr_at(5, cfg) == pytest.approx(0.5) and F.lr_at(10, cfg) == pytest.approx(1.0)
    assert F.lr_at(110, cfg) == pytest.approx(cfg.min_lr_frac) and F.lr_at(60, cfg) == pytest.approx((1 + cfg.min_lr_frac) / 2)
    assert F.parse_trainable("all") is None and F.parse_trainable("head") == 0 and F.parse_trainable("last:3") == 3
    with pytest.raises(ValueError):
        F.FinetuneConfig(trainable="some")


def test_training_path_is_the_inference_path(tiny_model_dir):
    from chud_predictor.model import load_forecaster, predict

    handle = load_forecaster(tiny_model_dir, device="cpu", batch_size=4)
    rng = np.random.default_rng(0)
    ctx = (11.5 + np.cumsum(rng.normal(0, 3e-4, (3, 256)), axis=1)).astype(np.float32)
    _, q = predict(handle, ctx, 64)
    out = F.forward_train(handle.obj.model, torch.from_numpy(ctx), torch.zeros(3, 256, dtype=torch.bool), 64)
    assert out.requires_grad and out.shape == (3, 64, 9)
    assert np.allclose(np.sort(out.detach().numpy(), axis=-1), q, atol=1e-6)


def test_set_trainable_freezes_the_rest(tiny_model_dir):
    from chud_predictor.model import load_forecaster

    model = load_forecaster(tiny_model_dir, device="cpu").obj.model
    total = sum(p.numel() for p in model.parameters())
    head = F.set_trainable(model, "head")
    assert head == sum(p.numel() for p in model.output_head.parameters())
    assert head < F.set_trainable(model, "last:1") < F.set_trainable(model, "all") == total


def test_run_finetune_end_to_end(settings, tiny_model_dir):
    from chud_predictor.model import load_forecaster, predict

    cfg = F.FinetuneConfig(**CFG | {"model_id": tiny_model_dir, "run_id": "t"})
    before = {k: v.clone() for k, v in load_forecaster(tiny_model_dir, device="cpu").obj.model.state_dict().items()}
    out = F.run_finetune(settings, cfg)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["steps_run"] == 4 and summary["skipped_steps"] == 0 and len(summary["history"]) == 3  # step 0 + two evaluations
    for split in ("val", "test"):
        m = summary["zero_shot"][split]
        assert np.isfinite([m["pinball"], m["pinball_rw"], m["mase_h"]]).all() and 0 <= m["cover80"] <= 1
    assert {"train", "val", "test"} <= set(summary["splits"]["origins"])
    text = (out / "summary.txt").read_text()
    assert "== Splits" in text and all(s in text for s in ("train", "val", "test"))

    last = load_forecaster(str(out / "last"), device="cpu")                        # loadable by the backtest
    after = last.obj.model.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)               # weights moved
    med, q = predict(last, [np.full(256, 11.5, dtype=np.float32) + np.linspace(0, 1e-3, 256, dtype=np.float32)], 64)
    assert np.isfinite(med).all() and np.isfinite(q).all()
    if summary["best_step"] > 0:
        assert (out / "best" / "model.safetensors").exists() and "--model-id" in text and summary["finetuned"]["test"]["n_origins"] > 0
    else:
        assert "no checkpoint was promoted" in text
    lines = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    assert sum("train_pinball" in x for x in lines) == 4
