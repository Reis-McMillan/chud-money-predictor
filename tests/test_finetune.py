"""Fine-tuning on the contract price: split hygiene, batch construction, the loss, the training path
and an end-to-end run on a tiny randomly initialised TimesFM (no weights download)."""

import json
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from conftest import synthetic_candles, synthetic_ticks, write_raw

from chud_predictor import finetune as F
from chud_predictor.features import build_frame
from chud_predictor.resample import build
from chud_predictor.settings import Settings

torch = pytest.importorskip("torch")
pytest.importorskip("timesfm3")

D1 = datetime(2025, 12, 20)
# three synthetic days: train = day 1, validation = day 2, test = day 3 (fractions cannot resolve midnights on so little data)
CFG = dict(context=64, start=None, val_start=date(2025, 12, 21), test_start=date(2025, 12, 22), embargo_bars=60, device="cpu",
           max_steps=4, batch_size=8, micro_batch=4, eval_every=2, eval_samples=30, eval_batch=8, warmup_steps=1, lr=1e-3, log_every=1, patience=0)


@pytest.fixture(scope="module")
def settings(tmp_path_factory) -> Settings:
    s = Settings(data_dir=tmp_path_factory.mktemp("ft") / "data")
    ticks = synthetic_ticks(D1, 3 * 86_400, seed=40, sigma=1.0)
    write_raw(ticks, s.raw_dir)
    write_raw(synthetic_candles(ticks, seed=41), s.contracts_raw_dir)
    build(s.raw_dir, s.processed_dir, "1m")
    build_frame(s.processed_dir, s.contracts_raw_dir, vol_lookback=60)
    return s


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
    samples, splits, bounds, frame = F.prepare_data(settings, cfg)
    ts, k = frame["ts"], frame["k"].to_numpy()
    n_steps = {s: 15 - (k[splits[s]] + 1) % 15 for s in F.SPLITS}
    first_target = {s: ts[int(splits[s].min()) + 1] for s in F.SPLITS}
    last_target = {s: ts[int((splits[s] + n_steps[s]).max())] for s in F.SPLITS}
    emb = timedelta(minutes=cfg.embargo_bars)
    assert last_target["train"] + emb < first_target["val"] and last_target["val"] + emb < first_target["test"]
    for s in F.SPLITS:  # every target row of every origin is inside its own split's range
        lo, hi = getattr(bounds, s)
        assert lo <= first_target[s] and last_target[s] < hi
    assert bounds.val[0] == datetime(2025, 12, 21) and bounds.test[0] == datetime(2025, 12, 22)   # UTC midnights -> backtest --start/--end
    assert last_target["test"] == ts[-1]                                                           # test runs to the end of the data
    assert not set(splits["train"]) & set(splits["val"]) and not set(splits["val"]) & set(splits["test"])
    # every minute of a window is a sample
    assert set(((k[splits["val"]] + 1) % 15).tolist()) == set(range(15))
    # the persistence fan is fitted on the train split only
    assert samples.fan is not None and samples.fan["n"].max() <= 96 and F.B.fan_cells(samples.fan) == 120


def test_fractions_and_explicit_dates():
    cfg = F.FinetuneConfig(val_frac=0.15, test_frac=0.15)
    b = F.split_bounds(datetime(2025, 12, 15, 4, 16), datetime(2026, 9, 17, 2), cfg)
    assert b.val[0] == datetime(2026, 6, 26) and b.test[0] == datetime(2026, 8, 7) and b.train[1] == datetime(2026, 6, 25)
    c = F.FinetuneConfig(**CFG)
    b2 = F.split_bounds(datetime(2025, 12, 20, 2), datetime(2025, 12, 23), c)
    assert b2.val == (datetime(2025, 12, 21), datetime(2025, 12, 21, 23)) and b2.test[0] == datetime(2025, 12, 22)
    with pytest.raises(ValueError, match="empty train"):  # a one-day embargo swallows the 22h train day
        F.split_bounds(datetime(2025, 12, 20, 2), datetime(2025, 12, 23), F.FinetuneConfig(**CFG | {"embargo_bars": 1440}))
    with pytest.raises(ValueError):
        F.FinetuneConfig(loss_scale="sigma")


def test_batch_shapes_and_step_mask(settings):
    cfg = F.FinetuneConfig(**CFG)
    samples, splits, _, frame = F.prepare_data(settings, cfg)
    o = splits["train"][100:130]
    b = samples.batch(o)
    assert b["context"].shape == (30, 64) and b["po"].shape == (30, 17, 64) and b["pf"].shape == (30, 7, 128)
    assert b["target"].shape == b["valid"].shape == b["scale"].shape == (30, 15) and b["fan"].shape == (30, 15, 9)
    assert all(np.isfinite(b[key]).all() for key in ("context", "po", "pf", "target", "scale")) and not b["mask"].any()
    m = (frame["k"].to_numpy()[o] + 1) % 15
    assert (b["valid"].sum(axis=1) == 15 - m).all()                       # exactly the steps to expiry
    mid = frame["mid_close"].to_numpy()
    assert b["target"][0, 0] == pytest.approx(mid[o[0] + 1]) and b["last_mid"][0] == mid[o[0]]
    assert np.allclose(b["scale"], 1.0)
    fan_scaled = F.Samples(samples.fa, F.FinetuneConfig(**CFG | {"loss_scale": "fan"}), samples.fan).batch(o)["scale"]
    assert (fan_scaled >= F.MIN_SCALE).all() and fan_scaled.max() > 0.05


def test_pinball_loss_matches_definition():
    pred = torch.tensor([[[0.0] * 9, [1.0] * 9]])                # (1, 2, 9)
    target = torch.tensor([[0.2, 0.7]])
    both = F.pinball_loss(pred, target, torch.tensor([[True, True]]))
    # step 0: under-prediction by 0.2 -> mean(q) * 0.2 = 0.1; step 1: over-prediction by 0.3 -> mean(1 - q) * 0.3 = 0.15
    assert float(both) == pytest.approx(0.125)
    assert float(F.pinball_loss(pred, target, torch.tensor([[True, False]]))) == pytest.approx(0.1)
    assert float(F.pinball_loss(pred, target, torch.tensor([[True, True]]), scale=torch.tensor([[0.1, 0.3]]))) == pytest.approx(0.75)


def test_lr_schedule_and_trainable_spec():
    cfg = F.FinetuneConfig(**CFG | {"warmup_steps": 10, "max_steps": 110, "lr": 1.0})
    assert F.lr_at(5, cfg) == pytest.approx(0.5) and F.lr_at(10, cfg) == pytest.approx(1.0)
    assert F.lr_at(110, cfg) == pytest.approx(cfg.min_lr_frac) and F.lr_at(60, cfg) == pytest.approx((1 + cfg.min_lr_frac) / 2)
    assert F.parse_trainable("all") is None and F.parse_trainable("head") == 0 and F.parse_trainable("last:3") == 3
    with pytest.raises(ValueError):
        F.FinetuneConfig(trainable="some")


def test_training_path_is_the_inference_path_with_covariates(settings, tiny_model_dir):
    from chud_predictor.model import load_forecaster, predict

    cfg = F.FinetuneConfig(**CFG)
    samples, splits, _, _ = F.prepare_data(settings, cfg)
    b = samples.batch(splits["val"][[5, 300, 900]])
    handle = load_forecaster(tiny_model_dir, device="cpu", batch_size=4)
    _, q = predict(handle, b["context"], 64, past_only=b["po"], past_future=b["pf"])
    t = F._to_torch(b, "cpu")
    out = F.forward_train(handle.obj.model, t["context"], t["mask"], 64, t["po"], t["pf"])
    assert out.requires_grad and out.shape == (3, 64, 9)
    assert np.allclose(np.sort(out.detach().numpy(), axis=-1), q, atol=1e-5)
    uni = F.forward_train(handle.obj.model, t["context"], t["mask"], 64)
    assert not np.allclose(uni.detach().numpy(), out.detach().numpy(), atol=1e-5)          # the covariates are used
    with pytest.raises(ValueError, match="known-future"):
        F.forward_train(handle.obj.model, t["context"], t["mask"], 64, t["po"], t["pf"][..., :-10])


def test_set_trainable_freezes_the_rest(tiny_model_dir):
    from chud_predictor.model import load_forecaster

    model = load_forecaster(tiny_model_dir, device="cpu").obj.model
    total = sum(p.numel() for p in model.parameters())
    head = F.set_trainable(model, "head")
    assert head == sum(p.numel() for p in model.output_head.parameters())
    assert head < F.set_trainable(model, "last:1") < F.set_trainable(model, "all") == total


def test_run_finetune_end_to_end(settings, tiny_model_dir):
    from chud_predictor.model import load_forecaster

    cfg = F.FinetuneConfig(**CFG | {"model_id": tiny_model_dir, "run_id": "t"})
    before = {k: v.clone() for k, v in load_forecaster(tiny_model_dir, device="cpu").obj.model.state_dict().items()}
    out = F.run_finetune(settings, cfg)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["steps_run"] == 4 and summary["skipped_steps"] == 0 and len(summary["history"]) == 3  # step 0 + two evaluations
    for split in ("val", "test"):
        m = summary["zero_shot"][split]
        assert np.isfinite([m["pinball_c"], m["pinball_c_fan"], m["rmse_c"], m["rmse_c_persist"], m["skill_mse"]]).all() and 0 <= m["cover80"] <= 1
        assert m["n_steps"] > m["n_origins"]
    text = (out / "summary.txt").read_text()
    assert "== Splits" in text and all(s in text for s in ("train", "val", "test"))
    conf = json.loads((out / "config.json").read_text())
    assert conf["context"] == 64 and conf["covariates"] == "full" and conf["n_variates"] == 25       # what the backtest checks a checkpoint against
    assert pl.read_parquet(out / "baselines.parquet").filter(pl.col("bucket") == -1).height == 120

    after = load_forecaster(str(out / "last"), device="cpu").obj.model.state_dict()  # loadable by the backtest
    assert any(not torch.equal(before[k], after[k]) for k in before)               # weights moved
    assert all(torch.isfinite(v).all() for v in after.values())
    if summary["best_step"] > 0:
        assert (out / "best" / "model.safetensors").exists() and "--model-id" in text and "--fan-from" in text
    else:
        assert "no checkpoint was promoted" in text
    lines = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    assert sum("train_loss" in x for x in lines) == 4
