"""The training loop itself, on the synthetic stand-in so it stays fast."""
import json
from pathlib import Path

import pytest
import torch

from pgan.config import ARMS, make
from pgan.train import build_models, ema_update, real_at_stage, train
from pgan.schedule import stages_for


def tiny(arm, tmp_path, **kw):
    opts = dict(steps=6, batch_size=8, data="synthetic", out_dir=str(tmp_path),
                num_workers=0, sample_every=0, log_every=2, steps_per_stage=2)
    opts.update(kw)
    return make(arm, **opts)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_each_arm_trains_and_writes_a_loadable_checkpoint(arm, tmp_path):
    cfg = tiny(arm, tmp_path)
    out = train(cfg, "cpu", log=lambda *_: None)
    ck = torch.load(Path(out) / "final.pt", map_location="cpu", weights_only=False)
    assert set(ck) >= {"g", "g_ema", "d", "cfg", "steps"}
    g, _ = build_models(cfg)
    g.load_state_dict(ck["g"])
    g.load_state_dict(ck["g_ema"])  # the EMA copy must match the same architecture


def test_metrics_are_written_one_json_row_per_log_interval(tmp_path):
    cfg = tiny("dcgan", tmp_path)
    out = train(cfg, "cpu", log=lambda *_: None)
    rows = [json.loads(l) for l in (Path(out) / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == cfg.steps // cfg.log_every
    assert all({"step", "loss_d", "loss_g", "res", "alpha"} <= set(r) for r in rows)
    assert [r["step"] for r in rows] == sorted(r["step"] for r in rows)


def test_the_config_is_saved_next_to_the_weights(tmp_path):
    cfg = tiny("progan-grow", tmp_path)
    out = train(cfg, "cpu", log=lambda *_: None)
    saved = json.loads((Path(out) / "config.json").read_text())
    assert saved["arm"] == "progan-grow" and saved["grow"] is True


def test_growing_run_logs_more_than_one_resolution(tmp_path):
    cfg = tiny("progan-grow", tmp_path, steps=12, log_every=1, steps_per_stage=2)
    out = train(cfg, "cpu", log=lambda *_: None)
    rows = [json.loads(l) for l in (Path(out) / "metrics.jsonl").read_text().splitlines()]
    assert len({r["res"] for r in rows}) > 1


def test_ema_moves_toward_the_live_weights_and_stays_behind_them():
    cfg = make("dcgan")
    g, _ = build_models(cfg)
    import copy
    ema = copy.deepcopy(g)
    with torch.no_grad():
        for p in g.parameters():
            p.add_(1.0)
    before = torch.cat([p.flatten() for p in ema.parameters()]).clone()
    ema_update(ema, g, decay=0.9)
    after = torch.cat([p.flatten() for p in ema.parameters()])
    live = torch.cat([p.flatten() for p in g.parameters()])
    assert not torch.allclose(before, after), "EMA did not move"
    assert (after - live).abs().mean() > 0, "EMA jumped straight onto the live weights"


def test_real_batches_are_resized_to_the_active_stage():
    stages = stages_for(64)
    x = torch.randn(4, 3, 64, 64)
    for stage, res in enumerate(stages):
        assert real_at_stage(x, stage, 1.0, stages).shape[-1] == res


def test_fading_blends_reals_toward_the_lower_resolution():
    """At alpha=0 the reals must equal their own downsample-upsample, or D wins
    the fade window on sharpness alone."""
    stages = stages_for(64)
    x = torch.randn(4, 3, 64, 64)
    sharp = real_at_stage(x, 4, 1.0, stages)
    blurred = real_at_stage(x, 4, 0.0, stages)
    expect = torch.nn.functional.interpolate(
        torch.nn.functional.avg_pool2d(sharp, 2), scale_factor=2, mode="nearest")
    assert torch.allclose(blurred, expect, atol=1e-6)
    assert not torch.allclose(blurred, sharp)


def test_same_seed_gives_the_same_losses(tmp_path):
    a = train(tiny("dcgan", tmp_path / "a", seed=3), "cpu", log=lambda *_: None)
    b = train(tiny("dcgan", tmp_path / "b", seed=3), "cpu", log=lambda *_: None)
    ra = (Path(a) / "metrics.jsonl").read_text()
    rb = (Path(b) / "metrics.jsonl").read_text()
    la = [json.loads(l)["loss_d"] for l in ra.splitlines()]
    lb = [json.loads(l)["loss_d"] for l in rb.splitlines()]
    assert la == pytest.approx(lb, rel=1e-9)
