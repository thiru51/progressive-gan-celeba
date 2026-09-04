"""Progressive-growing schedule.

Total step budget is fixed, so a growing run and a non-growing run see exactly
the same number of generator updates. The growing run simply spends the early
ones at low resolution, which is the paper's actual claim: the same compute
buys more when it is spent coarse-to-fine.
"""
from __future__ import annotations

from pgan.models.progan import RESOLUTIONS


def stages_for(resolution):
    return [r for r in RESOLUTIONS if r <= resolution]


def schedule(step, cfg):
    """Return (stage_index, alpha) for a given global step.

    Stage 0 has no fade -- there is no earlier resolution to blend with. Every
    later stage, the final one included, spends its first half fading the new
    block in (alpha 0 -> 1) and its second half stabilising at alpha = 1. Any
    budget left after the last stage's window runs at full resolution.
    """
    n_stages = len(stages_for(cfg.resolution))
    last = n_stages - 1
    if not cfg.grow:
        return last, 1.0

    per = cfg.steps_per_stage
    stage = min(step // per, last)
    if stage == 0:
        return 0, 1.0
    # Past the last stage's own window the network is fully grown and stable.
    if step >= per * n_stages:
        return last, 1.0
    within = step - per * stage
    half = per // 2
    alpha = min(1.0, within / half) if half else 1.0
    return stage, alpha


def resolution_at(step, cfg):
    stage, _ = schedule(step, cfg)
    return stages_for(cfg.resolution)[stage]
