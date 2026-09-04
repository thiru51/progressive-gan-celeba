"""The growing schedule. Getting this wrong silently changes the experiment."""
import pytest

from pgan.config import make
from pgan.schedule import resolution_at, schedule, stages_for


def test_non_growing_arms_sit_at_full_resolution_forever():
    cfg = make("progan-fixed", steps=1000)
    assert all(schedule(s, cfg) == (4, 1.0) for s in range(0, 1000, 37))


def test_dcgan_arms_never_grow():
    cfg = make("dcgan", steps=500)
    assert schedule(0, cfg) == schedule(499, cfg) == (4, 1.0)


def test_growing_visits_every_resolution_in_order():
    cfg = make("progan-grow", steps=12000, steps_per_stage=2000)
    seen = []
    for s in range(cfg.steps):
        r = resolution_at(s, cfg)
        if not seen or seen[-1] != r:
            seen.append(r)
    assert seen == [4, 8, 16, 32, 64]


def test_every_stage_after_the_first_fades_in_from_zero():
    cfg = make("progan-grow", steps=12000, steps_per_stage=2000)
    for stage in range(1, len(stages_for(cfg.resolution))):
        start = stage * cfg.steps_per_stage
        assert schedule(start, cfg) == (stage, 0.0)
        # Half the window later it must be fully faded in.
        assert schedule(start + cfg.steps_per_stage // 2, cfg)[1] == 1.0


def test_the_final_resolution_also_fades_rather_than_snapping_on():
    """The last stage is the easy one to forget, and forgetting it means the
    64x64 block is switched on at full strength with untrained weights."""
    cfg = make("progan-grow", steps=12000, steps_per_stage=2000)
    last = len(stages_for(cfg.resolution)) - 1
    assert schedule(last * cfg.steps_per_stage, cfg) == (last, 0.0)


def test_alpha_is_monotone_within_a_stage():
    cfg = make("progan-grow", steps=12000, steps_per_stage=2000)
    alphas = [schedule(s, cfg)[1] for s in range(2000, 4000)]
    assert alphas == sorted(alphas)
    assert alphas[0] == 0.0 and alphas[-1] == 1.0


def test_alpha_never_leaves_the_unit_interval():
    cfg = make("progan-grow", steps=20000, steps_per_stage=1500)
    assert all(0.0 <= schedule(s, cfg)[1] <= 1.0 for s in range(20000))


def test_leftover_budget_runs_at_full_resolution():
    cfg = make("progan-grow", steps=12000, steps_per_stage=2000)
    n = len(stages_for(cfg.resolution))
    for s in range(n * cfg.steps_per_stage, cfg.steps):
        assert schedule(s, cfg) == (n - 1, 1.0)


@pytest.mark.parametrize("per", [200, 1000, 2000, 5000])
def test_schedule_is_well_formed_for_any_stage_length(per):
    cfg = make("progan-grow", steps=12000, steps_per_stage=per)
    n = len(stages_for(cfg.resolution))
    for s in range(0, 12000, 7):
        stage, alpha = schedule(s, cfg)
        assert 0 <= stage < n and 0.0 <= alpha <= 1.0
