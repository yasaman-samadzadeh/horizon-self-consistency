"""The per-draw recorder must reproduce hsc.audit.gap exactly."""

import numpy as np

from hsc.audit.baselines import ExactForecaster
from hsc.audit.gap import self_consistency_gap
from hsc.audit.mechanism import BUMP_LENGTH, INNER_IDX, Slide, bump_windows, defect_with_draws
from hsc.audit.windows import CONTEXT, H_MAX, synthetic_windows
from hsc.common.quantiles import ppf


def test_recorder_reproduces_composition_defect_and_keeps_the_draws():
    contexts, truth = synthetic_windows("ar2", 8, seed=2)
    oracle = ExactForecaster("ar2")
    ref = self_consistency_gap(oracle, contexts, truth, inner=16)
    res = defect_with_draws(oracle, contexts, truth, inner=16)
    for key in ("D", "spread", "crps", "comp", "direct"):
        assert np.array_equal(res[key], ref[key]), key
    u = (np.arange(16) + 0.5) / 16
    assert np.allclose(res["y"], ppf(ref["direct"][:, 0, :], np.broadcast_to(u, (8, 16))))
    ext = np.concatenate([contexts[:1].repeat(16, axis=0), res["y"][0][:, None]], axis=1)
    assert np.allclose(res["inner_fc"][0], oracle(ext, H_MAX - 1, None)[:, INNER_IDX])


def test_bump_leaves_the_recent_context_and_the_exact_forecast_unchanged():
    base, truth = synthetic_windows("ar1", 6, seed=20261104)
    bumped, truth_b = bump_windows(40.0, 6, seed=20261104)
    assert np.array_equal(bump_windows(0.0, 6, seed=20261104)[0], base)
    assert np.array_equal(bumped[:, BUMP_LENGTH:], base[:, BUMP_LENGTH:]) and np.array_equal(truth, truth_b)
    assert bumped.std(axis=1).min() > 4 * base.std(axis=1).max()
    oracle = ExactForecaster("ar1")
    assert np.allclose(oracle(bumped, H_MAX, None), oracle(base, H_MAX, None))


def test_slide_changes_nothing_for_a_markov_law():
    contexts, truth = synthetic_windows("ar2", 8, seed=3)
    oracle = ExactForecaster("ar2")
    ref = self_consistency_gap(oracle, contexts, truth, inner=16)
    slid = self_consistency_gap(Slide(oracle, CONTEXT), contexts, truth, inner=16)
    assert np.allclose(slid["D"], ref["D"])
