"""Checks for the Chronos-T5 sample pipeline and its call budget."""

import numpy as np
import torch

from hsc.audit.baselines import ExactForecaster, Patchwork
from hsc.audit.chronos_t5 import ChronosT5Sampler, ExactSampler, QuantileSampler, sample_gap, sample_quantiles
from hsc.audit.windows import synthetic_windows
from hsc.common.quantiles import LEVELS


class FakePipe:
    """Stands in for ChronosPipeline: every path equals its context's last value."""

    def __init__(self):
        self.sizes, self.top_k = [], set()

    def predict(self, ctx, prediction_length, num_samples, limit_prediction_length, top_k=None):
        self.sizes.append(ctx.shape[0] * num_samples)
        self.top_k.add(top_k)
        return ctx[:, -1][:, None, None] + torch.zeros(ctx.shape[0], num_samples, prediction_length)


def test_t5_sampler_caps_sequences_per_call_and_keeps_order():
    contexts = np.arange(50, dtype=float).reshape(5, 10)
    for n_samples in (4096, 256, 16):
        pipe = FakePipe()
        s = ChronosT5Sampler(pipe, max_sequences=128).samples(contexts, 7, n_samples, seed=0)
        assert s.shape == (5, n_samples, 7)
        assert max(pipe.sizes) <= 128 and sum(pipe.sizes) == 5 * n_samples
        assert np.array_equal(s, np.broadcast_to(contexts[:, -1, None, None], s.shape))
        assert pipe.top_k == {None}  # None: the model's own setting
    pipe = FakePipe()
    ChronosT5Sampler(pipe, top_k=4096).samples(contexts, 7, 16, seed=0)
    assert pipe.top_k == {4096}


def test_oracle_sampler_has_exact_moments():
    contexts, _ = synthetic_windows("ar2", 4, seed=3)
    mean, sd = ExactForecaster("ar2").law.moments(contexts, 8)
    s = ExactSampler("ar2").samples(contexts, 8, 40_000, seed=0)
    assert np.allclose(s.mean(axis=1), mean, atol=5 * sd.max() / np.sqrt(40_000))
    assert np.allclose(s.std(axis=1), sd, rtol=0.03)


def test_quantile_sampler_draws_from_the_forecast():
    contexts, _ = synthetic_windows("ar1", 2, seed=4)
    oracle = ExactForecaster("ar1")
    q = sample_quantiles(QuantileSampler(oracle).samples(contexts, 3, 200_000, seed=0))
    assert np.allclose(q, oracle(contexts, 3, None), atol=0.03)


def test_sample_floor_shrinks_with_samples_and_detects_patchwork():
    contexts, truth = synthetic_windows("ar2", 16, seed=1)
    ratio = lambda r: (r["D"] / r["spread"]).mean(axis=0)  # noqa: E731
    floor = ratio(sample_gap(ExactSampler("ar2"), contexts, truth, 16, 4096, 256))
    finer = ratio(sample_gap(ExactSampler("ar2"), contexts, truth, 16, 16384, 1024))
    patch = ratio(sample_gap(QuantileSampler(Patchwork("ar2")), contexts, truth, 16, 4096, 256))
    assert 0.004 < floor.mean() < 0.02, floor
    assert finer.mean() < 0.7 * floor.mean(), (finer, floor)  # Monte Carlo error ~ 1/sqrt(samples)
    assert patch[0] > 3 * floor[0], (patch, floor)
    assert np.allclose(LEVELS, np.arange(1, 10) / 10)
