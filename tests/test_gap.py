"""Exact checks for the measurement pipeline (quantiles, AR laws, the gap), run before any GPU time is spent."""

import numpy as np
from scipy.stats import norm

from hsc.audit.baselines import ExactForecaster, Patchwork
from hsc.audit.gap import self_consistency_gap
from hsc.audit.windows import synthetic_windows
from hsc.common.processes import ARForecaster, ar_autocovariance
from hsc.common.quantiles import LEVELS, cdf, gaussian_quantiles, mixture_quantiles, ppf


def test_ppf_cdf_are_inverse_and_hit_knots():
    q = gaussian_quantiles(np.array([0.3, -1.0]), np.array([1.0, 2.5]))
    u = np.array([[0.01, 0.1, 0.35, 0.9, 0.99]] * 2)
    z = ppf(q, u)
    assert np.allclose(cdf(q, z), u, atol=1e-9)
    assert np.allclose(ppf(q, np.broadcast_to(LEVELS, q.shape)), q, atol=1e-6)
    # Gaussian tails make the extension exact for a Gaussian forecast
    assert np.allclose(z[0, [0, 4]], norm.ppf([0.01, 0.99], 0.3, 1.0), atol=0.05)


def test_mixture_of_identical_components_is_the_component():
    q = gaussian_quantiles(np.zeros((3, 5)), np.ones((3, 5)) * 1.7)
    assert np.allclose(mixture_quantiles(q), q[:, 0], atol=1e-6)


def test_mixture_matches_monte_carlo():
    rng = np.random.default_rng(0)
    means, sds = np.array([-1.0, 0.5, 2.0]), np.array([0.5, 1.0, 0.8])
    mix = mixture_quantiles(gaussian_quantiles(means, sds)[None])[0]
    draws = np.concatenate([rng.normal(m, s, 400_000) for m, s in zip(means, sds)])
    assert np.allclose(mix, np.quantile(draws, LEVELS), atol=0.02)


def test_ar_moments_match_closed_form_ar1():
    law = ARForecaster([0.9], 1.0)
    mean, sd = law.moments(np.array([[0.0, 2.0]]), 3)
    assert np.allclose(mean[0], 2.0 * 0.9 ** np.arange(1, 4))
    assert np.allclose(sd[0] ** 2, (1 - 0.81 ** np.arange(1, 4)) / (1 - 0.81))


def test_ar2_autocorrelation_identity():
    gam = ar_autocovariance([0.5, 0.4], 1.0, 3)
    rho = gam / gam[0]
    assert np.isclose(rho[1], 0.5 / 0.6)
    assert np.isclose(rho[2] - rho[1] ** 2, 0.4 * (1 - rho[1] ** 2))


def test_oracle_floor_is_small_and_patchwork_is_detected():
    contexts, truth = synthetic_windows("ar2", 64, seed=1)
    floor = self_consistency_gap(ExactForecaster("ar2"), contexts, truth, inner=32)
    patch = self_consistency_gap(Patchwork("ar2"), contexts, truth, inner=32)
    floor_ratio = (floor["D"] / floor["spread"]).mean(axis=0)
    patch_ratio = (patch["D"] / patch["spread"]).mean(axis=0)
    assert floor_ratio.max() < 0.02, floor_ratio
    assert patch_ratio[0] > 5 * floor_ratio[0], (patch_ratio, floor_ratio)
