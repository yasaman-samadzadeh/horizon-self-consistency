"""Checks for the rigour module: the residual-noise AR(8) baseline and the whole-series bootstrap."""

import numpy as np
from scipy.stats import norm

from hsc.audit.gap import self_consistency_gap
from hsc.audit.mechanism_report import cluster_bootstrap, spearman_ci
from hsc.audit.rigour import ARResid, ar_residuals, cluster_mean, cluster_spearman, impulse_response, paired_ratio
from hsc.audit.windows import H_MAX, synthetic_windows
from hsc.common.processes import ARForecaster, fit_ar, simulate_ar
from hsc.common.quantiles import gaussian_quantiles

# Sample-quantile sd at level 0.1 or 0.9 of a Gaussian: sqrt(0.09 / n) / pdf(z_0.9) sd, i.e. 0.67 / sqrt(n) widths.
QUANTILE_SE = 0.67


def test_residuals_and_impulse_response_match_the_fit():
    contexts, _ = synthetic_windows("ar2", 6, seed=5)
    phi, sigma, mu = fit_ar(contexts, 2)
    assert np.allclose(ar_residuals(contexts, phi, mu).std(axis=1, ddof=2), sigma)
    assert np.allclose(impulse_response(np.array([[0.9], [0.5]]), 5), [0.9 ** np.arange(5), 0.5 ** np.arange(5)])
    p1, p2 = 0.5, 0.4
    assert np.allclose(impulse_response(np.array([[p1, p2]]), 4)[0], [1, p1, p1**2 + p2, p1**3 + 2 * p1 * p2])
    phi8, sigma8, mu8 = fit_ar(contexts, 8)
    _, sd = ARForecaster(np.zeros(8), 1.0).moments(contexts, H_MAX, phi=phi8, sigma=sigma8, mu=mu8)
    assert np.allclose(sigma8[:, None] ** 2 * np.cumsum(impulse_response(phi8, H_MAX) ** 2, axis=1), sd**2)


def test_arresid_with_gaussian_noise_matches_the_gaussian_forecaster():
    contexts, _ = synthetic_windows("ar1", 4, seed=11)
    phi, sigma, mu = fit_ar(contexts, 1)
    z = norm.ppf((np.arange(100_000) + 0.5) / 100_000)  # a noise pool whose distribution is N(0, 1) to 1e-5
    n_paths = 400_000
    ar = ARResid(contexts, p=1, n_paths=n_paths, seed=3, pools=sigma[:, None] * z / z.std())
    extended = np.concatenate([contexts, contexts[:, -1:] + 0.5], axis=1)
    for ctx in (contexts, extended):
        mean, sd = ARForecaster(np.zeros(1), 1.0).moments(ctx, H_MAX, phi=phi, sigma=sigma, mu=mu)
        want = gaussian_quantiles(mean, sd)
        width = want[..., -1:] - want[..., :1]
        assert (np.abs(ar(ctx, H_MAX, np.arange(4)) - want) / width).max() < 5 * QUANTILE_SE / np.sqrt(n_paths)


def test_arresid_on_its_own_gaussian_residuals_matches_the_gaussian_forecaster():
    contexts = simulate_ar([0.9], 1.0, 2, 40_000, np.random.default_rng(2))
    n_paths = 200_000
    ar = ARResid(contexts, p=1, n_paths=n_paths, seed=4)
    mean, sd = ARForecaster(np.zeros(1), 1.0).moments(contexts, H_MAX, phi=ar.phi, sigma=ar.sigma, mu=ar.mu)
    want = gaussian_quantiles(mean, sd)
    width = want[..., -1:] - want[..., :1]
    # Monte Carlo error plus the 40,000 residuals' own departure from Gaussian (about 0.003 widths at h = 1)
    assert (np.abs(ar(contexts, H_MAX, np.arange(2)) - want) / width).max() < 0.02


def test_arresid_defect_on_synthetic_data_is_at_the_floor():
    contexts, truth = synthetic_windows("ar1t", 16, seed=7)
    res = self_consistency_gap(ARResid(contexts, p=8, n_paths=200_000, seed=0), contexts, truth, inner=32)
    ratio = (res["D"] / res["spread"]).mean(axis=0)
    # the main run's self-consistent baselines are <= 0.002 at h >= 8 (t(3) oracle 0.0105 at h = 2); every model is >= 0.0075
    assert ratio[2:].max() < 0.004, ratio
    assert ratio.max() < 0.02, ratio


def test_series_bootstrap_helpers_reproduce_mechanism_report():
    rng = np.random.default_rng(0)
    cluster = np.repeat(np.arange(5), [10, 3, 7, 1, 9])
    v = rng.normal(size=(30, 2)) + cluster[:, None]
    ref = cluster_bootstrap(v, cluster, n_boot=300, seed=4)
    got = cluster_mean(v, cluster, within=False, n_boot=300, seed=4)
    assert np.allclose(got["lo"], ref["lo"]) and np.allclose(got["hi"], ref["hi"])
    a, b = v[:, 0], v[:, 1] + rng.normal(size=30)
    sp = spearman_ci(a, b, cluster, n_boot=300, seed=4)
    mine = cluster_spearman(a, b, cluster, n_boot=300, seed=4)
    assert np.isclose(mine["rho"], sp[0]) and np.allclose(mine["ci95"], sp[1:])
    r = paired_ratio(3 * np.abs(v), np.abs(v), cluster, within=True, n_boot=200)
    assert np.allclose(r["mean"], 3) and np.allclose(r["lo"], 3) and np.allclose(r["hi"], 3)
