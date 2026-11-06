"""Exact checks for the Markov-switching source and its exact (Bayes) forecaster."""

import itertools

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from hsc.audit.gap import self_consistency_gap
from hsc.audit.regime import LAWS, MSAR, RegimeForecaster, filter_regimes, one_step_ppf, regime_windows, simulate_msar
from hsc.common.processes import ARForecaster
from hsc.common.quantiles import LEVELS, gaussian_quantiles


def test_simulator_switch_rate_and_stationary_regimes():
    law = MSAR(p=0.05)
    x, r = simulate_msar(law, 2000, 300, np.random.default_rng(0))
    assert abs((r[:, 1:] != r[:, :-1]).mean() - 0.05) < 0.002
    assert abs(r.mean() - 0.5) < 0.02
    mu, phi, sigma = law.params()
    stay0 = (r[:, 1:] == 0) & (r[:, :-1] == 0)
    resid = x[:, 1:][stay0] - phi[0] * x[:, :-1][stay0]  # mu_0 = 0
    assert abs(resid.std() - sigma[0]) < 0.01


def _path_log_density(law, x, path):
    mu, phi, sigma = law.params()
    trans = law.transition()
    j = path[0]
    lp = np.log(law.pi0[j]) + norm.logpdf(x[0], mu[j], sigma[j] / np.sqrt(1 - phi[j] ** 2))
    for t in range(1, len(x)):
        i, j = path[t - 1], path[t]
        lp += np.log(trans[i, j]) + norm.logpdf(x[t], mu[j] + phi[j] * (x[t - 1] - mu[j]), sigma[j])
    return lp


def test_filter_matches_enumeration_of_regime_paths():
    law = MSAR(p=0.1)
    x, _ = simulate_msar(law, 3, 10, np.random.default_rng(1))
    post = filter_regimes(law, x)
    for b in range(3):
        paths = list(itertools.product((0, 1), repeat=10))
        lp = np.array([_path_log_density(law, x[b], path) for path in paths])
        weight = np.exp(lp - lp.max())
        last = np.array([path[-1] for path in paths])
        exact = np.array([weight[last == j].sum() for j in (0, 1)]) / weight.sum()
        assert np.allclose(post[b], exact, atol=1e-12), (post[b], exact)


def test_known_regime_without_switching_is_the_gaussian_ar():
    for regime in (0, 1):
        law = MSAR(p=0.0, pi0=(1.0 - regime, float(regime)))
        x, r = simulate_msar(law, 6, 200, np.random.default_rng(2))
        assert (r == regime).all()
        mu, phi, sigma = law.params()
        mean, sd = ARForecaster([phi[regime]], sigma[regime], mu[regime]).moments(x, 64)
        q = RegimeForecaster(law)(x, 64)
        assert np.abs(q - gaussian_quantiles(mean, sd)).max() < 2e-4


def test_identical_regimes_with_switching_are_the_gaussian_ar():
    law = MSAR(mu=(1.0, 1.0), phi=(0.7, 0.7), sigma=(0.8, 0.8), p=0.3)
    x, _ = simulate_msar(law, 6, 200, np.random.default_rng(3))
    mean, sd = ARForecaster([0.7], 0.8, 1.0).moments(x, 64)
    assert np.abs(RegimeForecaster(law)(x, 64) - gaussian_quantiles(mean, sd)).max() < 2e-4


def test_predictive_matches_enumeration_of_future_regime_paths():
    law = MSAR(p=0.2)  # frequent switches, so many future paths carry weight
    x, _ = simulate_msar(law, 3, 40, np.random.default_rng(4))
    q = RegimeForecaster(law)(x, 6)
    mu, phi, sigma = law.params()
    trans = law.transition()
    w_next = filter_regimes(law, x) @ trans
    for b in range(3):
        for k in range(1, 7):
            weights, means, sds = [], [], []
            for path in itertools.product((0, 1), repeat=k):
                wt = w_next[b, path[0]] * np.prod([trans[path[i - 1], path[i]] for i in range(1, k)])
                m, v = x[b, -1], 0.0
                for j in path:
                    m, v = mu[j] + phi[j] * (m - mu[j]), phi[j] ** 2 * v + sigma[j] ** 2
                weights.append(wt), means.append(m), sds.append(np.sqrt(v))
            weights, means, sds = map(np.array, (weights, means, sds))
            assert np.isclose(weights.sum(), 1.0)
            exact = [brentq(lambda z: weights @ norm.cdf(z, means, sds) - lv, -30, 30, xtol=1e-12) for lv in LEVELS]
            assert np.abs(q[b, k - 1] - exact).max() < 1e-4, (b, k)


def test_long_horizon_matches_simulation():
    law = LAWS["msar"]
    contexts, _, _ = regime_windows("msar", 64, seed=5)
    post = filter_regimes(law, contexts)
    b = int(np.argmax(post.min(axis=1)))  # the context whose current regime is most uncertain
    q = RegimeForecaster(law)(contexts[b : b + 1], 64)[0]
    mu, phi, sigma = law.params()
    rng = np.random.default_rng(6)
    m = 400_000
    r = (rng.random(m) < post[b, 1]).astype(int)  # r_T from the filter
    x = np.full(m, contexts[b, -1])
    sims = []
    for _ in range(64):
        r = np.where(rng.random(m) < law.p, 1 - r, r)
        x = mu[r] + phi[r] * (x - mu[r]) + sigma[r] * rng.standard_normal(m)
        sims.append(np.quantile(x, LEVELS))
    assert np.abs(q - np.array(sims)).max() < 0.03


def test_exact_one_step_quantiles():
    law = MSAR(p=0.2)
    x, _ = simulate_msar(law, 4, 40, np.random.default_rng(8))
    u = np.array([0.01, 0.1, 0.33, 0.5, 0.9, 0.99])
    q = one_step_ppf(law, x, u)
    mu, phi, sigma = law.params()
    w = filter_regimes(law, x) @ law.transition()
    for b in range(4):
        mean = mu + phi * (x[b, -1] - mu)
        exact = [brentq(lambda z: w[b] @ norm.cdf(z, mean, sigma) - lv, -30, 30, xtol=1e-12) for lv in u]
        assert np.allclose(q[b], exact, atol=1e-9)
    assert np.abs(one_step_ppf(law, x, LEVELS) - RegimeForecaster(law)(x, 1)[:, 0]).max() < 1e-4


def test_floor_is_small_on_switching_contexts():
    contexts, truth, _ = regime_windows("msar", 12, seed=7)
    res = self_consistency_gap(RegimeForecaster(LAWS["msar"]), contexts, truth, inner=32)
    ratio = (res["D"] / res["spread"]).mean(axis=0)
    assert ratio.max() < 0.005, ratio
