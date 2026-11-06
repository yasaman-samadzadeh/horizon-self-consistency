"""Closed-form forecasters, and the adapter that puts a TSFM behind the same call.

Every forecaster is a callable f(contexts, horizon, parent) -> quantiles (B, horizon, 9) at
LEVELS. `parent` gives, for each row, the index of the base context it was extended from; only
ARFit uses it, to keep the AR law fitted to the base context for its extended contexts.

On the command line: oracle = ExactForecaster, patchwork = Patchwork, arfit = ARFit.
"""

from __future__ import annotations

import numpy as np

from hsc.audit.windows import H_MAX, SYNTHETIC
from hsc.common.processes import ARForecaster, ar_autocovariance, ar_noise, fit_ar
from hsc.common.quantiles import LEVELS, gaussian_quantiles


class ExactForecaster:
    """The true AR law: the self-consistent baseline whose gap is the pipeline's own error."""

    def __init__(self, source: str):
        spec = SYNTHETIC[source]
        self.law = ARForecaster(spec["phi"], spec["sigma"])
        self.noise_q = None
        if spec.get("noise") == "t3":
            # AR(1) with t noise: x_{t+h} = phi^h x_t + sum_j phi^j e_j, a location family in x_t.
            # The noise-sum quantiles are computed once by Monte Carlo (2 million paths).
            phi = spec["phi"][0]
            rng = np.random.default_rng(12345)
            total = np.zeros(2_000_000)
            qs = []
            for _ in range(H_MAX):
                total = phi * total + ar_noise("t3", spec["sigma"], total.shape, rng)
                qs.append(np.quantile(total, LEVELS))
            self.noise_q = np.stack(qs)  # (H, 9)
            self.phi_pow = phi ** np.arange(1, H_MAX + 1)

    def __call__(self, contexts, horizon, parent):
        if self.noise_q is not None:
            return contexts[:, -1:, None] * self.phi_pow[None, :horizon, None] + self.noise_q[None, :horizon]
        mean, sd = self.law.moments(contexts, horizon)
        return gaussian_quantiles(mean, sd)


class Patchwork:
    """Best Gaussian forecast of each horizon from the last value alone, fitted horizon by horizon.

    Each horizon is optimal for its own loss, but on AR(2) data the horizons are not the
    rollouts of one kernel (rho(2) != rho(1)^2), so this is a positive control: the
    pipeline must detect that it is not self-consistent. On AR(1) data it coincides with
    the exact forecaster.
    """

    def __init__(self, source: str):
        spec = SYNTHETIC[source]
        gam = ar_autocovariance(spec["phi"], spec["sigma"], H_MAX)
        self.rho = gam / gam[0]
        self.var = gam[0]

    def __call__(self, contexts, horizon, parent):
        rho = self.rho[1 : horizon + 1]
        mean = contexts[:, -1:] * rho
        sd = np.sqrt(self.var * (1 - rho**2)) * np.ones_like(mean)
        return gaussian_quantiles(mean, sd)


class ARFit:
    """AR(8) fitted to each base context and then held fixed: a self-consistent baseline for real data."""

    def __init__(self, contexts: np.ndarray, p: int = 8):
        self.phi, self.sigma, self.mu = fit_ar(contexts, p)
        self.law = ARForecaster(np.zeros(p), 1.0)

    def __call__(self, contexts, horizon, parent):
        mean, sd = self.law.moments(contexts, horizon, phi=self.phi[parent], sigma=self.sigma[parent],
                                    mu=self.mu[parent])
        return gaussian_quantiles(mean, sd)


class TSFMAdapter:
    """A model with forecast(contexts, horizon) behind the forecaster call (parent is ignored)."""

    def __init__(self, model):
        self.model = model

    def __call__(self, contexts, horizon, parent):
        return self.model.forecast(contexts, horizon)
