"""Simulators and exact forecasts for the synthetic AR processes, and a per-context AR fit."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_lyapunov


def companion(phi: np.ndarray) -> np.ndarray:
    p = len(phi)
    a = np.zeros((p, p))
    a[0] = phi
    a[1:, :-1] = np.eye(p - 1)
    return a


def ar_noise(kind: str, sigma: float, size, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise, or Student-t with 3 degrees of freedom scaled to the same variance."""
    if kind == "t3":
        return sigma * rng.standard_t(3, size=size) / np.sqrt(3.0)
    return rng.normal(0.0, sigma, size=size)


def simulate_ar(phi, sigma: float, n: int, length: int, rng: np.random.Generator, burn: int = 500,
                noise: str = "gauss") -> np.ndarray:
    """n independent stationary AR(p) series of the given length (zero mean)."""
    phi = np.asarray(phi, dtype=np.float64)
    p = len(phi)
    total = length + burn
    x = np.zeros((n, total + p))
    eps = ar_noise(noise, sigma, (n, total), rng)
    for t in range(total):
        x[:, t + p] = x[:, t : t + p][:, ::-1] @ phi + eps[:, t]
    return x[:, -length:]


class ARForecaster:
    """Exact conditional Gaussian forecasts of a known AR(p) law (the exact forecaster).

    Horizon h: mean e1' A^h s, variance sigma^2 sum_{j<h} (A^j)_{00}^2, where s holds the
    last p values (most recent first). These are the rollouts of one one-step kernel, so the
    family is self-consistent by construction.
    """

    def __init__(self, phi, sigma: float, mean: float | np.ndarray = 0.0):
        self.phi = np.asarray(phi, dtype=np.float64)
        self.sigma = float(sigma)
        self.mu = mean

    def moments(self, contexts: np.ndarray, horizon: int, phi=None, sigma=None, mu=None):
        phi = self.phi if phi is None else phi
        sigma = self.sigma if sigma is None else sigma
        mu = self.mu if mu is None else mu
        p = phi.shape[-1]
        state = (contexts[:, -p:][:, ::-1] - np.asarray(mu).reshape(-1, 1))  # (B, p)
        means, variances = [], []
        if phi.ndim == 1:
            a = companion(phi)
            power = np.eye(p)
            var = 0.0
            for _ in range(horizon):
                var = var + power[0, 0] ** 2
                power = a @ power
                means.append(state @ power[0])
                variances.append(np.full(len(contexts), sigma**2 * var))
        else:  # one fitted law per row
            b = len(contexts)
            a = np.zeros((b, p, p))
            a[:, 0, :] = phi
            if p > 1:
                a[:, 1:, :-1] = np.eye(p - 1)
            power = np.broadcast_to(np.eye(p), (b, p, p)).copy()
            var = np.zeros(b)
            for _ in range(horizon):
                var = var + power[:, 0, 0] ** 2
                power = a @ power
                means.append(np.einsum("bp,bp->b", state, power[:, 0, :]))
                variances.append(np.asarray(sigma) ** 2 * var)
        mean = np.stack(means, axis=1) + np.asarray(mu).reshape(-1, 1)
        sd = np.sqrt(np.stack(variances, axis=1))
        return mean, sd


def ar_autocovariance(phi, sigma: float, max_lag: int) -> np.ndarray:
    """gamma(0..max_lag) of a stationary AR(p) from the companion-form Lyapunov equation."""
    phi = np.asarray(phi, dtype=np.float64)
    a = companion(phi)
    q = np.zeros_like(a)
    q[0, 0] = sigma**2
    cov = solve_discrete_lyapunov(a, q)
    gam = []
    power = np.eye(len(phi))
    for _ in range(max_lag + 1):
        gam.append((power @ cov)[0, 0])
        power = a @ power
    return np.array(gam)


def fit_ar(contexts: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares AR(p) with intercept, fitted separately to each context row.

    Returns phi (B, p), sigma (B,), mean (B,). Used as a self-consistent baseline on real
    series: its horizons are rollouts of its own one-step kernel.
    """
    b, t = contexts.shape
    mu = contexts.mean(axis=1)
    xc = contexts - mu[:, None]
    lags = np.stack([xc[:, p - 1 - i : t - 1 - i] for i in range(p)], axis=-1)  # (B, T-p, p)
    target = xc[:, p:]
    gram = np.einsum("btp,btq->bpq", lags, lags) + 1e-8 * np.eye(p)
    rhs = np.einsum("btp,bt->bp", lags, target)
    phi = np.linalg.solve(gram, rhs[..., None])[..., 0]
    resid = target - np.einsum("btp,bp->bt", lags, phi)
    sigma = resid.std(axis=1, ddof=p)
    return phi, sigma, mu
