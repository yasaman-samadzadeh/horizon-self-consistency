"""Quantile-function utilities: the representation every forecast goes through.

Every forecast is stored as its quantiles at LEVELS = 0.1, 0.2, ..., 0.9, which is what
Chronos-Bolt outputs natively. To sample from a forecast, or to mix several forecasts, we
need a full distribution. Between the knots the quantile function is interpolated
linearly; below 0.1 and above 0.9 we attach Gaussian tails whose scale matches the slope
of the outermost pair of knots. Every forecaster, including the exact forecaster, goes
through exactly this representation, so the error it introduces shows up in the
self-consistent baselines' gap instead of being blamed on a model.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

LEVELS = np.round(np.arange(1, 10) / 10, 10)
_Z = ndtri(LEVELS)


def prepare(q: np.ndarray) -> np.ndarray:
    """Make quantile knots non-crossing and strictly increasing along the last axis."""
    q = np.maximum.accumulate(np.asarray(q, dtype=np.float64), axis=-1)
    scale = 1.0 + np.abs(q).max(axis=-1, keepdims=True)
    return q + 1e-9 * scale * np.arange(q.shape[-1])


def _tail_scales(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = (q[..., 1] - q[..., 0]) / (_Z[1] - _Z[0])
    hi = (q[..., -1] - q[..., -2]) / (_Z[-1] - _Z[-2])
    return lo, hi


def ppf(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Quantile function. q has shape (..., K); u has shape (..., S) with the same leading dims."""
    q = prepare(q)
    u = np.asarray(u, dtype=np.float64)
    lo, hi = _tail_scales(q)
    j = np.clip(np.searchsorted(LEVELS, u, side="right") - 1, 0, len(LEVELS) - 2)
    q_j = np.take_along_axis(q, j, axis=-1)
    q_j1 = np.take_along_axis(q, j + 1, axis=-1)
    w = (u - LEVELS[j]) / (LEVELS[j + 1] - LEVELS[j])
    out = q_j + w * (q_j1 - q_j)
    below = u < LEVELS[0]
    above = u > LEVELS[-1]
    zu = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
    out = np.where(below, q[..., :1] + lo[..., None] * (zu - _Z[0]), out)
    out = np.where(above, q[..., -1:] + hi[..., None] * (zu - _Z[-1]), out)
    return out


def cdf(q: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Distribution function, the inverse of ppf. q (..., K); z (..., S)."""
    q = prepare(q)
    z = np.asarray(z, dtype=np.float64)
    lo, hi = _tail_scales(q)
    j = np.clip((q[..., None, :] <= z[..., :, None]).sum(axis=-1) - 1, 0, len(LEVELS) - 2)
    q_j = np.take_along_axis(q, j, axis=-1)
    q_j1 = np.take_along_axis(q, j + 1, axis=-1)
    out = LEVELS[j] + (z - q_j) / (q_j1 - q_j) * (LEVELS[j + 1] - LEVELS[j])
    out = np.where(z < q[..., :1], ndtr(_Z[0] + (z - q[..., :1]) / lo[..., None]), out)
    out = np.where(z > q[..., -1:], ndtr(_Z[-1] + (z - q[..., -1:]) / hi[..., None]), out)
    return out


def mixture_quantiles(q_components: np.ndarray, iters: int = 60) -> np.ndarray:
    """Quantiles at LEVELS of the equal-weight mixture of M forecasts.

    q_components has shape (..., M, K). The mixture's tau-quantile lies between the
    smallest and largest component tau-quantiles, so bisection on that bracket converges.
    """
    qc = prepare(q_components)
    levels = np.broadcast_to(LEVELS, qc.shape[:-1] + (len(LEVELS),))
    at_levels = ppf(qc, levels)  # (..., M, L)
    lo = at_levels.min(axis=-2)
    hi = at_levels.max(axis=-2)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = cdf(qc, np.broadcast_to(mid[..., None, :], at_levels.shape)).mean(axis=-2)
        go_up = f < LEVELS
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)
    return 0.5 * (lo + hi)


def band_w1(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Wasserstein-1 distance restricted to the 0.1-0.9 band, from quantiles at LEVELS.

    The exact band distance is the integral of |Qa(u) - Qb(u)| over u in [0.1, 0.9];
    with knots every 0.1 we use the mean absolute difference over the nine knots,
    which is the same up to discretisation and the constant 0.8.
    """
    return np.abs(np.asarray(qa) - np.asarray(qb)).mean(axis=-1)


def quantile_crps(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Quantile-score approximation of CRPS: twice the mean pinball loss over LEVELS."""
    diff = np.asarray(y)[..., None] - np.asarray(q)
    pinball = np.maximum(LEVELS * diff, (LEVELS - 1) * diff)
    return 2.0 * pinball.mean(axis=-1)


def gaussian_quantiles(mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Quantiles at LEVELS of N(mean, sd^2); output shape (..., K)."""
    return np.asarray(mean)[..., None] + np.asarray(sd)[..., None] * _Z
