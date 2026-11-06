"""The measured self-consistency gap D_h of a forecaster, and its summary over contexts.

For each context c the forecaster gives its direct forecast q_h(.|c) for h = 1..64. The
chained forecast T_h(.|c) is the equal-weight mixture of its own (h-1)-step forecasts from the
extended contexts c+y_m, where y_m = Q_1((m - 0.5)/M | c) are M evenly spaced quantiles of its
one-step forecast. D_h(c) is the mean absolute difference between the direct and chained
forecasts at the nine quantile levels 0.1..0.9. It needs no labels. A forecaster whose horizons
all come from one probabilistic model has D_h = 0 up to the error of this pipeline, which the
self-consistent baselines measure.
"""

from __future__ import annotations

import time

import numpy as np

from hsc.audit.windows import H_MAX
from hsc.common.quantiles import band_w1, mixture_quantiles, ppf, quantile_crps
from hsc.common.stats import bootstrap_mean, bootstrap_ratio

HORIZONS = (2, 4, 8, 16, 32, 64)


def self_consistency_gap(forecaster, contexts: np.ndarray, truth: np.ndarray, inner: int) -> dict:
    n = len(contexts)
    t0 = time.time()
    direct = forecaster(contexts, H_MAX, np.arange(n))  # (n, H, 9)
    u = np.broadcast_to((np.arange(inner) + 0.5) / inner, (n, inner))
    y = ppf(direct[:, 0, :], u)  # evenly spaced quantiles of the model's own one-step forecast
    ext = np.concatenate([np.repeat(contexts, inner, axis=0), y.reshape(-1, 1)], axis=1)
    parent = np.repeat(np.arange(n), inner)
    ext_fc = forecaster(ext, H_MAX - 1, parent).reshape(n, inner, H_MAX - 1, -1)
    t_model = time.time() - t0
    rows = {"D": [], "spread": [], "crps": [], "comp": []}
    for h in HORIZONS:
        comp = mixture_quantiles(ext_fc[:, :, h - 2, :])
        rows["D"].append(band_w1(direct[:, h - 1, :], comp))
        rows["spread"].append(direct[:, h - 1, -1] - direct[:, h - 1, 0])
        rows["crps"].append(quantile_crps(direct[:, h - 1, :], truth[:, h - 1]))
        rows["comp"].append(comp)
    out = {k: np.stack(v, axis=1) for k, v in rows.items()}
    out["direct"] = direct
    out["seconds_model"] = t_model
    return out


def summarise(res: dict) -> dict:
    d, spread, crps = res["D"], res["spread"], res["crps"]
    return {
        "horizons": list(HORIZONS),
        "D": bootstrap_mean(d),
        "D_over_spread": bootstrap_mean(d / spread),
        "D_over_crps": bootstrap_ratio(d, crps),
        "crps": bootstrap_mean(crps),
        "spread": bootstrap_mean(spread),
    }
