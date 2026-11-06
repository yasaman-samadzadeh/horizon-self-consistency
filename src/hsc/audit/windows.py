"""The forecasting windows: 500-point contexts and the next 64 values, synthetic or from TIME.

Synthetic sources are stationary AR laws simulated with a fixed seed. Real sources are three
TIME datasets; windows are cut at random positions of randomly chosen series, after filling
gaps of at most 5 missing values by linear interpolation.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from hsc.common.paths import TIME_DIR
from hsc.common.processes import simulate_ar

H_MAX = 64
CONTEXT = 500  # extended contexts (501) stay inside every model's limit, so nothing slides out

SYNTHETIC = {
    "ar1": {"phi": [0.9], "sigma": 1.0},
    "ar2": {"phi": [0.5, 0.4], "sigma": 1.0},
    "ar2osc": {"phi": [1.5, -0.75], "sigma": 1.0},
    "ar1t": {"phi": [0.9], "sigma": 1.0, "noise": "t3"},  # heavy tails: tests the baseline for non-Gaussian forecasts
}
REAL = {
    "epf": ("epf_electricity_price/H", None),
    "covid": ("ECDC_COVID/D", None),
    "weather": ("SG_Weather/D", 1),  # air temperature
}
MAX_FILLED_GAP = 5


def fill_short_gaps(x: np.ndarray, max_gap: int = MAX_FILLED_GAP) -> np.ndarray:
    """Linearly interpolate runs of at most max_gap missing values; longer runs stay missing."""
    x = x.copy()
    missing = ~np.isfinite(x)
    if not missing.any():
        return x
    idx = np.arange(len(x))
    filled = np.interp(idx, idx[~missing], x[~missing])
    run_start = None
    for i in range(len(x) + 1):
        if i < len(x) and missing[i]:
            run_start = i if run_start is None else run_start
        elif run_start is not None:
            if i - run_start <= max_gap and run_start > 0 and i < len(x):
                x[run_start:i] = filled[run_start:i]
            run_start = None
    return x


def synthetic_windows(source: str, n: int, seed: int):
    spec = SYNTHETIC[source]
    rng = np.random.default_rng(seed)
    x = simulate_ar(spec["phi"], spec["sigma"], n, CONTEXT + H_MAX, rng, noise=spec.get("noise", "gauss"))
    return x[:, :CONTEXT], x[:, CONTEXT:]


def real_windows(source: str, n: int, seed: int):
    rel, variate = REAL[source]
    with pa.memory_map(str(TIME_DIR / rel / "data-00000-of-00001.arrow")) as src:
        table = pa.ipc.open_stream(src).read_all()
    series = []
    for target in table.column("target").to_pylist():
        arr = np.asarray(target, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[variate or 0]
        series.append(fill_short_gaps(arr))
    rng = np.random.default_rng(seed)
    need = CONTEXT + H_MAX
    windows = []
    attempts = 0
    while len(windows) < n and attempts < 100 * n:
        attempts += 1
        s = series[rng.integers(len(series))]
        if len(s) < need:
            continue
        start = rng.integers(0, len(s) - need + 1)
        w = s[start : start + need]
        if np.isfinite(w).all() and w[:CONTEXT].std() > 0:
            windows.append(w)
    w = np.stack(windows)
    return w[:, :CONTEXT], w[:, CONTEXT:]


def load_windows(source: str, n: int, seed: int):
    return synthetic_windows(source, n, seed) if source in SYNTHETIC else real_windows(source, n, seed)
