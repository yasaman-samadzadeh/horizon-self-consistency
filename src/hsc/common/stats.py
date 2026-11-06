"""Bootstrap confidence intervals and small I/O helpers shared by every experiment."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np


def bootstrap_mean(values: np.ndarray, n_boot: int = 2000, seed: int = 0, axis: int = 0) -> dict:
    """Mean and percentile 95% interval, resampling along `axis` (the independent units)."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = values.shape[axis]
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.take(values, idx, axis=axis).mean(axis=axis + 1)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {"mean": values.mean(axis=axis).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}


def bootstrap_ratio(num: np.ndarray, den: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Ratio of means num.mean()/den.mean() with a paired bootstrap over units (axis 0)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    n = len(num)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {"mean": (num.mean(axis=0) / den.mean(axis=0)).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}


def run_metadata() -> dict:
    meta = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        meta["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        meta["gpu"] = None
    return meta


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
