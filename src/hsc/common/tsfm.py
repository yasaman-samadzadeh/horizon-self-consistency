"""Time-series foundation models behind one interface: quantiles at LEVELS for horizons 1..H.

forecast(contexts, horizon) takes a float array (B, T) and returns (B, horizon, 9).
Models run in float32 so that reduced precision cannot masquerade as a self-consistency gap.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from hsc.common.quantiles import LEVELS

TSFM_REPOS = {
    "chronos-bolt-tiny": "amazon/chronos-bolt-tiny",
    "chronos-bolt-mini": "amazon/chronos-bolt-mini",
    "chronos-bolt-small": "amazon/chronos-bolt-small",
    "chronos-bolt-base": "amazon/chronos-bolt-base",
    "timestral-28m": "mistralai/Timestral-28M",
}


class ChronosBolt:
    """Chronos-Bolt: a patch-based encoder-decoder that outputs 9 quantiles for 64 steps in one shot."""

    def __init__(self, repo: str, device: str, batch_size: int = 1024):
        from chronos import BaseChronosPipeline

        self.pipe = BaseChronosPipeline.from_pretrained(repo, device_map=device, torch_dtype=torch.float32)
        assert np.allclose(self.pipe.quantiles, LEVELS), self.pipe.quantiles
        self.batch_size = batch_size

    def forecast(self, contexts: np.ndarray, horizon: int) -> np.ndarray:
        assert horizon <= 64, "beyond 64 steps Chronos-Bolt feeds back its median (not a direct forecast)"
        out = []
        for start in range(0, len(contexts), self.batch_size):
            batch = torch.tensor(contexts[start : start + self.batch_size], dtype=torch.float32)
            q = self.pipe.predict(batch, prediction_length=horizon, limit_prediction_length=True)
            out.append(q.permute(0, 2, 1).numpy().astype(np.float64))
        return np.concatenate(out, axis=0)


class Timestral:
    """Timestral: patches of 32 steps, 21 quantiles per step; one shot covers up to 1024 steps."""

    def __init__(self, repo: str, device: str, batch_size: int = 512):
        from timestral import TimestralForecaster

        self.model = TimestralForecaster.from_pretrained(repo, device=device)
        knots = list(self.model.quantile_levels)
        self.idx = [int(np.argmin(np.abs(np.asarray(knots) - lv))) for lv in LEVELS]
        assert np.allclose(np.asarray(knots)[self.idx], LEVELS)
        self.batch_size = batch_size

    def forecast(self, contexts: np.ndarray, horizon: int) -> np.ndarray:
        out = []
        for start in range(0, len(contexts), self.batch_size):
            batch = np.asarray(contexts[start : start + self.batch_size], dtype=np.float32)
            q = self.model.forecast(batch, prediction_length=horizon).quantiles  # (b, 1, H, 21)
            out.append(q[:, 0][..., self.idx].float().cpu().numpy().astype(np.float64))
        return np.concatenate(out, axis=0)


def load_tsfm(name: str, device: str):
    started = time.time()
    repo = TSFM_REPOS[name]
    model = ChronosBolt(repo, device) if name.startswith("chronos-bolt") else Timestral(repo, device)
    print(f"loaded {name} on {device} in {time.time() - started:.1f}s", flush=True)
    return model
