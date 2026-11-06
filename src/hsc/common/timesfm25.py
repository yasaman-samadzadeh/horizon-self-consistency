"""TimesFM 2.5 (200M, PyTorch) behind the tsfm.py interface: forecast(contexts, horizon) -> (B, H, 9).

Needs the timesfm package (3.0.2, the optional `timesfm` extra: `uv sync --extra timesfm`). What its code
(timesfm/timesfm_2p5/) does with our inputs:
- each series is left-padded with masked zeros to max_context and cut into 32-step patches, so the
  last patch ends at the last observation; fully padded patches are excluded from attention;
- one forward pass of the causal decoder gives, from the last patch, a 128-step output patch with
  10 channels (0 = mean, 1..9 = the 0.1..0.9 quantiles, 5 = median) and a 1024-step quantile head;
- steps beyond max_horizon (rounded up to 128) would be decoded autoregressively by feeding back the
  median. With max_horizon = 128, every horizon up to 64 comes from the one direct output patch.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from hsc.common.quantiles import LEVELS

REPO = "google/timesfm-2.5-200m-pytorch"
OUTPUT_PATCH = 128


def to_levels(quantile_forecast: np.ndarray) -> np.ndarray:
    """TimesFM's (B, H, 10) output (mean, then q0.1..q0.9) -> (B, H, 9) quantiles at LEVELS."""
    assert quantile_forecast.shape[-1] == len(LEVELS) + 1, quantile_forecast.shape
    return np.asarray(quantile_forecast[..., 1:], dtype=np.float64)


class TimesFM25:
    """TimesFM 2.5 with the README's forecasting flags, except infer_is_positive (see run_timesfm.py).

    Weights and activations are float32; TF32 matmuls (the README's "high" precision) are disabled.
    """

    def __init__(self, max_context: int = 512, batch_size: int = 256, infer_is_positive: bool = False,
                 max_horizon: int = OUTPUT_PATCH, force_flip_invariance: bool = True, repo: str = REPO):
        import timesfm
        import torch

        torch.set_float32_matmul_precision("highest")
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo, torch_compile=False)
        assert np.allclose(self.model.model.config.quantiles, LEVELS)
        self.model.compile(timesfm.ForecastConfig(
            max_context=max_context, max_horizon=max_horizon, normalize_inputs=True,
            use_continuous_quantile_head=True, force_flip_invariance=force_flip_invariance,
            infer_is_positive=infer_is_positive, fix_quantile_crossing=True, per_core_batch_size=batch_size))
        self.config = dataclasses.asdict(self.model.forecast_config)  # after compile() rounds the sizes
        self.device = str(self.model.model.device)

    def forecast(self, contexts: np.ndarray, horizon: int) -> np.ndarray:
        assert horizon <= self.config["max_horizon"], horizon
        assert contexts.shape[1] <= self.config["max_context"], "contexts would be truncated"
        _, q = self.model.forecast(horizon=horizon, inputs=list(np.asarray(contexts, dtype=np.float64)))
        return to_levels(q)
