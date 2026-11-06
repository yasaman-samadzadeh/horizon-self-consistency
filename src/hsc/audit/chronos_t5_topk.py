"""Chronos-T5 sampler sweep: how wide are Chronos-T5's forecasts on the AR sources, and is top-k sampling the reason?

Chronos-T5 scales a context by its mean |x| and maps it to 4,093 uniform bins on [-15, 15], so on
these AR series one noise standard deviation spans about 75 bins. Its default sampler keeps only the
50 most likely tokens per step. This compares the 80% interval width of its h-step samples with the
true law's, for several top_k (4096 keeps every token).

    python -m hsc.audit.chronos_t5_topk --device cuda   # writes results/controls/t5_topk_check.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hsc.audit.baselines import ExactForecaster
from hsc.audit.chronos_t5 import T5_REPOS, ChronosT5Sampler, sample_quantiles
from hsc.audit.windows import load_windows
from hsc.common.paths import RESULTS
from hsc.common.quantiles import LEVELS
from hsc.common.stats import bootstrap_mean, run_metadata, write_json

CHECK_H = (1, 2, 8, 16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["ar1", "ar2"])
    ap.add_argument("--models", nargs="+", default=["chronos-t5-small"])
    ap.add_argument("--top-k", nargs="+", type=int, default=[50, 200, 1000, 4096])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(RESULTS / "controls" / "t5_topk_check.json"))
    args = ap.parse_args()

    out = {"config": vars(args) | run_metadata(), "horizons": list(CHECK_H), "results": {}}
    for name in args.models:
        sampler = ChronosT5Sampler.load(T5_REPOS[name], args.device)
        for source in args.sources:
            contexts, _ = load_windows(source, args.n, args.seed)
            mean, sd = ExactForecaster(source).law.moments(contexts, max(CHECK_H))
            idx = [h - 1 for h in CHECK_H]
            oracle_width = 2 * 1.2815515655446004 * sd[:, idx]
            bins_per_sd = 1 / (30 / 4092 * np.abs(contexts).mean(axis=1))
            for k in args.top_k:
                sampler.top_k = k
                q = sample_quantiles(sampler.samples(contexts, max(CHECK_H), args.samples, seed=1))[:, idx]
                width = (q[..., -1] - q[..., 0]) / oracle_width
                median_err = np.abs(q[..., LEVELS.tolist().index(0.5)] - mean[:, idx]) / oracle_width
                key = f"{name}/{source}/top_k={k}"
                out["results"][key] = {"width_over_oracle": bootstrap_mean(width),
                                       "abs_median_error_over_oracle_width": bootstrap_mean(median_err),
                                       "bins_per_noise_sd": float(bins_per_sd.mean())}
                print(f"{key:40s} 80% width / oracle by h {dict(zip(CHECK_H, np.round(width.mean(0), 3).tolist()))}"
                      f"  |median - mean| / oracle width {dict(zip(CHECK_H, np.round(median_err.mean(0), 3).tolist()))}",
                      flush=True)
    write_json(Path(args.out), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
