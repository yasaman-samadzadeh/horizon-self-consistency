"""Chronos-T5 sample pipeline: a forecaster that samples step by step, against direct ones.

Chronos-T5 draws sample paths token by token, so its direct h-step forecast is the rollout of its
own one-step sampler. The chained forecast still differs from it in how y is read: inside a sample
path the decoder reads y in the scale of c, while the chained forecast re-reads c+y through the
encoder after re-scaling (mean |x| of the 501 values). Its forecasts are samples, so every quantile
is estimated from samples here, and the exact forecaster goes through the same sample-based pipeline
(same sample sizes) to give this pipeline's own error. A direct quantile model can be put through it
too, by drawing from its quantile function ("<model>-samples"), which compares the two kinds of model
like with like.

    python -m hsc.audit.chronos_t5 --sources ar1 ar2 --models oracle-samples chronos-t5-small --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from hsc.audit.baselines import ExactForecaster, TSFMAdapter
from hsc.audit.gap import HORIZONS, summarise
from hsc.audit.windows import CONTEXT, H_MAX, SYNTHETIC, load_windows
from hsc.common.paths import RESULTS
from hsc.common.quantiles import LEVELS, band_w1, mixture_quantiles, ppf, quantile_crps
from hsc.common.stats import run_metadata, write_json

T5_REPOS = {"chronos-t5-small": "amazon/chronos-t5-small", "chronos-t5-mini": "amazon/chronos-t5-mini"}


class ChronosT5Sampler:
    """Chronos-T5 sample paths, drawn in predict calls of at most max_sequences sequences.

    generate() copies the encoder output and the cross-attention cache once per returned sample,
    so memory grows with contexts x samples in one call (1,024 sequences ran out of 16 GB).
    top_k=None keeps the model's own setting (50 of 4,096 tokens); 4096 samples every token.
    """

    def __init__(self, pipe, max_sequences: int = 128, log_every: int = 0, top_k: int | None = None):
        self.pipe = pipe
        self.max_sequences = max_sequences
        self.log_every = log_every
        self.top_k = top_k
        self.calls = 0

    @classmethod
    def load(cls, repo: str, device: str, max_sequences: int = 128, log_every: int = 0, top_k: int | None = None):
        from chronos import BaseChronosPipeline

        pipe = BaseChronosPipeline.from_pretrained(repo, device_map=device, torch_dtype=torch.float32)
        return cls(pipe, max_sequences, log_every, top_k)

    def samples(self, contexts: np.ndarray, horizon: int, n_samples: int, seed: int) -> np.ndarray:
        """(B, S, H) sample paths."""
        torch.manual_seed(seed)
        per_call = min(n_samples, self.max_sequences)
        assert n_samples % per_call == 0, (n_samples, per_call)
        rows = max(1, self.max_sequences // per_call)
        total = -(-len(contexts) // rows) * (n_samples // per_call)
        out = np.empty((len(contexts), n_samples, horizon))
        started, done = time.time(), 0
        for start in range(0, len(contexts), rows):
            ctx = torch.tensor(contexts[start : start + rows], dtype=torch.float32)
            for k in range(0, n_samples, per_call):
                s = self.pipe.predict(ctx, prediction_length=horizon, num_samples=per_call, top_k=self.top_k,
                                      limit_prediction_length=True)
                out[start : start + rows, k : k + per_call] = s.numpy()
                done += 1
                if self.log_every and done % self.log_every == 0:
                    rate = (time.time() - started) / done
                    print(f"    {done}/{total} calls, {rate:.2f}s per call, {rate * (total - done) / 60:.0f} min left",
                          flush=True)
        self.calls += done
        return out


class ExactSampler:
    """Draws from the true AR law, for the self-consistent baseline of the sample-based pipeline."""

    def __init__(self, source: str):
        self.exact = ExactForecaster(source)

    def samples(self, contexts, horizon, n_samples, seed):
        mean, sd = self.exact.law.moments(contexts, horizon)
        rng = np.random.default_rng(seed)
        # only each horizon's marginal law is used, so horizons are drawn independently
        return mean[:, None, :] + sd[:, None, :] * rng.standard_normal((len(contexts), n_samples, horizon))


class QuantileSampler:
    """Draws from a quantile forecaster's own quantile function (ppf), horizon by horizon.

    forecaster follows the hsc.audit.baselines interface f(contexts, horizon, parent) and must ignore parent.
    """

    def __init__(self, forecaster):
        self.forecaster = forecaster

    def samples(self, contexts, horizon, n_samples, seed):
        q = self.forecaster(contexts, horizon, None)  # (B, H, 9)
        u = np.random.default_rng(seed).random((len(contexts), horizon, n_samples))
        return np.moveaxis(ppf(q, u), -1, 1)


def sample_quantiles(samples: np.ndarray) -> np.ndarray:
    """(B, S, H) samples -> (B, H, 9) empirical quantiles at LEVELS."""
    return np.moveaxis(np.quantile(samples, LEVELS, axis=1), 0, -1)


def sample_gap(sampler, contexts: np.ndarray, truth: np.ndarray, inner: int, s_direct: int,
                  s_inner: int) -> dict:
    """hsc.audit.gap.self_consistency_gap with every forecast's quantiles estimated from samples."""
    n = len(contexts)
    t0 = time.time()
    direct = sample_quantiles(sampler.samples(contexts, H_MAX, s_direct, seed=1))  # (n, H, 9)
    u = np.broadcast_to((np.arange(inner) + 0.5) / inner, (n, inner))
    y = ppf(direct[:, 0, :], u)  # evenly spaced quantiles of the estimated one-step forecast
    ext = np.concatenate([np.repeat(contexts, inner, axis=0), y.reshape(-1, 1)], axis=1)
    ext_q = sample_quantiles(sampler.samples(ext, H_MAX - 1, s_inner, seed=2)).reshape(n, inner, H_MAX - 1, -1)
    t_model = time.time() - t0
    rows = {"D": [], "spread": [], "crps": [], "comp": []}
    for h in HORIZONS:
        comp = mixture_quantiles(ext_q[:, :, h - 2, :])
        rows["D"].append(band_w1(direct[:, h - 1, :], comp))
        rows["spread"].append(direct[:, h - 1, -1] - direct[:, h - 1, 0])
        rows["crps"].append(quantile_crps(direct[:, h - 1, :], truth[:, h - 1]))
        rows["comp"].append(comp)
    out = {k: np.stack(v, axis=1) for k, v in rows.items()}
    out["direct"] = direct
    out["seconds_model"] = t_model
    return out


def build_sampler(name: str, source: str, args, cache: dict):
    if name == "oracle-samples":
        return ExactSampler(source) if source in SYNTHETIC else None
    if name not in cache:
        if name in T5_REPOS:
            cache[name] = ChronosT5Sampler.load(T5_REPOS[name], args.device, args.max_sequences, args.log_every,
                                                args.top_k)
        elif name.endswith("-samples"):
            from hsc.common.tsfm import load_tsfm

            cache[name] = QuantileSampler(TSFMAdapter(load_tsfm(name.removesuffix("-samples"), args.device)))
        else:
            raise ValueError(f"unknown model {name}")
    return cache[name]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["ar1", "ar2"])
    ap.add_argument("--models", nargs="+", default=["oracle-samples", "chronos-t5-small"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--inner", type=int, default=16)
    ap.add_argument("--s-direct", type=int, default=4096)
    ap.add_argument("--s-inner", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-sequences", type=int, default=128, help="contexts x samples per Chronos-T5 call")
    ap.add_argument("--log-every", type=int, default=256, help="Chronos-T5 progress line every this many calls")
    ap.add_argument("--top-k", type=int, default=None, help="Chronos-T5 sampling; default: the model's own (50)")
    ap.add_argument("--out", default=str(RESULTS / "controls"))
    args = ap.parse_args()

    cache: dict = {}
    for source in args.sources:
        contexts, truth = load_windows(source, args.n, args.seed)
        for name in args.models:
            sampler = build_sampler(name, source, args, cache)
            if sampler is None:
                continue
            started = time.time()
            res = sample_gap(sampler, contexts, truth, args.inner, args.s_direct, args.s_inner)
            summary = summarise(res)
            summary.update({"model": name, "source": source, "n": len(contexts), "inner": args.inner,
                            "context": CONTEXT, "seconds": time.time() - started, "pipeline": "samples",
                            "s_direct": args.s_direct, "s_inner": args.s_inner, "seed": args.seed,
                            **run_metadata()})
            if isinstance(sampler, ChronosT5Sampler):
                top_k = sampler.top_k or sampler.pipe.model.config.top_k
                summary.update({"repo": T5_REPOS[name], "max_sequences": args.max_sequences, "top_k": top_k,
                                "temperature": sampler.pipe.model.config.temperature})
                if sampler.top_k is not None:
                    name = f"{name}-topk{sampler.top_k}"
                    summary["model"] = name
            out = Path(args.out) / source
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / f"{name}.npz", D=res["D"], spread=res["spread"], crps=res["crps"],
                                comp=res["comp"], direct=res["direct"], truth=truth)
            write_json(out / f"{name}.json", summary)
            ds = np.round(summary["D_over_spread"]["mean"], 4)
            print(f"{source:8s} {name:28s} D/spread by h {dict(zip(HORIZONS, ds.tolist()))} "
                  f"({summary['seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
