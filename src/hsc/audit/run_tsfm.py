"""Main run: the self-consistency gap of the baselines and of Chronos-Bolt / Timestral on every source.

Writes <out>/<source>/<model>.json (summary) and .npz (per-context arrays).

    python -m hsc.audit.run_tsfm --models oracle patchwork arfit
    python -m hsc.audit.run_tsfm --models chronos-bolt-small chronos-bolt-base timestral-28m --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from hsc.audit.baselines import ARFit, ExactForecaster, Patchwork, TSFMAdapter
from hsc.audit.gap import HORIZONS, self_consistency_gap, summarise
from hsc.audit.windows import CONTEXT, REAL, SYNTHETIC, load_windows
from hsc.common.paths import RESULTS
from hsc.common.stats import run_metadata, write_json


def build_forecaster(name: str, source: str, contexts: np.ndarray, device: str, cache: dict):
    if name == "oracle":
        return ExactForecaster(source) if source in SYNTHETIC else None
    if name == "patchwork":
        return Patchwork(source) if source in SYNTHETIC else None
    if name == "arfit":
        return ARFit(contexts)
    if name not in cache:
        from hsc.common.tsfm import load_tsfm

        cache[name] = load_tsfm(name, device)
    return TSFMAdapter(cache[name])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sources", nargs="+", default=list(SYNTHETIC) + list(REAL))
    ap.add_argument("--n", type=int, default=256, help="contexts per source")
    ap.add_argument("--inner", type=int, default=32, help="evenly spaced one-step quantiles per context")
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(RESULTS / "main"))
    args = ap.parse_args()

    cache: dict = {}
    for source in args.sources:
        contexts, truth = load_windows(source, args.n, args.seed)
        for name in args.models:
            fc = build_forecaster(name, source, contexts, args.device, cache)
            if fc is None:
                continue
            started = time.time()
            res = self_consistency_gap(fc, contexts, truth, args.inner)
            summary = summarise(res)
            summary.update({"model": name, "source": source, "n": len(contexts), "inner": args.inner,
                            "context": CONTEXT, "seconds": time.time() - started, **run_metadata()})
            out = Path(args.out) / source
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / f"{name}.npz", D=res["D"], spread=res["spread"], crps=res["crps"],
                                comp=res["comp"], direct=res["direct"], truth=truth)
            write_json(out / f"{name}.json", summary)
            ds = np.round(summary["D_over_spread"]["mean"], 4)
            print(f"{source:8s} {name:20s} D/spread by h {dict(zip(HORIZONS, ds))} "
                  f"({summary['seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
