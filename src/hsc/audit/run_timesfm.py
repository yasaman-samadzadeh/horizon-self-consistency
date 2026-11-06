"""TimesFM 2.5 through the same windows, gap and summary as hsc.audit.run_tsfm.

The forecasting flags are those of the TimesFM README (normalize_inputs, continuous quantile head, flip
invariance, fix_quantile_crossing), except that infer_is_positive is off by default. That flag clamps the
forecast at 0 only when the whole context is non-negative, and the chained forecast appends values from
the model's own one-step forecast (whose lower tail can cross 0), so it can switch on for a base context
and off for some of its extended contexts. --infer-is-positive runs the README setting exactly.
--no-flip turns off force_flip_invariance, the average of f(x) and -f(-x) over two forward passes.

Needs the timesfm package (uv sync --extra timesfm):
    python -m hsc.audit.run_timesfm --out results/controls
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from hsc.audit.baselines import TSFMAdapter
from hsc.audit.gap import HORIZONS, self_consistency_gap, summarise
from hsc.audit.windows import CONTEXT, REAL, SYNTHETIC, load_windows
from hsc.common.paths import RESULTS
from hsc.common.stats import run_metadata, write_json
from hsc.common.timesfm25 import REPO, TimesFM25


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(SYNTHETIC) + list(REAL))
    ap.add_argument("--n", type=int, default=256, help="contexts per source")
    ap.add_argument("--inner", type=int, default=32, help="evenly spaced one-step quantiles per context")
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--infer-is-positive", action="store_true")
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default=str(RESULTS / "controls"))
    args = ap.parse_args()

    name = "timesfm-2.5-200m" + ("-pos" if args.infer_is_positive else "") + ("-noflip" if args.no_flip else "")
    started = time.time()
    model = TimesFM25(batch_size=args.batch, infer_is_positive=args.infer_is_positive,
                      force_flip_invariance=not args.no_flip)
    print(f"loaded {name} on {model.device} in {time.time() - started:.1f}s; {model.config}", flush=True)
    for source in args.sources:
        contexts, truth = load_windows(source, args.n, args.seed)
        started = time.time()
        res = self_consistency_gap(TSFMAdapter(model), contexts, truth, args.inner)
        summary = summarise(res)
        summary.update({"model": name, "source": source, "n": len(contexts), "inner": args.inner,
                        "context": CONTEXT, "seconds": time.time() - started, "seed": args.seed, "repo": REPO,
                        "forecast_config": model.config, **run_metadata()})
        out = Path(args.out) / source
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{name}.npz", D=res["D"], spread=res["spread"], crps=res["crps"],
                            comp=res["comp"], direct=res["direct"], truth=truth)
        write_json(out / f"{name}.json", summary)
        ds = np.round(summary["D_over_spread"]["mean"], 4)
        print(f"{source:8s} {name:20s} D/spread by h {dict(zip(HORIZONS, ds.tolist()))} "
              f"({summary['seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
