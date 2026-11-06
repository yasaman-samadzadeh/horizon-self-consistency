"""Mechanism runs: keep every inner forecast to see where a model's self-consistency gap comes from.

hsc.audit.gap.self_consistency_gap mixes the model's (h-1)-step forecasts from the extended contexts
c+y_m and returns only the mixture. Here a Recorder wraps the forecaster, so the identical
computation also yields each one-step value y_m and the forecast made from c+y_m. Two extra passes
check whether the gap is tied to the change of input length that appending y_m implies (the
input-length control):
  slide:      the extended contexts drop their oldest point, so they keep the base length (500);
  drop-first: the direct forecast from c without its oldest point, compared with the one from c.

Sources are those of hsc.audit.windows, the switching laws of hsc.audit.regime, and "ar1bump<A>": the ar1
windows with a smooth bump of height A over the first half of each context. The bump raises the
context's standard deviation (the scale Chronos-Bolt normalises by) while the last 250 points,
and so the exact AR(1) forecast, stay unchanged (the COVID probe).

    python -m hsc.audit.mechanism --models chronos-bolt-small chronos-bolt-base timestral-28m arfit \
        --sources covid epf weather ar1 --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from hsc.audit.gap import HORIZONS, self_consistency_gap, summarise
from hsc.audit.regime import LAWS, RegimeForecaster, regime_windows
from hsc.audit.run_tsfm import build_forecaster
from hsc.audit.windows import CONTEXT, H_MAX, load_windows
from hsc.common.paths import RESULTS
from hsc.common.quantiles import band_w1
from hsc.common.stats import bootstrap_mean, run_metadata, write_json

INNER_IDX = [h - 2 for h in HORIZONS]  # index of the (h-1)-step inner forecast that D_h uses


class Recorder:
    """Wraps a forecaster and keeps the inputs and outputs of every call."""

    def __init__(self, forecaster):
        self.forecaster, self.calls = forecaster, []

    def __call__(self, contexts, horizon, parent):
        out = self.forecaster(contexts, horizon, parent)
        self.calls.append((contexts, out))
        return out


class Slide:
    """Gives the forecaster only the last `length` points of each context."""

    def __init__(self, forecaster, length: int):
        self.forecaster, self.length = forecaster, length

    def __call__(self, contexts, horizon, parent):
        return self.forecaster(contexts[:, -self.length :], horizon, parent)


BUMP_LENGTH = 250


def bump_windows(amplitude: float, n: int, seed: int):
    """The ar1 windows with A sin^2(pi t / 250) added to the first 250 context points."""
    contexts, truth = load_windows("ar1", n, seed)
    t = np.arange(BUMP_LENGTH)
    contexts = contexts.copy()
    contexts[:, :BUMP_LENGTH] += amplitude * np.sin(np.pi * t / BUMP_LENGTH) ** 2
    return contexts, truth


def windows_and_forecaster(source: str, name: str, n: int, seed: int, device: str, cache: dict):
    if source in LAWS:
        contexts, truth, _ = regime_windows(source, n, seed)
        if name == "oracle":
            return contexts, truth, RegimeForecaster(LAWS[source])
        return contexts, truth, build_forecaster(name, "msar", contexts, device, cache)
    if source.startswith("ar1bump"):
        contexts, truth = bump_windows(float(source[len("ar1bump"):]), n, seed)
        return contexts, truth, build_forecaster(name, "ar1", contexts, device, cache)
    contexts, truth = load_windows(source, n, seed)
    return contexts, truth, build_forecaster(name, source, contexts, device, cache)


def defect_with_draws(forecaster, contexts: np.ndarray, truth: np.ndarray, inner: int) -> dict:
    """self_consistency_gap plus the one-step values y (n, M) and the inner forecasts at the horizons D uses."""
    rec = Recorder(forecaster)
    res = self_consistency_gap(rec, contexts, truth, inner)
    (_, direct), (ext, ext_fc) = rec.calls
    n = len(contexts)
    assert np.array_equal(direct, res["direct"])
    res["y"] = ext[:, -1].reshape(n, inner)
    res["inner_fc"] = ext_fc.reshape(n, inner, H_MAX - 1, -1)[:, :, INNER_IDX]  # (n, M, len(HORIZONS), 9)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sources", nargs="+", default=["covid", "epf", "weather", "ar1"])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--inner", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(RESULTS / "mechanism"))
    ap.add_argument("--main-run", default=str(RESULTS / "main"),
                    help="main-run results (hsc.audit.run_tsfm), to check reproduction")
    args = ap.parse_args()

    cache: dict = {}
    for source in args.sources:
        for name in args.models:
            contexts, truth, fc = windows_and_forecaster(source, name, args.n, args.seed, args.device, cache)
            n = len(contexts)
            if fc is None:
                continue
            started = time.time()
            res = defect_with_draws(fc, contexts, truth, args.inner)
            slid = self_consistency_gap(Slide(fc, CONTEXT), contexts, truth, args.inner)
            dropped = fc(contexts[:, 1:], H_MAX, np.arange(n))
            idx = np.array(HORIZONS) - 1
            shift = band_w1(res["direct"][:, idx], dropped[:, idx])  # (n, len(HORIZONS))
            summary = summarise(res)
            summary["slide_D_over_spread"] = bootstrap_mean(slid["D"] / slid["spread"])
            summary["dropfirst_over_spread"] = bootstrap_mean(shift / res["spread"])
            main_run = Path(args.main_run) / source / f"{name}.npz"
            if main_run.exists():
                d_main = np.load(main_run)["D"]
                summary["main_run_max_abs_diff_D_over_spread"] = float(np.abs((res["D"] - d_main) / res["spread"]).max())
            summary.update({"model": name, "source": source, "n": n, "inner": args.inner, "context": CONTEXT,
                            "seed": args.seed, "seconds": time.time() - started, **run_metadata()})
            out = Path(args.out) / source
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / f"{name}.npz", D=res["D"], spread=res["spread"], crps=res["crps"],
                                comp=res["comp"], direct=res["direct"], truth=truth, y=res["y"],
                                inner_fc=res["inner_fc"].astype(np.float32), D_slide=slid["D"],
                                comp_slide=slid["comp"], dropfirst=shift, horizons=np.array(HORIZONS))
            write_json(out / f"{name}.json", summary)
            show = {k: np.round(summary[k]["mean"], 4).tolist() for k in
                    ("D_over_spread", "slide_D_over_spread", "dropfirst_over_spread")}
            print(f"{source:8s} {name:20s} {show} main-run diff {summary.get('main_run_max_abs_diff_D_over_spread')} "
                  f"({summary['seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
