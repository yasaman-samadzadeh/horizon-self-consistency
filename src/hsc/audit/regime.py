"""Regime switching: a Markov-switching AR(1) and its exact (Bayes) forecaster.

A hidden regime r_t in {0, 1} switches with probability p at each step. Given the regime,
x_t = mu_r + phi_r (x_{t-1} - mu_r) + sigma_r e_t with e_t ~ N(0, 1). The exact forecaster runs
the Hamilton filter over the context to get P(r_T | x_1..x_T), then pushes the joint density of
(regime, value) forward on a fine grid. Its h-step forecast therefore averages over every future
regime path, including a switch inside the forecast window, and all its horizons are marginals
of one law. Through hsc.audit.gap.self_consistency_gap it shows only the pipeline's own error.

    python -m hsc.audit.regime --models oracle arfit --sources msar msar_fast msar_p0
    python -m hsc.audit.regime --models chronos-bolt-small chronos-bolt-base timestral-28m --device cuda
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from hsc.audit.baselines import ARFit, TSFMAdapter
from hsc.audit.gap import HORIZONS, self_consistency_gap, summarise
from hsc.audit.windows import CONTEXT, H_MAX
from hsc.common.paths import RESULTS
from hsc.common.quantiles import LEVELS, band_w1, mixture_quantiles
from hsc.common.stats import bootstrap_mean, run_metadata, write_json


@dataclass(frozen=True)
class MSAR:
    """Two-regime Markov-switching AR(1); the regime switches with probability p at each step."""

    mu: tuple[float, float] = (0.0, 3.0)
    phi: tuple[float, float] = (0.9, 0.5)
    sigma: tuple[float, float] = (0.5, 1.5)
    p: float = 0.005
    pi0: tuple[float, float] = (0.5, 0.5)  # law of r_1; (0.5, 0.5) is stationary for symmetric switching

    def params(self):
        return np.array(self.mu), np.array(self.phi), np.array(self.sigma)

    def transition(self) -> np.ndarray:
        return np.array([[1 - self.p, self.p], [self.p, 1 - self.p]])


LAWS = {
    "msar": MSAR(p=0.005),  # regimes last 200 steps on average
    "msar_fast": MSAR(p=0.02),  # 50 steps: a switch inside the 64-step window has probability 0.73
    "msar_p0": MSAR(p=0.0),  # control: the regime is drawn once per window and never changes
}


def simulate_msar(law: MSAR, n: int, length: int, rng: np.random.Generator):
    """n independent windows: values (n, length) and regimes (n, length).

    r_1 ~ pi0 and x_1 ~ N(mu_r, sigma_r^2 / (1 - phi_r^2)), the stationary law of regime r_1,
    which is exactly the start that filter_regimes assumes.
    """
    mu, phi, sigma = law.params()
    r = np.zeros((n, length), dtype=np.int64)
    x = np.zeros((n, length))
    r[:, 0] = rng.random(n) < law.pi0[1]
    x[:, 0] = mu[r[:, 0]] + sigma[r[:, 0]] / np.sqrt(1 - phi[r[:, 0]] ** 2) * rng.standard_normal(n)
    switch = rng.random((n, length)) < law.p
    eps = rng.standard_normal((n, length))
    for t in range(1, length):
        r[:, t] = np.where(switch[:, t], 1 - r[:, t - 1], r[:, t - 1])
        j = r[:, t]
        x[:, t] = mu[j] + phi[j] * (x[:, t - 1] - mu[j]) + sigma[j] * eps[:, t]
    return x, r


def regime_windows(source: str, n: int, seed: int):
    """Contexts (n, CONTEXT), the next H_MAX values, and the regime path of each whole window."""
    x, r = simulate_msar(LAWS[source], n, CONTEXT + H_MAX, np.random.default_rng(seed))
    return x[:, :CONTEXT], x[:, CONTEXT:], r


def _log_normal(x, mean, sd):
    return -0.5 * ((x - mean) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)


def filter_regimes(law: MSAR, x: np.ndarray) -> np.ndarray:
    """Hamilton filter: P(r_T = j | x_1..x_T) for every row of x (B, T); returns (B, 2)."""
    mu, phi, sigma = law.params()
    trans = law.transition()
    ll = _log_normal(x[:, :1], mu, sigma / np.sqrt(1 - phi**2))
    w = np.asarray(law.pi0) * np.exp(ll - ll.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    for t in range(1, x.shape[1]):
        ll = _log_normal(x[:, t : t + 1], mu + phi * (x[:, t - 1 : t] - mu), sigma)
        # predict r_t from r_{t-1}, then weigh by the density of x_t; the row max cancels on normalising
        w = (w @ trans) * np.exp(ll - ll.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
    return w


class RegimeForecaster:
    """Exact Bayes forecaster of an MSAR law, as a callable (contexts, horizon, parent) -> (B, H, 9).

    With w_j = P(r_{T+1} = j | context), the joint density of (r_{T+1}, x_{T+1}) is
    p_1(j, x) = w_j N(x; mu_j + phi_j (x_T - mu_j), sigma_j^2), and each further step is
    p_{k+1}(j, x) = int N(x; mu_j + phi_j (x' - mu_j), sigma_j^2) sum_i P_ij p_k(i, x') dx'.
    The integral is a sum over a uniform grid, which is accurate to rounding error for smooth
    densities that vanish at the grid ends (checked on every call). The k-step forecast is the
    marginal sum_j p_k(j, .); its quantiles come from the trapezoid-rule CDF. parent is unused:
    an extended context c+y is filtered from its start, so its forecast conditions on y.
    """

    def __init__(self, law: MSAR, lo: float = -12.0, hi: float = 18.0, dx: float = 0.01, batch: int = 512):
        self.law, self.dx, self.batch = law, dx, batch
        self.grid = lo + dx * np.arange(int(round((hi - lo) / dx)) + 1)
        mu, phi, sigma = law.params()
        # kernel[j][a, b] = dx * N(grid[b]; mu_j + phi_j (grid[a] - mu_j), sigma_j^2), so next = density @ kernel[j]
        self.kernel = [dx * np.exp(_log_normal(self.grid[None, :], mu[j] + phi[j] * (self.grid[:, None] - mu[j]),
                                               sigma[j])) for j in range(2)]

    def __call__(self, contexts: np.ndarray, horizon: int, parent=None) -> np.ndarray:
        return np.concatenate([self._forecast(contexts[s : s + self.batch], horizon)
                               for s in range(0, len(contexts), self.batch)])

    def _forecast(self, x: np.ndarray, horizon: int) -> np.ndarray:
        mu, phi, sigma = self.law.params()
        trans = self.law.transition()
        w = filter_regimes(self.law, x) @ trans  # P(r_{T+1} = j | x), (B, 2)
        mean = mu + phi * (x[:, -1:] - mu)  # (B, 2)
        dens = w[:, :, None] * np.exp(_log_normal(self.grid, mean[:, :, None], sigma[:, None]))  # (B, 2, G)
        out = np.empty((len(x), horizon, len(LEVELS)))
        for k in range(horizon):
            out[:, k] = self._quantiles(dens.sum(axis=1))
            if k + 1 < horizon:
                mixed = np.einsum("ij,big->bjg", trans, dens)  # sum_i P_ij p_k(i, .)
                dens = np.stack([mixed[:, j] @ self.kernel[j] for j in range(2)], axis=1)
        return out

    def _quantiles(self, f: np.ndarray) -> np.ndarray:
        """Quantiles at LEVELS of densities f (B, G) on the grid: trapezoid CDF, linear inversion."""
        cdf = np.zeros_like(f)
        cdf[:, 1:] = np.cumsum(0.5 * (f[:, 1:] + f[:, :-1]), axis=1) * self.dx
        lost = np.abs(cdf[:, -1] - 1).max()
        if lost > 1e-6:
            raise ValueError(f"forecast mass {lost:.2e} away from 1: widen the grid")
        j = (cdf[:, None, :] < LEVELS[:, None]).sum(axis=-1) - 1  # last grid point below each level, (B, 9)
        c0 = np.take_along_axis(cdf, j, axis=1)
        c1 = np.take_along_axis(cdf, j + 1, axis=1)
        return self.grid[j] + self.dx * (LEVELS - c0) / (c1 - c0)


def one_step_ppf(law: MSAR, contexts: np.ndarray, u: np.ndarray, iters: int = 80) -> np.ndarray:
    """Exact quantiles (n, S) of the one-step forecast, a two-Gaussian mixture, by bisection."""
    mu, phi, sigma = law.params()
    w = filter_regimes(law, contexts) @ law.transition()
    mean = mu + phi * (contexts[:, -1:] - mu)  # (n, 2)
    lo = np.broadcast_to((mean - 10 * sigma).min(axis=1, keepdims=True), (len(contexts), len(u))).copy()
    hi = np.broadcast_to((mean + 10 * sigma).max(axis=1, keepdims=True), (len(contexts), len(u))).copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = (w[:, None, :] * ndtr((mid[:, :, None] - mean[:, None, :]) / sigma)).sum(axis=-1)
        lo, hi = np.where(f < u, mid, lo), np.where(f < u, hi, mid)
    return 0.5 * (lo + hi)


def exact_draw_floor(law: MSAR, contexts: np.ndarray, inner: int) -> dict:
    """The baseline with the one-step values y taken from the exact one-step law, not its 9-quantile
    interpolation.

    Everything else matches self_consistency_gap, so the difference between the two baselines is the
    part caused by taking y from the interpolated one-step quantiles.
    """
    fc = RegimeForecaster(law)
    n = len(contexts)
    direct = fc(contexts, H_MAX)
    y = one_step_ppf(law, contexts, (np.arange(inner) + 0.5) / inner)
    ext = np.concatenate([np.repeat(contexts, inner, axis=0), y.reshape(-1, 1)], axis=1)
    ext_fc = fc(ext, H_MAX - 1).reshape(n, inner, H_MAX - 1, -1)
    d, spread = [], []
    for h in HORIZONS:
        d.append(band_w1(direct[:, h - 1], mixture_quantiles(ext_fc[:, :, h - 2])))
        spread.append(direct[:, h - 1, -1] - direct[:, h - 1, 0])
    return {"D": np.stack(d, axis=1), "spread": np.stack(spread, axis=1)}


def build(name: str, law: MSAR, contexts: np.ndarray, device: str, cache: dict):
    if name == "oracle":
        return RegimeForecaster(law)
    if name == "arfit":
        return ARFit(contexts)
    if name not in cache:
        from hsc.common.tsfm import load_tsfm

        cache[name] = load_tsfm(name, device)
    return TSFMAdapter(cache[name])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="oracle, arfit and/or TSFM names")
    ap.add_argument("--sources", nargs="+", default=list(LAWS))
    ap.add_argument("--n", type=int, default=256, help="contexts per source")
    ap.add_argument("--inner", type=int, default=32, help="evenly spaced one-step quantiles per context")
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(RESULTS / "regime"))
    args = ap.parse_args()

    cache: dict = {}
    for source in args.sources:
        law = LAWS[source]
        contexts, truth, regimes = regime_windows(source, args.n, args.seed)
        posterior = filter_regimes(law, contexts)  # P(r_T | context), for the analysis splits
        for name in args.models:
            started = time.time()
            out = Path(args.out) / source
            out.mkdir(parents=True, exist_ok=True)
            if name == "oracle_exactdraws":
                res = exact_draw_floor(law, contexts, args.inner)
                summary = {"horizons": list(HORIZONS), "D_over_spread": bootstrap_mean(res["D"] / res["spread"]),
                           "model": name, "source": source, "n": len(contexts), "inner": args.inner,
                           "seed": args.seed, "seconds": time.time() - started, **run_metadata()}
                np.savez_compressed(out / f"{name}.npz", D=res["D"], spread=res["spread"])
                write_json(out / f"{name}.json", summary)
                print(f"{source:10s} {name:20s} D/spread by h "
                      f"{dict(zip(HORIZONS, np.round(summary['D_over_spread']['mean'], 4).tolist()))}", flush=True)
                continue
            fc = build(name, law, contexts, args.device, cache)
            res = self_consistency_gap(fc, contexts, truth, args.inner)
            summary = summarise(res)
            summary.update({"model": name, "source": source, "law": asdict(law), "n": len(contexts),
                            "inner": args.inner, "context": CONTEXT, "seed": args.seed,
                            "seconds": time.time() - started, **run_metadata()})
            np.savez_compressed(out / f"{name}.npz", D=res["D"], spread=res["spread"], crps=res["crps"],
                                comp=res["comp"], direct=res["direct"], truth=truth, contexts=contexts,
                                regimes=regimes, posterior=posterior)
            write_json(out / f"{name}.json", summary)
            ds = np.round(summary["D_over_spread"]["mean"], 4)
            print(f"{source:10s} {name:20s} D/spread by h {dict(zip(HORIZONS, ds.tolist()))} "
                  f"({summary['seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
