"""Rigour checks on the real sources (electricity, COVID, weather): intervals by whole series, and a
non-Gaussian (residual-noise) self-consistent baseline.

The main run's 95% intervals resample the 256 windows of a source as if they were independent, but windows
cut from one series overlap. `report` recomputes every real-source interval by resampling whole series (a
cluster bootstrap), using the series recovered by hsc.audit.mechanism_report.window_origins, for every stored
direct model and baseline ("floor" below).

`floors` runs new self-consistent baselines through the unchanged hsc.audit.gap.self_consistency_gap on the
main run's windows (n = 256, inner = 32, seed 20261104):
  - ARResid: the AR(p) of hsc.audit.baselines.ARFit, fitted per base context and held fixed for its extended contexts,
    but with i.i.d. noise resampled from that context's own centred residuals instead of Gaussian noise;
  - the Gaussian ARFit with p = 2 and p = 24.

    python -m hsc.audit.rigour floors --sources epf --workers 16     # about 4 min per source on 16 CPUs
    python -m hsc.audit.rigour report                                 # tables.md and summary.json
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, skew, spearmanr

from hsc.audit.baselines import ARFit
from hsc.audit.gap import HORIZONS, self_consistency_gap, summarise
from hsc.audit.mechanism_report import cluster_bootstrap, window_origins
from hsc.audit.windows import CONTEXT, H_MAX, load_windows
from hsc.common.paths import RESULTS
from hsc.common.processes import ARForecaster, companion, fit_ar
from hsc.common.quantiles import LEVELS
from hsc.common.stats import bootstrap_mean, run_metadata, write_json

SEED = 20261104
N_CONTEXTS = 256
INNER = 32
SOURCES = ("epf", "covid", "weather")
OUT = RESULTS / "rigour"
MAIN_PATHS = 9_000_000
CHECK_PATHS = 2_250_000
# model -> results folder holding <source>/<model>.npz
MODELS = {"chronos-bolt-tiny": "controls", "chronos-bolt-mini": "controls", "chronos-bolt-small": "main",
          "chronos-bolt-base": "main", "timestral-28m": "main", "timesfm-2.5-200m": "controls"}
EXTRA_MODELS = {"timesfm-2.5-200m-noflip": "controls", "timesfm-2.5-200m-pos": "controls"}
FLOORS = {"arfit": "main", "arresid-p8": "rigour", "arresid-p8-check": "rigour", "arfit-p2": "rigour",
          "arfit-p24": "rigour"}
LABEL = {"chronos-bolt-tiny": "Chronos-Bolt tiny", "chronos-bolt-mini": "Chronos-Bolt mini",
         "chronos-bolt-small": "Chronos-Bolt small", "chronos-bolt-base": "Chronos-Bolt base",
         "timestral-28m": "Timestral-28M", "timesfm-2.5-200m": "TimesFM 2.5",
         "timesfm-2.5-200m-noflip": "TimesFM 2.5, no flip", "timesfm-2.5-200m-pos": "TimesFM 2.5, infer_is_positive",
         "arfit": "Gaussian AR(8)", "arresid-p8": "residual-noise AR(8)",
         "arresid-p8-check": "residual-noise AR(8), 2.25M paths, other seed", "arfit-p2": "Gaussian AR(2)",
         "arfit-p24": "Gaussian AR(24)"}


# ---------------------------------------------------------------- the residual-noise AR floor
def ar_residuals(contexts: np.ndarray, phi: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """In-sample one-step residuals of each row's fitted AR(p), with the same lags as processes.fit_ar."""
    p = phi.shape[-1]
    t = contexts.shape[1]
    xc = contexts - mu[:, None]
    lags = np.stack([xc[:, p - 1 - i : t - 1 - i] for i in range(p)], axis=-1)  # (B, T-p, p)
    return xc[:, p:] - np.einsum("btp,bp->bt", lags, phi)


def impulse_response(phi: np.ndarray, horizon: int) -> np.ndarray:
    """psi_j = (A^j)_00 for j < horizon, one row per law: x_{t+h} minus its mean is sum_{j<h} psi_j e_{t+h-j}."""
    b, p = phi.shape
    a = np.zeros((b, p, p))
    a[:, 0, :] = phi
    if p > 1:
        a[:, 1:, :-1] = np.eye(p - 1)
    power = np.broadcast_to(np.eye(p), (b, p, p)).copy()
    psi = np.empty((b, horizon))
    for j in range(horizon):
        psi[:, j] = power[:, 0, 0]
        power = a @ power
    return psi


def noise_sum_quantiles(pool: np.ndarray, psi: np.ndarray, n_paths: int, seed) -> np.ndarray:
    """Quantiles at LEVELS of S_h = sum_{j<h} psi_j e_j for h = 1..len(psi), e_j i.i.d. from pool.

    h = 1 is exact (S_1 = psi_0 e is discrete, with one atom per pool value). For h >= 2 they are Monte Carlo
    estimates from one set of paths that is extended by one term per horizon.
    """
    rng = np.random.default_rng(seed)
    total = np.zeros(n_paths)
    out = np.empty((len(psi), len(LEVELS)))
    for h, w in enumerate(psi):
        total += w * pool[rng.integers(0, len(pool), n_paths)]
        out[h] = np.quantile(total, LEVELS)
    out[0] = np.quantile(psi[0] * pool, LEVELS, method="inverted_cdf")
    return out


def _noise_sum_quantiles(args) -> np.ndarray:
    return noise_sum_quantiles(*args)


class ARResid:
    """AR(p) fitted to each base context and held fixed, with i.i.d. noise drawn from that context's residuals.

    Self-consistent by construction (one linear law, i.i.d. noise). The h-step forecast is the deterministic mean
    recursion of ARForecaster.moments plus the law of S_h = sum_{j<h} psi_j e_j. The quantiles of S_h are
    computed once per base context by Monte Carlo and reused for its extended contexts, as ExactForecaster does for
    t(3) noise. `pools` (one row per base context) replaces the residuals; the tests use it.
    """

    def __init__(self, contexts: np.ndarray, p: int = 8, n_paths: int = MAIN_PATHS, seed: int = 1,
                 workers: int = 1, pools: np.ndarray | None = None):
        self.phi, self.sigma, self.mu = fit_ar(contexts, p)
        resid = ar_residuals(contexts, self.phi, self.mu) if pools is None else np.asarray(pools, dtype=np.float64)
        self.pools = resid - resid.mean(axis=1, keepdims=True)
        self.psi = impulse_response(self.phi, H_MAX)
        self.law = ARForecaster(np.zeros(p), 1.0)
        jobs = [(self.pools[i], self.psi[i], n_paths, [seed, i]) for i in range(len(contexts))]
        if workers > 1:
            with ProcessPoolExecutor(workers) as ex:
                qs = list(ex.map(_noise_sum_quantiles, jobs, chunksize=2))
        else:
            qs = [noise_sum_quantiles(*job) for job in jobs]
        self.noise_q = np.stack(qs)  # (B, H_MAX, 9)

    def __call__(self, contexts, horizon, parent):
        mean, _ = self.law.moments(contexts, horizon, phi=self.phi[parent], sigma=self.sigma[parent],
                                   mu=self.mu[parent])
        return mean[..., None] + self.noise_q[parent, :horizon]


def run_floors(sources, workers: int, n_paths: int, check_paths: int, out: Path = OUT) -> None:
    for source in sources:
        contexts, truth = load_windows(source, N_CONTEXTS, SEED)
        _, ctx, _ = window_origins(source, N_CONTEXTS)
        assert np.array_equal(ctx, contexts), source
        builds = {"arfit-p2": lambda: ARFit(contexts, p=2), "arfit-p24": lambda: ARFit(contexts, p=24),
                  "arresid-p8": lambda: ARResid(contexts, 8, n_paths, seed=1, workers=workers),
                  "arresid-p8-check": lambda: ARResid(contexts, 8, check_paths, seed=2, workers=workers)}
        for name, build in builds.items():
            started = time.time()
            fc = build()
            res = self_consistency_gap(fc, contexts, truth, INNER)
            summary = summarise(res)
            summary.update({"model": name, "source": source, "n": len(contexts), "inner": INNER, "context": CONTEXT,
                            "seed": SEED, "seconds": time.time() - started, **run_metadata()})
            extra = {}
            if isinstance(fc, ARResid):
                summary.update({"p": 8, "n_paths": n_paths if name == "arresid-p8" else check_paths,
                                "mc_seed": 1 if name == "arresid-p8" else 2})
                extra = {"noise_q": fc.noise_q, "resid_skew": skew(fc.pools, axis=1),
                         "resid_exkurt": kurtosis(fc.pools, axis=1), "sigma": fc.sigma}
            dest = out / source
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / f"{name}.npz", D=res["D"], spread=res["spread"], crps=res["crps"],
                                comp=res["comp"], direct=res["direct"], truth=truth, **extra)
            write_json(dest / f"{name}.json", summary)
            ds = np.round(summary["D_over_spread"]["mean"], 5)
            print(f"{source:8s} {name:18s} D/spread by h {dict(zip(HORIZONS, ds))} ({summary['seconds']:.0f}s)",
                  flush=True)


# ---------------------------------------------------------------- resampling by whole series
def cluster_indices(cluster: np.ndarray, n_boot: int = 2000, seed: int = 0, within: bool = False):
    """Index arrays of a cluster bootstrap. Whole series are resampled; with within=True the windows inside each
    drawn series are resampled too (two-stage). within=False draws exactly as mechanism_report.cluster_bootstrap."""
    rng = np.random.default_rng(seed)
    ids = np.unique(cluster)
    members = [np.nonzero(cluster == c)[0] for c in ids]
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        if within:
            yield np.concatenate([members[k][rng.integers(0, len(members[k]), len(members[k]))] for k in pick])
        else:
            yield np.concatenate([members[k] for k in pick])


def _interval(boots: np.ndarray, level: float = 95.0) -> tuple[list, list]:
    lo, hi = np.percentile(boots, [50 - level / 2, 50 + level / 2], axis=0)
    return lo.tolist(), hi.tolist()


def cluster_mean(values: np.ndarray, cluster: np.ndarray, within: bool, n_boot: int = 2000, seed: int = 0) -> dict:
    boots = np.array([values[i].mean(axis=0) for i in cluster_indices(cluster, n_boot, seed, within)])
    lo, hi = _interval(boots)
    return {"mean": values.mean(axis=0).tolist(), "lo": lo, "hi": hi}


def paired_ratio(num: np.ndarray, den: np.ndarray, cluster: np.ndarray, within: bool = True, n_boot: int = 2000,
                 seed: int = 0) -> dict:
    """num.mean(0) / den.mean(0) with a paired cluster-bootstrap 95% interval (one draw used for both)."""
    boots = np.array([num[i].mean(axis=0) / den[i].mean(axis=0)
                      for i in cluster_indices(cluster, n_boot, seed, within)])
    lo, hi = _interval(boots)
    return {"mean": (num.mean(axis=0) / den.mean(axis=0)).tolist(), "lo": lo, "hi": hi}


def cluster_spearman(a: np.ndarray, b: np.ndarray, cluster: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Spearman rho with 95% and 99% intervals from resampling whole series (the draws of spearman_ci)."""
    boots = np.array([spearmanr(a[i], b[i])[0] for i in cluster_indices(cluster, n_boot, seed)])
    lo99, lo95, hi95, hi99 = np.nanpercentile(boots, [0.5, 2.5, 97.5, 99.5])
    return {"rho": float(spearmanr(a, b)[0]), "ci95": [float(lo95), float(hi95)], "ci99": [float(lo99), float(hi99)]}


# ---------------------------------------------------------------- the report
def load_run(model: str, source: str) -> dict | None:
    folder = {**MODELS, **EXTRA_MODELS, **FLOORS}[model]
    path = RESULTS / folder / source / f"{model}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return {k: z[k] for k in z.files if k != "comp"}


SHAPE_HORIZONS = (1, 2, 8, 16, 64)


def forecast_shape(direct: np.ndarray) -> dict:
    """Median over contexts of the quantile skewness (q90 + q10 - 2 q50) / (q90 - q10) (Gaussian 0) and of the
    tail ratio (q90 - q10) / (q70 - q30) (Gaussian 2.444) of the direct h-step forecasts; also the median |skew|."""
    q = direct[:, [h - 1 for h in SHAPE_HORIZONS]]
    width = q[..., -1] - q[..., 0]
    qskew = (q[..., -1] + q[..., 0] - 2 * q[..., 4]) / width
    tail = width / (q[..., 6] - q[..., 2])
    return {"quantile_skew_median": dict(zip((f"h{h}" for h in SHAPE_HORIZONS), np.median(qskew, axis=0).tolist())),
            "abs_quantile_skew_median": dict(zip((f"h{h}" for h in SHAPE_HORIZONS),
                                                 np.median(np.abs(qskew), axis=0).tolist())),
            "tail_ratio_median": dict(zip((f"h{h}" for h in SHAPE_HORIZONS), np.median(tail, axis=0).tolist()))}


def persistence(contexts: np.ndarray, p: int = 8) -> dict:
    """How much the first shock still matters at horizon h for the fitted AR(p) laws: the median share
    psi_{h-1}^2 / sum_{j<h} psi_j^2 of the h-step noise variance, and the companion matrix's spectral radius."""
    phi, _, _ = fit_ar(contexts, p)
    psi = impulse_response(phi, H_MAX)
    share = psi**2 / np.cumsum(psi**2, axis=1)
    radius = np.array([np.abs(np.linalg.eigvals(companion(f))).max() for f in phi])
    return {"first_shock_share_median": {f"h{h}": float(np.median(share[:, h - 1])) for h in (2, 8, 16, 32)},
            "spectral_radius_median": float(np.median(radius)), "share_radius_above_0.99": float((radius > 0.99).mean())}


def series_summary(ratio: np.ndarray, cluster: np.ndarray) -> dict:
    return {str(k): ratio[cluster == k].mean(axis=0).tolist() for k in np.unique(cluster)}


def analyse_source(source: str) -> dict:
    contexts, truth = load_windows(source, N_CONTEXTS, SEED)
    origin, ctx, names = window_origins(source, N_CONTEXTS)
    assert np.array_equal(ctx, contexts), source
    cluster = origin[:, 0]
    runs = {m: load_run(m, source) for m in {**MODELS, **EXTRA_MODELS, **FLOORS}}
    runs = {m: r for m, r in runs.items() if r is not None}
    for m, r in runs.items():
        assert np.array_equal(r["truth"], truth), (source, m)
    ref = ARFit(contexts)(contexts, H_MAX, np.arange(len(contexts)))
    arfit_err = float(np.abs(runs["arfit"]["direct"] - ref).max() / np.abs(ref).max())
    assert arfit_err < 1e-9, arfit_err
    ids, counts = np.unique(cluster, return_counts=True)
    out = {"verified": {"origins_reproduce_load_windows_contexts": True,
                        "stored_truth_equal_for": sorted(runs), "arfit_direct_recomputed_max_rel_err": arfit_err},
           "series": {"in_dataset": len(names), "sampled": len(ids),
                      "windows_per_series": {names[k]: int(c) for k, c in zip(ids, counts)}},
           "ar8_persistence": persistence(contexts),
           "runs": {}}
    floors = {f: runs[f]["D"] / runs[f]["spread"] for f in ("arfit", "arresid-p8") if f in runs}
    for m, r in runs.items():
        ratio = r["D"] / r["spread"]
        row = {"D_over_spread": bootstrap_mean(ratio),
               "by_series": cluster_bootstrap(ratio, cluster),
               "by_series_two_stage": cluster_mean(ratio, cluster, within=True),
               "per_series_mean": {names[int(k)]: v for k, v in series_summary(ratio, cluster).items()}}
        for f, fr in floors.items():
            if m == f:
                continue
            row[f"over_{f}"] = {"two_stage": paired_ratio(ratio, fr, cluster, within=True),
                                "whole_series": paired_ratio(ratio, fr, cluster, within=False),
                                "per_window": paired_ratio(ratio, fr, np.arange(len(ratio)), within=False),
                                "min_over_series": np.min([ratio[cluster == k].mean(axis=0) / fr[cluster == k].mean(axis=0)
                                                           for k in ids], axis=0).tolist()}
            fb = cluster_bootstrap(fr, cluster)
            row[f"over_{f}"]["separated_by_series"] = (np.array(row["by_series"]["lo"]) > np.array(fb["hi"])).tolist()
        if m in MODELS or m in EXTRA_MODELS:
            row["spearman_D_vs_crps"] = {}
            for h in (8, 16, 32):
                i = HORIZONS.index(h)
                a, b = ratio[:, i], (r["crps"] / r["spread"])[:, i]
                rho, p = spearmanr(a, b)
                within_rho = [(spearmanr(a[cluster == k], b[cluster == k])[0], c) for k, c in zip(ids, counts) if c >= 5]
                row["spearman_D_vs_crps"][f"h{h}"] = {
                    "rho": float(rho), "p_windows_independent": float(p),
                    "by_window": cluster_spearman(a, b, np.arange(len(a))),
                    "by_series": cluster_spearman(a, b, cluster),
                    "mean_within_series_rho": float(np.average([w[0] for w in within_rho],
                                                               weights=[w[1] for w in within_rho]))}
        row["forecast_shape"] = forecast_shape(r["direct"])
        if "resid_skew" in r:
            row["residual_shape"] = {
                "skew_quartiles": np.percentile(r["resid_skew"], [25, 50, 75]).tolist(),
                "excess_kurtosis_quartiles": np.percentile(r["resid_exkurt"], [25, 50, 75]).tolist()}
        out["runs"][m] = row
    if "arresid-p8" in runs and "arresid-p8-check" in runs:
        out["monte_carlo"] = monte_carlo_error(runs["arresid-p8"], runs["arresid-p8-check"])
    return out


def monte_carlo_error(main: dict, check: dict) -> dict:
    """Quantile MC error of the main run (MAIN_PATHS), from its difference with an independent run with
    CHECK_PATHS = MAIN_PATHS / 4: var(q_check - q_main) = 4 var_main + var_main when the variance scales as 1/n.
    h = 1 is exact in both runs and must agree exactly."""
    qm, qc = main["noise_q"], check["noise_q"]
    width = qm[..., -1] - qm[..., 0]
    diff = (qc - qm) / width[..., None]  # (B, H, 9)
    sd_main = diff[:, 1:].std(axis=0) / np.sqrt(MAIN_PATHS / CHECK_PATHS + 1.0)  # (H - 1, 9), over contexts
    return {"h1_max_abs_diff_over_width": float(np.abs(diff[:, 0]).max()),
            "sd_over_width_by_level_max_over_h_ge_2": sd_main.max(axis=0).tolist(),
            "sd_over_width_max_h_ge_2": float(sd_main.max()),
            "sd_over_width_median_h_ge_2": float(np.median(sd_main)),
            "share_of_cells_with_abs_diff_over_0.003": float((np.abs(diff[:, 1:]) > 0.003).mean()),
            "D_over_spread_main": (main["D"] / main["spread"]).mean(axis=0).tolist(),
            "D_over_spread_check": (check["D"] / check["spread"]).mean(axis=0).tolist()}


def f4(x: float) -> str:
    return f"{x:.4f}"


def ci(d: dict, i: int, fmt=f4) -> str:
    return f"{fmt(d['mean'][i])} [{fmt(d['lo'][i])}, {fmt(d['hi'][i])}]"


def tables(summary: dict) -> str:
    hs = (8, 16, 64)
    src = summary["sources"]
    lines = ["# Rigour checks on the real sources: tables\n", "Generated by `python -m hsc.audit.rigour report`. D/spread is the per-context defect "
             "over the direct 80% width, averaged over the 256 windows. Intervals are 95% percentile bootstraps "
             "(2,000 draws).\n", "## Series behind the windows\n", "| Source | series in dataset | series sampled | "
             "windows per series |", "|---|---|---|---|"]
    for s, r in src.items():
        w = r["series"]["windows_per_series"]
        lines.append(f"| {s} | {r['series']['in_dataset']} | {r['series']['sampled']} | "
                     + ", ".join(f"{k}: {v}" for k, v in w.items()) + " |")
    lines += ["\n## Does every model stay above the floor at every h >= 8 by series?\n",
              "Separated = the model's whole-series CI lies entirely above the floor's whole-series CI; "
              "ratio > 1 = the lower end of the paired two-stage CI of the multiple exceeds 1.\n",
              "| Floor | Source | separated at all h >= 8 | ratio > 1 at all h >= 8 | smallest lower CI end of the "
              "multiple, h = 8 / 16 |", "|---|---|---|---|---|"]
    for floor, rows in summary["conclusions"].items():
        for s, c in rows.items():
            lines.append(f"| {LABEL[floor]} | {s} | {c['all_separated_h_ge_8']} | {c['all_ratio_above_1_h_ge_8']} | "
                         f"{c['smallest_lower_ci_h8']:.1f} ({c['smallest_lower_ci_h8_model']}) / "
                         f"{c['smallest_lower_ci_h16']:.1f} ({c['smallest_lower_ci_h16_model']}) |")
    for h in hs:
        i = HORIZONS.index(h)
        lines += [f"\n## D/spread at h = {h}: mean [per-window 95% CI] / [whole-series CI] / [two-stage series CI]\n",
                  "| Model | " + " | ".join(src) + " |", "|---|" + "---|" * len(src)]
        for m in list(MODELS) + ["arfit", "arresid-p8"]:
            cells = []
            for s in src:
                row = src[s]["runs"].get(m)
                cells.append("-" if row is None else f"{ci(row['D_over_spread'], i)} / "
                             f"[{f4(row['by_series']['lo'][i])}, {f4(row['by_series']['hi'][i])}] / "
                             f"[{f4(row['by_series_two_stage']['lo'][i])}, {f4(row['by_series_two_stage']['hi'][i])}]")
            lines.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    i8 = HORIZONS.index(8)
    for s in src:
        names = list(src[s]["series"]["windows_per_series"])
        lines += [f"\n## {s}: D/spread at h = 8 in each series (windows per series in brackets)\n",
                  "| Model | " + " | ".join(f"{n} ({src[s]['series']['windows_per_series'][n]})" for n in names) + " |",
                  "|---|" + "---|" * len(names)]
        for m in list(MODELS) + ["arfit", "arresid-p8"]:
            row = src[s]["runs"].get(m)
            if row is not None:
                lines.append(f"| {LABEL[m]} | " + " | ".join(f4(row["per_series_mean"][n][i8]) for n in names) + " |")
    for floor in ("arfit", "arresid-p8"):
        lines += [f"\n## Multiple of the {LABEL[floor]} floor, h = 8 / 16 / 64: ratio of mean D/spread "
                  "[paired two-stage series bootstrap 95% CI]; `*` = model's series CI entirely above the floor's; "
                  "last number = smallest per-series multiple\n",
                  "| Model | " + " | ".join(src) + " |", "|---|" + "---|" * len(src)]
        for m in MODELS:
            cells = []
            for s in src:
                o = src[s]["runs"][m].get(f"over_{floor}")
                if o is None:
                    cells.append("-")
                    continue
                parts = []
                for h in hs:
                    i = HORIZONS.index(h)
                    t = o["two_stage"]
                    star = "*" if o["separated_by_series"][i] else ""
                    parts.append(f"{t['mean'][i]:.1f} [{t['lo'][i]:.1f}, {t['hi'][i]:.1f}]{star} "
                                 f"(min {o['min_over_series'][i]:.1f})")
                cells.append("<br>".join(parts))
            lines.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    lines += ["\n## Spearman rho of per-context D/spread with per-context CRPS/spread\n",
              "rho over all 256 windows; p treats windows as independent; CIs resample windows or whole series; "
              "last column is the window-weighted mean of the rho within each series.\n",
              "| Source | Model | h | rho | p (windows) | 95% CI, windows | 95% CI, series | 99% CI, series | "
              "within-series rho |", "|---|---|---|---|---|---|---|---|---|"]
    for s in src:
        for m in MODELS:
            for h, sp in src[s]["runs"][m]["spearman_D_vs_crps"].items():
                bw, bs = sp["by_window"], sp["by_series"]
                lines.append(f"| {s} | {LABEL[m]} | {h[1:]} | {sp['rho']:+.3f} | {sp['p_windows_independent']:.3g} | "
                             f"[{bw['ci95'][0]:+.2f}, {bw['ci95'][1]:+.2f}] | [{bs['ci95'][0]:+.2f}, {bs['ci95'][1]:+.2f}] | "
                             f"[{bs['ci99'][0]:+.2f}, {bs['ci99'][1]:+.2f}] | {sp['mean_within_series_rho']:+.3f} |")
    lines += ["\n## Coherent floors: D/spread by horizon, mean [per-window CI] / [whole-series CI]\n",
              "| Source | Floor | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |", "|---|---|" + "---|" * len(HORIZONS)]
    for s in src:
        for f in ("arfit", "arfit-p2", "arfit-p24", "arresid-p8", "arresid-p8-check"):
            row = src[s]["runs"].get(f)
            if row is None:
                continue
            lines.append(f"| {s} | {LABEL[f]} | " + " | ".join(
                f"{ci(row['D_over_spread'], i)} / [{f4(row['by_series']['lo'][i])}, {f4(row['by_series']['hi'][i])}]"
                for i in range(len(HORIZONS))) + " |")
    lines += ["\n## Residual-noise AR(8) floor over the Gaussian AR(8) floor (paired, two-stage series CI)\n",
              "| Source | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |", "|---|" + "---|" * len(HORIZONS)]
    for s in src:
        o = src[s]["runs"].get("arresid-p8", {}).get("over_arfit")
        if o is not None:
            lines.append(f"| {s} | " + " | ".join(ci(o["two_stage"], i, lambda x: f"{x:.2f}")
                                                for i in range(len(HORIZONS))) + " |")
    lines += ["\n## How non-Gaussian the AR(8) residuals are (over the 256 base contexts)\n",
              "Skewness and excess kurtosis of each context's centred AR(8) residuals: median (quartiles) over "
              "contexts. Gaussian: 0 and 0.\n", "| Source | residual skewness | residual excess kurtosis |",
              "|---|---|---|"]
    for s in src:
        rs = src[s]["runs"].get("arresid-p8", {}).get("residual_shape")
        if rs is not None:
            sk, ku = rs["skew_quartiles"], rs["excess_kurtosis_quartiles"]
            lines.append(f"| {s} | {sk[1]:+.2f} ({sk[0]:+.2f}, {sk[2]:+.2f}) | {ku[1]:.1f} ({ku[0]:.1f}, {ku[2]:.1f}) |")
    lines += ["\n## Persistence of the fitted AR(8) laws: how much the first shock still matters at horizon h\n",
              "Median over contexts of psi_{h-1}^2 / sum_{j<h} psi_j^2 (the first shock's share of the h-step noise "
              "variance), and of the companion matrix's spectral radius.\n",
              "| Source | first-shock share, h = 2 / 8 / 16 / 32 | spectral radius (median) | share of contexts above 0.99 |",
              "|---|---|---|---|"]
    for s in src:
        ps = src[s]["ar8_persistence"]
        lines.append(f"| {s} | " + " / ".join(f"{v:.3f}" for v in ps["first_shock_share_median"].values())
                     + f" | {ps['spectral_radius_median']:.3f} | {ps['share_radius_above_0.99']:.2f} |")
    hz = " / ".join(str(h) for h in SHAPE_HORIZONS)
    lines += ["\n## Shape of the direct forecasts: the floors' and the models' own\n",
              "Median over contexts of the quantile skewness (q90 + q10 - 2 q50)/(q90 - q10) (Gaussian 0), of its "
              "absolute value, and of the tail ratio (q90 - q10)/(q70 - q30) (Gaussian 2.444).\n",
              f"| Source | Forecaster | skew, h = {hz} | abs skew | tail ratio |", "|---|---|---|---|---|"]
    for s in src:
        for m in ["arfit", "arresid-p8"] + list(MODELS):
            fs = src[s]["runs"].get(m, {}).get("forecast_shape")
            if fs is not None:
                lines.append(f"| {s} | {LABEL[m]} | " + " / ".join(f"{v:+.3f}" for v in fs["quantile_skew_median"].values())
                             + " | " + " / ".join(f"{v:.3f}" for v in fs["abs_quantile_skew_median"].values())
                             + " | " + " / ".join(f"{v:.2f}" for v in fs["tail_ratio_median"].values()) + " |")
    main_m, check_m = MAIN_PATHS / 1e6, CHECK_PATHS / 1e6
    lines += [f"\n## Monte Carlo error of the residual-noise floor ({main_m:g}M paths, from an independent "
              f"{check_m:g}M-path run)\n",
              "Quantile sd in units of the forecast's own 80% width, over the 256 contexts, for h >= 2 "
              "(h = 1 is computed exactly; the largest h = 1 difference between the runs is shown).\n",
              "| Source | h = 1 max diff | max sd / width, h >= 2 | median sd / width | max over h >= 2 by level 0.1..0.9 | "
              "share of cells with diff > 0.003 | "
              f"D/spread {main_m:g}M (h = 2 / 8 / 16 / 64) | D/spread {check_m:g}M |", "|---|---|---|---|---|---|---|---|"]
    for s in src:
        mc = src[s].get("monte_carlo")
        if mc is None:
            continue
        sel = [HORIZONS.index(h) for h in (2, 8, 16, 64)]
        lines.append(f"| {s} | {mc['h1_max_abs_diff_over_width']:.1e} | {mc['sd_over_width_max_h_ge_2']:.5f} | "
                     f"{mc['sd_over_width_median_h_ge_2']:.5f} | "
                     + " ".join(f"{v:.5f}" for v in mc["sd_over_width_by_level_max_over_h_ge_2"]) + " | "
                     + f"{mc['share_of_cells_with_abs_diff_over_0.003']:.1e} | "
                     + " / ".join(f"{mc['D_over_spread_main'][i]:.5f}" for i in sel) + " | "
                     + " / ".join(f"{mc['D_over_spread_check'][i]:.5f}" for i in sel) + " |")
    return "\n".join(lines) + "\n"


def conclusions(sources: dict) -> dict:
    """Per floor and source: is every model above the floor at every h >= 8, and the weakest multiple."""
    late = [i for i, h in enumerate(HORIZONS) if h >= 8]
    out = {}
    for floor in ("arfit", "arresid-p8"):
        for s, r in sources.items():
            rows = {m: r["runs"][m].get(f"over_{floor}") for m in MODELS}
            rows = {m: o for m, o in rows.items() if o is not None}
            if not rows:
                continue
            c = {"all_separated_h_ge_8": all(o["separated_by_series"][i] for o in rows.values() for i in late),
                 "all_ratio_above_1_h_ge_8": all(o["two_stage"]["lo"][i] > 1 for o in rows.values() for i in late)}
            for h in (8, 16):
                i = HORIZONS.index(h)
                weakest = min(rows, key=lambda m: rows[m]["two_stage"]["lo"][i])
                c[f"smallest_lower_ci_h{h}"] = rows[weakest]["two_stage"]["lo"][i]
                c[f"smallest_lower_ci_h{h}_model"] = LABEL[weakest]
            out.setdefault(floor, {})[s] = c
    return out


def report(out: Path = OUT) -> dict:
    summary = {"config": {"n": N_CONTEXTS, "inner": INNER, "seed": SEED, "horizons": list(HORIZONS), "n_boot": 2000,
                          "bootstrap_seed": 0, "main_paths": MAIN_PATHS, "check_paths": CHECK_PATHS},
               "sources": {s: analyse_source(s) for s in SOURCES}}
    summary["conclusions"] = conclusions(summary["sources"])
    write_json(out / "summary.json", summary)
    (out / "tables.md").write_text(tables(summary))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("floors", "report"))
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--paths", type=int, default=MAIN_PATHS)
    ap.add_argument("--check-paths", type=int, default=CHECK_PATHS)
    args = ap.parse_args()
    if args.command == "floors":
        run_floors(args.sources, args.workers, args.paths, args.check_paths)
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
