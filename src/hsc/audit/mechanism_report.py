"""Where the self-consistency gap on daily COVID cases comes from (reads results/mechanism/).

Descriptive only. Chronos-Bolt standardises every context by its own mean and standard
deviation (InstanceNorm in chronos_bolt.py), so D / sd(context) is the gap in the units the
network works in, and spread / sd(context) is how wide the forecast is in those units. For each
source (covid, epf, weather, and ar1 as a synthetic reference) and model this module reports:
  - D / spread, D / sd and spread / sd by horizon;
  - how much of D is a pure location shift of the chained forecast ("composite") against the direct one;
  - the gap at each quantile level, and the inner forecasts as a function of the draw's level u;
  - a counterfactual in which the forecasts from tail draws (u outside 0.1-0.9) are replaced by
    the straight-line trend in y of the central draws' forecasts, and the converse;
  - Spearman correlations of the per-context D / spread with properties of the context;
  - the scale probe: ar1 contexts with a bump of height A in their first half (hsc.audit.mechanism).

    python -m hsc.audit.mechanism_report
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
from scipy.stats import spearmanr

from hsc.audit.gap import HORIZONS
from hsc.audit.mechanism import bump_windows
from hsc.audit.regime import LAWS, regime_windows
from hsc.audit.windows import CONTEXT, H_MAX, REAL, fill_short_gaps, load_windows
from hsc.common.paths import RESULTS, TIME_DIR
from hsc.common.quantiles import LEVELS, band_w1, mixture_quantiles
from hsc.common.stats import bootstrap_mean, write_json

RES = RESULTS / "mechanism"
SEED = 20261104
SOURCES = ("covid", "epf", "weather", "ar1")
MODELS = ("chronos-bolt-small", "chronos-bolt-base", "timestral-28m", "arfit")
BUMPS = (0, 10, 20, 40, 80)
IDX = np.array(HORIZONS) - 1
LABEL = {"chronos-bolt-small": "Chronos-Bolt small", "chronos-bolt-base": "Chronos-Bolt base",
         "timestral-28m": "Timestral-28M", "arfit": "AR(8) fit (coherent)", "oracle": "exact Bayes (floor)"}
SRC_COLOR = {"covid": "C3", "epf": "C0", "weather": "C2", "ar1": "0.4", "msar": "C4"}


def windows(source: str, n: int):
    if source in LAWS:
        return regime_windows(source, n, SEED)[0]
    if source.startswith("ar1bump"):
        return bump_windows(float(source[len("ar1bump"):]), n, SEED)[0]
    return load_windows(source, n, SEED)[0]


def window_origins(source: str, n: int, seed: int = SEED):
    """(series index, start) of each window that hsc.audit.windows.real_windows draws, by replaying its sampler."""
    rel, variate = REAL[source]
    with pa.memory_map(str(TIME_DIR / rel / "data-00000-of-00001.arrow")) as src:
        table = pa.ipc.open_stream(src).read_all()
    series = []
    for target in table.column("target").to_pylist():
        arr = np.asarray(target, dtype=np.float64)
        series.append(fill_short_gaps(arr[variate or 0] if arr.ndim == 2 else arr))
    rng = np.random.default_rng(seed)
    need = CONTEXT + H_MAX
    origin, windows_ = [], []
    attempts = 0
    while len(windows_) < n and attempts < 100 * n:
        attempts += 1
        k = rng.integers(len(series))
        s = series[k]
        if len(s) < need:
            continue
        start = rng.integers(0, len(s) - need + 1)
        w = s[start : start + need]
        if np.isfinite(w).all() and w[:CONTEXT].std() > 0:
            windows_.append(w)
            origin.append((int(k), int(start)))
    return np.array(origin), np.stack(windows_)[:, :CONTEXT], table.column("item_id").to_pylist()


def cluster_bootstrap(values: np.ndarray, cluster: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean over units with a 95% interval from resampling whole clusters (series), keeping all their units."""
    rng = np.random.default_rng(seed)
    ids = np.unique(cluster)
    members = [np.nonzero(cluster == c)[0] for c in ids]
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        idx = np.concatenate([members[p] for p in pick])
        boots.append(values[idx].mean(axis=0))
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    return {"mean": values.mean(axis=0).tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "clusters": int(len(ids))}


_CACHE: dict = {}


def load(source: str, model: str) -> dict | None:
    path = RES / source / f"{model}.npz"
    if not path.exists():
        return None
    z = dict(np.load(path))
    if source not in _CACHE:
        _CACHE[source] = windows(source, len(z["D"]))
    z["contexts"] = _CACHE[source]
    z["sd"] = z["contexts"].std(axis=1)  # Chronos-Bolt's scale (population sd over the context)
    return z


# ---------------------------------------------------------------- per-run quantities
def shape_of_defect(z: dict) -> dict:
    """Signed quantile differences d = composite - direct, and what part of D is a pure shift."""
    d = z["comp"] - z["direct"][:, IDX]  # (n, H, 9)
    spread = z["spread"]
    width = z["comp"][..., -1] - z["comp"][..., 0]
    return {"shift_over_spread": bootstrap_mean(d.mean(axis=-1) / spread),
            "abs_shift_over_spread": bootstrap_mean(np.abs(d.mean(axis=-1)) / spread),
            "shift_share_of_D": (np.abs(d.mean(axis=-1)).mean(axis=0) / z["D"].mean(axis=0)).tolist(),
            "width_change": bootstrap_mean(width / spread - 1),
            "by_level": (np.abs(d) / spread[..., None]).mean(axis=0).tolist()}  # (H, 9)


def draw_profiles(z: dict) -> dict:
    """Per draw m (level u_m): mean (median shift, width) of its inner forecast, in direct-spread units."""
    fc = z["inner_fc"].astype(np.float64)  # (n, M, H, 9)
    med = z["direct"][:, IDX, 4]  # (n, H)
    spread = z["spread"]
    return {"shift": ((fc[..., 4] - med[:, None]) / spread[:, None]).mean(axis=0).tolist(),  # (M, H)
            "width": ((fc[..., -1] - fc[..., 0]) / spread[:, None]).mean(axis=0).tolist()}


def smoothed_defects(z: dict) -> dict:
    """D after replacing the tail draws' (or the central draws') forecasts by the central linear trend in y."""
    y, fc = z["y"], z["inner_fc"].astype(np.float64)  # (n, M), (n, M, H, 9)
    m = y.shape[1]
    u = (np.arange(m) + 0.5) / m
    central = (u > LEVELS[0]) & (u < LEVELS[-1])
    yc = y[:, central]
    dy = yc - yc.mean(axis=1, keepdims=True)
    fcc = fc[:, central]
    fbar = fcc.mean(axis=1)  # (n, H, 9)
    var = (dy**2).sum(axis=1)
    slope = np.einsum("nm,nmhk->nhk", dy, fcc - fbar[:, None]) / np.where(var > 0, var, np.inf)[:, None, None]
    line = fbar[:, None] + slope[:, None] * (y - yc.mean(axis=1, keepdims=True))[:, :, None, None]
    out = {}
    for key, keep in (("tails_on_line", central), ("centre_on_line", ~central)):
        mixed = np.where(keep[None, :, None, None], fc, line)
        comp = mixture_quantiles(mixed.transpose(0, 2, 1, 3))  # mixture over draws, (n, H, 9)
        out[key] = bootstrap_mean(band_w1(z["direct"][:, IDX], comp) / z["spread"])
    out["tail_draws"] = int((~central).sum())
    return out


def features(z: dict) -> dict:
    c, y, sd = z["contexts"], z["y"], z["sd"]
    loc = c.mean(axis=1)
    n_ext = c.shape[1] + 1
    ext_mean = (c.sum(axis=1, keepdims=True) + y) / n_ext
    ext_sd = np.sqrt(np.maximum(((c**2).sum(axis=1, keepdims=True) + y**2) / n_ext - ext_mean**2, 0))
    one = z["direct"][:, 0]
    return {"log10 sd": np.log10(sd),
            "mean / sd": loc / sd,
            "end level (x_T - mean) / sd": (c[:, -1] - loc) / sd,
            "weekly change / sd": (c[:, -7:].mean(axis=1) - c[:, -14:-7].mean(axis=1)) / sd,
            "sd of last 28 / sd": c[:, -28:].std(axis=1) / sd,
            "one-step width / sd": (one[:, -1] - one[:, 0]) / sd,
            "h=8 width / sd": z["spread"][:, HORIZONS.index(8)] / sd,
            "largest |y - x_T| / sd": np.abs(y - c[:, -1:]).max(axis=1) / sd,
            "largest change of sd from appending y": np.abs(ext_sd / sd[:, None] - 1).max(axis=1),
            "share of draws outside context range": ((y < c.min(axis=1, keepdims=True))
                                                     | (y > c.max(axis=1, keepdims=True))).mean(axis=1)}


def spearman_ci(a: np.ndarray, b: np.ndarray, cluster: np.ndarray | None = None, n_boot: int = 500,
                seed: int = 0) -> list:
    """Spearman rho with a 95% interval; resamples whole clusters (series) when they are given."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    cluster = np.arange(len(a)) if cluster is None else cluster
    ids = np.unique(cluster)
    members = [np.nonzero(cluster == c)[0] for c in ids]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[p] for p in rng.integers(0, len(ids), len(ids))])
        boots.append(spearmanr(a[idx], b[idx])[0])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return [float(spearmanr(a, b)[0]), float(lo), float(hi)]


def summarise_run(source: str, model: str) -> dict | None:
    z = load(source, model)
    if z is None:
        return None
    ratio = z["D"] / z["spread"]
    out = {"D_over_spread": bootstrap_mean(ratio),
           "D_over_sd": bootstrap_mean(z["D"] / z["sd"][:, None]),
           "median_D_over_sd": np.median(z["D"] / z["sd"][:, None], axis=0).tolist(),
           "spread_over_sd": bootstrap_mean(z["spread"] / z["sd"][:, None]),
           "median_spread_over_sd": np.median(z["spread"] / z["sd"][:, None], axis=0).tolist(),
           "slide_D_over_spread": bootstrap_mean(z["D_slide"] / z["spread"]),
           "dropfirst_over_spread": bootstrap_mean(z["dropfirst"] / z["spread"]),
           "shape": shape_of_defect(z), "draws": draw_profiles(z), "smoothed": smoothed_defects(z)}
    cluster = None
    if source in ("covid", "epf", "weather"):
        origin, ctx, names = window_origins(source, len(ratio))
        assert np.array_equal(ctx, z["contexts"])
        cluster = origin[:, 0]
        out["series_names"] = names
        out["windows_per_series"] = np.bincount(cluster, minlength=len(names)).tolist()
        out["D_over_spread_cluster"] = cluster_bootstrap(ratio, cluster)
        out["D_over_sd_cluster"] = cluster_bootstrap(z["D"] / z["sd"][:, None], cluster)
        out["per_series_D_over_spread"] = {names[k]: ratio[cluster == k].mean(axis=0).tolist()
                                           for k in np.unique(cluster)}
    if model != "arfit" and model != "oracle":
        c = z["contexts"]
        change28 = np.polyfit(np.arange(28), c[:, -28:].T, 1)[0] * 28 / z["sd"]  # straight-line rise over the last 28 steps, in sd
        shift = (z["comp"] - z["direct"][:, IDX]).mean(axis=-1) / z["spread"]  # signed shift of the composite
        out["shift_vs_recent_change"] = {f"h{h}": spearman_ci(change28, shift[:, HORIZONS.index(h)], cluster)
                                         for h in (8, 16)}
        out["shift_negative_share"] = (shift < 0).mean(axis=0).tolist()
        feats = features(z)
        out["spearman_D_over_spread"] = {k: {f"h{h}": spearman_ci(v, ratio[:, HORIZONS.index(h)], cluster)
                                             for h in (8, 16)} for k, v in feats.items()}
        out["feature_medians"] = {k: float(np.median(v)) for k, v in feats.items()}
    return out


# ---------------------------------------------------------------- figures
def fig_units(path: Path) -> None:
    """D/spread and D/sd by horizon, per source: the defect in the network's own units."""
    models = ("chronos-bolt-small", "timestral-28m")
    fig, axes = plt.subplots(len(models), 3, figsize=(13, 3.4 * len(models)))
    for row, model in zip(axes, models):
        for source in SOURCES:
            z = load(source, model)
            if z is None:
                continue
            for ax, val in zip(row, (z["D"] / z["spread"], z["D"] / z["sd"][:, None], z["spread"] / z["sd"][:, None])):
                b = bootstrap_mean(val)
                ax.plot(HORIZONS, b["mean"], "o-", ms=3, color=SRC_COLOR[source], label=source)
                ax.fill_between(HORIZONS, b["lo"], b["hi"], color=SRC_COLOR[source], alpha=0.2)
        for ax, title in zip(row, ("D / 80% width", "D / sd(context)", "80% width / sd(context)")):
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(HORIZONS, [str(h) for h in HORIZONS])
            ax.set_title(f"{LABEL[model]}: {title}", fontsize=9)
            ax.grid(alpha=0.3, which="both")
            ax.set_xlabel("horizon h")
        row[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_scatter(path: Path) -> None:
    """Per context: D/sd against width/sd at h = 8 and 16. Grey lines are constant D/width."""
    models = ("chronos-bolt-small", "chronos-bolt-base", "timestral-28m")
    fig, axes = plt.subplots(len(models), 2, figsize=(9, 3.3 * len(models)), sharex=True, sharey=True)
    for row, model in zip(axes, models):
        for ax, h in zip(row, (8, 16)):
            i = HORIZONS.index(h)
            for source in SOURCES:
                z = load(source, model)
                if z is None:
                    continue
                ax.scatter(z["spread"][:, i] / z["sd"], z["D"][:, i] / z["sd"], s=4, alpha=0.5,
                           color=SRC_COLOR[source], label=source)
            xs = np.array([0.02, 10])
            for r in (0.01, 0.03, 0.1, 0.3):
                ax.plot(xs, r * xs, color="0.7", lw=0.6, ls="--")
                ax.text(9, r * 9, f"{r:g}", fontsize=6, color="0.5", va="bottom", ha="right")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{LABEL[model]}, h = {h}", fontsize=9)
            ax.grid(alpha=0.3, which="both")
        row[0].set_ylabel("D / sd(context)")
    for ax in axes[-1]:
        ax.set_xlabel("80% width / sd(context)")
    axes[0, 0].legend(fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_draws(summary: dict, path: Path) -> None:
    hs = (8, 16)
    fig, axes = plt.subplots(2, len(hs), figsize=(10, 6.4), sharex=True)
    for col, h in enumerate(hs):
        i = HORIZONS.index(h)
        for source in SOURCES:
            for model, ls in (("chronos-bolt-small", "-"), ("arfit", ":")):
                run = summary.get(source, {}).get(model)
                if run is None:
                    continue
                m = len(run["draws"]["shift"])
                u = (np.arange(m) + 0.5) / m
                lab = f"{source}, {LABEL[model]}" if col == 0 else None
                axes[0, col].plot(u, np.array(run["draws"]["shift"])[:, i], ls, color=SRC_COLOR[source], label=lab)
                axes[1, col].plot(u, np.array(run["draws"]["width"])[:, i], ls, color=SRC_COLOR[source])
        for ax in axes[:, col]:
            ax.axvspan(0, LEVELS[0], color="0.9")
            ax.axvspan(LEVELS[-1], 1, color="0.9")
            ax.grid(alpha=0.3)
        axes[0, col].set_title(f"inner forecasts used for h = {h} (step h-1 from c+y)", fontsize=9)
        axes[1, col].set_xlabel("level u of the draw y (grey: Gaussian-tail region)")
    axes[0, 0].set_ylabel("(median from c+y - direct median) / width")
    axes[1, 0].set_ylabel("width from c+y / direct width")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_levels(summary: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, h in zip(axes, (8, 16)):
        i = HORIZONS.index(h)
        for source in SOURCES:
            for model, ls in (("chronos-bolt-small", "o-"), ("timestral-28m", "s--")):
                run = summary.get(source, {}).get(model)
                if run is not None:
                    ax.plot(LEVELS, np.array(run["shape"]["by_level"])[i], ls, ms=3, color=SRC_COLOR[source],
                            label=f"{source}, {LABEL[model]}")
        ax.set_title(f"|composite - direct| / width by quantile level, h = {h}", fontsize=9)
        ax.set_xlabel("quantile level")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_examples(path: Path) -> None:
    z = load("covid", "chronos-bolt-small")
    ratio = (z["D"] / z["spread"])[:, HORIZONS.index(8)]
    order = np.argsort(ratio)
    picks = [order[len(order) // 2], order[int(0.9 * len(order))], order[-1], order[len(order) // 10]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.2))
    for ax, b in zip(axes.ravel(), picks):
        c = z["contexts"][b]
        t_ctx = np.arange(CONTEXT - 120, CONTEXT)
        t_fc = CONTEXT + np.arange(1, 65)
        ax.plot(np.arange(CONTEXT), c, color="0.75", lw=0.6)
        ax.plot(t_ctx, c[-120:], color="0.2", lw=0.8, label="context")
        q = z["direct"][b]
        ax.fill_between(t_fc, q[:, 0], q[:, -1], color="C0", alpha=0.2, label="direct 10-90%")
        ax.plot(t_fc, q[:, 4], color="C0", lw=1, label="direct median")
        hz = CONTEXT + np.array(HORIZONS)
        ax.errorbar(hz, z["comp"][b, :, 4], yerr=[z["comp"][b, :, 4] - z["comp"][b, :, 0],
                                                  z["comp"][b, :, -1] - z["comp"][b, :, 4]],
                    fmt="o", color="C3", ms=3, capsize=2, label="composite median, 10-90%")
        ax.set_xlim(CONTEXT - 120, CONTEXT + 66)
        lo = min(c[-120:].min(), q[:, 0].min())
        hi = max(c[-120:].max(), q[:, -1].max())
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        ax.set_title(f"COVID context {b}: D/width at h=8 = {ratio[b]:.3f}; "
                     f"width/sd = {z['spread'][b, 2] / z['sd'][b]:.2f}", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_features(summary: dict, path: Path) -> None:
    keys = ("one-step width / sd", "sd of last 28 / sd", "end level (x_T - mean) / sd", "weekly change / sd",
            "largest |y - x_T| / sd", "largest change of sd from appending y")
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharey=True)
    for ax, key in zip(axes.ravel(), keys):
        for source in SOURCES:
            z = load(source, "chronos-bolt-small")
            if z is None:
                continue
            x = features(z)[key]
            ax.scatter(x, (z["D"] / z["spread"])[:, HORIZONS.index(8)], s=4, alpha=0.5, color=SRC_COLOR[source],
                       label=source)
        if "width" in key or "sd of" in key or "largest" in key:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(key, fontsize=9)
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].set_ylabel("Chronos-Bolt small, D / width at h = 8")
    axes[1, 0].set_ylabel("Chronos-Bolt small, D / width at h = 8")
    axes[0, 0].legend(fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bump_summary() -> dict:
    out = {}
    for model in ("oracle", "arfit", "chronos-bolt-small", "chronos-bolt-base", "timestral-28m"):
        rows = {}
        for a in BUMPS:
            z = load(f"ar1bump{a}", model)
            if z is None:
                continue
            rows[str(a)] = {"D_over_spread": bootstrap_mean(z["D"] / z["spread"]),
                            "D_over_sd": bootstrap_mean(z["D"] / z["sd"][:, None]),
                            "spread_over_sd": bootstrap_mean(z["spread"] / z["sd"][:, None]),
                            "median_sd": float(np.median(z["sd"])), "spread": bootstrap_mean(z["spread"])}
        if rows:
            out[model] = rows
    return out


def fig_bump(bumps: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for model, rows in bumps.items():
        amps = [int(a) for a in rows]
        for ax, key in zip(axes, ("D_over_spread", "D_over_sd", "spread")):
            for h, ls in ((8, "o-"), (16, "s--")):
                i = HORIZONS.index(h)
                vals = [rows[str(a)][key]["mean"][i] for a in amps]
                ax.plot(amps, vals, ls, ms=3, color={"oracle": "k", "arfit": "0.55", "chronos-bolt-small": "C0",
                                                     "chronos-bolt-base": "C1", "timestral-28m": "C2"}[model],
                        label=f"{LABEL[model]}, h={h}")
    for ax, title in zip(axes, ("D / 80% width", "D / sd(context)", "80% width (data units)")):
        ax.set_xlabel("bump height A (AR(1) marginal sd = 2.29)")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def elasticities(n_boot: int = 1000, seed: int = 0) -> dict:
    """Least-squares slope of log(D/sd) on log(width/sd) per context: pooled over the four sources and within each."""
    out = {}
    rng = np.random.default_rng(seed)
    for model in ("chronos-bolt-small", "chronos-bolt-base", "timestral-28m"):
        out[model] = {}
        for h in (8, 16):
            i = HORIZONS.index(h)
            xs, ys, row = [], [], {}
            for source in SOURCES:
                z = load(source, model)
                x, y = np.log(z["spread"][:, i] / z["sd"]), np.log(z["D"][:, i] / z["sd"])
                row[source] = float(np.polyfit(x, y, 1)[0])
                xs.append(x), ys.append(y)
            x, y = np.concatenate(xs), np.concatenate(ys)
            boots = [np.polyfit(x[k], y[k], 1)[0] for k in rng.integers(0, len(x), (n_boot, len(x)))]
            row["pooled"] = [float(np.polyfit(x, y, 1)[0]), *np.percentile(boots, [2.5, 97.5]).tolist()]
            out[model][f"h{h}"] = row
    return out


def tables(summary: dict, bumps: dict, slopes: dict) -> str:
    hs = (8, 16, 64)
    ix = [HORIZONS.index(h) for h in hs]
    f3 = lambda d, i: f"{d['mean'][i]:.4f}"  # noqa: E731
    ci = lambda d, i: f"{d['mean'][i]:.4f} [{d['lo'][i]:.4f}, {d['hi'][i]:.4f}]"  # noqa: E731
    lines = ["### D / width, D / sd(context) and width / sd(context), h = 8 / 16 / 64\n",
             "| Source | Model | D / width | D / sd | width / sd | slide: D / width | drop-first shift / width |",
             "|---|---|---|---|---|---|---|"]
    for source in SOURCES:
        for model, r in summary.get(source, {}).items():
            lines.append(f"| {source} | {LABEL[model]} | " + " | ".join(
                " / ".join(f3(r[k], i) for i in ix) for k in ("D_over_spread", "D_over_sd", "spread_over_sd",
                                                               "slide_D_over_spread", "dropfirst_over_spread")) + " |")
    lines += ["\n### 95% intervals at h = 8: over contexts, and over whole series (cluster bootstrap)\n",
              "| Source | Model | windows per series | D / width, contexts | D / width, series | D / sd, series |",
              "|---|---|---|---|---|---|"]
    for source in ("covid", "epf", "weather"):
        for model, r in summary.get(source, {}).items():
            lines.append(f"| {source} | {LABEL[model]} | {r['windows_per_series']} | {ci(r['D_over_spread'], 2)} | "
                         f"{ci(r['D_over_spread_cluster'], 2)} | {ci(r['D_over_sd_cluster'], 2)} |")
    names = summary["covid"]["chronos-bolt-small"]["series_names"]
    lines += ["\n### COVID, D / width at h = 8 by country\n", "| Model | " + " | ".join(names) + " |",
              "|---|" + "---|" * len(names)]
    for model in ("chronos-bolt-small", "chronos-bolt-base", "timestral-28m"):
        per = summary["covid"][model]["per_series_D_over_spread"]
        lines.append(f"| {LABEL[model]} | " + " | ".join(f"{per[n][2]:.3f}" if n in per else "-" for n in names) + " |")
    lines += ["\n### Shape of the defect, h = 8 / 16 / 64\n",
              "| Source | Model | share of D that is a pure shift | composite width / direct width - 1 | "
              "signed shift / width |", "|---|---|---|---|---|"]
    for source in SOURCES + ("msar",):
        for model, r in summary.get(source, {}).items():
            sh = r["shape"]
            lines.append(f"| {source} | {LABEL.get(model, model)} | "
                         + " / ".join(f"{sh['shift_share_of_D'][i]:.2f}" for i in ix) + " | "
                         + " / ".join(f"{sh['width_change']['mean'][i]:+.3f}" for i in ix) + " | "
                         + " / ".join(f"{sh['shift_over_spread']['mean'][i]:+.3f}" for i in ix) + " |")
    lines += ["\n### Direction of the shift: Spearman rho of the signed shift with the straight-line change over the "
              "last 28 steps (in sd), 95% interval over whole series for real sources\n",
              "| Source | Model | h = 8 | h = 16 | share of contexts with the composite below |", "|---|---|---|---|---|"]
    for source in SOURCES + ("msar",):
        for model, r in summary.get(source, {}).items():
            if "shift_vs_recent_change" in r:
                s8, s16 = r["shift_vs_recent_change"]["h8"], r["shift_vs_recent_change"]["h16"]
                lines.append(f"| {source} | {LABEL[model]} | {s8[0]:+.2f} [{s8[1]:+.2f}, {s8[2]:+.2f}] | "
                             f"{s16[0]:+.2f} [{s16[1]:+.2f}, {s16[2]:+.2f}] | "
                             + " / ".join(f"{r['shift_negative_share'][i]:.2f}" for i in ix) + " |")
    lines += ["\n### Tail draws against central draws: D / width after putting one group on the central line in y "
              "(h = 8 / 16 / 64)\n", "| Source | Model | actual | tail draws on the line | central draws on the line |",
              "|---|---|---|---|---|"]
    for source in SOURCES + ("msar",):
        for model, r in summary.get(source, {}).items():
            sm = r["smoothed"]
            lines.append(f"| {source} | {LABEL.get(model, model)} | " + " / ".join(f3(r["D_over_spread"], i) for i in ix)
                         + " | " + " / ".join(f3(sm["tails_on_line"], i) for i in ix) + " | "
                         + " / ".join(f3(sm["centre_on_line"], i) for i in ix) + " |")
    keys = list(summary["covid"]["chronos-bolt-small"]["spearman_D_over_spread"])
    for model in ("chronos-bolt-small", "timestral-28m"):
        lines += [f"\n### Spearman rho of per-context D / width at h = 8 with context properties, {LABEL[model]} "
                  "(95% interval over whole series for real sources)\n",
                  "| Property | " + " | ".join(f"{s} median | {s} rho" for s in SOURCES) + " |",
                  "|---|" + "---|" * 2 * len(SOURCES)]
        for k in keys:
            cells = []
            for s in SOURCES:
                r = summary[s][model]
                rho = r["spearman_D_over_spread"][k]["h8"]
                cells += [f"{r['feature_medians'][k]:.3g}", f"{rho[0]:+.2f} [{rho[1]:+.2f}, {rho[2]:+.2f}]"]
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["\n### Slope of log(D / sd) on log(width / sd) across contexts\n",
              "| Model | h | pooled over 4 sources [95% CI] | " + " | ".join(SOURCES) + " |", "|---|---|---|" + "---|" * 4]
    for model, rows in slopes.items():
        for h, row in rows.items():
            p = row["pooled"]
            lines.append(f"| {LABEL[model]} | {h[1:]} | {p[0]:+.2f} [{p[1]:+.2f}, {p[2]:+.2f}] | "
                         + " | ".join(f"{row[s]:+.2f}" for s in SOURCES) + " |")
    if bumps:
        lines += ["\n### Scale probe: ar1 contexts with a bump of height A over their first 250 points (h = 8 / 16)\n",
                  "| Model | A | median sd(context) | width / sd | D / sd | D / width [95% CI at h = 8] |",
                  "|---|---|---|---|---|---|"]
        for model, rows in bumps.items():
            for a, r in rows.items():
                lines.append(f"| {LABEL[model]} | {a} | {r['median_sd']:.2f} | "
                             + " / ".join(f"{r['spread_over_sd']['mean'][i]:.3f}" for i in (2, 3)) + " | "
                             + " / ".join(f"{r['D_over_sd']['mean'][i]:.4f}" for i in (2, 3)) + " | "
                             + f"{ci(r['D_over_spread'], 2)} / {f3(r['D_over_spread'], 3)} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    summary: dict = {}
    for source in SOURCES + ("msar",):
        for model in MODELS + ("oracle",):
            run = summarise_run(source, model)
            if run is not None:
                summary.setdefault(source, {})[model] = run
    bumps = bump_summary()
    slopes = elasticities()
    write_json(RES / "summary.json", {"runs": summary, "bump": bumps, "elasticities": slopes})
    (RES / "tables.md").write_text(tables(summary, bumps, slopes))
    figs = RES / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig_units(figs / "defect_units.png")
    fig_scatter(figs / "defect_vs_width_per_context.png")
    fig_draws(summary, figs / "draw_profiles.png")
    fig_levels(summary, figs / "defect_by_level.png")
    fig_examples(figs / "covid_examples.png")
    fig_features(summary, figs / "features_bolt_small.png")
    if bumps:
        fig_bump(bumps, figs / "scale_probe.png")
    for source, runs in summary.items():
        for model, r in runs.items():
            fmt = lambda key: " ".join(f"{v:.4f}" for v in np.array(r[key]["mean"])[[2, 3, 5]])  # noqa: E731
            print(f"{source:8s} {model:20s} D/w {fmt('D_over_spread')} | D/sd {fmt('D_over_sd')} | "
                  f"w/sd {fmt('spread_over_sd')} | slide {fmt('slide_D_over_spread')} | "
                  f"tails-on-line {' '.join(f'{v:.4f}' for v in np.array(r['smoothed']['tails_on_line']['mean'])[[2, 3, 5]])} | "
                  f"centre-on-line {' '.join(f'{v:.4f}' for v in np.array(r['smoothed']['centre_on_line']['mean'])[[2, 3, 5]])} | "
                  f"shift share {' '.join(f'{v:.2f}' for v in np.array(r['shape']['shift_share_of_D'])[[2, 3, 5]])} | "
                  f"width chg {' '.join(f'{v:+.3f}' for v in np.array(r['shape']['width_change']['mean'])[[2, 3, 5]])}")
    for model, rows in bumps.items():
        print(model, {a: [round(v, 4) for v in np.array(r["D_over_spread"]["mean"])[[2, 3, 5]]] for a, r in rows.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
