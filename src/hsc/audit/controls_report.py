"""Tables and figures for every direct model (results/main/ and results/controls/) and the Chronos-T5
sample pipeline: the numbers behind Table 1.

Every number is D/spread per context, averaged over contexts, with a percentile bootstrap over
contexts. Self-consistent baselines ("floors" below): the exact forecaster on synthetic sources and the
per-context Gaussian AR(8) fit on real ones. Model and baseline share contexts, so differences use a
paired bootstrap.

    python -m hsc.audit.controls_report   # writes results/controls/{tables.md, summary.json, *.png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hsc.audit.baselines import ExactForecaster  # noqa: E402
from hsc.audit.gap import HORIZONS  # noqa: E402
from hsc.audit.windows import H_MAX, REAL, SYNTHETIC  # noqa: E402
from hsc.common.paths import RESULTS  # noqa: E402
from hsc.common.stats import bootstrap_mean, write_json  # noqa: E402

MAIN = RESULTS / "main"
OUT = RESULTS / "controls"
SOURCES = list(SYNTHETIC) + list(REAL)
SHOW_H = (8, 16, 64)

# name: (results folder, label, colour)
DIRECT = {
    "chronos-bolt-tiny": (OUT, "Chronos-Bolt tiny (9M)", "#9ecae1"),
    "chronos-bolt-mini": (OUT, "Chronos-Bolt mini (21M)", "#6baed6"),
    "chronos-bolt-small": (MAIN, "Chronos-Bolt small (48M)", "#2171b5"),
    "chronos-bolt-base": (MAIN, "Chronos-Bolt base (205M)", "#08306b"),
    "timestral-28m": (MAIN, "Timestral-28M", "#d9822b"),
    "timesfm-2.5-200m": (OUT, "TimesFM 2.5 (231M)", "#2e7d32"),
    "timesfm-2.5-200m-pos": (OUT, "TimesFM 2.5, infer_is_positive", "#81c784"),
    "timesfm-2.5-200m-noflip": (OUT, "TimesFM 2.5, no flip averaging", "#c5e1a5"),
}
FLOORS = {"oracle": "Oracle (floor)", "arfit": "AR(8) fit (floor)"}
SAMPLED = {
    "oracle-samples": ("Oracle, sampled (floor)", "#444444"),
    "chronos-t5-small": ("Chronos-T5 small (46M)", "#b03a2e"),
    "chronos-t5-small-topk4096": ("Chronos-T5 small, top_k = 4096", "#e57368"),
    "chronos-t5-mini": ("Chronos-T5 mini (21M)", "#7b1fa2"),
    "chronos-t5-mini-topk4096": ("Chronos-T5 mini, top_k = 4096", "#ba68c8"),
    "chronos-bolt-small-samples": ("Chronos-Bolt small, sampled", "#2171b5"),
    "chronos-bolt-base-samples": ("Chronos-Bolt base, sampled", "#08306b"),
    "timestral-28m-samples": ("Timestral-28M, sampled", "#d9822b"),
}
LABELS = {**{k: v[1] for k, v in DIRECT.items()}, **{k: v[0] for k, v in SAMPLED.items()}}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def ratio(folder: Path, source: str, model: str) -> np.ndarray | None:
    path = folder / source / f"{model}.npz"
    if not path.exists():
        return None
    r = np.load(path)
    return r["D"] / r["spread"]  # (n contexts, len(HORIZONS))


def floor_name(source: str) -> str:
    return "oracle" if source in SYNTHETIC else "arfit"


def fmt(st: dict, i: int, digits: int = 4) -> str:
    return f"{st['mean'][i]:.{digits}f} [{st['lo'][i]:.{digits}f}, {st['hi'][i]:.{digits}f}]"


def compare(r: np.ndarray, floor: np.ndarray) -> dict:
    """Model against its floor on the same contexts."""
    st, fst, diff = bootstrap_mean(r), bootstrap_mean(floor), bootstrap_mean(r - floor)
    return {"D_over_spread": st, "floor": fst, "paired_difference": diff,
            "fold": (np.asarray(st["mean"]) / np.asarray(fst["mean"])).tolist(),
            "separated": [bool(st["lo"][i] > fst["hi"][i]) for i in range(len(HORIZONS))],
            "difference_excludes_0": [bool(diff["lo"][i] > 0) for i in range(len(HORIZONS))]}


# ---------------------------------------------------------------- direct models, quantile pipeline
def direct_models() -> tuple[dict, list[str]]:
    out, lines = {}, []
    for source in SOURCES:
        floor = ratio(MAIN, source, floor_name(source))
        out[source] = {"floor": floor_name(source), "models": {}}
        for name, (folder, _, _) in DIRECT.items():
            r = ratio(folder, source, name)
            if r is not None and floor is not None:
                out[source]["models"][name] = compare(r, floor)
        for name in FLOORS:
            r = ratio(MAIN, source, name)
            if r is not None:
                out[source][f"{name}_D_over_spread"] = bootstrap_mean(r)

    for h in SHOW_H:
        i = HORIZONS.index(h)
        lines += [f"### D/spread at h = {h} (n = 256 contexts per source, inner = 32, quantile pipeline)", "",
                  "| Model | " + " | ".join(SOURCES) + " |", "|---|" + "---|" * len(SOURCES)]
        for name, label in FLOORS.items():
            cells = [fmt(out[s][f"{name}_D_over_spread"], i) if f"{name}_D_over_spread" in out[s] else "–"
                     for s in SOURCES]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        for name, (_, label, _) in DIRECT.items():
            cells = [fmt(out[s]["models"][name]["D_over_spread"], i) if name in out[s]["models"] else "–"
                     for s in SOURCES]
            if any(c != "–" for c in cells):
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    # at h = 64 the floor is ~0 (both forecasts sit at the stationary law), so multiples are not shown
    lines += ["### Multiple of the floor at h = 8 / 16 (* = 95% CI above the floor's CI, the pre-declared criterion)", "",
              "| Model | " + " | ".join(SOURCES) + " |", "|---|" + "---|" * len(SOURCES)]
    for name, (_, label, _) in DIRECT.items():
        cells = []
        for s in SOURCES:
            c = out[s]["models"].get(name)
            if c is None:
                cells.append("–")
                continue
            idx = [HORIZONS.index(h) for h in (8, 16)]
            cells.append(" / ".join(f"{c['fold'][i]:.1f}" + ("*" if c["separated"][i] else "") for i in idx))
        if any(c != "–" for c in cells):
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return out, lines


# ---------------------------------------------------------------- Chronos-T5 control, sample pipeline
def sampled_models() -> tuple[dict, list[str]]:
    out, lines = {}, []
    for source in ("ar1", "ar2"):
        floor = ratio(OUT, source, "oracle-samples")
        if floor is None:
            continue
        out[source] = {"oracle-samples": bootstrap_mean(floor)}
        for name in SAMPLED:
            r = ratio(OUT, source, name)
            if r is not None and name != "oracle-samples":
                out[source][name] = compare(r, floor)
        head = " | ".join(f"h = {h}" for h in HORIZONS)
        lines += [f"### {source}: sample-based pipeline (n = 64, inner = 16, 4096 direct and 256 inner samples)", "",
                  f"| Model | {head} |", "|---|" + "---|" * len(HORIZONS),
                  f"| {SAMPLED['oracle-samples'][0]} | " + " | ".join(fmt(out[source]['oracle-samples'], i)
                                                              for i in range(len(HORIZONS))) + " |"]
        models = [m for m in SAMPLED if m in out[source] and m != "oracle-samples"]
        for name in models:
            st = out[source][name]["D_over_spread"]
            lines.append(f"| {LABELS[name]} | " + " | ".join(fmt(st, i) for i in range(len(HORIZONS))) + " |")
        lines += ["", f"Paired difference from the sampled oracle on the same 64 contexts ({source}):", "",
                  f"| Model | {head} |", "|---|" + "---|" * len(HORIZONS)]
        for name in models:
            diff = out[source][name]["paired_difference"]
            lines.append(f"| {LABELS[name]} | " + " | ".join(fmt(diff, i) for i in range(len(HORIZONS))) + " |")
        lines.append("")
    return out, lines


def like_with_like() -> tuple[dict, list[str]]:
    """Each direct model's excess over its own floor, in the quantile pipeline (all 256 contexts and
    the 64 shared with the control) and in the sample pipeline, next to Chronos-T5's."""
    out, lines = {}, []
    lines += ["### Excess over the floor: quantile pipeline against sample pipeline (paired mean difference [95% CI])",
              "", "| Source | Model | Pipeline | " + " | ".join(f"h = {h}" for h in SHOW_H) + " |",
              "|---|---|---|" + "---|" * len(SHOW_H)]
    idx = [HORIZONS.index(h) for h in SHOW_H]
    for source in ("ar1", "ar2"):
        qfloor, sfloor = ratio(MAIN, source, "oracle"), ratio(OUT, source, "oracle-samples")
        if qfloor is None or sfloor is None:
            continue
        n = len(sfloor)
        out[source] = {}
        for name in ("chronos-bolt-small", "chronos-bolt-base", "timestral-28m", "chronos-t5-small"):
            entry = {}
            q = ratio(MAIN, source, name)
            if q is not None:
                entry["quantile_n256"] = bootstrap_mean(q - qfloor)
                entry["quantile_first64"] = bootstrap_mean(q[:n] - qfloor[:n])
            s = ratio(OUT, source, name if name == "chronos-t5-small" else f"{name}-samples")
            if s is not None:
                entry["samples_n64"] = bootstrap_mean(s - sfloor)
            out[source][name] = entry
            label = LABELS[name]
            for key, desc in (("quantile_n256", "quantiles, n = 256"), ("quantile_first64", "quantiles, same 64"),
                              ("samples_n64", "samples, n = 64")):
                if key in entry:
                    lines.append(f"| {source} | {label} | {desc} | " + " | ".join(fmt(entry[key], i) for i in idx) + " |")
    lines.append("")
    return out, lines


def yardstick() -> tuple[dict, list[str]]:
    """On ar1/ar2, D and each model's own 80% width in units of the true law's 80% width.

    The true width is the same for every model on a context, so D/true-width compares the size of
    the composition gap across models without crediting or penalising their own spread.
    """
    out, lines = {}, []
    idx = [HORIZONS.index(h) for h in (2, 8, 16, 64)]
    lines += ["### Common yardstick on ar1/ar2: D and the model's own 80% width, in units of the true law's 80% width",
              "", "| Source | Model | n | D / true width at h = 2, 8, 16, 64 | own width / true width at h = 2, 8, 16, 64 |",
              "|---|---|---|---|---|"]
    for source in ("ar1", "ar2"):
        out[source] = {}
        _, sd = ExactForecaster(source).law.moments(np.zeros((1, 8)), H_MAX)  # Gaussian AR: sd does not depend on c
        sd = sd[0]
        models = [(MAIN, "oracle"), (OUT, "oracle-samples")] + [(f, m) for m, (f, _, _) in DIRECT.items()] \
            + [(OUT, m) for m in SAMPLED if m != "oracle-samples"]
        for folder, name in models:
            path = folder / source / f"{name}.npz"
            if not path.exists():
                continue
            r = np.load(path)
            true_width = 2 * 1.2815515655446004 * sd[np.asarray(HORIZONS) - 1]  # same for every context
            d, w = r["D"] / true_width, r["spread"] / true_width
            out[source][name] = {"D_over_true_width": bootstrap_mean(d), "width_over_true_width": bootstrap_mean(w)}
            label = LABELS.get(name, FLOORS.get(name, name)) + (" [sample pipeline]" if folder == OUT and
                                                                name in SAMPLED else "")
            lines.append(f"| {source} | {label} | {len(d)} | "
                         + ", ".join(f"{np.mean(d[:, i]):.4f}" for i in idx) + " | "
                         + ", ".join(f"{np.mean(w[:, i]):.2f}" for i in idx) + " |")
    lines.append("")
    return out, lines


# ---------------------------------------------------------------- figures
def band(ax, st, color, label, lw=1.4, alpha=0.18, ls="-"):
    hs = np.asarray(HORIZONS)
    ax.plot(hs, st["mean"], color=color, lw=lw, label=label, ls=ls, marker="o", ms=2.5)
    ax.fill_between(hs, st["lo"], st["hi"], color=color, alpha=alpha, lw=0)


def fig_direct(direct: dict) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.8), sharex=True)
    for ax, source in zip(axes.flat, SOURCES):
        fname = floor_name(source)
        fst = direct[source].get(f"{fname}_D_over_spread")
        if fst is not None:
            band(ax, fst, "#777777", FLOORS[fname], lw=1, alpha=0.35)
        for name, (_, label, color) in DIRECT.items():
            c = direct[source]["models"].get(name)
            if c is not None:
                band(ax, c["D_over_spread"], color, label, ls="--" if name.endswith("-pos") else "-")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(HORIZONS, [str(h) for h in HORIZONS])
        ax.set_title(source)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[-2].get_legend_handles_labels()
    for ax in axes.flat[:-1]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("D / spread")
    for ax in axes[-1]:
        ax.set_xlabel("horizon h")
    fig.suptitle("Self-consistency gap of direct models (quantile pipeline, n = 256; bands are 95% bootstrap CIs)")
    fig.tight_layout()
    path = OUT / "controls_direct.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_h16(direct: dict) -> Path:
    i = HORIZONS.index(16)
    names = [n for n in DIRECT if any(n in direct[s]["models"] for s in SOURCES)]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    width = 0.8 / (len(names) + 1)
    for k, source in enumerate(SOURCES):
        fname = floor_name(source)
        fst = direct[source].get(f"{fname}_D_over_spread")
        if fst is not None:
            ax.errorbar(k - 0.4 + width / 2, fst["mean"][i], yerr=[[fst["mean"][i] - fst["lo"][i]],
                        [fst["hi"][i] - fst["mean"][i]]], fmt="s", color="#777777", ms=4,
                        label="floor (oracle / AR(8) fit)" if k == 0 else None)
        for j, name in enumerate(names):
            c = direct[source]["models"].get(name)
            if c is None:
                continue
            st = c["D_over_spread"]
            ax.errorbar(k - 0.4 + width * (j + 1.5), st["mean"][i],
                        yerr=[[st["mean"][i] - st["lo"][i]], [st["hi"][i] - st["mean"][i]]],
                        fmt="o", color=DIRECT[name][2], ms=4, label=DIRECT[name][1] if k == 0 else None)
    ax.set_yscale("log")
    ax.set_xticks(range(len(SOURCES)), SOURCES)
    ax.set_ylabel("D / spread at h = 16")
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")
    ax.set_title("Self-consistency gap at h = 16 by source (95% bootstrap CIs; n = 256)")
    fig.tight_layout()
    path = OUT / "controls_h16.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_t5(sampled: dict, yard: dict) -> Path:
    """Rows: D / own spread (the pre-specified metric), D / true width, own width / true width."""
    rows = (("D / own 80% width", lambda s, m: sampled[s]["oracle-samples"] if m == "oracle-samples"
             else sampled[s][m]["D_over_spread"]),
            ("D / true 80% width", lambda s, m: yard[s][m]["D_over_true_width"]),
            ("own width / true width", lambda s, m: yard[s][m]["width_over_true_width"]))
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True, sharey="row")
    for col, source in enumerate(("ar1", "ar2")):
        if source not in sampled:
            continue
        for row, (ylabel, get) in enumerate(rows):
            ax = axes[row, col]
            for name, (label, color) in SAMPLED.items():
                if name not in yard[source]:
                    continue
                t5 = name.startswith("chronos-t5")
                band(ax, get(source, name), color, label, lw=2 if t5 else 1,
                     ls="-" if t5 or name == "oracle-samples" else "--", alpha=0.35 if name == "oracle-samples" else 0.18)
            ax.set_xscale("log", base=2)
            ax.set_xticks(HORIZONS, [str(h) for h in HORIZONS])
            if row == 0:
                ax.set_title(source)
            if col == 0:
                ax.set_ylabel(ylabel)
        axes[-1, col].set_xlabel("horizon h")
    axes[0, 1].legend(fontsize=7, frameon=False)
    fig.suptitle("Sample-based pipeline (n = 64 contexts; 95% bootstrap CIs)")
    fig.tight_layout()
    path = OUT / "controls_t5.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    direct, d_lines = direct_models()
    sampled, s_lines = sampled_models()
    lwl, l_lines = like_with_like()
    yard, y_lines = yardstick()
    figs = [fig_direct(direct), fig_h16(direct), fig_t5(sampled, yard)]
    (OUT / "tables.md").write_text("\n".join(["## Chronos-T5 control", ""] + s_lines + l_lines + y_lines
                                             + ["## Direct models", ""] + d_lines))
    write_json(OUT / "summary.json", {"horizons": list(HORIZONS), "direct": direct, "sampled": sampled,
                                      "like_with_like": lwl, "yardstick": yard,
                                      "figures": [str(p.relative_to(RESULTS.parent)) for p in figs]})
    print("\n".join(s_lines + l_lines + y_lines + d_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
