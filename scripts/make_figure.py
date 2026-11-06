"""Figure 2 of the report: the measured self-consistency gap D_h against the horizon h.

Four panels. Synthetic sources: the exact forecaster as the baseline, intervals over contexts.
Real sources: the Gaussian and residual-noise AR(8) baselines, and every interval resamples
whole series (from results/rigour/summary.json). Reads only stored summaries (.json) and writes
results/figures/gap_vs_horizon.{pdf,png}.

    python scripts/make_figure.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hsc.common.paths import RESULTS  # noqa: E402

RES = RESULTS
OUT = RESULTS / "figures"
HORIZONS = [2, 4, 8, 16, 32, 64]

# (folder, file stem, label, colour, marker)
MODELS = [
    ("main", "chronos-bolt-small", "Chronos-Bolt small", "#1f4e9c", "o"),
    ("main", "timestral-28m", "Timestral-28M", "#e07b00", "s"),
    ("controls", "timesfm-2.5-200m", "TimesFM 2.5", "#2a8a3e", "^"),
]


def load(folder, source, stem):
    p = RES / folder / source / f"{stem}.json"
    return json.loads(p.read_text())["D_over_spread"] if p.exists() else None


def errbar(ax, r, colour, marker, label):
    ax.errorbar(HORIZONS, r["mean"],
                yerr=[[m - lo for m, lo in zip(r["mean"], r["lo"])],
                      [hi - m for m, hi in zip(r["mean"], r["hi"])]],
                color=colour, marker=marker, ms=3, lw=1.1, capsize=1.5, label=label)


def style(ax, title):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.5)
    ax.minorticks_off()


def figure(out_stem="gap_vs_horizon", ref="self-consistent baseline", ylabel="measured gap $D_h$"):
    rig = json.loads((RES / "rigour" / "summary.json").read_text())["sources"]
    panels = [("ar1", "AR(1)"), ("ar2", "AR(2)"), ("covid", "COVID-19 cases"), ("weather", "Air temperature")]
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7.5})
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
    for ax, (src, title) in zip(axes, panels):
        real = not src.startswith("ar")
        if real:
            runs = rig[src]["runs"]
            refs = [(runs["arfit"]["by_series"], "-", f"{ref} (Gaussian)"),
                    (runs["arresid-p8"]["by_series"], ":", f"{ref} (residual noise)")]
        else:
            refs = [(load("main", src, "oracle"), "-", f"{ref} (Gaussian)")]
        for r, ls, label in refs:
            ax.fill_between(HORIZONS, r["lo"], r["hi"], color="0.6", alpha=0.45, lw=0)
            ax.plot(HORIZONS, r["mean"], color="0.3", lw=1.2, ls=ls, label=label)
        if src == "ar2":
            pw = load("main", src, "patchwork")
            ax.plot(HORIZONS, pw["mean"], color="#b2182b", lw=1.2, ls="--", marker=".", ms=4,
                    label="patchwork (positive control)")
        for folder, stem, label, colour, marker in MODELS:
            r = rig[src]["runs"][stem]["by_series"] if real else load(folder, src, stem)
            errbar(ax, r, colour, marker, label)
        style(ax, title)
        ax.set_xlabel("horizon $h$")
    axes[0].set_ylabel(ylabel)
    handles, labels = [], []
    for ax in (axes[1], axes[2]):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    order = ["Chronos-Bolt small", "Timestral-28M", "TimesFM 2.5", "patchwork (positive control)",
             f"{ref} (Gaussian)", f"{ref} (residual noise)"]
    fig.legend([handles[labels.index(lab)] for lab in order], order, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(pad=0.3, w_pad=0.5, rect=(0, 0.17, 1, 1))
    fig.savefig(OUT / f"{out_stem}.pdf")
    fig.savefig(OUT / f"{out_stem}.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figure()


if __name__ == "__main__":
    main()
