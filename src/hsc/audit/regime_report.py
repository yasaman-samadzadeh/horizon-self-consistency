"""Tables and figures for regime switching, Table S1 (reads results/regime/, written by hsc.audit.regime).

Splits use what the context shows (a switch inside it, how recent the last one is, how sure the
exact filter is of the current regime) and, as a placebo, whether a switch happens in the
forecast window. Under this law the window switch is independent of the context (the switch
probability is the same in both regimes), so no label-free quantity such as D can depend on it.

    python -m hsc.audit.regime_report
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hsc.audit.gap import HORIZONS
from hsc.audit.windows import CONTEXT
from hsc.common.paths import RESULTS
from hsc.common.stats import bootstrap_mean, bootstrap_ratio, write_json

RES = RESULTS / "regime"
SOURCES = ("msar", "msar_fast", "msar_p0")
MODELS = ("oracle", "arfit", "chronos-bolt-small", "chronos-bolt-base", "timestral-28m")
TSFMS = MODELS[2:]
LABEL = {"oracle": "exact Bayes (floor)", "arfit": "AR(8) fit (coherent)", "chronos-bolt-small": "Chronos-Bolt small",
         "chronos-bolt-base": "Chronos-Bolt base", "timestral-28m": "Timestral-28M"}
COLOR = {"oracle": "k", "arfit": "0.55", "chronos-bolt-small": "C0", "chronos-bolt-base": "C1", "timestral-28m": "C2"}
RECENT = 50  # a switch in the last RECENT steps of the context counts as recent


def load(source: str) -> dict:
    runs = {}
    for name in MODELS:
        path = RES / source / f"{name}.npz"
        if path.exists():
            runs[name] = dict(np.load(path))
    ref = runs["oracle"]
    for name, z in runs.items():
        assert np.array_equal(z["contexts"], ref["contexts"]), (source, name)
    return runs


def context_groups(regimes: np.ndarray, posterior: np.ndarray) -> dict:
    """Boolean masks over contexts, from the true regime path and the exact filter."""
    flips = regimes[:, 1:] != regimes[:, :-1]  # flips[:, t-1] is a switch into step t
    in_context = flips[:, : CONTEXT - 1]
    last = np.where(in_context.any(axis=1), CONTEXT - 1 - np.argmax(in_context[:, ::-1], axis=1), -1)
    age = np.where(last >= 0, CONTEXT - 1 - last, np.inf)  # steps since the last switch
    uncertain = posterior.min(axis=1) > 0.01
    return {"no switch in context": ~in_context.any(axis=1),
            f"last switch > {RECENT} steps before end": in_context.any(axis=1) & (age > RECENT),
            f"last switch <= {RECENT} steps before end": age <= RECENT,
            "filter sure of regime (min P <= 0.01)": ~uncertain,
            "filter unsure of regime (min P > 0.01)": uncertain}


def window_switch(regimes: np.ndarray) -> np.ndarray:
    """(n, len(HORIZONS)): a switch happens within the first h steps of the forecast window."""
    flips = regimes[:, CONTEXT:] != regimes[:, CONTEXT - 1 : -1]
    return np.stack([flips[:, :h].any(axis=1) for h in HORIZONS], axis=1)


def summarise_source(source: str) -> dict:
    runs = load(source)
    floor = runs["oracle"]["D"] / runs["oracle"]["spread"]
    out = {"n": int(len(floor)), "models": {}}
    regimes, posterior = runs["oracle"]["regimes"], runs["oracle"]["posterior"]
    groups = context_groups(regimes, posterior)
    future = window_switch(regimes)
    out["group_sizes"] = {k: int(v.sum()) for k, v in groups.items()}
    out["window_switch_frac"] = future.mean(axis=0).tolist()
    for name, z in runs.items():
        ratio = z["D"] / z["spread"]
        crps_rel = z["crps"].mean(axis=0) / runs["oracle"]["crps"].mean(axis=0)
        m = {"D_over_spread": bootstrap_mean(ratio), "median_D_over_spread": np.median(ratio, axis=0).tolist(),
             "crps": bootstrap_mean(z["crps"]), "crps_over_oracle": crps_rel.tolist(),
             "spread_over_oracle": (z["spread"].mean(axis=0) / runs["oracle"]["spread"].mean(axis=0)).tolist()}
        if name != "oracle":
            m["times_floor"] = bootstrap_ratio(ratio, floor)
        m["groups"] = {k: bootstrap_mean(ratio[mask]) for k, mask in groups.items() if mask.sum() >= 5}
        # placebo: realised switch in the window, horizon by horizon
        mean = lambda v, mask: float(v[mask].mean()) if mask.any() else float("nan")  # noqa: E731
        cols = range(len(HORIZONS))
        m["window_switch"] = {"n_yes": future.sum(axis=0).tolist(),
                              "yes": [mean(ratio[:, i], future[:, i]) for i in cols],
                              "no": [mean(ratio[:, i], ~future[:, i]) for i in cols],
                              "crps_yes": [mean(z["crps"][:, i], future[:, i]) for i in cols],
                              "crps_no": [mean(z["crps"][:, i], ~future[:, i]) for i in cols],
                              "diff_ci": [_diff_ci(ratio[future[:, i], i], ratio[~future[:, i], i]) for i in cols]}
        out["models"][name] = m
    exact = RES / source / "oracle_exactdraws.npz"
    if exact.exists():
        e = np.load(exact)
        out["floor_exact_draws"] = bootstrap_mean(e["D"] / e["spread"])
    return out


def _diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list:
    """Difference of means a - b with a percentile interval from independent resampling of each group."""
    if len(a) < 5 or len(b) < 5:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    boots = (a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
             - b[rng.integers(0, len(b), (n_boot, len(b)))].mean(axis=1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return [float(a.mean() - b.mean()), float(lo), float(hi)]


def fig_by_horizon(summary: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(SOURCES), figsize=(4.2 * len(SOURCES), 3.6), sharey=True)
    titles = {"msar": "switching, p = 0.005", "msar_fast": "switching, p = 0.02", "msar_p0": "no switching, p = 0"}
    for ax, source in zip(axes, SOURCES):
        for name, m in summary[source]["models"].items():
            d = m["D_over_spread"]
            ax.plot(HORIZONS, d["mean"], "o-", color=COLOR[name], label=LABEL[name], ms=3)
            ax.fill_between(HORIZONS, d["lo"], d["hi"], color=COLOR[name], alpha=0.2)
        if "floor_exact_draws" in summary[source]:
            ax.plot(HORIZONS, summary[source]["floor_exact_draws"]["mean"], "k--", lw=0.9,
                    label="floor, draws from exact one-step law")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(HORIZONS, [str(h) for h in HORIZONS])
        ax.set_xlabel("horizon h")
        ax.set_title(titles[source])
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("D_h / 80% interval width")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_groups(summary: dict, source: str, path: Path) -> None:
    keys = [k for k in summary[source]["group_sizes"] if not k.startswith("filter")]
    keys += [k for k in summary[source]["group_sizes"] if k.startswith("filter")]
    hs = (8, 16, 64)
    fig, axes = plt.subplots(1, len(hs), figsize=(13, 3.8), sharey=True)
    names = [n for n in MODELS if n in summary[source]["models"]]
    width = 0.8 / len(names)
    for ax, h in zip(axes, hs):
        i = HORIZONS.index(h)
        for k, name in enumerate(names):
            groups = summary[source]["models"][name]["groups"]
            xs, ys, err = [], [], []
            for g, key in enumerate(keys):
                if key in groups:
                    d = groups[key]
                    xs.append(g + (k - (len(names) - 1) / 2) * width)
                    ys.append(d["mean"][i])
                    err.append([d["mean"][i] - d["lo"][i], d["hi"][i] - d["mean"][i]])
            ax.bar(xs, ys, width, color=COLOR[name], label=LABEL[name], yerr=np.array(err).T, capsize=2,
                   error_kw={"lw": 0.8})
        sizes = summary[source]["group_sizes"]
        ax.set_xticks(range(len(keys)), [f"{k}\n(n={sizes[k]})" for k in keys], rotation=30, ha="right", fontsize=6.5)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 5e-2)
        ax.set_title(f"h = {h}")
        ax.set_axisbelow(True)
        ax.grid(alpha=0.3, axis="y", which="both")
    axes[0].set_ylabel("D_h / 80% interval width")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(names), fontsize=7, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle(f"{source}: defect by what the context shows (95% bootstrap intervals)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_example(source: str, path: Path) -> None:
    """The context whose current regime the exact filter is least sure of: exact forecast against Chronos-Bolt small."""
    runs = load(source)
    b = int(np.argmax(runs["oracle"]["posterior"].min(axis=1)))
    x = runs["oracle"]["contexts"][b]
    truth = runs["oracle"]["truth"][b]
    t_ctx = np.arange(CONTEXT - 150, CONTEXT)
    t_fc = CONTEXT + np.arange(len(truth))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(t_ctx, x[-150:], color="0.3", lw=0.8, label="context")
    ax.plot(t_fc, truth, color="0.6", lw=0.8, label="realised")
    r = runs["oracle"]["regimes"][b]
    ax.scatter(np.arange(CONTEXT - 150, CONTEXT + len(truth)), np.full(150 + len(truth), x.min() - 0.8),
               c=np.where(r[CONTEXT - 150 :] == 1, "C3", "C9"), s=2, label="true regime (red = 1)")
    for name in ("oracle", "chronos-bolt-small"):
        if name in runs:
            q = runs[name]["direct"][b]
            ax.plot(t_fc, q[:, 4], color=COLOR[name], lw=1.2, label=f"{LABEL[name]} median")
            ax.fill_between(t_fc, q[:, 0], q[:, -1], color=COLOR[name], alpha=0.15)
    ax.set_title(f"{source}, context {b}: P(regime 1 at end) = {runs['oracle']['posterior'][b, 1]:.2f}", fontsize=9)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _cell(d: dict, i: int, digits: int = 4) -> str:
    return f"{d['mean'][i]:.{digits}f} [{d['lo'][i]:.{digits}f}, {d['hi'][i]:.{digits}f}]"


def tables(summary: dict) -> str:
    """Markdown tables of the key numbers (written to tables.md, copied into SUMMARY.md)."""
    lines = []
    hs = (8, 16, 64)
    for source, s in summary.items():
        lines += [f"\n### {source}: D_h / 80% width, mean [95% bootstrap interval], n = {s['n']}\n",
                  "| Model | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |",
                  "|---|" + "---|" * len(HORIZONS)]
        for name, m in s["models"].items():
            lines.append(f"| {LABEL[name]} | " + " | ".join(_cell(m["D_over_spread"], i) for i in range(len(HORIZONS)))
                         + " |")
        if "floor_exact_draws" in s:
            lines.append("| floor, draws from exact one-step law | "
                         + " | ".join(_cell(s["floor_exact_draws"], i) for i in range(len(HORIZONS))) + " |")
        lines += ["\n| Model | " + " | ".join(f"x floor, h = {h}" for h in hs) + " | "
                  + " | ".join(f"CRPS / exact, h = {h}" for h in hs) + " |", "|---|" + "---|" * 2 * len(hs)]
        for name, m in s["models"].items():
            if name == "oracle":
                continue
            tf = m["times_floor"]
            lines.append(f"| {LABEL[name]} | "
                         + " | ".join(f"{tf['mean'][HORIZONS.index(h)]:.1f} [{tf['lo'][HORIZONS.index(h)]:.1f}, "
                                      f"{tf['hi'][HORIZONS.index(h)]:.1f}]" for h in hs) + " | "
                         + " | ".join(f"{m['crps_over_oracle'][HORIZONS.index(h)]:.3f}" for h in hs) + " |")
        if source == "msar_p0":
            continue
        groups = [k for k in s["group_sizes"] if s["group_sizes"][k] >= 5]
        lines += [f"\n{source}, D_h / width by what the context shows (h = 8 / 16 / 64):\n",
                  "| Group (n) | " + " | ".join(LABEL[name] for name in s["models"]) + " |",
                  "|---|" + "---|" * len(s["models"])]
        for g in groups:
            cells = []
            for name, m in s["models"].items():
                d = m["groups"][g]
                cells.append(" / ".join(f"{d['mean'][HORIZONS.index(h)]:.4f}" for h in hs))
            lines.append(f"| {g} ({s['group_sizes'][g]}) | " + " | ".join(cells) + " |")
        n_yes = s["models"]["oracle"]["window_switch"]["n_yes"]
        lines += [f"\n{source}, placebo: realised switch in the first h steps of the window (h = 8 / 16 / 64; "
                  f"windows with a switch: {' / '.join(str(n_yes[HORIZONS.index(h)]) for h in hs)}):\n",
                  "| Model | D/width, switch | D/width, none | difference [95% CI] | CRPS, switch | CRPS, none |",
                  "|---|---|---|---|---|---|"]
        for name, m in s["models"].items():
            w = m["window_switch"]
            ix = [HORIZONS.index(h) for h in hs]
            lines.append(f"| {LABEL[name]} | " + " / ".join(f"{w['yes'][i]:.4f}" for i in ix) + " | "
                         + " / ".join(f"{w['no'][i]:.4f}" for i in ix) + " | "
                         + " / ".join(f"{w['diff_ci'][i][0]:+.4f} [{w['diff_ci'][i][1]:+.4f}, {w['diff_ci'][i][2]:+.4f}]"
                                      for i in ix) + " | "
                         + " / ".join(f"{w['crps_yes'][i]:.2f}" for i in ix) + " | "
                         + " / ".join(f"{w['crps_no'][i]:.2f}" for i in ix) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    summary = {s: summarise_source(s) for s in SOURCES if (RES / s / "oracle.npz").exists()}
    write_json(RES / "summary.json", summary)
    (RES / "tables.md").write_text(tables(summary))
    figs = RES / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    if len(summary) == len(SOURCES):
        fig_by_horizon(summary, figs / "defect_by_horizon.png")
    for source in summary:
        if source != "msar_p0":
            fig_groups(summary, source, figs / f"groups_{source}.png")
    fig_example("msar", figs / "example_msar.png")
    for source, s in summary.items():
        print(f"\n== {source} (n={s['n']}) groups {s['group_sizes']} window switch frac {np.round(s['window_switch_frac'], 2)}")
        for name, m in s["models"].items():
            d = m["D_over_spread"]
            cells = [f"{d['mean'][i]:.4f} [{d['lo'][i]:.4f}, {d['hi'][i]:.4f}]" for i in range(len(HORIZONS))]
            tf = m.get("times_floor")
            ratio = "" if tf is None else " x floor " + ", ".join(f"{v:.1f}" for v in tf["mean"])
            print(f"  {name:20s} " + " | ".join(cells) + ratio)
    print(json.dumps({s: v["group_sizes"] for s, v in summary.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
