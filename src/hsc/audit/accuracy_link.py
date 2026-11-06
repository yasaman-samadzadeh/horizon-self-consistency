"""Accuracy link: how the self-consistency gap relates to error against the exact distribution.

A descriptive analysis of stored outputs, added after the main results (no new model runs and no
criterion fixed in advance). Per context and horizon, with every distance divided by
the exact law's 80% width (a yardstick that is the same for every model):

    D      = band_w1(direct, composed)   the self-consistency gap (composed = chained forecast)
    E_dir  = band_w1(direct, exact)      error of the direct forecast
    E_comp = band_w1(composed, exact)    error of the composed forecast

band_w1 is a scaled L1 norm on the nine quantiles, so D <= E_dir + E_comp holds exactly.

On AR(1) it also scores the conditional forecasts that hsc.audit.mechanism stored, the model's
forecast of y_{t+h} from c + y_m ("what if the next value is y_m?"), against the exact conditional.

    python -m hsc.audit.accuracy_link
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from hsc.audit.baselines import ExactForecaster
from hsc.audit.gap import HORIZONS
from hsc.audit.windows import SYNTHETIC, load_windows
from hsc.common.paths import RESULTS
from hsc.common.quantiles import band_w1, gaussian_quantiles, mixture_quantiles, quantile_crps
from hsc.common.stats import bootstrap_mean, write_json

RES = RESULTS
OUT = RES / "accuracy"
SEED, N = 20261104, 256
IDX = [h - 1 for h in HORIZONS]
MODELS = {  # model -> results folder holding <source>/<model>.npz
    "chronos-bolt-tiny": "controls",
    "chronos-bolt-mini": "controls",
    "chronos-bolt-small": "main",
    "chronos-bolt-base": "main",
    "timestral-28m": "main",
    "timesfm-2.5-200m": "controls",
    "patchwork": "main",
    "oracle": "main",
}
WHATIF_MODELS = ("chronos-bolt-small", "chronos-bolt-base", "timestral-28m", "oracle")
TAILS = [0, 1, 30, 31]  # stratified draws at u = 0.016, 0.047, 0.953, 0.984
CENTRE = list(range(8, 24))  # u between 0.27 and 0.73


def by_model(source: str) -> dict:
    contexts, future = load_windows(source, N, SEED)
    exact = ExactForecaster(source)(contexts, max(HORIZONS), None)
    ex = exact[:, IDX, :]
    yard = ex[..., -1] - ex[..., 0]
    out = {}
    for model, folder in MODELS.items():
        path = RES / folder / source / f"{model}.npz"
        if not path.exists():
            continue
        z = np.load(path)
        assert np.allclose(z["truth"], future), f"{path}: windows differ from load_windows"
        direct, comp = z["direct"][:, IDX, :], z["comp"]
        d = band_w1(direct, comp)
        assert np.allclose(d, z["D"], rtol=1e-6, atol=1e-9), f"{path}: stored D not reproduced"
        e_dir, e_comp = band_w1(direct, ex), band_w1(comp, ex)
        rho = [float(spearmanr(d[:, j], e_dir[:, j])[0]) if d[:, j].std() > 0 else float("nan")
               for j in range(len(HORIZONS))]
        out[model] = {
            "D": bootstrap_mean(d / yard),
            "E_dir": bootstrap_mean(e_dir / yard),
            "E_comp": bootstrap_mean(e_comp / yard),
            "share": (d.mean(0) / (e_dir + e_comp).mean(0)).tolist(),
            "comp_closer": (e_comp < e_dir).mean(0).tolist(),
            "rho_D_Edir": rho,
            "crps": (quantile_crps(direct, future[:, IDX]).mean(0) / yard.mean(0)).tolist(),
        }
    return out


def ranking(rows: dict, j: int) -> dict:
    names = [m for m in rows if m not in ("patchwork", "oracle")]
    d = [rows[m]["D"]["mean"][j] for m in names]
    e = [rows[m]["E_dir"]["mean"][j] for m in names]
    c = [rows[m]["crps"][j] for m in names]
    return {"models": names, "rho_D_Edir": float(spearmanr(d, e)[0]), "rho_D_crps": float(spearmanr(d, c)[0])}


def real_ranking(source: str) -> dict:
    """Real series have no exact law, so models are ranked by mean defect and by mean CRPS (data units)."""
    names, d_rel, d_abs, crps = [], [], [], []
    for model, folder in MODELS.items():
        path = RES / folder / source / f"{model}.npz"
        if model in ("patchwork", "oracle") or not path.exists():
            continue
        z = np.load(path)
        names.append(model)
        d_rel.append((z["D"] / z["spread"]).mean(0))
        d_abs.append(z["D"].mean(0))
        crps.append(z["crps"].mean(0))
    d_rel, d_abs, crps = map(np.array, (d_rel, d_abs, crps))
    return {"models": names, **{f"h{h}": {"rho_Drel_crps": float(spearmanr(d_rel[:, j], crps[:, j])[0]),
                                          "rho_Dabs_crps": float(spearmanr(d_abs[:, j], crps[:, j])[0])}
                                for j, h in enumerate(HORIZONS)}}


def whatif(phi: float = 0.9, sigma: float = 1.0) -> dict:
    """AR(1): the models' conditional forecasts from c + y_m against the exact conditional law."""
    contexts, _ = load_windows("ar1", N, SEED)
    exact = ExactForecaster("ar1")(contexts, max(HORIZONS), None)
    out = {}
    for model in WHATIF_MODELS:
        z = np.load(RES / "mechanism" / "ar1" / f"{model}.npz")
        y, inner, direct = z["y"], z["inner_fc"].astype(np.float64), z["direct"]
        rows = {"E_lead": [], "E_cond_all": [], "E_cond_centre": [], "E_cond_tails": []}
        for j, h in enumerate(HORIZONS):
            assert np.allclose(mixture_quantiles(inner[:, :, j]), z["comp"][:, j], atol=1e-4)
            lead = h - 1
            sd = sigma * np.sqrt((1 - phi ** (2 * lead)) / (1 - phi**2))
            cond = gaussian_quantiles(phi**lead * y, np.full_like(y, sd))
            e_cond = band_w1(inner[:, :, j], cond) / (cond[..., -1] - cond[..., 0])
            ex_lead = exact[:, lead - 1]
            e_lead = band_w1(direct[:, lead - 1], ex_lead) / (ex_lead[:, -1] - ex_lead[:, 0])
            rows["E_lead"].append(float(e_lead.mean()))
            rows["E_cond_all"].append(float(e_cond.mean()))
            rows["E_cond_centre"].append(float(e_cond[:, CENTRE].mean()))
            rows["E_cond_tails"].append(float(e_cond[:, TAILS].mean()))
        out[model] = rows
    return out


def fmt(x: float) -> str:
    return f"{x:.4f}"


def tables(summary: dict) -> str:
    hs = HORIZONS
    lines = ["# Self-consistency gap against error from the exact law (synthetic sources)\n",
             "All values are divided by the exact law's 80% width. Means over 256 contexts.\n"]
    for source, rows in summary["by_model"].items():
        lines += [f"\n## {source}\n",
                  "| Model | D, h = " + " / ".join(map(str, hs)) + " | E_dir | E_comp | D / (E_dir + E_comp) |"
                  " composed closer to exact | rho(D, E_dir) over contexts |",
                  "|---|---|---|---|---|---|---|"]
        for m, r in rows.items():
            cells = [" / ".join(fmt(v) for v in r[k]["mean"]) for k in ("D", "E_dir", "E_comp")]
            lines.append(f"| {m} | " + " | ".join(cells)
                         + " | " + " / ".join(f"{v:.2f}" for v in r["share"])
                         + " | " + " / ".join(f"{v:.2f}" for v in r["comp_closer"])
                         + " | " + " / ".join(f"{v:+.2f}" for v in r["rho_D_Edir"]) + " |")
        for h in (8, 16):
            rk = summary["ranking"][source][f"h{h}"]
            lines.append(f"\nAcross the {len(rk['models'])} direct models at h = {h}: Spearman(mean D, mean E_dir)"
                         f" = {rk['rho_D_Edir']:+.2f}; Spearman(mean D, mean CRPS) = {rk['rho_D_crps']:+.2f}.")
    lines += ["\n## Real series: ranking of the direct models by defect and by CRPS\n",
              "Spearman over models between mean defect and mean CRPS (no exact law exists here).\n",
              "| Source | h = 8: D/width vs CRPS | h = 8: D vs CRPS (data units) | h = 16: D/width vs CRPS |"
              " h = 16: D vs CRPS (data units) |", "|---|---|---|---|---|"]
    for source, r in summary["real_ranking"].items():
        lines.append(f"| {source} ({len(r['models'])} models) | {r['h8']['rho_Drel_crps']:+.2f} | "
                     f"{r['h8']['rho_Dabs_crps']:+.2f} | {r['h16']['rho_Drel_crps']:+.2f} | "
                     f"{r['h16']['rho_Dabs_crps']:+.2f} |")
    lines += ["\n## What-if forecasts on AR(1)\n",
              "Error of the forecast of y_{t+h} from c + y_m against the exact conditional law, divided by that"
              " law's 80% width, next to the error of the direct forecast at the same lead h - 1 from c.\n",
              "| Model | lead = " + " / ".join(str(h - 1) for h in hs) + ": direct from c | from c + y, all draws |"
              " central draws (u 0.27-0.73) | tail draws (u < 0.05 or > 0.95) |",
              "|---|---|---|---|---|"]
    for m, r in summary["whatif_ar1"].items():
        lines.append(f"| {m} | " + " | ".join(" / ".join(fmt(v) for v in r[k])
                                               for k in ("E_lead", "E_cond_all", "E_cond_centre", "E_cond_tails"))
                     + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    summary = {"by_model": {}, "ranking": {}, "horizons": list(HORIZONS)}
    for source in SYNTHETIC:
        rows = by_model(source)
        summary["by_model"][source] = rows
        summary["ranking"][source] = {f"h{h}": ranking(rows, HORIZONS.index(h)) for h in (8, 16)}
    summary["real_ranking"] = {s: real_ranking(s) for s in ("epf", "covid", "weather")}
    summary["whatif_ar1"] = whatif()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "summary.json", summary)
    (OUT / "tables.md").write_text(tables(summary))
    print(tables(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
