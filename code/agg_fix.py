"""Aggregate exp_fix.py -> results/rev3_fix_summary.json and the revised Figure 4."""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = ROOT + "/results"; FIG = ROOT + "/figs"
DSL = ["NSL-KDD", "UNSW-NB15", "TON_IoT"]
CS = ["0.0", "0.005", "0.01", "0.02", "0.05"]
STRAT = ["naive", "labeltrim", "margin"]


def ms(v):
    v = np.asarray(v, float)
    return [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]


S = {"contam": {}, "drift": {}, "host": {}}
for ds in DSL:
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"{RES}/rev3_fix/{ds}_*.json"))]
    S["contam"][ds] = {}
    for en in runs[0]["contam"]:
        S["contam"][ds][en] = {}
        for c in CS:
            rec = {}
            for s in STRAT:
                f = [x for r in runs for x in r["contam"][en][c][s]["FPR"]]; d = [x for r in runs for x in r["contam"][en][c][s]["DR"]]
                rec[s] = dict(FPR=ms(f), DR=ms(d), FPR_p975=float(np.percentile(f, 97.5)))
                if s == "labeltrim":
                    ins = [x for r in runs for x in r["contam"][en][c][s]["inspected"]]
                    rec[s]["inspected"] = ms(ins); rec[s]["inspected_max"] = int(max(ins))
            S["contam"][ds][en][c] = rec
    S["drift"][ds] = {en: {k: ms([r["drift"][en][k] for r in runs if r["drift"][en][k] is not None])
                           for k in runs[0]["drift"][en] if runs[0]["drift"][en][k] is not None} for en in runs[0]["drift"]}
    S.setdefault("n_seeds", {})[ds] = len(runs)

H = [json.load(open(f)) for f in sorted(glob.glob(f"{RES}/rev3_fix/host_*.json"))]
hs = {"n_seeds": len(H), "test_hosts": [len(r["test_hosts"]) for r in H], "n_test_benign": [r["n_test_benign"] for r in H],
      "cal_hosts": [len(r["hostcal"]["cal_hosts"]) for r in H], "lam": [r["standard"]["lam"] for r in H]}
for v in ["standard", "hostcal"]:
    hs[v] = {en: {a: {m: dict(FPR=ms([r[v][en][a][m][0] for r in H]),
                                DR=ms([r[v][en][a][m][1] for r in H]) if H[0][v][en][a][m][1] is not None else None,
                                FPR_seeds=[r[v][en][a][m][0] for r in H])
                      for m in ["CH-IDS", "LightGBM"]} for a in H[0][v][en]} for en in ["hold-out", "unseen hosts"]}
for v in ["sensor", "adapt"]:
    hs[v] = {n: {m: dict(FPR=ms([x for r in H for x in r[v][n][m]["FPR"]]), DR=ms([x for r in H for x in r[v][n][m]["DR"]]),
                         FPR_p975=float(np.percentile([x for r in H for x in r[v][n][m]["FPR"]], 97.5)))
                 for m in ["CH-IDS", "LightGBM"]} for n in H[0][v]}
H2 = [json.load(open(f)) for f in sorted(glob.glob(f"{RES}/rev3_fix/host2_*.json"))]
hs["refresh"] = {n: dict(FPR=ms([x for r in H2 for x in r["refresh"][n]["FPR"]]), DR=ms([x for r in H2 for x in r["refresh"][n]["DR"]]),
                         DR_min=float(min(x for r in H2 for x in r["refresh"][n]["DR"])),
                         FPR_p975=float(np.percentile([x for r in H2 for x in r["refresh"][n]["FPR"]], 97.5)),
                         DR_seed=[float(np.mean(r["refresh"][n]["DR"])) for r in H2]) for n in H2[0]["refresh"]}
hs["adapt2"] = {n: {m: dict(FPR=ms([x for r in H2 for x in r["adapt2"][n][m]["FPR"]]), DR=ms([x for r in H2 for x in r["adapt2"][n][m]["DR"]]),
                            FPR_range=[float(min(x for r in H2 for x in r["adapt2"][n][m]["FPR"])), float(max(x for r in H2 for x in r["adapt2"][n][m]["FPR"]))],
                            DR_range=[float(min(x for r in H2 for x in r["adapt2"][n][m]["DR"])), float(max(x for r in H2 for x in r["adapt2"][n][m]["DR"]))])
                    for m in ["CH-IDS", "LightGBM"]} for n in H2[0]["adapt2"]}
hs["sensor_seed_DR"] = {n: {m: [float(np.mean(r["sensor"][n][m]["DR"])) for r in H] for m in ["CH-IDS", "LightGBM"]} for n in H[0]["sensor"]}
hs["drift_unseen_hosts"] = {k: ms([r["drift_unseen_hosts"][k] for r in H]) for k in H[0]["drift_unseen_hosts"]}
hs["drift_holdout"] = {k: ms([r["drift_holdout"][k] for r in H]) for k in H[0]["drift_holdout"]}
S["host"] = hs
json.dump(S, open(f"{RES}/rev3_fix_summary.json", "w"), indent=1)

# ------------------------------------------------------------------ Figure 4 (revised)
R3 = json.load(open(f"{RES}/rev3_summary.json"))
INK2, GRID = "#52514e", "#e4e3df"
COL = {"naive": "#eb6834", "labeltrim": "#2a78d6", "margin": "#1baf7a"}
LAB = {"naive": "Naive re-calibration", "labeltrim": "Analyst-inspected (label-trim)", "margin": "Rank margin, no labels (c̄ = 1%)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False})
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.95))
x = [100 * float(c) for c in CS]
for s in STRAT:
    for ds, en, ls in [("UNSW-NB15", "UNSW-NB15 test", "-"), ("NSL-KDD", "KDDTest+", "--")]:
        d = S["contam"][ds][en]
        axs[0].plot(x, [100 * d[c][s]["FPR"][0] for c in CS], ls=ls, marker="o", ms=2.8, lw=1.2, color=COL[s])
        axs[1].plot(x, [100 * d[c][s]["DR"][0] for c in CS], ls=ls, marker="o", ms=2.8, lw=1.2, color=COL[s],
                    label=f"{LAB[s]}" if ds == "UNSW-NB15" else None)
axs[0].axhline(1, ls=":", color=INK2, lw=0.8)
axs[0].set_xlabel("Attack share in window (%)"); axs[0].set_ylabel("Realised FPR (%)"); axs[0].set_title("(a) FPR after re-calibration", fontsize=8)
axs[1].set_xlabel("Attack share in window (%)"); axs[1].set_ylabel("Detection rate (%)"); axs[1].set_title("(b) DR after re-calibration", fontsize=8)
cmap = {"NSL-KDD": "#1baf7a", "UNSW-NB15": "#eb6834", "TON_IoT": "#6250d6"}
for i, ds in enumerate(DSL):
    cs = R3["calsize"][ds]["data"]; ns = sorted(int(n) for n in cs)
    y = [100 * cs[str(n)]["0.01"]["CH-IDS"]["hold_fpr"][0] for n in ns]
    lo = [100 * cs[str(n)]["0.01"]["CH-IDS"]["hold_fpr_p05"] for n in ns]; hi = [100 * cs[str(n)]["0.01"]["CH-IDS"]["hold_fpr_p95"] for n in ns]
    axs[2].errorbar(np.array(ns) * (1 + 0.05 * (i - 1)), y, yerr=[np.array(y) - lo, np.array(hi) - y], marker="o", ms=3, lw=1.2,
                    capsize=2, color=cmap[ds], label=ds)
axs[2].axhline(1, ls=":", color=INK2, lw=0.8); axs[2].set_xscale("log"); axs[2].set_ylim(top=7.2)
axs[2].set_xlabel("Benign calibration flows n"); axs[2].set_ylabel("Hold-out FPR at α = 1% (%)"); axs[2].set_title("(c) Calibration size", fontsize=8)
for ax in axs:
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
h, l = axs[1].get_legend_handles_labels()
from matplotlib.lines import Line2D
h += [Line2D([], [], color=INK2, ls="-", lw=1.2), Line2D([], [], color=INK2, ls="--", lw=1.2)]
l += ["UNSW-NB15 test", "KDDTest+"]
fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=1.0)
axs[2].legend(fontsize=6.5, loc="upper center", ncol=3, handletextpad=0.3, columnspacing=0.8)
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(f"{FIG}/fig_rev3_contam.png", dpi=600)
print("ok")
