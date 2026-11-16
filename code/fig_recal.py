"""Re-calibration figure (cleaner version) from results/calib.json."""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = json.load(open(ROOT + "/results/calib.json"))["recal"]
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
INK2, GRID = "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "savefig.dpi": 600})
SER = [("NSL-KDD", "KDDTest+", "KDDTest+ (official test)"), ("NSL-KDD", "KDDTest-21", "KDDTest-21 (official test)"),
       ("UNSW-NB15", "UNSW-NB15 test", "UNSW-NB15 (official test)"), ("TON_IoT", "heavy flows (bytes > P90)", "TON_IoT heavy flows")]
MK = "osD^"
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
pos = {"none": 60, "100": 100, "250": 250, "500": 500, "1000": 1000, "2000": 2000}
for i, (ds, t, lab) in enumerate(SER):
    d = C[ds][t]
    ws = [w for w in ["100", "250", "500", "1000", "2000"] if w in d]
    x = np.array([pos[w] for w in ws]) * (1 + 0.04 * (i - 1.5))
    y = np.array([d[w]["FPR"] for w in ws]); lo = np.array([max(d[w]["FPR_lo"], 1.2e-3) for w in ws]); hi = np.array([d[w]["FPR_hi"] for w in ws])
    axs[0].errorbar(x, y, yerr=[np.maximum(y - lo, 0), hi - y], marker=MK[i], ms=3.5, lw=1.2, capsize=1.5, color=COL[i], label=lab, mec="white", mew=0.4)
    axs[0].scatter([pos["none"] * (1 + 0.04 * (i - 1.5))], [d["none"]["FPR"]], marker=MK[i], color=COL[i], s=18, edgecolor="black", linewidth=0.4, zorder=5)
    if d[ws[0]]["DR"] is not None:
        axs[1].plot(x, [d[w]["DR"] for w in ws], marker=MK[i], ms=3.5, lw=1.2, color=COL[i], mec="white", mew=0.4)
        axs[1].scatter([pos["none"]], [d["none"]["DR"]], marker=MK[i], color=COL[i], s=18, edgecolor="black", linewidth=0.4, zorder=5)
axs[0].axhline(0.01, color=INK2, ls="--", lw=0.8)
axs[0].text(2300, 0.0105, "α", color=INK2, fontsize=7, va="bottom")
for ax in axs:
    ax.set_xscale("log"); ax.set_xticks([60, 100, 250, 500, 1000, 2000]); ax.set_xticklabels(["none", "100", "250", "500", "1000", "2000"])
    ax.minorticks_off(); ax.set_xlim(48, 2600)
    ax.set_xlabel("Benign re-calibration window (flows)")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
axs[0].set_yscale("log"); axs[0].set_ylim(1e-3, 0.5)
axs[0].set_ylabel("Realised FPR at α = 0.01"); axs[0].set_title("(a) False-positive rate", fontsize=7.5)
axs[1].set_ylabel("Detection rate on test attacks"); axs[1].set_ylim(0, 1); axs[1].set_title("(b) Detection rate", fontsize=7.5)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5)
fig.savefig(ROOT + "/figs/fig_recal.png")
print("ok")
