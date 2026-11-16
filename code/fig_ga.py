import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
S = json.load(open(ROOT + "/results/summary.json"))
C = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig = plt.figure(figsize=(9.6, 4.3), dpi=200)
ax0 = fig.add_axes([0.0, 0.02, 0.36, 0.96]); ax0.imshow(mpimg.imread(ROOT + "/figs/fig1_arch.png")); ax0.axis("off")
ax = fig.add_axes([0.43, 0.34, 0.25, 0.5])
tests = ["KDDTest+", "KDDTest-21", "UNSW-NB15 test", "TON_IoT test"]; lab = ["KDD+", "KDD-21", "UNSW", "TON"]
ms = [("Seq. hybrid: rules->RF+SMOTE [default]", "Thesis design (rules→RF, 0.5 threshold)"), ("LightGBM [conformal]", "GBDT, conformal"), ("CH-IDS (proposed)", "CH-IDS (proposed)")]
x = np.arange(4); w = 0.26
for i, (m, l) in enumerate(ms):
    vals = [100 * S["standard"][t][m]["MCC"][0] for t in tests]
    ax.bar(x + (i - 1) * w, vals, w * 0.92, color=[C[1], "#9a9a9a", C[0]][i], label=l)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("MCC (%)"); ax.set_ylim(0, 105)
ax.set_title("Official test sets (5 seeds)", fontsize=9)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, frameon=False)
ax2 = fig.add_axes([0.76, 0.34, 0.22, 0.5])
a = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
sw = S["extra"]["TON_IoT"]["sweep_all"]["alpha_sweep_TON_IoT test"]
f = [100 * sw[str(v)]["CH"]["FPR"][0] for v in a]
ax2.plot([0.1, 10], [0.1, 10], "--", color="#52514e", lw=1); ax2.plot([100 * v for v in a], f, "o-", color=C[0], lw=2)
ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlabel("False-alarm budget α (%)"); ax2.set_ylabel("Observed FPR (%)")
ax2.set_title("FPR tracks the budget (TON_IoT)", fontsize=9)
for s in ["top", "right"]: ax2.spines[s].set_visible(False)
fig.savefig(ROOT + "/figs/graphical_abstract.png", dpi=200)
