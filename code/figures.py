"""Manuscript figures from results/summary.json and results/runtime.json."""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = ROOT + "/results"; FIG = ROOT + "/figs"
S = json.load(open(f"{RES}/" + os.environ.get("SUMMARY", "summary_v2.json")))
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "legend.frameon": False,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 600})


def style(ax, grid_axis="y"):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.5); ax.set_axisbelow(True)


# ---------------------------------------------------------------- Fig. 2: false-alarm calibration
def fig_alpha():
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    tests = [("NSL-KDD", "alpha_sweep_KDDTest+", "KDDTest+"), ("NSL-KDD", "alpha_sweep_KDDTest-21", "KDDTest-21"),
             ("UNSW-NB15", "alpha_sweep_UNSW-NB15 test", "UNSW-NB15"), ("TON_IoT", "alpha_sweep_TON_IoT test", "TON_IoT")]
    mk = ["o", "s", "^", "D"]
    al = None
    for i, (ds, key, lab) in enumerate(tests):
        sw = S["extra"].get(ds, {}).get("sweep_all", {}).get(key)
        if not sw:
            continue
        al = np.array([float(a) for a in sw])
        f = np.array([sw[a]["CH"]["FPR"][0] for a in sw]); e = np.array([sw[a]["CH"]["FPR"][1] for a in sw])
        ax.errorbar(al, f, yerr=e, color=C[i], marker=mk[i], ms=4, lw=1.4, capsize=1.5, label=lab, mec="white", mew=0.5)
    lo, hi = 0.0007, 0.6
    ax.plot([lo, 0.15], [lo, 0.15], color=INK2, lw=0.8, ls="--")
    ax.text(0.0012, 0.00085, "ideal: FPR = α", color=INK2, fontsize=6, rotation=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.0007, 0.15); ax.set_ylim(lo, hi)
    ax.set_xlabel("False-alarm budget α (log scale)"); ax.set_ylabel("Empirical test FPR (log scale)")
    style(ax, "both"); ax.legend(loc="upper left", fontsize=6.3, ncol=2, handlelength=1.8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_alpha.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 3: zero-day per family
LAB = {"CH-IDS (proposed)": "CH-IDS", "GBDT [conformal]": "LightGBM (conformal)", "RF [conformal]": "RF (conformal)",
       "AE [conformal]": "AE (conformal)", "Seq. hybrid: rules->RF+SMOTE [default]": "Seq. hybrid (default)"}
LAB_CD = dict(LAB, **{"IForest [conformal]": "IForest (conformal)", "RF [default]": "RF (default)", "CH-IDS lambda=0.0": "CH-IDS, λ=0", "CH-IDS lambda=0.25": "CH-IDS, λ=0.25", "CH-IDS lambda=0.5": "CH-IDS, λ=0.5"})


def fig_loafo():
    L = S["loafo"]
    order = [d for d in ["NSL-KDD", "UNSW-NB15", "TON_IoT"] if d in L]
    rows = []
    for d in order:
        for f in sorted(L[d]["families"]):
            rows.append((d, f))
    methods = list(LAB)
    mk = ["o", "s", "^", "D", "v"]
    h = 0.17 * len(rows) + 0.9
    fig, ax = plt.subplots(figsize=(3.45, h))
    y = np.arange(len(rows))[::-1]
    for j, m in enumerate(methods):
        xs = [L[d]["families"][f].get(m, {}).get("DR", [np.nan])[0] for d, f in rows]
        ax.scatter(xs, y + (j - 2) * 0.12, s=13 if j else 20, marker=mk[j], color=C[j], label=LAB[m],
                   edgecolor="white", linewidth=0.4, zorder=3 if j else 4)
    ax.set_yticks(y); ax.set_yticklabels([f for d, f in rows], fontsize=6.3)
    # dataset separators
    pos = 0
    for d in order:
        n = len(L[d]["families"])
        top = len(rows) - pos - 0.5
        ax.axhline(top, color=INK2, lw=0.5)
        ax.text(1.04, top - n / 2, d, transform=ax.get_yaxis_transform(), rotation=270, va="center", fontsize=6.5, color=INK)
        pos += n
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Detection rate on the unseen (held-out) family")
    style(ax, "x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.08 * 7 / h), ncol=3, fontsize=6, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_loafo.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 4: lambda sensitivity
def fig_lambda():
    L = S["loafo"]
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.05), sharex=True)
    lams = [0.0, 0.25, 0.5]
    mk = ["o", "s", "^"]
    for i, d in enumerate([d for d in ["NSL-KDD", "UNSW-NB15", "TON_IoT"] if d in L]):
        sm = L[d]["summary"]
        dr = [sm[f"CH-IDS lambda={l}"]["meanDR"][0] for l in lams]
        fp = [sm[f"CH-IDS lambda={l}"]["FPR"][0] for l in lams]
        axs[0].plot(lams, dr, color=C[i], marker=mk[i], ms=3.5, lw=1.3, label=d)
        axs[1].plot(lams, fp, color=C[i], marker=mk[i], ms=3.5, lw=1.3)
        axs[0].scatter([0.6], [sm["CH-IDS (proposed)"]["meanDR"][0]], marker="*", s=40, color=C[i], edgecolor=INK, linewidth=0.3, zorder=5)
        axs[1].scatter([0.6], [sm["CH-IDS (proposed)"]["FPR"][0]], marker="*", s=40, color=C[i], edgecolor=INK, linewidth=0.3, zorder=5)
    for ax, t in zip(axs, ["Mean DR, unseen family", "Mean FPR, benign test"]):
        ax.set_xticks([0, 0.25, 0.5, 0.6]); ax.set_xticklabels(["0", ".25", ".5", "λ*"])
        ax.set_title(t, fontsize=6.8, color=INK); style(ax)
        ax.set_xlabel("Anomaly budget share λ")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=6, bbox_to_anchor=(0.5, 1.02), handlelength=1.6)
    fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.9)); fig.savefig(f"{FIG}/fig4_lambda.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 5: critical-difference diagram
def fig_cd():
    F = S.get("friedman_DR")
    if not F:
        return
    names = [LAB_CD.get(m, m).replace(" (proposed)", "") for m in F["methods"]]
    r = np.array(F["avg_rank"]); cd = F["CD"]; k = len(r)
    fig, ax = plt.subplots(figsize=(3.45, 1.45))
    ax.set_xlim(-1.6, k + 2.6); ax.set_ylim(0.2, 1.08); ax.axis("off")
    ax.plot([1, k], [0.82, 0.82], color=INK, lw=0.7)
    for t in range(1, k + 1):
        ax.plot([t, t], [0.82, 0.86], color=INK, lw=0.7); ax.text(t, 0.9, str(t), ha="center", fontsize=6.5)
    ax.plot([1, 1 + cd], [0.98, 0.98], color=INK, lw=1.2)
    ax.text(1 + cd / 2, 1.0, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=6)
    order = np.argsort(r)
    half = (k + 1) // 2
    for idx, j in enumerate(order):
        x = r[j]
        if idx < half:
            yy = 0.62 - idx * 0.12; tx = 0.8; ha = "right"
        else:
            yy = 0.62 - (k - 1 - idx) * 0.12; tx = k + 0.2; ha = "left"
        ax.plot([x, x], [0.82, yy], color=INK2, lw=0.6); ax.plot([x, tx], [yy, yy], color=INK2, lw=0.6)
        ax.text(tx + (0.05 if ha == "left" else -0.05), yy, f"{names[j]} ({x:.2f})", ha=ha, va="center", fontsize=6,
                color=C[0] if names[j] == "CH-IDS" else INK)
    # cliques: groups whose rank span < CD
    sr = r[order]; lines = []
    for a in range(k):
        b = a
        while b + 1 < k and sr[b + 1] - sr[a] < cd:
            b += 1
        if b > a and not any(a >= x and b <= y for x, y in lines):
            lines.append((a, b))
    for n, (a, b) in enumerate(lines):
        yy = 0.78 - 0.04 * n
        ax.plot([sr[a] - 0.03, sr[b] + 0.03], [yy, yy], color=INK, lw=1.6)
    fig.savefig(f"{FIG}/fig5_cd.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 6: runtime
def fig_runtime():
    fn = f"{RES}/runtime_v2.json"
    if not os.path.exists(fn):
        return
    R = json.load(open(fn))
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.9))
    ds = [d for d in ["NSL-KDD", "UNSW-NB15", "TON_IoT"] if d in R]
    mk = ["o", "s", "^"]
    for i, d in enumerate(ds):
        b = R[d]["batch_scaling"]
        xs = [int(k) for k in b]; ys = [b[k] for k in b]
        axs[0].plot(xs, ys, color=C[i], marker=mk[i], ms=3, lw=1.2, label=d)
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("Batch size (flows)"); axs[0].set_ylabel("Throughput (flows/s)")
    axs[0].legend(fontsize=5.6); style(axs[0], "both")
    for i, d in enumerate(ds):
        t = R[d]["train_scaling"]
        xs = [int(k) for k in t]; ys = [t[k] for k in t]
        axs[1].plot(xs, ys, color=C[i], marker=mk[i], ms=3, lw=1.2)
    axs[1].set_xlabel("Training flows"); axs[1].set_ylabel("Training time (s)"); style(axs[1], "both")
    axs[1].ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    fig.tight_layout(w_pad=0.8); fig.savefig(f"{FIG}/fig6_runtime.png"); plt.close(fig)


if __name__ == "__main__":
    fig_loafo(); fig_lambda(); fig_cd(); fig_runtime()
    print("figures done")
