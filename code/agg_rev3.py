"""Aggregate revision-3 outputs -> results/rev3_summary.json and two figures."""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, glob
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = ROOT + "/results"
FIG = ROOT + "/figs"
DSL = ["NSL-KDD", "UNSW-NB15", "TON_IoT"]
ALPHAS = ["0.001", "0.005", "0.01", "0.02", "0.05", "0.1"]


def ms(v):
    v = np.asarray(v, float)
    return [float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0]


def band(m, n2, a):
    k = int(np.floor(float(a) * (n2 + 1) + 1e-12))
    if k < 1:
        return 0.0, 0.0
    d = stats.betabinom(m, k, n2 + 1 - k)
    return d.ppf(0.025) / m, d.ppf(0.975) / m


S = {"repro": {}, "matched": {}, "oracle": {}, "family": {}, "calsize": {}, "drift": {}, "contam": {}, "rules": {}}
for ds in DSL:
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"{RES}/rev3/{ds}_*.json"))]
    S["repro"][ds] = all(abs(v["now"] - v["stored"]) < 1e-9 for r in runs for v in r["repro"].values())
    S["n_seeds"] = len(runs)
    S["matched"][ds] = {en: {m: {k: ms([r["matched"][en][m][k] for r in runs]) for k in ["FPR", "DR", "MCC", "F1"]}
                             for m in runs[0]["matched"][en]} for en in runs[0]["matched"]}
    S["oracle"][ds] = {en: {m: {k: ms([r["oracle"][en][m][k] for r in runs]) for k in ["AUC", "DR_at_1", "DR_at_5"]}
                            for m in runs[0]["oracle"][en]} for en in runs[0]["oracle"]}
    fam = {}
    for en in runs[0]["family"]:
        fam[en] = {}
        for f in runs[0]["family"][en]:
            if f == "_benign":
                fam[en][f] = {k: ms([r["family"][en][f][k] for r in runs]) for k in ["FPR_ch", "FPR_lgb"]}
                fam[en][f].update({k: int(sum(r["family"][en][f][k] for r in runs)) for k in ["only_ch", "only_lgb", "n"]})
            else:
                fam[en][f] = {k: ms([r["family"][en][f][k] for r in runs]) for k in ["DR_ch", "DR_lgb"]}
                fam[en][f].update({k: int(sum(r["family"][en][f][k] for r in runs)) for k in ["only_ch", "only_lgb", "only_ch_rule", "n"]})
    S["family"][ds] = fam
    cs = {}
    for n in runs[0]["calsize"]:
        cs[n] = {}
        for a in runs[0]["calsize"][n]:
            cs[n][a] = {}
            for m in runs[0]["calsize"][n][a]:
                arr = np.concatenate([np.array(r["calsize"][n][a][m]) for r in runs])   # rows: repetitions x seeds
                cs[n][a][m] = dict(hold_fpr=ms(arr[:, 0]), hold_fpr_p95=float(np.percentile(arr[:, 0], 97.5)),
                                   hold_fpr_p05=float(np.percentile(arr[:, 0], 2.5)),
                                   test_fpr=ms(arr[:, 1]), test_dr=ms(arr[:, 2]))
    S["calsize"][ds] = dict(test=runs[0]["calsize_test"], data=cs)
    S["drift"][ds] = {en: {k: ms([r["drift"][en][k] for r in runs if k in r["drift"][en]]) for k in runs[0]["drift"][en]}
                      for en in runs[0]["drift"]}
    S["contam"][ds] = {en: {k: dict(FPR=ms([x for r in runs for x in r["contam"][en][k]["FPR"]]),
                                    DR=ms([x for r in runs for x in r["contam"][en][k]["DR"]]))
                            for k in runs[0]["contam"][en]} for en in runs[0]["contam"]}
    S["rules"][ds] = next(r["rules"] for r in runs if r["seed"] == 0)
    S.setdefault("lam", {})[ds] = [r["lam"] for r in runs]

# duplicates, deduplicated and host-aware splits
S["dup"] = {ds: json.load(open(f"{RES}/rev3_split/dup_{ds}.json")) for ds in DSL}
for tag, pattern in [("dedup", "dedup_*_*.json"), ("host", "host_TON_IoT_*.json")]:
    out = {}
    for fn in sorted(glob.glob(f"{RES}/rev3_split/{pattern}")):
        R = json.load(open(fn))
        ds = fn.split("/")[-1].split("_")[1] if tag == "dedup" else "TON_IoT"
        if tag == "dedup":
            ds = "_".join(fn.split("/")[-1][len("dedup_"):].split("_")[:-1])
        out.setdefault(ds, []).append(R)
    agg = {}
    for ds, runs in out.items():
        a = {"info": runs[0].get("info"), "n2": [r["n2"] for r in runs], "sets": {}}
        for en in runs[0]["sets"]:
            e = {"m_benign": runs[0]["sets"][en]["m_benign"], "n_attack": runs[0]["sets"][en]["n_attack"]}
            for al in ALPHAS:
                rec = {}
                for m in ["CH-IDS", "LightGBM (Dcal2)"]:
                    fpr = [r["sets"][en][al][m][0] / r["sets"][en]["m_benign"] for r in runs]
                    pos = []
                    for r in runs:
                        lo, hi = band(r["sets"][en]["m_benign"], r["n2"], al)
                        x = r["sets"][en][al][m][0] / r["sets"][en]["m_benign"]
                        pos.append("in" if lo <= x <= hi else ("below" if x < lo else "above"))
                    rec[m] = dict(FPR=ms(fpr), inside=pos.count("in"), below=pos.count("below"), above=pos.count("above"),
                                  DR=ms([r["sets"][en][al][m][1] for r in runs]) if runs[0]["sets"][en][al][m][1] is not None else None)
                e[al] = rec
            if "metrics" in runs[0]["sets"][en]:
                e["metrics"] = {m: {k: ms([r["sets"][en]["metrics"][m][k] for r in runs]) for k in ["F1", "MCC", "FPR", "DR"]}
                                for m in ["CH-IDS", "LightGBM (Dcal2)"]}
            a["sets"][en] = e
        agg[ds] = a
    S[tag] = agg
json.dump(S, open(f"{RES}/rev3_summary.json", "w"), indent=1)

# ---------------- figure: contaminated re-calibration and calibration-set size
INK2, GRID = "#52514e", "#e4e3df"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False})
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9))
ser = [("UNSW-NB15", "hold-out", "UNSW-NB15 hold-out"), ("UNSW-NB15", "UNSW-NB15 test", "UNSW-NB15 test"),
       ("NSL-KDD", "KDDTest+", "KDDTest+"), ("TON_IoT", "TON_IoT test", "TON_IoT test")]
cvals = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
for i, (ds, en, lab) in enumerate(ser):
    d = S["contam"][ds][en]
    x = [100 * c for c in cvals]
    axs[0].plot(x, [100 * d[f"n1000_c{c}"]["FPR"][0] for c in cvals], marker="o", ms=3, lw=1.2, color=C[i], label=lab)
    axs[1].plot(x, [100 * d[f"n1000_c{c}"]["DR"][0] for c in cvals], marker="o", ms=3, lw=1.2, color=C[i], label=lab)
axs[0].axhline(1, ls="--", color=INK2, lw=0.8)
axs[0].set_xlabel("Attack share in window (%)"); axs[0].set_ylabel("Realised FPR (%)"); axs[0].set_title("(a) Contaminated window: FPR", fontsize=8)
axs[1].set_xlabel("Attack share in window (%)"); axs[1].set_ylabel("Detection rate (%)"); axs[1].set_title("(b) Contaminated window: DR", fontsize=8)
for i, ds in enumerate(DSL):
    cs = S["calsize"][ds]["data"]
    ns = sorted(int(n) for n in cs)
    y = [100 * cs[str(n)]["0.01"]["CH-IDS"]["hold_fpr"][0] for n in ns]
    lo = [100 * cs[str(n)]["0.01"]["CH-IDS"]["hold_fpr_p05"] for n in ns]; hi = [100 * cs[str(n)]["0.01"]["CH-IDS"]["hold_fpr_p95"] for n in ns]
    axs[2].errorbar(np.array(ns) * (1 + 0.05 * (i - 1)), y, yerr=[np.array(y) - lo, np.array(hi) - y], marker="o", ms=3, lw=1.2, capsize=2, color={"NSL-KDD": C[2], "UNSW-NB15": C[1], "TON_IoT": C[3]}[ds], label=ds)
axs[2].axhline(1, ls="--", color=INK2, lw=0.8); axs[2].set_xscale("log")
axs[2].set_xlabel("Benign calibration flows n"); axs[2].set_ylabel("Hold-out FPR at α = 1% (%)"); axs[2].set_title("(c) Calibration size", fontsize=8)
for ax in axs:
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.36, 0.0))
axs[2].set_ylim(top=7.2); axs[2].legend(fontsize=6.5, loc="upper center", ncol=3, handletextpad=0.3, columnspacing=0.8)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(f"{FIG}/fig_rev3_contam.png", dpi=600)
print("ok", S["repro"])
