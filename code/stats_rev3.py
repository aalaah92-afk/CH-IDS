"""Revision 3: independence-aware re-analysis of Protocols A and B (no new training).

Protocol A: paired seed-level differences per test set (descriptive 95% t-intervals over the five seeds; KDDTest-21 is
            a subset of KDDTest+ and is reported separately). No p-values pooled over test sets.
Protocol B: the three seeds of each held-out family are averaged first, so the family (22) is the unit. Wilcoxon
            signed-rank test on the 22 family-level differences (Holm), Hodges-Lehmann estimate with a 95% confidence
            interval, matched-pairs rank-biserial correlation, and a hierarchical bootstrap interval for the mean
            difference (datasets, then families within datasets, then seeds within families). Friedman / Nemenyi on
            the 22 family-level means.
Output: results/stats_rev3.json and figs/fig_cd_rev3.png"""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, glob
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = ROOT + "/results"
OUTF = ROOT + "/figs/fig_cd_rev3.png"
rng = np.random.default_rng(2026)
P = "CH-IDS (proposed)"
TESTS = [("NSL-KDD", "KDDTest+"), ("NSL-KDD", "KDDTest-21"), ("UNSW-NB15", "UNSW-NB15 test"), ("TON_IoT", "TON_IoT test")]
BASE_A = ["LR [default]", "DT [default]", "RF [default]", "SVM-RBF [default]", "MLP [default]", "XGBoost [default]",
          "LightGBM [default]", "IForest [default]", "AE [default]", "Seq. hybrid: rules->RF+SMOTE [default]",
          "RF [conformal]", "XGBoost [conformal]", "LightGBM [conformal]", "AE [conformal]",
          "Seq. hybrid: rules->RF+SMOTE [conformal]"]
BASE_B = ["GBDT [conformal]", "RF [conformal]", "RF [default]", "Seq. hybrid: rules->RF+SMOTE [default]", "AE [conformal]",
          "IForest [conformal]", "CH-IDS lambda=0.0", "CH-IDS lambda=0.25", "CH-IDS lambda=0.5"]
CD_METHODS = [P, "GBDT [conformal]", "RF [conformal]", "Seq. hybrid: rules->RF+SMOTE [default]", "AE [conformal]", "IForest [conformal]"]


def tci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); s = x.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    return float(m), float(m - h), float(m + h)


def hodges_lehmann(d):
    """HL estimate (median of Walsh averages) and the distribution-free 95% CI from the signed-rank distribution."""
    d = np.asarray(d, float); n = len(d)
    w = np.sort([(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)])
    N = len(w)
    # normal approximation to the signed-rank critical value
    c = int(np.floor(n * (n + 1) / 4 - 1.96 * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    c = max(c, 0)
    return float(np.median(w)), float(w[c]), float(w[N - 1 - c])


def rank_biserial(d):
    d = np.asarray(d, float); d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, k in enumerate(o):
        run = max(run, (m - i) * ps[k]); adj[k] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ Protocol A
S = {}
for fn in glob.glob(f"{RES}/standard_v2/*.json"):
    R = json.load(open(fn)); S.setdefault(R["dataset"], {})[R["seed"]] = R
A = {}
for ds, t in TESTS:
    seeds = sorted(S[ds])
    A[t] = {}
    for b in BASE_A:
        d = [S[ds][s]["results"][t][P]["MCC"] - S[ds][s]["results"][t][b]["MCC"] for s in seeds]
        A[t][b] = dict(zip(["mean", "lo", "hi"], tci(d)))
summaryA = {}
for b in BASE_A:
    ind = [A[t][b] for _, t in TESTS if t != "KDDTest-21"]
    summaryA[b] = dict(above=sum(x["lo"] > 0 for x in ind), below=sum(x["hi"] < 0 for x in ind),
                       overlap=sum(x["lo"] <= 0 <= x["hi"] for x in ind))

# ------------------------------------------------------------------ Protocol B
L = {}
for fn in glob.glob(f"{RES}/loafo_v2/*.json"):
    R = json.load(open(fn)); L.setdefault(R["dataset"], {})[R["seed"]] = R
fam_units = []          # (dataset, family, {method: [per-seed values]})
for ds, runs in L.items():
    fams = sorted(next(iter(runs.values()))["families"])
    for f in fams:
        rec = {}
        for s, R in runs.items():
            for m, v in R["families"][f]["results"].items():
                rec.setdefault(m, {"DR": [], "AUC": [], "FPR": []})
                for k in ["DR", "AUC", "FPR"]:
                    rec[m][k].append(v.get(k, np.nan))
        fam_units.append((ds, f, rec))
B = {}
for met in ["DR", "AUC"]:
    rows = {}
    for b in BASE_B:
        d = np.array([np.nanmean(rec[P][met]) - np.nanmean(rec[b][met]) for _, _, rec in fam_units])
        nz = d[d != 0]
        p = stats.wilcoxon(nz).pvalue if len(nz) >= 5 else 1.0
        hl = hodges_lehmann(d)
        # hierarchical bootstrap of the mean family-level difference
        dsets = sorted(set(x[0] for x in fam_units)); boots = []
        per_ds = {ds: [(np.array(rec[P][met]) - np.array(rec[b][met])) for (dd, _, rec) in fam_units if dd == ds] for ds in dsets}
        for _ in range(5000):
            vals = []
            for ds in rng.choice(dsets, len(dsets), replace=True):
                fl = per_ds[ds]
                for i in rng.integers(0, len(fl), len(fl)):
                    sd = fl[i]; vals.append(np.nanmean(sd[rng.integers(0, len(sd), len(sd))]))
            boots.append(np.mean(vals))
        rows[b] = dict(mean=float(d.mean()), hl=hl[0], hl_lo=hl[1], hl_hi=hl[2], r_rb=rank_biserial(d),
                       wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()), p=float(p),
                       boot_lo=float(np.percentile(boots, 2.5)), boot_hi=float(np.percentile(boots, 97.5)))
    adj = holm([rows[b]["p"] for b in BASE_B])
    for b, a in zip(BASE_B, adj):
        rows[b]["p_holm"] = float(a)
    B[met] = rows

# Friedman / Nemenyi on 22 family-level means
M = np.array([[np.nanmean(rec[m]["DR"]) for m in CD_METHODS] for _, _, rec in fam_units])
ranks = np.array([stats.rankdata(-row) for row in M])
avg = ranks.mean(0)
fr = stats.friedmanchisquare(*M.T)
k, N = len(CD_METHODS), len(M)
q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}[k]
CD = q * np.sqrt(k * (k + 1) / (6 * N))
F = dict(methods=CD_METHODS, avg_rank=avg.tolist(), CD=float(CD), p=float(fr.pvalue), N=int(N))

# critical-difference figure (larger, for readability)
LAB = {P: "CH-IDS", "GBDT [conformal]": "LightGBM (conformal)", "RF [conformal]": "RF (conformal)",
       "Seq. hybrid: rules->RF+SMOTE [default]": "Seq. hybrid (default)", "AE [conformal]": "AE (conformal)",
       "IForest [conformal]": "IForest (conformal)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(8.0, 2.9))
lo, hi = 1, k
ax.set_xlim(lo - 0.3, hi + 0.3); ax.set_ylim(0.12, 1.05); ax.axis("off")
ax.hlines(0.8, lo, hi, color="black", lw=1)
for r in range(lo, hi + 1):
    ax.vlines(r, 0.78, 0.82, color="black", lw=1); ax.text(r, 0.86, str(r), ha="center", fontsize=10)
order = np.argsort(avg)
for i, j in enumerate(order):
    y = 0.46 - 0.13 * (i % 3) if i < k / 2 else 0.46 - 0.13 * ((i - (k + 1) // 2) % 3)
    left = i < (k + 1) // 2
    x = avg[j]
    ax.vlines(x, y, 0.8, color="#2a78d6" if CD_METHODS[j] == P else "#444444", lw=1)
    tx = lo - 0.25 if left else hi + 0.25
    ax.hlines(y, min(x, tx), max(x, tx), color="#2a78d6" if CD_METHODS[j] == P else "#444444", lw=1)
    ax.text(tx + (-0.02 if left else 0.02), y, f"{LAB[CD_METHODS[j]]} ({x:.2f})", ha="right" if left else "left",
            va="center", fontsize=10, color="#2a78d6" if CD_METHODS[j] == P else "black",
            fontweight="bold" if CD_METHODS[j] == P else None)
# cliques: groups of consecutive methods whose rank range < CD
srt = np.sort(avg); cl = []
for a in range(k):
    b = a
    while b + 1 < k and srt[b + 1] - srt[a] < CD:
        b += 1
    if b > a and not any(c[0] <= a and c[1] >= b for c in cl):
        cl.append((a, b))
for i, (a, b) in enumerate(cl):
    ax.hlines(0.72 - 0.06 * i, srt[a] - 0.03, srt[b] + 0.03, color="black", lw=3)
ax.hlines(0.97, lo, lo + CD, color="black", lw=1.5); ax.text(lo + CD / 2, 0.99, f"CD = {CD:.2f}", ha="center", va="bottom", fontsize=10)
fig.tight_layout(); fig.savefig(OUTF, dpi=600)

json.dump(dict(A=A, summaryA=summaryA, B=B, friedman=F, n_family_units=len(fam_units)),
          open(f"{RES}/stats_rev3.json", "w"), indent=1)
print(json.dumps(summaryA)); print(json.dumps(F))
for met in B:
    for b, v in B[met].items():
        print(met, b, {k: round(x, 4) if isinstance(x, float) else x for k, x in v.items()})
