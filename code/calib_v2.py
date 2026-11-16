"""False-alarm calibration analysis for the split-calibrated CH-IDS (results/v2) -> results/calib.json + figures."""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, glob
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = ROOT + "/results"; FIG = ROOT + "/figs"
ALPHAS = ["0.001", "0.005", "0.01", "0.02", "0.05", "0.1"]
V = {}
for fn in sorted(glob.glob(f"{RES}/v2/*.json")):
    r = json.load(open(fn)); V.setdefault(r["dataset"], []).append(r)


def cp(x, n, conf=0.95):
    """Two-sided Clopper-Pearson interval for a binomial proportion."""
    a = 1 - conf
    lo = 0.0 if x == 0 else stats.beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else stats.beta.ppf(1 - a / 2, x + 1, n - x)
    return float(lo), float(hi)


def kk(a, n2):
    return int(np.floor(float(a) * (n2 + 1) + 1e-12))


def ref_band(m, n2, a):
    """Proposition (realised rate): given D_fit and D_cal1, with i.i.d. benign flows and distinct values of G, the number
    of false alarms among m new benign flows is BetaBinomial(m, k, n2 + 1 - k), k = floor(a (n2 + 1)).
    Returns the 2.5% and 97.5% quantiles (as rates) and the exact probability that the count falls in [lo, hi]
    (>= 0.95 because the distribution is discrete)."""
    k = kk(a, n2)
    if k < 1:
        return 0.0, 0.0, 1.0
    d = stats.betabinom(m, k, n2 + 1 - k)
    lo, hi = d.ppf(0.025), d.ppf(0.975)
    cov = float(d.cdf(hi) - (d.cdf(lo - 1) if lo >= 1 else 0.0))
    return float(lo / m), float(hi / m), cov


RNG = np.random.default_rng(12345)


def mean_band(counts, sims=200000):
    """95% reference band for the across-seed mean of the realised rate, by simulation from the Proposition
    (independent calibrations; each seed: F ~ Beta(k, n2 + 1 - k), X ~ Bin(m, F))."""
    tot = np.zeros(sims)
    for fp, m, n2, a in counts:
        k = kk(a, n2)
        F = RNG.beta(k, n2 + 1 - k, sims) if k >= 1 else np.zeros(sims)
        tot += RNG.binomial(m, F) / m
    tot /= len(counts)
    return float(np.quantile(tot, 0.025)), float(np.quantile(tot, 0.975))


def entry(counts):
    """counts: list of (fp, m, n2, alpha), one per seed.
    Reported: per-seed rate with Clopper-Pearson interval for the conditional false-alarm rate of that calibration;
    across-seed mean with a t-interval (seeds are independent calibrations, so the across-seed spread contains
    both the calibration and the test-sampling variability); per-seed position relative to the reference band;
    simulated reference band for the across-seed mean; the bound floor(a(n+1))/(n+1) of Theorem 1."""
    rows = [c for c in counts if c[1] > 0]
    per = np.array([c[0] / c[1] for c in rows])
    nseed = len(per)
    mean = float(per.mean())
    sd = float(per.std(ddof=1)) if nseed > 1 else 0.0
    t = stats.t.ppf(0.975, nseed - 1) if nseed > 1 else 0.0
    bands = [ref_band(c[1], c[2], c[3]) for c in rows]
    pos = ["inside" if b[0] <= r <= b[1] else ("below" if r < b[0] else "above") for r, b in zip(per, bands)]
    return dict(mean=mean, sd=sd, ci=[max(0.0, mean - t * sd / np.sqrt(nseed)), mean + t * sd / np.sqrt(nseed)],
                seeds=per.tolist(), seed_ci=[cp(c[0], c[1]) for c in rows], m_per_seed=float(np.mean([c[1] for c in rows])),
                inside=pos.count("inside"), below=pos.count("below"), above=pos.count("above"), n=nseed,
                band=[float(np.mean([b[0] for b in bands])), float(np.mean([b[1] for b in bands]))],
                band_cov=[b[2] for b in bands], mean_band=mean_band(rows),
                bound=float(np.mean([kk(c[3], c[2]) / (c[2] + 1) for c in rows])))


OUT = {"sets": {}, "rules": {}, "recal": {}, "n2": {}, "lam": {}, "rule_benign": {}, "split": {}, "benign_shift": {}}
for ds, runs in V.items():
    OUT["n2"][ds] = [r["n_cal2_benign"] for r in runs]
    OUT["lam"][ds] = [r["lam"] for r in runs]
    OUT["split"][ds] = {k: [r["split"][k] for r in runs] for k in runs[0]["split"]}
    OUT["benign_shift"][ds] = {s: {k: float(np.mean([r["benign_shift"][s][k] for r in runs])) if not isinstance(runs[0]["benign_shift"][s][k], list)
                                   else np.mean([r["benign_shift"][s][k] for r in runs], axis=0).tolist() for k in runs[0]["benign_shift"][s]}
                               for s in runs[0]["benign_shift"]}
    h = sum(r["rule_hits_cal2"] for r in runs); n = sum(r["n_cal2_benign"] for r in runs)
    OUT["rules"][ds] = dict(hits=h, n=n, rate=h / n, cp_upper=cp(h, n)[1], per_seed=[r["rule_hits_cal2"] for r in runs], n_rules=[r["n_rules"] for r in runs])
    rb = {}
    for t in runs[0]["rule_benign"]:
        rb[t] = [int(sum(r["rule_benign"][t][0] for r in runs)), int(sum(r["rule_benign"][t][1] for r in runs))]
    OUT["rule_benign"][ds] = rb
    sets = {}
    for s in ["in-distribution", "UDP-only (protocol mix)", "heavy flows (bytes > P90)"]:
        sets[s] = {a: [(r["shift"][s][a]["fp"], r["shift"][s][a]["m"], r["n_cal2_benign"], a) for r in runs] for a in ALPHAS}
    for t in runs[0]["sweep"]:
        sets["test: " + t] = {a: [(r["sweep"][t][a]["fp"], r["sweep"][t][a]["m"], r["n_cal2_benign"], a) for r in runs] for a in ALPHAS}
    OUT["sets"][ds] = {s: {a: entry(c) for a, c in d.items()} for s, d in sets.items()}
    OUT["recal"][ds] = {}
    for tname in runs[0]["recal"]:
        d = {}
        for w in runs[0]["recal"][tname]:
            f = [x for r in runs if tname in r["recal"] and w in r["recal"][tname] for x in r["recal"][tname][w]["FPR"]]
            g = [x for r in runs if tname in r["recal"] and w in r["recal"][tname] for x in r["recal"][tname][w]["DR"]]
            d[w] = dict(FPR=float(np.mean(f)), FPR_lo=float(np.percentile(f, 2.5)), FPR_hi=float(np.percentile(f, 97.5)),
                        DR=float(np.mean(g)) if g else None, m_eval=runs[0]["recal"][tname][w]["m_eval"], reps=len(f))
        OUT["recal"][ds][tname] = d
# summary of the per-seed band check on in-distribution data
tot = dict(inside=0, below=0, above=0, n=0, expected=0.0)
for ds in OUT["sets"]:
    for a in ALPHAS:
        e = OUT["sets"][ds]["in-distribution"][a]
        for k in ["inside", "below", "above", "n"]:
            tot[k] += e[k]
        tot["expected"] += float(np.sum(e["band_cov"]))
OUT["id_band_summary"] = tot
json.dump(OUT, open(f"{RES}/calib.json", "w"), indent=1)

# ------------------------------------------------------------------ figures
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK2, GRID = "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "savefig.dpi": 600})
al = np.array([float(a) for a in ALPHAS])


def panel(ax, series, title):
    ax.plot([5e-4, 0.2], [5e-4, 0.2], ls="--", color=INK2, lw=0.8)
    for i, (lab, d) in enumerate(series):
        y = np.array([d[a]["mean"] for a in ALPHAS]); lo = np.array([d[a]["ci"][0] for a in ALPHAS]); hi = np.array([d[a]["ci"][1] for a in ALPHAS])
        y = np.maximum(y, 6e-4); lo = np.maximum(lo, 6e-4)
        ax.errorbar(al * (1 + 0.06 * (i - 1)), y, yerr=[y - lo, hi - y], marker="osD^v"[i % 5], ms=3.5, lw=1.2, capsize=1.5, color=C[i], label=lab, mec="white", mew=0.4)
        if i == 0:
            b = np.array([d[a]["mean_band"] for a in ALPHAS])
            ax.fill_between(al, np.maximum(b[:, 0], 6e-4), b[:, 1], color="#bbbbbb", alpha=0.35, lw=0, label="95% band, 5-seed mean")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(7e-4, 0.14); ax.set_ylim(5e-4, 0.7)
    ax.set_title(title, fontsize=7.5); ax.set_xlabel("Target false-alarm rate α")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
    ax.legend(fontsize=5.8, loc="upper left")


fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.55), sharey=True)
S = OUT["sets"]
panel(axs[0], [(d, S[d]["in-distribution"]) for d in ["NSL-KDD", "UNSW-NB15", "TON_IoT"] if d in S], "(a) In-distribution hold-out")
nat = []
for d, t in [("NSL-KDD", "test: KDDTest+"), ("NSL-KDD", "test: KDDTest-21"), ("UNSW-NB15", "test: UNSW-NB15 test"), ("TON_IoT", "test: TON_IoT test")]:
    if d in S and t in S[d]:
        nat.append((t.replace("test: ", "").replace(" test", ""), S[d][t]))
panel(axs[1], nat, "(b) Official test sets")
panel(axs[2], [(d, S[d]["heavy flows (bytes > P90)"]) for d in ["NSL-KDD", "UNSW-NB15", "TON_IoT"] if d in S], "(c) Heavy-flow sub-population")
axs[0].set_ylabel("Realised FPR (mean of 5 seeds)")
fig.tight_layout(w_pad=0.4); fig.savefig(f"{FIG}/fig_calib.png"); plt.close(fig)

print(json.dumps(OUT["id_band_summary"]))
for ds in OUT["sets"]:
    for st, d in OUT["sets"][ds].items():
        print(ds, "|", st, "|", " ".join(f"{a}:{100*e['mean']:.2f}[{100*e['ci'][0]:.2f}-{100*e['ci'][1]:.2f}] i{e['inside']}/b{e['below']}/a{e['above']}" for a, e in d.items()))
