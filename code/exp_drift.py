"""Revision 3: label-free drift monitor based on randomised system p-values.

For every flow the randomised conformal p-value  P~(x) = (#{G_j < G(x)} + U (1 + #{G_j = G(x)})) / (n + 1),
U ~ Uniform(0,1), is uniform for exchangeable benign flows even when G has ties. A monitoring window of w flows is
flagged when the p-values above tau (which are mostly benign, because attacks receive small p-values) are not uniform
on (tau, 1] (one-sample Kolmogorov-Smirnov test at level 0.01). Windows are drawn at random from the hold-out (no
shift), from the official test sets (shift) and, for TON_IoT, from benign flows of unseen hosts.
usage: python exp_drift.py DATASET SEED...   |   python exp_drift.py host SEED..."""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import sys, os, json, warnings
import numpy as np
from scipy.stats import kstest
warnings.filterwarnings("ignore")
import data, models
from exp_v2 import split_cal

OUT = ROOT + "/results/rev3_drift"
os.makedirs(OUT, exist_ok=True)


def rand_p(Gcal, Gx, rng):
    s = np.sort(Gcal); n = len(s)
    lt = np.searchsorted(s, Gx, "left"); le = np.searchsorted(s, Gx, "right")
    return (lt + rng.random(len(Gx)) * (1 + le - lt)) / (n + 1)


def monitor(p, rng, w, tau=0.1, reps=200):
    if len(p) < w:
        return None
    rej = []
    for _ in range(reps):
        q = p[rng.choice(len(p), w, replace=False)]; q = q[q > tau]
        rej.append(float(kstest((q - tau) / (1 - tau), "uniform").pvalue < 0.01))
    return float(np.mean(rej))


def run(dsname, seed, host=False):
    fn = f"{OUT}/{'host_' if host else ''}{dsname}_{seed}.json"
    if os.path.exists(fn):
        return
    rng = np.random.default_rng(777 + seed)
    if host:
        import exp_split
        ds = exp_split.load_host(seed)
    else:
        ds = data.LOADERS[dsname]()
    P = data.prepare(ds, seed)
    Xf, yf, ff, Xc, yc, fc = P["Xf"], P["yf"], P["ff"], P["Xc"], P["yc"], P["fc"]
    i1, i2, iid = split_cal(P, seed)
    ch = models.SplitCHIDS(alpha=0.01, lam="auto", seed=seed).fit(Xf, yf, ff, Xc[i1], yc[i1], fc[i1], Xc[i2], yc[i2])
    Gcal = ch.G(ch.comp2)
    sets = {"hold-out": (Xc[iid], yc[iid])}
    for k, v in P["tests"].items():
        sets[k] = (v[0], v[1])
    R = {"dataset": dsname, "seed": seed, "host": host, "sets": {}}
    for en, (Xe, ye) in sets.items():
        p = rand_p(Gcal, ch.G(ch.components(Xe)), rng)
        rec = {"attack_share": float(ye.mean())}
        for w in [250, 500, 1000]:
            rec[f"mixed_w{w}"] = monitor(p, rng, w)
            rec[f"benign_w{w}"] = monitor(p[ye == 0], rng, w)
        R["sets"][en] = rec
    json.dump(R, open(fn, "w")); print(fn, json.dumps(R["sets"])[:400], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "host":
        for s in sys.argv[2:]:
            run("TON_IoT", int(s), host=True)
    else:
        for s in sys.argv[2:]:
            run(sys.argv[1], int(s))
