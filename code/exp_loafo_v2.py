"""Unseen-attack-family protocol for the split-calibrated CH-IDS. Baseline rows are taken from the earlier
run (results/loafo, same data splits and seeds); only the CH-IDS rows are recomputed.
usage: python exp_loafo_v2.py DATASET SEED [SEED ...]"""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import sys, os, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
import data, models
from exp_v2 import split_cal

OUT = ROOT + "/results/loafo_v2"
os.makedirs(OUT, exist_ok=True)


def run(dsname, seed):
    fn = f"{OUT}/{dsname}_{seed}.json"
    if os.path.exists(fn):
        print("skip", fn); return
    old = json.load(open(f"{ROOT}/results/loafo/{dsname}_{seed}.json"))
    ds = data.LOADERS[dsname](); P = data.prepare(ds, seed)
    Xf, yf, ff, Xc, yc, fc = P["Xf"], P["yf"], P["ff"], P["Xc"], P["yc"], P["fc"]
    i1, i2, iid = split_cal(P, seed)
    tname = list(P["tests"])[0]
    Xt, yt, ft, st = P["tests"][tname]
    ae = models.AutoEncoder(seed=seed).fit(Xf[yf == 0])
    out = dict(dataset=dsname, seed=seed, test=tname, families={})
    for fam in [f for f in np.unique(ff) if f != "Normal"]:
        t0 = time.time()
        k = ff != fam
        k1, k2 = fc[i1] != fam, fc[i2] != fam
        sel = (yt == 0) | (ft == fam)
        X, y = Xt[sel], yt[sel]; att = y == 1
        ch = models.SplitCHIDS(alpha=0.01, lam="auto", seed=seed, ano=ae).fit(
            Xf[k], yf[k], ff[k], Xc[i1][k1], yc[i1][k1], fc[i1][k1], Xc[i2][k2], yc[i2][k2])
        c = ch.components(X)
        res = dict(old["families"][fam]["results"])
        for key in list(res):
            if key.startswith("CH-IDS") or key == "Rule tier only":
                del res[key]

        def rec(name, pred, score):
            res[name] = dict(DR=float(pred[att].mean()), FPR=float(pred[~att].mean()), AUC=float(roc_auc_score(y, score)))
        pred, z, _ = ch.decide(c, 0.01, "full")
        rec("CH-IDS (proposed)", pred, z)
        for lam in [0.0, 0.25, 0.5]:
            pred, z, _ = ch.decide(c, 0.01, "full", lam)
            rec(f"CH-IDS lambda={lam}", pred, z)
        pred, z, _ = ch.decide(c, 0.01, "rules_only")
        rec("Rule tier only", pred, c["r"].astype(float))  # binary rule match: AUC of the match indicator
        out["families"][fam] = dict(n_attack=int(att.sum()), n_benign=int((~att).sum()), lam=ch.lam_,
                                    inner_dr=ch.inner_dr, n_rules=int(len(ch.rl.keep_arr)),
                                    rule_hits_cal2=ch.rule_benign_hits()[0], results=res)
        print(dsname, seed, fam, "lam", ch.lam_, round(res["CH-IDS (proposed)"]["DR"], 3), round(time.time() - t0), "s", flush=True)
    json.dump(out, open(fn, "w"), default=float)
    print("done", fn, flush=True)


if __name__ == "__main__":
    for sd in sys.argv[2:]:
        run(sys.argv[1], int(sd))
