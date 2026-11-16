"""Experiment 2: zero-day simulation by leave-one-attack-family-out (LOAFO).
For each attack family f: every method is trained/calibrated without any flow of f and tested on
benign test flows + test flows of f.   usage: python exp_loafo.py DATASET SEED [SEED ...]"""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import sys, os, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import data, models, metrics
from models import tippett, conformal_pvalues
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

ALPHA = 0.01
OUT = ROOT + "/results/loafo"
os.makedirs(OUT, exist_ok=True)


def run(dsname, seed):
    fn = f"{OUT}/{dsname}_{seed}.json"
    if os.path.exists(fn):
        print("skip", fn); return
    ds = data.LOADERS[dsname]()
    P = data.prepare(ds, seed)
    Xf, yf, ff, Xc, yc, fc = P["Xf"], P["yf"], P["ff"], P["Xc"], P["yc"], P["fc"]
    tname = list(P["tests"])[0]
    Xt, yt, ft, st = P["tests"][tname]
    cb = yc == 0
    ae = models.AutoEncoder(seed=seed).fit(Xf[yf == 0])
    ifo = models.IForest(seed).fit(Xf[yf == 0])
    cal_ae, cal_if = ae.score(Xc[cb]), ifo.score(Xc[cb])
    out = dict(dataset=dsname, seed=seed, test=tname, families={})
    part = fn + ".partial"
    if os.path.exists(part):
        out = json.load(open(part))
    fams = [f for f in np.unique(ff) if f != "Normal"]
    for fam in fams:
        if str(fam) in out["families"]:
            continue
        t_start = time.time()
        k, kc = ff != fam, fc != fam
        sel = (yt == 0) | (ft == fam)
        X, y = Xt[sel], yt[sel]
        att = y == 1
        res = {}

        def rec(name, pred, score):
            pred = np.asarray(pred).astype(bool)
            res[name] = dict(DR=float(pred[att].mean()), FPR=float(pred[~att].mean()),
                             AUC=float(roc_auc_score(y, score)))
        ch = models.CHIDS(alpha=ALPHA, lam="auto", seed=seed, ano=ae).fit(Xf[k], yf[k], Xc[kc], yc[kc], ff[k], fc[kc])
        pred, z, comp = ch.predict_score(X)
        rec("CH-IDS (proposed)", pred, z)
        r, ps, pa, rs, ra = ch.raw_components(X)
        tb = 1e-3 * rs / (1 + rs) + 1e-3 * ra / (1 + ra)
        for lam in [0.0, 0.25, 0.5]:
            T = tippett(ps, pa, lam)
            zz = -np.log(T) + tb; zz[r] = 1e6
            rec(f"CH-IDS lambda={lam}", r | (T <= ch.threshold_for(lam)), zz)
        rec("GBDT [conformal]", ps <= ch.threshold_for(0.0), rs)
        rec("AE [conformal]", conformal_pvalues(cal_ae, ra) <= ALPHA, ra)
        si = ifo.score(X)
        rec("IForest [conformal]", conformal_pvalues(cal_if, si) <= ALPHA, si)
        rec("Rule tier only", r, r.astype(float) + 1e-6 * rs)
        for name in ["RF"]:
            m = models.make_baseline(name, seed).set_params(n_estimators=100).fit(Xf[k], yf[k])
            s = m.predict_proba(X)[:, 1]
            cs = m.predict_proba(Xc[kc & cb])[:, 1]
            rec(f"{name} [default]", s >= 0.5, s)
            rec(f"{name} [conformal]", conformal_pvalues(cs, s) <= ALPHA, s)
        Xs, ys = SMOTE(random_state=seed).fit_resample(Xf[k], yf[k])
        rfs = models.make_baseline("RF", seed).set_params(n_estimators=100).fit(Xs, ys)
        del Xs, ys
        s = rfs.predict_proba(X)[:, 1]; sc = s.copy(); sc[r] = 2.0
        rec("Seq. hybrid: rules->RF+SMOTE [default]", r | (s >= 0.5), sc)
        out["families"][str(fam)] = dict(n_attack=int(att.sum()), n_benign=int((~att).sum()),
                                         lam=ch.lam_, inner_dr=ch.inner_dr, n_rules=int(len(ch.rl.keep_arr)),
                                         results=res)
        json.dump(out, open(part, "w"), default=float)
        print(dsname, seed, fam, "lam", ch.lam_, {kk: round(v["DR"], 3) for kk, v in res.items()},
              round(time.time() - t_start), "s", flush=True)
    json.dump(out, open(fn, "w"), default=float)
    if os.path.exists(part):
        os.remove(part)
    print("done", fn, flush=True)


if __name__ == "__main__":
    for sd in sys.argv[2:]:
        run(sys.argv[1], int(sd))
