"""Experiment 1: standard (official split) protocol, 5 seeds, all methods, two operating points.
usage: python exp_standard.py DATASET SEED [SEED ...]"""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import sys, os, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import data, models, metrics
from models import tippett, conformal_pvalues
from imblearn.over_sampling import SMOTE

ALPHA = 0.01
OUT = ROOT + "/results/standard"
os.makedirs(OUT, exist_ok=True)


def thr_conformal(cal_benign_scores, alpha=ALPHA):
    # flag iff conformal p-value <= alpha
    return lambda s: conformal_pvalues(cal_benign_scores, s) <= alpha


def per_family(fam, y, pred):
    out = {}
    for f in np.unique(fam[y == 1]):
        m = fam == f
        out[str(f)] = [float(pred[m].mean()), int(m.sum())]
    return out


def run(dsname, seed):
    fn = f"{OUT}/{dsname}_{seed}.json"
    if os.path.exists(fn):
        print("skip", fn); return
    ds = data.LOADERS[dsname]()
    P = data.prepare(ds, seed)
    Xf, yf, ff, Xc, yc, fc = P["Xf"], P["yf"], P["ff"], P["Xc"], P["yc"], P["fc"]
    cb = yc == 0
    R = dict(dataset=dsname, seed=seed, n_fit=len(yf), n_cal=len(yc), n_feat=Xf.shape[1],
             results={}, timing={}, extra={})
    novel = None
    if dsname == "NSL-KDD":
        train_types = ds["train_attacks"]

    # ---------------- proposed
    t0 = time.time(); ae = models.AutoEncoder(seed=seed).fit(Xf[yf == 0]); t_ae = time.time() - t0
    t0 = time.time()
    ch = models.CHIDS(alpha=ALPHA, lam="auto", seed=seed, ano=ae).fit(Xf, yf, Xc, yc, ff, fc)
    R["timing"]["CH-IDS_fit"] = time.time() - t0 + t_ae
    R["extra"]["lambda"] = ch.lam_
    R["extra"]["inner_dr"] = ch.inner_dr
    R["extra"]["n_rules"] = int(len(ch.rl.keep_arr))
    if seed == 0:
        R["extra"]["rules"] = [([(c[0], c[1], float(c[2])) for c in conds], sup)
                               for conds, sup in ch.rl.rules(P["names"])]
        imp = ch.sup.booster_.feature_importance("gain")
        order = np.argsort(-imp)[:15]
        R["extra"]["importance"] = [(P["names"][i], float(imp[i])) for i in order]

    # baselines
    fitted = {}
    for name in ["LR", "DT", "RF", "SVM-RBF", "MLP", "XGBoost"]:
        m = models.make_baseline(name, seed)
        t0 = time.time()
        if name == "SVM-RBF":
            rng = np.random.default_rng(seed)
            sub = rng.choice(len(yf), min(20000, len(yf)), replace=False)
            m.fit(Xf[sub], yf[sub])
        else:
            m.fit(Xf, yf)
        R["timing"][name + "_fit"] = time.time() - t0
        fitted[name] = m
        print(dsname, seed, name, "fit", round(time.time() - t0, 1), flush=True)
    fitted["LightGBM"] = ch.sup
    t0 = time.time(); ifo = models.IForest(seed).fit(Xf[yf == 0]); R["timing"]["IForest_fit"] = time.time() - t0
    # student's design: rule pre-filter -> RF trained on SMOTE-balanced data, default threshold
    t0 = time.time()
    Xs, ys = SMOTE(random_state=seed).fit_resample(Xf, yf)
    rfs = models.make_baseline("RF", seed).fit(Xs, ys)
    R["timing"]["SeqHybrid_fit"] = time.time() - t0
    del Xs, ys

    cal = {k: models.sup_score(m, Xc[cb]) for k, m in fitted.items()}
    cal["IForest"] = ifo.score(Xc[cb]); cal["AE"] = ae.score(Xc[cb])
    cal["SeqHybrid"] = rfs.predict_proba(Xc[cb])[:, 1]
    fit_benign_99 = {"IForest": np.quantile(ifo.score(Xf[yf == 0]), 0.99),
                     "AE": np.quantile(ae.score(Xf[yf == 0]), 0.99)}

    for tname, (Xt, yt, ft, st) in P["tests"].items():
        res = {}
        nov = None
        if dsname == "NSL-KDD":
            nov = np.array([(a not in train_types) and (a != "normal") for a in st])
            R["extra"][f"novel_n_{tname}"] = int(nov.sum())

        def rec(k, op, pr, sc):
            pr = np.asarray(pr).astype(int)
            d = dict(op=op, **metrics.binary_metrics(yt, pr, sc), fam=per_family(ft, yt, pr))
            if nov is not None:
                d["novel_DR"] = float(pr[nov].mean())
                d["known_DR"] = float(pr[(yt == 1) & ~nov].mean())
            res[k] = d
        # proposed + ablations from one set of components
        t0 = time.time(); pred, z, comp = ch.predict_score(Xt, fast_path=True); t_pred = time.time() - t0
        R["timing"][f"CH-IDS_pred_{tname}"] = t_pred
        R["extra"][f"rule_coverage_{tname}"] = float(comp["rule"].mean())
        R["extra"][f"rule_precision_{tname}"] = float(yt[comp["rule"]].mean()) if comp["rule"].any() else None
        r, ps, pa, rs, ra = ch.raw_components(Xt)
        lam = ch.lam_
        def zsc(T, rr):
            zz = -np.log(T) + 1e-3 * rs / (1 + rs) + 1e-3 * ra / (1 + ra)
            zz = zz.copy(); zz[rr] = 1e6; return zz
        T = tippett(ps, pa, lam)
        variants = {
            "CH-IDS (proposed)": (pred.astype(bool), z),
            "CH-IDS w/o rule tier": (T <= ch.t_, zsc(T, np.zeros(len(r), bool))),
            "CH-IDS w/o anomaly tier": (r | (ps <= ch.threshold_for(0.0)), zsc(ps, r)),
            "CH-IDS w/o supervised tier": (r | (pa <= ch.threshold_for(1.0)), zsc(pa, r)),
            "CH-IDS fixed lambda=0.5": (r | (tippett(ps, pa, .5) <= ch.threshold_for(.5)), zsc(tippett(ps, pa, .5), r)),
            "Rule tier only": (r, r.astype(float)),
        }
        for k, (pr, sc) in variants.items():
            rec(k, "conformal", pr, sc)
        # baselines
        for k, m in fitted.items():
            t0 = time.time(); s = models.sup_score(m, Xt); R["timing"][f"{k}_pred_{tname}"] = time.time() - t0
            dflt = (s > 0) if k == "SVM-RBF" else (s >= 0.5)
            cf = thr_conformal(cal[k])(s)
            rec(k + " [default]", "default", dflt, s)
            rec(k + " [conformal]", "conformal", cf, s)
        for k, f in [("IForest", ifo.score), ("AE", ae.score)]:
            t0 = time.time(); s = f(Xt); R["timing"][f"{k}_pred_{tname}"] = time.time() - t0
            dflt = s > fit_benign_99[k]; cf = thr_conformal(cal[k])(s)
            rec(k + " [default]", "default", dflt, s)
            rec(k + " [conformal]", "conformal", cf, s)
        t0 = time.time()
        rr = ch.rl.match(Xt); s = np.ones(len(Xt)); s[~rr] = rfs.predict_proba(Xt[~rr])[:, 1]
        R["timing"][f"SeqHybrid_pred_{tname}"] = time.time() - t0
        sp = rfs.predict_proba(Xt)[:, 1]; sc = sp.copy(); sc[rr] = 2.0
        dflt = rr | (sp >= 0.5); cf = rr | thr_conformal(cal["SeqHybrid"])(sp)
        rec("Seq. hybrid: rules->RF+SMOTE [default]", "default", dflt, sc)
        rec("Seq. hybrid: rules->RF+SMOTE [conformal]", "conformal", cf, sc)
        # alpha sweep (validity of the false-alarm control)
        sweep = {}
        for a in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
            ta = ch.threshold_for(lam, a)
            pr = r | (T <= ta)
            t0_ = ch.threshold_for(0.0, a)
            pr0 = ps <= t0_
            sweep[str(a)] = dict(CH=dict(FPR=float(pr[yt == 0].mean()), DR=float(pr[yt == 1].mean())),
                                 GBDT=dict(FPR=float(pr0[yt == 0].mean()), DR=float(pr0[yt == 1].mean())))
        R["extra"][f"alpha_sweep_{tname}"] = sweep
        R["results"][tname] = res
    json.dump(R, open(fn, "w"), default=float)
    print("done", fn, flush=True)


if __name__ == "__main__":
    for sd in sys.argv[2:]:
        run(sys.argv[1], int(sd))
