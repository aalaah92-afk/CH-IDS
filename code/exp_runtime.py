"""Protocol D: computational cost. Run alone, with CHIDS_NJOBS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1, so that
every model trains and predicts on one thread.

Timed quantities (seed 0, same D_fit / D_cal1 / D_cal2 split as exp_v2.py):
  * training: CH-IDS = rule induction + LightGBM + autoencoder + lambda search + scoring of D_cal2;
              CH-IDS fixed lambda = the same without the lambda search;
              Seq. hybrid = rule induction + SMOTE + RF; the other baselines = model fit only.
  * batch throughput on the first official test set (best of three), single-flow latency (median and 95th
    percentile over 300 flows).
  * CH-IDS predictions use SplitCHIDS.predict_fast, which is checked to give exactly the decisions of
    SplitCHIDS.decide (Equation (6)) on the whole test set.
usage: python exp_runtime.py"""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, time, warnings, os
import numpy as np
warnings.filterwarnings("ignore")
import data, models
from imblearn.over_sampling import SMOTE
from exp_v2 import split_cal

OUT = ROOT + "/results/runtime_v2.json"
ALPHA = 0.01
R = {"threads": int(models.NJ)}
for dsname in ["NSL-KDD", "UNSW-NB15", "TON_IoT"]:
    ds = data.LOADERS[dsname](); P = data.prepare(ds, 0)
    Xf, yf, ff, Xc, yc, fc = P["Xf"], P["yf"], P["ff"], P["Xc"], P["yc"], P["fc"]
    Xt = list(P["tests"].values())[0][0]
    i1, i2, iid = split_cal(P, 0)
    rec = {"fit": {}, "throughput": {}, "latency_ms": {}}

    t0 = time.time(); ae = models.AutoEncoder(seed=0).fit(Xf[yf == 0]); tae = time.time() - t0
    t0 = time.time(); ch = models.SplitCHIDS(seed=0, ano=ae).fit(Xf, yf, ff, Xc[i1], yc[i1], fc[i1], Xc[i2], yc[i2])
    rec["fit"]["CH-IDS"] = time.time() - t0 + tae
    t0 = time.time(); chf = models.SplitCHIDS(seed=0, ano=ae, lam=0.5).fit(Xf, yf, ff, Xc[i1], yc[i1], fc[i1], Xc[i2], yc[i2])
    rec["fit"]["CH-IDS (fixed lambda, no inner search)"] = time.time() - t0 + tae
    Gcal = ch.G(ch.comp2)

    # exactness of the fast path
    full = ch.decide(ch.components(Xt), ALPHA, "full", None, Gcal)[0]
    fast = ch.predict_fast(Xt, ALPHA, Gcal)
    k, gk = ch.threshold(Gcal, ALPHA)
    rec["fast_path"] = dict(identical=bool(np.array_equal(full, fast)), k=k, h=int((Gcal <= ch.RULE_G).sum()),
                            n=int(len(Gcal)), rule_coverage=float(ch.rl.match(Xt).mean()))

    fitted = {}
    for name in ["RF", "XGBoost", "LightGBM", "MLP"]:
        t0 = time.time(); fitted[name] = models.make_baseline(name, 0).fit(Xf, yf); rec["fit"][name] = time.time() - t0
    t0 = time.time()
    rl = models.RuleLayer(seed=0).fit(Xf, yf, Xc[i1], yc[i1])
    Xs, ys = SMOTE(random_state=0).fit_resample(Xf, yf)      # as in exp_standard.py
    rfs = models.make_baseline("RF", 0).fit(Xs, ys)
    rec["fit"]["Seq. hybrid (rules->RF+SMOTE)"] = time.time() - t0
    rec["fit"]["AE"] = tae

    def seq_predict(X):
        r = rl.match(X); out = r.copy()
        if (~r).any():
            out[~r] = rfs.predict_proba(X[~r])[:, 1] >= 0.5
        return out

    def thr(f, X, reps=3):
        best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter(); f(X); best = min(best, time.perf_counter() - t0)
        return len(X) / best
    rec["throughput"]["CH-IDS"] = thr(lambda X: ch.predict_fast(X, ALPHA, Gcal), Xt)
    rec["throughput"]["CH-IDS (all tiers evaluated)"] = thr(lambda X: ch.decide(ch.components(X), ALPHA, "full", None, Gcal)[0], Xt)
    for name, m in fitted.items():
        rec["throughput"][name] = thr(lambda X, m=m: m.predict_proba(X), Xt)
    rec["throughput"]["AE"] = thr(ae.score, Xt)
    rec["throughput"]["Seq. hybrid (rules->RF+SMOTE)"] = thr(seq_predict, Xt)

    rng = np.random.default_rng(0); idx = rng.choice(len(Xt), 300, replace=False)

    def lat(f):
        ts = []
        for i in idx:
            x = Xt[i:i + 1]; t0 = time.perf_counter(); f(x); ts.append(time.perf_counter() - t0)
        return [float(np.median(ts) * 1e3), float(np.percentile(ts, 95) * 1e3)]
    rec["latency_ms"]["CH-IDS"] = lat(lambda x: ch.predict_fast(x, ALPHA, Gcal))
    for name in ["LightGBM", "XGBoost", "RF"]:
        rec["latency_ms"][name] = lat(lambda x, m=fitted[name]: m.predict_proba(x))
    rec["latency_ms"]["Seq. hybrid (rules->RF+SMOTE)"] = lat(seq_predict)

    rec["batch_scaling"] = {}
    for b in [1, 10, 100, 1000, 10000, len(Xt)]:
        Xb = Xt[:b]; reps = 50 if b <= 100 else 3; best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter(); ch.predict_fast(Xb, ALPHA, Gcal); best = min(best, time.perf_counter() - t0)
        rec["batch_scaling"][str(b)] = b / best
    rec["train_scaling"] = {}
    for frac in [0.1, 0.25, 0.5, 1.0]:
        n = int(frac * len(yf)); sub = rng.choice(len(yf), n, replace=False)
        t0 = time.time()
        a2 = models.AutoEncoder(seed=0).fit(Xf[sub][yf[sub] == 0])
        models.SplitCHIDS(seed=0, ano=a2).fit(Xf[sub], yf[sub], ff[sub], Xc[i1], yc[i1], fc[i1], Xc[i2], yc[i2])
        rec["train_scaling"][str(n)] = time.time() - t0
    R[dsname] = rec
    print(dsname, json.dumps(rec["fast_path"]), json.dumps(rec["fit"]), flush=True)
json.dump(R, open(OUT, "w"), indent=1)
print("done")
