"""Revision 3 (reviewer comments): additional analyses on the same splits as exp_v2.py.

Per dataset and seed (same D_fit / D_cal1 / D_cal2 / hold-out as the paper):
  repro      : CH-IDS test MCC recomputed in this environment (check against results/v2)
  matched    : conformal LightGBM, RF and AE calibrated on (i) the benign flows of D_cal2 only (same calibration size as
               CH-IDS), (ii) D_cal1 + D_cal2, (iii) the whole calibration partition (as in the paper's baselines)
  calsize    : calibration-set size sensitivity (n = 100 ... all of D_cal2 benign, 20 random subsets per n) for CH-IDS
               and LightGBM at alpha = 0.01 and 0.001: realised hold-out FPR, and test FPR / DR
  oracle     : detection rate when every score is thresholded so that the realised FPR on the test benign flows is
               exactly 1% or 5% (budget-matched comparison that does not depend on calibration), plus ROC-AUC
  family     : per attack family on each test set: DR of CH-IDS and of matched LightGBM, flows caught only by one of
               them, and for flows caught only by CH-IDS whether a rule matched them
  drift      : drift monitor on unlabeled windows: KS test of the system p-values P(x) of flows with P(x) > tau
               against Uniform(tau, 1], on windows from the hold-out (no shift) and from the test sets (shift)
  contam     : re-calibration windows of benign flows contaminated with a fraction c of attack flows
  rules      : (seed 0) per-rule statistics and one true-positive and one false-positive example
usage: python exp_rev3.py DATASET SEED [SEED ...]"""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import sys, os, json, time, warnings
import numpy as np
from scipy.stats import kstest
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import data, models, metrics
from exp_v2 import split_cal

OUT = ROOT + "/results/rev3"
os.makedirs(OUT, exist_ok=True)
A = 0.01


def conf_alerts(score_cal_benign, score_x, alpha):
    """conformal decision for a score where larger = more attack-like (same integer rule as Eq. (7))."""
    return models.SplitCHIDS.alerts(-np.asarray(score_cal_benign), -np.asarray(score_x), alpha)


def rates(pred, y):
    m = metrics.binary_metrics(y, pred)
    return dict(FPR=float(m["FPR"]), DR=float(m["DR"]), MCC=float(m["MCC"]), F1=float(m["F1"]),
                fp=int(((pred == 1) & (y == 0)).sum()), m=int((y == 0).sum()))


def oracle_dr(score, y, target):
    """DR at the threshold that gives realised FPR <= target on these benign flows (score: larger = attack)."""
    b = np.sort(score[y == 0])[::-1]
    k = int(np.floor(target * len(b)))
    thr = b[k] if k < len(b) else -np.inf          # alert iff score > thr  -> at most k benign alerts
    pred = score > thr
    return float(pred[y == 1].mean()), float(pred[y == 0].mean())


def inverse(pre, name, thr):
    kind, feat = name.split("__", 1)
    if kind == "num":
        num = pre.named_transformers_["num"]
        sc = num.named_steps["sc"]
        i = list(pre.transformers_[0][2]).index(feat)
        v = thr * sc.scale_[i] + sc.mean_[i]
        return feat, float(np.sign(v) * np.expm1(abs(v)))
    return feat, None


def run(dsname, seed):
    fn = f"{OUT}/{dsname}_{seed}.json"
    if os.path.exists(fn):
        print("skip", fn); return
    t0 = time.time()
    rng = np.random.default_rng(10_000 + seed)
    ds = data.LOADERS[dsname](); P = data.prepare(ds, seed)
    Xf, yf, ff, Xc, yc, fc = P["Xf"], P["yf"], P["ff"], P["Xc"], P["yc"], P["fc"]
    i1, i2, iid = split_cal(P, seed)
    ch = models.SplitCHIDS(alpha=A, lam="auto", seed=seed).fit(Xf, yf, ff, Xc[i1], yc[i1], fc[i1], Xc[i2], yc[i2])
    Gcal = ch.G(ch.comp2)
    R = dict(dataset=dsname, seed=seed, lam=ch.lam_, repro={}, matched={}, calsize={}, oracle={}, family={},
             drift={}, contam={}, rules=None)

    # ---------------- models compared under matched calibration
    lgbm = ch.sup                                   # identical to the LightGBM baseline (same data, settings, seed)
    rf = models.make_baseline("RF", seed).fit(Xf, yf)
    ae = ch.ano
    b2 = Xc[i2][yc[i2] == 0]; b12 = Xc[np.r_[i1, i2]][yc[np.r_[i1, i2]] == 0]; ball = Xc[yc == 0]
    cal_sets = {"Dcal2": b2, "Dcal1+Dcal2": b12, "whole partition": ball}
    scorers = {"LightGBM": lambda X: lgbm.predict_proba(X)[:, 1], "RF": lambda X: rf.predict_proba(X)[:, 1],
               "AE": lambda X: ae.score(X)}
    calscores = {(m, c): f(Xs) for m, f in scorers.items() for c, Xs in cal_sets.items()}

    evalsets = {"hold-out": (Xc[iid], yc[iid], fc[iid])}
    for k, (Xt, yt, ft, _) in P["tests"].items():
        evalsets[k] = (Xt, yt, ft)
    stored = None
    try:
        stored = json.load(open(f"{ROOT}/results/v2/{dsname}_{seed}.json"))["results"]
    except Exception:
        pass
    comp_cache, P_cache = {}, {}
    for en, (Xe, ye, fe) in evalsets.items():
        c = ch.components(Xe); comp_cache[en] = c
        Ge = ch.G(c)
        pred_ch = ch.alerts(Gcal, Ge, A)
        P_cache[en] = ch.system_pvalues(Gcal, Ge)
        R["matched"].setdefault(en, {})["CH-IDS (Dcal2)"] = rates(pred_ch, ye)
        if en != "hold-out" and stored and en in stored:
            R["repro"][en] = dict(now=float(metrics.binary_metrics(ye, pred_ch)["MCC"]),
                                  stored=float(stored[en]["CH-IDS (proposed)"]["MCC"]))
        sc = {m: f(Xe) for m, f in scorers.items()}
        for (m, cn), s_cal in calscores.items():
            if en == "hold-out" and cn == "whole partition":
                continue                             # the hold-out is part of the whole partition
            R["matched"][en][f"{m} ({cn})"] = rates(conf_alerts(s_cal, sc[m], A), ye)
        # budget-matched (oracle) comparison and AUC
        allsc = {"CH-IDS": -Ge, **sc}
        R["oracle"][en] = {m: dict(AUC=float(roc_auc_score(ye, s)),
                                   DR_at_1=oracle_dr(s, ye, 0.01)[0], DR_at_5=oracle_dr(s, ye, 0.05)[0])
                           for m, s in allsc.items()}
        # per-family complementarity (nominal alpha, both calibrated on D_cal2)
        pl = conf_alerts(calscores[("LightGBM", "Dcal2")], sc["LightGBM"], A)
        fam = {}
        for f in np.unique(fe[ye == 1]):
            mk = (fe == f) & (ye == 1)
            only_ch = mk & pred_ch & ~pl
            fam[str(f)] = dict(n=int(mk.sum()), DR_ch=float(pred_ch[mk].mean()), DR_lgb=float(pl[mk].mean()),
                               only_ch=int(only_ch.sum()), only_lgb=int((mk & pl & ~pred_ch).sum()),
                               only_ch_rule=int((only_ch & c["r"]).sum()))
        bm = ye == 0
        fam["_benign"] = dict(n=int(bm.sum()), FPR_ch=float(pred_ch[bm].mean()), FPR_lgb=float(pl[bm].mean()),
                              only_ch=int((bm & pred_ch & ~pl).sum()), only_lgb=int((bm & pl & ~pred_ch).sum()))
        R["family"][en] = fam

    # ---------------- calibration-set size sensitivity (subsets of the benign flows of D_cal2)
    sl_cal = scorers["LightGBM"](b2)
    hold = evalsets["hold-out"]; tname = list(P["tests"])[0]; tst = evalsets[tname]
    G_hold, G_test = ch.G(comp_cache["hold-out"]), ch.G(comp_cache[tname])
    s_hold, s_test = scorers["LightGBM"](hold[0]), scorers["LightGBM"](tst[0])
    for n in [100, 250, 500, 1000, 2000, len(b2)]:
        if n > len(b2):
            continue
        rec = {}
        for alpha in [0.01, 0.001]:
            out = {"CH-IDS": [], "LightGBM": []}
            for r in range(20 if n < len(b2) else 1):
                sub = rng.choice(len(b2), n, replace=False) if n < len(b2) else np.arange(len(b2))
                for m, (gc, gh, gt) in {"CH-IDS": (Gcal[sub], G_hold, G_test),
                                        "LightGBM": (-sl_cal[sub], -s_hold, -s_test)}.items():
                    ph = ch.alerts(gc, gh, alpha); pt = ch.alerts(gc, gt, alpha)
                    out[m].append([float(ph[hold[1] == 0].mean()), float(pt[tst[1] == 0].mean()),
                                   float(pt[tst[1] == 1].mean())])
            rec[str(alpha)] = {m: np.array(v).tolist() for m, v in out.items()}
        R["calsize"][str(n)] = rec
    R["calsize_test"] = tname

    # ---------------- drift monitor on unlabeled windows (hold-out = no shift; tests = shift)
    for en, Pv in P_cache.items():
        ye = evalsets[en][1]
        res = {}
        for w in [500, 2000]:
            if w > len(Pv):
                continue
            for tau in [0.1, 0.5]:
                rej = []
                for r in range(200):
                    idx = rng.choice(len(Pv), w, replace=False)
                    p = Pv[idx]; p = p[p > tau]
                    if len(p) < 20:
                        rej.append(np.nan); continue
                    u = (p - tau) / (1 - tau)
                    rej.append(float(kstest(u, "uniform").pvalue < 0.01))
                res[f"w{w}_tau{tau}"] = float(np.nanmean(rej))
        res["attack_share"] = float(ye.mean())
        R["drift"][en] = res
    # benign-only windows of the hold-out with the same test (sanity check of the null)
    Pb = P_cache["hold-out"][hold[1] == 0]
    res = {}
    for tau in [0.1, 0.5]:
        rej = []
        for r in range(200):
            p = Pb[rng.choice(len(Pb), min(2000, len(Pb)), replace=False)]; p = p[p > tau]
            rej.append(float(kstest((p - tau) / (1 - tau), "uniform").pvalue < 0.01))
        res[f"w2000_tau{tau}"] = float(np.mean(rej))
    R["drift"]["hold-out (benign only)"] = res

    # ---------------- contaminated re-calibration windows
    targets = [k for k in P["tests"] if k != "KDDTest-21"] + ["hold-out"]
    for en in targets:
        Xe, ye, fe = evalsets[en]
        Ge = ch.G(comp_cache[en])
        if en == "hold-out":                          # no shift: window from D_cal2 (benign + attacks), evaluate on hold-out
            Gpool_b, Gpool_a = Gcal, ch.G(ch.components(Xc[i2][yc[i2] == 1]))
            Gev_b, Gev_a = Ge[ye == 0], Ge[ye == 1]
        else:
            ib, ia = np.where(ye == 0)[0], np.where(ye == 1)[0]
            pb = rng.permutation(ib); pa = rng.permutation(ia)
            Gpool_b, Gev_b = Ge[pb[:len(pb) // 2]], Ge[pb[len(pb) // 2:]]
            Gpool_a, Gev_a = Ge[pa[:len(pa) // 2]], Ge[pa[len(pa) // 2:]]
        res = {}
        for n in [1000]:
            if n > len(Gpool_b):
                continue
            for cfrac in [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]:
                na = int(round(cfrac * n)); f_, d_ = [], []
                for r in range(20):
                    win = np.r_[Gpool_b[rng.choice(len(Gpool_b), n - na, replace=False)],
                                Gpool_a[rng.choice(len(Gpool_a), na, replace=False)]]
                    f_.append(float(ch.alerts(win, Gev_b, A).mean())); d_.append(float(ch.alerts(win, Gev_a, A).mean()))
                res[f"n{n}_c{cfrac}"] = dict(FPR=f_, DR=d_)
        R["contam"][en] = res

    # ---------------- rule statistics and examples (seed 0)
    if seed == 0:
        pre = P["pre"]; names = P["names"]
        t = ch.rl.tree.tree_
        leaves_f = ch.rl.tree.apply(Xf); leaves_2 = ch.rl.tree.apply(Xc[i2])
        Xt, yt, ft, _ = P["tests"][tname]; leaves_t = ch.rl.tree.apply(Xt)
        rules = []
        for conds, sup in ch.rl.rules(names):
            # find leaf id by walking the tree
            node = 0
            for (nm, op, th) in conds:
                node = t.children_left[node] if op == "<=" else t.children_right[node]
            readable = []
            for nm, op, th in conds:
                f, v = inverse(pre, nm, th)
                readable.append(f"{f} {'<=' if op == '<=' else '>'} {v:.4g}" if v is not None else
                                f"{nm.split('__', 1)[1]} {'= 0' if op == '<=' else '= 1'}")
            mf = leaves_f == node
            rules.append(dict(conds=readable, support=int(mf.sum()), fit_precision=float(yf[mf].mean()),
                              cal2_benign=int(((leaves_2 == node) & (yc[i2] == 0)).sum()),
                              test_benign=int(((leaves_t == node) & (yt == 0)).sum()),
                              test_attack=int(((leaves_t == node) & (yt == 1)).sum()),
                              test_attack_families={str(k): int(v) for k, v in zip(*np.unique(ft[(leaves_t == node) & (yt == 1)], return_counts=True))}))
        # examples: a true positive and a false positive of the rule tier on the test set, raw feature values
        Xraw_t = ds["tests"][tname][0].reset_index(drop=True)
        rmatch = ch.rl.match(Xt)
        ex = {}
        for kind, mask in [("true_positive", rmatch & (yt == 1)), ("false_positive", rmatch & (yt == 0))]:
            idx = np.where(mask)[0]
            if len(idx):
                j = int(idx[0]); leaf = int(leaves_t[j])
                used = sorted({nm.split("__", 1)[1] for conds, s in ch.rl.rules(names) for nm, _, _ in conds})
                cols = [c for c in Xraw_t.columns if c in used or any(u.startswith(c + "_") for u in used)]
                ex[kind] = dict(index=j, family=str(ft[j]), leaf=leaf,
                                raw={c: (str(Xraw_t.iloc[j][c])) for c in cols},
                                lgb_score=float(lgbm.predict_proba(Xt[j:j + 1])[0, 1]),
                                ae_score=float(ae.score(Xt[j:j + 1])[0]))
        # benign false matches inside D_cal2 (seed 0) for the same purpose
        Xraw_c = ds["train"][0].iloc[P["ical"]].reset_index(drop=True)
        rm2 = ch.rl.match(Xc[i2]) & (yc[i2] == 0)
        if rm2.any():
            j = int(np.where(rm2)[0][0]); jj = int(i2[j])
            used = sorted({nm.split("__", 1)[1] for conds, s in ch.rl.rules(names) for nm, _, _ in conds})
            cols = [c for c in Xraw_c.columns if c in used or any(u.startswith(c + "_") for u in used)]
            ex["false_positive_cal2"] = dict(raw={c: str(Xraw_c.iloc[jj][c]) for c in cols}, leaf=int(leaves_2[j]))
        R["rules"] = dict(test=tname, rules=rules, examples=ex)
    R["time"] = time.time() - t0
    json.dump(R, open(fn, "w"))
    print(dsname, seed, "done", round(R["time"]), "s", json.dumps(R["repro"]), flush=True)


if __name__ == "__main__":
    for s in sys.argv[2:]:
        run(sys.argv[1], int(s))
