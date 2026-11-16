"""Revision 3b: remedies for the four weaknesses found in revision 3.

  contam  : re-calibration windows (n = 1000) with a fraction c of attack flows, three threshold rules
            naive      k = floor(alpha (n+1))-th smallest G of the window (Eq. (7))
            labeltrim  analyst inspects window flows in increasing order of G and removes confirmed attacks
                       until k benign flows have been confirmed; threshold = k-th smallest benign G
            margin     no labels: rank k + ceil(cbar n) with cbar = 1% (tolerates up to cbar n attacks)
  drift   : two-sample Kolmogorov-Smirnov monitor (window G versus reference G), level 0.01
            mixed   : unlabeled windows versus the unlabeled reference pool (all flows of D_cal2)
            benign  : benign-only windows versus the benign flows of D_cal2
  host    : TON_IoT with benign test flows from unseen hosts
            standard  : calibration hosts = training hosts (as in revision 3)
            hostcal   : benign calibration flows from hosts held out from fitting (host-disjoint calibration)
            sensor    : standard model re-calibrated on n clean benign flows of the monitored (unseen) hosts
usage: python exp_fix.py DATASET SEED...  |  python exp_fix.py host SEED..."""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import sys, os, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
import data, models, exp_split
from exp_v2 import split_cal

OUT = ROOT + "/results/rev3_fix"
os.makedirs(OUT, exist_ok=True)
A = 0.01
CBAR = 0.01


def alerts_rank(Gw, Gx, j):
    """alert iff G(x) < j-th smallest value of Gw (same integer rule as models.SplitCHIDS.alerts)"""
    if j < 1:
        return np.zeros(len(Gx), bool)
    s = np.sort(Gw); le = np.searchsorted(s, Gx, "right")
    return (le + 1) <= j


def k_of(n, a=A):
    return int(np.floor(a * (n + 1) + 1e-12))


def labeltrim(Gw, yw):
    """inspect flows in increasing G; drop confirmed attacks until k benign flows are confirmed.
    returns cleaned window and number of flows inspected"""
    o = np.argsort(Gw, kind="stable"); n = len(Gw); att = 0; ben = 0; i = 0
    while i < n:
        if ben >= k_of(n - att):
            break
        if yw[o[i]] == 1:
            att += 1
        else:
            ben += 1
        i += 1
    drop = o[:i][yw[o[:i]] == 1]
    keep = np.ones(n, bool); keep[drop] = False
    return Gw[keep], i


def contam_block(rng, Gpool_b, Gpool_a, Gev_b, Gev_a, n=1000, reps=20):
    res = {}
    for c in [0.0, 0.005, 0.01, 0.02, 0.05]:
        na = int(round(c * n)); rec = {s: dict(FPR=[], DR=[]) for s in ["naive", "labeltrim", "margin"]}; lab = []
        for _ in range(reps):
            ib = rng.choice(len(Gpool_b), n - na, replace=False); ia = rng.choice(len(Gpool_a), na, replace=False)
            Gw = np.r_[Gpool_b[ib], Gpool_a[ia]]; yw = np.r_[np.zeros(n - na, int), np.ones(na, int)]
            k = k_of(n)
            for s, (W, j) in {"naive": (Gw, k), "margin": (Gw, k + int(np.ceil(CBAR * n)))}.items():
                rec[s]["FPR"].append(float(alerts_rank(W, Gev_b, j).mean())); rec[s]["DR"].append(float(alerts_rank(W, Gev_a, j).mean()))
            Wc, used = labeltrim(Gw, yw); lab.append(used)
            j = k_of(len(Wc))
            rec["labeltrim"]["FPR"].append(float(alerts_rank(Wc, Gev_b, j).mean())); rec["labeltrim"]["DR"].append(float(alerts_rank(Wc, Gev_a, j).mean()))
        rec["labeltrim"]["inspected"] = lab
        res[str(c)] = rec
    return res


def ks_rate(rng, Gsrc, Gref, w, reps=200):
    if len(Gsrc) < w:
        return None
    return float(np.mean([ks_2samp(Gsrc[rng.choice(len(Gsrc), w, replace=False)], Gref).pvalue < 0.01 for _ in range(reps)]))


def drift_block(rng, Gref_all, Gref_b, Ge, ye):
    out = {"attack_share": float(ye.mean())}
    for w in [250, 500, 1000]:
        out[f"mixed_w{w}"] = ks_rate(rng, Ge, Gref_all, w)
        out[f"benign_w{w}"] = ks_rate(rng, Ge[ye == 0], Gref_b, w)
    return out


def fit(P, seed):
    Xf, yf, ff, Xc, yc, fc = P["Xf"], P["yf"], P["ff"], P["Xc"], P["yc"], P["fc"]
    i1, i2, iid = split_cal(P, seed)
    ch = models.SplitCHIDS(alpha=A, lam="auto", seed=seed).fit(Xf, yf, ff, Xc[i1], yc[i1], fc[i1], Xc[i2], yc[i2])
    return ch, i1, i2, iid


def run_dataset(dsname, seed):
    fn = f"{OUT}/{dsname}_{seed}.json"
    if os.path.exists(fn):
        return
    rng = np.random.default_rng(4242 + seed)
    P = data.prepare(data.LOADERS[dsname](), seed)
    ch, i1, i2, iid = fit(P, seed)
    Xc, yc = P["Xc"], P["yc"]
    G2 = ch.G(ch.components(Xc[i2])); y2 = yc[i2]
    Gb2, Ga2 = G2[y2 == 0], G2[y2 == 1]
    R = dict(dataset=dsname, seed=seed, lam=ch.lam_, contam={}, drift={})
    sets = {"hold-out": (Xc[iid], yc[iid])}
    for k, v in P["tests"].items():
        sets[k] = (v[0], v[1])
    for en, (Xe, ye) in sets.items():
        Ge = ch.G(ch.components(Xe))
        R["drift"][en] = drift_block(rng, G2, Gb2, Ge, ye)
        if en == "KDDTest-21":
            continue
        if en == "hold-out":                       # no shift: window from D_cal2, evaluation on the hold-out
            R["contam"][en] = contam_block(rng, Gb2, Ga2, Ge[ye == 0], Ge[ye == 1])
        else:                                       # shift: window from one half of the test set, evaluation on the other
            pb = rng.permutation(np.where(ye == 0)[0]); pa = rng.permutation(np.where(ye == 1)[0])
            hb, ha = len(pb) // 2, len(pa) // 2
            R["contam"][en] = contam_block(rng, Ge[pb[:hb]], Ge[pa[:ha]], Ge[pb[hb:]], Ge[pa[ha:]])
    json.dump(R, open(fn, "w")); print(fn, flush=True)


# ------------------------------------------------------------------ TON_IoT, unseen hosts
def host_frames(seed):
    df, X, y, fam, num, cat = exp_split.ton_frame()
    keep, _ = exp_split.dedup_mask(X[num + cat], y)
    df, X, y, fam = df[keep].reset_index(drop=True), X[keep].reset_index(drop=True), y[keep], fam[keep]
    rng = np.random.default_rng(500 + seed)                    # identical to exp_split.load_host
    ben = np.where(y == 0)[0]; hosts = df["src_ip"].values
    hb = pd.Series(hosts[ben]).value_counts(); order = rng.permutation(hb.index.values)
    test_hosts, tot = [], 0
    for h in order:
        if tot >= 0.15 * len(ben):
            break
        if tot + hb[h] > 0.30 * len(ben):
            continue
        test_hosts.append(h); tot += hb[h]
    bte = ben[np.isin(hosts[ben], test_hosts)]; btr = ben[~np.isin(hosts[ben], test_hosts)]
    att = np.where(y == 1)[0]
    atr, ate = train_test_split(att, test_size=0.15, stratify=fam[att], random_state=seed)
    return df, X, y, fam, num, cat, hosts, test_hosts, bte, btr, atr, ate


def prepare_idx(X, y, fam, num, cat, ifit, ical, tests):
    pre = data.make_preprocessor(num, cat)
    Xf = pre.fit_transform(X.iloc[ifit]).astype(np.float32); Xc = pre.transform(X.iloc[ical]).astype(np.float32)
    T = {k: (pre.transform(X.iloc[ii]).astype(np.float32), y[ii], fam[ii], fam[ii]) for k, ii in tests.items()}
    return dict(Xf=Xf, yf=y[ifit], ff=fam[ifit], Xc=Xc, yc=y[ical], fc=fam[ical], tests=T, pre=pre)


def host_eval(ch, P, iid, Xt, yt, tag):
    Xc, yc = P["Xc"], P["yc"]
    G2b = ch.G(ch.comp2); s2b = ch.sup.predict_proba(Xc[ch_i2[tag]][yc[ch_i2[tag]] == 0])[:, 1]
    out = {}
    for en, (Xe, ye) in {"hold-out": (Xc[iid], yc[iid]), "unseen hosts": (Xt, yt)}.items():
        Ge = ch.G(ch.components(Xe)); se = ch.sup.predict_proba(Xe)[:, 1]
        rec = {}
        for a in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
            pc = ch.alerts(G2b, Ge, a); pl = models.SplitCHIDS.alerts(-s2b, -se, a)
            rec[str(a)] = {"CH-IDS": [float(pc[ye == 0].mean()), float(pc[ye == 1].mean()) if (ye == 1).any() else None],
                           "LightGBM": [float(pl[ye == 0].mean()), float(pl[ye == 1].mean()) if (ye == 1).any() else None]}
        out[en] = rec
    return out


ch_i2 = {}


def run_host(seed):
    fn = f"{OUT}/host_{seed}.json"
    if os.path.exists(fn):
        return
    df, X, y, fam, num, cat, hosts, test_hosts, bte, btr, atr, ate = host_frames(seed)
    ite = np.r_[bte, ate]
    R = dict(seed=seed, test_hosts=[str(h) for h in test_hosts], n_test_benign=int(len(bte)))
    rng = np.random.default_rng(9000 + seed)

    # (1) standard: random fit / calibration split over the training hosts (as in revision 3)
    itr = np.r_[btr, atr]
    ifit_l, ical_l = train_test_split(np.arange(len(itr)), test_size=0.15 / 0.85, stratify=fam[itr], random_state=seed)
    P = prepare_idx(X, y, fam, num, cat, itr[ifit_l], itr[ical_l], {"t": ite})
    ch, i1, i2, iid = fit(P, seed); ch_i2["standard"] = i2
    Xt, yt = P["tests"]["t"][0], P["tests"]["t"][1]
    R["standard"] = host_eval(ch, P, iid, Xt, yt, "standard"); R["standard"]["lam"] = ch.lam_

    # (3) per-sensor re-calibration of the standard model on clean benign flows of the unseen hosts
    Gt = ch.G(ch.components(Xt)); st = ch.sup.predict_proba(Xt)[:, 1]
    tb = np.where(yt == 0)[0]; ta = np.where(yt == 1)[0]
    sens = {}
    for n in [250, 500, 1000, 2000]:
        rec = {"CH-IDS": dict(FPR=[], DR=[]), "LightGBM": dict(FPR=[], DR=[])}
        for _ in range(20):
            p = rng.permutation(tb); w, ev = p[:n], p[n:]
            for m, sc in [("CH-IDS", Gt), ("LightGBM", -st)]:
                rec[m]["FPR"].append(float(models.SplitCHIDS.alerts(sc[w], sc[ev], A).mean()))
                rec[m]["DR"].append(float(models.SplitCHIDS.alerts(sc[w], sc[ta], A).mean()))
        sens[str(n)] = rec
    R["sensor"] = sens
    # (4) sensor adaptation: n clean benign flows of the monitored hosts; half is added to the fitting data, the other
    #     half replaces the benign calibration flows of D_cal2; lambda kept at the value chosen by the standard model
    Xc, yc, fc = P["Xc"], P["yc"], P["fc"]
    adapt = {}
    for n in [500, 1000, 2000]:
        rec = {"CH-IDS": dict(FPR=[], DR=[]), "LightGBM": dict(FPR=[], DR=[])}
        for r in range(3):
            p = rng.permutation(tb); w, ev = p[:n], p[n:]
            wf, wc = w[: n // 2], w[n // 2:]
            Xf2 = np.r_[P["Xf"], Xt[wf]]; yf2 = np.r_[P["yf"], yt[wf]]; blab = np.asarray(P["ff"], dtype=object)[P["yf"] == 0][0]; ff2 = np.r_[np.asarray(P["ff"], dtype=object), np.array([blab] * len(wf), dtype=object)]
            a2 = i2[yc[i2] == 1]
            X2 = np.r_[Xt[wc], Xc[a2]]; y2_ = np.r_[yt[wc], yc[a2]]
            m = models.SplitCHIDS(alpha=A, lam=ch.lam_, seed=seed).fit(Xf2, yf2, ff2, Xc[i1], yc[i1], fc[i1], X2, y2_)
            Gc = m.G(m.comp2); Ge = m.G(m.components(Xt[ev])); Ga = m.G(m.components(Xt[ta]))
            rec["CH-IDS"]["FPR"].append(float(m.alerts(Gc, Ge, A).mean())); rec["CH-IDS"]["DR"].append(float(m.alerts(Gc, Ga, A).mean()))
            sc = m.sup.predict_proba(Xt[wc])[:, 1]
            rec["LightGBM"]["FPR"].append(float(models.SplitCHIDS.alerts(-sc, -m.sup.predict_proba(Xt[ev])[:, 1], A).mean()))
            rec["LightGBM"]["DR"].append(float(models.SplitCHIDS.alerts(-sc, -m.sup.predict_proba(Xt[ta])[:, 1], A).mean()))
        adapt[str(n)] = rec
    R["adapt"] = adapt
    # drift monitor on unseen hosts (two-sample, reference D_cal2 of the standard model)
    G2 = ch.G(ch.components(P["Xc"][i2])); y2 = P["yc"][i2]
    R["drift_unseen_hosts"] = drift_block(rng, G2, G2[y2 == 0], Gt, yt)
    R["drift_holdout"] = drift_block(rng, G2, G2[y2 == 0], ch.G(ch.components(P["Xc"][iid])), P["yc"][iid])

    # (2) host-disjoint calibration: benign calibration flows from hosts that are not used for fitting
    hr = np.random.default_rng(700 + seed)
    hb = pd.Series(hosts[btr]).value_counts(); order = hr.permutation(hb.index.values)
    cal_hosts, tot = [], 0
    for h in order:
        if tot >= 0.1765 * len(btr):
            break
        if tot + hb[h] > 0.35 * len(btr):
            continue
        cal_hosts.append(h); tot += hb[h]
    bcal = btr[np.isin(hosts[btr], cal_hosts)]; bfit = btr[~np.isin(hosts[btr], cal_hosts)]
    afit, acal = train_test_split(atr, test_size=0.15 / 0.85, stratify=fam[atr], random_state=seed)
    P2 = prepare_idx(X, y, fam, num, cat, np.r_[bfit, afit], np.r_[bcal, acal], {"t": ite})
    ch2, j1, j2, jid = fit(P2, seed); ch_i2["hostcal"] = j2
    R["hostcal"] = host_eval(ch2, P2, jid, P2["tests"]["t"][0], P2["tests"]["t"][1], "hostcal")
    R["hostcal"]["lam"] = ch2.lam_; R["hostcal"]["cal_hosts"] = [str(h) for h in cal_hosts]
    R["hostcal"]["n_cal_benign"] = int(len(bcal)); R["hostcal"]["n_fit_benign"] = int(len(bfit))
    json.dump(R, open(fn, "w")); print(fn, flush=True)


if __name__ == "__main__" and sys.argv[1] != "host2":
    if sys.argv[1] == "host":
        for s in sys.argv[2:]:
            run_host(int(s))
    else:
        for s in sys.argv[2:]:
            run_dataset(sys.argv[1], int(s))


# ------------------------------------------------------------------ v2: refresh the per-tier references as well as the threshold
import copy


def refreshed(ch, Xb):
    """copy of a fitted CH-IDS whose p-value references include the benign flows Xb and whose rules are re-screened on Xb
    (rules that match any flow of Xb are dropped, as in the original screening on D_cal1)"""
    m = copy.copy(ch)
    m.ref_s = np.sort(np.r_[ch.ref_s, ch.sup.predict_proba(Xb)[:, 1]])
    m.ref_a = np.sort(np.r_[ch.ref_a, ch.ano.score(Xb)])
    rl = copy.copy(ch.rl)
    hit = set(np.unique(rl.tree.apply(Xb)).tolist())
    rl.keep = set(l for l in ch.rl.keep if l not in hit); rl.keep_arr = np.array(sorted(rl.keep))
    m.rl = rl
    return m


def run_host2(seed):
    fn = f"{OUT}/host2_{seed}.json"
    if os.path.exists(fn):
        return
    df, X, y, fam, num, cat, hosts, test_hosts, bte, btr, atr, ate = host_frames(seed)
    ite = np.r_[bte, ate]; itr = np.r_[btr, atr]
    ifit_l, ical_l = train_test_split(np.arange(len(itr)), test_size=0.15 / 0.85, stratify=fam[itr], random_state=seed)
    P = prepare_idx(X, y, fam, num, cat, itr[ifit_l], itr[ical_l], {"t": ite})
    ch, i1, i2, iid = fit(P, seed)
    Xt, yt = P["tests"]["t"][0], P["tests"]["t"][1]
    Xc, yc, fc = P["Xc"], P["yc"], P["fc"]
    tb = np.where(yt == 0)[0]; ta = np.where(yt == 1)[0]
    rng = np.random.default_rng(9100 + seed)
    R = dict(seed=seed, lam=ch.lam_, n_rules=int(len(ch.rl.keep_arr)), refresh={}, adapt2={})
    # (a) refresh: half of the window extends the per-tier references and re-screens the rules, half sets the threshold
    for n in [250, 500, 1000, 2000]:
        rec = dict(FPR=[], DR=[], rules_left=[])
        for _ in range(20):
            p = rng.permutation(tb); w, ev = p[:n], p[n:]; wr, wc = w[: n // 2], w[n // 2:]
            m = refreshed(ch, Xt[wr])
            Gc = m.G(m.components(Xt[wc]))
            rec["FPR"].append(float(m.alerts(Gc, m.G(m.components(Xt[ev])), A).mean()))
            rec["DR"].append(float(m.alerts(Gc, m.G(m.components(Xt[ta])), A).mean()))
            rec["rules_left"].append(int(len(m.rl.keep_arr)))
        R["refresh"][str(n)] = rec
    # (b) adaptation: 40% of the window joins D_fit, 30% joins D_cal1 (references and rule screening), 30% is D_cal2
    blab = np.asarray(P["ff"], dtype=object)[P["yf"] == 0][0]
    for n in [500, 1000, 2000]:
        rec = {"CH-IDS": dict(FPR=[], DR=[]), "LightGBM": dict(FPR=[], DR=[])}
        for _ in range(3):
            p = rng.permutation(tb); w, ev = p[:n], p[n:]
            a, b = int(0.4 * n), int(0.7 * n); wf, w1, w2 = w[:a], w[a:b], w[b:]
            Xf2 = np.r_[P["Xf"], Xt[wf]]; yf2 = np.r_[P["yf"], yt[wf]]
            ff2 = np.r_[np.asarray(P["ff"], dtype=object), np.array([blab] * len(wf), dtype=object)]
            X1 = np.r_[Xc[i1], Xt[w1]]; y1 = np.r_[yc[i1], yt[w1]]; f1 = np.r_[np.asarray(fc[i1], dtype=object), np.array([blab] * len(w1), dtype=object)]
            a2 = i2[yc[i2] == 1]
            X2 = np.r_[Xt[w2], Xc[a2]]; y2_ = np.r_[yt[w2], yc[a2]]
            m = models.SplitCHIDS(alpha=A, lam=ch.lam_, seed=seed).fit(Xf2, yf2, ff2, X1, y1, f1, X2, y2_)
            Gc = m.G(m.comp2)
            rec["CH-IDS"]["FPR"].append(float(m.alerts(Gc, m.G(m.components(Xt[ev])), A).mean()))
            rec["CH-IDS"]["DR"].append(float(m.alerts(Gc, m.G(m.components(Xt[ta])), A).mean()))
            sc = m.sup.predict_proba(Xt[w2])[:, 1]
            rec["LightGBM"]["FPR"].append(float(models.SplitCHIDS.alerts(-sc, -m.sup.predict_proba(Xt[ev])[:, 1], A).mean()))
            rec["LightGBM"]["DR"].append(float(models.SplitCHIDS.alerts(-sc, -m.sup.predict_proba(Xt[ta])[:, 1], A).mean()))
        R["adapt2"][str(n)] = rec
    json.dump(R, open(fn, "w")); print(fn, flush=True)


if __name__ == "__main__" and sys.argv[1] == "host2":
    for s in sys.argv[2:]:
        run_host2(int(s))
