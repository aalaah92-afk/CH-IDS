"""Revision 3: duplicates and split robustness.

  dupstats DATASET           exact and near-duplicate overlap between D_fit, D_cal2, the hold-out and the test set
                             (seed-0 split, model input features)
  dedup DATASET SEED...      CH-IDS and matched LightGBM after removing exact duplicates (training partition for
                             NSL-KDD and UNSW-NB15, whole release for TON_IoT); vectors with conflicting labels removed
  host SEED...               TON_IoT, deduplicated, with benign test flows from hosts (src_ip) never seen in training
                             or calibration (host-disjoint benign split); attacks split at random as before
"""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import sys, os, json, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")
import data, models, metrics
from exp_v2 import split_cal

OUT = ROOT + "/results/rev3_split"
os.makedirs(OUT, exist_ok=True)
ALPHAS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
TON_DROP = ["src_ip", "dst_ip", "src_port", "dst_port", "dns_query", "ssl_subject", "ssl_issuer",
            "http_uri", "http_user_agent", "weird_name", "weird_addl", "label", "type"]


def ton_frame():
    df = pd.read_csv(data.D + "train_test_network.csv", low_memory=False)
    df.columns = [c.replace("﻿", "") for c in df.columns]
    fam = df["type"].str.strip().str.capitalize().replace({"Ddos": "DDoS", "Dos": "DoS", "Xss": "XSS", "Mitm": "MITM"}).values
    y = df["label"].values.astype(int)
    X = df.drop(columns=TON_DROP)
    cat = [c for c in X.columns if X[c].dtype == object]
    for c in cat:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cat]
    X[num] = X[num].apply(pd.to_numeric, errors="coerce").fillna(0)   # identical to data.load_ton
    return df, X, y, fam, num, cat


def keys(X):
    return pd.util.hash_pandas_object(X.reset_index(drop=True), index=False).values


def dedup_mask(X, y):
    """keep the first occurrence of every feature vector; drop vectors that occur with both labels."""
    k = keys(X)
    s = pd.DataFrame({"k": k, "y": y})
    conflict = s.groupby("k")["y"].transform("nunique") > 1
    first = ~s.duplicated("k")
    return (first & ~conflict).values, int(conflict.sum())


def load_dedup(dsname):
    if dsname == "TON_IoT":
        df, X, y, fam, num, cat = ton_frame()
        keep, nconf = dedup_mask(X[num + cat], y)
        X, y, fam = X[keep].reset_index(drop=True), y[keep], fam[keep]
        itr, ite = train_test_split(np.arange(len(y)), test_size=0.15, stratify=fam, random_state=42)
        mk = lambda ii: (X.iloc[ii].reset_index(drop=True), y[ii], fam[ii], fam[ii])
        return dict(name="TON_IoT", train=mk(itr), tests={"TON_IoT test": mk(ite)}, num=num, cat=cat,
                    info=dict(n_before=int(len(keep)), n_after=int(keep.sum()), conflicting=nconf))
    ds = data.LOADERS[dsname]()
    X, y, fam, sub = ds["train"]
    keep, nconf = dedup_mask(X, y)
    trk = set(keys(X[keep]))
    ds["train"] = (X[keep].reset_index(drop=True), y[keep], fam[keep], sub[keep])
    tests = {}
    for k, (Xt, yt, ft, st) in ds["tests"].items():
        tests[k] = (Xt, yt, ft, st)
        new = ~np.isin(keys(Xt), list(trk))            # test flows whose vector does not occur in training
        tests[k + " (unseen vectors)"] = (Xt[new].reset_index(drop=True), yt[new], ft[new], st[new])
    ds["tests"] = tests
    ds["info"] = dict(n_before=int(len(keep)), n_after=int(keep.sum()), conflicting=nconf)
    return ds


def load_host(seed):
    df, X, y, fam, num, cat = ton_frame()
    keep, _ = dedup_mask(X[num + cat], y)
    df, X, y, fam = df[keep].reset_index(drop=True), X[keep].reset_index(drop=True), y[keep], fam[keep]
    rng = np.random.default_rng(500 + seed)
    ben = np.where(y == 0)[0]
    hosts = df["src_ip"].values
    hb = pd.Series(hosts[ben]).value_counts()
    order = rng.permutation(hb.index.values)
    test_hosts, tot = [], 0
    for h in order:                                  # add hosts until ~15% of benign flows are in the test set
        if tot >= 0.15 * len(ben):
            break
        if tot + hb[h] > 0.30 * len(ben):            # skip hosts that would make the test set very large
            continue
        test_hosts.append(h); tot += hb[h]
    bte = ben[np.isin(hosts[ben], test_hosts)]; btr = ben[~np.isin(hosts[ben], test_hosts)]
    att = np.where(y == 1)[0]
    atr, ate = train_test_split(att, test_size=0.15, stratify=fam[att], random_state=seed)
    itr, ite = np.r_[btr, atr], np.r_[bte, ate]
    mk = lambda ii: (X.iloc[ii].reset_index(drop=True), y[ii], fam[ii], fam[ii])
    return dict(name="TON_IoT", train=mk(itr), tests={"TON_IoT test (unseen hosts)": mk(ite)}, num=num, cat=cat,
                info=dict(test_hosts=[str(h) for h in test_hosts], n_test_benign=int(len(bte)),
                          n_train_benign=int(len(btr)), n_benign_hosts=int(len(hb))))


def evaluate(ds, seed, tag):
    P = data.prepare(ds, seed)
    Xf, yf, ff, Xc, yc, fc = P["Xf"], P["yf"], P["ff"], P["Xc"], P["yc"], P["fc"]
    i1, i2, iid = split_cal(P, seed)
    ch = models.SplitCHIDS(alpha=0.01, lam="auto", seed=seed).fit(Xf, yf, ff, Xc[i1], yc[i1], fc[i1], Xc[i2], yc[i2])
    Gcal = ch.G(ch.comp2)
    s2 = ch.sup.predict_proba(Xc[i2][yc[i2] == 0])[:, 1]
    R = dict(tag=tag, seed=seed, lam=ch.lam_, n2=int(len(Gcal)), info=ds.get("info"), sets={})
    evs = {"hold-out": (Xc[iid], yc[iid])}
    for k, (Xt, yt, ft, st) in P["tests"].items():
        evs[k] = (Xt, yt)
    for en, (Xe, ye) in evs.items():
        if len(ye) == 0:
            continue
        Ge = ch.G(ch.components(Xe)); se = ch.sup.predict_proba(Xe)[:, 1]
        rec = {"m_benign": int((ye == 0).sum()), "n_attack": int((ye == 1).sum())}
        for a in ALPHAS:
            pc = ch.alerts(Gcal, Ge, a); pl = models.SplitCHIDS.alerts(-s2, -se, a)
            rec[str(a)] = {"CH-IDS": [int(pc[ye == 0].sum()), float(pc[ye == 1].mean()) if (ye == 1).any() else None],
                           "LightGBM (Dcal2)": [int(pl[ye == 0].sum()), float(pl[ye == 1].mean()) if (ye == 1).any() else None]}
        if (ye == 1).any() and (ye == 0).any():
            pc = ch.alerts(Gcal, Ge, 0.01); pl = models.SplitCHIDS.alerts(-s2, -se, 0.01)
            rec["metrics"] = {"CH-IDS": metrics.binary_metrics(ye, pc), "LightGBM (Dcal2)": metrics.binary_metrics(ye, pl)}
            rec["metrics"] = {m: {k: float(v) for k, v in d.items()} for m, d in rec["metrics"].items()}
        R["sets"][en] = rec
    return R


def dupstats(dsname):
    ds = data.LOADERS[dsname]() if dsname != "TON_IoT" else None
    if dsname == "TON_IoT":
        df, X, y, fam, num, cat = ton_frame()
        itr, ite = train_test_split(np.arange(len(y)), test_size=0.15, stratify=fam, random_state=42)
        Xtr, ytr, ftr = X.iloc[itr].reset_index(drop=True), y[itr], fam[itr]
        tests = {"TON_IoT test": (X.iloc[ite].reset_index(drop=True), y[ite])}
        dsd = dict(name="TON_IoT", train=(Xtr, ytr, ftr, ftr), tests={"TON_IoT test": (tests["TON_IoT test"][0], y[ite], fam[ite], fam[ite])}, num=num, cat=cat)
    else:
        Xtr, ytr, ftr, _ = ds["train"]
        tests = {k: (v[0], v[1]) for k, v in ds["tests"].items()}
        dsd = ds
    P = data.prepare(dsd, 0)
    i1, i2, iid = split_cal(P, 0)
    ktr = keys(Xtr)
    kfit = ktr[P["ifit"]]; kcal = ktr[P["ical"]]
    sfit = set(kfit)
    out = dict(n_train=int(len(ktr)), share_train_rows_with_duplicate=float(pd.Series(ktr).duplicated(keep=False).mean()),
               distinct_train_vectors=int(len(set(ktr))))
    s = pd.DataFrame({"k": ktr, "y": ytr})
    out["conflicting_label_vectors"] = int((s.groupby("k")["y"].nunique() > 1).sum())
    for nm, ii in [("Dcal1", i1), ("Dcal2", i2), ("hold-out", iid)]:
        kk = kcal[ii]; yy = P["yc"][ii]
        dup = np.isin(kk, list(sfit))
        out[f"{nm}_in_Dfit"] = dict(all=float(dup.mean()), benign=float(dup[yy == 0].mean()), attack=float(dup[yy == 1].mean()))
    out["hold-out_in_Dcal2"] = float(np.isin(kcal[iid], list(set(kcal[i2]))).mean())
    for k, (Xt, yt) in tests.items():
        dup = np.isin(keys(Xt), list(set(ktr)))
        out[f"{k}_in_training"] = dict(all=float(dup.mean()), benign=float(dup[yt == 0].mean()), attack=float(dup[yt == 1].mean()))
    # near duplicates in the model's input space (standardised), 2000 sampled query flows
    rng = np.random.default_rng(0)
    nn = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(P["Xf"])
    queries = {"hold-out": P["Xc"][iid]}
    for k, v in P["tests"].items():
        queries[k] = v[0]
    for nm, Xq in queries.items():
        sel = rng.choice(len(Xq), min(2000, len(Xq)), replace=False)
        d, _ = nn.kneighbors(Xq[sel])
        d = d[:, 0]
        out[f"nn_{nm}"] = dict(exact=float((d < 1e-6).mean()), lt_0_1=float((d < 0.1).mean()), lt_0_5=float((d < 0.5).mean()),
                               median=float(np.median(d)))
    json.dump(out, open(f"{OUT}/dup_{dsname}.json", "w"), indent=1)
    print(dsname, json.dumps(out)[:600], flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "dupstats":
        dupstats(sys.argv[2])
    elif cmd == "dedup":
        dsname = sys.argv[2]; ds = load_dedup(dsname)
        for s in sys.argv[3:]:
            fn = f"{OUT}/dedup_{dsname}_{s}.json"
            if os.path.exists(fn):
                continue
            t0 = time.time(); R = evaluate(ds, int(s), "dedup"); R["time"] = time.time() - t0
            json.dump(R, open(fn, "w")); print("dedup", dsname, s, round(R["time"]), flush=True)
    elif cmd == "host":
        for s in sys.argv[2:]:
            fn = f"{OUT}/host_TON_IoT_{s}.json"
            if os.path.exists(fn):
                continue
            ds = load_host(int(s)); t0 = time.time(); R = evaluate(ds, int(s), "host"); R["time"] = time.time() - t0
            json.dump(R, open(fn, "w")); print("host", s, round(R["time"]), json.dumps(ds["info"])[:300], flush=True)
