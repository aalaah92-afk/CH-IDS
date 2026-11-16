"""Revision experiments with the split-calibrated CH-IDS.

For each (dataset, seed):
  * the training partition is split (data.prepare, stratified by family) into D_fit (a fraction
    1 - 0.15/0.85 = 82.35%) and a calibration partition (17.65%); the calibration partition is split again
    into D_cal1 (30%), D_cal2 (30%) and an in-distribution hold-out D_id (40%), i.e. 5.29%, 5.29% and 7.06% of the
    training partition. D_id is never used for training or calibration;
  * SplitCHIDS and its ablations are fitted; results on the official test sets are produced in the same format as
    exp_standard.py (so they can replace the CH-IDS rows there);
  * target vs realised false-alarm rate is recorded for alpha in {0.001,...,0.1} on: the in-distribution hold-out,
    the official test sets (natural shift), and two controlled sub-population shifts (UDP-only benign traffic,
    heavy flows above the 90th percentile of benign byte counts in D_cal1);
  * re-calibration with windows of benign flows drawn from the shifted distribution.
usage: python exp_v2.py DATASET SEED [SEED ...]"""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import sys, os, json, time, warnings
import numpy as np
from scipy.stats import ks_2samp
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import data, models, metrics

ALPHAS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
OUT = ROOT + "/results/v2"
os.makedirs(OUT, exist_ok=True)
PROTO = {"NSL-KDD": "protocol_type", "UNSW-NB15": "proto", "TON_IoT": "proto"}
BYTES = {"NSL-KDD": "src_bytes", "UNSW-NB15": "sbytes", "TON_IoT": "src_bytes"}
WINDOWS = [100, 250, 500, 1000, 2000]


def split_cal(P, seed):
    idx = np.arange(len(P["yc"]))
    i12, iid = train_test_split(idx, test_size=0.4, stratify=P["fc"], random_state=1000 + seed)
    i1, i2 = train_test_split(i12, test_size=0.5, stratify=P["fc"][i12], random_state=2000 + seed)
    return i1, i2, iid


def per_family(fam, y, pred):
    return {str(f): [float(pred[fam == f].mean()), int((fam == f).sum())] for f in np.unique(fam[y == 1])}


def fpr_counts(pred, ben_mask):
    return int(pred[ben_mask].sum()), int(ben_mask.sum())


def run(dsname, seed):
    fn = f"{OUT}/{dsname}_{seed}.json"
    if os.path.exists(fn):
        print("skip", fn); return
    t_all = time.time()
    ds = data.LOADERS[dsname](); P = data.prepare(ds, seed)
    Xf, yf, ff, Xc, yc, fc = P["Xf"], P["yf"], P["ff"], P["Xc"], P["yc"], P["fc"]
    i1, i2, iid = split_cal(P, seed)
    Xraw_tr = ds["train"][0]
    raw_cal = Xraw_tr.iloc[P["ical"]].reset_index(drop=True)
    ch = models.SplitCHIDS(alpha=0.01, lam="auto", seed=seed).fit(Xf, yf, ff, Xc[i1], yc[i1], fc[i1], Xc[i2], yc[i2])
    n2 = int((yc[i2] == 0).sum())
    h2, _ = ch.rule_benign_hits()
    R = dict(dataset=dsname, seed=seed, lam=ch.lam_, inner_dr=ch.inner_dr, n_rules=int(len(ch.rl.keep_arr)),
             n_cal1_benign=int((yc[i1] == 0).sum()), n_cal2_benign=n2, rule_hits_cal2=h2,
             fit_time=ch.fit_time, results={}, sweep={}, shift={}, recal={}, benign_shift={},
             split=dict(n_train=int(len(yf) + len(yc)), n_train_benign=int((yf == 0).sum() + (yc == 0).sum()),
                        n_fit=int(len(yf)), n_fit_benign=int((yf == 0).sum()),
                        n_cal1=int(len(i1)), n_cal1_benign=int((yc[i1] == 0).sum()),
                        n_cal2=int(len(i2)), n_cal2_benign=int((yc[i2] == 0).sum()),
                        n_id=int(len(iid)), n_id_benign=int((yc[iid] == 0).sum())))
    variants = {"CH-IDS (proposed)": ("full", None), "CH-IDS w/o rule tier": ("no_rules", None),
                "CH-IDS w/o anomaly tier": ("no_ano", None), "CH-IDS w/o supervised tier": ("no_sup", None),
                "CH-IDS fixed lambda=0.5": ("full", 0.5), "Rule tier only": ("rules_only", None)}
    Gcal = {k: ch.G(ch.comp2, v, l) for k, (v, l) in variants.items()}
    G2, rs2, ra2 = Gcal["CH-IDS (proposed)"], ch.comp2["rs"], ch.comp2["ra"]

    def benign_diag(name, c, bm):
        """Benign-score shift between the calibration benign flows (D_cal2) and a set of benign flows."""
        if bm.sum() < 20:
            return
        Gb = ch.G(c, "full")[bm]
        R["benign_shift"][name] = dict(
            m=int(bm.sum()), lgb_gt05=float((c["rs"][bm] > 0.5).mean()), lgb_gt05_cal2=float((rs2 > 0.5).mean()),
            ks_G=float(ks_2samp(G2, Gb).statistic), ks_lgb=float(ks_2samp(rs2, c["rs"][bm]).statistic),
            ks_ae=float(ks_2samp(ra2, c["ra"][bm]).statistic), rule_rate=float(c["r"][bm].mean()),
            lgb_q=[float(x) for x in np.quantile(c["rs"][bm], [0.5, 0.9, 0.95, 0.99])],
            lgb_q_cal2=[float(x) for x in np.quantile(rs2, [0.5, 0.9, 0.95, 0.99])])

    # ---------------- official test sets (same format as exp_standard)
    comps = {}
    for tname, (Xt, yt, ft, st) in P["tests"].items():
        c = ch.components(Xt); comps[tname] = (c, yt, ft, st)
        res = {}
        nov = None
        if dsname == "NSL-KDD":
            nov = np.array([(a not in ds["train_attacks"]) and (a != "normal") for a in st])
        for k, (v, l) in variants.items():
            pred, z, _ = ch.decide(c, 0.01, v, l, Gcal[k])
            d = dict(op="conformal", **metrics.binary_metrics(yt, pred.astype(int), z), fam=per_family(ft, yt, pred.astype(int)))
            if nov is not None:
                d["novel_DR"] = float(pred[nov].mean()); d["known_DR"] = float(pred[(yt == 1) & ~nov].mean())
            res[k] = d
        R["results"][tname] = res
        R["sweep"][tname] = {str(a): dict(zip(["fp", "m"], fpr_counts(ch.decide(c, a, "full", None, Gcal["CH-IDS (proposed)"])[0], yt == 0)))
                             | dict(DR=float(ch.decide(c, a, "full", None, Gcal["CH-IDS (proposed)"])[0][yt == 1].mean())) for a in ALPHAS}
        R["results"][tname]["_rule_cov"] = float(c["r"].mean())
        R.setdefault("rule_benign", {})[tname] = [int(c["r"][yt == 0].sum()), int((yt == 0).sum())]
        benign_diag(tname, c, yt == 0)

    # ---------------- in-distribution hold-out and controlled shifts (benign flows only)
    Xid, yid = Xc[iid], yc[iid]
    raw_id = raw_cal.iloc[iid].reset_index(drop=True)
    if "raw_proto_train" in ds:
        raw_id[PROTO[dsname]] = ds["raw_proto_train"][P["ical"]][iid]
    cid = ch.components(Xid)
    ben = yid == 0
    raw_c1 = raw_cal.iloc[i1].reset_index(drop=True)
    heavy_thr = float(np.quantile(raw_c1[BYTES[dsname]][yc[i1] == 0].astype(float), 0.9))
    sets = {"in-distribution": ben,
            "UDP-only (protocol mix)": ben & (raw_id[PROTO[dsname]].astype(str).str.lower() == "udp").values,
            "heavy flows (bytes > P90)": ben & (raw_id[BYTES[dsname]].astype(float) > heavy_thr).values}
    Gfull = Gcal["CH-IDS (proposed)"]
    for sname, mask in sets.items():
        benign_diag(sname, cid, mask)
    for sname, mask in sets.items():
        R["shift"][sname] = {}
        for a in ALPHAS:
            pred = ch.decide(cid, a, "full", None, Gfull)[0]
            fp, m = fpr_counts(pred, mask)
            R["shift"][sname][str(a)] = dict(fp=fp, m=m)
    R["shift"]["_attack_DR_in_distribution"] = {str(a): float(ch.decide(cid, a, "full", None, Gfull)[0][yid == 1].mean()) for a in ALPHAS}
    R["heavy_thr"] = heavy_thr
    R.setdefault("rule_benign", {})["in-distribution"] = [int(cid["r"][ben].sum()), int(ben.sum())]

    # ---------------- re-calibration on windows of benign traffic from the shifted distribution
    rng = np.random.default_rng(seed)
    targets = []
    for tname, (c, yt, ft, st) in comps.items():
        if dsname != "TON_IoT":
            targets.append((tname, c, yt == 0, yt == 1))
    for sname in ["UDP-only (protocol mix)", "heavy flows (bytes > P90)"]:
        targets.append((sname, cid, sets[sname], None))
    for tname, c, bmask, amask in targets:
        bidx = np.where(bmask)[0]
        if len(bidx) < 220:
            continue
        perm = rng.permutation(bidx)
        pool, evalb = perm[: len(perm) // 2], perm[len(perm) // 2:]
        Gx = ch.G(c, "full")
        out = {}
        for w in WINDOWS:
            if w > len(pool):
                continue
            fprs, drs = [], []
            for rep in range(10):
                win = rng.choice(pool, w, replace=False)
                pred = ch.alerts(Gx[win], Gx, 0.01)
                fprs.append(float(pred[evalb].mean()))
                if amask is not None:
                    drs.append(float(pred[amask].mean()))
            out[str(w)] = dict(FPR=fprs, DR=drs, m_eval=int(len(evalb)))
        # reference: original calibration (no re-calibration) on the same evaluation half
        pred0 = ch.decide(c, 0.01, "full", None, Gfull)[0]
        out["none"] = dict(FPR=[float(pred0[evalb].mean())], DR=[float(pred0[amask].mean())] if amask is not None else [],
                           m_eval=int(len(evalb)))
        R["recal"][tname] = out
    R["total_time"] = time.time() - t_all
    json.dump(R, open(fn, "w"), default=float)
    print("done", fn, round(time.time() - t_all), "s", "lam", ch.lam_, "n2", n2, "h2", h2, flush=True)


if __name__ == "__main__":
    for sd in sys.argv[2:]:
        run(sys.argv[1], int(sd))
