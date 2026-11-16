"""Aggregate all experiment outputs into tables (JSON) and figures for the manuscript."""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, glob, os, sys
import numpy as np
from scipy import stats

RES = ROOT + "/results"
FIG = ROOT + "/figs"
TESTS = [("NSL-KDD", "KDDTest+"), ("NSL-KDD", "KDDTest-21"), ("UNSW-NB15", "UNSW-NB15 test"), ("TON_IoT", "TON_IoT test")]
TEST_SHORT = {"KDDTest+": "KDDTest+", "KDDTest-21": "KDDTest-21", "UNSW-NB15 test": "UNSW-NB15", "TON_IoT test": "TON_IoT"}

PROPOSED = "CH-IDS (proposed)"
MAIN_METHODS = [
    ("LR [default]", "LR"), ("DT [default]", "DT"), ("RF [default]", "RF"), ("SVM-RBF [default]", "SVM-RBF"),
    ("MLP [default]", "MLP"), ("XGBoost [default]", "XGBoost"), ("LightGBM [default]", "LightGBM"),
    ("IForest [default]", "IForest"), ("AE [default]", "AE"),
    ("Seq. hybrid: rules->RF+SMOTE [default]", "Seq. hybrid {wajs2025}"),
    ("RF [conformal]", "RF"), ("XGBoost [conformal]", "XGBoost"), ("LightGBM [conformal]", "LightGBM"),
    ("AE [conformal]", "AE"), ("Seq. hybrid: rules->RF+SMOTE [conformal]", "Seq. hybrid {wajs2025}"),
    (PROPOSED, "CH-IDS (proposed)"),
]
ABL = [PROPOSED, "CH-IDS w/o rule tier", "CH-IDS w/o anomaly tier", "CH-IDS w/o supervised tier",
       "CH-IDS fixed lambda=0.5", "Rule tier only"]


def load_standard():
    S = {}
    for fn in sorted(glob.glob(f"{RES}/{os.environ.get('STD_DIR', 'standard')}/*.json")):
        R = json.load(open(fn))
        S.setdefault(R["dataset"], []).append(R)
    return S


def load_loafo():
    L = {}
    for fn in sorted(glob.glob(f"{RES}/{os.environ.get('LO_DIR', 'loafo')}/*.json")):
        R = json.load(open(fn))
        L.setdefault(R["dataset"], []).append(R)
    return L


def agg(vals):
    v = np.array(vals, float)
    return float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0, len(v)


def standard_tables(S):
    out = {}
    for ds, t in TESTS:
        runs = S.get(ds, [])
        if not runs:
            continue
        d = {}
        for m in set(k for r in runs for k in r["results"][t]):
            d[m] = {}
            for met in ["Acc", "Prec", "DR", "FPR", "F1", "MCC", "ROC_AUC", "PR_AUC", "novel_DR", "known_DR"]:
                vals = [r["results"][t][m].get(met, np.nan) for r in runs if m in r["results"][t]]
                d[m][met] = agg(vals)
            fam = {}
            for r in runs:
                for f, (dr, n) in r["results"][t][m]["fam"].items():
                    fam.setdefault(f, []).append(dr)
            d[m]["fam"] = {f: agg(v) for f, v in fam.items()}
        out[t] = d
    return out


def holm(pvals):
    p = np.array(pvals); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def wilcoxon_block(pairs_a, pairs_b):
    a, b = np.array(pairs_a), np.array(pairs_b)
    diff = a - b
    if np.allclose(diff, 0):
        return 1.0, 0.0, (0, len(a), 0)
    st = stats.wilcoxon(a, b, zero_method="pratt", alternative="two-sided")
    w = int((diff > 1e-9).sum()); l = int((diff < -1e-9).sum()); tie = len(diff) - w - l
    return float(st.pvalue), float(np.median(diff)), (w, tie, l)


def standard_stats(S, metric="MCC"):
    comps = [m for m, _ in MAIN_METHODS if m != PROPOSED]
    res = {}
    for m in comps:
        a, b = [], []
        for ds, t in TESTS:
            for r in S.get(ds, []):
                rr = r["results"][t]
                a.append(rr[PROPOSED][metric]); b.append(rr[m][metric])
        res[m] = wilcoxon_block(a, b) + (len(a),)
    adj = holm([v[0] for v in res.values()])
    return {m: dict(p=v[0], p_holm=float(pa), med_diff=v[1], wtl=v[2], n=v[3]) for (m, v), pa in zip(res.items(), adj)}


LO_METHODS = ["CH-IDS (proposed)", "GBDT [conformal]", "RF [conformal]", "RF [default]",
              "Seq. hybrid: rules->RF+SMOTE [default]", "AE [conformal]", "IForest [conformal]",
              "CH-IDS lambda=0.0", "CH-IDS lambda=0.25", "CH-IDS lambda=0.5", "Rule tier only"]


def loafo_tables(L):
    out = {}
    for ds, runs in L.items():
        fams = sorted(runs[0]["families"])
        d = {"families": {}, "summary": {}, "lam": {}}
        for f in fams:
            d["families"][f] = {}
            d["lam"][f] = [r["families"][f]["lam"] for r in runs if f in r["families"]]
            for m in LO_METHODS:
                vals = [r["families"][f]["results"][m] for r in runs if f in r["families"] and m in r["families"][f]["results"]]
                if vals:
                    d["families"][f][m] = {k: agg([v[k] for v in vals]) for k in ["DR", "FPR", "AUC"]}
        for m in LO_METHODS:
            per_seed_mean, per_seed_min, fprs, aucs = [], [], [], []
            for r in runs:
                drs = [r["families"][f]["results"][m]["DR"] for f in fams if f in r["families"] and m in r["families"][f]["results"]]
                if not drs:
                    continue
                per_seed_mean.append(np.mean(drs)); per_seed_min.append(np.min(drs))
                fprs.append(np.mean([r["families"][f]["results"][m]["FPR"] for f in fams if f in r["families"]]))
                aucs.append(np.mean([r["families"][f]["results"][m]["AUC"] for f in fams if f in r["families"]]))
            if per_seed_mean:
                d["summary"][m] = dict(meanDR=agg(per_seed_mean), minDR=agg(per_seed_min), FPR=agg(fprs), AUC=agg(aucs))
        out[ds] = d
    return out


def loafo_stats(L, metric="DR"):
    comps = [m for m in LO_METHODS if m != PROPOSED and m != "Rule tier only"]
    res = {}
    for m in comps:
        a, b = [], []
        for ds, runs in L.items():
            for r in runs:
                for f, v in r["families"].items():
                    if m in v["results"]:
                        a.append(v["results"][PROPOSED][metric]); b.append(v["results"][m][metric])
        res[m] = wilcoxon_block(a, b) + (len(a),)
    adj = holm([v[0] for v in res.values()])
    return {m: dict(p=v[0], p_holm=float(pa), med_diff=v[1], wtl=v[2], n=v[3]) for (m, v), pa in zip(res.items(), adj)}


def friedman_ranks(L, methods, metric="AUC"):
    rows = []
    for ds, runs in L.items():
        for r in runs:
            for f, v in r["families"].items():
                if all(m in v["results"] for m in methods):
                    rows.append([v["results"][m][metric] for m in methods])
    X = np.array(rows)
    ranks = np.array([stats.rankdata(-x) for x in X])
    avg = ranks.mean(0)
    chi = stats.friedmanchisquare(*X.T)
    k, N = len(methods), len(X)
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}[k]
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    return dict(methods=methods, avg_rank=avg.tolist(), chi2=float(chi.statistic), p=float(chi.pvalue), N=N, CD=float(cd))


if __name__ == "__main__":
    S, L = load_standard(), load_loafo()
    out = dict(standard=standard_tables(S), loafo=loafo_tables(L) if L else {},
               n_seeds={k: len(v) for k, v in S.items()}, n_seeds_loafo={k: len(v) for k, v in L.items()})
    if S:
        out["stats_standard_MCC"] = standard_stats(S, "MCC")
        out["stats_standard_F1"] = standard_stats(S, "F1")
    if L:
        out["stats_loafo_DR"] = loafo_stats(L, "DR")
        out["stats_loafo_AUC"] = loafo_stats(L, "AUC")
        fm = ["CH-IDS (proposed)", "GBDT [conformal]", "RF [conformal]", "Seq. hybrid: rules->RF+SMOTE [default]",
              "AE [conformal]", "IForest [conformal]"]
        out["friedman_AUC"] = friedman_ranks(L, fm, "AUC")
        out["friedman_DR"] = friedman_ranks(L, fm, "DR")
    # extras: lambda, rules, coverage
    ex = {}
    for ds, runs in S.items():
        ex[ds] = dict(lam=[r["extra"]["lambda"] for r in runs], n_rules=[r["extra"]["n_rules"] for r in runs],
                      cov={k: agg([r["extra"][k] for r in runs]) for k in runs[0]["extra"] if k.startswith("rule_coverage")},
                      prec={k: agg([r["extra"][k] for r in runs if r["extra"][k] is not None]) for k in runs[0]["extra"] if k.startswith("rule_precision")},
                      sweep={k: runs[0]["extra"][k] for k in runs[0]["extra"] if k.startswith("alpha_sweep")})
        ex[ds]["sweep_all"] = {}
        for k in runs[0]["extra"]:
            if k.startswith("alpha_sweep"):
                agg_s = {}
                for a in runs[0]["extra"][k]:
                    agg_s[a] = {meth: {mm: agg([r["extra"][k][a][meth][mm] for r in runs]) for mm in ["FPR", "DR"]} for meth in ["CH", "GBDT"]}
                ex[ds]["sweep_all"][k] = agg_s
        r0 = [r for r in runs if r["seed"] == 0]
        if r0:
            ex[ds]["rules"] = r0[0]["extra"].get("rules")
            ex[ds]["importance"] = r0[0]["extra"].get("importance")
        ex[ds]["timing"] = {k: agg([r["timing"][k] for r in runs]) for k in runs[0]["timing"]}
    out["extra"] = ex
    json.dump(out, open(f"{RES}/{os.environ.get('SUMMARY', 'summary.json')}", "w"), indent=1, default=float)
    print("summary written", out["n_seeds"], out["n_seeds_loafo"])
