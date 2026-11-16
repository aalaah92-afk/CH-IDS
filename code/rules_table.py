"""Convert induced rules (seed 0) back to raw feature units for the manuscript."""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, sys
import numpy as np
import data

OUT = ROOT + "/results/rules_readable_v2.json"
import models
from exp_v2 import split_cal


def inverse(pre, name, thr):
    kind, feat = name.split("__", 1)
    if kind == "num":
        num = pre.named_transformers_["num"]
        sc = num.named_steps["sc"]
        i = list(pre.transformers_[0][2]).index(feat)
        v = thr * sc.scale_[i] + sc.mean_[i]
        return feat, float(np.sign(v) * np.expm1(abs(v)))
    return feat, None


def fmt(feat, op, val):
    if val is None:  # one-hot: feat = "<col>_<category>"
        return feat, op
    return feat, op


res = {}
for dsname in ["NSL-KDD", "UNSW-NB15", "TON_IoT"]:
    ds = data.LOADERS[dsname](); P = data.prepare(ds, 0)
    pre = P["pre"]
    i1, i2, iid = split_cal(P, 0)
    rl = models.RuleLayer(seed=0).fit(P["Xf"], P["yf"], P["Xc"][i1], P["yc"][i1])
    cat_cols = ds["cat"]
    rules = sorted(rl.rules(P["names"]), key=lambda r: -r[1])
    Xraw = ds["train"][0]
    is_int = {c: bool(np.all(np.mod(Xraw[c].astype(float), 1) == 0)) for c in ds["num"]}
    out = []
    for conds, sup in rules:
        # merge conditions on the same feature into an interval / category statement
        bounds, cats = {}, []
        for name, op, thr in conds:
            kind, feat = name.split("__", 1)
            if kind == "num":
                f, v = inverse(pre, name, thr)
                lo, hi = bounds.get(f, (-np.inf, np.inf))
                if op == "<=":
                    hi = min(hi, v)
                else:
                    lo = max(lo, v)
                bounds[f] = (lo, hi)
            else:
                col = next(c for c in cat_cols if feat.startswith(c + "_"))
                catv = feat[len(col) + 1:]
                cats.append(f"{col} {'≠' if op == '<=' else '='} {catv}")
        parts = []
        for f, (lo, hi) in bounds.items():
            def r(x):
                return f"{x:.3g}"
            if is_int.get(f):
                lo_s = (f"{int(np.floor(lo)) + 1}" if lo > -np.inf else None)
                hi_s = (f"{int(np.floor(hi))}" if hi < np.inf else None)
                if lo_s and hi_s:
                    parts.append(f"{lo_s} ≤ {f} ≤ {hi_s}" if lo_s != hi_s else f"{f} = {lo_s}")
                elif lo_s:
                    parts.append(f"{f} ≥ {lo_s}")
                else:
                    parts.append(f"{f} = 0" if hi_s == "0" and (Xraw[f] >= 0).all() else f"{f} ≤ {hi_s}")
                continue
            if lo > -np.inf and hi < np.inf:
                parts.append(f"{r(lo)} < {f} ≤ {r(hi)}")
            elif lo > -np.inf:
                parts.append(f"{f} > {r(lo)}")
            else:
                parts.append(f"{f} ≤ {r(hi)}")
        # keep positive category statements, drop most negative ones for brevity
        pos = [c for c in cats if " = " in c]
        neg = [c for c in cats if "≠" in c]
        out.append(dict(support=sup, conds=pos + parts + neg, n_conds=len(conds)))
    res[dsname] = out
    print(dsname, len(out))
    for o in out[:3]:
        print("  ", o["support"], " AND ".join(o["conds"]))
json.dump(res, open(OUT, "w"), indent=1)
