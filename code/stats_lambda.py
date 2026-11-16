"""Revision 3b: is the searched lambda* worth having? Family-level non-inferiority / superiority analysis (no new training).
Unit: held-out family (22), seeds averaged within family. For each fixed lambda in {0, 0.25, 0.5}:
  d_f = DR(lambda*) - DR(lambda fixed); Hodges-Lehmann estimate and 95% CI; Wilcoxon superiority p (two-sided);
  non-inferiority with margin delta = 1 percentage point: one-sided Wilcoxon on d_f + delta (H1: median > -delta);
  Holm within the three lambda comparisons and within all nine comparisons of Table 18.
Also: regret of lambda* relative to the best fixed lambda chosen in hindsight on the test results of each dataset."""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, glob
import numpy as np
from scipy import stats
from stats_rev3 import hodges_lehmann, holm, rank_biserial

RES = ROOT + "/results"; P = "CH-IDS (proposed)"; DELTA = 0.01
L = {}
for fn in glob.glob(f"{RES}/loafo_v2/*.json"):
    R = json.load(open(fn)); L.setdefault(R["dataset"], {})[R["seed"]] = R
units = []
for ds, runs in L.items():
    for f in sorted(next(iter(runs.values()))["families"]):
        rec = {m: np.mean([runs[s]["families"][f]["results"][m]["DR"] for s in runs]) for m in [P, "CH-IDS lambda=0.0", "CH-IDS lambda=0.25", "CH-IDS lambda=0.5"]}
        rec["lams"] = [runs[s]["families"][f]["lam"] for s in runs]
        rec["sd"] = float(np.std([runs[s]["families"][f]["results"][P]["DR"] for s in runs], ddof=1))
        units.append((ds, f, rec))
out = {"delta": DELTA, "comparisons": {}}
ps = []
for lam in ["0.0", "0.25", "0.5"]:
    d = np.array([u[2][P] - u[2][f"CH-IDS lambda={lam}"] for u in units])
    nz = d[d != 0]
    p_sup = float(stats.wilcoxon(nz).pvalue)
    p_ni = float(stats.wilcoxon(d + DELTA, alternative="greater").pvalue)
    hl = hodges_lehmann(d)
    out["comparisons"][lam] = dict(mean=float(d.mean()), hl=hl[0], hl_lo=hl[1], hl_hi=hl[2], wins=int((d > 0).sum()),
                                   ties=int((d == 0).sum()), losses=int((d < 0).sum()), p_sup=p_sup, p_noninf=p_ni,
                                   r_rb=rank_biserial(d), worst=float(d.min()))
    ps.append(p_sup)
for lam, a in zip(["0.0", "0.25", "0.5"], holm(ps)):
    out["comparisons"][lam]["p_sup_holm_lambda_family"] = float(a)
for lam, a in zip(["0.0", "0.25", "0.5"], holm([out["comparisons"][l]["p_noninf"] for l in ["0.0", "0.25", "0.5"]])):
    out["comparisons"][lam]["p_noninf_holm"] = float(a)
# hindsight regret: best fixed lambda per dataset chosen on the test results
reg = {}
for ds in L:
    U = [u for u in units if u[0] == ds]
    means = {lam: np.mean([u[2][f"CH-IDS lambda={lam}"] for u in U]) for lam in ["0.0", "0.25", "0.5"]}
    best = max(means, key=means.get)
    reg[ds] = dict(best_fixed=best, best_fixed_DR=float(means[best]), searched_DR=float(np.mean([u[2][P] for u in U])),
                   lam_star=sorted(set(l for u in U for l in u[2]["lams"])))
out["hindsight"] = reg
out["median_seed_sd"] = float(np.median([u[2]["sd"] for u in units]))
json.dump(out, open(f"{RES}/stats_lambda.json", "w"), indent=1)
print(json.dumps(out, indent=1))
