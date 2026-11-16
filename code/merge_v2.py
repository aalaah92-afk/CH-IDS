"""Merge split-calibrated CH-IDS results (results/v2) into the baseline result files -> results/standard_v2."""
import os as _os
ROOT = _os.environ.get('CHIDS_ROOT', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, glob, os
os.makedirs(ROOT + "/results/standard_v2", exist_ok=True)
for fn in glob.glob(ROOT + "/results/v2/*.json"):
    V = json.load(open(fn))
    ds, sd = V["dataset"], V["seed"]
    O = json.load(open(f"{ROOT}/results/standard/{ds}_{sd}.json"))
    for t, res in V["results"].items():
        for k, v in res.items():
            if k.startswith("_"):
                continue
            O["results"][t][k] = v
        O["extra"][f"rule_coverage_{t}"] = res["_rule_cov"]
        O["extra"][f"alpha_sweep_{t}"] = {a: {"CH": {"FPR": x["fp"] / x["m"], "DR": x["DR"]}, "GBDT": {"FPR": x["fp"] / x["m"], "DR": x["DR"]}}
                                          for a, x in V["sweep"][t].items()}
    O["extra"]["lambda"] = V["lam"]; O["extra"]["n_rules"] = V["n_rules"]; O["extra"]["inner_dr"] = V["inner_dr"]
    json.dump(O, open(f"{ROOT}/results/standard_v2/{ds}_{sd}.json", "w"), default=float)
print("merged", len(glob.glob(ROOT + "/results/standard_v2/*.json")))
