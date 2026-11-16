"""Recompute the headline numbers of the main tables from the stored result files (no retraining, runs in seconds).
usage: python code/verify_results.py"""
import os, json
ROOT = os.environ.get("CHIDS_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
R = os.path.join(ROOT, "results")
L = lambda f: json.load(open(os.path.join(R, f)))
S, FX, C, ST, LB, RT = L("rev3_summary.json"), L("rev3_fix_summary.json"), L("calib.json"), L("stats_rev3.json"), L("stats_lambda.json"), L("runtime_v2.json")
pc = lambda x, d=1: f"{100 * x:.{d}f}"
DS = {"NSL-KDD": "KDDTest+", "UNSW-NB15": "UNSW-NB15 test", "TON_IoT": "TON_IoT test"}

print("Table 8  calibration-matched MCC (%), CH-IDS vs LightGBM on the same D_cal2 flows; DR at realised FPR 1%")
for ds, t in DS.items():
    m = S["matched"][ds][t]; o = S["oracle"][ds][t]
    print(f"  {t:16s} MCC {pc(m['CH-IDS (Dcal2)']['MCC'][0])} vs {pc(m['LightGBM (Dcal2)']['MCC'][0])} | "
          f"DR@1% {pc(o['CH-IDS']['DR_at_1'][0])} vs {pc(o['LightGBM']['DR_at_1'][0])}")
print("Table 11 seed-budget cases inside the Proposition 1 band (random hold-outs):", C.get("id_band_summary"))
print("Table 12 deduplicated hold-out FPR at alpha = 1% (%):",
      {ds: pc(S["dedup"][ds]["sets"]["hold-out"]["0.01"]["CH-IDS"]["FPR"][0], 2) for ds in DS})
h = FX["host"]
print("Table 13 unseen TON_IoT hosts at alpha = 1% (%), FPR / DR")
print("  standard               ", pc(h["standard"]["unseen hosts"]["0.01"]["CH-IDS"]["FPR"][0]), "/", pc(h["standard"]["unseen hosts"]["0.01"]["CH-IDS"]["DR"][0]))
print("  host-disjoint calib.   ", pc(h["hostcal"]["unseen hosts"]["0.01"]["CH-IDS"]["FPR"][0]), "/", pc(h["hostcal"]["unseen hosts"]["0.01"]["CH-IDS"]["DR"][0]))
for n in ["500", "1000", "2000"]:
    a = h["adapt2"][n]["CH-IDS"]
    print(f"  adaptation n={n:5s}     {pc(a['FPR'][0], 2)} / {pc(a['DR'][0])}  (DR range {pc(a['DR_range'][0])}-{pc(a['DR_range'][1])})")
print("Table 17 contaminated windows, UNSW-NB15 test, FPR / DR (%) [flows inspected]")
for c in ["0.0", "0.01", "0.05"]:
    e = FX["contam"]["UNSW-NB15"]["UNSW-NB15 test"][c]
    print(f"  c={float(c)*100:g}%: usual rank {pc(e['naive']['FPR'][0], 2)} / {pc(e['naive']['DR'][0])} | "
          f"inspected {pc(e['labeltrim']['FPR'][0], 2)} / {pc(e['labeltrim']['DR'][0])} [{e['labeltrim']['inspected'][0]:.0f}]")
print("Table 18 drift monitor, windows of 500 flows flagged (%):",
      {k: pc(FX["drift"][ds][k2]["mixed_w500"][0]) for k, (ds, k2) in
       {"UNSW hold-out (no shift)": ("UNSW-NB15", "hold-out"), "KDDTest+ (shift)": ("NSL-KDD", "KDDTest+"),
        "UNSW test (shift)": ("UNSW-NB15", "UNSW-NB15 test")}.items()})
b = ST["B"]["DR"]["GBDT [conformal]"]
print(f"Table 20 CH-IDS vs conformal LightGBM, 22 families: W/T/L {b['wins']}/{b['ties']}/{b['losses']}, "
      f"HL {pc(b['hl'])} [{pc(b['hl_lo'])}, {pc(b['hl_hi'])}] pp, Holm p = {b['p_holm']:.2f}")
print("Section 4.4 lambda* non-inferiority (margin 1 pp), Holm p:",
      {k: round(v["p_noninf_holm"], 3) for k, v in LB["comparisons"].items()})
print("Table 25 throughput UNSW-NB15 (flows/s): CH-IDS", round(RT["UNSW-NB15"]["throughput"]["CH-IDS"]),
      "| LightGBM", round(RT["UNSW-NB15"]["throughput"]["LightGBM"]))
