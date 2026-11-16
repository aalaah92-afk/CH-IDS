# CH-IDS: conformal false-alarm budgeting for hybrid intrusion detection

Code, configurations, seeds and per-seed results for the article
*"A Conformal Framework for System-Level False-Alarm Budgeting in Hybrid Intrusion Detection"*
(A. Haddad and S. A. Wajs, Multimedia University; *Computers*, MDPI, manuscript computers-4681053).

## Quick check (no data download, no training, a few seconds)
```bash
pip install -r code/requirements.txt
python code/verify_results.py
```
This recomputes the headline numbers of Tables 8, 11, 12, 13, 17, 18, 20 and 25 and the λ* analysis of Section 4.4
directly from the stored per-seed result files in `Results/`. The full re-run is described below.

## Layout
```
code/      all scripts (Python 3.13); verify_results.py recomputes the main tables
configs/   models.json (every hyperparameter), protocols.json (seeds, budgets, protocol settings)
Results/   every per-seed result file used in the manuscript (JSON)
figs/      figures are written here
data/      put the public datasets here (not redistributed), or set CHIDS_DATA
```
Set `CHIDS_ROOT` if you move `code/` away from this folder.

## Environment used for all reported numbers
KVM virtual machine, 2 vCPUs of an Intel Xeon @ 2.10 GHz (family 6, model 207), 8 GB RAM, Ubuntu 24.04.5,
Python 3.13.16, scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.4.1, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1,
imbalanced-learn 0.14.2, matplotlib. Install with `pip install -r code/requirements.txt`.
Features are stored as float32. Timing runs use one thread per model
(`CHIDS_NJOBS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`).

## Data
* NSL-KDD: `KDDTrain+.txt`, `KDDTest+.txt`, `KDDTest-21.txt` (University of New Brunswick, CIC).
* UNSW-NB15: `UNSW_NB15_training-set.csv` (175,341 records), `UNSW_NB15_testing-set.csv` (82,332 records) (UNSW Canberra).
* TON_IoT: `train_test_network.csv` (network train/test release, 211,043 records).

## Configuration (all fixed, see `models.py` and `data.py`)
| Component | Setting |
|---|---|
| Pre-processing | one-hot encoding (`handle_unknown="infrequent_if_exist"`, `min_frequency=20`), signed log transform, standardisation; TON_IoT identifier fields dropped, string fields coerced to 0 |
| Partitions | 𝒟fit 82.35% / calibration partition 17.65% of the training partition, stratified by family; calibration partition → 𝒟cal1 30%, 𝒟cal2 30%, hold-out 40% |
| Rule tier | CART (Gini), max depth 8, min 30 flows per leaf, random_state = seed; rule kept if leaf precision ≥ 0.999, support ≥ 50 and no benign match in 𝒟cal1 |
| Supervised tier | LightGBM, 300 trees, learning rate 0.05, 63 leaves, subsample 0.8, colsample 0.8 |
| Anomaly tier | MLPRegressor autoencoder (64, 16, 64), ReLU, linear output, Adam, lr 1e-3, batch 256, L2 1e-4, ≤ 60 epochs, early stopping (10% validation, patience 5), inputs clipped to [−10, 10], ≤ 60,000 benign flows |
| Fusion | weighted minimum of conformal p-values, piecewise at λ = 0 and λ = 1; λ grid {0, 0.1, 0.25, 0.5, 0.75, 0.9} chosen by inner leave-one-family-out on 𝒟cal1 |
| Calibration | order-statistic threshold on 𝒟cal2, k = ⌊α(n+1)⌋ |
| Baselines | RF 150 trees; XGBoost 300 trees, depth 8, lr 0.1; LightGBM as above; LR, DT, SVM-RBF, MLP, IForest, AE, thesis hybrid (rules → RF + SMOTE) |
| Seeds | Protocols A, C, E: 0–4; Protocol B: 0–2 |

## Scripts and outputs
| Script | Produces | Manuscript |
|---|---|---|
| `exp_standard.py DS seeds` | `results/standard/` baselines, both operating points | Tables 6, 7, A1 |
| `exp_v2.py DS seeds`, `merge_v2.py` | `results/v2/`, `results/standard_v2/` CH-IDS with split calibration, ablations, α sweep, shifts, re-calibration | Tables 6, 7, 10, 11, 21; Figures 2, 3 |
| `exp_loafo.py`, `exp_loafo_v2.py DS seeds` | `results/loafo*/` leave-one-attack-family-out | Table 19; Figures 6, 7 |
| `calib_v2.py` | `results/calib.json` target vs realised FPR, reference bands, rule false matches | Tables 11, 16, A2 |
| `exp_rev3.py DS seeds` | `results/rev3/` reproducibility check, matched calibration, budget-matched comparison, calibration-set size, per-family complementarity, contaminated windows, rule audit | Tables 8, 22, 24; Figure 4c |
| `exp_split.py dupstats DS` / `dedup DS seeds` / `host seeds` | `results/rev3_split/` duplicate statistics, deduplicated and host-disjoint runs | Table 12 |
| `exp_drift.py DS seeds` / `exp_drift.py host seeds` | `results/rev3_drift/` first (one-sample) drift monitor, superseded by `exp_fix.py` | — |
| `stats_rev3.py` | `results/stats_rev3.json`, critical-difference diagram | Tables 9, 20; Figure 5 |
| `agg_rev3.py` | `results/rev3_summary.json` | Tables 8, 12, 22 |
| `exp_fix.py DS seeds` / `exp_fix.py host seeds` / `exp_fix.py host2 seeds` | `results/rev3_fix/` contaminated windows with three threshold rules (usual rank, analyst inspection, rank margin), two-sample drift monitor, unseen-host remedies (host-disjoint calibration, re-calibration, reference refresh, adaptation) | Tables 13, 17, 18; Proposition 3 |
| `agg_fix.py` | `results/rev3_fix_summary.json`, Figure 4 | Figure 4 |
| `stats_lambda.py` | `results/stats_lambda.json` non-inferiority of the λ search | Section 4.4 |
| `exp_runtime.py` | `results/runtime_v2.json` cost and fast-path equivalence check | Table 25 |
| `analysis.py`, `figures.py`, `fig_recal.py`, `fig_arch.py`, `rules_table.py` | summaries, figures, readable rules | Figures 1–3, 6, 7; Table 23 |

## Full run
```bash
export CHIDS_NJOBS=1 OMP_NUM_THREADS=1
cd code
for d in NSL-KDD UNSW-NB15 TON_IoT; do python exp_standard.py $d 0 1 2 3 4; python exp_v2.py $d 0 1 2 3 4; done
for d in NSL-KDD UNSW-NB15 TON_IoT; do python exp_loafo.py $d 0 1 2; python exp_loafo_v2.py $d 0 1 2; done
python merge_v2.py
STD_DIR=standard_v2 LO_DIR=loafo_v2 SUMMARY=summary_v2.json python analysis.py
python calib_v2.py && python fig_recal.py && python figures.py && python rules_table.py
for d in NSL-KDD UNSW-NB15 TON_IoT; do python exp_rev3.py $d 0 1 2 3 4; python exp_split.py dupstats $d; python exp_split.py dedup $d 0 1 2 3 4; python exp_drift.py $d 0 1 2 3 4; done
python exp_split.py host 0 1 2 3 4; python exp_drift.py host 0 1 2 3 4
python stats_rev3.py && python agg_rev3.py
for d in NSL-KDD UNSW-NB15 TON_IoT; do python exp_fix.py $d 0 1 2 3 4; done
python exp_fix.py host 0 1 2 3 4; python exp_fix.py host2 0 1 2 3 4
python agg_fix.py && python stats_lambda.py
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python exp_runtime.py   # alone on an idle machine
```
`exp_rev3.py` first re-trains CH-IDS and checks that every test MCC equals the stored value exactly.
All random choices are seeded. Per-seed values for every table are also collected in Supplementary Data S1.

## Licence
MIT (see LICENSE). The datasets remain under the terms of their providers.
