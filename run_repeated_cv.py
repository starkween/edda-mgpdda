"""5x5 repeated stratified CV ablation (N=25 paired observations per configuration).

Configurations (name, SVD k, IC weighting, negative sampling):
  baseline_k30   30   no   random   <- paper baseline
  svd_only_k107  107  no   random
  ic_only_k30    30   yes  random
  rnss_only_k30  30   no   rnss
  ic_only        107  yes  random   (reference)
  rnss_only      107  no   rnss     (reference)
  ic_svd_noRNSS  122  yes  random
  full_method    122  yes  rnss

Resumable: results are written to --out after every (config, seed).
"""
import argparse, json, os, time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from common import Data, pu_bagging_fold

CONFIGS = [("baseline_k30", 30, False, "random"), ("svd_only_k107", 107, False, "random"),
           ("ic_only", 107, True, "random"), ("rnss_only", 107, False, "rnss"),
           ("ic_only_k30", 30, True, "random"), ("rnss_only_k30", 30, False, "rnss"),
           ("ic_svd_noRNSS", 122, True, "random"), ("full_method", 122, True, "rnss")]
SEEDS = [42, 1, 2, 3, 4]

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--positives-csv", default="Supplementary_material_S1.csv")
ap.add_argument("--out", default="results/repeated_cv_results.json")
ap.add_argument("--T", type=int, default=5, help="PU-bagging iterations (5 in the reported runs)")
ap.add_argument("--only", nargs="*", help="subset of configuration names")
a = ap.parse_args()

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
D = Data(a.data_dir, a.positives_csv)
print(f"positives={D.n_pos}  RNSS pool={len(D.pool)}", flush=True)
res = json.load(open(a.out)) if os.path.exists(a.out) else {}
for name, *_ in CONFIGS:
    res.setdefault(name, {"fold_aucs": [], "fold_aps": []})

for si, seed in enumerate(SEEDS):
    splits = list(StratifiedKFold(5, shuffle=True, random_state=seed)
                  .split(np.zeros(D.n_pos), np.ones(D.n_pos)))
    for name, k, use_ic, mode in CONFIGS:
        if a.only and name not in a.only:
            continue
        if len(res[name]["fold_aucs"]) >= (si + 1) * 5:
            continue
        t0 = time.time()
        for fold, (tr, te) in enumerate(splits):
            auc, ap_ = pu_bagging_fold(
                D, tr, te, k, use_ic, mode, a.T,
                tr_seed_fn=lambda t, f=fold, s=seed: f * 1000 + t + s * 10000,
                te_seed=fold * 999 + seed)
            res[name]["fold_aucs"].append(auc)
            res[name]["fold_aps"].append(ap_)
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"[{name}] seed={seed}: AUROC={np.mean(res[name]['fold_aucs'][-5:]):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
