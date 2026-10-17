"""Nested CV for hyper-parameter selection (RNSS + IC; k fixed by --k).
Inner: 80/20 split of the outer-training positives, grid {default, more_trees, deeper}.
Outer: refit chosen configuration on all outer-training positives, test on held-out fold."""
import argparse, json, os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from common import Data, pu_bagging_fold

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--positives-csv", default="Supplementary_material_S1.csv")
ap.add_argument("--k", type=int, default=122)
ap.add_argument("--out", default="results/nested_cv.json")
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
D = Data(a.data_dir, a.positives_csv)
GRID = [("default", 100, 6), ("more_trees", 200, 6), ("deeper", 100, 8)]
res = {"selected_hp": [], "outer_test_aucs": [], "inner_scores": []}
for fold, (otr, ote) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42)
                                  .split(np.zeros(D.n_pos), np.ones(D.n_pos))):
    perm = np.random.default_rng(1000 + fold).permutation(len(otr))
    nv = int(0.2 * len(otr))
    ival, itr = otr[perm[:nv]], otr[perm[nv:]]
    inner = {}
    for name, ne, md in GRID:
        sb = fold * 10
        inner[name] = pu_bagging_fold(D, itr, ival, a.k, True, "rnss", 3,
                                      tr_seed_fn=lambda t, s=sb: s * 1000 + t, te_seed=sb * 777,
                                      n_estimators=ne, max_depth=md)[0]
        print(f"fold{fold+1} inner {name}: {inner[name]:.4f}", flush=True)
    best = max(inner, key=inner.get)
    _, ne, md = next(g for g in GRID if g[0] == best)
    sb = fold * 100
    auc = pu_bagging_fold(D, otr, ote, a.k, True, "rnss", 5,
                          tr_seed_fn=lambda t, s=sb: s * 1000 + t, te_seed=sb * 777,
                          n_estimators=ne, max_depth=md)[0]
    print(f"fold{fold+1} outer ({best}): {auc:.4f}", flush=True)
    res["selected_hp"].append(best); res["outer_test_aucs"].append(auc); res["inner_scores"].append(inner)
    json.dump(res, open(a.out, "w"), indent=2)
print(f"nested CV k={a.k}: {np.mean(res['outer_test_aucs']):.4f} ± {np.std(res['outer_test_aucs']):.4f}")
