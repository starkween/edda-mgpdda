"""Table of mean±std, paired t-test and sign-flip permutation test vs the baseline."""
import argparse, json
import numpy as np
from scipy import stats
ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/repeated_cv_results.json")
ap.add_argument("--baseline", default="baseline_k30")
a = ap.parse_args()
r = json.load(open(a.results))
base = np.array(r[a.baseline]["fold_aucs"][:25])
print(f"{'config':16s} {'AUROC':>14s} {'AUPRC':>14s} {'delta':>8s} {'t':>8s} {'p(t)':>9s} {'p(perm)':>8s}")
for name, v in r.items():
    x = np.array(v["fold_aucs"][:25]); ap_ = np.array(v["fold_aps"][:25])
    if len(x) < 25:
        print(f"{name:16s} incomplete ({len(x)}/25)"); continue
    row = f"{name:16s} {x.mean():.4f}±{x.std():.4f} {ap_.mean():.4f}±{ap_.std():.4f}"
    if name != a.baseline:
        d = x - base
        t, p = stats.ttest_rel(x, base) if d.std() > 0 else (float('nan'), float('nan'))
        flips = np.array([np.random.default_rng(i + 42).choice([-1, 1], 25) for i in range(10000)])
        pperm = np.mean((flips * d).mean(1) >= d.mean())
        row += f" {d.mean():+.4f} {t:8.3f} {p:9.5f} {pperm:8.5f}"
    print(row)
