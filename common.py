"""Shared data loading, IC weighting, RNSS pool and feature construction."""
import gc
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb


class Data:
    def __init__(self, data_dir, positives_csv="Supplementary_material_S1.csv"):
        P = lambda f: f"{data_dir}/{f}"
        self.A_rg = np.loadtxt(P("drug-function_association_matrix.txt"), dtype=np.float32)
        self.A_dg = np.loadtxt(P("disease-function_association_matrix.txt"), dtype=np.float32)
        df = pd.read_csv(P(positives_csv))
        self.drugs = list(pd.read_csv(P("drug_list.csv"))["drugBankId"])
        self.diseases = list(pd.read_csv(P("disease_list.csv"))["omimId"])
        di = {d: i for i, d in enumerate(self.drugs)}
        dj = {d: i for i, d in enumerate(self.diseases)}
        self.pos_di = np.array([di[d] for d in df["drugBankId"]])
        self.pos_dj = np.array([dj[d] for d in df["omimId"]])
        self.n_pos = len(self.pos_di)
        self.pos_set = set(zip(self.pos_di.tolist(), self.pos_dj.tolist()))
        # Information-content weights (shared by all IC configs)
        f = self.A_rg.sum(0) + self.A_dg.sum(0)
        ic = -np.log2(f / (f.sum() + 1e-12) + 1e-12)
        ic /= ic.max() + 1e-12
        self.IC3 = np.tile(ic, 3).astype(np.float32)
        # RNSS pool: random pairs that are non-positive and have M1 == 0
        rng0 = np.random.default_rng(0)
        pool, checked = [], 0
        while len(pool) < 50000:
            ri = int(rng0.integers(len(self.drugs)))
            cj = int(rng0.integers(len(self.diseases)))
            if (ri, cj) in self.pos_set:
                continue
            checked += 1
            if (self.A_rg[ri] * self.A_dg[cj]).sum() == 0:
                pool.append((ri, cj))
            if checked > 3_000_000:
                break
        self.pool = np.array(pool)

    def neg_random(self, n, seed):
        rng = np.random.default_rng(seed)
        out = []
        while len(out) < n:
            ri = int(rng.integers(len(self.drugs)))
            cj = int(rng.integers(len(self.diseases)))
            if (ri, cj) not in self.pos_set:
                out.append((ri, cj))
        a = np.array(out)
        return a[:, 0], a[:, 1]

    def neg_rnss(self, n, seed):
        idx = np.random.default_rng(seed).choice(len(self.pool), size=n, replace=False)
        return self.pool[idx, 0], self.pool[idx, 1]

    def negatives(self, mode, n, seed):
        return self.neg_rnss(n, seed) if mode == "rnss" else self.neg_random(n, seed)

    def build(self, di, dj, A_rd_tr, use_ic):
        """Meta-path profiles M1|M2|M3; A_rd_tr contains TRAINING positives only."""
        dng = (A_rd_tr.T @ self.A_rg).astype(np.float32)
        png = (A_rd_tr @ self.A_dg).astype(np.float32)
        X = sp.hstack([sp.csr_matrix(self.A_rg[di] * self.A_dg[dj]),
                       sp.csr_matrix(self.A_rg[di] * dng[dj]),
                       sp.csr_matrix(png[di] * self.A_dg[dj])]).tocsr()
        return X.multiply(self.IC3) if use_ic else X

    def train_matrix(self, tr_idx):
        A = np.zeros((len(self.drugs), len(self.diseases)), dtype=np.float32)
        A[self.pos_di[tr_idx], self.pos_dj[tr_idx]] = 1.0
        return A


def pu_bagging_fold(D, tr_idx, te_idx, k, use_ic, neg_mode, T, tr_seed_fn, te_seed,
                    n_estimators=100, max_depth=6, learning_rate=0.3, min_child_weight=3,
                    te_neg_mode=None):
    """One train/test evaluation. SVD is fit on the training matrix only."""
    A_tr = D.train_matrix(tr_idx)
    tni, tnj = D.negatives(te_neg_mode or neg_mode, len(te_idx), te_seed)
    te_di = np.concatenate([D.pos_di[te_idx], tni]).astype(int)
    te_dj = np.concatenate([D.pos_dj[te_idx], tnj]).astype(int)
    y_te = np.array([1] * len(te_idx) + [0] * len(te_idx))
    X_te = D.build(te_di, te_dj, A_tr, use_ic)
    preds = np.zeros(len(y_te))
    for t in range(T):
        ni, nj = D.negatives(neg_mode, len(tr_idx), tr_seed_fn(t))
        tr_di = np.concatenate([D.pos_di[tr_idx], ni]).astype(int)
        tr_dj = np.concatenate([D.pos_dj[tr_idx], nj]).astype(int)
        y_tr = np.array([1] * len(tr_idx) + [0] * len(tr_idx))
        X_tr = D.build(tr_di, tr_dj, A_tr, use_ic)
        svd = TruncatedSVD(k, random_state=t)
        Xr = svd.fit_transform(X_tr)
        Xt = svd.transform(X_te)
        clf = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                learning_rate=learning_rate, min_child_weight=min_child_weight,
                                eval_metric="logloss", n_jobs=1, random_state=t)
        clf.fit(Xr, y_tr)
        preds += clf.predict_proba(Xt)[:, 1]
        del X_tr, Xr, svd, clf
        gc.collect()
    preds /= T
    return float(roc_auc_score(y_te, preds)), float(average_precision_score(y_te, preds))
