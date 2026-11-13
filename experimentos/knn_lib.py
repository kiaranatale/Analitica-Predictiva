import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

def knn_feats(Q, I, yI, k=10, chunk=1500, exclude_self=False):
    """Q, I: matrices tf-idf normalizadas. Devuelve top1 label, top1 sim, media de k vecinos ponderada, sim media, n vecinos sim>0.8."""
    out = np.zeros((Q.shape[0], 6), dtype=np.float32); IT = I.T.tocsr()
    for s in range(0, Q.shape[0], chunk):
        S = (Q[s:s + chunk] @ IT).toarray()
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        sims = np.take_along_axis(S, idx, 1); order = np.argsort(-sims, 1)
        idx = np.take_along_axis(idx, order, 1); sims = np.take_along_axis(sims, order, 1); lab = yI[idx]
        w = sims + 1e-6
        out[s:s + chunk] = np.c_[lab[:, 0], sims[:, 0], (lab * w).sum(1) / w.sum(1), lab[:, :3].mean(1), sims.mean(1), (S > 0.8).sum(1)]
    return out

NAMES = ["top1", "sim1", "wmean", "mean3", "simmean", "n80"]

def fit_knn(dtr, ytr, make_vec, view, tag, n_splits=5, seed=42):
    vec = make_vec().fit(view(dtr)); M = vec.transform(view(dtr)).tocsr()
    oof = np.zeros((len(dtr), 6), dtype=np.float32)
    for a, b in StratifiedKFold(n_splits, shuffle=True, random_state=seed + 2).split(M, ytr):
        oof[b] = knn_feats(M[b], M[a], ytr[a])
    return pd.DataFrame(oof, columns=[f"{tag}_{n}" for n in NAMES]), (vec, M, ytr, tag)

def apply_knn(dq, art, view):
    vec, M, ytr, tag = art
    return pd.DataFrame(knn_feats(vec.transform(view(dq)).tocsr(), M, ytr), columns=[f"{tag}_{n}" for n in NAMES])
