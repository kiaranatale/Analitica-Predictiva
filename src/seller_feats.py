"""Features a nivel vendedor.
1) Agregados sin etiqueta sobre train+test (volumen, diversidad, precios, etc.).
2) Vecinos dentro del vendedor: etiqueta de los items de train mas parecidos (titulo TF-IDF) del mismo vendedor, out-of-fold.
(Se descarto promediar predicciones OOF de otros items del vendedor: filtra la etiqueta propia via modelos con embedding de vendedor.)"""
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

def seller_agg(df_all):
    d = df_all.assign(pl=df_all["price_log"], used_w=df_all["warr_usado"], q1=(df_all["initial_quantity"] > 1).astype(int),
                      gold=df_all["listing"].astype(str).str.startswith("gold").astype(int), free=df_all["listing"].eq("free").astype(int))
    g = d.groupby("seller")
    out = pd.DataFrame({
        "s_n": g["seller"].transform("size"), "s_ncat": g["category"].transform("nunique"),
        "s_pl_mean": g["pl"].transform("mean"), "s_pl_std": g["pl"].transform("std"),
        "s_pl_dev": d["pl"] - g["pl"].transform("mean"),
        "s_q1": g["q1"].transform("mean"), "s_gold": g["gold"].transform("mean"), "s_free": g["free"].transform("mean"),
        "s_warr": g["has_warranty"].transform("mean"), "s_warr_usado": g["used_w"].transform("mean"),
        "s_sold": g["sold_quantity"].transform("mean"), "s_pics": g["n_pics"].transform("mean"),
        "s_nwarr": g["warr_norm"].transform("nunique"), "s_npay": g["pay_combo"].transform("nunique"),
        "sc_n": d.groupby(["seller", "category"])["seller"].transform("size"),
    })
    out["sc_frac"] = out["sc_n"] / out["s_n"]
    return out.reset_index(drop=True)

def _topk_within_group(X, groups, cand_mask, query_idx, yfull, k=5):
    """Para cada query, similitud coseno con candidatos (cand_mask) del mismo grupo; devuelve top1 sim, label top1, media ponderada top-k, n cand."""
    res = np.full((len(query_idx), 4), np.nan); res[:, 3] = 0
    pos = {q: i for i, q in enumerate(query_idx)}
    by_g = pd.Series(np.arange(len(groups))).groupby(groups).indices
    qset = np.zeros(len(groups), bool); qset[query_idx] = True
    for gk, idx in by_g.items():
        qs = idx[qset[idx]]; cs = idx[cand_mask[idx]]
        if len(qs) == 0 or len(cs) == 0: continue
        S = (X[qs] @ X[cs].T).toarray()
        same = qs[:, None] == cs[None, :]; S[same] = -1  # nunca a si mismo
        for r, q in enumerate(qs):
            s = S[r]; ok = s > -1
            if not ok.any(): continue
            o = np.argsort(-s)[:k]; o = o[s[o] > -1]
            w = s[o] + 1e-3
            res[pos[q]] = [s[o[0]], yfull[cs[o[0]]], (w * yfull[cs[o]]).sum() / w.sum(), ok.sum()]
    return res

def seller_knn(df_all, y, n_tr, seed=42, n_splits=5):
    """OOF para train (candidatos = otros folds), test usa todo train."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(df_all["title"]).tocsr()
    groups = df_all["seller"].astype(str).values
    yfull = np.concatenate([y, np.zeros(len(df_all) - n_tr)]).astype(float)
    out = np.zeros((len(df_all), 4))
    for a, b in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(n_tr), y):
        cand = np.zeros(len(df_all), bool); cand[a] = True
        out[b] = _topk_within_group(X, groups, cand, b, yfull)
    cand = np.zeros(len(df_all), bool); cand[:n_tr] = True
    te_idx = np.arange(n_tr, len(df_all)); out[te_idx] = _topk_within_group(X, groups, cand, te_idx, yfull)
    return pd.DataFrame(out, columns=["sk_sim1", "sk_y1", "sk_ywk", "sk_ncand"])

def seller_pred_mean(df_all, P, cols):
    """Media de predicciones de nivel 1 de los OTROS items del vendedor (y del par vendedor-categoria)."""
    out = {}
    for keyname, key in (("s", df_all["seller"].astype(str)), ("sc", df_all["seller"].astype(str) + "|" + df_all["category"].astype(str))):
        cnt = key.map(key.value_counts()).values
        for c in cols:
            tot = pd.Series(P[c].values).groupby(key.values).transform("sum").values
            out[f"{keyname}m_{c}"] = np.where(cnt > 1, (tot - P[c].values) / np.maximum(cnt - 1, 1), np.nan)
    return pd.DataFrame(out)
