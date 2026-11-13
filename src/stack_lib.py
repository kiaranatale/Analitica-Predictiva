import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from features import CAT_COLS

SEED = 42
TE_KEYS = {"seller": ["seller"], "category": ["category"], "city": ["city"],
           "seller_cat": ["seller", "category"], "seller_listing": ["seller", "listing"],
           "cat_listing": ["category", "listing"], "pm": ["pay_combo"], "wr": ["warr_norm"], "cat_pm": ["category", "pay_combo"], "seller_pm": ["seller", "pay_combo"], "cat_fw": ["category", "first_word"], "cat_lw": ["category", "last_word"], "cat_pb": ["category", "price_bin"], "cat_qty": ["category", "qty_bin"], "cat_ship": ["category", "ship_key"], "cat_war": ["category", "warr_norm"], "seller_pb": ["seller", "price_bin"], "city_cat": ["city", "category"]}
SM = 2.0

def _key(df, cols):
    return df[cols].astype(str).agg("|".join, axis=1) if len(cols) > 1 else df[cols[0]].astype(str)

def te_fit(keys, y, prior):
    g = pd.DataFrame({"k": keys.values, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return g, prior

def te_apply(keys, tab, prior):
    g = tab
    s = keys.map(g["sum"]).fillna(0.0).values; n = keys.map(g["count"]).fillna(0.0).values
    return (s + SM * prior) / (n + SM), n

def text_views(df):
    return {"w": df["title"], "c": df["title"], "a": df["attr_text"] + " " + df["tag_text"] + " " + df["title"]}

def new_vecs():
    return {"w": TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=150000),
            "c": TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=250000),
            "a": TfidfVectorizer(token_pattern=r"[^ ]+", ngram_range=(1, 1), min_df=2, sublinear_tf=True, max_features=150000)}

def fit_stack(df, y, n_splits=5):
    """Ajusta TE y LR de texto sobre df; devuelve (features OOF para df, artefactos para aplicar a otros)."""
    prior = float(np.mean(y))
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    keys = {n: _key(df, c) for n, c in TE_KEYS.items()}
    feats = pd.DataFrame(index=range(len(df)))
    for n in TE_KEYS: feats["te_" + n] = 0.0; feats["n_" + n] = 0.0
    vecs = new_vecs(); views = text_views(df)
    X = {k: vecs[k].fit_transform(views[k]) for k in vecs}
    for k in vecs: feats["lr_" + k] = 0.0
    for tr, va in skf.split(df, y):
        for n in TE_KEYS:
            tab, _ = te_fit(keys[n].iloc[tr], y[tr], prior)
            te, cnt = te_apply(keys[n].iloc[va], tab, prior)
            feats.loc[va, "te_" + n] = te; feats.loc[va, "n_" + n] = cnt
        for k in vecs:
            lr = LogisticRegression(C=4, max_iter=300, solver="liblinear").fit(X[k][tr], y[tr])
            feats.loc[va, "lr_" + k] = lr.predict_proba(X[k][va])[:, 1]
    art = {"prior": prior, "te": {n: te_fit(keys[n], y, prior)[0] for n in TE_KEYS}, "vecs": vecs,
           "lr": {k: LogisticRegression(C=4, max_iter=300, solver="liblinear").fit(X[k], y) for k in vecs}}
    return feats, art

def apply_stack(df, art):
    feats = pd.DataFrame(index=range(len(df)))
    for n, c in TE_KEYS.items():
        te, cnt = te_apply(_key(df, c), art["te"][n], art["prior"])
        feats["te_" + n] = te; feats["n_" + n] = cnt
    views = text_views(df)
    for k in art["vecs"]:
        feats["lr_" + k] = art["lr"][k].predict_proba(art["vecs"][k].transform(views[k]))[:, 1]
    return feats

def dense_frame(df):
    num = df.drop(columns=CAT_COLS + ["attr_text", "tag_text", "title"]).copy()
    return num
