import numpy as np, pandas as pd, scipy.sparse as sp, pickle, lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from features import load
from stack_lib import fit_stack, apply_stack, dense_frame, SEED
from train_model import make_df, with_cats
rows = load("/mnt/user-data/uploads/train_data.jsonlines"); y = np.array([r["condition"] == "used" for r in rows]).astype(int)
df = make_df(rows); tr, va = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y)
dtr, dva = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True); ytr, yva = y[tr], y[va]
ftr, art = fit_stack(dtr, ytr); fva = apply_stack(dva, art)
Xtr = pd.concat([dense_frame(dtr), ftr], axis=1); Xva = pd.concat([dense_frame(dva), fva], axis=1)
pickle.dump((Xtr, ytr, Xva, yva), open("cache_eval8.pkl", "wb"))
Ctr, Cva = with_cats(Xtr, dtr), with_cats(Xva, dva); cc = [c for c in Ctr.columns if c.startswith("cf_")]
res = {}
for name, kw in {"cb d8 lr.03": dict(depth=8, learning_rate=0.03, l2_leaf_reg=5), "cb d6 lr.03": dict(depth=6, learning_rate=0.03, l2_leaf_reg=3), "cb d7 lr.06 l2=10": dict(depth=7, learning_rate=0.06, l2_leaf_reg=10)}.items():
    m = CatBoostClassifier(iterations=3000, thread_count=2, random_seed=SEED, verbose=0, od_type="Iter", od_wait=200, eval_metric="Accuracy", **kw)
    m.fit(Ctr, ytr, cat_features=cc, eval_set=(Cva, yva)); p = m.predict_proba(Cva)[:, 1]; res[name] = p
    print(name, "acc", round(((p > .5) == yva).mean(), 4), "best_it", m.get_best_iteration(), flush=True)
pickle.dump(res, open("cb_tuned.pkl", "wb"))
# LGBM con unigramas TF-IDF dispersos (interaccion palabra x categoria)
v = TfidfVectorizer(min_df=5, max_features=4000, sublinear_tf=True).fit(dtr["title"])
Str = sp.hstack([sp.csr_matrix(Xtr.fillna(-1).values.astype(np.float32)), v.transform(dtr["title"])], format="csr")
Sva = sp.hstack([sp.csr_matrix(Xva.fillna(-1).values.astype(np.float32)), v.transform(dva["title"])], format="csr")
m = lgb.LGBMClassifier(n_estimators=4000, learning_rate=0.03, num_leaves=31, min_child_samples=30, colsample_bytree=0.4, subsample=0.8, subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=2)
m.fit(Str, ytr, eval_set=[(Sva, yva)], eval_metric="binary_error", callbacks=[lgb.early_stopping(200, verbose=False)])
pw = m.predict_proba(Sva)[:, 1]; print("lgbm +tfidf sparse", round(((pw > .5) == yva).mean(), 4), m.best_iteration_, flush=True)
best = res["cb d8 lr.03"]
print("blend cb8+lgb_sparse", round((((.6 * best + .4 * pw) > .5) == yva).mean(), 4))
pa = np.mean(list(res.values()), axis=0) * .7 + .3 * pw
print("avg all cb + sparse", round(((pa > .5) == yva).mean(), 4))
