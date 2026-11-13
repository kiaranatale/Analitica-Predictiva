import pickle, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from features import load
from train_model import make_df
from knn_lib import *
rows = load("/mnt/user-data/uploads/train_data.jsonlines"); y = np.array([r["condition"] == "used" for r in rows]).astype(int)
df = make_df(rows); tr, va = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)
dtr, dva = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
Xtr, ytr, Xva, yva = pickle.load(open("cache_eval4.pkl", "rb"))
mk_w = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.03, sublinear_tf=True)
mk_c = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_df=0.03, sublinear_tf=True, max_features=300000)
v_title = lambda d: d["title"]
v_all = lambda d: d["title"] + " " + d["attr_text"].str.replace("=", "_") 
import time; t=time.time()
ktr1, a1 = fit_knn(dtr, ytr, mk_w, v_title, "kw"); kva1 = apply_knn(dva, a1, v_title); print("knn word", time.time()-t, flush=True)
print("knn word wmean acc alone", ((kva1.kw_wmean > .5) == yva).mean(), flush=True)
ktr2, a2 = fit_knn(dtr, ytr, mk_c, v_title, "kc"); kva2 = apply_knn(dva, a2, v_title); print("knn char", time.time()-t, flush=True)
print("knn char wmean acc alone", ((kva2.kc_wmean > .5) == yva).mean(), flush=True)
pickle.dump((ktr1, kva1, ktr2, kva2), open("knn_cache.pkl", "wb"))
def lg(Xa, Xb, tag):
    m = lgb.LGBMClassifier(n_estimators=4000, learning_rate=0.03, num_leaves=31, min_child_samples=40, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, random_state=42, verbose=-1, n_jobs=2)
    m.fit(Xa, ytr, eval_set=[(Xb, yva)], eval_metric="binary_error", callbacks=[lgb.early_stopping(200, verbose=False)])
    print(tag, round((m.predict(Xb) == yva).mean(), 4), m.best_iteration_, flush=True)
lg(Xtr, Xva, "base")
lg(pd.concat([Xtr, ktr1], axis=1), pd.concat([Xva, kva1], axis=1), "+knn word")
lg(pd.concat([Xtr, ktr1, ktr2], axis=1), pd.concat([Xva, kva1, kva2], axis=1), "+knn word+char")
