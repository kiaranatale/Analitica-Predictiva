import sys, pickle, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from features import load, build_frame
rows = load('/mnt/user-data/uploads/train_data.jsonlines'); y = np.array([r["condition"] == "used" for r in rows]).astype(int)
df = build_frame(rows)
tr, va = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)
Xtr, ytr, Xva, yva = pickle.load(open('cache_eval4.pkl','rb'))
assert (ytr==y[tr]).all()
cats = ["seller","category","city","state","listing","ship_mode","buying_mode"]
def add(X, d):
    X = X.copy()
    for c in cats: X["cf_"+c] = d[c].astype(str).values
    X["cf_seller_cat"] = (d["seller"].astype(str)+"|"+d["category"].astype(str)).values
    X["cf_cat_listing"] = (d["category"].astype(str)+"|"+d["listing"].astype(str)).values
    return X
Ctr, Cva = add(Xtr, df.iloc[tr]), add(Xva, df.iloc[va])
cc = [c for c in Ctr.columns if c.startswith("cf_")]
m = CatBoostClassifier(iterations=2500, learning_rate=0.06, depth=7, thread_count=2, random_seed=42, verbose=250, od_type="Iter", od_wait=150, eval_metric="Accuracy")
m.fit(Ctr, ytr, cat_features=cc, eval_set=(Cva, yva))
pc = m.predict_proba(Cva)[:,1]; print("catboost acc", ((pc>.5)==yva).mean(), m.get_best_iteration())
ml = lgb.LGBMClassifier(n_estimators=565,learning_rate=0.03,num_leaves=63,colsample_bytree=0.5,subsample=0.8,subsample_freq=1,random_state=42,verbose=-1,n_jobs=2).fit(Xtr,ytr)
pl = ml.predict_proba(Xva)[:,1]; print("lgbm", ((pl>.5)==yva).mean(), "blend", (((pl+pc)/2>.5)==yva).mean())
np.save("pc.npy", pc); np.save("pl.npy", pl)
