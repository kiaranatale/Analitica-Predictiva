import pickle, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb, itertools
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from features import load
from train_model import make_df, with_cats
Xtr, ytr, Xva, yva = pickle.load(open("cache_eval8.pkl", "rb")); lp, la, cp, ca = pickle.load(open("optuna_best.pkl", "rb"))
rows = load("/mnt/user-data/uploads/train_data.jsonlines"); y = np.array([r["condition"] == "used" for r in rows]).astype(int)
df = make_df(rows); tr, va = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)
dtr, dva = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
Ctr, Cva = with_cats(Xtr, dtr), with_cats(Xva, dva); cc = [c for c in Ctr.columns if c.startswith("cf_")]
acc = lambda p: round(((p > .5) == yva).mean(), 4); P = {}
# LGBM tuned (lr 0.06 -> 0.03 con el doble de arboles)
P["lgb_t"] = np.mean([lgb.LGBMClassifier(n_estimators=int(la["its"] * 2.2), learning_rate=0.03, num_leaves=lp["num_leaves"], min_child_samples=lp["mcs"], colsample_bytree=lp["col"], subsample=lp["sub"], subsample_freq=1,
    reg_lambda=lp["l2"], reg_alpha=lp["l1"], verbose=-1, n_jobs=2, random_state=s).fit(Xtr, ytr).predict_proba(Xva)[:, 1] for s in (1, 2, 3)], axis=0); print("lgb tuned x3", acc(P["lgb_t"]), flush=True)
P["cb_t"] = np.mean([CatBoostClassifier(iterations=int(ca["its"] * 1.1), learning_rate=0.08, depth=cp["depth"], l2_leaf_reg=cp["l2"], random_strength=cp["rs"], bagging_temperature=cp["bt"], border_count=cp["bc"],
    one_hot_max_size=cp["oh"], thread_count=2, random_seed=s, verbose=0).fit(Ctr, ytr, cat_features=cc).predict_proba(Cva)[:, 1] for s in (1, 2, 3)], axis=0); print("cb tuned x3", acc(P["cb_t"]), flush=True)
P["xgb"] = np.mean([xgb.XGBClassifier(n_estimators=700, learning_rate=0.03, max_depth=6, reg_alpha=1, reg_lambda=5, colsample_bytree=0.4, subsample=0.8, tree_method="hist", n_jobs=2, random_state=s).fit(Xtr, ytr).predict_proba(Xva)[:, 1] for s in (1, 2)], axis=0); print("xgb x2", acc(P["xgb"]), flush=True)
P["et"] = ExtraTreesClassifier(400, min_samples_leaf=2, max_features=0.4, n_jobs=2, random_state=1).fit(Xtr.fillna(-1), ytr).predict_proba(Xva.fillna(-1))[:, 1]; print("et", acc(P["et"]), flush=True)
pickle.dump(P, open("blend_preds.pkl", "wb"))
print("baseline (.3 lgb_t + .7 cb_t)", acc(.3 * P["lgb_t"] + .7 * P["cb_t"]))
print("promedio simple 4 modelos", acc(np.mean(list(P.values()), axis=0)))
print("cb .6 lgb .2 xgb .2", acc(.6 * P["cb_t"] + .2 * P["lgb_t"] + .2 * P["xgb"]))
# grilla gruesa de pesos
best = []
for w in itertools.product([0, .1, .2, .3], [0, .1, .2, .3], [0, .1, .2]):
    wc = 1 - sum(w)
    if wc < .3: continue
    p = wc * P["cb_t"] + w[0] * P["lgb_t"] + w[1] * P["xgb"] + w[2] * P["et"]; best.append((acc(p), round(wc, 2), w))
best.sort(reverse=True); print(best[:6])
