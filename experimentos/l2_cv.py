"""CV 5-fold del nivel 2 sobre features cacheadas. Uso: python experimentos/l2_cv.py --variant base|seller ..."""
import os, sys, time, argparse, joblib, numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import with_cats
from stack_lib import dense_frame
ap = argparse.ArgumentParser(); ap.add_argument("--feats", nargs="*", default=[]); ap.add_argument("--bert", nargs="+", default=["bert", "beto"])
ap.add_argument("--models", default="lgb,cb"); ap.add_argument("--tag", default="run"); a = ap.parse_args()
c = joblib.load("cache/l1.pkl"); y = c["y"]; t0 = time.time()
X = pd.concat([dense_frame(c["df"]), c["ftr"]], axis=1)
for i, b in enumerate(a.bert): X[f"bert{i}"] = np.load(f"colab/oof_{b}.npy")
X["nn"] = c["nn_oof"]
for f in a.feats: X = pd.concat([X, joblib.load(f"cache/{f}.pkl")["tr"].reset_index(drop=True)], axis=1)
C = with_cats(X, c["df"]); cc = [k for k in C.columns if k.startswith("cf_")]
print(a.tag, "n features", X.shape[1], flush=True)
pl, pc = np.zeros(len(y)), np.zeros(len(y))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(X, y)):
    if "lgb" in a.models:
        pl[va] = lgb.LGBMClassifier(n_estimators=620, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
                                    random_state=42, verbose=-1, n_jobs=4).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    if "cb" in a.models:
        pc[va] = CatBoostClassifier(iterations=360, learning_rate=0.06, depth=7, thread_count=4, random_seed=42, verbose=0).fit(
            C.iloc[tr], y[tr], cat_features=cc).predict_proba(C.iloc[va])[:, 1]
    print(" fold", k, round(time.time() - t0), "s", flush=True)
acc = lambda p: round(((p > .5) == y).mean(), 5)
res = {"lgb": acc(pl), "cb": acc(pc), "blend": acc(.3 * pl + .7 * pc)}
print(a.tag, res, flush=True); np.save(f"cache/l2_{a.tag}.npy", np.c_[pl, pc])
