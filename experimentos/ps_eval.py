import sys, numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from features import load
from stack_lib import fit_stack, apply_stack, dense_frame, SEED
from train_model import make_df, with_cats

def fit_pred(dtr, ytr, dq, seeds=(42,), n_lgb=565, n_cb=330):
    ftr, art = fit_stack(dtr, ytr); fq = apply_stack(dq, art)
    Xtr = pd.concat([dense_frame(dtr), ftr], axis=1); Xq = pd.concat([dense_frame(dq), fq], axis=1)
    Ctr, Cq = with_cats(Xtr, dtr), with_cats(Xq, dq); cc = [c for c in Ctr.columns if c.startswith("cf_")]
    pl = lgb.LGBMClassifier(n_estimators=n_lgb, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5, subsample=0.8,
                            subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=2).fit(Xtr, ytr).predict_proba(Xq)[:, 1]
    pc = np.mean([CatBoostClassifier(iterations=n_cb, learning_rate=0.06, depth=7, thread_count=2, random_seed=s, verbose=0)
                  .fit(Ctr, ytr, cat_features=cc).predict_proba(Cq)[:, 1] for s in seeds], axis=0)
    return 0.3 * pl + 0.7 * pc

if __name__ == "__main__":
    rows = load("/mnt/user-data/uploads/train_data.jsonlines"); y = np.array([r["condition"] == "used" for r in rows]).astype(int)
    test = load("/mnt/user-data/uploads/test_data.jsonlines")
    df = make_df(rows); dte = make_df(test)
    tr, va = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y)
    dtr, dva = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
    pool = pd.concat([dva, dte], ignore_index=True)
    p0 = fit_pred(dtr, y[tr], pool, seeds=(42, 7, 123)); nv = len(dva)
    print("base 3 seeds acc", ((p0[:nv] > .5) == y[va]).mean(), flush=True)
    for th in (0.9, 0.97):
        m = (p0 > th) | (p0 < 1 - th)
        print("th", th, "pseudo rows", m.sum(), "pseudo acc on va part", ((p0[:nv][m[:nv]] > .5) == y[va][m[:nv]]).mean(), flush=True)
        d2 = pd.concat([dtr, pool[m]], ignore_index=True); y2 = np.concatenate([y[tr], (p0[m] > .5).astype(int)])
        p1 = fit_pred(d2, y2, pool, seeds=(42, 7, 123))
        print("pseudo th", th, "acc", ((p1[:nv] > .5) == y[va]).mean(), flush=True)
