import pickle, numpy as np, pandas as pd, optuna, lightgbm as lgb, xgboost as xgb, time
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from features import load
from train_model import make_df, with_cats
optuna.logging.set_verbosity(optuna.logging.WARNING)
Xtr, ytr, Xva, yva = pickle.load(open("cache_eval8.pkl", "rb"))
rows = load("/mnt/user-data/uploads/train_data.jsonlines"); y = np.array([r["condition"] == "used" for r in rows]).astype(int)
df = make_df(rows); tr, va = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)
dtr, dva = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
Ctr, Cva = with_cats(Xtr, dtr), with_cats(Xva, dva); cc = [c for c in Ctr.columns if c.startswith("cf_")]
skf = list(StratifiedKFold(3, shuffle=True, random_state=1).split(Xtr, ytr))
t0 = time.time()
# ---- LGBM
def obj_l(t):
    p = dict(n_estimators=3000, learning_rate=0.06, num_leaves=t.suggest_int("num_leaves", 7, 127, log=True), min_child_samples=t.suggest_int("mcs", 10, 100),
             colsample_bytree=t.suggest_float("col", 0.2, 0.8), subsample=t.suggest_float("sub", 0.5, 1.0), subsample_freq=1,
             reg_lambda=t.suggest_float("l2", 1e-2, 30, log=True), reg_alpha=t.suggest_float("l1", 1e-3, 5, log=True), verbose=-1, n_jobs=2, random_state=42)
    accs, its = [], []
    for a, b in skf[:2]:
        m = lgb.LGBMClassifier(**p).fit(Xtr.iloc[a], ytr[a], eval_set=[(Xtr.iloc[b], ytr[b])], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(80, verbose=False)])
        accs.append((m.predict(Xtr.iloc[b]) == ytr[b]).mean()); its.append(m.best_iteration_)
    t.set_user_attr("its", int(np.mean(its))); return float(np.mean(accs))
sl = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1)); sl.optimize(obj_l, n_trials=18)
print("LGBM best cv", round(sl.best_value, 4), sl.best_params, sl.best_trial.user_attrs, round(time.time() - t0), "s", flush=True)
# ---- CatBoost
def obj_c(t):
    a, b = skf[0]
    p = dict(iterations=1500, learning_rate=0.08, depth=t.suggest_int("depth", 5, 9), l2_leaf_reg=t.suggest_float("l2", 1, 20, log=True),
             random_strength=t.suggest_float("rs", 0.1, 5, log=True), bagging_temperature=t.suggest_float("bt", 0, 3), border_count=t.suggest_categorical("bc", [64, 128, 254]),
             one_hot_max_size=t.suggest_categorical("oh", [2, 10]), thread_count=2, random_seed=42, verbose=0, od_type="Iter", od_wait=60, eval_metric="Accuracy")
    m = CatBoostClassifier(**p).fit(Ctr.iloc[a], ytr[a], cat_features=cc, eval_set=(Ctr.iloc[b], ytr[b]))
    t.set_user_attr("its", m.get_best_iteration() + 1); return float(((m.predict_proba(Ctr.iloc[b])[:, 1] > .5) == ytr[b]).mean())
sc = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=2)); sc.optimize(obj_c, n_trials=10)
print("CB best cv", round(sc.best_value, 4), sc.best_params, sc.best_trial.user_attrs, round(time.time() - t0), "s", flush=True)
pickle.dump((sl.best_params, sl.best_trial.user_attrs, sc.best_params, sc.best_trial.user_attrs), open("optuna_best.pkl", "wb"))
