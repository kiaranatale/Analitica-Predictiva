import sys, numpy as np, pandas as pd, scipy.sparse as sp, lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from features import load
from stack_lib import fit_stack, apply_stack, dense_frame, SEED, new_vecs, text_views
from train_model import make_df, with_cats

rows = load("/mnt/user-data/uploads/train_data.jsonlines"); y = np.array([r["condition"] == "used" for r in rows]).astype(int)
df = make_df(rows)
tr, va = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y)
dtr, dva = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True); ytr, yva = y[tr], y[va]
ftr, art = fit_stack(dtr, ytr); fva = apply_stack(dva, art)
Xtr = pd.concat([dense_frame(dtr), ftr], axis=1); Xva = pd.concat([dense_frame(dva), fva], axis=1)

# --- temas extra: SVM, NB, reduccion (SVD), con OOF ---
vecs = art["vecs"]; views_tr, views_va = text_views(dtr), text_views(dva)
Mtr = sp.hstack([vecs[k].transform(views_tr[k]) for k in ["w", "c", "a"]], format="csr")
Mva = sp.hstack([vecs[k].transform(views_va[k]) for k in ["w", "c", "a"]], format="csr")
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
def oof(make, fn):
    o = np.zeros(len(ytr))
    for a, b in skf.split(Mtr, ytr): o[b] = fn(make().fit(Mtr[a], ytr[a]), Mtr[b])
    m = make().fit(Mtr, ytr); return o, fn(m, Mva)
dec = lambda m, X: m.decision_function(X); pp = lambda m, X: m.predict_proba(X)[:, 1]
o, v = oof(lambda: LinearSVC(C=0.3), dec); Xtr["svm_all"], Xva["svm_all"] = o, v; print("svm", ((v > 0) == yva).mean(), flush=True)
o, v = oof(lambda: ComplementNB(alpha=0.3), pp); Xtr["nb_all"], Xva["nb_all"] = o, v; print("nb", ((v > .5) == yva).mean(), flush=True)
svd = TruncatedSVD(64, random_state=SEED).fit(Mtr)
S_tr, S_va = svd.transform(Mtr), svd.transform(Mva)
for i in range(64): Xtr[f"svd{i}"], Xva[f"svd{i}"] = S_tr[:, i], S_va[:, i]
print("features", Xtr.shape, flush=True)

def run(Xtr, Xva, tag):
    Ctr, Cva = with_cats(Xtr, dtr), with_cats(Xva, dva); cc = [c for c in Ctr.columns if c.startswith("cf_")]
    pl = lgb.LGBMClassifier(n_estimators=565, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=2).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
    pc = CatBoostClassifier(iterations=330, learning_rate=0.06, depth=7, thread_count=2, random_seed=SEED, verbose=0).fit(Ctr, ytr, cat_features=cc).predict_proba(Cva)[:, 1]
    print(tag, "lgbm", ((pl > .5) == yva).mean(), "cb", ((pc > .5) == yva).mean(), "blend", (((.3 * pl + .7 * pc) > .5) == yva).mean(), flush=True)
    return pl, pc
base_cols = [c for c in Xtr.columns if not c.startswith("svd") and c not in ("svm_all", "nb_all")]
run(Xtr[base_cols + ["svm_all", "nb_all"]], Xva[base_cols + ["svm_all", "nb_all"]], "+svm+nb")
pl, pc = run(Xtr, Xva, "+svm+nb+svd")
# arboles (ExtraTrees) y red (MLP) como modelos de diversidad
Z = lambda X: X.fillna(-1).values
et = ExtraTreesClassifier(400, min_samples_leaf=2, max_features=0.4, n_jobs=2, random_state=SEED).fit(Z(Xtr), ytr); pe = et.predict_proba(Z(Xva))[:, 1]
print("extratrees", ((pe > .5) == yva).mean(), flush=True)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(Z(Xtr)); mlp = MLPClassifier((128, 64), early_stopping=True, max_iter=60, random_state=SEED, alpha=1e-3).fit(sc.transform(Z(Xtr)), ytr); pm = mlp.predict_proba(sc.transform(Z(Xva)))[:, 1]
print("mlp", ((pm > .5) == yva).mean(), flush=True)
for w in [(0.3, 0.7, 0, 0), (0.2, 0.5, 0.2, 0.1), (0.2, 0.4, 0.3, 0.1), (0.15, 0.55, 0.2, 0)]:
    p = w[0] * pl + w[1] * pc + w[2] * pe + w[3] * pm; print("blend", w, ((p > .5) == yva).mean())
