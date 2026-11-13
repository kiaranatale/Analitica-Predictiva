"""Entrena el modelo final y lo guarda en models/.
Uso: python train.py --train data/train_data.jsonlines --bert-oof colab/oof_bert.npy"""
import argparse, os, sys, time, joblib
import numpy as np, pandas as pd, torch, lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from common import load, make_df, with_cats, labels
from stack_lib import fit_stack, dense_frame, SEED
from nn_lib import Prep, train_nn, predict_nn

ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--bert-oof", required=True); ap.add_argument("--out", default="models")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
rows = load(a.train); y = labels(rows); df = make_df(rows)

# 1) Stacking: target encoding + regresiones logisticas de texto (out-of-fold)
ftr, stack = fit_stack(df, y); print("stack", round(time.time() - t0), "s", flush=True)

# 2) Red neuronal propia (titulo + categoria + vendedor + numericas): OOF para entrenar, 2 redes full para test
D = dense_frame(df); pp = Prep().fit(df, D); A = pp.transform(df, D); nt, na = A[4], A[5]
sub = lambda d, i: ([d[0][k] for k in i], [d[1][k] for k in i], d[2][i], d[3][i])
oof = np.zeros(len(y))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED + 5).split(D, y)):
    oof[va] = predict_nn(train_nn(pp, sub(A, tr), y[tr], nt, na, epochs=7, seed=k), sub(A, va))
print("NN OOF acc", round(((oof > .5) == y).mean(), 4), round(time.time() - t0), "s", flush=True)
nets = [train_nn(pp, A[:4], y, nt, na, epochs=7, seed=s) for s in (11, 12)]

# 3) Matriz final: variables + stack + BERT (OOF generado en Colab) + NN
X = pd.concat([D, ftr], axis=1); X["bert"] = np.load(a.bert_oof); X["nn"] = oof
C = with_cats(X, df); cc = [c for c in C.columns if c.startswith("cf_")]

# 4) LightGBM + CatBoost (2 semillas); blend 0.3 / 0.7 en predict.py
lg = lgb.LGBMClassifier(n_estimators=620, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
                        random_state=SEED, verbose=-1, n_jobs=2).fit(X, y)
cbs = [CatBoostClassifier(iterations=360, learning_rate=0.06, depth=7, thread_count=2, random_seed=s, verbose=0).fit(C, y, cat_features=cc) for s in (42, 7)]
joblib.dump({"stack": stack, "pp": pp, "nt": nt, "na": na, "cols": list(X.columns), "lgb": lg, "cbs": cbs}, os.path.join(a.out, "model.pkl"))
torch.save([n.state_dict() for n in nets], os.path.join(a.out, "nets.pt"))
print("modelo guardado en", a.out, round(time.time() - t0), "s")
