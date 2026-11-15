"""Entrena el modelo final y lo guarda en models/.
Uso: python train.py --train data/train_data.jsonlines --test data/test_data.jsonlines --bert-oof colab/oof_bert.npy colab/oof_beto.npy
(--test solo se usa para los agregados por vendedor, que no usan etiquetas)"""
import argparse, os, sys, time, joblib
import numpy as np, pandas as pd, torch
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from common import load, make_df, with_cats, labels
from stack_lib import fit_stack, dense_frame, SEED
from nn_lib import Prep, train_nn, predict_nn
from seller_feats import seller_agg

ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--test", required=True); ap.add_argument("--bert-oof", required=True, nargs="+", help="uno o mas .npy con probabilidades OOF de BERT"); ap.add_argument("--out", default="models")
ap.add_argument("--cb-iters", type=int, default=360); ap.add_argument("--cb-lr", type=float, default=0.06)  # v7 (final): 360 / 0.06; v8: 1000 / 0.03
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
rows = load(a.train); y = labels(rows); df = make_df(rows)
SA = seller_agg(pd.concat([df, make_df(load(a.test))], ignore_index=True)).iloc[:len(df)].reset_index(drop=True)

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
X = pd.concat([D, ftr], axis=1)
for i, f in enumerate(a.bert_oof): X[f"bert{i}"] = np.load(f)
X["nn"] = oof
X = pd.concat([X, SA], axis=1)  # 5) agregados por vendedor sobre train+test (sin etiquetas)
C = with_cats(X, df); cc = [c for c in C.columns if c.startswith("cf_")]

# 4) CatBoost (2 semillas). En CV (5 folds) CatBoost solo (94,66%) supera al blend con LightGBM (94,63%), por eso se usa solo.
cbs = [CatBoostClassifier(iterations=a.cb_iters, learning_rate=a.cb_lr, depth=7, thread_count=2, random_seed=s, verbose=0).fit(C, y, cat_features=cc) for s in (42, 7)]
joblib.dump({"stack": stack, "pp": pp, "nt": nt, "na": na, "cols": list(X.columns), "cbs": cbs}, os.path.join(a.out, "model.pkl"))
torch.save([n.state_dict() for n in nets], os.path.join(a.out, "nets.pt"))
print("modelo guardado en", a.out, round(time.time() - t0), "s")
