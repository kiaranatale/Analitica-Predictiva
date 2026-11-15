"""Calcula y cachea las features de nivel 1 (OOF en train, aplicadas en test) para experimentar rapido con el nivel 2."""
import os, sys, time, joblib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import load, make_df, labels
from stack_lib import fit_stack, apply_stack, dense_frame, SEED
from nn_lib import Prep, train_nn, predict_nn
t0 = time.time()
tr = load("data/train_data.jsonlines"); te = load("data/test_data.jsonlines"); y = labels(tr)
df, dt = make_df(tr), make_df(te); print("frames", round(time.time() - t0), flush=True)
ftr, art = fit_stack(df, y); fte = apply_stack(dt, art); print("stack", round(time.time() - t0), flush=True)
D, Dt = dense_frame(df), dense_frame(dt)
pp = Prep().fit(df, D); A = pp.transform(df, D); T = pp.transform(dt, Dt); nt, na = A[4], A[5]
sub = lambda d, i: ([d[0][k] for k in i], [d[1][k] for k in i], d[2][i], d[3][i])
oof = np.zeros(len(y))
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED + 5).split(D, y)):
    oof[b] = predict_nn(train_nn(pp, sub(A, a), y[a], nt, na, epochs=7, seed=k), sub(A, b))
print("NN OOF acc", ((oof > .5) == y).mean(), round(time.time() - t0), flush=True)
nn_te = np.mean([predict_nn(train_nn(pp, A[:4], y, nt, na, epochs=7, seed=s), T[:4]) for s in (11, 12)], axis=0)
joblib.dump(dict(df=df, dt=dt, y=y, ftr=ftr, fte=fte, nn_oof=oof, nn_te=nn_te), "cache/l1.pkl")
print("done", round(time.time() - t0), flush=True)
