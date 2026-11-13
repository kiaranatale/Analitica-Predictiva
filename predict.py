"""Aplica el modelo final al test y genera submission.csv.
Uso: python predict.py --test data/test_data.jsonlines --bert-test colab/test_bert.npy colab/test_beto.npy"""
import argparse, os, sys, joblib
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from common import load, make_df, with_cats
from stack_lib import apply_stack, dense_frame
from nn_lib import Net, CAT_FIELDS, predict_nn

ap = argparse.ArgumentParser(); ap.add_argument("--test", required=True); ap.add_argument("--bert-test", required=True, nargs="+")
ap.add_argument("--models", default="models"); ap.add_argument("--out", default="submission.csv"); a = ap.parse_args()
m = joblib.load(os.path.join(a.models, "model.pkl")); rows = load(a.test); df = make_df(rows); D = dense_frame(df)
pp = m["pp"]; T = pp.transform(df, D)
nets = []
for sd in torch.load(os.path.join(a.models, "nets.pt")):
    n = Net(m["nt"], m["na"], [len(pp.cats[f]) for f in CAT_FIELDS], T[3].shape[1]); n.load_state_dict(sd); nets.append(n)
X = pd.concat([D, apply_stack(df, m["stack"])], axis=1)
for i, f in enumerate(a.bert_test): X[f"bert{i}"] = np.load(f)
X["nn"] = np.mean([predict_nn(n, T[:4]) for n in nets], axis=0); X = X[m["cols"]]
C = with_cats(X, df)
p = 0.3 * m["lgb"].predict_proba(X)[:, 1] + 0.7 * np.mean([c.predict_proba(C)[:, 1] for c in m["cbs"]], axis=0)
pd.DataFrame({"ID": range(1, len(rows) + 1), "condition": np.where(p > .5, "used", "new")}).to_csv(a.out, index=False)
print("escrito", a.out, len(rows), "filas")
