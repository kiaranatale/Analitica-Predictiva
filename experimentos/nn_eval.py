import numpy as np, pandas as pd, pickle, time
from sklearn.model_selection import train_test_split
from features import load
from stack_lib import dense_frame, SEED
from train_model import make_df
from nn_lib import *
rows = load("/mnt/user-data/uploads/train_data.jsonlines"); y = np.array([r["condition"] == "used" for r in rows]).astype(int)
df = make_df(rows); tr, va = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y)
dtr, dva = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
Dtr, Dva = dense_frame(dtr), dense_frame(dva)
t = time.time(); pp = Prep().fit(dtr, Dtr)
ti, ai, cat, num, nt, na = pp.transform(dtr, Dtr); vti, vai, vcat, vnum, _, _ = pp.transform(dva, Dva); print("prep", round(time.time() - t), "s", flush=True)
net = train_nn(pp, (ti, ai, cat, num), y[tr], nt, na, epochs=8, va_data=(vti, vai, vcat, vnum), yva=y[va])
p = predict_nn(net, (vti, vai, vcat, vnum)); np.save("nn_val.npy", p); print("NN solo", ((p > .5) == y[va]).mean(), round(time.time() - t), "s")
