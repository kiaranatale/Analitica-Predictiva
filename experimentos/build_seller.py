import os, sys, time, joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from seller_feats import seller_agg, seller_knn, seller_pred_mean
t0 = time.time(); c = joblib.load("cache/l1.pkl"); y = c["y"]; n = len(y)
A = pd.concat([c["df"], c["dt"]], ignore_index=True)
save = lambda name, F: joblib.dump({"tr": F.iloc[:n].reset_index(drop=True), "te": F.iloc[n:].reset_index(drop=True)}, f"cache/{name}.pkl")
save("s_agg", seller_agg(A)); print("agg", round(time.time() - t0), flush=True)
save("s_knn", seller_knn(A, y, n)); print("knn", round(time.time() - t0), flush=True)
