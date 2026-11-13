import numpy as np, pandas as pd
from features import load, build_frame, pic_frame

CATS = ["first_word", "last_word", "price_bin", "qty_bin", "ship_key", "pay_combo", "warr_norm", "seller", "category", "city",
        "state", "listing", "ship_mode", "buying_mode"]

def make_df(rows):
    return pd.concat([build_frame(rows), pic_frame(rows)], axis=1)

def with_cats(X, d):
    """Agrega las columnas categoricas crudas (cf_*) que usa CatBoost."""
    X = X.copy()
    for c in CATS: X["cf_" + c] = d[c].astype(str).values
    X["cf_seller_cat"] = (d["seller"].astype(str) + "|" + d["category"].astype(str)).values
    X["cf_cat_listing"] = (d["category"].astype(str) + "|" + d["listing"].astype(str)).values
    X["cf_seller_listing"] = (d["seller"].astype(str) + "|" + d["listing"].astype(str)).values
    return X

def labels(rows):
    return np.array([r["condition"] == "used" for r in rows]).astype(int)
