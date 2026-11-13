import sys, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from features import load, build_frame
from model_lib import *
rows = load(sys.argv[1]); y = np.array([r["condition"] == "used" for r in rows]).astype(int)
df = build_frame(rows)
tr, va = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y)
# baseline: arbol con 8 numericas
base = ["price", "initial_quantity", "sold_quantity", "available_quantity", "n_pics", "has_warranty", "free_ship", "title_len"]
X = df[base].fillna(-1).values
m = DecisionTreeClassifier(max_depth=6, random_state=SEED).fit(X[tr], y[tr])
print("baseline tree acc", (m.predict(X[va]) == y[va]).mean())
# logistica solo titulo
v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True).fit(df.title.iloc[tr])
lr = LogisticRegression(C=3, max_iter=1000).fit(v.transform(df.title.iloc[tr]), y[tr])
print("logreg title acc", (lr.predict(v.transform(df.title.iloc[va])) == y[va]).mean())
# lgbm completo
enc, tf = fit_encoders(df.iloc[tr]); X, ci = transform(df, enc, tf)
m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63, colsample_bytree=0.3, subsample=0.8, subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=2)
m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="binary_error", callbacks=[lgb.early_stopping(100, verbose=False)])
print("lgbm acc", (m.predict(X[va]) == y[va]).mean(), "best_iter", m.best_iteration_)
