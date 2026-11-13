import pickle, numpy as np, xgboost as xgb
Xtr,ytr,Xva,yva=pickle.load(open('cache_eval4.pkl','rb'))
pc=np.load('pc.npy'); pl=np.load('pl.npy')
for n,c in {"depth6":dict(max_depth=6,reg_alpha=1,reg_lambda=5),"depth8":dict(max_depth=8,reg_alpha=1,reg_lambda=10,min_child_weight=3)}.items():
    m=xgb.XGBClassifier(n_estimators=3000,learning_rate=0.03,colsample_bytree=0.4,subsample=0.8,tree_method="hist",n_jobs=2,random_state=42,early_stopping_rounds=150,eval_metric="error",**c)
    m.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False); px=m.predict_proba(Xva)[:,1]
    print(n,"xgb",round(((px>.5)==yva).mean(),4),"iters",m.best_iteration,"| blend cb.6+xgb.4",round((((.6*pc+.4*px)>.5)==yva).mean(),4),"| cb.5+lgb.2+xgb.3",round((((.5*pc+.2*pl+.3*px)>.5)==yva).mean(),4),flush=True)
