import pickle, numpy as np, lightgbm as lgb
Xtr,ytr,Xva,yva=pickle.load(open('cache_eval4.pkl','rb'))
cfgs={"actual 63 hojas":dict(num_leaves=63),
 "15 hojas":dict(num_leaves=15,min_child_samples=40),
 "31 hojas+L2":dict(num_leaves=31,min_child_samples=40,reg_lambda=5),
 "7 hojas":dict(num_leaves=7,min_child_samples=60,reg_lambda=5),
 "31 hojas+L2+colsample .3":dict(num_leaves=31,min_child_samples=40,reg_lambda=10,colsample_bytree=0.3)}
for n,c in cfgs.items():
    p=dict(n_estimators=4000,learning_rate=0.03,colsample_bytree=0.5,subsample=0.8,subsample_freq=1,random_state=42,verbose=-1,n_jobs=2); p.update(c)
    m=lgb.LGBMClassifier(**p).fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="binary_error",callbacks=[lgb.early_stopping(200,verbose=False)])
    print(n,"val",round((m.predict(Xva)==yva).mean(),4),"train",round((m.predict(Xtr)==ytr).mean(),4),"iters",m.best_iteration_,flush=True)
