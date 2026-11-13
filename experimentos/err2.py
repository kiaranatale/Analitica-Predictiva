import numpy as np, json, collections
from sklearn.model_selection import train_test_split
rows=[json.loads(l) for l in open('/mnt/user-data/uploads/train_data.jsonlines')]
y=np.array([r['condition']=='used' for r in rows]).astype(int)
tr,va=train_test_split(np.arange(len(rows)),test_size=0.2,random_state=42,stratify=y)
vr=[rows[i] for i in va]; yv=y[va]
p=0.7*np.load('pc.npy')+0.3*np.load('pl.npy'); err=(p>.5)!=yv
print('acc',1-err.mean())
conf=np.abs(p-.5)
for lo,hi in [(0,.1),(.1,.2),(.2,.3),(.3,.4),(.4,.5)]:
    m=(conf>=lo)&(conf<hi); print(f'|p-.5| in [{lo},{hi}) n={m.sum()} err={err[m].mean():.3f} share_of_errors={err[m].sum()/err.sum():.2f}')
# errores por categoria raiz/ fraccion de errores en top categorias
c=collections.Counter(); n=collections.Counter()
for r,e in zip(vr,err): n[r['category_id']]+=1; c[r['category_id']]+=e
top=sorted(c.items(),key=lambda x:-x[1])[:10]; print([(k,v,n[k]) for k,v in top], 'share top10', sum(v for _,v in top)/err.sum())
# confident errors: ver ejemplos
idx=np.where(err&(conf>.4))[0]
print(len(idx),'errores muy confiados')
for i in idx[:25]:
    r=vr[i]; print(r['condition'],round(p[i],2),'|',r['title'][:70],'|',r['price'],r['category_id'],r['listing_type_id'],r['seller_id'])
