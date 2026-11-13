import re, zlib, numpy as np, pandas as pd, torch, torch.nn as nn
from collections import Counter
from sklearn.preprocessing import QuantileTransformer
torch.set_num_threads(2)
CAT_FIELDS = ["category", "seller", "city", "state", "listing", "ship_key", "pay_combo", "warr_norm", "first_word", "price_bin"]
NB = 2 ** 17

def toks_title(t):
    w = re.findall(r"\w+", t); return w, [a + "_" + b for a, b in zip(w, w[1:])]

class Prep:
    def fit(self, d, dense):
        self.vocab = {w: i + 1 for i, (w, c) in enumerate(Counter(w for t in d["title"] for w in toks_title(t)[0]).items()) if c >= 2}
        self.vocab = {w: i + 1 for i, w in enumerate(self.vocab)}
        self.avoc = {w: i + 1 for i, w in enumerate([w for w, c in Counter(x for t in (d["attr_text"] + " " + d["tag_text"]) for x in t.split()).items() if c >= 3])}
        self.cats = {f: {v: i + 1 for i, v in enumerate([v for v, c in d[f].astype(str).value_counts().items() if c >= 2])} for f in CAT_FIELDS}
        X = dense.astype(float); X = X.loc[:, X.notna().any()]; self.dcols = list(X.columns)
        self.med = X.median().fillna(0); self.qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=10**9, random_state=0).fit(X.fillna(self.med))
        return self
    def bags(self, d):
        title_ids, attr_ids = [], []
        nv = len(self.vocab) + 1
        for t, a in zip(d["title"], d["attr_text"] + " " + d["tag_text"]):
            w, b = toks_title(t)
            title_ids.append([self.vocab[x] for x in w if x in self.vocab] + [nv + zlib.crc32(x.encode()) % NB for x in b] or [0])
            attr_ids.append([self.avoc[x] for x in a.split() if x in self.avoc] or [0])
        return title_ids, attr_ids, nv + NB, len(self.avoc) + 1
    def transform(self, d, dense):
        ti, ai, nt, na = self.bags(d)
        cat = np.stack([d[f].astype(str).map(self.cats[f]).fillna(0).astype(int).values for f in CAT_FIELDS], 1)
        num = self.qt.transform(dense[self.dcols].astype(float).fillna(self.med)).astype(np.float32)
        return ti, ai, cat, num, nt, na

def _bag(lists):
    flat = torch.tensor([i for l in lists for i in l], dtype=torch.long); off = torch.tensor(np.cumsum([0] + [len(l) for l in lists[:-1]]), dtype=torch.long)
    return flat, off

class Net(nn.Module):
    def __init__(self, nt, na, ncats, nnum, dim=48):
        super().__init__()
        self.et = nn.EmbeddingBag(nt, dim, mode="mean"); self.ea = nn.EmbeddingBag(na, 24, mode="mean")
        self.ec = nn.ModuleList([nn.Embedding(n + 1, 24 if f == "seller" else 16, padding_idx=0) for f, n in zip(CAT_FIELDS, ncats)])
        tot = dim + 24 + sum(24 if f == "seller" else 16 for f in CAT_FIELDS) + 64
        self.num = nn.Sequential(nn.Linear(nnum, 64), nn.ReLU())
        self.mlp = nn.Sequential(nn.BatchNorm1d(tot), nn.Dropout(0.3), nn.Linear(tot, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, tb, ab, cat, num):
        z = [self.et(*tb), self.ea(*ab)] + [e(cat[:, i]) for i, e in enumerate(self.ec)] + [self.num(num)]
        return self.mlp(torch.cat(z, 1)).squeeze(1)

def batches(data, idx):
    ti, ai, cat, num = data
    return (_bag([ti[i] for i in idx]), _bag([ai[i] for i in idx]), torch.tensor(cat[idx]), torch.tensor(num[idx]))

def train_nn(pp, tr_data, ytr, nt, na, epochs=6, va_data=None, yva=None, seed=0, bs=256, lr=2e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    ncats = [len(pp.cats[f]) for f in CAT_FIELDS]
    net = Net(nt, na, ncats, tr_data[3].shape[1]); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(ytr) + bs - 1) // bs))
    yt = torch.tensor(ytr, dtype=torch.float32); lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(len(ytr))
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            if len(idx) < 2: continue
            opt.zero_grad(); loss = lossf(net(*batches(tr_data, idx)), yt[idx]); loss.backward(); opt.step(); sched.step()
        if va_data is not None:
            p = predict_nn(net, va_data); print(f"  epoch {ep} val acc {((p > .5) == yva).mean():.4f}", flush=True)
    return net

def predict_nn(net, data, bs=2048):
    net.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(data[3]), bs):
            idx = np.arange(s, min(s + bs, len(data[3]))); out.append(torch.sigmoid(net(*batches(data, idx))).numpy())
    return np.concatenate(out)
