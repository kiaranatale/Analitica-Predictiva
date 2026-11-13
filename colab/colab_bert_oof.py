# ====== COLAB (Runtime > Change runtime type > T4 GPU). Subir train_data.jsonlines y test_data.jsonlines a /content ======
# !pip -q install transformers
import json, numpy as np, torch, time
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

MODEL = "dccuchile/bert-base-spanish-wwm-cased"   # si falla: "bert-base-multilingual-cased"
EPOCHS, BS, LR, MAXLEN = 2, 64, 3e-5, 48
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
tr, te = load("/content/train_data.jsonlines"), load("/content/test_data.jsonlines")
txt = lambda r: f"{r['title']} | {r['category_id']} | {r['listing_type_id']}"
Xtr, Xte = [txt(r) for r in tr], [txt(r) for r in te]
y = np.array([r["condition"] == "used" for r in tr]).astype(int)
tok = AutoTokenizer.from_pretrained(MODEL); dev = "cuda"
enc = lambda T: tok(T, truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt")
Etr, Ete = enc(Xtr), enc(Xte)

def predict(model, E, idx):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for s in range(0, len(idx), 256):
            b = idx[s:s + 256]
            lg = model(input_ids=E["input_ids"][b].to(dev), attention_mask=E["attention_mask"][b].to(dev)).logits
            out.append(torch.softmax(lg.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)

oof = np.zeros(len(y)); test_pred = np.zeros(len(Xte)); t0 = time.time()
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(Xtr, y)):
    torch.manual_seed(k)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(a) + BS - 1) // BS); sch = get_linear_schedule_with_warmup(opt, int(.06 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(); yt = torch.tensor(y)
    for ep in range(EPOCHS):
        model.train(); perm = np.random.RandomState(ep + 10 * k).permutation(a)
        for s in range(0, len(perm), BS):
            i = perm[s:s + BS]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=Etr["input_ids"][i].to(dev), attention_mask=Etr["attention_mask"][i].to(dev), labels=yt[i].to(dev)).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
    oof[b] = predict(model, Etr, b); test_pred += predict(model, Ete, np.arange(len(Xte))) / 5
    print(f"fold {k} acc {((oof[b] > .5) == y[b]).mean():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    del model, opt; torch.cuda.empty_cache()
print("OOF acc", ((oof > .5) == y).mean())
np.save("oof_bert.npy", oof); np.save("test_bert.npy", test_pred)
from google.colab import files; files.download("oof_bert.npy"); files.download("test_bert.npy")
