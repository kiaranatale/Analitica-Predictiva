# ====== COLAB (GPU T4). Subir train_data.jsonlines y test_data.jsonlines a /content ======
# !pip -q install transformers
# BERT v3 (todo en uno): corre en orden beto3 -> xlmr -> beto3b (otra semilla) y descarga los .npy de cada uno al terminar.
# BERT v3: como v2 pero la entrada incluye VENDEDOR, ciudad, estado y fecha de publicacion (para que BERT aprenda patrones por vendedor junto con el texto). 3 epocas.
# Genera oof_<nombre>.npy y test_<nombre>.npy (5 folds). Corre BETO; poner RUN_XLMR = True para sumar un segundo modelo (~35 min mas).
import json, re, numpy as np, torch, time
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

MODELS = [("beto3", "dccuchile/bert-base-spanish-wwm-cased", 3e-5, 100),   # ~70 min
          ("xlmr", "xlm-roberta-base", 2e-5, 200),                          # ~70 min
          ("beto3b", "dccuchile/bert-base-spanish-wwm-cased", 3e-5, 300)]   # ~70 min (otra semilla)
EPOCHS, BS, MAXLEN = 3, 64, 128        # si falta tiempo: EPOCHS = 2
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
tr, te = load("/content/train_data.jsonlines"), load("/content/test_data.jsonlines")

def txt(r):
    at = " ; ".join(f"{a.get('name','')}: {a.get('value_name') or ''}" for a in (r.get("attributes") or [])[:8])
    pay = ",".join(p.get("description", "") for p in (r.get("non_mercado_pago_payment_methods") or []))
    sh = r.get("shipping") or {}
    sa = r.get("seller_address") or {}; city = (sa.get("city") or {}).get("name", ""); st = (sa.get("state") or {}).get("name", "")
    return (f"vendedor {r['seller_id']} | {r['title']} | {r['category_id']} | {city} {st} | {str(r.get('date_created'))[:7]} | {r['listing_type_id']} | precio {r.get('price')} | cantidad {r.get('initial_quantity')} "
            f"| garantia: {(r.get('warranty') or 'ninguna')[:50]} | envio {sh.get('mode')} gratis {sh.get('free_shipping')} | pagos: {pay} | {at}")
Xtr, Xte = [txt(r) for r in tr], [txt(r) for r in te]
y = np.array([r["condition"] == "used" for r in tr]).astype(int); dev = "cuda"

def run(name, MODEL, LR, SEED):
    tok = AutoTokenizer.from_pretrained(MODEL)
    Itr = tok(Xtr, truncation=True, max_length=MAXLEN)["input_ids"]; Ite = tok(Xte, truncation=True, max_length=MAXLEN)["input_ids"]
    pad = tok.pad_token_id
    def batch(I, idx):
        L = max(len(I[i]) for i in idx); ids = torch.full((len(idx), L), pad); 
        for k, i in enumerate(idx): ids[k, :len(I[i])] = torch.tensor(I[i])
        return ids.to(dev), (ids != pad).long().to(dev)
    def predict(model, I, idx):
        model.eval(); out = []; order = sorted(idx, key=lambda i: len(I[i])); 
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for s in range(0, len(order), 256):
                ids, m = batch(I, order[s:s + 256]); out.append(torch.softmax(model(input_ids=ids, attention_mask=m).logits.float(), -1)[:, 1].cpu().numpy())
        p = np.concatenate(out); res = np.zeros(len(idx)); pos = {i: k for k, i in enumerate(idx)}
        for o, v in zip(order, p): res[pos[o]] = v
        return res
    oof = np.zeros(len(y)); tp = np.zeros(len(Xte)); t0 = time.time()
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(Xtr, y)):
        torch.manual_seed(k + SEED); model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01); steps = EPOCHS * ((len(a) + BS - 1) // BS)
        sch = get_linear_schedule_with_warmup(opt, int(.06 * steps), steps); scaler = torch.amp.GradScaler(); yt = torch.tensor(y)
        for ep in range(EPOCHS):
            model.train(); perm = np.random.RandomState(ep + 10 * k + SEED).permutation(a)
            for s in range(0, len(perm), BS):
                i = list(perm[s:s + BS]); ids, m = batch(Itr, i)
                with torch.autocast("cuda", dtype=torch.float16): loss = model(input_ids=ids, attention_mask=m, labels=yt[i].to(dev)).loss
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step()
        oof[b] = predict(model, Itr, list(b)); tp += predict(model, Ite, list(range(len(Xte)))) / 5
        print(f"{name} fold {k} acc {((oof[b] > .5) == y[b]).mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
        del model, opt; torch.cuda.empty_cache()
    print(name, "OOF acc", ((oof > .5) == y).mean()); np.save(f"oof_{name}.npy", oof); np.save(f"test_{name}.npy", tp)

from google.colab import files
for n, m, lr, sd in MODELS:
    run(n, m, lr, sd); files.download(f"oof_{n}.npy"); files.download(f"test_{n}.npy")   # baja cada modelo apenas termina
