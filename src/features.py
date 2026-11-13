import json, re
import numpy as np
import pandas as pd

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def build_frame(rows):
    out = []
    for r in rows:
        sh = r.get("shipping") or {}
        sa = r.get("seller_address") or {}
        attrs = r.get("attributes") or []
        pics = r.get("pictures") or []
        title = (r.get("title") or "")
        wr = r.get("warranty")
        dc = pd.Timestamp(r["date_created"]).timestamp() if r.get("date_created") else np.nan
        lu = pd.Timestamp(r["last_updated"]).timestamp() if r.get("last_updated") else np.nan
        st = (r.get("start_time") or np.nan)
        sp = (r.get("stop_time") or np.nan)
        pays = sorted((p.get("id") or "") for p in (r.get("non_mercado_pago_payment_methods") or []))
        wnorm = re.sub(r"[^a-z0-9 ]", "", wr.lower())[:40] if isinstance(wr, str) else "none"
        var_tok = " ".join(f"var_{(c.get('name') or '').replace(' ', '_')}_{(c.get('value_name') or '').replace(' ', '_')}"
                           for v in (r.get("variations") or [])[:3] for c in (v.get("attribute_combinations") or []))
        pr = r.get("price") or 0.0
        toks = re.findall(r"\w+", title.lower()); slug = re.sub(r"^.*?MLA\d+-", "", (r.get("permalink") or "")); slug = re.sub(r"-?_JM$", "", slug)
        stoks = set(slug.split("-")); ttoks = set(toks)
        iq, aq, sq = (r.get("initial_quantity") or 0), (r.get("available_quantity") or 0), (r.get("sold_quantity") or 0)
        th = r.get("thumbnail") or ""; p0 = (pics[0].get("id") if pics else "") or ""
        msz = (pics[0].get("max_size") or "").split("x") if pics else []
        out.append(dict(
            first_word=toks[0] if toks else "", last_word=toks[-1] if toks else "", price_bin=str(int(np.log1p(max(pr, 0)) * 2)),
            qty_bin=str(min(iq, 6)), ship_key=f"{int(bool(sh.get('free_shipping')))}{int(bool(sh.get('local_pick_up')))}{sh.get('mode')}",
            slug_match=len(stoks & ttoks) / max(len(ttoks), 1), slug_len_diff=len(stoks) - len(ttoks),
            thumb_pic_same=int(bool(p0) and p0 in th), q_gap=iq - aq - sq, q_sold_gap=iq - aq,
            start_gap=((st - dc * 1000) / 86400000.0) if st == st and dc == dc else np.nan,
            n_exclam=title.count("!"), n_dollar=title.count("$"), n_dash=title.count("-"), n_dot=title.count("."), n_comma=title.count(","),
            n_paren=title.count("(") + title.count("["), n_slash=title.count("/") + title.count("\\"), n_plus=title.count("+"),
            n_caps_words=sum(w.isupper() and len(w) > 1 for w in title.split()), max_tok=max((len(w) for w in toks), default=0),
            n_numtok=sum(w.isdigit() for w in toks), has_year=int(bool(re.search(r"\b(19|20)\d\d\b", title))),
            title_alpha=sum(c.isalpha() for c in title) / max(len(title), 1), nonascii=sum(ord(c) > 127 for c in title),
            pic_max_h=int(msz[1]) if len(msz) == 2 and msz[1].isdigit() else np.nan,
            pay_combo="|".join(pays), warr_norm=wnorm,
            price_cents=round(pr - int(pr), 2), price_r10=int(pr % 10 == 0), price_r100=int(pr % 100 == 0), price_r1000=int(pr % 1000 == 0),
            price_9=int(int(pr) % 10 == 9), price_99=int(int(pr) % 100 == 99), price_int_len=len(str(int(pr))),
            has_deal=int(bool(r.get("deal_ids"))), n_var_pics=sum(len(v.get("picture_ids") or []) for v in (r.get("variations") or [])),
            var_sold=sum((v.get("sold_quantity") or 0) for v in (r.get("variations") or [])),
            price=r.get("price"), base_price=r.get("base_price"),
            has_original_price=int(r.get("original_price") is not None),
            initial_quantity=r.get("initial_quantity"), sold_quantity=r.get("sold_quantity"),
            available_quantity=r.get("available_quantity"),
            n_pics=len(pics), n_attrs=len(attrs), n_vars=len(r.get("variations") or []),
            n_tags=len(r.get("tags") or []), n_pay=len(r.get("non_mercado_pago_payment_methods") or []),
            n_desc=len(r.get("descriptions") or []), n_sub=len(r.get("sub_status") or []),
            has_warranty=int(wr is not None), warranty_len=len(wr) if isinstance(wr, str) else 0,
            warr_nuevo=int(isinstance(wr, str) and "nuevo" in wr.lower()),
            warr_usado=int(isinstance(wr, str) and "usad" in wr.lower()),
            warr_sin=int(isinstance(wr, str) and ("sin garant" in wr.lower())),
            has_subtitle=int(bool(r.get("subtitle"))), has_video=int(r.get("video_id") is not None),
            has_catalog=int(r.get("catalog_product_id") is not None),
            has_official=int(r.get("official_store_id") is not None),
            accepts_mp=int(bool(r.get("accepts_mercadopago"))),
            auto_relist=int(bool(r.get("automatic_relist"))),
            free_ship=int(bool(sh.get("free_shipping"))), pickup=int(bool(sh.get("local_pick_up"))),
            n_ship_methods=len(sh.get("methods") or []),
            title_len=len(title), title_words=len(title.split()),
            title_upper=sum(c.isupper() for c in title) / max(len(title), 1),
            title_digits=sum(c.isdigit() for c in title),
            date_created=dc, life_days=(sp - st) / 86400.0 if st == st and sp == sp else np.nan,
            upd_gap=(lu - dc) / 86400.0 if lu == lu and dc == dc else np.nan,
            listing=r.get("listing_type_id"), buying_mode=r.get("buying_mode"),
            currency=r.get("currency_id"), status=r.get("status"),
            ship_mode=sh.get("mode"), intl=r.get("international_delivery_mode"),
            state=(sa.get("state") or {}).get("id"), city=(sa.get("city") or {}).get("id"),
            category=r.get("category_id"), seller=r.get("seller_id"),
            attr_text=" ".join(f"{a.get('id','')}={a.get('value_id') or a.get('value_name') or ''}" for a in attrs) + " " + var_tok,
            tag_text=" ".join(r.get("tags") or []) + " " + " ".join("pay_" + p for p in pays) + " " + " ".join("w_" + w for w in wnorm.split()) + (" has_deal" if r.get("deal_ids") else ""),
            title=title.lower(),
        ))
    df = pd.DataFrame(out)
    df["price_log"] = np.log1p(df["price"].astype(float).clip(lower=0))
    df["sold_ratio"] = df["sold_quantity"] / df["initial_quantity"].clip(lower=1)
    return df

CAT_COLS = ["first_word", "last_word", "price_bin", "qty_bin", "ship_key", "pay_combo", "warr_norm", "listing", "buying_mode", "currency", "status", "ship_mode", "intl", "state", "city", "category", "seller"]

def pic_frame(rows):
    out = []
    for r in rows:
        pics = r.get("pictures") or []
        own = r["id"]
        d = dict(pic_own=np.nan, pic_other_n=0, pic_prefix=np.nan, pic_ym=np.nan, pic_mm_gap=np.nan,
                 pic_w=np.nan, pic_h=np.nan, pic_ar=np.nan, pic_secure=0, th_server=-1, th_secure=0,
                 th_pic_same=np.nan, pic_nsizes=0, pic_maxw=np.nan)
        if pics:
            m = [re.match(r"(\d+)-(MLA\d+)_(\w+)$", p.get("id", "")) for p in pics]
            ok = [x for x in m if x]
            if ok:
                f = ok[0]
                d["pic_own"] = int(f.group(2) == own)
                d["pic_other_n"] = sum(x.group(2) != own for x in ok)
                d["pic_prefix"] = int(f.group(1))
                s = f.group(3)
                if re.fullmatch(r"\d{6}", s):
                    mo, yr = int(s[:2]), int(s[2:])
                    d["pic_ym"] = yr * 12 + mo
                    try:
                        dc = pd.Timestamp(r["date_created"])
                        d["pic_mm_gap"] = dc.year * 12 + dc.month - d["pic_ym"]
                    except Exception: pass
            sz = (pics[0].get("size") or "").split("x"); ms = (pics[0].get("max_size") or "").split("x")
            if len(sz) == 2 and sz[0].isdigit(): d["pic_w"], d["pic_h"] = int(sz[0]), int(sz[1]); d["pic_ar"] = d["pic_w"] / max(d["pic_h"], 1)
            if len(ms) == 2 and ms[0].isdigit(): d["pic_maxw"] = int(ms[0])
            d["pic_nsizes"] = len({p.get("size") for p in pics})
            d["pic_secure"] = int("a248" in (pics[0].get("secure_url") or ""))
        th = r.get("thumbnail") or ""
        mm = re.search(r"mla-s(\d)-p", th); d["th_server"] = int(mm.group(1)) if mm else -1
        d["th_secure"] = int(th.startswith("https"))
        d["id_num"] = int(r["id"][3:])
        pm = re.search(r"MLA(\d+)", r.get("parent_item_id") or "")
        d["parent_num"] = int(pm.group(1)) if pm else np.nan
        d["has_parent"] = int(pm is not None)
        d["parent_id_gap"] = d["id_num"] - d["parent_num"] if pm else np.nan
        out.append(d)
    return pd.DataFrame(out)
